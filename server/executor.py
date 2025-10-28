from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from spider.config import JobConfig, OutputMode
from .backends.factory import create_backend
from .sources import collect_prompts
from .writers import JSONLBatchWriter
from .hf_upload import HFUploadError, publish_to_hub

class JobExecutionError(Exception):
    pass

@dataclass
class JobExecutionResult:
    artifacts_path: Path
    remote_artifact: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

def run_generation_job(
    job_id: str, job: JobConfig, *, workspace: Path
) -> JobExecutionResult:
    workspace.mkdir(parents=True, exist_ok=True)
    artifact_path = workspace / "result.jsonl"

    _ensure_tensor_parallel(job)
    backend = create_backend(job.model)
    prompts = collect_prompts(job.sources)
    batch_size = _resolve_batch_size(job, prompts)
    if not prompts:
        artifact_path.write_text("", encoding="utf-8")
        return JobExecutionResult(
            artifacts_path=artifact_path,
            metrics={"records": 0},
            messages=["No prompts found; nothing generated."]
        )

    aggregated_metrics = {}
    records_written = 0
    metadata_path = workspace / "metadata.json"
    payload = _base_metadata(job_id, job)
    _write_metadata(metadata_path, payload, records_written)

    try:
        with JSONLBatchWriter(artifact_path) as writer:
            for chunk_start in range(0, len(prompts), batch_size):
                chunk = prompts[chunk_start : chunk_start + batch_size]
                chunk_generations = backend.generate(
                    chunk, parameters=job.generation.parameters
                )
                if len(chunk_generations) != len(chunk):
                    raise JobExecutionError(
                        f"Generation count mismatch within batch: "
                        f"expected {len(chunk)}, received {len(chunk_generations)}"
                    )
                writer.write_batch(chunk, chunk_generations)
                records_written = writer.count

                batch_metrics = dict(backend.metrics() or {})
                for key, value in batch_metrics.items():
                    if isinstance(value, (int, float)):
                        aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + float(value)

                payload["metrics"] = _summarize_metrics(records_written, aggregated_metrics)
                _write_metadata(metadata_path, payload, records_written)
    except Exception as exc:
        raise JobExecutionError(f"Generation pipeline failed: {exc}") from exc

    if records_written != len(prompts):
        raise JobExecutionError(f"Generation count mismatch, expected {len(prompts)} but received {records_written}")

    metrics = _summarize_metrics(records_written, aggregated_metrics)
    payload["metrics"] = metrics
    _write_metadata(metadata_path, payload, records_written)
    
    hf_url = None
    if job.output.mode == OutputMode.HF_UPLOAD and job.output.hf:
        try:
            hf_url = publish_to_hub(
                job_id=job_id,
                artifact=artifact_path,
                metadata=metadata_path,
                config=job.output.hf
            )
        except HFUploadError as exc:
            raise JobExecutionError(str(exc)) from exc

    return JobExecutionResult(
        artifacts_path=artifact_path,
        remote_artifact=hf_url,
        metrics=metrics,
        messages=["Generation pipeline completed."]
    )

def _resolve_batch_size(job: JobConfig, prompts: List[str]) -> int:
    requested = job.generation.max_batch_size
    if isinstance(requested, int) and requested > 0:
        return max(1, min(requested, len(prompts) or 1))
    base = 4
    gpu_factor = job.model.parameters.get("tensor_parallel_size", 1)
    if not isinstance(gpu_factor, int) or gpu_factor < 1:
        gpu_factor = 1
    inferred = base * gpu_factor
    return max(1, min(inferred, len(prompts) or 1))

def _ensure_tensor_parallel(job: JobConfig) -> None:
    params = job.model.parameters
    if params.get("tensor_parallel_size"):
        return
    params["tensor_parallel_size"] = _detect_available_gpus()

def _detect_available_gpus() -> int:
    try:
        import torch
        count = torch.cuda.device_count()
        if count > 0:
            return count
    except Exception:
        return 1
    return 1

def _base_metadata(job_id: str, job: JobConfig) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "model": job.model.model_dump(exclude_none=True),
        "output": job.output.model_dump(exclude_none=True),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "records": 0,
        "metrics": {"records": 0}
    }

def _write_metadata(path: Path, payload: Dict[str, Any], records: int) -> None:
    payload["records"] = records
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _summarize_metrics(records: int, aggregated: Dict[str, float]) -> Dict[str, Any]:
    metrics = {"records": records}
    for key, value in aggregated.items():
        metrics[key] = int(value) if float(value).is_integer() else value
    return metrics