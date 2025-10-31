from __future__ import annotations

import json, os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Iterable, Tuple
from types import MappingProxyType

from spider.config import JobConfig, OutputMode, ProcessorConfig
from .backends.factory import create_backend
from .sources import collect_prompts
from .writers import JSONLBatchWriter
from .hf_upload import HFUploadError, publish_to_hub
from .on_policy import run_on_policy_job

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
    if job.generation.on_policy:
        return run_on_policy_job(job_id, job, workspace=workspace)
    return _run_off_policy_job(job_id, job, workspace=workspace)

def _run_off_policy_job(
    job_id: str, job: JobConfig, *, workspace: Path
) -> JobExecutionResult:
    artifact_path = workspace / "result.jsonl"

    _ensure_tensor_parallel(job)
    backend = create_backend(job.model)
    prompts = collect_prompts(job.source)
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
        processor = _resolve_processor(job.processor) if job.processor else None
        pending = {}
        next_index = 0

        with JSONLBatchWriter(artifact_path) as writer:
            executor_context = (
                ThreadPoolExecutor(max_workers=_processing_worker_count()) 
                if processor else None
            )

            try:
                for batch_index, chunk_start in enumerate(range(0, len(prompts), batch_size)):
                    chunk = prompts[chunk_start : chunk_start + batch_size]
                    chunk_generations = backend.generate(
                        chunk, parameters=job.generation.parameters
                    )
                    if len(chunk_generations) != len(chunk):
                        raise JobExecutionError(
                            f"Generation count mismatch within batch: "
                            f"expected {len(chunk)}, received {len(chunk_generations)}"
                        )
                    if processor and executor_context:
                        future = executor_context.submit(_process_batch, chunk, chunk_generations, processor)
                    else:
                        future = _immediate_future(_pair_records(chunk, chunk_generations))

                    batch_metrics = dict(backend.metrics() or {})
                    pending[batch_index] = (future, batch_metrics)
                    next_index = _drain_ready_batches(
                        pending, next_index, writer, aggregated_metrics, payload,
                        metadata_path, block=False
                    )
                next_index = _drain_ready_batches(
                    pending, next_index, writer, aggregated_metrics, payload,
                    metadata_path, block=True
                )
            finally:
                if executor_context:
                    executor_context.shutdown(wait=True)
            
            records_written = writer.count

    except Exception as exc:
        raise JobExecutionError(f"Generation pipeline failed: {exc}") from exc

    filtered_records = max(0, len(prompts) - records_written)
    metrics = _summarize_metrics(records_written, aggregated_metrics)
    if filtered_records:
        metrics["filtered_records"] = filtered_records
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

def _pair_records(prompts: Iterable[str], generations: Iterable[str]) -> List[Dict[str, Any]]:
    paired = []
    for prompt, completion in zip(prompts, generations):
        paired.append({"prompt": prompt, "completion": completion})
    return paired

def _resolve_processor(spec: Optional[ProcessorConfig]) -> Optional[Callable[[Iterable[Dict[str, Any]]], Iterable[Dict[str, Any]]]]:
    import ast, math, numpy as np, pandas as pd, random, re
    if spec is None:
        return None
    safe_builtins = MappingProxyType({
        "len": len, "enumerate": enumerate, "range": range, "min": min, "max": max,
        "sum": sum, "any": any, "all": all, "sorted": sorted, "zip": zip, "map": map,
        "filter": filter, "list": list, "dict": dict, "set": set, "tuple": tuple, "str": str,
        "int": int, "float": float, "bool": bool,
        "isinstance": isinstance, "type": type
    })
    globals_dict = {
        "__builtins__": safe_builtins, "ast": ast, "math": math, "np": np, "pd": pd, 
        "random": random, "re": re
    }
    locals_dict = {}
    try:
        compiled = compile(spec.source, "<processor>", "exec")
        exec(compiled, globals_dict, locals_dict)
    except Exception as exc:
        raise JobExecutionError(f"Failed to load processor source: {exc}") from exc
    
    func = locals_dict.get(spec.name)
    if not callable(func):
        raise JobExecutionError(f"Processor source did not define the expected callable")
    
    def wrapped(records: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        return func(records, **spec.kwargs)
    
    return wrapped

def _process_batch(
    prompts: Iterable[str], generations: Iterable[str], 
    processor: Callable[[Iterable[Dict[str, Any]]], Iterable[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    records = _pair_records(prompts, generations)
    processed = processor(records)
    return list(processed)

def _drain_ready_batches(
    pending: Dict[int, Tuple[Future[List[Dict[str, Any]]], Dict[str, Any]]],
    next_index: int, writer: JSONLBatchWriter, aggregated_metrics: Dict[str, float],
    payload: Dict[str, Any], metadata_path: Path, *, block: bool
) -> int:
    while next_index in pending:
        future, batch_metrics = pending[next_index]
        if not block and not future.done():
            break
        try:
            records = future.result()
        except Exception as exc:
            raise JobExecutionError(f"Processor failed: {exc}") from exc
        writer.write_records(records)

        for key, value in batch_metrics.items():
            if isinstance(value, (int, float)):
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + float(value)
        
        payload["metrics"] = _summarize_metrics(writer.count, aggregated_metrics)
        _write_metadata(metadata_path, payload, writer.count)

        del pending[next_index]
        next_index += 1
    return next_index

def _processing_worker_count() -> int:
    cpu_total = os.cpu_count() or 1
    return max(1, min(4, cpu_total // 2 or 1))

def _immediate_future(result: Iterable[Dict[str, Any]]) -> Future[List[Dict[str, Any]]]:
    future = Future()
    future.set_result(list(result))
    return future

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
    output_config = job.output.model_dump(exclude_none=True)
    hf_config = output_config.get("hf")
    if isinstance(hf_config, dict) and "token" in hf_config:
        hf_config = dict(hf_config)
        hf_config.pop("token", None)
        output_config["hf"] = hf_config

    return {
        "job_id": job_id,
        "model": job.model.model_dump(exclude_none=True),
        "source": job.source.model_dump(exclude_none=True),
        "output": output_config,
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