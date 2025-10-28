from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from spider.config import JobConfig, SourceConfig, SourceType
from .backends.factory import create_backend
from .sources import resolve_sources

class JobExecutionError(Exception):
    pass

@dataclass
class JobExecutionResult:
    artifact_path: Path
    metrics: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

def run_generation_job(
    job_id: str, job: JobConfig, *, workspace: Path
) -> JobExecutionResult:
    workspace.mkdir(parents=True, exist_ok=True)
    artifact_path = workspace / "result.jsonl"

    try:
        record_count, metrics = _generate_records(
            backend=backend,
            sources=sources,
            job=job,
            output_path=artifact_path,
        )
    except JobExecutionError:
        raise
    except Exception as exc:
        raise JobExecutionError(f"Generation pipeline failed: {exc}") from exc

    payload = {
        "job_id": job_id,
        "model": job.model.model_dump(exclude_none=True),
        "output": job.output.model_dump(exclude_none=True),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "records": record_count,
    }

    metadata_path = workspace / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return JobExecutionResult(
        artifact_path=artifact_path,
        metrics={"records": record_count, **metrics},
        messages=["Generation pipeline completed."]
    )

def _generate_records(
    *, backend, sources: Sequence[Sequence[str]], job: JobConfig, output_path: Path
) -> tuple[int, Dict[str, Any]]:
    from .writers import write_jsonl

    prompts: List[str] = []
    for source_batch in sources:
        prompts.extend(source_batch)

    if not prompts:
        return 0, {}
    
    generations = backend.generate(
        prompts, parameters=job.generation.parameters
    )

    record_count = write_jsonl(output_path, prompts, generations)
    return record_count, backend.metrics()