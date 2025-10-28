from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from spider.config import JobConfig
from .backends.factory import create_backend
from .sources import collect_prompts
from .writers import write_jsonl

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

    backend = create_backend(job.model)
    prompts = collect_prompts(job,sources)
    if not prompts:
        artifact_path.write_text("", encoding="utf-8")
        return JobExecutionResult(
            artifact_path=artifact_path,
            metrics={"records": 0},
            messages=["No prompts found; nothing generated."]
        )

    try:
        generations = backend.generate(
            prompts,
            parameters=job.generation.parameters
        )
    except Exception as exc:
        raise JobExecutionError(f"Generation pipeline failed: {exc}") from exc

    record_count = write_jsonl(artifact_path, prompts, generations)
    metrics = {"records": record_count, **backend.metrics()}

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
        metrics=metrics,
        messages=["Generation pipeline completed."]
    )