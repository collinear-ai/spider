from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from spider.config import JobConfig
from .executor import JobExecutionError, run_generation_job

app = FastAPI(title="Spider Data Generation Service", version="0.1.0")

ARTIFACT_ROOT = Path("./.artifacts")
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobRecord(BaseModel):
    job: JobConfig
    status: JobStatus
    submitted_at: datetime
    updated_at: datetime
    artifacts_path: Optional[str] = None
    error_message: Optional[str] = None
    messages: List[str] = []
    metrics: Dict[str, Any] = Field(default_factory=dict)

_JOB_STORE: Dict[str, JobRecord] = {}

@app.post("/v1/jobs")
async def create_job(job: JobConfig, background: BackgroundTasks):
    job_id = str(uuid4())
    now = datetime.utcnow()
    record = JobRecord(
        job=job, status=JobStatus.QUEUED, submitted_at=now, updated_at=now, messages=["Job accepted."]
    )
    _JOB_STORE[job_id] = record
    background.add_task(_execute_job, job_id)
    return {"job_id": job_id, "status": record.status}

def _execute_job(job_id: str) -> None:
    record = _JOB_STORE.get(job_id)
    if record is None or record.status != JobStatus.QUEUED:
        return
    record.status = JobStatus.RUNNING
    record.updated_at = datetime.utcnow()
    record.messages.append("Job started.")
    try:
        result = run_generation_job(job_id, record.job, workspace=ARTIFACT_ROOT / job_id)
        record.artifacts_path = str(result.artifacts_path)
        record.metrics = result.metrics
        if result.messages:
            record.messages.extend(result.messages)
        record.status = JobStatus.COMPLETED
    except JobExecutionError as exc:
        record.status = JobStatus.FAILED
        record.error_message = str(exc)
        record.messages.append(f"Job failed: {exc}")
    except Exception as exc:
        record.status = JobStatus.FAILED
        record.error_message = str(exc)
        record.messages.append(f"Job crashed: {exc}")
    finally:
        record.updated_at = datetime.utcnow()

@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
    record = _JOB_STORE.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id, 
        "status": record.status, 
        "submitted_at": record.submitted_at,
        "updated_at": record.updated_at, 
        "messages": record.messages, 
        "error": record.error_message,
        "artifacts_path": record.artifacts_path,
        "metrics": record.metrics,
    }

@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    record = _JOB_STORE.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if record.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        return {"job_id": job_id, "status": record.status}
    record.status = JobStatus.CANCELLED
    record.updated_at = datetime.utcnow()
    record.messages.append("Cancellation requested.")
    return {"job_id": job_id, "status": record.status}

@app.get("/v1/jobs/{job_id}/result")
async def download_result(job_id: str):
    record = _JOB_STORE.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not record.artifacts_path:
        raise HTTPException(status_code=404, detail="Result not available yet")
    payload_text = Path(record.artifacts_path).read_text(encoding="utf-8")
    try:
        artifact = json.loads(payload_text)
    except json.JSONDecodeError:
        artifact = {"raw": payload_text}
    return JSONResponse(
        content={
            "job_id": job_id, 
            "payload": record.status,
            "metrics": record.metrics,
            "error": record.error_message,
            "artifact": artifact,
        },
        media_type="application/json",
    )