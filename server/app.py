from __future__ import annotations

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from spider.config import JobConfig

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

_JOB_STORE: Dict[str, JobRecord] = {}

@app.post("/v1/jobs")
async def create_job(job: JobConfig, background: BackgroundTasks):
    job_id = str(uuid4())
    now = datetime.utcnow()
    _JOB_STORE[job_id] = JobRecord(
        job=job, status=JobStatus.QUEUED, submitted_at=now, updated_at=now, messages=["Job accepted."]
    )
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
        artifact_path = _write_stub_artifact(job_id, record.job)
        record.artifacts_path = artifact_path
        record.messages.append("Job completed successfully.")
    except Exception as exc:
        record.status = JobStatus.FAILED
        record.error_message = str(exc)
        record.messages.append(f"Job failed: {exc}")
    finally:
        record.updated_at = datetime.utcnow()

def _write_stub_artifact(job_id: str, job: JobConfig) -> Path:
    job_dir = ARTIFACT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = job_dir / "result.json"
    payload = {
        "job_id": job_id,
        "model": job.model.model_dump(exclude_none=True),
        "sources": [source.model_dump(exclude_none=True) for source in job.sources],
        "output": job.output.model_dump(exclude_none=True),
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }
    artifact_path.write_text(
        f"{payload}\n", encoding="utf-8"
    )
    return artifact_path

@app.post("/v1/jobs/{job_id}/upload-chunk")
async def upload_source_chunk(
    job_id: str,
    source_name: str,
    is_last: bool,
    file: UploadFile,
    sequence: Optional[int] = None,
):
    record = _JOB_STORE.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    chunk_dir = ARTIFACT_ROOT / job_id / "sources" / source_name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_name = f"{sequence or 0:06d}.chunk"
    chunk_path = chunk_dir / chunk_name
    chunk_path.write_bytes(await file.read())
    record.messages.append(f"Received chunk for source={source_name}, sequence={sequence} last={is_last}")
    if is_last:
        record.messages.append(f"Source {source_name} upload completed.")
    record.updated_at = datetime.utcnow()
    return {"job_id": job_id, "ack": True}

@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
    record = _JOB_STORE.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id, "status": record.status, "submitted_at": record.submitted_at,
        "updated_at": record.updated_at, "messages": record.messages, "error": record.error_message,
        "artifacts_path": str(record.artifacts_path) if record.artifacts_path else None
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
    if record.artifacts_path is None:
        raise HTTPException(status_code=404, detail="Result not available yet")
    return FileResponse(
        record.artifacts_path, media_type="application/json",
        filename=record.artifacts_path.name,
    )