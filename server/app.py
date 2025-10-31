from __future__ import annotations

import json, os
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4
from fastapi import BackgroundTasks, FastAPI, HTTPException, Header, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from spider.config import JobConfig
from .executor import JobExecutionError, run_generation_job
from .api_key_manager import APIKeyManager, ResolvedAPIKey

app = FastAPI(title="Spider Data Generation Service", version="0.1.0")

ARTIFACT_ROOT = Path("./.artifacts")
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_ALLOWLIST = {
  "demo-001" : "demo-test",
}

PLATFORM_BASE_URL = os.environ.get("PLATFORM_BASE_URL", "https://api.collinear.ai")
PLATFORM_API_KEY_LOOKUP_PATH = os.environ.get("PLATFORM_API_KEY_LOOKUP_PATH", "/api/v1/auth/identity")
API_KEY_LOOKUP_TIMEOUT_MS = int(os.environ.get("API_KEY_LOOKUP_TIMEOUT_MS", "2000"))
API_KEY_CACHE_MAX_SIZE = int(os.environ.get("API_KEY_CACHE_MAX_SIZE", "500"))
API_KEY_CACHE_TTL_VALID = int(os.environ.get("API_KEY_CACHE_TTL_VALID", "300"))

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
    remote_artifact: Optional[str] = None
    error_message: Optional[str] = None
    messages: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

_JOB_STORE: Dict[str, JobRecord] = {}

@app.on_event("startup")
async def startup_event():
    global api_key_manager
    api_key_manager = APIKeyManager(
        allowlist=DEFAULT_ALLOWLIST,
        cache_max_size=API_KEY_CACHE_MAX_SIZE,
        valid_ttl_seconds=API_KEY_CACHE_TTL_VALID,
        platform_base_url=PLATFORM_BASE_URL,
        platform_lookup_path=PLATFORM_API_KEY_LOOKUP_PATH,
        lookup_timeout_seconds=API_KEY_LOOKUP_TIMEOUT_MS / 1000.0,
    )

async def get_api_key(authorization: Optional[str] = Header(None, alias="Authorization")) -> ResolvedAPIKey:
    if api_key_manager is None:
        raise HTTPException(status_code=503, detail="API key validation unavailable")
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing", headers={"WWW-Authenticate": "Bearer"})
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "Bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header", headers={"WWW-Authenticate": "Bearer"})
    token = parts[1]
    return await api_key_manager.resolve(token)

@app.post("/v1/jobs")
async def create_job(job: JobConfig, background: BackgroundTasks, api_key: ResolvedAPIKey = Depends(get_api_key)):
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
        record.remote_artifact = result.remote_artifact
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
async def get_job(job_id: str, api_key: ResolvedAPIKey = Depends(get_api_key)):
    record = _JOB_STORE.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _serialize_job(job_id, record)

@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, api_key: ResolvedAPIKey = Depends(get_api_key)):
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
async def download_result(job_id: str, api_key: ResolvedAPIKey = Depends(get_api_key)):
    record = _JOB_STORE.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not record.artifacts_path:
        raise HTTPException(status_code=404, detail="Result not available yet")
    artifact_path = Path(record.artifacts_path)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found on disk")

    artifact_records = []
    with artifact_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                artifact_records.append(json.loads(line))
            except json.JSONDecodeError:
                artifact_records.append({"raw": line})

    payload = _serialize_job(job_id, record)
    payload["artifact_path"] = record.artifacts_path
    payload["artifact"] = artifact_records
    return JSONResponse(jsonable_encoder(payload))


def _serialize_job(job_id: str, record: JobRecord) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "status": record.status,
        "submitted_at": record.submitted_at,
        "updated_at": record.updated_at,
        "messages": record.messages,
        "error": record.error_message,
        "artifacts_path": record.artifacts_path,
        "remote_artifact": record.remote_artifact,
        "metrics": record.metrics,
    }