from __future__ import annotations

import json, os, traceback, logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from spider.config import JobConfig, OutputMode
from .executor import JobExecutionError, run_generation_job, _job_snapshot
from . import events

def _configure_executor_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_configure_executor_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Spider Data Generation Service", version="0.1.0")

ARTIFACT_ROOT = Path("./.artifacts")
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

EVENT_BUFFER_LIMIT = 200

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
    events: List[Dict[str, Any]] = Field(default_factory=list)
    events_truncated: bool = False
    next_event_offset: int = 0

_JOB_STORE: Dict[str, JobRecord] = {}
def _job_snapshot_path(job_id: str) -> Path:
    return ARTIFACT_ROOT / job_id / "job.json"

def _persist_job_record(job_id: str, record: JobRecord) -> None:
    snapshot_path = _job_snapshot_path(job_id)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = jsonable_encoder(record)
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

def _load_job_record(job_id: str) -> Optional[JobRecord]:
    snapshot_path = _job_snapshot_path(job_id)
    if not snapshot_path.exists():
        return None
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return JobRecord.model_validate(payload)
    except Exception as exc:
        logger.warning("Failed to load snapshot for job %s: %s", job_id, exc)
        return None

def _get_job_record(job_id: str) -> Optional[JobRecord]:
    record = _JOB_STORE.get(job_id)
    if record is not None:
        return record
    logger.info("_JOB_STORE is empty, loading record from disk for job %s", job_id)
    record = _load_job_record(job_id)
    if record is not None:
        _JOB_STORE[job_id] = record
    return record

def _store_job_event(job_id: str, event: Dict[str, Any]) -> None:
    record = _JOB_STORE.get(job_id)
    if record is None:
        return
    event_copy = dict(event)
    event_copy["offset"] = record.next_event_offset
    record.next_event_offset += 1
    record.events.append(event_copy)
    if len(record.events) > EVENT_BUFFER_LIMIT:
        drop = len(record.events) - EVENT_BUFFER_LIMIT
        record.events = record.events[drop:]
        record.events_truncated = True

events.configure_event_sink(_store_job_event)

@app.post("/v1/jobs")
async def create_job(job: JobConfig, background: BackgroundTasks):
    job_id = str(uuid4())
    now = datetime.utcnow()
    record = JobRecord(
        job=job, status=JobStatus.QUEUED, submitted_at=now, updated_at=now, messages=["Job accepted."]
    )
    _JOB_STORE[job_id] = record
    _persist_job_record(job_id, record)
    events.emit_for_job(job_id, "Job accepted.", code="job.accepted")
    background.add_task(_execute_job, job_id)
    return {"job_id": job_id, "status": record.status}

def _execute_job(job_id: str) -> None:
    record = _JOB_STORE.get(job_id)
    if record is None or record.status != JobStatus.QUEUED:
        return
    record.status = JobStatus.RUNNING
    record.updated_at = datetime.utcnow()
    record.messages.append("Job started.")
    token = events.bind_job(job_id)
    events.emit("Job started.", code="job.started")
    try:
        result = run_generation_job(job_id, record.job, workspace=ARTIFACT_ROOT / job_id)
        record.artifacts_path = str(result.artifacts_path)
        record.remote_artifact = result.remote_artifact
        record.metrics = result.metrics
        if result.messages:
            record.messages.extend(result.messages)
        record.status = JobStatus.COMPLETED
        events.emit("Job completed successfully.", code="job.completed")
    except JobExecutionError as exc:
        logger.exception("Job %s failed.", job_id)
        record.status = JobStatus.FAILED
        record.error_message = str(exc)
        record.messages.append(f"Job failed: {exc}")
        events.emit(
            "Job failed.",
            level="error",
            code="job.failed",
            data={"error": str(exc)}
        )
    except Exception as exc:
        logger.exception("Job %s crashed with unexpected error.", job_id)
        record.status = JobStatus.FAILED
        crash_tb = traceback.format_exc().strip()
        record.error_message = crash_tb
        record.messages.append(f"Job crashed: \n{crash_tb}")
        events.emit(
            "Job crashed.",
            level="error",
            code="job.crashed",
            data={"error": crash_tb.splitlines()[-1]}
        )
    finally:
        record.updated_at = datetime.utcnow()
        _persist_job_record(job_id, record)
        events.reset_job(token)

@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str, since: Optional[int] = Query(None, ge=0)):
    record = _get_job_record(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _serialize_job(job_id, record, since=since)

@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    record = _get_job_record(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if record.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        return {"job_id": job_id, "status": record.status}
    record.status = JobStatus.CANCELLED
    record.updated_at = datetime.utcnow()
    record.messages.append("Cancellation requested.")
    events.emit_for_job(job_id, "Cancellation requested.", code="job.cancel_requested")
    _persist_job_record(job_id, record)
    return {"job_id": job_id, "status": record.status}

@app.get("/v1/jobs/{job_id}/result")
async def download_result(job_id: str):
    record = _get_job_record(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not record.artifacts_path:
        raise HTTPException(status_code=404, detail="Result not available yet")
    artifact_path = Path(record.artifacts_path)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found on disk")

    include_records = record.job.output.mode != OutputMode.HF_UPLOAD
    artifact_records = []
    if include_records:
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
    if include_records:
        payload["artifact"] = artifact_records
    return JSONResponse(jsonable_encoder(payload))


def _serialize_job(job_id: str, record: JobRecord, *, since: Optional[int] = None) -> Dict[str, Any]:
    events_payload, truncated = _render_events(record, since)
    events_start = record.events[0]["offset"] if record.events else record.next_event_offset
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
        "job": _job_snapshot(record.job),
        "events": events_payload,
        "events_next_offset": record.next_event_offset,
        "events_start_offset": events_start,
        "events_truncated": truncated
    }

def _render_events(record: JobRecord, since: Optional[int]) -> Tuple[List[Dict[str, Any]], bool]:
    events_list = record.events
    truncated = record.events_truncated
    if since is not None:
        events_list = [evt for evt in events_list if evt.get("offset", 0) >= since]
        if record.events and record.events[0].get("offset", 0) > since:
            truncated = True
    return [dict(evt) for evt in events_list], truncated
