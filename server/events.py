from __future__ import annotations

import contextvars
from datetime import datetime
from typing import Any, Callable, Dict, Optional

_EVENT_SINK: Optional[Callable[[str, Dict[str, Any]], None]] = None
_JOB_CONTEXT = contextvars.ContextVar("spider_job_id", default=None)

def configure_event_sink(sink: Callable[[str, Dict[str, Any]], None]) -> None:
    global _EVENT_SINK
    _EVENT_SINK = sink

def bind_job(job_id: str):
    return _JOB_CONTEXT.set(job_id)

def reset_job(token) -> None:
    _JOB_CONTEXT.reset(token)

def emit(
    message: str,
    *,
    level: str = "info",
    code: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None
) -> None:
    job_id = _JOB_CONTEXT.get()
    if not job_id or _EVENT_SINK is None:
        return
    _dispatch(job_id, message, level=level, code=code, data=data)

def emit_for_job(
    job_id: str,
    message: str,
    *,
    level: str = "info",
    code: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None
) -> None:
    if not job_id or _EVENT_SINK is None:
        return
    _dispatch(job_id, message, level=level, code=code, data=data)

def _dispatch(
    job_id: str,
    message: str,
    *,
    level: str,
    code: Optional[str],
    data: Optional[Dict[str, Any]],
) -> None:
    level_name = (level or "info").upper()
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "message": message,
        "level": level_name,
    }
    if code:
        payload["code"] = code
    if data:
        payload["data"] = data
    _EVENT_SINK(job_id, payload)