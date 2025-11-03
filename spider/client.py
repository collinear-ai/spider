from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, List, Tuple
import httpx, time, inspect, textwrap, sys, json

from .config import AppConfig

class SpiderClient:
    def __init__(
        self, config: AppConfig, *, client: Optional[httpx.Client] = None,
        processor: Optional[callable[[Iterable[Dict[str, Any]]], Iterable[Dict[str, Any]]]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._config = config
        self._client: Optional[httpx.Client] = client
        self._owns_client = client is None
        self._processor = processor
        self._processor_kwargs = dict(processor_kwargs or {})

    def __enter__(self) -> "SpiderClient":
        self._ensure_client()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @classmethod
    def from_config(cls, path: str, overrides: Optional[Dict[str, Any]] = None):
        config = AppConfig.load(path, overrides=overrides)
        return cls(config=config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        config = AppConfig.model_validate(config_dict)
        return cls(config=config)

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
        self._client = None

    def submit_job(self, *, job_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = self._config.job.model_dump(exclude_none=True)
        if job_overrides:
            payload = self._deep_merge(payload, job_overrides)

        if self._processor is not None:
            payload["processor"] = self._serialize_processor()

        response = self._ensure_client().post("/v1/jobs", json=payload)
        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_id: str, *, since: Optional[int] = None) -> Dict[str, Any]:
        params = {"since": since} if since is not None else None
        response = self._ensure_client().get(f"/v1/jobs/{job_id}", params=params)
        response.raise_for_status()
        return response.json()

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        response = self._ensure_client().post(f"/v1/jobs/{job_id}/cancel")
        response.raise_for_status()
        return response.json()

    def download_result(
        self, job_id: str, *, destination: Optional[Union[str, Path]] = None
    ) -> Union[bytes, Path]:
        client = self._ensure_client()
        stream = client.stream("GET", f"/v1/jobs/{job_id}/result")
        with stream as response:
            response.raise_for_status()
            if destination is None:
                buffer = bytearray()
                for block in response.iter_bytes():
                    buffer.extend(block)
                return bytes(buffer)

            path = Path(destination).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                for block in response.iter_bytes():
                    handle.write(block)
            return path

    def poll_job(
        self, 
        job_id: str, 
        *, 
        interval: float = 5.0, 
        timeout: Optional[float] = None,
        on_update: Optional[Callable[[Dict[str, Any]], None]] = None, 
        wait_for_completion: bool = False,
        stream_events: bool = True,
    ) -> Dict[str, Any]:
        terminal_states = {"completed", "failed", "cancelled"}
        deadline = (time.monotonic() + timeout) if timeout is not None else None

        follow_events = (wait_for_completion and stream_events)
        event_offset = 0 if follow_events else None
        truncated_notified = False

        status = self.get_job_status(job_id, since=event_offset)
        if follow_events:
            event_offset, truncated_notified = self._handle_events(
                status, stream_events, event_offset, truncated_notified
            )
        if on_update:
            on_update(status)

        state = str(status.get("status", "")).lower()
        if not wait_for_completion or state in terminal_states:
            return status
            
        while state not in terminal_states:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

            time.sleep(max(interval, 0.1))
            status = self.get_job_status(job_id, since=event_offset)
            if follow_events:
                event_offset, truncated_notified = self._handle_events(
                    status, stream_events, event_offset, truncated_notified
                )
            if on_update:
                on_update(status)    
            state = str(status.get("status", "")).lower()   

        return status 

    def _handle_events(
        self,
        status: Dict[str, Any],
        stream_events: bool,
        current_offset: Optional[int],
        truncated_notified: bool,
    ) -> Tuple[Optional[int], bool]:
        events = status.get("events")
        if isinstance(events, list):
            for event in events:
                self._dispatch_event(dict(event), stream_events)
        truncated = bool(status.get("events_truncated"))
        if truncated and not truncated_notified:
            notice = {
                "level": "WARNING",
                "message": "Some earlier job events are no longer available.",
                "code": "events.truncated",
            }
            self._dispatch_event(notice, stream_events)
            truncated_notified = True
        next_offset = status.get("events_next_offset")
        if isinstance(next_offset, int):
            current_offset = next_offset
        elif isinstance(events, list) and events:
            last_offset = events[-1].get("offset")
            if isinstance(last_offset, int):
                current_offset = last_offset + 1
        return current_offset, truncated_notified

    def _dispatch_event(
        self,
        event: Dict[str, Any],
        stream_events: bool
    ) -> None:
        if not stream_events:
            return
        timestamp = event.get("timestamp")
        level = event.get("level", "INFO")
        message = event.get("message", "")
        data = event.get("data")
        if timestamp:
            line = f"{timestamp} [{level}] {message}"
        else:
            line = f"[{level}] {message}"
        if data:
            line = f"{line} {json.dumps(data)}"
        print(line, file=sys.stdout, flush=True)

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            server = self._config.server
            headers = self._build_headers()
            self._client = httpx.Client(
                base_url=str(server.base_url),
                timeout=server.request_timeout,
                headers=headers,
                verify=server.verify_tls,
            )
            self._owns_client = True
        return self._client

    def _build_headers(self) -> Dict[str, str]:
        headers = dict(self._config.server.headers)
        if self._config.server.api_key:
            headers.setdefault("Authorization", f"Bearer {self._config.server.api_key}")
        headers.setdefault("Accept", "application/json")
        return headers

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base)
        for key, value in overrides.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = SpiderClient._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _serialize_processor(self) -> Dict[str, Any]:
        if self._processor is None:
            raise ValueError("Processor callable is not set")
        name = getattr(self._processor, "__name__", None)
        if not name:
            raise ValueError("Processor callable must have a __name__ attribute")
        try:
            source = inspect.getsource(self._processor)
        except (OSError, TypeError) as exc:
            raise ValueError("Unable to retrieve source for processor callable") from exc
        source = textwrap.dedent(source)
        return {
            "name": name,
            "source": source,
            "kwargs": dict(self._processor_kwargs)
        }