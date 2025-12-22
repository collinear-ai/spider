from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, List, Tuple, Iterable
import httpx, time, sys, json, shutil, textwrap, warnings

from .config import AppConfig
from .processor_bundle import bundle_processor_source, ProcessorBundlingError

class SpiderClient:
    def __init__(
        self, 
        config: AppConfig, 
        *, 
        client: Optional[httpx.Client] = None,
        env: Optional[Dict[str, str]] = None,
        pre_processor: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
        pre_processor_kwargs: Optional[Dict[str, Any]] = None,
        post_processor: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
        post_processor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._config = config
        self._client: Optional[httpx.Client] = client
        self._owns_client = client is None

        self._pre_processor = pre_processor
        self._pre_processor_kwargs = dict(pre_processor_kwargs or {})
        self._post_processor = post_processor
        self._post_processor_kwargs = dict(post_processor_kwargs or {})

        self._tool_payloads: Dict[str, Dict[str, Any]] = {}
        raw_env = dict(env or {})
        self._job_env = {key: value for key, value in raw_env.items() if value is not None}

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

        for key, value in self._job_env.items():
            if not isinstance(value, str):
                raise ValueError(f"Environment variable `{key}` must be a string, got {type(value).__name__}.")

        if self._pre_processor is not None:
            payload["pre_processor"] = self._serialize_processor(
                kind="pre_processor",
                func=self._pre_processor,
                kwargs=self._pre_processor_kwargs,
            )

        if self._post_processor is not None:
            payload["post_processor"] = self._serialize_processor(
                kind="post_processor",
                func=self._post_processor,
                kwargs=self._post_processor_kwargs,
            )

        if self._tool_payloads:
            payload_tools = list(payload.get("tools") or [])
            payload_tools.extend(self._tool_payloads.values())
            payload["tools"] = payload_tools

        request_body = {"job": payload, "env": self._job_env}
        response = self._ensure_client().post("/v1/jobs", json=request_body)
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
            items = []
            for key in data.keys():
                value = data[key]
                serialized = json.dumps(value, ensure_ascii=True)
                items.append(f"{key}={serialized}")
            if items:
                line = f"{line} | " + " | ".join(items)
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

    def _serialize_processor(
        self,
        kind: str,
        func: Callable[[Dict[str, Any]], Any],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        if func is None:
            raise ValueError(f"{kind} callable is not set")
        name = getattr(func, "__name__", None)
        if not name:
            raise ValueError(f"{kind} callable must have a __name__ attribute")
        try:
            source = bundle_processor_source(func)
        except ProcessorBundlingError as exc:
            raise ValueError(str(exc)) from exc
        self._print_bundle_preview(kind, name, source)
        return {
            "name": name,
            "source": source,
            "kwargs": dict(kwargs)
        }

    def add_tool(
        self,
        *,
        description: str,
        json_schema: Dict[str, Any],
        func: Callable[..., Any],
        kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        if func is None:
            raise ValueError("Tool callable is required")
        func_name = getattr(func, "__name__", None)
        if not func_name:
            raise ValueError("Tool callable must have a __name__ attribute")
        tool_name = name or func_name
        if tool_name != func_name:
            raise ValueError(f"Tool name `{tool_name}` must match the callable name `{func_name}` so that the server can load it.")
        if not description or not description.strip():
            raise ValueError("Tool description is required.")

        schema = self._prepare_tool_schema(json_schema)
        try:
            source = bundle_processor_source(func)
        except ProcessorBundlingError as exc:
            raise ValueError(str(exc)) from exc

        self._print_bundle_preview("tool", tool_name, source)
        self._tool_payloads[tool_name] = {
            "name": tool_name,
            "description": description.strip(),
            "json_schema": schema,
            "source": source,
            "kwargs": dict(kwargs or {})
        }

    def _prepare_tool_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(schema, dict):
            raise ValueError("Tool JSON schema must be a dictionary.")
        try:
            normalized = json.loads(json.dumps(schema))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Tool JSON schema must be JSON-serializable: {exc}") from exc
        schema_type = normalized.get("type")
        if schema_type is None:
            normalized["type"] = "object"
        elif schema_type != "object":
            raise ValueError(f"Tool JSON schema must be 'object'.")
        properties = normalized.get("properties")
        if properties is None:
            normalized["properties"] = {}
        elif not isinstance(properties, dict):
            raise ValueError("Tool JSON schema 'properties' must be a dictionary.")
        return normalized
    
    def _print_bundle_preview(self, kind: str, name: str, source: str) -> None:
        try:
            width = shutil.get_terminal_size().columns
        except OSError:
            width = 80
        border = "-" * min(width, 80)
        header = f"[Spider client] Bundled {kind} `{name}`:"
        print(f"\033[34m{border}\033[0m", file=sys.stdout)
        print(f"\033[36m{header}\033[0m", file=sys.stdout)
        print(textwrap.indent(source.rstrip(), "  "), file=sys.stdout)
        print(f"\033[34m{border}\033[0m", file=sys.stdout)