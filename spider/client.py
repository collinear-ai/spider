from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import httpx

from .config import AppConfig

class SpiderClient:
    def __init__(self, config: AppConfig, *, client: Optional[httpx.Client] = None):
        self._config = config
        self._client: Optional[httpx.Client] = client
        self._owns_client = client is None

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

        response = self._ensure_client().post("/v1/jobs", json=payload)
        response.raise_for_status()
        return response.json()

    def upload_source_chunk(
        self, job_id: str, source_name: str, chunk: bytes,
        *, sequence: Optional[int] = None, is_last: bool = False,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        data: Dict[str, str] = {
            "source_name": source_name,
            "is_last": "true" if is_last else "false",
        }
        if sequence is not None:
            data["sequence"] = str(sequence)

        files = {
            "file": (filename or f"{source_name}.chunk", chunk),
        }
        response = self._ensure_client().post(
            f"/v1/jobs/{job_id}/upload-chunk",
            data=data,
            files=files,
        )
        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        response = self._ensure_client().get(f"/v1/jobs/{job_id}")
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