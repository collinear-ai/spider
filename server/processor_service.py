from __future__ import annotations

import json
import re
import socket
import subprocess
import threading
import time
import textwrap
import logging
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, Optional, List

import requests
from requests.adapters import HTTPAdapter

from spider.config import ProcessorConfig

logger = logging.getLogger(__name__)


class ProcessorServiceError(Exception):
    """Raised when the processor service fails to start or respond."""


@dataclass
class _ProcessorSpec:
    name: str
    kwargs: Dict[str, Any]
    source: str
    requirements: List[str]


class DockerProcessorService:
    """Launches a per-job FastAPI processor service inside a Docker container."""

    def __init__(
        self,
        job_id: str,
        workspace: Path,
        *,
        pre_processor: Optional[ProcessorConfig] = None,
        post_processor: Optional[ProcessorConfig] = None,
        request_timeout: float = 15.0,
        max_workers: int = 2,
    ) -> None:
        self._job_id = job_id
        self._workspace = workspace
        self._pre = self._to_spec(pre_processor)
        self._post = self._to_spec(post_processor)
        self._request_timeout = request_timeout
        self._worker_count = max(1, int(max_workers))

        self._context_dir = self._workspace / "processor_service"
        self._image_tag = self._build_image_tag(job_id)
        self._container_name = f"spider-processor-{self._sanitize(job_id)}"
        self._port: Optional[int] = None
        self._started = False
        self._thread_local = threading.local()
        self._session_lock = threading.Lock()
        self._sessions: list[requests.Session] = []
        self._session_pool_size = max(4, self._worker_count * 4)

    @staticmethod
    def _sanitize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "job"

    @classmethod
    def _build_image_tag(cls, job_id: str) -> str:
        safe = cls._sanitize(job_id)
        return f"spider-processor-{safe}:latest"

    @staticmethod
    def _to_spec(config: Optional[ProcessorConfig]) -> Optional[_ProcessorSpec]:
        if not config:
            return None
        return _ProcessorSpec(
            name=config.name,
            kwargs=dict(config.kwargs or {}),
            source=config.source,
            requirements=list(config.requirements or []),
        )

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise ProcessorServiceError("Processor service has not been started")
        return f"http://127.0.0.1:{self._port}"

    def is_running(self) -> bool:
        return self._started

    def start(self) -> None:
        if not self._pre and not self._post:
            return
        self._context_dir.mkdir(parents=True, exist_ok=True)
        self._write_context()
        self._build_image()
        self._run_container()
        self._wait_until_ready()
        self._started = True

    def stop(self) -> None:
        try:
            if self._container_name:
                subprocess.run(
                    ["docker", "rm", "-f", self._container_name],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
        finally:
            self._close_sessions()
            self._started = False
            self.cleanup_all()

    def invoke_preprocess(self, record: Dict[str, Any]) -> Any:
        if not self._pre:
            raise ProcessorServiceError("Pre-processor not configured")
        return self._invoke("preprocess", record)

    def invoke_postprocess(self, record: Dict[str, Any]) -> Any:
        if not self._post:
            raise ProcessorServiceError("Post-processor not configured")
        return self._invoke("postprocess", record)

    # Internal helpers -------------------------------------------------

    def _write_context(self) -> None:
        self._write_user_code()
        self._write_requirements()
        self._write_app_module()
        self._write_dockerfile()

    def _write_user_code(self) -> None:
        parts = []
        if self._pre:
            parts.append(self._pre.source.rstrip())
        if self._post:
            parts.append(self._post.source.rstrip())
        if not parts:
            parts.append("# no user processors provided\n")
        payload = "\n\n".join(parts) + "\n"
        (self._context_dir / "user_code.py").write_text(payload, encoding="utf-8")

    def _write_requirements(self) -> None:
        base = [
            "fastapi",
            "uvicorn[standard]",
            "pydantic",
        ]
        extras: List[str] = []
        for spec in (self._pre, self._post):
            if spec and spec.requirements:
                extras.extend(spec.requirements)

        ordered: List[str] = []
        for item in base + extras:
            normalized = item.strip()
            if not normalized or normalized in ordered:
                continue
            ordered.append(normalized)

        if not ordered:
            ordered = base

        payload = "\n".join(ordered) + "\n"
        (self._context_dir / "requirements.txt").write_text(payload, encoding="utf-8")

    def _write_app_module(self) -> None:
        pre_name = self._pre.name if self._pre else None
        post_name = self._post.name if self._post else None
        try:
            pre_kwargs = json.dumps(self._pre.kwargs if self._pre else {})
            post_kwargs = json.dumps(self._post.kwargs if self._post else {})
        except TypeError as exc:
            raise ProcessorServiceError(f"Processor kwargs must be JSON serializable: {exc}") from exc

        template = Template(
            textwrap.dedent(
                """
                from __future__ import annotations

                import importlib.util
                from pathlib import Path
                from typing import Any, Dict

                from fastapi import FastAPI, HTTPException
                from fastapi.encoders import jsonable_encoder
                from pydantic import BaseModel

                MODULE_PATH = Path(__file__).parent / "user_code.py"
                module_spec = importlib.util.spec_from_file_location("user_code", MODULE_PATH)
                if module_spec is None or module_spec.loader is None:
                    raise RuntimeError("Failed to load user processors")
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)

                PREPROCESSOR_NAME = $pre_name
                POSTPROCESSOR_NAME = $post_name
                PREPROCESSOR = getattr(module, PREPROCESSOR_NAME, None) if PREPROCESSOR_NAME else None
                POSTPROCESSOR = getattr(module, POSTPROCESSOR_NAME, None) if POSTPROCESSOR_NAME else None
                PRE_KWARGS = $pre_kwargs
                POST_KWARGS = $post_kwargs

                if PREPROCESSOR_NAME and not callable(PREPROCESSOR):
                    raise RuntimeError(f"Preprocessor '{PREPROCESSOR_NAME}' is not callable")
                if POSTPROCESSOR_NAME and not callable(POSTPROCESSOR):
                    raise RuntimeError(f"Postprocessor '{POSTPROCESSOR_NAME}' is not callable")

                app = FastAPI(title="Spider Processor Service")

                class ProcessorRequest(BaseModel):
                    record: Dict[str, Any]

                @app.get("/healthz")
                def healthz() -> Dict[str, str]:
                    return {"status": "ok"}

                @app.post("/preprocess")
                def preprocess(req: ProcessorRequest):
                    if PREPROCESSOR is None:
                        raise HTTPException(status_code=404, detail="Preprocessor not configured")
                    try:
                        result = PREPROCESSOR(dict(req.record), **PRE_KWARGS)
                    except Exception as exc:  # noqa: BLE001
                        raise HTTPException(status_code=500, detail=str(exc)) from exc
                    return {"result": jsonable_encoder(result)}

                @app.post("/postprocess")
                def postprocess(req: ProcessorRequest):
                    if POSTPROCESSOR is None:
                        raise HTTPException(status_code=404, detail="Postprocessor not configured")
                    try:
                        result = POSTPROCESSOR(dict(req.record), **POST_KWARGS)
                    except Exception as exc:  # noqa: BLE001
                        raise HTTPException(status_code=500, detail=str(exc)) from exc
                    return {"result": jsonable_encoder(result)}
                """
            ).strip()
            + "\n"
        )
        app_source = template.substitute(
            pre_name=repr(pre_name),
            post_name=repr(post_name),
            pre_kwargs=pre_kwargs,
            post_kwargs=post_kwargs,
        )
        (self._context_dir / "app.py").write_text(app_source, encoding="utf-8")

    def _write_dockerfile(self) -> None:
        worker_args = ""
        if self._worker_count > 1:
            worker_args = f', "--workers", "{self._worker_count}"'
        dockerfile = f"""
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"{worker_args}]
"""
        (self._context_dir / "Dockerfile").write_text(dockerfile.strip() + "\n", encoding="utf-8")

    def _build_image(self) -> None:
        result = subprocess.run(
            ["docker", "build", "-t", self._image_tag, "."],
            cwd=self._context_dir,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise ProcessorServiceError(
                "Failed to build processor Docker image: " f"{result.stderr.strip()}"
            )

    def _run_container(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            self._port = sock.getsockname()[1]
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self._container_name,
                "-p",
                f"{self._port}:8080",
                self._image_tag,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise ProcessorServiceError(
                "Failed to start processor container: " f"{result.stderr.strip()}"
            )

    def _wait_until_ready(self) -> None:
        if self._port is None:
            raise ProcessorServiceError("Processor container port not assigned")
        deadline = time.time() + 60.0
        url = f"{self.base_url}/healthz"
        session = self._get_session()
        while time.time() < deadline:
            try:
                resp = session.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise ProcessorServiceError("Processor service did not become ready in time")

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is not None:
            return session
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=self._session_pool_size,
            pool_maxsize=self._session_pool_size,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        with self._session_lock:
            self._sessions.append(session)
        self._thread_local.session = session
        return session

    def _close_sessions(self) -> None:
        with self._session_lock:
            for session in self._sessions:
                try:
                    session.close()
                except Exception:
                    pass
            self._sessions.clear()
        self._thread_local = threading.local()

    @staticmethod
    def cleanup_all() -> None:
        try:
            ps = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-aq",
                    "--filter",
                    "name=spider-processor-",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            logger.debug("Docker binary not available; skipping processor cleanup")
            return

        if ps.returncode != 0:
            logger.warning("Failed to list processor containers: %s", ps.stderr.strip())
            return

        container_ids = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
        for container_id in container_ids:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

    def _invoke(self, endpoint: str, record: Dict[str, Any]) -> Any:
        if not self._started or self._port is None:
            raise ProcessorServiceError("Processor service is not running")
        url = f"{self.base_url}/{endpoint}"
        session = self._get_session()
        try:
            response = session.post(
                url,
                json={"record": record},
                timeout=self._request_timeout,
            )
        except requests.RequestException as exc:
            raise ProcessorServiceError(f"Processor request failed: {exc}") from exc
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            detail = response.text.strip() or response.reason
            raise ProcessorServiceError(
                f"Processor endpoint '{endpoint}' returned {response.status_code}: {detail}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProcessorServiceError("Processor response was not valid JSON") from exc
        return payload.get("result")