from __future__ import annotations

import os
import sys
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Tuple

DEFAULT_ZENDESK_COMMAND = ["mcp-zendesk"]
DEFAULT_JIRA_COMMAND = ["atlassian-jira-mcp-server"]
DEFAULT_PROXY_COMMAND = [sys.executable, "-m", "workloads.mcp_support.stdio_proxy"]
_REPO_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class MCPServerSpec:
    name: str
    command: List[str]
    env: Dict[str, str]
    cwd: Optional[Path] = None
    port: Optional[int] = None
    startup_timeout_s: float = 30.0

@dataclass
class MCPServerHandle:
    spec: MCPServerSpec
    proc: subprocess.Popen[str]

    def stop(self, timeout_s: float = 10.0) -> None:
        if self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()

def _wait_for_port(
    port: int,
    host: str = "127.0.0.1",
    timeout_s: float = 30.0,
) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {host}:{port}") from last_error

def _start_server(spec: MCPServerSpec) -> MCPServerHandle:
    env = os.environ.copy()
    env.update(spec.env)
    proc = subprocess.Popen(
        spec.command,
        cwd=str(spec.cwd) if spec.cwd else None,
        env=env,
        text=True,
    )
    if spec.port is not None:
        _wait_for_port(spec.port, timeout_s=spec.startup_timeout_s)
    return MCPServerHandle(spec=spec, proc=proc)

def _require_env(env: Dict[str, str], required: Sequence[str], name: str) -> None:
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise ValueError(f"{name} missing env vars: {', '.join(missing)}")

def start_mcp_remote_proxy(
    *,
    port: int, 
    host: str = "127.0.0.1",
    mcp_url_env: str,
    remote_url: str,
    proxy_command: Optional[Sequence[str]] = None,
    mcp_remote_command: Optional[Sequence[str]] = None,
) -> Tuple[MCPServerHandle, str]:
    proxy_command = list(proxy_command or DEFAULT_PROXY_COMMAND)
    mcp_cmd = list(
        mcp_remote_command
        or ["npx", "-y", "mcp-remote", remote_url]
    )
    command = proxy_command + [
        "--host", host,
        "--port", str(port),
        "--command",
        *mcp_cmd,
    ]
    spec = MCPServerSpec(
        name="mcp-remote-proxy",
        command=command,
        env={},
        cwd=_REPO_ROOT,
        port=port,
    )
    handle = _start_server(spec)
    mcp_url = f"http://{host}:{port}/mcp"
    os.environ[mcp_url_env] = mcp_url
    return handle, mcp_url

def stop_mcp_support_servers(handles: Sequence[MCPServerHandle]) -> None:
    for handle in handles:
        handle.stop()