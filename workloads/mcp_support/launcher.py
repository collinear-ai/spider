from __future__ import annotations

from dataclasses import dataclass
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Tuple

DEFAULT_ZENDESK_COMMAND = ["mcp-zendesk"]
DEFAULT_JIRA_COMMAND = ["atlassian-jira-mcp-server"]

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

def _zendesk_env() -> Dict[str, str]:
    env = {
        "ZENDESK_BASE_URL": os.environ.get("ZENDESK_BASE_URL", ""),
        "ZENDESK_EMAIL": os.environ.get("ZENDESK_EMAIL", ""),
        "ZENDESK_API_TOKEN": os.environ.get("ZENDESK_API_TOKEN", ""),
    }

    _require_env(env, ["ZENDESK_BASE_URL", "ZENDESK_EMAIL", "ZENDESK_API_TOKEN"], "Zendesk")
    return env

def _jira_env() -> Dict[str, str]:
    env = {
        "JIRA_BASE_URL": os.environ.get("JIRA_BASE_URL", ""),
        "JIRA_USERNAME": os.environ.get("JIRA_USERNAME", ""),
        "JIRA_API_TOKEN": os.environ.get("JIRA_API_TOKEN", ""),
    }

    default_project = os.environ.get("JIRA_DEFAULT_PROJECT")
    if default_project:
        env["JIRA_DEFAULT_PROJECT"] = default_project

    _require_env(env, ["JIRA_BASE_URL", "JIRA_USERNAME", "JIRA_API_TOKEN"], "Jira")
    return env

def start_mcp_support_server(
    *,
    name: str,
    port: Optional[int] = None,
    command: Optional[Sequence[str]] = None,
    args: Optional[Sequence[str]] = None,
) -> Tuple[MCPServerHandle, MCPServerSpec]:
    normalized = name.lower().strip()
    if normalized == "zendesk":
        env = _zendesk_env()
        default_command = DEFAULT_ZENDESK_COMMAND
    elif normalized == "jira":
        env = _jira_env()
        default_command = DEFAULT_JIRA_COMMAND
    else:
        raise ValueError(f"Unsupported MCP server name: {name}")
    
    spec = MCPServerSpec(
        name=normalized,
        command=list(command or default_command) + list(args or []),
        env=env,
        port=port,
    )
    handle = _start_server(spec)
    return handle, spec

def stop_mcp_support_servers(handles: Sequence[MCPServerHandle]) -> None:
    for handle in handles:
        handle.stop()