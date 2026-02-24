import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
from huggingface_hub import HfApi

from workloads.mcp_support.public_readonly_servers import (
    ensure_server_runtime_ready,
    headers_for_server,
    list_public_readonly_mcp_servers,
    resolve_stdio_cwd,
    server_requires_local_proxy,
)
from workloads.mcp_support.tool_schemas import tool_config_from_server


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parent
PROMPT_TEMPLATE_PATH = BASE_DIR / "prompts" / "genq_from_tools_single_server_multi_tools.md"
ENV_PATH = Path(__file__).resolve().parent / ".env"
OUTPUT_PATH = BASE_DIR / "artifacts" / "model_prompts.jsonl"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, required=True)
    parser.add_argument("--dataset-id", type=str, default=None)
    return parser.parse_args()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


def format_tools_for_prompt(tools: List[object]) -> str:
    blocks: List[str] = []
    for t in tools:
        schema = json.dumps(t.json_schema or {}, ensure_ascii=True, separators=(",", ":"))
        desc = (t.description or "").strip()
        blocks.append(f"- {t.name}\n  Description: {desc}\n  Signature: {schema}")
    return "\n".join(blocks)


def wait_for_port(host: str, port: int, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    last_exc: Optional[Exception] = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_exc = exc
            time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for {host}:{port}") from last_exc


def start_stdio_proxy(server: object) -> Tuple[subprocess.Popen[str], str]:
    if not server.stdio_command:
        raise RuntimeError("missing stdio_command")
    parsed = urlparse(server.mcp_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if not port:
        raise RuntimeError(f"missing port in mcp_url: {server.mcp_url}")
    stdio_cmd = list(server.stdio_command)
    if server.key == "scientific-computation-mcp":
        stdio_cmd = [x for x in stdio_cmd if x not in {"--key", "<SMITHERY_API_KEY>"}]
    cmd = [
        sys.executable,
        "-m",
        "workloads.mcp_support.stdio_proxy",
        "--host",
        host,
        "--port",
        str(port),
        "--command-cwd",
        str(resolve_stdio_cwd(server, REPO_ROOT) or REPO_ROOT),
        "--command",
        *stdio_cmd,
    ]
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=os.environ.copy(), text=True)
    wait_for_port(host, port)
    return proc, f"http://{host}:{port}/mcp/"


def fetch_tool_cache() -> List[Tuple[object, List[object]]]:
    servers = list_public_readonly_mcp_servers()
    cache: List[Tuple[object, List[object]]] = []
    original_headers = os.environ.get("MCP_HEADERS_JSON")
    for server in servers:
        proc: Optional[subprocess.Popen[str]] = None
        try:
            ensure_server_runtime_ready(server, REPO_ROOT)
            headers = headers_for_server(server)
            if headers:
                os.environ["MCP_HEADERS_JSON"] = json.dumps(headers)
            else:
                os.environ.pop("MCP_HEADERS_JSON", None)
            url = server.mcp_url
            if server_requires_local_proxy(server):
                proc, url = start_stdio_proxy(server)
                tools = tool_config_from_server(url, mcp_url_env=server.mcp_url_env)
            else:
                tools = tool_config_from_server(url, mcp_url_env=server.mcp_url_env)
            if len(tools) >= 2:
                cache.append((server, tools))
                print(f"[ok] {server.key}: {len(tools)} tools")
            else:
                print(f"[skip] {server.key}: fewer than 2 tools")
        except Exception as exc:
            print(f"[skip] {server.key}: {exc}")
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
    if original_headers is not None:
        os.environ["MCP_HEADERS_JSON"] = original_headers
    else:
        os.environ.pop("MCP_HEADERS_JSON", None)
    return cache


def build_prompt(template: str, server: object, tools: List[object], num_turns: int) -> str:
    tool_count = len(tools)
    return template.format(
        NUM_TOOLS=tool_count,
        NUM_TURNS=num_turns,
        MCP_SERVER_NAME=server.display_name,
        MCP_SERVER_DESCRIPTION=server.description,
        TOOL_LIST=format_tools_for_prompt(tools),
    )


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0:
        raise SystemExit("--num-examples must be > 0")
    if not PROMPT_TEMPLATE_PATH.exists():
        raise SystemExit(f"Prompt template not found: {PROMPT_TEMPLATE_PATH}")

    load_env(ENV_PATH)
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    cache = fetch_tool_cache()
    if not cache:
        raise SystemExit("No reachable remote MCP servers with >=2 tools.")

    rows = []
    for _ in range(args.num_examples):
        server, all_tools = random.choice(cache)
        desired = random.randint(2, 3)
        k = min(desired, len(all_tools))
        sampled = random.sample(all_tools, k)
        num_turns = random.randint(1, 2)
        rows.append(
            {
                "prompt": build_prompt(template, server, sampled, num_turns),
                "server": asdict(server),
                "tools": [t.name for t in sampled],
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")
    if args.dataset_id:
        parquet_path = OUTPUT_PATH.with_suffix(".parquet")
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise SystemExit("HF_TOKEN is required when --dataset-id is set.")
        api = HfApi(token=token)
        api.create_repo(repo_id=args.dataset_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=parquet_path.name,
            repo_id=args.dataset_id,
            repo_type="dataset",
        )
        print(f"Uploaded {parquet_path} to hf://datasets/{args.dataset_id}")


if __name__ == "__main__":
    main()
