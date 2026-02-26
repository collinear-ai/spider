import argparse
import json
import os
import socket
import subprocess
import random
import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import anyio
import httpx
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from tqdm.auto import tqdm

from spider.config import JobConfig, ModelConfig
from server.backends.factory import create_backend
from server.executor import _initial_chat_history, _run_tool_turn
from workloads.mcp_support.public_readonly_servers import (
    ensure_server_runtime_ready,
    get_public_readonly_mcp_server,
    headers_for_server,
    resolve_stdio_cwd,
    server_requires_local_proxy,
)

MAX_TOOL_CONTENT_CHARS = 40000
MAX_TOOL_CONTENT_TOTAL_CHARS = 60000
CHECKPOINT_EVERY = 2000

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
ARTIFACT_DIR = BASE_DIR / "artifacts"
REPO_ROOT = Path(__file__).resolve().parents[2]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-in", required=True)
    parser.add_argument("--dataset-out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=1024)
    parser.add_argument("--row-timeout-sec", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()

def _truncate_and_cap_tool_message(history):
    total = 0
    for msg in reversed(history):
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if len(content) > MAX_TOOL_CONTENT_CHARS:
                content = content[:MAX_TOOL_CONTENT_CHARS]
            msg["content"] = content
            total += len(content)
            if total > MAX_TOOL_CONTENT_TOTAL_CHARS:
                return True
    return False

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
        os.environ[key.strip()] = value.strip().strip("'").strip('"')


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


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def start_stdio_proxy_if_needed(server: Any) -> tuple[Optional[subprocess.Popen[str]], str]:
    url = server.mcp_url
    if not server_requires_local_proxy(server):
        return None, url
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if not port:
        raise RuntimeError(f"missing port in mcp_url: {url}")
    if port_open(host, port):
        return None, f"http://{host}:{port}/mcp/"
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
        *list(server.stdio_command),
    ]
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=os.environ.copy(), text=True)
    wait_for_port(host, port)
    return proc, f"http://{host}:{port}/mcp/"


def list_tools_from_server(server_url: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    async def _run() -> List[Dict[str, Any]]:
        async with httpx.AsyncClient(headers=headers or None, timeout=60.0) as http_client:
            async with streamable_http_client(server_url, http_client=http_client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in result.tools]
    return anyio.run(_run)

def call_mcp_tool(server_url: str, headers: Dict[str, str], tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    async def _run() -> Dict[str, Any]:
        max_attempts = 5
        base_delay = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient(headers=headers or None, timeout=120.0) as http_client:
                    async with streamable_http_client(server_url, http_client=http_client) as (read_stream, write_stream, _):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            result = await session.call_tool(tool_name, arguments)
                            payload = result.model_dump()
                            return payload
            except Exception as exc:
                if attempt >= max_attempts:
                    raise
                sleep_s = base_delay * (2 ** (attempt - 1)) * (0.5 + random.random())
                await asyncio.sleep(sleep_s)
    return anyio.run(_run)


def parse_server_key(row: Dict[str, Any]) -> str:
    server = row.get("server")
    if isinstance(server, str):
        try:
            server = json.loads(server)
        except json.JSONDecodeError:
            server = {"key": server}
    if isinstance(server, dict):
        key = server.get("key")
        if isinstance(key, str) and key:
            return key
    raise ValueError("row missing server key")


def parse_tools(row: Dict[str, Any]) -> List[str]:
    tools = row.get("tools")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            tools = [tools]
    if not isinstance(tools, list):
        return []
    return [str(t) for t in tools if str(t).strip()]


def parse_questions(row: Dict[str, Any]) -> List[str]:
    questions = row.get("generated_questions")
    if isinstance(questions, str):
        try:
            questions = json.loads(questions)
        except json.JSONDecodeError:
            questions = [questions]
    if not isinstance(questions, list):
        return []
    return [str(q).strip() for q in questions if str(q).strip()]


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("server", "tools", "generated_questions"):
        value = row.get(key)
        if isinstance(value, str):
            try:
                row[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    return row


def has_tool_turn(history: Any) -> bool:
    if not isinstance(history, list):
        return False
    for msg in history:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            return True
    return False


def main() -> None:
    args = parse_args()
    load_env(ENV_PATH)

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise SystemExit("HF_TOKEN is required.")
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise SystemExit("OPENROUTER_API_KEY is required.")

    ds_kwargs: Dict[str, Any] = {"path": args.dataset_in, "split": args.split}
    if args.config_name:
        ds_kwargs["name"] = args.config_name
    dataset = load_dataset(**ds_kwargs)
    rows = [normalize_row(dict(x)) for x in dataset]
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    if not rows:
        raise SystemExit("No rows found.")

    server_keys = sorted({parse_server_key(row) for row in rows})
    server_ctx: Dict[str, Dict[str, Any]] = {}
    procs: List[subprocess.Popen[str]] = []
    for key in server_keys:
        server = get_public_readonly_mcp_server(key)
        ensure_server_runtime_ready(server, REPO_ROOT)
        proc, url = start_stdio_proxy_if_needed(server)
        if proc:
            procs.append(proc)
        headers = headers_for_server(server)
        tools = list_tools_from_server(url, headers)
        tool_meta = {t.get("name"): t for t in tools if t.get("name")}
        server_ctx[key] = {
            "server": server,
            "url": url,
            "headers": headers,
            "tool_meta": tool_meta,
        }
        print(f"[ok] {key}: tools={len(tool_meta)}")

    backend = create_backend(ModelConfig(provider="openrouter", name="moonshotai/kimi-k2-0905"))
    job = JobConfig.model_validate(
        {
            "model": {"provider": "openrouter", "name": "moonshotai/kimi-k2-0905"},
            "source": {"dataset": "x"},
            "generation": {
                "max_tool_turns": 16,
                "max_batch_size": 1,
                "parameters": {"temperature": args.temperature},
            },
            "output": {"mode": "return"},
        }
    )

    results: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    partial_histories: Dict[int, List[Dict[str, Any]]] = {}
    partial_tool_defs: Dict[int, List[Dict[str, Any]]] = {}
    start_times: Dict[int, float] = {}

    def run_one(index: int, row: Dict[str, Any]) -> Dict[str, Any]:
        key = parse_server_key(row)
        tool_names = parse_tools(row)
        questions = parse_questions(row)
        if not questions:
            return {**row, "trajectory": []}

        ctx = server_ctx[key]
        tool_defs = []
        tool_registry: Dict[str, Any] = {}
        for name in tool_names:
            meta = ctx["tool_meta"].get(name)
            if not meta:
                continue
            schema = meta.get("inputSchema") or meta.get("input_schema") or {}
            tool_defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": meta.get("description", ""),
                        "parameters": schema,
                    },
                }
            )

            def make_tool(tool_name: str):
                def _tool(**kwargs):
                    return call_mcp_tool(ctx["url"], ctx["headers"], tool_name, kwargs)
                return _tool
            tool_registry[name] = make_tool(name)
        partial_tool_defs[index] = list(tool_defs)

        history = _initial_chat_history(questions[0], system_prompt=None)
        partial_histories[index] = list(history)
        turn_limit = max(1, job.generation.max_tool_turns or 16)
        for q_idx, question in enumerate(questions):
            if q_idx > 0:
                history.append({"role": "user", "content": question})
                partial_histories[index] = list(history)
            for turn_idx in range(turn_limit):
                finished = _run_tool_turn(
                    backend=backend,
                    job=job,
                    history=history,
                    tool_defs=tool_defs,
                    tool_registry=tool_registry,
                    turn_idx=turn_idx,
                    prompt=question,
                )
                force_finish = _truncate_and_cap_tool_message(history)
                if force_finish:
                    finished = True
                partial_histories[index] = list(history)
                if finished:
                    break
        return {**row, "tool_def": tool_defs, "trajectory": history}

    workers = max(1, min(args.max_workers, len(rows)))
    executor: Optional[ThreadPoolExecutor] = None
    try:
        executor = ThreadPoolExecutor(max_workers=workers)
        api = HfApi(token=hf_token)
        
        def _checkpoint_upload():
            final_rows = [r for r in results if r is not None]
            final_rows = [
                {"trajectory": row.get("trajectory"), **{k: v for k, v in row.items() if k != "trajectory"}}
                for row in final_rows
            ]

            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            jsonl_path = ARTIFACT_DIR / "tool_calls.jsonl"
            parquet_path = ARTIFACT_DIR / "train-00000-of-00001.parquet"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for row in final_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            pd.DataFrame(final_rows).to_parquet(parquet_path, index=False)

            api.create_repo(repo_id=args.dataset_out, repo_type="dataset", private=True, exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(parquet_path),
                path_in_repo=parquet_path.name,
                repo_id=args.dataset_out,
                repo_type="dataset",
            )

        with tqdm(total=len(rows), desc="tool_calls", unit="row") as pbar:
            last_checkpoint = 0
            for batch_start in range(0, len(rows), workers):
                batch = list(enumerate(rows[batch_start:batch_start + workers], start=batch_start))
                future_to_idx = {executor.submit(run_one, i, row): i for i, row in batch}
                active = dict(future_to_idx)
                for future, idx in future_to_idx.items():
                    start_times[idx] = time.monotonic()

                while active:
                    done = [f for f in active if f.done()]
                    for future in done:
                        idx = active.pop(future)
                        try:
                            row_result = future.result()
                        except Exception as exc:
                            row_result = {
                                **rows[idx],
                                "tool_def": partial_tool_defs.get(idx, []),
                                "trajectory": partial_histories.get(idx, []),
                                "error": str(exc),
                            }
                        if not has_tool_turn(row_result.get("trajectory")):
                            results[idx] = None
                            pbar.update(1)
                            continue
                        results[idx] = row_result
                        pbar.update(1)

                        if pbar.n - last_checkpoint >= CHECKPOINT_EVERY:
                            _checkpoint_upload()
                            last_checkpoint = pbar.n

                    if args.row_timeout_sec and args.row_timeout_sec > 0:
                        now = time.monotonic()
                        timed_out: List[Any] = []
                        for future, idx in list(active.items()):
                            if now - start_times[idx] > args.row_timeout_sec:
                                timed_out.append(future)
                        for future in timed_out:
                            idx = active.pop(future)
                            partial = partial_histories.get(idx, [])
                            if not has_tool_turn(partial):
                                results[idx] = None
                            else: 
                                results[idx] = {
                                    **rows[idx],
                                    "tool_def": partial_tool_defs.get(idx, []),
                                    "trajectory": partial,
                                    "error": "timeout",
                                }
                            future.cancel()
                            pbar.update(1)
                            if pbar.n - last_checkpoint >= CHECKPOINT_EVERY:
                                _checkpoint_upload()
                                last_checkpoint = pbar.n

                    if active:
                        time.sleep(0.2)
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        close = getattr(backend, "close", None)
        if callable(close):
            close()
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    final_rows = [r for r in results if r is not None]
    final_rows = [
        {"trajectory": row.get("trajectory"), **{k: v for k, v in row.items() if k != "trajectory"}}
        for row in final_rows
    ]
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = ARTIFACT_DIR / "tool_calls.jsonl"
    parquet_path = ARTIFACT_DIR / "train-00000-of-00001.parquet"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(final_rows).to_parquet(parquet_path, index=False)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=args.dataset_out, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo=parquet_path.name,
        repo_id=args.dataset_out,
        repo_type="dataset",
    )
    print(f"records={len(final_rows)}")
    print(f"max_workers={workers}")
    print(f"local_jsonl={jsonl_path}")
    print(f"local_parquet={parquet_path}")
    print(f"uploaded=hf://datasets/{args.dataset_out}/{parquet_path.name}")


if __name__ == "__main__":
    main()
