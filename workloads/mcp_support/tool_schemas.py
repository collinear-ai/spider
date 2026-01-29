from __future__ import annotations

from typing import Any, Dict, List, Optional
import re
import os
import json

import anyio
import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from spider.config import ToolConfig

_SOURCE_TEMPLATE = """
import os 
import anyio
from anyio import fail_after
import json
import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

def _load_mcp_headers():
    raw = os.environ.get("MCP_HEADERS_JSON", "").strip()
    if not raw:
        return None
    try:
        headers = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("MCP_HEADERS_JSON must be valid JSON.") from exc
    if not isinstance(headers, dict):
        raise ValueError("MCP_HEADERS_JSON must decode to an object.")
    return headers

def call_mcp_tool(server_url, tool_name, arguments):
    async def _run():
        headers = _load_mcp_headers()
        last_exc = None
        for attempt in range(1, 4):
            try:
                with fail_after(120):
                    async with httpx.AsyncClient(headers=headers) as http_client:
                        async with streamable_http_client(server_url, http_client=http_client) as (read_stream, write_stream, _):
                            async with ClientSession(read_stream, write_stream) as session:
                                await session.initialize()
                                result = await session.call_tool(tool_name, arguments)
                                return result.model_dump()
            except (TimeoutError, httpx.HTTPError) as exc:
                last_exc = exc
                if attempt == 3:
                    raise
                await anyio.sleep(1.0 * attempt)
        raise last_exc
    return anyio.run(_run)

def {func_name}(**kwargs):
    return call_mcp_tool(os.environ["{mcp_url_env}"], "{tool_name}", kwargs)
"""

def _load_mcp_headers() -> Optional[Dict[str, str]]:
    raw = os.environ.get("MCP_HEADERS_JSON", "").strip()
    if not raw:
        return None
    try:
        headers = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("MCP_HEADERS_JSON must be valid JSON.") from exc
    if not isinstance(headers, dict):
        raise ValueError("MCP_HEADERS_JSON must decode to an object.")
    return headers

def _sanitize_tool_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name).strip("_")
    if not cleaned:
        raise ValueError(f"Tool name is empty or invalid after sanitization: {name}.")
    if cleaned[0].isdigit():
        cleaned = f"tool_{cleaned}"
    return cleaned

def tool_config_from_server(
    server_url: str,
    *,
    mcp_url_env: str,
) -> List[ToolConfig]:
    if not server_url:
        raise ValueError("server_url is required to query MCP tools.")
    headers = _load_mcp_headers()
        
    async def _run() -> List[ToolConfig]:
        async with httpx.AsyncClient(headers=headers) as http_client:
            async with streamable_http_client(server_url, http_client=http_client) as (
                read_stream,
                write_stream,
                _get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    payload = result.model_dump()

        tools = payload.get("tools") or []
        configs = []
        for tool in tools:
            raw_name = tool.get("name") or ""
            func_name = _sanitize_tool_name(raw_name)
            description = tool.get("description") or ""
            schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            source = _SOURCE_TEMPLATE.format(
                func_name=func_name,
                tool_name=raw_name,
                mcp_url_env=mcp_url_env,
            )
            configs.append(
                ToolConfig(
                    name=func_name,
                    description=description,
                    json_schema=schema,
                    source=source,
                    kwargs={},
                )
            )
        return configs

    return anyio.run(_run)