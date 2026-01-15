from __future__ import annotations

import argparse
import os
from typing import List

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Route
import uvicorn

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proxy stdio MCP server to streamable HTTP."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True, help="Stdio MCP command")
    parser.add_argument("--json-response", action="store_true", help="Use JSON responses over SSE.")

    return parser.parse_args()

async def _serve(args: argparse.Namespace) -> None:
    if not args.command:
        raise SystemExit("Missing --command for stdio MCP server.")

    server = Server("mcp-stdio-proxy")
    env = os.environ.copy()
    lock = anyio.Lock()

    async with stdio_client(args.command, env=env) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            @server.list_tools()
            async def _list_tools():
                async with lock:
                    result = await session.list_tools()
                    return result.tools

            @server.call_tool()
            async def _call_tool(name: str, arguments: dict):
                async with lock:
                    return await session.call_tool(name, arguments)
            
            session_manager = StreamableHTTPSessionManager(
                server,
                json_response=args.json_response,
            )
            
            async def lifespan(app: Starlette):
                async with session_manager.run():
                    yield

            app = Starlette(
                routes=[
                    Route(
                        "/mcp",
                        session_manager.handle_request,
                        methods=["GET", "POST", "DELETE"],
                    ),
                ],
                lifespan=lifespan,
            )
            config = uvicorn.Config(
                app, host=args.host, port=args.port, log_level="info"
            )
            
            uvicorn_server = uvicorn.Server(config)
            await uvicorn_server.serve()

def main() -> None:
    args = _parse_args()
    anyio.run(_serve, args)

if __name__ == "__main__":
    main()