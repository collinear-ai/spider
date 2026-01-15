from __future__ import annotations

import argparse
import os
from typing import List

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
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
    lock = anyio.Lock()

    server_params = StdioServerParameters(command=args.command[0], args=args.command[1:])
    async with stdio_client(server_params) as (read_stream, write_stream):
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

            async def mcp_app(scope, receive, send):
                await session_manager.handle_request(scope, receive, send)

            async def lifespan(app: Starlette):
                async with session_manager.run():
                    yield

            app = Starlette(
                routes=[
                    Mount("/mcp", app=mcp_app),
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