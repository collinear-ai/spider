import os

from spider.client import SpiderClient
from spider.config import AppConfig

from workloads.mcp_support.tool_schemas import tool_config_from_server
from workloads.mcp_support.launcher import (
    use_mcp_remote_url,
    stop_mcp_support_servers
)

MCP_URL_ENV = "EXPEDIA_MCP_URL" # Change it to any MCP server URL
REMOTE_MCP_URL = "http://127.0.0.1:9900/mcp"

def main() -> None:
    config = AppConfig.load("config/generate_mcp_expedia.yaml")
    config.job.ensure_runtime().add_packages(
        "mcp",
        "httpx",
        "anyio",
    )

    handle, mcp_url = use_mcp_remote_url(
        mcp_url_env=MCP_URL_ENV,
        remote_url=REMOTE_MCP_URL,
    )

    try:
        config.job.tools = tool_config_from_server(
            mcp_url,
            mcp_url_env=MCP_URL_ENV,
        )

        with SpiderClient(
            config=config, 
            env=(MCP_URL_ENV, "EXPEDIA_API_KEY", "HF_TOKEN", "OPENAI_API_KEY", "OPENROUTER_API_KEY"),
        ) as client:
            submission = client.submit_job()
            status = client.poll_job(
                submission["job_id"], interval=5.0, wait_for_completion=True,
            )
            if status["status"] == "completed":
                client.download_result(
                    submission["job_id"],
                    destination="artifacts/generate_mcp_expedia.json",
                )
    finally:
        if handle:
            stop_mcp_support_servers([handle])

if __name__ == "__main__":
    main()