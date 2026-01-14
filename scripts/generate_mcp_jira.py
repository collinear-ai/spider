import os

from spider.client import SpiderClient
from spider.config import AppConfig

from workloads.mcp_support.tool_schemas import tool_config_from_server

MCP_URL_ENV = "JIRA_MCP_URL" # Change it to any MCP server URL
MCP_NAME = "jira"

def main() -> None:
    config = AppConfig.load("config/generate_mcp_jira.yaml")
    config.job.ensure_runtime().add_packages(
        "mcp",
        "httpx",
        "anyio",
    )

    mcp_url = os.environ.get(MCP_URL_ENV)

    config.job.tools = tool_config_from_server(
        mcp_url,
        mcp_url_env=MCP_URL_ENV,
    )
    
    with SpiderClient(
        config=config, 
        env=(MCP_URL_ENV, "HF_TOKEN"),
    ) as client:
        submission = client.submit_job()
        status = client.poll_job(
            submission["job_id"], interval=5.0, wait_for_completion=True,
        )
        if status["status"] == "completed":
            client.download_result(
                submission["job_id"],
                destination="artifacts/generate_mcp_zendesk.json",
            )

if __name__ == "__main__":
    main()