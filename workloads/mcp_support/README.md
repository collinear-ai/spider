# MCP support for spider workflow

This directory contains helpers that will set up MCP servers and provide tools for a spider job to run. 

For off-policy distillation on jira, see [this script](../../scripts/generate_mcp_jira.py).

For off-policy distillation on Financial Datasets, see [this script](../../scripts/generate_mcp_finance.py).

## Set up

For each MCP server, two things need to be set up: the MCP server dependency and environmental variables needed for that server.

For jira and the like which make calls to a stdio server:

```bash
pip install mcp anyio httpx starlette uvicorn sse-starlette
npx -y mcp-remote https://mcp.atlassian.com/v1/mcp # auth
python scripts/generate_mcp_jira.py
```

For Financial Datasets and the like which make calls to a streambale HTTP server:

```bash
pip install mcp anyio httpx
export MCP_HEADERS_JSON='{"X-API-KEY": "YOUR_API_KEY"}' # auth
python scripts/generate_mcp_finance.py
```

Look up on the relevant MCP server doc for what headers and auth options to choose specifically.