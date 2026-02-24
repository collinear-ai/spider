# MCP support for spider workflow

This directory contains helpers that will set up MCP servers and provide tools for a spider job to run. Example domains:

- Jira [script](../../scripts/generate_mcp_jira.py)

- Financial Datasets [script](../../scripts/generate_mcp_finance.py)

- Expedia [script](../../scripts/generate_mcp_expedia.py)

- Public MCP catalog (20 domains; includes read-only + API-key-authenticated endpoints): [research](./PUBLIC_READONLY_MCP_RESEARCH.md), [script](../../scripts/generate_mcp_public_readonly.py)

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

# Financial Datasets
export MCP_HEADERS_JSON='{"X-API-KEY": "YOUR_API_KEY"}' # auth
python scripts/generate_mcp_finance.py

# Expedia
export EXPEDIA_API_KEY="YOUR_API_KEY"
uvx expedia_travel_recommendations_mcp --protocol "streamable-http"
```

Look up on the relevant MCP server doc for what headers and auth options to choose specifically.

## Public read-only catalog quick start

List available servers:

```bash
python scripts/generate_mcp_public_readonly.py --list
```

Run with one of the curated public read-only servers:

```bash
python scripts/generate_mcp_public_readonly.py \
  --server openstreetmap \
  --config config/generate_mcp_finance.yaml \
  --output artifacts/generate_mcp_openstreetmap.json
```

The script resolves remote URL + env key from `workloads/mcp_support/public_readonly_servers.py`,
then loads tools through `tool_config_from_server(...)` so no server-specific tool stubs are needed.
