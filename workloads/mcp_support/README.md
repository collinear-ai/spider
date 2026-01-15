# MCP support for spider workflow

This directory contains helpers that will set up MCP servers and provide tools for a spider job to run. 

For off-policy distillation on jira, see [this script](../../scripts/generate_mcp_jira.py).

## Set up

For each MCP server, two things need to be set up: the MCP server dependency and environmental variables needed for that server.

For jira, the following should be executed in the **client** environment:

```bash
pip install mcp anyio httpx
npx -y mcp-remote https://mcp.atlassian.com/v1/mcp # auth
```

Other MCP servers should follow the same pattern.