# MCP support for spider workflow

This directory contains helpers that will set up MCP servers and provide tools for a spider job to run. 

For off-policy distillation on jira, see [this script](../../scripts/generate_mcp_jira.py).

## Set up

For each MCP server, two things need to be set up: the MCP server dependency and environmental variables needed for that server.

For jira, the following should be executed in the **client** environment:

```bash
pip install mcp anyio httpx
pip install atlassian-jira-mcp-server 

export JIRA_BASE_URL="https://spider-mcp-tool-call.atlassian.net"
export JIRA_USERNAME="muyu@collinear.ai"
export JIRA_API_TOKEN="YOUR_NEW_TOKEN"
export JIRA_DEFAULT_PROJECT="SCRUM"
export JIRA_MCP_URL="http://127.0.0.1:8080/mcp" # this should match the port selection in the run script
```

Since the MCP server is hosted on the client machine, the whole workload currently only supports the spider server being on the same local machine as the client. 

Other MCP servers should follow the same pattern.