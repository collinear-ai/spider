# MCP Support

## Set up

From repo root, install the client package:

```bash
pip install -e ".[client]"
```

Required system tools: `python`,`pip`, `git`, `node`, `npm`, `npx`, `uvx`

Create `workloads/mcp_support/.env`:

```bash
# Required for Hugging Face dataset read/write
HF_TOKEN=...

# Required for generation with OpenRouter
OPENROUTER_API_KEY=...

# Server auth keys
GOOGLE_MAPS_API_KEY=...  # google-maps
FINANCIAL_API_KEY=...    # financialdatasets
TAVILY_API_KEY=...       # tavily
SMITHERY_API_KEY=...     # scientific-computation-mcp
```

## Workflow

Three-step generation pipeline. There is no need to do extra manual server setup; stdio servers are auto-bootstrapped from `public_readonly_servers.py`.

Run end-to-end with one command:

```bash
cd workloads/mcp_support
./run.sh <NUM_EXAMPLES>
```

Or run each stage manually:

1. Generate model-facing prompt templates from MCP servers/tools:

```bash
python workloads/mcp_support/generate_model_prompts.py --num-examples <N>
```

2. Generate user questions from those prompts:

```bash
python workloads/mcp_support/generate_prompts.py --dataset-in <HF_DATASET> --dataset-out <HF_DATASET_OUT>
```

3. Execute tool-calling trajectories against MCP servers:

```bash
python workloads/mcp_support/generate_tool_calls.py --dataset-in <HF_DATASET> --dataset-out <HF_DATASET_OUT>
```

## Servers

Server definitions, auth env vars, runtime/bootstrap commands, and MCP URLs are documented in [servers.md](/Users/muyuhe/Documents/spider/workloads/mcp_support/servers.md) and [public_readonly_servers.py](/Users/muyuhe/Documents/spider/workloads/mcp_support/public_readonly_servers.py)

| MCP server | # tools |
|---|---:|
| `arxiv-paper-mcp` | 4 |
| `context7` | 2 |
| `deepwiki` | 3 |
| `financialdatasets` | 13 |
| `google-maps` | 18 |
| `leetcode` | 9 |
| `open-weather` | 2 |
| `pubmed` | 16 |
| `scientific-computation-mcp` | 26 |
| `tavily` | 5 |
| `time-mcp` | 6 |
| `weather-mcp` | 8 |
| `wikipedia` | 10 |
