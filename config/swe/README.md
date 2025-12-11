# SWE Trajectory Generation Quick Start

## Configuration Files

This directory contains example configurations for different use cases:

- **[config-reference.yaml](config-reference.yaml)** - Complete reference with ALL available options
- **[swebench-example.yaml](swebench-example.yaml)** - SWE-bench with generic image (recommended for train split)
- **[swesmith-example.yaml](swesmith-example.yaml)** - SWE-smith with per-instance images
- **[vllm-example.yaml](vllm-example.yaml)** - vLLM configuration

**Quick start:** Copy an example config and modify it for your needs.

**Need to see all options?** Check [config-reference.yaml](config-reference.yaml) for comprehensive documentation.

## Simple Standalone Usage

### 1. Install Dependencies

```bash
pip install spider[swe-scaffolds]
```

### 2. Set Up OpenHands

**Install from Source** (required - evaluation utilities not in PyPI package)
```bash
git clone https://github.com/All-Hands-AI/OpenHands.git
cd OpenHands
pip install -e .
```

### 3. Set API Keys

```bash
export OPENAI_API_KEY=your_key_here
# Or for vLLM:
export VLLM_API_KEY=your_key_here
# Or for Anthropic:
export ANTHROPIC_API_KEY=your_key_here
```

### 4. Create/Edit Config File

Copy an example config file:
```bash
cp config/swe/example-openai-config.yaml config/swe/my-swe-config.yaml
# Or for vLLM:
cp config/swe/example-vllm-config.yaml config/swe/my-swe-config.yaml
```

Then edit for your needs:
```yaml
# Dataset configuration
dataset: "SWE-bench/SWE-smith"
split: "train"
max_instances: 10

# Agent configuration
agent_class: "CodeActAgent"
max_iterations: 30
llm_model: "gpt-4o"
llm_api_key_env: "OPENAI_API_KEY"

# For vLLM or other providers, also set:
# llm_base_url: "http://localhost:8000/v1"

# HuggingFace upload
hf_repo_id: "your-org/your-dataset-name"
hf_private: true
```

### 5. Run Trajectory Generation

```bash
spider-scaffold openhands --config config/swe/my-swe-config.yaml
```

Or with Python:
```bash
python -m spider.server.scaffolds.cli openhands --config config/swe/my-swe-config.yaml
```

## Config File Options

### Required Fields
- `scaffold.type`: Scaffold name (`"openhands"`)
- `scaffold.dataset`: HuggingFace dataset name (e.g., `"SWE-bench/SWE-smith"`)
- `scaffold.agent_class`: Agent name (e.g., `"CodeActAgent"`)
- `scaffold.llm_model`: LLM model string OR `llm_config_name`

### Optional Fields
- `scaffold.split`: Dataset split (default: `"train"`)
- `scaffold.max_instances`: Limit number of instances (default: `null` = all)
- `scaffold.num_workers`: Parallel workers (default: `1`)
- `scaffold.timeout_seconds`: Timeout per instance (default: `null`)
- `scaffold.max_retries`: Retry attempts (default: `5`)

## Using with Spider Job System

If you want to use Spider's full job system (with HF upload, job tracking, etc.):

1. Start Spider server:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 9000
```

2. Use `config/swe/spider-openhands-job.yaml` config

3. Submit job via Python client:
```python
from spider.client import SpiderClient
from spider.config import AppConfig

config = AppConfig.load("config/swe/spider-openhands-job.yaml")
with SpiderClient(config=config) as client:
    submission = client.submit_job()
    job_id = submission["job_id"]
    status = client.poll_job(job_id, wait_for_completion=True)
    if status["status"] == "completed":
        client.download_result(job_id, destination="./trajectories.jsonl")
```

## Example Config Files

- `example-openai-config.yaml` - For OpenAI/Anthropic (recommended for quick start)
- `example-vllm-config.yaml` - For vLLM, Fireworks, Together.ai, or other providers
- `openhands-config.yaml` - Advanced example with all options
- `spider-openhands-job.yaml` - Config for Spider job system (not typically needed)

