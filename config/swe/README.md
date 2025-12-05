# SWE Trajectory Generation Quick Start

## Simple Standalone Usage

### 1. Install Dependencies

```bash
pip install spider[swe-scaffolds]
```

### 2. Set Up OpenHands

**Option A: Install from PyPI**
```bash
pip install openhands-ai
```

**Option B: Install from Source** (if you need evaluation utils)
```bash
git clone https://github.com/OpenHands/OpenHands.git
cd OpenHands
pip install -e .
export OPENHANDS_EVAL_PATH=/path/to/OpenHands/evaluation  # Optional
```

### 3. Set API Keys

```bash
export ANTHROPIC_API_KEY=your_key_here
# Or set llm_api_key in the config file
```

### 4. Edit Config File

Edit `config/swe/my-swe-config.yaml`:

```yaml
scaffold:
  type: openhands
  dataset: "SWE-bench/SWE-smith"     # Your dataset
  agent_class: "CodeActAgent"        # Agent name
  llm_model: "anthropic/claude-sonnet-4"  # Your LLM
  output_dir: "./trajectories"
  # ... other settings
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

- `my-swe-config.yaml` - Simple standalone config (recommended for quick start)
- `openhands-example.yaml` - Detailed example with all options
- `spider-openhands-job.yaml` - Config for Spider job system

