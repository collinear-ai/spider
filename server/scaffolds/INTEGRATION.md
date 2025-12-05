# SWE Scaffold Integration into Spider

This document explains how SWE scaffold trajectory generation has been integrated into Spider's job execution system.

## Overview

Spider now supports SWE (Software Engineering) trajectory generation using external scaffolds (OpenHands, SWE-agent, mini-swe-agent) as an alternative to its standard model-based generation pipeline.

## Architecture

### Integration Points

1. **Config System** (`spider/config.py`):
   - Added `ScaffoldConfig` class to `GenerationConfig`
   - Scaffold jobs are detected when `generation.scaffold` is set
   - Scaffold jobs cannot be combined with `on_policy` mode

2. **Job Execution** (`server/executor.py`):
   - Added `_run_scaffold_job()` function
   - Routes scaffold jobs through a separate execution path
   - Integrates with Spider's existing output handling (HF upload, metadata, etc.)

3. **Scaffold Wrappers** (`server/scaffolds/`):
   - Base scaffold interface (`base.py`)
   - OpenHands wrapper (`openhands_wrapper.py`)
   - Future: SWE-agent and mini-swe-agent wrappers

## Usage

### Example Config

```yaml
server:
  base_url: http://localhost:9000
  api_key: 
  request_timeout: 600

job:
  model:
    provider: "vllm"
    name: "dummy"  # Not used for scaffold jobs
  
  source:
    dataset: "SWE-bench/SWE-smith"
    split: "train"
  
  generation:
    scaffold:
      type: "openhands"
      agent_class: "CodeActAgent"
      max_iterations: 50
      llm_model: "anthropic/claude-sonnet-4"
      num_workers: 1
      timeout_seconds: 3600
      max_retries: 5
      scaffold_specific:
        enable_browser: false
        runtime: "docker"
  
  output:
    mode: "upload_hf"
    hf:
      repo_id: "your-org/swe-trajectories"
      repo_type: "dataset"
```

### Python Client Usage

```python
from spider.client import SpiderClient
from spider.config import AppConfig

config = AppConfig.load("config/swe/spider-openhands-job.yaml")

with SpiderClient(config=config) as client:
    submission = client.submit_job()
    job_id = submission["job_id"]
    
    status = client.poll_job(job_id, interval=5.0, timeout=3600, wait_for_completion=True)
    
    if status["status"] == "completed":
        client.download_result(job_id, destination="./trajectories.jsonl")
```

## How It Works

1. **Job Submission**: User submits a job with `generation.scaffold` configured
2. **Job Routing**: `run_generation_job()` detects scaffold config and routes to `_run_scaffold_job()`
3. **Scaffold Execution**:
   - Loads dataset from HuggingFace (using `source.dataset` and `source.split`)
   - Creates scaffold-specific config from job config
   - Instantiates scaffold wrapper (e.g., `OpenHandsScaffold`)
   - Runs scaffold on dataset instances
   - Collects trajectories
4. **Output Handling**: 
   - Writes trajectories to `result.jsonl` (Spider's standard format)
   - Generates metadata
   - Uploads to HuggingFace if configured (using Spider's existing upload logic)

## Differences from Standalone Scaffold Usage

- **Integrated with Spider's job system**: Uses Spider's config, output handling, HF upload, etc.
- **Unified interface**: Same client API for scaffold jobs and model generation jobs
- **Consistent output format**: Trajectories written to `result.jsonl` like other Spider jobs
- **Metadata tracking**: Uses Spider's metadata system for job tracking

## Current Limitations

1. **OpenHands only**: Currently only OpenHands scaffold is implemented
2. **Output format**: Scaffold outputs are copied to Spider's standard format; native scaffold formats are not preserved
3. **No instance filtering**: Instance filtering via config not yet implemented (can be added)

## Future Work

1. Add SWE-agent scaffold wrapper
2. Add mini-swe-agent scaffold wrapper
3. Support instance filtering via config
4. Preserve scaffold-native output formats as an option
5. Add scaffold-specific metrics and evaluation

