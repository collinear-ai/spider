# SWE Task Generation Configs

This directory contains example configurations for SWE task generation using Spider.

## Overview

Spider integrates with [SWE-smith](https://github.com/SWE-bench/SWE-smith) to generate software engineering task instances from GitHub repositories. The pipeline:

1. **Generates bugs** using various methods (LM-based, procedural, PR mirroring)
2. **Validates bugs** by running test harnesses
3. **Gathers tasks** into a standardized format
4. **Generates issue text** (optional) using LLMs
5. **Uploads tasks** to HuggingFace datasets (optional)

## Configuration Files

### `task-generation-example.yaml`

Complete example for generating tasks from a repository.

**Key sections:**
- `task_generation.repository`: GitHub repository to generate tasks for
- `task_generation.bug_generation.methods`: List of bug generation methods
- `task_generation.validation`: Test harness validation configuration
- `task_generation.gather`: Task gathering configuration
- `task_generation.issue_generation`: Optional issue text generation
- `task_output`: Configuration for uploading tasks to HuggingFace

## Usage

```bash
# Start Spider server
cd /home/ubuntu/spider
uvicorn server.app:app --host 0.0.0.0 --port 9000

# Submit task generation job
python -c "
from spider.client import SpiderClient
from spider.config import AppConfig

config = AppConfig.load('config/swe/task-generation-example.yaml')
with SpiderClient(config=config) as client:
    submission = client.submit_job()
    job_id = submission['job_id']
    status = client.poll_job(job_id, wait_for_completion=True)
    print(f'Job completed: {status}')
"
```

## Requirements

- SWE-smith must be installed: `pip install swesmith`
- SWE-smith config files must be available (e.g., `configs/bug_gen/lm_modify.yml`)
- Docker must be installed and running (for validation)
- HuggingFace token (if uploading to HF): `export HF_TOKEN=your_token`

## Bug Generation Methods

### LM Modify
Uses language models to modify code entities to introduce bugs.

### LM Rewrite
Uses language models to rewrite code entities from scratch with bugs.

### Procedural
Uses AST-based transformations to procedurally generate bugs.

### PR Mirror
Reverses real pull requests to create bug instances.

## Next Steps

After generating tasks, you can:
1. Use them for trajectory generation with scaffolds (swe-agent, openhands, etc.)
2. Upload them to HuggingFace for sharing
3. Use them for training SWE agents

See `trajectory-generation-example.yaml` for generating trajectories from tasks.

