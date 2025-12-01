# SWE Task Generation Implementation

## Overview

This document describes the implementation of SWE (Software Engineering) task generation pipeline in Spider, which integrates with [SWE-smith](https://github.com/SWE-bench/SWE-smith) to generate bug instances from GitHub repositories.

## Architecture

### Config Structure

SWE-specific configs are kept separate from the main Spider configs to maintain backward compatibility:

- **`TaskGenerationConfig`**: Configuration for the task generation pipeline
- **`TaskSourceConfig`**: Configuration for loading existing tasks
- **`ScaffoldConfig`**: Configuration for SWE agent scaffolds (for trajectory generation)
- **`TaskOutputConfig`**: Configuration for uploading generated tasks

These are optional fields in `JobConfig`, so existing Spider configs continue to work.

### Module Structure

```
server/task_generation/
├── __init__.py              # Module exports
├── swesmith_integration.py   # SWE-smith pipeline wrapper
└── format_converter.py       # Format conversion utilities
```

### Integration Points

1. **Executor** (`server/executor.py`):
   - Added `_run_task_generation_job()` function
   - Integrated into `run_generation_job()` with priority check

2. **Config** (`spider/config.py`):
   - Added SWE-specific config classes
   - Extended `JobConfig` with optional SWE fields
   - Added validation to ensure correct config combinations

## Pipeline Flow

```
0. Docker Image Creation (if enabled)
   └─> Build Docker image with base repo + dependencies
       (Required before validation, optional rebuild after tasks)

1. Bug Generation
   ├─> LM Modify (LLM modifies code to introduce bugs)
   ├─> LM Rewrite (LLM rewrites code with bugs)
   ├─> Procedural (AST-based transformations)
   └─> PR Mirror (Reverse real PRs)

2. Patch Collection
   └─> Collect all bug patches into single file

3. Validation
   └─> Run test harness to ensure bugs break tests
       (Uses Docker image from step 0)

4. Task Gathering
   └─> Collect validated tasks into task instances
       - Creates Git branches on GitHub mirror
       - Each branch contains a bug
       - Sets image_name in task instances

5. Issue Generation (Optional)
   └─> Generate GitHub-style issue text using LLM

6. Docker Image Rebuild (Optional)
   └─> Rebuild image to include task branches
       (Makes branches available in Docker image)

7. Output
   ├─> Save to JSON/JSONL file
   └─> Upload to HuggingFace (optional)
```

## Usage

### Basic Example

```yaml
# config/swe/task-generation-example.yaml
server:
  base_url: http://localhost:9000

job:
  task_generation:
    enabled: true
    repository:
      github_url: "pandas-dev/pandas"
      commit: "70c3acf6"
    bug_generation:
      methods:
        - type: "lm_modify"
          model: "claude-3-7-sonnet-20250219"
          n_bugs: 10
    validation:
      enabled: true
      workers: 8
    gather:
      enabled: true
  
  task_output:
    mode: "upload_hf"
    hf:
      repo_id: "my-org/pandas-tasks"
```

### Python API

```python
from spider.client import SpiderClient
from spider.config import AppConfig

config = AppConfig.load("config/swe/task-generation-example.yaml")
with SpiderClient(config=config) as client:
    submission = client.submit_job()
    job_id = submission["job_id"]
    status = client.poll_job(job_id, wait_for_completion=True)
    
    if status["status"] == "completed":
        client.download_result(job_id, destination="./tasks.jsonl")
```

## Dependencies

- **SWE-smith**: Must be installed (`pip install swesmith`)
- **Docker**: Required for validation and environment setup
- **HuggingFace Hub**: Required for uploading tasks (`pip install huggingface_hub`)

## Configuration Options

### Repository Config

```yaml
repository:
  github_url: "owner/repo"  # Required
  commit: "abc123"          # Optional: commit, branch, or tag
  mirror_org: "your-org"    # Optional: GitHub org/user for mirrors
  mirror_repo_template: "custom-{owner}-{repo}-{commit}"  # Optional: Custom repo name format
```

**Mirror Repository Options:**
- `mirror_org: "username"` → Creates `username/{owner}__{repo}.{commit}`
- `mirror_org: "org/prefix"` → Creates `org/prefix-{owner}__{repo}.{commit}`
- `mirror_repo_template` → Fully custom format with `{owner}`, `{repo}`, `{commit}` placeholders

### Docker Image Config

```yaml
docker_image:
  enabled: true
  build_before_tasks: true   # Required: build before validation
  rebuild_after_tasks: false # Optional: rebuild after tasks to include branches
  push: false                # Optional: push to Docker Hub
  options:
    workers: 4               # Parallel workers for building
```

**Important Notes:**
- `build_before_tasks: true` is **required** for validation to work
- Docker images contain the base repository at a specific commit
- Task branches are created on GitHub mirror (not in initial image)
- `rebuild_after_tasks: true` includes task branches in the image (optional)

### Bug Generation Methods

#### LM Modify
```yaml
- type: "lm_modify"
  config_file: "configs/bug_gen/lm_modify.yml"  # SWE-smith config
  model: "claude-3-7-sonnet-20250219"
  n_bugs: 10
  n_workers: 20
```

#### Procedural
```yaml
- type: "procedural"
  max_bugs: 20
```

#### PR Mirror
```yaml
- type: "pr_mirror"
  file: "path/to/task-instances.jsonl.all"
  model: "o3-mini"
```

### Validation

```yaml
validation:
  enabled: true
  workers: 8
  options: {}  # Additional SWE-smith validation options
```

### Issue Generation

```yaml
issue_generation:
  enabled: true
  config_file: "configs/issue_gen/ig_v2.yaml"
  model: "claude-3-7-sonnet-20250219"
  workers: 2
```

## Output Format

### SWE-smith Format

Tasks are generated in SWE-smith format:

```json
{
  "instance_id": "pandas-dev__pandas.70c3acf6__lm_modify__abc123",
  "repo": "swesmith/pandas-dev__pandas.70c3acf6",
  "image_name": "jyangballin/swesmith.x86_64.pandas-dev__pandas.70c3acf6",
  "patch": "git diff...",
  "FAIL_TO_PASS": ["tests/test_xyz.py::test_something"],
  "PASS_TO_PASS": ["tests/test_abc.py::test_other"],
  "problem_statement": "GitHub issue text...",
  "base_commit": "70c3acf6"
}
```

### HF Dataset Format

If `task_output.mode: upload_hf`, tasks are converted to HF dataset format:

```json
{
  "org": "pandas-dev",
  "repo": "pandas",
  "instance_id": "...",
  "fix_patch": "...",
  "f2p_tests": {"test": [...]},
  "p2p_tests": {"test": [...]},
  ...
}
```

## Error Handling

- `TaskGenerationError`: Raised for task generation-specific errors
- `JobExecutionError`: Wrapped and raised for executor-level errors
- Validation errors are logged and can be configured to fail or continue

## Future Work

- [ ] Scaffold integration for trajectory generation
- [ ] Support for loading tasks from HF datasets
- [ ] Parallel task generation across multiple repos
- [ ] Caching and incremental generation
- [ ] Integration with SWE-bench format

## References

- [SWE-smith](https://github.com/SWE-bench/SWE-smith)
- [SWE-bench](https://github.com/SWE-bench/SWE-bench)
- [Multi-SWE-bench Dataset](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench)

