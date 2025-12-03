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

### Required for All Methods
- SWE-smith must be installed: `pip install swesmith`
- Docker must be installed and running (for validation)
- **GitHub token**: Set in `.env` file or `export GITHUB_TOKEN=your_token` (for creating mirror repos)
  - Token needs `repo` scope (full control of private repositories)
  - If using an organization, token needs `admin:org` scope and the token owner must be an org admin
- **SSH keys for GitHub**: Required for git clone/push operations
  - SWE-smith uses SSH (`git@github.com:...`) for all git operations
  - Generate SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
  - Add to SSH agent: `eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519`
  - Add public key to GitHub: https://github.com/settings/keys
  - Test: `ssh -T git@github.com` (should show "You've successfully authenticated")

**Note on Mirror Repositories:**
- SWE-smith creates mirror repositories in your specified GitHub organization/user
- These mirrors contain the repository code at the specified commit
- The mirror repository must be populated with code before bug generation can proceed
- If a mirror repository exists but is empty, you may need to manually populate it or delete it and let SWE-smith recreate it

### Optional
- HuggingFace token (if uploading to HF): `export HF_TOKEN=your_token`
- LLM API keys (only if using `lm_modify` or `lm_rewrite`):
  - `export ANTHROPIC_API_KEY=your_key` (for Claude)
  - `export OPENAI_API_KEY=your_key` (for GPT-4)
- PR data file (for `pr_mirror` method)

### Important: No vllm Needed!
- **Task generation does NOT require vllm or LLM hosting**
- The `model` field in job config is just a placeholder (not used)
- For `pr_mirror` and `procedural`: **No LLM needed at all**
- For `lm_modify`/`lm_rewrite`: SWE-smith uses API keys directly (not vllm)

See `BUG_GENERATION_METHODS.md` for detailed comparison of all four methods.

## Bug Generation Methods

SWE-smith supports four bug generation methods, with different trade-offs:

### PR Mirror (Recommended for Training) ⭐
**Best for training effectiveness** - Inverts real pull requests to create realistic bug instances.

- **Training Performance**: 9.2% ± 1.7 resolve rate (best)
- **Realism**: Most reflective of SWE-bench
- **Cost**: Free (uses existing PRs)
- **Requirements**: PR data file (SWE-bench task instances format, see `PR_MIRROR_FILE_FORMAT.md`)

```yaml
bug_generation:
  methods:
    - type: "pr_mirror"
      file: "path/to/pr_data.json"  # PR data file
```

### LM Rewrite
Uses language models to rewrite code entities from scratch with bugs.

- **Training Performance**: 8.8% ± 1.7 resolve rate (strong)
- **Requirements**: LLM API key (Claude/OpenAI)
- **Cost**: API costs per bug

```yaml
bug_generation:
  methods:
    - type: "lm_rewrite"
      model: "claude-3-7-sonnet-20250219"
      n_bugs: 10
```

### Procedural
Uses AST-based transformations to procedurally generate bugs.

- **Training Performance**: 8.6% ± 1.8 resolve rate (strong)
- **Requirements**: None (no LLM needed)
- **Cost**: Free
- **Scalability**: High

```yaml
bug_generation:
  methods:
    - type: "procedural"
      max_bugs: 20
```

### LM Modify
Uses language models to modify code entities to introduce bugs.

- **Training Performance**: 5.7% ± 1.5 resolve rate (weaker)
- **Requirements**: LLM API key (Claude/OpenAI)
- **Cost**: API costs per bug

```yaml
bug_generation:
  methods:
    - type: "lm_modify"
      model: "claude-3-7-sonnet-20250219"
      n_bugs: 10
```

## Recommended Strategy

Based on SWE-smith research (arXiv:2504.21798):
- **Best overall**: PR Mirror (realism + training effectiveness)
- **Strong complements**: Procedural + LM Rewrite (scalable)
- **Avoid alone**: LM Modify (weaker training signal despite decent yield)

## Next Steps

After generating tasks, you can:
1. Use them for trajectory generation with scaffolds (swe-agent, openhands, etc.)
2. Upload them to HuggingFace for sharing
3. Use them for training SWE agents

See `trajectory-generation-example.yaml` for generating trajectories from tasks.

