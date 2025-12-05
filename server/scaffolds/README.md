# SWE Scaffolds for Trajectory Generation

This module provides wrappers around different SWE agent scaffolds (OpenHands, SWE-agent, mini-swe-agent) for generating SFT trajectories from SWE task datasets.

## OpenHands Scaffold

The OpenHands scaffold generates trajectories using OpenHands agents on SWE task datasets.

### Installation

1. Install Spider with SWE scaffold dependencies:
```bash
pip install spider[swe-scaffolds]
```

2. Install OpenHands (if not already installed):
```bash
pip install openhands-ai
```

Or install from source:
```bash
git clone https://github.com/OpenHands/OpenHands.git
cd OpenHands
pip install -e .
```

If installing from source, the evaluation utilities will be available automatically. If using the PyPI package, you may need to set `OPENHANDS_EVAL_PATH` to point to the OpenHands evaluation directory.

### Configuration

Create a YAML config file (see `config/swe/openhands-example.yaml`):

```yaml
scaffold:
  type: openhands
  output_dir: "./trajectories/openhands"
  dataset: "SWE-bench/SWE-smith"
  split: "train"
  agent_class: "CodeActAgent"
  max_iterations: 50
  llm_model: "anthropic/claude-sonnet-4"
  num_workers: 1
```

### Usage

#### Python API

```python
from pathlib import Path
from spider.server.scaffolds.openhands_wrapper import (
    OpenHandsScaffold,
    OpenHandsScaffoldConfig,
)

# Create config
config = OpenHandsScaffoldConfig(
    output_dir=Path("./trajectories"),
    dataset="SWE-bench/SWE-smith",
    split="train",
    agent_class="CodeActAgent",
    max_iterations=50,
    llm_model="anthropic/claude-sonnet-4",
    num_workers=1,
)

# Create scaffold and run
scaffold = OpenHandsScaffold(config)
output_path = scaffold.run_batch(
    dataset_name="SWE-bench/SWE-smith",
    split="train",
    instance_filter=None,  # Optional regex filter
)

print(f"Trajectories saved to: {output_path}")
```

#### Command Line (if CLI is added)

```bash
python -m spider.server.scaffolds.openhands_wrapper \
    --config config/swe/openhands-example.yaml
```

### Output Format

Trajectories are saved as JSONL files with OpenHands native format:

```json
{
  "instance_id": "django__django-11333",
  "instruction": "...",
  "instance": {...},
  "test_result": {
    "git_patch": "..."
  },
  "history": [...],  # Full trajectory of agent actions
  "metrics": {...},
  "error": null
}
```

### Supported Datasets

- `SWE-bench/SWE-smith` - SWE-smith training dataset
- `princeton-nlp/SWE-bench` - SWE-bench dataset
- `princeton-nlp/SWE-bench_Lite` - SWE-bench Lite
- Any HuggingFace dataset with SWE-bench format

### Dataset Format Requirements

The dataset should have these fields:
- `instance_id` (str): Unique identifier
- `problem_statement` (str): Task description
- `repo` (str): Repository identifier
- `image_name` (str, optional): Docker image name (required for SWE-smith)
- `FAIL_TO_PASS` (list[str], optional): Tests that should fail before fix
- `PASS_TO_PASS` (list[str], optional): Tests that should pass before and after

### Docker Images

The scaffold expects Docker images to be pre-built and available:
- SWE-smith: Images are specified in `image_name` field
- SWE-bench: Images are auto-generated from `instance_id`

Make sure Docker images are accessible (pulled or built) before running.

### Troubleshooting

1. **Import errors**: Make sure OpenHands is installed and evaluation utils are accessible
2. **Docker errors**: Ensure Docker is running and images are available
3. **API key errors**: Set LLM API keys via environment variables or config

