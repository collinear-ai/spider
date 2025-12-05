# OpenHands Scaffold Implementation Summary

## What Was Built

A complete wrapper around OpenHands for generating SFT trajectories from SWE task datasets (like SWE-bench/SWE-smith).

## Files Created

1. **`spider/server/scaffolds/base.py`**
   - Base `Scaffold` abstract class
   - `ScaffoldConfig` base configuration class

2. **`spider/server/scaffolds/openhands_wrapper.py`**
   - `OpenHandsScaffold` class - main wrapper
   - `OpenHandsScaffoldConfig` - configuration class
   - Handles dataset loading, OpenHands config conversion, trajectory generation

3. **`spider/server/scaffolds/__init__.py`**
   - Module exports

4. **`spider/server/scaffolds/cli.py`**
   - CLI interface for running scaffolds

5. **`spider/server/scaffolds/README.md`**
   - Documentation

6. **`spider/config/swe/openhands-example.yaml`**
   - Example configuration file

7. **`spider/examples/openhands_trajectory_generation.py`**
   - Usage example

## Key Features

✅ **Dataset Loading**: Loads datasets from HuggingFace (e.g., `SWE-bench/SWE-smith`)
✅ **OpenHands Integration**: Uses OpenHands as a dependency (not copied code)
✅ **Docker Management**: Leverages OpenHands' Docker runtime management
✅ **Trajectory Generation**: Generates full trajectories with history, metrics, patches
✅ **Error Handling**: Retries, timeouts, multiprocessing support
✅ **Configurable**: YAML config files + Python API
✅ **Output Format**: Native OpenHands format (JSONL)

## Architecture

```
Spider Config (YAML)
    ↓
OpenHandsScaffoldConfig
    ↓
OpenHandsScaffold
    ↓
OpenHands APIs (create_runtime, run_controller, run_evaluation)
    ↓
Docker Containers + Agent Execution
    ↓
Trajectory Output (JSONL)
```

## Usage

### Python API
```python
from spider.server.scaffolds.openhands_wrapper import (
    OpenHandsScaffold,
    OpenHandsScaffoldConfig,
)

config = OpenHandsScaffoldConfig(
    output_dir=Path("./trajectories"),
    dataset="SWE-bench/SWE-smith",
    llm_model="anthropic/claude-sonnet-4",
)

scaffold = OpenHandsScaffold(config)
output_path = scaffold.run_batch()
```

### YAML Config
```yaml
scaffold:
  type: openhands
  dataset: "SWE-bench/SWE-smith"
  output_dir: "./trajectories"
  llm_model: "anthropic/claude-sonnet-4"
```

## Dependencies

- `openhands-ai>=0.62.0` - OpenHands library
- `datasets` - HuggingFace datasets
- `pandas` - Data manipulation

Install with: `pip install spider[swe-scaffolds]`

## Next Steps

1. **Add SWE-agent scaffold** - Similar wrapper for SWE-agent
2. **Add mini-swe-agent scaffold** - Similar wrapper for mini-swe-agent
3. **Add trajectory conversion utilities** - Convert between scaffold formats if needed
4. **Add evaluation metrics** - Compare trajectories across scaffolds

## Notes

- OpenHands handles all Docker/runtime management
- Trajectories are saved in OpenHands native format
- No normalization needed - each scaffold keeps its own format
- Docker images must be pre-built and available

