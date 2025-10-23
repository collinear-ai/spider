# Tinker for Data

Lightweight synthetic data generation framework with a single client interface.

## Python API

```python
from tinker_data import SyntheticDataClient

client = SyntheticDataClient.from_config("config/basic.yaml")
client.run()
```

### CLI

```bash
tinker-data run --config configs/basic.yaml
```

### Config Snapshot

```yaml
model:
    name: Qwen/Qwen3-8B
    backend: vllm
    parameters:
        max_tokens: 1024
sources:
    type: jsonl
    path: data/questions.jsonl
    field: question
generation:
    duplications: 3
output:
    format: parquet
    desination: outputs/qwen3-8b-rollouts.parquet
```

Python API and CLI share the same schema. CLI flags will override config values once implemented.