# Spider

Lightweight synthetic data generation framework with a single client interface.

## Python API

```python
from spider.client import SpiderClient

client = SpiderClient.from_config("config/qwen3-8B-ocr2.yaml")
submission = client.submit_job()
job_id = submission["job_id"]

status = client.poll_job(job_id, interval=30.0)
if status["status"] == "completed":
    client.download_result(job_id, destionation="./artifacts/result.jsonl")
```

### CLI (planned)

```bash
spider submit --config configs/qwen3-8B-ocr2.yaml
```

### Config Snapshot

```yaml
job:
    model:
        provider: vllm
        name: Qwen/Qwen3-8B
        parameters:
            max_tokens: 1024
            temperature: 0.7
    sources:
        -   type: hf_dataset
            name: opencodereasoning
            dataset: nvidia/OpenCodeReasoning-2
            split: python
            field: question
    generation:
        parameters:
            temperature: 0.8
            top_9: 0.9
    output:
        mode: return
        format: jsonl
        local_path: ./artifacts/result.jsonl
```

Python API and CLI share the same schema. CLI flags will override config values once implemented.

### TODOs

- [ ] Implement backend connection to remote server
- [ ] Implement batching and streaming generation in the remote endpoint
- [ ] Integrate Tinker for server-side on-policy distillation 
- [ ] Enable server-side filtering logic for data