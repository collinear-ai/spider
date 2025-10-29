# Spider

Lightweight synthetic data generation framework with a single client interface.

## Python API

```python
from spider.client import SpiderClient

client = SpiderClient.from_config("config/qwen3-8B-ocr2.yaml")
submission = client.submit_job()
job_id = submission["job_id"]

status = client.poll_job(job_id, interval=5.0, timeout=600, wait_for_completion=True)
if status["status"] == "completed":
    client.download_result(job_id, destionation="./artifacts/result.json")
```

### CLI (planned)

```bash
spider submit --config configs/qwen3-8B-ocr2.yaml
```

### Config Snapshot

```yaml
server:
  base_url: 
  request_timeout: 120
job:
  model: 
    provider: vllm
    name: "Qwen/Qwen3-8B"
    parameters:
      tensor_parallel_size: 4
  source:
    dataset: "RiddleHe/OpenCodeReasoning-2-questions-dedup-34k-sample-1024"
    split: "train[0:10]"
    field: "question"
  generation:
    parameters:
      temperature: 0.7
      top_p: 0.9
      max_tokens: 256
  output:
    mode: "return"
```

Python API and CLI share the same schema. CLI flags will override config values once implemented.

### TODOs

- [ ] Integrate Tinker for server-side on-policy distillation 
- [ ] Enable server-side filtering logic for data