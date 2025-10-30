# Spider

Lightweight synthetic data generation framework with a single client interface.

## Python API

```python
from spider.client import SpiderClient
from spider.config import AppConfig

def filter_rows(records): # define custom data filtering logic
  return records

config = AppConfig.load("config/test-remote-processor.yaml") # define rollout hyperparams

with SpiderClient(config=config, processor=filter_rows) as client:
    submission = client.submit_job()
    job_id = submission["job_id"]

    status = client.poll_job(job_id, interval=5.0, timeout=600, wait_for_completion=True)
    if status["status"] == "completed":
        client.download_result(job_id, destionation="./artifacts/result.json") # return full data with metadata, optionally upload to HF
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
    name: "Qwen/Qwen2.5-7B-Instruct"
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
      max_tokens: 8196
  output:
    mode: "upload_hf"
    hf:
      repo_id: collinear-ai/spider-rollouts-qwen2.5-7b-instruct-ocr2-ast-filter
      token: 
      private: false
```

Python API and CLI share the same schema. CLI flags will override config values once implemented.

### TODOs

- [ ] Integrate Tinker for server-side on-policy distillation 
- [ ] Enable server-side filtering logic for data