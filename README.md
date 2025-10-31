# Spider

Lightweight on/off-policy distillation framework with a single client interface.

If `on_policy: false`, a complete pipeline for generating distillation data will be set up. If `on_policy: true`, a complete pipeline for online training job will be set up.

`spider` takes care the whole workflow from dataset preparation, rollouts, kl supervision (on-policy only), and data post-processing in a few lines of code.

## Install

```
pip install -e .[server] # launch a server
pip install -e .[client] # launch a client (available for cpu machines)
```

## Python client API

The following snippet showcases a complete job cycle for a distillation job.

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

### Config Snapshot

The following is the config file for an off-policy distillation job.

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

The following is the config file for an off-policy distillation job.

```yaml
server:
  base_url: 
  request_timeout: 180

job:
  model:
    provider: tinker
    name: "Qwen/Qwen3-8B"
  source:
    dataset: "RiddleHe/OpenCodeReasoning-2-questions-dedup-34k-sample-1024"
    split: "train[0:8]"
    field: "question"
  generation:
    on_policy: true
    on_policy_options:
      teacher: "Qwen/Qwen3-30B-A3B"
      api_key: 
      learning_rate: 5e-5
      groups_per_batch: 4
      group_size: 2
      max_tokens: 1024
      lora_rank: 16
      num_substeps: 1
      kl_penalty_coef: 0.1
      kl_discount_factor: 0.0
      loss_fn: "importance_sampling"
      compute_post_kl: true
      eval_every: 5
      save_every: 20
  output:
    mode: "return"
```