# Spider

Lightweight on/off-policy distillation framework with a single client interface.

If `on_policy: false`, a complete pipeline for generating distillation data will be set up. 

If `on_policy: true`, a complete pipeline for online training job will be set up, with cross-tokenizer teacher support.

`spider` takes care the whole workflow from dataset preparation, rollouts, kl supervision (on-policy only), and data post-processing in a few lines of code.

## Install

```bash
# install & launch a server
pip install -e ".[server]"
uvicorn server.app:app --host 0.0.0.0 --port 9000 --workers 1

# install a client
pip install -e ".[client]" # (available for cpu machines)
```

## Python client API

The following snippet showcases a complete job cycle for a distillation job.

```python
from spider.client import SpiderClient
from spider.config import AppConfig

def post_process_row(row): # define custom data filtering logic
  """
  Per-row filtering function after a rollout is generated.
  Can reference arbitrary helpers defined in the same script
  """
  return row # or None, if unwanted

config = AppConfig.load("config/test-remote-processor.yaml") # define rollout hyperparams

with SpiderClient(config=config, processor=post_process_row) as client:
    submission = client.submit_job()
    job_id = submission["job_id"]

    # pool_job streams the distillation process back to client
    status = client.poll_job(job_id, interval=5.0, timeout=600, wait_for_completion=True)
    
    if status["status"] == "completed":
        client.download_result(job_id, destionation="./artifacts/result.json") # return full data with metadata, optionally upload to HF
```

### Config Snapshot

The following is the config file for an off-policy distillation job.

```yaml
server:
  base_url: 
  api_key:
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
      repo_type: "dataset"
      token: 
      private: false
```

The following is the config file for an off-policy distillation job.

```yaml
server:
  base_url: 
  api_key:
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
    mode: "upload_hf"
    hf:
      repo_id: RiddleHe/qwen3-8B-on-policy-distill-teacher-qwen3-30B-A3B-OCR2-8-examples
      repo_type: "model"
      private: false
      token: 
```