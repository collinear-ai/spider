# Spider

Lightweight on/off-policy distillation engine with a single client interface. Runnable in minimal lines of code.

`spider` supports two types of jobs:

- **Off-policy distillation**, which is to create a dataset with rollouts from a good teacher model. [This script](scripts/generate_tulu_precise_if.py) demonstrates how to generate a single-turn instruction dataset. [This script](scripts/generate_tool_search_nemo.py) demonstrates how to generate a multi-turn tool-enabled trajectory dataset, with minimally defined custom sandbox, dependencies, and tools.

- **On-policy distillation**, which is to create a training run with online supervision from a good teacher model. [This script](scripts/train_on_policy_precise_if.py) demonstrates how to train on-policy with any teacher model with a different tokenizer, which is a novel feature of this repo. We have also enabled training on-policy with multi-turn tool rollouts, with a script coming soon.

Highlighted features of the engine includes:

- Plug-and-play with any tool definitions and custom pre/post-filtering functions. The user only needs to pass a fixed template of tool and filter definitions to the client. The client will recursively parse and package referenced modules, and the server will spin up a sandbox with dependencies to run the generation.

- On-policy distillation with any chosen model. The backend will realign tokenization differences between student and teacher models to ensure the KL divergence loss is correct.

## Install

```bash
# install & launch a server
pip install -e .[server]
python -m uvicorn server.app:app --host 0.0.0.0 --port 9000
```

```bash
# install a client
pip install -e .[client] # (available on cpu machines)z
```

## Python client API

The following snippet showcases a complete job cycle for a distillation job.

For complete examples across on/off-policy distillation scenarios, see `/scripts` (which references config files in `/config`). Each script is responsible for an independent large-scale distillation run. 

```python
from spider.client import SpiderClient
from spider.config import AppConfig

def pre_prcess_row(row) -> str: # custom transform of input prompt
  return ""

def post_process_row(row) -> Dict[str, Any]: # custom transform of outputs
  return row # or None, if unwanted

config = AppConfig.load("config/test-remote-processor.yaml") # define rollout hyperparams
env = {"HF_TOKEN": "", "OPENAI_API_KEY":""} # define env variables (can also fetch from local env)

with SpiderClient(
  config=config, 
  env=env,
  pre_processor=pre_process_row,
  post_processor=post_process_row
) as client:
    submission = client.submit_job()
    job_id = submission["job_id"]

    # pool_job streams the distillation process back to client
    status = client.poll_job(job_id, interval=5.0, wait_for_completion=True)

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
      private: false
```

The following is the config file for an on-policy distillation job.

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
```