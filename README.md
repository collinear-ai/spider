# Spider

Lightweight on/off-policy distillation engine with a single client interface. Runnable in minimal lines of code.

`spider` supports two types of jobs:

- **Off-policy distillation**, which is to create a dataset with rollouts from a good teacher model. 
  - [This script](scripts/generate_tulu_precise_if.py) demonstrates how to generate a single-turn instruction dataset with custom processors to create prompt variations. 
  - [This script](scripts/generate_tool_search_nemo.py) demonstrates how to generate a **multi-turn tool-enabled trajectory dataset**, with runnable sandbox, dependencies, and tools.
  - [This script](scripts/generate_multiturn_hotpotqa.py) demonstrates how to generate a **multi-turn user-simulated trajectory dataset**, where an LLM is configured to play the role of a user to ask follow-up questions.

- **On-policy distillation**, which is to create a training run with online supervision from a good teacher model. 
  - [This script](scripts/train_on_policy_precise_if.py) demonstrates how to train on-policy **with any teacher model with a different tokenizer**, ensuring the correct chat template is used by both models. 
  - [This script](scripts/train_on_policy_tool_search_nemo.py) demonstrates how to train on-policy with a specified set of tools so that the teacher can supervise the student's **multi-turn tool-execution trajectories**. 
  - [This script](scripts/train_on_policy_swe.py) demonstrates how to train on-policy for **SWE-agent** tasks, executing tool trajectories in **concurrent docker environments** following standard agent scaffolds.

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
pip install -e .[client] # (available on cpu machines)
```

## Python client API

The following snippet showcases a complete job cycle for a distillation job.

For complete examples across on/off-policy distillation scenarios, see `/scripts` (which references config files in `/config`). Each script is responsible for an independent large-scale distillation run. 

```python
from spider.client import SpiderClient
from spider.config import AppConfig

def pre_prcess_row(row) -> Dict[str, ANy]: # custom transform of inputs
  return row

def post_process_row(row) -> Dict[str, Any]: # custom transform of outputs
  return row # or None, if unwanted

def tool_call(arg): # custom tool that will execute in sandbox
  return ""

TOOL_SCHEMA = {}

config = AppConfig.load("config/test-remote-processor.yaml") # define rollout hyperparams
env = {"HF_TOKEN": "", "OPENAI_API_KEY":""} # define env variables (can also fetch from local env)

with SpiderClient(
  config=config, 
  env=env,
  pre_processor=pre_process_row,
  post_processor=post_process_row
) as client:
    client.add_tool( # add tool
      description="",
      json_schema=TOOL_SCHEMA,
      func=tool_call,
    )

    submission = client.submit_job()
    job_id = submission["job_id"]

    # pool_job streams the distillation process back to client
    status = client.poll_job(job_id, interval=5.0, wait_for_completion=True)

    if status["status"] == "completed":
        client.download_result(job_id, destionation="./artifacts/result.json") # return full data with metadata, optionally upload to HF
```

### Config Snapshot

The following is the config file for an off-policy distillation job, enabling multi-turn user simulations.

```yaml
server:
  base_url: http://127.0.0.1:9000
job:
  model: 
    provider: vllm
    name: "openai/gpt-oss-120b"
    parameters:
      tensor_parallel_size: 8
      gpu_memory_utilization: 0.85
  source:
    dataset: "hotpotqa/hotpot_qa"
    config_name: "fullwiki"
    split: "train"
    max_examples: 100
    multi_turn: true
    user_simulation_prompt: You are a user prompt generator.
    user_model:
      name: "gpt-5-nano-2025-08-07"
      provider: openai
  generation:
    max_turns: 4
    parameters:
      temperature: 0.7
      top_p: 0.9
      max_tokens: 16384
  output:
    mode: "upload_hf"
    hf:
      repo_id: collinear-ai/spider-openqa-hotpot-gptoss-samples
      private: true
```

The following is the config file for an on-policy distillation job.

```yaml
server:
  base_url: http://127.0.0.1:9000
  request_timeout: 120
job:
  model:
    provider: tinker
    name: "Qwen/Qwen3-8B"
  source:
    dataset: nvidia/Nemotron-RL-knowledge-web_search-mcqa
    split: train[0:128]
  generation:
    on_policy: true
    parameters:
      top_p: 0.9
      max_tokens: 32768
      tool_choice: auto
    on_policy_options:
      teacher: moonshotai/Kimi-K2-Thinking
      learning_rate: 1e-7
      groups_per_batch: 64
      lora_rank: 16
      num_substeps: 1
      kl_penalty_coef: 1.0
      kl_discount_factor: 0.0
      loss_fn: importance_sampling
      save_every: 20
  output:
    mode: upload_hf
    hf:
      repo_id: collinear-ai/spider-on-policy-tool-search-qwen-teacher-kimi-k2
      repo_type: model
```