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

def pre_prcess_row(row) -> str: # custom transform of input prompt
  return ""

def post_process_row(row) -> Dict[str, Any]: # custom transform of outputs
  return row # or None, if unwanted

config = AppConfig.load("config/test-remote-processor.yaml") # define rollout hyperparams

with SpiderClient(
  config=config, 
  pre_processor=pre_process_row,
  post_processor=post_process_row
) as client:
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

## SWE Trajectory Generation

Spider includes an OpenHands wrapper for generating agent trajectories on SWE datasets (e.g., SWE-smith). This allows you to generate high-quality coding agent trajectories and automatically push them to HuggingFace.

### Installation Requirements

Since OpenHands evaluation utilities are not included in the PyPI package, you need to install OpenHands from source:

```bash
# Clone and install OpenHands
git clone https://github.com/All-Hands-AI/OpenHands.git
cd OpenHands
pip install -e .

# Install Spider
cd /path/to/spider
pip install -e ".[server]"
```

### Configuration

Create a config file for your SWE trajectory generation. See example configs:
- `config/swe/example-openai-config.yaml` - For OpenAI/Anthropic, etc..
- `config/swe/example-vllm-config.yaml` - For vLLM or other providers

**Basic example (OpenAI):**

```yaml
# Dataset configuration
dataset: "SWE-bench/SWE-smith"
split: "train"
max_instances: 50
instance_filter: "conan-io__conan.*"  # Optional: filter by repo

# Agent configuration
agent_class: "CodeActAgent"
max_iterations: 30
llm_model: "gpt-4o"

# Parallel processing
num_workers: 2  # Increase for faster processing (balance with resource constraints)

# Performance tuning
max_retries: 3
timeout_seconds: 1800

# HuggingFace upload (automatic)
hf_repo_id: "your-org/your-dataset-name"
hf_private: true
# hf_config_name: "optional_config_name"  # Auto-generated if not specified

# Output directory
eval_output_dir: "trajectories"
```

**Using vLLM or OpenAI-Compatible APIs:**

You can use any OpenAI-compatible API (vLLM, Fireworks, Together.ai, etc.) by specifying the base URL and API key:

```yaml
# Agent configuration with custom API
agent_class: "CodeActAgent"
max_iterations: 30
llm_model: "your-org/your-model"  # Model name as recognized by your API
llm_base_url: "http://localhost:8000/v1"  # vLLM endpoint
llm_api_key: "your-api-key"  # Or use llm_api_key_env: "VLLM_API_KEY"

# Example for Fireworks AI
# llm_model: "accounts/fireworks/models/llama-v3p1-70b-instruct"
# llm_base_url: "https://api.fireworks.ai/inference/v1"
# llm_api_key_env: "FIREWORKS_API_KEY"

# Example for Together.ai
# llm_model: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
# llm_base_url: "https://api.together.xyz/v1"
# llm_api_key_env: "TOGETHER_API_KEY"
```

**API Key Options:**
- `llm_api_key`: Direct API key in config (not recommended for sensitive keys)
- `llm_api_key_env`: Environment variable name (e.g., `"OPENAI_API_KEY"`)

### Running Trajectory Generation

```bash
# Set your HuggingFace token for automatic upload
export HF_TOKEN="hf_..."

# Run trajectory generation
spider-scaffold openhands --config config/swe/my-swe-config.yaml
```

This will:
1. Download the SWE-smith dataset
2. Generate agent trajectories using OpenHands
3. Save results to `trajectories/` with logs per instance
4. Automatically upload to HuggingFace Hub 

### Output Structure

```
trajectories/
└── SWE-bench__SWE-smith-train/
    └── CodeActAgent/
        └── gpt-4o_maxiter_30/
            ├── output.jsonl          # Raw trajectories
            ├── metadata.json         # Run metadata
            └── infer_logs/
                ├── instance_1.log    # Separate log per instance
                ├── instance_2.log
                └── ...
```

### Dataset Format

The uploaded HuggingFace Dataset includes:

- **`messages`**: Chat format for training (list of `{"role": "user/assistant", "content": "..."}`)
- **`trajectory`**: Raw OpenHands events for reference
- **`instance_id`**, **`repo`**, **`image_name`**: Task identifiers
- **`instruction`**, **`problem_statement`**: Task descriptions
- **`git_patch`**: Generated code changes
- **`metrics`**, **`resolved`**: Evaluation results
- **`model`**, **`agent_class`**, **`max_iterations`**: Generation metadata



### HuggingFace Upload

**Automatic Upload (during generation):**
- Happens automatically after trajectory generation completes
- Requires `HF_TOKEN` environment variable
- Converts trajectories to training-ready `messages` format

**Manual Upload:**

```bash
# Edit the script to set your paths
nano scripts/push_openhands_traj_dataset.py

# Run the upload
export HF_TOKEN="hf_..."
python3 scripts/push_openhands_traj_dataset.py
```

To evaluate trajectories and add the `resolved` field, run OpenHands evaluation:

```bash
cd /path/to/OpenHands
# Evaluate generated trajectories
python -m evaluation.swe_bench.eval_infer \
  --dataset "your-org/your-dataset-name" \
  --split "train" \
  --input-file "path/to/output.jsonl" \
  --output-dir "evaluation_results" \
  --num-workers 8
```

This will apply patches, run tests in Docker, and update `output.jsonl` with `resolved=True/False`.


### Tips

- Use `instance_filter` to focus on a single repository for faster iteration
- Monitor logs in `trajectories/.../infer_logs/` for debugging
- The first run per unique Docker image will be slower due to image building
- Most trajectories will have `resolved=None` until you run evaluation