"""Train on-policy SWE-rebench using Hugging Face Accelerate.

This script uses the Accelerate-based on-policy training instead of Tinker.

Required environment variables:
- HF_TOKEN: Hugging Face token for model access

Usage:
    python scripts/train_on_policy_swe_accelerate.py
"""

# IMPORTANT: Prevent CUDA initialization before vLLM starts.
# vLLM needs to spawn workers with clean CUDA state.
# This must be done BEFORE importing torch or any package that uses CUDA.
import os
_original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
# Temporarily hide CUDA devices to prevent initialization during imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import sys
from pathlib import Path
import yaml
import logging

from spider.config import JobConfig
from workloads.swe_rebench_openhands.runner_accelerate import run_server_only

# Restore CUDA_VISIBLE_DEVICES after imports but before main execution
if _original_cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _original_cuda_visible_devices
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def _load_job_config(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    job = raw.get("job")
    return JobConfig.model_validate(job)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    # Use accelerate-specific config
    config_path = Path("config/train_on_policy_swe_accelerate.yaml")
    if not config_path.exists():
        # Fallback to original config
        config_path = Path("config/train_on_policy_swe.yaml")
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Please ensure config/train_on_policy_swe_accelerate.yaml exists.")
        sys.exit(1)

    print(f"Using config: {config_path}")

    job = _load_job_config(config_path)

    # Run training with accelerate
    run_server_only(
        job=job,
        workspace=Path("./workspace_accelerate"),
        split="filtered",
        on_batch_start_lookahead=2,
        prefetch_max_workers=2,
        max_batches_keep=2,
    )


if __name__ == "__main__":
    main()
