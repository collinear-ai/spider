from __future__ import annotations

import json, logging, os
from pathlib import Path
from typing import Any, Dict

from spider.client import SpiderClient
from spider.config import AppConfig

def load_manifest(path):
    if not path.exists():
        raise FileNotFoundError(f"Stage 1 manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage 1 manifest must be a JSON object.")
    return payload

def configure_on_policy(
    config,
    checkpoint_path,
):
    generation = config.job.generation
    if not generation.on_policy or not generation.on_policy_options:
        raise ValueError("Stage 2 config must enable on-policy generation.")

    updates = {"student_checkpoint_path": checkpoint_path}
    generation.on_policy_options = generation.on_policy_options.model_copy(update=updates)
    config.job.generation = generation

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = load_manifest(Path("logs/stage1_sft_manifest.json"))
    config = AppConfig.load("config/train_if_stage_2_on_policy.yaml")

    checkpoint_uri = manifest.get("state_checkpoint")
    if not checkpoint_uri:
        raise ValueError("Stage 1 manifest missing state checkpoint URI.")
    configure_on_policy(config, checkpoint_uri)
    
    logger = logging.getLogger(__name__)
    with SpiderClient(config=config) as client:
        logger.info("Submitting stage 2 on-policy job with checkpoint %s", checkpoint_uri)
        submission = client.submit_job()
        job_id = submission["job_id"]
        status = client.poll_job(
            job_id,
            interval=20.0,
            wait_for_completion=True,
        )
        if status["status"] == "completed":
            client.download_result(job_id, destination="artifacts/train_if_stage2_on_policy.json")
        else:
            raise RuntimeError(status.get("error") or status.get("messages"))

if __name__ == "__main__":
    main()