from __future__ import annotations

import json, logging, os
from pathlib import Path
from typing import Any, Dict

from spider.client import SpiderClient
from spider.config import AppConfig

def main():
    config = AppConfig.load("config/train_on_policy_precise_if.yaml")

    env = {"TINKER_API_KEY": os.environ.get("TINKER_API_KEY")}
    
    with SpiderClient(config=config, env=env) as client:
        submission = client.submit_job()
        job_id = submission["job_id"]
        status = client.poll_job(
            job_id,
            interval=20.0,
            wait_for_completion=True,
        )
        if status["status"] == "completed":
            client.download_result(job_id, destination="artifacts/train_on_policy_precise_if.json")
        else:
            raise RuntimeError(status.get("error") or status.get("messages"))

if __name__ == "__main__":
    main()