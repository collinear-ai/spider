import json, os
from typing import Iterable, Dict, Any, List, Optional

from openai import OpenAI

from spider.client import SpiderClient
from spider.config import AppConfig

# == main function for client call ==

def main() -> None:
    config = AppConfig.load("config/generate_tulu_precise_if.yaml")

    with SpiderClient(
        config=config, 
        env=("HF_TOKEN", "HF_HOME"),
    ) as client:
        submission = client.submit_job()
        job_id = submission["job_id"]
        print(f"Job submitted: {job_id}")

        status = client.poll_job(job_id, interval=5.0, wait_for_completion=False)
        print(f"Final status: {status['status']}")

        if status["status"] == "completed":
            artifact_path = client.download_result(job_id, destination="artifacts/generate_tulu_precise_if.json")
            print(f"Artifacts saved to {artifact_path}")
        else:
            print("Messages: ", status.get("messages", []))
            if status.get("error"):
                print("Error: ", status["error"])

if __name__ == "__main__":
    main()