import json, os
from typing import Iterable, Dict, Any, List, Optional

from spider.client import SpiderClient
from spider.config import AppConfig

def _preprocess_row(row):
    question, context = row["question"], row["context"]
    context_sent = context["sentences"]
    context_sent_flattened = [sent for group in context_sent for sent in group]
    context_text = "\n".join(context_sent_flattened)
    updated = dict(row)
    updated["prompt"] = f"{context_text}\n\nQuestion: {question}"
    return updated

# == main function for client call ==

def main() -> None:
    config = AppConfig.load("config/generate_multiturn_hotpotqa.yaml")
    env = {"HF_TOKEN": os.getenv("HF_TOKEN"), "HF_HOME": os.getenv("HF_HOME"), "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}
    with SpiderClient(
        config=config, 
        env=env,
        pre_processor=_preprocess_row,
    ) as client:
        submission = client.submit_job()
        job_id = submission["job_id"]
        print(f"Job submitted: {job_id}")

        status = client.poll_job(job_id, interval=5.0, wait_for_completion=False)
        print(f"Final status: {status['status']}")

        if status["status"] == "completed":
            artifact_path = client.download_result(job_id, destination="artifacts/generate_multiturn.json")
            print(f"Artifacts saved to {artifact_path}")

if __name__ == "__main__":
    main()