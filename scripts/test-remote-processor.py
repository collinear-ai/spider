import re

from spider.client import SpiderClient
from spider.config import AppConfig

def strip_think_tags(record: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    _THINK_TAG_PATTERN = re.compile(r"</?think>")
    cleaned = []
    for record in records:
        updated = dict(record)
        value = updated.get("completion")
        if isinstance(value, str):
            updated[field] = _THINK_TAG_PATTERN.sub("", value)
        cleaned.append(updated)
    return cleaned

def main() -> None:
    config = AppConfig.load("config/test-remote-processor.yaml")
    with SpiderClient(config=config, processor=strip_think_tags) as client:
        submission = client.submit_job()
        job_id = submission["job_id"]
        print(f"Job submitted: {job_id}")

        status = client.poll_job(job_id, interval=5.0, timeout=600, wait_for_completion=True)
        print(f"Final status: {status['status']}")

        if status["status"] == "completed":
            artifact_path = client.download_result(job_id, destination="artifacts/test-remote.json")
            print(f"Artifacts saved to {artifact_path}")
        else:
            print("Messages: ", status.get("messages", []))
            if status.get("error"):
                print("Error: ", status["error"])

if __name__ == "__main__":
    main()