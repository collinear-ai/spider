import re, ast
from typing import Iterable, Dict, Any, List

from spider.client import SpiderClient
from spider.config import AppConfig

def filter_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []
    for record in records:
        updated = dict(record)
        value = updated.get("completion")
        if not isinstance(value, str):
            continue
        lower_value = value.lower()
        start_marker = lower_value.rfind("```python")
        if start_marker == -1:
            continue
        code_start = start_marker + len("```python")
        closing_marker = value.find("```", code_start)
        if closing_marker == -1:
            continue
        code = value[code_start:closing_marker].lstrip("\r\n").rstrip()
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        updated["completion"] = value
        updated["code"] = code
        filtered.append(updated)
    return filtered

def main() -> None:
    config = AppConfig.load("config/test-remote-processor.yaml")
    with SpiderClient(config=config, processor=filter_rows) as client:
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