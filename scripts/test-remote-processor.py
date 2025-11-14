import re, ast
from typing import Iterable, Dict, Any, List, Optional

from spider.client import SpiderClient
from spider.config import AppConfig

# == sample post processor ==

LANG_MARKER = "```python"

def _extract_code_block(text):
    lower_text = text.lower()
    start_marker = lower_text.rfind(LANG_MARKER)
    if start_marker == -1:
        return None
    code_start = start_marker + len(LANG_MARKER)
    closing_marker = text.find("```", code_start)
    if closing_marker == -1:
        return None
    snippet = text[code_start:closing_marker].lstrip("\r\n").rstrip()
    try:
        ast.parse(snippet)
    except SyntaxError:
        return None
    return snippet

def filter_row(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    params:
    -- record (dict): a single rollout record with a "completion" field (str)

    return:
    -- enriched record (dict): the updated record with arbitrary fields added / edited
    -- None: if the record is unwanted
    """
    completion = record.get("completion")
    if not isinstance(completion, str):
        return None
    snippet = _extract_code_block(completion)
    if snippet is None:
        return None
    enriched = dict(record)
    enriched["code"] = snippet
    return enriched

# == main function for client call ==

def main() -> None:
    config = AppConfig.load("config/test-remote-processor.yaml")
    with SpiderClient(config=config, processor=filter_row) as client:
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