import sys
from pathlib import Path
import yaml

from spider.config import JobConfig
from workloads.swe_rebench_openhands.runner import run_server_only

def _load_job_config(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    job = raw.get("job")
    return JobConfig.model_validate(job)

def main():
    job = _load_job_config(Path("config/train_on_policy_swe.yaml"))
    run_server_only(
        job=job,
        workspace=Path("./workspace"),
        split="filtered",
    )

if __name__ == "__main__":
    main()