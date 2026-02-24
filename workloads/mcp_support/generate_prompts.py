import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from spider.config import JobConfig
from server.executor import run_generation_job


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
ARTIFACT_DIR = BASE_DIR / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-in", required=True)
    parser.add_argument("--dataset-out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=None)
    return parser.parse_args()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("'").strip('"')


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_response(row: dict) -> str:
    trajectory = row.get("trajectory")
    if isinstance(trajectory, list) and trajectory:
        for turn in reversed(trajectory):
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                content = turn.get("content")
                if isinstance(content, str):
                    return content
    content = row.get("content")
    return content if isinstance(content, str) else ""


def extract_generated_questions(text: str) -> list[str]:
    if not text:
        return []
    matches = re.findall(r"<question>(.*?)</question>", text, flags=re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches if m.strip()]


def infer_batch_size(
    dataset: str,
    *,
    split: str,
    config_name: str | None,
    max_examples: int | None,
) -> int:
    kwargs = {"path": dataset, "split": split}
    if config_name:
        kwargs["name"] = config_name
    ds = load_dataset(**kwargs)
    size = len(ds)
    if max_examples:
        size = min(size, max_examples)
    return max(1, size)


def main() -> None:
    args = parse_args()
    load_env(ENV_PATH)

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is required.")
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise SystemExit("OPENROUTER_API_KEY is required.")

    source = {
        "dataset": args.dataset_in,
        "split": args.split,
        "field": "prompt",
    }
    if args.config_name:
        source["config_name"] = args.config_name
    if args.max_examples:
        source["max_examples"] = args.max_examples
    max_batch_size = args.max_batch_size or infer_batch_size(
        args.dataset_in,
        split=args.split,
        config_name=args.config_name,
        max_examples=args.max_examples,
    )

    job = JobConfig.model_validate(
        {
            "model": {
                "provider": "openrouter",
                "name": "moonshotai/kimi-k2-0905",
            },
            "source": source,
            "generation": {
                "max_batch_size": max_batch_size,
                "parameters": {
                    "temperature": 0.7,
                }
            },
            "output": {
                "mode": "return",
            },
        }
    )

    job_id = f"generate_prompts_{int(time.time())}"
    workspace = ARTIFACT_DIR / job_id
    result = run_generation_job(
        job_id=job_id,
        job=job,
        workspace=workspace,
        job_env={
            "HF_TOKEN": token,
            "OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"],
        },
    )

    rows = read_jsonl(result.artifacts_path)
    rows = [
        {
            **row,
            "generated_questions": extract_generated_questions(extract_response(row)),
        }
        for row in rows
    ]
    parquet_path = ARTIFACT_DIR / "train-00000-of-00001.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    api = HfApi(token=token)
    api.create_repo(repo_id=args.dataset_out, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo=parquet_path.name,
        repo_id=args.dataset_out,
        repo_type="dataset",
    )
    print(f"records={len(rows)}")
    print(f"max_batch_size={max_batch_size}")
    print(f"local_jsonl={result.artifacts_path}")
    print(f"local_parquet={parquet_path}")
    print(f"uploaded=hf://datasets/{args.dataset_out}/{parquet_path.name}")


if __name__ == "__main__":
    main()
