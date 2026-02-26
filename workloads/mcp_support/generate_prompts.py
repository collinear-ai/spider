import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from spider.config import ModelConfig
from server.backends.factory import create_backend
from tqdm.auto import tqdm


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
    parser.add_argument("--max-workers", type=int, default=1024)
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


def main() -> None:
    args = parse_args()
    load_env(ENV_PATH)

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is required.")
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise SystemExit("OPENROUTER_API_KEY is required.")

    ds_kwargs = {"path": args.dataset_in, "split": args.split}
    if args.config_name:
        ds_kwargs["name"] = args.config_name
    dataset = load_dataset(**ds_kwargs)
    rows = [dict(x) for x in dataset]
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    model = ModelConfig.model_validate(
        {
            "provider": "openrouter",
            "name": "moonshotai/kimi-k2-0905",
        }
    )
    backend = create_backend(model)
    parameters = {"temperature": 1.0}

    def run_one(row: dict) -> dict | None:
        messages = []
        system_prompt = row.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": row["prompt"]})
        try:
            resp = backend.chat(messages=messages, parameters=parameters)
        except Exception:
            return None
        record = {"prompt": row["prompt"]}
        if resp.get("content"):
            record["content"] = resp["content"]
        if resp.get("reasoning"):
            record["reasoning"] = resp["reasoning"]
        for key, value in row.items():
            if key in {"prompt", "content", "reasoning", "trajectory"}:
                continue
            record[key] = value
        return record

    max_workers = max(1, min(args.max_workers, len(rows)))
    results: list[dict] = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, row) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="generation", unit="row"):
            record = future.result()
            if record is None:
                skipped += 1
                continue
            results.append(record)

    job_id = f"generate_prompts_{int(time.time())}"
    workspace = ARTIFACT_DIR / job_id
    workspace.mkdir(parents=True, exist_ok=True)
    result_path = workspace / "result.jsonl"
    with result_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = results
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
    print(f"skipped={skipped}")
    print(f"local_jsonl={result_path}")
    print(f"local_parquet={parquet_path}")
    print(f"uploaded=hf://datasets/{args.dataset_out}/{parquet_path.name}")


if __name__ == "__main__":
    main()
