import json
from pathlib import Path
from typing import Iterable, List

def write_jsonl(path: Path, prompts: Iterable[str], generations: Iterable[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for prompt, completion in zip(prompts, generations):
            record = {"prompt": prompt, "response": completion}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count