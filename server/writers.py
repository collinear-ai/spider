import json
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Iterator, Tuple

_MISSING = object()

class JSONLBatchWriter:
    def __init__(self, path: Path):
        self._path = path
        self._handle = None
        self._count = 0

    def __enter__(self) -> "JSONLBatchWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle:
            self._handle.close()

    @property
    def count(self) -> int:
        return self._count

    def write_batch(self, prompts: Iterable[str], generations: Iterable[str]) -> int:
        if self._handle is None:
            raise RuntimeError("JSONLBatchWriter must be used as a context manager")
        written = 0
        for prompt, completion in zip_longest(prompts, generations, fillvalue=_MISSING):
            if prompt is _MISSING or completion is _MISSING:
                raise ValueError("Prompts and generations must have the same length")
            payload = json.dumps({"prompt": prompt, "response": completion}, ensure_ascii=False)
            self._handle.write(payload + "\n")
            written += 1
        self._handle.flush()
        self._count += written
        return written

def iter_pairs(prompts: Iterable[str], generations: Iterable[str]) -> Iterator[Tuple[str, str]]:
    for prompt, completion in zip(prompts, generations):
        yield prompt, completion
