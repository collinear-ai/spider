import json
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Iterator, Tuple, Dict

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

    def write_records(self, records: Iterable[Dict[str, object]]) -> int:
        if self._handle is None:
            raise RuntimeError("JSONLBatchWriter must be used as a context manager")
        written = 0
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Records must be a mapping")
            payload = json.dumps(record, ensure_ascii=False)
            self._handle.write(payload + "\n")
            written += 1
        self._handle.flush()
        self._count += written
        return written
