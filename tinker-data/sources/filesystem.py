from __future__ import annotations

import json, csv
from pathlib import Path
from typing import Iterator, Mapping, Any, Optional, Dict

from .base import DataSource

class JSONLSource(DataSource):
    def __init__(self, path: str):
        self.path = Path(path)

    def records(self) -> Iterator[Mapping[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL file source does not exist: {self.path}")

        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON record at line {line_number} n {self.path}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object at line {line_number} in {self.path}")
                yield payload


class CSVSource(DataSource):
    def __init__(self, path: str, *, dialect: Optional[str], **csv_options: Any):
        self.path = Path(path)
        self.dialect = dialect
        self.csv_options = csv_options

    def records(self) -> Iterator[Mapping[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV source does not exist: {self.path}")
        
        reader_options = dict(self.csv_options)
        if self.dialect is not None:
            reader_options["dialect"] = self.dialect

        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, **reader_options)
            if reader.fieldnames is None:
                raise ValueError(f"No header row found in CSV file: {self.path}")
            for row in reader:
                yield row