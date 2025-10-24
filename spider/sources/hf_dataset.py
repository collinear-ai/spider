from __future__ import annotations

from typing import Any, Dict, Optional, Iterator, Mapping

from datasets import load_dataset

from .base import DataSource

class HFDatasetSource(DataSource):
    def __init__(
        self, dataset: str, *,
        split: str = "train",
        field: Optional[str] = None,
        streaming: bool = False,
        revision: Optional[str] = None,
        **load_kwargs: Any,
    ):
        self.dataset = dataset
        self.split = split
        self.field = field
        self.streaming = streaming
        self.revision = revision
        self.load_kwargs = load_kwargs

    def records(self) -> Iterator[Mapping[str, Any]]:
        load_args = dict(self.load_kwargs)
        if self.revision is not None:
            load_args["revision"] = self.revision
        
        dataset = load_dataset(
            path=self.dataset,
            split=self.split,
            streaming=self.streaming,
            **load_args
        )

        for example in dataset:
            if not isinstance(example, Mapping):
                raise ValueError(f"Expected mapping records from dataset {self.dataset}, received {type(example)}.")

            if self.field is None:
                yield example
                continue

            if self.field not in example:
                raise KeyError(f"Field {self.field} missing in dataset record for {self.dataset}.")

            field_value = example[self.field]
            if isinstance(field_value, Mapping):
                yield field_value
            else:
                yield {self.field: field_value}