from __future__ import annotations

from typing import List

from spider.config import SourceConfig

def collect_prompts(source: SourceConfig) -> List[str]:
    return _load_hf_dataset(source)

def _load_hf_dataset(source: SourceConfig) -> List[str]:
    from datasets import load_dataset

    load_kwargs = {
        "path": source.dataset,
        "split": source.split,
        "streaming": source.streaming,
    }
    if source.config_name:
        load_kwargs["name"] = source.config_name
    if source.revision:
        load_kwargs["revision"] = source.revision
    dataset = load_dataset(
        **load_kwargs,
        **source.options,
    )
    prompts: List[str] = []
    for example in dataset:
        if source.field is None:
            prompts.append(str(example))
        else:
            value = example[source.field]
            prompts.append(value if isinstance(value, str) else str(value))
    return prompts