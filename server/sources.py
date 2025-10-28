from __future__ import annotations

from typing import Iterable, List, Sequence

from spider.config import SourceConfig, SourceType

def collect_prompts(sources: Sequence[SourceConfig]) -> List[str]:
    prompts: List[str] = []
    for source in sources:
        if source.type == SourceType.HF_DATASET:
            prompts.extend(_load_hf_dataset(source))
        elif source.type == SourceType.REMOTE_URI:
            prompts.extend(_load_remote_uri(source))
        else:
            raise ValueError(f"Unsupported source type: {source.type}")
    return prompts

def _load_hf_dataset(source: SourceConfig) -> List[str]:
    from datasets import load_dataset

    dataset = load_dataset(
        path=source.dataset,
        split=source.split,
        streaming=source.streaming,
        revision=source.revision.
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

def _load_remote_uri(source: SourceConfig) -> List[str]:
    import requests

    response = requests.get(source.uri, timeout=source.options.get("timeout", 30))
    response.raise_for_status()
    body = response.text.strip()
    if not body:
        return []
    if source.options.get("splitlines", True):
        return [line for line in body.splitlines() if line.strip()]
    return [body]