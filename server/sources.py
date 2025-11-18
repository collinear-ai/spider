from __future__ import annotations

from typing import List, Any, Callable, Dict, Optional
from tqdm.auto import tqdm

from spider.config import SourceConfig

PreProcessor = Callable[[Dict[str, Any]], Optional[str]]

def collect_prompts(
    source: SourceConfig,
    *,
    pre_processor: Optional[PreProcessor] = None,
) -> List[str]:
    return _load_hf_dataset(source, pre_processor=pre_processor)

def _load_hf_dataset(
    source: SourceConfig,
    *,
    pre_processor: Optional[PreProcessor] = None,
) -> List[str]:
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

    total = None
    iterable = dataset
    if tqdm and not source.streaming:
        try:
            total = len(dataset)
        except Exception:
            total = None
        iterable = tqdm(dataset, total=total, desc="Collecting prompts", leave=False)

    for example in iterable:
        record = dict(example)
        if pre_processor:
            print("Apply pre processor to record: ", record[:32])
            prompt = pre_processor(record)
            if prompt is None:
                continue
            prompts.append(prompt if isinstance(prompt, str) else str(prompt))
            continue
        if source.field is None:
            prompts.append(str(record))
        else:
            value = example[source.field]
            prompts.append(value if isinstance(value, str) else str(value))
    return prompts