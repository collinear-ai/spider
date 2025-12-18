from __future__ import annotations

from typing import List, Any, Callable, Dict, Optional
from tqdm.auto import tqdm
import random

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
    max_examples = source.max_examples
    shuffle_requested = source.shuffle

    total = None
    iterable = dataset
    if tqdm and not source.streaming:
        try:
            total = len(dataset)
        except Exception:
            total = None
        iterable = tqdm(dataset, total=total, desc="Collecting prompts", leave=False)

    collected = 0
    for i, example in enumerate(iterable):
        record = dict(example)
        if pre_processor:
            prompt = pre_processor(record)
            if prompt is None:
                continue
            value = prompt if isinstance(prompt, str) else str(prompt)
        elif source.field is None:
            value = str(record)
        else:
            value = example[source.field]
            value = value if isinstance(value, str) else str(value)
        
        collected += 1
        if max_examples is None or not shuffle_requested:
            prompts.append(value)
            if max_examples is not None and len(prompts) >= max_examples:
                break
            continue

        # reservoir sampling
        if len(prompts) < max_examples:
            prompts.append(value)
        else:
            idx = random.randint(0, collected - 1)
            if idx < max_examples:
                prompts[idx] = value

        if shuffle_requested:
            random.shuffle(prompts) # shuffle the biased order in reservoir sampling
    return prompts