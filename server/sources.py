from __future__ import annotations

from typing import List, Any, Callable, Dict, Optional
from tqdm.auto import tqdm
import random

from spider.config import SourceConfig

PreProcessor = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]

def collect_prompts(
    source: SourceConfig,
    *,
    pre_processor: Optional[PreProcessor] = None,
) -> List[Dict[str, Any]]:
    return _load_hf_dataset(source, pre_processor=pre_processor)

def _load_hf_dataset(
    source: SourceConfig,
    *,
    pre_processor: Optional[PreProcessor] = None,
) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    field = source.field or "prompt"

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

    prompts = []
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
        row = record
        if pre_processor:
            row = pre_processor(record)
            if row is None:
                continue
            if not isinstance(row, dict):
                raise ValueError("Pre-processor must return a dict (can be empty) per record.")
        if field not in row:
            raise ValueError(f"`source.field` was set to `{field}` but the field is missing.")
        value = row[field]
        value = value if isinstance(value, str) else str(value)
        row = _build_prompt_record(row, prompt=value, drop_field=field)
        
        collected += 1
        if max_examples is None or not shuffle_requested:
            prompts.append(row)
            if max_examples is not None and len(prompts) >= max_examples:
                break
            continue

        # reservoir sampling
        if len(prompts) < max_examples:
            prompts.append(row)
        else:
            idx = random.randint(0, collected - 1)
            if idx < max_examples:
                prompts[idx] = row

        if shuffle_requested:
            random.shuffle(prompts) # shuffle the biased order in reservoir sampling
    return prompts

def _build_prompt_record(
    record: Dict[str, Any],
    *,
    prompt: str,
    drop_field: Optional[str] = None,
) -> Dict[str, Any]:
    ordered = {"prompt": prompt}
    for key, value in record.items():
        if key == "prompt":
            continue
        if drop_field is not None and key == drop_field:
            continue
        ordered[key] = value
    return ordered