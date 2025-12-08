from __future__ import annotations

import argparse, json, asyncio, hashlib, logging
from pathlib import Path
from typing import Iterable, List

from datasets import Dataset, load_dataset

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.renderers import Message, TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
)

LOGGER = logging.getLogger(__name__)

class OpenThoughts3DatasetBuilder(ChatDatasetBuilder):
    dataset_name: str
    split: str
    
    def __call__(self):
        LOGGER.info("Loading %s split=%s", self.dataset_name, self.split)
        ds = load_dataset(self.dataset_name, split=self.split)

        renderer = self.renderer
        train_on = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_row(row: dict) -> renderers.tinker.Datum:
            convo = [
                Message(role="user", content=row.get("input", "")),
                Message(role="assistant", content=row.get("output", "")),
            ]
            return conversation_to_datum(
                convo,
                renderer,
                self.common_config.max_length,
                train_on,
            )

        dataset = SupervisedDatasetFromHFDataset(
            ds,
            batch_size=self.common_config.batch_size,
            map_fn=map_row,
        )
        return dataset, None

def build_dataset_builder(
    dataset_name,
    split,
    base_model,
    max_length,
    batch_size,
) -> ChatDatasetBuilder:
    renderer_name = model_info.get_recommended_renderer_name(base_model)
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=base_model,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    return OpenThoughts3DatasetBuilder(
        common_config=common,
        dataset_name=dataset_name,
        split=split,
    )

def run_training(
    dataset_name,
    split,
    base_model,
    max_length,
    batch_size,
    lr,
    num_epochs,
    save_every,
    eval_every,
) -> dict:
    dataset_builder = build_dataset_builder(
        dataset_name=dataset_name,
        split=split,
        base_model=base_model,
        max_length=max_length,
        batch_size=batch_size,
    )
    log_dir = Path("logs")
    run_log_dir = log_dir / "stage1_sft"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    config = train.Config(
        log_path=str(run_log_dir),
        model_name=base_model,
        dataset_builder=dataset_builder,
        learning_rate=lr,
        num_epochs=num_epochs,
        save_every=save_every,
        eval_every=eval_every,
    )
    LOGGER.info("Starting Stage 1 SFT training. Log at %s", run_log_dir)
    asyncio.run(train.main(config))

    checkpoint = checkpoint_utils.get_last_checkpoint(str(run_log_dir), required_key="state_path")
    if not checkpoint:
        raise RuntimeError("No checkpoint produced. Cannot continue.")
    return checkpoint

def write_manifest(manifest_path, checkpoint):
    payload = {
        "state_checkpoint": checkpoint.get("state_path"),
        "sampler_checkpoint": checkpoint.get("sampler_path"),
        "last_batch": checkpoint.get("batch"), 
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Stage 1 SFT training manifest written to %s", manifest_path)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    checkpoint = run_training(
        dataset_name="collinear-ai/OpenThoughts3-1.2M-code-only",
        split="random_50k[0:10000]",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        max_length=16384,
        batch_size=128,
        lr=8e-5,
        num_epochs=3,
        save_every=256,
        eval_every=256,
    )
    write_manifest(Path("logs/stage1_sft_manifest.json"), checkpoint)

if __name__ == "__main__":
    main()