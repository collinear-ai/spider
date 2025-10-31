from __future__ import annotations

import asyncio, logging, os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import tinker
from spider.config import JobConfig, OutputMode
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDataset,
    TeacherConfig,
)
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .hf_upload import HFUploadError, publish_to_hub
from .sources import collect_prompts
from .writers import JSONLBatchWriter

logger = logging.getLogger(__name__)

class PromptListDatasetBuilder(RLDatasetBuilder):
    def __init__(
        self, 
        *, 
        prompts: List[str], 
        dataset_name: str,
        groups_per_batch: int, 
        group_size: int,
        renderer_name: str, 
        model_name_for_tokenizer: str,
    ) -> None:
        self._prompts = prompts
        self._dataset_name = dataset_name
        self._groups_per_batch = groups_per_batch
        self._group_size = group_size
        self._renderer_name = renderer_name
        self._model_name = model_name_for_tokenizer

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tokenizer = get_tokenizer(self._model_name)
        renderer = renderers.get_renderer(self._renderer_name, tokenizer=tokenizer)
        dataset = PromptOnlyDataset(
            prompts=self._prompts,
            batch_size=self._groups_per_batch,
            group_size=self._group_size,
            renderer=renderer,
            tokenizer=tokenizer,
            max_prompt_tokens=None,
            convo_prefix=None,
            dataset_name=self._dataset_name,
        )
        return dataset, None

@contextmanager
def _temporary_api_key(api_key: str | None):
    if not api_key:
        yield
        return

    previous = os.environ.get("TINKER_API_KEY")
    if previous == api_key:
        yield
        return

    os.environ["TINKER_API_KEY"] = api_key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TINKER_API_KEY", None)
        else:
            os.environ["TINKER_API_KEY"] = previous

def run_on_policy_job(
    job_id: str, 
    job: JobConfig, 
    *, 
    workspace: Path
):
    from .executor import (
        JobExecutionResult,
        JobExecutionError,
        _base_metadata,
        _summarize_metrics,
        _write_metadata,
    )
    options = job.generation.on_policy_options
    if not options:
        raise JobExecutionError("on_policy_options must be provided")
    
    student_model = job.model.name
    if not student_model:
        raise JobExecutionError("job.model.name must be provided")

    artifact_path = workspace / "result.jsonl"
    metadata_path = workspace / "metadata.json"
    
    prompts = collect_prompts(job.source)
    logger.info(
        "Job %s: on-policy distillation for student=%s collected %d prompts",
        job_id,
        student_model,
        len(prompts)
    )

    payload = _base_metadata(job_id, job)
    payload["generation_mode"] = "on_policy"
    sanitized_options = options.model_dump(exclude_none=True).copy()
    sanitized_options.pop("api_key", None)
    payload["on_policy"] = sanitized_options
    _write_metadata(metadata_path, payload, 0)

    if not prompts:
        artifact_path.write_text("", encoding="utf-8")
        payload["metrics"] = {"records": 0}
        _write_metadata(metadata_path, payload, 0)

        logger.info("Job %s: no prompts found. skipping on-policy distillation.", job_id)
        return JobExecutionResult(
            artifacts_path=artifact_path,
            metrics={"records": 0},
            messages=["No prompts found, skipped on-policy distillation."]
        )
    
    renderer_name = model_info.get_recommended_renderer_name(student_model)
    dataset_builder = PromptListDatasetBuilder(
        prompts=prompts,
        dataset_name=(job.source.dataset or "prompts"),
        groups_per_batch=options.groups_per_batch,
        group_size=options.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=student_model,
    )

    teacher_config = TeacherConfig(base_model=options.teacher)
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=options.groups_per_batch,
    )

    training_dir = workspace / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Job %s: starting Tinker training (teacher=%s, prompts=%d) logs=%s",
        job_id,
        options.teacher,
        len(prompts),
        training_dir
    )

    train_config = train_on_policy.Config(
        learning_rate=options.learning_rate,
        dataset_configs=[dataset_config],
        model_name=student_model,
        max_tokens=options.max_tokens,
        compute_post_kl=options.compute_post_kl,
        lora_rank=options.lora_rank,
        evaluator_builders=[],
        kl_penalty_coef=options.kl_penalty_coef,
        kl_discount_factor=options.kl_discount_factor,
        loss_fn=options.loss_fn,
        num_substeps=options.num_substeps,
        wandb_project=None,
        wandb_name=None,
        log_path=str(training_dir),
        base_url=None,
        enable_trace=False,
        eval_every=options.eval_every,
        save_every=options.save_every,
        load_checkpoint_path=None,
    )

    _configure_tinker_logging()

    with _temporary_api_key(options.api_key):
        try:
            asyncio.run(train_on_policy.main(train_config))
        except Exception as exc:
            raise JobExecutionError(f"On-policy distillation failed: {exc}") from exc

    checkpoint = checkpoint_utils.get_last_checkpoint(
        str(training_dir), required_key="sampler_path"
    )
    if not checkpoint or "sampler_path" not in checkpoint:
        raise JobExecutionError("On-policy training did not produce a sampler checkpoint")

    artifact_path.write_text("", encoding="utf-8")
    metrics = {
        "records": 0,
        "training_batches": float(checkpoint.get("batch", 0))
    }
    payload["metrics"] = metrics
    payload["training"] = {
        "sampler_checkpoint": checkpoint.get("sampler_path"),
        "state_checkpoint": checkpoint.get("state_path"),
        "last_batch": checkpoint.get("batch"),
        "teacher_model": options.teacher,
    }
    _write_metadata(metadata_path, payload, 0)

    logger.info(
        "Job %s: training finished at batch=%s, sampler=%s",
        job_id,
        checkpoint.get("batch"),
        checkpoint.get("sampler_path")
    )
    return JobExecutionResult(
        artifacts_path=artifact_path,
        metrics=metrics,
        messages=["On-policy distillation completed."]
    )

def _configure_tinker_logging() -> None:
    stream_formatter = logging.Formatter("[tinker] %(message)s")
    for name in ("tinker", "tinker_cookbook"):
        tinker_logger = logging.getLogger(name)
        tinker_logger.setLevel(logging.INFO)
        if not any(getattr(handler, "_spider_tinker", False) for handler in tinker_logger.handlers):
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(stream_formatter)
            handler._spider_tinker = True
            tinker_logger.addHandler(handler)