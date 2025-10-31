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
        self, *, prompts: List[str], dataset_name: str,
        groups_per_batch: int, group_size: int,
        renderer_name: str, model_name_for_tokenizer: str,
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
    pass

def build_sampling_params(
    *, max_tokens: int, generation_parameters: Mapping[str, object],
) -> tinker.SamplingParams:
    allowed = {"temperature", "top_p"}
    kwargs = {}
    for key in allowed:
        kwargs[key] = generation_parameters.get(key)
    return tinker.SamplingParams(max_tokens=max_tokens, **kwargs)

async def _emit_student_generations(
    *, prompts: Iterable[str], sampling_client: tinker.SamplingClient,
    tokenizer, sampling_params: tinker.SamplingParams,
    writer: JSONLBatchWriter, metadata_payload: Dict[str, object],
    metdata_path: Path, summarize_metrics, write_metadata,
) -> None:
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt)
        model_input = tinker.ModelInput.from_ints(prompt_ids)
        result = await sampling_client.sample_async(
            prompt=model_input, sampling_params=sampling_params,
            num_samples=1,
        )
        if not result.sequences:
            continue
        completion = tokenizer.decode(result.sequences[0].tokens)
        writer.write_records([{
            "prompt": prompt,
            "completion": completion,
        }])
        metadata_payload["metrics"] = summarize_metrics(writer.count, {})
        write_metadata(metadata_path, metadata_payload, writer.count)

def run_on_policy_job(
    job_id: str, job: JobConfig, *, workspace: Path
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
    
    prompts = collect_prompts(job_id, job)
    payload["generation_mode"] = "on_policy"
    sanitized_options = options.model_dump(exclude_none=True).copy()
    sanitized_options.pop("api_key", None)
    payload["on_policy"] = sanitized_options
    _write_metadata(metadata_path, payload, 0)

    if not prompts:
        artifact_path.write_text("", encoding="utf-8")
        payload["metrics"] = {"records": 0}
        _write_metadata(metadata_path, payload, 0)
        return JobExecutionResult(
            artifacts_path=artifact_path,
            metrics={"records": 0},
            messages=["No prompts found, skipped on-policy distillation."]
        )
    
    renderer_name = model_info.get_recommended_renderer_name(student_model)
    dataset_builder = PromptListDatasetBuilder(
        prompts=prompts,
        dataset_name-job.source.dataset or "prompts",
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
    sampler_path = checkpoint["sampler_path"]
    logger.info("Sampling student outputs from checkpoint %s", sampler_path)

    tokenizer = get_tokenizer(student_model)
    sampling_params = _build_sampling_params(
        max_tokens=options.max_tokens,
        generation_parameters=job.generation.parameters,
    )
    with JSONLBatchWriter(artifact_path) as writer:
        with _temporary_api_key(options.api_key):
            service_client = tinker.ServiceClient()
            sampling_client = service_client.create_sampling_client(model_path=sampler_path)
            try:
                asyncio.run(
                    _emit_student_generations(
                        prompts=prompts,
                        sampling_client=sampling_client,
                        tokenizer=tokenizer,
                        sampling_params=sampling_params,
                        writer=writer,
                        metadata_payload=payload,
                        metadata_path=metadata_path,
                        summarize_metrics=_summarize_metrics,
                        write_metadata=_write_metadata,
                    )
                )
            except Exception as exc:
                rais JobExecutionError(f"Sampling student outputs failed: {exc}") from exc
        records_written = writer.count

    metrics = _summarize_metrics(
        records_written, 
        {"training_batches": float(checkpoint.get("batch", 0))},
    )
    payload["metrics"] = metrics
    _write_metadata(metadata_path, payload, records_written)

    hf_url = None
    if job.output.mode == OutputMode.HF_UPLOAD and job.output.hf:
        try:
            hf_url = publish_to_hub(
                job_id=job_id,
                artifact=artifact_path,
                metadata=metadata_path,
                config=job.output.hf
            )
        except HFUploadError as exc:
            raise JobExecutionError(str(exc)) from exc

    return JobExecutionResult(
        artifacts_path=artifact_path,
        remote_artifact=hf_url,
        metrics=metrics,
        messages=["On-policy distillation completed."]
    )