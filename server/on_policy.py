from __future__ import annotations

import asyncio, logging, os, shutil, tarfile, urllib.request, zipfile
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Any, Iterable, Optional, Callable, TypedDict
from urllib.parse import urlparse
import blobfile

import tinker
from spider.config import JobConfig
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDataset,
    TeacherConfig,
)
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .on_policy_utils import compute_teacher_alignment_for_rewards
from .sources import collect_prompts
from . import events
from .writers import JSONLBatchWriter

RolloutBatch = List[Dict[str, Any]]

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
        object.__setattr__(self, "_prompts", prompts)
        object.__setattr__(self, "_dataset_name", dataset_name)
        object.__setattr__(self, "_groups_per_batch", groups_per_batch)
        object.__setattr__(self, "_group_size", group_size)
        object.__setattr__(self, "_renderer_name", renderer_name)
        object.__setattr__(self, "_model_name", model_name_for_tokenizer)

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

class _StudentSamplerContext:
    def __init__(self, *, job_id: str, job: JobConfig) -> None:
        self._job_id = job_id
        self._job = job
        self._client = None
        self._student_client = None
        self._service_client = None

    def __enter__(self) -> Any:
        from .executor import JobExecutionError

        if self._student_client is not None:
            return self._student_client

        model_name = getattr(self._job.model, "name", None)
        checkpoint_path = getattr(self._job.model, "student_checkpoint_path", None)
        if not model_name:
            raise JobExecutionError(
                "job.model.name must be provided to create a Tinker sampling client."
            )

        self._service_client = tinker.ServiceClient()
        if checkpoint_path:
            self._student_client = self._service_client.create_sampling_client(
                model_path=checkpoint_path,
            )
        else:
            self._student_client = self._service_client.create_sampling_client(
                base_model=model_name,
            )

        return self._student_client

    def refresh_from_sampler_path(self, sampler_path: str) -> Any:
        if not self._service_client:
            raise RuntimeError("Student sampler is not initialized.")

        self._student_client = self._service_client.create_sampling_client(
            model_path=sampler_path,
        )
        return self._student_client

    def __exit__(self, exc_type, exc, tb) -> None:
        self._student_client = None
        self._service_client = None
        
def _create_shared_student_sampler(
    *,
    job_id: str,
    job: JobConfig,
) -> tuple[_StudentSamplerContext, Any]:
    ctx = _StudentSamplerContext(job_id=job_id, job=job)
    client = ctx.__enter__()
    return ctx, client

def _needs_gold_alignment(student_model: str, teacher_model: str) -> bool:
    _ = (student_model, teacher_model)
    return True

def run_on_policy_job(
    job_id: str, 
    job: JobConfig, 
    *, 
    workspace: Path,
    job_env: dict[str, str],
    prompts: Optional[List[Dict[str, Any]]] = None,
    tool_registry: Optional[Dict[str, Callable[..., Any]]] = None,
):
    from .executor import (
        JobExecutionResult,
        JobExecutionError,
        _base_metadata,
        _resolve_processor,
        _summarize_metrics,
        _write_metadata,
        _shutdown_backend,
        _tool_batch_worker,
    )
    options = job.generation.on_policy_options
    if not options:
        raise JobExecutionError("on_policy_options must be provided")
    
    student_model = job.model.name
    student_checkpoint = job.model.student_checkpoint_path
    if not student_model:
        raise JobExecutionError("job.model.name must be provided")

    artifact_path = workspace / "result.jsonl"
    metadata_path = workspace / "metadata.json"
    
    pre_processor = _resolve_processor(job.pre_processor) if job.pre_processor else None
    prompt_rows = prompts if prompts is not None else collect_prompts(job.source, pre_processor=pre_processor)
    prompt_list = [row["prompt"] for row in prompt_rows]

    use_gold_alignment = _needs_gold_alignment(student_model, options.teacher)
    logger.info(
        "Job %s: on-policy distillation for student=%s collected %d prompts, use_gold_alignment=%s, tool_rollouts=%s",
        job_id,
        student_model,
        len(prompt_list),
        use_gold_alignment,
        bool(tool_registry),
    )

    sampler_ctx = None
    student_client = None
    rollout_stream = None
    if tool_registry:
        sampler_ctx, student_client = _create_shared_student_sampler(job_id=job_id, job=job)

        rollout_stream = _tool_rollout_stream(
            job_id=job_id,
            job=job,
            student_client=student_client,
            prompts=prompt_list,
            tool_registry=tool_registry,
        )
        events.emit(
            "Finished Setup for tool rollouts streaming for on-policy distillation.",
            code="on_policy.tool_rollouts_stream"
        )
    else:
        events.emit(
            "Collected prompts for on-policy distillation.",
            code="on_policy.prompts_collected",
            data={"total_prompts": len(prompt_list)}
        )

    payload = _base_metadata(job_id, job)
    payload["generation_mode"] = "on_policy"
    on_policy_options = options.model_dump(exclude_none=True)
    on_policy_options["use_gold_alignment"] = use_gold_alignment
    payload["on_policy"] = on_policy_options
    _write_metadata(metadata_path, payload, 0)

    if not prompt_list:
        artifact_path.write_text("", encoding="utf-8")
        payload["metrics"] = {"records": 0}
        _write_metadata(metadata_path, payload, 0)

        logger.info("Job %s: no prompts found. skipping on-policy distillation.", job_id)
        events.emit(
            "No prompts found for on-policy distillation.",
            level="warning",
            code="on_policy.no_prompts",
        )
        result = JobExecutionResult(
            artifacts_path=artifact_path,
            metrics={"records": 0},
            messages=["No prompts found, skipped on-policy distillation."]
        )
        if sampler_ctx:
            sampler_ctx.__exit__(None, None, None)
        return result
    
    if rollout_stream:
        checkpoint_info = _run_tool_on_policy_stream(
            job_id=job_id,
            job=job,
            options=options,
            workspace=workspace,
            rollout_stream=rollout_stream,
            sampler_ctx=sampler_ctx,
            metadata_path=metadata_path,
            artifact_path=artifact_path,
        )
        checkpoint = checkpoint_info["checkpoint"]
        training_dir = checkpoint_info["training_dir"]
    
    else:   
        renderer_name = model_info.get_recommended_renderer_name(student_model)

        dataset_builder: RLDatasetBuilder
        dataset_builder = PromptListDatasetBuilder(
            prompts=prompt_list,
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
            len(prompt_list),
            training_dir
        )
        events.emit(
            "Starting on-policy distillation training.",
            code="on_policy.training_start",
            data={"teacher": options.teacher}
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
            wandb_project=None,
            wandb_name=None,
            log_path=str(training_dir),
            base_url=None,
            enable_trace=False,
            save_every=options.save_every,
            load_checkpoint_path=student_checkpoint,
            use_gold_alignment=use_gold_alignment,
        )

        _configure_tinker_logging()

        try:
            asyncio.run(train_on_policy.main(train_config))
        except Exception as exc:
            raise JobExecutionError(f"On-policy distillation failed: {exc}") from exc

        checkpoint = checkpoint_utils.get_last_checkpoint(
            str(training_dir), required_key="sampler_path"
        )
        if not checkpoint or "sampler_path" not in checkpoint:
            raise JobExecutionError("On-policy training did not produce a sampler checkpoint")

    hf_payload_dir, manifest, checkpoints_index_text = _prepare_hf_payload(
        training_dir=training_dir,
        checkpoint=checkpoint,
        workspace=workspace,
    )
    if not manifest:
        raise JobExecutionError("On-policy training did not produce uploadable checkpoint artifacts")
        
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
    if checkpoints_index_text is not None:
        payload["training"]["checkpoints_index_contents"] = checkpoints_index_text
    payload["hf_upload"] = {
        "repo_id": job.output.hf.repo_id if job.output.hf else None,
        "relative_dir": hf_payload_dir.relative_to(workspace).as_posix(),
        "manifest": manifest,
    }
    _write_metadata(metadata_path, payload, 0)

    summary_record = {
        "job_id": job_id,
        "metrics": metrics,
        "hf_manifest": manifest,
        "hf_payload_dir": hf_payload_dir.relative_to(workspace).as_posix(),
    }
    if job.output.hf:
        summary_record["hf_repo_id"] = job.output.hf.repo_id
    with JSONLBatchWriter(artifact_path) as writer:
        writer.write_records([summary_record])

    logger.info(
        "Job %s: training finished at batch=%s, sampler=%s",
        job_id,
        checkpoint.get("batch"),
        checkpoint.get("sampler_path")
    )
    events.emit(
        "On-policy distillation completed.",
        code="on_policy.completed",
        data={"training_batches": metrics["training_batches"]}
    )
    result = JobExecutionResult(
        artifacts_path=artifact_path,
        metadata_path=metadata_path,
        upload_source=hf_payload_dir,
        metrics=metrics,
        messages=["On-policy distillation completed."]
    )
    if sampler_ctx:
        sampler_ctx.__exit__(None, None, None)
    return result

def _run_tool_on_policy_stream(
    *,
    job_id: str,
    job: JobConfig,
    options: Any,
    workspace: Path,
    rollout_stream: Iterable[List[Dict[str, Any]]],
    sampler_ctx: _StudentSamplerContext,
    metadata_path: Path,
    artifact_path: Path,
):
    import torch
    from .executor import (
        JobExecutionError,
        tool_descriptors
    )
    from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
    from .executor import _initial_chat_history, _execute_tool_calls, _call_backend_chat

    tool_defs = tool_descriptors(job.tools)
    service_client = tinker.ServiceClient()
    teacher_client = service_client.create_sampling_client(base_model=options.teacher)
    training_client = service_client.create_lora_training_client(
        base_model=job.model.name,
        rank=options.lora_rank,
    )

    training_dir = workspace / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    async def _process_batch(batch_index: int, trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items = []
        data_D = []

        async def _process_turn(
            turn_index: int, 
            turn: Dict[str, Any]
        ) -> tuple[Dict[str, Any], tinker.Datum]:
            messages = turn["messages"]
            token_ids = turn["token_ids"]
            logprobs = turn["logprobs"]
            reward_mask = turn["reward_mask"]

            student_logprobs = torch.tensor(logprobs, dtype=torch.float32)
            
            teacher_alignment = await compute_teacher_alignment_for_rewards(
                sampling_client=teacher_client,
                messages=messages,
                tools=tool_defs,
                teacher_model=options.teacher,
                student_model=job.model.name,
                student_token_ids=token_ids,
                student_logprobs=student_logprobs,
                reward_mask=reward_mask,
                assistant_raw_text=turn.get("assistant_raw_text"),
            )

            item = dict(turn)
            item.update({
                "student_logprobs": student_logprobs,
                "teacher_logprobs": teacher_alignment.get("teacher_logprobs"),
                "kl_adjustments": teacher_alignment.get("kl_adjustments"),
                "kl_mask": teacher_alignment.get("kl_mask"),
                "teacher_token_ids": teacher_alignment.get("teacher_token_ids"),
            })

            kl_adj = teacher_alignment.get("kl_adjustments") or [0.0] * len(token_ids)
            kl_mask = teacher_alignment.get("kl_mask") or [0.0] * len(token_ids) 
            if len(kl_mask) != len(reward_mask):
                raise JobExecutionError("KL mask length must match reward mask length.")
            if any(int(a) != int(b) for a, b in zip(kl_mask, reward_mask)):
                raise JobExecutionError("KL mask must match reward mask positions.")
            
            kl_tensor = torch.tensor(kl_adj, device=student_logprobs.device, dtype=student_logprobs.dtype)
            
            kl_coef = float(getattr(options, "kl_penalty_coef", 1.0))
            kl_discount = float(getattr(options, "kl_discount_factor", 0.0))
            advantage = -kl_coef * kl_tensor

            if kl_discount > 0:
                advantage = torch.tensor(
                    discounted_future_sum_vectorized(advantage.detach().cpu().numpy(), kl_discount),
                    device=student_logprobs.device,
                    dtype=student_logprobs.dtype,
                )

            datum = tinker.Datum( # turn kl loss into CE loss with logprobs + kl delta
                model_input=tinker.ModelInput.from_ints(list(token_ids)),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_list(list(token_ids)),
                    "mask": tinker.TensorData.from_list(list(reward_mask)),
                    "logprobs": tinker.TensorData.from_torch(student_logprobs),
                    "advantages": tinker.TensorData.from_torch(advantage),
                } # use importance sampling as a surrogate
            )

            logger.info(
                "Job %s: tool rollout batch=%d traj=%d tokens=%d reward_tokens=%d",
                job_id,
                batch_index,
                turn_index,
                len(token_ids),
                sum(reward_mask),
            )

            return item, datum

        results = await asyncio.gather(*[asyncio.create_task(_process_turn(i, turn)) for i, turn in enumerate(trajectories)])
        for item, datum in results:
            items.append(item)
            data_D.append(datum)
        
        # train!
        fwd_bwd_future = await training_client.forward_backward_async(
            data_D,
            loss_fn=getattr(options, "loss_fn", "importance_sampling"),
        )
        _ = await fwd_bwd_future.result_async()

        adam_params = tinker.AdamParams(
            learning_rate=options.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        step_future = await training_client.optim_step_async(adam_params)
        await step_future.result_async()

        logger.info(
            "Job %s: tool rollout batch=%d training step complete.",
            job_id,
            batch_index,
        )
        events.emit(
            "Tool rollout batch KL training step complete.",
            code="tool_on_policy.training_step_complete",
            data={"batch_index": batch_index},
        )

        return items

    async def _train_stream():
        save_every = max(1, getattr(options, "save_every", 1))
        last_checkpoint = None
        for batch_index, trajectories in enumerate(rollout_stream):
            batch_items = await _process_batch(batch_index, trajectories)
            
            if (batch_index + 1) % save_every == 0:
                last_checkpoint = await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=f"{batch_index:06d}",
                    log_path=str(training_dir),
                    loop_state={"batch": batch_index + 1},
                    kind="sampler",
                )
                sampler_path = last_checkpoint.get("sampler_path")
                if sampler_path:
                    sampler_ctx.refresh_from_sampler_path(sampler_path)
                    logger.info(
                        "Job %s: refreshed student sampler from checkpoint at batch=%d path=%s.",
                        job_id,
                        batch_index,
                        sampler_path,
                    )
                    events.emit(
                        "Refreshed student sampler from checkpoint.",
                        code="tool_on_policy.sampler_refreshed",
                        data={"batch_index": batch_index, "sampler_path": sampler_path},
                    )
        return last_checkpoint
        
    last_checkpoint = asyncio.run(_train_stream())
    if not last_checkpoint:
        last_checkpoint = asyncio.run(
            checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name="final",
                log_path=str(training_dir),
                loop_state={"batch": 0},
                kind="sampler",
            )
        )
    if not last_checkpoint or "sampler_path" not in last_checkpoint:
        raise JobExecutionError("Tool on-policy training did not produce a sampler checkpoint.")

    return {
        "checkpoint": last_checkpoint,
        "training_dir": training_dir,
    }

def _tool_rollout_stream(
    *,
    job_id: str,
    job: JobConfig,
    student_client: Any,
    prompts: List[str],
    tool_registry: Dict[str, Callable[..., Any]],
) -> Iterable[List[Dict[str, Any]]]:
    from concurrent.futures import ThreadPoolExecutor
    from .executor import _initial_chat_history, _call_backend_chat, _execute_tool_calls, tool_descriptors, JobExecutionError

    tool_defs = tool_descriptors(job.tools)
    turn_limit = max(1, job.generation.max_turns or 16)
    batch_size = getattr(job.generation.on_policy_options, "groups_per_batch", 64) # TODO: rename to batch size
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    def _run_prompt(prompt: str) -> List[Dict[str, Any]]:
        history = _initial_chat_history(prompt, job)
        transcript = []
        turn_items = []

        for turn_idx in range(turn_limit):
            assistant_message = _call_backend_chat(
                backend=student_client,
                messages=history,
                tools=tool_defs,
                parameters=job.generation.parameters,
                include_logprobs=True,
                model_name=job.model.name,
            )
            token_ids = assistant_message.get("token_ids") or []
            logprobs = assistant_message.get("logprobs") or [0.0] * len(token_ids)
            reward_mask = assistant_message.get("reward_mask") or [1] * len(token_ids)

            if len(logprobs) != len(token_ids):
                raise JobExecutionError(
                    f"logprobs/token_ids mismatch: logprobs={len(logprobs)} tokens={len(token_ids)}"
                )

            if len(reward_mask) != len(token_ids):
                raise JobExecutionError(
                    f"reward_mask/token_ids mismatch: reward_mask={len(reward_mask)} tokens={len(token_ids)}"
                )

            bad_indices = [
                i for i, (lp, mask) in enumerate(zip(logprobs, reward_mask))
                if int(mask) == 1 and lp is None
            ]
            if bad_indices:
                raise JobExecutionError(
                    f"logprobs None on masked tokens (count={len(bad_indices)})"
                )

            none_count = sum(lp is None for lp in logprobs)
            masked_count = sum(1 for m in reward_mask if int(m) == 1)
            none_on_masked = sum(
                1 for lp, m in zip(logprobs, reward_mask)
                if lp is None and int(m) == 1
            )
            logger.info(
                "Job %s: logprobs None=%d masked=%d none_on_masked=%d",
                job_id,
                none_count,
                masked_count,
                none_on_masked,
            )

            content = assistant_message.get("content", "")
            tool_calls = assistant_message.get("tool_calls")

            snapshot = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
            reasoning = assistant_message.get("reasoning")
            if reasoning:
                snapshot["reasoning"] = reasoning

            messages = list(history) + [snapshot]
            turn_items.append(
                {
                    "prompt": prompt,
                    "messages": messages,
                    "token_ids": token_ids,
                    "logprobs": logprobs,
                    "reward_mask": reward_mask,
                    "assistant_content": content,
                    "assistant_reasoning": reasoning,
                    "assistant_tool_calls": tool_calls,
                    "assistant_raw_text": assistant_message.get("assistant_raw_text"),
                    "prompt_token_count": assistant_message.get("prompt_token_count", 0),
                    "parser_fallback": assistant_message.get("parser_fallback", False),
                    "turn_index": turn_idx,
                }
            )

            transcript.append(snapshot)
            history.append(snapshot)

            if not tool_calls:
                break

            _execute_tool_calls(
                tool_calls=tool_calls,
                tool_registry=tool_registry,
                transcript=transcript,
                history=history,
                job_id=job_id,
            )

        return turn_items

    for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
        chunk = prompts[start: start + batch_size]
        with ThreadPoolExecutor(max_workers=min(len(chunk), 8)) as pool:
            results = list(pool.map(_run_prompt, chunk))
        turns = [turn for per_prompt in results for turn in per_prompt]
        events.emit(
            "Tool rollout batch ready.",
            code="tool_on_policy.batch_ready",
            data={
                "batch_index": batch_index,
                "batch_size": len(chunk),
                "total_batches": total_batches,
            }
        )
        if turns:
            yield turns

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

def _prepare_hf_payload(
    *,
    training_dir: Path,
    checkpoint: Mapping[str, object],
    workspace: Path,
) -> Tuple[Path, Dict[str, str], str | None]:
    payload_dir = workspace / "hf_upload"
    if payload_dir.exists():
        shutil.rmtree(payload_dir)
    payload_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    sampler_rel = _copy_checkpoint_artifact(checkpoint.get("sampler_path"), payload_dir)
    if sampler_rel:
        manifest.update(_stage_lora_artifacts(payload_dir, sampler_rel))

    checkpoints_index = training_dir / "checkpoints.jsonl"
    checkpoints_index_text = None
    if checkpoints_index.exists():
        try:
            checkpoints_index_text = checkpoints_index.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read checkpoints index %s: %s", checkpoints_index, exc)

    return payload_dir, manifest, checkpoints_index_text

def _copy_checkpoint_artifact(path_value: object, dest_root: Path) -> str | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    if _is_remote_uri(path_value):
        try:
            dest = _download_remote_artifact(path_value, dest_root)
        except Exception as exc:
            logger.warning("Failed to download remote checkpoint artifact %s: %s", path_value, exc)
            return None
        if not dest:
            return None
        finalized = _extract_artifact_if_archive(dest)
        return finalized.relative_to(dest_root).as_posix() if dest else None

    src = Path(path_value)
    if not src.exists():
        logger.warning('Checkpoint artifact missing: %s', src)
        return None
    dest = dest_root / src.name
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    finalized = _extract_artifact_if_archive(dest)
    return finalized.relative_to(dest_root).as_posix()

def _is_remote_uri(path: str) -> bool:
    parsed = urlparse(path)
    if not parsed.scheme:
        return False
    if parsed.scheme == "file":
        return False
    if len(parsed.scheme) == 1 and parsed.scheme.isalpha():
        return False
    return True

def _download_remote_artifact(uri: str, dest_root: Path) -> Path | None:
    if _is_tinker_uri(uri):
        return _download_tinker_artifact(uri, dest_root)

    if not blobfile.exists(uri):
        logger.warning("Remote checkpoint artifact missing: %s", uri)
        return None
    
    name = Path(urlparse(uri).path.rstrip("/")).name or "artifact"
    dest = dest_root / name
    if blobfile.isdir(uri):
        _download_remote_directory(uri, dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with blobfile.BlobFile(uri, "rb") as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return _extract_artifact_if_archive(dest)

def _download_remote_directory(uri: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for entry in blobfile.scandir(uri):
        child_uri = f"{uri.rstrip('/')}/{entry.name}"
        child_dest = dest / entry.name
        if entry.is_dir():
            _download_remote_directory(child_uri, child_dest)
        else:
            child_dest.parent.mkdir(parents=True, exist_ok=True)
            with blobfile.BlobFile(child_uri, "rb") as src, child_dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)

def _is_tinker_uri(uri: str) -> bool:
    return urlparse(uri).scheme == "tinker"

def _download_tinker_artifact(uri: str, dest_root: Path) -> Path | None:
    parsed = urlparse(uri)
    run_id = parsed.netloc.strip()
    artifact_path = parsed.path.lstrip("/")
    if not run_id or not artifact_path:
        logger.warning("Malformed Tinker checkpoint URI: %s", uri)
        return None
    
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    
    try:
        future = rest_client.get_checkpoint_archive_url_from_tinker_path(uri)
        response = future.result(timeout=300)
    except Exception as exc:
        logger.warning("Failed to resolve Tinker checkpoint URL for %s: %s", uri, exc)
        return None

    signed_url = getattr(response, "url", None)
    if not signed_url:
        logger.warning("Tinker checkpoint response missing signed URL for %s", uri)
        return None

    dest_root.mkdir(parents=True, exist_ok=True)
    archive_path = dest_root / _checkpoint_archive_name(run_id, artifact_path)
    try:
        urllib.request.urlretrieve(signed_url, archive_path)
    except Exception as exc:
        logger.warning("Failed to download Tinker checkpoint archive %s: %s", uri, exc)
        return None

    try:
        extracted = _extract_artifact_if_archive(archive_path)
    except Exception as exc:
        logger.warning("Failed to extract archive %s: %s", archive_path, exc)
        return None

    if archive_path.exists() and archive_path.is_file():
        archive_path.unlink()
    return extracted

def _extract_artifact_if_archive(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Archive not found: {path}")
    if not path.is_file() or \
        not path.name.lower().endswith((".tar", ".tar.gz", ".zip")):
        return path
    
    extract_dir = _derive_extract_dir(path)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if tarfile.is_tarfile(path):
        with tarfile.open(path, mode="r:*") as archive:
            _safe_extract_tar(archive, extract_dir)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, mode="r") as archive:
            _safe_extract_zip(archive, extract_dir)
    else:
        raise RuntimeError(f"Unsupported archive type: {path}")
    return extract_dir

def _stage_lora_artifacts(payload_dir: Path, extracted_rel: str) -> Dict[str, str]:
    staged = {}
    extracted_path = payload_dir / extracted_rel
    if not extracted_path.exists():
        return staged

    if extracted_path.is_file():
        raise RuntimeError(f"Expected checkpoint archive to extract into a directory, got file: {extracted_path}")
    
    weight_file = next((p for p in extracted_path.rglob("*.safetensors") if p.is_file()))
    if weight_file:
        weight_target = payload_dir / weight_file.name
        if weight_target != weight_file:
            shutil.move(weight_file, weight_target)
        staged["weights"] = weight_target.relative_to(payload_dir).as_posix()
    else:
        logger.warning("No .safetensors file found in HF payload directory %s", extracted_path)

    adapter_file = _find_adapter_config(extracted_path)
    if adapter_file:
        adapter_target = payload_dir / adapter_file.name
        if adapter_target != adapter_file:
            shutil.move(adapter_file, adapter_target)
        staged["adapter_config"] = adapter_target.relative_to(payload_dir).as_posix()
    else:
        logger.warning("No adapter config found in HF payload directory %s", extracted_path)
    
    shutil.rmtree(extracted_path, ignore_errors=True)
    return staged

def _find_adapter_config(payload_dir: Path) -> Path | None:
    candidate_names = (
        "adapter_config.json", 
        "adapter_config.yaml",
        "config.json",
    )
    for name in candidate_names:
        candidate = payload_dir / name
        if candidate.exists():
            return candidate
    for candidate in payload_dir.rglob("adapter_config.json"):
        return candidate
    return None

def _checkpoint_archive_name(run_id: str, artifact_path: str) -> str:
    sanitized = artifact_path.strip().replace("/", "_") or "checkpoint"
    if not sanitized.endswith((".tar", ".zip")):
        sanitized = f"{sanitized}.tar"
    return f"{run_id}_{sanitized}"

def _derive_extract_dir(path: Path) -> Path:
    suffixes = (".tar.gz", ".tgz", ".tar.xz", ".tar", ".zip")
    base_name = path.name
    for suffix in suffixes:
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break
    if not base_name:
        base_name = "extracted"
    return path.parent / base_name

def _safe_extract_tar(archive: tarfile.TarFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for member in archive.getmembers():
        member_target = (dest_resolved / member.name).resolve(strict=False)
        if not str(member_target).startswith(str(dest_resolved)):
            raise RuntimeError(f"Unsafe member path detected in archive: {member.name}")
    archive.extractall(dest_resolved)

def _safe_extract_zip(archive: zipfile.ZipFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for name in archive.namelist():
        member_target = (dest_resolved / name).resolve(strict=False)
        if not str(member_target).startswith(str(dest_resolved)):
            raise RuntimeError(f"Unsafe member path detected in archive: {name}")
    archive.extractall(dest_resolved)
