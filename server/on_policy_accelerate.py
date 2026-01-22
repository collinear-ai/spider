"""On-policy distillation using Hugging Face Accelerate + PEFT.

This module provides the same functionality as on_policy.py but uses
Accelerate and PEFT instead of Tinker for training.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate import DeepSpeedPlugin
from peft import PeftModel

from tqdm.auto import tqdm

from spider.config import JobConfig

from .on_policy_accelerate_utils import (
    AccelerateTeacherContext,
    FireworksTeacherContext,
    TransformersSamplerContext,
    compute_teacher_alignment_for_rewards_direct,
    importance_sampling_loss,
    importance_sampling_loss_with_clip,
    load_model_with_lora,
    save_checkpoint_accelerate,
)
from .hf_upload import _prepare_hf_payload
from .sources import collect_prompts
from . import events
from .writers import JSONLBatchWriter
from .on_policy_vllm_rollouts import VLLMRolloutCollector, rollout_results_to_dicts
from .weight_synchronizer import WeightSynchronizer
from .backends.vllm_backend import VLLMBackend
from .vllm_parsers import parse_assistant_turn
from .backends.vllm_backend import _default_tool_parser, _default_reasoning_parser
from .metrics import discounted_future_sum_vectorized

RolloutBatch = List[Dict[str, Any]]

logger = logging.getLogger(__name__)


class _AccelerateStudentSamplerContext:
    """Student sampler context using transformers generate()."""

    def __init__(self, *, job_id: str, job: JobConfig) -> None:
        self._job_id = job_id
        self._job = job
        self._model = None
        self._tokenizer = None
        self._sampler = None
        self._accelerator = None

    def __enter__(self) -> TransformersSamplerContext:
        from .executor import JobExecutionError

        if self._sampler is not None:
            return self._sampler

        model_name = getattr(self._job.model, "name", None)
        checkpoint_path = getattr(self._job.model, "student_checkpoint_path", None)
        lora_rank = getattr(self._job.generation.on_policy_options, "lora_rank", 16)

        if not model_name:
            raise JobExecutionError(
                "job.model.name must be provided to create a sampling context."
            )

        # Load model with LoRA
        self._model, self._tokenizer = load_model_with_lora(
            model_name=model_name,
            lora_rank=lora_rank,
            checkpoint_path=checkpoint_path,
        )

        self._sampler = TransformersSamplerContext(
            model=self._model,
            tokenizer=self._tokenizer,
            model_name=model_name,
        )

        return self._sampler

    def refresh_from_sampler_path(self, sampler_path: str) -> TransformersSamplerContext:
        """Refresh the sampler from a checkpoint path."""
        if self._sampler is None:
            raise RuntimeError("Student sampler is not initialized.")

        self._sampler.refresh_from_checkpoint(sampler_path)
        return self._sampler

    def get_model(self) -> PeftModel:
        """Get the underlying model for training."""
        return self._model

    def __exit__(self, exc_type, exc, tb) -> None:
        self._model = None
        self._tokenizer = None
        self._sampler = None


def _create_shared_student_sampler(
    *,
    job_id: str,
    job: JobConfig,
) -> Tuple[_AccelerateStudentSamplerContext, TransformersSamplerContext]:
    """Create a shared student sampler context."""
    ctx = _AccelerateStudentSamplerContext(job_id=job_id, job=job)
    sampler = ctx.__enter__()
    return ctx, sampler


def _needs_gold_alignment(student_model: str, teacher_model: str) -> bool:
    """Check if gold alignment is needed (always True for now)."""
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
    runtime_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    on_batch_start: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    on_batch_start_lookahead: int = 0,
    on_batch_complete: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
):
    """Run on-policy distillation job using Accelerate.

    This is the accelerate-based version of run_on_policy_job from on_policy.py.
    """
    from .executor import (
        JobExecutionResult,
        JobExecutionError,
        _base_metadata,
        _resolve_processor,
        _write_metadata,
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
    prompt_rows = (
        prompts if prompts is not None else collect_prompts(job.source, pre_processor=pre_processor)
    )
    prompt_list = [row["prompt"] for row in prompt_rows]

    use_gold_alignment = _needs_gold_alignment(student_model, options.teacher)
    logger.info(
        "Job %s: on-policy distillation (accelerate) for student=%s collected %d prompts, "
        "use_gold_alignment=%s, tool_rollouts=%s",
        job_id,
        student_model,
        len(prompt_list),
        use_gold_alignment,
        bool(tool_registry),
    )

    sampler_ctx = None
    student_sampler = None
    rollout_stream = None

    use_vllm_inference = getattr(options, "use_vllm_inference", False)

    if tool_registry:
        if use_vllm_inference:
            # Use the new vLLM-based separated architecture
            logger.info(
                "Job %s: Using vLLM-based inference with separated training (accelerate)",
                job_id,
            )
            events.emit(
                "Using vLLM-based separated inference/training architecture.",
                code="on_policy_accelerate.vllm_inference",
            )
            # We don't need sampler_ctx for vLLM-based inference
            sampler_ctx = None
            rollout_stream = None  # Will be created inside the training function
        else:
            # Use the original transformers-based architecture
            sampler_ctx, student_sampler = _create_shared_student_sampler(job_id=job_id, job=job)

            rollout_stream = _tool_rollout_stream(
                job_id=job_id,
                job=job,
                student_sampler=student_sampler,
                prompts=prompt_rows,
                tool_registry=tool_registry,
                runtime_factory=runtime_factory,
                on_batch_start=on_batch_start,
                on_batch_start_lookahead=on_batch_start_lookahead,
                on_batch_complete=on_batch_complete,
            )
            events.emit(
                "Finished Setup for tool rollouts streaming for on-policy distillation (accelerate).",
                code="on_policy_accelerate.tool_rollouts_stream",
            )
    else:
        raise JobExecutionError(
            "Non-tool on-policy training not yet supported in accelerate version. "
            "Please use tool_registry or the tinker-based on_policy.py."
        )

    payload = _base_metadata(job_id, job)
    payload["generation_mode"] = "on_policy_accelerate"
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
            code="on_policy_accelerate.no_prompts",
        )
        result = JobExecutionResult(
            artifacts_path=artifact_path,
            metrics={"records": 0},
            messages=["No prompts found, skipped on-policy distillation."],
        )
        if sampler_ctx:
            sampler_ctx.__exit__(None, None, None)
        return result

    if use_vllm_inference:
        checkpoint_info = _run_tool_on_policy_vllm_accelerate(
            job_id=job_id,
            job=job,
            options=options,
            workspace=workspace,
            prompt_rows=prompt_rows,
            tool_registry=tool_registry,
            runtime_factory=runtime_factory,
            on_batch_start=on_batch_start,
            on_batch_start_lookahead=on_batch_start_lookahead,
            on_batch_complete=on_batch_complete,
            metadata_path=metadata_path,
            artifact_path=artifact_path,
        )
        checkpoint = checkpoint_info["checkpoint"]
        training_dir = checkpoint_info["training_dir"]
    elif rollout_stream:
        checkpoint_info = _run_tool_on_policy_stream_accelerate(
            job_id=job_id,
            job=job,
            options=options,
            workspace=workspace,
            total_batches=_compute_batch_stats(prompt_list, options)[1],
            rollout_stream=rollout_stream,
            sampler_ctx=sampler_ctx,
            metadata_path=metadata_path,
            artifact_path=artifact_path,
        )
        checkpoint = checkpoint_info["checkpoint"]
        training_dir = checkpoint_info["training_dir"]
    else:
        raise JobExecutionError(
            "No rollout stream available for on-policy training."
        )

    hf_payload_dir, manifest, checkpoints_index_text = _prepare_hf_payload(
        training_dir=training_dir,
        checkpoint=checkpoint,
        workspace=workspace,
    )
    if not manifest:
        raise JobExecutionError(
            "On-policy training did not produce uploadable checkpoint artifacts"
        )

    metrics = {
        "records": 0,
        "training_batches": float(checkpoint.get("batch", 0)),
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
        checkpoint.get("sampler_path"),
    )
    events.emit(
        "On-policy distillation (accelerate) completed.",
        code="on_policy_accelerate.completed",
        data={"training_batches": metrics["training_batches"]},
    )
    result = JobExecutionResult(
        artifacts_path=artifact_path,
        metadata_path=metadata_path,
        upload_source=hf_payload_dir,
        metrics=metrics,
        messages=["On-policy distillation completed."],
    )
    if sampler_ctx:
        sampler_ctx.__exit__(None, None, None)
    return result


def _run_tool_on_policy_stream_accelerate(
    *,
    job_id: str,
    job: JobConfig,
    options: Any,
    workspace: Path,
    total_batches: int,
    rollout_stream: Iterable[List[Dict[str, Any]]],
    sampler_ctx: _AccelerateStudentSamplerContext,
    metadata_path: Path,
    artifact_path: Path,
):
    """Run tool-based on-policy training using Accelerate."""
    import wandb
    from .executor import JobExecutionError, tool_descriptors

    tool_defs = tool_descriptors(job.tools)

    verbose_turns = bool(job.generation.verbose)

    # Initialize DeepSpeed plugin if enabled
    use_deepspeed = getattr(options, "use_deepspeed", False)
    deepspeed_plugin = None
    
    if use_deepspeed:
        deepspeed_config_path = getattr(options, "deepspeed_config_path", None)
        if deepspeed_config_path and Path(deepspeed_config_path).exists():
            # Load DeepSpeed config from file
            import json
            with open(deepspeed_config_path) as f:
                ds_config = json.load(f)
            
            zero_stage = ds_config.get("zero_optimization", {}).get("stage", 2)
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=ds_config,
                zero3_init_flag=zero_stage == 3,
            )
            logger.info(
                "Using DeepSpeed config from file: %s (ZeRO stage %d)",
                deepspeed_config_path,
                zero_stage,
            )
        else:
            # Create DeepSpeed plugin programmatically
            zero_stage = getattr(options, "deepspeed_zero_stage", 2)
            offload_optimizer = getattr(options, "deepspeed_offload_optimizer", False)
            offload_param = getattr(options, "deepspeed_offload_param", False)
            
            deepspeed_plugin = DeepSpeedPlugin(
                zero_stage=zero_stage,
                offload_optimizer_device="cpu" if offload_optimizer else None,
                offload_param_device="cpu" if offload_param else None,
            )
            logger.info(
                "Using DeepSpeed ZeRO stage %d (optimizer offload: %s, param offload: %s)",
                zero_stage,
                offload_optimizer,
                offload_param,
            )

    # Initialize Accelerator with DeepSpeed
    accelerator = Accelerator(
        mixed_precision="bf16",
        deepspeed_plugin=deepspeed_plugin,
    )

    # Get the model from sampler context
    model = sampler_ctx.get_model()

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Prepare with accelerator
    model, optimizer = accelerator.prepare(model, optimizer)

    # Load teacher model (use Fireworks API if fireworks_model is specified)
    fireworks_model = getattr(options, "fireworks_model", None)
    if fireworks_model:
        teacher_ctx = FireworksTeacherContext(
            model_name=options.teacher,
            fireworks_model=fireworks_model,
        )
        logger.info("Using Fireworks API for teacher: %s", fireworks_model)
    else:
        teacher_ctx = AccelerateTeacherContext(
            model_name=options.teacher,
            torch_dtype=torch.bfloat16,
        )

    training_dir = workspace / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb_project = getattr(options, "wandb_project", None)
    wandb_name = getattr(options, "wandb_name", None) or f"tool-on-policy-accelerate-{job_id[:8]}"
    token_budget = getattr(options, "token_budget", None)

    wandb_run = None
    if wandb_project:
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "job_id": job_id,
                "student_model": job.model.name,
                "teacher_model": options.teacher,
                "learning_rate": options.learning_rate,
                "lora_rank": options.lora_rank,
                "kl_penalty_coef": getattr(options, "kl_penalty_coef", 1.0),
                "kl_discount_factor": getattr(options, "kl_discount_factor", 0.0),
                "loss_fn": getattr(options, "loss_fn", "importance_sampling"),
                "total_batches": total_batches,
                "token_budget": token_budget,
                "framework": "accelerate",
            },
        )
        logger.info(
            "Job %s: wandb initialized. project=%s name=%s url=%s",
            job_id,
            wandb_project,
            wandb_name,
            wandb_run.url,
        )
        events.emit(
            "Wandb logging initialized (accelerate).",
            code="tool_on_policy_accelerate.wandb_initialized",
            data={"project": wandb_project, "name": wandb_name, "url": wandb_run.url},
        )
    else:
        logger.info("Job %s: wandb_project not set, skipping wandb logging.", job_id)

    device = accelerator.device

    # Helper to call teacher alignment (handles both sync and async contexts)
    def _compute_teacher_alignment(
        messages,
        tools,
        student_token_ids,
        student_logprobs,
        reward_mask,
        assistant_raw_text,
    ):
        if isinstance(teacher_ctx, FireworksTeacherContext):
            # Fireworks uses synchronous requests
            return teacher_ctx.compute_teacher_alignment(
                messages=messages,
                tools=tools,
                student_model=job.model.name,
                student_token_ids=student_token_ids,
                student_logprobs=student_logprobs,
                reward_mask=reward_mask,
                assistant_raw_text=assistant_raw_text,
            )
        else:
            # AccelerateTeacherContext - use direct function call
            return compute_teacher_alignment_for_rewards_direct(
                model=teacher_ctx.model,
                messages=messages,
                tools=tools,
                teacher_model=options.teacher,
                student_model=job.model.name,
                student_token_ids=student_token_ids,
                student_logprobs=student_logprobs,
                reward_mask=reward_mask,
                assistant_raw_text=assistant_raw_text,
                device=device,
            )

    def _process_batch(
        batch_index: int, trajectories: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Process a batch of trajectories and perform a training step."""
        items = []
        batch_metrics = {
            "kl_sum": 0.0,
            "kl_count": 0,
            "advantage_sum": 0.0,
            "advantage_sq_sum": 0.0,
            "advantage_count": 0,
        }

        # Collect data for batch training
        all_input_ids = []
        all_target_ids = []
        all_sampling_logprobs = []
        all_advantages = []
        all_loss_masks = []

        for turn_index, turn in enumerate(trajectories):
            messages = turn["messages"]
            token_ids = turn["token_ids"]
            logprobs = turn["logprobs"]
            reward_mask = turn["reward_mask"]

            student_logprobs = torch.tensor(logprobs, dtype=torch.float32, device=device)

            # Check if this is a combined multi-turn item
            combined_turns = turn.get("_combined_turns")
            if combined_turns:
                # Compute teacher alignment for each turn separately and combine
                kl_adjustments_combined = [0.0] * len(token_ids)
                kl_mask_combined = [0.0] * len(token_ids)
                teacher_logprobs_list = []
                teacher_token_ids_list = []

                for turn_info in combined_turns:
                    turn_messages = turn_info["messages"]
                    turn_assistant_raw_text = turn_info.get("assistant_raw_text")
                    completion_start = turn_info["completion_start"]
                    completion_end = turn_info["completion_end"]

                    if not turn_assistant_raw_text:
                        continue

                    # Create a reward mask for just this turn's completion
                    turn_reward_mask = [0] * len(token_ids)
                    for idx in range(completion_start, completion_end):
                        if idx < len(turn_reward_mask):
                            turn_reward_mask[idx] = 1

                    # Compute teacher alignment for this turn
                    turn_alignment = _compute_teacher_alignment(
                        messages=turn_messages,
                        tools=tool_defs,
                        student_token_ids=token_ids,
                        student_logprobs=student_logprobs,
                        reward_mask=turn_reward_mask,
                        assistant_raw_text=turn_assistant_raw_text,
                    )

                    # Extract KL adjustments and mask for this turn's region
                    turn_kl_adj = turn_alignment.get("kl_adjustments") or [0.0] * len(token_ids)
                    turn_kl_mask = turn_alignment.get("kl_mask") or [0.0] * len(token_ids)

                    # Combine into the full sequence
                    for idx in range(completion_start, completion_end):
                        if idx < len(kl_adjustments_combined):
                            kl_adjustments_combined[idx] = turn_kl_adj[idx]
                            if idx < len(turn_kl_mask):
                                kl_mask_combined[idx] = turn_kl_mask[idx]

                    # Collect teacher logprobs and token_ids
                    if turn_alignment.get("teacher_logprobs"):
                        teacher_logprobs_list = turn_alignment.get("teacher_logprobs")
                    if turn_alignment.get("teacher_token_ids"):
                        teacher_token_ids_list = turn_alignment.get("teacher_token_ids")

                teacher_alignment = {
                    "kl_adjustments": kl_adjustments_combined,
                    "kl_mask": kl_mask_combined,
                    "teacher_logprobs": teacher_logprobs_list,
                    "teacher_token_ids": teacher_token_ids_list,
                }
            else:
                # Single turn - compute teacher alignment normally
                teacher_alignment = _compute_teacher_alignment(
                    messages=messages,
                    tools=tool_defs,
                    student_token_ids=token_ids,
                    student_logprobs=student_logprobs,
                    reward_mask=reward_mask,
                    assistant_raw_text=turn.get("assistant_raw_text"),
                )

            item = dict(turn)
            item.update(
                {
                    "student_logprobs": student_logprobs,
                    "teacher_logprobs": teacher_alignment.get("teacher_logprobs"),
                    "kl_adjustments": teacher_alignment.get("kl_adjustments"),
                    "kl_mask": teacher_alignment.get("kl_mask"),
                    "teacher_token_ids": teacher_alignment.get("teacher_token_ids"),
                }
            )
            items.append(item)

            kl_adj = teacher_alignment.get("kl_adjustments") or [0.0] * len(token_ids)
            kl_mask = teacher_alignment.get("kl_mask") or [0.0] * len(token_ids)
            if len(kl_mask) != len(reward_mask):
                raise JobExecutionError("KL mask length must match reward mask length.")
            kl_tensor = torch.tensor(kl_adj, device=device, dtype=torch.float32)

            kl_coef = float(getattr(options, "kl_penalty_coef", 1.0))
            kl_discount = float(getattr(options, "kl_discount_factor", 0.0))
            advantage = -kl_coef * kl_tensor

            if kl_discount > 0:
                advantage = torch.tensor(
                    discounted_future_sum_vectorized(advantage.detach().cpu().numpy(), kl_discount),
                    device=device,
                    dtype=torch.float32,
                )

            # Prepare shifted sequences for training
            input_tokens = list(token_ids)[:-1]
            target_tokens = list(token_ids)[1:]
            mask_tokens = list(reward_mask)[1:]
            logprobs_tokens = student_logprobs[1:]
            advantages_tokens = advantage[1:]

            target_tensor = torch.tensor(target_tokens, dtype=torch.long, device=device)
            mask_tensor = torch.tensor(mask_tokens, dtype=torch.float32, device=device)
            advantages_tokens = advantages_tokens * mask_tensor

            all_input_ids.append(torch.tensor(input_tokens, dtype=torch.long, device=device))
            all_target_ids.append(target_tensor)
            all_sampling_logprobs.append(logprobs_tokens)
            all_advantages.append(advantages_tokens)
            all_loss_masks.append(mask_tensor)

            # Compute metrics
            n_tokens = len(token_ids)
            n_reward = sum(reward_mask)
            kl_vals = (
                kl_tensor[torch.tensor(reward_mask, dtype=torch.bool, device=device)].tolist()
                if kl_tensor.numel() > 0
                else []
            )
            adv_vals = (
                advantages_tokens[mask_tensor.bool()].tolist()
                if advantages_tokens.numel() > 0
                else []
            )

            batch_metrics["kl_sum"] += sum(kl_vals)
            batch_metrics["kl_count"] += len(kl_vals)
            batch_metrics["advantage_sum"] += sum(adv_vals)
            batch_metrics["advantage_sq_sum"] += sum(v * v for v in adv_vals)
            batch_metrics["advantage_count"] += len(adv_vals)

            if verbose_turns:
                logger.info(
                    "tool rollout batch=%d batch_item_idx=%d n_tokens=%d n_reward_tokens=%d",
                    batch_index,
                    turn_index,
                    n_tokens,
                    n_reward,
                )

        # Perform training step
        loss_token_count = sum(m.sum().item() for m in all_loss_masks)

        # Accumulate loss over all items in batch
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for input_ids, target_ids, sampling_lp, advantages, loss_mask in zip(
            all_input_ids, all_target_ids, all_sampling_logprobs, all_advantages, all_loss_masks
        ):
            # Add batch dimension
            input_ids = input_ids.unsqueeze(0)
            target_ids = target_ids.unsqueeze(0)
            sampling_lp = sampling_lp.unsqueeze(0)
            advantages = advantages.unsqueeze(0)
            loss_mask = loss_mask.unsqueeze(0)

            loss, _ = importance_sampling_loss(
                model=model,
                input_ids=input_ids,
                target_ids=target_ids,
                sampling_logprobs=sampling_lp,
                advantages=advantages,
                loss_mask=loss_mask,
            )
            total_loss = total_loss + loss

        # Normalize by number of items
        total_loss = total_loss / len(trajectories)

        # Backward pass
        optimizer.zero_grad()
        accelerator.backward(total_loss)
        optimizer.step()

        loss_value = total_loss.item()
        batch_metrics["loss"] = loss_value

        logger.info(
            "tool rollout batch=%d training step complete (accelerate). loss=%s",
            batch_index,
            loss_value,
        )
        events.emit(
            "Tool rollout batch KL training step complete (accelerate).",
            code="tool_on_policy_accelerate.training_step_complete",
            data={"batch_index": batch_index},
        )

        return items, batch_metrics

    def _train_stream():
        save_every = max(1, getattr(options, "save_every", 1))
        if total_batches > 0 and save_every > total_batches:
            save_every = total_batches

        last_checkpoint = None
        global_step = 0
        token_cum = 0
        
        # Create progress bar for training batches
        # Use position=0 and file=sys.stderr to persist at bottom of terminal
        train_pbar = tqdm(
            total=total_batches,
            desc="Training",
            unit="batch",
            initial=0,
            ncols=120,
            position=0,  # Fixed position at bottom
            file=sys.stderr,  # Use stderr so it doesn't interfere with stdout logging
            leave=True,  # Keep visible after completion
            mininterval=0.5,  # Update at least every 0.5 seconds
        )

        for batch_index, trajectories in enumerate(rollout_stream):
            batch_start = time.time()
            batch_items, batch_metrics = _process_batch(batch_index, trajectories)
            batch_time = time.time() - batch_start
            global_step += 1

            step_tokens = sum(int(sum(turn.get("reward_mask") or [])) for turn in trajectories)
            token_cum += step_tokens
            
            # Calculate throughput metrics
            tokens_per_sec = step_tokens / batch_time if batch_time > 0 else 0

            # Update progress bar
            train_pbar.update(1)
            train_pbar.set_postfix({
                "loss": f"{batch_metrics.get('loss', 0):.4f}" if batch_metrics.get('loss') else "N/A",
                "tokens": step_tokens,
                "cum_tokens": token_cum,
                "time": f"{batch_time:.1f}s",
                "tokens/s": f"{tokens_per_sec:.1f}",
            })

            logger.info(
                "tool rollout batch=%d/%d step_time=%.3fs token_step=%d token_cum=%d "
                "tokens/s=%.1f token_budget=%s",
                batch_index + 1,
                total_batches,
                batch_time,
                step_tokens,
                token_cum,
                tokens_per_sec,
                token_budget,
            )

            # Log to wandb
            if wandb_run:
                log_data = {
                    "progress/done_frac": (batch_index + 1) / total_batches
                    if total_batches > 0
                    else 1.0,
                    "optim/lr": options.learning_rate,
                    "rollouts": len(trajectories),
                    "time / total_seconds": batch_time,
                    "time / per_turn_seconds": batch_time / len(trajectories),
                    "token_step": step_tokens,
                    "token_cum": token_cum,
                }

                if token_budget is not None:
                    log_data["token_budget"] = token_budget
                if batch_metrics.get("loss") is not None:
                    log_data["loss"] = batch_metrics["loss"]
                if batch_metrics["kl_count"] > 0:
                    log_data["kl_mean"] = batch_metrics["kl_sum"] / batch_metrics["kl_count"]
                if batch_metrics["advantage_count"] > 0:
                    adv_mean = batch_metrics["advantage_sum"] / batch_metrics["advantage_count"]
                    adv_var = (
                        batch_metrics["advantage_sq_sum"] / batch_metrics["advantage_count"]
                    ) - adv_mean**2
                    log_data["advantage_mean"] = adv_mean
                    log_data["advantage_std"] = adv_var**0.5 if adv_var > 0 else 0.0

                # Flatten keys
                flat_log_data = {}
                for k, v in log_data.items():
                    key = k.split("/")[-1] if "/" in k else k
                    flat_log_data[key] = v
                wandb_run.log(flat_log_data, step=global_step)

            if token_budget is not None and token_cum >= token_budget:
                logger.info(
                    "tool rollout token budget reached. stopping training. token_cum=%d budget=%d",
                    token_cum,
                    token_budget,
                )
                events.emit(
                    "Tool rollout token budget reached (accelerate).",
                    code="tool_on_policy_accelerate.token_budget_reached",
                    data={"token_cum": token_cum, "token_budget": token_budget},
                )
                train_pbar.close()
                break

            if (batch_index + 1) % save_every == 0:
                train_pbar.write(f"💾 Saving checkpoint at batch {batch_index + 1}")
                last_checkpoint = save_checkpoint_accelerate(
                    model=model,
                    optimizer=optimizer,
                    accelerator=accelerator,
                    name=f"{batch_index:06d}",
                    log_path=str(training_dir),
                    loop_state={"batch": batch_index + 1},
                )
                sampler_path = last_checkpoint.get("sampler_path")
                if sampler_path:
                    sampler_ctx.refresh_from_sampler_path(sampler_path)
                    logger.info(
                        "Refreshed student sampler from checkpoint at batch=%d path=%s.",
                        batch_index,
                        sampler_path,
                    )
                    events.emit(
                        "Refreshed student sampler from checkpoint (accelerate).",
                        code="tool_on_policy_accelerate.sampler_refreshed",
                        data={"batch_index": batch_index, "sampler_path": sampler_path},
                    )
        
        train_pbar.close()
        logger.info("Training loop complete: total_batches=%d total_tokens=%d", total_batches, token_cum)
        
        return last_checkpoint

    try:
        last_checkpoint = _train_stream()
        if not last_checkpoint:
            last_checkpoint = save_checkpoint_accelerate(
                model=model,
                optimizer=optimizer,
                accelerator=accelerator,
                name="final",
                log_path=str(training_dir),
                loop_state={"batch": 0},
            )
        if not last_checkpoint or "sampler_path" not in last_checkpoint:
            raise JobExecutionError(
                "Tool on-policy training (accelerate) did not produce a sampler checkpoint."
            )

        return {
            "checkpoint": last_checkpoint,
            "training_dir": training_dir,
        }
    finally:
        if wandb_run:
            wandb_run.finish()


def _run_tool_on_policy_vllm_accelerate(
    *,
    job_id: str,
    job: JobConfig,
    options: Any,
    workspace: Path,
    prompt_rows: List[Dict[str, Any]],
    tool_registry: Dict[str, Callable[..., Any]],
    runtime_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    on_batch_start: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    on_batch_start_lookahead: int = 0,
    on_batch_complete: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    metadata_path: Path,
    artifact_path: Path,
):
    """Run tool-based on-policy training using separated vLLM inference + Accelerate training.

    This function implements the separated architecture where:
    - vLLM server handles inference with LoRA adapters
    - ThreadPoolExecutor collects rollouts in parallel
    - Accelerate handles training with PEFT
    - WeightSynchronizer syncs weights between training and inference
    """
    import wandb
    from .executor import JobExecutionError, tool_descriptors
    from spider.config import ModelConfig

    tool_defs = tool_descriptors(job.tools)
    verbose_turns = bool(job.generation.verbose)

    student_model = job.model.name
    student_checkpoint = job.model.student_checkpoint_path

    batch_size, total_batches = _compute_batch_stats(prompt_rows, options)

    # GPU allocation
    vllm_gpu_ids = getattr(options, "vllm_gpu_ids", [0])
    training_gpu_ids = getattr(options, "training_gpu_ids", [1])

    logger.info(
        "GPU allocation: vLLM=%s, training=%s",
        vllm_gpu_ids,
        training_gpu_ids,
    )

    # IMPORTANT: Start vLLM server BEFORE setting CUDA_VISIBLE_DEVICES for training
    # vLLM subprocess will get its own CUDA_VISIBLE_DEVICES via its environment
    vllm_config = ModelConfig(
        provider="vllm",
        name=student_model,
        parameters={
            "tensor_parallel_size": getattr(options, "vllm_tensor_parallel_size", 1),
            "gpu_memory_utilization": getattr(options, "vllm_gpu_memory_utilization", 0.9),
            # Performance optimizations for better throughput
            "max_num_batched_tokens": getattr(options, "vllm_max_num_batched_tokens", None),
            "max_num_seqs": getattr(options, "vllm_max_num_seqs", None),
        },
    )

    logger.info(
        "Starting vLLM server with LoRA support for model %s on GPUs %s",
        student_model,
        vllm_gpu_ids,
    )

    # Save original CUDA_VISIBLE_DEVICES to restore if needed
    original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    # CRITICAL: Ensure CUDA is not initialized before starting vLLM subprocess.
    # vLLM workers need a clean CUDA context. If torch has already initialized
    # CUDA, the subprocess workers may inherit corrupted state.
    if torch.cuda.is_initialized():
        logger.warning(
            "CUDA was already initialized before starting vLLM. "
            "This may cause issues with vLLM worker processes. "
            "Consider setting CUDA_VISIBLE_DEVICES='' before imports."
        )

    vllm_backend = VLLMBackend(
        config=vllm_config,
        enable_lora=True,
        max_lora_rank=options.lora_rank * 2,  # Allow some headroom
        gpu_ids=vllm_gpu_ids,
    )

    # Initialize wandb_run before try block so it's always defined in finally
    wandb_run = None

    try:
        # NOW set training GPUs via environment variable after vLLM has started
        # This ensures the training model loads on the correct GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, training_gpu_ids))
        logger.info(
            "Set CUDA_VISIBLE_DEVICES=%s for training",
            os.environ["CUDA_VISIBLE_DEVICES"],
        )

        # Initialize DeepSpeed plugin if enabled
        use_deepspeed = getattr(options, "use_deepspeed", False)
        deepspeed_plugin = None
        
        if use_deepspeed:
            deepspeed_config_path = getattr(options, "deepspeed_config_path", None)
            if deepspeed_config_path and Path(deepspeed_config_path).exists():
                # Load DeepSpeed config from file
                import json
                with open(deepspeed_config_path) as f:
                    ds_config = json.load(f)
                
                zero_stage = ds_config.get("zero_optimization", {}).get("stage", 2)
                deepspeed_plugin = DeepSpeedPlugin(
                    hf_ds_config=ds_config,
                    zero3_init_flag=zero_stage == 3,
                )
                logger.info(
                    "Using DeepSpeed config from file: %s (ZeRO stage %d)",
                    deepspeed_config_path,
                    zero_stage,
                )
            else:
                # Create DeepSpeed plugin programmatically
                zero_stage = getattr(options, "deepspeed_zero_stage", 2)
                offload_optimizer = getattr(options, "deepspeed_offload_optimizer", False)
                offload_param = getattr(options, "deepspeed_offload_param", False)
                
                deepspeed_plugin = DeepSpeedPlugin(
                    zero_stage=zero_stage,
                    offload_optimizer_device="cpu" if offload_optimizer else None,
                    offload_param_device="cpu" if offload_param else None,
                )
                logger.info(
                    "Using DeepSpeed ZeRO stage %d (optimizer offload: %s, param offload: %s)",
                    zero_stage,
                    offload_optimizer,
                    offload_param,
                )

        # Initialize Accelerator with DeepSpeed
        accelerator = Accelerator(
            mixed_precision="bf16",
            deepspeed_plugin=deepspeed_plugin,
        )

        # Delay loading training model until first batch (for testing inference speed)
        # Model will be loaded lazily when first needed for training
        model = None
        tokenizer = None
        optimizer = None
        training_model_loaded = False

        def _ensure_training_model_loaded():
            """Lazily load training model when first needed."""
            nonlocal model, tokenizer, optimizer, training_model_loaded
            if training_model_loaded:
                return
            
            logger.info("Loading training model (deferred until first training step)...")
            # When using DeepSpeed, don't use device_map (Accelerate/DeepSpeed will handle device placement)
            # For ZeRO-3, load on CPU first (DeepSpeed will shard during prepare)
            if use_deepspeed:
                zero_stage = getattr(options, "deepspeed_zero_stage", 2)
                if zero_stage == 3:
                    device_map = "cpu"  # ZeRO-3 needs CPU loading
                else:
                    device_map = None  # ZeRO-1/2: let Accelerate/DeepSpeed handle device placement
            else:
                device_map = "auto"  # No DeepSpeed: use auto device mapping
            
            model, tokenizer = load_model_with_lora(
                model_name=student_model,
                lora_rank=options.lora_rank,
                checkpoint_path=student_checkpoint,
                device_map=device_map,
            )

            # Create optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=options.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
            )

            # Prepare with accelerator
            model, optimizer = accelerator.prepare(model, optimizer)
            training_model_loaded = True
            logger.info("Training model loaded and prepared.")

        training_dir = workspace / "training"
        training_dir.mkdir(parents=True, exist_ok=True)

        # Create weight synchronizer (will sync after model is loaded)
        weight_sync_steps = getattr(options, "weight_sync_steps", 10)
        synchronizer = WeightSynchronizer(
            sync_every_n_steps=weight_sync_steps,
            checkpoint_dir=training_dir / "lora_checkpoints",
            vllm_base_url=vllm_backend.base_url,
            lora_name="student",
            vllm_backend=vllm_backend,
        )

        # Skip initial sync - will sync after first batch when model is loaded
        logger.info("Skipping initial weight sync (model not loaded yet). Will sync after first batch.")

        # Create rollout collector
        # Note: lora_name will be None initially (no LoRA adapter loaded yet)
        # vLLM will work without LoRA for initial rollouts
        rollout_workers = getattr(options, "rollout_workers", 8)
        collector = VLLMRolloutCollector(
            vllm_base_url=vllm_backend.base_url,
            model_name=student_model,
            tools=tool_defs,
            tool_registry=tool_registry,
            tool_parser_name=vllm_backend.tool_parser,
            reasoning_parser_name=vllm_backend.reasoning_parser,
            max_workers=rollout_workers,
            max_tool_turns=max(1, job.generation.max_tool_turns or 16),
            max_tokens=job.generation.parameters.get("max_tokens", 4096),
            temperature=job.generation.parameters.get("temperature", 1.0),
            lora_name=None,  # Will be set after first batch when model is loaded
            runtime_factory=runtime_factory,
            verbose=verbose_turns,
        )

        # Load teacher model
        fireworks_model = getattr(options, "fireworks_model", None)
        if fireworks_model:
            teacher_ctx = FireworksTeacherContext(
                model_name=options.teacher,
                fireworks_model=fireworks_model,
            )
            logger.info("Using Fireworks API for teacher: %s", fireworks_model)
        else:
            teacher_ctx = AccelerateTeacherContext(
                model_name=options.teacher,
                torch_dtype=torch.bfloat16,
            )

        # Initialize wandb
        wandb_project = getattr(options, "wandb_project", None)
        wandb_name = getattr(options, "wandb_name", None) or f"tool-on-policy-vllm-{job_id[:8]}"
        token_budget = getattr(options, "token_budget", None)
        clip_ratio = getattr(options, "importance_sampling_clip", 0.2)

        if wandb_project:
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_name,
                config={
                    "job_id": job_id,
                    "student_model": student_model,
                    "teacher_model": options.teacher,
                    "learning_rate": options.learning_rate,
                    "lora_rank": options.lora_rank,
                    "kl_penalty_coef": getattr(options, "kl_penalty_coef", 1.0),
                    "kl_discount_factor": getattr(options, "kl_discount_factor", 0.0),
                    "loss_fn": getattr(options, "loss_fn", "importance_sampling"),
                    "clip_ratio": clip_ratio,
                    "rollout_workers": rollout_workers,
                    "weight_sync_steps": weight_sync_steps,
                    "total_batches": total_batches,
                    "token_budget": token_budget,
                    "framework": "accelerate_vllm",
                },
            )
            logger.info(
                "Job %s: wandb initialized. project=%s name=%s url=%s",
                job_id,
                wandb_project,
                wandb_name,
                wandb_run.url,
            )
            events.emit(
                "Wandb logging initialized (accelerate + vLLM).",
                code="tool_on_policy_vllm_accelerate.wandb_initialized",
                data={"project": wandb_project, "name": wandb_name, "url": wandb_run.url},
            )
        else:
            logger.info("Job %s: wandb_project not set, skipping wandb logging.", job_id)

        device = accelerator.device

        # Helper to call teacher alignment
        def _compute_teacher_alignment(
            messages,
            tools,
            student_token_ids,
            student_logprobs,
            reward_mask,
            assistant_raw_text,
        ):
            if isinstance(teacher_ctx, FireworksTeacherContext):
                # Fireworks uses synchronous requests
                return teacher_ctx.compute_teacher_alignment(
                    messages=messages,
                    tools=tools,
                    student_model=student_model,
                    student_token_ids=student_token_ids,
                    student_logprobs=student_logprobs,
                    reward_mask=reward_mask,
                    assistant_raw_text=assistant_raw_text,
                )
            else:
                return compute_teacher_alignment_for_rewards_direct(
                    model=teacher_ctx.model,
                    messages=messages,
                    tools=tools,
                    teacher_model=options.teacher,
                    student_model=student_model,
                    student_token_ids=student_token_ids,
                    student_logprobs=student_logprobs,
                    reward_mask=reward_mask,
                    assistant_raw_text=assistant_raw_text,
                    device=device,
                )

        def _process_batch(
            batch_index: int, trajectories: List[Dict[str, Any]]
        ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
            # Load training model lazily on first batch
            _ensure_training_model_loaded()
            
            # Do initial weight sync on first batch (if not done yet)
            if batch_index == 0:
                synchronizer.force_sync(model, accelerator, step=0)
                # Update collector to use the LoRA adapter after sync
                collector.lora_name = synchronizer.get_lora_name()
                logger.info("Initial weight sync completed after first batch. LoRA adapter loaded: %s", collector.lora_name)
            """Process a batch of trajectories and perform a training step."""
            items = []
            batch_metrics = {
                "kl_sum": 0.0,
                "kl_count": 0,
                "advantage_sum": 0.0,
                "advantage_sq_sum": 0.0,
                "advantage_count": 0,
                "clip_fraction_sum": 0.0,
                "clip_count": 0,
            }

            all_input_ids = []
            all_target_ids = []
            all_sampling_logprobs = []
            all_advantages = []
            all_loss_masks = []

            for turn_index, turn in enumerate(trajectories):
                messages = turn["messages"]
                token_ids = turn["token_ids"]
                logprobs = turn["logprobs"]
                reward_mask = turn["reward_mask"]

                student_logprobs = torch.tensor(logprobs, dtype=torch.float32, device=device)

                # Check if this is a combined multi-turn item
                combined_turns = turn.get("_combined_turns")
                if combined_turns:
                    kl_adjustments_combined = [0.0] * len(token_ids)
                    kl_mask_combined = [0.0] * len(token_ids)

                    for turn_info in combined_turns:
                        turn_messages = turn_info["messages"]
                        turn_assistant_raw_text = turn_info.get("assistant_raw_text")
                        completion_start = turn_info["completion_start"]
                        completion_end = turn_info["completion_end"]

                        if not turn_assistant_raw_text:
                            continue

                        turn_reward_mask = [0] * len(token_ids)
                        for idx in range(completion_start, completion_end):
                            if idx < len(turn_reward_mask):
                                turn_reward_mask[idx] = 1

                        turn_alignment = _compute_teacher_alignment(
                            messages=turn_messages,
                            tools=tool_defs,
                            student_token_ids=token_ids,
                            student_logprobs=student_logprobs,
                            reward_mask=turn_reward_mask,
                            assistant_raw_text=turn_assistant_raw_text,
                        )

                        turn_kl_adj = turn_alignment.get("kl_adjustments") or [0.0] * len(token_ids)
                        turn_kl_mask = turn_alignment.get("kl_mask") or [0.0] * len(token_ids)

                        for idx in range(completion_start, completion_end):
                            if idx < len(kl_adjustments_combined):
                                kl_adjustments_combined[idx] = turn_kl_adj[idx]
                                if idx < len(turn_kl_mask):
                                    kl_mask_combined[idx] = turn_kl_mask[idx]

                    teacher_alignment = {
                        "kl_adjustments": kl_adjustments_combined,
                        "kl_mask": kl_mask_combined,
                    }
                else:
                    teacher_alignment = _compute_teacher_alignment(
                        messages=messages,
                        tools=tool_defs,
                        student_token_ids=token_ids,
                        student_logprobs=student_logprobs,
                        reward_mask=reward_mask,
                        assistant_raw_text=turn.get("assistant_raw_text"),
                    )

                item = dict(turn)
                item.update({
                    "student_logprobs": student_logprobs,
                    "kl_adjustments": teacher_alignment.get("kl_adjustments"),
                    "kl_mask": teacher_alignment.get("kl_mask"),
                })
                items.append(item)

                kl_adj = teacher_alignment.get("kl_adjustments") or [0.0] * len(token_ids)
                kl_mask = teacher_alignment.get("kl_mask") or [0.0] * len(token_ids)
                kl_tensor = torch.tensor(kl_adj, device=device, dtype=torch.float32)

                kl_coef = float(getattr(options, "kl_penalty_coef", 1.0))
                kl_discount = float(getattr(options, "kl_discount_factor", 0.0))
                advantage = -kl_coef * kl_tensor

                if kl_discount > 0:
                    advantage = torch.tensor(
                        discounted_future_sum_vectorized(advantage.detach().cpu().numpy(), kl_discount),
                        device=device,
                        dtype=torch.float32,
                    )

                input_tokens = list(token_ids)[:-1]
                target_tokens = list(token_ids)[1:]
                mask_tokens = list(reward_mask)[1:]
                logprobs_tokens = student_logprobs[1:]
                advantages_tokens = advantage[1:]

                target_tensor = torch.tensor(target_tokens, dtype=torch.long, device=device)
                mask_tensor = torch.tensor(mask_tokens, dtype=torch.float32, device=device)
                advantages_tokens = advantages_tokens * mask_tensor

                all_input_ids.append(torch.tensor(input_tokens, dtype=torch.long, device=device))
                all_target_ids.append(target_tensor)
                all_sampling_logprobs.append(logprobs_tokens)
                all_advantages.append(advantages_tokens)
                all_loss_masks.append(mask_tensor)

                n_tokens = len(token_ids)
                n_reward = sum(reward_mask)
                kl_vals = (
                    kl_tensor[torch.tensor(reward_mask, dtype=torch.bool, device=device)].tolist()
                    if kl_tensor.numel() > 0
                    else []
                )
                adv_vals = (
                    advantages_tokens[mask_tensor.bool()].tolist()
                    if advantages_tokens.numel() > 0
                    else []
                )

                batch_metrics["kl_sum"] += sum(kl_vals)
                batch_metrics["kl_count"] += len(kl_vals)
                batch_metrics["advantage_sum"] += sum(adv_vals)
                batch_metrics["advantage_sq_sum"] += sum(v * v for v in adv_vals)
                batch_metrics["advantage_count"] += len(adv_vals)

                if verbose_turns:
                    logger.info(
                        "vllm rollout batch=%d batch_item_idx=%d n_tokens=%d n_reward_tokens=%d",
                        batch_index,
                        turn_index,
                        n_tokens,
                        n_reward,
                    )

            # Perform training step with clipped importance sampling and gradient accumulation
            num_trajectories = len(all_input_ids)
            gradient_accumulation_steps = getattr(options, "gradient_accumulation_steps", 1)
            if gradient_accumulation_steps is None or gradient_accumulation_steps <= 0:
                gradient_accumulation_steps = num_trajectories  # Process all at once if not set
            
            # Zero gradients at start
            optimizer.zero_grad()
            
            accumulated_loss_value = 0.0
            total_clip_fraction = 0.0
            
            # Process trajectories in chunks for gradient accumulation
            for chunk_start in range(0, num_trajectories, gradient_accumulation_steps):
                chunk_end = min(chunk_start + gradient_accumulation_steps, num_trajectories)
                chunk_size = chunk_end - chunk_start
                
                chunk_loss = torch.tensor(0.0, device=device, requires_grad=True)
                chunk_clip_frac = 0.0
                
                for idx in range(chunk_start, chunk_end):
                    input_ids = all_input_ids[idx].unsqueeze(0)
                    target_ids = all_target_ids[idx].unsqueeze(0)
                    sampling_lp = all_sampling_logprobs[idx].unsqueeze(0)
                    advantages = all_advantages[idx].unsqueeze(0)
                    loss_mask = all_loss_masks[idx].unsqueeze(0)

                    loss, _, metrics = importance_sampling_loss_with_clip(
                        model=model,
                        input_ids=input_ids,
                        target_ids=target_ids,
                        sampling_logprobs=sampling_lp,
                        advantages=advantages,
                        loss_mask=loss_mask,
                        clip_ratio=clip_ratio,
                    )
                    chunk_loss = chunk_loss + loss
                    chunk_clip_frac += metrics["clip_fraction"]
                    
                    # Clear intermediate tensors after each sequence
                    del input_ids, target_ids, sampling_lp, advantages, loss_mask, loss
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Average chunk loss and scale by number of chunks
                chunk_loss = chunk_loss / chunk_size
                num_chunks = (num_trajectories + gradient_accumulation_steps - 1) // gradient_accumulation_steps
                chunk_loss = chunk_loss / num_chunks
                
                # Backward on chunk (gradients accumulate)
                accelerator.backward(chunk_loss)
                accumulated_loss_value += chunk_loss.item() * chunk_size
                total_clip_fraction += chunk_clip_frac
                
                del chunk_loss
                torch.cuda.empty_cache()
                gc.collect()
            
            # Step optimizer once after all chunks
            optimizer.step()
            
            # Average loss for logging
            loss_value = accumulated_loss_value / num_trajectories if num_trajectories > 0 else 0.0
            batch_metrics["clip_fraction_sum"] += total_clip_fraction
            batch_metrics["clip_count"] += num_trajectories
            
            # Aggressive memory cleanup after backward pass
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            batch_metrics["loss"] = loss_value

            logger.info(
                "vllm rollout batch=%d training step complete. loss=%s clip_frac=%.3f",
                batch_index,
                loss_value,
                total_clip_fraction / len(trajectories) if trajectories else 0,
            )
            events.emit(
                "Tool rollout batch KL training step complete (vLLM + accelerate).",
                code="tool_on_policy_vllm_accelerate.training_step_complete",
                data={"batch_index": batch_index},
            )

            return items, batch_metrics

        # Main training loop
        save_every = max(1, getattr(options, "save_every", 1))
        if total_batches > 0 and save_every > total_batches:
            save_every = total_batches

        last_checkpoint = None
        global_step = 0
        token_cum = 0
        
        # Create progress bar for training batches
        # Use position=0 and file=sys.stderr to persist at bottom of terminal
        train_pbar = tqdm(
            total=total_batches,
            desc="Training",
            unit="batch",
            initial=0,
            ncols=120,
            position=0,  # Fixed position at bottom
            file=sys.stderr,  # Use stderr so it doesn't interfere with stdout logging
            leave=True,  # Keep visible after completion
            mininterval=0.5,  # Update at least every 0.5 seconds
        )

        for batch_index, start in enumerate(range(0, len(prompt_rows), batch_size)):
            batch_start = time.time()

            # Handle lookahead callbacks
            if on_batch_start and on_batch_start_lookahead > 0:
                for ahead in range(1, on_batch_start_lookahead + 1):
                    next_start = start + ahead * batch_size
                    if next_start >= len(prompt_rows):
                        break
                    next_chunk = prompt_rows[next_start : next_start + batch_size]
                    on_batch_start(next_chunk)

            chunk = prompt_rows[start : start + batch_size]

            # Collect rollouts in parallel using vLLM
            rollout_start = time.time()
            logger.info("Collecting rollouts for batch %d/%d (%d prompts)", batch_index + 1, total_batches, len(chunk))
            rollout_results = collector.collect_batch(chunk)
            rollout_time = time.time() - rollout_start
            trajectories = rollout_results_to_dicts(rollout_results)
            
            logger.info(
                "Rollout collection complete for batch %d: time=%.2fs trajectories=%d",
                batch_index,
                rollout_time,
                len(trajectories),
            )

            if on_batch_complete is not None:
                on_batch_complete(chunk)

            if not trajectories:
                logger.warning("No trajectories collected for batch %d", batch_index)
                continue

            events.emit(
                "Tool rollout batch ready (vLLM + accelerate).",
                code="tool_on_policy_vllm_accelerate.batch_ready",
                data={
                    "batch_index": batch_index,
                    "batch_size": len(chunk),
                    "total_batches": total_batches,
                },
            )

            # Process batch (training step)
            training_start = time.time()
            batch_items, batch_metrics = _process_batch(batch_index, trajectories)
            training_time = time.time() - training_start
            batch_time = time.time() - batch_start
            global_step += 1

            step_tokens = sum(int(sum(turn.get("reward_mask") or [])) for turn in trajectories)
            token_cum += step_tokens
            
            # Calculate throughput metrics
            tokens_per_sec = step_tokens / batch_time if batch_time > 0 else 0
            rollout_throughput = len(chunk) / rollout_time if rollout_time > 0 else 0

            # Update progress bar
            train_pbar.update(1)
            train_pbar.set_postfix({
                "loss": f"{batch_metrics.get('loss', 0):.4f}" if batch_metrics.get('loss') else "N/A",
                "tokens": step_tokens,
                "cum_tokens": token_cum,
                "time": f"{batch_time:.1f}s",
                "tokens/s": f"{tokens_per_sec:.1f}",
            })

            logger.info(
                "vllm rollout batch=%d/%d step_time=%.3fs (rollout=%.2fs train=%.2fs) "
                "token_step=%d token_cum=%d tokens/s=%.1f token_budget=%s",
                batch_index + 1,
                total_batches,
                batch_time,
                rollout_time,
                training_time,
                step_tokens,
                token_cum,
                tokens_per_sec,
                token_budget,
            )

            # Log to wandb
            if wandb_run:
                log_data = {
                    "progress/done_frac": (batch_index + 1) / total_batches if total_batches > 0 else 1.0,
                    "optim/lr": options.learning_rate,
                    "rollouts": len(trajectories),
                    "time / total_seconds": batch_time,
                    "time / per_turn_seconds": batch_time / len(trajectories) if trajectories else 0,
                    "token_step": step_tokens,
                    "token_cum": token_cum,
                }

                if token_budget is not None:
                    log_data["token_budget"] = token_budget
                if batch_metrics.get("loss") is not None:
                    log_data["loss"] = batch_metrics["loss"]
                if batch_metrics["kl_count"] > 0:
                    log_data["kl_mean"] = batch_metrics["kl_sum"] / batch_metrics["kl_count"]
                if batch_metrics["advantage_count"] > 0:
                    adv_mean = batch_metrics["advantage_sum"] / batch_metrics["advantage_count"]
                    adv_var = (batch_metrics["advantage_sq_sum"] / batch_metrics["advantage_count"]) - adv_mean**2
                    log_data["advantage_mean"] = adv_mean
                    log_data["advantage_std"] = adv_var**0.5 if adv_var > 0 else 0.0
                if batch_metrics["clip_count"] > 0:
                    log_data["clip_fraction"] = batch_metrics["clip_fraction_sum"] / batch_metrics["clip_count"]

                flat_log_data = {}
                for k, v in log_data.items():
                    key = k.split("/")[-1] if "/" in k else k
                    flat_log_data[key] = v
                wandb_run.log(flat_log_data, step=global_step)

            if token_budget is not None and token_cum >= token_budget:
                logger.info(
                    "vllm rollout token budget reached. stopping training. token_cum=%d budget=%d",
                    token_cum,
                    token_budget,
                )
                events.emit(
                    "Tool rollout token budget reached (vLLM + accelerate).",
                    code="tool_on_policy_vllm_accelerate.token_budget_reached",
                    data={"token_cum": token_cum, "token_budget": token_budget},
                )
                train_pbar.close()
                break

            # Sync weights to vLLM and save checkpoint
            if (batch_index + 1) % save_every == 0:
                last_checkpoint = save_checkpoint_accelerate(
                    model=model,
                    optimizer=optimizer,
                    accelerator=accelerator,
                    name=f"{batch_index:06d}",
                    log_path=str(training_dir),
                    loop_state={"batch": batch_index + 1},
                )

            # Sync weights to vLLM for inference (only if model is loaded)
            if training_model_loaded and synchronizer.maybe_sync(model, accelerator, batch_index + 1):
                # Update the collector to use the new LoRA adapter
                collector.lora_name = synchronizer.get_lora_name()
                logger.info(
                    "Synced LoRA weights to vLLM at batch=%d",
                    batch_index,
                )
                events.emit(
                    "Synced LoRA weights to vLLM (accelerate).",
                    code="tool_on_policy_vllm_accelerate.weights_synced",
                    data={"batch_index": batch_index},
                )

        # Close progress bar
        train_pbar.close()
        logger.info("Training loop complete: total_batches=%d total_tokens=%d", total_batches, token_cum)

        # Final checkpoint (only if model was loaded)
        if training_model_loaded and not last_checkpoint:
            last_checkpoint = save_checkpoint_accelerate(
                model=model,
                optimizer=optimizer,
                accelerator=accelerator,
                name="final",
                log_path=str(training_dir),
                loop_state={"batch": global_step},
            )

        if not last_checkpoint or "sampler_path" not in last_checkpoint:
            raise JobExecutionError(
                "Tool on-policy training (vLLM + accelerate) did not produce a sampler checkpoint."
            )

        return {
            "checkpoint": last_checkpoint,
            "training_dir": training_dir,
        }

    finally:
        # Clean up resources
        if wandb_run:
            wandb_run.finish()
        if 'collector' in locals():
            collector.close()
        if 'synchronizer' in locals():
            synchronizer.close()
        vllm_backend.close()


def _tool_rollout_stream(
    *,
    job_id: str,
    job: JobConfig,
    student_sampler: TransformersSamplerContext,
    prompts: List[Dict[str, Any]],
    tool_registry: Dict[str, Callable[..., Any]],
    runtime_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    on_batch_start: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    on_batch_start_lookahead: int = 0,
    on_batch_complete: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Iterable[List[Dict[str, Any]]]:
    """Generate tool rollouts using transformers generate().

    This is the accelerate version of _tool_rollout_stream from on_policy.py.
    It uses TransformersSamplerContext instead of tinker.SamplingClient.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from .executor import (
        _initial_chat_history,
        _execute_tool_calls,
        tool_descriptors,
        JobExecutionError,
    )

    tool_defs = tool_descriptors(job.tools)
    turn_limit = max(1, job.generation.max_tool_turns or 16)
    verbose_turns = bool(job.generation.verbose)

    batch_size, total_batches = _compute_batch_stats(prompts, job.generation.on_policy_options)

    tokenizer = student_sampler.tokenizer
    model = student_sampler.model
    model_name = student_sampler.model_name
    device = next(model.parameters()).device

    def _run_prompt(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = row["prompt"]
        history = _initial_chat_history(prompt, row.get("system_prompt"))
        turn_items = []
        runtime = None

        if runtime_factory:
            runtime = runtime_factory(row)

        try:
            for turn_idx in range(turn_limit):
                try:
                    # Generate using transformers
                    messages = list(history)
                    max_tokens = job.generation.parameters.get("max_tokens", 4096)
                    temperature = job.generation.parameters.get("temperature", 1.0)

                    # Apply chat template and generate
                    input_text = tokenizer.apply_chat_template(
                        messages,
                        tools=tool_defs if tool_defs else None,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
                    prompt_token_count = input_ids.shape[1]

                    with torch.no_grad():
                        output = model.generate(
                            input_ids,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            return_dict_in_generate=True,
                            output_scores=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    full_ids = output.sequences[0].tolist()
                    generated_ids = full_ids[prompt_token_count:]

                    # Compute logprobs for generated tokens
                    scores = output.scores
                    gen_logprobs = []
                    for i, score in enumerate(scores):
                        log_probs = F.log_softmax(score, dim=-1)
                        token_id = generated_ids[i] if i < len(generated_ids) else 0
                        gen_logprobs.append(log_probs[0, token_id].item())

                    # Full sequence logprobs (0 for prompt tokens)
                    all_logprobs = [0.0] * prompt_token_count + gen_logprobs

                    # Decode generated text
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

                    # Parse response for tool calls using vLLM parsers
                    tool_parser_name = _default_tool_parser(model_name)
                    reasoning_parser_name = _default_reasoning_parser(model_name)
                    parsed = parse_assistant_turn(
                        messages=messages,
                        assistant_text=generated_text,
                        tools=tool_defs if tool_defs else None,
                        tool_choice="auto" if tool_defs else None,
                        tool_parser_name=tool_parser_name,
                        reasoning_parser_name=reasoning_parser_name,
                        tokenizer=tokenizer,
                        token_ids=generated_ids,
                        chat_template_kwargs=None,
                    )
                    content = parsed.content
                    tool_calls = parsed.tool_calls
                    reasoning_content = parsed.reasoning

                    # Build reward mask (1 for generated tokens, 0 for prompt)
                    reward_mask = [0] * prompt_token_count + [1] * len(generated_ids)

                except Exception as exc:
                    if _is_context_window_error(exc):
                        logger.warning(
                            "Prompt=`%s...` turn=%d exceeded context window; ending trajectory.",
                            prompt[:20],
                            turn_idx,
                        )
                        break
                    raise

                if len(full_ids) <= 1:
                    raise JobExecutionError(
                        f"token_ids too short for shifted training: tokens={len(full_ids)}"
                    )

                snapshot = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
                if reasoning_content:
                    snapshot["reasoning_content"] = reasoning_content

                if verbose_turns:
                    logger.info(
                        "prompt=`%s...` turn=%d tool_returned=%s",
                        prompt[:8],
                        turn_idx,
                        bool(tool_calls),
                    )

                history.append(snapshot)
                turn_items.append(
                    {
                        "prompt": prompt,
                        "messages": list(history),
                        "token_ids": full_ids,
                        "logprobs": all_logprobs,
                        "reward_mask": reward_mask,
                        "assistant_content": content,
                        "assistant_reasoning_content": reasoning_content,
                        "assistant_tool_calls": tool_calls,
                        "assistant_raw_text": generated_text,
                        "prompt_token_count": prompt_token_count,
                        "parser_fallback": False,
                        "turn_index": turn_idx,
                        "retokenize_match": True,
                    }
                )

                if not tool_calls:
                    break

                _execute_tool_calls(
                    tool_calls=tool_calls,
                    tool_registry=tool_registry,
                    history=history,
                )

        finally:
            if runtime is not None:
                runtime.cleanup()

        # Combine all turns into a single item for batch training
        if len(turn_items) == 0:
            return []

        if len(turn_items) == 1:
            return turn_items

        # Use the last turn's full sequence
        last_turn = turn_items[-1]
        combined_token_ids = list(last_turn["token_ids"])
        combined_logprobs = list(last_turn["logprobs"])

        # Initialize reward_mask (all masked)
        combined_reward_mask = [0] * len(combined_token_ids)

        # For each turn, unmask its completion region
        for turn_item in turn_items:
            prompt_count = turn_item["prompt_token_count"]
            turn_token_count = len(turn_item["token_ids"])

            completion_start = prompt_count
            completion_end = turn_token_count

            for idx in range(completion_start, completion_end):
                if idx < len(combined_reward_mask):
                    combined_reward_mask[idx] = 1

        # Combine metadata
        all_tool_calls = []
        all_assistant_contents = []
        all_assistant_reasoning = []
        all_assistant_raw_texts = []

        for turn_item in turn_items:
            if turn_item.get("assistant_tool_calls"):
                all_tool_calls.extend(turn_item["assistant_tool_calls"])
            if turn_item.get("assistant_content"):
                all_assistant_contents.append(turn_item["assistant_content"])
            if turn_item.get("assistant_reasoning_content"):
                all_assistant_reasoning.append(turn_item["assistant_reasoning_content"])
            if turn_item.get("assistant_raw_text"):
                all_assistant_raw_texts.append(turn_item["assistant_raw_text"])

        # Store turn information for teacher alignment computation
        turn_info_for_alignment = []
        for turn_item in turn_items:
            turn_info_for_alignment.append(
                {
                    "messages": turn_item["messages"],
                    "assistant_raw_text": turn_item.get("assistant_raw_text"),
                    "completion_start": turn_item["prompt_token_count"],
                    "completion_end": len(turn_item["token_ids"]),
                }
            )

        # Create combined item
        combined_item = {
            "prompt": turn_items[0]["prompt"],
            "messages": last_turn["messages"],
            "token_ids": combined_token_ids,
            "logprobs": combined_logprobs,
            "reward_mask": combined_reward_mask,
            "assistant_content": all_assistant_contents[-1] if all_assistant_contents else "",
            "assistant_reasoning_content": all_assistant_reasoning[-1]
            if all_assistant_reasoning
            else None,
            "assistant_tool_calls": all_tool_calls if all_tool_calls else None,
            "assistant_raw_text": all_assistant_raw_texts[-1] if all_assistant_raw_texts else None,
            "prompt_token_count": turn_items[0]["prompt_token_count"],
            "parser_fallback": last_turn.get("parser_fallback", False),
            "turn_index": len(turn_items) - 1,
            "retokenize_match": True,
            "_combined_turns": turn_info_for_alignment,
        }

        logger.info(
            "prompt=`%s...` combined %d turns into single item: total_tokens=%d reward_tokens=%d",
            prompt[:8],
            len(turn_items),
            len(combined_token_ids),
            sum(combined_reward_mask),
        )

        return [combined_item]

    for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
        if on_batch_start and on_batch_start_lookahead > 0:
            for ahead in range(1, on_batch_start_lookahead + 1):
                next_start = start + ahead * batch_size
                if next_start >= len(prompts):
                    break
                next_chunk = prompts[next_start : next_start + batch_size]
                on_batch_start(next_chunk)

        chunk = prompts[start : start + batch_size]

        # Note: Using single-threaded execution for transformers generation
        # as the model is not thread-safe without additional synchronization
        results = [_run_prompt(row) for row in chunk]

        if on_batch_complete is not None:
            on_batch_complete(chunk)

        turns = [turn for per_prompt in results for turn in per_prompt]
        events.emit(
            "Tool rollout batch ready (accelerate).",
            code="tool_on_policy_accelerate.batch_ready",
            data={
                "batch_index": batch_index,
                "batch_size": len(chunk),
                "total_batches": total_batches,
            },
        )
        if turns:
            yield turns

def _compute_batch_stats(prompts: List[Any], options: Any) -> Tuple[int, int]:
    """Compute batch size and total batches."""
    batch_size = max(1, getattr(options, "groups_per_batch", 64))
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    return batch_size, total_batches


def _is_context_window_error(exc: Exception) -> bool:
    """Check if exception is a context window error."""
    text = str(exc).lower()
    return "context window" in text or "max_tokens" in text
