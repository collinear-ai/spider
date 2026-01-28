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
import concurrent.futures
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from transformers import get_scheduler

from tqdm.auto import tqdm

# DISABLED: Liger kernel causes corrupted logits with Qwen3 + LoRA + gradient checkpointing
# Symptoms: logits range from -260 to +300, predictions are garbage
# Result: importance sampling ratio ≈ 0, gradient explosion, NaN loss
# from liger_kernel.transformers import apply_liger_kernel_to_qwen3
from spider.config import JobConfig

from .on_policy_accelerate_utils import (
    FireworksTeacherContext,
    importance_sampling_loss_with_clip,
    load_model_with_lora,
    save_checkpoint_accelerate,
)
from .hf_upload import _prepare_hf_payload, publish_to_hub
from .sources import collect_prompts
from . import events
from .writers import JSONLBatchWriter
from .on_policy_vllm_rollouts import VLLMRolloutCollector, rollout_results_to_dicts
from .weight_synchronizer import WeightSynchronizer
from .backends.vllm_backend import VLLMBackend

RolloutBatch = List[Dict[str, Any]]

logger = logging.getLogger(__name__)


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

    if not tool_registry:
        raise JobExecutionError(
            "tool_registry is required for on-policy training (accelerate)."
        )

    # Use vLLM-based separated architecture
    logger.info(
        "Job %s: Using vLLM-based inference with separated training (accelerate)",
        job_id,
    )
    events.emit(
        "Using vLLM-based separated inference/training architecture.",
        code="on_policy_accelerate.vllm_inference",
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
        return JobExecutionResult(
            artifacts_path=artifact_path,
            metrics={"records": 0},
            messages=["No prompts found, skipped on-policy distillation."],
        )

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

    # Upload to HuggingFace if configured
    if job.output.mode == "upload_hf" and job.output.hf and job.output.hf.repo_id:
        try:
            logger.info("Uploading checkpoint to HuggingFace: %s", job.output.hf.repo_id)
            hf_url = publish_to_hub(
                job_id=job_id,
                artifact=hf_payload_dir,
                metadata=metadata_path,
                config=job.output.hf,
            )
            logger.info("Uploaded to HuggingFace: %s", hf_url)
        except Exception as exc:
            logger.error("Failed to upload to HuggingFace: %s", exc)
            # Don't fail the job, just log the error

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
    return result


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
    num_epochs = max(1, getattr(options, "num_epochs", 1))
    total_training_steps = total_batches * num_epochs

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
        import os
        # NOW set training GPUs via environment variable after vLLM has started
        # This ensures the training model loads on the correct GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, training_gpu_ids))
        logger.info(
            "Set CUDA_VISIBLE_DEVICES=%s for training",
            os.environ["CUDA_VISIBLE_DEVICES"],
        )

        # Initialize Accelerator with DeepSpeed
        training_precision = getattr(options, "training_precision", "bf16")
        deepspeed_config = getattr(options, "deepspeed_config", None)
        
        if deepspeed_config:
            # Use DeepSpeed with config file
            config_path = Path(deepspeed_config)
            if not config_path.is_absolute():
                config_path = Path("/home/ubuntu/spider") / config_path
            
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=str(config_path),
                zero_stage=3,
                gradient_accumulation_steps=1,
                gradient_clipping=1.0,
            )
            accelerator = Accelerator(
                mixed_precision=training_precision,
                deepspeed_plugin=deepspeed_plugin,
            )
            logger.info("Accelerator initialized with DeepSpeed (config=%s) and mixed_precision=%s", config_path, training_precision)
        else:
            accelerator = Accelerator(mixed_precision=training_precision)
            logger.info("Accelerator initialized with mixed_precision=%s", training_precision)

        # Delay loading training model until first batch (for testing inference speed)
        # Model will be loaded lazily when first needed for training
        model = None
        tokenizer = None
        optimizer = None
        scheduler = None
        training_model_loaded = False
        training_model_future = None
        training_model_lock = threading.Lock()
        training_model_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def _load_training_model():
            nonlocal model, tokenizer, optimizer, scheduler, training_model_loaded
            with training_model_lock:
                if training_model_loaded:
                    return
            
            logger.info("Loading training model (async)...")
            load_start = time.time()
            pbar = tqdm(
                total=3,
                desc="Loading training model",
                unit="step",
                position=1,
                file=sys.stderr,
                leave=True,
            )

            phase_start = time.time()

            # Determine torch dtype based on training precision
            training_precision = getattr(options, "training_precision", "bf16")
            if training_precision == "fp32":
                model_dtype = torch.float32
            elif training_precision == "fp16":
                model_dtype = torch.float16
            else:  # bf16 or anything else
                model_dtype = torch.bfloat16
            logger.info("Loading model with dtype %s (training_precision=%s)", model_dtype, training_precision)

            model, tokenizer = load_model_with_lora(
                model_name=student_model,
                lora_rank=options.lora_rank,
                checkpoint_path=student_checkpoint,
                torch_dtype=model_dtype,
            )
            pbar.update(1)
            logger.info("Training model weights loaded in %.2fs", time.time() - phase_start)

            # Create optimizer
            phase_start = time.time()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=options.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
            
            # LR scheduler: cosine with warmup (fixed 10 steps)
            warmup_steps = int(getattr(options, "warmup_steps", 10) or 10)
            scheduler = get_scheduler(
                name="cosine",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps,
            )
            logger.info(
                "LR scheduler: cosine with %d warmup steps out of %d total steps",
                warmup_steps, total_training_steps,
            )
            pbar.update(1)
            logger.info("Training optimizer created in %.2fs", time.time() - phase_start)

            # Prepare with accelerator
            phase_start = time.time()
            model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
            pbar.update(1)
            logger.info("Accelerator prepare completed in %.2fs", time.time() - phase_start)
            pbar.close()

            training_model_loaded = True
            logger.info("Training model loaded and prepared in %.2fs", time.time() - load_start)

        def _start_training_model_load_async():
            nonlocal training_model_future
            with training_model_lock:
                if training_model_loaded or training_model_future:
                    return
                training_model_future = training_model_executor.submit(_load_training_model)

        def _ensure_training_model_loaded():
            nonlocal training_model_future
            if training_model_loaded:
                return
            if training_model_future is not None:
                training_model_future.result()
                return
            _load_training_model()

        _start_training_model_load_async()

        training_dir = workspace / "training"
        training_dir.mkdir(parents=True, exist_ok=True)

        # Create weight synchronizer (will sync after model is loaded)
        weight_sync_steps = getattr(options, "weight_sync_steps", 1)
        save_checkpoint_steps = getattr(options, "save_checkpoint_steps", 5)
        synchronizer = WeightSynchronizer(
            sync_every_n_steps=weight_sync_steps,
            checkpoint_dir=training_dir / "lora_checkpoints",
            vllm_base_url=vllm_backend.base_url,
            lora_name="student",
            vllm_backend=vllm_backend,
            save_every_n_steps=save_checkpoint_steps,
        )

        # CRITICAL: Wait for training model to load and sync BEFORE collecting rollouts
        # This ensures importance ratio starts at ~1.0 (same model for sampling and training)
        logger.info("Waiting for training model to load before collecting rollouts...")
        _ensure_training_model_loaded()
        synchronizer.force_sync(model, accelerator, step=0)
        initial_lora_name = synchronizer.get_lora_name()
        logger.info("Initial weight sync completed. LoRA adapter loaded: %s", initial_lora_name)

        # Create rollout collector with the synced LoRA adapter
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
            max_tokens=job.generation.parameters.get("max_tokens", 16384),  # High default, let model generate freely
            temperature=job.generation.parameters.get("temperature", 1.0),
            tool_timeout=getattr(options, "tool_timeout", 60.0),  # None = no timeout
            lora_name=initial_lora_name,  # Use synced LoRA from the start
            runtime_factory=runtime_factory,
            verbose=verbose_turns,
        )

        # Load teacher model using Fireworks API
        fireworks_model = getattr(options, "fireworks_model", None)
        if not fireworks_model:
            raise ValueError("fireworks_model must be specified in options")
        
        teacher_ctx = FireworksTeacherContext(
            model_name=options.teacher,
            fireworks_model=fireworks_model,
        )
        logger.info("Using Fireworks API for teacher: %s", fireworks_model)
        
        # Async teacher alignment: submit turns as they complete during rollouts
        alignment_futures = []  # Shared list for async alignment futures
        alignment_lock = threading.Lock()
        
        def _on_turn_complete(turn_result):
            """Callback to submit turn for teacher alignment while rollouts continue."""
            try:
                future = teacher_ctx._executor.submit(
                    teacher_ctx.compute_teacher_alignment,
                    messages=turn_result.messages,
                    tools=tool_defs,
                    student_model=student_model,
                    student_token_ids=turn_result.token_ids,
                    reward_mask=turn_result.reward_mask,
                    assistant_raw_text=turn_result.assistant_raw_text,
                )
                with alignment_lock:
                    alignment_futures.append((turn_result, future))
                    logger.debug(f"Queued async alignment: trace_id={turn_result.trace_id}, total_queued={len(alignment_futures)}")
            except Exception as e:
                logger.error(f"Error in _on_turn_complete callback: {e}")
        
        # Update collector to use the callback
        collector.on_turn_complete = _on_turn_complete

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

        def _process_batch(
            batch_index: int, trajectories: List[Dict[str, Any]]
        ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
            """Process a batch of trajectories and perform a training step."""
            
            items = []
            batch_metrics = {
                "kl_sum": 0.0,
                "kl_count": 0,
                "advantage_sum": 0.0,
                "advantage_count": 0,
                "clip_fraction_sum": 0.0,
                "clip_count": 0,
            }

            # Build trace_id -> turn_index mapping for trajectories
            trace_id_to_idx = {}
            for turn_index, turn in enumerate(trajectories):
                trace_id = turn.get("trace_id")
                if trace_id:
                    trace_id_to_idx[trace_id] = turn_index
            
            # Wait for async alignment futures (submitted during rollouts)
            alignment_results = {}
            with alignment_lock:
                pending_futures = list(alignment_futures)
                alignment_futures.clear()  # Clear for next batch
            
            logger.info(f"Processing batch: {len(trajectories)} trajectories, {len(pending_futures)} async alignment futures")
            
            if pending_futures:
                logger.info("Waiting for async teacher alignment: %d tasks (started during rollouts)", len(pending_futures))
                teacher_start = time.time()
                completed_count = 0
                last_log_time = time.time()
                
                for turn_result, future in pending_futures:
                    completed_count += 1
                    now = time.time()
                    if now - last_log_time >= 10.0:
                        logger.info(
                            "Teacher alignment progress: %d/%d (%.0f%%), %.1fs elapsed",
                            completed_count, len(pending_futures), 100.0 * completed_count / len(pending_futures),
                            now - teacher_start
                        )
                        last_log_time = now
                    
                    # Find matching trajectory turn using trace_id
                    turn_idx = trace_id_to_idx.get(turn_result.trace_id, -1)
                    
                    if turn_idx < 0:
                        logger.warning(f"No match for trace_id={turn_result.trace_id}")
                    
                    try:
                        alignment = future.result()
                    except Exception as e:
                        logger.error(f"Error in async teacher alignment: {e}")
                        alignment = {
                            "kl_adjustments": [0.0] * len(turn_result.token_ids),
                            "kl_mask": [0.0] * len(turn_result.token_ids),
                        }
                    
                    if turn_idx >= 0:
                        if turn_idx not in alignment_results:
                            alignment_results[turn_idx] = []
                        alignment_results[turn_idx].append({
                            "task": {
                                "turn_index": turn_idx,
                                "turn_info": None,
                                "token_ids": turn_result.token_ids,
                            },
                            "alignment": alignment,
                        })
                
                teacher_elapsed = time.time() - teacher_start
                logger.info(
                    "Teacher alignment complete: %d tasks in %.1fs (%.1f tasks/s, overlapped with rollouts)",
                    len(pending_futures), teacher_elapsed, len(pending_futures) / teacher_elapsed if teacher_elapsed > 0 else 0
                )
            else:
                logger.warning("No async alignments found, using zero KL for all")
            
            # Training setup
            num_trajectories = len(trajectories)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            optimizer.zero_grad()
            total_loss_value = 0.0
            total_clip_fraction = 0.0
            valid_count = 0
            
            debug_loss_dir = Path("/home/ubuntu/spider/debug_loss")
            debug_loss_dir.mkdir(exist_ok=True)
            
            # Process each trajectory: build tensors -> train -> free memory (one at a time)
            for turn_index, turn in enumerate(trajectories):
                token_ids = turn["token_ids"]
                logprobs = turn["logprobs"]
                reward_mask = turn["reward_mask"]
                
                # Skip sequences that are too long (OOM prevention)
                seq_len = len(token_ids) - 1  # input_ids length
                if seq_len > 16384:
                    logger.warning(f"Skipping traj {turn_index}: seq_len={seq_len} > 16384")
                    continue
                
                # Get alignment (fallback to zero aligned teacher logprobs)
                results = alignment_results.get(turn_index)
                if results:
                    teacher_alignment = results[0]["alignment"]
                else:
                    teacher_alignment = {
                        "kl_adjustments": [0.0] * len(token_ids),
                        "kl_mask": [0.0] * len(token_ids),
                        "aligned_teacher_logprobs": [0.0] * len(token_ids),
                    }
                
                # Get aligned teacher logprobs for KL computation inside loss function
                aligned_teacher_lp = teacher_alignment.get("aligned_teacher_logprobs") or [0.0] * len(token_ids)
                kl_coef = float(getattr(options, "kl_penalty_coef", 1.0))
                
                input_tokens = token_ids[:-1]
                target_tokens = token_ids[1:]
                mask_tokens = reward_mask[1:]
                logprobs_tokens = torch.tensor(logprobs, dtype=torch.float32)[1:]
                aligned_teacher_tokens = torch.tensor(aligned_teacher_lp, dtype=torch.float32)[1:]
                
                mask_tensor_cpu = torch.tensor(mask_tokens, dtype=torch.float32)
                
                # Move to GPU only for this trajectory
                input_ids = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
                target_ids = torch.tensor(target_tokens, dtype=torch.long, device=device).unsqueeze(0)
                sampling_lp = logprobs_tokens.to(device).unsqueeze(0)
                aligned_teacher_gpu = aligned_teacher_tokens.to(device).unsqueeze(0)
                loss_mask = mask_tensor_cpu.to(device).unsqueeze(0)
                
                # Metrics tracking (before GPU tensors get deleted)
                kl_adj = teacher_alignment.get("kl_adjustments") or [0.0] * len(token_ids)
                kl_tensor_cpu = torch.tensor(kl_adj, dtype=torch.float32)
                kl_tensor = kl_tensor_cpu.to(device)
                mask_bool = torch.tensor(reward_mask, dtype=torch.bool, device=device)
                kl_vals = kl_tensor[mask_bool].tolist() if kl_tensor.numel() > 0 else []
                
                # Note: advantages are now computed inside loss function from aligned_teacher_logprobs
                # Pre-compute expected advantage for metrics (will be recomputed with fresh logprobs in loss fn)
                batch_metrics["kl_sum"] += sum(kl_vals)
                batch_metrics["kl_count"] += len(kl_vals)
                
                # Store item for return (student_logprobs no longer needed, computed fresh in loss fn)
                item = dict(turn)
                item.update({
                    "kl_adjustments": teacher_alignment.get("kl_adjustments"),
                    "kl_mask": teacher_alignment.get("kl_mask"),
                    "aligned_teacher_logprobs": teacher_alignment.get("aligned_teacher_logprobs"),
                })
                items.append(item)
                
                if verbose_turns:
                    logger.info(
                        "vllm rollout batch=%d batch_item_idx=%d n_tokens=%d n_reward_tokens=%d",
                        batch_index, turn_index, len(token_ids), sum(reward_mask),
                    )
                
                # Log GPU memory before forward pass
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(
                    "Before loss fn: traj=%d/%d seq_len=%d mem_allocated=%.2fGB mem_reserved=%.2fGB",
                    turn_index, num_trajectories, input_ids.shape[1], mem_allocated, mem_reserved
                )
                
                # Forward pass - KL and advantages computed inside with gradients
                loss, current_logprobs, metrics = importance_sampling_loss_with_clip(
                    model=model, input_ids=input_ids, target_ids=target_ids,
                    sampling_logprobs=sampling_lp, aligned_teacher_logprobs=aligned_teacher_gpu,
                    loss_mask=loss_mask, clip_ratio=clip_ratio, kl_coef=kl_coef,
                )
                
                # Track advantage metrics from loss function output
                batch_metrics["advantage_sum"] += metrics.get("mean_advantage", 0.0) * loss_mask.sum().item()
                batch_metrics["advantage_count"] += int(loss_mask.sum().item())
                
                # Guard against NaN/Inf
                if not torch.isfinite(loss):
                    logger.error("NaN/Inf loss at batch %d traj %d, skipping", batch_index, turn_index)
                    del input_ids, target_ids, sampling_lp, aligned_teacher_gpu, loss_mask, loss, current_logprobs, metrics
                    torch.cuda.empty_cache()
                    continue
                
                # Backward pass immediately (micro-batching)
                scaled_loss = loss / num_trajectories
                accelerator.backward(scaled_loss)
                
                total_loss_value += loss.detach().item()
                total_clip_fraction += metrics["clip_fraction"]
                valid_count += 1
                
                # Free GPU memory immediately
                del input_ids, target_ids, sampling_lp, aligned_teacher_gpu, loss_mask, loss, scaled_loss, current_logprobs, metrics
                del kl_tensor, mask_bool
                gc.collect()
                torch.cuda.empty_cache()
            
            if valid_count == 0:
                logger.warning("No valid trajectories in batch %d, skipping", batch_index)
                return items, batch_metrics
            
            # Average loss value for logging
            loss_value = total_loss_value / valid_count
            
            # Gradient clipping and optimizer step
            max_grad_norm = float(getattr(options, "max_grad_norm", 1.0))
            
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    has_nan_grad = True
                    logger.error("NaN gradient in %s", name)
                    break
            
            if has_nan_grad:
                logger.error("NaN gradient encountered at batch %d, aborting", batch_index)
                optimizer.zero_grad()
                sys.exit(1)
            else:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
            
            batch_metrics["clip_fraction_sum"] += total_clip_fraction
            batch_metrics["clip_count"] += valid_count
            
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
        shuffle = getattr(job.source, "shuffle", False)

        last_checkpoint = None
        global_step = 0
        token_cum = 0
        
        # Total batches across all epochs
        total_batches_all_epochs = total_batches * num_epochs
        
        # Create progress bar for training batches
        # Use position=0 and file=sys.stderr to persist at bottom of terminal
        train_pbar = tqdm(
            total=total_batches_all_epochs,
            desc="Training",
            unit="batch",
            initial=0,
            ncols=120,
            position=0,  # Fixed position at bottom
            file=sys.stderr,  # Use stderr so it doesn't interfere with stdout logging
            leave=True,  # Keep visible after completion
            mininterval=0.5,  # Update at least every 0.5 seconds
        )

        for epoch in range(num_epochs):
            logger.info("Starting epoch %d/%d", epoch + 1, num_epochs)
            
            # Shuffle dataset at start of each epoch (except first if already shuffled)
            if shuffle and epoch > 0:
                import random
                random.shuffle(prompt_rows)
                logger.info("Shuffled dataset for epoch %d", epoch + 1)
            
            for batch_index_in_epoch, start in enumerate(range(0, len(prompt_rows), batch_size)):
                # Global batch index across all epochs
                batch_index = epoch * total_batches + batch_index_in_epoch
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
                logger.info(
                    "Epoch %d/%d: Collecting rollouts for batch %d/%d (%d prompts)",
                    epoch + 1, num_epochs, batch_index_in_epoch + 1, total_batches, len(chunk)
                )
                rollout_results = collector.collect_batch(chunk)
                rollout_time = time.time() - rollout_start
                trajectories = rollout_results_to_dicts(rollout_results)
                
                logger.info(
                    "Epoch %d/%d: Rollout collection complete for batch %d/%d: time=%.2fs trajectories=%d",
                    epoch + 1, num_epochs, batch_index_in_epoch + 1, total_batches,
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
                try:
                    import os
                    import pickle
                    from pathlib import Path
                    debug_dir = Path("debug_traj_b4_training")
                    debug_dir.mkdir(exist_ok=True)
                    fname = f"trajectories_batch{batch_index}_step{global_step}.pkl"
                    fpath = debug_dir / fname
                    with open(fpath, "wb") as f:
                        pickle.dump(trajectories, f)

                    batch_items, batch_metrics = _process_batch(batch_index, trajectories)
                except RuntimeError as e:
                    if "Non-finite loss" in str(e):
                        logger.error("Aborting training at batch %d due to non-finite loss", batch_index)
                        break
                    raise

                training_time = time.time() - training_start
                batch_time = time.time() - batch_start
                global_step += 1

                step_tokens = sum(int(sum(turn.get("reward_mask") or [])) for turn in trajectories)
                token_cum += step_tokens
                
                # Calculate throughput metrics
                tokens_per_sec = step_tokens / batch_time if batch_time > 0 else 0

                # Update progress bar
                train_pbar.update(1)
                train_pbar.set_description(f"Training (epoch {epoch + 1}/{num_epochs})")
                train_pbar.set_postfix({
                    "loss": f"{batch_metrics.get('loss', 0):.4f}" if batch_metrics.get('loss') else "N/A",
                    "tokens": step_tokens,
                    "cum_tokens": token_cum,
                    "time": f"{batch_time:.1f}s",
                    "tokens/s": f"{tokens_per_sec:.1f}",
                })

                logger.info(
                    "Epoch %d/%d: vllm rollout batch=%d/%d step_time=%.3fs (rollout=%.2fs train=%.2fs) "
                    "token_step=%d token_cum=%d tokens/s=%.1f token_budget=%s",
                    epoch + 1, num_epochs,
                    batch_index_in_epoch + 1,
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
                    # Get current learning rate from scheduler
                    current_lr = scheduler.get_last_lr()[0] if scheduler else options.learning_rate
                    log_data = {
                        "epoch": epoch + 1,
                        "progress/done_frac": (batch_index + 1) / total_batches_all_epochs if total_batches_all_epochs > 0 else 1.0,
                        "progress/epoch_done_frac": (batch_index_in_epoch + 1) / total_batches if total_batches > 0 else 1.0,
                        "optim/lr": current_lr,
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
                        log_data["advantage_mean"] = adv_mean
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
                        loop_state={"epoch": epoch + 1, "batch": batch_index + 1},
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
            
            logger.info("Completed epoch %d/%d", epoch + 1, num_epochs)

        # Close progress bar
        train_pbar.close()
        logger.info(
            "Training complete: %d epochs, %d batches/epoch, %d total batches, %d total tokens",
            num_epochs, total_batches, total_batches_all_epochs, token_cum
        )

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
        if 'training_model_executor' in locals():
            training_model_executor.shutdown(wait=False)
        if 'synchronizer' in locals():
            synchronizer.close()
        vllm_backend.close()


def _compute_batch_stats(prompts: List[Any], options: Any) -> Tuple[int, int]:
    """Compute batch size and total batches."""
    batch_size = max(1, getattr(options, "groups_per_batch", 64))
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    return batch_size, total_batches


