from __future__ import annotations

import json
import logging
import os
import requests
import time
from tqdm.auto import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from accelerate import Accelerator

logger = logging.getLogger(__name__)


@dataclass
class TrainingBatch:
    """Batch of training data for importance sampling."""

    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    target_ids: torch.Tensor  # [batch, seq_len-1]
    sampling_logprobs: torch.Tensor  # [batch, seq_len-1]
    advantages: torch.Tensor  # [batch, seq_len-1]
    loss_mask: torch.Tensor  # [batch, seq_len-1]


def load_model_with_lora(
    model_name: str,
    lora_rank: int,
    checkpoint_path: str | None = None,
    device_map: str | None = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PeftModel, PreTrainedTokenizer]:
    """Load base model and apply LoRA (or load existing adapter).

    Args:
        model_name: HuggingFace model name or path
        lora_rank: LoRA rank for adapter
        checkpoint_path: Optional path to existing LoRA adapter
        device_map: Device mapping strategy ("auto" for automatic placement)
        torch_dtype: Torch dtype for model weights

    Returns:
        Tuple of (PeftModel, Tokenizer)
    """
    logger.info(
        "Loading model %s with LoRA rank=%d, checkpoint=%s, device_map=%s",
        model_name,
        lora_rank,
        checkpoint_path,
        device_map,
    )

    step_timings = {}

    def _timed_step(step_name):
        start_time = time.perf_counter()
        logger.info("Starting %s", step_name)
        return start_time

    def _finish_step(step_name, start_time, progress_bar):
        elapsed = time.perf_counter() - start_time
        step_timings[step_name] = round(elapsed, 3)
        logger.info("Completed %s in %.2fs", step_name, elapsed)
        progress_bar.update(1)
        progress_bar.set_postfix_str(f"{step_name} {elapsed:.1f}s", refresh=True)

    total_steps = 6
    pbar = tqdm(total=total_steps, desc="load_model_with_lora", unit="step")

    # Build kwargs for from_pretrained
    try:
        step = "build_model_kwargs"
        start = _timed_step(step)
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "device_map": device_map,
            "attn_implementation": "flash_attention_2",
        }
        logger.info("Flash Attention 2 enabled")

        _finish_step(step, start, pbar)
        
        step = "load_base_model"
        start = _timed_step(step)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        _finish_step(step, start, pbar)

        step = "load_tokenizer"
        start = _timed_step(step)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _finish_step(step, start, pbar)

        if checkpoint_path and Path(checkpoint_path).exists():
            step = "load_lora_adapter"
            start = _timed_step(step)
            logger.info("Loading existing LoRA adapter from %s", checkpoint_path)
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            _finish_step(step, start, pbar)
    
        else:
            step = "apply_lora"
            start = _timed_step(step)
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_dropout=0.0,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(base_model, lora_config)
            _finish_step(step, start, pbar)

            step = "print_trainable_parameters"
            start = _timed_step(step)
            model.print_trainable_parameters()
            _finish_step(step, start, pbar)

        step = "enable_gradient_checkpointing"
        start = _timed_step(step)
        # Enable gradient checkpointing to save VRAM during training
        # This trades compute for memory by recomputing activations during backward pass
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for memory efficiency")
        elif hasattr(model, "base_model") and hasattr(model.base_model, "gradient_checkpointing_enable"):
            # For PEFT models, enable on the base model
            model.base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on base model for memory efficiency")
        elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "gradient_checkpointing_enable"):
            # Some PEFT models have base_model.model structure
            model.base_model.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on base_model.model for memory efficiency")
        else:
            logger.warning("Could not enable gradient checkpointing - model may not support it")
        model.config.use_cache = False  # VERY IMPORTANT
        _finish_step(step, start, pbar)
    
    finally:
        pbar.close()
        logger.info(
            "load_model_with_lora timing summary: %s",
            json.dumps(step_timings, sort_keys=False)
        )

    return model, tokenizer


def compute_logprobs_direct(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute logprobs via direct forward pass.

    Args:
        model: Model to use for forward pass
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Optional attention mask [batch, seq_len]

    Returns:
        Log probabilities for each token [batch, seq_len-1]
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # outputs.logits: [batch, seq_len, vocab_size]
    # Shift to get logits for predicting next token
    logits = outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len-1, vocab_size]

    # Gather the logprobs for the actual next tokens
    target_ids = input_ids[:, 1:]  # [batch, seq_len-1]
    gathered_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(
        -1
    )  # [batch, seq_len-1]

    return gathered_logprobs


def compute_logprobs_for_sequence(
    model: torch.nn.Module,
    token_ids: List[int],
    device: torch.device,
) -> List[float]:
    """Compute logprobs for a single sequence.

    Args:
        model: Model to use for forward pass
        token_ids: Token IDs for the sequence
        device: Device to run on

    Returns:
        List of logprobs (one per token, first token has logprob 0)
    """
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_tensor)

    logits = outputs.logits[0, :-1, :]  # [seq_len-1, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)  # [seq_len-1, vocab_size]

    # Gather logprobs for actual tokens
    target_ids = torch.tensor(token_ids[1:], dtype=torch.long, device=device)
    gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # Prepend 0 for the first token (no prediction for it)
    result = [0.0] + gathered.cpu().tolist()
    return result


def importance_sampling_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute importance sampling loss.

    Loss = -sum(exp(target_lp - sampling_lp) * advantages * mask)

    Args:
        model: Model to compute current logprobs
        input_ids: Input token IDs [batch, seq_len] (already shifted, excludes last token)
        target_ids: Target token IDs [batch, seq_len] (predictions for input_ids)
        sampling_logprobs: Log probs from sampling [batch, seq_len]
        advantages: Advantage values [batch, seq_len]
        loss_mask: Mask for which tokens to include [batch, seq_len]

    Returns:
        Tuple of (loss, current_logprobs)
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [batch, seq_len, vocab_size] - no slicing needed, input is already aligned
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather logprobs for target tokens
    current_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # Importance weight: exp(current - sampling)
    importance_weights = torch.exp(current_logprobs - sampling_logprobs)

    # Loss: -sum(importance_weight * advantage * mask)
    loss_per_token = -importance_weights * advantages * loss_mask
    loss = loss_per_token.sum() / (loss_mask.sum() + 1e-8)

    return loss, current_logprobs


def importance_sampling_loss_with_clip(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_ratio: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Compute importance sampling loss with PPO-style clipping.

    This function implements PPO-style clipping to prevent large policy updates
    that could destabilize training. The loss is computed as the pessimistic
    (maximum) of the clipped and unclipped objectives.

    Loss = max(-ratio * advantage, -clipped_ratio * advantage) where:
    - ratio = exp(current_logprob - sampling_logprob)
    - clipped_ratio = clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

    Args:
        model: Model to compute current logprobs
        input_ids: Input token IDs [batch, seq_len] (already shifted, excludes last token)
        target_ids: Target token IDs [batch, seq_len] (predictions for input_ids)
        sampling_logprobs: Log probs from sampling [batch, seq_len]
        advantages: Advantage values [batch, seq_len]
        loss_mask: Mask for which tokens to include [batch, seq_len]
        clip_ratio: PPO clipping ratio (default 0.2)

    Returns:
        Tuple of (loss, current_logprobs, metrics_dict)
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [batch, seq_len, vocab_size]
    
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather logprobs for target tokens
    current_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # Importance weight: exp(current - sampling)
    ratio = torch.exp(current_logprobs - sampling_logprobs)

    # Clip the ratio for stability
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    # Pessimistic loss (take worse of clipped/unclipped)
    loss_unclipped = -ratio * advantages * loss_mask
    loss_clipped = -clipped_ratio * advantages * loss_mask

    # Take the maximum (pessimistic) loss
    loss_per_token = torch.max(loss_unclipped, loss_clipped)
    loss = loss_per_token.sum() / (loss_mask.sum() + 1e-8)

    # Compute metrics for logging
    with torch.no_grad():
        mask_sum = loss_mask.sum().item()
        if mask_sum > 0:
            mean_ratio = (ratio * loss_mask).sum().item() / mask_sum
            mean_clipped_ratio = (clipped_ratio * loss_mask).sum().item() / mask_sum
            clipped_low = ((ratio < 1.0 - clip_ratio) * loss_mask).sum().item()
            clipped_high = ((ratio > 1.0 + clip_ratio) * loss_mask).sum().item()
            clip_fraction = (clipped_low + clipped_high) / mask_sum
        else:
            mean_ratio = 1.0
            mean_clipped_ratio = 1.0
            clip_fraction = 0.0

    metrics = {
        "mean_ratio": mean_ratio,
        "mean_clipped_ratio": mean_clipped_ratio,
        "clip_fraction": clip_fraction,
    }

    return loss, current_logprobs, metrics


def save_checkpoint_accelerate(
    model: PeftModel,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    name: str,
    log_path: str,
    loop_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Save checkpoint in format compatible with tinker-cookbook checkpoints.jsonl.

    Args:
        model: PEFT model to save
        optimizer: Optimizer to save state
        accelerator: Accelerator instance
        name: Checkpoint name
        log_path: Path to log directory
        loop_state: Additional state to save (e.g. batch number)

    Returns:
        Dict with checkpoint paths
    """
    log_dir = Path(log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = log_dir / f"checkpoint-{name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter weights
    adapter_path = checkpoint_dir / "adapter"
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(str(adapter_path))

    # Save optimizer state
    state_path = checkpoint_dir / "state"
    state_path.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(state_path))

    paths = {
        "sampler_path": str(adapter_path),
        "state_path": str(state_path),
    }

    # Write to checkpoints.jsonl for compatibility
    full_dict = {"name": name, **loop_state, **paths}
    checkpoints_file = log_dir / "checkpoints.jsonl"
    with open(checkpoints_file, "a") as f:
        f.write(json.dumps(full_dict) + "\n")

    logger.info("Saved checkpoint: %s", paths)
    return full_dict


class TransformersSamplerContext:
    """Context for generating samples using transformers generate().

    This replaces the Tinker sampling client for student rollouts,
    using the PEFT model directly for inference.
    """

    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._device = next(model.parameters()).device

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None,
        max_tokens: int,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[List[int], List[float]]:
        """Generate completion with logprobs.

        Args:
            messages: Chat messages
            tools: Tool definitions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation kwargs

        Returns:
            Tuple of (token_ids, logprobs) for full sequence
        """
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self._device
        )
        prompt_len = input_ids.shape[1]

        # Generate with logprobs
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        # Get full sequence
        full_ids = output.sequences[0].tolist()

        # Compute logprobs for the generated tokens
        scores = output.scores  # list of [1, vocab_size] tensors
        gen_logprobs = []
        for i, score in enumerate(scores):
            log_probs = F.log_softmax(score, dim=-1)
            token_id = full_ids[prompt_len + i]
            gen_logprobs.append(log_probs[0, token_id].item())

        # Combine: prompt tokens have 0 logprob, generated tokens have actual logprobs
        all_logprobs = [0.0] * prompt_len + gen_logprobs

        return full_ids, all_logprobs

    def refresh_from_checkpoint(self, adapter_path: str) -> None:
        """Reload LoRA weights from a checkpoint.

        Args:
            adapter_path: Path to the adapter checkpoint
        """
        logger.info("Refreshing model from checkpoint: %s", adapter_path)
        self.model.load_adapter(adapter_path, adapter_name="default")


class FireworksTeacherContext:
    """Context for computing teacher logprobs using Fireworks API.

    This avoids loading the teacher model locally, saving significant RAM.
    Requires FIREWORKS_API_KEY environment variable.
    """

    def __init__(
        self,
        model_name: str,
        fireworks_model: str,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize Fireworks teacher context.

        Args:
            model_name: HuggingFace model name (for tokenizer)
            fireworks_model: Fireworks model ID (e.g. "accounts/fireworks/models/llama-v3p1-70b-instruct")
            base_url: Fireworks API base URL
            api_key: Fireworks API key (defaults to FIREWORKS_API_KEY env var)
        """

        self.model_name = model_name
        self.fireworks_model = fireworks_model
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")

        if not self.api_key:
            raise ValueError(
                "FIREWORKS_API_KEY environment variable must be set or api_key must be provided"
            )

        logger.info(
            "Using Fireworks API for teacher model: %s (fireworks: %s)",
            model_name,
            fireworks_model,
        )

        # Thread pool for parallel API calls
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=32)

    def compute_logprobs_for_input_ids(self, input_ids: Sequence[int]) -> List[float]:
        """Compute logprobs for input_ids using Fireworks completions API."""
        fireworks_url = f"{self.base_url}/completions"
        payload = {
            "model": self.fireworks_model,
            "max_tokens": 1,
            "echo": True,
            "logprobs": True,
            "input_ids": list(input_ids),
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.post(fireworks_url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        # Extract logprobs from response
        lp_list = []
        for item in response_json["choices"][0]["logprobs"]["content"]:
            lp_list.append(item.get("logprob", 0.0))

        # Align lengths - use prompt_tokens from usage to slice logprobs
        prompt_tokens = response_json["usage"]["prompt_tokens"]
        # Fireworks returns the prefix/completion tokens + 1 extra completion token.
        # So we will just consider the prompt tokens (prefix/completion).
        # Extract completion logprobs starting after prefix_tokens
        logprobs = lp_list[:prompt_tokens]

        # Log warning if Fireworks returns token_ids that don't match input_ids.
        fw_tokens = response_json.get("choices", [{}])[0].get("logprobs", {}).get("token_ids")
        if fw_tokens is not None:
            if list(fw_tokens) != list(input_ids):
                mismatch_at = None
                for idx, (fw_id, in_id) in enumerate(zip(fw_tokens, input_ids)):
                    if fw_id != in_id:
                        mismatch_at = idx
                        break
                logger.warning(
                    "Fireworks token_ids mismatch: input_len=%d fw_len=%d first_mismatch=%s",
                    len(input_ids),
                    len(fw_tokens),
                    mismatch_at,
                )

        if len(logprobs) != len(input_ids):
            logger.warning(
                "Token count mismatch: API returned %d, input_ids has %d",
                len(logprobs),
                len(input_ids),
            )
            if len(logprobs) < len(input_ids):
                logprobs.extend([0.0] * (len(input_ids) - len(logprobs)))
            else:
                logprobs = logprobs[: len(input_ids)]

        return [lp if lp is not None else 0.0 for lp in logprobs]

    def compute_teacher_alignment(
        self,
        input_ids: Sequence[int],
        student_logprobs: torch.Tensor,
        completion_mask: Sequence[int],
    ) -> Dict[str, object]:
        """Compute teacher alignment for rewards using Fireworks API."""
        if not completion_mask or not any(completion_mask):
            return {
                "kl_adjustments": [0.0] * len(input_ids),
                "kl_mask": [0.0] * len(input_ids),
                "teacher_token_ids": list(input_ids),
                "teacher_logprobs": [0.0] * len(input_ids),
            }

        teacher_logprobs = self.compute_logprobs_for_input_ids(input_ids)
        if len(teacher_logprobs) != len(input_ids):
            if len(teacher_logprobs) < len(input_ids):
                teacher_logprobs.extend([0.0] * (len(input_ids) - len(teacher_logprobs)))
            else:
                teacher_logprobs = teacher_logprobs[: len(input_ids)]

        kl_adjustments = [0.0] * len(input_ids)
        kl_mask = [0.0] * len(input_ids)
        limit = min(len(input_ids), len(student_logprobs), len(completion_mask))
        for idx in range(limit):
            if completion_mask[idx]:
                kl_adjustments[idx] = float(student_logprobs[idx].item()) - teacher_logprobs[idx]
                kl_mask[idx] = 1.0

        return {
            "kl_adjustments": kl_adjustments,
            "kl_mask": kl_mask,
            "teacher_token_ids": list(input_ids),
            "teacher_logprobs": teacher_logprobs,
        }
