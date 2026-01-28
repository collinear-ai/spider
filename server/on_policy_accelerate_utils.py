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

from tinker_cookbook.distillation.train_on_policy import _compute_groupwise_reverse_kl, compute_aligned_teacher_logprobs

logger = logging.getLogger(__name__)


def _normalize_tool_calls_args_for_qwen3(msgs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Convert string arguments in tool_calls to dict for Qwen3 chat template compatibility.
    
    Qwen3 chat template expects arguments as dict, but OpenAI protocol (and vLLM) returns
    arguments as a JSON string. This function normalizes the format.
    
    Args:
        msgs: List of message dicts that may contain tool_calls
        
    Returns:
        List of normalized message dicts with arguments converted to dict
    """
    normalized = []
    for msg in msgs:
        msg_copy = dict(msg)
        if msg_copy.get("role") == "assistant" and msg_copy.get("tool_calls"):
            tool_calls = []
            for tc in msg_copy["tool_calls"]:
                tc_copy = dict(tc)
                func = tc_copy.get("function", {})
                if isinstance(func, dict):
                    func_copy = dict(func)
                    args = func_copy.get("arguments")
                    # Convert string arguments to dict (Qwen3 bug fix)
                    if isinstance(args, str):
                        try:
                            func_copy["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError) as e:
                            # If parsing fails, keep as string but log warning
                            logger.warning(
                                "Failed to parse tool call arguments as JSON (keeping as string): %s",
                                args[:100] if args else None
                            )
                            # Keep original string if parsing fails
                    tc_copy["function"] = func_copy
                tool_calls.append(tc_copy)
            msg_copy["tool_calls"] = tool_calls
        normalized.append(msg_copy)
    return normalized


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

        step = "configure_for_training"
        start = _timed_step(step)
        # Enable input gradients for LoRA (required for gradients to flow through frozen layers)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        # Disable KV cache (incompatible with training)
        model.config.use_cache = False
        _finish_step(step, start, pbar)
        # Note: Gradient/activation checkpointing handled by DeepSpeed if enabled
    
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
    aligned_teacher_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    kl_coef: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Compute importance sampling loss with PPO-style clipping.

    This function implements PPO-style clipping to prevent large policy updates
    that could destabilize training. The loss is computed as the pessimistic
    (maximum) of the clipped and unclipped objectives.

    Key design: Gradients flow through the advantages (computed from reverse KL),
    while the importance ratio is stop-graded. This allows the model to learn
    to match the teacher's distribution.

    Loss = max(-ratio * advantage, -clipped_ratio * advantage) where:
    - ratio = exp(current_logprob - sampling_logprob).detach()  # stop-grad
    - advantage = -kl_coef * (current_logprob - teacher_logprob)  # has gradients
    - clipped_ratio = clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

    Args:
        model: Model to compute current logprobs
        input_ids: Input token IDs [batch, seq_len] (already shifted, excludes last token)
        target_ids: Target token IDs [batch, seq_len] (predictions for input_ids)
        sampling_logprobs: Log probs from sampling [batch, seq_len]
        aligned_teacher_logprobs: Teacher log probs aligned to student tokens [batch, seq_len]
        loss_mask: Mask for which tokens to include [batch, seq_len]
        clip_ratio: PPO clipping ratio (default 0.2)
        kl_coef: Coefficient for KL penalty (default 1.0)

    Returns:
        Tuple of (loss, current_logprobs, metrics_dict)
    """
    # Log input shape for debugging OOM
    seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else len(input_ids)
    with open("/home/ubuntu/spider/debug_training_shapes.txt", "a") as f:
        f.write(f"input_ids.shape={list(input_ids.shape)} seq_len={seq_len}\n")
    
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [batch, seq_len, vocab_size]
    
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather logprobs for target tokens (has gradients)
    current_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    
    # Clear logits and log_probs immediately after gathering (they're large tensors)
    # These can be huge: [batch, seq_len, vocab_size] where vocab_size ~50k+
    del logits, log_probs

    # Compute reverse KL: log p_student - log p_teacher (has gradients through current_logprobs)
    reverse_kl = current_logprobs - aligned_teacher_logprobs
    
    # Compute advantages from KL (gradients flow through here)
    # Negative KL means: if student > teacher, advantage is negative (discourage)
    # if student < teacher, advantage is positive (encourage)
    advantages = -kl_coef * reverse_kl

    # Importance weight: exp(current - sampling) with STOP GRAD
    # We don't want gradients through the ratio, only through advantages
    ratio = torch.exp(current_logprobs - sampling_logprobs).detach()

    # Clip the ratio for stability
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    # Pessimistic loss (take worse of clipped/unclipped)
    # Gradients flow through advantages -> reverse_kl -> current_logprobs -> model
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
            mean_reverse_kl = (reverse_kl * loss_mask).sum().item() / mask_sum
            mean_advantage = (advantages * loss_mask).sum().item() / mask_sum
        else:
            mean_ratio = 1.0
            mean_clipped_ratio = 1.0
            clip_fraction = 0.0
            mean_reverse_kl = 0.0
            mean_advantage = 0.0

    metrics = {
        "mean_ratio": mean_ratio,
        "mean_clipped_ratio": mean_clipped_ratio,
        "clip_fraction": clip_fraction,
        "mean_reverse_kl": mean_reverse_kl,
        "mean_advantage": mean_advantage,
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


def render_chat_tokens(
    *,
    messages: Sequence[Dict[str, object]],
    tools: Sequence[Dict[str, object]] | None,
    model_name: str,
    add_generation_prompt: bool = False,
) -> Tuple[List[int], int]:
    """Render chat messages to tokens using the model's chat template."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    # Fix Qwen3 chat template bug: convert string arguments to dict
    normalized_messages = _normalize_tool_calls_args_for_qwen3(messages)
    prompt_text = tokenizer.apply_chat_template(
        normalized_messages,
        tools=tools or None,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    tokens = tokenizer.encode(prompt_text)
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    prompt_len = len(tokens)
    return tokens, prompt_len


def compute_teacher_logprobs_direct(
    *,
    model: torch.nn.Module,
    messages: Sequence[Dict[str, object]],
    tools: Sequence[Dict[str, object]] | None,
    teacher_model: str,
    device: torch.device,
) -> Tuple[List[int], int, List[float]]:
    """Compute teacher logprobs using direct forward pass.

    Args:
        model: Teacher model
        messages: Chat messages
        tools: Tool definitions
        teacher_model: Teacher model name (for tokenizer)
        device: Device to run on

    Returns:
        Tuple of (token_ids, prompt_len, logprobs)
    """
    token_ids, prompt_len = render_chat_tokens(
        messages=messages,
        tools=tools,
        model_name=teacher_model,
        add_generation_prompt=False,
    )

    logprobs = compute_logprobs_for_sequence(model, token_ids, device)

    return token_ids, prompt_len, logprobs


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
        # Fix Qwen3 chat template bug: convert string arguments to dict
        normalized_messages = _normalize_tool_calls_args_for_qwen3(messages)
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            normalized_messages,
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

        # Load tokenizer locally (small memory footprint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Thread pool for parallel API calls (reduced from 32 to avoid rate limits)
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=16)

    def compute_logprobs(self, text: str) -> Tuple[List[int], List[float]]:
        """Compute logprobs for text using Fireworks completions API.

        Args:
            text: Full text to compute logprobs for

        Returns:
            Tuple of (token_ids, logprobs)
        """

        fireworks_url = f"{self.base_url}/completions"
        payload = {
            "model": self.fireworks_model,
            "max_tokens": 1,
            "echo": True,
            "logprobs": True,
            "prompt": text,
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Retry with exponential backoff
        max_retries = 5
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                response = requests.post(fireworks_url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                response_json = response.json()
                break
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:
                    raise  # Re-raise on final attempt
                delay = base_delay * (2 ** attempt)  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                logger.warning(
                    "Fireworks API error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, max_retries, str(e)[:100], delay
                )
                time.sleep(delay)

        # Extract logprobs from response
        lp_list = [
            item["logprob"]
            for item in response_json["choices"][0]["logprobs"]["content"]
        ]

        # Get token IDs using local tokenizer
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Align lengths - use prompt_tokens from usage to slice logprobs
        prompt_tokens = response_json["usage"]["prompt_tokens"]
        # Fireworks returns the prefix/completion tokens + 1 extra completion token.
        # So we will just consider the prompt tokens (prefix/completion).
        # Extract completion logprobs starting after prefix_tokens
        logprobs = lp_list[:prompt_tokens]

        # Align with tokenizer output
        if len(logprobs) != len(token_ids):
            logger.warning(
                "Token count mismatch: API returned %d, tokenizer has %d",
                len(logprobs),
                len(token_ids),
            )
            if len(logprobs) < len(token_ids):
                logprobs.extend([0.0] * (len(token_ids) - len(logprobs)))
            else:
                logprobs = logprobs[: len(token_ids)]

        return token_ids, [lp if lp is not None else 0.0 for lp in logprobs]

    def compute_logprobs_batch(self, texts: List[str]) -> List[Tuple[List[int], List[float]]]:
        """Compute logprobs for multiple texts in parallel.
        
        Args:
            texts: List of texts to compute logprobs for
            
        Returns:
            List of (token_ids, logprobs) tuples
        """
        from concurrent.futures import as_completed
        
        futures = {self._executor.submit(self.compute_logprobs, text): i 
                  for i, text in enumerate(texts)}
        
        results = [None] * len(texts)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Error computing logprobs for text {idx}: {e}")
                # Fallback: return empty result
                token_ids = self.tokenizer.encode(texts[idx], add_special_tokens=False)
                results[idx] = (token_ids, [0.0] * len(token_ids))
        
        return results

    def compute_teacher_alignment(
        self,
        messages: Sequence[Dict[str, object]],
        tools: Sequence[Dict[str, object]] | None,
        student_model: str,
        student_token_ids: Sequence[int],
        reward_mask: Sequence[int],
        assistant_raw_text: str,
    ) -> Dict[str, object]:
        """Compute teacher alignment for rewards using Fireworks API.
        
        Note: student_logprobs are no longer needed since KL is computed
        in the loss function using fresh logprobs from forward pass.
        """
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Messages must end with an assistant turn.")

        student_tokenizer = AutoTokenizer.from_pretrained(student_model, use_fast=True, trust_remote_code=True)
        teacher_tokenizer = self.tokenizer

        prefix_msgs = list(messages[:-1])
        
        # Fix Qwen3 chat template bug: convert string arguments to dict
        prefix_msgs = _normalize_tool_calls_args_for_qwen3(prefix_msgs)

        prefix_text = teacher_tokenizer.apply_chat_template(
            prefix_msgs,
            tools=tools if tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )
        prefix_tokens = teacher_tokenizer.encode(prefix_text, add_special_tokens=False)

        if assistant_raw_text is None:
            raise ValueError("Assistant raw text is required for teacher alignment")

        completion_tokens = teacher_tokenizer.encode(
            assistant_raw_text,
            add_special_tokens=False,
        )

        full_text = prefix_text + assistant_raw_text

        # Get logprobs from Fireworks (uses thread pool for parallelization)
        _, full_logprobs = self.compute_logprobs(full_text)

        completion_lp = full_logprobs[len(prefix_tokens):]

        if len(completion_lp) < len(completion_tokens):
            completion_lp.extend([0.0] * (len(completion_tokens) - len(completion_lp)))
        elif len(completion_lp) > len(completion_tokens):
            completion_lp = completion_lp[:len(completion_tokens)]

        try:
            start = list(reward_mask).index(1)
            end = len(reward_mask) - list(reversed(reward_mask)).index(1)
        except ValueError:
            return {
                "aligned_teacher_logprobs": [0.0] * len(student_token_ids),
                "kl_mask": [0.0] * len(student_token_ids),
                "teacher_token_ids": list(completion_tokens),
                "teacher_logprobs": list(completion_lp),
            }

        student_ids = list(student_token_ids)[start:end]
        student_mask = torch.tensor(
            list(reward_mask)[start:end],
            dtype=torch.float32,
        )

        teacher_lp_tensor = torch.tensor(
            completion_lp,
            dtype=torch.float32,
        )

        aligned_teacher_slice, kl_mask_slice = compute_aligned_teacher_logprobs(
            student_tokenizer,
            student_ids,
            teacher_tokenizer,
            completion_tokens,
            teacher_lp_tensor,
            student_mask,
        )

        # Pad results to full sequence length
        aligned_teacher_logprobs = torch.zeros(len(student_token_ids), dtype=torch.float32)
        kl_mask = torch.zeros(len(student_token_ids), dtype=torch.float32)
        aligned_teacher_logprobs[start:end] = aligned_teacher_slice
        kl_mask[start:end] = kl_mask_slice

        return {
            "aligned_teacher_logprobs": aligned_teacher_logprobs.tolist(),
            "kl_mask": kl_mask.tolist(),
            "teacher_token_ids": list(completion_tokens),
            "teacher_logprobs": list(completion_lp),
        }
