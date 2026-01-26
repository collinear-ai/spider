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

import flash_attn
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from accelerate import Accelerator

from tinker_cookbook.distillation.train_on_policy import _compute_groupwise_reverse_kl

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
        device_map: Device mapping strategy (None means no device mapping, for DeepSpeed)
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
        }
        # Only add device_map if it's not None (None means let Accelerate/DeepSpeed handle it)
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        else:
            # When device_map is None, load on CPU first to avoid default GPU placement
            # Accelerate will move it to the correct device during prepare()
            model_kwargs["device_map"] = "cpu"
    
        # Enable Flash Attention 2 for memory efficiency (if available)
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled for memory efficiency")

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
    logits = outputs.logits  # [batch, seq_len, vocab_size] - no slicing needed, input is already aligned
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather logprobs for target tokens
    current_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    
    # Delete logits and log_probs immediately - they're large and not needed for backward
    # Only current_logprobs is needed for the computation graph
    del logits, log_probs, outputs

    # Importance weight: exp(current - sampling)
    ratio = torch.exp(current_logprobs - sampling_logprobs)

    # Clip the ratio for stability
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    # Pessimistic loss (take worse of clipped/unclipped)
    # For positive advantages: we want to maximize ratio * advantage
    # For negative advantages: we want to minimize ratio * (-advantage)
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
            # Count how many tokens were clipped
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
    
    # DeepSpeed handles checkpointing differently
    if accelerator.state.deepspeed_plugin is not None:
        # DeepSpeed checkpoint - save using DeepSpeed's method
        # Note: DeepSpeed checkpoints are saved automatically during training
        # We still save the LoRA adapter separately for vLLM sync
        accelerator.save_state(str(state_path))
    else:
        # Standard Accelerate checkpoint
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
    prompt_text = tokenizer.apply_chat_template(
        messages,
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

        # Load tokenizer locally (small memory footprint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Thread pool for parallel API calls
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=8)

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

        response = requests.post(fireworks_url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

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
        student_logprobs: torch.Tensor,
        reward_mask: Sequence[int],
        assistant_raw_text: str,
    ) -> Dict[str, object]:
        """Compute teacher alignment for rewards using Fireworks API."""
        # Validate messages are proper dicts
        if not messages:
            logger.warning("Empty messages list passed to compute_teacher_alignment")
            return {
                "kl_adjustments": [0.0] * len(student_token_ids),
                "kl_mask": [0.0] * len(student_token_ids),
            }
        
        # Ensure all messages are dicts (not other types)
        validated_messages = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                logger.error("Message %d is not a dict: type=%s value=%s", i, type(msg), str(msg)[:100])
                return {
                    "kl_adjustments": [0.0] * len(student_token_ids),
                    "kl_mask": [0.0] * len(student_token_ids),
                }
            validated_messages.append(msg)
        
        if validated_messages[-1].get("role") != "assistant":
            raise ValueError("Messages must end with an assistant turn.")

        student_tokenizer = AutoTokenizer.from_pretrained(student_model, use_fast=True, trust_remote_code=True)
        teacher_tokenizer = self.tokenizer

        prefix_msgs = list(validated_messages[:-1])
        
        # Fix tool_calls format for Qwen tokenizer compatibility
        # Qwen expects arguments as dict, not JSON string
        def _fix_message_for_template(msg):
            """Convert message to format compatible with Qwen chat template."""
            fixed = dict(msg)  # Shallow copy
            
            if "tool_calls" in fixed and fixed["tool_calls"]:
                fixed_tool_calls = []
                for tc in fixed["tool_calls"]:
                    if not isinstance(tc, dict):
                        continue
                    fixed_tc = dict(tc)
                    if "function" in fixed_tc and isinstance(fixed_tc["function"], dict):
                        fixed_func = dict(fixed_tc["function"])
                        # Convert arguments from JSON string to dict if needed
                        if "arguments" in fixed_func and isinstance(fixed_func["arguments"], str):
                            try:
                                fixed_func["arguments"] = json.loads(fixed_func["arguments"])
                            except (json.JSONDecodeError, TypeError):
                                pass  # Keep as string if not valid JSON
                        fixed_tc["function"] = fixed_func
                    fixed_tool_calls.append(fixed_tc)
                fixed["tool_calls"] = fixed_tool_calls
            
            return fixed
        
        fixed_prefix_msgs = [_fix_message_for_template(msg) for msg in prefix_msgs]

        prefix_text = teacher_tokenizer.apply_chat_template(
            fixed_prefix_msgs,
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
                "kl_adjustments": [0.0] * len(student_token_ids),
                "kl_mask": [0.0] * len(student_token_ids),
                "teacher_token_ids": list(completion_tokens),
                "teacher_logprobs": list(completion_lp),
            }

        student_ids = list(student_token_ids)[start:end]
        student_lp_slice = student_logprobs[start:end]
        student_mask = torch.tensor(
            list(reward_mask)[start:end],
            device=student_logprobs.device,
            dtype=student_logprobs.dtype,
        )

        teacher_lp_tensor = torch.tensor(
            completion_lp,
            device=student_logprobs.device,
            dtype=student_logprobs.dtype,
        )

        kl_slice, kl_mask_slice = _compute_groupwise_reverse_kl(
            student_tokenizer,
            student_ids,
            student_lp_slice,
            teacher_tokenizer,
            completion_tokens,
            teacher_lp_tensor,
            student_mask,
        )

        # Log KL and associated tensors for inspection and debugging using pickle
        kl_adjustments = torch.zeros_like(student_logprobs)
        kl_mask = torch.zeros_like(student_logprobs)
        kl_adjustments[start:end] = kl_slice
        try:
            kl_mask[start:end] = kl_mask_slice
        except Exception as e:
            import pickle

            debug_save = {
                "kl_mask": kl_mask,
                "kl_mask_slice": kl_mask_slice,
                "start": start,
                "end": end,
                "student_logprobs": student_logprobs,
                "student_ids": list(student_ids),
                "student_lp_slice": student_lp_slice,
                "completion_tokens": list(completion_tokens),
                "teacher_lp_tensor": teacher_lp_tensor,
                "student_mask": student_mask,
            }

            with open("debug_fireworks_teacher_alignment.pkl", "wb") as f:
                pickle.dump(debug_save, f)
            raise e


        return {
            "kl_adjustments": kl_adjustments.tolist(),
            "kl_mask": kl_mask.tolist(),
            "teacher_token_ids": list(completion_tokens),
            "teacher_logprobs": list(completion_lp),
        }
