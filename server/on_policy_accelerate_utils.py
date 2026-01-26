from __future__ import annotations

import json
import logging
import os
import requests
import threading
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

from tinker_cookbook.distillation.train_on_policy import _compute_groupwise_reverse_kl

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
    
    # Clear logits and log_probs immediately after gathering (they're large tensors)
    # These can be huge: [batch, seq_len, vocab_size] where vocab_size ~50k+
    del logits, log_probs

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
        base_url: str = "http://10.234.201.144:8000/v1",
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
        self.api_key = ""

        logger.info(
            "Using Fireworks API for teacher model: %s (fireworks: %s)",
            model_name,
            fireworks_model,
        )

        # Load tokenizer locally (small memory footprint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Cache for student tokenizers (avoid reloading for every call)
        self._student_tokenizer_cache: Dict[str, Any] = {}
        self._tokenizer_cache_lock = threading.Lock()
        
        # Thread pool for parallel API calls
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=64)
        
        # HTTP session with connection pooling for Fireworks API
        import requests
        self._http_session = requests.Session()
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=64,
            pool_maxsize=64,
            max_retries=3,
        )
        self._http_session.mount('https://', adapter)
        self._http_session.mount('http://', adapter)
    
    def _get_student_tokenizer(self, student_model: str):
        """Get cached student tokenizer, loading if necessary."""
        # Fast path: check without lock first
        if student_model in self._student_tokenizer_cache:
            return self._student_tokenizer_cache[student_model]
        
        # Slow path: acquire lock and load
        with self._tokenizer_cache_lock:
            # Double-check after acquiring lock
            if student_model not in self._student_tokenizer_cache:
                logger.info("Loading student tokenizer for: %s", student_model)
                self._student_tokenizer_cache[student_model] = AutoTokenizer.from_pretrained(
                    student_model, use_fast=True, trust_remote_code=True
                )
            return self._student_tokenizer_cache[student_model]

    def compute_logprobs(self, text: str) -> Tuple[List[int], List[float]]:
        """Compute logprobs for text using Fireworks completions API.

        Args:
            text: Full text to compute logprobs for

        Returns:
            Tuple of (token_ids, logprobs)
        """

        api_url = f"{self.base_url}/completions"
        payload = {
            "model": self.fireworks_model,
            "max_tokens": 1,
            "echo": True,
            "logprobs": 1,  # vLLM uses integer for logprobs count
            "prompt": text,
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Use session with connection pooling for faster requests
        response = self._http_session.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        # Extract logprobs from response - handle both vLLM and Fireworks formats
        logprobs_data = response_json["choices"][0]["logprobs"]
        
        # vLLM format: {"tokens": [...], "token_logprobs": [...], ...}
        # Fireworks format: {"content": [{"token": ..., "logprob": ...}, ...]}
        if "token_logprobs" in logprobs_data:
            # vLLM format
            lp_list = logprobs_data["token_logprobs"]
            # First token logprob is None in vLLM, replace with 0.0
            lp_list = [lp if lp is not None else 0.0 for lp in lp_list]
        elif "content" in logprobs_data:
            # Fireworks format
            lp_list = [item["logprob"] for item in logprobs_data["content"]]
        else:
            raise ValueError(f"Unknown logprobs format: {logprobs_data.keys()}")

        # Get token IDs using local tokenizer
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # For vLLM with echo=True, we get logprobs for all tokens including prompt
        # The lp_list already contains prompt tokens, just need to align with our token_ids
        # Use length of lp_list minus 1 (the generated token) as prompt_tokens count
        prompt_tokens = len(lp_list) - 1  # Exclude the 1 generated token
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
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Messages must end with an assistant turn.")

        # Use cached tokenizer instead of loading every time
        student_tokenizer = self._get_student_tokenizer(student_model)
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
        import time
        t0 = time.perf_counter()
        _, full_logprobs = self.compute_logprobs(full_text)
        t1 = time.perf_counter()

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
        t2 = time.perf_counter()
        
        logging.info(f"TIMING: fireworks_api={t1-t0:.3f}s alignment={t2-t1:.3f}s total={t2-t0:.3f}s tokens={len(completion_tokens)}")

        # Log KL and associated tensors for inspection and debugging using pickle
        kl_adjustments = torch.zeros_like(student_logprobs)
        kl_mask = torch.zeros_like(student_logprobs)
        kl_adjustments[start:end] = kl_slice
        kl_mask[start:end] = kl_mask_slice

        return {
            "kl_adjustments": kl_adjustments.tolist(),
            "kl_mask": kl_mask.tolist(),
            "teacher_token_ids": list(completion_tokens),
            "teacher_logprobs": list(completion_lp),
        }

    def compute_teacher_alignment_full_trajectory(
        self,
        full_messages: Sequence[Dict[str, object]],
        combined_turns: Sequence[Dict[str, object]],
        tools: Sequence[Dict[str, object]] | None,
        student_model: str,
        student_token_ids: Sequence[int],
        student_logprobs: torch.Tensor,
        reward_mask: Sequence[int],
    ) -> Dict[str, object]:
        """Compute teacher alignment for a full multi-turn trajectory with a single API call.
        
        Instead of making one API call per turn, this makes a single call for the entire
        trajectory and extracts logprobs for each assistant turn using token positions.
        
        Args:
            full_messages: Complete conversation history
            combined_turns: List of turn info dicts with completion_start, completion_end, assistant_raw_text
            tools: Tool definitions
            student_model: Student model name for tokenizer
            student_token_ids: Full token sequence
            student_logprobs: Student logprobs for full sequence
            reward_mask: Full reward mask
            
        Returns:
            Dict with kl_adjustments, kl_mask, teacher_token_ids, teacher_logprobs
        """
        import time
        t0 = time.perf_counter()
        
        student_tokenizer = self._get_student_tokenizer(student_model)
        teacher_tokenizer = self.tokenizer
        
        # Normalize tool calls for Qwen3
        normalized_messages = _normalize_tool_calls_args_for_qwen3(list(full_messages))
        
        # Build full text from all messages
        full_text = teacher_tokenizer.apply_chat_template(
            normalized_messages,
            tools=tools if tools else None,
            add_generation_prompt=False,
            tokenize=False,
        )
        
        # Get logprobs for entire trajectory in ONE API call
        t_api_start = time.perf_counter()
        full_token_ids_teacher, full_logprobs = self.compute_logprobs(full_text)
        t_api_end = time.perf_counter()
        
        # Now we need to find the token positions for each assistant turn's completion
        # We'll use the chat template to figure out where each turn starts/ends
        kl_adjustments = [0.0] * len(student_token_ids)
        kl_mask = [0.0] * len(student_token_ids)
        all_teacher_token_ids = []
        all_teacher_logprobs = []
        
        for turn_info in combined_turns:
            assistant_raw_text = turn_info.get("assistant_raw_text")
            if not assistant_raw_text:
                continue
            
            completion_start = turn_info["completion_start"]
            completion_end = turn_info["completion_end"]
            turn_messages = turn_info["messages"]
            
            # Find where this turn's assistant text appears in the full teacher tokenization
            # Build prefix text (messages up to but not including assistant response)
            prefix_msgs = _normalize_tool_calls_args_for_qwen3(list(turn_messages[:-1]))
            prefix_text = teacher_tokenizer.apply_chat_template(
                prefix_msgs,
                tools=tools if tools else None,
                add_generation_prompt=True,
                tokenize=False,
            )
            prefix_tokens = teacher_tokenizer.encode(prefix_text, add_special_tokens=False)
            
            # Tokenize the assistant completion
            completion_tokens = teacher_tokenizer.encode(
                assistant_raw_text,
                add_special_tokens=False,
            )
            
            # Extract logprobs for this turn's completion from the full logprobs
            turn_start_idx = len(prefix_tokens)
            turn_end_idx = turn_start_idx + len(completion_tokens)
            
            completion_lp = full_logprobs[turn_start_idx:turn_end_idx]
            
            # Pad/truncate to match completion tokens
            if len(completion_lp) < len(completion_tokens):
                completion_lp = list(completion_lp) + [0.0] * (len(completion_tokens) - len(completion_lp))
            elif len(completion_lp) > len(completion_tokens):
                completion_lp = completion_lp[:len(completion_tokens)]
            
            # Create turn-specific reward mask
            turn_reward_mask = [0] * len(student_token_ids)
            for idx in range(completion_start, completion_end):
                if idx < len(turn_reward_mask):
                    turn_reward_mask[idx] = 1
            
            # Find reward region
            try:
                start = turn_reward_mask.index(1)
                end = len(turn_reward_mask) - list(reversed(turn_reward_mask)).index(1)
            except ValueError:
                continue
            
            student_ids = list(student_token_ids)[start:end]
            student_lp_slice = student_logprobs[start:end]
            student_mask = torch.tensor(
                turn_reward_mask[start:end],
                device=student_logprobs.device,
                dtype=student_logprobs.dtype,
            )
            
            teacher_lp_tensor = torch.tensor(
                completion_lp,
                device=student_logprobs.device,
                dtype=student_logprobs.dtype,
            )
            
            # Compute KL for this turn
            kl_slice, kl_mask_slice = _compute_groupwise_reverse_kl(
                student_tokenizer,
                student_ids,
                student_lp_slice,
                teacher_tokenizer,
                completion_tokens,
                teacher_lp_tensor,
                student_mask,
            )
            
            # Fill in the KL adjustments for this turn's region
            kl_adj_tensor = torch.zeros(len(student_token_ids), device=student_logprobs.device, dtype=student_logprobs.dtype)
            kl_mask_tensor = torch.zeros(len(student_token_ids), device=student_logprobs.device, dtype=student_logprobs.dtype)
            kl_adj_tensor[start:end] = kl_slice
            kl_mask_tensor[start:end] = kl_mask_slice
            
            for idx in range(completion_start, completion_end):
                if idx < len(kl_adjustments):
                    kl_adjustments[idx] = kl_adj_tensor[idx].item()
                    kl_mask[idx] = kl_mask_tensor[idx].item()
            
            all_teacher_token_ids.extend(completion_tokens)
            all_teacher_logprobs.extend(completion_lp)
        
        t2 = time.perf_counter()
        logging.info(
            f"TIMING (full trajectory): api={t_api_end-t_api_start:.3f}s "
            f"alignment={t2-t_api_end:.3f}s total={t2-t0:.3f}s "
            f"turns={len(combined_turns)} tokens={len(full_token_ids_teacher)}"
        )
        
        return {
            "kl_adjustments": kl_adjustments,
            "kl_mask": kl_mask,
            "teacher_token_ids": all_teacher_token_ids,
            "teacher_logprobs": all_teacher_logprobs,
        }
