from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from accelerate import Accelerator

from tinker_cookbook.tokenizer_utils import get_tokenizer
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
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PeftModel, PreTrainedTokenizer]:
    """Load base model and apply LoRA (or load existing adapter).

    Args:
        model_name: HuggingFace model name or path
        lora_rank: LoRA rank for adapter
        checkpoint_path: Optional path to existing LoRA adapter
        device_map: Device mapping strategy
        torch_dtype: Torch dtype for model weights

    Returns:
        Tuple of (PeftModel, Tokenizer)
    """
    logger.info(
        "Loading model %s with LoRA rank=%d, checkpoint=%s",
        model_name,
        lora_rank,
        checkpoint_path,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info("Loading existing LoRA adapter from %s", checkpoint_path)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
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
        model.print_trainable_parameters()

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
    tokenizer = get_tokenizer(model_name)
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


def compute_teacher_alignment_for_rewards_direct(
    *,
    model: torch.nn.Module,
    messages: Sequence[Dict[str, object]],
    tools: Sequence[Dict[str, object]] | None,
    teacher_model: str,
    student_model: str,
    student_token_ids: Sequence[int],
    student_logprobs: torch.Tensor,
    reward_mask: Sequence[int],
    assistant_raw_text: str,
    device: torch.device,
) -> Dict[str, object]:
    """Compute teacher alignment for rewards using direct forward pass.

    This is the accelerate version of compute_teacher_alignment_for_rewards
    from on_policy_utils.py. It uses direct model forward pass instead of
    tinker.SamplingClient.

    Args:
        model: Teacher model
        messages: Chat messages (must end with assistant turn)
        tools: Tool definitions
        teacher_model: Teacher model name
        student_model: Student model name
        student_token_ids: Student's token IDs
        student_logprobs: Student's logprobs
        reward_mask: Mask for reward tokens
        assistant_raw_text: Raw text of assistant response
        device: Device to run on

    Returns:
        Dict with kl_adjustments, kl_mask, teacher_token_ids, teacher_logprobs
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Messages must end with an assistant turn.")

    student_tokenizer = get_tokenizer(student_model)
    teacher_tokenizer = get_tokenizer(teacher_model)

    prefix_msgs = messages[:-1]

    prefix_tokens, _ = render_chat_tokens(
        messages=prefix_msgs,
        tools=tools,
        model_name=teacher_model,
        add_generation_prompt=True,
    )

    if assistant_raw_text is None:
        raise ValueError("Assistant raw text is required for teacher alignment")

    completion_tokens = teacher_tokenizer.encode(
        assistant_raw_text,
        add_special_tokens=False,
    )
    full_tokens = list(prefix_tokens) + list(completion_tokens)

    # Compute logprobs via direct forward pass
    logprobs = compute_logprobs_for_sequence(model, full_tokens, device)
    completion_lp = logprobs[len(prefix_tokens) :]

    # Find reward region
    start = reward_mask.index(1)
    end = len(reward_mask) - list(reversed(reward_mask)).index(1)

    student_ids = list(student_token_ids)[start:end]
    student_lp_slice = student_logprobs[start:end]
    student_mask = torch.tensor(
        reward_mask[start:end],
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


class AccelerateTeacherContext:
    """Context for computing teacher logprobs using accelerate/transformers."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_name = model_name
        logger.info("Loading teacher model: %s", model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._device = next(self.model.parameters()).device

    def compute_logprobs(self, token_ids: List[int]) -> List[float]:
        """Compute logprobs for a sequence of tokens."""
        return compute_logprobs_for_sequence(self.model, token_ids, self._device)

    async def compute_teacher_alignment(
        self,
        messages: Sequence[Dict[str, object]],
        tools: Sequence[Dict[str, object]] | None,
        student_model: str,
        student_token_ids: Sequence[int],
        student_logprobs: torch.Tensor,
        reward_mask: Sequence[int],
        assistant_raw_text: str,
    ) -> Dict[str, object]:
        """Compute teacher alignment for rewards (async interface for compatibility)."""
        return compute_teacher_alignment_for_rewards_direct(
            model=self.model,
            messages=messages,
            tools=tools,
            teacher_model=self.model_name,
            student_model=student_model,
            student_token_ids=student_token_ids,
            student_logprobs=student_logprobs,
            reward_mask=reward_mask,
            assistant_raw_text=assistant_raw_text,
            device=self._device,
        )


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
        import httpx

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

        # Create async HTTP client
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False

    async def compute_logprobs_async(self, text: str) -> Tuple[List[int], List[float]]:
        """Compute logprobs for text using Fireworks completions API.

        Args:
            text: Full text to compute logprobs for

        Returns:
            Tuple of (token_ids, logprobs)
        """
        # Use completions API with echo=True to get logprobs for input tokens
        payload = {
            "model": self.fireworks_model,
            "prompt": text,
            "max_tokens": 1,
            "echo": True,
            "logprobs": 1,
        }

        response = await self._client.post("/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs", {})

        token_logprobs = logprobs_data.get("token_logprobs", [])

        # First token has no logprob
        if token_logprobs and token_logprobs[0] is None:
            token_logprobs[0] = 0.0

        # Get token IDs using local tokenizer
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Align lengths
        if len(token_logprobs) != len(token_ids):
            logger.warning(
                "Token count mismatch: API returned %d, tokenizer has %d",
                len(token_logprobs),
                len(token_ids),
            )
            if len(token_logprobs) < len(token_ids):
                token_logprobs.extend([0.0] * (len(token_ids) - len(token_logprobs)))
            else:
                token_logprobs = token_logprobs[: len(token_ids)]

        return token_ids, [lp if lp is not None else 0.0 for lp in token_logprobs]

    async def compute_teacher_alignment(
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

        student_tokenizer = get_tokenizer(student_model)
        teacher_tokenizer = self.tokenizer

        prefix_msgs = list(messages[:-1])

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

        # Get logprobs from Fireworks
        _, full_logprobs = await self.compute_logprobs_async(full_text)

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

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
