from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import logging

import torch
import tinker
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.distillation.train_on_policy import _compute_groupwise_reverse_kl

logger = logging.getLogger(__name__)

def render_chat_tokens(
    *,
    messages: Sequence[Dict[str, object]],
    tools: Sequence[Dict[str, object]] | None,
    model_name: str,
    add_generation_prompt: bool = False,
) -> Tuple[List[int], int]:
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

async def compute_teacher_logprobs(
    *,
    sampling_client: tinker.SamplingClient,
    messages: Sequence[Dict[str, object]],
    tools: Sequence[Dict[str, object]] | None,
    teacher_model: str,
) -> Tuple[List[int], int, List[float]]:
    token_ids, prompt_len = render_chat_tokens(
        messages=messages,
        tools=tools,
        model_name=teacher_model,
        add_generation_prompt=False,
    )
    model_input = tinker.ModelInput.from_ints(list(token_ids))
    lp_resp = await sampling_client.compute_logprobs_async(model_input)
    teacher_logprobs = list(lp_resp)
    return token_ids, prompt_len, teacher_logprobs

async def compute_teacher_alignment_for_rewards( 
    *,
    sampling_client: tinker.SamplingClient,
    messages: Sequence[Dict[str, object]],
    tools: Sequence[Dict[str, object]] | None,
    teacher_model: str,
    student_model: str,
    student_token_ids: Sequence[int],
    student_logprobs: torch.Tensor,
    reward_mask: Sequence[int],
    assistant_raw_text: str,
) -> Dict[str, object]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Messages must end with an assistant turn.")
    
    student_tokenizer = get_tokenizer(student_model)
    teacher_tokenizer = get_tokenizer(teacher_model)

    prefix_msgs = messages[:-1]
    full_msgs = messages

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

    model_input = tinker.ModelInput.from_ints(list(full_tokens))
    lp_resp = await sampling_client.compute_logprobs_async(model_input)
    lp_list = list(lp_resp)
    completion_lp = lp_list[len(prefix_tokens):] # only completion tokens for teacher

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
        "kl_mask": kl_mask.tolist(), # full token length
        "teacher_token_ids": list(completion_tokens),
        "teacher_logprobs": list(completion_lp), # completion token length
    }