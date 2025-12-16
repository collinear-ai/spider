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

def reward_spans_from_mask(mask: Sequence[int]) -> List[Tuple[int, int]]:
    spans = []
    start = None
    for idx, flag in enumerate(mask): # flag is 1
        if flag and start is None:
            start = idx
        if not flag and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans

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
) -> Dict[str, object]:
    spans = reward_spans_from_mask(reward_mask)
    assistant_positions = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if len(spans) != len(assistant_positions):
        raise ValueError(f"assistant_turns={len(assistant_positions)} reward_spans={len(spans)} mismatch")

    student_tokenizer = get_tokenizer(student_model)
    teacher_tokenizer = get_tokenizer(teacher_model)

    kl_adjustments = torch.zeros_like(student_logprobs)
    kl_mask = torch.zeros_like(student_logprobs)
    all_teacher_token_ids = []
    all_teacher_logprobs = []

    for turn_idx, span in enumerate(spans):
        assistant_idx = assistant_positions[turn_idx]
        prefix_msgs = messages[:assistant_idx]
        full_msgs = messages[:assistant_idx + 1]

        prefix_tokens, _ = render_chat_tokens(
            messages=prefix_msgs,
            tools=tools,
            model_name=teacher_model,
            add_generation_prompt=True,
        )
        full_tokens, _ = render_chat_tokens(
            messages=full_msgs,
            tools=tools,
            model_name=teacher_model,
            add_generation_prompt=False,
        )
        if len(full_tokens) < len(prefix_tokens):
            continue
        completion_tokens = full_tokens[len(prefix_tokens):]

        def _preview(tokens, tokenizer, start=True, window=16):
            if start:
                slice_tokens = tokens[:min(len(tokens), window)]
            else:
                slice_tokens = tokens[max(0, len(tokens) - window):]
            return tokenizer.decode(
                slice_tokens,
                skip_special_tokens=False,
            ).replace("\n", "\\n")

        logger.info(
            "Teacher alignment turn=%d prefix_end='%s' completion_start='%s'",
            turn_idx,
            _preview(prefix_tokens, teacher_tokenizer, start=False),
            _preview(completion_tokens, teacher_tokenizer, start=True),
        )

        model_input = tinker.ModelInput.from_ints(list(full_tokens))
        lp_resp = await sampling_client.compute_logprobs_async(model_input)
        lp_list = list(lp_resp)
        completion_lp = lp_list[len(prefix_tokens):] # only completion tokens for teacher

        start, end = span
        student_ids_slice = list(student_token_ids[start:end]) # only completion tokens for student
        student_lp_slice = student_logprobs[start:end]
        student_mask_slice = torch.tensor(
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
            student_ids_slice,
            student_lp_slice,
            teacher_tokenizer,
            completion_tokens,
            teacher_lp_tensor,
            student_mask_slice,
        )

        kl_adjustments[start:end] = kl_slice
        kl_mask[start:end] = kl_mask_slice

        all_teacher_token_ids.extend(completion_tokens)
        all_teacher_logprobs.extend(completion_lp)
    
    return {
        "kl_adjustments": kl_adjustments.tolist(),
        "kl_mask": kl_mask.tolist(),
        "teacher_token_ids": list(all_teacher_token_ids),
        "teacher_logprobs": list(all_teacher_logprobs),
        "reward_spans": spans,
    }

async def compute_student_logprobs_trainable(
    *,
    training_client: tinker.TrainingClient,
    token_ids: Sequence[int],
    reward_mask: Sequence[int],
    loss_fn: str = "importance_sampling",
) -> Tuple[torch.Tensor, torch.Tensor]:
    model_input = tinker.ModelInput.from_ints(list(token_ids))
    target_tokens = tinker.TensorData.from_list(list(token_ids))
    mask = tinker.TensorData.from_list(list(reward_mask))

    datum = tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "mask": mask,
        }
    )
    fwd_bwd_future = await training_client.forward_backward_async(
        [datum],
        loss_fn=loss_fn,
    )
    fwd_bwd_result = await fwd_bwd_future.result_async()
    logprobs = fwd_bwd_result.loss_fn_outputs[0]["logprobs"].to_torch()

    torch_mask = torch.tensor(reward_mask, device=logprobs.device, dtype=logprobs.dtype)
    return logprobs, torch_mask