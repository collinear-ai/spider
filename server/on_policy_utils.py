from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import logging
import os
import requests
import torch
import json
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer as Tokenizer

logger = logging.getLogger(__name__)

def render_chat_tokens(
    *,
    messages: Sequence[Dict[str, object]],
    tools: Sequence[Dict[str, object]] | None,
    model_name: str,
    add_generation_prompt: bool = False,
) -> Tuple[List[int], int]:
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

def _token_pieces(tokenizer: Tokenizer, token_ids: List[int]) -> List[str]:
    pieces = []
    prev = ""
    for idx in range(len(token_ids)):
        cur = tokenizer.decode(
            token_ids[: idx + 1],
            skip_special_tokens=False,
        )
        pieces.append(cur[len(prev):])
        prev = cur
    return pieces

def _compute_groupwise_reverse_kl(
    student_tokenizer: Tokenizer,
    student_token_ids: List[int],
    student_logprobs: torch.Tensor,
    teacher_tokenizer: Tokenizer,
    teacher_token_ids: List[int],
    teacher_logprobs: torch.Tensor,
    base_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_groups, teacher_groups = _build_alignment_groups(
        student_tokenizer,
        student_token_ids,
        teacher_tokenizer,
        teacher_token_ids,
    )
    logger.info(
        "GOLD alignment (post-trim): student_tokens=%d teacher_tokens=%d student_groups=%d teacher_groups=%d",
        len(student_token_ids),
        len(teacher_token_ids),
        len(student_groups),
        len(teacher_groups)
    )

    reverse_kl = torch.zeros_like(student_logprobs)
    mask = torch.zeros_like(base_mask)

    for i, (s_group, t_group) in enumerate(zip(student_groups, teacher_groups)):
        if i < 5:
            student_slice = student_tokenizer.decode(
                [student_token_ids[j] for j in s_group],
                skip_special_tokens=False,
            )
            teacher_slice = teacher_tokenizer.decode(
                [teacher_token_ids[j] for j in t_group],
                skip_special_tokens=False,
            )
            logger.info(
                "GOLD group %d: student_tokens=%d teacher_tokens=%d student_text='%s', teacher_text='%s'",
                i,
                len(s_group),
                len(t_group),
                student_slice,
                teacher_slice,
            )

        teacher_indices = [idx for idx in t_group if idx < len(teacher_logprobs)]
        student_indices = [idx for idx in s_group if idx < len(student_logprobs) and base_mask[idx] > 0]
        if not teacher_indices or not student_indices:
            continue

        student_log_sum = student_logprobs[student_indices[0]]
        for s_idx in student_indices[1:]:
            student_log_sum = student_log_sum + student_logprobs[s_idx]
            
        teacher_log_sum = teacher_logprobs[teacher_indices[0]]
        for t_idx in teacher_indices[1:]:
            teacher_log_sum = teacher_log_sum + teacher_logprobs[t_idx]

        delta = student_log_sum - teacher_log_sum
        share = delta / len(student_indices)

        for s_idx in student_indices:
            reverse_kl[s_idx] = share
            mask[s_idx] = base_mask[s_idx]

    return reverse_kl, mask

def _build_alignment_groups(
    student_tokenizer: Tokenizer,
    student_token_ids: List[int],
    teacher_tokenizer: Tokenizer,
    teacher_token_ids: List[int],
) -> tuple[List[List[int]], List[List[int]]]:
    student_pieces = _token_pieces(student_tokenizer, student_token_ids)
    teacher_pieces = _token_pieces(teacher_tokenizer, teacher_token_ids)

    student_groups = []
    teacher_groups = []

    i = j = 0
    s_buf = t_buf = ""
    cur_s = []
    cur_t = []

    def flush() -> None:
        nonlocal s_buf, t_buf, cur_s, cur_t
        if cur_s and cur_t:
            student_groups.append(cur_s.copy())
            teacher_groups.append(cur_t.copy())
        s_buf = t_buf = ""
        cur_s = []
        cur_t = []

    def check_match() -> bool:
        """Check if current groups match by comparing full decoded strings"""
        if not cur_s or not cur_t:
            return False
        import unicodedata
        s_full = unicodedata.normalize('NFC', 
            student_tokenizer.decode([student_token_ids[idx] for idx in cur_s], skip_special_tokens=False))
        t_full = unicodedata.normalize('NFC',
            teacher_tokenizer.decode([teacher_token_ids[idx] for idx in cur_t], skip_special_tokens=False))
        return s_full == t_full and s_full

    while i < len(student_pieces) or j < len(teacher_pieces):
        if check_match():
            flush()
            continue
        
        if i < len(student_pieces) and (s_buf == "" or len(s_buf) <= len(t_buf)):
            s_buf += student_pieces[i]
            cur_s.append(i)
            i += 1
            continue
        
        if j < len(teacher_pieces):
            t_buf += teacher_pieces[j]
            cur_t.append(j)
            j += 1
            continue
        
        break

    if check_match():
        flush()

    return student_groups, teacher_groups


async def compute_teacher_alignment_for_rewards( 
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
    
    student_tokenizer = AutoTokenizer.from_pretrained(student_model, use_fast=True, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model, use_fast=True, trust_remote_code=True)
    
    prefix_tokens, _ = render_chat_tokens(
        messages=messages[:-1],
        tools=tools,
        model_name=teacher_model,
        add_generation_prompt=True,
    )
    completion_tokens = teacher_tokenizer.encode(assistant_raw_text, add_special_tokens=False)
    full_text = teacher_tokenizer.decode(list(prefix_tokens) + list(completion_tokens))
    
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("Fireworks API key required (FIREWORKS_API_KEY env var or parameter)")
    
    fireworks_url = os.getenv("FIREWORKS_URL")
    if not fireworks_url:
        raise ValueError("Fireworks URL required (FIREWORKS_URL env var or parameter)")

    payload = {
        "model": fireworks_url,
        "max_tokens": 5120,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "echo": True,
        "logprobs": True,
        "temperature": 0.1,
        "prompt": full_text
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.request("POST", fireworks_url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    response_json = response.json()
    lp_list = [item["logprob"] for item in response_json["choices"][0]["logprobs"]["content"]]

    # Fireworks returns the prefix/completion tokens + 2 extra completion tokens.
    # So we will just consider the prompt tokens (prefix/completion).
    completion_lp = lp_list[:response_json['usage']['prompt_tokens']][len(prefix_tokens):]

    start, end = reward_mask.index(1), len(reward_mask) - list(reversed(reward_mask)).index(1)
    student_ids = list(student_token_ids)[start:end]
    student_lp_slice = student_logprobs[start:end]
    student_mask = torch.tensor(reward_mask[start:end], device=student_logprobs.device, dtype=student_logprobs.dtype)
    teacher_lp_tensor = torch.tensor(completion_lp, device=student_logprobs.device, dtype=student_logprobs.dtype)
    
    kl_slice, kl_mask_slice = _compute_groupwise_reverse_kl(
        student_tokenizer, student_ids, student_lp_slice,
        teacher_tokenizer, completion_tokens, teacher_lp_tensor, student_mask,
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