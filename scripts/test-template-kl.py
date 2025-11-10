from __future__ import annotations

import argparse, asyncio, logging
from dataclasses import dataclass
from typing import Sequence

import tinker
from tinker import TensorData
import torch

from tinker_cookbook import model_info, renderers
from tinker_cookbook.distillation.train_on_policy import (
    _teacher_input_and_completion_start,
    _student_completion_to_teacher_tokens,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer

logger = logging.getLogger(__name__)

@dataclass
class Example:
    prompt: str
    completion: str
    system: str | None = None

    def convo_messages(self) -> list[renderers.Message]:
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.prompt})
        return msgs

def _build_student_datum(
    renderer: renderers.Renderer,
    tokenizer: Tokenizer,
    example: Example,
) -> tuple[tinker.Datum, int]:
    prompt_input = renderer.build_generation_prompt(example.convo_messages())
    prompt_tokens = list(prompt_input.to_ints())
    completion_tokens = tokenizer.encode(example.completion, add_special_tokens=False)
    if not completion_tokens:
        raise ValueError(f"Completion text must produce at least one token.")

    sequence = prompt_tokens + completion_tokens
    model_input = tinker.ModelInput.from_ints(tokens=sequence)
    targets = torch.tensor(sequence[1:], dtype=torch.long)
    mask = torch.zeros_like(targets, dtype=torch.float32)
    mask[-len(completion_tokens):] = 1.0

    datum = tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(targets),
            "logprobs": TensorData.from_torch(torch.zeros_like(targets, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.zeros_like(targets, dtype=torch.float32)),
            "mask": TensorData.from_torch(mask),
        },
    )
    return datum, len(prompt_tokens)

async def _fetch_teacher_logprobs(
    client: tinker.SamplingClient,
    teacher_input: tinker.ModelInput,
) -> torch.Tensor:
    raw = await client.compute_logprobs_async(teacher_input)
    return torch.tensor(raw[1:], dtype=torch.float32) # skip bos

async def _run_probe(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    example = Example(
        prompt=args.prompt,
        completion=args.completion,
        system=args.system_prompt,
    )
    student_tokenizer = get_tokenizer(args.student_model)
    student_renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(args.student_model),
        tokenizer=student_tokenizer,
    )
    teacher_tokenizer = get_tokenizer(args.teacher_model)
    teacher_renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(args.teacher_model),
        tokenizer=teacher_tokenizer,
    )

    datum, student_prompt_len = _build_student_datum(
        student_renderer, student_tokenizer, example
    )

    service = tinker.ServiceClient()
    teacher_client = service.create_sampling_client(
        base_model=args.teacher_model,
    )

    teacher_input_correct, completion_start = _teacher_input_and_completion_start(
        raw_prompt_text=example.prompt,
        student_completion_text=example.completion,
        convo_prefix=None,
        teacher_renderer=teacher_renderer,
        teacher_tokenizer=teacher_tokenizer,
    )
    teacher_logprobs_correct = await _fetch_teacher_logprobs(
        teacher_client, teacher_input_correct,
    )

    student_sequence_text = student_renderer.tokenizer.decode(
        datum.model_input.to_ints(),
        skip_special_tokens=False,
    )
    teacher_input_reused = tinker.ModelInput.from_ints(
        tokens=teacher_tokenizer.encode(student_sequence_text, add_special_tokens=False),
    )
    teacher_logprobs_reused = await _fetch_teacher_logprobs(
        teacher_client, teacher_input_reused,
    )
    completion_start_reused = student_prompt_len

    teacher_completion_ids = _student_completion_to_teacher_tokens(
        example.completion, teacher_tokenizer,
    )
    trim_correct = teacher_logprobs_correct[
        completion_start: completion_start + len(teacher_completion_ids)
    ]
    trim_reused = teacher_logprobs_reused[
        completion_start_reused: completion_start_reused + len(teacher_completion_ids)
    ]
    
    per_token_delta = trim_correct - trim_reused
    total_kl = float(per_token_delta.sum())
    avg_kl = float(per_token_delta.mean())

    logger.info("Compared %d completion tokens", trim_correct.shape[0])
    logger.info("Sum(log p_correct - log p_reused) = %.6f", total_kl)
    logger.info("Avg per-token delta = %.6f", avg_kl)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--prompt", default="Explains how rainbows form.")
    parser.add_argument("--completion", default="Rainbows form when sunlight refracts and reflects through water droplets.")
    parser.add_argument("--system-prompt", default=None)
    args = parser.parse_args()

    asyncio.run(_run_probe(args))

if __name__ == "__main__":
    main()