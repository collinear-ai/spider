from __future__ import annotations

import argparse, asyncio, logging, time
from dataclasses import dataclass
from typing import Sequence

import tinker
from tinker import TensorData, types
import torch

from tinker_cookbook import model_info, renderers
from tinker_cookbook.distillation.train_on_policy import (
    _teacher_input_and_completion_start,
    _student_completion_to_teacher_tokens,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer

logger = logging.getLogger(__name__)

PROMPTS = [
    "Explain how rainbows form.",
    "Describe why deciduous leaves change color in the fall.",
    "How does a total solar eclipse occur?",
    "Explain how a sailboat can move against the wind.",
    "Why does bread dough rise when yeast is added?",
    "How do vaccines train the immune system?",
    "Explain how a refrigerator removes heat from food.",
    "Why does a magnet attract metal objects?",
]
SYSTEM_PROMPT = None
DEFAULT_MAX_STUDENT_TOKENS = 128
DEFAULT_STUDENT_TEMPERATURE = 0.7
DEFAULT_STUDENT_TOP_P = 0.9

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

async def _sample_student_completion(
    client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    prompt: str,
    system: str | None,
    sampling_params: types.SamplingParams,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    prompt_input = renderer.build_generation_prompt(messages)
    logger.info(
        "Sampling student completion (prompt len=%d tokens)",
        len(list(prompt_input.to_ints()))
    )

    t0 = time.perf_counter()
    response = await client.sample_async(
        prompt=prompt_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    logger.info(
        "Finished sampling in %.2fs (response tokens=%d)",
        time.perf_counter() - t0,
        len(response.sequences[0].tokens),
    )

    parsed, _ = renderer.parse_response(response.sequences[0].tokens)
    return parsed["content"]

async def _fetch_teacher_logprobs(
    client: tinker.SamplingClient,
    teacher_input: tinker.ModelInput,
) -> torch.Tensor:
    raw = await client.compute_logprobs_async(teacher_input)
    return torch.tensor(raw[1:], dtype=torch.float32) # skip bos

async def _run_probe(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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

    sampling_params = types.SamplingParams(
        max_tokens=DEFAULT_MAX_STUDENT_TOKENS,
        temperature=DEFAULT_STUDENT_TEMPERATURE,
        top_p=DEFAULT_STUDENT_TOP_P,
        stop=student_renderer.get_stop_sequences(),
    )

    service = tinker.ServiceClient()
    student_client = service.create_sampling_client(base_model=args.student_model)
    teacher_client = service.create_sampling_client(base_model=args.teacher_model)

    examples = []
    logger.info("Sampling student completions...")
    for prompt in PROMPTS:
        completion = await _sample_student_completion(
            student_client,
            student_renderer,
            prompt,
            SYSTEM_PROMPT,
            sampling_params,
        )
        examples.append(Example(prompt=prompt, completion=completion, system=SYSTEM_PROMPT))

    total_delta = 0.0
    total_tokens = 0
    for idx, example in enumerate(examples):
        logger.info("==== Prompt %d ====", idx)

        student_prompt_input = student_renderer.build_generation_prompt(
            example.convo_messages(),
        )
        student_prompt_tokens = list(student_prompt_input.to_ints())
        student_prompt_text = student_tokenizer.decode(
            student_prompt_tokens,
            skip_special_tokens=False,
        )
        student_completion_tokens = student_tokenizer.encode(
            example.completion,
            add_special_tokens=False,
        )
        student_sequence_tokens = student_prompt_tokens + student_completion_tokens
        student_sequence_text = student_tokenizer.decode(
            student_sequence_tokens,
            skip_special_tokens=False,
        )

        # get correct logprobs
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

        # get reused logprobs
        teacher_input_reused = tinker.ModelInput.from_ints(
            tokens=teacher_tokenizer.encode(student_sequence_text, add_special_tokens=False),
        )
        teacher_logprobs_reused = await _fetch_teacher_logprobs(
            teacher_client, teacher_input_reused,
        )
        completion_start_reused = len(
            teacher_tokenizer.encode(student_prompt_text, add_special_tokens=False),
        )

        teacher_completion_ids = _student_completion_to_teacher_tokens(
            example.completion, teacher_tokenizer,
        )

        # log text
        teacher_tokens_correct = list(teacher_input_correct.to_ints())
        teacher_tokens_reused = list(teacher_input_reused.to_ints())

        prompt_header_correct = teacher_tokenizer.decode(
            teacher_tokens_correct[:min(completion_start, 10)],
            skip_special_tokens=False,
        )
        prompt_header_reused = teacher_tokenizer.decode(
            teacher_tokens_reused[:min(completion_start_reused, 10)],
            skip_special_tokens=False,
        )
        logger.info("Teacher prompt header (correct): %r", prompt_header_correct)
        logger.info("Teacher prompt header (reused): %r", prompt_header_reused)
        
        completion_text_correct = teacher_tokenizer.decode(
            teacher_tokens_correct[
                completion_start: completion_start + len(teacher_completion_ids)
            ],
            skip_special_tokens=False,
        )
        completion_text_reused = teacher_tokenizer.decode(
            teacher_tokens_reused[
                completion_start_reused: completion_start_reused + len(teacher_completion_ids)
            ],
            skip_special_tokens=False,
        )
        logger.info("Teacher completion text (correct): %s ...[truncated]", completion_text_correct[:32])
        logger.info("Teacher completion text (reused): %s ...[truncated]", completion_text_reused[:32])

        # align logprobs
        trim_correct = teacher_logprobs_correct[
            completion_start: completion_start + len(teacher_completion_ids)
        ]
        trim_reused = teacher_logprobs_reused[
            completion_start_reused: completion_start_reused + len(teacher_completion_ids)
        ]
        
        # compute KL
        per_token_delta = trim_correct - trim_reused
        prompt_delta = float(per_token_delta.sum())
        avg_delta = float(per_token_delta.mean()) if per_token_delta.numel() else 0.0
        total_delta += prompt_delta
        total_tokens += per_token_delta.numel()

        logger.info(
            "Prompt %d: compared %d completion tokens, sum delta %.6f, avg delta %.6f",
            idx,
            per_token_delta.numel(),
            prompt_delta,
            avg_delta,
        )

    if total_tokens == 0:
        logger.warning("No completion tokens compared.")
        return

    logger.info(
        "Mean sum(log p_correct - log p_reused) per prompt = %.6f",
        total_delta / len(examples),
    )
    logger.info(
        "Mean per-token delta across prompts = %.6f",
        total_delta / total_tokens,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--teacher-model", default="meta-llama/Llama-3.3-70B-Instruct")
    args = parser.parse_args()

    asyncio.run(_run_probe(args))

if __name__ == "__main__":
    main()