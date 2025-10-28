from __future__ import annotations

from typing import Dict, Iterable, List

from vllm import LLM, SamplingParams

from spider.config import ModelConfig

class VLLMBackend:
    def __init__(self, config: ModelConfig):
        self._config = config
        llm_kwargs = {k: v for k, v in config.parameters.items() if k != "system_prompt"}
        self._llm = LLM(
            model=config.name, **llm_kwargs
        )
        self._last_metrics: Dict[str, object] = {}

    def generate(self, prompts: Iterable[str], *, parameters: Dict[str, object]) -> List[str]:
        tokenizer = self._llm.get_tokenizer()
        system_prompt = self._config.parameters.get("system_prompt")
        chat_prompts: List[str] = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            chat_prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        sampling = SamplingParams(**parameters)
        outputs = self._llm.generate(chat_prompts, sampling_params=sampling)
        completions: List[str] = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            completions.append(text)
        self._last_metrics = {
            "prompt_tokens": sum(len(getattr(o, "prompt_token_ids", []) or []) for o in outputs),
            "completion_tokens": sum(sum(len(getattr(chunk, "token_ids", []) or []) for chunk in getattr(o, "outputs", [])) for o in outputs),
        }
        return completions

    def metrics(self) -> Dict[str, object]:
        return dict(self._last_metrics)