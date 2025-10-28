from __future__ import annotations

from typing import Dict, Iterable, List

from vllm import LLM, SamplingParams

from spider.config import ModelConfig

class VLLMBackend:
    def __init__(self, config: ModelConfig):
        self._config = config
        self._llm = LLM(
            model=config.name, **config.parameters
        )
        self._last_metrics: Dict[str, object] = {}

    def generate(self, prompts: Iterable[str], *, parameters: Dict[str, object]) -> List[str]:
        sampling = SamplingParams(**parameters)
        outputs = self._llm.generate(prompts, sampling_params=sampling)
        completions = List[str] = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            completions.append(text)
        self._last_metrics = {
            "prompt_tokens": sum(
                o.prompt_token_ids and len(o.prompt_token_ids) or 0 for o in outputs
            ),
            "completion_tokens": sum(
                sum(len(chunk.token_ids) for chunk in o.outputs) for o in outputs
            ),
        }
        return completions

    def metrics(self) -> Dict[str, object]:
        return dict(self._last_metrics)