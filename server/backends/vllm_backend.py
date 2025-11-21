from __future__ import annotations

from typing import Dict, Iterable, List, Any, Optional
import threading, logging

from vllm import LLM, SamplingParams

from spider.config import ModelConfig

logger = logging.getLogger(__name__)

class VLLMBackend:
    def __init__(self, config: ModelConfig):
        self._config = config
        llm_kwargs = {k: v for k, v in config.parameters.items() if k != "system_prompt"}
        self._llm = LLM(
            model=config.name, **llm_kwargs
        )
        self._last_metrics: Dict[str, object] = {}
        self._metrics_lock = threading.Lock()

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
        metrics = {
            "prompt_tokens": sum(
                len(getattr(o, "prompt_token_ids", []) or []) 
                for o in outputs
            ),
            "completion_tokens": sum(
                sum(len(getattr(chunk, "token_ids", []) or []) 
                for chunk in getattr(o, "outputs", [])) 
                for o in outputs
            ),
        }
        with self._metrics_lock:
            self._last_metrics = metrics
        return completions

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        sampling = SamplingParams(**parameters)
        logger.info(
            "vLLM chat caled with %d messages(s), tools=%s",
            len(messages),
            bool(tools),
        )
        outputs = self._llm.chat(
            [messages], 
            sampling_params=sampling,
            tools=tools,
        )
        logger.info(
            "vLLM chat return %d output batch(es)",
            len(outputs) if outputs is not None else 0
        )
        response = {}
        if not outputs:
            response = {"content": "", "tool_calls": None}
        else:
            first = outputs[0]
            choice = first.outputs[0] if first.outputs else None
            content = ""
            tool_calls = None
            if choice is not None:
                message = getattr(choice, "message", None)
                content = message.get("content") or ""
                tool_calls = message.get("tool_calls")
            response = {"content": content, "tool_calls": tool_calls}

        return response

    def metrics(self) -> Dict[str, object]:
        with self._metrics_lock:
            return dict(self._last_metrics)

    def close(self) -> None:
        engine = getattr(self._llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()
        self._llm = None