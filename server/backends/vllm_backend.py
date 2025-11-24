from __future__ import annotations

from typing import Dict, Iterable, List, Any, Optional
import threading, logging

from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.tool_parsers import ToolParserManager

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
        self._tool_parser_cls = ToolParserManager.get_tool_parser(
            self._default_tool_parser(config.name or "")
        )

    @staticmethod
    def _default_tool_parser(model_name: str) -> Optional[str]:
        lower = (model_name or "").lower()
        if "llama" in lower:
            return "llama_tool_parser"
        if "qwen3" in lower:
            return "qwen3coder_tool_parser"
        if "deepseek" in lower:
            return "deepseek_v31" if "v3.1" in lower else "deepseek_v3"
        if "mistral" in lower:
            return "mistral_tool_parser"
        if "glm" in lower:
            return "glm45"
        return "openai"

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
            "vLLM chat called with %d messages(s), tools=%s",
            len(messages),
            bool(tools),
        )
        outputs = self._llm.chat(
            [messages], 
            sampling_params=sampling,
            tools=tools,
        )
        logger.info(
            "vLLM chat return %d output batch(es) type=%s first=%s",
            len(outputs) if outputs is not None else 0,
            type(outputs),
            type(outputs[0])
        )
        response = {}
        if not outputs:
            response = {"content": "", "tool_calls": None}
        else:
            first = outputs[0]
            if not first.outputs:
                raise RuntimeError("vLLM chat returned no candidate outputs.")

            completion = first.outputs[0]
            content = completion.text or ""
            tool_calls = None

            if self._tool_parser_cls and tools:
                parser = self._tool_parser_cls(self._llm.get_tokenizer())
                request = ChatCompletionRequest(
                    messages=messages,
                    model=self._config.name,
                    tools=tools,
                    tool_choice="auto",
                    temperature=parameters.get("temperature"),
                    top_p=parameters.get("top_p"),
                    top_k=parameters.get("top_k"),
                    max_tokens=parameters.get("max_tokens"),
                )
                request = parser.adjust_request(request)
                parsed = parser.extract_tool_calls(content, request=request)

                if parsed is not None and parsed.tools_called:
                    tool_calls = []
                    for idx, call in enumerate(parsed.tool_calls):
                        tool_calls.append(
                            {
                                "id": call.id or f"call_{idx}",
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments or "{}",
                                },
                            }
                        )
                    content = parsed.content or ""

            response = {"content": content, "tool_calls": tool_calls}

        logger.info(
            "vLLM chat returning content length %d tool_calls=%s",
            len(response.get("content") or ""),
            bool(response.get("tool_calls"))
        )
        return response

    def metrics(self) -> Dict[str, object]:
        with self._metrics_lock:
            return dict(self._last_metrics)

    def close(self) -> None:
        engine = getattr(self._llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()
        self._llm = None