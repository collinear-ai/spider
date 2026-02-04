from __future__ import annotations

from typing import Dict, List, Any, Optional
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from openai import OpenAI

from spider.config import ModelConfig

logger = logging.getLogger(__name__)


class OpenRouterBackend:
    def __init__(self, config: ModelConfig):
        if not config.name:
            raise ValueError("`model.name` is required for OpenRouter backend.")
        self._config = config

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("`OPENROUTER_API_KEY` is not set.")

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = dict(parameters or {})
        payload["model"] = self._config.name

        if tools:
            payload["tools"] = [_as_dict(tool) for tool in tools]

        logger.info(
            "OpenRouter chat called with %d message(s)",
            len(messages),
        )

        response = self._client.chat.completions.create(
            messages=messages,
            **payload,
        )
        content = _extract_content(response)
        reasoning = _extract_reasoning(response)
        tool_calls = _extract_tool_calls(response)

        return {
            "content": content or "",
            "reasoning": reasoning or None,
            "tool_calls": tool_calls,
        }

    def chat_batch(
        self,
        prompts: List[str],
        *,
        parameters: Dict[str, Any],
        system_prompts: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not prompts:
            return []

        total = len(prompts)
        progress = {"done": 0}
        log_every = max(1, total // 20)
        last_log = {"ts": 0.0}
        progress_lock = threading.Lock()

        def run_one(args: Any) -> tuple[int, Optional[Dict[str, Any]]]:
            idx, prompt = args
            messages = []
            if system_prompts and system_prompts[idx]:
                messages.append({"role": "system", "content": system_prompts[idx]})
            messages.append({"role": "user", "content": prompt})
            try:
                return idx, self.chat(messages=messages, parameters=parameters)
            except Exception as exc:
                logger.warning(
                    "OpenRouter chat_batch skipped prompt due to error: %s",
                    exc,
                )
                return idx, None
            finally:
                with progress_lock:
                    progress["done"] += 1
                    done = progress["done"]
                    now = time.monotonic()
                    if done == total or done % log_every == 0 or now - last_log["ts"] >= 5.0:
                        bar = _render_progress_bar(done, total)
                        logger.info("OpenRouter progress %s %d/%d", bar, done, total)
                        last_log["ts"] = now

        results = [None] * len(prompts)
        max_workers = min(len(prompts), 8)
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for idx, resp in pool.map(run_one, enumerate(prompts)):
                    results[idx] = resp
        finally:
            pass

        return results

    def metrics(self) -> Dict[str, object]:
        return {}

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        pass


def _extract_content(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    choice = choices[0]
    message = getattr(choice, "message", None)
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", None) or ""

def _extract_reasoning(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    choice = choices[0]
    message = getattr(choice, "message", None)
    if isinstance(message, dict):
        return message.get("reasoning") or message.get("reasoning_content") or ""
    return (
        getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or ""
    )

def _extract_tool_calls(response: Any) -> Optional[List[Dict[str, Any]]]:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    choice = choices[0]
    message = getattr(choice, "message", None)
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls") or []
    else:
        tool_calls = getattr(message, "tool_calls", None) or []

    normalized = []
    for call in tool_calls:
        call = _as_dict(call)
        function = _as_dict(call.get("function") or {})
        normalized.append(
            {
                "id": call.get("id"),
                "type": call.get("type") or "function",
                "function": {
                    "name": function.get("name") or "",
                    "arguments": function.get("arguments") or "",
                },
            }
        )
    return normalized or None

def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)

def _render_progress_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[{}]".format("-" * width)
    ratio = min(1.0, max(0.0, done / total))
    filled = int(ratio * width)
    return "[{}{}] {:3d}%".format("#" * filled, "-" * (width - filled), int(ratio * 100))
