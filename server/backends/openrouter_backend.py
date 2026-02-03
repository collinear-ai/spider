from __future__ import annotations

from typing import Dict, List, Any, Optional
import os
import logging
from concurrent.futures import ThreadPoolExecutor

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
            logger.info(
                "OpenRouter backend ignoring %d tool(s) (tool calls disabled).",
                len(tools),
            )

        logger.info(
            "OpenRouter chat called with %d message(s)",
            len(messages),
        )

        response = self._client.chat.completions.create(
            messages=messages,
            **payload,
        )
        content = _extract_content(response)

        return {
            "content": content or "",
            "reasoning": None,
            "tool_calls": None,
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

        def run_one(args: Any) -> tuple[int, Dict[str, Any]]:
            idx, prompt = args
            messages = []
            if system_prompts and system_prompts[idx]:
                messages.append({"role": "system", "content": system_prompts[idx]})
            messages.append({"role": "user", "content": prompt})
            return idx, self.chat(messages=messages, parameters=parameters)

        results = [None] * len(prompts)
        max_workers = min(len(prompts), 16)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for idx, resp in pool.map(run_one, enumerate(prompts)):
                results[idx] = resp

        return [r or {} for r in results]

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
