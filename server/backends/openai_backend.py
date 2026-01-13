from __future__ import annotations

from typing import Dict, List, Any, Optional
import os
import logging
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from spider.config import ModelConfig

logger = logging.getLogger(__name__)

class OpenAIBackend:
    def __init__(self, config: ModelConfig):
        if not config.name:
            raise ValueError("`model.name` is required for OpenAI backend.")
        self._config = config

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("`OPENAI_API_KEY` is not set.")

        self._client = OpenAI(api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = dict(parameters or {})
        payload["model"] = self._config.name
        payload["input"] = _to_response_input(messages)
        if tools:
            payload["tools"] = _normalize_tools(tools)

        logger.info(
            "OpenAI responses called with %d message(s), tools=%s",
            len(messages),
            bool(tools),
        )

        response = self._client.chat.responses.create(**payload)
        content = response.output_text
        tool_calls = _response_tool_calls(response)

        return {
            "content": content or "",
            "reasoning": None,
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

def _normalize_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for tool in tools:
        tool = _as_dict(tool)
        function = _as_dict(tool.get("function") or {})
        normalized.append(
            {
                "type": "function",
                "name": function.get("name") or "",
                "description": function.get("description") or "",
                "parameters": function.get("parameters") or {},
                "strict": function.get("strict"),
            }
        )
    return normalized

def _to_response_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for msg in messages:
        msg = _as_dict(msg)
        role = msg["role"]
        if role == "tool":
            call_id = msg.get("tool_call_id")
            output = msg.get("content") or ""
            items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": str(output), 
            })
            continue
        if role:
            items.append({
                "role": role,
                "content": msg.get("content") or "",
            })
        for call in msg.get("tool_calls") or []:
            call = _as_dict(call)
            function = _as_dict(call.get("function") or {})
            items.append(
                {
                    "type": "function_call",
                    "call_id": call.get("id"),
                    "name": function.get("name"),
                    "arguments": function.get("arguments") or {},
                }
            ) # TODO: do we have deterministic guarantee for the type of each field so that we don't have to use fallbacks?

    return items

def _response_tool_calls(response: Any) -> Optional[List[Dict[str, Any]]]:
    output = response.output or []
    calls = []
    for item in output:
        item = _as_dict(item)
        if item.get("type") != "function_call":
            continue
        calls.append({
            "id": item.get("call_id"),
            "type": "function",
            "function": {
                "name": item.get("name") or "",
                "arguments": item.get("arguments") or "",
            }
        })
    return calls or None

def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)