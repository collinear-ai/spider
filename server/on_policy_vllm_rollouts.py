"""Parallel rollout collection using vLLM HTTP server.

This module provides VLLMRolloutCollector for collecting rollouts in parallel
using ThreadPoolExecutor and vLLM's HTTP API with logprobs support.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx

from .vllm_parsers import parse_assistant_turn

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Result from a single rollout trajectory."""

    prompt: str
    messages: List[Dict[str, Any]]
    token_ids: List[int]
    logprobs: List[float]
    reward_mask: List[int]
    assistant_content: str
    assistant_reasoning_content: Optional[str]
    assistant_tool_calls: Optional[List[Dict[str, Any]]]
    assistant_raw_text: str
    prompt_token_count: int
    parser_fallback: bool
    turn_index: int
    retokenize_match: bool = True
    combined_turns: Optional[List[Dict[str, Any]]] = None


@dataclass
class VLLMRolloutCollector:
    """Parallel rollout collection using vLLM HTTP server.

    This collector uses ThreadPoolExecutor to run multiple prompts in parallel,
    each making HTTP calls to a vLLM server with logprobs enabled.
    """

    vllm_base_url: str
    model_name: str
    tools: List[Dict[str, Any]]
    tool_registry: Dict[str, Callable[..., Any]]
    tool_parser_name: Optional[str] = None
    reasoning_parser_name: Optional[str] = None
    max_workers: int = 8
    max_tool_turns: int = 16
    max_tokens: int = 4096
    temperature: float = 1.0
    lora_name: Optional[str] = None
    runtime_factory: Optional[Callable[[Dict[str, Any]], Any]] = None
    verbose: bool = False

    _client: httpx.Client = field(init=False, default=None)
    _executor: ThreadPoolExecutor = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.vllm_base_url,
            timeout=httpx.Timeout(480.0, connect=60.0),
        )
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Load tokenizer for parsing
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self._tokenizer = get_tokenizer(self.model_name)

    def collect_batch(self, prompts: List[Dict[str, Any]]) -> List[RolloutResult]:
        """Collect rollouts in parallel using ThreadPoolExecutor.

        Args:
            prompts: List of prompt dicts with 'prompt' and optional 'system_prompt'

        Returns:
            List of RolloutResult objects for each prompt
        """
        futures = [self._executor.submit(self._run_prompt, p) for p in prompts]
        results = []
        for future in futures:
            try:
                result = future.result()
                if result:
                    results.extend(result)
            except Exception as exc:
                logger.error("Rollout failed: %s", exc)
        return results

    def _run_prompt(self, row: Dict[str, Any]) -> List[RolloutResult]:
        """Run single prompt with multi-turn tool calling.

        Args:
            row: Dict with 'prompt' and optional 'system_prompt'

        Returns:
            List of RolloutResult objects for this trajectory
        """
        prompt = row["prompt"]
        system_prompt = row.get("system_prompt")

        history: List[Dict[str, Any]] = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": prompt})

        turn_items: List[RolloutResult] = []
        runtime = None

        if self.runtime_factory:
            runtime = self.runtime_factory(row)

        try:
            for turn_idx in range(self.max_tool_turns):
                try:
                    response = self._generate_with_logprobs(history)
                except Exception as exc:
                    if self._is_context_window_error(exc):
                        logger.warning(
                            "Prompt=`%s...` turn=%d exceeded context window; ending trajectory.",
                            prompt[:20],
                            turn_idx,
                        )
                        break
                    raise

                content = response["content"]
                reasoning = response.get("reasoning")
                tool_calls = response.get("tool_calls")
                token_ids = response["token_ids"]
                logprobs = response["logprobs"]
                raw_text = response["raw_text"]
                prompt_token_count = response["prompt_token_count"]

                # Build reward mask (1 for generated tokens, 0 for prompt)
                reward_mask = [0] * prompt_token_count + [1] * len(token_ids)

                # Full token sequence
                full_token_ids = response["full_token_ids"]

                if len(full_token_ids) <= 1:
                    logger.warning(
                        "Token sequence too short for training: tokens=%d",
                        len(full_token_ids),
                    )
                    break

                # Build assistant message snapshot
                snapshot = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
                if reasoning:
                    snapshot["reasoning_content"] = reasoning

                if self.verbose:
                    logger.info(
                        "prompt=`%s...` turn=%d tool_returned=%s",
                        prompt[:8],
                        turn_idx,
                        bool(tool_calls),
                    )

                history.append(snapshot)

                turn_items.append(
                    RolloutResult(
                        prompt=prompt,
                        messages=list(history),
                        token_ids=full_token_ids,
                        logprobs=[0.0] * prompt_token_count + logprobs,
                        reward_mask=reward_mask,
                        assistant_content=content,
                        assistant_reasoning_content=reasoning,
                        assistant_tool_calls=tool_calls,
                        assistant_raw_text=raw_text,
                        prompt_token_count=prompt_token_count,
                        parser_fallback=response.get("parser_fallback", False),
                        turn_index=turn_idx,
                    )
                )

                if not tool_calls:
                    break

                # Execute tool calls
                self._execute_tool_calls(tool_calls, history)

        finally:
            if runtime is not None:
                runtime.cleanup()

        # Combine all turns into single item for batch training
        if len(turn_items) == 0:
            return []

        if len(turn_items) == 1:
            return [turn_items[0]]

        return [self._combine_turns(turn_items)]

    def _generate_with_logprobs(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate completion with logprobs from vLLM.

        Args:
            messages: Chat messages

        Returns:
            Dict with content, tool_calls, token_ids, logprobs, raw_text
        """
        model_name = self.lora_name if self.lora_name else self.model_name

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "logprobs": True,
            "top_logprobs": 1,
        }

        if self.tools:
            payload["tools"] = self.tools

        response = self._client.post("/v1/chat/completions", json=payload)

        if response.status_code >= 400:
            body = (response.text or "").strip()
            raise RuntimeError(
                f"vLLM chat failed (status={response.status_code}): {body[:512]}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM chat returned no candidate outputs.")

        choice = choices[0]
        message = choice.get("message") or {}

        # Extract content and tool calls from message
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        tool_calls = message.get("tool_calls")

        # Extract logprobs
        logprobs_data = choice.get("logprobs") or {}
        logprobs_content = logprobs_data.get("content") or []

        token_ids = []
        logprobs = []
        for lp_entry in logprobs_content:
            token_logprob = lp_entry.get("logprob", 0.0)
            # Get token ID
            top_lps = lp_entry.get("top_logprobs") or []
            token = lp_entry.get("token", "")
            token_id = None
            for top_lp in top_lps:
                if top_lp.get("token") == token:
                    token_id = top_lp.get("token_id")
                    break
            if token_id is None:
                token_id = lp_entry.get("token_id", 0)

            token_ids.append(token_id)
            logprobs.append(token_logprob)

        # Get raw text by decoding token_ids
        raw_text = self._tokenizer.decode(token_ids, skip_special_tokens=False)

        # Parse the response using official vLLM parsers
        parsed = parse_assistant_turn(
            messages=messages,
            assistant_text=raw_text,
            tools=self.tools if self.tools else None,
            tool_parser_name=self.tool_parser_name,
            reasoning_parser_name=self.reasoning_parser_name,
            tokenizer=self._tokenizer,
            token_ids=token_ids,
        )

        # Use parsed content/tool_calls if parser succeeded
        if not parsed.parser_fallback:
            content = parsed.content
            tool_calls = parsed.tool_calls
            if parsed.reasoning:
                reasoning = parsed.reasoning

        # Get prompt token count from usage
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0)

        # Build full token sequence (prompt + generated)
        # We need to get the prompt tokens from the input
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tools=self.tools if self.tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_token_ids = self._tokenizer.encode(prompt_text)
        full_token_ids = list(prompt_token_ids) + token_ids

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "token_ids": token_ids,
            "logprobs": logprobs,
            "raw_text": raw_text,
            "prompt_token_count": len(prompt_token_ids),
            "full_token_ids": full_token_ids,
            "parser_fallback": parsed.parser_fallback,
        }

    def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
    ) -> None:
        """Execute tool calls and append results to history.

        Args:
            tool_calls: List of tool call objects
            history: Chat history to append results to
        """
        for tool_call in tool_calls:
            func = tool_call.get("function") or {}
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")
            tool_call_id = tool_call.get("id", "")

            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                args = {}

            handler = self.tool_registry.get(name)
            if handler is None:
                result = f"Tool '{name}' not found in registry."
            else:
                try:
                    result = handler(**args)
                    if not isinstance(result, str):
                        result = json.dumps(result)
                except Exception as exc:
                    result = f"Tool execution error: {exc}"

            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "content": result,
                }
            )

    def _combine_turns(self, turn_items: List[RolloutResult]) -> RolloutResult:
        """Combine multiple turns into a single result for batch training.

        Args:
            turn_items: List of turn results

        Returns:
            Combined RolloutResult
        """
        if not turn_items:
            raise ValueError("No turns to combine")

        # Use the last turn's full sequence
        last_turn = turn_items[-1]
        combined_token_ids = list(last_turn.token_ids)
        combined_logprobs = list(last_turn.logprobs)

        # Initialize reward_mask (all masked)
        combined_reward_mask = [0] * len(combined_token_ids)

        # For each turn, unmask its completion region
        for turn_item in turn_items:
            prompt_count = turn_item.prompt_token_count
            turn_token_count = len(turn_item.token_ids)

            completion_start = prompt_count
            completion_end = turn_token_count

            for idx in range(completion_start, completion_end):
                if idx < len(combined_reward_mask):
                    combined_reward_mask[idx] = 1

        # Collect all tool calls and contents
        all_tool_calls = []
        all_assistant_contents = []
        all_assistant_reasoning = []
        all_assistant_raw_texts = []

        for turn_item in turn_items:
            if turn_item.assistant_tool_calls:
                all_tool_calls.extend(turn_item.assistant_tool_calls)
            if turn_item.assistant_content:
                all_assistant_contents.append(turn_item.assistant_content)
            if turn_item.assistant_reasoning_content:
                all_assistant_reasoning.append(turn_item.assistant_reasoning_content)
            if turn_item.assistant_raw_text:
                all_assistant_raw_texts.append(turn_item.assistant_raw_text)

        # Store turn information for teacher alignment computation
        turn_info_for_alignment = []
        for turn_item in turn_items:
            turn_info_for_alignment.append(
                {
                    "messages": turn_item.messages,
                    "assistant_raw_text": turn_item.assistant_raw_text,
                    "completion_start": turn_item.prompt_token_count,
                    "completion_end": len(turn_item.token_ids),
                }
            )

        logger.info(
            "prompt=`%s...` combined %d turns into single item: total_tokens=%d reward_tokens=%d",
            turn_items[0].prompt[:8],
            len(turn_items),
            len(combined_token_ids),
            sum(combined_reward_mask),
        )

        return RolloutResult(
            prompt=turn_items[0].prompt,
            messages=last_turn.messages,
            token_ids=combined_token_ids,
            logprobs=combined_logprobs,
            reward_mask=combined_reward_mask,
            assistant_content=all_assistant_contents[-1] if all_assistant_contents else "",
            assistant_reasoning_content=all_assistant_reasoning[-1] if all_assistant_reasoning else None,
            assistant_tool_calls=all_tool_calls if all_tool_calls else None,
            assistant_raw_text=all_assistant_raw_texts[-1] if all_assistant_raw_texts else "",
            prompt_token_count=turn_items[0].prompt_token_count,
            parser_fallback=last_turn.parser_fallback,
            turn_index=len(turn_items) - 1,
            combined_turns=turn_info_for_alignment,
        )

    def _is_context_window_error(self, exc: Exception) -> bool:
        """Check if exception is a context window error."""
        text = str(exc).lower()
        return "context window" in text or "max_tokens" in text

    def close(self) -> None:
        """Close the collector and release resources."""
        if self._client:
            self._client.close()
            self._client = None
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def rollout_results_to_dicts(results: List[RolloutResult]) -> List[Dict[str, Any]]:
    """Convert RolloutResult objects to dicts for compatibility with existing code.

    Args:
        results: List of RolloutResult objects

    Returns:
        List of dicts with the same structure as existing rollout items
    """
    return [
        {
            "prompt": r.prompt,
            "messages": r.messages,
            "token_ids": r.token_ids,
            "logprobs": r.logprobs,
            "reward_mask": r.reward_mask,
            "assistant_content": r.assistant_content,
            "assistant_reasoning_content": r.assistant_reasoning_content,
            "assistant_tool_calls": r.assistant_tool_calls,
            "assistant_raw_text": r.assistant_raw_text,
            "prompt_token_count": r.prompt_token_count,
            "parser_fallback": r.parser_fallback,
            "turn_index": r.turn_index,
            "retokenize_match": r.retokenize_match,
            "_combined_turns": r.combined_turns,
        }
        for r in results
    ]
