"""Parallel rollout collection using vLLM HTTP server.

This module provides VLLMRolloutCollector for collecting rollouts in parallel
using ThreadPoolExecutor and vLLM's HTTP API with logprobs support.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import httpx
from tqdm.auto import tqdm

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
    _thread_local: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Use connection pooling and keep-alive for better performance
        # Increase connection limits to match rollout_workers and account for multiple turns per prompt
        # Each worker can have multiple concurrent requests (one per turn), so we need more connections
        max_concurrent_requests = self.max_workers * min(self.max_tool_turns, 10)  # Cap at 10 concurrent turns per worker
        self._client = httpx.Client(
            base_url=self.vllm_base_url,
            timeout=httpx.Timeout(480.0, connect=60.0),
            limits=httpx.Limits(
                max_keepalive_connections=max(200, max_concurrent_requests),  # Keep more connections alive
                max_connections=max(400, max_concurrent_requests * 2),  # Allow more concurrent connections
            ),
            http2=False,  # HTTP/1.1 is faster for vLLM
        )
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Use thread-local storage for per-thread HTTP clients to avoid contention
        import threading
        self._thread_local = threading.local()

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
        start_time = time.time()
        total_prompts = len(prompts)
        
        # Submit all tasks
        futures = {self._executor.submit(self._run_prompt, p): i for i, p in enumerate(prompts)}
        
        results = []
        completed = 0
        failed = 0
        
        # Create progress bar for rollout collection
        # Use position=0 and file=sys.stderr to persist at bottom of terminal
        pbar = tqdm(
            total=total_prompts,
            desc="Rollouts",
            unit="prompt",
            leave=True,  # Keep visible after completion
            ncols=100,
            position=0,  # Fixed position at bottom
            file=sys.stderr,  # Use stderr so it doesn't interfere with stdout logging
            mininterval=0.5,  # Update at least every 0.5 seconds
        )
        
        # Process completed futures with progress tracking
        for future in as_completed(futures):
            prompt_idx = futures[future]
            try:
                result = future.result()
                if result:
                    results.extend(result)
                    completed += 1
                else:
                    logger.warning("Rollout for prompt %d returned no results", prompt_idx)
            except Exception as exc:
                failed += 1
                logger.error("Rollout failed for prompt %d: %s", prompt_idx, exc, exc_info=True)
            
            pbar.update(1)
            pbar.set_postfix({
                "completed": completed,
                "failed": failed,
                "results": len(results)
            })
        
        pbar.close()
        
        elapsed = time.time() - start_time
        throughput = total_prompts / elapsed if elapsed > 0 else 0
        
        logger.info(
            "Rollout collection complete: prompts=%d completed=%d failed=%d results=%d "
            "time=%.2fs throughput=%.2f prompts/s",
            total_prompts,
            completed,
            failed,
            len(results),
            elapsed,
            throughput,
        )
        
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

        turn_start_time = time.time()
        try:
            for turn_idx in range(self.max_tool_turns):
                try:
                    turn_gen_start = time.time()
                    response = self._generate_with_logprobs(history)
                    turn_gen_time = time.time() - turn_gen_start
                    
                    # Log slow generations to identify bottlenecks
                    if turn_gen_time > 5.0:  # Log if generation takes > 5 seconds
                        logger.warning(
                            "Slow generation: prompt=`%s...` turn=%d generation_time=%.2fs tokens=%d",
                            prompt[:20],
                            turn_idx,
                            turn_gen_time,
                            len(response.get("token_ids", [])),
                        )
                    elif self.verbose:
                        logger.debug(
                            "Prompt=`%s...` turn=%d generation_time=%.2fs tokens=%d",
                            prompt[:20],
                            turn_idx,
                            turn_gen_time,
                            len(response.get("token_ids", [])),
                        )
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
                full_logprobs = [0.0] * prompt_token_count + logprobs

                # Full token sequence
                full_token_ids = response["full_token_ids"]

                if not (len(full_logprobs) == len(full_token_ids) == len(reward_mask)):
                    raise ValueError(
                        "Length mismatch: full_logprobs=%d full_token_ids=%d reward_mask=%d logprobs=%d token_ids=%d prompt_token_count=%d",
                        len(full_logprobs), len(full_token_ids), len(reward_mask), len(logprobs), len(token_ids), prompt_token_count
                    )

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
                        logprobs=full_logprobs,
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

                # Execute tool calls (measure time to identify slow tools)
                tool_exec_start = time.time()
                self._execute_tool_calls(tool_calls, history)
                tool_exec_time = time.time() - tool_exec_start
                if tool_exec_time > 1.0:  # Log if tool execution takes > 1 second
                    logger.warning(
                        "Slow tool execution: prompt=`%s...` turn=%d tool_time=%.2fs num_tools=%d",
                        prompt[:20],
                        turn_idx,
                        tool_exec_time,
                        len(tool_calls),
                    )

        finally:
            if runtime is not None:
                runtime.cleanup()

        # Combine all turns into single item for batch training
        if len(turn_items) == 0:
            return []

        if len(turn_items) == 1:
            return [turn_items[0]]

        return [self._combine_turns(turn_items)]

    def _get_client(self) -> httpx.Client:
        """Get thread-local HTTP client to avoid contention on shared client."""
        if not hasattr(self._thread_local, 'client'):
            # Create per-thread client with same configuration
            self._thread_local.client = httpx.Client(
                base_url=self.vllm_base_url,
                timeout=httpx.Timeout(480.0, connect=60.0),
                limits=httpx.Limits(
                    max_keepalive_connections=50,  # Per-thread connection pool
                    max_connections=100,  # Per-thread max connections
                ),
                http2=False,
            )
        return self._thread_local.client

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
            # "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "logprobs": True,
            "top_logprobs": 1,
        }

        if self.tools:
            payload["tools"] = self.tools

        # Use thread-local client to avoid contention
        client = self._get_client()
        
        # Retry logic for JSON serialization errors
        max_retries = 3
        for attempt in range(max_retries):
            response = client.post("/v1/chat/completions", json=payload)
            
            if response.status_code >= 400:
                body = (response.text or "").strip()
                
                # Check if it's the JSON serialization error (inf/-inf in logprobs)
                is_json_error = (
                    response.status_code == 500 and
                    ("Out of range float values" in body or "JSON compliant" in body)
                )
                
                if is_json_error and attempt < max_retries - 1:
                    logger.warning(
                        "vLLM JSON serialization error (attempt %d/%d), retrying...",
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                
                # Log error and raise
                import pickle
                from datetime import datetime
                log_dir = Path("/home/ubuntu/spider/vllm_call_logs")
                log_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                log_file = log_dir / f"vllm_call_input_{timestamp}.pkl"
                with open(log_file, "wb") as f:
                    pickle.dump({
                        "payload": payload,
                        "model_name": model_name,
                        "messages": messages,
                        "error": body[:500],
                    }, f)
                logger.debug("Saved vLLM call input to %s", log_file)
                
                raise RuntimeError(
                    f"vLLM chat failed (status={response.status_code}): {body[:512]}"
                )
            
            # Success - break out of retry loop
            break

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

        token_strings = []
        logprobs = []
        for lp_entry in logprobs_content:
            try:
                token_strings.append(lp_entry["token"])
            except KeyError:
                token_strings.append("")
            try:
                lp_val = lp_entry["logprob"]
                # Sanitize inf/-inf/nan logprobs (vLLM can return these)
                import math
                if lp_val is None or not math.isfinite(lp_val):
                    lp_val = -100.0  # Use a very negative but finite value
                logprobs.append(lp_val)
            except KeyError:
                logprobs.append(0.0)

        import unicodedata
        concatenated = "".join(token_strings)
        normalized_text = unicodedata.normalize('NFC', concatenated)
        token_ids = self._tokenizer.encode(normalized_text, add_special_tokens=False)
        
        if len(token_ids) != len(token_strings):
            logger.debug(
                "Token count mismatch: original=%d re-encoded=%d",
                len(token_strings),
                len(token_ids),
            )

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
