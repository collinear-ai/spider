"""Parallel rollout collection using vLLM HTTP server.

This module provides VLLMRolloutCollector for collecting rollouts in parallel
using ThreadPoolExecutor and vLLM's HTTP API with logprobs support.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
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
    tool_timeout: Optional[float] = None  # Timeout for tool execution (None = no timeout)
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
        
        # Shared progress tracking for turns (thread-safe)
        import threading
        turn_counter = {"total": 0, "lock": threading.Lock()}
        
        def track_turn():
            with turn_counter["lock"]:
                turn_counter["total"] += 1
        
        # Submit all tasks (pass prompt_idx for debugging)
        futures = {self._executor.submit(self._run_prompt, p, i, track_turn): i for i, p in enumerate(prompts)}
        
        results = []
        completed = 0
        failed = 0
        last_log_time = time.time()
        
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
                with open("/home/ubuntu/spider/error.txt", "w") as f:
                    f.write(str(exc))
                logger.error("Rollout failed for prompt %d: %s", prompt_idx, exc, exc_info=True)
            
            # Log progress every 10 seconds or on completion
            now = time.time()
            if now - last_log_time >= 10.0 or completed + failed == total_prompts:
                elapsed = now - start_time
                logger.info(
                    "Rollout progress: %d/%d prompts (%.0f%%), %d turns, %d results, %.1fs elapsed",
                    completed + failed,
                    total_prompts,
                    100.0 * (completed + failed) / total_prompts,
                    turn_counter["total"],
                    len(results),
                    elapsed,
                )
                last_log_time = now
        
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

    def _run_prompt(self, row: Dict[str, Any], prompt_idx: int = 0, on_turn: Optional[Callable[[], None]] = None) -> List[RolloutResult]:
        """Run single prompt with multi-turn tool calling.

        Args:
            row: Dict with 'prompt' and optional 'system_prompt'
            prompt_idx: Index of this prompt in the batch (for debugging)
            on_turn: Optional callback to track turn completion

        Returns:
            List of RolloutResult objects for this trajectory
        """
        prompt = row["prompt"]
        system_prompt = row.get("system_prompt")

        # Full history for rollout (grows unbounded during generation)
        full_history: List[Dict[str, Any]] = []
        if system_prompt:
            full_history.append({"role": "system", "content": system_prompt})
        full_history.append({"role": "user", "content": prompt})

        turn_items: List[RolloutResult] = []
        runtime = None
        max_history_turns = 10  # Max turns to keep in sliding window
        max_history_tokens = 16384  # Max tokens for history

        if self.runtime_factory:
            runtime = self.runtime_factory(row)

        try:
            for turn_idx in range(self.max_tool_turns):
                # Truncate history for generation (sliding window)
                history = self._truncate_history(full_history, max_history_turns, max_history_tokens)
                
                # Retry loop for parsing failures
                max_parse_retries = 3
                for parse_attempt in range(max_parse_retries):
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
                    
                    # Check if parsing failed - retry if so
                    parser_fallback = response.get("parser_fallback", False)
                    tool_calls = response.get("tool_calls")
                    if parser_fallback and not tool_calls and parse_attempt < max_parse_retries - 1:
                        logger.warning(
                            "Tool call parse failed, retrying: prompt=`%s...` turn=%d attempt=%d",
                            prompt[:20],
                            turn_idx,
                            parse_attempt + 1,
                        )
                        continue
                    break  # Success or exhausted retries
                else:
                    # Exhausted retries due to context window error
                    break

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

                full_history.append(snapshot)

                turn_items.append(
                    RolloutResult(
                        prompt=prompt,
                        messages=list(history) + [snapshot],  # history used for this turn + response
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
                
                # Track turn completion for progress reporting
                if on_turn:
                    on_turn()

                if not tool_calls:
                    break

                # Execute tool calls (measure time to identify slow tools)
                tool_exec_start = time.time()
                self._execute_tool_calls(tool_calls, full_history, prompt_idx, turn_idx)
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

        # Return each turn as separate training item (no combining)
        # Each turn already has truncated history context
        if turn_items:
            self._debug_save_rollout(prompt[:20], turn_items)
            logger.info(
                "prompt=`%s...` returning %d turns as training items (max_tokens=%d per turn)",
                prompt[:8], len(turn_items), max(len(t.token_ids) for t in turn_items)
            )
        return turn_items

    def _truncate_history(
        self, history: List[Dict[str, Any]], max_turns: int, max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Truncate history to fit within turn and token limits.
        
        Keeps system message + user message + last N assistant/tool turns.
        """
        if len(history) <= 2:  # Just system + user
            return list(history)
        
        # Always keep system (if present) and first user message
        prefix = []
        rest = list(history)
        
        if rest and rest[0].get("role") == "system":
            prefix.append(rest.pop(0))
        if rest and rest[0].get("role") == "user":
            prefix.append(rest.pop(0))
        
        # Keep last max_turns worth of messages from rest
        # Each "turn" is roughly: assistant + tool responses
        if len(rest) > max_turns * 2:
            rest = rest[-(max_turns * 2):]
        
        truncated = prefix + rest
        
        # Check token count and trim further if needed
        token_count = self._estimate_history_tokens(truncated)
        while token_count > max_tokens and len(rest) > 2:
            rest = rest[2:]  # Remove oldest turn (assistant + tool)
            truncated = prefix + rest
            token_count = self._estimate_history_tokens(truncated)
        
        return truncated

    def _estimate_history_tokens(self, history: List[Dict[str, Any]]) -> int:
        """Rough token estimate for history (4 chars ~ 1 token)."""
        text = json.dumps(history)
        return len(text) // 4

    def _debug_save_rollout(self, prompt_prefix: str, turn_items: List[RolloutResult]) -> None:
        """Save rollout debug info to pkl."""
        import pickle
        from pathlib import Path
        debug_dir = Path("/home/ubuntu/spider/debug_chunk_history")
        debug_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_prefix = "".join(c if c.isalnum() else "_" for c in prompt_prefix)
        filepath = debug_dir / f"{timestamp}_{safe_prefix}.pkl"
        
        debug_data = {
            "prompt_prefix": prompt_prefix,
            "num_turns": len(turn_items),
            "turns": [
                {
                    # Summary stats
                    "turn_idx": t.turn_index,
                    "token_count": len(t.token_ids),
                    "reward_tokens": sum(t.reward_mask),
                    "prompt_tokens": t.prompt_token_count,
                    "messages_count": len(t.messages),
                    # Actual data
                    # input
                    "token_ids": t.token_ids,
                    "logprobs": t.logprobs,
                    "reward_mask": t.reward_mask,
                    "messages": t.messages,
                    # output
                    "assistant_content": t.assistant_content,
                    "assistant_reasoning_content": t.assistant_reasoning_content,
                    "assistant_tool_calls": t.assistant_tool_calls,
                    "assistant_raw_text": t.assistant_raw_text,
                    "parser_fallback": t.parser_fallback,
                }
                for t in turn_items
            ],
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(debug_data, f)

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
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "logprobs": True,
            "top_logprobs": 1,
        }

        if self.tools:
            payload["tools"] = self.tools

        # Use thread-local client to avoid contention
        client = self._get_client()
        
        response = client.post("/v1/chat/completions", json=payload)
        
        if response.status_code >= 400:
            body = (response.text or "").strip()
            raise RuntimeError(
                f"vLLM chat failed (status={response.status_code}): {body}"
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



        # Extract logprobs and token IDs
        # CRITICAL: We must keep logprobs aligned with token_ids for importance sampling
        # vLLM returns logprobs[i] for its token[i], so we need the exact token IDs vLLM used
        logprobs_data = choice.get("logprobs") or {}
        logprobs_content = logprobs_data.get("content") or []



        token_strings = []
        logprobs = []
        token_ids = []
        
        for lp_entry in logprobs_content:
            # Extract token string
            try:
                token_str = lp_entry["token"]
                token_strings.append(token_str)
            except KeyError:
                token_strings.append("")
                token_str = ""
            
            # Extract logprob
            try:
                logprobs.append(lp_entry["logprob"])
            except KeyError:
                logprobs.append(0.0)
            
            # Convert token string to token ID
            # Method 1: Try convert_tokens_to_ids (handles tokenizer-specific token formats)
            token_id = self._tokenizer.convert_tokens_to_ids(token_str)
            if token_id != self._tokenizer.unk_token_id:
                token_ids.append(token_id)
                continue
            
            # Method 2: Try using bytes if available (most accurate for edge cases)
            token_bytes = lp_entry.get("bytes")
            if token_bytes is not None:
                try:
                    token_text = bytes(token_bytes).decode("utf-8")
                    # Try convert_tokens_to_ids first
                    token_id = self._tokenizer.convert_tokens_to_ids(token_text)
                    if token_id != self._tokenizer.unk_token_id:
                        token_ids.append(token_id)
                        continue
                    # Fall back to encode
                    ids = self._tokenizer.encode(token_text, add_special_tokens=False)
                    if ids:
                        token_ids.append(ids[0])
                        if len(ids) > 1:
                            logger.debug("Token from bytes encoded to multiple IDs: %s -> %s", token_text[:20], ids[:5])
                        continue
                except Exception as e:
                    logger.debug("Failed to process token bytes: %s", e)
            
            # Method 3: Fall back to encoding the token string
            ids = self._tokenizer.encode(token_str, add_special_tokens=False)
            if ids:
                token_ids.append(ids[0])
                if len(ids) > 1:
                    logger.debug("Token string encoded to multiple IDs: '%s' -> %s", token_str[:20], ids[:5])
            else:
                # Last resort: use unknown token
                token_ids.append(self._tokenizer.unk_token_id or 0)
                logger.debug("Could not convert token to ID, using unk: '%s'", token_str[:20])
        
        # Verify alignment - this is critical for importance sampling
        if len(token_ids) != len(logprobs):
            logger.error(
                "CRITICAL: Token/logprob count mismatch: token_ids=%d logprobs=%d. "
                "This will cause importance sampling ratio errors!",
                len(token_ids),
                len(logprobs),
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
        if parsed.parser_fallback:
            content = parsed.content
            tool_calls = parsed.tool_calls
            if parsed.reasoning:
                reasoning = parsed.reasoning

        # Get prompt token count from vLLM's usage - this is the authoritative count
        # CRITICAL: We must use vLLM's count, not local tokenization, because vLLM may use
        # a different chat template that produces different token counts
        usage = data.get("usage") or {}
        vllm_prompt_tokens = usage.get("prompt_tokens", 0)
        vllm_completion_tokens = usage.get("completion_tokens", 0)

        # Build full token sequence (prompt + generated)
        # Compute prompt tokens locally for the full sequence
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tools=self.tools if self.tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_token_ids = self._tokenizer.encode(prompt_text)
        
        # Log diagnostic info
        local_prompt_count = len(prompt_token_ids)
        logger.info(
            "Token counts: vLLM_prompt=%d vLLM_completion=%d local_prompt=%d generated=%d logprobs=%d",
            vllm_prompt_tokens,
            vllm_completion_tokens,
            local_prompt_count,
            len(token_ids),
            len(logprobs),
        )
        
        # Log first few logprobs to verify they're not all zeros
        if logprobs:
            sample_lps = logprobs[:5]
            logger.info("First 5 generated logprobs from vLLM: %s", sample_lps)
        
        # Check for mismatch between local and vLLM tokenization
        if vllm_prompt_tokens > 0 and local_prompt_count != vllm_prompt_tokens:
            logger.warning(
                "Prompt token count mismatch: local=%d vLLM=%d (diff=%d). "
                "Adjusting to match vLLM.",
                local_prompt_count,
                vllm_prompt_tokens,
                local_prompt_count - vllm_prompt_tokens,
            )
            # Adjust prompt_token_ids to match vLLM's count
            # This ensures full_token_ids, full_logprobs, and reward_mask all align
            if vllm_prompt_tokens > local_prompt_count:
                # vLLM has more tokens - pad with unk tokens
                pad_count = vllm_prompt_tokens - local_prompt_count
                pad_token = self._tokenizer.unk_token_id or 0
                prompt_token_ids = prompt_token_ids + [pad_token] * pad_count
            else:
                # vLLM has fewer tokens - truncate
                prompt_token_ids = prompt_token_ids[:vllm_prompt_tokens]
        elif vllm_prompt_tokens == 0:
            logger.warning("vLLM did not return prompt_tokens in usage, using local count=%d", local_prompt_count)
        
        # Use consistent prompt token count for all arrays
        prompt_token_count = len(prompt_token_ids)
        
        # Build full token IDs
        full_token_ids = list(prompt_token_ids) + token_ids
        
        # Debug log vLLM call
        self._log_vllm_call(
            messages=messages,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            token_ids=token_ids,
            logprobs=logprobs,
            raw_text=raw_text,
            prompt_token_count=prompt_token_count,
        )

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "token_ids": token_ids,
            "logprobs": logprobs,
            "raw_text": raw_text,
            "prompt_token_count": prompt_token_count,
            "full_token_ids": full_token_ids,
            "parser_fallback": parsed.parser_fallback,
        }

    def _log_vllm_call(
        self,
        messages: List[Dict[str, Any]],
        content: str,
        reasoning: Optional[str],
        tool_calls: Optional[List[Dict[str, Any]]],
        token_ids: List[int],
        logprobs: List[float],
        raw_text: str,
        prompt_token_count: int,
    ) -> None:
        """Log vLLM request/response to debug folder."""
        import pickle
        debug_dir = Path("/home/ubuntu/spider/debug_vllm_calls")
        debug_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
        log_data = {
            "timestamp": timestamp,
            "request": {
                "messages": messages,
                "model": self.lora_name or self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            "response": {
                "content": content,
                "reasoning": reasoning,
                "tool_calls": tool_calls,
                "raw_text": raw_text,
                "token_ids": token_ids,
                "logprobs": logprobs,
                "prompt_token_count": prompt_token_count,
                "completion_token_count": len(token_ids),
            },
        }
        
        log_file = debug_dir / f"vllm_call_{timestamp}.pkl"
        try:
            with open(log_file, "wb") as f:
                pickle.dump(log_data, f)
        except Exception as e:
            logger.warning("Failed to write vLLM call log: %s", e)

    def _log_tool_exec(
        self,
        name: str,
        args: Dict[str, Any],
        result: str,
        duration: float,
        error: Optional[str] = None,
        prompt_idx: int = 0,
        turn_idx: int = 0,
    ) -> None:
        """Log tool execution to debug folder."""
        debug_dir = Path("debug_tool_exec")
        debug_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_entry = {
            "timestamp": timestamp,
            "prompt_idx": prompt_idx,
            "turn_idx": turn_idx,
            "tool_name": name,
            "args": args,
            "result": result,
            "duration_s": round(duration, 3),
            "error": error,
        }
        
        log_file = debug_dir / f"p{prompt_idx}_t{turn_idx}_{name}_{timestamp}.json"
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to write tool exec log: %s", e)

    def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
        prompt_idx: int = 0,
        turn_idx: int = 0,
    ) -> None:
        """Execute tool calls and append results to history with timeout.

        Args:
            tool_calls: List of tool call objects
            history: Chat history to append results to
            prompt_idx: Index of prompt in batch for logging
            turn_idx: Turn index for logging
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
            error = None
            exec_start = time.time()
            
            if handler is None:
                result = f"Tool '{name}' not found in registry."
                error = "not_found"
            else:
                # Execute tool (with optional timeout)
                try:
                    if self.tool_timeout:
                        # Use ThreadPoolExecutor to enforce timeout
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(handler, **args)
                            result = future.result(timeout=self.tool_timeout)
                    else:
                        # No timeout - execute directly
                        result = handler(**args)
                    
                    if not isinstance(result, str):
                        result = json.dumps(result)
                except FuturesTimeoutError:
                    result = f"Tool '{name}' execution timed out after {self.tool_timeout}s"
                    error = "timeout"
                    logger.warning(
                        "Tool '%s' timed out after %.1fs, skipping result",
                        name, self.tool_timeout
                    )
                except Exception as exc:
                    result = f"Tool execution error: {exc}"
                    error = str(exc)
                    logger.warning("Tool '%s' raised exception: %s", name, exc)

            exec_duration = time.time() - exec_start
            self._log_tool_exec(name, args, result, exec_duration, error, prompt_idx, turn_idx)

            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "content": result,
                }
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
