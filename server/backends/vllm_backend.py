from __future__ import annotations

from typing import Dict, Iterable, List, Any, Optional, Union
import threading, logging, contextlib, os, socket, subprocess, sys, time, httpx, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from spider.config import ModelConfig

logger = logging.getLogger(__name__)

class VLLMBackend:
    def __init__(
        self,
        config: ModelConfig,
        *,
        enable_lora: bool = False,
        max_lora_rank: int = 64,
        lora_modules: Optional[str] = None,
        gpu_ids: Optional[List[int]] = None,
    ):
        if not config.name:
            raise ValueError("`model.name` is required for vLLM backend.")
        self._config = config

        self._server_host = "127.0.0.1"
        self._server_port = _reserve_port()
        self._client_timeout = 480.0 # 8 minutes
        self._model_params = {
            k: v for k, v in config.parameters.items()
            if k not in ("system_prompt",)
        }
        self._base_url = f"http://{self._server_host}:{self._server_port}"
        self._client = None
        self._server_proc = None

        self._last_metrics: Dict[str, object] = {}
        self._metrics_lock = threading.Lock()
        self._tool_parser = config.parameters.get("tool_parser") or _default_tool_parser(config.name or "")
        self._chat_template = _default_chat_template(config.name or "")
        self._reasoning_parser = config.parameters.get("reasoning_parser") or _default_reasoning_parser(config.name or "")

        # LoRA configuration
        self._enable_lora = enable_lora
        self._max_lora_rank = max_lora_rank
        self._lora_modules = lora_modules
        self._gpu_ids = gpu_ids

        try:
            self._start_server()
        except Exception:
            self.close()
            raise

    @property
    def base_url(self) -> str:
        """Return the base URL of the vLLM server."""
        return self._base_url

    @property
    def tool_parser(self) -> Optional[str]:
        """Return the tool parser name for this model."""
        return self._tool_parser

    @property
    def reasoning_parser(self) -> Optional[str]:
        """Return the reasoning parser name for this model."""
        return self._reasoning_parser

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self._client:
            raise RuntimeError("vLLM HTTP client is not initialized.")
        payload = {
            "model": self._config.name,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        payload.update(dict(parameters or {}))
        logger.info(
            "vLLM chat called with %d messages(s), tools=%s",
            len(messages),
            bool(tools),
        )

        response = self._client.post("/v1/chat/completions", json=payload)

        if response.status_code >= 400:
            body = (response.text or "").strip()
            payload_preview = json.dumps(payload)[:2048]
            logger.error(
                "vLLM chat request failed (status=%s) payload=%s body=%s",
                response.status_code,
                payload_preview,
                body[:2048].replace("\n", "\\n"),
            )
            raise RuntimeError(
                f"vLLM chat failed (status={response.status_code}): "
                f"{body[:512]} | payload={payload_preview}"
            )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM chat returned no candidate outputs.")

        message = choices[0].get("message") or {}
        content = _normalize_content(message.get("content")) or ""
        reasoning = _normalize_content(
            message.get("reasoning") or message.get("reasoning_content")
        )
        tool_calls = message.get("tool_calls")
        
        self._update_metrics(data.get("usage"))
        logger.info(
            "vLLM chat returned content length %d tool_calls=%s",
            len(content),
            bool(tool_calls),
        )
        return {
            "content": content,
            "reasoning": reasoning or None,
            "tool_calls": tool_calls
        }

    def chat_with_logprobs(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        parameters: Dict[str, Any],
        top_logprobs: int = 1,
        lora_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Chat with logprobs returned for each token.

        Args:
            messages: Chat messages
            tools: Optional tool definitions
            parameters: Generation parameters
            top_logprobs: Number of top logprobs to return per token
            lora_name: Optional LoRA adapter name to use

        Returns:
            Dict with content, tool_calls, token_ids, logprobs
        """
        if not self._client:
            raise RuntimeError("vLLM HTTP client is not initialized.")

        model_name = self._config.name
        if lora_name:
            model_name = lora_name

        payload = {
            "model": model_name,
            "messages": messages,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        if tools:
            payload["tools"] = tools
        payload.update(dict(parameters or {}))

        logger.info(
            "vLLM chat_with_logprobs called with %d messages(s), tools=%s, lora=%s",
            len(messages),
            bool(tools),
            lora_name,
        )

        response = self._client.post("/v1/chat/completions", json=payload)

        if response.status_code >= 400:
            body = (response.text or "").strip()
            payload_preview = json.dumps(payload)[:2048]
            logger.error(
                "vLLM chat request failed (status=%s) payload=%s body=%s",
                response.status_code,
                payload_preview,
                body[:2048].replace("\n", "\\n"),
            )
            raise RuntimeError(
                f"vLLM chat failed (status={response.status_code}): "
                f"{body[:512]} | payload={payload_preview}"
            )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM chat returned no candidate outputs.")

        message = choices[0].get("message") or {}
        content = _normalize_content(message.get("content")) or ""
        reasoning = _normalize_content(
            message.get("reasoning") or message.get("reasoning_content")
        )
        tool_calls = message.get("tool_calls")

        # Extract logprobs from response
        logprobs_data = choices[0].get("logprobs") or {}
        logprobs_content = logprobs_data.get("content") or []

        token_ids = []
        logprobs = []
        for lp_entry in logprobs_content:
            token = lp_entry.get("token", "")
            token_logprob = lp_entry.get("logprob", 0.0)
            # Get token ID from top_logprobs
            top_lps = lp_entry.get("top_logprobs") or []
            token_id = None
            for top_lp in top_lps:
                if top_lp.get("token") == token:
                    token_id = top_lp.get("token_id")
                    break
            if token_id is None:
                # Fallback: try to get from the entry itself
                token_id = lp_entry.get("token_id", 0)

            token_ids.append(token_id)
            logprobs.append(token_logprob)

        self._update_metrics(data.get("usage"))
        logger.info(
            "vLLM chat_with_logprobs returned content length %d tokens=%d tool_calls=%s",
            len(content),
            len(token_ids),
            bool(tool_calls),
        )

        return {
            "content": content,
            "reasoning": reasoning or None,
            "tool_calls": tool_calls,
            "token_ids": token_ids,
            "logprobs": logprobs,
        }

    def load_lora_adapter(
        self,
        lora_name: str,
        lora_path: str,
    ) -> bool:
        """Load a LoRA adapter dynamically.

        Requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=True environment variable.

        Args:
            lora_name: Name to assign to the adapter
            lora_path: Path to the adapter checkpoint

        Returns:
            True if successful, False otherwise
        """
        if not self._client:
            raise RuntimeError("vLLM HTTP client is not initialized.")

        payload = {
            "lora_name": lora_name,
            "lora_path": lora_path,
        }

        logger.info("Loading LoRA adapter: name=%s path=%s", lora_name, lora_path)

        try:
            response = self._client.post("/v1/load_lora_adapter", json=payload)
            if response.status_code != 200:
                logger.error(
                    "Failed to load LoRA adapter (status=%s): %s",
                    response.status_code,
                    response.text[:512],
                )
                return False

            logger.info("Successfully loaded LoRA adapter: %s", lora_name)
            return True
        except Exception as exc:
            logger.error("Error loading LoRA adapter: %s", exc)
            return False

    def unload_lora_adapter(self, lora_name: str) -> bool:
        """Unload a LoRA adapter dynamically.

        Args:
            lora_name: Name of the adapter to unload

        Returns:
            True if successful, False otherwise
        """
        if not self._client:
            raise RuntimeError("vLLM HTTP client is not initialized.")

        payload = {"lora_name": lora_name}

        logger.info("Unloading LoRA adapter: %s", lora_name)

        try:
            response = self._client.post("/v1/unload_lora_adapter", json=payload)
            if response.status_code != 200:
                logger.warning(
                    "Failed to unload LoRA adapter (status=%s): %s",
                    response.status_code,
                    response.text[:512],
                )
                return False

            logger.info("Successfully unloaded LoRA adapter: %s", lora_name)
            return True
        except Exception as exc:
            logger.warning("Error unloading LoRA adapter: %s", exc)
            return False

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
        max_workers = min(len(prompts), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for idx, resp in pool.map(run_one, enumerate(prompts)):
                results[idx] = resp
        
        return [r or {} for r in results]

    def metrics(self) -> Dict[str, object]:
        with self._metrics_lock:
            return dict(self._last_metrics)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._server_proc is not None:
            self._stop_server()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _start_server(self) -> None:
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self._config.name,
            "--host",
            self._server_host,
            "--port",
            str(self._server_port),
        ]
        if self._tool_parser:
            command.append("--enable-auto-tool-choice")
            command.extend(["--tool-call-parser", self._tool_parser])
        # if self._chat_template: # special template for auto-tool-choice
        #     command.extend(["--chat-template", str(self._chat_template)])
        if self._reasoning_parser:
            command.extend(["--reasoning-parser", self._reasoning_parser])

        # Add LoRA support flags
        if self._enable_lora:
            command.append("--enable-lora")
            command.extend(["--max-lora-rank", str(self._max_lora_rank)])
            if self._lora_modules:
                command.extend(["--lora-modules", self._lora_modules])

        # Removed --enforce-eager to enable CUDA graphs for better performance (2-5x faster)
        # If you encounter CUDA graph errors, re-enable this flag
        # command.append("--enforce-eager")

        for key, value in self._model_params.items():
            if value is None: continue
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    command.append(flag)
                continue
            if isinstance(value, (list, tuple)):
                for item in value:
                    command.extend([flag, str(item)])
                continue
            command.extend([flag, str(value)])

        # Check if CUDA was already initialized - this can cause issues with vLLM subprocess
        try:
            import torch
            if torch.cuda.is_initialized():
                logger.warning(
                    "CUDA was already initialized before starting vLLM server. "
                    "This may cause issues with GPU allocation. Consider starting "
                    "vLLM before any CUDA operations."
                )
        except ImportError:
            pass

        logger.info(
            "Starting vLLM HTTP server for %s on %s (enable_lora=%s, gpu_ids=%s)",
            self._config.name,
            self._base_url,
            self._enable_lora,
            self._gpu_ids,
        )

        env = os.environ.copy()
        # Enable dynamic LoRA loading if LoRA is enabled
        if self._enable_lora:
            env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
            # Enable fused MoE LoRA kernel for better performance (faster inference)
            # If you encounter Triton compilation errors, set this back to "0"
            env["VLLM_MOE_LORA_USE_FUSED_KERNEL"] = "1"
        # Set GPU visibility for vLLM server
        if self._gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self._gpu_ids))
        # Use spawn method for multiprocessing to avoid CUDA context issues
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # Use V0 engine for better stability with subprocess spawning
        # V1 engine has known issues with CUDA initialization in multiprocessing
        env["VLLM_USE_V1"] = "0"
        # Also set the multiprocessing start method to spawn to avoid fork issues
        env["VLLM_MULTIPROC_METHOD"] = "spawn"
        # Disable async output processing which can cause issues with multiprocessing
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Ensure clean CUDA state in subprocess
        env.pop("CUDA_DEVICE_ORDER", None)
        # Clear any CUDA initialization flags that might interfere
        env.pop("CUDA_CACHE_PATH", None)

        logger.info("vLLM subprocess env: VLLM_USE_V1=%s, CUDA_VISIBLE_DEVICES=%s",
                    env.get("VLLM_USE_V1"), env.get("CUDA_VISIBLE_DEVICES"))
        # Start in new session to fully isolate from parent process CUDA context
        self._server_proc = subprocess.Popen(
            command,
            env=env,
            start_new_session=True,
        )

        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                self._client_timeout,
                connect=self._client_timeout,
            )
        )
        self._wait_for_startup()

    def _wait_for_startup(self) -> None:
        assert self._client is not None
        while True:
            if self._server_proc and self._server_proc.poll() is not None:
                raise RuntimeError("vLLM HTTP server exited before it became ready.")
            try:
                response = self._client.get("/health")
                if response.status_code == 200:
                    logger.info("vLLM HTTP server ready on %s", self._base_url)
                    return
                if response.status_code == 503:
                    time.sleep(1)
                    continue
                raise RuntimeError(
                    f"vLLM HTTP server returned unexpected status {response.status_code}: "
                    f"{(response.text or '').strip()[:200]}"
                )
            except httpx.HTTPError:
                pass
            time.sleep(1)

    def _stop_server(self) -> None:
        assert self._server_proc is not None
        self._server_proc.terminate()
        try:
            self._server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._server_proc.kill()
        finally:
            self._server_proc = None

    def _update_metrics(self, usage: Optional[Dict[str, Any]]) -> None:
        if not usage:
            return
        with self._metrics_lock:
            self._last_metrics = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }

def _normalize_content(value: Union[str, List[Any], Dict[str, Any], None]) -> str:
    if value is None:
        return ""
    return value # TODO: placeholder for vision model if returning a list

def _reserve_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

def _default_tool_parser(model_name: str) -> Optional[str]:
    lower = (model_name or "").lower()
    if "llama" in lower:
        return "llama3_json"
    if "qwen3-coder" in lower:
        return "qwen3_xml"
    if "qwen3" in lower: 
        return "hermes"
    if "qwen2.5" in lower:
        return "hermes"
    if "qwen" in lower:
        return "hermes"
    if "deepseek" in lower:
        return "deepseek_v31" if "v3.1" in lower else "deepseek_v3"
    if "glm-4" in lower:
        return "glm45"
    if "mistral" in lower:
        return "mistral"
    if "gpt-oss" in lower:
        return "openai"
    return "hermes"

def _default_chat_template(model_name: str) -> Optional[Path]:
    lower = (model_name or "").lower()
    supported_models = ["mistral", "qwen3"]
    if not any(model in lower for model in supported_models):
        return None

    template_root = Path(__file__).resolve().parent / "templates"
    if "mistral" in lower:
        import vllm
        template_root = Path(vllm.__file__).resolve().parent / "examples"
        template = template_root / "tool_chat_template_mistral_parallel.jinja"
        if template.exists():
            return template
        logger.warning("Could not find mistral chat template at %s", template)
    if "qwen3" in lower:
        template = template_root / "qwen3_no_dummy_think.jinja"
        if template.exists():
            return template
        logger.warning("Could not find qwen3 chat template at %s", template)
    return None

def _default_reasoning_parser(model_name: str) -> Optional[str]:
    lower = (model_name or "").lower()
    if "deepseek-r1" in lower:
        return "deepseek_r1"
    if "qwen3" in lower or "qwq" in lower:
        return "qwen3"
    if "glm-4" in lower:
        return "glm45"
    if "kimi" in lower:
        return "kimi_k2"
    if "gpt-oss" in lower:
        return "openai_gptoss"
    return None
