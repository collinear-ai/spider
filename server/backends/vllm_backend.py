from __future__ import annotations

from typing import Dict, Iterable, List, Any, Optional
import threading, logging, contextlib, os, socket, subprocess, sys, time, httpx
from pathlib import Path

from spider.config import ModelConfig

logger = logging.getLogger(__name__)

class VLLMBackend:
    def __init__(self, config: ModelConfig):
        if not config.name:
            raise ValueError("`model.name` is required for vLLM backend.")
        self._config = config

        self._server_host = "127.0.0.1"
        self._server_port = _reserve_port()
        self._client_timeout = 120.0
        self._system_prompt = config.parameters.get("system_prompt")
        self._model_params = {
            k: v for k, v in config.parameters.items()
            if k not in ("system_prompt")
        }
        self._base_url = f"http://{self._server_host}:{self._server_port}"
        self._client = None
        self._server_proc = None
    
        self._last_metrics: Dict[str, object] = {}
        self._metrics_lock = threading.Lock()
        self._tool_parser = _default_tool_parser(config.name or "")
        self._chat_template = _default_chat_template(config.name or "")

        try:
            self._start_server()
        except Exception:
            self.close()
            raise

    def generate(self, prompts: Iterable[str], *, parameters: Dict[str, object]) -> List[str]:
        if not self._client:
            raise RuntimeError("vLLM HTTP client is not initialized.")
        prompt_list = list(prompts)
        if not prompt_list:
            return []

        payload = self._build_completion_payload(prompt_list, parameters)
        response = self._client.post("/v1/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        
        if len(choices) != len(prompt_list):
            raise RuntimeError(
                f"Generation count mismatch within batch: expected {len(prompt_list)}, received {len(choices)}"
            )

        ordered = sorted(choices, key=lambda entry: entry.get("index", 0))
        completions = [(choice.get("text") or "") for choice in ordered]
        self._update_metrics(data.get("usage"))
        return completions

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
            logger.error(
                "vLLM chat request failed (status=%s) payload=%s body=%s",
                response.status_code,
                payload,
                response.text[:2048].replace("\n", "\\n"),
            )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM chat returned no candidate outputs.")

        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls")
        
        self._update_metrics(data.get("usage"))
        logger.info(
            "vLLM chat returned content length %d tool_calls=%s",
            len(content),
            bool(tool_calls),
        )
        return {"content": content, "tool_calls": tool_calls}

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
        if self._chat_template:
            command.extend(["--chat-template", str(self._chat_template)])

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
        
        logger.info(
            "Starting vLLM HTTP server for %s on %s",
            self._config.name,
            self._base_url,
        )

        env = os.environ.copy()
        self._server_proc = subprocess.Popen(command, env=env)

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

    def _build_prompt(self, prompt: str) -> str:
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_completion_payload(
        self, prompts: List[str], parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = {
            "model": self._config.name,
            "prompt": [self._build_prompt(prompt) for prompt in prompts],
            "n": 1,
            "stream": False,
        }
        payload.update(dict(parameters or {}))
        return payload

    def _update_metrics(self, usage: Optional[Dict[str, Any]]) -> None:
        if not usage:
            return
        with self._metrics_lock:
            self._last_metrics = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }

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
    if "deepseek" in lower:
        return "deepseek_v31" if "v3.1" in lower else "deepseek_v3"
    if "glm" in lower:
        return "glm45"
    if "mistral" in lower:
        return "mistral"
    if "qwen2.5" in lower:
        return "hermes"
    return None

def _default_chat_template(model_name: str) -> Optional[Path]:
    lower = (model_name or "").lower()
    supported_models = ["mistral"]
    if not any(model in lower for model in supported_models):
        return None

    import vllm
    template_root = (
        Path(vllm.__file__).resolve().parent 
        / "examples"
    )
    if "mistral" in lower:
        template = template_root / "tool_chat_template_mistral_parallel.jinja"
        if template.exists():
            return template
        logger.warning("Could not find mistral chat template at %s", template)
    return None