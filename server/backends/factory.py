from __future__ import annotations

from spider.config import ModelConfig
from .vllm_backend import VLLMBackend
from .openai_backend import OpenAIBackend
from .openrouter_backend import OpenRouterBackend

def create_backend(config: ModelConfig):
    provider = config.provider.lower()
    if provider == "vllm":
        return VLLMBackend(config)
    if provider == "openai":
        return OpenAIBackend(config)
    if provider == "openrouter":
        return OpenRouterBackend(config)
    if provider == "tinker":
        raise ValueError(
            "`provider: tinker` is only valid for on-policy jobs. Set generation.on_policy=true."
        )
    raise ValueError(f"Unsupported provider: {config.provider}")
