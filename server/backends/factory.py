from __future__ import annotations

from spider.config import ModelConfig
from .vllm_backend import VLLMBackend

def create_backend(config: ModelConfig):
    provider = config.provider.lower()
    if provider == "vllm":
        return VLLMBackend(config)
    if provider == "tinker":
        raise ValueError(
            "`provider: tinker` is only valid for on-policy jobs. Set generation.on_policy=true."
        )
    raise ValueError(f"Unsupported provider: {config.provider}")