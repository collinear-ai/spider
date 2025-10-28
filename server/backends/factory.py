from __future__ import annotations

from spider.config import ModelConfig
from .vllm_backend import VLLMBackend

def create_backend(config: ModelConfig):
    provider = config.provider.lower()
    if provider == "vllm":
        return VLLMBackend(config)
    raise ValueError(f"Unsupported provider: {config.provider}")