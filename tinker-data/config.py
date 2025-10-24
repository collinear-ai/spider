from __future__ import annotations
import json, yaml
from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import Any, Dict, List, Optional

class SourceConfig(BaseModel):
    type: str
    path: str
    field: str
    options: Dict[str, Any] = Field(default_factory=dict)

class OutputConfig(BaseModel):
    format: str
    destination: str
    push_to_hub: Optional[Dict[str, Any]] = None

class GenerationConfig(BaseModel):
    duplications: int = Field(default=1, ge=1)
    max_batch_size: int = Field(default=8, ge=1)
    seed = Optional[int] = None

class ModelConfig(BaseModel):
    name: str
    backend: str = Field(default="vllm")
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @validator("backend")
    def ensure_local_backend(cls, value: str):
        if value != "vllm":
            raise ValueError("Only local vllm backend is supported right now.")
        return value

class AppConfig(BaseModel):
    model: ModelConfig
    sources: List[SourceConfig]
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    output: OutputConfig

    @classmethod
    def load(cls, path: str, overrides: Optional[Dict[str, Any]] = None):
        data = _read_config_file(path)
        if overrides:
            data = _deep_merge(data, overrides)
        return cls.model_validate(data)

def _read_config_file(path: str):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    suffix = config_path.suffix.lower()
    if suffix == ".yaml":
        return yaml.safe_load(content) or {}
    raise ValueError(f"Unsupported config file format: {suffix}")

def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]):
    result = dict(base)
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result