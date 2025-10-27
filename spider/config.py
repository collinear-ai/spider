from __future__ import annotations
import json, yaml
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import AnyHttpUrl, BaseModel, Field, model_validator

class SourceType(str, Enum):
    INLINE_UPLOAD = "inline_upload"
    HF_DATASET = "hf_dataset"
    REMOTE_URI = "remote_uri"

class SourceConfig(BaseModel):
    type: SourceType
    name: str = Field(..., description="Unique id for the data source within a job")
    dataset: Optional[str] = Field(
        default=None,
        description="HF dataset repo ID when type='hf_dataset'"
    )
    field: Optional[str] = Field(
        default=None,
        description="Field to extract from each record when using an HF dataset"
    )
    revision: Optional[str] = Field(
        default=None,
        description="Revision for HF dataset"
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream the dataset instead of download"
    )
    uri: Optional[str] = Field(
        default=None,
        description="Remote URI fetch when type='remote_uri'"
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra backend-specific options for this source"
    )

    @model_validator(mode="after")
    def validate_source(self) -> "SourceConfig":
        if self.type == SourceType.HF_DATASET:
            if not self.dataset:
                raise ValueError("`dataset` is required when type='hf_dataset'")
        elif self.type == SourceType.REMOTE_URI:
            if not self.uri:
                raise ValueError("`uri` is required when type='remote_uri'")
        elif self.type == SourceType.INLINE_UPLOAD:
            pass
        else:
            raise ValueError(f"Unsupported source type: {self.type}")
        return self

class GenerationConfig(BaseModel):
    duplications: int = Field(default=1, ge=1)
    max_batch_size: int = Field(default=8, ge=1)
    seed: Optional[int] = Field(default=None)
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sampler parameters forwarded to the remote backend"
    )

class ModelConfig(BaseModel):
    provider: str = Field(..., description="Identifier for the remote inference provider")
    name: Optional[str] = Field(
        default=None,
        description="Model name or engine identifier for the remote backend"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional backend params"
    )

class OutputMode(str, Enum):
    RETURN = "return"
    HF_UPLOAD = "upload_hf"

class HFUploadConfig(BaseModel):
    repo_id: str = Field(..., description="Target dataset repo on HF")
    config_name: Optional[str] = Field(
        default=None,
        description="Subset config name for the dataset upload"
    )
    token: Optional[str] = Field(
        default=None,
        description="Write token supplied by client for Hub access"
    )
    private: bool = Field(default=True, description="whether to keep the target repo private")

class OutputConfig(BaseModel):
    mode: OutputMode = Field(default=OutputMode.RETURN)
    local_path: Optional[str] = Field(
        default=None,
        description="Optional client-side path where downloaded artifacts should be stored"
    )
    hf: Optional[HFUploadConfig] = Field(default=None)
    format: str = Field(
        default="jsonl",
        description="Output format requested from the server"
    )

    @model_validator(mode="after")
    def validate_mode(self) -> "OutputConfig":
        if self.mode == OutputMode.RETURN:
            return self
        elif self.mode == OutputMode.HF_UPLOAD:
            if not self.hf:
                raise ValueError("`hf` is required when mode='upload_hf'")
        else:
            raise ValueError(f"Unsupported output mode: {self.mode}")
        return self

class JobConfig(BaseModel):
    model: ModelConfig
    sources: List[SourceConfig]
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata to attach to the job request"
    )

class ServerConfig(BaseModel):
    base_url: AnyHttpUrl
    api_key: Optional[str] = Field(
        default=None,
        description="API key used to authenticate with the remote server"
    )
    verify_tls: bool = Field(default=True, description="Whether to verify TLS certificates")
    request_timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Timeout in seconds for client-side HTTP requests"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional headers to include in HTTP requests"
    )

class AppConfig(BaseModel):
    server: ServerConfig
    job: JobConfig

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
    content = config_path.read_text(encoding="utf-8")
    if not content.strip():
        return {}
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(content) or {}
    if suffix == ".json":
        return json.loads(content)
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