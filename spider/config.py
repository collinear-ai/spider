from __future__ import annotations
import json, yaml
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, List, Literal
from pydantic import AnyHttpUrl, BaseModel, Field, model_validator

class SourceType(str, Enum):
    HF_DATASET = "hf_dataset"
    REMOTE_URI = "remote_uri"

class SourceConfig(BaseModel):
    dataset: str = Field(
        ...,
        description="HF dataset repo ID"
    )
    config_name: Optional[str] = Field(
        default=None,
        description="HF dataset config name"
    )
    split: str = Field(
        default="train",
        description="HF dataset split name"
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
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra dataset loading options"
    )

    @model_validator(mode="after")
    def validate_source(self) -> "SourceConfig":
        if not self.dataset:
            raise ValueError("`dataset` is required")
        return self

class OnPolicyConfig(BaseModel):
    teacher: str = Field(..., description="Teacher model to compute KL")
    api_key: Optional[str] = Field(default=None, description="API key for Tinker")
    learning_rate: float = Field(default=1e-4, gt=0.0)
    groups_per_batch: int = Field(default=512, ge=1)
    group_size: int = Field(default=4, ge=1)
    max_tokens: int = Field(default=4096, ge=1)
    lora_rank: int = Field(default=32, ge=1)
    num_substeps: int = Field(default=1, ge=1)
    kl_penalty_coef: float = Field(default=1.0)
    kl_discount_factor: float = Field(default=0.0)
    loss_fn: Literal["importance_sampling", "ppo"] = Field(
        default="importance_sampling",
        description="Loss for updating the student model"
    )
    compute_post_kl: bool = Field(
        default=False,
        description="Whether to compute post-kl metrics after each step"
    )
    eval_every: int = Field(default=20, ge=0)
    save_every: int = Field(default=20, ge=0)

class GenerationConfig(BaseModel):
    duplications: int = Field(default=1, ge=1)
    max_batch_size: Optional[int] = Field(default=None, ge=1)
    max_turns: Optional[int] = Field(
        default=8,
        ge=1,
        description="Maximum number of turns allowed for tool calls"
    )
    seed: Optional[int] = Field(default=None)
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sampler parameters forwarded to the remote backend"
    )
    on_policy: bool = Field(
        default=False,
        description="Enable on-policy training"
    )
    on_policy_options: Optional[OnPolicyConfig] = Field(
        default=None,
        description="Config for on-policy distillation workflow"
    )

    @model_validator(mode="after")
    def validate_on_policy(self) -> "GenerationConfig":
        if self.on_policy and not self.on_policy_options:
            raise ValueError("`on_policy_options` is required when `on_policy` is true")
        return self

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
    private: bool = Field(default=False, description="whether to keep the target repo private")
    repo_type: str = Field(
        default="dataset",
        description="HF repo type to upload to (eg. dataset or model)"
    )
    config_name: Optional[str] = Field(
        default=None,
        description="Subset config name for the dataset upload"
    )
    
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

class ProcessorConfig(BaseModel):
    name: str = Field(..., description="processor callable name")
    source: str = Field(..., description="Python source text defining the processor callable")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Kwargs forwarded to the processor")

class ToolConfig(BaseModel):
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    json_schema: Dict[str, Any] = Field(
        ..., description="JSON schema for the tool arguments"
    )
    source: str = Field(..., description="Python source text defining the tool callable")
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs forwarded to the tool callable"
    )

class RuntimeDependencyConfig(BaseModel):
    packages: List[str] = Field(
        default_factory=list,
        description="List of pip requirements that must be installed before execution."
    )
    env: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to expose the runetime sandbox when using dependencies"
    )
    
class JobConfig(BaseModel):
    model: ModelConfig
    source: SourceConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    pre_processor: Optional[ProcessorConfig] = Field(
        default=None,
        description="Optional callable to filter / transform dataset rows to generate final prompts"
    )
    post_processor: Optional[ProcessorConfig] = Field(
        default=None,
        description="Optional callable to filter / transform rollout rows to generate final completions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata to attach to the job request"
    )
    tools: List[ToolConfig] = Field(
        default_factory=list,
        description="Optional set of tools"
    )
    runtime: Optional[RuntimeDependencyConfig] = Field(
        default=None,
        description="Optional runtime dependency requirements declared by the client"
    )

    @model_validator(mode="after")
    def validate_on_policy_output(self) -> "JobConfig":
        if not self.generation.on_policy:
            return self
        if self.output.mode != OutputMode.HF_UPLOAD:
            raise ValueError("`output.mode` must be `upload_hf` when `generation.on_policy` is true")
        if not self.output.hf or not self.output.hf.repo_id.strip():
            raise ValueError("`output.hf.repo_id` is required when `generation.on_policy` is true")
        return self

class ServerConfig(BaseModel):
    base_url: AnyHttpUrl
    api_key: Optional[str] = Field(
        default=None,
        description="API key used to authenticate with the remote server"
    )
    verify_tls: bool = Field(default=False, description="Whether to verify TLS certificates")
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