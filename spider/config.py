from __future__ import annotations
from dataclasses import field
from itertools import filterfalse
import json, yaml
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, List, Literal
from pydantic import AnyHttpUrl, BaseModel, Field, model_validator, field_validator

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
    max_examples: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of examples to load from the split"
    )
    shuffle: bool = Field(
        default=False,
        description="Shuffle the dataset before sampling"
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra dataset loading options"
    )
    multi_turn: bool = Field(
        default=False,
        description="Enable multi-turn user simulation for each source prompt"
    )
    user_simulation_prompt: Optional[str] = Field(
        default=None,
        description="System prompt used by the user-simulation model"
    )
    user_model: Optional["ModelConfig"] = Field(
        default=None,
        description="Model config for the user-simulation model"
    )

    @model_validator(mode="after")
    def validate_source(self) -> "SourceConfig":
        if not self.dataset:
            raise ValueError("`dataset` is required")
        if self.multi_turn:
            if not self.user_simulation_prompt:
                raise ValueError("`user_simulation_prompt` is required when `multi_turn` is true")
            if not self.user_model:
                raise ValueError("`user_model` is required when `multi_turn` is true")
        return self

class OnPolicyConfig(BaseModel):
    teacher: str = Field(..., description="Teacher model to compute KL (HF model name for tokenizer)")
    learning_rate: float = Field(default=1e-4, gt=0.0)
    groups_per_batch: int = Field(default=512, ge=1)
    group_size: int = Field(default=4, ge=1)
    max_tokens: int = Field(default=4096, ge=1)
    lora_rank: int = Field(default=32, ge=1)
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
    token_budget: Optional[int] = Field(
        default=None,
        ge=1,
        description="Stop on-policy training after accumulating this many generated tokens"
    )
    save_every: int = Field(default=20, ge=0)
    wandb_project: Optional[str] = Field(default=None, description="Wandb project name for logging")
    wandb_name: Optional[str] = Field(default=None, description="Wandb run name")

    # DeepSpeed configuration
    use_deepspeed: bool = Field(
        default=False,
        description="Enable DeepSpeed ZeRO optimizer for memory-efficient training"
    )
    deepspeed_config_path: Optional[str] = Field(
        default=None,
        description="Path to DeepSpeed config JSON file (if None, uses programmatic config)"
    )
    deepspeed_zero_stage: int = Field(
        default=2,
        ge=1,
        le=3,
        description="DeepSpeed ZeRO stage (1, 2, or 3). Stage 2 recommended for LoRA training."
    )
    deepspeed_offload_optimizer: bool = Field(
        default=False,
        description="Offload optimizer states to CPU (saves GPU memory)"
    )
    deepspeed_offload_param: bool = Field(
        default=False,
        description="Offload parameters to CPU (ZeRO-3 only, saves more memory)"
    )

class GenerationConfig(BaseModel):
    duplications: int = Field(default=1, ge=1)
    max_batch_size: Optional[int] = Field(default=None, ge=1)
    max_tool_turns: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of assistant/tool turns allowed per user request"
    )
    max_user_turns: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of user turns in user simulation"
    )
    seed: Optional[int] = Field(default=None)
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sampler parameters forwarded to the remote backend"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose per-turn logging for generation"
    )
    system_prompt: Optional[List[str]] = Field(
        default=None,
        description="System prompt(s) prepended to each chat history",
    )
    on_policy: bool = Field(
        default=False,
        description="Enable on-policy training"
    )
    on_policy_options: Optional[OnPolicyConfig] = Field(
        default=None,
        description="Config for on-policy distillation workflow"
    )

    @field_validator("system_prompt", mode="before")
    @classmethod
    def normalize_system_prompt(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            if not value:
                return None
            if not all(isinstance(item, str) for item in value):
                raise ValueError("`system_prompt` must be a list of strings")
            return value
        raise ValueError("`system_prompt` must be a string or a list of strings")

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
    student_checkpoint_path: Optional[str] = Field(
        default=None,
        description="Optional tinker:// URI to resume to student from a prior checkpoint"
    )

    @model_validator(mode="after")
    def validate_tinker_student(self) -> "ModelConfig":
        provider = (self.provider or "").lower()
        if provider == "tinker" and not self.name:
            raise ValueError("`model.name` is required if provider='tinker'.")
        return self

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

    def add_packages(self, *packages: str) -> None:
        existing = set(self.packages or [])
        for pkg in packages:
            if not pkg:
                continue
            existing.add(pkg)
        self.packages = sorted(existing)
    
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

    def ensure_runtime(self) -> RuntimeDependencyConfig:
        if self.runtime is None:
            self.runtime = RuntimeDependencyConfig()
        return self.runtime

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
