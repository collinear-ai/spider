from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Tuple, Dict
from huggingface_hub import HfApi, upload_file, upload_folder
from huggingface_hub.utils import HfHubHTTPError
import shutil
import logging

from spider.config import HFUploadConfig

logger = logging.getLogger(__name__)


class HFUploadError(RuntimeError):
    pass

def publish_to_hub(
    *, job_id: str, artifact: Path, metadata: Path, config: HFUploadConfig
) -> str:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    repo_id = config.repo_id
    repo_type = (config.repo_type or "dataset").strip() or "dataset"
    repo_kwargs = {
        "repo_id": repo_id, 
        "token": token, 
        "private": config.private, 
        "repo_type": repo_type,
    }

    try:
        api.create_repo(**repo_kwargs, exist_ok=True)
    except Exception as exc:
        raise _wrap_hf_error(f"Failed to ensure HF repo {repo_id}", exc) from exc
    
    prefix = config.config_name.strip() if config.config_name else ""

    if not artifact.exists():
        raise HFUploadError(f"Artifact file not found at {artifact}")
    if not metadata.exists():
        raise HFUploadError(f"Metadata file not found at {metadata}")
    try:
        if artifact.is_dir():
            upload_folder(
                folder_path=str(artifact),
                repo_id=repo_id,
                repo_type=repo_type,
                path_in_repo=prefix or None,
                token=token,
            )
        else:
            artifact_remote = _compose_repo_path(prefix, artifact.name)
            upload_file(
                path_or_fileobj=str(artifact),
                path_in_repo=artifact_remote,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
            )
    except Exception as exc:
        raise _wrap_hf_error(f"Failed to upload artifacts for job {job_id}", exc) from exc

    if repo_type == "dataset":
        base_url = f"https://huggingface.co/datasets/{repo_id}"
    else:
        base_url = f"https://huggingface.co/{repo_id}"
        
    if artifact.is_dir():
        return f"{base_url}/tree/main/{prefix}" if prefix else base_url

    artifact_remote = _compose_repo_path(prefix, artifact.name)
    return f"{base_url}/blob/main/{artifact_remote}"

def _compose_repo_path(prefix: str, name: str) -> str:
    if prefix:
        return f"{prefix.rstrip('/')}/{name}"
    return name

def _wrap_hf_error(prefix: str, exc: Exception) -> HFUploadError:
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", "unknown")
        detail = getattr(exc, "message", None) or str(exc)
        return HFUploadError(f"{prefix} (status {status}): {detail}")
    return HFUploadError(f"{prefix}: {exc}")


def _prepare_hf_payload(
    *,
    training_dir: Path,
    checkpoint: Mapping[str, object],
    workspace: Path,
) -> Tuple[Path, Dict[str, str], str | None]:
    """Prepare HuggingFace upload payload from checkpoint."""
    payload_dir = workspace / "hf_upload"
    if payload_dir.exists():
        shutil.rmtree(payload_dir)
    payload_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    sampler_path = checkpoint.get("sampler_path")

    if sampler_path and Path(sampler_path).exists():
        # Copy LoRA adapter files
        adapter_path = Path(sampler_path)
        if adapter_path.is_dir():
            # Copy safetensors weights
            for sf_file in adapter_path.glob("*.safetensors"):
                dest = payload_dir / sf_file.name
                shutil.copy2(sf_file, dest)
                manifest["weights"] = sf_file.name

            # Copy adapter config
            config_file = adapter_path / "adapter_config.json"
            if config_file.exists():
                dest = payload_dir / "adapter_config.json"
                shutil.copy2(config_file, dest)
                manifest["adapter_config"] = "adapter_config.json"

    checkpoints_index = training_dir / "checkpoints.jsonl"
    checkpoints_index_text = None
    if checkpoints_index.exists():
        try:
            checkpoints_index_text = checkpoints_index.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read checkpoints index %s: %s", checkpoints_index, exc)

    return payload_dir, manifest, checkpoints_index_text
