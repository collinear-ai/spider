from __future__ import annotations

from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, HfFolder, upload_file, upload_folder
from huggingface_hub.utils import HfHubHTTPError

from spider.config import HFUploadConfig

class HFUploadError(RuntimeError):
    pass

def publish_to_hub(
    *, job_id: str, artifact: Path, metadata: Path, config: HFUploadConfig
) -> str:
    token = _resolve_token(config)
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

def _resolve_token(config: HFUploadConfig) -> str:
    if config.token:
        return config.token
    token = HfFolder.get_token()
    if token:
        return token
    raise HFUploadError("Hugging Face token must be supplied via config or `huggingface-cli login`")

def _wrap_hf_error(prefix: str, exc: Exception) -> HFUploadError:
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", "unknown")
        detail = getattr(exc, "message", None) or str(exc)
        return HFUploadError(f"{prefix} (status {status}): {detail}")
    return HFUploadError(f"{prefix}: {exc}")