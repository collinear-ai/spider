from __future__ import annotations

import asyncio, logging, os, shutil, tarfile, urllib.request, zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Mapping, Tuple
from urllib.parse import urlparse
import blobfile

import tinker
from spider.config import JobConfig
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDataset,
    TeacherConfig,
)
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .sources import collect_prompts
from . import events
from .writers import JSONLBatchWriter

logger = logging.getLogger(__name__)

class PromptListDatasetBuilder(RLDatasetBuilder):
    def __init__(
        self, 
        *, 
        prompts: List[str], 
        dataset_name: str,
        groups_per_batch: int, 
        group_size: int,
        renderer_name: str, 
        model_name_for_tokenizer: str,
    ) -> None:
        object.__setattr__(self, "_prompts", prompts)
        object.__setattr__(self, "_dataset_name", dataset_name)
        object.__setattr__(self, "_groups_per_batch", groups_per_batch)
        object.__setattr__(self, "_group_size", group_size)
        object.__setattr__(self, "_renderer_name", renderer_name)
        object.__setattr__(self, "_model_name", model_name_for_tokenizer)

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tokenizer = get_tokenizer(self._model_name)
        renderer = renderers.get_renderer(self._renderer_name, tokenizer=tokenizer)
        dataset = PromptOnlyDataset(
            prompts=self._prompts,
            batch_size=self._groups_per_batch,
            group_size=self._group_size,
            renderer=renderer,
            tokenizer=tokenizer,
            max_prompt_tokens=None,
            convo_prefix=None,
            dataset_name=self._dataset_name,
        )
        return dataset, None

@contextmanager
def _temporary_api_key(api_key: str | None):
    if not api_key:
        yield
        return

    previous = os.environ.get("TINKER_API_KEY")
    if previous == api_key:
        yield
        return

    os.environ["TINKER_API_KEY"] = api_key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TINKER_API_KEY", None)
        else:
            os.environ["TINKER_API_KEY"] = previous

def run_on_policy_job(
    job_id: str, 
    job: JobConfig, 
    *, 
    workspace: Path
):
    from .executor import (
        JobExecutionResult,
        JobExecutionError,
        _base_metadata,
        _summarize_metrics,
        _write_metadata,
    )
    options = job.generation.on_policy_options
    if not options:
        raise JobExecutionError("on_policy_options must be provided")
    
    student_model = job.model.name
    if not student_model:
        raise JobExecutionError("job.model.name must be provided")

    artifact_path = workspace / "result.jsonl"
    metadata_path = workspace / "metadata.json"
    
    prompts = collect_prompts(job.source)
    logger.info(
        "Job %s: on-policy distillation for student=%s collected %d prompts",
        job_id,
        student_model,
        len(prompts)
    )
    events.emit(
        "Collected prompts for on-policy distillation.",
        code="on_policy.prompts_collected",
        data={"total_prompts": len(prompts)}
    )

    payload = _base_metadata(job_id, job)
    payload["generation_mode"] = "on_policy"
    sanitized_options = options.model_dump(exclude_none=True).copy()
    sanitized_options.pop("api_key", None)
    payload["on_policy"] = sanitized_options
    _write_metadata(metadata_path, payload, 0)

    if not prompts:
        artifact_path.write_text("", encoding="utf-8")
        payload["metrics"] = {"records": 0}
        _write_metadata(metadata_path, payload, 0)

        logger.info("Job %s: no prompts found. skipping on-policy distillation.", job_id)
        events.emit(
            "No prompts found for on-policy distillation.",
            level="warning",
            code="on_policy.no_prompts",
        )
        return JobExecutionResult(
            artifacts_path=artifact_path,
            metrics={"records": 0},
            messages=["No prompts found, skipped on-policy distillation."]
        )
    
    renderer_name = model_info.get_recommended_renderer_name(student_model)
    dataset_builder = PromptListDatasetBuilder(
        prompts=prompts,
        dataset_name=(job.source.dataset or "prompts"),
        groups_per_batch=options.groups_per_batch,
        group_size=options.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=student_model,
    )

    teacher_config = TeacherConfig(base_model=options.teacher)
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=options.groups_per_batch,
    )

    training_dir = workspace / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Job %s: starting Tinker training (teacher=%s, prompts=%d) logs=%s",
        job_id,
        options.teacher,
        len(prompts),
        training_dir
    )
    events.emit(
        "Starting on-policy distillation training.",
        code="on_policy.training_start",
        data={"teacher": options.teacher}
    )

    train_config = train_on_policy.Config(
        learning_rate=options.learning_rate,
        dataset_configs=[dataset_config],
        model_name=student_model,
        max_tokens=options.max_tokens,
        compute_post_kl=options.compute_post_kl,
        lora_rank=options.lora_rank,
        evaluator_builders=[],
        kl_penalty_coef=options.kl_penalty_coef,
        kl_discount_factor=options.kl_discount_factor,
        loss_fn=options.loss_fn,
        num_substeps=options.num_substeps,
        wandb_project=None,
        wandb_name=None,
        log_path=str(training_dir),
        base_url=None,
        enable_trace=False,
        eval_every=options.eval_every,
        save_every=options.save_every,
        load_checkpoint_path=None,
    )

    _configure_tinker_logging()

    with _temporary_api_key(options.api_key):
        try:
            asyncio.run(train_on_policy.main(train_config))
        except Exception as exc:
            raise JobExecutionError(f"On-policy distillation failed: {exc}") from exc

        checkpoint = checkpoint_utils.get_last_checkpoint(
            str(training_dir), required_key="sampler_path"
        )
        if not checkpoint or "sampler_path" not in checkpoint:
            raise JobExecutionError("On-policy training did not produce a sampler checkpoint")

        hf_payload_dir, manifest = _prepare_hf_payload(
            training_dir=training_dir,
            checkpoint=checkpoint,
            workspace=workspace,
        )
        if not manifest:
            raise JobExecutionError("On-policy training did not produce uploadable checkpoint artifacts")
        
    metrics = {
        "records": 0,
        "training_batches": float(checkpoint.get("batch", 0))
    }
    payload["metrics"] = metrics
    payload["training"] = {
        "sampler_checkpoint": checkpoint.get("sampler_path"),
        "state_checkpoint": checkpoint.get("state_path"),
        "last_batch": checkpoint.get("batch"),
        "teacher_model": options.teacher,
    }
    payload["hf_upload"] = {
        "repo_id": job.output.hf.repo_id if job.output.hf else None,
        "relative_dir": hf_payload_dir.relative_to(workspace).as_posix(),
        "manifest": manifest,
    }
    _write_metadata(metadata_path, payload, 0)

    summary_record = {
        "job_id": job_id,
        "metrics": metrics,
        "hf_manifest": manifest,
        "hf_payload_dir": hf_payload_dir.relative_to(workspace).as_posix(),
    }
    if job.output.hf:
        summary_record["hf_repo_id"] = job.output.hf.repo_id
    with JSONLBatchWriter(artifact_path) as writer:
        writer.write_records([summary_record])

    logger.info(
        "Job %s: training finished at batch=%s, sampler=%s",
        job_id,
        checkpoint.get("batch"),
        checkpoint.get("sampler_path")
    )
    events.emit(
        "On-policy distillation completed.",
        code="on_policy.completed",
        data={"training_batches": metrics["training_batches"]}
    )
    return JobExecutionResult(
        artifacts_path=artifact_path,
        metadata_path=metadata_path,
        upload_source=hf_payload_dir,
        metrics=metrics,
        messages=["On-policy distillation completed."]
    )

def _configure_tinker_logging() -> None:
    stream_formatter = logging.Formatter("[tinker] %(message)s")
    for name in ("tinker", "tinker_cookbook"):
        tinker_logger = logging.getLogger(name)
        tinker_logger.setLevel(logging.INFO)
        if not any(getattr(handler, "_spider_tinker", False) for handler in tinker_logger.handlers):
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(stream_formatter)
            handler._spider_tinker = True
            tinker_logger.addHandler(handler)

def _prepare_hf_payload(
    *,
    training_dir: Path,
    checkpoint: Mapping[str, object],
    workspace: Path,
) -> Tuple[Path, Dict[str, str]]:
    payload_dir = workspace / "hf_upload"
    if payload_dir.exists():
        shutil.rmtree(payload_dir)
    payload_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for label, key in (("sampler", "sampler_path"), ("state", "state_path")):
        copied = _copy_checkpoint_artifact(checkpoint.get(key), payload_dir)
        if copied:
            manifest[label] = copied

    _ensure_lora_artifacts(payload_dir, manifest)

    checkpoints_index = training_dir / "checkpoints.jsonl"
    if checkpoints_index.exists():
        shutil.copy2(checkpoints_index, payload_dir / "checkpoints.jsonl")
        manifest["checkpoints_index"] = "checkpoints.jsonl"

    return payload_dir, manifest

def _copy_checkpoint_artifact(path_value: object, dest_root: Path) -> str | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    if _is_remote_uri(path_value):
        try:
            dest = _download_remote_artifact(path_value, dest_root)
        except Exception as exc:
            logger.warning("Failed to download remote checkpoint artifact %s: %s", path_value, exc)
            return None
        if not dest:
            return None
        finalized = _extract_artifact_if_archive(dest)
        return finalized.relative_to(dest_root).as_posix() if dest else None

    src = Path(path_value)
    if not src.exists():
        logger.warning('Checkpoint artifact missing: %s', src)
        return None
    dest = dest_root / src.name
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    finalized = _extract_artifact_if_archive(dest)
    return finalized.relative_to(dest_root).as_posix()

def _is_remote_uri(path: str) -> bool:
    parsed = urlparse(path)
    if not parsed.scheme:
        return False
    if parsed.scheme == "file":
        return False
    if len(parsed.scheme) == 1 and parsed.scheme.isalpha():
        return False
    return True

def _download_remote_artifact(uri: str, dest_root: Path) -> Path | None:
    if _is_tinker_uri(uri):
        return _download_tinker_artifact(uri, dest_root)

    if not blobfile.exists(uri):
        logger.warning("Remote checkpoint artifact missing: %s", uri)
        return None
    
    name = Path(urlparse(uri).path.rstrip("/")).name or "artifact"
    dest = dest_root / name
    if blobfile.isdir(uri):
        _download_remote_directory(uri, dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with blobfile.BlobFile(uri, "rb") as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return _extract_artifact_if_archive(dest)

def _download_remote_directory(uri: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for entry in blobfile.scandir(uri):
        child_uri = f"{uri.rstrip('/')}/{entry.name}"
        child_dest = dest / entry.name
        if entry.is_dir():
            _download_remote_directory(child_uri, child_dest)
        else:
            child_dest.parent.mkdir(parents=True, exist_ok=True)
            with blobfile.BlobFile(child_uri, "rb") as src, child_dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)

def _is_tinker_uri(uri: str) -> bool:
    return urlparse(uri).scheme == "tinker"

def _download_tinker_artifact(uri: str, dest_root: Path) -> Path | None:
    parsed = urlparse(uri)
    run_id = parsed.netloc.strip()
    artifact_path = parsed.path.lstrip("/")
    if not run_id or not artifact_path:
        logger.warning("Malformed Tinker checkpoint URI: %s", uri)
        return None
    
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    
    try:
        future = rest_client.get_checkpoint_archive_url_from_tinker_path(uri)
        response = future.result(timeout=300)
    except Exception as exc:
        logger.warning("Failed to resolve Tinker checkpoint URL for %s: %s", uri, exc)
        return None

    signed_url = getattr(response, "url", None)
    if not signed_url:
        logger.warning("Tinker checkpoint response missing signed URL for %s", uri)
        return None

    dest_root.mkdir(parents=True, exist_ok=True)
    archive_path = dest_root / _checkpoint_archive_name(run_id, artifact_path)
    try:
        urllib.request.urlretrieve(signed_url, archive_path)
    except Exception as exc:
        logger.warning("Failed to download Tinker checkpoint archive %s: %s", uri, exc)
        return None

    try:
        extracted = _extract_artifact_if_archive(archive_path)
    except Exception as exc:
        logger.warning("Failed to extract archive %s: %s", archive_path, exc)
        return None

    return extracted

def _extract_artifact_if_archive(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Archive not found: {path}")
    if not path.is_file() or \
        not path.name.lower().endswith((".tar", ".tar.gz", ".zip")):
        return path
    
    extract_dir = _derive_extract_dir(path)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if tarfile.is_tarfile(path):
        with tarfile.open(path, mode="r:*") as archive:
            _safe_extract_tar(archive, extract_dir)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, mode="r") as archive:
            _safe_extract_zip(archive, extract_dir)
    else:
        raise RuntimeError(f"Unsupported archive type: {path}")

def _ensure_lora_artifacts(payload_dir: Path, manifest: Dict[str, str]) -> None:
    weight_files = sorted(
        p.relative_to(payload_dir).as_posix()
        for p in payload_dir.rglob("*.safetensors")
        if p.is_file()
    )
    if weight_files:
        manifest.setdefault("weights", weight_files[0])
    else:
        logger.warning("No .safetensors files found in HF payload directory %s", payload_dir)

    adapter_config = _find_adapter_config(payload_dir)
    if adapter_config:
        manifest.setdefault("adapter_config", adapter_config.relative_to(payload_dir).as_posix())
    else:
        logger.warning("No adapter config found in HF payload directory %s", payload_dir)

def _find_adapter_config(payload_dir: Path) -> Path | None:
    candidate_names = (
        "adapter_config.json", 
        "adapter_config.yaml",
        "config.json",
    )
    for name in candidate_names:
        candidate = payload_dir / name
        if candidate.exists():
            return candidate
    for candidate in payload_dir.rglob("adapter_config.json"):
        return candidate
    return None

def _checkpoint_archive_name(run_id: str, artifact_path: str) -> str:
    sanitized = artifact_path.strip().replace("/", "_") or "checkpoint"
    if not sanitized.endswith((".tar", ".zip")):
        sanitized = f"{sanitized}.tar"
    return f"{run_id}_{sanitized}"

def _derive_extract_dir(path: Path) -> Path:
    suffixes = (".tar.gz", ".tgz", ".tar.xz", ".tar", ".zip")
    base_name = path.name
    for suffix in suffixes:
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break
    if not base_name:
        base_name = "extracted"
    return path.parent / base_name

def _safe_extract_tar(archive: tarfile.TarFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for member in archive.getmembers():
        member_target = (dest_resolved / member.name).resolve(strict=False)
        if not str(member_target).startswith(str(dest_resolved)):
            raise RuntimeError(f"Unsafe member path detected in archive: {member.name}")
    archive.extractall(dest_resolved)

def _safe_extract_zip(archive: zipfile.ZipFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for name in archive.namelist():
        member_target = (dest_resolved / name).resolve(strict=False)
        if not str(member_target).startswith(str(dest_resolved)):
            raise RuntimeError(f"Unsafe member path detected in archive: {name}")
    archive.extractall(dest_resolved)