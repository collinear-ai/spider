from __future__ import annotations

import contextvars
import json
import time
import logging
import subprocess
import statistics
from dataclasses import dataclass, field
from collections import deque
from datasets import load_dataset
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Deque

from spider.config import JobConfig, ToolConfig
from server.on_policy import run_on_policy_job

from .container_manager import ContainerManager, image_exists
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

_CURRENT_RUNTIME = contextvars.ContextVar(
    "swe_rebench_runtime"
)

@dataclass
class _RuntimeHandle:
    runtime: ContainerManager
    token: contextvars.Token[ContainerManager]

    def cleanup(self) -> None:
        _CURRENT_RUNTIME.reset(self.token)
        self.runtime.cleanup()

@dataclass
class ImageCacheState:
    max_batches_keep: int
    recent_batches: Deque[Set[str]] = field(default_factory=deque)

    def add(self, batch_images: Set[str]) -> None:
        if not batch_images:
            return
        self.recent_batches.append(batch_images)
        while len(self.recent_batches) > self.max_batches_keep:
            self.recent_batches.popleft()

    def recent_union(self) -> Set[str]:
        union = set()
        for batch_images in self.recent_batches:
            union.update(batch_images)
        return union

def _runtime_factory(row: Dict[str, Any]) -> _RuntimeHandle:
    instance_id = row.get("instance_id", "unknown")
    image = row.get("docker_image") or row.get("image_name") or "unknown"
    logger.info("Dispatching runtime for instance_id=%s image=%s", instance_id, image)

    runtime = ContainerManager.from_row(row)
    runtime.create()
    token = _CURRENT_RUNTIME.set(runtime)
    return _RuntimeHandle(runtime=runtime, token=token)

def _load_tool_schemas(schema_path: Path) -> List[Dict[str, Any]]:
    return json.loads(schema_path.read_text(encoding="utf-8"))

def _build_tool_configs(schemas: List[Dict[str, Any]]) -> List[ToolConfig]:
    configs = []
    for spec in schemas:
        fn = spec.get("function") or {}
        configs.append(
            ToolConfig(
                name=fn.get("name", ""),
                description=fn.get("description", ""),
                json_schema=fn.get("parameters", {}),
                source="",
                kwargs={},
            )
        )
    return configs

def _tool_wrapper(method_name: str) -> Callable[..., Any]:
    def _call(**kwargs: Any) -> Any:
        runtime = _CURRENT_RUNTIME.get()
        registry = ToolRegistry(runtime=runtime)
        method = getattr(registry, method_name)
        return method(kwargs)
    return _call

def _build_tool_registry() -> Dict[str, Callable[..., Any]]:
    return {
        "execute_bash": _tool_wrapper("_execute_bash"),
        "str_replace_editor": _tool_wrapper("_str_replace_editor"),
        "think": _tool_wrapper("_think"),
        "finish": _tool_wrapper("_finish"),
        "task_tracker": _tool_wrapper("_task_tracker"),
    }

def _cleanup_existing_containers(prefix: str = "swe-rebench-") -> None:
    proc = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name={prefix}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    container_ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not container_ids:
        return
    
    logger.info(
        "Cleaning up %d existing SWE-rebench container(s).", len(container_ids),
    )
    subprocess.run(
        ["docker", "rm", "-f", *container_ids],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

def _load_swe_rebench_instances(
    *,
    split: str = "filtered",
    instance_ids: Optional[Sequence[str]] = None,
    dataset_name: str = "nebius/SWE-rebench",
    dataset_config: str = "default",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    if max_examples:
        ds = ds.select(range(max_examples))
    rows = [dict(row) for row in ds]
    if instance_ids:
        wanted = set(instance_ids)
        rows = [row for row in rows if row.get("instance_id") in wanted]
    return rows

def _maybe_pull_image(image: str) -> None:
    if image_exists(image):
        return
    try:
        _pull_image(image)
    except Exception as exc:
        logger.warning("Failed to prefetch image %s: %s", image, exc)

def _prefetch_images_async(
    images: Iterable[str],
    *,
    pool: ThreadPoolExecutor,
) -> None:
    for image in images:
        pool.submit(_maybe_pull_image, image)

def _pull_image(image: str) -> str:
    proc = subprocess.run(
        ["docker", "pull", image],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout.strip())
    return proc.stdout

def _collect_batch_images(rows: List[Dict[str, Any]]) -> Set[str]:
    images = set()
    for row in rows:
        image = _image_for_row(row)
        if image:
            images.add(image)
    return images

def _image_for_row(row: Dict[str, Any]) -> Optional[str]:
    image = row.get("docker_image") or row.get("image_name")
    return str(image) if image else None

def _prepull_images(
    rows: List[Dict[str, Any]],
    *,
    workspace: Path,
    max_parallel: int = 2,
) -> None:
    images = sorted({img for img in (_image_for_row(row) for row in rows) if img})
    if not images:
        return

    workspace.mkdir(parents=True, exist_ok=True)
    manifest_path = workspace / "swe-rebench-image-pulls.json"
    
    start_ts = time.time()
    results = []

    def _maybe_pull(image: str) -> Dict[str, Any]:
        if image_exists(image):
            return {"image": image, "status": "present"}
        output = _pull_image(image)
        return {
            "image": image, 
            "status": "pulled", 
            "output_tail": output[-2000:],
        }

    total = len(images)
    completed = 0
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(_maybe_pull, image) for image in images]
        for fut in as_completed(futures):
            results.append(fut.result())
            completed += 1
            print(f"[prepull] pulled {completed}/{total} images", flush=True)

    payload = {
        "started_at": start_ts,
        "finished_at": time.time(),
        "images": results,
        "note": "Images are stored in the local Docker daemon cache.",
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _build_prompt(row: Dict[str, Any]) -> str:
    repo = row.get("repo", "")
    base_commit = row.get("base_commit", "")
    problem_statement = row.get("problem_statement", "")
    hints_text = row.get("hints_text", "")
    install_cfg = row.get("install_config") or {}
    test_cmd = install_cfg.get("test_cmd") if isinstance(install_cfg, dict) else ""

    repo_dir = repo.replace("/", "__") if repo else "repo"
    uploaded_path = f"/workspace/{repo_dir}"

    template_path = Path(__file__).parent / "prompts" / "user.txt"
    template = template_path.read_text(encoding="utf-8")

    hints_block = ""
    if hints_text:
        hints_block = f"\nHints:\n{hints_text}\n"

    test_cmd_block = ""
    if test_cmd:
        test_cmd_block = f"\nTest command:\n{test_cmd}\n"

    return template.format(
        uploaded_path=uploaded_path,
        repo_dir=repo_dir,
        problem_statement=problem_statement or "(no issue text provided)",
        base_commit=base_commit or "(unknown base commit)",
        hints_block=hints_block,
        test_cmd_block=test_cmd_block,
    )

def run_server_only(
    *,
    job: JobConfig,
    workspace: Path,
    split: str = "filtered",
    instance_ids: Optional[List[str]] = None,
    schema_path: Optional[Path] = None,
    job_env: Optional[Dict[str, str]] = None,
    on_batch_start_lookahead: int = 2,
    prefetch_max_workers: int = 2,
    max_batches_keep: int = 2,
) -> Any:
    schema_path = schema_path or (Path(__file__).parent / "schemas.json")
    tool_schemas = _load_tool_schemas(schema_path)
    job.tools = _build_tool_configs(tool_schemas)

    system_prompt_path = Path(__file__).parent / "prompts" / "system.txt"
    if system_prompt_path.exists():
        job.generation.system_prompt = system_prompt_path.read_text(encoding="utf-8")

    _cleanup_existing_containers()

    rows = _load_swe_rebench_instances(
        split=split, 
        instance_ids=instance_ids,
        max_examples=job.source.max_examples,
    )
    for row in rows:
        row["prompt"] = _build_prompt(row)

    tool_registry = _build_tool_registry()
    image_cache_state = ImageCacheState(max_batches_keep=max_batches_keep)
    prefetch_pool = ThreadPoolExecutor(max_workers=prefetch_max_workers)

    def _on_batch_start(rows):
        images = _collect_batch_images(rows)
        _prefetch_images_async(images, pool=prefetch_pool)

    def _on_batch_complete(rows):
        image_cache_state.add(_collect_batch_images(rows))

    return run_on_policy_job(
        job_id="swe-rebench-openhands",
        job=job,
        workspace=workspace,
        job_env=job_env or {},
        prompts=rows,
        tool_registry=tool_registry,
        runtime_factory=_runtime_factory,
        on_batch_start=_on_batch_start,
        on_batch_start_lookahead=on_batch_start_lookahead,
        on_batch_complete=_on_batch_complete,
    )