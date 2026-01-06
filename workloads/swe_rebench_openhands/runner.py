from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass
from datasets import load_dataset
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from spider.config import JobConfig, ToolConfig
from server.on_policy import run_on_policy_job

from .container_manager import ContainerManager
from .tool_registry import ToolRegistry

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

def _runtime_factory(row: Dict[str, Any]) -> _RuntimeHandle:
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

def _load_swe_rebench_instances(
    *,
    split: str = "filtered",
    instance_ids: Optional[Sequence[str]] = None,
    dataset_name: str = "nebius/SWE-rebench",
    dataset_config: str = "default",
) -> List[Dict[str, Any]]:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    rows = [dict(row) for row in ds]
    if instance_ids:
        wanted = set(instance_ids)
        rows = [row for row in rows if row.get("instance_id") in wanted]
    return rows

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
) -> Any:
    schema_path = schema_path or (Path(__file__).parent / "schemas.json")
    tool_schemas = _load_tool_schemas(schema_path)
    job.tools = _build_tool_configs(tool_schemas)

    rows = _load_swe_rebench_instances(split=split, instance_ids=instance_ids)
    for row in rows:
        row["prompt"] = _build_prompt(row)

    tool_registry = _build_tool_registry()
    return run_on_policy_job(
        job_id="swe-rebench-openhands",
        job=job,
        workspace=workspace,
        job_env=job_env or {},
        prompts=rows,
        tool_registry=tool_registry,
        runtime_factory=_runtime_factory,
    )