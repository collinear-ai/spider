from __future__ import annotations

import json, os, logging, inspect, time, threading, traceback
import concurrent
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Iterable, Tuple
from types import MappingProxyType

from spider.config import JobConfig, OutputMode, ProcessorConfig, ToolConfig, ModelConfig
from .runtime_env import RuntimeEnvironment, RuntimeEnvironmentError
from .backends.factory import create_backend
from . import events
from .sources import collect_prompts
from .writers import JSONLBatchWriter
from .hf_upload import HFUploadError, publish_to_hub
from .on_policy import run_on_policy_job
try:
    from .scaffolds.base import Scaffold, ScaffoldConfig
    from .scaffolds.openhands_wrapper import OpenHandsScaffold, OpenHandsScaffoldConfig
    SCAFFOLDS_AVAILABLE = True
except ImportError:
    SCAFFOLDS_AVAILABLE = False

logger = logging.getLogger(__name__)
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)

class JobExecutionError(Exception):
    def __init__(self, message: str):
        tb = traceback.format_exc()
        if not tb.strip() or tb.strip() == "NoneType: None":
            tb = "".join(traceback.format_stack()[:-1])
        payload = f"{message}\n{tb}".strip()
        super().__init__(payload)

@dataclass
class JobExecutionResult:
    artifacts_path: Path
    metadata_path: Optional[Path] = None
    upload_source: Optional[Path] = None
    remote_artifact: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

def run_generation_job(
    job_id: str, 
    job: JobConfig, 
    *, 
    workspace: Path,
    job_env: Optional[Dict[str, str]] = None,
) -> JobExecutionResult:
    with _job_env_context(job_env or {}):
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        if job.generation.on_policy:
            if (
                job.output.mode != OutputMode.HF_UPLOAD
                or not job.output.hf
                or not job.output.hf.repo_id.strip()
            ):
                raise JobExecutionError(
                    "On-policy jobs require `output.mode: upload_hf` with a populated `output.hf.repo_id`"
                )

            if job.tools:
                logger.info("Job %s: starting tool-aware on-policy distillation", job_id)
                events.emit(
                    "Launching tool-aware on-policy pipeline.",
                    code="job.pipeline",
                    data={"mode": "on_policy_tool"},
                )
                result = _run_tool_on_policy_job(
                    job_id,
                    job,
                    workspace=workspace,
                    job_env=job_env or {},
                )
            
            else:
                logger.info("Job %s: starting on-policy distillation", job_id)
                events.emit("Launching on-policy distillation pipeline.", code="job.pipeline", data={"mode": "on_policy"})
                result = run_on_policy_job(
                    job_id, 
                    job, 
                    workspace=workspace,
                    job_env=job_env or {},
                )
        elif job.generation.scaffold:
            if not SCAFFOLDS_AVAILABLE:
                raise JobExecutionError(
                    "Scaffold support not available. Install with: pip install spider[swe-scaffolds]"
                )
            logger.info("Job %s: starting scaffold trajectory generation", job_id)
            events.emit(
                "Launching scaffold trajectory generation pipeline.",
                code="job.pipeline",
                data={"mode": "scaffold", "scaffold_type": job.generation.scaffold.type}
            )
            result = _run_scaffold_job(
                job_id,
                job,
                workspace=workspace,
                job_env=job_env or {},
            )
        else:
            logger.info("Job %s: starting off-policy generation pipeline", job_id)
            events.emit("Launching off-policy generation pipeline.", code="job.pipeline", data={"mode": "off_policy"})
            result = _run_off_policy_job(
                job_id, 
                job, 
                workspace=workspace,
                job_env=job_env or {},
            )

        if job.output.mode == OutputMode.HF_UPLOAD and job.output.hf:
            artifact_source = result.upload_source or result.artifacts_path
            metadata_path = result.metadata_path or (workspace / "metadata.json")
            try:
                remote = publish_to_hub(
                    job_id=job_id,
                    artifact=artifact_source,
                    metadata=metadata_path,
                    config=job.output.hf
                )
            except HFUploadError as exc:
                raise JobExecutionError(str(exc)) from exc
            result.remote_artifact = remote
            events.emit(
                "Published artifacts to remote storage.",
                code="artifact.uploaded",
                data={"remote_artifact": remote}
            )
    return result

def _run_off_policy_job(
    job_id: str, 
    job: JobConfig, 
    *, 
    workspace: Path, 
    job_env: Dict[str, str]
) -> JobExecutionResult:
    artifact_path = workspace / "result.jsonl"
    metadata_path = workspace / "metadata.json"
    runtime_env, runtime_stack, _ = _prepare_runtime_env(job, job_env)

    _ensure_tensor_parallel(job)
    backend = create_backend(job.model)

    pre_processor, post_processor, prompts = _prepare_processors_and_prompts(
        job=job,
        runtime_env=runtime_env,
        pre_processor=job.pre_processor,
        post_processor=job.post_processor,
    )

    if not prompts:
        artifact_path.write_text("", encoding="utf-8")
        events.emit(
            "No prompts found for generation.",
            level="warning",
            code="generation.no_prompts",
        )
        return JobExecutionResult(
            artifacts_path=artifact_path,
            metadata_path=metadata_path,
            metrics={"records": 0},
            messages=["No prompts found; nothing generated."]
        )

    logger.info("Job %s: collected %d prompts for generation", job_id, len(prompts))
    events.emit(
        "Collected prompts for generation.",
        code="generation.prompts_collected",
        data={"total_prompts": len(prompts)}
    )

    tool_registry = _resolve_tools(job.tools, runtime_env=runtime_env)
    try:
        batch_worker = _build_batch_worker(
            job_id=job_id,
            job=job,
            backend=backend,
            post_processor=post_processor,
            tool_registry=tool_registry,
        )
        if tool_registry:
            events.emit(
                "Tool-aware generation enabled.",
                code="generation.tools_enabled",
                data={"tool_names": sorted(tool_registry.keys())}
            )
        return _run_batched_generation(
            job_id=job_id,
            job=job,
            prompts=prompts,
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            batch_worker=batch_worker,
        )
    finally:
        _shutdown_backend(job_id, backend)
        runtime_stack.close()
        if runtime_env:
            runtime_env.cleanup()
            events.emit(
                "Runtime sandbox cleaned.",
                code="runtime.cleaned"
            )

def _run_tool_on_policy_job(
    job_id: str,
    job: JobConfig,
    *,
    workspace: Path,
    job_env: Dict[str, str],
) -> JobExecutionResult:
    if not job.tools:
        raise JobExecutionError("Tool-aware on-policy jobs require at least one tool.")
    
    runtime_env, runtime_stack, _ = _prepare_runtime_env(job, job_env)
    
    pre_processor, _, prompts = _prepare_processors_and_prompts(
        job=job,
        runtime_env=runtime_env,
        pre_processor=job.pre_processor,
    )

    tool_registry = _resolve_tools(job.tools, runtime_env=runtime_env)
    if tool_registry:
        events.emit(
            "Tool-aware rollout registry ready.",
            code="tool_on_policy.tools_ready",
            data={"tool_names": sorted(tool_registry.keys())}
        )

    try:
        return run_on_policy_job(
            job_id,
            job,
            workspace=workspace,
            job_env=job_env,
            prompts=prompts,
            tool_registry=tool_registry,
        )
    finally:
        runtime_stack.close()
        if runtime_env:
            runtime_env.cleanup()
            events.emit("Runtime sandbox cleaned.", code="runtime.cleaned")


def _run_scaffold_job(
    job_id: str,
    job: JobConfig,
    *,
    workspace: Path,
    job_env: Dict[str, str],
) -> JobExecutionResult:
    """Run scaffold-based trajectory generation job."""
    artifact_path = workspace / "result.jsonl"
    metadata_path = workspace / "metadata.json"
    
    scaffold_config = job.generation.scaffold
    if not scaffold_config:
        raise JobExecutionError("Scaffold config is required for scaffold jobs")
    
    # Get dataset info from source config
    dataset_name = job.source.dataset
    split = job.source.split
    
    # Convert JobConfig scaffold config to ScaffoldConfig
    scaffold_base_config = ScaffoldConfig(
        output_dir=workspace / "trajectories",
        max_instances=job.generation.max_batch_size,
        num_workers=scaffold_config.num_workers,
        timeout_seconds=scaffold_config.timeout_seconds,
        max_retries=scaffold_config.max_retries,
    )
    
    # Create scaffold-specific config
    if scaffold_config.type == "openhands":
        oh_config = OpenHandsScaffoldConfig(
            **scaffold_base_config.model_dump(),
            dataset=dataset_name,
            split=split,
            agent_class=scaffold_config.agent_class or "CodeActAgent",
            max_iterations=scaffold_config.max_iterations or 50,
            llm_config_name=scaffold_config.llm_config_name,
            llm_model=scaffold_config.llm_model,
            llm_api_key=scaffold_config.llm_api_key or job_env.get("LLM_API_KEY"),
            llm_base_url=scaffold_config.llm_base_url or job_env.get("LLM_BASE_URL"),
            enable_browser=scaffold_config.scaffold_specific.get("enable_browser", False),
            enable_llm_editor=scaffold_config.scaffold_specific.get("enable_llm_editor", False),
            runtime=scaffold_config.scaffold_specific.get("runtime", "docker"),
            platform=scaffold_config.scaffold_specific.get("platform", "linux/amd64"),
            remote_runtime_resource_factor=scaffold_config.scaffold_specific.get("remote_runtime_resource_factor", 1),
        )
        scaffold: Scaffold = OpenHandsScaffold(oh_config)
    else:
        raise JobExecutionError(f"Unsupported scaffold type: {scaffold_config.type}")
    
    # Run scaffold
    events.emit(
        f"Running {scaffold_config.type} scaffold on dataset {dataset_name}",
        code="scaffold.started",
        data={"scaffold_type": scaffold_config.type, "dataset": dataset_name}
    )
    
    try:
        output_path = scaffold.run_batch(
            dataset_name=dataset_name,
            split=split,
            instance_filter=None,  # Could add to config later
        )
        
        # Copy output to artifact path (Spider expects result.jsonl)
        import shutil
        if output_path != artifact_path:
            shutil.copy(output_path, artifact_path)
        
        # Count records
        record_count = 0
        if artifact_path.exists():
            with artifact_path.open("r", encoding="utf-8") as f:
                record_count = sum(1 for line in f if line.strip())
        
        events.emit(
            f"Generated {record_count} trajectories",
            code="scaffold.completed",
            data={"records": record_count}
        )
        
        # Write metadata
        _write_metadata(
            metadata_path,
            _job_snapshot(job),
            record_count
        )
        
        return JobExecutionResult(
            artifacts_path=artifact_path,
            metadata_path=metadata_path,
            metrics={"records": record_count},
            messages=[f"Generated {record_count} trajectories using {scaffold_config.type} scaffold"]
        )
    except Exception as exc:
        logger.exception("Scaffold job %s failed", job_id)
        raise JobExecutionError(f"Scaffold execution failed: {exc}") from exc


@contextmanager
def _job_env_context(job_env: Dict[str, str]):
    if not job_env:
        yield 
        return

    previous = {}
    try:
        for key, value in job_env.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

def _run_batched_generation(
    *,
    job_id: str,
    job: JobConfig,
    prompts: List[str],
    artifact_path: Path,
    metadata_path: Path,
    batch_worker: Callable[[List[str]], Tuple[List[Dict[str, Any]], Dict[str, Any]]],
) -> JobExecutionResult:
    aggregated_metrics = {}
    records_written = 0
    payload = _base_metadata(job_id, job)
    if job.runtime and job.runtime.packages:
        payload.setdefault("runtime", {})["packages"] = list(job.runtime.packages)
    _write_metadata(metadata_path, payload, records_written)

    batch_size = _resolve_batch_size(job, prompts)

    try:
        pending = {}
        next_index = 0
        total_batches = (len(prompts) + batch_size -1) // batch_size

        with JSONLBatchWriter(artifact_path) as writer:
            executor_context = ThreadPoolExecutor(max_workers=_processing_worker_count())
            try:
                for batch_index, chunk_start in enumerate(range(0, len(prompts), batch_size)):
                    chunk = prompts[chunk_start : chunk_start + batch_size]
                    events.emit(
                        "Batch started.",
                        code="batch.started",
                        data={
                            "batch_index": batch_index,
                            "batch_size": batch_size,
                            "total_batches": total_batches,
                            "records_written": writer.count,
                        }
                    )
                    future, batch_metrics = batch_worker(chunk)
                    pending[batch_index] = (future, batch_metrics)
                    next_index = _drain_ready_batches(
                        pending, 
                        next_index, 
                        writer, 
                        aggregated_metrics, 
                        payload,
                        metadata_path, 
                        block=False,
                        batch_started=batch_index,
                        job_id=job_id,
                    )
                next_index = _drain_ready_batches(
                    pending, 
                    next_index, 
                    writer, 
                    aggregated_metrics, 
                    payload,
                    metadata_path, 
                    block=True,
                    batch_started=batch_index,
                    job_id=job_id,
                )
            finally:
                if executor_context:
                    executor_context.shutdown(wait=True)
            
            records_written = writer.count

    except Exception as exc:
        raise JobExecutionError(f"Generation pipeline failed: {exc}") from exc

    filtered_records = max(0, len(prompts) - records_written)
    metrics = _summarize_metrics(records_written, aggregated_metrics)
    if filtered_records:
        metrics["filtered_records"] = filtered_records
    payload["metrics"] = metrics
    _write_metadata(metadata_path, payload, records_written)

    logger.info(
        "Job %s: generation complete; wrote %d records (filtered=%d)",
        job_id,
        records_written,
        filtered_records
    )
    events.emit(
        "Generation completed.",
        code="generation.completed",
        data={"records_written": records_written, "filtered_records": filtered_records}
    )
    return JobExecutionResult(
        artifacts_path=artifact_path,
        metadata_path=metadata_path,
        upload_source=artifact_path,
        metrics=metrics,
        messages=["Generation pipeline completed."]
    )

def _build_batch_worker(
    *,
    job_id: str,
    job: JobConfig,
    backend: Any,
    post_processor: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]],
    tool_registry: Dict[str, Callable[..., Any]],
) -> Callable[[List[str]], Tuple[Future[List[Dict[str, Any]]], Dict[str, Any]]]:
    if not tool_registry:
        return lambda prompts: (
            _immediate_future(_text_batch_worker(
                prompts=prompts,
                backend=backend,
                job=job,
                post_processor=post_processor,
                )
            ),
            dict(backend.metrics() or {})
        )

    return lambda prompts: (
        _tool_batch_worker(
            job_id=job_id,
            prompts=prompts,
            backend=backend,
            job=job,
            post_processor=post_processor,
            tool_registry=tool_registry,
        ),
        dict(backend.metrics() or {})
    )

def _pair_records(prompts: Iterable[str], generations: Iterable[str]) -> List[Dict[str, Any]]:
    paired = []
    for prompt, completion in zip(prompts, generations):
        paired.append({"prompt": prompt, "completion": completion})
    return paired

def _resolve_processor(spec: Optional[ProcessorConfig], *, runtime_env: Optional["RuntimeEnvironment"] = None) -> Optional[Callable[[Dict[str, Any]], Any]]:
    if spec is None:
        return None
    exec_ns = {}
    try:
        compiled = compile(spec.source, "<processor>", "exec")
        exec(compiled, exec_ns, exec_ns)
    except Exception as exc:
        raise JobExecutionError(f"Failed to load processor `{spec.name}`: {exc}") from exc
    
    func = exec_ns.get(spec.name)
    if not callable(func):
        raise JobExecutionError(f"Processor source did not define the expected callable")
    
    def wrapped(record: Dict[str, Any]):
        return func(record, **spec.kwargs)
    
    return _wrap_runtime_callable(wrapped, runtime_env)

def _resolve_tools(tool_specs: Optional[List[ToolConfig]], *, runtime_env: Optional["RuntimeEnvironment"] = None) -> Dict[str, Callable[..., Any]]:
    registry = {}
    for spec in tool_specs or []:
        if spec.name in registry:
            raise JobExecutionError(f"Duplicate tool name: `{spec.name}`")
        exec_ns = {}
        try:
            compiled = compile(spec.source, "<tool>", "exec")
            exec(compiled, exec_ns, exec_ns)
        except Exception as exc:
            raise JobExecutionError(f"Failed to load tool `{spec.name}`: {exc}") from exc
        func = exec_ns.get(spec.name)
        if not callable(func):
            raise JobExecutionError(f"Tool `{spec.name}` did not define a callable named {spec.name}")

        def _make_wrapper(f: Callable[..., Any], kwargs: Dict[str, Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **inner_kwargs: Any) -> Any:
                bound = dict(kwargs)
                bound.update(inner_kwargs)
                return f(*args, **bound)
            return _wrap_runtime_callable(wrapper, runtime_env)
        registry[spec.name] = _make_wrapper(func, dict(spec.kwargs or {}))
    return registry

def _wrap_runtime_callable(
    func: Callable[..., Any],
    runtime_env: Optional["RuntimeEnvironment"],
) -> Callable[..., Any]:
    if runtime_env is None:
        return func

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with runtime_env.activate():
            return func(*args, **kwargs)

    return wrapped

def _process_batch(
    prompts: Iterable[str], generations: Iterable[str], 
    processor: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    processed = []
    for prompt, completion in zip(prompts, generations):
        record = {"prompt": prompt, "completion": completion}
        result = processor(record)
        if result is None:
            continue
        if not isinstance(result, dict):
            raise JobExecutionError(f"Processor must return a dict or None per record")
        processed.append(result)
    return processed

def _drain_ready_batches(
    pending: Dict[int, Tuple[Future[List[Dict[str, Any]]], Dict[str, Any]]],
    next_index: int, 
    writer: JSONLBatchWriter, 
    aggregated_metrics: Dict[str, float],
    payload: Dict[str, Any], 
    metadata_path: Path, 
    *, 
    block: bool, 
    batch_started: Optional[int],
    job_id: str,
) -> int:
    while next_index in pending:
        future, batch_metrics = pending[next_index]
        if not block and not future.done():
            break
        try:
            records = future.result()
        except Exception as exc:
            raise JobExecutionError(f"Processor failed: {exc}") from exc
        writer.write_records(records)
        events.emit(
            "Batch completed.",
            code="batch.completed",
            data={
                "batch_index": next_index,
                "records_written": writer.count,
                "records_in_batch": len(records),
            }
        )

        for key, value in batch_metrics.items():
            if isinstance(value, (int, float)):
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + float(value)
        
        payload["metrics"] = _summarize_metrics(writer.count, aggregated_metrics)
        _write_metadata(metadata_path, payload, writer.count)

        del pending[next_index]
        next_index += 1
    return next_index

def _processing_worker_count() -> int:
    cpu_total = os.cpu_count() or 1
    return max(1, min(4, cpu_total // 2 or 1))

def _immediate_future(result: Iterable[Dict[str, Any]]) -> Future[List[Dict[str, Any]]]:
    future = Future()
    future.set_result(list(result))
    return future

def _text_batch_worker(
    *,
    prompts: List[str],
    backend: Any,
    job: JobConfig,
    post_processor: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    generations = backend.generate(prompts, parameters=job.generation.parameters)

    if len(generations) != len(prompts):
        raise JobExecutionError(
            f"Generation count mismatch within batch: expected {len(prompts)}, received {len(generations)}"
        )

    if post_processor:
        records = _process_batch(prompts, generations, post_processor)
    else:
        records = _pair_records(prompts, generations)

    return records

def _tool_batch_worker(
    *,
    job_id: str,
    prompts: List[str],
    backend: Any,
    job: JobConfig,
    post_processor: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]],
    tool_registry: Dict[str, Callable[..., Any]],
    include_logprobs: bool = False,
) -> List[Dict[str, Any]]:
    tool_defs = _tool_descriptors(job.tools)

    turn_limit = max(1, job.generation.max_turns or 16)
    max_concurrency = max(1, job.generation.max_batch_size or 4)
    max_concurrency = min(max_concurrency, len(prompts))

    executor = ThreadPoolExecutor(max_workers=max_concurrency)

    def run_prompt(prompt):
        try:
            transcript, token_ids, logprobs, reward_mask = _run_prompt_with_tools(
                backend=backend,
                job=job,
                prompt=prompt,
                tool_defs=tool_defs,
                tool_registry=tool_registry,
                turn_limit=turn_limit,
                job_id=job_id,
                include_logprobs=include_logprobs,
            )
        except Exception as exc:
            logger.exception("Job %s: prompt worker crashed while handling `%s`.", job_id, prompt[:20])
            events.emit_for_job(
                job_id,
                "Tool prompt worker crashed.",
                level="error",
                code="batch.worker_failed",
                data={"error": str(exc)}
            )
            raise
        record = {"prompt": prompt, "completion": transcript}
        if include_logprobs:
            record = {
                "prompt": prompt,
                "completion": transcript,
                "token_ids": token_ids,
                "logprobs": logprobs,
                "reward_mask": reward_mask,
            }

        if post_processor:
            processed = post_processor(record)
            if processed is None:
                return {}
            if not isinstance(processed, dict):
                raise JobExecutionError("Post-processor must return a dict (can be empty) per record")
            record = processed
        return record

    futures = []
    for prompt in prompts:
        futures.append(executor.submit(run_prompt, prompt))

    def gather_results():
        try:
            ordered_records = [None] * len(futures)
            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_EXCEPTION
                )
                for future in done:
                    idx = futures.index(future)
                    ordered_records[idx] = future.result()
            return [record for record in ordered_records if record]
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

    aggregate = Future()

    def finalize():
        try:
            aggregate.set_result(gather_results())
        except Exception as exc:
            aggregate.set_exception(exc)

    threading.Thread(target=finalize, daemon=True).start()
    return aggregate

def _tool_descriptors(tool_specs: Optional[List[ToolConfig]]) -> List[Dict[str, Any]]:
    descriptors = []
    for spec in tool_specs or []:
        descriptors.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.json_schema,
                }
            }
        )
    return descriptors

def _serialize_tool_output(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list, tuple, int, float, bool)):
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            pass
    return str(result)

def _run_prompt_with_tools(
    *,
    backend: Any,
    job: JobConfig,
    prompt: str,
    tool_defs: List[Dict[str, Any]],
    tool_registry: Dict[str, Callable[..., Any]],
    turn_limit: int,
    job_id: str,
    include_logprobs: bool = False,
) -> Tuple[List[Dict[str, Any]], List[int], List[float], List[int]]:
    history = _initial_chat_history(prompt, job)
    transcript = []

    all_token_ids = []
    all_logprobs = []
    all_reward_masks = []
    
    prompt_preview = prompt.replace("\n", "\\n")
    prompt_preview = prompt_preview[:40] + ("..." if len(prompt_preview) > 80 else "")
    logger.info("Job %s: Tool-runner invoked for prompt `%s`", job_id, prompt_preview)

    for turn_idx in range(turn_limit):
        logger.info(
            "Job %s: starting turn %d for prompt `%s`",
            job_id,
            turn_idx,
            prompt_preview
        )
        # TODO: only return full logprob for the last turn
        assistant_message = _call_backend_chat(
            backend=backend,
            messages=history,
            tools=tool_defs,
            parameters=job.generation.parameters,
            include_logprobs=include_logprobs,
        )
        logger.info(
            "Job %s: assistant turn %d for prompt `%s` returned keys=%s tool_calls=%s",
            job_id,
            turn_idx,
            prompt_preview,
            sorted(list(assistant_message.keys())),
            bool(assistant_message.get("tool_calls")) 
        )
        snapshot = {
            "role": "assistant",
            "content": assistant_message.get("content", ""),
            "tool_calls": assistant_message.get("tool_calls"),
        }

        token_ids = assistant_message.get("token_ids") or []
        logprobs = assistant_message.get("logprobs") or []
        reward_mask = assistant_message.get("reward_mask") or []
        if include_logprobs:
            all_token_ids.extend(token_ids)
            all_logprobs.extend(logprobs)
            all_reward_masks.extend(reward_mask)

        transcript.append(snapshot)
        history.append(snapshot)
        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            logger.info(
                "Job %s: trajectory finished for prompt `%s'",
                job_id,
                prompt_preview,
            )
            return transcript, all_token_ids, all_logprobs, all_reward_masks
        
        for call in tool_calls:
            function_call = call.get("function") or {}
            tool_name = function_call.get("name")
            if not tool_name or tool_name not in tool_registry:
                warning = (
                    f"Tool `{tool_name or 'unknown'}` is not available. "
                    "Please choose one of the provided tools."
                )
                logger.warning(
                    "Job %s: assistant requested unknown tool `%s` for prompt `%s`",
                    job_id,
                    tool_name,
                    prompt_preview,
                )
                tool_message = {
                    "role": "tool",
                    "name": tool_name or "unknown_tool",
                    "content": warning,
                    "tool_call_id": call.get("id"),
                }
                transcript.append(tool_message)
                history.append(dict(tool_message))
                continue

            raw_args = function_call.get("arguments") or "{}"
            turn_index = len(transcript) - 1
            logger.info(
                "Job %s: invoking tool `%s` (turn=%d) args=%s for prompt '%s'",
                job_id,
                tool_name,
                turn_index,
                raw_args,
                prompt_preview
            )
            try:
                tool_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (TypeError, ValueError) as exc:
                raise JobExecutionError(f"Tool `{tool_name}` argument must be valid JSON")

            success = True
            try:
                result = tool_registry[tool_name](**tool_args)
            except Exception as exc:
                success = False
                result = {"error": str(exc)}
            payload = _serialize_tool_output(result)
            tool_message = {
                "role": "tool",
                "name": tool_name,
                "content": payload,
                "tool_call_id": call.get("id"),
            }
            transcript.append(tool_message)
            history.append(dict(tool_message))
            logger.info(
                "Job %s: Tool %s invoked. Success: %s for prompt `%s`", 
                job_id,
                tool_name, 
                success,
                prompt_preview,
            )

    raise JobExecutionError(f"Tool-enabled generation exceeded {turn_limit} turns without reaching a final response.")

def _prepare_runtime_env(
    job: JobConfig,
    job_env: Dict[str, str],
) -> tuple[Optional["RuntimeEnvironment"], ExitStack, Dict[str, str]]:
    runtime_env = None
    runtime_stack = ExitStack()
    merged_env = dict(job_env or {})

    if job.runtime:
        runtime_env = RuntimeEnvironment()
        try:
            runtime_env.create()
            runtime_env.install(job.runtime.packages)
        except RuntimeEnvironmentError as exc:
            raise JobExecutionError(f"Failed to prepare runtime environment: {exc}") from exc

        merged_env = dict(job_env or {})
        merged_env.update(job.runtime.env)
        runtime_stack.enter_context(runtime_env.activate(merged_env))
        events.emit(
            "Runtime sandbox prepared.",
            code="runtime.ready",
        )
        logger.info("Job %s: runtime sandbox prepared and activated.", job.metadata.get("job_id", ""))

    return runtime_env, runtime_stack, merged_env

def _prepare_processors_and_prompts(
    *,
    job: JobConfig,
    runtime_env: Optional["RuntimeEnvironment"],
    pre_processor: Optional[ProcessorConfig],
    post_processor: Optional[ProcessorConfig],
) -> tuple[Optional[Callable[[Dict[str, Any]], Any]], List[str]]:
    pre_processor = _resolve_processor(job.pre_processor, runtime_env=runtime_env) if pre_processor else None
    post_processor = _resolve_processor(job.post_processor, runtime_env=runtime_env) if post_processor else None
    prompts = collect_prompts(job.source, pre_processor=pre_processor)

    return pre_processor, post_processor, prompts

def _call_backend_chat(
    *,
    backend: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    parameters: Dict[str, Any],
    include_logprobs: bool = False,
) -> Dict[str, Any]:
    if _is_tinker_backend(backend):
        return _tinker_chat_and_logprobs(
            sampling_client=backend,
            messages=messages,
            tools=tools,
            parameters=parameters,
            include_logprobs=include_logprobs
        )

    chat_fn = getattr(backend, "chat", None)
    if not callable(chat_fn):
        raise JobExecutionError("Backend does not support chat tool calls.")
    logger.info(
        "Dispatching chat calls with %d message(s), tools=%s",
        len(messages),
        bool(tools)
    )
    
    params = dict(parameters or {})

    try:
        response = chat_fn(
            messages=messages,
            tools=tools or None,
            parameters=params,
        )
        logger.info(
            "Chat call returned successfully (tool_calls=%s)",
            bool(response.get("tool_calls"))
        )
        return response
    except Exception as exc:
        logger.exception("Chat backend failed: %s", exc)
        raise JobExecutionError(f"Chat backend failed: {exc}") from exc

def _tinker_chat_and_logprobs(
    *,
    sampling_client: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    parameters: Dict[str, Any],
    include_logprobs: bool,
) -> Dict[str, Any]:
    import tinker
    from tinker_cookbook import renderers, model_info
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    base_model = getattr(sampling_client, "base_model", None) or getattr(sampling_client, "model_id", None)
    renderer_name = model_info.get_recommended_renderer_name(base_model or "")
    tokenizer = get_tokenizer(base_model or "")
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    convo = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            convo.append({"role": "system", "content": content})
        elif role == "user":
            convo.append({"role": "user", "content": content})
        elif role == "assistant":
            convo.append({"role": "assistant", "content": content})
        elif role == "tool":
            convo.append({"role": "assistant", "content": content}) # TODO: verify role is tool vs assistant
    prompt_text = renderer.build_generation_prompt(convo)

    sampling_params = parameters.copy()
    max_tokens = sampling_params.pop("max_tokens", None)
    temperature = sampling_params.pop("temperature", None)
    top_p = sampling_params.pop("top_p", None)
    stop = renderer.get_stop_sequences()

    sample_resp = sampling_client.sample(
        prompt=prompt_text,
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=max_tokens or 1024,
            temperature=temperature if temperature is not None else 0.7,
            top_p=top_p if top_p is not None else 0.95,
            stop=stop,
        )
    )
    seq = sample_resp.sequences[0]
    completion_tokens = seq.tokens
    assistant_text, _ = renderer.parse_response(completion_tokens)

    result = {
        "role": "assistant",
        "content": assistant_text["content"],
        "tool_calls": None, # TODO: need to support tool calls
        "token_ids": [],
        "logprobs": [],
        "reward_masks": [], 
    }

    if include_logprobs:
        prompt_tokens = tokenizer.encode(prompt_text)
        full_tokens = prompt_tokens + completion_tokens
        lp_resp = sampling_client.compute_logprobs(
            prompt=prompt_text,
            tokens=full_tokens,
            include_prompt_logprobs=True, # TODO: skip this to be faster (does it matter?)
        )
        result["token_ids"] = full_tokens
        result["logprobs"] = lp_resp.logprobs
        result["reward_masks"] = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

    return result

def _initial_chat_history(prompt: str, job: JobConfig) -> List[Dict[str, Any]]:
    history = []
    system_prompt = job.model.parameters.get("system_prompt")
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})
    history.append({"role": "user", "content": prompt})
    return history

def _resolve_batch_size(job: JobConfig, prompts: List[str]) -> int:
    requested = job.generation.max_batch_size
    if isinstance(requested, int) and requested > 0:
        return max(1, min(requested, len(prompts) or 1))
    base = 4
    gpu_factor = job.model.parameters.get("tensor_parallel_size", 1)
    if not isinstance(gpu_factor, int) or gpu_factor < 1:
        gpu_factor = 1
    inferred = base * gpu_factor
    return max(1, min(inferred, len(prompts) or 1))

def _ensure_tensor_parallel(job: JobConfig) -> None:
    params = job.model.parameters
    if params.get("tensor_parallel_size"):
        return
    params["tensor_parallel_size"] = _detect_available_gpus()

def _detect_available_gpus() -> int:
    try:
        import torch
        count = torch.cuda.device_count()
        if count > 0:
            return count
    except Exception:
        return 1
    return 1

def _processor_snapshot(spec: ProcessorConfig) -> Dict[str, Any]:
    return {"name": spec.name, "kwargs": dict(spec.kwargs or {})}

def _sanitize_generation_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    options = payload.get("on_policy_options")
    if isinstance(options, dict):
        options.pop("api_key", None)
    return payload

def _sanitize_output_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    hf_config = payload.get("hf")
    if isinstance(hf_config, dict):
        payload["hf"] = dict(hf_config)
    return payload

def _job_snapshot(job: JobConfig) -> Dict[str, Any]:
    snapshot = {
        "model": job.model.model_dump(exclude_none=True),
        "source": job.source.model_dump(exclude_none=True),
        "generation": _sanitize_generation_config(job.generation.model_dump(exclude_none=True)),
        "output": _sanitize_output_config(job.output.model_dump(exclude_none=True)),
        "metadata": dict(job.metadata)
    }
    if job.runtime:
        snapshot["runtime"] = job.runtime.model_dump(exclude_none=True)
    processors = {}
    if job.pre_processor:
        processors["pre_processor"] = _processor_snapshot(job.pre_processor)
    if job.post_processor:
        processors["post_processor"] = _processor_snapshot(job.post_processor)
    if processors:
        snapshot["processors"] = processors
    if job.tools:
        snapshot["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "json_schema": tool.json_schema,
                "kwargs": dict(tool.kwargs or {})
            }
            for tool in job.tools
        ]
    return snapshot

def _base_metadata(job_id: str, job: JobConfig) -> Dict[str, Any]:
    snapshot = _job_snapshot(job)
    payload = {
        "job_id": job_id,
        "model": snapshot["model"],
        "source": snapshot["source"],
        "generation": snapshot["generation"],
        "output": snapshot["output"],
        "metadata": snapshot["metadata"],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "records": 0,
        "metrics": {"records": 0}
    }
    processors = snapshot.get("processors")
    if processors:
        payload["processors"] = processors
    tool_snapshot = snapshot.get("tools")
    if tool_snapshot:
        payload["tools"] = tool_snapshot
    return payload

def _write_metadata(path: Path, payload: Dict[str, Any], records: int) -> None:
    payload["records"] = records
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _summarize_metrics(records: int, aggregated: Dict[str, float]) -> Dict[str, Any]:
    metrics = {"records": records}
    for key, value in aggregated.items():
        metrics[key] = int(value) if float(value).is_integer() else value
    return metrics

def _shutdown_backend(job_id: str, backend: Any) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.warning("Job %s: backend shutdown raised %s", job_id, exc)

def _is_tinker_backend(backend: Any) -> bool:
    import tinker
    return isinstance(backend, tinker.SamplingClient)