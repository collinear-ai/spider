"""OpenHands scaffold wrapper for SWE trajectory generation.

This module provides a wrapper around OpenHands for generating SFT trajectories
from SWE task datasets like SWE-bench/SWE-smith.
"""

import asyncio
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from datasets import load_dataset
from pydantic import BaseModel, Field

try:
    import sys
    
    import openhands.agenthub  # noqa F401
    from openhands.controller.state.state import State
    from openhands.core.config import (
        AgentConfig,
        OpenHandsConfig,
        get_agent_config_arg,
        get_evaluation_parser,
        get_llm_config_arg,
    )
    from openhands.core.config.condenser_config import NoOpCondenserConfig
    from openhands.core.logger import openhands_logger as logger
    from openhands.core.main import create_runtime, run_controller
    from openhands.events.action import CmdRunAction, MessageAction
    from openhands.events.observation import CmdOutputObservation, ErrorObservation
    from openhands.events.serialization.event import event_to_dict
    from openhands.runtime.base import Runtime
    from openhands.utils.async_utils import call_async_from_sync
    from openhands.utils.shutdown_listener import sleep_if_should_continue
    
    # Try to import evaluation utils - they're in the OpenHands repo structure
    # We need to add the OpenHands evaluation directory to the path
    try:
        # Try direct import first (if installed as package with evaluation)
        from evaluation.utils.shared import (
            EvalException,
            EvalMetadata,
            EvalOutput,
            assert_and_raise,
            codeact_user_response,
            get_default_sandbox_config_for_eval,
            get_metrics,
            get_openhands_config_for_eval,
            is_fatal_evaluation_error,
            make_metadata,
            prepare_dataset,
            reset_logger_for_multiprocessing,
            run_evaluation,
            update_llm_config_for_completions_logging,
        )
    except ImportError:
        # Try to find OpenHands repo and add evaluation to path
        import openhands
        openhands_path = Path(openhands.__file__).parent.parent
        eval_path = openhands_path / "evaluation"
        if eval_path.exists():
            if str(eval_path) not in sys.path:
                sys.path.insert(0, str(eval_path))
            from utils.shared import (
                EvalException,
                EvalMetadata,
                EvalOutput,
                assert_and_raise,
                codeact_user_response,
                get_default_sandbox_config_for_eval,
                get_metrics,
                get_openhands_config_for_eval,
                is_fatal_evaluation_error,
                make_metadata,
                prepare_dataset,
                reset_logger_for_multiprocessing,
                run_evaluation,
                update_llm_config_for_completions_logging,
            )
        else:
            raise ImportError(
                f"OpenHands evaluation utilities not found. "
                f"Tried: {eval_path}. "
                f"Make sure OpenHands is installed from source with evaluation directory, "
                f"or set OPENHANDS_EVAL_PATH environment variable."
            )
    
    OPENHANDS_AVAILABLE = True
except ImportError as e:
    OPENHANDS_AVAILABLE = False
    IMPORT_ERROR = str(e)

from server.scaffolds.base import Scaffold, ScaffoldConfig


class OpenHandsScaffoldConfig(ScaffoldConfig):
    """Configuration for OpenHands scaffold."""
    
    dataset: str = Field(
        default="SWE-bench/SWE-smith",
        description="HuggingFace dataset name"
    )
    split: str = Field(default="train", description="Dataset split")
    instance_filter: Optional[str] = Field(
        default=None,
        description="Optional regex filter for instance IDs"
    )
    
    agent_class: str = Field(default="CodeActAgent", description="OpenHands agent class")
    max_iterations: int = Field(default=50, description="Maximum iterations per task")
    llm_config_name: Optional[str] = Field(
        default=None,
        description="LLM config name (from OpenHands config.toml) or model string"
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="LLM model name (e.g., 'anthropic/claude-sonnet-4', 'gpt-4o')"
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for LLM provider"
    )
    llm_api_key_env: Optional[str] = Field(
        default=None,
        description="Environment variable name to read LLM API key from (takes precedence if set)"
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for LLM API (e.g., 'https://api.fireworks.ai/inference/v1' for Fireworks, 'http://localhost:8000/v1' for vLLM)"
    )
    enable_browser: bool = Field(default=False, description="Enable browser tools")
    enable_llm_editor: bool = Field(default=False, description="Enable LLM editor")
    runtime: str = Field(default="docker", description="Runtime type (docker/local/etc)")
    platform: str = Field(default="linux/amd64", description="Docker platform")
    remote_runtime_resource_factor: int = Field(
        default=1,
        description="Resource factor for remote runtime (1, 2, 4, or 8)"
    )
    runtime_container_image: Optional[str] = Field(
        default=None,
        description="Pre-built runtime container image to use (skips building). If None, builds from base_container_image."
    )
    workspace_dir: Optional[str] = Field(
        default=None,
        description="Working directory in the container (e.g., 'testbed' for SWE-smith, 'workspace/{repo}' for SWE-bench). If None, auto-detects."
    )
    base_container_image_override: Optional[str] = Field(
        default=None,
        description="Override base container image (ignores dataset's image_name). Use this to avoid per-task image rebuilds. E.g., 'nikolaik/python-nodejs:python3.12-nodejs22'"
    )
    eval_output_dir: Optional[Path] = Field(
        default=None,
        description="Output directory for evaluation results (defaults to output_dir)"
    )
    
    # HuggingFace upload configuration
    hf_repo_id: Optional[str] = Field(
        default=None,
        description="HuggingFace repo ID to upload trajectories to (e.g., 'username/dataset-name'). If None, no upload."
    )
    hf_private: bool = Field(
        default=True,
        description="Whether to make the HuggingFace repo private"
    )
    hf_config_name: Optional[str] = Field(
        default=None,
        description="Subset config name for HuggingFace dataset (e.g., 'gpt-4o_swe-smith_50iter'). If None, auto-generated from config."
    )


def _get_workspace_dir_name(instance: pd.Series) -> str:
    """Get workspace directory name from instance.
    
    Handles both SWE-bench format (repo__version) and SWE-smith format (swesmith/repo).
    For SWE-smith, parses the repo name from the 'repo' field.
    """
    if 'repo' in instance and 'version' in instance:
        # SWE-bench format
        return f'{instance.repo}__{instance.version}'.replace('/', '__')
    elif 'repo' in instance:
        # SWE-smith format: swesmith/oauthlib__oauthlib.1fd52536
        repo = instance.repo
        if repo.startswith('swesmith/'):
            repo = repo[len('swesmith/'):]
        return repo.replace('/', '__')
    else:
        # Fallback to instance_id
        return instance.instance_id.replace('/', '__').replace('__', '_')


def _get_instruction(instance: pd.Series) -> str:
    """Generate instruction for SWE task instance."""
    workspace_dir_name = _get_workspace_dir_name(instance)
    
    instruction = (
        '<uploaded_files>\n'
        f'/workspace/{workspace_dir_name}\n'
        '</uploaded_files>\n'
        f"I've uploaded a code repository in the directory {workspace_dir_name}. "
        "Consider the following issue description:\n\n"
        f'<issue_description>\n'
        f'{instance.problem_statement}\n'
        '</issue_description>\n\n'
        'Can you help me implement the necessary changes to the repository so that '
        'the requirements specified in the <issue_description> are met?\n'
        "I've already taken care of all changes to any of the test files described "
        "in the <issue_description>. This means you DON'T have to modify the testing "
        "logic or any of the tests in any way!\n"
        "Also the development environment is already set up for you (i.e., all "
        "dependencies already installed), so you don't need to install other packages.\n"
        'Your task is to make the minimal changes to non-test files in the /workspace '
        'directory to ensure the <issue_description> is satisfied.\n'
        'Follow these steps to resolve the issue:\n'
        '1. As a first step, it might be a good idea to explore the repo to familiarize '
        'yourself with its structure.\n'
        '2. Create a script to reproduce the error and execute it to confirm the error\n'
        '3. Edit the sourcecode of the repo to resolve the issue\n'
        '4. Rerun your reproduce script and confirm that the error is fixed!\n'
        '5. Think about edgecases, add comprehensive tests for them in your reproduce '
        'script, and run them to make sure your fix handles them as well\n'
        "Your thinking should be thorough and so it's fine if it's very long.\n"
    )
    
    return instruction


def _get_instance_docker_image(instance: pd.Series, base_image_override: Optional[str] = None) -> str:
    """Get Docker image name from instance.
    
    Args:
        instance: Dataset instance
        base_image_override: If provided, use this instead of dataset's image_name
    
    Returns:
        Docker image name to use as base
    """
    # Use override if provided (allows using generic image for all tasks)
    if base_image_override:
        return base_image_override
    
    if 'image_name' in instance and instance.get('image_name'):
        return str(instance['image_name']).lower()
    
    # Fallback: generate from instance_id (SWE-bench style)
    instance_id = instance['instance_id']
    if '__' in instance_id:
        repo, name = instance_id.split('__', 1)
        # SWE-bench official format
        image_name = f'swebench/sweb.eval.x86_64.{repo}_1776_{name}:latest'.lower()
    else:
        # Generic fallback
        image_name = f'swebench/sweb.eval.x86_64.{instance_id}:latest'.lower()
    
    return image_name


def _initialize_runtime(runtime: Runtime, instance: pd.Series):
    """Initialize runtime for SWE task instance."""
    workspace_dir_name = _get_workspace_dir_name(instance)
    
    # Set instance ID and git config
    action = CmdRunAction(
        command=(
            f"echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && "
            "echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && "
            "echo \"alias git='git --no-pager'\" >> ~/.bashrc && "
            'git config --global core.pager "" && '
            'git config --global diff.binary false'
        )
    )
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to export SWE_INSTANCE_ID and configure git: {str(obs)}',
    )
    
    # Set USER
    action = CmdRunAction(command="export USER=$(whoami); echo USER=${USER}")
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    assert_and_raise(obs.exit_code == 0, f'Failed to export USER: {str(obs)}')
    
    # For SWE-smith: copy /testbed to /workspace/{workspace_dir_name}
    if 'image_name' in instance and instance.get('image_name'):
        logger.info(f'Copying /testbed to /workspace/{workspace_dir_name} for SWE-smith dataset')
        action = CmdRunAction(command=f'mkdir -p /workspace && cp -r /testbed /workspace/{workspace_dir_name}')
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to copy /testbed to /workspace/{workspace_dir_name}: {str(obs)}',
        )
    
    # Navigate to workspace
    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )
    
    # Reset git state
    action = CmdRunAction(command='git reset --hard')
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    assert_and_raise(obs.exit_code == 0, f'Failed to git reset --hard: {str(obs)}')


def _complete_runtime(runtime: Runtime, instance: pd.Series) -> Dict[str, Any]:
    """Complete runtime and extract git patch."""
    workspace_dir_name = _get_workspace_dir_name(instance)
    
    # Navigate to workspace
    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    
    if obs.exit_code == -1:
        # Previous command still running, kill it
        action = CmdRunAction(command='C-c')
        runtime.run_action(action)
        action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)
    
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )
    
    # Configure git
    action = CmdRunAction(command='git config --global core.pager ""')
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to configure git: {str(obs)}',
    )
    
    # Remove nested git repos
    action = CmdRunAction(command='find . -type d -name .git -not -path "./.git"')
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
        git_dirs = [p for p in obs.content.strip().split('\n') if p]
        for git_dir in git_dirs:
            action = CmdRunAction(command=f'rm -rf "{git_dir}"')
            action.set_hard_timeout(600)
            runtime.run_action(action)
    
    # Get git diff
    action = CmdRunAction(command='git add -A')
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git add -A: {str(obs)}',
    )
    
    # Get diff from HEAD (works for both SWE-bench and SWE-smith)
    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(command='git diff --no-color --cached HEAD')
        action.set_hard_timeout(max(300 + 100 * n_retries, 600))
        obs = runtime.run_action(action)
        n_retries += 1
        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                git_patch = obs.content.strip()
                break
            else:
                logger.info('Failed to get git diff, retrying...')
                sleep_if_should_continue(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error occurred: {obs.content}. Retrying...')
            sleep_if_should_continue(10)
    
    assert_and_raise(git_patch is not None, 'Failed to get git diff')
    
    return {'git_patch': git_patch}


class OpenHandsScaffold(Scaffold):
    """OpenHands scaffold for generating SWE trajectories."""
    
    def __init__(self, config: OpenHandsScaffoldConfig):
        if not OPENHANDS_AVAILABLE:
            raise ImportError(
                f"OpenHands is not available. Install with: pip install openhands-ai\n"
                f"Original error: {IMPORT_ERROR}"
            )
        super().__init__(config)
        self.oh_config = config
        
        # Set eval_output_dir if not provided
        if self.oh_config.eval_output_dir is None:
            self.oh_config.eval_output_dir = self.oh_config.output_dir
    
    def _create_llm_config(self):
        """Create LLM config from scaffold config."""
        # Resolve API key: explicit key > env var override > None
        resolved_api_key = self.oh_config.llm_api_key
        if not resolved_api_key and self.oh_config.llm_api_key_env:
            resolved_api_key = os.getenv(self.oh_config.llm_api_key_env)

        if self.oh_config.llm_config_name:
            # Use OpenHands config system
            llm_config = get_llm_config_arg(self.oh_config.llm_config_name)
            # If base_url override provided in YAML, apply it
            if self.oh_config.llm_base_url:
                llm_config.base_url = self.oh_config.llm_base_url
            # If API key override provided, apply it
            if resolved_api_key:
                llm_config.api_key = resolved_api_key
        elif self.oh_config.llm_model:
            # Create from model string
            from openhands.core.config import LLMConfig
            llm_config = LLMConfig(
                model=self.oh_config.llm_model,
                api_key=resolved_api_key,
                base_url=self.oh_config.llm_base_url,
            )
        else:
            raise ValueError(
                "Either llm_config_name or llm_model must be provided"
            )
        
        llm_config.log_completions = True
        llm_config.modify_params = False  # For reproducibility
        
        return llm_config
    
    def _create_metadata(self, dataset_name: str, split: str) -> EvalMetadata:
        """Create evaluation metadata."""
        llm_config = self._create_llm_config()
        
        dataset_description = dataset_name.replace('/', '__') + '-' + split.replace('/', '__')
        
        agent_config = None
        if self.oh_config.agent_class:
            agent_config = get_agent_config_arg(self.oh_config.agent_class)
        
        metadata = make_metadata(
            llm_config=llm_config,
            dataset_name=dataset_description,
            agent_class=self.oh_config.agent_class,
            max_iterations=self.oh_config.max_iterations,
            eval_note=None,
            eval_output_dir=str(self.oh_config.eval_output_dir),
            data_split=split,
            details={},
            agent_config=agent_config,
            condenser_config=NoOpCondenserConfig(),
        )
        
        return metadata
    
    def _get_config(self, instance: pd.Series, metadata: EvalMetadata) -> OpenHandsConfig:
        """Get OpenHands config for an instance."""
        base_container_image = _get_instance_docker_image(
            instance, 
            self.oh_config.base_container_image_override
        )
        
        if self.oh_config.base_container_image_override:
            logger.info(
                f'Using override base image: {base_container_image} '
                f'(ignoring dataset image_name)'
            )
        else:
            logger.info(
                f'Using instance container image: {base_container_image}. '
                f'Please make sure this image exists.'
            )
        
        sandbox_config = get_default_sandbox_config_for_eval()
        sandbox_config.base_container_image = base_container_image
        sandbox_config.enable_auto_lint = True
        sandbox_config.use_host_network = False
        sandbox_config.platform = self.oh_config.platform
        sandbox_config.remote_runtime_resource_factor = self.oh_config.remote_runtime_resource_factor
        
        # Use pre-built runtime image if specified (skips slow build process)
        if self.oh_config.runtime_container_image:
            sandbox_config.runtime_container_image = self.oh_config.runtime_container_image
            logger.info(f'Using pre-built runtime image: {self.oh_config.runtime_container_image}')
        
        config = get_openhands_config_for_eval(
            metadata=metadata,
            enable_browser=self.oh_config.enable_browser,
            runtime=self.oh_config.runtime,
            sandbox_config=sandbox_config,
        )
        
        config.set_llm_config(
            update_llm_config_for_completions_logging(
                metadata.llm_config,
                metadata.eval_output_dir,
                instance['instance_id']
            )
        )
        
        agent_config = AgentConfig(
            enable_jupyter=False,
            enable_browsing=self.oh_config.enable_browser,
            enable_llm_editor=self.oh_config.enable_llm_editor,
            enable_mcp=False,
            condenser=metadata.condenser_config,
            enable_prompt_extensions=False,
        )
        config.set_agent_config(agent_config)
        
        return config
    
    def _process_instance(
        self,
        instance: pd.Series,
        metadata: EvalMetadata,
        reset_logger: bool = True,
        runtime_failure_count: int = 0,
    ) -> EvalOutput:
        """Process a single instance."""
        config = self._get_config(instance, metadata)
        
        # Setup logger
        if reset_logger:
            log_dir = Path(metadata.eval_output_dir) / 'infer_logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            reset_logger_for_multiprocessing(
                logger, instance.instance_id, str(log_dir)
            )
        else:
            logger.info(f'Starting evaluation for instance {instance.instance_id}.')
        
        # Increase resource factor on retries
        if runtime_failure_count > 0:
            config.sandbox.remote_runtime_resource_factor = min(
                config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
                8,
            )
        
        metadata = copy.deepcopy(metadata)
        metadata.details['runtime_failure_count'] = runtime_failure_count
        
        runtime = create_runtime(config)
        call_async_from_sync(runtime.connect)
        
        try:
            _initialize_runtime(runtime, instance)
            
            instruction = _get_instruction(instance)
            message_action = MessageAction(content=instruction)
            
            # Run agent
            state: State | None = asyncio.run(
                run_controller(
                    config=config,
                    initial_user_action=message_action,
                    runtime=runtime,
                    fake_user_response_fn=codeact_user_response,
                )
            )
            
            # Check for fatal errors
            if is_fatal_evaluation_error(state.last_error if state else None):
                raise EvalException('Fatal error detected: ' + (state.last_error if state else 'Unknown'))
            
            # Get git patch
            return_val = _complete_runtime(runtime, instance)
            git_patch = return_val['git_patch']
            logger.info(
                f'Got git diff for instance {instance.instance_id}:\n'
                f'--------\n{git_patch}\n--------'
            )
        finally:
            runtime.close()
        
        # Build output
        test_result = {'git_patch': git_patch}
        
        if state is None:
            raise ValueError('State should not be None.')
        
        histories = [event_to_dict(event) for event in state.history]
        metrics = get_metrics(state)
        
        output = EvalOutput(
            instance_id=instance.instance_id,
            instruction=instruction,
            instance=instance.to_dict(),
            test_result=test_result,
            metadata=metadata,
            history=histories,
            metrics=metrics,
            error=state.last_error if state and state.last_error else None,
        )
        
        return output
    
    def run_batch(
        self,
        dataset_name: Optional[str] = None,
        split: Optional[str] = None,
        instance_filter: Optional[str] = None,
    ) -> Path:
        """Run OpenHands on a dataset and generate trajectories.
        
        Args:
            dataset_name: HuggingFace dataset name. If None, uses config default.
            split: Dataset split. If None, uses config default.
            instance_filter: Optional regex filter for instance IDs.
        """
        # Use config defaults if not provided
        if dataset_name is None:
            dataset_name = self.oh_config.dataset
        if split is None:
            split = self.oh_config.split
        
        # Load dataset
        logger.info(f'Loading dataset {dataset_name} split {split}')
        dataset = load_dataset(dataset_name, split=split)
        df = dataset.to_pandas()
        
        # Filter by instance_id if provided
        if instance_filter is None:
            instance_filter = self.oh_config.instance_filter
        if instance_filter:
            import re
            df = df[df['instance_id'].str.match(instance_filter, na=False)]
            logger.info(f'Filtered to {len(df)} instances matching {instance_filter}')
        
        # Create metadata
        metadata = self._create_metadata(dataset_name, split)
        
        # Prepare output file (use metadata.eval_output_dir which has full path)
        output_file = Path(metadata.eval_output_dir) / 'output.jsonl'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f'Output file: {output_file}')
        
        # Prepare dataset
        instances = prepare_dataset(
            df,
            str(output_file),
            self.oh_config.max_instances,
        )
        
        # Convert PASS_TO_PASS and FAIL_TO_PASS to strings if needed
        if len(instances) > 0:
            for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
                if col in instances.columns:
                    if not isinstance(instances[col].iloc[0], str):
                        instances[col] = instances[col].apply(lambda x: str(x) if x is not None else '[]')
        
        # Run evaluation
        run_evaluation(
            instances,
            metadata,
            str(output_file),
            self.oh_config.num_workers,
            self._process_instance,
            timeout_seconds=self.oh_config.timeout_seconds,
            max_retries=self.oh_config.max_retries,
        )
        
        # Upload to HuggingFace if configured
        if self.oh_config.hf_repo_id:
            hf_url = self._upload_to_huggingface(output_file, metadata)
            logger.info(f'Trajectories uploaded to HuggingFace: {hf_url}')
        
        return output_file
    
    def _clean_dict_for_parquet(self, d: Any) -> Any:
        """Recursively clean dict for Parquet compatibility (remove empty dicts)."""
        if not isinstance(d, dict):
            return d
        
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, dict):
                if len(v) == 0:
                    # Skip empty dicts (Parquet can't handle them)
                    continue
                else:
                    cleaned[k] = self._clean_dict_for_parquet(v)
            elif isinstance(v, list):
                cleaned[k] = [self._clean_dict_for_parquet(item) if isinstance(item, dict) else item for item in v]
            else:
                cleaned[k] = v
        return cleaned if cleaned else None
    
    def _trajectory_to_messages(self, trajectory: list[Dict[str, Any]]) -> list[Dict[str, str]]:
        """Convert OpenHands trajectory to OpenAI function calling format.
        
        Returns list of messages in OpenAI format with tool_calls.
        Matches SWE-Gym/OpenHands-Sampled-Trajectories format.
        """
        import json
        
        messages = []
        i = 0
        
        # Add system message first (from first event if it's a system message)
        if trajectory and trajectory[0].get('action') == 'system':
            system_content = trajectory[0].get('args', {}).get('content', '') or trajectory[0].get('message', '')
            if system_content:
                messages.append({
                    "role": "system",
                    "content": system_content,
                    "function_call": None,
                    "name": None,
                    "tool_call_id": None,
                    "tool_calls": None
                })
            i = 1  # Skip the system message
        
        while i < len(trajectory):
            turn = trajectory[i]
            source = turn.get("source", "")
            action = turn.get("action", "")
            
            # Handle user messages
            if source == "user" and action == "message":
                content = turn.get("args", {}).get("content", "") or turn.get("message", "")
                if content:
                    messages.append({
                        "role": "user",
                        "content": content,
                        "function_call": None,
                        "name": None,
                        "tool_call_id": None,
                        "tool_calls": None
                    })
                i += 1
                continue
            
            # Handle agent tool calls (actions)
            if source == "agent" and action in ["run", "run_ipython", "read", "write", "edit", "browse", "str_replace", "task_tracking"]:
                tool_call_metadata = turn.get('tool_call_metadata', {})
                tool_call_id = tool_call_metadata.get('tool_call_id', turn.get('id', f"call_{i}"))
                
                # Create tool call
                args = turn.get("args", {})
                tool_calls = [{
                    "function": {
                        "arguments": json.dumps(args),
                        "name": action
                    },
                    "id": tool_call_id,
                    "index": None,
                    "type": "function"
                }]
                
                messages.append({
                    "role": "assistant",
                    "content": turn.get("message", "") or "",
                    "function_call": None,
                    "name": None,
                    "tool_call_id": None,
                    "tool_calls": tool_calls
                })
                
                # Check for observation in next event
                if i + 1 < len(trajectory):
                    next_turn = trajectory[i + 1]
                    if 'content' in next_turn and next_turn.get('source') in ['agent', 'environment']:
                        obs_content = next_turn.get("content", "")
                        # Limit observation length
                        if len(obs_content) > 50000:
                            obs_content = obs_content[:50000] + "\n... (output truncated)"
                        
                        messages.append({
                            "role": "tool",
                            "content": f"OBSERVATION:\n{obs_content}",
                            "function_call": None,
                            "name": action,
                            "tool_call_id": tool_call_id,
                            "tool_calls": None
                        })
                        i += 1  # Skip observation
                
                i += 1
                continue
            
            # Handle agent messages (non-tool responses)
            if source == "agent" and action == "message":
                content = turn.get("args", {}).get("content", "") or turn.get("message", "")
                if content:
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "function_call": None,
                        "name": None,
                        "tool_call_id": None,
                        "tool_calls": None
                    })
                i += 1
                continue
            
            # Skip other events
            i += 1
        
        return messages
    
    def _extract_tools_from_trajectory(self, trajectory: list[Dict[str, Any]]) -> list:
        """Extract tools definition from trajectory.
        
        OpenHands includes tool definitions in the first system message.
        """
        if trajectory and trajectory[0].get('action') == 'system':
            tools = trajectory[0].get('args', {}).get('tools', [])
            if tools:
                return tools
        
        # If not in system message, return empty list
        return []
    
    def _transform_output_for_dataset(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform to OpenAI function calling format (matches SWE-Gym).
        
        Creates a dataset with 'messages' field (OpenAI format with tool_calls)
        and 'tools' field (tools definition).
        """
        # Get raw trajectory
        trajectory = record.get("history", [])
        
        # Convert to messages format (OpenAI function calling)
        messages = self._trajectory_to_messages(trajectory)
        
        # Extract tools definition
        tools = self._extract_tools_from_trajectory(trajectory)
        
        # Also clean trajectory for reference
        cleaned_trajectory = []
        for turn in trajectory:
            cleaned_turn = self._clean_dict_for_parquet(turn)
            if cleaned_turn:
                cleaned_trajectory.append(cleaned_turn)
        
        # Get resolved from test_result.report.resolved (set by evaluation step)
        # If not evaluated yet, this will be None
        resolved = None
        test_result = record.get("test_result", {})
        if test_result and "report" in test_result:
            resolved = test_result.get("report", {}).get("resolved")
        
        result = {
            "instance_id": record.get("instance_id"),
            "messages": messages,  # OpenAI format with tool_calls
            "tools": tools,  # Tools definition
            "resolved": resolved,  # True/False if evaluated, None if not
        }
        
        # Add optional metadata fields
        if record.get("instance", {}).get("repo"):
            result["repo"] = record.get("instance", {}).get("repo")
        if record.get("instance", {}).get("image_name"):
            result["image_name"] = record.get("instance", {}).get("image_name")
        if record.get("instruction"):
            result["instruction"] = record.get("instruction")
        if record.get("test_result", {}).get("git_patch"):
            result["git_patch"] = record.get("test_result", {}).get("git_patch")
        if record.get("metadata", {}).get("agent_class"):
            result["agent_class"] = record.get("metadata", {}).get("agent_class")
        if record.get("metadata", {}).get("llm_config", {}).get("model"):
            result["model"] = record.get("metadata", {}).get("llm_config", {}).get("model")
        if record.get("metadata", {}).get("max_iterations"):
            result["max_iterations"] = record.get("metadata", {}).get("max_iterations")
        if self._clean_dict_for_parquet(record.get("metrics", {})):
            result["metrics"] = self._clean_dict_for_parquet(record.get("metrics", {}))
        if record.get("error"):
            result["error"] = record.get("error")
        if record.get("instance", {}).get("problem_statement"):
            result["problem_statement"] = record.get("instance", {}).get("problem_statement")
        
        # Include raw trajectory for debugging
        result["trajectory"] = cleaned_trajectory
        
        return result
    
    def _upload_to_huggingface(self, output_file: Path, metadata: EvalMetadata) -> str:
        """Upload trajectories to HuggingFace Hub as a proper Dataset.
        
        Args:
            output_file: Path to output.jsonl file
            metadata: Evaluation metadata
            
        Returns:
            URL to the uploaded dataset
        """
        from datasets import Dataset
        import json
        import os
        
        logger.info('Preparing to upload trajectories to HuggingFace Hub...')
        
        # Check for HF token
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError('HF_TOKEN environment variable not set. Cannot upload to HuggingFace.')
        
        # Load and transform data
        logger.info(f'Loading and transforming {output_file.name}...')
        data = []
        with open(output_file, 'r') as f:
            for i, line in enumerate(f, 1):
                record = json.loads(line)
                transformed = self._transform_output_for_dataset(record)
                # Remove None values but keep empty lists
                transformed = {k: v for k, v in transformed.items() if v is not None}
                data.append(transformed)
        
        logger.info(f'  ✓ Loaded {len(data)} records')
        
        # Create HuggingFace Dataset
        logger.info('Creating HuggingFace Dataset...')
        dataset = Dataset.from_list(data)
        logger.info(f'  ✓ Created dataset with {len(dataset)} rows')
        
        # Push to Hub
        logger.info(f'Pushing to HuggingFace Hub...')
        logger.info(f'  Repo: {self.oh_config.hf_repo_id} ({"private" if self.oh_config.hf_private else "public"})')
        
        try:
            dataset.push_to_hub(
                repo_id=self.oh_config.hf_repo_id,
                private=self.oh_config.hf_private,
                token=hf_token
            )
            
            hf_url = f"https://huggingface.co/datasets/{self.oh_config.hf_repo_id}"
            logger.info(f'✓ Upload successful: {hf_url}')
            return hf_url
            
        except Exception as e:
            logger.error(f'Failed to upload to HuggingFace: {e}')
            raise
    
    def run_single(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """Run OpenHands on a single instance."""
        instance_series = pd.Series(instance)
        metadata = self._create_metadata("single_instance", "train")
        
        output = self._process_instance(instance_series, metadata, reset_logger=False)
        
        return output.model_dump()

