"""SWE-smith integration for task generation.

This module wraps SWE-smith's task generation pipeline to create bug instances
from GitHub repositories.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from spider.config import TaskGenerationConfig
from .. import events

logger = logging.getLogger(__name__)

class TaskGenerationError(Exception):
    """Error during task generation"""
    pass


class SWESmithTaskGenerator:
    """Wrapper around SWE-smith task generation pipeline"""
    
    def __init__(self, config: TaskGenerationConfig, workspace: Path):
        self.config = config
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for SWE-smith artifacts
        self.bug_gen_dir = self.workspace / "bug_gen"
        self.validation_dir = self.workspace / "run_validation"
        self.tasks_dir = self.workspace / "task_insts"
        self.issue_gen_dir = self.workspace / "issue_gen"
        
        for dir_path in [self.bug_gen_dir, self.validation_dir, self.tasks_dir, self.issue_gen_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set mirror organization for SWE-smith
        self._setup_mirror_org()
    
    def _setup_mirror_org(self) -> None:
        """Configure SWE-smith to use custom mirror repository location.
        
        SWE-smith uses the ORG_NAME_GH constant from swesmith.constants.
        We patch this at runtime before SWE-smith creates any profiles.
        
        Mirror location can be:
        - Just org/user: "spider" -> creates "spider/{owner}__{repo}.{commit}"
        - Full path: "spider/mirrors" -> creates "spider/mirrors-{owner}__{repo}.{commit}"
        """
        # Get mirror org from config (docker_image.mirror_org or repository.mirror_org)
        mirror_org = None
        mirror_repo_template = None
        
        if self.config.docker_image:
            if self.config.docker_image.mirror_org:
                mirror_org = self.config.docker_image.mirror_org
            if self.config.docker_image.mirror_repo_template:
                mirror_repo_template = self.config.docker_image.mirror_repo_template
        
        if not mirror_org and self.config.repository.mirror_org:
            mirror_org = self.config.repository.mirror_org
        if not mirror_repo_template and self.config.repository.mirror_repo_template:
            mirror_repo_template = self.config.repository.mirror_repo_template
        
        # Parse mirror_org - could be "org" or "org/repo"
        mirror_org_name = None
        mirror_repo_prefix = None
        
        if mirror_org:
            if "/" in mirror_org:
                # Full path like "spider/mirrors"
                parts = mirror_org.split("/", 1)
                mirror_org_name = parts[0]
                mirror_repo_prefix = parts[1]
            else:
                # Just org/user like "spider"
                mirror_org_name = mirror_org
        
        if mirror_org_name:
            # Patch SWE-smith's constant before any profiles are created
            try:
                import swesmith.constants as swesmith_constants
                # Store original value
                original_org = getattr(swesmith_constants, 'ORG_NAME_GH', 'swesmith')
                swesmith_constants.ORG_NAME_GH = mirror_org_name
                logger.info(f"Patched SWE-smith ORG_NAME_GH: '{original_org}' -> '{mirror_org_name}'")
                
                # If we have a custom repo template or prefix, we need to patch the mirror_name property
                # SWE-smith uses: f"{self.org_gh}/{self.repo_name}"
                # We want to customize the repo_name part
                if mirror_repo_template or mirror_repo_prefix:
                    self._patch_mirror_repo_name(mirror_repo_template, mirror_repo_prefix)
                
                # Update existing profiles
                try:
                    from swesmith.profiles import registry
                    for profile in registry.values():
                        profile.org_gh = mirror_org_name
                        if mirror_repo_template or mirror_repo_prefix:
                            # Store custom template for later use
                            profile._spider_mirror_template = mirror_repo_template
                            profile._spider_mirror_prefix = mirror_repo_prefix
                    logger.info(f"Updated mirror org for {len(registry)} existing profiles")
                except Exception as e:
                    logger.debug(f"Could not update existing profiles: {e}")
                    
            except Exception as e:
                logger.warning(f"Could not patch SWE-smith constants: {e}")
                logger.warning("Mirror repos will use default 'swesmith' organization")
                os.environ["SWESMITH_MIRROR_ORG"] = mirror_org_name or mirror_org
    
    def _patch_mirror_repo_name(self, template: Optional[str], prefix: Optional[str]) -> None:
        """Patch the repo_name and mirror_name properties to use custom template/prefix"""
        try:
            from swesmith.profiles.base import RepoProfile
            
            # Store original properties
            original_repo_name = RepoProfile.repo_name
            original_mirror_name = RepoProfile.mirror_name
            
            def custom_repo_name(self):
                """Custom repo_name that uses template or prefix"""
                if hasattr(self, '_spider_mirror_template') and self._spider_mirror_template:
                    # Use custom template
                    return self._spider_mirror_template.format(
                        owner=self.owner,
                        repo=self.repo,
                        commit=self.commit[:8]
                    )
                elif hasattr(self, '_spider_mirror_prefix') and self._spider_mirror_prefix:
                    # Use prefix + default format
                    return f"{self._spider_mirror_prefix}-{self.owner}__{self.repo}.{self.commit[:8]}"
                else:
                    # Fall back to original
                    return original_repo_name.fget(self)
            
            def custom_mirror_name(self):
                """Custom mirror_name that uses patched repo_name"""
                return f"{self.org_gh}/{self.repo_name}"
            
            # Replace the properties
            RepoProfile.repo_name = property(custom_repo_name)
            RepoProfile.mirror_name = property(custom_mirror_name)
            logger.info("Patched RepoProfile.repo_name and mirror_name to use custom template")
            
        except Exception as e:
            logger.warning(f"Could not patch repo_name/mirror_name properties: {e}")
    
    def generate_tasks(self) -> List[Dict[str, Any]]:
        """Run the complete task generation pipeline.
        
        Returns:
            List of task instances in SWE-smith format
        """
        logger.info("Starting SWE task generation pipeline")
        events.emit("Starting task generation pipeline.", code="task_gen.start")
        
        # Step 0: Build Docker image (if enabled and required)
        docker_config = self.config.docker_image
        if docker_config and docker_config.enabled and docker_config.build_before_tasks:
            events.emit("Building Docker image...", code="task_gen.docker_build")
            self._build_docker_image()
            events.emit("Docker image built successfully.", code="task_gen.docker_build_done")
        
        # Step 1: Generate bugs
        events.emit("Generating bugs...", code="task_gen.bug_generation")
        bug_patches = self._generate_bugs()
        
        if not bug_patches:
            logger.warning("No bugs generated")
            events.emit("No bugs generated.", code="task_gen.no_bugs", level="warning")
            return []
        
        events.emit(f"Generated {len(bug_patches)} bug patches.", code="task_gen.bugs_generated", data={"count": len(bug_patches)})
        
        # Step 2: Collect patches
        events.emit("Collecting patches...", code="task_gen.collect")
        collected_patches = self._collect_patches(bug_patches)
        events.emit("Patches collected.", code="task_gen.collect_done")
        
        # Step 3: Validate bugs
        if self.config.validation.enabled:
            events.emit("Validating bugs (running tests)...", code="task_gen.validation")
            validated_patches = self._validate_bugs(collected_patches)
            events.emit("Validation complete.", code="task_gen.validation_done")
        else:
            logger.info("Skipping validation")
            events.emit("Skipping validation.", code="task_gen.validation_skipped")
            validated_patches = collected_patches
        
        # Step 4: Gather tasks
        if self.config.gather.enabled:
            events.emit("Gathering tasks into instances...", code="task_gen.gather")
            tasks = self._gather_tasks(validated_patches)
            events.emit(f"Gathered {len(tasks)} task instances.", code="task_gen.gather_done", data={"count": len(tasks)})
        else:
            logger.info("Skipping gather, using validated patches as tasks")
            events.emit("Skipping gather.", code="task_gen.gather_skipped")
            tasks = validated_patches
        
        # Step 5: Generate issue text (optional)
        if self.config.issue_generation and self.config.issue_generation.enabled:
            events.emit("Generating issue text...", code="task_gen.issue_gen")
            tasks = self._generate_issues(tasks)
            events.emit("Issue text generated.", code="task_gen.issue_gen_done")
        
        # Step 6: Rebuild Docker image with task branches (optional)
        if docker_config and docker_config.enabled and docker_config.rebuild_after_tasks:
            events.emit("Rebuilding Docker image with task branches...", code="task_gen.docker_rebuild")
            self._rebuild_docker_image()
            events.emit("Docker image rebuilt.", code="task_gen.docker_rebuild_done")
        
        logger.info(f"Generated {len(tasks)} task instances")
        events.emit(f"Task generation complete: {len(tasks)} tasks generated.", code="task_gen.complete", data={"task_count": len(tasks)})
        return tasks
    
    def _generate_bugs(self) -> List[Path]:
        """Generate bugs using configured methods.
        
        Returns:
            List of paths to bug patch files
        """
        repo_name = self._get_repo_name()
        bug_patches = []
        
        for method in self.config.bug_generation.methods:
            logger.info(f"Running bug generation method: {method.type}")
            events.emit(f"Running bug generation: {method.type}...", code="task_gen.bug_method", data={"method": method.type})
            
            try:
                if method.type == "lm_modify":
                    patches = self._run_lm_modify(repo_name, method)
                elif method.type == "lm_rewrite":
                    patches = self._run_lm_rewrite(repo_name, method)
                elif method.type == "procedural":
                    patches = self._run_procedural(repo_name, method)
                elif method.type == "pr_mirror":
                    patches = self._run_pr_mirror(method)
                else:
                    raise TaskGenerationError(f"Unknown bug generation method: {method.type}")
                
                bug_patches.extend(patches)
                logger.info(f"Generated {len(patches)} bugs using {method.type}")
                events.emit(f"Generated {len(patches)} bugs using {method.type}.", code="task_gen.bug_method_done", data={"method": method.type, "count": len(patches)})
                
            except Exception as e:
                logger.error(f"Error in {method.type}: {e}", exc_info=True)
                events.emit(f"Error in {method.type}: {str(e)}", code="task_gen.bug_method_error", level="error", data={"method": method.type})
                raise TaskGenerationError(f"Failed to generate bugs with {method.type}: {e}") from e
        
        return bug_patches
    
    def _run_lm_modify(self, repo_name: str, method_config) -> List[Path]:
        """Run LM modify bug generation"""
        cmd = [
            sys.executable, "-m", "swesmith.bug_gen.llm.modify",
            repo_name,
        ]
        
        if method_config.config_file:
            cmd.extend(["--config_file", method_config.config_file])
        if method_config.model:
            cmd.extend(["--model", method_config.model])
        if method_config.n_bugs:
            cmd.extend(["--n_bugs", str(method_config.n_bugs)])
        if method_config.n_workers:
            cmd.extend(["--n_workers", str(method_config.n_workers)])
        
        # Add any additional options
        for key, value in method_config.options.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
        
        if result.returncode != 0:
            raise TaskGenerationError(
                f"LM modify failed: {result.stderr}\n{result.stdout}"
            )
        
        # Find generated bug patches
        bug_dir = self.bug_gen_dir / repo_name
        if bug_dir.exists():
            return list(bug_dir.rglob("bug__*.diff"))
        return []
    
    def _run_lm_rewrite(self, repo_name: str, method_config) -> List[Path]:
        """Run LM rewrite bug generation"""
        cmd = [
            sys.executable, "-m", "swesmith.bug_gen.llm.rewrite",
            repo_name,
        ]
        
        if method_config.config_file:
            cmd.extend(["--config_file", method_config.config_file])
        if method_config.model:
            cmd.extend(["--model", method_config.model])
        if method_config.n_bugs:
            cmd.extend(["--n_bugs", str(method_config.n_bugs)])
        if method_config.n_workers:
            cmd.extend(["--n_workers", str(method_config.n_workers)])
        
        for key, value in method_config.options.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
        
        if result.returncode != 0:
            raise TaskGenerationError(
                f"LM rewrite failed: {result.stderr}\n{result.stdout}"
            )
        
        bug_dir = self.bug_gen_dir / repo_name
        if bug_dir.exists():
            return list(bug_dir.rglob("bug__*.diff"))
        return []
    
    def _run_procedural(self, repo_name: str, method_config) -> List[Path]:
        """Run procedural bug generation"""
        cmd = [
            sys.executable, "-m", "swesmith.bug_gen.procedural.generate",
            repo_name,
        ]
        
        if method_config.max_bugs:
            cmd.extend(["--max_bugs", str(method_config.max_bugs)])
        
        for key, value in method_config.options.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
        
        if result.returncode != 0:
            raise TaskGenerationError(
                f"Procedural generation failed: {result.stderr}\n{result.stdout}"
            )
        
        bug_dir = self.bug_gen_dir / repo_name
        if bug_dir.exists():
            return list(bug_dir.rglob("bug__*.diff"))
        return []
    
    def _run_pr_mirror(self, method_config) -> List[Path]:
        """Run PR mirror bug generation"""
        # If file is not provided and auto_collect is enabled, collect PRs automatically
        pr_data_file = method_config.file
        if not pr_data_file and method_config.auto_collect_prs:
            logger.info("No PR data file provided, automatically collecting PRs from repository...")
            pr_data_file = self._collect_and_convert_prs(method_config)
        elif not pr_data_file:
            raise TaskGenerationError(
                "PR mirror method requires 'file' parameter or enable 'auto_collect_prs'"
            )
        
        # Check if file exists
        pr_data_path = Path(pr_data_file)
        if not pr_data_path.is_absolute():
            # Try relative to workspace first, then current directory
            pr_data_path = self.workspace / pr_data_file
            if not pr_data_path.exists():
                pr_data_path = Path(pr_data_file)
        
        if not pr_data_path.exists():
            raise TaskGenerationError(f"PR data file not found: {pr_data_file}")
        
        cmd = [
            sys.executable, "-m", "swesmith.bug_gen.mirror.generate",
            str(pr_data_path),
        ]
        
        if method_config.model:
            cmd.extend(["--model", method_config.model])
        
        for key, value in method_config.options.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Running PR mirror: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
        
        if result.returncode != 0:
            raise TaskGenerationError(
                f"PR mirror failed: {result.stderr}\n{result.stdout}"
            )
        
        # PR mirror output location may vary
        bug_dir = self.bug_gen_dir
        if bug_dir.exists():
            return list(bug_dir.rglob("bug__*.diff"))
        return []
    
    def _collect_and_convert_prs(self, method_config) -> str:
        """Automatically collect PRs from repository and convert to task instance format.
        
        Returns:
            Path to the generated task instances file
        """
        repo_name = self._get_repo_name()
        repo_url = self.config.repository.github_url
        
        # Create directory for PR data
        pr_data_dir = self.workspace / "pr_data"
        pr_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Collect PRs from GitHub
        raw_prs_file = pr_data_dir / f"{repo_name.replace('/', '__')}-prs.jsonl"
        
        if not raw_prs_file.exists():
            logger.info(f"Collecting PRs from {repo_url}...")
            events.emit(f"Collecting PRs from {repo_url}...", code="task_gen.pr_collect")
            cmd = [
                sys.executable, "-m", "swesmith.bug_gen.mirror.collect.print_pulls",
                repo_url,
                str(raw_prs_file),
            ]
            
            if method_config.max_pulls:
                cmd.extend(["--max_pulls", str(method_config.max_pulls)])
            
            # Get GitHub token from environment
            github_token = os.environ.get("GITHUB_TOKEN")
            if not github_token:
                logger.warning("GITHUB_TOKEN not set - PR collection may be rate-limited")
            else:
                cmd.extend(["--token", github_token])
            
            logger.info(f"Running PR collection: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
            
            if result.returncode != 0:
                raise TaskGenerationError(
                    f"PR collection failed: {result.stderr}\n{result.stdout}"
                )
            
            if not raw_prs_file.exists():
                raise TaskGenerationError(f"PR collection completed but file not found: {raw_prs_file}")
            
            logger.info(f"✓ Collected PRs saved to {raw_prs_file}")
            events.emit(f"PRs collected and saved.", code="task_gen.pr_collect_done")
        else:
            logger.info(f"PR data already exists at {raw_prs_file}, skipping collection")
            events.emit("Using existing PR data.", code="task_gen.pr_collect_cached")
        
        # Step 2: Convert PRs to task instance format
        task_instances_file = pr_data_dir / f"{repo_name.replace('/', '__')}-task-instances.jsonl"
        
        if not task_instances_file.exists():
            logger.info("Converting PRs to task instance format...")
            events.emit("Converting PRs to task instance format...", code="task_gen.pr_convert")
            cmd = [
                sys.executable, "-m", "swesmith.bug_gen.mirror.collect.build_dataset",
                str(raw_prs_file),
                str(task_instances_file),
            ]
            
            # Get GitHub token from environment
            github_token = os.environ.get("GITHUB_TOKEN")
            if github_token:
                cmd.extend(["--token", github_token])
            
            logger.info(f"Running dataset build: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
            
            if result.returncode != 0:
                raise TaskGenerationError(
                    f"Dataset build failed: {result.stderr}\n{result.stdout}"
                )
            
            logger.info(f"✓ Task instances saved to {task_instances_file}")
            events.emit("PRs converted to task instance format.", code="task_gen.pr_convert_done")
        else:
            logger.info(f"Task instances already exist at {task_instances_file}, skipping conversion")
            events.emit("Using existing task instances.", code="task_gen.pr_convert_cached")
        
        # Return relative path from workspace
        return str(task_instances_file.relative_to(self.workspace))
    
    def _collect_patches(self, bug_patches: List[Path]) -> Path:
        """Collect all bug patches into a single file"""
        repo_name = self._get_repo_name()
        bug_dir = self.bug_gen_dir / repo_name
        
        if not bug_dir.exists():
            raise TaskGenerationError(f"Bug directory not found: {bug_dir}")
        
        output_file = bug_dir / f"{repo_name}_all_patches.json"
        
        cmd = [
            sys.executable, "-m", "swesmith.bug_gen.collect_patches",
            str(bug_dir),
        ]
        
        logger.info(f"Collecting patches: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
        
        if result.returncode != 0:
            raise TaskGenerationError(
                f"Patch collection failed: {result.stderr}\n{result.stdout}"
            )
        
        # Find the collected patches file
        collected_file = bug_dir / f"{repo_name}_all_patches.json"
        if not collected_file.exists():
            # Try alternative naming
            collected_file = bug_dir.parent / f"{repo_name}_all_patches.json"
        
        if not collected_file.exists():
            raise TaskGenerationError(f"Collected patches file not found after collection")
        
        return collected_file
    
    def _validate_bugs(self, collected_patches: Path) -> Path:
        """Validate bugs using SWE-smith harness"""
        validation_output = self.validation_dir / collected_patches.stem
        
        cmd = [
            sys.executable, "-m", "swesmith.harness.valid",
            str(collected_patches),
            "--workers", str(self.config.validation.workers),
        ]
        
        for key, value in self.config.validation.options.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Validating bugs: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
        
        if result.returncode != 0:
            raise TaskGenerationError(
                f"Validation failed: {result.stderr}\n{result.stdout}"
            )
        
        # Validation output is in validation_dir
        return self.validation_dir
    
    def _gather_tasks(self, validation_output: Path) -> List[Dict[str, Any]]:
        """Gather validated tasks into task instances.
        
        SWE-smith stores tasks as a JSON file (not JSONL) containing an array of task objects.
        Location: logs/task_insts/{run_id}.json
        """
        import json
        
        # Find the validation run directory
        validation_dirs = [d for d in self.validation_dir.iterdir() if d.is_dir()]
        if not validation_dirs:
            raise TaskGenerationError("No validation output directories found")
        
        # Use the most recent or specified validation run
        validation_run_dir = validation_dirs[-1]  # Most recent
        
        cmd = [
            sys.executable, "-m", "swesmith.harness.gather",
            str(validation_run_dir),
        ]
        
        # Add repush_image flag if Docker rebuild is enabled
        docker_config = self.config.docker_image
        if docker_config and docker_config.enabled and docker_config.rebuild_after_tasks:
            cmd.append("--repush_image")
        
        for key, value in self.config.gather.options.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Gathering tasks: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
        
        if result.returncode != 0:
            raise TaskGenerationError(
                f"Gather failed: {result.stderr}\n{result.stdout}"
            )
        
        # SWE-smith saves to logs/task_insts/{run_id}.json
        # The run_id is the validation_run_dir.name
        tasks_file = self.tasks_dir / f"{validation_run_dir.name}.json"
        if not tasks_file.exists():
            raise TaskGenerationError(f"Gathered tasks file not found: {tasks_file}")
        
        # Load tasks - SWE-smith stores as JSON array (not JSONL)
        with open(tasks_file) as f:
            tasks = json.load(f)  # This loads an array of task dictionaries
        
        if not isinstance(tasks, list):
            raise TaskGenerationError(f"Expected tasks file to contain a JSON array, got {type(tasks)}")
        
        logger.info(f"Loaded {len(tasks)} tasks from {tasks_file}")
        return tasks
    
    def _generate_issues(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate issue text for tasks"""
        if not self.config.issue_generation:
            return tasks
        
        issue_config = self.config.issue_generation
        
        # Save tasks to temp file for issue generation
        import json
        temp_tasks_file = self.workspace / "temp_tasks_for_issue_gen.json"
        with open(temp_tasks_file, "w") as f:
            json.dump(tasks, f)
        
        cmd = [
            sys.executable, "-m", "swesmith.issue_gen.generate",
            "--dataset", str(temp_tasks_file),
            "--workers", str(issue_config.workers),
        ]
        
        if issue_config.config_file:
            cmd.extend(["--config_file", issue_config.config_file])
        if issue_config.model:
            cmd.extend(["--model", issue_config.model])
        
        for key, value in issue_config.options.items():
            cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Generating issues: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"Issue generation failed: {result.stderr}\n{result.stdout}")
            logger.warning("Continuing without issue text")
            return tasks
        
        # Reload tasks with issue text
        # Issue generation modifies tasks in place or creates new files
        # This is a simplified version - may need adjustment based on SWE-smith behavior
        return tasks
    
    def _build_docker_image(self) -> None:
        """Build Docker image for the repository before task generation"""
        repo_name = self._get_repo_name()
        docker_config = self.config.docker_image
        
        logger.info(f"Building Docker image for {repo_name}")
        
        # Use SWE-smith's build_repo.create_images
        # First, we need to ensure the repo profile is registered
        # SWE-smith auto-detects based on repo, but we can also call it directly
        
        cmd = [
            sys.executable, "-m", "swesmith.build_repo.create_images",
            "--repos", self.config.repository.github_url,
        ]
        
        if docker_config and docker_config.push:
            cmd.append("--push")
        
        if docker_config and docker_config.options.get("workers"):
            cmd.extend(["--workers", str(docker_config.options["workers"])])
        
        if docker_config and docker_config.options.get("proceed"):
            cmd.append("--proceed")
        
        logger.info(f"Building Docker image: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
        
        if result.returncode != 0:
            # Check if image already exists (that's OK)
            if "already exists" in result.stderr.lower() or "ImageNotFound" not in result.stderr:
                logger.warning(f"Docker image build had issues (may already exist): {result.stderr}")
            else:
                raise TaskGenerationError(
                    f"Docker image build failed: {result.stderr}\n{result.stdout}"
                )
        else:
            logger.info("Docker image built successfully")
    
    def _rebuild_docker_image(self) -> None:
        """Rebuild Docker image after task generation to include task branches"""
        repo_name = self._get_repo_name()
        docker_config = self.config.docker_image
        
        logger.info(f"Rebuilding Docker image for {repo_name} to include task branches")
        
        # Use SWE-smith's profile system to rebuild
        # The gather step should have set repush_image, but we can also do it explicitly
        # This requires the repo profile, which SWE-smith manages internally
        
        # For now, we'll rely on SWE-smith's gather to handle this if repush_image is set
        # Or we can call the profile's push_image method directly
        logger.info("Docker image will be rebuilt by gather step if repush_image is enabled")
        # Note: This is handled by gather.py if --repush_image flag is passed
    
    def _get_repo_name(self) -> str:
        """Extract repo name from GitHub URL"""
        repo_url = self.config.repository.github_url
        # Remove https://github.com/ if present
        repo_url = repo_url.replace("https://github.com/", "").replace("http://github.com/", "")
        # Replace / with __ for SWE-smith format
        return repo_url.replace("/", "__")

