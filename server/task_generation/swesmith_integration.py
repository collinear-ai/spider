"""SWE-smith integration for task generation.

This module wraps SWE-smith's task generation pipeline to create bug instances
from GitHub repositories.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, make_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env file if it exists (for GITHUB_TOKEN, etc.)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)  # Don't override existing env vars
except ImportError:
    pass  # python-dotenv not available, skip

from spider.config import TaskGenerationConfig, CustomProfileConfig
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
        
        # Apply runtime patches to SWE-smith FIRST (before any other operations)
        self._apply_swesmith_patches()
        
        # Register custom profiles before any SWE-smith operations
        self._register_custom_profiles()
        
        # Set mirror organization for SWE-smith
        self._setup_mirror_org()
    
    def _apply_swesmith_patches(self) -> None:
        """Apply runtime monkey-patches to fix bugs in pip-installed SWE-smith."""
        logger.info("Applying runtime patches to SWE-smith...")
        
        # PATCH 1: Fix empty change list bug
        try:
            from swesmith.bug_gen import utils
            original_apply = utils.apply_code_change
            
            def patched_apply(candidate, bug):
                with open(candidate.file_path, "r") as f:
                    lines = f.readlines()
                if candidate.line_start < 1 or candidate.line_end > len(lines) or candidate.line_start > candidate.line_end:
                    raise ValueError("Invalid line range")
                change = [f"{' ' * candidate.indent_level * candidate.indent_size}{x}" if len(x.strip()) > 0 else x for x in bug.rewrite.splitlines(keepends=True)]
                if not change:
                    logger.warning(f"Empty rewrite from LLM, skipping")
                    return
                curr_last = lines[candidate.line_end - 1]
                num_newlines = len(curr_last) - len(curr_last.rstrip("\n"))
                change[-1] = change[-1].rstrip("\n") + "\n" * num_newlines
                with open(candidate.file_path, "w") as f:
                    f.writelines(lines[:candidate.line_start - 1] + change + lines[candidate.line_end:])
            
            utils.apply_code_change = patched_apply
            logger.info("  ✓ Patched swesmith.bug_gen.utils.apply_code_change")
        except Exception as e:
            logger.warning(f"  ✗ Failed to patch apply_code_change: {e}")
        
        # PATCH 2: Fix user account mirror creation
        try:
            from swesmith.profiles.base import RepoProfile
            import shutil
            
            def patched_create_mirror(self):
                import subprocess
                logger.info(f"Creating mirror: {self.org_gh}/{self.repo_name}")
                if self.repo_name in os.listdir():
                    shutil.rmtree(self.repo_name)
                
                # Check if mirror already exists and has content
                mirror_exists = False
                try:
                    repo_info = self.api.repos.get(owner=self.org_gh, repo=self.repo_name)
                    mirror_exists = True
                    if repo_info.size > 0:
                        logger.info(f"Mirror already exists with content (size={repo_info.size}), skipping")
                        return
                    else:
                        logger.warning(f"Mirror exists but is EMPTY (size=0), will delete and recreate properly")
                        # Delete the empty repo to start fresh
                        try:
                            self.api.repos.delete(owner=self.org_gh, repo=self.repo_name)
                            logger.info(f"Deleted empty mirror repo: {self.org_gh}/{self.repo_name}")
                            mirror_exists = False
                        except Exception as del_err:
                            logger.warning(f"Could not delete empty repo: {del_err}, will try to populate it")
                            # If we can't delete it, we'll try to populate it below
                except Exception:
                    # Repo doesn't exist, which is fine - we'll create it
                    pass
                
                # Clone and prepare the content FIRST (before creating GitHub repo)
                clone_url = f"https://github.com/{self.owner}/{self.repo}.git"
                logger.info(f"Cloning source repo: {clone_url}")
                subprocess.run(["git", "clone", "--no-single-branch", clone_url, self.repo_name], check=True)
                
                cwd = os.getcwd()
                try:
                    os.chdir(self.repo_name)
                    subprocess.run(["git", "checkout", self.commit], check=True)
                    subprocess.run(["git", "config", "user.name", "spider"], check=True)
                    subprocess.run(["git", "config", "user.email", "spider@example.com"], check=True)
                    
                    # NOW create the GitHub repo (only after we have content ready to push)
                    if not mirror_exists:
                        is_org = False
                        try:
                            self.api.orgs.get(self.org_gh)
                            is_org = True
                        except Exception:
                            pass
                        
                        try:
                            if is_org:
                                self.api.repos.create_in_org(self.org_gh, self.repo_name, private=False)
                                logger.info(f"✓ Created mirror as org repo: {self.org_gh}/{self.repo_name}")
                            else:
                                self.api.users.get_by_username(self.org_gh)
                                self.api.repos.create_for_authenticated_user(name=self.repo_name, private=False)
                                logger.info(f"✓ Created mirror as user repo: {self.org_gh}/{self.repo_name}")
                        except Exception as create_err:
                            if "already exists" in str(create_err).lower():
                                logger.info(f"Mirror was created by another process: {self.org_gh}/{self.repo_name}")
                            else:
                                raise RuntimeError(f"Failed to create mirror: {create_err}")
                    
                    # Push content immediately after creation
                    mirror_url = f"git@github.com:{self.mirror_name}.git"
                    subprocess.run(["git", "remote", "add", "mirror", mirror_url], check=True)
                    subprocess.run(["git", "push", "-f", "mirror", "HEAD:refs/heads/main"], check=True)
                    logger.info(f"✓ Mirror complete with content: {self.org_gh}/{self.repo_name}")
                finally:
                    os.chdir(cwd)
                    # Clean up local clone
                    if os.path.exists(self.repo_name):
                        shutil.rmtree(self.repo_name)
            
            RepoProfile.create_mirror = patched_create_mirror
            logger.info("  ✓ Patched RepoProfile.create_mirror")
        except Exception as e:
            logger.warning(f"  ✗ Failed to patch create_mirror: {e}")
    
    def _register_custom_profiles(self) -> None:
        """Dynamically create and register custom profiles with SWE-smith's registry.
        
        This allows users to add repos not in SWE-smith's built-in registry
        without modifying SWE-smith code.
        """
        from swesmith.profiles import registry
        
        # Collect all custom profiles to register
        profiles_to_register = []
        
        # Add profile from repository.custom_profile if specified
        if self.config.repository.custom_profile:
            profiles_to_register.append(self.config.repository.custom_profile)
        
        # Add profiles from task_generation.custom_profiles
        if self.config.custom_profiles:
            profiles_to_register.extend(self.config.custom_profiles)
        
        if not profiles_to_register:
            return
        
        logger.info(f"Registering {len(profiles_to_register)} custom profile(s) with SWE-smith")
        
        for profile_config in profiles_to_register:
            try:
                profile_class = self._create_profile_class(profile_config)
                registry.register_profile(profile_class)
                
                # Verify registration by checking repo_name
                profile_instance = profile_class()
                repo_name = profile_instance.repo_name
                mirror_name = profile_instance.mirror_name
                logger.info(f"Registered custom profile: {profile_config.owner}/{profile_config.repo} ({profile_config.language})")
                logger.info(f"  Profile repo_name: {repo_name}, mirror_name: {mirror_name}")
                
                # Verify it's in the registry
                if repo_name in registry.data:
                    logger.info(f"  ✓ Verified: {repo_name} is in registry")
                else:
                    logger.warning(f"  ⚠ Warning: {repo_name} not found in registry after registration")
                    
            except Exception as e:
                logger.error(f"Failed to register custom profile {profile_config.owner}/{profile_config.repo}: {e}", exc_info=True)
                raise TaskGenerationError(f"Failed to register custom profile: {e}") from e
    
    def _create_profile_class(self, profile_config: CustomProfileConfig):
        """Dynamically create a RepoProfile subclass from config.
        
        Args:
            profile_config: Custom profile configuration
            
        Returns:
            A dataclass subclass of the appropriate base profile
        """
        # Import base profiles
        from swesmith.profiles.python import PythonProfile
        from swesmith.profiles.golang import GoProfile
        from swesmith.profiles.rust import RustProfile
        from swesmith.profiles.javascript import JavaScriptProfile
        from swesmith.profiles.java import JavaProfile
        from swesmith.profiles.cpp import CppProfile
        from swesmith.profiles.c import CProfile
        from swesmith.profiles.csharp import CSharpProfile
        from swesmith.profiles.php import PhpProfile
        
        # Map language to base profile class
        language_map = {
            "python": PythonProfile,
            "golang": GoProfile,
            "rust": RustProfile,
            "javascript": JavaScriptProfile,
            "java": JavaProfile,
            "cpp": CppProfile,
            "c": CProfile,
            "csharp": CSharpProfile,
            "php": PhpProfile,
        }
        
        base_class = language_map.get(profile_config.language)
        if not base_class:
            raise ValueError(f"Unsupported language: {profile_config.language}")
        
        # Generate a unique class name based on owner, repo, and commit
        class_name = f"{profile_config.owner.capitalize()}{profile_config.repo.capitalize()}{profile_config.commit[:8]}"
        # Remove special characters that aren't valid in class names
        class_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in class_name)
        
        # Build field definitions for make_dataclass
        fields = [
            ("owner", str, field(default=profile_config.owner)),
            ("repo", str, field(default=profile_config.repo)),
            ("commit", str, field(default=profile_config.commit)),
        ]
        
        # Add optional fields
        if profile_config.install_cmds:
            install_cmds_val = profile_config.install_cmds.copy()
            fields.append(("install_cmds", "list[str]", field(default_factory=lambda: install_cmds_val.copy())))
        if profile_config.timeout:
            fields.append(("timeout", int, field(default=profile_config.timeout)))
        if profile_config.timeout_ref:
            fields.append(("timeout_ref", int, field(default=profile_config.timeout_ref)))
        if profile_config.test_cmd:
            fields.append(("test_cmd", str, field(default=profile_config.test_cmd)))
        if profile_config.language == "python" and profile_config.python_version:
            fields.append(("python_version", str, field(default=profile_config.python_version)))
        
        # Create the dataclass using make_dataclass
        profile_class = make_dataclass(
            class_name,
            fields=fields,
            bases=(base_class,),
            frozen=False
        )
        
        # Override build_image for custom profiles to auto-generate env and clone from mirror
        if profile_config.language == "python":
            import types
            
            def custom_build_image(self):
                """Build Docker image without requiring pre-generated environment file."""
                import docker
                from pathlib import Path
                from swebench.harness.docker_build import build_image as build_image_sweb
                from swebench.harness.dockerfiles import get_dockerfile_env
                from swesmith.constants import ENV_NAME, LOG_DIR_ENV
                
                BASE_IMAGE_KEY = "jyangballin/swesmith.x86_64"
                client = docker.from_env()
                python_version = getattr(self, 'python_version', '3.10')
                
                setup_commands = [
                    "#!/bin/bash",
                    "set -euxo pipefail",
                    f"git clone -o origin https://github.com/{self.mirror_name} /{ENV_NAME}",
                    f"cd /{ENV_NAME}",
                    "source /opt/miniconda3/bin/activate",
                    f"conda create -n {ENV_NAME} python={python_version} -y",
                    f"conda activate {ENV_NAME}",
                    'echo "Current environment: $CONDA_DEFAULT_ENV"',
                ] + self.install_cmds
                
                dockerfile = get_dockerfile_env(self.pltf, self.arch, "py", base_image_key=BASE_IMAGE_KEY)
                build_dir = LOG_DIR_ENV / self.repo_name
                build_dir.mkdir(parents=True, exist_ok=True)
                
                build_image_sweb(
                    image_name=self.image_name,
                    setup_scripts={"setup_env.sh": "\n".join(setup_commands) + "\n"},
                    dockerfile=dockerfile,
                    platform=self.pltf,
                    client=client,
                    build_dir=build_dir,
                )
            
            # Bind the custom method to ALL instances of this class
            # This ensures it works even with SWE-smith's singleton pattern
            original_init = profile_class.__init__
            def new_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.build_image = types.MethodType(custom_build_image, self)
            profile_class.__init__ = new_init
        
        return profile_class
    
    def _create_profile_wrapper_script(self, module_name: str, cmd_args: List[str]) -> str:
        """Create a Python script that registers custom profiles and runs a SWE-smith command.
        
        This is needed because subprocess calls start fresh Python interpreters that don't
        have access to profiles registered in the parent process.
        """
        script_lines = [
            "#!/usr/bin/env python3",
            "import sys",
            "import os",
            "from pathlib import Path",
            "from dataclasses import dataclass, field, make_dataclass",
            "",
            "# Load .env file if it exists (for GITHUB_TOKEN, etc.)",
            "try:",
            "    from dotenv import load_dotenv",
            "    # Try to find .env file relative to spider directory",
            "    env_paths = [",
            "        Path(__file__).parent.parent.parent.parent / '.env',  # spider/.env",
            "        Path(__file__).parent.parent.parent / '.env',  # Alternative path",
            "        Path.cwd() / '.env',  # Current working directory",
            "    ]",
            "    for env_path in env_paths:",
            "        if env_path.exists():",
            "            load_dotenv(env_path, override=False)",
            "            print(f'Loaded .env from {env_path}')",
            "            break",
            "except ImportError:",
            "    pass  # python-dotenv not available",
            "",
        ]
        
        # Add profile registration code for each custom profile
        profiles_to_register = []
        if self.config.repository.custom_profile:
            profiles_to_register.append(self.config.repository.custom_profile)
        if self.config.custom_profiles:
            profiles_to_register.extend(self.config.custom_profiles)
        
        # Import base profiles (only import what we need)
        base_imports = set()
        for profile_config in profiles_to_register:
            base_imports.add(profile_config.language)
        
        base_import_map = {
            "python": "from swesmith.profiles.python import PythonProfile",
            "golang": "from swesmith.profiles.golang import GoProfile",
            "rust": "from swesmith.profiles.rust import RustProfile",
            "javascript": "from swesmith.profiles.javascript import JavaScriptProfile",
            "java": "from swesmith.profiles.java import JavaProfile",
            "cpp": "from swesmith.profiles.cpp import CppProfile",
            "c": "from swesmith.profiles.c import CProfile",
            "csharp": "from swesmith.profiles.csharp import CSharpProfile",
            "php": "from swesmith.profiles.php import PhpProfile",
        }
        
        for lang in base_imports:
            if lang in base_import_map:
                script_lines.append(base_import_map[lang])
        
        script_lines.extend([
            "from swesmith.profiles import registry",
            "",
        ])
        
        # Apply mirror org configuration if specified
        # Get mirror org from config (same logic as _setup_mirror_org)
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
        
        if mirror_org:
            # Parse mirror_org - could be "org" or "org/repo"
            if "/" in mirror_org:
                parts = mirror_org.split("/", 1)
                mirror_org_name = parts[0]
                mirror_repo_prefix = parts[1]
            else:
                mirror_org_name = mirror_org
                mirror_repo_prefix = None
            
            # Also get Docker Hub org if specified
            docker_hub_org = None
            if self.config.docker_image and self.config.docker_image.docker_hub_org:
                docker_hub_org = self.config.docker_image.docker_hub_org
                if "/" in docker_hub_org:
                    docker_hub_org = docker_hub_org.split("/", 1)[0]
            
            script_lines.extend([
                "# Apply mirror org configuration BEFORE importing profiles",
                "import swesmith.constants as swesmith_constants",
                f"swesmith_constants.ORG_NAME_GH = '{mirror_org_name}'",
                f"print('Set GitHub mirror org to: {mirror_org_name}')",
            ])
            
            if docker_hub_org:
                script_lines.extend([
                    f"swesmith_constants.ORG_NAME_DH = '{docker_hub_org}'",
                    f"print('Set Docker Hub org to: {docker_hub_org}')",
                ])
            
            script_lines.extend([
                "",
                "# Patch git user config to use 'spider' instead of 'swesmith'",
                "from swesmith.profiles.base import RepoProfile",
                "import shutil",
                "import subprocess",
                "import os",
                "",
                "original_create_mirror = RepoProfile.create_mirror",
                "",
                "def patched_create_mirror(self):",
                "    \"\"\"Patched version that uses 'spider' as git user name\"\"\"",
                "    if self._mirror_exists():",
                "        return",
                "    if self.repo_name in os.listdir():",
                "        shutil.rmtree(self.repo_name)",
                "    self.api.repos.create_in_org(self.org_gh, self.repo_name)",
                "",
                "    # Clone the repository",
                "    subprocess.run(",
                "        f'git clone git@github.com:{self.owner}/{self.repo}.git {self.repo_name}',",
                "        shell=True,",
                "        check=True,",
                "        stdout=subprocess.DEVNULL,",
                "        stderr=subprocess.DEVNULL,",
                "    )",
                "",
                "    # Build the git commands (using 'spider' instead of 'swesmith')",
                "    git_cmds = [",
                "        f'cd {self.repo_name}',",
                "        f'git checkout {self.commit}',",
                "    ]",
                "",
                "    # Add submodule update if submodules exist",
                "    if os.path.exists(os.path.join(self.repo_name, '.gitmodules')):",
                "        git_cmds.append('git submodule update --init --recursive')",
                "",
                "    # Add the rest of the commands (with 'spider' as user name)",
                "    git_cmds.extend([",
                "        'rm -rf .git',",
                "        'git init',",
                "        'git config user.name \"spider\"',",
                "        'git config user.email \"spider@collinear.ai\"',",
                "        'rm -rf .github/workflows',",
                "        'rm -rf .github/dependabot.y*',",
                "        'git add .',",
                "        'git commit --no-gpg-sign -m \\'Initial commit\\'',",
                "        'git branch -M main',",
                "        f'git remote add origin git@github.com:{self.mirror_name}.git',",
                "        'git push -u origin main',",
                "    ])",
                "",
                "    # Execute the commands",
                "    subprocess.run(",
                "        '; '.join(git_cmds),",
                "        shell=True,",
                "        check=True,",
                "        stdout=subprocess.DEVNULL,",
                "        stderr=subprocess.DEVNULL,",
                "    )",
                "",
                "RepoProfile.create_mirror = patched_create_mirror",
                "print('Patched RepoProfile.create_mirror() to use \\'spider\\' as git user name')",
                "",
            ])
        
        # Generate profile class definitions and registration
        for profile_config in profiles_to_register:
            class_name = f"{profile_config.owner.capitalize()}{profile_config.repo.capitalize()}{profile_config.commit[:8]}"
            class_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in class_name)
            
            base_class_map = {
                "python": "PythonProfile",
                "golang": "GoProfile",
                "rust": "RustProfile",
                "javascript": "JavaScriptProfile",
                "java": "JavaProfile",
                "cpp": "CppProfile",
                "c": "CProfile",
                "csharp": "CSharpProfile",
                "php": "PhpProfile",
            }
            base_class = base_class_map.get(profile_config.language, "PythonProfile")
            
            # Build fields list
            fields_parts = [
                f'("owner", str, field(default="{profile_config.owner}"))',
                f'("repo", str, field(default="{profile_config.repo}"))',
                f'("commit", str, field(default="{profile_config.commit}"))',
            ]
            if profile_config.install_cmds:
                install_cmds_str = str(profile_config.install_cmds)
                fields_parts.append(f'("install_cmds", "list[str]", field(default_factory=lambda: {install_cmds_str}))')
            if profile_config.timeout:
                fields_parts.append(f'("timeout", int, field(default={profile_config.timeout}))')
            if profile_config.timeout_ref:
                fields_parts.append(f'("timeout_ref", int, field(default={profile_config.timeout_ref}))')
            if profile_config.test_cmd:
                fields_parts.append(f'("test_cmd", str, field(default="{profile_config.test_cmd}"))')
            if profile_config.language == "python" and profile_config.python_version:
                fields_parts.append(f'("python_version", str, field(default="{profile_config.python_version}"))')
            
            fields_str = "[" + ", ".join(fields_parts) + "]"
            
            script_lines.extend([
                f"# Register {profile_config.owner}/{profile_config.repo}",
                f"{class_name} = make_dataclass(",
                f'    "{class_name}",',
                f"    fields={fields_str},",
                f"    bases=({base_class},),",
                "    frozen=False",
                ")",
                f"registry.register_profile({class_name})",
                "",
            ])
        
        # After all profiles are registered, update their org_gh if mirror_org is set
        if mirror_org:
            script_lines.extend([
                "# Update org_gh for all registered profiles",
                "for profile_class in registry.values():",
                f"    profile_class.org_gh = '{mirror_org_name}'",
                "    profile_class._cache_mirror_exists = None  # Clear cache",
                f"print('Updated all profiles org_gh to: {mirror_org_name}')",
                "",
            ])
        
        # Add the actual command execution
        # SWE-smith's main() functions are called via argparse from __main__
        # We need to use argparse to parse the arguments properly
        script_lines.extend([
            "# Run the SWE-smith command",
            f"from {module_name} import main",
            "import argparse",
            "",
            "# Parse command arguments using argparse (same as SWE-smith does)",
            f"sys.argv = ['{module_name.split('.')[-1]}'] + {repr(cmd_args)}",
            "",
            "# Create parser and parse args (matching SWE-smith's argument structure)",
            "parser = argparse.ArgumentParser()",
            "parser.add_argument('repo', type=str)",
            "parser.add_argument('-c', '--config_file', required=True)",
            "parser.add_argument('--model', type=str, default=None)",
            "parser.add_argument('-w', '--n_workers', type=int, default=1)",
            "parser.add_argument('--redo_existing', action='store_true')",
            "parser.add_argument('-m', '--max_bugs', type=int, default=None)",
            "args = parser.parse_args()",
            "",
            "# Ensure mirror repository exists before running",
            "# SWE-smith requires mirror repos to exist before cloning",
            "github_token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GITHUB_JWT_TOKEN')",
            "if not github_token:",
            "    print('ERROR: GITHUB_TOKEN or GITHUB_JWT_TOKEN environment variable is required')",
            "    print('Please set it in your .env file or as an environment variable')",
            "    print(f'Current working directory: {os.getcwd()}')",
            "    print(f'Checked .env paths but token not found')",
            "    sys.exit(1)",
            "else:",
            "    print(f'Found GITHUB_TOKEN (length: {len(github_token)})')",
            "",
            "try:",
            "    profile = registry.get(args.repo)",
            f"    # Ensure org_gh is set correctly",
            f"    if '{mirror_org_name if mirror_org else ''}':",
            f"        if profile.org_gh != '{mirror_org_name if mirror_org else ''}':",
            f"            profile.org_gh = '{mirror_org_name if mirror_org else ''}'",
            f"            print('Updated profile.org_gh from ' + str(profile.org_gh) + f' to {mirror_org_name if mirror_org else ''}')",
            f"    print('Using mirror org: ' + str(profile.org_gh) + ', repo: ' + str(profile.mirror_name))",
            "    # Clear cache to force fresh check",
            "    profile._cache_mirror_exists = None",
            "    ",
            "    # Check if mirror exists (with retry in case of race condition)",
            "    import time",
            "    mirror_exists = False",
            "    for attempt in range(3):",
            "        # Force fresh check by clearing cache each time",
            "        profile._cache_mirror_exists = None",
            "        if profile._mirror_exists():",
            "            mirror_exists = True",
            "            print(f'Mirror repository exists: {profile.mirror_name}')",
            "            # Check if repository is empty (has no commits)",
            "            try:",
            "                import subprocess",
            "                result = subprocess.run(",
            "                    ['git', 'ls-remote', f'https://github.com/{profile.mirror_name}.git'],",
            "                    capture_output=True,",
            "                    text=True,",
            "                    timeout=10",
            "                )",
            "                if result.returncode == 0 and result.stdout.strip():",
            "                    print('Repository has content')",
            "                else:",
            "                    print('WARNING: Repository exists but appears to be empty')",
            "                    print('This may cause issues. The repository should be populated.')",
            "            except Exception as e:",
            "                print(f'Could not check repository content: {e}')",
            "            break",
            "        if attempt < 2:",
            "            time.sleep(1)  # Wait a bit before retrying",
            "    ",
            "    if not mirror_exists:",
            "        print(f'Mirror repository not found, attempting to create: {profile.mirror_name}...')",
            "        try:",
            "            profile.create_mirror()",
            "            print(f'Mirror repository created: {profile.mirror_name}')",
            "        except Exception as create_error:",
            "            error_str = str(create_error)",
            "            # If creation fails due to SSH/auth issues, check if repo exists but is empty",
            "            if 'Permission denied' in error_str or 'publickey' in error_str:",
            "                print('WARNING: Git operations require SSH keys to be set up')",
            "                print('The repository may exist but be empty. Please ensure:')",
            "                print('  1. SSH keys are configured for git@github.com')",
            "                print('  2. Or manually populate the repository')",
            "                # Continue anyway - SWE-smith will try to clone and may fail",
            "            else:",
            "                # Check if repo was created despite the error (race condition or permission issue)",
            "                time.sleep(2)  # Give GitHub API time to propagate",
            "                profile._cache_mirror_exists = None  # Clear cache before checking",
            "                if profile._mirror_exists():",
            "                    print(f'Mirror repository exists (may have been created by another process): {profile.mirror_name}')",
            "                    print('Continuing with bug generation...')",
            "                else:",
            "                    # Real error - repo doesn't exist",
            "                    print(f'ERROR: Could not create mirror repository: {create_error}')",
            "                    if '403' in error_str or 'Forbidden' in error_str:",
            "                        print('')",
            "                        print('Permission denied. This usually means:')",
            "                        print('  1. The token does not have admin access to the organization')",
            "                        print('  2. OR the repository already exists but you do not have access to it')",
            "                        print('')",
            "                        print('Solutions:')",
            "                        print('  - Use a personal GitHub username instead of an organization')",
            "                        print('  - Grant admin permissions to your token for the organization')",
            "                        print('  - Manually create the repository and ensure your token has access')",
            "                    else:",
            "                        print('This is required for bug generation. Please check:')",
            "                        print('  1. GITHUB_TOKEN is set and valid')",
            "                        print('  2. Token has permissions to create repos')",
            "                    sys.exit(1)",
            "except Exception as e:",
            "    print(f'ERROR: Could not ensure mirror exists: {e}')",
            "    # Check if repo exists despite the error",
            "    try:",
            "        import time",
            "        time.sleep(1)",
            "        if profile._mirror_exists():",
            "            print(f'Mirror repository exists: {profile.mirror_name}')",
            "            print('Continuing despite error...')",
            "        else:",
            "            print('This is required for bug generation. Please check:')",
            "            print('  1. GITHUB_TOKEN is set and valid')",
            "            print('  2. Token has permissions to create repos in the organization')",
            "            sys.exit(1)",
            "    except:",
            "        sys.exit(1)",
            "",
            "# Call main with parsed arguments",
            "main(**vars(args))",
        ])
        
        return "\n".join(script_lines)
    
    def _create_validation_wrapper_script(self, patches_file: str, workers: int) -> str:
        """Create a Python script that registers custom profiles and runs validation.
        
        This is needed because subprocess calls start fresh Python interpreters that don't
        have access to profiles registered in the parent process.
        """
        script_lines = [
            "#!/usr/bin/env python3",
            "import sys",
            "import os",
            "from pathlib import Path",
            "from dataclasses import dataclass, field, make_dataclass",
            "",
            "# Load .env file if it exists (for GITHUB_TOKEN, etc.)",
            "try:",
            "    from dotenv import load_dotenv",
            "    env_paths = [",
            "        Path(__file__).parent.parent.parent.parent / '.env',  # spider/.env",
            "        Path(__file__).parent.parent.parent / '.env',  # Alternative path",
            "        Path.cwd() / '.env',  # Current working directory",
            "    ]",
            "    for env_path in env_paths:",
            "        if env_path.exists():",
            "            load_dotenv(env_path, override=False)",
            "            print(f'Loaded .env from {env_path}')",
            "            break",
            "except ImportError:",
            "    pass  # python-dotenv not available",
            "",
        ]
        
        # Add profile registration code (reuse logic from _create_profile_wrapper_script)
        profiles_to_register = []
        if self.config.repository.custom_profile:
            profiles_to_register.append(self.config.repository.custom_profile)
        if self.config.custom_profiles:
            profiles_to_register.extend(self.config.custom_profiles)
        
        if not profiles_to_register:
            # No custom profiles, just run the command directly
            script_lines.extend([
                "# No custom profiles to register, running validation directly",
                "from swesmith.harness.valid import main",
                "",
                "# Run validation with required arguments",
                f"main('{patches_file}', {workers})",
            ])
            return "\n".join(script_lines)
        
        # Import base profiles
        base_imports = set()
        for profile_config in profiles_to_register:
            base_imports.add(profile_config.language)
        
        base_import_map = {
            "python": "from swesmith.profiles.python import PythonProfile",
            "golang": "from swesmith.profiles.golang import GoProfile",
            "rust": "from swesmith.profiles.rust import RustProfile",
            "javascript": "from swesmith.profiles.javascript import JavaScriptProfile",
            "java": "from swesmith.profiles.java import JavaProfile",
            "cpp": "from swesmith.profiles.cpp import CppProfile",
            "c": "from swesmith.profiles.c import CProfile",
            "csharp": "from swesmith.profiles.csharp import CSharpProfile",
            "php": "from swesmith.profiles.php import PhpProfile",
        }
        
        for lang in base_imports:
            if lang in base_import_map:
                script_lines.append(base_import_map[lang])
        
        script_lines.extend([
            "from swesmith.profiles import registry",
            "",
        ])
        
        # Apply mirror org configuration if specified
        mirror_org = None
        if self.config.docker_image:
            if self.config.docker_image.mirror_org:
                mirror_org = self.config.docker_image.mirror_org
        
        if not mirror_org and self.config.repository.mirror_org:
            mirror_org = self.config.repository.mirror_org
        
        mirror_org_name = None
        if mirror_org:
            if "/" in mirror_org:
                parts = mirror_org.split("/", 1)
                mirror_org_name = parts[0]
            else:
                mirror_org_name = mirror_org
            
            script_lines.extend([
                "# Apply mirror org configuration",
                "import swesmith.constants as swesmith_constants",
                f"swesmith_constants.ORG_NAME_GH = '{mirror_org_name}'",
                f"print('Set mirror org to: {mirror_org_name}')",
                "",
            ])
        
        # Generate profile class definitions and registration
        for profile_config in profiles_to_register:
            class_name = f"{profile_config.owner.capitalize()}{profile_config.repo.capitalize()}{profile_config.commit[:8]}"
            class_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in class_name)
            
            base_class_map = {
                "python": "PythonProfile",
                "golang": "GoProfile",
                "rust": "RustProfile",
                "javascript": "JavaScriptProfile",
                "java": "JavaProfile",
                "cpp": "CppProfile",
                "c": "CProfile",
                "csharp": "CSharpProfile",
                "php": "PhpProfile",
            }
            base_class = base_class_map.get(profile_config.language, "PythonProfile")
            
            # Build fields list
            fields_parts = [
                f'("owner", str, field(default="{profile_config.owner}"))',
                f'("repo", str, field(default="{profile_config.repo}"))',
                f'("commit", str, field(default="{profile_config.commit}"))',
            ]
            if profile_config.install_cmds:
                install_cmds_str = str(profile_config.install_cmds)
                fields_parts.append(f'("install_cmds", "list[str]", field(default_factory=lambda: {install_cmds_str}))')
            if profile_config.timeout:
                fields_parts.append(f'("timeout", int, field(default={profile_config.timeout}))')
            if profile_config.timeout_ref:
                fields_parts.append(f'("timeout_ref", int, field(default={profile_config.timeout_ref}))')
            if profile_config.test_cmd:
                fields_parts.append(f'("test_cmd", str, field(default="{profile_config.test_cmd}"))')
            if profile_config.language == "python" and profile_config.python_version:
                fields_parts.append(f'("python_version", str, field(default="{profile_config.python_version}"))')
            
            fields_str = "[" + ", ".join(fields_parts) + "]"
            
            script_lines.extend([
                f"# Register {profile_config.owner}/{profile_config.repo}",
                f"{class_name} = make_dataclass(",
                f'    "{class_name}",',
                f"    fields={fields_str},",
                f"    bases=({base_class},),",
                "    frozen=False",
                ")",
                f"registry.register_profile({class_name})",
                "",
            ])
        
        # Update org_gh for all registered profiles if mirror_org is set
        if mirror_org_name:
            script_lines.extend([
                "# Update org_gh for all registered profiles",
                "for profile_class in registry.values():",
                f"    profile_class.org_gh = '{mirror_org_name}'",
                "    profile_class._cache_mirror_exists = None  # Clear cache",
                f"print('Updated all profiles org_gh to: {mirror_org_name}')",
                "",
            ])
        
        # Run the validation command
        # Parse arguments using argparse (same as validation harness does)
        script_lines.extend([
            "# Run the validation harness",
            "from swesmith.harness.valid import main",
            "",
            "# Run validation with required arguments",
            "# The validation harness expects: main(bug_patches, workers)",
            f"main('{patches_file}', {workers})",
        ])
        
        return "\n".join(script_lines)
    
    def _patch_git_user_config(self) -> None:
        """Patch SWE-smith's create_mirror() to use 'spider' instead of 'swesmith' for git commits."""
        try:
            from swesmith.profiles.base import RepoProfile
            original_create_mirror = RepoProfile.create_mirror
            
            def patched_create_mirror(self):
                """Patched version that uses 'spider' as git user name"""
                # Call original method but we need to patch the git commands
                # Since create_mirror builds commands as a list, we'll need to intercept it differently
                # Instead, let's patch it after the fact by overriding the method
                import shutil
                import subprocess
                import os
                
                if self._mirror_exists():
                    return
                if self.repo_name in os.listdir():
                    shutil.rmtree(self.repo_name)
                self.api.repos.create_in_org(self.org_gh, self.repo_name)

                # Clone the repository
                subprocess.run(
                    f"git clone git@github.com:{self.owner}/{self.repo}.git {self.repo_name}",
                    shell=True,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Build the git commands (using 'spider' instead of 'swesmith')
                git_cmds = [
                    f"cd {self.repo_name}",
                    f"git checkout {self.commit}",
                ]

                # Add submodule update if submodules exist
                if os.path.exists(os.path.join(self.repo_name, ".gitmodules")):
                    git_cmds.append("git submodule update --init --recursive")

                # Add the rest of the commands (with 'spider' as user name)
                git_cmds.extend(
                    [
                        "rm -rf .git",
                        "git init",
                        'git config user.name "spider"',
                        'git config user.email "spider@collinear.ai"',
                        "rm -rf .github/workflows",
                        "rm -rf .github/dependabot.y*",
                        "git add .",
                        "git commit --no-gpg-sign -m 'Initial commit'",
                        "git branch -M main",
                        f"git remote add origin git@github.com:{self.mirror_name}.git",
                        "git push -u origin main",
                    ]
                )

                # Execute the commands
                subprocess.run(
                    "; ".join(git_cmds),
                    shell=True,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            
            # Replace the method
            RepoProfile.create_mirror = patched_create_mirror
            logger.info("Patched RepoProfile.create_mirror() to use 'spider' as git user name")
        except Exception as e:
            logger.warning(f"Could not patch git user config: {e}")
    
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
        
        # Patch Docker Hub organization if configured
        docker_hub_org = None
        if self.config.docker_image and self.config.docker_image.docker_hub_org:
            docker_hub_org = self.config.docker_image.docker_hub_org
            try:
                import swesmith.constants as swesmith_constants
                original_dh_org = getattr(swesmith_constants, 'ORG_NAME_DH', 'jyangballin')
                swesmith_constants.ORG_NAME_DH = docker_hub_org
                logger.info(f"Patched SWE-smith ORG_NAME_DH: '{original_dh_org}' -> '{docker_hub_org}'")
                # Store for use in subprocesses
                self._docker_hub_org = docker_hub_org
            except Exception as e:
                logger.warning(f"Could not patch SWE-smith ORG_NAME_DH: {e}")
                self._docker_hub_org = None
        else:
            self._docker_hub_org = None
        
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
            tasks = self._gather_tasks(validated_patches, collected_patches)
            events.emit(f"Gathered {len(tasks)} task instances.", code="task_gen.gather_done", data={"count": len(tasks)})
            
            # Provide helpful feedback if no tasks passed validation
            if len(tasks) == 0:
                logger.warning("No tasks passed validation")
                logger.info("This is normal - many generated bugs don't pass repository tests")
                logger.info("To get valid tasks, try:")
                logger.info("  • Increase n_bugs (try 50-100 for better chances)")
                logger.info("  • Use a repository with simpler tests")
                logger.info("  • Try 'procedural' bug generation (higher pass rate)")
                events.emit(
                    "No tasks passed validation. This is normal - try generating more bugs or using a simpler repository.",
                    level="warning",
                    code="task_gen.no_valid_tasks"
                )
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
        
        # Ensure mirror repository exists before generating bugs
        # This is required for all bug generation methods
        self._ensure_mirror_exists(repo_name)
        
        bug_patches = []
        
        for method in self.config.bug_generation.methods:
            logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"Running bug generation method: {method.type}")
            logger.info(f"  Method config: {method}")
            events.emit(f"Running bug generation: {method.type}...", code="task_gen.bug_method", data={"method": method.type})
            
            try:
                if method.type == "lm_modify":
                    logger.info(f"Starting lm_modify for {repo_name}...")
                    logger.info(f"  Model: {method.model}")
                    logger.info(f"  Max bugs: {method.max_bugs or method.n_bugs}")
                    patches = self._run_lm_modify(repo_name, method)
                elif method.type == "lm_rewrite":
                    logger.info(f"Starting lm_rewrite for {repo_name}...")
                    logger.info(f"  Model: {method.model}")
                    logger.info(f"  Max bugs: {method.max_bugs or method.n_bugs}")
                    logger.info(f"  Workers: {method.n_workers}")
                    patches = self._run_lm_rewrite(repo_name, method)
                elif method.type == "procedural":
                    logger.info(f"Starting procedural bug generation for {repo_name}...")
                    logger.info(f"  Max bugs: {method.max_bugs}")
                    patches = self._run_procedural(repo_name, method)
                elif method.type == "pr_mirror":
                    logger.info(f"Starting PR mirror bug generation...")
                    logger.info(f"  File: {method.file}")
                    logger.info(f"  Auto collect PRs: {method.auto_collect_prs}")
                    patches = self._run_pr_mirror(method)
                else:
                    raise TaskGenerationError(f"Unknown bug generation method: {method.type}")
                
                logger.info(f"✓ Generated {len(patches)} bug patches using {method.type}")
                if patches:
                    logger.info(f"  Patch files found: {len(patches)}")
                    logger.info(f"  First few: {[str(p.name) for p in patches[:3]]}{'...' if len(patches) > 3 else ''}")
                bug_patches.extend(patches)
                events.emit(f"Generated {len(patches)} bugs using {method.type}.", code="task_gen.bug_method_done", data={"method": method.type, "count": len(patches)})
                
            except Exception as e:
                logger.error(f"✗ Error in {method.type}: {e}", exc_info=True)
                events.emit(f"Error in {method.type}: {str(e)}", code="task_gen.bug_method_error", level="warning", data={"method": method.type})
                logger.warning(f"Continuing with next method despite error in {method.type}")
                # Don't raise - continue with other methods or with bugs generated so far
        
        return bug_patches
    
    def _run_lm_modify(self, repo_name: str, method_config) -> List[Path]:
        """Run LM modify bug generation"""
        # Ensure mirror repository exists before running
        self._ensure_mirror_exists(repo_name)
        
        cmd_args = [repo_name]
        
        # config_file is required for lm_modify, use default if not provided
        if method_config.config_file:
            config_file = method_config.config_file
        else:
            # Use default SWE-smith config file
            default_config = Path("/home/ubuntu/SWE-smith/configs/bug_gen/lm_modify.yml")
            if not default_config.exists():
                # Try relative to current directory or find it
                import swesmith
                swesmith_path = Path(swesmith.__file__).parent.parent
                default_config = swesmith_path / "configs" / "bug_gen" / "lm_modify.yml"
            if not default_config.exists():
                raise TaskGenerationError(
                    "config_file is required for lm_modify. Please specify it in your config or ensure "
                    "SWE-smith's default config exists at configs/bug_gen/lm_modify.yml"
                )
            config_file = str(default_config)
        
        cmd_args.extend(["--config_file", config_file])
        
        if method_config.model:
            cmd_args.extend(["--model", method_config.model])
        if method_config.n_bugs:
            cmd_args.extend(["--n_bugs", str(method_config.n_bugs)])
        if method_config.n_workers:
            cmd_args.extend(["--n_workers", str(method_config.n_workers)])
        
        # Add any additional options
        for key, value in method_config.options.items():
            cmd_args.extend([f"--{key}", str(value)])
        
        # Use unified wrapper script if we have custom profiles
        if self.config.repository.custom_profile or self.config.custom_profiles:
            wrapper_script = self._create_unified_wrapper_script(
                "swesmith.bug_gen.llm.modify",
                cmd_args,
                "main()"
            )
            cmd = [sys.executable, str(wrapper_script)]
        else:
            cmd = [sys.executable, "-m", "swesmith.bug_gen.llm.modify"] + cmd_args
        
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
    
    def _ensure_mirror_exists(self, repo_name: str) -> None:
        """Ensure the mirror repository exists on GitHub before running operations.
        
        SWE-smith requires mirror repositories to exist before cloning.
        """
        from swesmith.profiles import registry
        
        try:
            profile = registry.get(repo_name)
            if not profile._mirror_exists():
                logger.info(f"Creating mirror repository for {repo_name}...")
                profile.create_mirror()
                logger.info(f"Mirror repository created: {profile.mirror_name}")
            else:
                logger.debug(f"Mirror repository already exists: {profile.mirror_name}")
        except Exception as e:
            logger.warning(f"Could not ensure mirror exists for {repo_name}: {e}")
            # Continue anyway - SWE-smith will handle the error
    
    def _run_lm_rewrite(self, repo_name: str, method_config) -> List[Path]:
        """Run LM rewrite bug generation"""
        # Ensure mirror repository exists before running
        self._ensure_mirror_exists(repo_name)
        
        # Create a wrapper script that registers custom profiles before running
        # This is needed because subprocess calls start fresh Python interpreters
        wrapper_script = self.workspace / "run_lm_rewrite.py"
        
        # Build the command arguments
        cmd_args = [repo_name]
        
        # config_file is required for lm_rewrite, use default if not provided
        if method_config.config_file:
            config_file = method_config.config_file
            logger.info(f"Using provided config_file: {config_file}")
        else:
            # Use default SWE-smith config file
            default_config = Path("/home/ubuntu/SWE-smith/configs/bug_gen/lm_rewrite.yml")
            if not default_config.exists():
                # Try relative to current directory or find it
                import swesmith
                swesmith_path = Path(swesmith.__file__).parent.parent
                default_config = swesmith_path / "configs" / "bug_gen" / "lm_rewrite.yml"
            if not default_config.exists():
                raise TaskGenerationError(
                    "config_file is required for lm_rewrite. Please specify it in your config or ensure "
                    "SWE-smith's default config exists at configs/bug_gen/lm_rewrite.yml"
                )
            config_file = str(default_config)
            logger.info(f"Using default config_file: {config_file}")
        
        cmd_args.extend(["--config_file", config_file])
        
        if method_config.model:
            cmd_args.extend(["--model", method_config.model])
        # lm_rewrite uses --max_bugs, not --n_bugs
        if method_config.max_bugs:
            cmd_args.extend(["--max_bugs", str(method_config.max_bugs)])
        elif method_config.n_bugs:
            # Fallback: use n_bugs as max_bugs for compatibility
            cmd_args.extend(["--max_bugs", str(method_config.n_bugs)])
        if method_config.n_workers:
            cmd_args.extend(["--n_workers", str(method_config.n_workers)])
        
        for key, value in method_config.options.items():
            cmd_args.extend([f"--{key}", str(value)])
        
        # Use unified wrapper script if we have custom profiles
        has_custom_profile = bool(self.config.repository.custom_profile or self.config.custom_profiles)
        logger.info(f"Has custom profile: {has_custom_profile}")
        logger.info(f"  repository.custom_profile: {self.config.repository.custom_profile}")
        logger.info(f"  custom_profiles: {self.config.custom_profiles}")
        
        if has_custom_profile:
            logger.info("Creating unified wrapper script for custom profiles...")
            # For lm_rewrite, we need to parse args and pass them to main()
            main_call_code = """import argparse
parser = argparse.ArgumentParser()
parser.add_argument('repo')
parser.add_argument('--config_file', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--n_workers', type=int, default=1)
parser.add_argument('--redo_existing', action='store_true')
parser.add_argument('--max_bugs', type=int, default=None)
args, unknown = parser.parse_known_args(sys.argv[1:])
main(args.repo, args.config_file, args.model, args.n_workers, args.redo_existing, args.max_bugs)"""
            
            wrapper_script = self._create_unified_wrapper_script(
                "swesmith.bug_gen.llm.rewrite",
                cmd_args,
                main_call_code
            )
            cmd = [sys.executable, str(wrapper_script)]
            logger.info(f"Using wrapper script: {wrapper_script}")
        else:
            # No custom profiles, can use direct command
            cmd = [sys.executable, "-m", "swesmith.bug_gen.llm.rewrite"] + cmd_args
            logger.info("Using direct command (no custom profiles)")
        
        logger.info(f"Running lm_rewrite command: {' '.join(cmd)}")
        logger.info(f"  Working directory: {self.workspace}")
        logger.info(f"  Config file: {config_file}")
        # Pass environment variables (especially GITHUB_TOKEN) to subprocess
        env = os.environ.copy()
        # Ensure GITHUB_TOKEN is available (load from .env if needed)
        if not env.get('GITHUB_TOKEN') and not env.get('GITHUB_JWT_TOKEN'):
            # Try to load from .env file
            try:
                from dotenv import load_dotenv
                env_path = Path(__file__).parent.parent.parent / ".env"
                if env_path.exists():
                    # Load into a temporary dict first
                    with open(env_path) as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"').strip("'")
                                if key == 'GITHUB_TOKEN' or key == 'GITHUB_JWT_TOKEN':
                                    env[key] = value
                                    logger.info(f"Loaded {key} from .env file")
            except Exception as e:
                logger.warning(f"Could not load GITHUB_TOKEN from .env: {e}")
        
        # Verify GITHUB_TOKEN is available
        if not env.get('GITHUB_TOKEN') and not env.get('GITHUB_JWT_TOKEN'):
            logger.warning("GITHUB_TOKEN not found in environment. Mirror creation may fail.")
            logger.warning("Please ensure GITHUB_TOKEN is set in .env file or as an environment variable.")
        else:
            token_key = 'GITHUB_TOKEN' if env.get('GITHUB_TOKEN') else 'GITHUB_JWT_TOKEN'
            logger.info(f"Using {token_key} for GitHub authentication")
        
        max_bugs = method_config.max_bugs or method_config.n_bugs or "unknown"
        model = method_config.model or "default"
        n_workers = method_config.n_workers or 1
        
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("Starting lm_rewrite subprocess")
        logger.info(f"  Model: {model}")
        logger.info(f"  Max bugs: {max_bugs}")
        logger.info(f"  Workers: {n_workers}")
        logger.info(f"  Config file: {config_file}")
        logger.info("  ⏳ This may take 5-30+ minutes depending on:")
        logger.info("     - Number of bugs requested")
        logger.info("     - LLM API response time")
        logger.info("     - Number of workers (sequential if n_workers=1)")
        logger.info("  📊 Streaming output in real-time...")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Stream output in real-time instead of capturing it all
        import threading
        import queue
        
        process = subprocess.Popen(
            cmd,
            cwd=str(self.workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,  # Line buffered
        )
        
        stdout_lines = []
        stderr_lines = []
        
        def read_output(pipe, lines_list, log_func):
            """Read from pipe and log in real-time"""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        line = line.rstrip()
                        lines_list.append(line)
                        # Log important lines immediately
                        if any(keyword in line.lower() for keyword in ['error', 'warning', 'bug', 'progress', 'generating', 'completed', 'failed']):
                            log_func(f"  [lm_rewrite] {line}")
                        # Also log every 10th line to show progress
                        elif len(lines_list) % 10 == 0:
                            log_func(f"  [lm_rewrite] {line}")
            except Exception as e:
                log_func(f"  [lm_rewrite] Error reading output: {e}")
            finally:
                pipe.close()
        
        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(
            target=read_output,
            args=(process.stdout, stdout_lines, logger.info),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=read_output,
            args=(process.stderr, stderr_lines, logger.warning),
            daemon=True
        )
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Wait for threads to finish reading
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        
        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"lm_rewrite subprocess completed")
        logger.info(f"  Return code: {return_code}")
        logger.info(f"  Captured {len(stdout_lines)} stdout lines, {len(stderr_lines)} stderr lines")
        
        # Create a result-like object for compatibility
        class ProcessResult:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
        
        result = ProcessResult(return_code, '\n'.join(stdout_lines), '\n'.join(stderr_lines))
        
        # Even if return code != 0, collect any bugs that were generated before the error
        # This makes the pipeline resilient - partial progress is better than complete failure
        bug_dir = self.workspace / "logs" / "bug_gen" / repo_name
        patches = []
        if bug_dir.exists():
            patches = list(bug_dir.rglob("bug__lm_rewrite*.diff"))
            logger.info(f"Found {len(patches)} bug patches in {bug_dir}")
        
        if result.returncode != 0:
            logger.warning(f"lm_rewrite completed with errors (code {result.returncode})")
            logger.warning(f"However, {len(patches)} bugs were successfully generated before the error")
            if len(patches) > 0:
                logger.info(f"Continuing with {len(patches)} successfully generated bugs")
                events.emit(
                    f"Bug generation had errors but {len(patches)} bugs were saved",
                    level="warning",
                    code="bug_gen.partial_success",
                    data={"count": len(patches), "method": "lm_rewrite"}
                )
                return patches  # Return the bugs we have
            else:
                logger.error(f"lm_rewrite failed and no bugs were generated")
                raise TaskGenerationError(
                    f"LM rewrite failed: {result.stderr}\n{result.stdout}"
                )
        
        logger.info("✓ lm_rewrite completed successfully")
        
        # SWE-smith stores bugs in logs/bug_gen/, not directly in bug_gen/
        # Check both locations for compatibility
        bug_dir = self.bug_gen_dir / repo_name
        logs_bug_dir = self.workspace / "logs" / "bug_gen" / repo_name
        
        logger.info(f"  Looking for bug patches in: {bug_dir}")
        logger.info(f"  Also checking logs directory: {logs_bug_dir}")
        
        # Log some diagnostic info about what happened
        if result.stdout:
            stdout_lines_list = result.stdout.strip().split('\n')
            logger.info(f"  SWE-smith stdout ({len(stdout_lines_list)} lines):")
            # Show last 30 lines to see what happened
            for line in stdout_lines_list[-30:]:
                if line.strip():  # Skip empty lines
                    logger.info(f"    {line}")
        
        # Search for bugs in both locations
        patches = []
        if bug_dir.exists():
            patches.extend(list(bug_dir.rglob("bug__*.diff")))
        if logs_bug_dir.exists():
            patches.extend(list(logs_bug_dir.rglob("bug__*.diff")))
        
        # Also search recursively from workspace root in case SWE-smith uses a different structure
        if not patches:
            all_patches = list(self.workspace.rglob("bug__*.diff"))
            if all_patches:
                logger.info(f"  Found {len(all_patches)} bug patches in alternative locations")
                patches = all_patches
        
        if patches:
            logger.info(f"  ✓ Found {len(patches)} bug patch files")
            logger.info(f"    Examples: {[p.name for p in patches[:3]]}")
            return patches
        else:
            logger.warning(f"  ⚠ No bug patches found")
            logger.warning(f"  Checked locations:")
            logger.warning(f"    - {bug_dir} (exists: {bug_dir.exists()})")
            logger.warning(f"    - {logs_bug_dir} (exists: {logs_bug_dir.exists()})")
            logger.warning(f"  This could mean:")
            logger.warning(f"    - SWE-smith's lm_rewrite couldn't generate bugs for this repo")
            logger.warning(f"    - The LLM failed to produce valid bug patches")
            logger.warning(f"    - Bugs were generated in a different location")
            return []
    
    def _create_unified_wrapper_script(self, module_name: str, cmd_args: List[str], main_call: str = "main()") -> Path:
        """Create a unified wrapper script that registers custom profiles and runs any SWE-smith command.
        
        This works for ALL bug generation methods (procedural, lm_rewrite, lm_modify, pr_mirror, etc.)
        """
        script_lines = [
            "#!/usr/bin/env python3",
            "import sys",
            "import os",
            "from pathlib import Path",
            "",
            "# Load .env file if it exists (for GITHUB_TOKEN, etc.)",
            "try:",
            "    from dotenv import load_dotenv",
            "    env_paths = [",
            "        Path(__file__).parent.parent.parent.parent / '.env',  # spider/.env",
            "        Path(__file__).parent.parent.parent / '.env',  # Alternative path",
            "        Path.cwd() / '.env',  # Current working directory",
            "    ]",
            "    for env_path in env_paths:",
            "        if env_path.exists():",
            "            load_dotenv(env_path, override=False)",
            "            print(f'Loaded .env from {env_path}')",
            "            break",
            "except ImportError:",
            "    pass  # python-dotenv not available",
            "",
        ]
        
        # Collect profiles to register
        profiles_to_register = []
        if self.config.repository.custom_profile:
            profiles_to_register.append(self.config.repository.custom_profile)
        if self.config.custom_profiles:
            profiles_to_register.extend(self.config.custom_profiles)
        
        # Determine mirror_org and docker_hub_org FIRST (needed for patching)
        mirror_org = None
        docker_hub_org = None
        if self.config.docker_image:
            if self.config.docker_image.mirror_org:
                mirror_org = self.config.docker_image.mirror_org
            if self.config.docker_image.docker_hub_org:
                docker_hub_org = self.config.docker_image.docker_hub_org
                if "/" in docker_hub_org:
                    docker_hub_org = docker_hub_org.split("/", 1)[0]
        if not mirror_org and self.config.repository.mirror_org:
            mirror_org = self.config.repository.mirror_org
        
        # Apply ORG_NAME patching REGARDLESS of custom profiles
        # This is critical for gather to create tasks with correct org names
        if mirror_org or docker_hub_org:
            if mirror_org:
                if "/" in mirror_org:
                    mirror_org_name = mirror_org.split("/", 1)[0]
                else:
                    mirror_org_name = mirror_org
            else:
                mirror_org_name = None
            
            script_lines.append("# Patch SWE-smith constants for org names")
            script_lines.append("import swesmith.constants as swesmith_constants")
            
            if mirror_org_name:
                script_lines.append(f"swesmith_constants.ORG_NAME_GH = '{mirror_org_name}'")
                script_lines.append(f"print('✓ Set GitHub org to: {mirror_org_name}')")
            
            if docker_hub_org:
                script_lines.append(f"swesmith_constants.ORG_NAME_DH = '{docker_hub_org}'")
                script_lines.append(f"print('✓ Set Docker Hub org to: {docker_hub_org}')")
            
            script_lines.append("")
        
        if profiles_to_register:
            # Import base profiles
            base_imports = set()
            for profile_config in profiles_to_register:
                base_imports.add(profile_config.language)
            
            base_import_map = {
                "python": "from swesmith.profiles.python import PythonProfile",
                "golang": "from swesmith.profiles.golang import GoProfile",
                "rust": "from swesmith.profiles.rust import RustProfile",
                "javascript": "from swesmith.profiles.javascript import JavaScriptProfile",
                "java": "from swesmith.profiles.java import JavaProfile",
                "cpp": "from swesmith.profiles.cpp import CppProfile",
                "c": "from swesmith.profiles.c import CProfile",
                "csharp": "from swesmith.profiles.csharp import CSharpProfile",
                "php": "from swesmith.profiles.php import PhpProfile",
            }
            
            for lang in base_imports:
                if lang in base_import_map:
                    script_lines.append(base_import_map[lang])
            
            script_lines.extend([
                "from swesmith.profiles import registry",
                "from dataclasses import make_dataclass, field",
                "",
            ])
            
            # Apply mirror org configuration
            mirror_org = None
            if self.config.docker_image and self.config.docker_image.mirror_org:
                mirror_org = self.config.docker_image.mirror_org
            if not mirror_org and self.config.repository.mirror_org:
                mirror_org = self.config.repository.mirror_org
            
            if mirror_org:
                if "/" in mirror_org:
                    mirror_org_name = mirror_org.split("/", 1)[0]
                else:
                    mirror_org_name = mirror_org
                
                script_lines.extend([
                    "# Apply mirror org configuration BEFORE importing profiles",
                    "import swesmith.constants as swesmith_constants",
                    f"swesmith_constants.ORG_NAME_GH = '{mirror_org_name}'",
                    f"print('Set mirror org to: {mirror_org_name}')",
                    "",
                ])
            
            # Generate profile class definitions and registration
            for profile_config in profiles_to_register:
                class_name = f"{profile_config.owner.capitalize()}{profile_config.repo.capitalize()}{profile_config.commit[:8]}"
                class_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in class_name)
                
                base_class_map = {
                    "python": "PythonProfile",
                    "golang": "GoProfile",
                    "rust": "RustProfile",
                    "javascript": "JavaScriptProfile",
                    "java": "JavaProfile",
                    "cpp": "CppProfile",
                    "c": "CProfile",
                    "csharp": "CSharpProfile",
                    "php": "PhpProfile",
                }
                base_class = base_class_map.get(profile_config.language, "PythonProfile")
                
                fields_parts = [
                    f'("owner", str, field(default="{profile_config.owner}"))',
                    f'("repo", str, field(default="{profile_config.repo}"))',
                    f'("commit", str, field(default="{profile_config.commit}"))',
                ]
                if profile_config.install_cmds:
                    install_cmds_str = str(profile_config.install_cmds)
                    fields_parts.append(f'("install_cmds", "list[str]", field(default_factory=lambda: {install_cmds_str}))')
                if profile_config.timeout:
                    fields_parts.append(f'("timeout", int, field(default={profile_config.timeout}))')
                if profile_config.timeout_ref:
                    fields_parts.append(f'("timeout_ref", int, field(default={profile_config.timeout_ref}))')
                if profile_config.test_cmd:
                    fields_parts.append(f'("test_cmd", str, field(default="{profile_config.test_cmd}"))')
                if profile_config.language == "python" and profile_config.python_version:
                    fields_parts.append(f'("python_version", str, field(default="{profile_config.python_version}"))')
                
                fields_str = "[" + ", ".join(fields_parts) + "]"
                
                script_lines.extend([
                    f"# Register {profile_config.owner}/{profile_config.repo}",
                    f"{class_name} = make_dataclass(",
                    f'    "{class_name}",',
                    f"    fields={fields_str},",
                    f"    bases=({base_class},),",
                    "    frozen=False",
                    ")",
                ])
                
                # Add custom build_image method for Python profiles
                if profile_config.language == "python":
                    script_lines.extend([
                        "",
                        f"# Override build_image for {class_name} to not require pre-generated env file",
                        f"def custom_build_image_{class_name}(self):",
                        "    import docker",
                        "    from pathlib import Path",
                        "    from swebench.harness.docker_build import build_image as build_image_sweb",
                        "    from swebench.harness.dockerfiles import get_dockerfile_env",
                        "    from swesmith.constants import ENV_NAME, LOG_DIR_ENV",
                        "    ",
                        "    BASE_IMAGE_KEY = 'jyangballin/swesmith.x86_64'",
                        "    client = docker.from_env()",
                        "    ",
                        f"    python_version = getattr(self, 'python_version', '3.10')",
                        "    ",
                        "    setup_commands = [",
                        "        '#!/bin/bash',",
                        "        'set -euxo pipefail',",
                        "        f'git clone -o origin https://github.com/{self.mirror_name} /{ENV_NAME}',",
                        "        f'cd /{ENV_NAME}',",
                        "        'source /opt/miniconda3/bin/activate',",
                        "        f'conda create -n {ENV_NAME} python={python_version} -y',",
                        "        f'conda activate {ENV_NAME}',",
                        "        'echo \"Current environment: $CONDA_DEFAULT_ENV\"',",
                        "    ] + self.install_cmds",
                        "    ",
                        "    dockerfile = get_dockerfile_env(self.pltf, self.arch, 'py', base_image_key=BASE_IMAGE_KEY)",
                        "    build_dir = LOG_DIR_ENV / self.repo_name",
                        "    build_dir.mkdir(parents=True, exist_ok=True)",
                        "    ",
                        "    build_image_sweb(",
                        "        image_name=self.image_name,",
                        "        setup_scripts={'setup_env.sh': '\\n'.join(setup_commands) + '\\n'},",
                        "        dockerfile=dockerfile,",
                        "        platform=self.pltf,",
                        "        client=client,",
                        "        build_dir=build_dir,",
                        "    )",
                        "",
                        f"{class_name}.build_image = custom_build_image_{class_name}",
                        "",
                    ])
                
                script_lines.extend([
                    f"registry.register_profile({class_name})",
                    f"print(f'✓ Registered custom profile: {profile_config.owner}__{profile_config.repo}.{profile_config.commit[:8]}')",
                    f"print(f'  Registry now contains: {{list(registry.keys())}}')",
                    "",
                ])
            
            # Update org_gh for all registered profiles if mirror_org is set
            if mirror_org:
                script_lines.extend([
                    "# Update org_gh for all registered profiles",
                    "for profile_class in registry.values():",
                    f"    profile_class.org_gh = '{mirror_org_name}'",
                    "    profile_class._cache_mirror_exists = None  # Clear cache",
                    f"print('Updated all profiles org_gh to: {mirror_org_name}')",
                    "",
                ])
        
        # Run the SWE-smith command
        module_short_name = module_name.split('.')[-1]
        
        # Check if main_call contains argparse (needs import at top level)
        needs_argparse = 'import argparse' in main_call or 'parser.parse_args' in main_call
        
        # Extract argparse import if needed
        argparse_import = ""
        if needs_argparse:
            # Extract import argparse line from main_call if present
            if 'import argparse' in main_call:
                argparse_import = "import argparse"
        
        script_lines.extend([
            f"# Run {module_name}",
        ])
        
        # Add argparse import at top if needed
        if argparse_import:
            script_lines.append(argparse_import)
        
        script_lines.extend([
            f"from {module_name} import main",
            "",
            "try:",
            f"    sys.argv = ['{module_short_name}'] + {repr(cmd_args)}",
        ])
        
        # Properly indent multi-line main_call strings
        if '\n' in main_call:
            # Split into lines and indent each non-empty line
            main_call_lines = main_call.split('\n')
            for line in main_call_lines:
                if line.strip():  # Non-empty line - indent it
                    # Skip import argparse line if we already added it at top
                    if needs_argparse and line.strip() == 'import argparse':
                        continue
                    script_lines.append(f"    {line}")
                else:
                    # Empty lines are preserved as-is (they don't need indentation)
                    script_lines.append("")
        else:
            # Single line, just indent it
            script_lines.append(f"    {main_call}")
        
        script_lines.extend([
            "except SystemExit as e:",
            "    sys.exit(e.code if e.code is not None else 0)",
            "except Exception as e:",
            f"    print('ERROR in {module_short_name}: ' + str(e), file=sys.stderr)",
            "    import traceback",
            "    traceback.print_exc()",
            "    sys.exit(1)",
        ])
        
        # Write wrapper script
        script_name = module_name.split('.')[-1] + "_wrapper.py"
        wrapper_script = self.workspace / script_name
        wrapper_content = "\n".join(script_lines)
        wrapper_script.write_text(wrapper_content)
        wrapper_script.chmod(0o755)
        
        logger.info(f"Created wrapper script at: {wrapper_script}")
        logger.debug(f"Wrapper script first 50 lines:\n{chr(10).join(script_lines[:50])}")
        
        return wrapper_script
    
    def _run_procedural(self, repo_name: str, method_config) -> List[Path]:
        """Run procedural bug generation"""
        # Ensure mirror repository exists before running
        self._ensure_mirror_exists(repo_name)
        
        cmd_args = [repo_name]
        
        if method_config.max_bugs:
            cmd_args.extend(["--max_bugs", str(method_config.max_bugs)])
        
        for key, value in method_config.options.items():
            cmd_args.extend([f"--{key}", str(value)])
        
        # Use unified wrapper script if we have custom profiles
        if self.config.repository.custom_profile or self.config.custom_profiles:
            # Procedural uses argparse, so parse arguments and call main with them
            # Use same defaults as procedural's __main__ block: seed=24, max_bugs=-1
            procedural_main_call = """import argparse
parser = argparse.ArgumentParser()
parser.add_argument('repo', type=str)
parser.add_argument('--seed', type=int, default=24)
parser.add_argument('--max_bugs', type=int, default=-1)
args = parser.parse_args()
# Call main with parsed arguments (same as procedural's __main__ block)
main(**vars(args))"""
            
            wrapper_script = self._create_unified_wrapper_script(
                "swesmith.bug_gen.procedural.generate",
                cmd_args,
                procedural_main_call
            )
            cmd = [sys.executable, str(wrapper_script)]
        else:
            cmd = [sys.executable, "-m", "swesmith.bug_gen.procedural.generate"] + cmd_args
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
        
        # Log output for debugging
        if result.stdout:
            logger.info(f"Procedural stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"Procedural stderr: {result.stderr}")
        
        if result.returncode != 0:
            raise TaskGenerationError(
                f"Procedural generation failed: {result.stderr}\n{result.stdout}"
            )
        
        # Procedural writes bugs to logs/bug_gen/ (SWE-smith's LOG_DIR_BUG_GEN)
        # Check both our bug_gen_dir and SWE-smith's logs/bug_gen location
        # Resolve paths to handle relative vs absolute path issues
        bug_dirs_to_check = [
            (self.bug_gen_dir / repo_name).resolve(),  # Our location
            (self.workspace / "logs" / "bug_gen" / repo_name).resolve(),  # SWE-smith's location
        ]
        
        for bug_dir in bug_dirs_to_check:
            logger.info(f"Looking for bug patches in: {bug_dir}")
            if bug_dir.exists():
                patches = list(bug_dir.rglob("bug__*.diff"))
                logger.info(f"Found {len(patches)} bug patch files in {bug_dir}")
                if patches:
                    logger.info(f"  First few patches: {[str(p.name) for p in patches[:3]]}")
                    return patches
            else:
                logger.debug(f"Bug directory does not exist: {bug_dir}")
        
        # Also check alternative locations (legacy)
        alt_dirs = [
            self.workspace / "run_procedural" / repo_name,
            self.workspace / "logs" / "run_procedural" / repo_name,
        ]
        for alt_dir in alt_dirs:
            if alt_dir.exists():
                patches = list(alt_dir.rglob("bug__*.diff"))
                if patches:
                    logger.info(f"Found {len(patches)} patches in alternative location: {alt_dir}")
                    return patches
        
        logger.warning(f"No bug patches found for {repo_name} in any checked location")
        return []
    
    def _create_pr_mirror_wrapper_script(self, cmd_args: List[str]) -> str:
        """Create a Python script that registers custom profiles and runs PR Mirror.
        
        This is needed because subprocess calls start fresh Python interpreters that don't
        have access to profiles registered in the parent process.
        """
        script_lines = [
            "#!/usr/bin/env python3",
            "import sys",
            "import os",
            "from pathlib import Path",
            "from dataclasses import dataclass, field, make_dataclass",
            "",
            "# Load .env file if it exists (for GITHUB_TOKEN, etc.)",
            "try:",
            "    from dotenv import load_dotenv",
            "    env_paths = [",
            "        Path(__file__).parent.parent.parent.parent / '.env',  # spider/.env",
            "        Path(__file__).parent.parent.parent / '.env',  # Alternative path",
            "        Path.cwd() / '.env',  # Current working directory",
            "    ]",
            "    for env_path in env_paths:",
            "        if env_path.exists():",
            "            load_dotenv(env_path, override=False)",
            "            print(f'Loaded .env from {env_path}')",
            "            break",
            "except ImportError:",
            "    pass  # python-dotenv not available",
            "",
        ]
        
        # Add profile registration code (reuse logic from _create_profile_wrapper_script)
        profiles_to_register = []
        if self.config.repository.custom_profile:
            profiles_to_register.append(self.config.repository.custom_profile)
        if self.config.custom_profiles:
            profiles_to_register.extend(self.config.custom_profiles)
        
        if not profiles_to_register:
            # No custom profiles - PR Mirror should use built-in profiles
            script_lines.extend([
                "# No custom profiles to register",
                "# PR Mirror will use built-in SWE-smith profiles",
                "",
            ])
        else:
            # Import base profiles (only import what we need)
            base_imports = set()
            for profile_config in profiles_to_register:
                base_imports.add(profile_config.language)
            
            base_import_map = {
                "python": "from swesmith.profiles.python import PythonProfile",
                "golang": "from swesmith.profiles.golang import GoProfile",
                "rust": "from swesmith.profiles.rust import RustProfile",
                "javascript": "from swesmith.profiles.javascript import JavaScriptProfile",
                "java": "from swesmith.profiles.java import JavaProfile",
                "cpp": "from swesmith.profiles.cpp import CppProfile",
                "c": "from swesmith.profiles.c import CProfile",
                "csharp": "from swesmith.profiles.csharp import CSharpProfile",
                "php": "from swesmith.profiles.php import PhpProfile",
            }
            
            for lang in base_imports:
                if lang in base_import_map:
                    script_lines.append(base_import_map[lang])
            
            script_lines.extend([
                "from swesmith.profiles import registry",
                "",
            ])
            
            # Apply mirror org configuration if specified
            mirror_org = None
            if self.config.docker_image:
                if self.config.docker_image.mirror_org:
                    mirror_org = self.config.docker_image.mirror_org
            
            if not mirror_org and self.config.repository.mirror_org:
                mirror_org = self.config.repository.mirror_org
            
            if mirror_org:
                if "/" in mirror_org:
                    parts = mirror_org.split("/", 1)
                    mirror_org_name = parts[0]
                else:
                    mirror_org_name = mirror_org
                
                script_lines.extend([
                    "# Apply mirror org configuration BEFORE importing profiles",
                    "import swesmith.constants as swesmith_constants",
                    f"swesmith_constants.ORG_NAME_GH = '{mirror_org_name}'",
                    f"print('Set mirror org to: {mirror_org_name}')",
                    "",
                ])
            
            # Generate profile class definitions and registration
            for profile_config in profiles_to_register:
                class_name = f"{profile_config.owner.capitalize()}{profile_config.repo.capitalize()}{profile_config.commit[:8]}"
                class_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in class_name)
                
                base_class_map = {
                    "python": "PythonProfile",
                    "golang": "GoProfile",
                    "rust": "RustProfile",
                    "javascript": "JavaScriptProfile",
                    "java": "JavaProfile",
                    "cpp": "CppProfile",
                    "c": "CProfile",
                    "csharp": "CSharpProfile",
                    "php": "PhpProfile",
                }
                base_class = base_class_map.get(profile_config.language, "PythonProfile")
                
                # Build fields list
                fields_parts = [
                    f'("owner", str, field(default="{profile_config.owner}"))',
                    f'("repo", str, field(default="{profile_config.repo}"))',
                    f'("commit", str, field(default="{profile_config.commit}"))',
                ]
                if profile_config.install_cmds:
                    install_cmds_str = str(profile_config.install_cmds)
                    fields_parts.append(f'("install_cmds", "list[str]", field(default_factory=lambda: {install_cmds_str}))')
                if profile_config.timeout:
                    fields_parts.append(f'("timeout", int, field(default={profile_config.timeout}))')
                if profile_config.timeout_ref:
                    fields_parts.append(f'("timeout_ref", int, field(default={profile_config.timeout_ref}))')
                if profile_config.test_cmd:
                    fields_parts.append(f'("test_cmd", str, field(default="{profile_config.test_cmd}"))')
                if profile_config.language == "python" and profile_config.python_version:
                    fields_parts.append(f'("python_version", str, field(default="{profile_config.python_version}"))')
                
                fields_str = "[" + ", ".join(fields_parts) + "]"
                
                script_lines.extend([
                    f"# Register {profile_config.owner}/{profile_config.repo}",
                    f"{class_name} = make_dataclass(",
                    f'    "{class_name}",',
                    f"    fields={fields_str},",
                    f"    bases=({base_class},),",
                    "    frozen=False",
                    ")",
                    f"registry.register_profile({class_name})",
                    f"print(f'Registered profile: {profile_config.owner}__{profile_config.repo}')",
                    "",
                ])
            
            # Update org_gh for all registered profiles if mirror_org is set
            if mirror_org:
                script_lines.extend([
                    "# Update org_gh for all registered profiles",
                    "for profile_class in registry.values():",
                    f"    profile_class.org_gh = '{mirror_org_name}'",
                    "    profile_class._cache_mirror_exists = None  # Clear cache",
                    f"print('Updated all profiles org_gh to: {mirror_org_name}')",
                    "",
                ])
        
        # Run PR Mirror command
        # PR Mirror's __main__ uses argparse, so set up sys.argv to match
        script_lines.extend([
            "# Run PR Mirror",
            "# PR Mirror's __main__ block expects sys.argv with file path and optional args",
            "import argparse",
            f"sys.argv = ['pr_mirror'] + {repr(cmd_args)}",
            "",
            "# Import PR Mirror module to trigger __main__ block when run as script",
            "# But we'll call main() directly with parsed arguments",
            "from swesmith.bug_gen.mirror.generate import main",
            "",
            "# Parse arguments like PR Mirror's __main__ block does",
            "parser = argparse.ArgumentParser()",
            "parser.add_argument('sweb_insts_files', type=str, nargs='+')",
            "parser.add_argument('--model', type=str, default=None)",
            "parser.add_argument('--redo_existing', action='store_true', default=False)",
            "parser.add_argument('--redo_skipped', action='store_true', default=False)",
            "parser.add_argument('--num_processes', type=int, default=1)",
            "args = parser.parse_args()",
            "",
            "# Call main with parsed arguments",
            "try:",
            "    main(**vars(args))",
            "except SystemExit as e:",
            "    sys.exit(e.code if e.code is not None else 0)",
            "except Exception as e:",
            "    print(f'ERROR in PR Mirror: {e}', file=sys.stderr)",
            "    import traceback",
            "    traceback.print_exc()",
            "    sys.exit(1)",
        ])
        
        return "\n".join(script_lines)
    
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
        
        # Build command arguments
        cmd_args = [str(pr_data_path.relative_to(self.workspace) if pr_data_path.is_relative_to(self.workspace) else pr_data_path)]
        
        if method_config.model:
            cmd_args.extend(["--model", method_config.model])
        
        for key, value in method_config.options.items():
            cmd_args.extend([f"--{key}", str(value)])
        
        # Use unified wrapper script if we have custom profiles
        if self.config.repository.custom_profile or self.config.custom_profiles:
            # PR Mirror uses argparse, so we need special main call
            pr_mirror_main_call = """import argparse
parser = argparse.ArgumentParser()
parser.add_argument('sweb_insts_files', type=str, nargs='+')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--redo_existing', action='store_true', default=False)
parser.add_argument('--redo_skipped', action='store_true', default=False)
parser.add_argument('--num_processes', type=int, default=1)
args = parser.parse_args()
main(**vars(args))"""
            
            wrapper_script = self._create_unified_wrapper_script(
                "swesmith.bug_gen.mirror.generate",
                cmd_args,
                pr_mirror_main_call
            )
            cmd = [sys.executable, str(wrapper_script)]
        else:
            # No custom profiles, run directly
            cmd = [
                sys.executable, "-m", "swesmith.bug_gen.mirror.generate",
            ] + cmd_args
            logger.info(f"Running PR mirror: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
        
        if result.returncode != 0:
            raise TaskGenerationError(
                f"PR mirror failed: {result.stderr}\n{result.stdout}"
            )
        
        # PR mirror output location may vary
        # SWE-smith stores bugs in logs/bug_gen/, not directly in bug_gen/
        repo_name = self._get_repo_name()
        bug_dir = self.bug_gen_dir / repo_name
        logs_bug_dir = self.workspace / "logs" / "bug_gen" / repo_name
        
        patches = []
        
        # Check logs directory first (preferred location)
        if logs_bug_dir.exists():
            logger.info(f"Checking for bugs in logs directory: {logs_bug_dir}")
            patches.extend(list(logs_bug_dir.rglob("bug__*.diff")))
        
        # Also check bug_gen directory
        if bug_dir.exists():
            logger.info(f"Checking for bugs in bug_gen directory: {bug_dir}")
            patches.extend(list(bug_dir.rglob("bug__*.diff")))
        
        # Fallback: search recursively from workspace root
        if not patches:
            logger.info("No bugs found in standard locations, searching recursively...")
            all_patches = list(self.workspace.rglob("bug__*.diff"))
            if all_patches:
                logger.info(f"Found {len(all_patches)} bug patches in workspace")
                patches.extend(all_patches)
        
        logger.info(f"Found {len(patches)} bug patches from PR mirror")
        return patches
    
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
        
        # Check both bug_gen_dir and logs/bug_gen (SWE-smith stores bugs in logs/bug_gen/)
        bug_dir = self.bug_gen_dir / repo_name
        logs_bug_dir = self.workspace / "logs" / "bug_gen" / repo_name
        
        # Use logs_bug_dir if it exists (preferred location), otherwise fall back to bug_dir
        if logs_bug_dir.exists():
            bug_dir = logs_bug_dir
            logger.info(f"Using bug directory: {bug_dir}")
        elif bug_dir.exists():
            logger.info(f"Using bug directory: {bug_dir}")
        else:
            # Neither exists - this shouldn't happen if patches were found
            # But let's check if patches exist in subdirectories
            all_patches_dir = None
            if bug_patches:
                # Find the common parent directory of all patches
                patch_dirs = {p.parent for p in bug_patches}
                if patch_dirs:
                    # Get the repo_name directory (should be parent of all patch dirs)
                    for patch_path in bug_patches:
                        # Walk up from patch to find the repo_name directory
                        current = patch_path.parent
                        while current != self.workspace and current.name != repo_name:
                            current = current.parent
                        if current.name == repo_name:
                            all_patches_dir = current
                            break
            
            if all_patches_dir and all_patches_dir.exists():
                bug_dir = all_patches_dir
                logger.info(f"Using bug directory from patch locations: {bug_dir}")
            else:
                raise TaskGenerationError(
                    f"Bug directory not found. Checked:\n"
                    f"  - {bug_dir} (exists: {bug_dir.exists()})\n"
                    f"  - {logs_bug_dir} (exists: {logs_bug_dir.exists()})\n"
                    f"  Found {len(bug_patches)} patch files but couldn't determine bug directory."
                )
        
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
        
        # Build command arguments (excluding the patches file)
        cmd_args = [
            "--workers", str(self.config.validation.workers),
        ]
        
        for key, value in self.config.validation.options.items():
            cmd_args.extend([f"--{key}", str(value)])
        
        # Extract workers value from cmd_args for easier passing to wrapper
        workers = self.config.validation.workers
        
        # If we have custom profiles, create a wrapper script that registers them
        # before running validation (subprocesses don't see parent's registry)
        if self.config.repository.custom_profile or self.config.custom_profiles:
            wrapper_script = self.workspace / "validate_wrapper.py"
            wrapper_content = self._create_validation_wrapper_script(
                str(collected_patches),
                workers
            )
            wrapper_script.write_text(wrapper_content)
            wrapper_script.chmod(0o755)
            cmd = [sys.executable, str(wrapper_script)]
            logger.info(f"Using validation wrapper script with custom profiles")
        else:
            # No custom profiles, can use direct command
            cmd = [
                sys.executable, "-m", "swesmith.harness.valid",
                str(collected_patches),
            ] + cmd_args
        
        logger.info(f"Validating bugs: {' '.join(cmd)}")
        # Count patches in the collected patches file for logging
        patch_count = "unknown"
        if collected_patches.exists():
            try:
                import json
                with open(collected_patches, 'r') as f:
                    patches_data = json.load(f)
                    if isinstance(patches_data, list):
                        patch_count = len(patches_data)
                    elif isinstance(patches_data, dict) and "patches" in patches_data:
                        patch_count = len(patches_data["patches"])
            except Exception:
                pass  # If we can't read it, just use "unknown"
        logger.info(f"Validation may take a while - processing {patch_count} bugs with {self.config.validation.workers} workers")
        
        # Run validation with timeout (default 2 hours for large batches)
        # Validation can take a long time, especially with many bugs
        result = subprocess.run(
            cmd, 
            cwd=str(self.workspace), 
            capture_output=True, 
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        # Log validation output for debugging
        if result.stdout:
            logger.debug(f"Validation stdout (last 1000 chars): {result.stdout[-1000:]}")
        if result.stderr:
            logger.debug(f"Validation stderr (last 1000 chars): {result.stderr[-1000:]}")
        
        if result.returncode != 0:
            raise TaskGenerationError(
                f"Validation failed: {result.stderr}\n{result.stdout}"
            )
        
        # Validation output is in validation_dir
        return self.validation_dir
    
    def _gather_tasks(self, validation_output: Path, collected_patches: Path) -> List[Dict[str, Any]]:
        """Gather validated tasks into task instances.
        
        SWE-smith stores tasks as a JSON file (not JSONL) containing an array of task objects.
        Location: logs/task_insts/{run_id}.json
        
        validation_output can be either:
        - A collected patches file path (when validation was skipped)
        - A validation directory (when validation ran)
        """
        import json
        
        # Check if validation_output is a file (collected patches) or directory
        if validation_output.is_file():
            # It's a collected patches file - gather can work with this directly
            logger.info(f"Using collected patches file for gather: {validation_output}")
            # Use the parent directory (where the patches file is) as the run directory
            validation_run_dir = validation_output.parent
        else:
            # It's a directory - find validation run directories
            # SWE-smith might store validation output in run_validation/ or logs/run_validation/
            validation_dirs = []
            
            # Check standard location
            if validation_output.exists():
                dirs = [d for d in validation_output.iterdir() if d.is_dir()]
                validation_dirs.extend(dirs)
                logger.info(f"Checked {validation_output}: found {len(dirs)} directories")
            
            # Check logs location (SWE-smith sometimes stores things in logs/)
            logs_validation_dir = self.workspace / "logs" / "run_validation"
            if logs_validation_dir.exists():
                logs_dirs = [d for d in logs_validation_dir.iterdir() if d.is_dir()]
                validation_dirs.extend(logs_dirs)
                logger.info(f"Checked {logs_validation_dir}: found {len(logs_dirs)} additional directories")
            
            # If no subdirectories found, try fallback options
            if not validation_dirs:
                # Check if validation_dir itself should be used (if it has files)
                if validation_output.exists() and any(validation_output.iterdir()):
                    logger.info(f"Using validation_dir directly: {validation_output}")
                    validation_run_dir = validation_output
                elif logs_validation_dir.exists() and any(logs_validation_dir.iterdir()):
                    logger.info(f"Using logs_validation_dir directly: {logs_validation_dir}")
                    validation_run_dir = logs_validation_dir
                elif collected_patches.exists() and collected_patches.is_file():
                    # Fallback: use collected patches file if validation directories don't exist
                    logger.warning(f"No validation directories found, using collected patches file: {collected_patches}")
                    validation_run_dir = collected_patches.parent
                else:
                    # Log what directories exist to help debug
                    logger.error(f"Validation directory exists: {validation_output.exists()}")
                    logger.error(f"Logs validation directory exists: {logs_validation_dir.exists()}")
                    logger.error(f"Collected patches file exists: {collected_patches.exists() if collected_patches else False}")
                    if validation_output.parent.exists():
                        workspace_contents = list(validation_output.parent.iterdir())
                        logger.error(f"Workspace contents: {[str(p.name) for p in workspace_contents]}")
                    if (self.workspace / "logs").exists():
                        logs_contents = list((self.workspace / "logs").iterdir())
                        logger.error(f"Logs directory contents: {[str(p.name) for p in logs_contents]}")
                    raise TaskGenerationError(
                        f"No validation output found. Checked:\n"
                        f"  - {validation_output} (exists: {validation_output.exists()})\n"
                        f"  - {logs_validation_dir} (exists: {logs_validation_dir.exists()})\n"
                        f"  - Collected patches: {collected_patches} (exists: {collected_patches.exists() if collected_patches else False})"
                    )
            else:
                # Use the most recent validation run directory
                validation_run_dir = validation_dirs[-1]
                logger.info(f"Using validation run directory: {validation_run_dir}")
        
        # Gather needs a fresh clone, but rp.clone() REUSES existing clones if they exist.
        # Why cleanup? Gather creates branches and pushes them. If the repo has leftover branches
        # from validation (that don't exist on remote), gather's branch operations can fail.
        #
        # Note: Bug generation and validation typically don't clone repos locally
        # (validation runs in Docker), so cleanup is mainly defensive. If nothing cloned
        # before gather, cleanup is harmless (just won't find anything to remove).
        repo_name_full = self._get_repo_name()
        
        # Clean up repo clones - check both full name and base name
        cleanup_paths = [
            self.workspace / repo_name_full,  # Full name: "pallets__click.fde47b4b"
            self.workspace / repo_name_full.split('.')[0],  # Base: "pallets__click"
        ]
        
        # Also check for any matching directories
        if self.workspace.exists():
            for item in self.workspace.iterdir():
                if item.is_dir() and item.name.startswith(repo_name_full.split('.')[0]):
                    cleanup_paths.append(item)
        
        # Remove duplicates
        cleanup_paths = list(set(cleanup_paths))
        
        for path_to_remove in cleanup_paths:
            if path_to_remove.exists() and path_to_remove.is_dir():
                logger.info(f"Cleaning up existing repo clone before gather: {path_to_remove}")
                logger.debug(f"  (rp.clone() reuses existing clones, so cleanup forces fresh clone)")
                import shutil
                try:
                    shutil.rmtree(path_to_remove)
                    logger.info(f"Successfully removed {path_to_remove}")
                except Exception as e:
                    logger.warning(f"Could not remove repo clone {path_to_remove}: {e}")
        
        # Gather expects validation_logs_path to be relative to the workspace
        # (it uses relative paths like ../{path_patch} from inside the repo)
        # Convert to relative path from workspace if needed
        validation_run_dir_abs = validation_run_dir.resolve()
        workspace_abs = self.workspace.resolve()
        
        try:
            # Try to get relative path from workspace
            gather_path = str(validation_run_dir_abs.relative_to(workspace_abs))
            logger.info(f"Using relative path for gather (from workspace): {gather_path}")
        except ValueError:
            # Not under workspace - use absolute path (gather should handle it)
            gather_path = str(validation_run_dir_abs)
            logger.warning(f"Validation path is not under workspace, using absolute path: {gather_path}")
        
        # Build gather command arguments
        gather_cmd_args = [gather_path]
        
        # Add repush_image flag if Docker rebuild is enabled
        docker_config = self.config.docker_image
        if docker_config and docker_config.enabled and docker_config.rebuild_after_tasks:
            gather_cmd_args.append("--repush_image")
        
        for key, value in self.config.gather.options.items():
            gather_cmd_args.extend([f"--{key}", str(value)])
        
        # Add override_branch flag to allow gather to work with existing branches
        # This helps when validation created branches that don't exist on remote
        if "--override_branch" not in gather_cmd_args:
            gather_cmd_args.append("--override_branch")
        
        # Always use wrapper script for gather to ensure mirror org is configured
        # Gather needs mirror org to push branches to the correct GitHub org
        # Without this, gather will push to default 'swesmith' org which user may not have access to
        mirror_org = None
        if self.config.docker_image and self.config.docker_image.mirror_org:
            mirror_org = self.config.docker_image.mirror_org
        if not mirror_org and self.config.repository.mirror_org:
            mirror_org = self.config.repository.mirror_org
        
        if mirror_org or self.config.repository.custom_profile or self.config.custom_profiles:
            # Gather uses argparse, so we need to parse arguments first
            # The wrapper imports main, sets sys.argv, then needs to parse args before calling main
            gather_main_call = """import argparse
import subprocess as original_subprocess

# Patch subprocess.run to make git push failures non-fatal
_original_run = original_subprocess.run
def _patched_run(cmd, *args, **kwargs):
    # Check if this is a git push command
    is_git_push = False
    if isinstance(cmd, list) and len(cmd) >= 2:
        is_git_push = (cmd[0] == 'git' and cmd[1] == 'push')
    elif isinstance(cmd, str):
        is_git_push = cmd.startswith('git push')
    
    if is_git_push:
        # Make git push non-fatal
        kwargs_copy = kwargs.copy()
        kwargs_copy['check'] = False  # Don't raise on error
        result = _original_run(cmd, *args, **kwargs_copy)
        if result.returncode != 0:
            print(f"WARNING: git push failed (non-fatal): {cmd}", file=sys.stderr)
        return result
    
    # Not a git push, call original
    return _original_run(cmd, *args, **kwargs)

original_subprocess.run = _patched_run
import subprocess
subprocess.run = _patched_run

parser = argparse.ArgumentParser(description="Convert validation logs to SWE-bench style dataset")
parser.add_argument("validation_logs_path", type=str, help="Path to the validation logs")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
parser.add_argument("-o", "--override_branch", action="store_true", help="Override existing branches")
parser.add_argument("-d", "--debug_subprocess", action="store_true", help="Debug mode")
parser.add_argument("-p", "--repush_image", action="store_true", help="Rebuild and push Docker image")
args = parser.parse_args()
main(**vars(args))"""
            
            wrapper_script = self._create_unified_wrapper_script(
                "swesmith.harness.gather",
                gather_cmd_args,
                gather_main_call
            )
            cmd = [sys.executable, str(wrapper_script)]
            logger.info(f"Using wrapper script for gather to ensure mirror org '{mirror_org}' is configured")
        else:
            cmd = [sys.executable, "-m", "swesmith.harness.gather"] + gather_cmd_args
        
        logger.info(f"Gathering tasks: {' '.join(cmd)}")
        logger.info(f"Gather will clone repo fresh - ensure GITHUB_TOKEN is set for authenticated git")
        
        # Ensure GITHUB_TOKEN is available for gather (it needs authenticated git)
        env = os.environ.copy()
        if "GITHUB_TOKEN" not in env:
            # Try to load from .env file
            try:
                from dotenv import load_dotenv
                env_path = Path(__file__).parent.parent.parent / ".env"
                if env_path.exists():
                    load_dotenv(env_path, override=False)
                    if "GITHUB_TOKEN" in os.environ:
                        env["GITHUB_TOKEN"] = os.environ["GITHUB_TOKEN"]
                        logger.info("Loaded GITHUB_TOKEN from .env for gather")
            except Exception:
                pass
        
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True, env=env)
        
        # Log gather command output for debugging
        if result.stdout:
            # Show more of the output for debugging gather failures
            logger.info(f"Gather command stdout (last 2000 chars): {result.stdout[-2000:]}")
        if result.stderr:
            logger.info(f"Gather command stderr (last 2000 chars): {result.stderr[-2000:]}")
        
        # Check if gather succeeded partially (tasks written but push failed)
        gather_had_errors = result.returncode != 0
        if gather_had_errors:
            error_msg = result.stderr + "\n" + result.stdout
            logger.warning(f"Gather completed with errors (return code {result.returncode})")
            logger.warning(f"Error output: {error_msg[-1000:]}")
            
            # Check if the error is just a git push failure
            if 'git push' in error_msg.lower() and ('128' in error_msg or 'denied' in error_msg.lower()):
                logger.warning("Gather encountered a git push error (likely authentication)")
                logger.warning("Will check if tasks were generated before the push failure")
            else:
                logger.warning(f"Gather had errors: {error_msg[-500:]}")
        
        # SWE-smith saves to logs/task_insts/{run_id}.json
        # The run_id is the validation_run_dir.name
        # Check both task_insts/ and logs/task_insts/
        run_id = validation_run_dir.name
        logger.info(f"Looking for tasks file with run_id: {run_id}")
        logger.info(f"Validation run directory: {validation_run_dir}")
        tasks_file = self.tasks_dir / f"{run_id}.json"
        logs_tasks_file = self.workspace / "logs" / "task_insts" / f"{run_id}.json"
        
        # Also check if there are any task files with different names
        if (self.workspace / "logs" / "task_insts").exists():
            all_task_files = list((self.workspace / "logs" / "task_insts").glob("*.json"))
            if all_task_files:
                logger.info(f"All task files in logs/task_insts: {[str(f.name) for f in all_task_files]}")
        
        # Prefer logs/task_insts/ (where SWE-smith actually saves)
        if logs_tasks_file.exists():
            tasks_file = logs_tasks_file
            logger.info(f"Found tasks file in logs: {tasks_file}")
        elif tasks_file.exists():
            logger.info(f"Found tasks file: {tasks_file}")
        else:
            # If gather had errors and no tasks file, that's a real failure
            if gather_had_errors:
                logger.error("Gather had errors AND no tasks file was created")
                raise TaskGenerationError(
                    f"Gather failed and no tasks were created:\n{result.stderr}\n{result.stdout}"
                )
            
            # Check if either directory exists and what files are in it
            if self.tasks_dir.exists():
                task_files = list(self.tasks_dir.glob("*.json"))
                logger.error(f"Task files in {self.tasks_dir}: {[str(f.name) for f in task_files]}")
            if (self.workspace / "logs" / "task_insts").exists():
                logs_task_files = list((self.workspace / "logs" / "task_insts").glob("*.json"))
                logger.error(f"Task files in logs/task_insts: {[str(f.name) for f in logs_task_files]}")
            raise TaskGenerationError(
                f"Gathered tasks file not found. Checked:\n"
                f"  - {tasks_file}\n"
                f"  - {logs_tasks_file}"
            )
        
        # Load tasks - SWE-smith stores as JSON array (not JSONL)
        with open(tasks_file) as f:
            tasks = json.load(f)  # This loads an array of task dictionaries
        
        if not isinstance(tasks, list):
            raise TaskGenerationError(f"Expected tasks file to contain a JSON array, got {type(tasks)}")
        
        logger.info(f"Loaded {len(tasks)} tasks from {tasks_file}")
        
        # If gather had errors but tasks were created, emit a warning
        if gather_had_errors and len(tasks) > 0:
            logger.warning(f"Gather completed with errors but {len(tasks)} tasks were successfully created")
            logger.warning("Task branches may not have been pushed to GitHub (git push failed)")
            logger.warning("To fix: ensure GITHUB_TOKEN is set and has write access")
            events.emit(
                f"Gather had git push errors but {len(tasks)} tasks were created locally",
                level="warning",
                code="gather.partial_success",
                data={"count": len(tasks)}
            )
        
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
    
    def _create_docker_build_wrapper_script(self, cmd_args: List[str]) -> Path:
        """Create a wrapper script for Docker building that patches Docker Hub org and registers profiles"""
        script_lines = [
            "#!/usr/bin/env python3",
            "# Wrapper script for Docker image building",
            "import sys",
            "import os",
            "",
            "# Load environment variables",
            "from dotenv import load_dotenv",
            "load_dotenv()",
            "",
        ]
        
        # Patch Docker Hub org if configured
        docker_hub_org = None
        if self.config.docker_image and self.config.docker_image.docker_hub_org:
            docker_hub_org = self.config.docker_image.docker_hub_org
            script_lines.extend([
                "# Patch Docker Hub organization",
                "import swesmith.constants as swesmith_constants",
                f"swesmith_constants.ORG_NAME_DH = '{docker_hub_org}'",
                f"print('Set Docker Hub org to: {docker_hub_org}')",
                "",
            ])
        
        # Register custom profiles if needed
        profiles_to_register = []
        if self.config.repository.custom_profile:
            profiles_to_register.append(self.config.repository.custom_profile)
        if self.config.custom_profiles:
            profiles_to_register.extend(self.config.custom_profiles)
        
        if profiles_to_register:
            # Import base profiles
            base_imports = set()
            for profile_config in profiles_to_register:
                base_imports.add(profile_config.language)
            
            base_import_map = {
                "python": "from swesmith.profiles.python import PythonProfile",
                "golang": "from swesmith.profiles.golang import GoProfile",
                "rust": "from swesmith.profiles.rust import RustProfile",
                "javascript": "from swesmith.profiles.javascript import JavaScriptProfile",
                "java": "from swesmith.profiles.java import JavaProfile",
                "cpp": "from swesmith.profiles.cpp import CppProfile",
                "c": "from swesmith.profiles.c import CProfile",
                "csharp": "from swesmith.profiles.csharp import CSharpProfile",
                "php": "from swesmith.profiles.php import PhpProfile",
            }
            
            for lang in base_imports:
                if lang in base_import_map:
                    script_lines.append(base_import_map[lang])
            
            script_lines.extend([
                "from swesmith.profiles import registry",
                "from dataclasses import make_dataclass, field",
                "",
            ])
            
            # Apply mirror org configuration
            mirror_org = None
            if self.config.docker_image and self.config.docker_image.mirror_org:
                mirror_org = self.config.docker_image.mirror_org
            if not mirror_org and self.config.repository.mirror_org:
                mirror_org = self.config.repository.mirror_org
            
            if mirror_org:
                if "/" in mirror_org:
                    mirror_org_name = mirror_org.split("/", 1)[0]
                else:
                    mirror_org_name = mirror_org
                
                script_lines.extend([
                    "# Apply mirror org configuration BEFORE importing profiles",
                    "import swesmith.constants as swesmith_constants",
                    f"swesmith_constants.ORG_NAME_GH = '{mirror_org_name}'",
                    f"print('Set mirror org to: {mirror_org_name}')",
                    "",
                ])
            
            # Generate profile class definitions and registration (similar to other wrappers)
            for profile_config in profiles_to_register:
                class_name = f"{profile_config.owner.capitalize()}{profile_config.repo.capitalize()}{profile_config.commit[:8]}"
                class_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in class_name)
                
                base_class_map = {
                    "python": "PythonProfile",
                    "golang": "GoProfile",
                    "rust": "RustProfile",
                    "javascript": "JavaScriptProfile",
                    "java": "JavaProfile",
                    "cpp": "CppProfile",
                    "c": "CProfile",
                    "csharp": "CSharpProfile",
                    "php": "PhpProfile",
                }
                base_class = base_class_map.get(profile_config.language, "PythonProfile")
                
                fields_parts = [
                    f'("owner", str, field(default="{profile_config.owner}"))',
                    f'("repo", str, field(default="{profile_config.repo}"))',
                    f'("commit", str, field(default="{profile_config.commit}"))',
                ]
                if profile_config.install_cmds:
                    install_cmds_str = str(profile_config.install_cmds)
                    fields_parts.append(f'("install_cmds", "list[str]", field(default_factory=lambda: {install_cmds_str}))')
                if profile_config.timeout:
                    fields_parts.append(f'("timeout", int, field(default={profile_config.timeout}))')
                if profile_config.timeout_ref:
                    fields_parts.append(f'("timeout_ref", int, field(default={profile_config.timeout_ref}))')
                if profile_config.test_cmd:
                    fields_parts.append(f'("test_cmd", str, field(default="{profile_config.test_cmd}"))')
                if profile_config.language == "python" and profile_config.python_version:
                    fields_parts.append(f'("python_version", str, field(default="{profile_config.python_version}"))')
                
                fields_str = "[" + ", ".join(fields_parts) + "]"
                
                script_lines.extend([
                    f"# Register {profile_config.owner}/{profile_config.repo}",
                    f"{class_name} = make_dataclass(",
                    f'    "{class_name}",',
                    f"    fields={fields_str},",
                    f"    bases=({base_class},),",
                    "    frozen=False",
                    ")",
                    f"registry.register_profile({class_name})",
                    f"print(f'Registered profile: {profile_config.owner}__{profile_config.repo}')",
                    "",
                ])
        
        # Run Docker build command
        script_lines.extend([
            "# Run Docker build command",
            "from swesmith.build_repo.create_images import main",
            "",
            "try:",
            f"    sys.argv = ['create_images'] + {repr(cmd_args)}",
            "    main()",
            "except Exception as e:",
            "    print(f'ERROR in Docker build: {e}', file=sys.stderr)",
            "    import traceback",
            "    traceback.print_exc()",
            "    sys.exit(1)",
        ])
        
        # Write wrapper script
        wrapper_script = self.workspace / "docker_build_wrapper.py"
        wrapper_script.write_text("\n".join(script_lines))
        wrapper_script.chmod(0o755)
        
        return wrapper_script
    
    def _build_docker_image(self) -> None:
        """Build Docker image for the repository before task generation"""
        repo_name = self._get_repo_name()
        docker_config = self.config.docker_image
        
        logger.info(f"Building Docker image for {repo_name}")
        
        # Check if this is a custom profile (not in SWE-smith's registry)
        has_custom_profile = (
            self.config.repository.custom_profile is not None or
            (self.config.custom_profiles and len(self.config.custom_profiles) > 0)
        )
        
        if has_custom_profile:
            # For custom profiles, we need to build the image directly using the profile
            # because create_images only works with pre-registered profiles
            logger.info(f"Building Docker image for custom profile: {repo_name}")
            self._build_custom_profile_image(repo_name, docker_config)
            return
        
        # For repos in SWE-smith's registry, use create_images
        # Find the correct profile image name from the registry
        # e.g., jyangballin/swesmith.x86_64.pallets_1776_click.fde47b4b
        image_name = self._find_profile_name(repo_name)
        
        # If push is enabled and we have a custom org, try to retag/push existing image first
        # This is independent of SWE-smith - we only care about our org
        if docker_config and docker_config.push and docker_config.docker_hub_org:
            docker_hub_org = docker_config.docker_hub_org
            logger.info(f"Docker push enabled. Target Docker Hub org: {docker_hub_org}")
            
            # Check Docker Hub authentication
            auth_check = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if auth_check.returncode != 0:
                logger.warning("Docker daemon may not be running or accessible. Push may fail.")
            else:
                login_check = subprocess.run(
                    ["docker", "system", "info"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if "Username" not in login_check.stdout and "docker.io" not in login_check.stdout:
                    logger.warning("Docker Hub authentication not detected. Ensure 'docker login' has been run before pushing.")
            
            # Try to retag and push existing image to our org
            if self._retag_and_push_if_needed(image_name, docker_hub_org):
                logger.info(f"Successfully handled Docker image for {docker_hub_org}")
                return  # Done - image is in our org and pushed
            
            # If retag/push failed, we'll continue to build from scratch
            logger.info(f"Will build Docker image from scratch for {docker_hub_org}")
        
        # Build command arguments
        # create_images uses --profiles with the image_name from the registry
        cmd_args = [
            "--profiles", image_name,
        ]
        
        if docker_config and docker_config.push:
            cmd_args.append("--push")
            docker_hub_org = docker_config.docker_hub_org or "jyangballin"
            logger.info(f"Building and pushing to Docker Hub org: {docker_hub_org}")
        
        if docker_config and docker_config.options.get("workers"):
            cmd_args.extend(["--workers", str(docker_config.options["workers"])])
        
        if docker_config and docker_config.options.get("proceed"):
            cmd_args.append("--proceed")
        
        # Use wrapper script if Docker Hub org is configured (to patch constant in subprocess)
        # or if we have custom profiles that need registration
        use_wrapper = (
            (docker_config and docker_config.docker_hub_org) or
            (self.config.repository.custom_profile or self.config.custom_profiles)
        )
        
        if use_wrapper:
            wrapper_script = self._create_docker_build_wrapper_script(cmd_args)
            cmd = [sys.executable, str(wrapper_script)]
        else:
            cmd = [
                sys.executable, "-m", "swesmith.build_repo.create_images",
            ] + cmd_args
        
        logger.info(f"Building Docker image: {' '.join(cmd)}")
        if docker_config and docker_config.push:
            logger.info("Starting Docker build and push operation...")
        
        result = subprocess.run(cmd, cwd=str(self.workspace), capture_output=True, text=True)
        
        # Log output for debugging (more verbose for push operations)
        log_level = logger.info if (docker_config and docker_config.push) else logger.debug
        if result.stdout:
            log_level(f"Docker build stdout: {result.stdout}")
        if result.stderr:
            log_level(f"Docker build stderr: {result.stderr}")
        
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
            if docker_config and docker_config.push:
                docker_hub_org = docker_config.docker_hub_org or "jyangballin"
                logger.info(f"Docker push completed. Checking push status...")
                
                # Extract image name from output if possible
                image_name = None
                combined_output = result.stdout + result.stderr
                
                # Look for image push patterns in output
                # Pattern: "pushed" or "Pushed" followed by image name
                push_patterns = [
                    r'pushed\s+([^\s]+)',
                    r'Pushed\s+([^\s]+)',
                    r'Successfully\s+pushed\s+([^\s]+)',
                    r'([^\s]+docker\.io/[^\s]+)',
                ]
                
                for pattern in push_patterns:
                    match = re.search(pattern, combined_output, re.IGNORECASE)
                    if match:
                        image_name = match.group(1) if match.lastindex else match.group(0)
                        break
                
                # Check for push success indicators
                push_succeeded = False
                push_failed = False
                
                if "pushed" in combined_output.lower() or "successfully pushed" in combined_output.lower():
                    push_succeeded = True
                    logger.info(f"Docker image push succeeded!")
                    if image_name:
                        logger.info(f"Pushed image: {image_name}")
                    else:
                        logger.info(f"Docker image should be available at: docker.io/{docker_hub_org}/<repo-name>")
                
                # Check for push failure indicators
                failure_keywords = ["denied", "unauthorized", "authentication", "login", "unauthorized: authentication required"]
                if any(keyword in combined_output.lower() for keyword in failure_keywords):
                    push_failed = True
                    logger.error("Docker push failed due to authentication error!")
                    logger.error("Please ensure you have run 'docker login' and are authenticated to Docker Hub.")
                    logger.error(f"Expected Docker Hub org: {docker_hub_org}")
                
                if "error" in combined_output.lower() and not push_succeeded:
                    logger.warning("Docker push may have encountered errors. Check the output above for details.")
                
                if not push_succeeded and not push_failed:
                    logger.warning("Could not determine Docker push status from output. Check Docker Hub manually.")
                    logger.info(f"Expected image location: docker.io/{docker_hub_org}/<repo-name>")
    
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
        """Extract repo name in SWE-smith format (owner__repo.commit).
        
        If custom_profile is provided, uses that. Otherwise constructs from github_url.
        """
        if self.config.repository.custom_profile:
            # Use custom profile format: owner__repo.commit (first 8 chars)
            cp = self.config.repository.custom_profile
            commit_short = cp.commit[:8] if len(cp.commit) >= 8 else cp.commit
            return f"{cp.owner}__{cp.repo}.{commit_short}"
        
        # Fallback: construct from github_url
        repo_url = self.config.repository.github_url
        # Remove https://github.com/ if present
        repo_url = repo_url.replace("https://github.com/", "").replace("http://github.com/", "")
        # Replace / with __ for SWE-smith format
        owner, repo = repo_url.split("/", 1)
        commit = self.config.repository.commit or "HEAD"
        commit_short = commit[:8] if len(commit) >= 8 else commit
        return f"{owner}__{repo}.{commit_short}"
    
    def _build_custom_profile_image(self, repo_name: str, docker_config) -> None:
        """Build Docker image for a custom profile (not in SWE-smith's registry).
        
        This directly uses the profile's build_image(), create_mirror(), and push_image() methods.
        """
        try:
            from swesmith.profiles import registry
            
            # The custom profile should already be registered by _register_custom_profiles()
            if repo_name not in registry:
                raise TaskGenerationError(f"Custom profile {repo_name} not found in registry. Registration may have failed.")
            
            profile_class = registry[repo_name]
            profile = profile_class()
            
            logger.info(f"Building Docker image for custom profile: {profile.image_name}")
            
            # Create mirror repository on GitHub (if it doesn't exist)
            try:
                logger.info(f"Creating mirror repository: {profile.mirror_name}")
                profile.create_mirror()
                logger.info(f"Mirror repository created: {profile.mirror_name}")
            except Exception as e:
                logger.warning(f"Mirror creation failed or already exists: {e}")
                # Continue anyway - mirror might already exist
            
            # Build the Docker image with custom method for custom profiles
            logger.info(f"Building Docker image: {profile.image_name}")
            logger.info(f"Profile class: {profile.__class__.__name__}")
            
            # Check if this is a Python profile (original or custom subclass)
            from swesmith.profiles.python import PythonProfile
            if isinstance(profile, PythonProfile):
                logger.info("Detected Python profile, using custom build method")
                self._build_custom_python_image(profile)
            else:
                logger.info(f"Non-Python profile ({profile.__class__.__name__}), using standard build")
                profile.build_image()
            
            logger.info(f"Docker image built successfully: {profile.image_name}")
            
            # Push to Docker Hub if requested
            if docker_config and docker_config.push:
                # Use custom image name if we built with a custom org
                image_name_to_push = getattr(self, '_custom_image_name', profile.image_name)
                logger.info(f"Attempting to push Docker image to Docker Hub: {image_name_to_push}")
                
                try:
                    import docker
                    client = docker.from_env()
                    
                    # Verify image exists
                    try:
                        image = client.images.get(image_name_to_push)
                        logger.info(f"Found image to push: {image.id[:12]}")
                    except docker.errors.ImageNotFound:
                        logger.warning(f"Image not found locally: {image_name_to_push}, skipping push")
                        events.emit(f"Docker image not found locally, skipping push", level="warning", code="docker.push_skipped")
                        return  # Exit the method, don't fail
                    
                    # Push the image
                    logger.info(f"Pushing {image_name_to_push} to Docker Hub...")
                    push_successful = False
                    for line in client.images.push(image_name_to_push, stream=True, decode=True):
                        if 'status' in line:
                            status = line['status']
                            if 'progress' in line:
                                logger.debug(f"Docker push: {status} {line['progress']}")
                            else:
                                logger.info(f"Docker push: {status}")
                        if 'error' in line:
                            error_msg = line['error']
                            # Common errors that are user-fixable, not pipeline-breaking
                            if 'denied' in error_msg.lower() or 'not found' in error_msg.lower():
                                logger.warning(f"Docker push failed: {error_msg}")
                                logger.warning("This is likely because:")
                                logger.warning(f"  1. The repository doesn't exist on Docker Hub yet")
                                logger.warning(f"  2. You need to create it at: https://hub.docker.com/repositories")
                                logger.warning(f"  3. Or ensure you're logged in: docker login")
                                logger.warning("The pipeline will continue without pushing the image.")
                                events.emit(f"Docker push failed (non-critical): {error_msg}", level="warning", code="docker.push_failed")
                                return  # Don't raise, just continue
                            else:
                                # Unexpected error, but still don't fail the pipeline
                                logger.error(f"Unexpected Docker push error: {error_msg}")
                                events.emit(f"Docker push error: {error_msg}", level="warning", code="docker.push_error")
                                return
                        if 'status' in line and line['status'].startswith('Pushed'):
                            push_successful = True
                    
                    if push_successful:
                        logger.info(f"✓ Docker image pushed successfully: {image_name_to_push}")
                        events.emit(f"Docker image pushed to Docker Hub", code="docker.push_success")
                    else:
                        logger.warning(f"Docker push may have failed (no error reported)")
                        
                except Exception as e:
                    # Catch any other Docker push errors and continue
                    logger.warning(f"Docker push failed: {e}")
                    logger.warning("The pipeline will continue - the Docker image is still available locally")
                    events.emit(f"Docker push failed (non-critical): {str(e)}", level="warning", code="docker.push_exception")
                    # Don't raise - continue the pipeline
        
        except Exception as e:
            raise TaskGenerationError(f"Failed to build custom profile image: {e}")
    
    def _build_custom_python_image(self, profile) -> None:
        """Build Docker image for custom Python profile without requiring pre-generated env.yml."""
        import docker
        from pathlib import Path
        from swebench.harness.docker_build import build_image as build_image_sweb
        from swebench.harness.dockerfiles import get_dockerfile_env
        from swesmith.constants import ENV_NAME, LOG_DIR_ENV
        
        BASE_IMAGE_KEY = "jyangballin/swesmith.x86_64"
        client = docker.from_env()
        python_version = getattr(profile, 'python_version', '3.10')
        
        # Override image_name with correct Docker Hub org if configured
        docker_hub_org = None
        if self.config.docker_image and self.config.docker_image.docker_hub_org:
            docker_hub_org = self.config.docker_image.docker_hub_org
        
        if docker_hub_org and docker_hub_org != "jyangballin":
            # Replace the org in the image name
            original_image_name = profile.image_name
            # Image name format: jyangballin/swesmith.x86_64.python_1776_typing_extensions.479dae13
            if "/" in original_image_name:
                image_base = original_image_name.split("/", 1)[1]
                image_name = f"{docker_hub_org}/{image_base}"
                logger.info(f"Overriding Docker Hub org: {original_image_name} -> {image_name}")
            else:
                image_name = original_image_name
        else:
            image_name = profile.image_name
        
        setup_commands = [
            "#!/bin/bash",
            "set -euxo pipefail",
            f"git clone -o origin https://github.com/{profile.mirror_name} /{ENV_NAME}",
            f"cd /{ENV_NAME}",
            "source /opt/miniconda3/bin/activate",
            f"conda create -n {ENV_NAME} python={python_version} -y",
            f"conda activate {ENV_NAME}",
            'echo "Current environment: $CONDA_DEFAULT_ENV"',
        ] + profile.install_cmds
        
        dockerfile = get_dockerfile_env(profile.pltf, profile.arch, "py", base_image_key=BASE_IMAGE_KEY)
        build_dir = LOG_DIR_ENV / profile.repo_name
        build_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Building custom Python Docker image: {image_name}")
        build_image_sweb(
            image_name=image_name,
            setup_scripts={"setup_env.sh": "\n".join(setup_commands) + "\n"},
            dockerfile=dockerfile,
            platform=profile.pltf,
            client=client,
            build_dir=build_dir,
        )
        logger.info(f"Custom Python Docker image built: {image_name}")
        
        # Store the custom image name for later use in push
        self._custom_image_name = image_name
    
    def _find_profile_name(self, repo_name: str) -> str:
        """Find the correct profile image name from the SWE-smith registry.
        
        For repos in the registry, we need to find the profile's image_name.
        e.g., jyangballin/swesmith.x86_64.pallets_1776_click.fde47b4b
        """
        # Try to find the profile in the registry
        try:
            from swesmith.profiles import registry
            
            # Look for a matching profile by repo_name key
            if repo_name in registry:
                profile_class = registry[repo_name]
                profile = profile_class()
                image_name = profile.image_name
                logger.info(f"Found profile in registry: {repo_name} -> {image_name}")
                return image_name
            else:
                logger.warning(f"Profile {repo_name} not found in registry")
        except Exception as e:
            logger.warning(f"Could not access profile registry: {e}")
        
        # Fallback: use our repo_name format
        logger.info(f"Using repo_name as profile: {repo_name}")
        return repo_name
    
    def _retag_and_push_if_needed(self, source_image_name: str, target_org: str) -> bool:
        """Retag and push Docker image to target org if it doesn't exist there.
        
        This is independent of SWE-smith - we only care about our org.
        Returns True if image was successfully handled, False if we need to build.
        """
        try:
            import docker
            client = docker.from_env()
            
            # Extract the image name parts
            # source_image_name format: jyangballin/swesmith.x86_64.pallets_1776_click.fde47b4b
            if '/' not in source_image_name:
                logger.warning(f"Invalid image name format: {source_image_name}")
                return False
            
            source_org, image_suffix = source_image_name.split('/', 1)
            target_image_name = f"{target_org}/{image_suffix}"
            
            # Check if image already exists in target org
            try:
                client.images.get(target_image_name)
                logger.info(f"Image already exists in target org: {target_image_name}")
                logger.info(f"Pushing existing image to Docker Hub: {target_image_name}")
                
                # Push the existing image
                push_successful = False
                error_message = None
                
                try:
                    for line in client.images.push(target_image_name, stream=True, decode=True):
                        logger.debug(f"Push stream: {line}")
                        
                        # Check for errors
                        if 'error' in line:
                            error_message = line.get('error', 'Unknown error')
                            logger.error(f"Push error: {error_message}")
                            break
                        
                        # Check for progress
                        if 'status' in line:
                            status = line['status']
                            if 'Pushed' in status or 'Layer already exists' in status:
                                logger.debug(f"Push progress: {status}")
                            # Check for final success message (digest indicates successful push)
                            if 'digest:' in status or ': digest:' in status:
                                push_successful = True
                                logger.info(f"Push completed: {status}")
                
                except Exception as push_error:
                    error_message = str(push_error)
                    logger.error(f"Exception during push: {push_error}")
                
                if error_message:
                    logger.error(f"Failed to push {target_image_name}: {error_message}")
                    return False
                
                if not push_successful:
                    logger.warning(f"Push stream completed but no digest confirmation received for {target_image_name}")
                    logger.warning("Push may have failed silently - returning False to trigger rebuild")
                    return False
                
                logger.info(f"Successfully pushed {target_image_name} to Docker Hub")
                return True
            
            except docker.errors.ImageNotFound:
                # Image doesn't exist in target org, check if source exists
                logger.info(f"Image not found in target org: {target_image_name}")
                
                try:
                    # Check if source image exists
                    source_image = client.images.get(source_image_name)
                    logger.info(f"Found source image: {source_image_name}")
                    logger.info(f"Retagging {source_image_name} -> {target_image_name}")
                    
                    # Retag the image
                    source_image.tag(target_image_name)
                    logger.info(f"Successfully retagged image to {target_image_name}")
                    
                    # Push the retagged image
                    logger.info(f"Pushing retagged image to Docker Hub: {target_image_name}")
                    push_successful = False
                    error_message = None
                    
                    try:
                        for line in client.images.push(target_image_name, stream=True, decode=True):
                            logger.debug(f"Push stream: {line}")
                            
                            # Check for errors
                            if 'error' in line:
                                error_message = line.get('error', 'Unknown error')
                                logger.error(f"Push error: {error_message}")
                                break
                            
                            # Check for progress
                            if 'status' in line:
                                status = line['status']
                                if 'Pushed' in status or 'Layer already exists' in status:
                                    logger.debug(f"Push progress: {status}")
                                # Check for final success message (digest indicates successful push)
                                if 'digest:' in status or ': digest:' in status:
                                    push_successful = True
                                    logger.info(f"Push completed: {status}")
                    
                    except Exception as push_error:
                        error_message = str(push_error)
                        logger.error(f"Exception during push: {push_error}")
                    
                    if error_message:
                        logger.error(f"Failed to push retagged {target_image_name}: {error_message}")
                        return False
                    
                    if not push_successful:
                        logger.warning(f"Push stream completed but no digest confirmation received for retagged {target_image_name}")
                        logger.warning("Push may have failed silently - returning False")
                        return False
                    
                    logger.info(f"Successfully pushed {target_image_name} to Docker Hub")
                    return True
                
                except docker.errors.ImageNotFound:
                    # Source image doesn't exist either, need to build
                    logger.info(f"Source image not found: {source_image_name}")
                    logger.info(f"Will need to build image from scratch")
                    return False
        
        except Exception as e:
            logger.warning(f"Failed to retag/push image: {e}")
            logger.warning("Will attempt to build image from scratch")
            return False

    def _retag_and_push_if_needed(self, source_image_name: str, target_org: str) -> bool:
        """Retag and push Docker image to target org if it doesn't exist there.
        
        This is independent of SWE-smith - we only care about our org.
        Returns True if image was successfully handled, False if we need to build.
        """
        try:
            import docker
            client = docker.from_env()
            
            # Extract the image name parts
            # source_image_name format: jyangballin/swesmith.x86_64.pallets_1776_click.fde47b4b
            if '/' not in source_image_name:
                logger.warning(f"Invalid image name format: {source_image_name}")
                return False
            
            source_org, image_suffix = source_image_name.split('/', 1)
            target_image_name = f"{target_org}/{image_suffix}"
            
            # Check if image already exists in target org
            try:
                client.images.get(target_image_name)
                logger.info(f"Image already exists in target org: {target_image_name}")
                logger.info(f"Pushing existing image to Docker Hub: {target_image_name}")
                
                # Push the existing image
                push_successful = False
                error_message = None
                
                try:
                    for line in client.images.push(target_image_name, stream=True, decode=True):
                        logger.debug(f"Push stream: {line}")
                        
                        # Check for errors
                        if 'error' in line:
                            error_message = line.get('error', 'Unknown error')
                            logger.error(f"Push error: {error_message}")
                            break
                        
                        # Check for progress
                        if 'status' in line:
                            status = line['status']
                            if 'Pushed' in status or 'Layer already exists' in status:
                                logger.debug(f"Push progress: {status}")
                            # Check for final success message (digest indicates successful push)
                            if 'digest:' in status or ': digest:' in status:
                                push_successful = True
                                logger.info(f"Push completed: {status}")
                
                except Exception as push_error:
                    error_message = str(push_error)
                    logger.error(f"Exception during push: {push_error}")
                
                if error_message:
                    logger.error(f"Failed to push {target_image_name}: {error_message}")
                    return False
                
                if not push_successful:
                    logger.warning(f"Push stream completed but no digest confirmation received for {target_image_name}")
                    logger.warning("Push may have failed silently - returning False to trigger rebuild")
                    return False
                
                logger.info(f"Successfully pushed {target_image_name} to Docker Hub")
                return True
            
            except docker.errors.ImageNotFound:
                # Image doesn't exist in target org, check if source exists
                logger.info(f"Image not found in target org: {target_image_name}")
                
                try:
                    # Check if source image exists
                    source_image = client.images.get(source_image_name)
                    logger.info(f"Found source image: {source_image_name}")
                    logger.info(f"Retagging {source_image_name} -> {target_image_name}")
                    
                    # Retag the image
                    source_image.tag(target_image_name)
                    logger.info(f"Successfully retagged image to {target_image_name}")
                    
                    # Push the retagged image
                    logger.info(f"Pushing retagged image to Docker Hub: {target_image_name}")
                    push_successful = False
                    error_message = None
                    
                    try:
                        for line in client.images.push(target_image_name, stream=True, decode=True):
                            logger.debug(f"Push stream: {line}")
                            
                            # Check for errors
                            if 'error' in line:
                                error_message = line.get('error', 'Unknown error')
                                logger.error(f"Push error: {error_message}")
                                break
                            
                            # Check for progress
                            if 'status' in line:
                                status = line['status']
                                if 'Pushed' in status or 'Layer already exists' in status:
                                    logger.debug(f"Push progress: {status}")
                                # Check for final success message (digest indicates successful push)
                                if 'digest:' in status or ': digest:' in status:
                                    push_successful = True
                                    logger.info(f"Push completed: {status}")
                    
                    except Exception as push_error:
                        error_message = str(push_error)
                        logger.error(f"Exception during push: {push_error}")
                    
                    if error_message:
                        logger.error(f"Failed to push retagged {target_image_name}: {error_message}")
                        return False
                    
                    if not push_successful:
                        logger.warning(f"Push stream completed but no digest confirmation received for retagged {target_image_name}")
                        logger.warning("Push may have failed silently - returning False")
                        return False
                    
                    logger.info(f"Successfully pushed {target_image_name} to Docker Hub")
                    return True
                
                except docker.errors.ImageNotFound:
                    # Source image doesn't exist either, need to build
                    logger.info(f"Source image not found: {source_image_name}")
                    logger.info(f"Will need to build image from scratch")
                    return False
        
        except Exception as e:
            logger.warning(f"Failed to retag/push image: {e}")
            logger.warning("Will attempt to build image from scratch")
            return False

