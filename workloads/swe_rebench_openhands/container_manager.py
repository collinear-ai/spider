from __future__ import annotations

from dataclasses import dataclass
import time
import json
import logging
import os
import shlex
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Pre-built swerebench images have conda environment already configured
SWEREBENCH_IMAGE_PREFIX = "swerebench/"

# Conda activation commands to prefix before each command
CONDA_ACTIVATE_PREFIX = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && "

def _run(cmd: Sequence[str], *, check: bool = True, text: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=text, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def _sanitize_container_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")

def _short_image_name(image: str) -> str:
    name = image.rsplit("/", 1)[-1]
    name = name.split(":", 1)[0]
    return name[-12:] if len(name) > 12 else name

def _ensure_str(value: Any, name: str) -> str:
    if not value:
        raise ValueError(f"missing required field: {name}")
    return str(value)

def image_exists(image: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode == 0

@dataclass
class ContainerSpec:
    instance_id: str
    docker_image: str
    image_name: Optional[str] = None
    install_config: Optional[Dict[str, Any]] = None

def _is_prebuilt_swerebench_image(image: str) -> bool:
    """Check if the image is a pre-built swerebench image.

    Pre-built images have the conda environment and dependencies already installed,
    so we don't need to run install_config commands.
    """
    return image.startswith(SWEREBENCH_IMAGE_PREFIX)


class ContainerManager:
    def __init__(
        self,
        spec: ContainerSpec,
        *,
        workdir: str = "/testbed",
    ) -> None:
        self.spec = spec
        self.workdir = workdir
        self._is_prebuilt = _is_prebuilt_swerebench_image(spec.docker_image)

        run_id = str(time.time_ns())
        image_short = _short_image_name(spec.docker_image)
        raw_name = f"swe-rebench-{spec.instance_id}-{image_short}-{run_id}"
        self.container_name = _sanitize_container_name(raw_name)

    @classmethod
    def from_row(cls, row: Dict[str, Any], *, workdir: str = "/testbed") -> "ContainerManager":
        spec = ContainerSpec(
            instance_id=_ensure_str(row.get("instance_id"), "instance_id"),
            docker_image=_ensure_str(row.get("docker_image") or row.get("image_name"), "docker_image"),
            image_name=row.get("image_name"),
            install_config=row.get("install_config"),
        )
        return cls(spec, workdir=workdir)

    def create(self) -> None:
        image = self.spec.docker_image
        if not image_exists(image):
            logger.info("Pulling Docker image: %s", image)
            _run(["docker", "pull", image])
        else:
            logger.info("Using cached Docker image: %s", image)

        proc = _run([
            "docker", "run", "-d",
            "--name", self.container_name,
            "-w", self.workdir,
            image,
            "sleep", "infinity",
        ])

        # Pre-built swerebench images already have the environment configured
        # Running install_config on them is unnecessary and may cause errors
        # due to Python version mismatches (e.g., packages requiring Python < 3.12)
        if self._is_prebuilt:
            logger.info(
                "Skipping install_config for pre-built swerebench image: %s",
                image,
            )
        else:
            self._apply_install_config()

    def cleanup(self) -> None:
        _run(["docker", "rm", "-f", self.container_name], check=False)

    def exec_bash(
        self,
        command: str,
        *,
        timeout: Optional[float] = None,
        activate_conda: Optional[bool] = None,
    ) -> str:
        """Execute a bash command in the container.

        Args:
            command: The bash command to execute.
            timeout: Optional timeout in seconds.
            activate_conda: Whether to activate the testbed conda environment before
                running the command. If None (default), activates for pre-built
                swerebench images automatically.
        """
        # For pre-built swerebench images, ensure conda environment is activated
        # The images have .bashrc configured, but bash -lc doesn't always source it
        should_activate = activate_conda if activate_conda is not None else self._is_prebuilt
        if should_activate:
            command = CONDA_ACTIVATE_PREFIX + command

        cmd = [
            "docker", "exec",
            "-w", self.workdir,
            self.container_name,
            "bash", "-c", command
        ]
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout.strip())
        return proc.stdout

    def read_file(
        self,
        path: str,
        *,
        view_range: Optional[List[int]] = None,
    ) -> str:
        if view_range:
            start, end = view_range
            if end == -1:
                cmd = f"sed -n '{start},$p' {shlex.quote(path)}"
            else:
                cmd = f"sed -n '{start},{end}p' {shlex.quote(path)}"
            return self.exec_bash(cmd)
        return self.exec_bash(f"cat -n {shlex.quote(path)}")

    def write_file(
        self,
        path: str,
        content: str,
        *,
        create: bool = False,
    ) -> None:
        if create:
            self.exec_bash(f"test ! -e {shlex.quote(path)}")
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            _run(["docker", "cp", tmp_path, f"{self.container_name}:{path}"])
        finally:
            os.unlink(tmp_path)

    def list_dir(self, path: str) -> List[str]:
        out = self.exec_bash(f"ls -a {shlex.quote(path)}")
        return [line for line in out.splitlines() if line.strip()]

    def is_dir(self, path: str) -> bool:
        try:
            self.exec_bash(f"test -d {shlex.quote(path)}")
            return True
        except Exception:
            return False

    def is_file(self, path: str) -> bool:
        try:
            self.exec_bash(f"test -f {shlex.quote(path)}")
            return True
        except Exception:
            return False

    def replace_in_file(self, path: str, old: str, new: str) -> int:
        if old == "":
            raise ValueError("old_str must be non-empty")
        count = int(self.exec_bash(
            "python - <<'PY'\n"
            "import re\n"
            "import sys\n"
            f"path={path!r}\n"
            f"old={old!r}\n"
            f"text=open(path, 'r', encoding='utf-8').read()\n"
            f"print(len(re.findall(re.escape(old), text)))\n"
            "PY"
        ).strip())
        if count == 0:
            return 0

        self.exec_bash(
            "python - <<'PY'\n"
            "path={path!r}\n"
            "old={old!r}\n"
            "new={new!r}\n"
            "text=open(path, 'r', encoding='utf-8').read()\n"
            "text=text.replace(old, new)\n"
            "open(path, 'w', encoding='utf-8').write(text)\n"
            "PY".format(path=path, old=old, new=new)
        )
        return count

    def insert_in_file(self, path: str, insert_line: int, new: str) -> None:
        self.exec_bash(
            "python - <<'PY'\n"
            "path={path!r}\n"
            "insert_line={insert_line}\n"
            "new={new!r}\n"
            "lines=open(path, 'r', encoding='utf-8').read().splitlines()\n"
            "idx=max(0, min(len(lines), insert_line))\n"
            "lines.insert(idx, new)\n"
            "open(path, 'w', encoding='utf-8').write('\\n'.join(lines) + ('\\n' if lines else ''))\n"
            "PY".format(path=path, insert_line=insert_line, new=new)
        )

    def apply_patch(self, path: str, patch: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(patch)
            tmp_path = tmp.name
        try:
            _run(["docker", "cp", tmp_path, f"{self.container_name}:{path}.patch"])
        finally:
            os.unlink(tmp_path)
        return self.exec_bash(f"patch -p0 -i {shlex.quote(path)}.patch")

    def _apply_install_config(self) -> None:
        cfg = self.spec.install_config or {}
        env_vars = cfg.get("env_vars") or {}
        for key, value in env_vars.items():
            self.exec_bash(f"export {shlex.quote(key)}={shlex.quote(str(value))}")
        pre_install = cfg.get("pre_install") or []
        for cmd in pre_install:
            self.exec_bash(str(cmd))
        install_cmd = cfg.get("install")
        if install_cmd:
            self.exec_bash(str(install_cmd))
        pip_packages = cfg.get("pip_packages") or []

        if pip_packages:
            # Handle both list and string formats
            if isinstance(pip_packages, str):
                # If it's a string, split it into a list
                pip_packages = pip_packages.split()
            # Join packages with spaces - pip install expects them as separate arguments
            # Package names are safe, so no need to quote them
            packages_str = ' '.join(str(pkg) for pkg in pip_packages)
            self.exec_bash(f"pip install {packages_str}")