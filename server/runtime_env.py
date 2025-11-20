from __future__ import annotations

import contextlib, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional

class RuntimeEnvironmentError(Exception):
    pass

class RuntimeEnvironment:
    def __init__(self, *, python_executable: Optional[str] = None) -> None:
        self._python = python_executable or sys.executable
        self._workspace = Path(tempfile.mkdtemp(prefix="spider-runtime-"))
        self._venv_dir = self._workspace / "venv"
        self._bin_dir = self._venv_dir / ("Scripts" if os.name == "nt" else "bin")
        self._site_packages: Optional[Path] = None

    def create(self) -> None:
        self._run([self._python, "-m", "venv", str(self._venv_dir)])
        self._site_packages = self._detect_site_packages()

    def install(self, packages: Iterable[str]) -> None:
        pkgs = [pkg for pkg in packages or [] if pkg]
        if not pkgs:
            return
        pip = self._bin_dir / ("pip.exe" if os.name == "nt" else "pip")
        self._run([str(pip), "install", "--upgrade", "pip"])
        self._run([str(pip), "install", *pkgs])

    @contextlib.contextmanager
    def activate(self, extra_env: Optional[Dict[str, str]] = None):
        if not self._site_packages:
            raise RuntimeEnvironmentError("Runtime environment not created.")
        previous_env = os.environ.copy()
        previous_path = list(sys.path)
        try:
            os.environ["VIRTUAL_ENV"] = str(self._venv_dir)
            os.environ["PATH"] = f"{self._bin_dir}{os.pathsep}{previous_env.get('PATH', '')}"
            if extra_env:
                os.environ.update(extra_env)
            sys.path.insert(0, str(self._site_packages))
            yield
        finally:
            os.environ.clear()
            os.environ.update(previous_env)
            sys.path[:] = previous_path

    def cleanup(self) -> None:
        shutil.rmtree(self._workspace, ignore_errors=True)

    def _detect_site_packages(self) -> Path:
        if os.name == "nt":
            candidate = self._venv_dir / "Lib" / "site-packages"
        else:
            version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            candidate = self._venv_dir / "lib" / version / "site-packages"
        if not candidate.exists():
            raise RuntimeEnvironmentError(f"Unable to locate site-packages at {candidate}")
        return candidate

    def _run(self, cmd: Iterable[str]) -> None:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeEnvironmentError(f"COmmand failed: {' '.join(cmd)}") from exc

