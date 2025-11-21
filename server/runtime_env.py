from __future__ import annotations

import contextlib, os, shutil, subprocess, sys, tempfile, threading
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List, Deque
from collections import deque

class RuntimeEnvironmentError(Exception):
    pass

class RuntimeEnvironment:
    def __init__(self, *, python_executable: Optional[str] = None) -> None:
        self._python = python_executable or sys.executable
        self._workspace = Path(tempfile.mkdtemp(prefix="spider-runtime-"))
        self._venv_dir = self._workspace / "venv"
        self._bin_dir = self._venv_dir / ("Scripts" if os.name == "nt" else "bin")
        self._site_packages: Optional[Path] = None
        self._lock = threading.RLock()
        self._thread_state = threading.local()
        self._global_snapshot: Optional[Tuple[Dict[str, str], List[str]]] = None
        self._cwd_snapshot: Optional[Path] = None

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
        state_stack: Deque[int] = getattr(self._thread_state, "stack", deque())
        setattr(self._thread_state, "stack", state_stack)
        reentrant = bool(state_stack)
        token = object()
        state_stack.append(token)

        if not reentrant:
            self._enter_global(extra_env)
        try:
            yield
        finally:
            popped = state_stack.pop()
            if popped is not token:
                state_stack.clear()
                raise RuntimeEnvironmentError("Runtime activation stack corrupted.")
            if not state_stack:
                self._exit_global()

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

    def _enter_global(self, extra_env: Optional[Dict[str, str]]) -> None:
        with self._lock:
            if self._global_snapshot is not None:
                return
            previous_env = os.environ.copy()
            previous_path = list(sys.path)
            self._global_snapshot = (previous_env, previous_path)
            self._cwd_snapshot = Path.cwd()
            os.environ["VIRTUAL_ENV"] = str(self._venv_dir)
            os.environ["PATH"] = f"{self._bin_dir}{os.pathsep}{previous_env.get('PATH', '')}"
            if extra_env:
                os.environ.update(extra_env)
            os.environ["HOME"] = str(self._workspace)
            os.environ["SPIDER_SANDBOX_ROOT"] = str(self._workspace)
            sys.path[:] = [str(self._workspace) + [str(self._site_packages)] + previous_path]
            os.chdir(self._workspace)

    def _exit_global(self) -> None:
        with self._lock:
            if not self._global_snapshot:
                return
            if self._cwd_snapshot is not None:
                os.chdir(self._cwd_snapshot)
                self._cwd_snapshot = None
            previous_env, previous_path = self._global_snapshot
            os.environ.clear()
            os.environ.update(previous_env)
            sys.path[:] = previous_path
            self._global_snapshot = None
