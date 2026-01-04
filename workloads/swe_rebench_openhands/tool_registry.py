from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Dict, List, Optional, Protocol

ToolFn = Callable[[Dict[str, Any]], str]

class RuntimeAdapter(Protocol):
    def exec_bash(
        self,
        command: str,
        *,
        timeout: Optional[float] = None,
        is_input: bool = False,
    ) -> str:
        ...

    def read_file(
        self,
        path: str,
        *,
        view_range: Optional[List[int]] = None,
    ) -> str:
        ...

    def write_file(
        self,
        path: str,
        content: str,
        *,
        create: bool = False,
    ) -> None:
        ...


    def list_dir(
        self,
        path: str,
    ) -> List[str]:
        ...

    def is_dir(
        self,
        path: str,
    ) -> bool:
        ...

    def is_file(
        self,
        path: str,
    ) -> bool:
        ...

    def replace_in_file(
        self,
        path: str,
        old: str,
        new: str,
    ) -> int:
        ...

    def insert_in_file(
        self,
        path: str,
        insert_line: int,
        new: str
    ) -> None:
        ...

    def apply_patch(
        self,
        path: str,
        patch: str,
    ) -> str:
        ...


@dataclass
class TaskTracker:
    tasks: List[Dict[str, Any]] = field(default_factory=list)

    def view(self) -> str:
        return json.dumps(self.tasks, indent=2, sort_keys=True)

    def plan(self, task_list: List[Dict[str, Any]]) -> str:
        self.tasks = task_list
        return json.dumps(self.tasks, indent=2, sort_keys=True)

@dataclass
class ToolRegistry:
    runtime: Optional[RuntimeAdapter] = None
    task_tracker: TaskTracker = field(default_factory=TaskTracker)

    def build(self) -> Dict[str, ToolFn]:
        return {
            "execute_bash": self._execute_bash,
            "str_replace_editor": self._str_replace_editor,
            "think": self._think,
            "finish": self._finish,
            "task_tracker": self._task_tracker,
        }

    def _require_runtime(self) -> RuntimeAdapter:
        if self.runtime is None:
            raise RuntimeError("Tool runtime is not configured")
        return self.runtime

    def _execute_bash(self, args: Dict[str, Any]) -> str:
        runtime = self._require_runtime()
        command = args.get("command", "")
        timeout = args.get("timeout")
        is_input = str(args.get("is_input", "false")).lower() == "true"
        return runtime.exec_bash(command, timeout=timeout, is_input=is_input)

    def _str_replace_editor(self, args: Dict[str, Any]) -> str:
        runtime = self._require_runtime()
        command = args.get("command")
        path = args.get("path")
        if not command or not path:
            raise ValueError("str_replace_editor requires 'command' and 'path'.")

        if command == "view":
            view_range = args.get("view_range")
            if runtime.is_dir(path):
                entries = runtime.list_dir(path)
                return "\n".join(entries)
            if runtime.is_file(path):
                return runtime.read_file(path, view_range=view_range)
            raise ValueError(f"view requires existing file or directory: {path}")

        if command == "create":
            file_text = args.get("file_text", "")
            runtime.write_file(path, file_text, create=True)
            return "created"

        if command == "str_replace":
            old_str = args.get("old_str", "")
            new_str = args.get("new_str", "")

            count = runtime.replace_in_file(path, old_str, new_str)
            return f"replaced {count} occurrence(s)"

        if command == "insert":
            new_str = args.get("new_str", "")
            insert_line = args.get("insert_line")
            if insert_line is None:
                raise ValueError("insert requires 'insert_line'.")
            runtime.insert_in_file(path, int(insert_line), new_str)
            return "inserted"

        if command == "undo_edit":
            return "undo not supported in this runtime"

        raise ValueError(f"Unsupported str_replace_editor command: {command}")

    def _finish(self, args: Dict[str, Any]) -> str:
        return str(args.get("message", ""))

    def _task_tracker(self, args: Dict[str, Any]) -> str:
        command = args.get("command")
        if command == "view":
            return self.task_tracker.view()
        if command == "plan":
            task_list = args.get("task_list", [])
            return self.task_tracker.plan(task_list)

        raise ValueError(f"Unsupported task_tracker command: {command}")