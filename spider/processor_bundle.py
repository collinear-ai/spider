from __future__ import annotations

import ast, builtins, inspect, textwrap
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional

__all__ = ["bundle_processor_source", "ProcessorBundlingError"]

class ProcessorBundlingError(ValueError):
    pass

@dataclass
class _Statement:
    source: str
    lineno: int

class _ModuleIndex:
    def __init__(self, source: str) -> None:
        self.source = source
        self.functions = {}
        self.imports = {}
        tree = ast.parse(source)
        for node in tree.body:
            segment = _statement_source(source, node)
            lineno = getattr(node, "lineno", 0)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[node.name] = _Statement(segment, lineno)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    self.imports[name] = _Statement(segment, lineno)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    self.imports[name] = _Statement(segment, lineno)

def bundle_processor_source(func: Callable) -> str:
    if not callable(func):
        raise ProcessorBundlingError("Processor must be callable.")
    
    module = inspect.getmodule(func)
    if module is None:
        raise ProcessorBundlingError("Processor must be defined in a module.")

    module_file = inspect.getsourcefile(func) or getattr(module, "__file__", None)
    if not module_file:
        raise ProcessorBundlingError("Unable to locate the processor's source file.")

    module_path = Path(module_file).resolve()

    try:
        module_source = module_path.read_text()
    except OSError as exc:
        raise ProcessorBundlingError(f"Failed to read processor module: {exc}") from exc

    index = _ModuleIndex(module_source)
    required_imports = {}
    required_functions = {}
    queue = deque([func])
    seen = set()

    while queue:
        current = queue.popleft()
        name = getattr(current, "__name__", None)
        if not name:
            raise ProcessorBundlingError("Every processor must have a __name__.")
        if name in seen:
            continue
        if not _defined_in_module(current, module_path):
            raise ProcessorBundlingError(f"Function `{name}` must be defined in {module_path} to be bundled.")

        statement = index.functions.get(name)
        if statement is None:
            raise ProcessorBundlingError(f"Function `{name}` was not found at the top level of {module_path}.")

        required_functions[name] = statement
        seen.add(name)

        for dep_name, obj in _function_dependency_map(current).items():
            if dep_name in _SKIP_NAMES:
                continue
            if dep_name in index.functions and _defined_in_module(obj, module_path):
                if dep_name not in seen:
                    queue.append(obj)
                continue
            if dep_name in index.imports:
                required_imports[dep_name] = index.imports[dep_name]
                continue
            raise ProcessorBundlingError(
                f"Processor `{name}` references `{dep_name}` which is not a "
                f"function or import defined in {module_path}."
            )

    import_block = _render_block(required_imports.values())
    function_block = _render_block(required_functions.values(), separator="\n\n")
    sections = [block for block in (import_block, function_block) if block]
    bundled = "\n\n".join(sections).strip()
    return bundled + ("\n" if bundled else "")

def _render_block(statements: Iterable[_Statement], *, separator: str = "\n") -> str:
    snippets = [stmt.source for stmt in sorted(statements, key=lambda item: item.lineno)]
    return separator.join(snippets).strip()

def _statement_source(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        lines = source.splitlines()
        start = max(getattr(node, "lineno", 1) - 1, 0)
        end = getattr(node, "end_lineno", start + 1)
        segment = "\n".join(lines[start:end])
    return textwrap.dedent(segment).rstrip()

def _function_dependency_map(func: Callable) -> Mapping[str, object]:
    closure = inspect.getclosurevars(func)
    names = dict(closure.globals)
    names.update(closure.nonlocals)
    names.pop("__builtins__", None)
    return names

def _defined_in_module(obj: object, module_path: Path) -> bool:
    try:
        obj_file = inspect.getsourcefile(obj)
    except (OSError, TypeError):
        return False
    if not obj_file:
        return False
    return Path(obj_file).resolve() == module_path

_SKIP_NAMES = set(dir(builtins)) | {"__builtins__"}