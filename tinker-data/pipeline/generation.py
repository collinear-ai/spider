from __future__ import annotations

from typing import Iterable, Mapping, Sequence
from ..backends.base import InferenceBackend

def run_generation(
    backend: InferenceBackend,
    prompts: Sequence[str],
):
    raise NotImplementedError