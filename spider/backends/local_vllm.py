from __future__ import annotations
from typing import Iterable, List
from .base import InferenceBackend

class LocalVLLMBackend(InferenceBackend):
    def generate(self, prompts: Iterable[str]):
        raise NotImplementedError