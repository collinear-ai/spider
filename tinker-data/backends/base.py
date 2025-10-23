from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, List

class InferenceBackend(ABC):
    @abstractmethod
    def generate(self, prompts: Iterable[str]):
        raise NotImplementedError
    