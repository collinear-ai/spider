from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Mapping

class DataSource(ABC):
    @abstractmethod
    def records(self):
        raise NotImplementedError