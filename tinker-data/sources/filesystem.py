from __future__ import annotations
from typing import Iterable, Mapping
from .base import DataSource

class JSONLSource(DataSource):
    def __init__(self, path: str):
        self.path = path

    def records(self):
        raise NotImplementedError


class CSVSource(DataSource):
    def __init__(self, path: str):
        self.path = path

    def records(self):
        raise NotImplementedError