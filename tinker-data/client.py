from __future__ import annotations
from typing import Any, Dict, Optional
from .config import AppConfig

class SyntheticDataClient:
    def __init__(self, config: AppConfig):
        self._config = config

    @classmethod
    def from_config(cls, path: str, overrides: Optional[Dict[str, Any]] = None):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError