from __future__ import annotations

from typing import Any, Dict, Optional

from .config import AppConfig

class SyntheticDataClient:
    def __init__(self, config: AppConfig):
        self._config = config

    @classmethod
    def from_config(cls, path: str, overrides: Optional[Dict[str, Any]] = None):
        config = AppConfig.load(path, overrides=overrides)
        return cls(config=config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        config = AppConfig.model_validate(config_dict)
        return cls(config=config)

    def run(self):
        raise NotImplementedError