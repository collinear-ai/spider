from __future__ import annotations

from typing import Dict, Iterable, List, Protocol

class InferenceBackend(Protocol):
    def generate(
        self, prompts: Iterable[str], *, parameters: Dict[str, object]
    ) -> List[str]: 
        return []

    def metrics(self) -> Dict[str, object]:
        return {}