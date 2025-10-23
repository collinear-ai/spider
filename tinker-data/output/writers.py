from __future__ import annotations

from typing import Iterable, Mapping

def write_outputs(
    records: Iterable[Mapping[str, object]],
    *,
    format: str,
    destination: str,
):
    raise NotImplementedError