from __future__ import annotations
from .base import InferenceBackend

def create_backend(config: object):
    raise NotImplementedError