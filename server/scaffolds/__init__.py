"""Scaffolds for SWE trajectory generation.

This module provides wrappers around different SWE agent scaffolds
(OpenHands, SWE-agent, mini-swe-agent) for generating trajectories
from SWE task datasets.
"""

from server.scaffolds.base import Scaffold, ScaffoldConfig

try:
    from server.scaffolds.openhands_wrapper import OpenHandsScaffold, OpenHandsScaffoldConfig
    __all__ = [
        "Scaffold",
        "ScaffoldConfig",
        "OpenHandsScaffold",
        "OpenHandsScaffoldConfig",
    ]
except ImportError:
    # OpenHands not available
    __all__ = [
        "Scaffold",
        "ScaffoldConfig",
    ]

