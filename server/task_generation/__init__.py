"""SWE task generation module for Spider.

This module provides integration with SWE-smith for generating software engineering
task instances from GitHub repositories.
"""

from .swesmith_integration import SWESmithTaskGenerator, TaskGenerationError
from .format_converter import convert_swesmith_to_hf_format

__all__ = [
    "SWESmithTaskGenerator",
    "TaskGenerationError",
    "convert_swesmith_to_hf_format",
]

