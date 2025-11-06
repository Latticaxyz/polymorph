"""Pipeline stages for data processing."""

# Legacy functional API (for backward compatibility)
from . import fetch, process

# New class-based API
from .fetch_stage import FetchStage
from .process_stage import ProcessStage

__all__ = [
    # Legacy modules
    "fetch",
    "process",
    # New classes
    "FetchStage",
    "ProcessStage",
]
