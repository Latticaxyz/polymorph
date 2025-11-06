# Legacy functional API (for backward compatibility)
from . import clob, gamma, subgraph

# New class-based API
from .gamma_source import GammaSource
from .clob_source import CLOBSource
from .subgraph_source import SubgraphSource

__all__ = [
    # Legacy modules
    "clob",
    "gamma",
    "subgraph",
    # New classes
    "GammaSource",
    "CLOBSource",
    "SubgraphSource",
]
