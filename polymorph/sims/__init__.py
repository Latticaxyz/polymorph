"""Simulation and analysis modules."""

# Legacy functional API (for backward compatibility)
from . import monte_carlo, param_search

# New class-based API
from .monte_carlo_simulator import MonteCarloSimulator
from .parameter_searcher import ParameterSearcher

__all__ = [
    # Legacy modules
    "monte_carlo",
    "param_search",
    # New classes
    "MonteCarloSimulator",
    "ParameterSearcher",
]
