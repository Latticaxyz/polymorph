"""Analysis result models for simulations and optimization."""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator


class SimulationResult(BaseModel):
    """Result from Monte Carlo simulation."""

    token_id: str
    n_trials: int
    n_days: int
    median_return: float
    percentile_5: float
    percentile_95: float
    prob_negative: float
    initial_price: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class OptimizationResult(BaseModel):
    """Result from parameter search optimization."""

    study_name: str
    n_trials: int
    best_params: dict[str, Any]
    best_value: float
    optimization_history: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
