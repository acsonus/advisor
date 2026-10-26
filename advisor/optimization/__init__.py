"""Optimization package for systematic strategy parameter search and risk management."""

from advisor.optimization.objectives import OptimizationObjective
from advisor.optimization.sampler import ParameterSampler
from advisor.optimization.search import OptimizationResult, StrategyOptimizer
from advisor.optimization.spaces import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ParameterSpace,
    get_default_strategy_space,
)

__all__ = [
    "CategoricalParam",
    "FloatParam",
    "IntParam",
    "OptimizationObjective",
    "OptimizationResult",
    "ParameterSampler",
    "ParameterSpace",
    "StrategyOptimizer",
    "get_default_strategy_space",
]
