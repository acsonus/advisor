"""Backtesting simulation and performance evaluation package."""

from advisor.backtesting.engine import BacktestEngine, backtest_strategy
from advisor.backtesting.metrics import calculate_metrics
from advisor.backtesting.models import BacktestConfig, BacktestResult, Trade

__all__ = [
    "BacktestConfig",
    "Trade",
    "BacktestResult",
    "BacktestEngine",
    "calculate_metrics",
    "backtest_strategy",
]
