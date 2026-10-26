"""Parameter search space definitions for strategies and risk management."""

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class IntParam:
    """Integer parameter range."""
    min_val: int
    max_val: int
    step: int = 1


@dataclass
class FloatParam:
    """Floating point parameter range."""
    min_val: float
    max_val: float
    step: float = 0.01


@dataclass
class CategoricalParam:
    """Categorical choice parameter."""
    choices: Sequence[Any]


@dataclass
class ParameterSpace:
    """Collection of parameters defining a strategy and risk space."""
    strategy_params: dict[str, IntParam | FloatParam | CategoricalParam] = field(default_factory=dict)
    risk_params: dict[str, IntParam | FloatParam | CategoricalParam] = field(default_factory=dict)


def get_default_strategy_space(strategy_name: str) -> ParameterSpace:
    """
    Retrieve standard optimization parameter space for a registered strategy.

    Goal:
    -----
    Provide sensible search ranges around standard technical indicator settings
    while avoiding unphysical indicator spans or inverted bands.
    """
    risk_space = {
        "stop_loss_pct": FloatParam(0.01, 0.06, step=0.005),
        "take_profit_pct": FloatParam(0.02, 0.15, step=0.01),
        "trailing_stop": CategoricalParam([False, True]),
        "max_hold_bars": CategoricalParam([None, 10, 20, 40, 60]),
    }

    if strategy_name == "ma_rsi":
        return ParameterSpace(
            strategy_params={
                "short_ema_period": IntParam(5, 20, step=1),
                "long_ema_period": IntParam(21, 60, step=2),
                "rsi_period": IntParam(7, 21, step=1),
                "rsi_oversold": FloatParam(20.0, 40.0, step=2.0),
                "rsi_overbought": FloatParam(60.0, 80.0, step=2.0),
            },
            risk_params=risk_space,
        )

    if strategy_name == "bollinger_squeeze":
        return ParameterSpace(
            strategy_params={
                "window": IntParam(10, 30, step=2),
                "num_std_dev": FloatParam(1.5, 2.5, step=0.1),
                "squeeze_window": IntParam(20, 80, step=10),
                "squeeze_threshold": FloatParam(0.2, 0.8, step=0.1),
            },
            risk_params=risk_space,
        )

    if strategy_name == "macd_histogram_reversal":
        return ParameterSpace(
            strategy_params={
                "fast_period": IntParam(6, 16, step=1),
                "slow_period": IntParam(18, 35, step=1),
                "signal_period": IntParam(5, 14, step=1),
            },
            risk_params=risk_space,
        )

    if strategy_name == "atr_trailing_stop":
        return ParameterSpace(
            strategy_params={
                "atr_period": IntParam(7, 21, step=1),
                "multiplier": FloatParam(1.5, 4.0, step=0.25),
            },
            risk_params=risk_space,
        )

    if strategy_name == "vwap_cross":
        return ParameterSpace(
            strategy_params={},
            risk_params=risk_space,
        )

    # Generic fallback
    return ParameterSpace(
        strategy_params={},
        risk_params=risk_space,
    )
