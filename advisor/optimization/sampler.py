"""Randomized parameter sampling with domain constraint validation."""

import random
from typing import Any

from advisor.optimization.spaces import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ParameterSpace,
)


class ParameterSampler:
    """Samples valid parameter combinations from a ParameterSpace."""

    def __init__(self, space: ParameterSpace, seed: int | None = None):
        self.space = space
        self.rng = random.Random(seed)

    def _sample_value(self, param: IntParam | FloatParam | CategoricalParam) -> Any:
        if isinstance(param, IntParam):
            steps = int((param.max_val - param.min_val) / param.step)
            return param.min_val + self.rng.randint(0, steps) * param.step
        elif isinstance(param, FloatParam):
            steps = int(round((param.max_val - param.min_val) / param.step))
            val = param.min_val + self.rng.randint(0, steps) * param.step
            return round(val, 4)
        elif isinstance(param, CategoricalParam):
            return self.rng.choice(param.choices)
        return None

    def sample_one(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Sample a valid strategy and risk parameter set.

        Enforces domain rules:
        - long_ema_period > short_ema_period + 4
        - slow_period > fast_period + 4
        - take_profit_pct >= stop_loss_pct * 1.2 (positive risk-to-reward ratio)
        """
        for _ in range(100):  # max attempts to satisfy constraints
            strategy_cfg = {
                k: self._sample_value(v) for k, v in self.space.strategy_params.items()
            }
            risk_cfg = {
                k: self._sample_value(v) for k, v in self.space.risk_params.items()
            }

            # Strategy constraints
            if "short_ema_period" in strategy_cfg and "long_ema_period" in strategy_cfg:
                if strategy_cfg["long_ema_period"] <= strategy_cfg["short_ema_period"] + 4:
                    continue

            if "fast_period" in strategy_cfg and "slow_period" in strategy_cfg:
                if strategy_cfg["slow_period"] <= strategy_cfg["fast_period"] + 4:
                    continue

            if "rsi_oversold" in strategy_cfg and "rsi_overbought" in strategy_cfg:
                if strategy_cfg["rsi_oversold"] >= strategy_cfg["rsi_overbought"] - 15.0:
                    continue

            # Risk constraints
            sl = risk_cfg.get("stop_loss_pct")
            tp = risk_cfg.get("take_profit_pct")
            if sl is not None and tp is not None and tp < sl * 1.2:
                continue

            return strategy_cfg, risk_cfg

        # Return whatever was sampled on last attempt as fallback
        return strategy_cfg, risk_cfg

    def sample_batch(self, n_iter: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Generate a unique batch of n_iter candidate parameter sets."""
        samples: list[tuple[dict[str, Any], dict[str, Any]]] = []
        seen = set()

        for _ in range(n_iter * 3):
            strat, risk = self.sample_one()
            key = (tuple(sorted(strat.items())), tuple(sorted(risk.items())))
            if key not in seen:
                seen.add(key)
                samples.append((strat, risk))
            if len(samples) >= n_iter:
                break

        return samples
