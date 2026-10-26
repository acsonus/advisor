"""Unit and integration tests for the strategy optimization module."""

import numpy as np
import pandas as pd
import pytest

from advisor.optimization.objectives import OptimizationObjective
from advisor.optimization.sampler import ParameterSampler
from advisor.optimization.search import StrategyOptimizer
from advisor.optimization.spaces import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ParameterSpace,
    get_default_strategy_space,
)
from advisor.strategies.param_store import (
    get_effective_params,
    load_optimal_params,
    save_optimal_params,
)


@pytest.fixture
def sample_ohlcv_data():
    """Generate 120 bars of synthetic trending price data."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    base_price = 100.0
    returns = np.random.normal(0.002, 0.02, n)
    prices = base_price * np.exp(np.cumsum(returns))
    highs = prices * (1.0 + np.abs(np.random.normal(0.005, 0.005, n)))
    lows = prices * (1.0 - np.abs(np.random.normal(0.005, 0.005, n)))
    volumes = np.random.randint(10000, 50000, n)

    df = pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_parameter_sampler_constraints():
    """Verify sampler enforces domain constraints."""
    space = get_default_strategy_space("ma_rsi")
    sampler = ParameterSampler(space, seed=123)

    samples = sampler.sample_batch(30)
    assert len(samples) == 30

    for strat_cfg, risk_cfg in samples:
        # Long EMA must exceed Short EMA + 4
        assert strat_cfg["long_ema_period"] > strat_cfg["short_ema_period"] + 4
        # RSI oversold must be significantly below overbought
        assert strat_cfg["rsi_overbought"] > strat_cfg["rsi_oversold"] + 15.0
        # Positive risk to reward
        if risk_cfg.get("stop_loss_pct") and risk_cfg.get("take_profit_pct"):
            assert risk_cfg["take_profit_pct"] >= risk_cfg["stop_loss_pct"] * 1.2


def test_optimization_objective_penalties():
    """Verify objective penalizes drawdown breaches and small trade counts."""
    obj = OptimizationObjective(max_drawdown_cap_pct=15.0, min_trades=5)

    # Valid run
    good_run = {
        "sharpe_ratio": 1.5,
        "profit_factor": 2.0,
        "max_drawdown_pct": 10.0,
        "total_return_pct": 25.0,
        "n_trades": 8,
    }
    fitness = obj.compute_fitness(good_run)
    assert fitness > 0.0

    # Drawdown cap breached
    bad_dd_run = dict(good_run, max_drawdown_pct=25.0)
    fitness_dd = obj.compute_fitness(bad_dd_run)
    assert fitness_dd < 0.0

    # Too few trades
    few_trades_run = dict(good_run, n_trades=2)
    fitness_trades = obj.compute_fitness(few_trades_run)
    assert fitness_trades < -50.0


def test_strategy_optimizer_run(sample_ohlcv_data):
    """Verify StrategyOptimizer runs randomized search and evaluates holdout metrics."""
    optimizer = StrategyOptimizer(
        train_ratio=0.70,
        initial_cash=10000.0,
        commission_bps=5.0,
    )

    result = optimizer.optimize(
        data=sample_ohlcv_data,
        strategy_name="ma_rsi",
        n_iter=15,
        seed=42,
    )

    assert result.strategy_name == "ma_rsi"
    assert "short_ema_period" in result.best_strategy_params
    assert "long_ema_period" in result.best_strategy_params
    assert result.bars_total == 120
    assert result.bars_train == 84
    assert result.bars_test == 36
    assert isinstance(result.optimized_in_sample, dict)
    assert isinstance(result.optimized_out_of_sample, dict)
    assert len(result.top_trials) > 0


def test_param_store_save_and_load(tmp_path):
    """Verify optimal parameters can be persisted and loaded from disk."""
    test_file = tmp_path / "test_optimal_params.json"

    strat_params = {"short_ema_period": 9, "long_ema_period": 35}
    risk_params = {"stop_loss_pct": 0.02, "take_profit_pct": 0.06, "trailing_stop": True}

    save_optimal_params(
        ticker="AAPL",
        strategy_name="ma_rsi",
        strategy_params=strat_params,
        risk_params=risk_params,
        filepath=test_file,
    )

    loaded = load_optimal_params("AAPL", "ma_rsi", filepath=test_file)
    assert loaded is not None
    assert loaded["strategy_params"] == strat_params
    assert loaded["risk_params"] == risk_params

    # Case insensitivity for ticker
    loaded_lower = load_optimal_params("aapl", "ma_rsi", filepath=test_file)
    assert loaded_lower is not None

    # Test get_effective_params
    eff_strat, eff_risk = get_effective_params("AAPL", "ma_rsi", filepath=test_file)
    assert eff_strat == strat_params
    assert eff_risk == risk_params

    # Unknown ticker
    eff_empty_strat, eff_empty_risk = get_effective_params("UNKNOWN", "ma_rsi", filepath=test_file)
    assert eff_empty_strat == {}
    assert eff_empty_risk == {}
