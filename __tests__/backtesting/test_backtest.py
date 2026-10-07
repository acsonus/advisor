"""Unit tests for the BacktestEngine and backtest_strategy function."""

import pandas as pd
import pytest

from utilities.test_helpers import uptrend
from advisor.backtesting import BacktestConfig, BacktestEngine, backtest_strategy


class TestBacktestStrategy:
    """
    Tests for the backtest_strategy() performance engine.

    Trades are executed at the close of the signal bar (same-bar execution).
    """

    def _df(self, closes, signals):
        assert len(closes) == len(signals), "closes and signals must have the same length"
        n = len(closes)
        return pd.DataFrame(
            {"Close": closes, "Signal": signals},
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )

    # --- Return keys ---

    def test_all_metric_keys_present(self):
        df = self._df(uptrend(30), ["Buy"] + ["Hold"] * 28 + ["Sell"])
        metrics = backtest_strategy(df)
        for key in (
            "total_return_pct",
            "annualized_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "win_rate_pct",
            "profit_factor",
            "expectancy_pct",
            "n_trades",
            "trade_returns",
        ):
            assert key in metrics, f"Missing key: '{key}'"

    # --- Correct return calculation ---

    def test_buy_low_sell_high_positive_return(self):
        prices = [100.0] * 5 + list(range(100, 151)) + [150.0] * 5
        signals = ["Hold"] * 5 + ["Buy"] + ["Hold"] * 49 + ["Sell"] + ["Hold"] * 5
        assert len(prices) == len(signals), f"{len(prices)} vs {len(signals)}"
        metrics = backtest_strategy(self._df(prices, signals))
        assert metrics["total_return_pct"] > 40.0
        assert metrics["n_trades"] >= 1

    def test_buy_high_sell_low_negative_return(self):
        prices = list(range(150, 99, -1)) + [100.0] * 5
        signals = ["Buy"] + ["Hold"] * 49 + ["Sell"] + ["Hold"] * 5
        metrics = backtest_strategy(self._df(prices, signals))
        assert metrics["total_return_pct"] < 0.0

    def test_all_hold_gives_zero_trades(self):
        df = self._df(uptrend(50), ["Hold"] * 50)
        metrics = backtest_strategy(df)
        assert metrics["n_trades"] == 0

    def test_open_position_closed_at_last_bar(self):
        df = self._df(uptrend(20), ["Buy"] + ["Hold"] * 19)
        metrics = backtest_strategy(df)
        assert metrics["n_trades"] == 1

    # --- Trade count and win rate ---

    def test_two_profitable_trades_win_rate_100(self):
        prices = [100, 110, 120, 110, 100, 110, 130, 130, 125]
        signals = ["Buy", "Hold", "Sell", "Buy", "Hold", "Hold", "Sell", "Hold", "Hold"]
        metrics = backtest_strategy(self._df(prices, signals))
        assert metrics["n_trades"] == 2
        assert metrics["win_rate_pct"] == 100.0

    def test_two_losing_trades_win_rate_0(self):
        prices = [120, 110, 100, 120, 110, 100, 90, 80, 75]
        signals = ["Buy", "Hold", "Sell", "Buy", "Hold", "Hold", "Sell", "Hold", "Hold"]
        metrics = backtest_strategy(self._df(prices, signals))
        assert metrics["n_trades"] == 2
        assert metrics["win_rate_pct"] == 0.0

    # --- Drawdown ---

    def test_max_drawdown_non_positive(self):
        df = self._df(uptrend(50), ["Buy"] + ["Hold"] * 48 + ["Sell"])
        metrics = backtest_strategy(df)
        assert metrics["max_drawdown_pct"] <= 0.0

    def test_flat_hold_zero_drawdown(self):
        df = self._df([100.0] * 20, ["Hold"] * 20)
        metrics = backtest_strategy(df)
        assert metrics["max_drawdown_pct"] == 0.0

    # --- Sharpe ratio ---

    def test_sharpe_positive_in_steady_uptrend(self):
        df = self._df(uptrend(120, step=1.0), ["Buy"] + ["Hold"] * 118 + ["Sell"])
        assert backtest_strategy(df)["sharpe_ratio"] > 0.0

    def test_sharpe_zero_when_no_trades(self):
        df = self._df(uptrend(50), ["Hold"] * 50)
        metrics = backtest_strategy(df)
        assert metrics["sharpe_ratio"] == 0.0

    # --- Edge cases ---

    def test_single_bar_returns_zeros(self):
        df = self._df([100.0], ["Hold"])
        metrics = backtest_strategy(df)
        assert metrics["total_return_pct"] == 0.0
        assert metrics["n_trades"] == 0

    def test_buy_on_last_bar_only_no_trade(self):
        df = self._df(uptrend(10), ["Hold"] * 9 + ["Buy"])
        metrics = backtest_strategy(df)
        assert metrics["total_return_pct"] == pytest.approx(0.0, abs=0.1)

    def test_trade_returns_list_length_matches_n_trades(self):
        df = self._df(uptrend(30), ["Buy"] + ["Hold"] * 28 + ["Sell"])
        metrics = backtest_strategy(df)
        assert len(metrics["trade_returns"]) == metrics["n_trades"]

    def test_custom_column_names(self):
        prices = uptrend(30)
        signals = ["Buy"] + ["Hold"] * 28 + ["Sell"]
        df = pd.DataFrame(
            {"price": prices, "action": signals},
            index=pd.date_range("2020-01-01", periods=30, freq="B"),
        )
        metrics = backtest_strategy(df, signal_col="action", close_col="price")
        assert metrics["n_trades"] >= 1

    def test_costs_reduce_total_return(self):
        df = self._df(uptrend(40), ["Buy"] + ["Hold"] * 38 + ["Sell"])
        baseline = backtest_strategy(df, fee_bps=0.0, slippage_bps=0.0)
        costly = backtest_strategy(df, fee_bps=10.0, slippage_bps=5.0)
        assert costly["total_return_pct"] <= baseline["total_return_pct"]

    def test_stop_loss_exits_early(self):
        prices = [100, 99, 98, 95, 92, 90, 89, 88, 87]
        signals = ["Buy"] + ["Hold"] * (len(prices) - 1)
        metrics = backtest_strategy(self._df(prices, signals), stop_loss_pct=0.03)
        assert metrics["n_trades"] == 1
        assert metrics["trade_returns"][0] <= -0.03

    def test_take_profit_exits_early(self):
        prices = [100, 103, 106, 107, 106, 105, 104]
        signals = ["Buy"] + ["Hold"] * (len(prices) - 1)
        metrics = backtest_strategy(self._df(prices, signals), take_profit_pct=0.05)
        assert metrics["n_trades"] == 1
        assert metrics["trade_returns"][0] >= 0.05

    def test_max_hold_bars_exits_trade(self):
        prices = [100, 101, 102, 103, 104, 105]
        signals = ["Buy"] + ["Hold"] * (len(prices) - 1)
        metrics = backtest_strategy(self._df(prices, signals), max_hold_bars=2)
        assert metrics["n_trades"] == 1

    def test_invalid_backtest_parameters_raise(self):
        df = self._df(uptrend(20), ["Buy"] + ["Hold"] * 18 + ["Sell"])
        with pytest.raises(ValueError):
            backtest_strategy(df, fee_bps=-1)
        with pytest.raises(ValueError):
            backtest_strategy(df, slippage_bps=-1)
        with pytest.raises(ValueError):
            backtest_strategy(df, stop_loss_pct=0)
        with pytest.raises(ValueError):
            backtest_strategy(df, take_profit_pct=0)
        with pytest.raises(ValueError):
            backtest_strategy(df, max_hold_bars=0)

    def test_oop_backtest_engine_class_matches_function(self):
        df = self._df(uptrend(30), ["Buy"] + ["Hold"] * 28 + ["Sell"])
        config = BacktestConfig()
        result = BacktestEngine(config).run(df)
        assert result.to_dict() == backtest_strategy(df)
