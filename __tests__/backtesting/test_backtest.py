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
        """
        Goal:
            Helper fixture generator that constructs a synthetic time-indexed DataFrame
            from parallel lists of closing prices and trading signals.

        Execution Principle:
            1. Validate that `closes` and `signals` share identical length.
            2. Build a pandas DataFrame indexed by business days from 2020-01-01.
            3. Return structured DataFrame with 'Close' and 'Signal' columns.
        """
        assert len(closes) == len(signals), "closes and signals must have the same length"
        n = len(closes)
        return pd.DataFrame(
            {"Close": closes, "Signal": signals},
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )

    # --- Return keys ---

    def test_all_metric_keys_present(self):
        """
        Goal:
            Verify the metrics dictionary returned by `backtest_strategy` contains all required performance keys.

        Execution Principle:
            1. Create a 30-bar dataset with a Buy-and-Sell cycle.
            2. Compute backtest metrics.
            3. Confirm presence of: `total_return_pct`, `annualized_return_pct`, `sharpe_ratio`, `max_drawdown_pct`, `win_rate_pct`, `profit_factor`, `expectancy_pct`, `n_trades`, and `trade_returns`.
        """
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
        """
        Goal:
            Verify a trade buying at low price and selling at high price generates positive return > 40%.

        Execution Principle:
            1. Build price curve transitioning from 100 to 150.
            2. Issue Buy at 100 and Sell at 150.
            3. Execute backtest and assert `total_return_pct > 40.0`.
        """
        prices = [100.0] * 5 + list(range(100, 151)) + [150.0] * 5
        signals = ["Hold"] * 5 + ["Buy"] + ["Hold"] * 49 + ["Sell"] + ["Hold"] * 5
        assert len(prices) == len(signals), f"{len(prices)} vs {len(signals)}"
        metrics = backtest_strategy(self._df(prices, signals))
        assert metrics["total_return_pct"] > 40.0
        assert metrics["n_trades"] >= 1

    def test_buy_high_sell_low_negative_return(self):
        """
        Goal:
            Verify a trade buying high and selling low produces negative total return.

        Execution Principle:
            1. Build downtrend price trajectory from 150 down to 100.
            2. Issue Buy at top and Sell at bottom.
            3. Assert `total_return_pct < 0.0`.
        """
        prices = list(range(150, 99, -1)) + [100.0] * 5
        signals = ["Buy"] + ["Hold"] * 49 + ["Sell"] + ["Hold"] * 5
        metrics = backtest_strategy(self._df(prices, signals))
        assert metrics["total_return_pct"] < 0.0

    def test_all_hold_gives_zero_trades(self):
        """
        Goal:
            Verify a signal series consisting entirely of 'Hold' executes zero trades.

        Execution Principle:
            1. Supply 50 bars of uptrend prices with uniform 'Hold' signals.
            2. Execute backtest.
            3. Assert `n_trades == 0`.
        """
        df = self._df(uptrend(50), ["Hold"] * 50)
        metrics = backtest_strategy(df)
        assert metrics["n_trades"] == 0

    def test_open_position_closed_at_last_bar(self):
        """
        Goal:
            Verify an unclosed position is automatically force-closed at the final available bar.

        Execution Principle:
            1. Issue 'Buy' on bar 0 followed by 'Hold' through bar 19 without explicit 'Sell'.
            2. Run backtest.
            3. Assert `n_trades == 1`, confirming termination at final bar.
        """
        df = self._df(uptrend(20), ["Buy"] + ["Hold"] * 19)
        metrics = backtest_strategy(df)
        assert metrics["n_trades"] == 1

    # --- Trade count and win rate ---

    def test_two_profitable_trades_win_rate_100(self):
        """
        Goal:
            Verify win rate evaluates to 100% when two distinct profitable round trips are completed.

        Execution Principle:
            1. Construct two consecutive winning Buy/Sell trade sequences.
            2. Execute backtest.
            3. Assert `n_trades == 2` and `win_rate_pct == 100.0`.
        """
        prices = [100, 110, 120, 110, 100, 110, 130, 130, 125]
        signals = ["Buy", "Hold", "Sell", "Buy", "Hold", "Hold", "Sell", "Hold", "Hold"]
        metrics = backtest_strategy(self._df(prices, signals))
        assert metrics["n_trades"] == 2
        assert metrics["win_rate_pct"] == 100.0

    def test_two_losing_trades_win_rate_0(self):
        """
        Goal:
            Verify win rate evaluates to 0% when two distinct losing round trips are completed.

        Execution Principle:
            1. Construct two consecutive losing trade sequences.
            2. Execute backtest.
            3. Assert `n_trades == 2` and `win_rate_pct == 0.0`.
        """
        prices = [120, 110, 100, 120, 110, 100, 90, 80, 75]
        signals = ["Buy", "Hold", "Sell", "Buy", "Hold", "Hold", "Sell", "Hold", "Hold"]
        metrics = backtest_strategy(self._df(prices, signals))
        assert metrics["n_trades"] == 2
        assert metrics["win_rate_pct"] == 0.0

    # --- Drawdown ---

    def test_max_drawdown_non_positive(self):
        """
        Goal:
            Verify maximum drawdown is strictly non-positive (<= 0.0%).

        Execution Principle:
            1. Compute backtest on uptrend with Buy/Sell cycle.
            2. Assert `max_drawdown_pct <= 0.0`.
        """
        df = self._df(uptrend(50), ["Buy"] + ["Hold"] * 48 + ["Sell"])
        metrics = backtest_strategy(df)
        assert metrics["max_drawdown_pct"] <= 0.0

    def test_flat_hold_zero_drawdown(self):
        """
        Goal:
            Verify an uninvested flat-price series exhibits exactly 0.0% drawdown.

        Execution Principle:
            1. Construct flat price series with uniform 'Hold' signals.
            2. Assert `max_drawdown_pct == 0.0`.
        """
        df = self._df([100.0] * 20, ["Hold"] * 20)
        metrics = backtest_strategy(df)
        assert metrics["max_drawdown_pct"] == 0.0

    # --- Sharpe ratio ---

    def test_sharpe_positive_in_steady_uptrend(self):
        """
        Goal:
            Verify Sharpe ratio is strictly positive for a steadily profitable trade in an uptrend.

        Execution Principle:
            1. Buy and hold across 120 bars of monotonic upward momentum.
            2. Compute backtest.
            3. Assert `sharpe_ratio > 0.0`.
        """
        df = self._df(uptrend(120, step=1.0), ["Buy"] + ["Hold"] * 118 + ["Sell"])
        assert backtest_strategy(df)["sharpe_ratio"] > 0.0

    def test_sharpe_zero_when_no_trades(self):
        """
        Goal:
            Verify Sharpe ratio safely returns 0.0 when zero trades are executed.

        Execution Principle:
            1. Run backtest with no trades.
            2. Assert `sharpe_ratio == 0.0` without division by zero errors.
        """
        df = self._df(uptrend(50), ["Hold"] * 50)
        metrics = backtest_strategy(df)
        assert metrics["sharpe_ratio"] == 0.0

    # --- Edge cases ---

    def test_single_bar_returns_zeros(self):
        """
        Goal:
            Verify engine handles single-bar dataset without IndexError or crashes.

        Execution Principle:
            1. Pass 1 bar DataFrame into `backtest_strategy`.
            2. Assert `total_return_pct == 0.0` and `n_trades == 0`.
        """
        df = self._df([100.0], ["Hold"])
        metrics = backtest_strategy(df)
        assert metrics["total_return_pct"] == 0.0
        assert metrics["n_trades"] == 0

    def test_buy_on_last_bar_only_no_trade(self):
        """
        Goal:
            Verify a Buy signal placed on the final bar closes immediately with near-zero return.

        Execution Principle:
            1. Place Buy on bar 9 of 10.
            2. Run backtest.
            3. Assert total return is approximately 0.0.
        """
        df = self._df(uptrend(10), ["Hold"] * 9 + ["Buy"])
        metrics = backtest_strategy(df)
        assert metrics["total_return_pct"] == pytest.approx(0.0, abs=0.1)

    def test_trade_returns_list_length_matches_n_trades(self):
        """
        Goal:
            Verify the length of the `trade_returns` detail array matches `n_trades`.

        Execution Principle:
            1. Execute strategy with completed trade cycle.
            2. Verify `len(metrics['trade_returns']) == metrics['n_trades']`.
        """
        df = self._df(uptrend(30), ["Buy"] + ["Hold"] * 28 + ["Sell"])
        metrics = backtest_strategy(df)
        assert len(metrics["trade_returns"]) == metrics["n_trades"]

    def test_custom_column_names(self):
        """
        Goal:
            Verify backtesting engine supports custom price and signal column names.

        Execution Principle:
            1. Create DataFrame with 'price' and 'action' columns.
            2. Pass `close_col='price'` and `signal_col='action'`.
            3. Assert successful execution with `n_trades >= 1`.
        """
        prices = uptrend(30)
        signals = ["Buy"] + ["Hold"] * 28 + ["Sell"]
        df = pd.DataFrame(
            {"price": prices, "action": signals},
            index=pd.date_range("2020-01-01", periods=30, freq="B"),
        )
        metrics = backtest_strategy(df, signal_col="action", close_col="price")
        assert metrics["n_trades"] >= 1

    def test_costs_reduce_total_return(self):
        """
        Goal:
            Verify transaction fees and slippage reduce total net return relative to frictionless baseline.

        Execution Principle:
            1. Run backtest with zero fees/slippage.
            2. Run backtest with 10 bps fee and 5 bps slippage.
            3. Assert costly return <= baseline return.
        """
        df = self._df(uptrend(40), ["Buy"] + ["Hold"] * 38 + ["Sell"])
        baseline = backtest_strategy(df, fee_bps=0.0, slippage_bps=0.0)
        costly = backtest_strategy(df, fee_bps=10.0, slippage_bps=5.0)
        assert costly["total_return_pct"] <= baseline["total_return_pct"]

    def test_stop_loss_exits_early(self):
        """
        Goal:
            Verify trailing stop-loss triggers early exit before price decline completes.

        Execution Principle:
            1. Generate descending price bars following Buy entry.
            2. Execute backtest with `stop_loss_pct=0.03`.
            3. Confirm trade exit occurred with return <= -3%.
        """
        prices = [100, 99, 98, 95, 92, 90, 89, 88, 87]
        signals = ["Buy"] + ["Hold"] * (len(prices) - 1)
        metrics = backtest_strategy(self._df(prices, signals), stop_loss_pct=0.03)
        assert metrics["n_trades"] == 1
        assert metrics["trade_returns"][0] <= -0.03

    def test_take_profit_exits_early(self):
        """
        Goal:
            Verify take-profit threshold triggers early exit once profit target is reached.

        Execution Principle:
            1. Generate rising prices after Buy entry.
            2. Execute backtest with `take_profit_pct=0.05`.
            3. Confirm trade exit occurred with return >= +5%.
        """
        prices = [100, 103, 106, 107, 106, 105, 104]
        signals = ["Buy"] + ["Hold"] * (len(prices) - 1)
        metrics = backtest_strategy(self._df(prices, signals), take_profit_pct=0.05)
        assert metrics["n_trades"] == 1
        assert metrics["trade_returns"][0] >= 0.05

    def test_max_hold_bars_exits_trade(self):
        """
        Goal:
            Verify time-based max hold duration triggers automatic position exit after specified bar count.

        Execution Principle:
            1. Open position with Buy signal followed by Hold signals.
            2. Set `max_hold_bars=2`.
            3. Confirm trade was closed after 2 bars.
        """
        prices = [100, 101, 102, 103, 104, 105]
        signals = ["Buy"] + ["Hold"] * (len(prices) - 1)
        metrics = backtest_strategy(self._df(prices, signals), max_hold_bars=2)
        assert metrics["n_trades"] == 1

    def test_invalid_backtest_parameters_raise(self):
        """
        Goal:
            Verify invalid backtest parameters raise `ValueError` upon validation.

        Execution Principle:
            1. Test negative fee, negative slippage, non-positive stop loss, non-positive take profit, and zero max hold bars.
            2. Assert `ValueError` is raised in each case.
        """
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
        """
        Goal:
            Verify object-oriented `BacktestEngine.run().to_dict()` outputs match `backtest_strategy`.

        Execution Principle:
            1. Run backtest with `BacktestEngine` and `backtest_strategy`.
            2. Assert dictionary equality.
        """
        df = self._df(uptrend(30), ["Buy"] + ["Hold"] * 28 + ["Sell"])
        config = BacktestConfig()
        result = BacktestEngine(config).run(df)
        assert result.to_dict() == backtest_strategy(df)
