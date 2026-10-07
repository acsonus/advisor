"""Unit tests for MA-RSI strategy."""

import pandas as pd
import pytest

from utilities.test_helpers import (
    assert_structure as _assert_structure,
    assert_valid_signals as _assert_valid_signals,
    downtrend,
    inverted_v,
    make_ohlcv,
    uptrend,
    v_shape,
)
from advisor.strategies import ma_rsi_strategy, MaRsiStrategy


class TestMaRsiStrategy:
    """
    Tests for ma_rsi_strategy().

    Verifies:
    - Buy signals on a V-shaped recovery.
    - Sell signals on an inverted-V.
    - RSI values inside bounds.
    """

    def test_returns_dataframe(self):
        """
        Goal:
            Verify that ma_rsi_strategy returns a pandas DataFrame.

        Execution Principle:
            1. Generate synthetic V-shaped recovery price series.
            2. Invoke `ma_rsi_strategy`.
            3. Verify output instance type is `pd.DataFrame`.
        """
        df = make_ohlcv(v_shape())
        assert isinstance(ma_rsi_strategy(df), pd.DataFrame)

    def test_required_columns_present(self):
        """
        Goal:
            Verify presence of all expected technical indicator and signal columns.

        Execution Principle:
            1. Execute strategy on V-shaped price history.
            2. Check for columns: `Close`, `Short_EMA`, `Long_EMA`, `RSI`, `Signal`.
        """
        result = ma_rsi_strategy(make_ohlcv(v_shape()))
        _assert_structure(result, ["Close", "Short_EMA", "Long_EMA", "RSI", "Signal"])

    def test_signals_are_valid_strings(self):
        """
        Goal:
            Verify all signals emitted by MA-RSI conform to the allowed discrete set.

        Execution Principle:
            1. Compute strategy signals over V-shaped recovery.
            2. Run `_assert_valid_signals` to check membership in `{'Buy', 'Sell', 'Hold'}`.
        """
        result = ma_rsi_strategy(make_ohlcv(v_shape()))
        _assert_valid_signals(result)

    def test_rsi_in_valid_range(self):
        """
        Goal:
            Verify RSI oscillator values remain strictly bounded within the standard [0, 100] interval.

        Execution Principle:
            1. Drop warmup NaNs from the RSI column.
            2. Assert min >= 0 and max <= 100.
        """
        result = ma_rsi_strategy(make_ohlcv(v_shape()))
        rsi = result["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), (
            f"RSI outside [0, 100]: min={rsi.min():.2f}, max={rsi.max():.2f}"
        )

    def test_rsi_high_in_uptrend(self):
        """
        Goal:
            Verify Wilder-smoothed RSI rises well into bullish territory (> 60) during extended uptrends.

        Execution Principle:
            1. Generate 80 bars of monotonic uptrend data.
            2. Calculate MA-RSI strategy.
            3. Assert final RSI value exceeds 60.
        """
        result = ma_rsi_strategy(make_ohlcv(uptrend(80)))
        last_rsi = result["RSI"].iloc[-1]
        assert last_rsi > 60, f"Expected elevated RSI in strong uptrend, got {last_rsi:.1f}."

    def test_rsi_low_in_downtrend(self):
        """
        Goal:
            Verify Wilder-smoothed RSI drops well into bearish territory (< 40) during extended downtrends.

        Execution Principle:
            1. Generate 80 bars of monotonic downtrend data.
            2. Calculate MA-RSI strategy.
            3. Assert final RSI value is below 40.
        """
        result = ma_rsi_strategy(make_ohlcv(downtrend(80)))
        last_rsi = result["RSI"].iloc[-1]
        assert last_rsi < 40, f"Expected depressed RSI in strong downtrend, got {last_rsi:.1f}."

    def test_short_ema_above_long_ema_in_uptrend(self):
        """
        Goal:
            Confirm Fast EMA (12) finishes above Slow EMA (26) in an extended sustained uptrend.

        Execution Principle:
            1. Compute strategy on an 80-bar uptrend.
            2. Assert last bar Short_EMA > Long_EMA.
        """
        result = ma_rsi_strategy(make_ohlcv(uptrend(80)))
        assert result["Short_EMA"].iloc[-1] > result["Long_EMA"].iloc[-1]

    def test_short_ema_below_long_ema_in_downtrend(self):
        """
        Goal:
            Confirm Fast EMA (12) finishes below Slow EMA (26) in an extended sustained downtrend.

        Execution Principle:
            1. Compute strategy on an 80-bar downtrend.
            2. Assert last bar Short_EMA < Long_EMA.
        """
        result = ma_rsi_strategy(make_ohlcv(downtrend(80)))
        assert result["Short_EMA"].iloc[-1] < result["Long_EMA"].iloc[-1]

    def test_buy_signal_in_v_shape(self):
        """
        Goal:
            Verify a bullish EMA crossover accompanied by non-overbought RSI produces a Buy signal in a V-shaped recovery.

        Execution Principle:
            1. Construct synthetic V-shape price path (40 down, 50 up).
            2. Compute `ma_rsi_strategy`.
            3. Assert `'Buy'` exists in the `Signal` column.
        """
        prices = v_shape(n_down=40, n_up=50, start=150.0, step=1.5)
        result = ma_rsi_strategy(make_ohlcv(prices))
        assert "Buy" in result["Signal"].values, "Expected at least one Buy signal in V-shape data."

    def test_sell_signal_in_inverted_v(self):
        """
        Goal:
            Verify a bearish EMA crossover accompanied by non-oversold RSI produces a Sell signal in an inverted-V reversal.

        Execution Principle:
            1. Construct synthetic inverted-V price path (40 up, 50 down).
            2. Compute `ma_rsi_strategy`.
            3. Assert `'Sell'` exists in the `Signal` column.
        """
        prices = inverted_v(n_up=40, n_down=50, start=100.0, step=1.5)
        result = ma_rsi_strategy(make_ohlcv(prices))
        assert "Sell" in result["Signal"].values, "Expected at least one Sell signal in inverted-V data."

    def test_mostly_hold_in_steady_uptrend(self):
        """
        Goal:
            Verify that once a trend is established, consecutive bars emit 'Hold' rather than repetitive buy signals.

        Execution Principle:
            1. Run strategy over 80 bars of monotonic uptrend.
            2. Calculate proportion of `'Hold'` signals.
            3. Assert hold percentage exceeds 80%.
        """
        result = ma_rsi_strategy(make_ohlcv(uptrend(80)))
        hold_pct = (result["Signal"] == "Hold").mean()
        assert hold_pct > 0.8, f"Expected >80 % Hold, got {hold_pct:.1%}"

    def test_does_not_mutate_input_dataframe(self):
        """
        Goal:
            Verify functional strategy execution does not mutate caller's DataFrame columns.

        Execution Principle:
            1. Snapshot columns of input DataFrame.
            2. Run `ma_rsi_strategy`.
            3. Assert caller DataFrame's columns match the original snapshot.
        """
        df = make_ohlcv(v_shape())
        original_cols = list(df.columns)
        ma_rsi_strategy(df)
        assert list(df.columns) == original_cols

    def test_custom_ema_periods_respected(self):
        """
        Goal:
            Verify customizing fast/slow EMA periods shifts the timing of the crossover signal earlier for shorter periods.

        Execution Principle:
            1. Compute strategy with fast periods (5, 10) vs default periods (12, 26).
            2. Locate first Buy signal index in each run.
            3. Assert fast configuration triggers on or before the slow configuration.
        """
        prices = v_shape()
        result_fast = ma_rsi_strategy(make_ohlcv(prices), short_ema_period=5, long_ema_period=10)
        result_slow = ma_rsi_strategy(make_ohlcv(prices), short_ema_period=12, long_ema_period=26)
        fast_first_buy = result_fast[result_fast["Signal"] == "Buy"].index.min()
        slow_first_buy = result_slow[result_slow["Signal"] == "Buy"].index.min()
        assert fast_first_buy <= slow_first_buy

    def test_oop_strategy_class_matches_function(self):
        """
        Goal:
            Verify the object-oriented `MaRsiStrategy` class produces signals and indicators identical to `ma_rsi_strategy`.

        Execution Principle:
            1. Generate synthetic OHLCV data.
            2. Execute `MaRsiStrategy().generate_signals(df)` and `ma_rsi_strategy(df)`.
            3. Assert DataFrame equality.
        """
        df = make_ohlcv(v_shape())
        strategy = MaRsiStrategy()
        assert strategy.generate_signals(df).equals(ma_rsi_strategy(df))
