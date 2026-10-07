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
        df = make_ohlcv(v_shape())
        assert isinstance(ma_rsi_strategy(df), pd.DataFrame)

    def test_required_columns_present(self):
        result = ma_rsi_strategy(make_ohlcv(v_shape()))
        _assert_structure(result, ["Close", "Short_EMA", "Long_EMA", "RSI", "Signal"])

    def test_signals_are_valid_strings(self):
        result = ma_rsi_strategy(make_ohlcv(v_shape()))
        _assert_valid_signals(result)

    def test_rsi_in_valid_range(self):
        result = ma_rsi_strategy(make_ohlcv(v_shape()))
        rsi = result["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), (
            f"RSI outside [0, 100]: min={rsi.min():.2f}, max={rsi.max():.2f}"
        )

    def test_rsi_high_in_uptrend(self):
        """After a long uptrend, RSI should stabilise well above 60."""
        result = ma_rsi_strategy(make_ohlcv(uptrend(80)))
        last_rsi = result["RSI"].iloc[-1]
        assert last_rsi > 60, f"Expected elevated RSI in strong uptrend, got {last_rsi:.1f}."

    def test_rsi_low_in_downtrend(self):
        """After a long downtrend, RSI should stabilise well below 40."""
        result = ma_rsi_strategy(make_ohlcv(downtrend(80)))
        last_rsi = result["RSI"].iloc[-1]
        assert last_rsi < 40, f"Expected depressed RSI in strong downtrend, got {last_rsi:.1f}."

    def test_short_ema_above_long_ema_in_uptrend(self):
        """In a sustained uptrend, Short EMA (12) must end above Long EMA (26)."""
        result = ma_rsi_strategy(make_ohlcv(uptrend(80)))
        assert result["Short_EMA"].iloc[-1] > result["Long_EMA"].iloc[-1]

    def test_short_ema_below_long_ema_in_downtrend(self):
        result = ma_rsi_strategy(make_ohlcv(downtrend(80)))
        assert result["Short_EMA"].iloc[-1] < result["Long_EMA"].iloc[-1]

    def test_buy_signal_in_v_shape(self):
        prices = v_shape(n_down=40, n_up=50, start=150.0, step=1.5)
        result = ma_rsi_strategy(make_ohlcv(prices))
        assert "Buy" in result["Signal"].values, "Expected at least one Buy signal in V-shape data."

    def test_sell_signal_in_inverted_v(self):
        prices = inverted_v(n_up=40, n_down=50, start=100.0, step=1.5)
        result = ma_rsi_strategy(make_ohlcv(prices))
        assert "Sell" in result["Signal"].values, "Expected at least one Sell signal in inverted-V data."

    def test_mostly_hold_in_steady_uptrend(self):
        result = ma_rsi_strategy(make_ohlcv(uptrend(80)))
        hold_pct = (result["Signal"] == "Hold").mean()
        assert hold_pct > 0.8, f"Expected >80 % Hold, got {hold_pct:.1%}"

    def test_does_not_mutate_input_dataframe(self):
        df = make_ohlcv(v_shape())
        original_cols = list(df.columns)
        ma_rsi_strategy(df)
        assert list(df.columns) == original_cols

    def test_custom_ema_periods_respected(self):
        prices = v_shape()
        result_fast = ma_rsi_strategy(make_ohlcv(prices), short_ema_period=5, long_ema_period=10)
        result_slow = ma_rsi_strategy(make_ohlcv(prices), short_ema_period=12, long_ema_period=26)
        fast_first_buy = result_fast[result_fast["Signal"] == "Buy"].index.min()
        slow_first_buy = result_slow[result_slow["Signal"] == "Buy"].index.min()
        assert fast_first_buy <= slow_first_buy

    def test_oop_strategy_class_matches_function(self):
        df = make_ohlcv(v_shape())
        strategy = MaRsiStrategy()
        assert strategy.generate_signals(df).equals(ma_rsi_strategy(df))
