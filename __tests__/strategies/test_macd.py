"""Unit tests for MACD Histogram Reversal strategy."""

import pandas as pd
import pytest

from utilities.test_helpers import (
    assert_structure as _assert_structure,
    assert_valid_signals as _assert_valid_signals,
    inverted_v,
    make_ohlcv,
    sine_wave,
    uptrend,
    v_shape,
)
from advisor.strategies import MacdHistogramReversalStrategy, macd_histogram_reversal_strategy


class TestMacdHistogramReversal:
    """Tests for macd_histogram_reversal_strategy()."""

    def test_returns_dataframe(self):
        assert isinstance(macd_histogram_reversal_strategy(make_ohlcv(uptrend(60))), pd.DataFrame)

    def test_required_columns_present(self):
        result = macd_histogram_reversal_strategy(make_ohlcv(uptrend(60)))
        _assert_structure(result, ["Close", "MACD", "Signal_Line", "MACD_Histogram", "Signal"])

    def test_signals_are_valid(self):
        result = macd_histogram_reversal_strategy(make_ohlcv(sine_wave(100)))
        _assert_valid_signals(result)

    def test_histogram_equals_macd_minus_signal_line(self):
        """Mathematical identity: Histogram = MACD − Signal_Line."""
        result = macd_histogram_reversal_strategy(make_ohlcv(sine_wave(100)))
        diff = (result["MACD_Histogram"] - (result["MACD"] - result["Signal_Line"])).abs()
        assert (diff < 1e-10).all(), "MACD_Histogram does not equal MACD - Signal_Line"

    def test_buy_signal_on_histogram_zero_cross_up(self):
        prices = v_shape(n_down=50, n_up=60, start=200.0, step=1.0)
        result = macd_histogram_reversal_strategy(make_ohlcv(prices))
        assert "Buy" in result["Signal"].values

    def test_sell_signal_on_histogram_zero_cross_down(self):
        prices = inverted_v(n_up=50, n_down=60, start=100.0, step=1.0)
        result = macd_histogram_reversal_strategy(make_ohlcv(prices))
        assert "Sell" in result["Signal"].values

    def test_buy_where_histogram_crosses_zero_upward(self):
        prices = v_shape(n_down=50, n_up=60, start=200.0, step=1.0)
        result = macd_histogram_reversal_strategy(make_ohlcv(prices))
        buys = result[result["Signal"] == "Buy"]
        for idx in buys.index:
            pos = result.index.get_loc(idx)
            if pos == 0:
                continue
            prev_hist = result["MACD_Histogram"].iloc[pos - 1]
            curr_hist = result["MACD_Histogram"].iloc[pos]
            assert prev_hist < 0 and curr_hist >= 0

    def test_sell_where_histogram_crosses_zero_downward(self):
        prices = inverted_v(n_up=50, n_down=60, start=100.0, step=1.0)
        result = macd_histogram_reversal_strategy(make_ohlcv(prices))
        sells = result[result["Signal"] == "Sell"]
        for idx in sells.index:
            pos = result.index.get_loc(idx)
            if pos == 0:
                continue
            prev_hist = result["MACD_Histogram"].iloc[pos - 1]
            curr_hist = result["MACD_Histogram"].iloc[pos]
            assert prev_hist > 0 and curr_hist <= 0

    def test_does_not_mutate_input_dataframe(self):
        df = make_ohlcv(uptrend(60))
        original_cols = list(df.columns)
        macd_histogram_reversal_strategy(df)
        assert list(df.columns) == original_cols

    def test_oop_strategy_class_matches_function(self):
        df = make_ohlcv(sine_wave(60))
        strategy = MacdHistogramReversalStrategy()
        assert strategy.generate_signals(df).equals(macd_histogram_reversal_strategy(df))
