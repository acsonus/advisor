"""Unit tests for VWAP strategy."""

import pandas as pd
import pytest

from utilities.test_helpers import (
    assert_structure as _assert_structure,
    assert_valid_signals as _assert_valid_signals,
    make_ohlcv,
    sine_wave,
)
from advisor.strategies import VwapStrategy, calculate_vwap


class TestVwapStrategy:
    """Tests for calculate_vwap()."""

    def test_returns_dataframe(self):
        assert isinstance(calculate_vwap(make_ohlcv(sine_wave(50))), pd.DataFrame)

    def test_required_columns_present(self):
        result = calculate_vwap(make_ohlcv(sine_wave(50)))
        _assert_structure(result, ["Close", "Volume", "VWAP", "Signal"])

    def test_signals_are_valid(self):
        _assert_valid_signals(calculate_vwap(make_ohlcv(sine_wave(100))))

    def test_vwap_within_reasonable_price_range(self):
        df = make_ohlcv(sine_wave(100))
        result = calculate_vwap(df)
        assert (result["VWAP"] >= df["Low"].min() * 0.9).all()
        assert (result["VWAP"] <= df["High"].max() * 1.1).all()

    def test_buy_signal_when_close_crosses_above_vwap(self):
        result = calculate_vwap(make_ohlcv(sine_wave(80)))
        assert "Buy" in result["Signal"].values

    def test_sell_signal_when_close_crosses_below_vwap(self):
        result = calculate_vwap(make_ohlcv(sine_wave(80)))
        assert "Sell" in result["Signal"].values

    def test_buy_only_on_upward_cross(self):
        result = calculate_vwap(make_ohlcv(sine_wave(80)))
        buys = result[result["Signal"] == "Buy"]
        for idx in buys.index:
            pos = result.index.get_loc(idx)
            if pos == 0:
                continue
            prev_close = result["Close"].iloc[pos - 1]
            prev_vwap = result["VWAP"].iloc[pos - 1]
            curr_close = result["Close"].iloc[pos]
            curr_vwap = result["VWAP"].iloc[pos]
            assert prev_close <= prev_vwap and curr_close > curr_vwap

    def test_cumulative_volume_is_monotone(self):
        df = make_ohlcv(sine_wave(50))
        result = calculate_vwap(df)
        assert result["Volume"].is_monotonic_increasing or (result["Volume"] == result["Volume"].iloc[0]).all()

    def test_does_not_mutate_input_dataframe(self):
        df = make_ohlcv(sine_wave(50))
        original_cols = list(df.columns)
        calculate_vwap(df)
        assert list(df.columns) == original_cols

    def test_oop_strategy_class_matches_function(self):
        df = make_ohlcv(sine_wave(50))
        strategy = VwapStrategy()
        assert strategy.generate_signals(df).equals(calculate_vwap(df))
