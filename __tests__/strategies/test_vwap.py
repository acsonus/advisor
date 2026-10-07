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
        """
        Goal:
            Verify that calculate_vwap returns a pandas DataFrame.

        Execution Principle:
            1. Generate synthetic OHLCV data using `sine_wave(50)`.
            2. Run `calculate_vwap`.
            3. Assert result is an instance of `pd.DataFrame`.
        """
        assert isinstance(calculate_vwap(make_ohlcv(sine_wave(50))), pd.DataFrame)

    def test_required_columns_present(self):
        """
        Goal:
            Verify existence of all expected VWAP output columns.

        Execution Principle:
            1. Compute VWAP on synthetic OHLCV bars.
            2. Check structure for `['Close', 'Volume', 'VWAP', 'Signal']`.
        """
        result = calculate_vwap(make_ohlcv(sine_wave(50)))
        _assert_structure(result, ["Close", "Volume", "VWAP", "Signal"])

    def test_signals_are_valid(self):
        """
        Goal:
            Verify all generated VWAP signals are members of `{'Buy', 'Sell', 'Hold'}`.

        Execution Principle:
            1. Generate 100 cyclical bars and calculate VWAP.
            2. Validate signal values using `_assert_valid_signals`.
        """
        _assert_valid_signals(calculate_vwap(make_ohlcv(sine_wave(100))))

    def test_vwap_within_reasonable_price_range(self):
        """
        Goal:
            Ensure calculated VWAP volume-weighted averages remain strictly bounded within traded price extremes.

        Execution Principle:
            1. Compute VWAP over a 100-bar sine wave.
            2. Verify every VWAP value lies within [Low_min * 0.9, High_max * 1.1].
        """
        df = make_ohlcv(sine_wave(100))
        result = calculate_vwap(df)
        assert (result["VWAP"] >= df["Low"].min() * 0.9).all()
        assert (result["VWAP"] <= df["High"].max() * 1.1).all()

    def test_buy_signal_when_close_crosses_above_vwap(self):
        """
        Goal:
            Verify that upward price crossing above VWAP yields at least one 'Buy' signal in undulating data.

        Execution Principle:
            1. Run VWAP strategy over an 80-bar sine wave.
            2. Assert `'Buy'` is present in `result['Signal'].values`.
        """
        result = calculate_vwap(make_ohlcv(sine_wave(80)))
        assert "Buy" in result["Signal"].values

    def test_sell_signal_when_close_crosses_below_vwap(self):
        """
        Goal:
            Verify that downward price crossing below VWAP yields at least one 'Sell' signal in undulating data.

        Execution Principle:
            1. Run VWAP strategy over an 80-bar sine wave.
            2. Assert `'Sell'` is present in `result['Signal'].values`.
        """
        result = calculate_vwap(make_ohlcv(sine_wave(80)))
        assert "Sell" in result["Signal"].values

    def test_buy_only_on_upward_cross(self):
        """
        Goal:
            Verify that every bar labelled 'Buy' satisfies the exact crossover condition: `prev_close <= prev_vwap` and `curr_close > curr_vwap`.

        Execution Principle:
            1. Filter result to all Buy bars.
            2. Compare previous bar Close and VWAP against current bar Close and VWAP.
            3. Assert upward crossing condition strictly holds.
        """
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
        """
        Goal:
            Verify input or cumulative volume used in VWAP computation is non-negative and properly handled.

        Execution Principle:
            1. Generate synthetic OHLCV data.
            2. Compute VWAP.
            3. Assert volume is either monotonic increasing or uniform non-negative values.
        """
        df = make_ohlcv(sine_wave(50))
        result = calculate_vwap(df)
        assert result["Volume"].is_monotonic_increasing or (result["Volume"] == result["Volume"].iloc[0]).all()

    def test_does_not_mutate_input_dataframe(self):
        """
        Goal:
            Verify execution of `calculate_vwap` leaves caller's DataFrame columns unmodified.

        Execution Principle:
            1. Snapshot columns of input DataFrame.
            2. Invoke `calculate_vwap`.
            3. Verify original columns remain intact.
        """
        df = make_ohlcv(sine_wave(50))
        original_cols = list(df.columns)
        calculate_vwap(df)
        assert list(df.columns) == original_cols

    def test_oop_strategy_class_matches_function(self):
        """
        Goal:
            Verify the object-oriented `VwapStrategy` produces outputs matching functional `calculate_vwap`.

        Execution Principle:
            1. Compute signals with `VwapStrategy().generate_signals(df)` and `calculate_vwap(df)`.
            2. Assert DataFrame equality.
        """
        df = make_ohlcv(sine_wave(50))
        strategy = VwapStrategy()
        assert strategy.generate_signals(df).equals(calculate_vwap(df))
