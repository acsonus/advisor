"""Unit tests for Bollinger Band Squeeze Breakout strategy."""

import pandas as pd
import pytest

from utilities.test_helpers import (
    assert_structure as _assert_structure,
    assert_valid_signals as _assert_valid_signals,
    make_ohlcv,
    sine_wave,
)
from advisor.strategies import BollingerSqueezeStrategy, bollinger_squeeze_strategy


class TestBollingerSqueezeStrategy:
    """
    Tests for bollinger_squeeze_strategy().

    Key invariant: verifies that Is_Squeeze uses rolling statistics without look-ahead bias.
    """

    _N = 150

    def _make(self, prices=None):
        if prices is None:
            prices = sine_wave(self._N)
        return make_ohlcv(prices)

    def test_returns_dataframe(self):
        assert isinstance(bollinger_squeeze_strategy(self._make()), pd.DataFrame)

    def test_required_columns_present(self):
        result = bollinger_squeeze_strategy(self._make())
        _assert_structure(
            result,
            ["Close", "Middle_Band", "Upper_Band", "Lower_Band", "BB_Width", "Is_Squeeze", "Signal"],
        )

    def test_signals_are_valid(self):
        _assert_valid_signals(bollinger_squeeze_strategy(self._make()))

    def test_upper_band_above_middle_band(self):
        result = bollinger_squeeze_strategy(self._make())
        valid = result[["Upper_Band", "Middle_Band"]].dropna()
        assert (valid["Upper_Band"] >= valid["Middle_Band"]).all()

    def test_lower_band_below_middle_band(self):
        result = bollinger_squeeze_strategy(self._make())
        valid = result[["Lower_Band", "Middle_Band"]].dropna()
        assert (valid["Lower_Band"] <= valid["Middle_Band"]).all()

    def test_bb_width_equals_upper_minus_lower(self):
        result = bollinger_squeeze_strategy(self._make())
        diff = (result["BB_Width"] - (result["Upper_Band"] - result["Lower_Band"])).abs()
        assert (diff.dropna() < 1e-10).all()

    def test_no_lookahead_bias(self):
        prices_short = sine_wave(self._N)
        prices_long = prices_short + sine_wave(40, amplitude=30.0)

        result_short = bollinger_squeeze_strategy(make_ohlcv(prices_short))
        result_long = bollinger_squeeze_strategy(make_ohlcv(prices_long))

        shared_idx = result_short.index
        mismatch = (
            result_short.loc[shared_idx, "Signal"]
            != result_long.loc[shared_idx, "Signal"]
        ).sum()
        assert mismatch == 0, f"{mismatch} signal(s) changed when future data was appended"

    def test_is_squeeze_column_is_boolean(self):
        result = bollinger_squeeze_strategy(self._make())
        valid = result["Is_Squeeze"].dropna()
        assert valid.isin([True, False]).all()

    def test_does_not_mutate_input_dataframe(self):
        df = self._make()
        original_cols = list(df.columns)
        bollinger_squeeze_strategy(df)
        assert list(df.columns) == original_cols

    def test_oop_strategy_class_matches_function(self):
        df = self._make()
        strategy = BollingerSqueezeStrategy()
        assert strategy.generate_signals(df).equals(bollinger_squeeze_strategy(df))
