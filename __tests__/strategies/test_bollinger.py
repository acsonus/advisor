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
        """
        Goal:
            Generate a standardized synthetic OHLCV DataFrame using sine wave variations for Bollinger testing.

        Execution Principle:
            1. If `prices` is not supplied, generate 150-bar sine wave values.
            2. Convert price series into OHLCV DataFrame via `make_ohlcv`.
        """
        if prices is None:
            prices = sine_wave(self._N)
        return make_ohlcv(prices)

    def test_returns_dataframe(self):
        """
        Goal:
            Verify that `bollinger_squeeze_strategy` returns a pandas DataFrame.

        Execution Principle:
            1. Generate synthetic OHLCV data.
            2. Execute strategy.
            3. Assert result is an instance of `pd.DataFrame`.
        """
        assert isinstance(bollinger_squeeze_strategy(self._make()), pd.DataFrame)

    def test_required_columns_present(self):
        """
        Goal:
            Verify presence of all expected bands, squeeze status, and signal columns.

        Execution Principle:
            1. Run strategy over synthetic data.
            2. Confirm DataFrame contains `Close`, `Middle_Band`, `Upper_Band`, `Lower_Band`, `BB_Width`, `Is_Squeeze`, and `Signal`.
        """
        result = bollinger_squeeze_strategy(self._make())
        _assert_structure(
            result,
            ["Close", "Middle_Band", "Upper_Band", "Lower_Band", "BB_Width", "Is_Squeeze", "Signal"],
        )

    def test_signals_are_valid(self):
        """
        Goal:
            Verify all generated Bollinger squeeze breakout signals are valid signal types ('Buy', 'Sell', 'Hold').

        Execution Principle:
            1. Run strategy on sine wave price data.
            2. Validate signal values using `_assert_valid_signals`.
        """
        _assert_valid_signals(bollinger_squeeze_strategy(self._make()))

    def test_upper_band_above_middle_band(self):
        """
        Goal:
            Verify Upper Band is greater than or equal to Middle Band on all bars.

        Execution Principle:
            1. Calculate Bollinger bands.
            2. Drop warmup NaNs.
            3. Assert `Upper_Band >= Middle_Band` on 100% of rows.
        """
        result = bollinger_squeeze_strategy(self._make())
        valid = result[["Upper_Band", "Middle_Band"]].dropna()
        assert (valid["Upper_Band"] >= valid["Middle_Band"]).all()

    def test_lower_band_below_middle_band(self):
        """
        Goal:
            Verify Lower Band is less than or equal to Middle Band on all bars.

        Execution Principle:
            1. Calculate Bollinger bands.
            2. Drop warmup NaNs.
            3. Assert `Lower_Band <= Middle_Band` on 100% of rows.
        """
        result = bollinger_squeeze_strategy(self._make())
        valid = result[["Lower_Band", "Middle_Band"]].dropna()
        assert (valid["Lower_Band"] <= valid["Middle_Band"]).all()

    def test_bb_width_equals_upper_minus_lower(self):
        """
        Goal:
            Verify mathematical identity: `BB_Width = Upper_Band - Lower_Band`.

        Execution Principle:
            1. Calculate strategy metrics.
            2. Evaluate difference between `BB_Width` and `(Upper_Band - Lower_Band)`.
            3. Assert absolute error is within 1e-10.
        """
        result = bollinger_squeeze_strategy(self._make())
        diff = (result["BB_Width"] - (result["Upper_Band"] - result["Lower_Band"])).abs()
        assert (diff.dropna() < 1e-10).all()

    def test_no_lookahead_bias(self):
        """
        Goal:
            Verify strict absence of lookahead bias; signals on historical bars must never alter when future bars are appended.

        Execution Principle:
            1. Calculate strategy signals on base price history of length N.
            2. Append 40 additional bars and calculate strategy on extended dataset.
            3. Compare signals on overlapping index range.
            4. Assert count of mismatched signals equals exactly 0.
        """
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
        """
        Goal:
            Verify the `Is_Squeeze` indicator is strictly boolean (`True` or `False`).

        Execution Principle:
            1. Calculate strategy metrics.
            2. Verify values in `Is_Squeeze` belong strictly to `[True, False]`.
        """
        result = bollinger_squeeze_strategy(self._make())
        valid = result["Is_Squeeze"].dropna()
        assert valid.isin([True, False]).all()

    def test_does_not_mutate_input_dataframe(self):
        """
        Goal:
            Verify execution does not modify caller's input DataFrame.

        Execution Principle:
            1. Capture input columns.
            2. Execute `bollinger_squeeze_strategy`.
            3. Confirm columns remain intact.
        """
        df = self._make()
        original_cols = list(df.columns)
        bollinger_squeeze_strategy(df)
        assert list(df.columns) == original_cols

    def test_oop_strategy_class_matches_function(self):
        """
        Goal:
            Verify `BollingerSqueezeStrategy` class generates identical results to `bollinger_squeeze_strategy`.

        Execution Principle:
            1. Compute signals with OOP class and standalone function.
            2. Assert DataFrames match using `equals()`.
        """
        df = self._make()
        strategy = BollingerSqueezeStrategy()
        assert strategy.generate_signals(df).equals(bollinger_squeeze_strategy(df))
