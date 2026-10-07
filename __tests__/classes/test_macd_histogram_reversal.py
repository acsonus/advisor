# ===========================================================================
# 5.  MACD Histogram Reversal
# ===========================================================================
import pandas as pd
import trading_strategy as ts
from utilities.test_helpers import make_ohlcv, uptrend, downtrend, v_shape, inverted_v, sine_wave, assert_structure as _assert_structure, assert_valid_signals as _assert_valid_signals
class TestMacdHistogramReversal:
    """Tests for macd_histogram_reversal_strategy()."""

    def test_returns_dataframe(self):
        """
        Goal:
            Verify that macd_histogram_reversal_strategy returns a pandas DataFrame.

        Execution Principle:
            1. Generate synthetic uptrend OHLCV data.
            2. Run `macd_histogram_reversal_strategy`.
            3. Assert output type is `pd.DataFrame`.
        """
        assert isinstance(ts.macd_histogram_reversal_strategy(make_ohlcv(uptrend(60))), pd.DataFrame)

    def test_required_columns_present(self):
        """
        Goal:
            Verify existence of all expected MACD indicators and output columns.

        Execution Principle:
            1. Execute strategy on synthetic OHLCV data.
            2. Confirm output contains `Close`, `MACD`, `Signal_Line`, `MACD_Histogram`, and `Signal`.
        """
        result = ts.macd_histogram_reversal_strategy(make_ohlcv(uptrend(60)))
        _assert_structure(result, ["Close", "MACD", "Signal_Line", "MACD_Histogram", "Signal"])

    def test_signals_are_valid(self):
        """
        Goal:
            Verify that all signals generated across cyclical sine wave data belong to the valid signal set.

        Execution Principle:
            1. Compute MACD signals over 100 bars of sine wave data.
            2. Run `_assert_valid_signals` to ensure every signal is 'Buy', 'Sell', or 'Hold'.
        """
        result = ts.macd_histogram_reversal_strategy(make_ohlcv(sine_wave(100)))
        _assert_valid_signals(result)

    def test_histogram_equals_macd_minus_signal_line(self):
        """
        Goal:
            Verify mathematical identity: Histogram = MACD - Signal_Line across all bars.

        Execution Principle:
            1. Compute indicators over 100 bars of sine wave data.
            2. Calculate absolute difference between histogram column and (MACD - Signal_Line).
            3. Assert maximum difference is strictly within floating point threshold (`< 1e-10`).
        """
        result = ts.macd_histogram_reversal_strategy(make_ohlcv(sine_wave(100)))
        diff = (result["MACD_Histogram"] - (result["MACD"] - result["Signal_Line"])).abs()
        assert (diff < 1e-10).all(), "MACD_Histogram does not equal MACD - Signal_Line"

    def test_buy_signal_on_histogram_zero_cross_up(self):
        """
        Goal:
            Verify a transition from negative to positive histogram in a V-shaped recovery yields at least one Buy signal.

        Execution Principle:
            1. Generate synthetic V-shaped price history.
            2. Run strategy and check for 'Buy' presence in the Signal series.
        """
        prices = v_shape(n_down=50, n_up=60, start=200.0, step=1.0)
        result = ts.macd_histogram_reversal_strategy(make_ohlcv(prices))
        assert "Buy" in result["Signal"].values

    def test_sell_signal_on_histogram_zero_cross_down(self):
        """
        Goal:
            Verify a transition from positive to negative histogram in an inverted-V decline yields at least one Sell signal.

        Execution Principle:
            1. Generate synthetic inverted-V price history.
            2. Run strategy and check for 'Sell' presence in the Signal series.
        """
        prices = inverted_v(n_up=50, n_down=60, start=100.0, step=1.0)
        result = ts.macd_histogram_reversal_strategy(make_ohlcv(prices))
        assert "Sell" in result["Signal"].values

    def test_buy_where_histogram_crosses_zero_upward(self):
        """
        Goal:
            Verify every bar labelled 'Buy' satisfies the exact upward zero-crossing predicate (`prev < 0 and curr >= 0`).

        Execution Principle:
            1. Extract all rows where Signal is 'Buy'.
            2. For each Buy bar, look up the previous bar's histogram value.
            3. Assert `prev_hist < 0` and `curr_hist >= 0`.
        """
        prices = v_shape(n_down=50, n_up=60, start=200.0, step=1.0)
        result = ts.macd_histogram_reversal_strategy(make_ohlcv(prices))
        buys = result[result["Signal"] == "Buy"]
        for idx in buys.index:
            pos = result.index.get_loc(idx)
            if pos == 0:
                continue
            prev_hist = result["MACD_Histogram"].iloc[pos - 1]
            curr_hist = result["MACD_Histogram"].iloc[pos]
            assert prev_hist < 0 and curr_hist >= 0, \
                f"Buy at {idx}: histogram did not cross zero upward " \
                f"(prev={prev_hist:.4f}, curr={curr_hist:.4f})"

    def test_sell_where_histogram_crosses_zero_downward(self):
        """
        Goal:
            Verify every bar labelled 'Sell' satisfies the exact downward zero-crossing predicate (`prev > 0 and curr <= 0`).

        Execution Principle:
            1. Extract all rows where Signal is 'Sell'.
            2. For each Sell bar, look up the previous bar's histogram value.
            3. Assert `prev_hist > 0` and `curr_hist <= 0`.
        """
        prices = inverted_v(n_up=50, n_down=60, start=100.0, step=1.0)
        result = ts.macd_histogram_reversal_strategy(make_ohlcv(prices))
        sells = result[result["Signal"] == "Sell"]
        for idx in sells.index:
            pos = result.index.get_loc(idx)
            if pos == 0:
                continue
            prev_hist = result["MACD_Histogram"].iloc[pos - 1]
            curr_hist = result["MACD_Histogram"].iloc[pos]
            assert prev_hist > 0 and curr_hist <= 0, \
                f"Sell at {idx}: histogram did not cross zero downward " \
                f"(prev={prev_hist:.4f}, curr={curr_hist:.4f})"

    def test_does_not_mutate_input_dataframe(self):
        """
        Goal:
            Verify that running `macd_histogram_reversal_strategy` does not modify input DataFrame columns.

        Execution Principle:
            1. Record columns of input OHLCV DataFrame.
            2. Execute strategy.
            3. Verify original DataFrame columns remain identical.
        """
        df = make_ohlcv(uptrend(60))
        original_cols = list(df.columns)
        ts.macd_histogram_reversal_strategy(df)
        assert list(df.columns) == original_cols
