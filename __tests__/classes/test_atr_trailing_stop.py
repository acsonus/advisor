# ===========================================================================
# 4.  ATR Trailing Stop
# ===========================================================================
import pandas as pd
import trading_strategy as ts
from utilities.test_helpers import make_ohlcv, uptrend, downtrend, v_shape, inverted_v, assert_structure as _assert_structure, assert_valid_signals as _assert_valid_signals
class TestAtrTrailingStop:
    """Tests for atr_trailing_stop()."""
        
    def test_returns_dataframe(self):
        """
        Goal:
            Verify that atr_trailing_stop returns a pandas DataFrame.

        Execution Principle:
            1. Generate synthetic OHLCV data using `uptrend(30)`.
            2. Run `atr_trailing_stop`.
            3. Assert the result is an instance of `pd.DataFrame`.
        """
        assert isinstance(ts.atr_trailing_stop(make_ohlcv(uptrend(30))), pd.DataFrame)

    def test_required_columns_present(self):
        """
        Goal:
            Ensure the returned DataFrame includes all required output columns.

        Execution Principle:
            1. Run `atr_trailing_stop` on synthetic OHLCV uptrend data.
            2. Verify existence of `['Close', 'ATR', 'Buy_Stop', 'Sell_Stop', 'Signal']` columns.
        """
        result = ts.atr_trailing_stop(make_ohlcv(uptrend(30)))
        _assert_structure(result, ["Close", "ATR", "Buy_Stop", "Sell_Stop", "Signal"])

    def test_signals_are_valid(self):
        """
        Goal:
            Verify all generated signals belong to the valid signal set.

        Execution Principle:
            1. Generate signals over 50 bars of synthetic uptrend data.
            2. Assert that all values in 'Signal' are strictly in `{'Buy', 'Sell', 'Hold'}`.
        """
        result = ts.atr_trailing_stop(make_ohlcv(uptrend(50)))
        _assert_valid_signals(result)

    def test_signal_column_no_nan(self):
        """
        Goal:
            Confirm the 'Signal' column contains no NaN values across the entire output series.

        Execution Principle:
            1. Compute strategy signals over 50 bars.
            2. Assert `result["Signal"].notna().all()` evaluates to True.
        """
        result = ts.atr_trailing_stop(make_ohlcv(uptrend(50)))
        assert result["Signal"].notna().all()

    def test_atr_is_strictly_positive(self):
        """
        Goal:
            Verify that ATR values are strictly positive for bars with price movement.

        Execution Principle:
            1. Drop early warmup NaN entries from the ATR column.
            2. Assert all computed ATR numbers are strictly greater than zero.
        """
        result = ts.atr_trailing_stop(make_ohlcv(uptrend(50)))
        atr = result["ATR"].dropna()
        assert (atr > 0).all(), "ATR must be strictly positive"

    def test_buy_signal_in_strong_uptrend(self):
        """
        Goal:
            Confirm a strong, sustained uptrend triggers at least one 'Buy' signal.

        Execution Principle:
            1. Generate steep upward pricing bars with `uptrend(60, step=3.0)`.
            2. Execute `atr_trailing_stop`.
            3. Assert that `'Buy'` is present in the `Signal` column.
        """
        result = ts.atr_trailing_stop(make_ohlcv(uptrend(60, step=3.0)))
        assert "Buy" in result["Signal"].values

    def test_sell_signal_in_strong_downtrend(self):
        """
        Goal:
            Confirm a strong, sustained downtrend triggers at least one 'Sell' signal.

        Execution Principle:
            1. Generate steep downward pricing bars with `downtrend(60, step=3.0)`.
            2. Execute `atr_trailing_stop`.
            3. Assert that `'Sell'` is present in the `Signal` column.
        """
        result = ts.atr_trailing_stop(make_ohlcv(downtrend(60, step=3.0)))
        assert "Sell" in result["Signal"].values

    def test_buy_stop_does_not_decrease_while_long(self):
        """
        Goal:
            Verify the trailing buy stop ratchets monotonically upward while holding a long position.

        Execution Principle:
            1. Execute strategy on an 80-bar uptrend.
            2. Calculate first differences (`diff()`) on `Buy_Stop`.
            3. Assert no discrete step decreases by more than floating point epsilon (`-1e-9`).
        """
        result = ts.atr_trailing_stop(make_ohlcv(uptrend(80, step=2.0)))
        buy_stops = result["Buy_Stop"].dropna()
        if len(buy_stops) > 1:
            drops = buy_stops.diff().dropna()
            assert (drops >= -1e-9).all(), \
                f"Buy stop decreased unexpectedly: worst drop = {drops.min():.4f}"

    def test_nan_guard_no_exception_on_first_entry(self):
        """
        Goal:
            Verify first-bar entry edge case does not raise TypeError or ValueError when comparing with NaN.

        Execution Principle:
            1. Generate aggressive price jump on early bars.
            2. Compute `atr_trailing_stop`.
            3. Ensure signals are generated without exceptions.
        """
        prices = uptrend(50, start=100.0, step=20.0)
        result = ts.atr_trailing_stop(make_ohlcv(prices))
        assert result["Signal"].notna().all()

    def test_does_not_mutate_input_dataframe(self):
        """
        Goal:
            Ensure the strategy execution function operates immutably on input DataFrames.

        Execution Principle:
            1. Capture columns of the input OHLCV DataFrame.
            2. Pass DataFrame into `atr_trailing_stop`.
            3. Verify columns of the original DataFrame are completely unchanged.
        """
        df = make_ohlcv(uptrend(30))
        original_cols = list(df.columns)
        ts.atr_trailing_stop(df)
        assert list(df.columns) == original_cols

    def test_custom_multiplier_widens_stop(self):
        """
        Goal:
            Verify increasing `atr_multiplier` widens the trailing stop distance below price.

        Execution Principle:
            1. Compute strategy with tight multiplier (1) and wide multiplier (4).
            2. Compare mean trailing stop levels.
            3. Assert the wide multiplier yields a significantly lower average stop level.
        """
        df = make_ohlcv(uptrend(60, step=2.0))
        result_tight = ts.atr_trailing_stop(df, atr_multiplier=1)
        result_wide  = ts.atr_trailing_stop(df, atr_multiplier=4)
        stops_tight  = result_tight["Buy_Stop"].dropna()
        stops_wide   = result_wide["Buy_Stop"].dropna()
        if len(stops_tight) > 0 and len(stops_wide) > 0:
            assert stops_wide.mean() < stops_tight.mean(), \
                "Wider multiplier should produce a lower trailing stop"

