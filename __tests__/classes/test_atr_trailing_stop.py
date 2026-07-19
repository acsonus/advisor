# ===========================================================================
# 4.  ATR Trailing Stop
# ===========================================================================
from turtle import pd
from advisor import ts
from src.backend.advisor.utilities import ohlvc, make_ohlcv, uptrend, downtrend, v_shape, inverted_v, _assert_structure, _assert_valid_signals
class TestAtrTrailingStop:
    """Tests for atr_trailing_stop()."""
        
    def test_returns_dataframe(self):
        assert isinstance(ts.atr_trailing_stop(make_ohlcv(uptrend(30))), pd.DataFrame)

    def test_required_columns_present(self):
        result = ts.atr_trailing_stop(make_ohlcv(uptrend(30)))
        _assert_structure(result, ["Close", "ATR", "Buy_Stop", "Sell_Stop", "Signal"])

    def test_signals_are_valid(self):
        result = ts.atr_trailing_stop(make_ohlcv(uptrend(50)))
        _assert_valid_signals(result)

    def test_signal_column_no_nan(self):
        """Signal column must be fully populated (no NaN)."""
        result = ts.atr_trailing_stop(make_ohlcv(uptrend(50)))
        assert result["Signal"].notna().all()

    def test_atr_is_strictly_positive(self):
        result = ts.atr_trailing_stop(make_ohlcv(uptrend(50)))
        atr = result["ATR"].dropna()
        assert (atr > 0).all(), "ATR must be strictly positive"

    def test_buy_signal_in_strong_uptrend(self):
        """A strong, sustained uptrend must trigger at least one Buy signal."""
        result = ts.atr_trailing_stop(make_ohlcv(uptrend(60, step=3.0)))
        assert "Buy" in result["Signal"].values

    def test_sell_signal_in_strong_downtrend(self):
        """A strong, sustained downtrend must trigger at least one Sell signal."""
        result = ts.atr_trailing_stop(make_ohlcv(downtrend(60, step=3.0)))
        assert "Sell" in result["Signal"].values

    def test_buy_stop_does_not_decrease_while_long(self):
        """
        Trailing stop must only ratchet upward while in a long position.
        No single step should lower the stop by more than floating-point noise.
        """
        result = ts.atr_trailing_stop(make_ohlcv(uptrend(80, step=2.0)))
        buy_stops = result["Buy_Stop"].dropna()
        if len(buy_stops) > 1:
            drops = buy_stops.diff().dropna()
            assert (drops >= -1e-9).all(), \
                f"Buy stop decreased unexpectedly: worst drop = {drops.min():.4f}"

    def test_nan_guard_no_exception_on_first_entry(self):
        """
        The ATR loop used Python max() on a NaN value before the fix.
        This test verifies the guard prevents that failure.
        """
        # Prices designed to enter long immediately on bar 1 (step >> ATR)
        prices = uptrend(50, start=100.0, step=20.0)
        result = ts.atr_trailing_stop(make_ohlcv(prices))
        assert result["Signal"].notna().all()

    def test_does_not_mutate_input_dataframe(self):
        df = make_ohlcv(uptrend(30))
        original_cols = list(df.columns)
        ts.atr_trailing_stop(df)
        assert list(df.columns) == original_cols

    def test_custom_multiplier_widens_stop(self):
        """A larger ATR multiplier should yield a wider (lower) buy stop."""
        df = make_ohlcv(uptrend(60, step=2.0))
        result_tight = ts.atr_trailing_stop(df, atr_multiplier=1)
        result_wide  = ts.atr_trailing_stop(df, atr_multiplier=4)
        stops_tight  = result_tight["Buy_Stop"].dropna()
        stops_wide   = result_wide["Buy_Stop"].dropna()
        if len(stops_tight) > 0 and len(stops_wide) > 0:
            assert stops_wide.mean() < stops_tight.mean(), \
                "Wider multiplier should produce a lower trailing stop"

