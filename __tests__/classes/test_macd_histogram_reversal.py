# ===========================================================================
# 5.  MACD Histogram Reversal
# ===========================================================================
from advisor import ts
from src.backend.advisor.utilities import ohlvc, make_ohlcv, uptrend, downtrend, v_shape, inverted_v, sine_wave, _assert_structure, _assert_valid_sign
from src.backend.advisor.utilities import _assert_valid_signals
class TestMacdHistogramReversal:
    """Tests for macd_histogram_reversal_strategy()."""

    def test_returns_dataframe(self):
        assert isinstance(ts.macd_histogram_reversal_strategy(make_ohlcv(uptrend(60))), pd.DataFrame)

    def test_required_columns_present(self):
        result = ts.macd_histogram_reversal_strategy(make_ohlcv(uptrend(60)))
        _assert_structure(result, ["Close", "MACD", "Signal_Line", "MACD_Histogram", "Signal"])

    def test_signals_are_valid(self):
        result = ts.macd_histogram_reversal_strategy(make_ohlcv(sine_wave(100)))
        _assert_valid_signals(result)

    def test_histogram_equals_macd_minus_signal_line(self):
        """Mathematical identity: Histogram = MACD − Signal_Line."""
        result = ts.macd_histogram_reversal_strategy(make_ohlcv(sine_wave(100)))
        diff = (result["MACD_Histogram"] - (result["MACD"] - result["Signal_Line"])).abs()
        assert (diff < 1e-10).all(), "MACD_Histogram does not equal MACD - Signal_Line"

    def test_buy_signal_on_histogram_zero_cross_up(self):
        """
        A V-shape causes the MACD histogram to cross from negative to positive,
        which must produce at least one Buy signal.
        """
        prices = v_shape(n_down=50, n_up=60, start=200.0, step=1.0)
        result = ts.macd_histogram_reversal_strategy(make_ohlcv(prices))
        assert "Buy" in result["Signal"].values

    def test_sell_signal_on_histogram_zero_cross_down(self):
        """
        An inverted-V causes the histogram to cross from positive to negative,
        producing at least one Sell signal.
        """
        prices = inverted_v(n_up=50, n_down=60, start=100.0, step=1.0)
        result = ts.macd_histogram_reversal_strategy(make_ohlcv(prices))
        assert "Sell" in result["Signal"].values

    def test_buy_where_histogram_crosses_zero_upward(self):
        """
        Every bar labelled Buy must have a histogram that flipped from negative
        on the previous bar to non-negative on the current bar.
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
        Every bar labelled Sell must have a histogram that flipped from positive
        on the previous bar to non-positive on the current bar.
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
        df = make_ohlcv(uptrend(60))
        original_cols = list(df.columns)
        ts.macd_histogram_reversal_strategy(df)
        assert list(df.columns) == original_cols
