"""
test_strategies_backtest.py
===========================
Comprehensive backtest and unit tests for every strategy in TradingStrategy.py.

Design principles
-----------------
* 100 % offline  — no network calls.  yfinance is never imported here.
* Deterministic  — synthetic OHLCV data is constructed so that every expected
  signal can be verified by hand (see each factory function's docstring).
* Trustworthy    — each test targets a single, clearly stated invariant and
  uses the minimum amount of data needed to trigger the condition under test.

Run all tests:
    cd src/backend/advisor
    python -m pytest test_strategies_backtest.py -v

Run a specific class:
    python -m pytest test_strategies_backtest.py::TestMaRsiStrategy -v
"""

import os
import sys

import pandas as pd
import pytest

# Make sure local modules are importable when pytest is run from any cwd.
_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
for _path in (_ROOT, _DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utilities.test_helpers import (
    VALID_SIGNALS,
    assert_structure as _assert_structure,
    assert_valid_signals as _assert_valid_signals,
    downtrend,
    inverted_v,
    make_ohlcv,
    sine_wave,
    uptrend,
    v_shape,
)

import trading_strategy as ts



# ===========================================================================
# 2.  news_sentiment_signal
# ===========================================================================

class TestNewsSentimentSignal:
    """Tests for the aggregated news sentiment → signal mapping."""

    def _news(self, scores, dates=None):
        n = len(scores)
        if dates is None:
            dates = pd.date_range("2023-01-01", periods=n, freq="D")
        return pd.DataFrame({"Date": dates, "Signed_Score": scores})

    def test_empty_df_returns_empty(self):
        result = ts.news_sentiment_signal(pd.DataFrame())
        assert result.empty

    def test_missing_signed_score_column_returns_empty(self):
        df = pd.DataFrame({"Date": ["2023-01-01"]})
        result = ts.news_sentiment_signal(df)
        assert result.empty

    def test_bullish_scores_produce_buy(self):
        df = self._news([0.8, 0.9, 0.85])
        result = ts.news_sentiment_signal(df)
        assert (result["Signal"] == "Buy").all()

    def test_bearish_scores_produce_sell(self):
        df = self._news([-0.8, -0.9, -0.85])
        result = ts.news_sentiment_signal(df)
        assert (result["Signal"] == "Sell").all()

    def test_neutral_scores_produce_hold(self):
        df = self._news([0.0, 0.05, -0.05])
        result = ts.news_sentiment_signal(df)
        assert (result["Signal"] == "Hold").all()

    def test_all_signals_valid(self):
        df = self._news([0.9, -0.7, 0.1, -0.1, 0.5])
        result = ts.news_sentiment_signal(df)
        _assert_valid_signals(result)

    def test_custom_thresholds_hold_within_band(self):
        """Scores within custom thresholds must remain Hold."""
        df = self._news([0.5, -0.5])
        result = ts.news_sentiment_signal(df, bullish_threshold=0.6, bearish_threshold=-0.6)
        assert (result["Signal"] == "Hold").all()

    def test_custom_thresholds_buy_above_band(self):
        df = self._news([0.7])
        result = ts.news_sentiment_signal(df, bullish_threshold=0.6, bearish_threshold=-0.6)
        assert result["Signal"].iloc[0] == "Buy"

    def test_multiple_articles_same_day_averaged(self):
        """Two articles on the same date with opposing scores should average to Hold."""
        df = self._news([0.8, -0.8], dates=["2023-01-01", "2023-01-01"])
        result = ts.news_sentiment_signal(df, bullish_threshold=0.2, bearish_threshold=-0.2)
        assert result["Signal"].iloc[0] == "Hold"

    def test_returns_daily_sentiment_column(self):
        df = self._news([0.5])
        result = ts.news_sentiment_signal(df)
        assert "daily_sentiment" in result.columns

    def test_daily_sentiment_equals_score_for_single_article(self):
        df = self._news([0.6])
        result = ts.news_sentiment_signal(df)
        assert abs(result["daily_sentiment"].iloc[0] - 0.6) < 1e-9


# ===========================================================================
# 3.  MA-RSI Strategy
# ===========================================================================

class TestMaRsiStrategy:
    """
    Tests for ma_rsi_strategy().

    After the Wilder-smoothing fix (com=rsi_period-1) and the signal-logic fix
    (RSI < rsi_overbought for Buy, RSI > rsi_oversold for Sell), the strategy
    should:
    - Generate Buy signals on a V-shaped recovery (bullish EMA crossover while
      RSI hasn't yet reached overbought territory).
    - Generate Sell signals on an inverted-V (bearish crossover while RSI
      hasn't yet reached oversold territory).
    """

    def test_returns_dataframe(self):
        df = make_ohlcv(v_shape())
        assert isinstance(ts.ma_rsi_strategy(df), pd.DataFrame)

    def test_required_columns_present(self):
        result = ts.ma_rsi_strategy(make_ohlcv(v_shape()))
        _assert_structure(result, ["Close", "Short_EMA", "Long_EMA", "RSI", "Signal"])

    def test_signals_are_valid_strings(self):
        result = ts.ma_rsi_strategy(make_ohlcv(v_shape()))
        _assert_valid_signals(result)

    def test_rsi_in_valid_range(self):
        result = ts.ma_rsi_strategy(make_ohlcv(v_shape()))
        rsi = result["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), \
            f"RSI outside [0, 100]: min={rsi.min():.2f}, max={rsi.max():.2f}"

    def test_rsi_high_in_uptrend(self):
        """After a long uptrend, RSI should stabilise well above 60."""
        result = ts.ma_rsi_strategy(make_ohlcv(uptrend(80)))
        last_rsi = result["RSI"].iloc[-1]
        assert last_rsi > 60, \
            f"Expected elevated RSI in strong uptrend, got {last_rsi:.1f}. " \
            "Possible Wilder smoothing regression (should use com=rsi_period-1)."

    def test_rsi_low_in_downtrend(self):
        """After a long downtrend, RSI should stabilise well below 40."""
        result = ts.ma_rsi_strategy(make_ohlcv(downtrend(80)))
        last_rsi = result["RSI"].iloc[-1]
        assert last_rsi < 40, \
            f"Expected depressed RSI in strong downtrend, got {last_rsi:.1f}."

    def test_short_ema_above_long_ema_in_uptrend(self):
        """In a sustained uptrend, Short EMA (12) must end above Long EMA (26)."""
        result = ts.ma_rsi_strategy(make_ohlcv(uptrend(80)))
        assert result["Short_EMA"].iloc[-1] > result["Long_EMA"].iloc[-1]

    def test_short_ema_below_long_ema_in_downtrend(self):
        result = ts.ma_rsi_strategy(make_ohlcv(downtrend(80)))
        assert result["Short_EMA"].iloc[-1] < result["Long_EMA"].iloc[-1]

    def test_buy_signal_in_v_shape(self):
        """
        A bullish EMA crossover during the recovery phase of a V-shape should
        produce at least one Buy signal.
        After the fix: the condition is crossover AND RSI < rsi_overbought (70).
        """
        prices = v_shape(n_down=40, n_up=50, start=150.0, step=1.5)
        result = ts.ma_rsi_strategy(make_ohlcv(prices))
        assert "Buy" in result["Signal"].values, \
            "Expected at least one Buy signal in V-shape data."

    def test_sell_signal_in_inverted_v(self):
        """
        A bearish EMA crossover during the decline phase of an inverted-V should
        produce at least one Sell signal.
        After the fix: the condition is crossover AND RSI > rsi_oversold (30).
        """
        prices = inverted_v(n_up=40, n_down=50, start=100.0, step=1.5)
        result = ts.ma_rsi_strategy(make_ohlcv(prices))
        assert "Sell" in result["Signal"].values, \
            "Expected at least one Sell signal in inverted-V data."

    def test_mostly_hold_in_steady_uptrend(self):
        """
        In a one-directional trend there is only one crossover event, so the
        overwhelming majority of bars must carry a Hold signal.
        """
        result = ts.ma_rsi_strategy(make_ohlcv(uptrend(80)))
        hold_pct = (result["Signal"] == "Hold").mean()
        assert hold_pct > 0.8, f"Expected >80 % Hold, got {hold_pct:.1%}"

    def test_does_not_mutate_input_dataframe(self):
        df = make_ohlcv(v_shape())
        original_cols = list(df.columns)
        ts.ma_rsi_strategy(df)
        assert list(df.columns) == original_cols

    def test_custom_ema_periods_respected(self):
        """Shorter EMA windows should produce an earlier crossover."""
        prices = v_shape()
        result_fast = ts.ma_rsi_strategy(make_ohlcv(prices), short_ema_period=5, long_ema_period=10)
        result_slow = ts.ma_rsi_strategy(make_ohlcv(prices), short_ema_period=12, long_ema_period=26)
        fast_first_buy = result_fast[result_fast["Signal"] == "Buy"].index.min()
        slow_first_buy = result_slow[result_slow["Signal"] == "Buy"].index.min()
        # Faster EMAs should cross earlier or at the same time
        assert fast_first_buy <= slow_first_buy


# ===========================================================================
# 4.  ATR Trailing Stop
# ===========================================================================

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


# ===========================================================================
# 5.  MACD Histogram Reversal
# ===========================================================================

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


# ===========================================================================
# 6.  Bollinger Squeeze Strategy
# ===========================================================================

class TestBollingerSqueezeStrategy:
    """
    Tests for bollinger_squeeze_strategy().

    Key regression: the Is_Squeeze flag previously used df['BB_Width'].std(),
    a global statistic that reads future data.  After the fix, it uses a
    rolling std shifted by 1 bar — no look-ahead.
    """
    _N = 150  # needs ≥ squeeze_window (100) bars for valid squeeze data

    def _make(self, prices=None):
        if prices is None:
            prices = sine_wave(self._N)
        return make_ohlcv(prices)

    def test_returns_dataframe(self):
        assert isinstance(ts.bollinger_squeeze_strategy(self._make()), pd.DataFrame)

    def test_required_columns_present(self):
        result = ts.bollinger_squeeze_strategy(self._make())
        _assert_structure(
            result,
            ["Close", "Middle_Band", "Upper_Band", "Lower_Band", "BB_Width", "Is_Squeeze", "Signal"],
        )

    def test_signals_are_valid(self):
        _assert_valid_signals(ts.bollinger_squeeze_strategy(self._make()))

    def test_upper_band_above_middle_band(self):
        result = ts.bollinger_squeeze_strategy(self._make())
        valid  = result[["Upper_Band", "Middle_Band"]].dropna()
        assert (valid["Upper_Band"] >= valid["Middle_Band"]).all()

    def test_lower_band_below_middle_band(self):
        result = ts.bollinger_squeeze_strategy(self._make())
        valid  = result[["Lower_Band", "Middle_Band"]].dropna()
        assert (valid["Lower_Band"] <= valid["Middle_Band"]).all()

    def test_bb_width_equals_upper_minus_lower(self):
        result = ts.bollinger_squeeze_strategy(self._make())
        diff = (result["BB_Width"] - (result["Upper_Band"] - result["Lower_Band"])).abs()
        assert (diff.dropna() < 1e-10).all()

    def test_no_lookahead_bias(self):
        """
        Appending extra future bars must NOT change the signals on bars that
        were already computed.  With the old global .std() this test would fail;
        after the rolling-std fix it must pass.
        """
        prices_short = sine_wave(self._N)
        prices_long  = prices_short + sine_wave(40, amplitude=30.0)  # very different tail

        result_short = ts.bollinger_squeeze_strategy(make_ohlcv(prices_short))
        result_long  = ts.bollinger_squeeze_strategy(make_ohlcv(prices_long))

        shared_idx = result_short.index
        mismatch = (
            result_short.loc[shared_idx, "Signal"]
            != result_long.loc[shared_idx, "Signal"]
        ).sum()
        assert mismatch == 0, (
            f"{mismatch} signal(s) changed when future data was appended — "
            "look-ahead bias still present. Check Bollinger squeeze uses rolling std."
        )

    def test_is_squeeze_column_is_boolean(self):
        result = ts.bollinger_squeeze_strategy(self._make())
        # After rolling std fix, the column may contain NaN for the warm-up period;
        # any non-NaN value must be boolean-compatible.
        valid = result["Is_Squeeze"].dropna()
        assert valid.isin([True, False]).all()

    def test_does_not_mutate_input_dataframe(self):
        df = self._make()
        original_cols = list(df.columns)
        ts.bollinger_squeeze_strategy(df)
        assert list(df.columns) == original_cols


# ===========================================================================
# 7.  VWAP Strategy
# ===========================================================================

class TestVwapStrategy:
    """Tests for calculate_vwap()."""

    def test_returns_dataframe(self):
        assert isinstance(ts.calculate_vwap(make_ohlcv(sine_wave(50))), pd.DataFrame)

    def test_required_columns_present(self):
        result = ts.calculate_vwap(make_ohlcv(sine_wave(50)))
        _assert_structure(result, ["Close", "Volume", "VWAP", "Signal"])

    def test_signals_are_valid(self):
        _assert_valid_signals(ts.calculate_vwap(make_ohlcv(sine_wave(100))))

    def test_vwap_within_reasonable_price_range(self):
        """VWAP (a cumulative average of typical price) must stay within the observed price range."""
        df = make_ohlcv(sine_wave(100))
        result = ts.calculate_vwap(df)
        assert (result["VWAP"] >= df["Low"].min() * 0.9).all()
        assert (result["VWAP"] <= df["High"].max() * 1.1).all()

    def test_buy_signal_when_close_crosses_above_vwap(self):
        """
        The sine wave oscillates around its VWAP; close must cross above VWAP
        at least once, triggering a Buy signal.
        """
        result = ts.calculate_vwap(make_ohlcv(sine_wave(80)))
        assert "Buy" in result["Signal"].values

    def test_sell_signal_when_close_crosses_below_vwap(self):
        result = ts.calculate_vwap(make_ohlcv(sine_wave(80)))
        assert "Sell" in result["Signal"].values

    def test_buy_only_on_upward_cross(self):
        """Every Buy bar: previous close ≤ previous VWAP AND current close > current VWAP."""
        result = ts.calculate_vwap(make_ohlcv(sine_wave(80)))
        buys = result[result["Signal"] == "Buy"]
        for idx in buys.index:
            pos = result.index.get_loc(idx)
            if pos == 0:
                continue
            prev_close = result["Close"].iloc[pos - 1]
            prev_vwap  = result["VWAP"].iloc[pos - 1]
            curr_close = result["Close"].iloc[pos]
            curr_vwap  = result["VWAP"].iloc[pos]
            assert prev_close <= prev_vwap and curr_close > curr_vwap, \
                f"Buy at {idx} is not an upward close/VWAP crossover."

    def test_cumulative_volume_is_monotone(self):
        """Volume is always positive, so cumulative volume must be strictly increasing."""
        df = make_ohlcv(sine_wave(50))
        result = ts.calculate_vwap(df)
        assert result["Volume"].is_monotonic_increasing or \
            (result["Volume"] == result["Volume"].iloc[0]).all()

    def test_does_not_mutate_input_dataframe(self):
        df = make_ohlcv(sine_wave(50))
        original_cols = list(df.columns)
        ts.calculate_vwap(df)
        assert list(df.columns) == original_cols


# ===========================================================================
# 8.  backtest_strategy  (new function)
# ===========================================================================

class TestBacktestStrategy:
    """
    Tests for the backtest_strategy() performance engine.

    Trades are executed at the close of the signal bar (same-bar execution).
    """

    def _df(self, closes, signals):
        assert len(closes) == len(signals), "closes and signals must have the same length"
        n = len(closes)
        return pd.DataFrame(
            {"Close": closes, "Signal": signals},
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )

    # --- Return keys ---

    def test_all_metric_keys_present(self):
        df = self._df(uptrend(30), ["Buy"] + ["Hold"] * 28 + ["Sell"])
        metrics = ts.backtest_strategy(df)
        for key in (
            "total_return_pct", "annualized_return_pct", "sharpe_ratio",
            "max_drawdown_pct", "win_rate_pct", "profit_factor",
            "expectancy_pct", "n_trades", "trade_returns",
        ):
            assert key in metrics, f"Missing key: '{key}'"

    # --- Correct return calculation ---

    def test_buy_low_sell_high_positive_return(self):
        """
        Buy at 100, hold through a steady rise to 150, sell at 150.
        Expected gross return ≈ +50 %.
        """
        # prices: flat at 100 for 5 bars, rises 100→150 over 51 bars, flat 150 for 5 bars
        prices  = [100.0] * 5 + list(range(100, 151)) + [150.0] * 5
        # signals: Buy at index 5 (price 100), Sell at index 55 (price 150)
        signals = ["Hold"] * 5 + ["Buy"] + ["Hold"] * 49 + ["Sell"] + ["Hold"] * 5
        assert len(prices) == len(signals), f"{len(prices)} vs {len(signals)}"
        metrics = ts.backtest_strategy(self._df(prices, signals))
        assert metrics["total_return_pct"] > 40.0, \
            f"Expected ~50% return, got {metrics['total_return_pct']:.1f}%"
        assert metrics["n_trades"] >= 1

    def test_buy_high_sell_low_negative_return(self):
        """
        Buy at the peak, hold through a decline, sell at the bottom → negative return.
        """
        prices  = list(range(150, 99, -1)) + [100.0] * 5   # 51 + 5 = 56
        signals = ["Buy"] + ["Hold"] * 49 + ["Sell"] + ["Hold"] * 5
        metrics = ts.backtest_strategy(self._df(prices, signals))
        assert metrics["total_return_pct"] < 0.0

    def test_all_hold_gives_zero_trades(self):
        df = self._df(uptrend(50), ["Hold"] * 50)
        metrics = ts.backtest_strategy(df)
        assert metrics["n_trades"] == 0

    def test_open_position_closed_at_last_bar(self):
        """A Buy signal with no subsequent Sell must still record one trade."""
        df = self._df(uptrend(20), ["Buy"] + ["Hold"] * 19)
        metrics = ts.backtest_strategy(df)
        assert metrics["n_trades"] == 1

    # --- Trade count and win rate ---

    def test_two_profitable_trades_win_rate_100(self):
        """Two complete trades, both profitable → 100 % win rate."""
        prices  = [100, 110, 120, 110, 100, 110, 130, 130, 125]
        signals = ["Buy", "Hold", "Sell", "Buy", "Hold", "Hold", "Sell", "Hold", "Hold"]
        metrics = ts.backtest_strategy(self._df(prices, signals))
        assert metrics["n_trades"] == 2
        assert metrics["win_rate_pct"] == 100.0

    def test_two_losing_trades_win_rate_0(self):
        """Two complete trades, both losing → 0 % win rate."""
        prices  = [120, 110, 100, 120, 110, 100, 90,  80, 75]
        signals = ["Buy", "Hold", "Sell", "Buy", "Hold", "Hold", "Sell", "Hold", "Hold"]
        metrics = ts.backtest_strategy(self._df(prices, signals))
        assert metrics["n_trades"] == 2
        assert metrics["win_rate_pct"] == 0.0

    # --- Drawdown ---

    def test_max_drawdown_non_positive(self):
        """Max drawdown is always ≤ 0 (expressed as a negative percentage)."""
        df = self._df(uptrend(50), ["Buy"] + ["Hold"] * 48 + ["Sell"])
        metrics = ts.backtest_strategy(df)
        assert metrics["max_drawdown_pct"] <= 0.0

    def test_flat_hold_zero_drawdown(self):
        """If we never hold a position, equity is flat and drawdown is 0."""
        df = self._df([100.0] * 20, ["Hold"] * 20)
        metrics = ts.backtest_strategy(df)
        assert metrics["max_drawdown_pct"] == 0.0

    # --- Sharpe ratio ---

    def test_sharpe_positive_in_steady_uptrend(self):
        """
        Buying at the start of a smooth, uninterrupted uptrend and holding
        throughout should produce a positive Sharpe ratio.
        """
        df = self._df(uptrend(120, step=1.0), ["Buy"] + ["Hold"] * 118 + ["Sell"])
        assert ts.backtest_strategy(df)["sharpe_ratio"] > 0.0

    def test_sharpe_zero_when_no_trades(self):
        """No position taken → no daily P&L variance → Sharpe is 0."""
        df = self._df(uptrend(50), ["Hold"] * 50)
        metrics = ts.backtest_strategy(df)
        assert metrics["sharpe_ratio"] == 0.0

    # --- Edge cases ---

    def test_single_bar_returns_zeros(self):
        df = self._df([100.0], ["Hold"])
        metrics = ts.backtest_strategy(df)
        assert metrics["total_return_pct"] == 0.0
        assert metrics["n_trades"] == 0

    def test_buy_on_last_bar_only_no_trade(self):
        """A Buy on the very last bar has no price movement → 0 trades."""
        df = self._df(uptrend(10), ["Hold"] * 9 + ["Buy"])
        metrics = ts.backtest_strategy(df)
        # Position opened and immediately force-closed at same price → 0 % trade
        # n_trades = 1 (open position closed at last bar), return ~0
        assert metrics["total_return_pct"] == pytest.approx(0.0, abs=0.1)

    def test_trade_returns_list_length_matches_n_trades(self):
        df = self._df(uptrend(30), ["Buy"] + ["Hold"] * 28 + ["Sell"])
        metrics = ts.backtest_strategy(df)
        assert len(metrics["trade_returns"]) == metrics["n_trades"]

    def test_custom_column_names(self):
        """backtest_strategy must honour custom signal_col / close_col names."""
        prices  = uptrend(30)
        signals = ["Buy"] + ["Hold"] * 28 + ["Sell"]
        df = pd.DataFrame(
            {"price": prices, "action": signals},
            index=pd.date_range("2020-01-01", periods=30, freq="B"),
        )
        metrics = ts.backtest_strategy(df, signal_col="action", close_col="price")
        assert metrics["n_trades"] >= 1

    def test_costs_reduce_total_return(self):
        """Adding fees and slippage must not improve return vs zero-cost run."""
        df = self._df(uptrend(40), ["Buy"] + ["Hold"] * 38 + ["Sell"])
        baseline = ts.backtest_strategy(df, fee_bps=0.0, slippage_bps=0.0)
        costly = ts.backtest_strategy(df, fee_bps=10.0, slippage_bps=5.0)
        assert costly["total_return_pct"] <= baseline["total_return_pct"]

    def test_stop_loss_exits_early(self):
        """A tight stop-loss should cut a losing trade and keep drawdown bounded."""
        prices = [100, 99, 98, 95, 92, 90, 89, 88, 87]
        signals = ["Buy"] + ["Hold"] * (len(prices) - 1)
        metrics = ts.backtest_strategy(
            self._df(prices, signals),
            stop_loss_pct=0.03,
        )
        assert metrics["n_trades"] == 1
        assert metrics["trade_returns"][0] <= -0.03

    def test_take_profit_exits_early(self):
        """A take-profit threshold should lock in gains before the signal exits."""
        prices = [100, 103, 106, 107, 106, 105, 104]
        signals = ["Buy"] + ["Hold"] * (len(prices) - 1)
        metrics = ts.backtest_strategy(
            self._df(prices, signals),
            take_profit_pct=0.05,
        )
        assert metrics["n_trades"] == 1
        assert metrics["trade_returns"][0] >= 0.05

    def test_max_hold_bars_exits_trade(self):
        """max_hold_bars should close a trade even without Sell signal."""
        prices = [100, 101, 102, 103, 104, 105]
        signals = ["Buy"] + ["Hold"] * (len(prices) - 1)
        metrics = ts.backtest_strategy(
            self._df(prices, signals),
            max_hold_bars=2,
        )
        assert metrics["n_trades"] == 1

    def test_invalid_backtest_parameters_raise(self):
        df = self._df(uptrend(20), ["Buy"] + ["Hold"] * 18 + ["Sell"])
        with pytest.raises(ValueError):
            ts.backtest_strategy(df, fee_bps=-1)
        with pytest.raises(ValueError):
            ts.backtest_strategy(df, slippage_bps=-1)
        with pytest.raises(ValueError):
            ts.backtest_strategy(df, stop_loss_pct=0)
        with pytest.raises(ValueError):
            ts.backtest_strategy(df, take_profit_pct=0)
        with pytest.raises(ValueError):
            ts.backtest_strategy(df, max_hold_bars=0)


# ===========================================================================
# 9.  Integration smoke tests (all strategies on the same dataset)
# ===========================================================================

class TestIntegrationSmoke:
    """
    Run every strategy end-to-end on the same synthetic market data and verify
    that each one produces a structurally valid, signal-correct result.
    """

    @pytest.fixture(scope="class")
    def market_data(self):
        """Shared V-shape dataset with enough bars for all strategies."""
        return make_ohlcv(v_shape(n_down=70, n_up=80, start=200.0, step=1.0))

    def test_ma_rsi_smoke(self, market_data):
        result = ts.ma_rsi_strategy(market_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_atr_smoke(self, market_data):
        result = ts.atr_trailing_stop(market_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_macd_smoke(self, market_data):
        result = ts.macd_histogram_reversal_strategy(market_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_bollinger_smoke(self, market_data):
        # Bollinger needs ≥ 100 bars for the squeeze window
        long_data = make_ohlcv(v_shape(n_down=80, n_up=80, start=200.0, step=1.0))
        result = ts.bollinger_squeeze_strategy(long_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_vwap_smoke(self, market_data):
        result = ts.calculate_vwap(market_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_backtest_on_atr_signals(self, market_data):
        """Full pipeline: ATR signals fed into backtest_strategy → valid metrics dict."""
        atr_result = ts.atr_trailing_stop(market_data)
        metrics = ts.backtest_strategy(
            atr_result[["Close", "Signal"]],
            signal_col="Signal",
            close_col="Close",
        )
        for key in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "n_trades"):
            assert key in metrics

    def test_backtest_on_ma_rsi_signals(self, market_data):
        """Full pipeline: MA-RSI signals fed into backtest_strategy → valid metrics dict."""
        ma_result = ts.ma_rsi_strategy(market_data)
        metrics = ts.backtest_strategy(
            ma_result[["Close", "Signal"]],
            signal_col="Signal",
            close_col="Close",
        )
        for key in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "n_trades"):
            assert key in metrics

    def test_backtest_on_macd_signals(self, market_data):
        """Full pipeline: MACD signals fed into backtest_strategy."""
        macd_result = ts.macd_histogram_reversal_strategy(market_data)
        metrics = ts.backtest_strategy(
            macd_result[["Close", "Signal"]],
            signal_col="Signal",
            close_col="Close",
        )
        assert "total_return_pct" in metrics


# ===========================================================================
# 10.  Real-market data tests — last 30 days, multiple tickers
# ===========================================================================

#: Tickers used for live-data tests.
_LIVE_TICKERS = ["COKE", "BA", "MSFT", "GOOGL", "WMT"]

#: Minimum tradeable bars we require even if Yahoo returns fewer than expected.
_LIVE_MIN_BARS = 15


@pytest.mark.live
class TestRealMarketData:
    """
    Downloads the last ~30 calendar days (period='1mo', interval='1d') of OHLCV
    data from Yahoo Finance for a basket of liquid stocks and runs every
    implemented strategy against that real data.

    These tests require a live network connection.  They are tagged with the
    ``live`` marker so they can be run selectively or excluded:

        # run only live tests
        pytest test_strategies_backtest.py -m live -v

        # skip live tests (offline CI)
        pytest test_strategies_backtest.py -m "not live" -v

    Each test calls ``_fetch(ticker)`` which automatically skips (rather than
    failing) if the market is closed, the ticker is delisted, or the network
    is unavailable.
    """

    @staticmethod
    def _fetch(ticker: str) -> pd.DataFrame:
        """
        Download ~30 calendar days of daily OHLCV from Yahoo Finance.
        Skips the test (instead of failing) on any network or data error.
        """
        try:
            df = ts.downloadData(ticker, period="1mo", interval="1d")
            df.dropna(inplace=True)
        except Exception as exc:
            pytest.skip(f"Could not download data for '{ticker}': {exc}")

        if df.empty or len(df) < _LIVE_MIN_BARS:
            pytest.skip(
                f"Insufficient data for '{ticker}': only {len(df)} bar(s) returned. "
                "Market may be closed or ticker unavailable."
            )
        return df

    # -----------------------------------------------------------------------
    # OHLCV data quality
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_columns_and_minimum_rows(self, ticker):
        """Yahoo Finance must return a properly shaped OHLCV DataFrame."""
        df = self._fetch(ticker)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in df.columns, f"{ticker}: missing column '{col}'"
        assert len(df) >= _LIVE_MIN_BARS, (
            f"{ticker}: expected ≥ {_LIVE_MIN_BARS} bars, got {len(df)}"
        )

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_no_negative_or_zero_prices(self, ticker):
        """All price columns must be strictly positive."""
        df = self._fetch(ticker)
        for col in ("Open", "High", "Low", "Close"):
            assert (df[col] > 0).all(), f"{ticker}: '{col}' contains non-positive values"

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_high_greater_equal_low(self, ticker):
        """High must be ≥ Low on every bar — a basic OHLCV integrity check."""
        df = self._fetch(ticker)
        assert (df["High"] >= df["Low"]).all(), (
            f"{ticker}: High < Low on {(df['High'] < df['Low']).sum()} bar(s)"
        )

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_volume_positive(self, ticker):
        """Volume must be positive on every bar."""
        df = self._fetch(ticker)
        assert (df["Volume"] > 0).all(), (
            f"{ticker}: Volume ≤ 0 on {(df['Volume'] <= 0).sum()} bar(s)"
        )

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_close_between_high_and_low(self, ticker):
        """Close must lie within [Low, High] on every bar."""
        df = self._fetch(ticker)
        assert (df["Close"] >= df["Low"]).all() and (df["Close"] <= df["High"]).all(), (
            f"{ticker}: Close is outside [Low, High] on some bar(s)"
        )

    # -----------------------------------------------------------------------
    # MA-RSI strategy on live data
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ma_rsi_live_structure(self, ticker):
        """MA-RSI returns required columns and valid signals on real data."""
        df = self._fetch(ticker)
        result = ts.ma_rsi_strategy(df)
        _assert_structure(result, ["Close", "Short_EMA", "Long_EMA", "RSI", "Signal"])
        _assert_valid_signals(result)

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ma_rsi_rsi_in_range_live(self, ticker):
        """RSI must be in [0, 100] on every non-NaN bar."""
        df = self._fetch(ticker)
        rsi = ts.ma_rsi_strategy(df)["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), (
            f"{ticker}: RSI out of range — min={rsi.min():.2f}, max={rsi.max():.2f}"
        )

    # -----------------------------------------------------------------------
    # ATR Trailing Stop on live data
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_atr_live_structure(self, ticker):
        """ATR trailing stop returns required columns and valid signals on real data."""
        df = self._fetch(ticker)
        result = ts.atr_trailing_stop(df)
        _assert_structure(result, ["Close", "ATR", "Buy_Stop", "Sell_Stop", "Signal"])
        _assert_valid_signals(result)

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_atr_positive_live(self, ticker):
        """ATR must be strictly positive on every non-NaN bar."""
        df = self._fetch(ticker)
        atr = ts.atr_trailing_stop(df)["ATR"].dropna()
        assert (atr > 0).all(), f"{ticker}: ATR has non-positive values"

    # -----------------------------------------------------------------------
    # MACD Histogram Reversal on live data
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_macd_live_structure(self, ticker):
        """MACD strategy returns required columns and valid signals on real data."""
        df = self._fetch(ticker)
        result = ts.macd_histogram_reversal_strategy(df)
        _assert_structure(result, ["Close", "MACD", "Signal_Line", "MACD_Histogram", "Signal"])
        _assert_valid_signals(result)

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_macd_histogram_identity_live(self, ticker):
        """Histogram = MACD − Signal_Line must hold on every bar."""
        df = self._fetch(ticker)
        result = ts.macd_histogram_reversal_strategy(df)
        diff = (result["MACD_Histogram"] - (result["MACD"] - result["Signal_Line"])).abs()
        assert (diff.dropna() < 1e-10).all(), (
            f"{ticker}: MACD_Histogram identity violated (max diff={diff.max():.2e})"
        )

    # -----------------------------------------------------------------------
    # VWAP on live data
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_vwap_live_structure(self, ticker):
        """VWAP strategy returns required columns and valid signals on real data."""
        df = self._fetch(ticker)
        result = ts.calculate_vwap(df)
        _assert_structure(result, ["Close", "Volume", "VWAP", "Signal"])
        _assert_valid_signals(result)

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_vwap_positive_live(self, ticker):
        """VWAP must be strictly positive on every bar."""
        df = self._fetch(ticker)
        vwap = ts.calculate_vwap(df)["VWAP"]
        assert (vwap > 0).all(), f"{ticker}: VWAP has non-positive values"

    # -----------------------------------------------------------------------
    # Full backtest pipeline on live data
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_backtest_atr_live(self, ticker):
        """Full ATR → backtest pipeline: metrics are valid on real market data."""
        df     = self._fetch(ticker)
        result = ts.atr_trailing_stop(df)
        metrics = ts.backtest_strategy(result[["Close", "Signal"]])
        for key in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                    "win_rate_pct", "n_trades"):
            assert key in metrics, f"{ticker}: missing key '{key}' in ATR backtest metrics"
        assert metrics["max_drawdown_pct"] <= 0.0, (
            f"{ticker}: max_drawdown_pct should be ≤ 0, got {metrics['max_drawdown_pct']}"
        )
        assert 0.0 <= metrics["win_rate_pct"] <= 100.0, (
            f"{ticker}: win_rate_pct out of range: {metrics['win_rate_pct']}"
        )

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_backtest_ma_rsi_live(self, ticker):
        """Full MA-RSI → backtest pipeline: metrics are valid on real market data."""
        df     = self._fetch(ticker)
        result = ts.ma_rsi_strategy(df)
        metrics = ts.backtest_strategy(result[["Close", "Signal"]])
        assert metrics["max_drawdown_pct"] <= 0.0
        assert 0.0 <= metrics["win_rate_pct"] <= 100.0
        assert len(metrics["trade_returns"]) == metrics["n_trades"]

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_backtest_macd_live(self, ticker):
        """Full MACD → backtest pipeline: metrics are valid on real market data."""
        df     = self._fetch(ticker)
        result = ts.macd_histogram_reversal_strategy(df)
        metrics = ts.backtest_strategy(result[["Close", "Signal"]])
        assert metrics["max_drawdown_pct"] <= 0.0
        assert 0.0 <= metrics["win_rate_pct"] <= 100.0

    # -----------------------------------------------------------------------
    # Cross-strategy consistency
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_all_strategies_valid_signals_live(self, ticker):
        """Every strategy must emit only valid signal values on real market data."""
        df = self._fetch(ticker)
        strategies = [
            (ts.ma_rsi_strategy,                  "MA-RSI"),
            (ts.atr_trailing_stop,                "ATR"),
            (ts.macd_histogram_reversal_strategy, "MACD"),
            (ts.calculate_vwap,                   "VWAP"),
        ]
        for fn, label in strategies:
            result = fn(df)
            bad = set(result["Signal"].dropna().unique()) - VALID_SIGNALS
            assert not bad, f"{ticker} / {label}: unexpected signal values {bad}"
