"""
test_intraday_contract.py
Contract tests for TradingStrategy.downloadData() intraday interval support.

Verifies:
  - Valid period/interval combinations download data without raising errors
    and return a properly shaped DataFrame.
  - Invalid combinations (period exceeds interval's data cap, unknown values)
    raise ValueError before any network call is made.
  - Downstream strategy functions (atr_trailing_stop, ma_rsi_strategy) run
    without errors on intraday OHLCV data.

Run with:
    pytest test_intraday_contract.py -v
"""

import sys
import os
import pytest
import pandas as pd

# Make sure local modules are importable when pytest is run from any cwd.
_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
for _path in (_ROOT, _DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utilities.test_helpers import assert_ohlcv as _assert_ohlcv

import trading_strategy as ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKER = "AAPL"   # liquid, always available


# ---------------------------------------------------------------------------
# 1. Validation — rejects bad inputs before network I/O
# ---------------------------------------------------------------------------

class TestValidationErrors:
    """Period/interval validation raises ValueError with no network call."""

    def test_unknown_period(self):
        with pytest.raises(ValueError, match="period"):
            ts.downloadData(TICKER, period="bad", interval="1d")

    def test_unknown_interval(self):
        with pytest.raises(ValueError, match="interval"):
            ts.downloadData(TICKER, period="1mo", interval="bad")

    def test_1m_exceeds_7_day_cap(self):
        """1m interval supports at most 7 calendar days."""
        with pytest.raises(ValueError, match="1m"):
            ts.downloadData(TICKER, period="1mo", interval="1m")

    def test_2m_exceeds_60_day_cap(self):
        with pytest.raises(ValueError, match="2m"):
            ts.downloadData(TICKER, period="3mo", interval="2m")

    def test_5m_exceeds_60_day_cap(self):
        with pytest.raises(ValueError, match="5m"):
            ts.downloadData(TICKER, period="6mo", interval="5m")

    def test_15m_exceeds_60_day_cap(self):
        with pytest.raises(ValueError, match="15m"):
            ts.downloadData(TICKER, period="1y", interval="15m")

    def test_30m_exceeds_60_day_cap(self):
        with pytest.raises(ValueError, match="30m"):
            ts.downloadData(TICKER, period="6mo", interval="30m")

    def test_90m_exceeds_60_day_cap(self):
        with pytest.raises(ValueError, match="90m"):
            ts.downloadData(TICKER, period="3mo", interval="90m")

    def test_60m_exceeds_730_day_cap(self):
        with pytest.raises(ValueError, match="60m"):
            ts.downloadData(TICKER, period="5y", interval="60m")

    def test_1h_exceeds_730_day_cap(self):
        with pytest.raises(ValueError, match="1h"):
            ts.downloadData(TICKER, period="5y", interval="1h")


# ---------------------------------------------------------------------------
# 2. Happy-path downloads — valid intraday combinations
# ---------------------------------------------------------------------------

class TestIntradayDownloadContract:
    """Valid intraday period/interval pairs return non-empty OHLCV data."""

    def test_1m_5d(self):
        df = ts.downloadData(TICKER, period="5d", interval="1m")
        _assert_ohlcv(df, min_rows=10)

    def test_2m_1mo(self):
        df = ts.downloadData(TICKER, period="1mo", interval="2m")
        _assert_ohlcv(df, min_rows=10)

    def test_5m_1mo(self):
        df = ts.downloadData(TICKER, period="1mo", interval="5m")
        _assert_ohlcv(df, min_rows=10)

    def test_15m_1mo(self):
        df = ts.downloadData(TICKER, period="1mo", interval="15m")
        _assert_ohlcv(df, min_rows=5)

    def test_30m_1mo(self):
        df = ts.downloadData(TICKER, period="1mo", interval="30m")
        _assert_ohlcv(df, min_rows=5)

    def test_60m_1y(self):
        df = ts.downloadData(TICKER, period="1y", interval="60m")
        _assert_ohlcv(df, min_rows=50)

    def test_90m_1mo(self):
        df = ts.downloadData(TICKER, period="1mo", interval="90m")
        _assert_ohlcv(df, min_rows=5)

    def test_1h_1y(self):
        df = ts.downloadData(TICKER, period="1y", interval="1h")
        _assert_ohlcv(df, min_rows=50)

    def test_1d_3mo(self):
        """Daily baseline — must always work."""
        df = ts.downloadData(TICKER, period="3mo", interval="1d")
        _assert_ohlcv(df, min_rows=50)

    def test_1wk_1y(self):
        df = ts.downloadData(TICKER, period="1y", interval="1wk")
        _assert_ohlcv(df, min_rows=40)

    def test_1mo_5y(self):
        df = ts.downloadData(TICKER, period="5y", interval="1mo")
        _assert_ohlcv(df, min_rows=50)


# ---------------------------------------------------------------------------
# 3. Strategy compatibility — intraday data runs through ATR & MA-RSI
# ---------------------------------------------------------------------------

class TestStrategyCompatibilityIntraday:
    """Strategies must not throw when fed intraday OHLCV bars."""

    @pytest.fixture(scope="class")
    def data_5m(self):
        df = ts.downloadData(TICKER, period="1mo", interval="5m")
        df.dropna(inplace=True)
        return df

    @pytest.fixture(scope="class")
    def data_1h(self):
        df = ts.downloadData(TICKER, period="1y", interval="1h")
        df.dropna(inplace=True)
        return df

    def test_atr_trailing_stop_5m_no_error(self, data_5m):
        result = ts.atr_trailing_stop(data_5m)
        assert "Signal" in result.columns
        assert set(result["Signal"].unique()).issubset({"Buy", "Sell", "Hold"})

    def test_ma_rsi_strategy_5m_no_error(self, data_5m):
        result = ts.ma_rsi_strategy(data_5m)
        assert "Signal" in result.columns
        assert set(result["Signal"].unique()).issubset({"Buy", "Sell", "Hold"})

    def test_atr_trailing_stop_1h_no_error(self, data_1h):
        result = ts.atr_trailing_stop(data_1h)
        assert "Signal" in result.columns
        assert set(result["Signal"].unique()).issubset({"Buy", "Sell", "Hold"})

    def test_ma_rsi_strategy_1h_no_error(self, data_1h):
        result = ts.ma_rsi_strategy(data_1h)
        assert "Signal" in result.columns
        assert set(result["Signal"].unique()).issubset({"Buy", "Sell", "Hold"})

    def test_news_sentiment_signal_empty_df_no_error(self):
        """news_sentiment_signal must gracefully handle empty input."""
        empty = pd.DataFrame(columns=["Date", "Signed_Score"])
        result = ts.news_sentiment_signal(empty)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["daily_sentiment", "Signal"]

    @pytest.mark.parametrize("period,interval", [
        ("5d",  "1m"),
        ("1mo", "5m"),
        ("1mo", "30m"),
        ("1y",  "1h"),
    ])
    def test_to_naive_s_strips_tz_on_intraday(self, period, interval):
        """
        to_naive_s() must not raise when the yfinance index is tz-aware (intraday)
        and must produce a tz-naive datetime64[s] Series.
        """
        df = ts.downloadData(TICKER, period=period, interval=interval)
        reset = df.reset_index()
        result = ts.to_naive_s(reset["Datetime"] if "Datetime" in reset.columns else reset["Date"])
        assert result.dtype == "datetime64[s]", f"Expected datetime64[s], got {result.dtype}"
        assert result.dt.tz is None, "Result must be tz-naive"


# ---------------------------------------------------------------------------
# 4. Column shape contract — every download returns the same column set
# ---------------------------------------------------------------------------

class TestColumnShapeContract:
    """downloadData always returns Open/High/Low/Close/Volume regardless of interval."""

    @pytest.mark.parametrize("period,interval", [
        ("5d",  "1m"),
        ("1mo", "5m"),
        ("1mo", "15m"),
        ("1mo", "30m"),
        ("1y",  "1h"),
        ("3mo", "1d"),
        ("1y",  "1wk"),
    ])
    def test_ohlcv_columns_present(self, period, interval):
        df = ts.downloadData(TICKER, period=period, interval=interval)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in df.columns, (
                f"Column '{col}' missing for period={period} interval={interval}"
            )

    @pytest.mark.parametrize("period,interval", [
        ("5d",  "1m"),
        ("1mo", "5m"),
        ("1mo", "15m"),
        ("1mo", "30m"),
        ("1y",  "1h"),
        ("3mo", "1d"),
    ])
    def test_no_multiindex_columns(self, period, interval):
        """Columns must be a flat Index, not a MultiIndex."""
        df = ts.downloadData(TICKER, period=period, interval=interval)
        assert not isinstance(df.columns, pd.MultiIndex), (
            f"MultiIndex columns returned for period={period} interval={interval}"
        )
