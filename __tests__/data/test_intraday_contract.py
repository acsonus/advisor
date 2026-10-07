"""Contract tests for intraday interval validation and market data downloading."""

import pandas as pd
import pytest

from utilities.test_helpers import assert_ohlcv as _assert_ohlcv
from advisor.data import download_data, to_naive_s
from advisor.sentiment import news_sentiment_signal
from advisor.strategies import atr_trailing_stop, ma_rsi_strategy

TICKER = "AAPL"


class TestValidationErrors:
    """Period/interval validation raises ValueError with no network call."""

    def test_unknown_period(self):
        with pytest.raises(ValueError, match="period"):
            download_data(TICKER, period="bad", interval="1d")

    def test_unknown_interval(self):
        with pytest.raises(ValueError, match="interval"):
            download_data(TICKER, period="1mo", interval="bad")

    def test_1m_exceeds_7_day_cap(self):
        with pytest.raises(ValueError, match="1m"):
            download_data(TICKER, period="1mo", interval="1m")

    def test_2m_exceeds_60_day_cap(self):
        with pytest.raises(ValueError, match="2m"):
            download_data(TICKER, period="3mo", interval="2m")

    def test_5m_exceeds_60_day_cap(self):
        with pytest.raises(ValueError, match="5m"):
            download_data(TICKER, period="3mo", interval="5m")

    def test_15m_exceeds_60_day_cap(self):
        with pytest.raises(ValueError, match="15m"):
            download_data(TICKER, period="3mo", interval="15m")

    def test_30m_exceeds_60_day_cap(self):
        with pytest.raises(ValueError, match="30m"):
            download_data(TICKER, period="3mo", interval="30m")

    def test_90m_exceeds_60_day_cap(self):
        with pytest.raises(ValueError, match="90m"):
            download_data(TICKER, period="3mo", interval="90m")

    def test_60m_exceeds_730_day_cap(self):
        with pytest.raises(ValueError, match="60m"):
            download_data(TICKER, period="5y", interval="60m")

    def test_1h_exceeds_730_day_cap(self):
        with pytest.raises(ValueError, match="1h"):
            download_data(TICKER, period="5y", interval="1h")


@pytest.mark.live
class TestIntradaySupported:
    """Valid period/interval combinations download without error."""

    def test_1m_within_cap(self):
        df = download_data(TICKER, period="5d", interval="1m")
        _assert_ohlcv(df, min_rows=100)

    def test_2m_within_cap(self):
        df = download_data(TICKER, period="1mo", interval="2m")
        _assert_ohlcv(df, min_rows=100)

    def test_5m_within_cap(self):
        df = download_data(TICKER, period="1mo", interval="5m")
        _assert_ohlcv(df, min_rows=100)

    def test_15m_within_cap(self):
        df = download_data(TICKER, period="1mo", interval="15m")
        _assert_ohlcv(df, min_rows=50)

    def test_30m_within_cap(self):
        df = download_data(TICKER, period="1mo", interval="30m")
        _assert_ohlcv(df, min_rows=50)

    def test_60m_within_cap(self):
        df = download_data(TICKER, period="6mo", interval="60m")
        _assert_ohlcv(df, min_rows=100)

    def test_90m_within_cap(self):
        df = download_data(TICKER, period="1mo", interval="90m")
        _assert_ohlcv(df, min_rows=20)

    def test_1h_within_cap(self):
        df = download_data(TICKER, period="1y", interval="1h")
        _assert_ohlcv(df, min_rows=200)


@pytest.mark.live
class TestDailyWeeklyBaseline:
    """Daily and weekly intervals remain supported without lookback restrictions."""

    def test_1d_baseline(self):
        df = download_data(TICKER, period="3mo", interval="1d")
        _assert_ohlcv(df, min_rows=50)

    def test_1wk_1y(self):
        df = download_data(TICKER, period="1y", interval="1wk")
        _assert_ohlcv(df, min_rows=40)

    def test_1mo_5y(self):
        df = download_data(TICKER, period="5y", interval="1mo")
        _assert_ohlcv(df, min_rows=50)


@pytest.mark.live
class TestStrategyCompatibilityIntraday:
    """Strategies must not throw when fed intraday OHLCV bars."""

    @pytest.fixture(scope="class")
    @classmethod
    def data_5m(cls):
        df = download_data(TICKER, period="1mo", interval="5m")
        df.dropna(inplace=True)
        return df

    @pytest.fixture(scope="class")
    @classmethod
    def data_1h(cls):
        df = download_data(TICKER, period="1y", interval="1h")
        df.dropna(inplace=True)
        return df

    def test_atr_trailing_stop_5m_no_error(self, data_5m):
        result = atr_trailing_stop(data_5m)
        assert "Signal" in result.columns
        assert set(result["Signal"].unique()).issubset({"Buy", "Sell", "Hold"})

    def test_ma_rsi_strategy_5m_no_error(self, data_5m):
        result = ma_rsi_strategy(data_5m)
        assert "Signal" in result.columns
        assert set(result["Signal"].unique()).issubset({"Buy", "Sell", "Hold"})

    def test_atr_trailing_stop_1h_no_error(self, data_1h):
        result = atr_trailing_stop(data_1h)
        assert "Signal" in result.columns
        assert set(result["Signal"].unique()).issubset({"Buy", "Sell", "Hold"})

    def test_ma_rsi_strategy_1h_no_error(self, data_1h):
        result = ma_rsi_strategy(data_1h)
        assert "Signal" in result.columns
        assert set(result["Signal"].unique()).issubset({"Buy", "Sell", "Hold"})

    def test_news_sentiment_signal_empty_df_no_error(self):
        empty = pd.DataFrame(columns=["Date", "Signed_Score"])
        result = news_sentiment_signal(empty)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["daily_sentiment", "Signal"]

    @pytest.mark.parametrize("period,interval", [
        ("5d", "1m"),
        ("1mo", "5m"),
        ("1mo", "30m"),
        ("1y", "1h"),
    ])
    def test_to_naive_s_strips_tz_on_intraday(self, period, interval):
        df = download_data(TICKER, period=period, interval=interval)
        reset = df.reset_index()
        result = to_naive_s(reset["Datetime"] if "Datetime" in reset.columns else reset["Date"])
        assert result.dtype == "datetime64[s]", f"Expected datetime64[s], got {result.dtype}"
        assert result.dt.tz is None, "Result must be tz-naive"


@pytest.mark.live
class TestColumnShapes:
    """Intraday and daily downloads produce identical column structure."""

    REQUIRED = ["Open", "High", "Low", "Close", "Volume"]

    @pytest.mark.parametrize("interval,period", [
        ("1m",  "1d"),
        ("5m",  "5d"),
        ("15m", "5d"),
        ("30m", "1mo"),
        ("60m", "1mo"),
        ("1h",  "1mo"),
        ("1d",  "1mo"),
    ])
    def test_has_required_columns(self, interval, period):
        df = download_data(TICKER, period=period, interval=interval)
        for col in self.REQUIRED:
            assert col in df.columns, f"Missing '{col}' in {interval} DataFrame"
        assert not isinstance(df.columns, pd.MultiIndex), "Columns should be a flat Index"
