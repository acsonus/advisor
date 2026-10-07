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
        """
        Goal:
            Verify unsupported period strings trigger ValueError prior to issuing network requests.

        Execution Principle:
            1. Pass an invalid period token ("bad") into `download_data`.
            2. Assert `ValueError` is raised with descriptive message containing "period".
        """
        with pytest.raises(ValueError, match="period"):
            download_data(TICKER, period="bad", interval="1d")

    def test_unknown_interval(self):
        """
        Goal:
            Verify unsupported interval strings trigger ValueError prior to issuing network requests.

        Execution Principle:
            1. Pass an invalid interval token ("bad") into `download_data`.
            2. Assert `ValueError` is raised with message mentioning "interval".
        """
        with pytest.raises(ValueError, match="interval"):
            download_data(TICKER, period="1mo", interval="bad")

    def test_1m_exceeds_7_day_cap(self):
        """
        Goal:
            Verify requested 1m interval exceeding Yahoo Finance's 7-day lookback cap is rejected immediately.

        Execution Principle:
            1. Request 1m interval with 1mo period (> 7 days).
            2. Assert `ValueError` is raised citing the 1m interval limitation.
        """
        with pytest.raises(ValueError, match="1m"):
            download_data(TICKER, period="1mo", interval="1m")

    def test_2m_exceeds_60_day_cap(self):
        """
        Goal:
            Verify requested 2m interval exceeding Yahoo Finance's 60-day lookback cap is rejected immediately.

        Execution Principle:
            1. Request 2m interval with 3mo period (> 60 days).
            2. Assert `ValueError` is raised citing the 2m interval cap.
        """
        with pytest.raises(ValueError, match="2m"):
            download_data(TICKER, period="3mo", interval="2m")

    def test_5m_exceeds_60_day_cap(self):
        """
        Goal:
            Verify requested 5m interval exceeding the 60-day lookback cap is rejected immediately.

        Execution Principle:
            1. Request 5m interval with 3mo period (> 60 days).
            2. Assert `ValueError` is raised citing the 5m interval cap.
        """
        with pytest.raises(ValueError, match="5m"):
            download_data(TICKER, period="3mo", interval="5m")

    def test_15m_exceeds_60_day_cap(self):
        """
        Goal:
            Verify requested 15m interval exceeding the 60-day lookback cap is rejected immediately.

        Execution Principle:
            1. Request 15m interval with 3mo period (> 60 days).
            2. Assert `ValueError` is raised citing the 15m interval cap.
        """
        with pytest.raises(ValueError, match="15m"):
            download_data(TICKER, period="3mo", interval="15m")

    def test_30m_exceeds_60_day_cap(self):
        """
        Goal:
            Verify requested 30m interval exceeding the 60-day lookback cap is rejected immediately.

        Execution Principle:
            1. Request 30m interval with 3mo period (> 60 days).
            2. Assert `ValueError` is raised citing the 30m interval cap.
        """
        with pytest.raises(ValueError, match="30m"):
            download_data(TICKER, period="3mo", interval="30m")

    def test_90m_exceeds_60_day_cap(self):
        """
        Goal:
            Verify requested 90m interval exceeding the 60-day lookback cap is rejected immediately.

        Execution Principle:
            1. Request 90m interval with 3mo period (> 60 days).
            2. Assert `ValueError` is raised citing the 90m interval cap.
        """
        with pytest.raises(ValueError, match="90m"):
            download_data(TICKER, period="3mo", interval="90m")

    def test_60m_exceeds_730_day_cap(self):
        """
        Goal:
            Verify requested 60m interval exceeding the 730-day (2-year) lookback cap is rejected immediately.

        Execution Principle:
            1. Request 60m interval with 5y period (> 730 days).
            2. Assert `ValueError` is raised citing the 60m interval cap.
        """
        with pytest.raises(ValueError, match="60m"):
            download_data(TICKER, period="5y", interval="60m")

    def test_1h_exceeds_730_day_cap(self):
        """
        Goal:
            Verify requested 1h interval exceeding the 730-day (2-year) lookback cap is rejected immediately.

        Execution Principle:
            1. Request 1h interval with 5y period (> 730 days).
            2. Assert `ValueError` is raised citing the 1h interval cap.
        """
        with pytest.raises(ValueError, match="1h"):
            download_data(TICKER, period="5y", interval="1h")


@pytest.mark.live
class TestIntradaySupported:
    """Valid period/interval combinations download without error."""

    def test_1m_within_cap(self):
        """
        Goal:
            Verify 1m bar downloads within the 7-day limit (5d) succeed and return OHLCV data.

        Execution Principle:
            1. Download AAPL 1m bars for 5d period.
            2. Verify OHLCV structure and minimum row threshold (>= 100 rows).
        """
        df = download_data(TICKER, period="5d", interval="1m")
        _assert_ohlcv(df, min_rows=100)

    def test_2m_within_cap(self):
        """
        Goal:
            Verify 2m bar downloads within the 60-day limit (1mo) succeed and return OHLCV data.

        Execution Principle:
            1. Download AAPL 2m bars for 1mo period.
            2. Verify OHLCV structure and minimum row threshold (>= 100 rows).
        """
        df = download_data(TICKER, period="1mo", interval="2m")
        _assert_ohlcv(df, min_rows=100)

    def test_5m_within_cap(self):
        """
        Goal:
            Verify 5m bar downloads within the 60-day limit (1mo) succeed and return OHLCV data.

        Execution Principle:
            1. Download AAPL 5m bars for 1mo period.
            2. Verify OHLCV structure and minimum row threshold (>= 100 rows).
        """
        df = download_data(TICKER, period="1mo", interval="5m")
        _assert_ohlcv(df, min_rows=100)

    def test_15m_within_cap(self):
        """
        Goal:
            Verify 15m bar downloads within the 60-day limit (1mo) succeed and return OHLCV data.

        Execution Principle:
            1. Download AAPL 15m bars for 1mo period.
            2. Verify OHLCV structure and minimum row threshold (>= 50 rows).
        """
        df = download_data(TICKER, period="1mo", interval="15m")
        _assert_ohlcv(df, min_rows=50)

    def test_30m_within_cap(self):
        """
        Goal:
            Verify 30m bar downloads within the 60-day limit (1mo) succeed and return OHLCV data.

        Execution Principle:
            1. Download AAPL 30m bars for 1mo period.
            2. Verify OHLCV structure and minimum row threshold (>= 50 rows).
        """
        df = download_data(TICKER, period="1mo", interval="30m")
        _assert_ohlcv(df, min_rows=50)

    def test_60m_within_cap(self):
        """
        Goal:
            Verify 60m bar downloads within the 730-day limit (6mo) succeed and return OHLCV data.

        Execution Principle:
            1. Download AAPL 60m bars for 6mo period.
            2. Verify OHLCV structure and minimum row threshold (>= 100 rows).
        """
        df = download_data(TICKER, period="6mo", interval="60m")
        _assert_ohlcv(df, min_rows=100)

    def test_90m_within_cap(self):
        """
        Goal:
            Verify 90m bar downloads within the 60-day limit (1mo) succeed and return OHLCV data.

        Execution Principle:
            1. Download AAPL 90m bars for 1mo period.
            2. Verify OHLCV structure and minimum row threshold (>= 20 rows).
        """
        df = download_data(TICKER, period="1mo", interval="90m")
        _assert_ohlcv(df, min_rows=20)

    def test_1h_within_cap(self):
        """
        Goal:
            Verify 1h bar downloads within the 730-day limit (1y) succeed and return OHLCV data.

        Execution Principle:
            1. Download AAPL 1h bars for 1y period.
            2. Verify OHLCV structure and minimum row threshold (>= 200 rows).
        """
        df = download_data(TICKER, period="1y", interval="1h")
        _assert_ohlcv(df, min_rows=200)


@pytest.mark.live
class TestDailyWeeklyBaseline:
    """Daily and weekly intervals remain supported without lookback restrictions."""

    def test_1d_baseline(self):
        """
        Goal:
            Verify daily bars download with standard 3mo lookback window.

        Execution Principle:
            1. Download AAPL 1d bars for 3mo.
            2. Assert OHLCV structure and >= 50 rows.
        """
        df = download_data(TICKER, period="3mo", interval="1d")
        _assert_ohlcv(df, min_rows=50)

    def test_1wk_1y(self):
        """
        Goal:
            Verify weekly bars download with standard 1y lookback window.

        Execution Principle:
            1. Download AAPL 1wk bars for 1y.
            2. Assert OHLCV structure and >= 40 rows.
        """
        df = download_data(TICKER, period="1y", interval="1wk")
        _assert_ohlcv(df, min_rows=40)

    def test_1mo_5y(self):
        """
        Goal:
            Verify monthly bars download with standard 5y lookback window.

        Execution Principle:
            1. Download AAPL 1mo bars for 5y.
            2. Assert OHLCV structure and >= 50 rows.
        """
        df = download_data(TICKER, period="5y", interval="1mo")
        _assert_ohlcv(df, min_rows=50)


@pytest.mark.live
class TestStrategyCompatibilityIntraday:
    """Strategies must not throw when fed intraday OHLCV bars."""

    @pytest.fixture(scope="class")
    @classmethod
    def data_5m(cls):
        """
        Goal:
            Fixture fetching live 5m bars for AAPL for strategy compatibility testing.

        Execution Principle:
            1. Download AAPL 5m bars for 1mo period.
            2. Drop rows with NaNs and return clean DataFrame.
        """
        df = download_data(TICKER, period="1mo", interval="5m")
        df.dropna(inplace=True)
        return df

    @pytest.fixture(scope="class")
    @classmethod
    def data_1h(cls):
        """
        Goal:
            Fixture fetching live 1h bars for AAPL for strategy compatibility testing.

        Execution Principle:
            1. Download AAPL 1h bars for 1y period.
            2. Drop rows with NaNs and return clean DataFrame.
        """
        df = download_data(TICKER, period="1y", interval="1h")
        df.dropna(inplace=True)
        return df

    def test_atr_trailing_stop_5m_no_error(self, data_5m):
        """
        Goal:
            Verify ATR Trailing Stop strategy executes on live 5m bars without error.

        Execution Principle:
            1. Run `atr_trailing_stop` on 5m market bars.
            2. Assert 'Signal' column exists and contains valid signals.
        """
        result = atr_trailing_stop(data_5m)
        assert "Signal" in result.columns
        assert set(result["Signal"].unique()).issubset({"Buy", "Sell", "Hold"})

    def test_ma_rsi_strategy_5m_no_error(self, data_5m):
        """
        Goal:
            Verify MA-RSI strategy executes on live 5m bars without error.

        Execution Principle:
            1. Run `ma_rsi_strategy` on 5m market bars.
            2. Assert 'Signal' column exists and contains valid signals.
        """
        result = ma_rsi_strategy(data_5m)
        assert "Signal" in result.columns
        assert set(result["Signal"].unique()).issubset({"Buy", "Sell", "Hold"})

    def test_atr_trailing_stop_1h_no_error(self, data_1h):
        """
        Goal:
            Verify ATR Trailing Stop strategy executes on live 1h bars without error.

        Execution Principle:
            1. Run `atr_trailing_stop` on 1h market bars.
            2. Assert 'Signal' column exists and contains valid signals.
        """
        result = atr_trailing_stop(data_1h)
        assert "Signal" in result.columns
        assert set(result["Signal"].unique()).issubset({"Buy", "Sell", "Hold"})

    def test_ma_rsi_strategy_1h_no_error(self, data_1h):
        """
        Goal:
            Verify MA-RSI strategy executes on live 1h bars without error.

        Execution Principle:
            1. Run `ma_rsi_strategy` on 1h market bars.
            2. Assert 'Signal' column exists and contains valid signals.
        """
        result = ma_rsi_strategy(data_1h)
        assert "Signal" in result.columns
        assert set(result["Signal"].unique()).issubset({"Buy", "Sell", "Hold"})

    def test_news_sentiment_signal_empty_df_no_error(self):
        """
        Goal:
            Verify news sentiment mapper handles empty DataFrame gracefully without raising error.

        Execution Principle:
            1. Pass empty DataFrame with expected columns into `news_sentiment_signal`.
            2. Assert returned DataFrame is structured with `['daily_sentiment', 'Signal']`.
        """
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
        """
        Goal:
            Verify `to_naive_s` removes timezones and yields `datetime64[s]` across all intraday frequencies.

        Execution Principle:
            1. Ingest intraday bars for specified period/interval.
            2. Normalize index timestamps via `to_naive_s`.
            3. Assert dtype is `datetime64[s]` and `dt.tz` is None.
        """
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
        """
        Goal:
            Verify ingested DataFrames across all intraday and daily intervals contain flat OHLCV columns.

        Execution Principle:
            1. Download bars for specified interval and period.
            2. Assert Open, High, Low, Close, Volume exist in columns.
            3. Assert DataFrame column index is flat (not MultiIndex).
        """
        df = download_data(TICKER, period=period, interval=interval)
        for col in self.REQUIRED:
            assert col in df.columns, f"Missing '{col}' in {interval} DataFrame"
        assert not isinstance(df.columns, pd.MultiIndex), "Columns should be a flat Index"
