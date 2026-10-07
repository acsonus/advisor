"""Live Yahoo Finance market data contract and strategy tests."""

import pandas as pd
import pytest

from utilities.test_helpers import (
    assert_structure as _assert_structure,
    assert_valid_signals as _assert_valid_signals,
)
from advisor.backtesting import backtest_strategy
from advisor.data import download_data
from advisor.strategies import (
    atr_trailing_stop,
    calculate_vwap,
    macd_histogram_reversal_strategy,
    ma_rsi_strategy,
)

_LIVE_TICKERS = ["COKE", "BA", "MSFT", "GOOGL", "WMT"]
_LIVE_MIN_BARS = 15


@pytest.mark.live
class TestRealMarketData:
    """Live network tests downloading real OHLCV data from Yahoo Finance."""

    @staticmethod
    def _fetch(ticker: str) -> pd.DataFrame:
        try:
            df = download_data(ticker, period="1mo", interval="1d")
            df.dropna(inplace=True)
        except Exception as exc:
            pytest.skip(f"Could not download data for '{ticker}': {exc}")

        if df.empty or len(df) < _LIVE_MIN_BARS:
            pytest.skip(f"Insufficient data for '{ticker}': only {len(df)} bar(s) returned.")
        return df

    # OHLCV data quality

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_columns_and_minimum_rows(self, ticker):
        df = self._fetch(ticker)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in df.columns, f"{ticker}: missing column '{col}'"
        assert len(df) >= _LIVE_MIN_BARS

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_no_negative_or_zero_prices(self, ticker):
        df = self._fetch(ticker)
        for col in ("Open", "High", "Low", "Close"):
            assert (df[col] > 0).all(), f"{ticker}: '{col}' contains non-positive values"

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_high_greater_equal_low(self, ticker):
        df = self._fetch(ticker)
        assert (df["High"] >= df["Low"]).all()

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_volume_positive(self, ticker):
        df = self._fetch(ticker)
        assert (df["Volume"] > 0).all()

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_close_between_high_and_low(self, ticker):
        df = self._fetch(ticker)
        assert (df["Close"] >= df["Low"]).all() and (df["Close"] <= df["High"]).all()

    # MA-RSI
    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ma_rsi_live_structure(self, ticker):
        df = self._fetch(ticker)
        result = ma_rsi_strategy(df)
        _assert_structure(result, ["Close", "Short_EMA", "Long_EMA", "RSI", "Signal"])
        _assert_valid_signals(result)

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ma_rsi_rsi_in_range_live(self, ticker):
        df = self._fetch(ticker)
        rsi = ma_rsi_strategy(df)["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    # ATR
    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_atr_live_structure(self, ticker):
        df = self._fetch(ticker)
        result = atr_trailing_stop(df)
        _assert_structure(result, ["Close", "ATR", "Buy_Stop", "Sell_Stop", "Signal"])
        _assert_valid_signals(result)

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_atr_positive_live(self, ticker):
        df = self._fetch(ticker)
        atr = atr_trailing_stop(df)["ATR"].dropna()
        assert (atr > 0).all()

    # MACD
    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_macd_live_structure(self, ticker):
        df = self._fetch(ticker)
        result = macd_histogram_reversal_strategy(df)
        _assert_structure(result, ["Close", "MACD", "Signal_Line", "MACD_Histogram", "Signal"])
        _assert_valid_signals(result)

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_macd_histogram_identity_live(self, ticker):
        df = self._fetch(ticker)
        result = macd_histogram_reversal_strategy(df)
        diff = (result["MACD_Histogram"] - (result["MACD"] - result["Signal_Line"])).abs()
        assert (diff.dropna() < 1e-10).all()

    # VWAP
    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_vwap_live_structure(self, ticker):
        df = self._fetch(ticker)
        result = calculate_vwap(df)
        _assert_structure(result, ["Close", "Volume", "VWAP", "Signal"])
        _assert_valid_signals(result)

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_vwap_positive_live(self, ticker):
        df = self._fetch(ticker)
        vwap = calculate_vwap(df)["VWAP"]
        assert (vwap > 0).all()

    # Full backtest pipeline on live data
    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_backtest_atr_live(self, ticker):
        df = self._fetch(ticker)
        result = atr_trailing_stop(df)
        metrics = backtest_strategy(result[["Close", "Signal"]])
        for key in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "n_trades"):
            assert key in metrics
        assert metrics["max_drawdown_pct"] <= 0.0
        assert 0.0 <= metrics["win_rate_pct"] <= 100.0

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_backtest_ma_rsi_live(self, ticker):
        df = self._fetch(ticker)
        result = ma_rsi_strategy(df)
        metrics = backtest_strategy(result[["Close", "Signal"]])
        assert metrics["max_drawdown_pct"] <= 0.0
        assert 0.0 <= metrics["win_rate_pct"] <= 100.0
        assert len(metrics["trade_returns"]) == metrics["n_trades"]

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_backtest_macd_live(self, ticker):
        df = self._fetch(ticker)
        result = macd_histogram_reversal_strategy(df)
        metrics = backtest_strategy(result[["Close", "Signal"]])
        assert metrics["max_drawdown_pct"] <= 0.0
        assert 0.0 <= metrics["win_rate_pct"] <= 100.0

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_all_strategies_valid_signals_live(self, ticker):
        df = self._fetch(ticker)
        strategies = [
            (ma_rsi_strategy, "MA-RSI"),
            (atr_trailing_stop, "ATR"),
            (macd_histogram_reversal_strategy, "MACD"),
            (calculate_vwap, "VWAP"),
        ]
        for strat, name in strategies:
            res = strat(df)
            _assert_valid_signals(res)
