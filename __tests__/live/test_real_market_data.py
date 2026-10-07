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
        """
        Goal:
            Safely fetch real market data for a given ticker, skipping gracefully if network or data is unavailable.

        Execution Principle:
            1. Dispatch `download_data` for 1 month of daily bars.
            2. Drop incomplete NaN rows.
            3. Catch network/API errors and trigger `pytest.skip`.
            4. Verify row count meets `_LIVE_MIN_BARS` threshold or skip.
        """
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
        """
        Goal:
            Verify real market data contains all standard OHLCV columns and meets minimum bar count.

        Execution Principle:
            1. Fetch live market bars for ticker.
            2. Assert Open, High, Low, Close, Volume exist in columns.
            3. Verify bar count >= _LIVE_MIN_BARS.
        """
        df = self._fetch(ticker)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in df.columns, f"{ticker}: missing column '{col}'"
        assert len(df) >= _LIVE_MIN_BARS

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_no_negative_or_zero_prices(self, ticker):
        """
        Goal:
            Verify real price data contains strictly positive numbers (> 0).

        Execution Principle:
            1. Fetch live data for ticker.
            2. Assert all price columns (Open, High, Low, Close) are strictly greater than zero.
        """
        df = self._fetch(ticker)
        for col in ("Open", "High", "Low", "Close"):
            assert (df[col] > 0).all(), f"{ticker}: '{col}' contains non-positive values"

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_high_greater_equal_low(self, ticker):
        """
        Goal:
            Verify High price is greater than or equal to Low price for all bars in real feeds.

        Execution Principle:
            1. Fetch live market data.
            2. Assert `df['High'] >= df['Low']` across all rows.
        """
        df = self._fetch(ticker)
        assert (df["High"] >= df["Low"]).all()

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_volume_positive(self, ticker):
        """
        Goal:
            Verify trading volume is strictly positive for active trading sessions.

        Execution Principle:
            1. Ingest live daily bars.
            2. Assert `df['Volume'] > 0` across all rows.
        """
        df = self._fetch(ticker)
        assert (df["Volume"] > 0).all()

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ohlcv_close_between_high_and_low(self, ticker):
        """
        Goal:
            Verify Close price lies strictly between Low and High price extremes.

        Execution Principle:
            1. Fetch live market bars.
            2. Assert Close >= Low and Close <= High for every bar.
        """
        df = self._fetch(ticker)
        assert (df["Close"] >= df["Low"]).all() and (df["Close"] <= df["High"]).all()

    # MA-RSI
    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ma_rsi_live_structure(self, ticker):
        """
        Goal:
            Verify MA-RSI strategy produces valid structural columns and signals on live market data.

        Execution Principle:
            1. Fetch live data and evaluate `ma_rsi_strategy`.
            2. Verify output column structure and signal validity.
        """
        df = self._fetch(ticker)
        result = ma_rsi_strategy(df)
        _assert_structure(result, ["Close", "Short_EMA", "Long_EMA", "RSI", "Signal"])
        _assert_valid_signals(result)

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_ma_rsi_rsi_in_range_live(self, ticker):
        """
        Goal:
            Verify live RSI calculation stays within [0, 100] bounds.

        Execution Principle:
            1. Calculate MA-RSI on live data.
            2. Drop warmup NaNs and assert 0 <= RSI <= 100.
        """
        df = self._fetch(ticker)
        rsi = ma_rsi_strategy(df)["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    # ATR
    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_atr_live_structure(self, ticker):
        """
        Goal:
            Verify ATR Trailing Stop strategy produces valid column structure and valid signals on live market data.

        Execution Principle:
            1. Fetch live data and compute `atr_trailing_stop`.
            2. Verify output column structure and signal validity.
        """
        df = self._fetch(ticker)
        result = atr_trailing_stop(df)
        _assert_structure(result, ["Close", "ATR", "Buy_Stop", "Sell_Stop", "Signal"])
        _assert_valid_signals(result)

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_atr_positive_live(self, ticker):
        """
        Goal:
            Verify live ATR volatility calculation is strictly positive.

        Execution Principle:
            1. Compute ATR on live data.
            2. Assert all non-NaN values are strictly > 0.
        """
        df = self._fetch(ticker)
        atr = atr_trailing_stop(df)["ATR"].dropna()
        assert (atr > 0).all()

    # MACD
    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_macd_live_structure(self, ticker):
        """
        Goal:
            Verify MACD Histogram Reversal strategy produces expected columns and valid signals on live market data.

        Execution Principle:
            1. Fetch live data and run `macd_histogram_reversal_strategy`.
            2. Verify output columns and valid signal values.
        """
        df = self._fetch(ticker)
        result = macd_histogram_reversal_strategy(df)
        _assert_structure(result, ["Close", "MACD", "Signal_Line", "MACD_Histogram", "Signal"])
        _assert_valid_signals(result)

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_macd_histogram_identity_live(self, ticker):
        """
        Goal:
            Verify MACD histogram identity holds exactly on real market data (`Histogram = MACD - Signal_Line`).

        Execution Principle:
            1. Evaluate MACD on live data.
            2. Assert absolute difference between histogram and (MACD - Signal_Line) is < 1e-10.
        """
        df = self._fetch(ticker)
        result = macd_histogram_reversal_strategy(df)
        diff = (result["MACD_Histogram"] - (result["MACD"] - result["Signal_Line"])).abs()
        assert (diff.dropna() < 1e-10).all()

    # VWAP
    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_vwap_live_structure(self, ticker):
        """
        Goal:
            Verify VWAP strategy produces expected columns and valid signals on live market data.

        Execution Principle:
            1. Run `calculate_vwap` on live data.
            2. Verify columns and valid signal values.
        """
        df = self._fetch(ticker)
        result = calculate_vwap(df)
        _assert_structure(result, ["Close", "Volume", "VWAP", "Signal"])
        _assert_valid_signals(result)

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_vwap_positive_live(self, ticker):
        """
        Goal:
            Verify live VWAP values are strictly positive.

        Execution Principle:
            1. Run VWAP on live data.
            2. Assert all VWAP entries are > 0.
        """
        df = self._fetch(ticker)
        vwap = calculate_vwap(df)["VWAP"]
        assert (vwap > 0).all()

    # Full backtest pipeline on live data
    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_backtest_atr_live(self, ticker):
        """
        Goal:
            Verify full backtest execution on live ATR strategy signals generates realistic bounded performance metrics.

        Execution Principle:
            1. Run ATR on live data.
            2. Backtest generated signals.
            3. Confirm metric keys and bound constraints (max_drawdown <= 0%, win_rate in [0, 100%]).
        """
        df = self._fetch(ticker)
        result = atr_trailing_stop(df)
        metrics = backtest_strategy(result[["Close", "Signal"]])
        for key in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "n_trades"):
            assert key in metrics
        assert metrics["max_drawdown_pct"] <= 0.0
        assert 0.0 <= metrics["win_rate_pct"] <= 100.0

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_backtest_ma_rsi_live(self, ticker):
        """
        Goal:
            Verify full backtest execution on live MA-RSI signals produces valid drawdown and win rate metrics.

        Execution Principle:
            1. Run MA-RSI on live data.
            2. Run backtest engine.
            3. Assert metrics constraints and trade array length consistency.
        """
        df = self._fetch(ticker)
        result = ma_rsi_strategy(df)
        metrics = backtest_strategy(result[["Close", "Signal"]])
        assert metrics["max_drawdown_pct"] <= 0.0
        assert 0.0 <= metrics["win_rate_pct"] <= 100.0
        assert len(metrics["trade_returns"]) == metrics["n_trades"]

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_backtest_macd_live(self, ticker):
        """
        Goal:
            Verify full backtest execution on live MACD reversal signals produces valid drawdown and win rate metrics.

        Execution Principle:
            1. Run MACD on live data.
            2. Run backtest engine.
            3. Assert metrics constraints.
        """
        df = self._fetch(ticker)
        result = macd_histogram_reversal_strategy(df)
        metrics = backtest_strategy(result[["Close", "Signal"]])
        assert metrics["max_drawdown_pct"] <= 0.0
        assert 0.0 <= metrics["win_rate_pct"] <= 100.0

    @pytest.mark.parametrize("ticker", _LIVE_TICKERS)
    def test_all_strategies_valid_signals_live(self, ticker):
        """
        Goal:
            Verify all four strategies emit strictly valid signals ('Buy', 'Sell', 'Hold') on real market data.

        Execution Principle:
            1. Loop through all 4 strategies (MA-RSI, ATR, MACD, VWAP).
            2. Execute each over live ticker data.
            3. Assert signal validity for each strategy.
        """
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
