"""Market data fetcher using Yahoo Finance."""

import pandas as pd
import yfinance as yf

from advisor.data.validator import validate_period_and_interval


def download_data(
    ticker_symbol: str = 'AAPL',
    period: str = '1mo',
    interval: str = '1d',
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker_symbol : str
        Ticker symbol, e.g. 'AAPL', 'BRK-B', '^GSPC'.
    period : str
        Lookback window. One of: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max.
    interval : str
        Bar size. One of: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo.
        Note: intraday intervals have Yahoo Finance limits on how far back
        data is available (e.g. 1m → max 7 days, 1h → max 730 days).
    """
    validate_period_and_interval(period, interval)

    data = yf.download(ticker_symbol, period=period, interval=interval, auto_adjust=True)
    if data is None:
        raise ValueError(f"No data returned for ticker '{ticker_symbol}'")

    # Flatten MultiIndex columns produced when a single ticker is downloaded.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    return data


# Backward-compatible alias
downloadData = download_data
