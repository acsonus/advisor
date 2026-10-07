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
    Download and prepare historical OHLCV market data from Yahoo Finance.

    Goal:
    -----
    Retrieve split- and dividend-adjusted historical price bars (Open, High, Low, Close,
    Volume) for a given symbol and bar granularity, returning a clean, flat single-index
    DataFrame ready for technical indicator calculation and backtesting.

    Execution Principle:
    --------------------
    1. Pre-validation: Calls `validate_period_and_interval(period, interval)` to enforce
       valid syntax and verify that intraday lookback requests respect Yahoo's limits.
    2. Data Retrieval: Invokes `yf.download()` with `auto_adjust=True`, which adjusts
       Open, High, Low, and Close for corporate actions (splits and dividends).
    3. Null & Empty Check: Verifies that returned data is not `None` or missing.
    4. Index Flattening: In modern `yfinance` versions, single-ticker queries return
       a 2-level MultiIndex on columns (e.g., `('Close', 'AAPL')`). If detected,
       flattens columns by dropping the second level (`data.columns.droplevel(1)`),
       yielding standard column names `['Open', 'High', 'Low', 'Close', 'Volume']`.
    5. Returns the prepared DataFrame with datetime index.

    Parameters:
    -----------
    ticker_symbol : str, default 'AAPL'
        Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'BRK-B', '^GSPC').
    period : str, default '1mo'
        Historical duration window. Supported: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max.
    interval : str, default '1d'
        Bar resolution. Supported: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo.

    Returns:
    --------
    pd.DataFrame
        Adjusted OHLCV DataFrame indexed by timestamp.

    Raises:
    -------
    ValueError
        If the combination of period/interval is invalid or no data is returned.
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
