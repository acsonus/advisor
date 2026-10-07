"""Datetime normalization utilities."""

import pandas as pd


def to_naive_s(series: pd.Series) -> pd.Series:
    """
    Normalise a datetime Series to tz-naive datetime64[s].

    yfinance returns tz-aware (UTC) timestamps for intraday intervals and
    tz-naive timestamps for daily/weekly intervals. Calling .astype('datetime64[s]')
    directly on a tz-aware Series raises TypeError, so we strip the timezone first.
    """
    dt = pd.to_datetime(series)
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt.astype("datetime64[s]")
