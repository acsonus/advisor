"""Datetime normalization utilities."""

import pandas as pd


def to_naive_s(series: pd.Series) -> pd.Series:
    """
    Normalise a datetime Series to timezone-naive datetime64[s] precision.

    Goal:
    -----
    Standardize timestamps across daily, weekly, and intraday Yahoo Finance data
    and news publication dates into a uniform, timezone-naive format suitable for
    merging (e.g., via `pd.merge_asof`).

    Execution Principle:
    --------------------
    1. Parse the input series with `pd.to_datetime` to ensure consistent datetime dtype.
    2. Check if the series is timezone-aware (`dt.tz is not None`).
       - If timezone-aware (typical for intraday Yahoo data in UTC): convert explicitly
         to UTC (`tz_convert("UTC")`) and then strip the timezone offset (`tz_localize(None)`).
         Directly casting a timezone-aware Series to `datetime64[s]` raises a `TypeError`
         in pandas, making explicit localization stripping mandatory.
       - If already timezone-naive (typical for daily/weekly bars): retain as-is.
    3. Truncate resolution to second precision (`datetime64[s]`) to avoid nanosecond
       precision discrepancies during time-series merges.

    Parameters:
    -----------
    series : pd.Series
        Series containing timestamps (strings, datetimes, or Timestamp objects).

    Returns:
    --------
    pd.Series
        Timezone-naive datetime series with `datetime64[s]` dtype.
    """
    dt = pd.to_datetime(series)
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt.astype("datetime64[s]")
