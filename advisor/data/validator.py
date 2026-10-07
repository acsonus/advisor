"""Input validators for tickers, periods, and intervals."""

import re
from advisor.data.constants import (
    INTERVAL_MAX_DAYS,
    PERIOD_DAYS,
    VALID_INTERVALS,
    VALID_PERIODS,
)


def validate_period_and_interval(period: str, interval: str) -> None:
    """
    Validate Yahoo Finance period and interval combinations against API constraints.

    Goal:
    -----
    Fail fast before issuing network calls by validating that the requested period and
    interval exist within Yahoo Finance supported sets, and that the lookback range
    does not violate Yahoo Finance hard retention caps on intraday data.

    Execution Principle:
    --------------------
    1. Check membership of `period` against `VALID_PERIODS`. If missing, raise ValueError
       listing valid supported choices.
    2. Check membership of `interval` against `VALID_INTERVALS`. If missing, raise ValueError.
    3. Check intraday data caps: Yahoo limits historical depth for granular intervals
       (e.g., 1m is capped at 7 days, 2m-30m at 60 days, 1h at 730 days). If `interval`
       has an enforced cap in `INTERVAL_MAX_DAYS`, compare the approximate duration
       (`PERIOD_DAYS[period]`) with `max_days`. If duration exceeds the cap, raise ValueError
       with explanatory diagnostic message.

    Parameters:
    -----------
    period : str
        Lookback timeframe token (e.g., '1mo', '1y').
    interval : str
        Bar resolution token (e.g., '1m', '1h', '1d').

    Raises:
    -------
    ValueError
        If the period or interval is invalid or exceeds intraday retention limits.
    """
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Choose from: {sorted(VALID_PERIODS)}")
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval '{interval}'. Choose from: {sorted(VALID_INTERVALS)}")
    if interval in INTERVAL_MAX_DAYS:
        max_days = INTERVAL_MAX_DAYS[interval]
        if PERIOD_DAYS.get(period, 0) > max_days:
            raise ValueError(
                f"Interval '{interval}' supports at most {max_days} days of history, "
                f"but period '{period}' requests ~{PERIOD_DAYS[period]} days."
            )


def validate_ticker(ticker: str) -> str:
    """
    Validate and sanitize stock ticker symbols.

    Goal:
    -----
    Ensure user-supplied ticker input represents a valid financial symbol format
    and prevent invalid characters, injection attempts, or malformed queries before
    calling external financial APIs.

    Execution Principle:
    --------------------
    1. Strip leading and trailing whitespace and convert string to uppercase.
    2. Evaluate sanitized string against regex `^[A-Z0-9.\\-\\^=]{1,10}$`, allowing
       alphanumeric characters, exchange dots (BRK.B), hyphens (BRK-B), indices (^GSPC),
       and futures/currencies (=X).
    3. If valid, return cleaned string; otherwise raise a `ValueError`.

    Parameters:
    -----------
    ticker : str
        Raw ticker symbol string.

    Returns:
    --------
    str
        Sanitized, uppercase ticker symbol.

    Raises:
    -------
    ValueError
        If ticker format does not conform to allowed financial symbol patterns.
    """
    cleaned = ticker.upper().strip()
    if not re.match(r'^[A-Z0-9.\-\^=]{1,10}$', cleaned):
        raise ValueError(f"Invalid ticker '{ticker}'.")
    return cleaned
