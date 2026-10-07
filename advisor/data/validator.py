"""Input validators for tickers, periods, and intervals."""

import re
from advisor.data.constants import (
    INTERVAL_MAX_DAYS,
    PERIOD_DAYS,
    VALID_INTERVALS,
    VALID_PERIODS,
)


def validate_period_and_interval(period: str, interval: str) -> None:
    """Validate Yahoo Finance period and interval combinations."""
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
    """Validate and clean stock ticker symbol."""
    cleaned = ticker.upper().strip()
    if not re.match(r'^[A-Z0-9.\-\^=]{1,10}$', cleaned):
        raise ValueError(f"Invalid ticker '{ticker}'.")
    return cleaned
