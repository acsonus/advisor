"""Market data ingestion, normalization, and validation."""

from advisor.data.constants import (
    INTERVAL_MAX_DAYS,
    PERIOD_DAYS,
    VALID_INTERVALS,
    VALID_PERIODS,
)
from advisor.data.fetcher import download_data, downloadData
from advisor.data.normalizers import to_naive_s
from advisor.data.validator import validate_period_and_interval, validate_ticker

# Alias constants matching private names in trading_strategy.py for 100% backward compatibility
_INTERVAL_MAX_DAYS = INTERVAL_MAX_DAYS
_PERIOD_DAYS = PERIOD_DAYS

__all__ = [
    "VALID_PERIODS",
    "VALID_INTERVALS",
    "INTERVAL_MAX_DAYS",
    "PERIOD_DAYS",
    "_INTERVAL_MAX_DAYS",
    "_PERIOD_DAYS",
    "to_naive_s",
    "validate_period_and_interval",
    "validate_ticker",
    "download_data",
    "downloadData",
]
