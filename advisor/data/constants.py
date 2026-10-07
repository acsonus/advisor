"""Constants for market data periods and intervals."""

# Yahoo Finance supported values
VALID_PERIODS = {'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'}
VALID_INTERVALS = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'}

# Maximum lookback in calendar days allowed per intraday interval by Yahoo Finance.
# Daily and weekly intervals have no enforced cap.
INTERVAL_MAX_DAYS: dict[str, int] = {
    '1m': 7,
    '2m': 60,
    '5m': 60,
    '15m': 60,
    '30m': 60,
    '60m': 730,
    '90m': 60,
    '1h': 730,
}

# Approximate number of calendar days each period covers (used for validation).
PERIOD_DAYS: dict[str, int] = {
    '1d': 1,
    '5d': 5,
    '1mo': 30,
    '3mo': 90,
    '6mo': 180,
    '1y': 365,
    '2y': 730,
    '5y': 1825,
    '10y': 3650,
    'ytd': 366,
    'max': 36500,
}
