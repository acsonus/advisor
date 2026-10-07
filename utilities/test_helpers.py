"""Shared helper methods used by advisor tests."""

import math
from typing import Sequence
import numpy as np
import pandas as pd

VALID_SIGNALS = {"Buy", "Sell", "Hold"}


def make_ohlcv(close_prices: Sequence[float], spread_pct: float = 0.005, volume: int = 1_000_000) -> pd.DataFrame:
    """
    Construct a synthetic but structurally realistic OHLCV DataFrame from close prices.

    Goal:
    -----
    Generate deterministic, offline test datasets that satisfy standard candlestick
    relationships (High >= Close/Open, Low <= Close/Open) without network I/O.

    Execution Principle:
    --------------------
    1. Casts close prices to a numpy float array of length `n`.
    2. Constructs open prices by shifting previous close forward (`opens[i] = closes[i-1]`).
    3. Calculates `High = Close * (1 + spread_pct)` and `Low = Close * (1 - spread_pct)`.
    4. Fills volume with a constant integer level.
    5. Indexes the DataFrame with business days starting from '2026-01-01'.

    Parameters:
    -----------
    close_prices : Sequence[float]
        Series of synthetic closing prices.
    spread_pct : float, default 0.005
        Fractional percentage used to create high/low spreads around close.
    volume : int, default 1,000,000
        Bar volume.

    Returns:
    --------
    pd.DataFrame
        OHLCV DataFrame with columns `['Open', 'High', 'Low', 'Close', 'Volume']`.
    """
    closes = np.array(close_prices, dtype=float)
    n = len(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = closes * (1 + spread_pct)
    lows = closes * (1 - spread_pct)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [volume] * n},
        index=pd.date_range("2026-01-01", periods=n, freq="B"),
    )


def v_shape(n_down: int = 40, n_up: int = 40, start: float = 150.0, step: float = 1.5) -> list[float]:
    """
    Generate synthetic price trajectory representing a sharp decline followed by a recovery.

    Goal:
    -----
    Provide deterministic price series to test bullish reversal patterns (such as
    EMA golden cross and MACD positive histogram flips).

    Execution Principle:
    --------------------
    1. Creates a descending linear sequence of length `n_down` stepping down by `step`.
    2. Appends an ascending linear sequence of length `n_up` stepping up from the trough.

    Parameters:
    -----------
    n_down : int, default 40
        Bar count for the declining leg.
    n_up : int, default 40
        Bar count for the recovery leg.
    start : float, default 150.0
        Initial starting price.
    step : float, default 1.5
        Price change per bar.

    Returns:
    --------
    list[float]
        Concatenated V-shaped price values.
    """
    down = [start - i * step for i in range(n_down)]
    up = [down[-1] + i * step for i in range(n_up)]
    return down + up


def inverted_v(n_up: int = 40, n_down: int = 40, start: float = 100.0, step: float = 1.5) -> list[float]:
    """
    Generate synthetic price trajectory representing a rally followed by a sharp drop.

    Goal:
    -----
    Provide deterministic price series to test bearish reversal patterns (such as
    EMA death cross and MACD negative histogram flips).

    Execution Principle:
    --------------------
    1. Creates an ascending linear sequence of length `n_up`.
    2. Appends a descending linear sequence of length `n_down` from the peak.

    Parameters:
    -----------
    n_up : int, default 40
        Bar count for the rally leg.
    n_down : int, default 40
        Bar count for the decline leg.
    start : float, default 100.0
        Initial starting price.
    step : float, default 1.5
        Price change per bar.

    Returns:
    --------
    list[float]
        Concatenated inverted-V price values.
    """
    up = [start + i * step for i in range(n_up)]
    down = [up[-1] - i * step for i in range(n_down)]
    return up + down


def uptrend(n: int = 50, start: float = 100.0, step: float = 2.0) -> list[float]:
    """
    Generate monotonic ascending price sequence.

    Goal:
    -----
    Test trend-continuation behavior, trailing stop ratcheting, and positive Sharpe calculation.

    Execution Principle:
    --------------------
    Generates list of `n` items where `price[i] = start + i * step`.

    Parameters:
    -----------
    n : int, default 50
        Total bar count.
    start : float, default 100.0
        Initial price.
    step : float, default 2.0
        Positive increment per bar.

    Returns:
    --------
    list[float]
        Monotonically rising price values.
    """
    return [start + i * step for i in range(n)]


def downtrend(n: int = 50, start: float = 200.0, step: float = 2.0) -> list[float]:
    """
    Generate monotonic descending price sequence.

    Goal:
    -----
    Test short entries, stop loss triggers, and negative return calculations.

    Execution Principle:
    --------------------
    Generates list of `n` items where `price[i] = start - i * step`.

    Parameters:
    -----------
    n : int, default 50
        Total bar count.
    start : float, default 200.0
        Initial price.
    step : float, default 2.0
        Negative decrement per bar.

    Returns:
    --------
    list[float]
        Monotonically falling price values.
    """
    return [start - i * step for i in range(n)]


def sine_wave(n: int = 200, amplitude: float = 20.0, center: float = 100.0, period: int = 30) -> list[float]:
    """
    Generate smooth sinusoidal price oscillation.

    Goal:
    -----
    Test oscillating oscillators (VWAP, Bollinger Band squeeze cycles, MACD) across
    repeating crests and troughs without persistent drift.

    Execution Principle:
    --------------------
    Evaluates `center + amplitude * sin(2 * pi * i / period)` for `i` from 0 to `n-1`.

    Parameters:
    -----------
    n : int, default 200
        Total bar count.
    amplitude : float, default 20.0
        Half-height of oscillation from center.
    center : float, default 100.0
        Baseline price average.
    period : int, default 30
        Bar duration of a complete sine cycle.

    Returns:
    --------
    list[float]
        Sinusoidal price values.
    """
    return [center + amplitude * math.sin(2 * math.pi * i / period) for i in range(n)]


def assert_structure(result: pd.DataFrame, required_cols: Sequence[str], min_rows: int = 1) -> None:
    """
    Assert DataFrame structural validity.

    Goal:
    -----
    Verify that an output DataFrame contains required columns and sufficient rows.

    Execution Principle:
    --------------------
    1. Asserts `isinstance(result, pd.DataFrame)`.
    2. Asserts row count `len(result) >= min_rows`.
    3. Iterates over `required_cols` asserting each exists in `result.columns`.

    Parameters:
    -----------
    result : pd.DataFrame
        DataFrame under test.
    required_cols : Sequence[str]
        Expected column names.
    min_rows : int, default 1
        Minimum row count.
    """
    assert isinstance(result, pd.DataFrame)
    assert len(result) >= min_rows
    for col in required_cols:
        assert col in result.columns, f"Missing column: '{col}'"


def assert_valid_signals(result: pd.DataFrame, signal_col: str = "Signal") -> None:
    """
    Assert that a Signal column contains exclusively valid signal strings.

    Goal:
    -----
    Ensure strategy output contains only supported signal labels and no stray strings or NaNs.

    Execution Principle:
    --------------------
    Extracts non-NaN unique values from `result[signal_col]` and asserts that the set
    difference against `VALID_SIGNALS` ({'Buy', 'Sell', 'Hold'}) is empty.

    Parameters:
    -----------
    result : pd.DataFrame
        DataFrame containing signal column.
    signal_col : str, default 'Signal'
        Name of the signal column.
    """
    bad = set(result[signal_col].dropna().unique()) - VALID_SIGNALS
    assert not bad, f"Unexpected signal values: {bad}"


def assert_ohlcv(df: pd.DataFrame, min_rows: int = 1) -> None:
    """
    Assert that a DataFrame conforms to valid financial OHLCV format.

    Goal:
    -----
    Verify contract compliance for market data fetchers.

    Execution Principle:
    --------------------
    1. Asserts `isinstance(df, pd.DataFrame)`.
    2. Asserts length is at least `min_rows`.
    3. Checks for presence of 'Open', 'High', 'Low', 'Close', 'Volume' columns.
    4. Asserts that 'Close' column is not entirely NaN.

    Parameters:
    -----------
    df : pd.DataFrame
        Candidate OHLCV DataFrame.
    min_rows : int, default 1
        Minimum required row count.
    """
    assert isinstance(df, pd.DataFrame), "Result must be a DataFrame"
    assert len(df) >= min_rows, f"Expected at least {min_rows} row(s), got {len(df)}"
    for col in ("Open", "High", "Low", "Close", "Volume"):
        assert col in df.columns, f"Missing column '{col}'"
    assert df["Close"].notna().any(), "Close column is entirely NaN"
