"""Shared helper methods used by advisor tests."""

import math

import numpy as np
import pandas as pd

VALID_SIGNALS = {"Buy", "Sell", "Hold"}


def make_ohlcv(close_prices, spread_pct: float = 0.005, volume: int = 1_000_000) -> pd.DataFrame:
    """Build a minimal but realistic OHLCV DataFrame from close prices."""
    closes = np.array(close_prices, dtype=float)
    n = len(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = closes * (1 + spread_pct)
    lows = closes * (1 - spread_pct)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [volume] * n},
        index=pd.date_range("2026-01-01", periods=n, freq="B"),
    )


def v_shape(n_down: int = 40, n_up: int = 40, start: float = 150.0, step: float = 1.5):
    """Prices fall then recover."""
    down = [start - i * step for i in range(n_down)]
    up = [down[-1] + i * step for i in range(n_up)]
    return down + up


def inverted_v(n_up: int = 40, n_down: int = 40, start: float = 100.0, step: float = 1.5):
    """Prices rise then fall."""
    up = [start + i * step for i in range(n_up)]
    down = [up[-1] - i * step for i in range(n_down)]
    return up + down


def uptrend(n: int = 50, start: float = 100.0, step: float = 2.0):
    return [start + i * step for i in range(n)]


def downtrend(n: int = 50, start: float = 200.0, step: float = 2.0):
    return [start - i * step for i in range(n)]


def sine_wave(n: int = 200, amplitude: float = 20.0, center: float = 100.0, period: int = 30):
    """Smooth sinusoidal oscillation."""
    return [center + amplitude * math.sin(2 * math.pi * i / period) for i in range(n)]


def assert_structure(result: pd.DataFrame, required_cols, min_rows: int = 1):
    assert isinstance(result, pd.DataFrame)
    assert len(result) >= min_rows
    for col in required_cols:
        assert col in result.columns, f"Missing column: '{col}'"


def assert_valid_signals(result: pd.DataFrame, signal_col: str = "Signal"):
    bad = set(result[signal_col].dropna().unique()) - VALID_SIGNALS
    assert not bad, f"Unexpected signal values: {bad}"


def assert_ohlcv(df: pd.DataFrame, min_rows: int = 1) -> None:
    """Assert the DataFrame looks like valid OHLCV data."""
    assert isinstance(df, pd.DataFrame), "Result must be a DataFrame"
    assert len(df) >= min_rows, f"Expected at least {min_rows} row(s), got {len(df)}"
    for col in ("Open", "High", "Low", "Close", "Volume"):
        assert col in df.columns, f"Missing column '{col}'"
    assert df["Close"].notna().any(), "Close column is entirely NaN"
