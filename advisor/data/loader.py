"""Historical market data loader for MetaTrader and platform CSV dumps."""

import csv
import io
import re
from pathlib import Path
from typing import Any
import pandas as pd


def _detect_delimiter(sample_text: str) -> str:
    """Auto-detect column delimiter (tab, comma, semicolon)."""
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=["\t", ",", ";"])
        return dialect.delimiter
    except Exception:
        if "\t" in sample_text:
            return "\t"
        if ";" in sample_text:
            return ";"
        return ","


def _is_header_row(tokens: list[str]) -> bool:
    """Check if row contains column names rather than data values."""
    header_keywords = {"date", "time", "open", "high", "low", "close", "vol", "tickvol"}
    cleaned = [re.sub(r"[<>\s]", "", t).lower() for t in tokens if t]
    return any(k in cleaned for k in header_keywords)


def load_metatrader_dump(filepath: str | Path) -> pd.DataFrame:
    """
    Load and parse a MetaTrader 4 / MetaTrader 5 historical bar export file.

    Supports:
    ---------
    - MT5 standard tab/comma export with `<DATE>`, `<TIME>`, `<OPEN>`, `<HIGH>`, `<LOW>`, `<CLOSE>`, `<TICKVOL>`
    - MT4 standard CSV exports with `Date,Time,Open,High,Low,Close,Volume`
    - Headerless MT4 raw exports (7 columns: Date, Time, Open, High, Low, Close, Volume)
    - Daily MT4/MT5 files without Time column
    - Standard MetaTrader `YYYY.MM.DD` date formats

    Parameters:
    -----------
    filepath : str or Path
        Path to the exported CSV or TXT file.

    Returns:
    --------
    pd.DataFrame
        Standardized OHLCV DataFrame indexed by datetime `Date`.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Historical data file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head_sample = "".join([f.readline() for _ in range(5)])

    if not head_sample.strip():
        raise ValueError(f"File is empty: {path}")

    delimiter = _detect_delimiter(head_sample)
    first_line_tokens = [t.strip() for t in head_sample.splitlines()[0].split(delimiter)]
    has_header = _is_header_row(first_line_tokens)

    if has_header:
        df = pd.read_csv(
            path,
            delimiter=delimiter,
            skipinitialspace=True,
            dtype=str,
        )
        # Normalize column names: strip '<', '>', whitespace, uppercase
        clean_col_map = {}
        for col in df.columns:
            clean_name = re.sub(r"[<>\s]", "", str(col)).upper()
            clean_col_map[col] = clean_name
        df.rename(columns=clean_col_map, inplace=True)
    else:
        # Headerless MT4 raw format
        first_row_len = len(first_line_tokens)
        if first_row_len >= 7:
            names = ["DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"] + [
                f"EXTRA_{i}" for i in range(first_row_len - 7)
            ]
        elif first_row_len == 6:
            names = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        else:
            names = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]

        df = pd.read_csv(
            path,
            delimiter=delimiter,
            names=names,
            skipinitialspace=True,
            dtype=str,
        )

    # Resolve Date/Time columns
    date_col = next((c for c in ["DATE", "DATETIME", "TIME_STAMP"] if c in df.columns), None)
    time_col = next((c for c in ["TIME", "HOUR"] if c in df.columns and c != date_col), None)

    if not date_col:
        raise ValueError(
            f"Could not identify Date column in MetaTrader dump. Columns found: {list(df.columns)}"
        )

    if time_col and time_col in df.columns:
        dt_strings = df[date_col].str.replace(".", "-", regex=False) + " " + df[time_col]
    else:
        dt_strings = df[date_col].str.replace(".", "-", regex=False)

    df["Date"] = pd.to_datetime(dt_strings, errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # Normalize tz
    if getattr(df["Date"].dtype, "tz", None) is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)

    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # Map OHLCV columns
    std_map = {}
    for col in df.columns:
        c_upper = col.upper()
        if c_upper == "OPEN":
            std_map[col] = "Open"
        elif c_upper == "HIGH":
            std_map[col] = "High"
        elif c_upper == "LOW":
            std_map[col] = "Low"
        elif c_upper in ("CLOSE", "LAST"):
            std_map[col] = "Close"
        elif c_upper in ("TICKVOL", "VOL", "VOLUME", "TOTAL_VOLUME"):
            if "Volume" not in std_map.values():
                std_map[col] = "Volume"

    df.rename(columns=std_map, inplace=True)

    if "Close" not in df.columns:
        raise ValueError(
            f"Could not identify Close price column in MetaTrader export. Columns: {list(df.columns)}"
        )

    # Convert numeric price columns
    for num_col in ["Open", "High", "Low", "Close", "Volume"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col].str.replace(",", "."), errors="coerce")

    # If Open, High, Low are missing, fall back to Close
    if "Open" not in df.columns:
        df["Open"] = df["Close"]
    if "High" not in df.columns:
        df["High"] = df["Close"]
    if "Low" not in df.columns:
        df["Low"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 1.0

    df.dropna(subset=["Close"], inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_historical_dump(filepath: str | Path, source: str = "auto") -> pd.DataFrame:
    """
    Load historical market data dump from trading platforms.

    Supports:
    ---------
    - source="metatrader": MetaTrader 4 / MetaTrader 5 exports
    - source="auto": auto-detects MetaTrader or generic OHLCV CSV formats
    """
    path = Path(filepath)
    if source.lower() == "metatrader" or source.lower() == "auto":
        try:
            return load_metatrader_dump(path)
        except Exception:
            if source.lower() == "metatrader":
                raise

    # Generic fallback loader
    df = pd.read_csv(path)
    # Match time/date
    date_col = next((c for c in df.columns if c.lower() in ("date", "time", "timestamp", "datetime")), None)
    if date_col:
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        df.set_index("Date", inplace=True)

    # Standardize column names
    col_map = {
        c: c.capitalize()
        for c in df.columns
        if c.lower() in ("open", "high", "low", "close", "volume")
    }
    df.rename(columns=col_map, inplace=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df.dropna(subset=["Close"], inplace=True)
    df.sort_index(inplace=True)
    return df
