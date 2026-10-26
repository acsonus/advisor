"""Unit tests for MetaTrader and platform historical data dumps loader."""

import pytest
import pandas as pd
from advisor.data.loader import load_historical_dump, load_metatrader_dump


def test_load_mt5_tab_delimited(tmp_path):
    """Test loading standard MT5 tab-delimited export with <TAGS>."""
    file_content = (
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2024.01.02\t00:00:00\t1.10425\t1.10440\t1.10390\t1.10410\t142\t0\t1\n"
        "2024.01.02\t01:00:00\t1.10410\t1.10480\t1.10350\t1.10450\t210\t0\t1\n"
        "2024.01.02\t02:00:00\t1.10450\t1.10510\t1.10420\t1.10490\t185\t0\t1\n"
    )
    dump_path = tmp_path / "EURUSD_H1.txt"
    dump_path.write_text(file_content, encoding="utf-8")

    df = load_metatrader_dump(dump_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert str(df.index[0]) == "2024-01-02 00:00:00"
    assert df["Close"].iloc[0] == 1.10410
    assert df["Volume"].iloc[0] == 142.0


def test_load_mt4_comma_delimited(tmp_path):
    """Test loading MT4 standard comma-separated export."""
    file_content = (
        "Date,Time,Open,High,Low,Close,Volume\n"
        "2024.01.02,10:00,100.50,101.20,100.10,101.00,500\n"
        "2024.01.02,11:00,101.00,101.80,100.90,101.60,650\n"
    )
    dump_path = tmp_path / "AAPL_H1.csv"
    dump_path.write_text(file_content, encoding="utf-8")

    df = load_historical_dump(dump_path, source="metatrader")

    assert len(df) == 2
    assert str(df.index[0]) == "2024-01-02 10:00:00"
    assert df["Close"].iloc[0] == 101.00
    assert df["High"].iloc[0] == 101.20


def test_load_mt4_headerless_format(tmp_path):
    """Test loading MT4 raw export without headers (7 columns)."""
    file_content = (
        "2024.01.02,09:30,150.20,151.00,149.80,150.90,1200\n"
        "2024.01.02,09:31,150.90,151.30,150.70,151.20,950\n"
    )
    dump_path = tmp_path / "RAW_M1.csv"
    dump_path.write_text(file_content, encoding="utf-8")

    df = load_metatrader_dump(dump_path)

    assert len(df) == 2
    assert str(df.index[0]) == "2024-01-02 09:30:00"
    assert df["Open"].iloc[0] == 150.20
    assert df["Close"].iloc[0] == 150.90


def test_load_daily_metatrader_without_time(tmp_path):
    """Test loading MT daily export without Time column."""
    file_content = (
        "<DATE>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>\n"
        "2024.01.02,1.0500,1.0600,1.0450,1.0580,12000\n"
        "2024.01.03,1.0580,1.0650,1.0520,1.0620,14500\n"
    )
    dump_path = tmp_path / "DAILY_D1.csv"
    dump_path.write_text(file_content, encoding="utf-8")

    df = load_metatrader_dump(dump_path)

    assert len(df) == 2
    assert str(df.index[0]) == "2024-01-02 00:00:00"
    assert df["Close"].iloc[0] == 1.0580


def test_file_not_found():
    """Test FileNotFoundError on missing file."""
    with pytest.raises(FileNotFoundError):
        load_metatrader_dump("/non/existent/path/dump.csv")
