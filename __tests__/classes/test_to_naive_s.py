# ===========================================================================
# 1.  to_naive_s 
# 
# ===========================================================================
import pandas as pd
import trading_strategy as ts
class TestToNaiveS:
    """Test suite for timestamp normalization function to_naive_s."""

    def test_tz_aware_utc_stripped(self):
        """
        Goal:
            Verify that UTC timezone-aware Series timestamps are successfully stripped of timezone info.

        Execution Principle:
            1. Construct a pandas Series with 5 UTC datetime timestamps.
            2. Pass through `to_naive_s`.
            3. Assert `dt.tz` is `None`.
        """
        s = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC"))
        result = ts.to_naive_s(s)
        assert result.dt.tz is None

    def test_tz_aware_eastern_stripped(self):
        """
        Goal:
            Verify that non-UTC timezone-aware Series (e.g. US/Eastern) are converted to UTC and stripped.

        Execution Principle:
            1. Construct a pandas Series with US/Eastern localized datetimes.
            2. Pass through `to_naive_s`.
            3. Verify timezone is removed (`dt.tz is None`).
        """
        s = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D", tz="US/Eastern"))
        result = ts.to_naive_s(s)
        assert result.dt.tz is None

    def test_tz_naive_unchanged(self):
        """
        Goal:
            Verify that already timezone-naive Series datetimes remain naive without error.

        Execution Principle:
            1. Construct a pandas Series with naive datetimes.
            2. Invoke `to_naive_s`.
            3. Assert result remains timezone-naive (`dt.tz is None`).
        """
        s = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D"))
        result = ts.to_naive_s(s)
        assert result.dt.tz is None

    def test_dtype_is_datetime64_s(self):
        """
        Goal:
            Verify that output Series preserves or conforms to a datetime64 dtype.

        Execution Principle:
            1. Process UTC timestamps through `to_naive_s`.
            2. Inspect dtype string representation and assert `datetime64` presence.
        """
        s = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC"))
        result = ts.to_naive_s(s)
        assert "datetime64" in str(result.dtype)