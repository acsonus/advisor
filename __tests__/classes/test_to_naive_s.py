# ===========================================================================
# 1.  to_naive_s 
# 
# ===========================================================================
import pandas as pd
from advisor import ts
class TestToNaiveS:
    def test_tz_aware_utc_stripped(self):
        s = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC"))
        result = ts.to_naive_s(s)
        assert result.dt.tz is None

    def test_tz_aware_eastern_stripped(self):
        s = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D", tz="US/Eastern"))
        result = ts.to_naive_s(s)
        assert result.dt.tz is None

    def test_tz_naive_unchanged(self):
        s = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D"))
        result = ts.to_naive_s(s)
        assert result.dt.tz is None

    def test_dtype_is_datetime64_s(self):
        s = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC"))
        result = ts.to_naive_s(s)
        assert "datetime64" in str(result.dtype)