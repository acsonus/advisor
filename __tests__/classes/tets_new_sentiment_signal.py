# ===========================================================================
# 2.  news_sentiment_signal
# ===========================================================================
import pandas as pd
from advisor import ts
from src.backend.advisor.utilities import _assert_valid_signals
class TestNewsSentimentSignal:
    """Tests for the aggregated news sentiment → signal mapping."""

    def _news(self, scores, dates=None):
        n = len(scores)
        if dates is None:
            dates = pd.date_range("2023-01-01", periods=n, freq="D")
        return pd.DataFrame({"Date": dates, "Signed_Score": scores})

    def test_empty_df_returns_empty(self):
        result = ts.news_sentiment_signal(pd.DataFrame())
        assert result.empty

    def test_missing_signed_score_column_returns_empty(self):
        df = pd.DataFrame({"Date": ["2023-01-01"]})
        result = ts.news_sentiment_signal(df)
        assert result.empty

    def test_bullish_scores_produce_buy(self):
        df = self._news([0.8, 0.9, 0.85])
        result = ts.news_sentiment_signal(df)
        assert (result["Signal"] == "Buy").all()

    def test_bearish_scores_produce_sell(self):
        df = self._news([-0.8, -0.9, -0.85])
        result = ts.news_sentiment_signal(df)
        assert (result["Signal"] == "Sell").all()

    def test_neutral_scores_produce_hold(self):
        df = self._news([0.0, 0.05, -0.05])
        result = ts.news_sentiment_signal(df)
        assert (result["Signal"] == "Hold").all()

    def test_all_signals_valid(self):
        df = self._news([0.9, -0.7, 0.1, -0.1, 0.5])
        result = ts.news_sentiment_signal(df)
        _assert_valid_signals(result)

    def test_custom_thresholds_hold_within_band(self):
        """Scores within custom thresholds must remain Hold."""
        df = self._news([0.5, -0.5])
        result = ts.news_sentiment_signal(df, bullish_threshold=0.6, bearish_threshold=-0.6)
        assert (result["Signal"] == "Hold").all()

    def test_custom_thresholds_buy_above_band(self):
        df = self._news([0.7])
        result = ts.news_sentiment_signal(df, bullish_threshold=0.6, bearish_threshold=-0.6)
        assert result["Signal"].iloc[0] == "Buy"

    def test_multiple_articles_same_day_averaged(self):
        """Two articles on the same date with opposing scores should average to Hold."""
        df = self._news([0.8, -0.8], dates=["2023-01-01", "2023-01-01"])
        result = ts.news_sentiment_signal(df, bullish_threshold=0.2, bearish_threshold=-0.2)
        assert result["Signal"].iloc[0] == "Hold"

    def test_returns_daily_sentiment_column(self):
        df = self._news([0.5])
        result = ts.news_sentiment_signal(df)
        assert "daily_sentiment" in result.columns

    def test_daily_sentiment_equals_score_for_single_article(self):
        df = self._news([0.6])
        result = ts.news_sentiment_signal(df)
        assert abs(result["daily_sentiment"].iloc[0] - 0.6) < 1e-9