"""Tests for news sentiment signal mapping."""

import pandas as pd
import pytest

from utilities.test_helpers import assert_valid_signals as _assert_valid_signals
from advisor.sentiment import news_sentiment_signal


class TestNewsSentimentSignal:
    """Tests for the aggregated news sentiment → signal mapping."""

    def _news(self, scores, dates=None):
        """
        Goal:
            Helper fixture generator that builds synthetic news sentiment DataFrames
            with specified Signed_Score values and optional Date series.

        Execution Principle:
            1. Determine count `n` from `scores`.
            2. If dates is omitted, generate an n-period daily DatetimeIndex starting from 2023-01-01.
            3. Return a DataFrame with 'Date' and 'Signed_Score' columns.
        """
        n = len(scores)
        if dates is None:
            dates = pd.date_range("2023-01-01", periods=n, freq="D")
        return pd.DataFrame({"Date": dates, "Signed_Score": scores})

    def test_empty_df_returns_empty(self):
        """
        Goal:
            Verify that providing an empty input DataFrame returns an empty DataFrame gracefully.

        Execution Principle:
            1. Pass `pd.DataFrame()` into `news_sentiment_signal`.
            2. Assert returned DataFrame is empty (`result.empty is True`).
        """
        result = news_sentiment_signal(pd.DataFrame())
        assert result.empty

    def test_missing_signed_score_column_returns_empty(self):
        """
        Goal:
            Verify that input missing the mandatory 'Signed_Score' column returns empty without unhandled KeyError.

        Execution Principle:
            1. Construct DataFrame without 'Signed_Score'.
            2. Pass to `news_sentiment_signal`.
            3. Assert `result.empty` is True.
        """
        df = pd.DataFrame({"Date": ["2023-01-01"]})
        result = news_sentiment_signal(df)
        assert result.empty

    def test_bullish_scores_produce_buy(self):
        """
        Goal:
            Verify that scores strictly exceeding bullish threshold (+0.2 default) produce 'Buy' signals.

        Execution Principle:
            1. Generate synthetic news with scores [0.8, 0.9, 0.85].
            2. Pass through `news_sentiment_signal`.
            3. Assert all elements in 'Signal' column equal 'Buy'.
        """
        df = self._news([0.8, 0.9, 0.85])
        result = news_sentiment_signal(df)
        assert (result["Signal"] == "Buy").all()

    def test_bearish_scores_produce_sell(self):
        """
        Goal:
            Verify that scores below bearish threshold (-0.2 default) produce 'Sell' signals.

        Execution Principle:
            1. Generate synthetic news with negative scores [-0.8, -0.9, -0.85].
            2. Pass through `news_sentiment_signal`.
            3. Assert all elements in 'Signal' column equal 'Sell'.
        """
        df = self._news([-0.8, -0.9, -0.85])
        result = news_sentiment_signal(df)
        assert (result["Signal"] == "Sell").all()

    def test_neutral_scores_produce_hold(self):
        """
        Goal:
            Verify that sentiment scores within [-0.2, +0.2] produce 'Hold' signals.

        Execution Principle:
            1. Generate synthetic news with near-zero scores [0.0, 0.05, -0.05].
            2. Compute sentiment signals.
            3. Assert all signals are 'Hold'.
        """
        df = self._news([0.0, 0.05, -0.05])
        result = news_sentiment_signal(df)
        assert (result["Signal"] == "Hold").all()

    def test_all_signals_valid(self):
        """
        Goal:
            Verify that heterogeneous sentiment inputs only output signals within the allowed set ('Buy', 'Sell', 'Hold').

        Execution Principle:
            1. Generate mixed sentiment score inputs.
            2. Run `news_sentiment_signal`.
            3. Verify outputs conform to valid signal enum using `_assert_valid_signals`.
        """
        df = self._news([0.9, -0.7, 0.1, -0.1, 0.5])
        result = news_sentiment_signal(df)
        _assert_valid_signals(result)

    def test_custom_thresholds_hold_within_band(self):
        """
        Goal:
            Verify configurable custom sentiment thresholds classify inside-band scores as 'Hold'.

        Execution Principle:
            1. Supply scores [0.5, -0.5] with widened threshold parameters (±0.6).
            2. Assert signals remain 'Hold'.
        """
        df = self._news([0.5, -0.5])
        result = news_sentiment_signal(df, bullish_threshold=0.6, bearish_threshold=-0.6)
        assert (result["Signal"] == "Hold").all()

    def test_custom_thresholds_buy_above_band(self):
        """
        Goal:
            Verify configurable thresholds classify scores surpassing customized upper bound as 'Buy'.

        Execution Principle:
            1. Supply score 0.7 with bullish_threshold=0.6.
            2. Assert resulting signal is 'Buy'.
        """
        df = self._news([0.7])
        result = news_sentiment_signal(df, bullish_threshold=0.6, bearish_threshold=-0.6)
        assert result["Signal"].iloc[0] == "Buy"

    def test_multiple_articles_same_day_averaged(self):
        """
        Goal:
            Verify same-day conflicting news sentiment articles are aggregated via mean pooling.

        Execution Principle:
            1. Supply opposing scores [+0.8, -0.8] on identical dates.
            2. Group by date and calculate average (0.0).
            3. Verify final daily signal evaluates to 'Hold'.
        """
        df = self._news([0.8, -0.8], dates=["2023-01-01", "2023-01-01"])
        result = news_sentiment_signal(df, bullish_threshold=0.2, bearish_threshold=-0.2)
        assert result["Signal"].iloc[0] == "Hold"

    def test_returns_daily_sentiment_column(self):
        """
        Goal:
            Verify that output DataFrame includes 'daily_sentiment' continuous score column.

        Execution Principle:
            1. Execute `news_sentiment_signal` on valid input.
            2. Assert `'daily_sentiment'` exists in output columns.
        """
        df = self._news([0.5])
        result = news_sentiment_signal(df)
        assert "daily_sentiment" in result.columns

    def test_daily_sentiment_equals_score_for_single_article(self):
        """
        Goal:
            Verify single article daily sentiment precisely preserves the original article signed score.

        Execution Principle:
            1. Supply single article with score 0.6.
            2. Assert `daily_sentiment` value equals 0.6 within floating point tolerance.
        """
        df = self._news([0.6])
        result = news_sentiment_signal(df)
        assert abs(result["daily_sentiment"].iloc[0] - 0.6) < 1e-9
