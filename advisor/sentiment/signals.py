"""Signal generation from sentiment scores."""

import pandas as pd


def news_sentiment_signal(
    df_news: pd.DataFrame,
    bullish_threshold: float = 0.2,
    bearish_threshold: float = -0.2,
) -> pd.DataFrame:
    """
    Aggregate per-article sentiment scores into daily trading signals.

    Goal:
    -----
    Transform unstructured article-level sentiment observations into an aggregated
    daily time-series with clear, actionable directional signals ('Buy', 'Sell', 'Hold')
    that can be merged onto price candles for strategy decision-making and reporting.

    Execution Principle:
    --------------------
    1. Empty & Schema Validation:
       If `df_news` is empty or lacks the required `'Signed_Score'` column, returns an empty
       DataFrame indexed by an empty Index named `'Date'` with columns `['daily_sentiment', 'Signal']`.
       This guarantees downstream operations like `reset_index()` and `pd.merge_asof(on='Date')`
       do not fail when no news is available.
    2. Grouping & Daily Averaging:
       Groups articles by `'Date'` and computes the arithmetic mean of `'Signed_Score'`.
       Renames the averaged column to `'daily_sentiment'`.
    3. Signal Assignment:
       - Default signal is initialized to `'Hold'`.
       - If `daily_sentiment > bullish_threshold`: signal set to `'Buy'`.
       - If `daily_sentiment < bearish_threshold`: signal set to `'Sell'`.
       - Neutral values between `[bearish_threshold, bullish_threshold]` remain `'Hold'`.
    4. Returns a DataFrame indexed by `'Date'` containing `['daily_sentiment', 'Signal']`.

    Parameters:
    -----------
    df_news : pd.DataFrame
        DataFrame containing articles with `'Date'` (datetime-like) and `'Signed_Score'`
        (float in range [-1.0, 1.0]).
    bullish_threshold : float, default 0.2
        Threshold above which daily sentiment triggers a 'Buy' signal.
    bearish_threshold : float, default -0.2
        Threshold below which daily sentiment triggers a 'Sell' signal.

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by `'Date'` with columns `['daily_sentiment', 'Signal']`.
    """
    if df_news.empty or 'Signed_Score' not in df_news.columns:
        # Index must be named 'Date' so downstream merge_asof(on="Date") calls
        # can reset_index() and find a 'Date' column even with no news rows.
        return pd.DataFrame(columns=['daily_sentiment', 'Signal'], index=pd.Index([], name='Date'))

    daily = (
        df_news.groupby('Date')['Signed_Score']
        .mean()
        .rename('daily_sentiment')
        .to_frame()
    )
    daily['Signal'] = 'Hold'
    daily.loc[daily['daily_sentiment'] > bullish_threshold, 'Signal'] = 'Buy'
    daily.loc[daily['daily_sentiment'] < bearish_threshold, 'Signal'] = 'Sell'
    return daily
