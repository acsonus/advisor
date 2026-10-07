"""Signal generation from sentiment scores."""

import pandas as pd


def news_sentiment_signal(
    df_news: pd.DataFrame,
    bullish_threshold: float = 0.2,
    bearish_threshold: float = -0.2,
) -> pd.DataFrame:
    """
    Aggregates per-article sentiment into a daily signal aligned with OHLCV dates.

    Expects df_news to have columns: ['Date', 'Signed_Score']
    Returns DataFrame indexed by Date with columns: ['daily_sentiment', 'Signal']
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
