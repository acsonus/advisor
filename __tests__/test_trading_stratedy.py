"""
Interactive plotting and exploratory visual verification.

Marked as live and headless to prevent blocking test runners.
"""

import matplotlib
matplotlib.use('Agg')  # Headless backend to avoid GUI window popup
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import pytest

from advisor.data import download_data, to_naive_s
from advisor.sentiment import YahooSentiments, news_sentiment_signal
from advisor.strategies import atr_trailing_stop, ma_rsi_strategy


@pytest.mark.live
def test_visual_pipeline_smoke():
    """
    Goal:
        Validate the complete real-data ingestion, sentiment analysis, strategy calculation,
        and as-of timestamp merging pipeline using live market feeds for AAPL.

    Execution Principle:
        1. Fetch 1 month of daily OHLCV bars for AAPL via `download_data`.
        2. Fetch recent Yahoo Finance headlines and evaluate FinBERT sentiments.
        3. Evaluate ATR trailing stop and MA-RSI strategies over the bars.
        4. Normalize datetime indices to naive UTC timestamps via `to_naive_s`.
        5. Merge sentiment scores onto OHLCV bars using backward `merge_asof`.
        6. Verify resulting unified dataset is non-empty and properly populated.
    """
    ticker = 'AAPL'
    data = download_data(ticker, period="1mo", interval="1d")

    sentiments = YahooSentiments()
    sentiments.downloadYahooNews(ticker)
    df_news = sentiments.analyze_news()

    data.dropna(inplace=True)
    data['atr_signal'] = atr_trailing_stop(data)['Signal']
    data['ma_rsi_signal'] = ma_rsi_strategy(data)['Signal']

    sentiment_df = news_sentiment_signal(df_news)
    data_reset = data.reset_index()
    data_reset['Date'] = to_naive_s(data_reset['Date'])
    sentiment_reset = sentiment_df.reset_index().rename(columns={'Signal': 'sentiment_signal'})
    sentiment_reset['Date'] = to_naive_s(sentiment_reset['Date'])
    merged = pd.merge_asof(
        data_reset.sort_values('Date'),
        sentiment_reset.sort_values('Date'),
        on='Date',
        direction='backward',
    )
    data = merged.set_index('Date')
    data['daily_sentiment'] = data['daily_sentiment'].fillna(0.0)
    assert not data.empty


@pytest.mark.live
def test_visual_multi_timeframe_plot():
    """
    Goal:
        Verify candlestick rendering and multi-timeframe strategy generation across 1h
        and 15m intervals without blocking headless test runners.

    Execution Principle:
        1. Ingest multi-frequency historical bars (1h and 15m) for AAPL.
        2. Partition data into training windows and run ATR trailing stop and MA-RSI strategies.
        3. Construct a 2x2 subplot canvas using Matplotlib and mplfinance candle renderers.
        4. Verify headless figure generation and close figure safely to avoid memory leaks.
        5. Assert strategy indicator outputs are non-empty across the evaluated intervals.
    """
    ticker = 'AAPL'
    data_1h = download_data(ticker, period="1mo", interval="1h")
    data_15m = download_data(ticker, period="1mo", interval="15m")

    split_index = int(len(data_1h) * 0.75)
    train_data_1h = data_1h.iloc[:split_index].copy().dropna()
    train_data_15m = data_15m.iloc[:split_index].copy().dropna()

    atr_result_1h = atr_trailing_stop(train_data_1h)
    ma_result_1h = ma_rsi_strategy(train_data_1h)

    fig, axes = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})
    mpf.plot(train_data_1h, type='candle', style='charles', ema=(12, 26), ax=axes[0][0], volume=False)
    mpf.plot(train_data_15m, type='candle', style='charles', ema=(12, 26), ax=axes[0][1], volume=False)
    plt.tight_layout()
    plt.close(fig)  # Close without blocking GUI
    assert len(atr_result_1h) > 0
    assert len(ma_result_1h) > 0


if __name__ == '__main__':
    test_visual_pipeline_smoke()
    test_visual_multi_timeframe_plot()