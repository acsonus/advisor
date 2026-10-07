"""
trading_report.py — Backward-compatibility facade.

Re-exports `generate_report` and reporting utilities from `advisor.reporting`.
"""

import os
import sys
import pandas as pd

from advisor.reporting import (
    PdfReportGenerator,
    build_report_styles,
    generate_report,
    get_signal_colour,
    render_sentiment_bar,
)
from advisor.reporting.styles import (
    ACCENT,
    DARK_BG,
    GREEN,
    LIGHT_GREY,
    MID_GREY,
    RED,
    WHITE,
    YELLOW,
)

__all__ = [
    "PdfReportGenerator",
    "generate_report",
    "build_report_styles",
    "get_signal_colour",
    "render_sentiment_bar",
    "DARK_BG",
    "ACCENT",
    "GREEN",
    "RED",
    "YELLOW",
    "WHITE",
    "LIGHT_GREY",
    "MID_GREY",
]

if __name__ == '__main__':
    from sentiments import YahooSentiments
    import trading_strategy as TradingStrategy

    ticker = sys.argv[1] if len(sys.argv) > 1 else 'COKE'
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    period = sys.argv[3] if len(sys.argv) > 3 else '1mo'
    interval = sys.argv[4] if len(sys.argv) > 4 else '1d'

    print(f'Fetching data for {ticker}  period={period}  interval={interval}…')
    data = TradingStrategy.downloadData(ticker, period=period, interval=interval)

    sentiments = YahooSentiments()
    sentiments.downloadYahooNews(ticker)
    df_news = sentiments.analyze_news()

    data.dropna(inplace=True)
    data['atr_signal'] = TradingStrategy.atr_trailing_stop(data)['Signal']
    data['ma_rsi_signal'] = TradingStrategy.ma_rsi_strategy(data)['Signal']

    sentiment_df = TradingStrategy.news_sentiment_signal(df_news)
    data_reset = data.reset_index()
    data_reset['Date'] = TradingStrategy.to_naive_s(data_reset['Date'])
    sentiment_reset = sentiment_df.reset_index().rename(columns={'Signal': 'sentiment_signal'})
    sentiment_reset['Date'] = TradingStrategy.to_naive_s(sentiment_reset['Date'])
    merged = pd.merge_asof(
        data_reset.sort_values('Date'),
        sentiment_reset.sort_values('Date'),
        on='Date',
        direction='backward',
    )
    data = merged.set_index('Date')
    data['sentiment_signal'] = data['sentiment_signal'].fillna('Hold')
    data['daily_sentiment'] = data['daily_sentiment'].fillna(0.0)

    path = generate_report(
        ticker=ticker,
        ohlcv_df=data,
        news_df=df_news,
        budget_eur=budget,
    )
    print(f'\nPDF report saved to:\n  {path}')
