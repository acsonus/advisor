from  sentiments import YahooSentiments
import trading_strategy as TradingStrategy
import pandas as pd
from pandas import DataFrame
from trading_report import generate_report
def runTest1():
    print("running test 1")
    ticker = 'AAPL'
    data = TradingStrategy.downloadData(ticker)

    sentiments = YahooSentiments()
    sentiments.downloadYahooNews(ticker)
    df_news = sentiments.analyze_news()

    data.dropna(inplace=True)
    data['atr_signal'] = TradingStrategy.atr_trailing_stop(data)['Signal']
    # fill gap algorythm is not relevant for this place
    #data['gap_fill_signal'] =  TradingStrategy.gap_fill_algorithm(data)['Signal']
    data['ma_rsi_signal'] = TradingStrategy.ma_rsi_strategy(data)['Signal']

    # Map daily sentiment signal onto the OHLCV date index.
    # merge_asof with direction='forward' assigns weekend/holiday news to the next trading day.
    sentiment_df = TradingStrategy.news_sentiment_signal(df_news)
    data_reset = data.reset_index()
    # Align datetime resolution — strip tz from intraday (tz-aware) and normalise to datetime64[s]
    data_reset['Date'] = TradingStrategy.to_naive_s(data_reset['Date'])
    sentiment_reset = sentiment_df.reset_index().rename(columns={'Signal': 'sentiment_signal'})
    sentiment_reset['Date'] = TradingStrategy.to_naive_s(sentiment_reset['Date'])
    merged = pd.merge_asof(
        data_reset.sort_values('Date'),
        sentiment_reset.sort_values('Date'),
        on='Date',
        direction='backward'  # assign most-recent sentiment to each bar (works for both daily and intraday)
    )
    data = merged.set_index('Date')
    #data['sentiment_signal'] = data['sentiment_signal'].fillna('Hold')
    data['daily_sentiment'] = data['daily_sentiment'].fillna(0.0)

    print(data)

    print("Test finished")
if __name__=='__main__':
    runTest1()