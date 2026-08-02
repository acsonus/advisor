#general imports 
import sys
import os
from matplotlib import axes, ticker
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pandas import DataFrame
from pathlib import Path



#code to include files from parent directory
_root = Path(__file__).parent.parent  # advisor/
for _p in [_root, _root.parent]:      # advisor/, Projects/
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
#Module dependent files to be included        
from sentiments import YahooSentiments
import trading_strategy as TradingStrategy
from trading_report import generate_report
#downlad data for a given ticker and period. Runs through all stratagies and give a recommendations. Current test is for APPL and for 1 day period
def test1():
    print("running test 1")
    ticker = 'AAPL'
    data = TradingStrategy.downloadData(ticker)

    sentiments = YahooSentiments()
    sentiments.downloadYahooNews(ticker)
    df_news = sentiments.analyze_news()

    data.dropna(inplace=True)
    data['atr_signal'] = TradingStrategy.atr_trailing_stop(data)['Signal']
    # fill gap algorythm is not relevant for this place
    #data['gap_fill_signal'] =  TradingStrategy.gap_fill_algorith m(data)['Signal']
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
    
def test2():
    print("running test 2")
    ticker = 'AAPL'
    data_1h = TradingStrategy.downloadData(ticker, period="1mo", interval="1h")
    data_15m = TradingStrategy.downloadData(ticker, period="1mo", interval="15m")
    # currently not using news sentiment for this test, but leaving the code here for future use
    # sentiments = YahooSentiments()
    # sentiments.downloadYahooNews(ticker)
    
    #Split data into 3/4 and 1/4 for training and testing for the set of 1h
    split_index = int(len(data_1h) * 0.75)
    train_data_1h = data_1h.iloc[:split_index]
    test_data_1h = data_1h.iloc[split_index:]
    #the same for 15 min data
    train_data_15m = data_15m.iloc[:split_index]
    test_data_15m = data_15m.iloc[split_index:]
    #print training and testing data
    # build marker series: show price only where signal fires
    train_data_1h_copy = train_data_1h.copy()
    train_data_1h_copy.dropna(inplace=True)
    # the same for 15 min data
    train_data_15m_copy = train_data_15m.copy()
    train_data_15m_copy.dropna(inplace=True)
    #run strategies on training data
    atr_result_1h = TradingStrategy.atr_trailing_stop(train_data_1h_copy)
    ma_result_1h = TradingStrategy.ma_rsi_strategy(train_data_1h_copy) 
    atr_result_15m = TradingStrategy.atr_trailing_stop(train_data_15m_copy)
    ma_result_15m = TradingStrategy.ma_rsi_strategy(train_data_15m_copy)
    #Siganl array for 1h data
    atr_buy_1h  = train_data_1h_copy['Close'].where(atr_result_1h['Signal'] == 'Buy')
    atr_sell_1h = train_data_1h_copy['Close'].where(atr_result_1h['Signal'] == 'Sell')
    ma_buy_1h  = train_data_1h_copy['Close'].where(ma_result_1h['Signal'] == 'Buy')
    ma_sell_1h = train_data_1h_copy['Close'].where(ma_result_1h['Signal'] == 'Sell')            
    atr_buy_15m  = train_data_15m_copy['Close'].where(atr_result_15m['Signal'] == 'Buy')
    atr_sell_15m = train_data_15m_copy['Close'].where(atr_result_15m['Signal'] == 'Sell')
    ma_buy_15m   = train_data_15m_copy['Close'].where(ma_result_15m['Signal']  == 'Buy')
    ma_sell_15m  = train_data_15m_copy['Close'].where(ma_result_15m['Signal']  == 'Sell')
    
    # adding signal array to the chart
    ap = [
    mpf.make_addplot(atr_buy_1h,  type='scatter', markersize=80, marker='^', color='green',  label='ATR Buy'),
    mpf.make_addplot(atr_sell_1h, type='scatter', markersize=80, marker='v', color='red',    label='ATR Sell'),
    mpf.make_addplot(ma_buy_1h,   type='scatter', markersize=80, marker='^', color='lime',   label='MA Buy'),
    mpf.make_addplot(ma_sell_1h,  type='scatter', markersize=80, marker='v', color='orange', label='MA Sell'),
    mpf.make_addplot(atr_buy_1h,  type='scatter', markersize=80, marker='^', color='green',  label='ATR Buy'),
    mpf.make_addplot(atr_sell_1h, type='scatter', markersize=80, marker='v', color='red',    label='ATR Sell'),
    mpf.make_addplot(ma_buy_1h,   type='scatter', markersize=80, marker='^', color='lime',   label='MA Buy'),
    mpf.make_addplot(ma_sell_1h,  type='scatter', markersize=80, marker='v', color ='orange', label='MA Sell'),
    
    ]
    #plot training data
    #plot in a form for OHLV candles
    #mpf.plot(train_data_1h, type='candle', style='charles', title=ticker + " Training Data", ylabel='Price', volume=False, addplot=ap)
    #try two plots in one figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})

    mpf.plot(train_data_1h_copy,  type='candle', style='charles', ema=(12,26), ax=axes[0][0], volume=False)
    mpf.plot(train_data_15m_copy, type='candle', style='charles', ema=(12,26), ax=axes[0][1], volume=False)

    plt.tight_layout()
    plt.show()
    #run strategies on training data
    train_data_1h.dropna(inplace=True)

    
    
if __name__=='__main__':
    #test1()
    test2()