"""Service layer orchestrating data fetching, strategy execution, sentiment, and reporting."""

import pandas as pd

from advisor.backtesting import backtest_strategy
from advisor.data import download_data, to_naive_s
from advisor.reporting import generate_report
from advisor.sentiment import YahooSentiments, news_sentiment_signal
from advisor.strategies import (
    atr_trailing_stop,
    bollinger_squeeze_strategy,
    calculate_vwap,
    macd_histogram_reversal_strategy,
    ma_rsi_strategy,
)

DEFAULT_SAMPLE_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]


def run_strategies_pipeline(
    ticker: str,
    budget_eur: float,
    period: str = '1mo',
    interval: str = '1d',
) -> tuple[dict, str]:
    """
    Core strategy workflow: download market data, compute signals, analyze sentiment, and build PDF.

    Returns
    -------
    tuple[dict, str]
        (signals_dict, generated_pdf_path)
    """
    data = download_data(ticker, period=period, interval=interval)
    data.dropna(inplace=True)

    if data.empty:
        raise ValueError(
            f"No data returned for '{ticker}' with period='{period}', interval='{interval}'. "
            "The market may be closed or the ticker may be invalid."
        )

    min_bars = 27
    if len(data) < min_bars:
        raise ValueError(
            f"Insufficient data: only {len(data)} bar(s) returned for period='{period}', "
            f"interval='{interval}'. Need at least {min_bars} bars for technical indicators. "
            "Try a longer period (e.g. period='5d' for 60m, 'ytd' for 1d)."
        )

    # Compute technical strategy signals
    data["atr_signal"] = atr_trailing_stop(data)["Signal"]
    data["ma_rsi_signal"] = ma_rsi_strategy(data)["Signal"]

    # Sentiment analysis
    sentiments = YahooSentiments()
    sentiments.downloadYahooNews(ticker)
    df_news = sentiments.analyze_news()
    sentiment_df = news_sentiment_signal(df_news)

    # Merge sentiment onto OHLCV index
    data_reset = data.reset_index()
    data_reset["Date"] = to_naive_s(data_reset["Date"])

    sentiment_reset = sentiment_df.reset_index().rename(columns={"Signal": "sentiment_signal"})
    sentiment_reset["Date"] = to_naive_s(sentiment_reset["Date"])

    merged = pd.merge_asof(
        data_reset.sort_values("Date"),
        sentiment_reset.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    data = merged.set_index("Date")
    data["sentiment_signal"] = data["sentiment_signal"].fillna("Hold")
    data["daily_sentiment"] = data["daily_sentiment"].fillna(0.0)

    # Signal summary
    latest = data.iloc[-1]
    signals = {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "latest_date": str(data.index[-1]),
        "close": float(latest["Close"]) if "Close" in data.columns else None,
        "atr_signal": str(latest.get("atr_signal", "N/A")),
        "ma_rsi_signal": str(latest.get("ma_rsi_signal", "N/A")),
        "sentiment_signal": str(latest.get("sentiment_signal", "N/A")),
        "daily_sentiment": float(latest.get("daily_sentiment", 0.0)),
    }

    # Generate PDF
    pdf_path = generate_report(
        ticker=ticker,
        ohlcv_df=data,
        news_df=df_news,
        budget_eur=budget_eur,
    )

    return signals, pdf_path


def run_simulation_for_ticker(
    ticker: str,
    period: str,
    interval: str,
    initial_cash: float,
    commission_bps: float,
) -> dict:
    """Run all strategies and backtest simulations for a single ticker."""
    data = download_data(ticker, period=period, interval=interval)
    data.dropna(inplace=True)

    if data.empty:
        raise ValueError(
            f"No data returned for '{ticker}' with period='{period}' and interval='{interval}'."
        )

    strategy_signals = {
        'atr_trailing_stop': atr_trailing_stop(data)['Signal'],
        'ma_rsi': ma_rsi_strategy(data)['Signal'],
        'bollinger_squeeze': bollinger_squeeze_strategy(data)['Signal'],
        'macd_histogram_reversal': macd_histogram_reversal_strategy(data)['Signal'],
        'vwap_cross': calculate_vwap(data)['Signal'],
    }

    strategy_results = {}
    for strategy_name, signal_series in strategy_signals.items():
        df_bt = pd.DataFrame({
            'Close': data['Close'],
            'Signal': signal_series,
        }).dropna()

        metrics = backtest_strategy(
            df_bt,
            signal_col='Signal',
            close_col='Close',
            fee_bps=commission_bps,
            slippage_bps=0.0,
            initial_cash=initial_cash,
        )

        profit_amount = round(initial_cash * (metrics['total_return_pct'] / 100.0), 2)
        strategy_results[strategy_name] = {
            'total_return_pct': metrics['total_return_pct'],
            'profit': profit_amount,
            'n_trades': metrics['n_trades'],
            'win_rate_pct': metrics['win_rate_pct'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'skipped_buys_due_to_cash': metrics.get('skipped_buys_due_to_cash', 0),
        }

    return {
        'ticker': ticker,
        'bars': int(len(data)),
        'strategies': strategy_results,
    }
