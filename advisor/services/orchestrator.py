"""Service layer orchestrating data fetching, strategy execution, sentiment, and reporting."""

import pandas as pd

from advisor.backtesting import backtest_strategy
from advisor.data import download_data, to_naive_s
from advisor.optimization import StrategyOptimizer
from advisor.reporting import generate_report
from advisor.sentiment import YahooSentiments, news_sentiment_signal
from advisor.strategies import (
    StrategyRegistry,
    atr_trailing_stop,
    bollinger_squeeze_strategy,
    calculate_vwap,
    get_effective_params,
    macd_histogram_reversal_strategy,
    ma_rsi_strategy,
    save_optimal_params,
)

DEFAULT_SAMPLE_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]


def run_strategies_pipeline(
    ticker: str,
    budget_eur: float,
    period: str = '1mo',
    interval: str = '1d',
) -> tuple[dict, str]:
    """
    Execute end-to-end trading advisory pipeline for a given stock ticker.

    Goal:
    -----
    Coordinate market data downloading, technical strategy signal generation,
    AI news sentiment analysis, time-series signal alignment, and executive PDF
    report compilation into a single business service invocation decoupled from HTTP.

    Execution Principle:
    --------------------
    1. Data Ingestion & Sanitization:
       - Downloads OHLCV bars via `download_data()`.
       - Drops missing rows (`dropna()`).
       - Enforces minimum bar count: requires at least 27 bars for technical indicator
         warm-up (MA-RSI requires at least 26 bars for slow EMA; ATR requires at least 14 bars).
    2. Technical Strategy Calculation:
       - Computes ATR Trailing Stop signals on price bars.
       - Computes MA-RSI crossover signals on price bars.
    3. Sentiment Analysis:
       - Queries Yahoo Finance news headlines for `ticker`.
       - Classifies article titles via FinBERT and computes per-article sentiment scores.
       - Aggregates article scores into daily signals using `news_sentiment_signal()`.
    4. Time-Series Alignment:
       - Normalizes both price index dates and sentiment dates to timezone-naive `datetime64[s]`.
       - Merges sentiment signals backward in time onto price bars using `pd.merge_asof(direction='backward')`,
         ensuring that weekend/holiday news maps to subsequent market trading bars without look-ahead bias.
       - Fills missing sentiment signals with `'Hold'` and sentiment score with `0.0`.
    5. Result Assembly:
       - Builds summary dictionary with latest bar close, signals, and dates.
       - Invokes `generate_report()` to compile and write the PDF report.
    6. Returns `(signals_dict, generated_pdf_path)`.

    Parameters:
    -----------
    ticker : str
        Stock symbol (e.g. 'AAPL').
    budget_eur : float
        Available investor budget in EUR.
    period : str, default '1mo'
        Historical duration.
    interval : str, default '1d'
        Bar resolution.

    Returns:
    --------
    tuple[dict, str]
        Tuple of (signals_summary_dict, absolute_path_to_pdf_report).

    Raises:
    -------
    ValueError
        If no data is returned or data has fewer than 27 bars.
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
    use_optimal_params: bool = False,
) -> dict:
    """
    Run backtest simulations across all core strategies for a single ticker.

    Goal:
    -----
    Evaluate and compare the quantitative performance of five distinct strategies
    (ATR, MA-RSI, Bollinger Squeeze, MACD, and VWAP) on the same asset under
    identical liquidity and commission constraints.

    Execution Principle:
    --------------------
    1. Downloads OHLCV data for `ticker` and drops missing bars.
    2. Runs each strategy against the price history:
       - Uses tuned parameters from `param_store` when `use_optimal_params=True`.
       - Otherwise runs baseline default parameters.
    3. Runs the `backtest_strategy` engine on each signal series using `initial_cash`,
       `commission_bps`, and any active risk management parameters (stop-loss, trailing stop).
    4. Computes total profit in currency: `round(initial_cash * (total_return_pct / 100.0), 2)`.
    5. Packages returns, profit, trade count, win rate, drawdown, Sharpe ratio,
       and skipped buys into a structured result dictionary.

    Parameters:
    -----------
    ticker : str
        Stock symbol to simulate.
    period : str
        Lookback timeframe.
    interval : str
        Bar resolution.
    initial_cash : float
        Account starting capital.
    commission_bps : float
        One-way trade fee in basis points.
    use_optimal_params : bool, default False
        Whether to load and apply stored optimal hyperparameters and risk limits.

    Returns:
    --------
    dict
        Dictionary containing ticker symbol, bar count, and strategy performance metrics.
    """
    data = download_data(ticker, period=period, interval=interval)
    data.dropna(inplace=True)

    if data.empty:
        raise ValueError(
            f"No data returned for '{ticker}' with period='{period}' and interval='{interval}'."
        )

    strategy_names = [
        'atr_trailing_stop',
        'ma_rsi',
        'bollinger_squeeze',
        'macd_histogram_reversal',
        'vwap_cross',
    ]

    strategy_results = {}
    for strategy_name in strategy_names:
        strat_params, risk_params = ({}, {})
        if use_optimal_params:
            strat_params, risk_params = get_effective_params(ticker, strategy_name)

        strategy = StrategyRegistry.get(strategy_name, **strat_params)
        signals_df = strategy.generate_signals(data)

        df_bt = pd.DataFrame({
            'Close': data['Close'],
            'Signal': signals_df['Signal'],
        }).dropna()
        if 'High' in data.columns:
            df_bt['High'] = data['High']
        if 'Low' in data.columns:
            df_bt['Low'] = data['Low']

        metrics = backtest_strategy(
            df_bt,
            signal_col='Signal',
            close_col='Close',
            fee_bps=commission_bps,
            slippage_bps=0.0,
            initial_cash=initial_cash,
            stop_loss_pct=risk_params.get('stop_loss_pct'),
            take_profit_pct=risk_params.get('take_profit_pct'),
            max_hold_bars=risk_params.get('max_hold_bars'),
            trailing_stop=bool(risk_params.get('trailing_stop', False)),
            stop_loss_atr_mult=risk_params.get('stop_loss_atr_mult'),
            take_profit_atr_mult=risk_params.get('take_profit_atr_mult'),
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
            'used_optimal_params': bool(strat_params or risk_params),
        }

    return {
        'ticker': ticker,
        'bars': int(len(data)),
        'strategies': strategy_results,
    }


def optimize_strategy_on_dataframe(
    data: pd.DataFrame,
    ticker: str,
    strategy_name: str,
    n_iter: int = 50,
    initial_cash: float = 10000.0,
    commission_bps: float = 0.0,
    save_params: bool = True,
    seed: int | None = 42,
) -> dict:
    """
    Execute systematic parameter search and risk limit optimization on an OHLCV DataFrame.

    Parameters:
    -----------
    data : pd.DataFrame
        Clean OHLCV price bars.
    ticker : str
        Symbol identifier.
    strategy_name : str
        Strategy name registered in StrategyRegistry.
    n_iter : int, default 50
        Number of randomized parameter configurations to test.
    initial_cash : float, default 10000.0
        Account starting capital.
    commission_bps : float, default 0.0
        Trading fee in basis points.
    save_params : bool, default True
        Whether to persist winning parameters to optimal_params.json.
    seed : int, optional, default 42
        Random seed for reproducible parameter sampling.

    Returns:
    --------
    dict
        Structured optimization report.
    """
    if data.empty:
        raise ValueError("Cannot optimize on empty DataFrame.")

    optimizer = StrategyOptimizer(
        initial_cash=initial_cash,
        commission_bps=commission_bps,
    )

    result = optimizer.optimize(
        data=data,
        strategy_name=strategy_name,
        n_iter=n_iter,
        seed=seed,
    )

    saved_path = None
    if save_params:
        saved_file = save_optimal_params(
            ticker=ticker,
            strategy_name=strategy_name,
            strategy_params=result.best_strategy_params,
            risk_params=result.best_risk_params,
            metrics=result.optimized_out_of_sample,
        )
        saved_path = str(saved_file)

    report = result.to_dict()
    report["ticker"] = ticker
    report["saved_to"] = saved_path
    return report


def optimize_strategy_for_ticker(
    ticker: str,
    strategy_name: str,
    period: str = '1y',
    interval: str = '1d',
    n_iter: int = 50,
    initial_cash: float = 10000.0,
    commission_bps: float = 0.0,
    save_params: bool = True,
    seed: int | None = 42,
) -> dict:
    """
    Execute systematic parameter search and risk limit optimization for a strategy and ticker.

    Goal:
    -----
    Find indicator configurations and risk controls (stop loss, profit target, trailing stop)
    that optimize risk-adjusted returns and verify that out-of-sample performance holds up.

    Parameters:
    -----------
    ticker : str
        Stock symbol to optimize (e.g. 'AAPL').
    strategy_name : str
        Strategy name registered in StrategyRegistry (e.g. 'ma_rsi').
    period : str, default '1y'
        Historical duration to download.
    interval : str, default '1d'
        Bar resolution.
    n_iter : int, default 50
        Number of randomized parameter configurations to test.
    initial_cash : float, default 10000.0
        Account starting capital.
    commission_bps : float, default 0.0
        Trading fee in basis points.
    save_params : bool, default True
        Whether to persist winning parameters to optimal_params.json.
    seed : int, optional, default 42
        Random seed for reproducible parameter sampling.

    Returns:
    --------
    dict
        Structured optimization report including best parameters, in-sample vs
        out-of-sample metrics, baseline comparison, and top trials.
    """
    data = download_data(ticker, period=period, interval=interval)
    data.dropna(inplace=True)

    if data.empty:
        raise ValueError(
            f"No data returned for '{ticker}' with period='{period}' and interval='{interval}'."
        )

    return optimize_strategy_on_dataframe(
        data=data,
        ticker=ticker,
        strategy_name=strategy_name,
        n_iter=n_iter,
        initial_cash=initial_cash,
        commission_bps=commission_bps,
        save_params=save_params,
        seed=seed,
    )
