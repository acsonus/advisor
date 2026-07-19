"""  
app.py
Flask API server exposing trading strategy endpoints.

Endpoints:
  GET /api/strategies/run?ticker=AAPL&budget=300
      Runs ATR, MA-RSI and sentiment strategies, returns signals as JSON
      and the path of the generated PDF report.

  GET /api/strategies/report?ticker=AAPL&budget=300
      Same as /run but streams the generated PDF back to the caller.

Common query parameters:
  ticker    str   default AAPL   Ticker symbol (letters, digits, . - ^ =)
  budget    float default 300    Available budget in EUR (informational)
  period    str   default 1mo    Yahoo Finance period:   1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
  interval  str   default 1d     Yahoo Finance interval: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo
"""

import os
import re
import sys

import pandas as pd
from flask import Flask, jsonify, request, send_file, abort

# Ensure local modules are importable when the server is launched from another
# working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import trading_strategy as TradingStrategy
from trading_report import generate_report
from sentiments import YahooSentiments

app = Flask(__name__)

DEFAULT_SAMPLE_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_strategies(ticker: str, budget_eur: float,
                    period: str = '1mo', interval: str = '1d') -> tuple[dict, str]:
    """
    Core logic: download data, compute signals, generate PDF.
    Returns (signals_dict, pdf_path).
    """
    # 1. Download OHLCV data
    data = TradingStrategy.downloadData(ticker, period=period, interval=interval)
    data.dropna(inplace=True)

    if data.empty:
        raise ValueError(
            f"No data returned for '{ticker}' with period='{period}', interval='{interval}'. "
            "The market may be closed or the ticker may be invalid."
        )

    # MA-RSI requires at least long_ema_period (26) bars; ATR requires at least atr_period (14).
    _MIN_BARS = 27
    if len(data) < _MIN_BARS:
        raise ValueError(
            f"Insufficient data: only {len(data)} bar(s) returned for period='{period}', "
            f"interval='{interval}'. Need at least {_MIN_BARS} bars for technical indicators. "
            "Try a longer period (e.g. period='5d' for 60m, 'ytd' for 1d)."
        )

    # 2. Technical strategy signals
    data["atr_signal"] = TradingStrategy.atr_trailing_stop(data)["Signal"]
    data["ma_rsi_signal"] = TradingStrategy.ma_rsi_strategy(data)["Signal"]

    # 3. Sentiment analysis
    sentiments = YahooSentiments()
    sentiments.downloadYahooNews(ticker)
    df_news = sentiments.analyze_news()

    sentiment_df = TradingStrategy.news_sentiment_signal(df_news)

    # 4. Merge sentiment onto OHLCV index
    data_reset = data.reset_index()
    data_reset["Date"] = TradingStrategy.to_naive_s(data_reset["Date"])

    sentiment_reset = (
        sentiment_df.reset_index()
        .rename(columns={"Signal": "sentiment_signal"})
    )
    sentiment_reset["Date"] = TradingStrategy.to_naive_s(sentiment_reset["Date"])

    merged = pd.merge_asof(
        data_reset.sort_values("Date"),
        sentiment_reset.sort_values("Date"),
        on="Date",
        direction="backward",  # assign most-recent sentiment to each bar (works for both daily and intraday)
    )
    data = merged.set_index("Date")
    data["sentiment_signal"] = data["sentiment_signal"].fillna("Hold")
    data["daily_sentiment"] = data["daily_sentiment"].fillna(0.0)

    # 5. Build the signal summary for the JSON response
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

    # 6. Generate PDF
    pdf_path = generate_report(
        ticker=ticker,
        ohlcv_df=data,
        news_df=df_news,
        budget_eur=budget_eur,
    )

    return signals, pdf_path


def _validate_ticker(ticker: str) -> str:
    cleaned = ticker.upper().strip()
    if not re.match(r'^[A-Z0-9.\-\^=]{1,10}$', cleaned):
        raise ValueError(f"Invalid ticker '{ticker}'.")
    return cleaned


def _run_simulation_for_ticker(
    ticker: str,
    period: str,
    interval: str,
    initial_cash: float,
    commission_bps: float,
) -> dict:
    data = TradingStrategy.downloadData(ticker, period=period, interval=interval)
    data.dropna(inplace=True)

    if data.empty:
        raise ValueError(
            f"No data returned for '{ticker}' with period='{period}' and interval='{interval}'."
        )

    strategy_signals = {
        'atr_trailing_stop': TradingStrategy.atr_trailing_stop(data)['Signal'],
        'ma_rsi': TradingStrategy.ma_rsi_strategy(data)['Signal'],
        'bollinger_squeeze': TradingStrategy.bollinger_squeeze_strategy(data)['Signal'],
        'macd_histogram_reversal': TradingStrategy.macd_histogram_reversal_strategy(data)['Signal'],
        'vwap_cross': TradingStrategy.calculate_vwap(data)['Signal'],
    }

    strategy_results = {}
    for strategy_name, signal_series in strategy_signals.items():
        df_bt = pd.DataFrame({
            'Close': data['Close'],
            'Signal': signal_series,
        }).dropna()

        metrics = TradingStrategy.backtest_strategy(
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


def _print_simulation_summary(results: list[dict], initial_cash: float, commission_bps: float) -> None:
    print('===== STRATEGY SIMULATION SUMMARY =====')
    print(f'Assumptions: initial_cash={initial_cash:.2f}, commission_bps={commission_bps:.2f}')

    for ticker_result in results:
        ticker = ticker_result['ticker']
        print(f'\nTicker: {ticker}')
        print('Strategy                     Return %      Profit        Trades')
        print('--------------------------------------------------------------')
        for strategy_name, metrics in ticker_result['strategies'].items():
            print(
                f"{strategy_name:28} "
                f"{metrics['total_return_pct']:9.2f}% "
                f"{metrics['profit']:11.2f} "
                f"{metrics['n_trades']:12d}"
            )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/strategies/run", methods=["GET"])
def run_strategies():
    """
    Run all strategies and return signals as JSON.

    Query params:
      ticker    (str,   default "AAPL")  ticker symbol
      budget    (float, default 300)     budget in EUR
      period    (str,   default "1mo")   Yahoo Finance period
      interval  (str,   default "1d")    Yahoo Finance interval
      
      VALID_PERIODS   = {'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'}
      VALID_INTERVALS = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'}
    """
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    try:
        budget_eur = float(request.args.get("budget", 300))
    except ValueError:
        abort(400, description="'budget' must be a number.")

    period   = request.args.get("period",   "1mo").strip()
    interval = request.args.get("interval", "1d").strip()

    if period not in TradingStrategy.VALID_PERIODS:
        abort(400, description=(
            f"'period' must be one of: {', '.join(sorted(TradingStrategy.VALID_PERIODS))}"
        ))
    if interval not in TradingStrategy.VALID_INTERVALS:
        abort(400, description=(
            f"'interval' must be one of: {', '.join(sorted(TradingStrategy.VALID_INTERVALS))}"
        ))
    if interval in TradingStrategy._INTERVAL_MAX_DAYS:
        max_d = TradingStrategy._INTERVAL_MAX_DAYS[interval]
        if TradingStrategy._PERIOD_DAYS.get(period, 0) > max_d:
            abort(400, description=(
                f"Interval '{interval}' supports at most {max_d} days of history, "
                f"but period '{period}' requests ~{TradingStrategy._PERIOD_DAYS[period]} days."
            ))

    if not re.match(r'^[A-Z0-9.\-\^=]{1,10}$', ticker):
        abort(400, description="'ticker' must be a valid ticker symbol (letters, digits, . - ^ =).")

    try:
        signals= _run_strategies(ticker, budget_eur, period, interval)
    except Exception as exc:
        abort(500, description=str(exc))

    return jsonify({
        "status": "ok",
        "signals": signals
    })


@app.route("/api/strategies/report", methods=["GET"])
def download_report():
    """
    Run all strategies and stream the generated PDF back to the caller.

    Query params:
      ticker    (str,   default "AAPL")  ticker symbol
      budget    (float, default 300)     budget in EUR
      period    (str,   default "1mo")   Yahoo Finance period
      interval  (str,   default "1d")    Yahoo Finance interval
    """
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    try:
        budget_eur = float(request.args.get("budget", 300))
    except ValueError:
        abort(400, description="'budget' must be a number.")

    period   = request.args.get("period",   "1mo").strip()
    interval = request.args.get("interval", "1d").strip()

    if period not in TradingStrategy.VALID_PERIODS:
        abort(400, description=(
            f"'period' must be one of: {', '.join(sorted(TradingStrategy.VALID_PERIODS))}"
        ))
    if interval not in TradingStrategy.VALID_INTERVALS:
        abort(400, description=(
            f"'interval' must be one of: {', '.join(sorted(TradingStrategy.VALID_INTERVALS))}"
        ))
    if interval in TradingStrategy._INTERVAL_MAX_DAYS:
        max_d = TradingStrategy._INTERVAL_MAX_DAYS[interval]
        if TradingStrategy._PERIOD_DAYS.get(period, 0) > max_d:
            abort(400, description=(
                f"Interval '{interval}' supports at most {max_d} days of history, "
                f"but period '{period}' requests ~{TradingStrategy._PERIOD_DAYS[period]} days."
            ))

    if not re.match(r'^[A-Z0-9.\-\^=]{1,10}$', ticker):
        abort(400, description="'ticker' must be a valid ticker symbol (letters, digits, . - ^ =).")

    try:
        _, pdf_path = _run_strategies(ticker, budget_eur, period, interval)
    except Exception as exc:
        abort(500, description=str(exc))

    if not os.path.isfile(pdf_path):
        abort(500, description="PDF was not generated.")

    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{ticker}_trading_report.pdf",
    )


@app.route('/api/strategies/simulate', methods=['GET'])
def simulate_strategies():
    """
    Simulate multiple strategies on sample stocks and print profits to console.

    Query params:
      tickers         (str,   optional)   comma-separated list, default sample set
      period          (str,   default 6mo)
      interval        (str,   default 1d)
      initial_cash    (float, default 10000)
      commission_bps  (float, default 0)
    """
    period = request.args.get('period', '6mo').strip()
    interval = request.args.get('interval', '1d').strip()

    if period not in TradingStrategy.VALID_PERIODS:
        abort(400, description=(
            f"'period' must be one of: {', '.join(sorted(TradingStrategy.VALID_PERIODS))}"
        ))
    if interval not in TradingStrategy.VALID_INTERVALS:
        abort(400, description=(
            f"'interval' must be one of: {', '.join(sorted(TradingStrategy.VALID_INTERVALS))}"
        ))

    try:
        initial_cash = float(request.args.get('initial_cash', 10000.0))
    except ValueError:
        abort(400, description="'initial_cash' must be a number.")
    if initial_cash <= 0:
        abort(400, description="'initial_cash' must be > 0.")

    try:
        commission_bps = float(request.args.get('commission_bps', 0.0))
    except ValueError:
        abort(400, description="'commission_bps' must be a number.")
    if commission_bps < 0:
        abort(400, description="'commission_bps' must be >= 0.")

    tickers_raw = request.args.get('tickers', '')
    if tickers_raw:
        try:
            tickers = [_validate_ticker(t) for t in tickers_raw.split(',') if t.strip()]
        except ValueError as exc:
            abort(400, description=str(exc))
    else:
        tickers = DEFAULT_SAMPLE_TICKERS

    if not tickers:
        abort(400, description='No valid tickers provided.')

    results = []
    errors = []
    for ticker in tickers:
        try:
            ticker_result = _run_simulation_for_ticker(
                ticker=ticker,
                period=period,
                interval=interval,
                initial_cash=initial_cash,
                commission_bps=commission_bps,
            )
            results.append(ticker_result)
        except Exception as exc:
            errors.append({'ticker': ticker, 'error': str(exc)})

    _print_simulation_summary(results, initial_cash=initial_cash, commission_bps=commission_bps)

    return jsonify({
        'status': 'ok' if results else 'error',
        'assumptions': {
            'period': period,
            'interval': interval,
            'initial_cash': initial_cash,
            'commission_bps': commission_bps,
        },
        'results': results,
        'errors': errors,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
