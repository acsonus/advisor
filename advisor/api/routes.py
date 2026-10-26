"""Flask Blueprint routes for trading strategies."""

import os
from flask import Blueprint, abort, jsonify, request, send_file

from advisor.data.validator import validate_period_and_interval, validate_ticker
from advisor.services.orchestrator import (
    DEFAULT_SAMPLE_TICKERS,
    optimize_strategy_for_ticker,
    run_simulation_for_ticker,
    run_strategies_pipeline,
)
from advisor.strategies.registry import StrategyRegistry

strategies_bp = Blueprint("strategies", __name__, url_prefix="/api/strategies")


def _validate_request_params() -> tuple[str, float, str, str]:
    """
    Extract and validate common HTTP query parameters for strategy runs.

    Goal:
    -----
    Normalize and validate incoming GET query string parameters (ticker, budget,
    period, interval), aborting early with HTTP 400 Bad Request on invalid inputs.

    Execution Principle:
    --------------------
    1. Extracts 'ticker' (default 'AAPL') and sanitizes via `validate_ticker()`.
    2. Parses 'budget' (default 300) into a float, aborting with 400 if malformed.
    3. Extracts 'period' (default '3mo') and 'interval' (default '1d').
    4. Validates period and interval combinations via `validate_period_and_interval()`.
    5. Returns sanitized `(ticker, budget_eur, period, interval)` tuple.

    Returns:
    --------
    tuple[str, float, str, str]
        Sanitized request parameters.
    """
    ticker_raw = request.args.get("ticker", "AAPL")
    try:
        ticker = validate_ticker(ticker_raw)
    except ValueError as exc:
        abort(400, description=str(exc))

    try:
        budget_eur = float(request.args.get("budget", 300))
    except ValueError:
        abort(400, description="'budget' must be a number.")

    period = request.args.get("period", "3mo").strip()
    interval = request.args.get("interval", "1d").strip()

    try:
        validate_period_and_interval(period, interval)
    except ValueError as exc:
        abort(400, description=str(exc))

    return ticker, budget_eur, period, interval


@strategies_bp.route("/run", methods=["GET"])
def run_strategies():
    """
    HTTP endpoint to execute strategies and return signals as JSON.

    Goal:
    -----
    Expose a REST endpoint (`GET /api/strategies/run`) allowing frontend dashboards
    and external consumers to retrieve current strategy signals and sentiment scores
    for any supported ticker symbol.

    Execution Principle:
    --------------------
    1. Validates query parameters using `_validate_request_params()`.
    2. Executes the core advisory service workflow: `run_strategies_pipeline()`.
    3. Serializes the generated signal summary into a JSON payload with status 'ok'.
    4. Catches unhandled service exceptions and aborts with HTTP 500.

    Returns:
    --------
    flask.Response
        JSON response containing strategy signals and status.
    """
    ticker, budget_eur, period, interval = _validate_request_params()

    try:
        signals, _ = run_strategies_pipeline(ticker, budget_eur, period, interval)
    except Exception as exc:
        abort(500, description=str(exc))

    return jsonify({
        "status": "ok",
        "signals": signals,
    })


@strategies_bp.route("/report", methods=["GET"])
def download_report():
    """
    HTTP endpoint to execute strategies and stream the generated PDF report.

    Goal:
    -----
    Expose a download endpoint (`GET /api/strategies/report`) allowing users to generate
    and directly download an executive PDF trading report attachment.

    Execution Principle:
    --------------------
    1. Validates parameters with `_validate_request_params()`.
    2. Runs `run_strategies_pipeline()`, which writes a compiled PDF report to disk.
    3. Verifies that the PDF file exists on the filesystem.
    4. Streams the file to the caller using `send_file()` with `mimetype='application/pdf'`,
       specifying `as_attachment=True` and formatted attachment filename.

    Returns:
    --------
    flask.Response
        Streaming binary PDF download response.
    """
    ticker, budget_eur, period, interval = _validate_request_params()

    try:
        _, pdf_path = run_strategies_pipeline(ticker, budget_eur, period, interval)
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


@strategies_bp.route("/simulate", methods=["GET"])
def simulate_strategies():
    """
    HTTP endpoint to run multi-strategy backtest simulations on sample or custom tickers.

    Goal:
    -----
    Expose a simulation endpoint (`GET /api/strategies/simulate`) to benchmark multiple
    strategies simultaneously across a basket of assets.

    Execution Principle:
    --------------------
    1. Extracts and validates period, interval, initial cash, and commission basis points.
    2. Parses comma-separated ticker symbols, defaulting to `DEFAULT_SAMPLE_TICKERS` if omitted.
    3. Iterates over each ticker, executing `run_simulation_for_ticker()`.
    4. Gathers successfully completed simulation metrics and isolates failures into an error array.
    5. Returns an aggregated JSON report containing assumptions, per-ticker results, and error logs.

    Returns:
    --------
    flask.Response
        JSON response with simulation performance across requested symbols.
    """
    period = request.args.get('period', '6mo').strip()
    interval = request.args.get('interval', '1d').strip()

    try:
        validate_period_and_interval(period, interval)
    except ValueError as exc:
        abort(400, description=str(exc))

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
            tickers = [validate_ticker(t) for t in tickers_raw.split(',') if t.strip()]
        except ValueError as exc:
            abort(400, description=str(exc))
    else:
        tickers = DEFAULT_SAMPLE_TICKERS

    if not tickers:
        abort(400, description='No valid tickers provided.')

    use_optimal = request.args.get('use_optimal', 'false').strip().lower() in ('true', '1', 'yes')

    results = []
    errors = []
    for ticker in tickers:
        try:
            ticker_result = run_simulation_for_ticker(
                ticker=ticker,
                period=period,
                interval=interval,
                initial_cash=initial_cash,
                commission_bps=commission_bps,
                use_optimal_params=use_optimal,
            )
            results.append(ticker_result)
        except Exception as exc:
            errors.append({'ticker': ticker, 'error': str(exc)})

    return jsonify({
        'status': 'ok' if results else 'error',
        'assumptions': {
            'period': period,
            'interval': interval,
            'initial_cash': initial_cash,
            'commission_bps': commission_bps,
            'use_optimal': use_optimal,
        },
        'results': results,
        'errors': errors,
    })


@strategies_bp.route("/optimize", methods=["GET", "POST"])
def optimize_strategy():
    """
    HTTP endpoint to run systematic parameter optimization and risk tuning.

    Goal:
    -----
    Expose a REST endpoint (`/api/strategies/optimize`) enabling automated hyperparameter
    and exit tuning (stop-loss, take-profit, trailing stop) for any registered strategy.

    Query/JSON Parameters:
    ----------------------
    - ticker: str (default 'AAPL')
    - strategy: str (default 'ma_rsi')
    - period: str (default '1y')
    - interval: str (default '1d')
    - iterations / n_iter: int (default 50)
    - initial_cash: float (default 10000.0)
    - commission_bps: float (default 0.0)
    - save_params: bool (default True)
    """
    # Accept params from JSON payload or query parameters
    payload = request.get_json(silent=True) or {}

    ticker_raw = payload.get("ticker", request.args.get("ticker", "AAPL"))
    try:
        ticker = validate_ticker(ticker_raw)
    except ValueError as exc:
        abort(400, description=str(exc))

    strategy_name = payload.get("strategy", request.args.get("strategy", "ma_rsi")).strip()
    available_strategies = StrategyRegistry.list_available()
    if strategy_name not in available_strategies:
        abort(400, description=f"Unknown strategy '{strategy_name}'. Available: {available_strategies}")

    period = payload.get("period", request.args.get("period", "1y")).strip()
    interval = payload.get("interval", request.args.get("interval", "1d")).strip()
    try:
        validate_period_and_interval(period, interval)
    except ValueError as exc:
        abort(400, description=str(exc))

    n_iter_raw = payload.get("n_iter", payload.get("iterations", request.args.get("iterations", request.args.get("n_iter", 50))))
    try:
        n_iter = int(n_iter_raw)
        if n_iter < 1:
            raise ValueError
    except ValueError:
        abort(400, description="'iterations' must be a positive integer.")

    initial_cash_raw = payload.get("initial_cash", request.args.get("initial_cash", 10000.0))
    try:
        initial_cash = float(initial_cash_raw)
        if initial_cash <= 0:
            raise ValueError
    except ValueError:
        abort(400, description="'initial_cash' must be > 0.")

    comm_raw = payload.get("commission_bps", request.args.get("commission_bps", 0.0))
    try:
        commission_bps = float(comm_raw)
        if commission_bps < 0:
            raise ValueError
    except ValueError:
        abort(400, description="'commission_bps' must be >= 0.")

    save_params_raw = payload.get("save_params", request.args.get("save_params", "true"))
    if isinstance(save_params_raw, bool):
        save_params = save_params_raw
    else:
        save_params = str(save_params_raw).strip().lower() in ("true", "1", "yes")

    try:
        report = optimize_strategy_for_ticker(
            ticker=ticker,
            strategy_name=strategy_name,
            period=period,
            interval=interval,
            n_iter=n_iter,
            initial_cash=initial_cash,
            commission_bps=commission_bps,
            save_params=save_params,
        )
    except Exception as exc:
        abort(500, description=str(exc))

    return jsonify({
        "status": "ok",
        "optimization": report,
    })
