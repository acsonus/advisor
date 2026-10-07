"""Flask Blueprint routes for trading strategies."""

import os
from flask import Blueprint, abort, jsonify, request, send_file

from advisor.data.constants import INTERVAL_MAX_DAYS, PERIOD_DAYS, VALID_INTERVALS, VALID_PERIODS
from advisor.data.validator import validate_period_and_interval, validate_ticker
from advisor.services.orchestrator import (
    DEFAULT_SAMPLE_TICKERS,
    run_simulation_for_ticker,
    run_strategies_pipeline,
)

strategies_bp = Blueprint("strategies", __name__, url_prefix="/api/strategies")


def _validate_request_params():
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
    """Run all strategies and return signals as JSON."""
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
    """Run all strategies and stream the generated PDF back to the caller."""
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
    """Simulate multiple strategies on sample stocks and return results."""
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
        },
        'results': results,
        'errors': errors,
    })
