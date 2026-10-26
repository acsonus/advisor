#!/usr/bin/env python3
"""
CLI runner for systematic trading strategy parameter and risk limit optimization.

Usage examples:
---------------
python optimize_strategy.py --ticker AAPL --strategy ma_rsi --period 1y -n 50
python optimize_strategy.py --ticker MSFT --strategy macd_histogram_reversal --period 2y -n 100
python optimize_strategy.py --ticker TSLA --strategy bollinger_squeeze --period 1y -n 50
"""

import argparse
import json
import sys

from pathlib import Path
from advisor.data.loader import load_historical_dump
from advisor.data.validator import validate_period_and_interval, validate_ticker
from advisor.services.orchestrator import (
    optimize_strategy_for_ticker,
    optimize_strategy_on_dataframe,
)
from advisor.strategies.registry import StrategyRegistry


def format_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:+.2f}%"


def format_num(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Systematically tune trading strategy parameters and risk limits with train/test validation."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (e.g. AAPL, EURUSD, BTCUSD). Default: AAPL.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to historical market data dump (MetaTrader 4/5 CSV/TXT, TradingView, or generic CSV).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "metatrader", "generic"],
        help="Data dump format. Default: auto (detects MetaTrader <DATE> or standard CSV).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ma_rsi",
        choices=StrategyRegistry.list_available(),
        help="Strategy to optimize. Default: ma_rsi.",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1y",
        help="Historical duration to download from Yahoo if --data-file is omitted. Default: 1y.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Bar resolution (e.g. 1d, 60m). Default: 1d.",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=50,
        help="Number of randomized parameter configurations to test. Default: 50.",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=10000.0,
        help="Starting capital for simulation. Default: 10000.0.",
    )
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=0.0,
        help="One-way commission in basis points. Default: 0.0.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist optimal parameters to optimal_params.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible parameter sampling. Default: 42.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON result to stdout instead of formatted table.",
    )

    args = parser.parse_args()

    if args.data_file:
        file_path = Path(args.data_file)
        if not file_path.is_file():
            print(f"Error: Data file not found: {file_path}", file=sys.stderr)
            sys.exit(1)

        ticker = args.ticker if args.ticker != "AAPL" else file_path.stem.upper()
        print(f"\n[+] Loading MetaTrader / platform dump from {file_path} (format={args.format})...")
        try:
            data = load_historical_dump(file_path, source=args.format)
        except Exception as exc:
            print(f"Error loading historical dump: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"    Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        print(f"[+] Running optimization for {ticker} ({args.strategy})...")
        print(f"    Search budget: {args.iterations} trials, capital: ${args.cash:,.2f}")

        try:
            report = optimize_strategy_on_dataframe(
                data=data,
                ticker=ticker,
                strategy_name=args.strategy,
                n_iter=args.iterations,
                initial_cash=args.cash,
                commission_bps=args.commission_bps,
                save_params=not args.no_save,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"\n[!] Optimization failed: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            ticker = validate_ticker(args.ticker)
            validate_period_and_interval(args.period, args.interval)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"\n[+] Running optimization for {ticker} ({args.strategy})...")
        print(f"    Timeframe: period={args.period}, interval={args.interval}")
        print(f"    Search budget: {args.iterations} trials, capital: ${args.cash:,.2f}")

        try:
            report = optimize_strategy_for_ticker(
                ticker=ticker,
                strategy_name=args.strategy,
                period=args.period,
                interval=args.interval,
                n_iter=args.iterations,
                initial_cash=args.cash,
                commission_bps=args.commission_bps,
                save_params=not args.no_save,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"\n[!] Optimization failed: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("\n" + "=" * 80)
    print(f"  OPTIMIZATION REPORT: {ticker} - {args.strategy.upper()}")
    print("=" * 80)
    print(f"Bars: Total={report['bars_total']}, Train={report['bars_train']}, Test (Holdout)={report['bars_test']}")
    print(f"Out-of-Sample Validated: {'YES (Holdout Profitable/Beats Baseline)' if report['out_of_sample_verified'] else 'NO (Caution: Overfitting Risk)'}")
    if report.get("saved_to"):
        print(f"Saved optimal parameters to: {report['saved_to']}")

    print("\n--- WINNING PARAMETERS ---")
    print(f"  Strategy Parameters: {report['best_strategy_params']}")
    print(f"  Risk Parameters:     {report['best_risk_params']}")
    print(f"  Composite Fitness:   {report['best_fitness']:.4f}")

    print("\n--- PERFORMANCE COMPARISON ---")
    header = f"{'Metric':<25} {'Baseline (Train)':<18} {'Optimized (Train)':<18} {'Baseline (Test)':<18} {'Optimized (Test)':<18}"
    print(header)
    print("-" * len(header))

    b_tr = report["baseline_in_sample"]
    o_tr = report["optimized_in_sample"]
    b_te = report["baseline_out_of_sample"]
    o_te = report["optimized_out_of_sample"]

    rows = [
        ("Total Return", format_pct(b_tr["total_return_pct"]), format_pct(o_tr["total_return_pct"]), format_pct(b_te["total_return_pct"]), format_pct(o_te["total_return_pct"])),
        ("Sharpe Ratio", format_num(b_tr["sharpe_ratio"]), format_num(o_tr["sharpe_ratio"]), format_num(b_te["sharpe_ratio"]), format_num(o_te["sharpe_ratio"])),
        ("Max Drawdown", format_pct(b_tr["max_drawdown_pct"]), format_pct(o_tr["max_drawdown_pct"]), format_pct(b_te["max_drawdown_pct"]), format_pct(o_te["max_drawdown_pct"])),
        ("Win Rate", format_pct(b_tr["win_rate_pct"]), format_pct(o_tr["win_rate_pct"]), format_pct(b_te["win_rate_pct"]), format_pct(o_te["win_rate_pct"])),
        ("Profit Factor", format_num(b_tr["profit_factor"]), format_num(o_tr["profit_factor"]), format_num(b_te["profit_factor"]), format_num(o_te["profit_factor"])),
        ("Trade Count", str(b_tr["n_trades"]), str(o_tr["n_trades"]), str(b_te["n_trades"]), str(o_te["n_trades"])),
    ]

    for label, b1, o1, b2, o2 in rows:
        print(f"{label:<25} {b1:<18} {o1:<18} {b2:<18} {o2:<18}")

    print("\n--- TOP CANDIDATE TRIALS (IN-SAMPLE) ---")
    for t in report.get("top_trials", []):
        print(f"  Rank #{t['rank']}: Fitness={t['fitness']:.3f} | Ret={format_pct(t['total_return_pct'])} | Sharpe={format_num(t['sharpe_ratio'])} | DD={format_pct(t['max_drawdown_pct'])} | Trades={t['n_trades']}")
        print(f"         Params: {t['strategy_params']} | Risk: {t['risk_params']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
