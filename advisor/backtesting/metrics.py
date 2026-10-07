"""Financial performance and risk metric calculators."""

import numpy as np

from advisor.backtesting.models import BacktestResult


def calculate_metrics(
    equity: np.ndarray,
    daily_rets: np.ndarray,
    trade_returns: list[float],
    n_bars: int,
    total_return_pct: float,
    skipped_buys: int = 0,
) -> BacktestResult:
    """
    Compute comprehensive portfolio performance, risk-adjusted returns, and trade statistics.

    Goal:
    -----
    Evaluate the quality, profitability, and risk profile of a strategy backtest run,
    producing standard quantitative metrics (CAGR, Sharpe, Max Drawdown, Win Rate,
    Profit Factor, Expectancy) from the generated equity path and round-trip trades.

    Execution Principle:
    --------------------
    1. Annualized Return (CAGR):
       - Assumes 252 trading days per calendar year (`n_years = n_bars / 252.0`).
       - Calculates CAGR as `(final_equity ** (1.0 / n_years) - 1.0) * 100.0`.
       - If equity non-positive or duration 0, defaults to 0.0%.
    2. Annualized Sharpe Ratio:
       - Computes standard deviation of bar-to-bar returns `std_dr = np.std(daily_rets)`.
       - Assuming zero risk-free rate: `(mean(daily_rets) / std_dr) * sqrt(252.0)`.
       - If volatility is zero (e.g. all hold bars), Sharpe is 0.0.
    3. Maximum Drawdown (MDD):
       - Computes running peak of equity curve: `peak = np.maximum.accumulate(equity)`.
       - Evaluates worst peak-to-trough decline: `((equity - peak) / peak).min() * 100.0`.
       - Result is always non-positive (≤ 0.0%).
    4. Win Rate & Trade Expectancy:
       - `win_rate = (winning_trades / n_trades) * 100.0`.
       - `expectancy_pct = (sum(trade_returns) / n_trades) * 100.0`.
    5. Profit Factor:
       - Sums all positive returns (`gross_profit`) and absolute sum of negative returns (`gross_loss`).
       - If `gross_loss > 0`: `gross_profit / gross_loss`.
       - If `gross_loss == 0` and `gross_profit > 0`: `float('inf')`.
       - Otherwise: `0.0`.
    6. Returns an assembled `BacktestResult` dataclass with standard rounding.

    Parameters:
    -----------
    equity : np.ndarray
        Array representing cumulative portfolio equity over time.
    daily_rets : np.ndarray
        Array of percentage returns per bar.
    trade_returns : list[float]
        Individual percentage returns for completed round-trip trades.
    n_bars : int
        Total bar count of the backtest sample.
    total_return_pct : float
        Cumulative strategy return percentage.
    skipped_buys : int, default 0
        Count of buy signals skipped due to insufficient purchasing cash.

    Returns:
    --------
    BacktestResult
        Dataclass containing formatted and rounded performance statistics.
    """
    n_years = n_bars / 252.0
    final_equity = equity[-1] if len(equity) > 0 else 1.0

    annualized_return = (
        (final_equity ** (1.0 / n_years) - 1.0) * 100.0
        if final_equity > 0 and n_years > 0
        else 0.0
    )

    std_dr = np.std(daily_rets)
    sharpe = (float(np.mean(daily_rets)) / std_dr * np.sqrt(252.0)) if std_dr > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min()) * 100.0

    n_trades = len(trade_returns)
    win_rate = (
        sum(1 for r in trade_returns if r > 0) / n_trades * 100.0
        if n_trades > 0
        else 0.0
    )

    gross_profit = sum(r for r in trade_returns if r > 0)
    gross_loss = abs(sum(r for r in trade_returns if r < 0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float('inf')
    else:
        profit_factor = 0.0

    expectancy_pct = (sum(trade_returns) / n_trades * 100.0) if n_trades > 0 else 0.0

    return BacktestResult(
        total_return_pct=round(total_return_pct, 2),
        annualized_return_pct=round(annualized_return, 2),
        sharpe_ratio=round(sharpe, 3),
        max_drawdown_pct=round(max_dd, 2),
        win_rate_pct=round(win_rate, 1),
        profit_factor=(round(profit_factor, 3) if np.isfinite(profit_factor) else float('inf')),
        expectancy_pct=round(expectancy_pct, 2),
        n_trades=n_trades,
        trade_returns=trade_returns,
        skipped_buys_due_to_cash=skipped_buys,
    )
