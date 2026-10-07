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
    Compute standard performance metrics from equity curve and trade returns.

    Parameters
    ----------
    equity : np.ndarray
        Equity curve over the simulation.
    daily_rets : np.ndarray
        Bar-to-bar returns.
    trade_returns : list[float]
        Returns of individual completed trades.
    n_bars : int
        Total number of bars.
    total_return_pct : float
        Total cumulative percentage return.
    skipped_buys : int
        Number of buy signals skipped due to insufficient cash.
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
