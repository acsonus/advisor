"""Optimization objectives, composite scoring, and constraint penalties."""

from dataclasses import dataclass
from advisor.backtesting.models import BacktestResult


@dataclass
class OptimizationObjective:
    """
    Evaluates backtest performance with a composite fitness function and risk penalties.
    """

    max_drawdown_cap_pct: float = 20.0
    min_trades: int = 3
    sharpe_weight: float = 0.6
    profit_factor_weight: float = 0.4

    def compute_fitness(self, metrics: BacktestResult | dict) -> float:
        """
        Calculate scalar fitness score. Higher is better.

        Goal:
        -----
        Balance risk-adjusted return (Sharpe) and payoff consistency (Profit Factor)
        while strictly rejecting parameter sets that blow through drawdown limits
        or fail to execute enough trades to be statistically valid.
        """
        if isinstance(metrics, dict):
            sharpe = float(metrics.get("sharpe_ratio", 0.0))
            drawdown = float(metrics.get("max_drawdown_pct", 0.0))
            profit_factor = float(metrics.get("profit_factor", 0.0))
            total_return = float(metrics.get("total_return_pct", 0.0))
            n_trades = int(metrics.get("n_trades", 0))
        else:
            sharpe = metrics.sharpe_ratio
            drawdown = metrics.max_drawdown_pct
            profit_factor = metrics.profit_factor
            total_return = metrics.total_return_pct
            n_trades = metrics.n_trades

        # Rule 1: Minimum trade count gate
        if n_trades < self.min_trades:
            return -100.0 - (self.min_trades - n_trades) * 10.0

        # Rule 2: Hard drawdown cap violation penalty
        if drawdown > self.max_drawdown_cap_pct:
            excess_dd = drawdown - self.max_drawdown_cap_pct
            return -50.0 - excess_dd * 5.0

        # Rule 3: Negative total return penalty
        if total_return <= 0.0:
            return -10.0 + total_return / 10.0

        # Rule 4: Payoff score with capped profit factor (to prevent 1-trade infinity)
        capped_pf = min(profit_factor, 6.0)

        # Composite score
        fitness = (sharpe * self.sharpe_weight) + (capped_pf * self.profit_factor_weight)
        return round(fitness, 4)
