"""Data models and configuration for the backtesting engine."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BacktestConfig:
    """Configuration options and risk management parameters for a backtest."""

    signal_col: str = 'Signal'
    close_col: str = 'Close'
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    initial_cash: float | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    max_hold_bars: int | None = None

    def validate(self) -> None:
        """
        Validate backtesting constraints and friction parameters.

        Goal:
        -----
        Prevent invalid execution parameters (such as negative costs or zero initial cash)
        from causing mathematical domain errors or unphysical simulation results.

        Execution Principle:
        --------------------
        1. Verifies that transaction fees (`fee_bps`) and slippage (`slippage_bps`) are non-negative.
        2. If `initial_cash` is specified, ensures capital > 0.
        3. If risk boundaries (`stop_loss_pct`, `take_profit_pct`) are specified, ensures percentages > 0.
        4. If `max_hold_bars` is specified, verifies it is a positive integer (> 0).
        5. Raises `ValueError` with informative message if any constraint is violated.

        Raises:
        -------
        ValueError
            If any parameter value is out of bounds.
        """
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be non-negative")
        if self.initial_cash is not None and self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive when provided")
        if self.stop_loss_pct is not None and self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive when provided")
        if self.take_profit_pct is not None and self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive when provided")
        if self.max_hold_bars is not None and self.max_hold_bars <= 0:
            raise ValueError("max_hold_bars must be a positive integer when provided")


@dataclass
class Trade:
    """Record of a single completed trade."""

    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    trade_return: float
    shares: float = 1.0
    exit_reason: str = "signal"


@dataclass
class BacktestResult:
    """Comprehensive performance metrics returned by a backtest run."""

    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    expectancy_pct: float
    n_trades: int
    trade_returns: list[float] = field(default_factory=list)
    skipped_buys_due_to_cash: int = 0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert performance metrics to a standard Python dictionary.

        Goal:
        -----
        Maintain backward-compatible format parity with existing reporting engines,
        JSON API endpoints, and assertions in test suites.

        Execution Principle:
        --------------------
        Extracts all metric attributes into key-value pairs matching the classical
        `backtest_strategy()` dictionary schema.

        Returns:
        --------
        dict[str, Any]
            Dictionary containing metrics including total return, Sharpe ratio, drawdown,
            win rate, trade counts, and trade return arrays.
        """
        return {
            'total_return_pct': self.total_return_pct,
            'annualized_return_pct': self.annualized_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'win_rate_pct': self.win_rate_pct,
            'profit_factor': self.profit_factor,
            'expectancy_pct': self.expectancy_pct,
            'n_trades': self.n_trades,
            'trade_returns': self.trade_returns,
            'skipped_buys_due_to_cash': self.skipped_buys_due_to_cash,
        }
