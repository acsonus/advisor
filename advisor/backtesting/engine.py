"""Backtesting simulation engine."""

import numpy as np
import pandas as pd

from advisor.backtesting.metrics import calculate_metrics
from advisor.backtesting.models import BacktestConfig, BacktestResult


class BacktestEngine:
    """Simulates trading strategy execution on historical OHLCV data."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.config.validate()

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Run backtest simulation on data containing signals and close prices."""
        cfg = self.config
        prices = df[cfg.close_col].to_numpy(dtype=float)
        signals = df[cfg.signal_col].to_numpy()
        n = len(prices)

        one_way_cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10000.0

        if n < 2:
            return BacktestResult(
                total_return_pct=0.0,
                annualized_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                expectancy_pct=0.0,
                n_trades=0,
                trade_returns=[],
                skipped_buys_due_to_cash=0,
            )

        if cfg.initial_cash is not None:
            return self._run_cash_constrained(
                prices, signals, n, one_way_cost_rate, float(cfg.initial_cash)
            )
        else:
            return self._run_legacy_unit(prices, signals, n, one_way_cost_rate)

    def _run_cash_constrained(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        n: int,
        one_way_cost_rate: float,
        initial_cash: float,
    ) -> BacktestResult:
        cfg = self.config
        cash = float(initial_cash)
        shares = 0.0
        in_position = False
        entry_total_cost = 0.0
        entry_price_raw = 0.0
        entry_index = -1
        equity = np.zeros(n, dtype=float)
        trade_returns: list[float] = []
        skipped_buys_due_to_cash = 0

        for i in range(n):
            price = prices[i]

            if in_position:
                bars_held = i - entry_index
                gross_ret = (price / entry_price_raw) - 1.0
                risk_exit = False

                if cfg.stop_loss_pct is not None and gross_ret <= -cfg.stop_loss_pct:
                    risk_exit = True
                elif cfg.take_profit_pct is not None and gross_ret >= cfg.take_profit_pct:
                    risk_exit = True
                elif cfg.max_hold_bars is not None and bars_held >= cfg.max_hold_bars:
                    risk_exit = True

                if risk_exit:
                    exit_value = shares * price * (1.0 - one_way_cost_rate)
                    trade_ret = (
                        (exit_value - entry_total_cost) / entry_total_cost
                        if entry_total_cost > 0
                        else 0.0
                    )
                    trade_returns.append(trade_ret)
                    cash += exit_value
                    shares = 0.0
                    in_position = False
                    entry_total_cost = 0.0
                    entry_price_raw = 0.0
                    entry_index = -1

            if signals[i] == 'Buy' and not in_position and price > 0:
                per_share_cost = price * (1.0 + one_way_cost_rate)
                max_shares = int(cash // per_share_cost)
                if max_shares >= 1:
                    shares = float(max_shares)
                    entry_total_cost = shares * per_share_cost
                    cash -= entry_total_cost
                    in_position = True
                    entry_price_raw = price
                    entry_index = i
                else:
                    skipped_buys_due_to_cash += 1
            elif signals[i] == 'Sell' and in_position and price > 0:
                exit_value = shares * price * (1.0 - one_way_cost_rate)
                trade_ret = (
                    (exit_value - entry_total_cost) / entry_total_cost
                    if entry_total_cost > 0
                    else 0.0
                )
                trade_returns.append(trade_ret)
                cash += exit_value
                shares = 0.0
                in_position = False
                entry_total_cost = 0.0
                entry_price_raw = 0.0
                entry_index = -1

            equity[i] = cash + shares * price

        if in_position:
            exit_value = shares * prices[-1] * (1.0 - one_way_cost_rate)
            trade_ret = (
                (exit_value - entry_total_cost) / entry_total_cost
                if entry_total_cost > 0
                else 0.0
            )
            trade_returns.append(trade_ret)
            cash += exit_value
            equity[-1] = cash

        daily_rets = np.zeros(n, dtype=float)
        daily_rets[1:] = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1e-12)
        total_return = ((equity[-1] / initial_cash) - 1.0) * 100.0

        return calculate_metrics(
            equity=equity,
            daily_rets=daily_rets,
            trade_returns=trade_returns,
            n_bars=n,
            total_return_pct=total_return,
            skipped_buys=skipped_buys_due_to_cash,
        )

    def _run_legacy_unit(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        n: int,
        one_way_cost_rate: float,
    ) -> BacktestResult:
        cfg = self.config
        in_position = False
        entry_price_raw = 0.0
        entry_price_exec = 0.0
        entry_index = -1
        positions = np.zeros(n)
        cost_rets = np.zeros(n)
        trade_returns: list[float] = []

        for i in range(n):
            if in_position:
                bars_held = i - entry_index
                gross_ret = (prices[i] / entry_price_raw) - 1.0
                risk_exit = False

                if cfg.stop_loss_pct is not None and gross_ret <= -cfg.stop_loss_pct:
                    risk_exit = True
                elif cfg.take_profit_pct is not None and gross_ret >= cfg.take_profit_pct:
                    risk_exit = True
                elif cfg.max_hold_bars is not None and bars_held >= cfg.max_hold_bars:
                    risk_exit = True

                if risk_exit:
                    exit_price_exec = prices[i] * (1.0 - one_way_cost_rate)
                    trade_returns.append((exit_price_exec - entry_price_exec) / entry_price_exec)
                    in_position = False
                    positions[i] = 1.0
                    cost_rets[i] += one_way_cost_rate
                    continue

            if signals[i] == 'Buy' and not in_position:
                in_position = True
                entry_price_raw = prices[i]
                entry_price_exec = prices[i] * (1.0 + one_way_cost_rate)
                entry_index = i
                positions[i] = 0.0
                cost_rets[i] += one_way_cost_rate
            elif signals[i] == 'Sell' and in_position:
                in_position = False
                exit_price_exec = prices[i] * (1.0 - one_way_cost_rate)
                trade_returns.append((exit_price_exec - entry_price_exec) / entry_price_exec)
                positions[i] = 1.0
                cost_rets[i] += one_way_cost_rate
            elif in_position:
                positions[i] = 1.0

        if in_position:
            exit_price_exec = prices[-1] * (1.0 - one_way_cost_rate)
            trade_returns.append((exit_price_exec - entry_price_exec) / entry_price_exec)
            cost_rets[-1] += one_way_cost_rate

        price_rets = np.concatenate(
            [[0.0], np.diff(prices) / np.where(prices[:-1] != 0, prices[:-1], 1e-12)]
        )
        daily_rets = positions * price_rets - cost_rets
        equity = np.cumprod(1.0 + daily_rets)
        total_return = (equity[-1] - 1.0) * 100.0

        return calculate_metrics(
            equity=equity,
            daily_rets=daily_rets,
            trade_returns=trade_returns,
            n_bars=n,
            total_return_pct=total_return,
            skipped_buys=0,
        )


def backtest_strategy(
    df: pd.DataFrame,
    signal_col: str = 'Signal',
    close_col: str = 'Close',
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    initial_cash: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    max_hold_bars: int | None = None,
) -> dict:
    """Backward-compatible function wrapper for the backtest engine."""
    config = BacktestConfig(
        signal_col=signal_col,
        close_col=close_col,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_cash=initial_cash,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_hold_bars=max_hold_bars,
    )
    result = BacktestEngine(config).run(df)
    return result.to_dict()
