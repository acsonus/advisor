"""Backtesting simulation engine."""

import numpy as np
import pandas as pd

from advisor.backtesting.metrics import calculate_metrics
from advisor.backtesting.models import BacktestConfig, BacktestResult


class BacktestEngine:
    """
    Simulates trading strategy execution on historical OHLCV data.

    Executes trades bar-by-bar using either realistic cash-constrained portfolio sizing
    or legacy unit-share return tracking, taking into account fees, slippage, and
    risk management constraints (stop-loss, take-profit, max holding duration).
    """

    def __init__(self, config: BacktestConfig | None = None):
        """
        Initialize the BacktestEngine with execution and risk parameters.

        Goal:
        -----
        Store simulation configuration and validate all constraint parameters before execution.

        Execution Principle:
        --------------------
        1. Assigns `config` or instantiates default `BacktestConfig()`.
        2. Calls `self.config.validate()` to guarantee non-negative costs and positive thresholds.

        Parameters:
        -----------
        config : BacktestConfig, optional
            Backtesting configuration settings.
        """
        self.config = config or BacktestConfig()
        self.config.validate()

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Execute backtest simulation across the supplied signal and price dataset.

        Goal:
        -----
        Route data into the appropriate execution engine mode (cash-constrained share-sizing
        versus legacy 1-unit mode) and produce complete performance metrics.

        Execution Principle:
        --------------------
        1. Extract numpy arrays for close prices (`close_col`) and signals (`signal_col`).
        2. Calculate one-way combined friction rate: `one_way_cost_rate = (fee_bps + slippage_bps) / 10000.0`.
        3. If sample length < 2 bars, short-circuit and return a null `BacktestResult` with zero metrics.
        4. If `initial_cash` is provided:
           Delegates to `_run_cash_constrained()` where purchases are sized by available liquidity.
        5. If `initial_cash` is None:
           Delegates to `_run_legacy_unit()` simulating 1-unit positions for backward compatibility.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing prices (`close_col`) and trading signals (`signal_col`).

        Returns:
        --------
        BacktestResult
            Computed simulation performance results and metrics.
        """
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

        atr_values = None
        if cfg.atr_col and cfg.atr_col in df.columns:
            atr_values = df[cfg.atr_col].to_numpy(dtype=float)
        elif (cfg.stop_loss_atr_mult is not None or cfg.take_profit_atr_mult is not None) and all(
            c in df.columns for c in ('High', 'Low', 'Close')
        ):
            high = df['High'].to_numpy(dtype=float)
            low = df['Low'].to_numpy(dtype=float)
            close = df['Close'].to_numpy(dtype=float)
            tr1 = high - low
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.zeros(n, dtype=float)
            tr[0] = tr1[0]
            tr[1:] = np.maximum(tr1[1:], np.maximum(tr2, tr3))
            atr_values = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy(dtype=float)

        if cfg.initial_cash is not None:
            return self._run_cash_constrained(
                prices, signals, n, one_way_cost_rate, float(cfg.initial_cash), atr_values=atr_values
            )
        else:
            return self._run_legacy_unit(prices, signals, n, one_way_cost_rate, atr_values=atr_values)

    def _run_cash_constrained(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        n: int,
        one_way_cost_rate: float,
        initial_cash: float,
        atr_values: np.ndarray | None = None,
    ) -> BacktestResult:
        """
        Simulate realistic cash-constrained execution where position size depends on capital.

        Goal:
        -----
        Model cash balance depletion, dynamic share allocation, cash-drag on portfolio equity,
        and transaction friction costs on each trade entry and exit.

        Execution Principle:
        --------------------
        1. Initialize cash account with `initial_cash`, position `shares = 0.0`, and `in_position = False`.
        2. Iterate bar-by-bar across all price bars:
           a. If in a position, evaluate risk management exits:
              - `gross_ret = (price / entry_price_raw) - 1.0`.
              - Stop loss exit: if `gross_ret <= -stop_loss_pct`.
              - Take profit exit: if `gross_ret >= take_profit_pct`.
              - Max holding period exit: if `bars_held >= max_hold_bars`.
              - On risk exit: liquidate shares, deduct exit transaction costs, compute net trade return,
                credit cash account, and reset position tracking.
           b. Signal processing:
              - If signal is 'Buy' and flat: compute maximum integer shares affordable from current cash
                including entry costs (`per_share_cost = price * (1 + cost_rate)`). If `max_shares >= 1`,
                deduct cost from cash and enter long; otherwise increment `skipped_buys_due_to_cash`.
              - If signal is 'Sell' and in position: liquidate shares net of exit fees, record trade return,
                credit cash, and reset position.
           c. Mark-to-market: record total portfolio equity at bar close (`equity[i] = cash + shares * price`).
        3. Force-close any open position on final bar at `prices[-1]`.
        4. Calculate bar-to-bar returns: `daily_rets = diff(equity) / equity[:-1]`.
        5. Forward results to `calculate_metrics()` to compute CAGR, Sharpe, drawdown, and win rate.

        Parameters:
        -----------
        prices : np.ndarray
            Close price array.
        signals : np.ndarray
            Signal string array ('Buy', 'Sell', 'Hold').
        n : int
            Bar count.
        one_way_cost_rate : float
            Fractional cost per transaction leg (fees + slippage).
        initial_cash : float
            Starting balance in currency units.

        Returns:
        --------
        BacktestResult
            Calculated backtest metrics.
        """
        cfg = self.config
        cash = float(initial_cash)
        shares = 0.0
        in_position = False
        entry_total_cost = 0.0
        entry_price_raw = 0.0
        peak_price = 0.0
        stop_price = None
        target_price = None
        entry_index = -1
        equity = np.zeros(n, dtype=float)
        trade_returns: list[float] = []
        skipped_buys_due_to_cash = 0

        for i in range(n):
            price = prices[i]

            if in_position:
                bars_held = i - entry_index
                gross_ret = (price / entry_price_raw) - 1.0
                peak_price = max(peak_price, price)
                risk_exit = False

                if cfg.trailing_stop:
                    trail_pct = (
                        cfg.trailing_stop_pct
                        if cfg.trailing_stop_pct is not None
                        else cfg.stop_loss_pct
                    )
                    if trail_pct is not None and ((price / peak_price) - 1.0) <= -trail_pct:
                        risk_exit = True
                elif cfg.stop_loss_pct is not None and gross_ret <= -cfg.stop_loss_pct:
                    risk_exit = True
                elif stop_price is not None and price <= stop_price:
                    risk_exit = True

                if cfg.take_profit_pct is not None and gross_ret >= cfg.take_profit_pct:
                    risk_exit = True
                elif target_price is not None and price >= target_price:
                    risk_exit = True

                if cfg.max_hold_bars is not None and bars_held >= cfg.max_hold_bars:
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
                    peak_price = 0.0
                    stop_price = None
                    target_price = None
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
                    peak_price = price
                    entry_index = i
                    stop_price = None
                    target_price = None
                    if atr_values is not None:
                        curr_atr = atr_values[i]
                        if cfg.stop_loss_atr_mult is not None:
                            stop_price = price - (cfg.stop_loss_atr_mult * curr_atr)
                        if cfg.take_profit_atr_mult is not None:
                            target_price = price + (cfg.take_profit_atr_mult * curr_atr)
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
                peak_price = 0.0
                stop_price = None
                target_price = None
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
        atr_values: np.ndarray | None = None,
    ) -> BacktestResult:
        """
        Simulate classical 1-unit position percentage return tracking without cash sizing.

        Goal:
        -----
        Provide pure percentage strategy return tracking matching legacy backtest models,
        independent of account capital size.

        Execution Principle:
        --------------------
        1. Maintain binary exposure array `positions` (1.0 when long, 0.0 when flat) and `cost_rets`.
        2. Iterate bar-by-bar:
           - Check risk management exits (stop loss, take profit, max hold bars, trailing stops, ATR stops).
             If triggered, record trade return net of execution friction and exit to flat.
           - If 'Buy' and flat: set `in_position = True`, execute at `price * (1 + cost_rate)`,
             record entry friction in `cost_rets`.
           - If 'Sell' and in position: set `in_position = False`, execute at `price * (1 - cost_rate)`,
             record trade return, record exit friction.
           - If still in position, set `positions[i] = 1.0`.
        3. Force close open position at `prices[-1]`.
        4. Calculate bar percentage price changes `price_rets = diff(prices) / prices[:-1]`.
        5. Composite bar returns: `daily_rets = positions * price_rets - cost_rets`.
        6. Compound portfolio equity curve: `equity = cumprod(1.0 + daily_rets)`.
        7. Compute and return metrics via `calculate_metrics()`.

        Parameters:
        -----------
        prices : np.ndarray
            Close prices.
        signals : np.ndarray
            Trading signals.
        n : int
            Bar count.
        one_way_cost_rate : float
            Friction per trade leg.
        atr_values : np.ndarray, optional
            Precalculated ATR series.

        Returns:
        --------
        BacktestResult
            Calculated backtest metrics.
        """
        cfg = self.config
        in_position = False
        entry_price_raw = 0.0
        entry_price_exec = 0.0
        peak_price = 0.0
        stop_price = None
        target_price = None
        entry_index = -1
        positions = np.zeros(n)
        cost_rets = np.zeros(n)
        trade_returns: list[float] = []

        for i in range(n):
            if in_position:
                bars_held = i - entry_index
                gross_ret = (prices[i] / entry_price_raw) - 1.0
                peak_price = max(peak_price, prices[i])
                risk_exit = False

                if cfg.trailing_stop:
                    trail_pct = (
                        cfg.trailing_stop_pct
                        if cfg.trailing_stop_pct is not None
                        else cfg.stop_loss_pct
                    )
                    if trail_pct is not None and ((prices[i] / peak_price) - 1.0) <= -trail_pct:
                        risk_exit = True
                elif cfg.stop_loss_pct is not None and gross_ret <= -cfg.stop_loss_pct:
                    risk_exit = True
                elif stop_price is not None and prices[i] <= stop_price:
                    risk_exit = True

                if cfg.take_profit_pct is not None and gross_ret >= cfg.take_profit_pct:
                    risk_exit = True
                elif target_price is not None and prices[i] >= target_price:
                    risk_exit = True

                if cfg.max_hold_bars is not None and bars_held >= cfg.max_hold_bars:
                    risk_exit = True

                if risk_exit:
                    exit_price_exec = prices[i] * (1.0 - one_way_cost_rate)
                    trade_returns.append((exit_price_exec - entry_price_exec) / entry_price_exec)
                    in_position = False
                    entry_price_raw = 0.0
                    entry_price_exec = 0.0
                    peak_price = 0.0
                    stop_price = None
                    target_price = None
                    entry_index = -1
                    positions[i] = 1.0
                    cost_rets[i] += one_way_cost_rate
                    continue

            if signals[i] == 'Buy' and not in_position:
                in_position = True
                entry_price_raw = prices[i]
                entry_price_exec = prices[i] * (1.0 + one_way_cost_rate)
                peak_price = prices[i]
                entry_index = i
                stop_price = None
                target_price = None
                if atr_values is not None:
                    curr_atr = atr_values[i]
                    if cfg.stop_loss_atr_mult is not None:
                        stop_price = prices[i] - (cfg.stop_loss_atr_mult * curr_atr)
                    if cfg.take_profit_atr_mult is not None:
                        target_price = prices[i] + (cfg.take_profit_atr_mult * curr_atr)
                positions[i] = 0.0
                cost_rets[i] += one_way_cost_rate
            elif signals[i] == 'Sell' and in_position:
                in_position = False
                exit_price_exec = prices[i] * (1.0 - one_way_cost_rate)
                trade_returns.append((exit_price_exec - entry_price_exec) / entry_price_exec)
                entry_price_raw = 0.0
                entry_price_exec = 0.0
                peak_price = 0.0
                stop_price = None
                target_price = None
                entry_index = -1
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
    trailing_stop: bool = False,
    trailing_stop_pct: float | None = None,
    stop_loss_atr_mult: float | None = None,
    take_profit_atr_mult: float | None = None,
    atr_col: str | None = None,
) -> dict:
    """
    Long-only backtest engine function for any strategy emitting Buy/Sell/Hold signals.

    Goal:
    -----
    Provide a convenient, backward-compatible functional interface to execute backtests,
    returning metrics in the classic Python dictionary format.

    Execution Principle:
    --------------------
    1. Bundles parameters into a `BacktestConfig` instance.
    2. Instantiates a `BacktestEngine` with that configuration.
    3. Runs the simulation on the input DataFrame `df`.
    4. Converts the resulting `BacktestResult` to a dictionary using `.to_dict()` and returns it.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing price and signal columns.
    signal_col : str, default 'Signal'
        Column name with 'Buy', 'Sell', 'Hold' labels.
    close_col : str, default 'Close'
        Column name with asset closing prices.
    fee_bps : float, default 0.0
        Transaction commission in basis points.
    slippage_bps : float, default 0.0
        Execution slippage friction in basis points.
    initial_cash : float, optional
        Starting capital. If provided, enables cash-constrained sizing.
    stop_loss_pct : float, optional
        Percentage loss threshold triggering risk exit.
    take_profit_pct : float, optional
        Percentage gain threshold triggering profit taking.
    max_hold_bars : int, optional
        Maximum bar count allowed per trade before forced exit.
    trailing_stop : bool, default False
        Whether to trail the stop loss behind the peak price.
    trailing_stop_pct : float, optional
        Trailing drop percentage from peak.
    stop_loss_atr_mult : float, optional
        Dynamic stop loss in ATR multiples below entry.
    take_profit_atr_mult : float, optional
        Dynamic profit target in ATR multiples above entry.
    atr_col : str, optional
        Column name for precomputed ATR in df.

    Returns:
    --------
    dict
        Dictionary containing all performance metrics (total_return_pct, sharpe_ratio,
        max_drawdown_pct, win_rate_pct, profit_factor, n_trades, trade_returns, etc.).
    """
    config = BacktestConfig(
        signal_col=signal_col,
        close_col=close_col,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_cash=initial_cash,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_hold_bars=max_hold_bars,
        trailing_stop=trailing_stop,
        trailing_stop_pct=trailing_stop_pct,
        stop_loss_atr_mult=stop_loss_atr_mult,
        take_profit_atr_mult=take_profit_atr_mult,
        atr_col=atr_col,
    )
    result = BacktestEngine(config).run(df)
    return result.to_dict()
