"""Strategy parameter and risk management optimization search engine."""

from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from advisor.backtesting.engine import BacktestEngine
from advisor.backtesting.models import BacktestConfig, BacktestResult
from advisor.optimization.objectives import OptimizationObjective
from advisor.optimization.sampler import ParameterSampler
from advisor.optimization.spaces import ParameterSpace, get_default_strategy_space
from advisor.strategies.registry import StrategyRegistry


@dataclass
class TrialResult:
    """Record of a single parameter evaluation trial."""
    strategy_params: dict[str, Any]
    risk_params: dict[str, Any]
    fitness: float
    metrics: dict[str, Any]


@dataclass
class OptimizationResult:
    """Complete summary of a strategy optimization run."""
    strategy_name: str
    best_strategy_params: dict[str, Any]
    best_risk_params: dict[str, Any]
    best_fitness: float
    baseline_in_sample: dict[str, Any]
    baseline_out_of_sample: dict[str, Any]
    optimized_in_sample: dict[str, Any]
    optimized_out_of_sample: dict[str, Any]
    out_of_sample_verified: bool
    top_trials: list[dict[str, Any]] = field(default_factory=list)
    bars_total: int = 0
    bars_train: int = 0
    bars_test: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to standard dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "best_strategy_params": self.best_strategy_params,
            "best_risk_params": self.best_risk_params,
            "best_fitness": self.best_fitness,
            "baseline_in_sample": self.baseline_in_sample,
            "baseline_out_of_sample": self.baseline_out_of_sample,
            "optimized_in_sample": self.optimized_in_sample,
            "optimized_out_of_sample": self.optimized_out_of_sample,
            "out_of_sample_verified": self.out_of_sample_verified,
            "bars_total": self.bars_total,
            "bars_train": self.bars_train,
            "bars_test": self.bars_test,
            "top_trials": self.top_trials,
        }


class StrategyOptimizer:
    """
    Randomized parameter search optimizer with train/test validation
    and risk-adjusted fitness scoring.
    """

    def __init__(
        self,
        objective: OptimizationObjective | None = None,
        train_ratio: float = 0.70,
        initial_cash: float = 10000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ):
        self.objective = objective or OptimizationObjective()
        self.train_ratio = train_ratio
        self.initial_cash = initial_cash
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    def _split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split historical time series sequentially into in-sample and out-of-sample."""
        n = len(data)
        split_idx = int(n * self.train_ratio)
        train_df = data.iloc[:split_idx].copy()
        test_df = data.iloc[split_idx:].copy()
        return train_df, test_df

    def _evaluate_candidate(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        strategy_params: dict[str, Any],
        risk_params: dict[str, Any],
    ) -> BacktestResult:
        """Run strategy signal generation and backtest for a specific parameter set."""
        strategy = StrategyRegistry.get(strategy_name, **strategy_params)
        signals_df = strategy.generate_signals(df)

        df_bt = pd.DataFrame({
            "Close": df["Close"],
            "Signal": signals_df["Signal"],
        })
        if "High" in df.columns:
            df_bt["High"] = df["High"]
        if "Low" in df.columns:
            df_bt["Low"] = df["Low"]

        config = BacktestConfig(
            signal_col="Signal",
            close_col="Close",
            fee_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            initial_cash=self.initial_cash,
            stop_loss_pct=risk_params.get("stop_loss_pct"),
            take_profit_pct=risk_params.get("take_profit_pct"),
            max_hold_bars=risk_params.get("max_hold_bars"),
            trailing_stop=bool(risk_params.get("trailing_stop", False)),
            stop_loss_atr_mult=risk_params.get("stop_loss_atr_mult"),
            take_profit_atr_mult=risk_params.get("take_profit_atr_mult"),
        )
        return BacktestEngine(config).run(df_bt)

    def optimize(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        n_iter: int = 50,
        space: ParameterSpace | None = None,
        seed: int | None = 42,
    ) -> OptimizationResult:
        """
        Execute randomized parameter search with holdout validation.

        Goal:
        -----
        Find strategy parameters and exit thresholds that maximize risk-adjusted
        performance on training data while verifying positive expectancy on holdout data.
        """
        if len(data) < 50:
            raise ValueError(
                f"Insufficient data for optimization: {len(data)} bars provided. Need at least 50 bars."
            )

        param_space = space or get_default_strategy_space(strategy_name)
        train_df, test_df = self._split_data(data)

        # Baseline evaluation (empty custom params = defaults, no risk rules)
        base_train_res = self._evaluate_candidate(strategy_name, train_df, {}, {})
        base_test_res = self._evaluate_candidate(strategy_name, test_df, {}, {})

        # Generate trial candidates
        sampler = ParameterSampler(param_space, seed=seed)
        candidates = sampler.sample_batch(n_iter)

        trials: list[TrialResult] = []
        for strat_cfg, risk_cfg in candidates:
            try:
                res = self._evaluate_candidate(strategy_name, train_df, strat_cfg, risk_cfg)
                fitness = self.objective.compute_fitness(res)
                trials.append(
                    TrialResult(
                        strategy_params=strat_cfg,
                        risk_params=risk_cfg,
                        fitness=fitness,
                        metrics=res.to_dict(),
                    )
                )
            except Exception:
                continue

        if not trials:
            raise RuntimeError(f"Optimization failed: 0 valid trials completed for {strategy_name}.")

        # Sort by in-sample fitness descending
        trials.sort(key=lambda t: t.fitness, reverse=True)
        best_trial = trials[0]

        # Evaluate winning parameters on out-of-sample data
        opt_test_res = self._evaluate_candidate(
            strategy_name,
            test_df,
            best_trial.strategy_params,
            best_trial.risk_params,
        )

        # Verify out-of-sample: beats zero or beats baseline return
        opt_test_dict = opt_test_res.to_dict()
        base_test_dict = base_test_res.to_dict()
        oos_verified = (
            opt_test_dict["total_return_pct"] > 0
            or opt_test_dict["total_return_pct"] > base_test_dict["total_return_pct"]
        )

        top_trials_summary = [
            {
                "rank": idx + 1,
                "strategy_params": t.strategy_params,
                "risk_params": t.risk_params,
                "fitness": t.fitness,
                "total_return_pct": t.metrics["total_return_pct"],
                "sharpe_ratio": t.metrics["sharpe_ratio"],
                "max_drawdown_pct": t.metrics["max_drawdown_pct"],
                "win_rate_pct": t.metrics["win_rate_pct"],
                "n_trades": t.metrics["n_trades"],
            }
            for idx, t in enumerate(trials[:5])
        ]

        return OptimizationResult(
            strategy_name=strategy_name,
            best_strategy_params=best_trial.strategy_params,
            best_risk_params=best_trial.risk_params,
            best_fitness=best_trial.fitness,
            baseline_in_sample=base_train_res.to_dict(),
            baseline_out_of_sample=base_test_dict,
            optimized_in_sample=best_trial.metrics,
            optimized_out_of_sample=opt_test_dict,
            out_of_sample_verified=oos_verified,
            top_trials=top_trials_summary,
            bars_total=len(data),
            bars_train=len(train_df),
            bars_test=len(test_df),
        )
