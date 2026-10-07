"""Integration smoke tests across all strategies."""

import pytest

from utilities.test_helpers import (
    assert_valid_signals as _assert_valid_signals,
    make_ohlcv,
    v_shape,
)
from advisor.backtesting import backtest_strategy
from advisor.strategies import (
    atr_trailing_stop,
    bollinger_squeeze_strategy,
    calculate_vwap,
    macd_histogram_reversal_strategy,
    ma_rsi_strategy,
)


class TestIntegrationSmoke:
    """
    Run every strategy end-to-end on synthetic market data and verify
    that each produces a structurally valid, signal-correct result.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def market_data(cls):
        """
        Goal:
            Fixture providing a shared V-shape OHLCV market dataset of sufficient length (150 bars) for warm-up of all strategies.

        Execution Principle:
            1. Generate a 150-bar V-shaped recovery series (70 bars down, 80 bars up).
            2. Convert to synthetic OHLCV DataFrame using `make_ohlcv`.
            3. Return cached class-scoped DataFrame.
        """
        return make_ohlcv(v_shape(n_down=70, n_up=80, start=200.0, step=1.0))

    def test_ma_rsi_smoke(self, market_data):
        """
        Goal:
            Verify MA-RSI strategy runs cleanly end-to-end and produces non-empty, valid signals.

        Execution Principle:
            1. Run `ma_rsi_strategy(market_data)`.
            2. Assert output DataFrame is not empty.
            3. Validate signal column against valid enum set using `_assert_valid_signals`.
        """
        result = ma_rsi_strategy(market_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_atr_smoke(self, market_data):
        """
        Goal:
            Verify ATR trailing stop strategy runs cleanly end-to-end and produces non-empty, valid signals.

        Execution Principle:
            1. Run `atr_trailing_stop(market_data)`.
            2. Assert output DataFrame is not empty.
            3. Validate signals using `_assert_valid_signals`.
        """
        result = atr_trailing_stop(market_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_macd_smoke(self, market_data):
        """
        Goal:
            Verify MACD histogram reversal strategy runs cleanly end-to-end and produces non-empty, valid signals.

        Execution Principle:
            1. Run `macd_histogram_reversal_strategy(market_data)`.
            2. Assert output DataFrame is not empty.
            3. Validate signals using `_assert_valid_signals`.
        """
        result = macd_histogram_reversal_strategy(market_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_bollinger_smoke(self, market_data):
        """
        Goal:
            Verify Bollinger Band squeeze breakout strategy runs cleanly on 160 bars and produces valid signals.

        Execution Principle:
            1. Generate 160-bar synthetic recovery data.
            2. Run `bollinger_squeeze_strategy`.
            3. Assert output is non-empty and signals are valid.
        """
        long_data = make_ohlcv(v_shape(n_down=80, n_up=80, start=200.0, step=1.0))
        result = bollinger_squeeze_strategy(long_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_vwap_smoke(self, market_data):
        """
        Goal:
            Verify VWAP strategy runs cleanly end-to-end and produces non-empty, valid signals.

        Execution Principle:
            1. Run `calculate_vwap(market_data)`.
            2. Assert output DataFrame is not empty.
            3. Validate signals using `_assert_valid_signals`.
        """
        result = calculate_vwap(market_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_backtest_on_atr_signals(self, market_data):
        """
        Goal:
            Verify seamless integration between ATR trailing stop signals and the backtesting engine.

        Execution Principle:
            1. Compute ATR signals over market data.
            2. Run `backtest_strategy` on resulting `['Close', 'Signal']` columns.
            3. Assert presence of key risk/return metrics: total return, Sharpe, max drawdown, and trade count.
        """
        atr_result = atr_trailing_stop(market_data)
        metrics = backtest_strategy(
            atr_result[["Close", "Signal"]],
            signal_col="Signal",
            close_col="Close",
        )
        for key in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "n_trades"):
            assert key in metrics

    def test_backtest_on_ma_rsi_signals(self, market_data):
        """
        Goal:
            Verify seamless integration between MA-RSI signals and the backtesting engine.

        Execution Principle:
            1. Compute MA-RSI signals over market data.
            2. Run `backtest_strategy`.
            3. Assert presence of core performance metrics in result dictionary.
        """
        ma_result = ma_rsi_strategy(market_data)
        metrics = backtest_strategy(
            ma_result[["Close", "Signal"]],
            signal_col="Signal",
            close_col="Close",
        )
        for key in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "n_trades"):
            assert key in metrics

    def test_backtest_on_macd_signals(self, market_data):
        """
        Goal:
            Verify seamless integration between MACD reversal signals and the backtesting engine.

        Execution Principle:
            1. Compute MACD signals over market data.
            2. Run `backtest_strategy`.
            3. Assert `'total_return_pct'` is present in output metrics.
        """
        macd_result = macd_histogram_reversal_strategy(market_data)
        metrics = backtest_strategy(
            macd_result[["Close", "Signal"]],
            signal_col="Signal",
            close_col="Close",
        )
        assert "total_return_pct" in metrics
