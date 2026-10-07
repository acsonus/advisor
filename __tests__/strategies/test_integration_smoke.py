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
        """Shared V-shape dataset with enough bars for all strategies."""
        return make_ohlcv(v_shape(n_down=70, n_up=80, start=200.0, step=1.0))

    def test_ma_rsi_smoke(self, market_data):
        result = ma_rsi_strategy(market_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_atr_smoke(self, market_data):
        result = atr_trailing_stop(market_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_macd_smoke(self, market_data):
        result = macd_histogram_reversal_strategy(market_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_bollinger_smoke(self, market_data):
        long_data = make_ohlcv(v_shape(n_down=80, n_up=80, start=200.0, step=1.0))
        result = bollinger_squeeze_strategy(long_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_vwap_smoke(self, market_data):
        result = calculate_vwap(market_data)
        assert not result.empty
        _assert_valid_signals(result)

    def test_backtest_on_atr_signals(self, market_data):
        atr_result = atr_trailing_stop(market_data)
        metrics = backtest_strategy(
            atr_result[["Close", "Signal"]],
            signal_col="Signal",
            close_col="Close",
        )
        for key in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "n_trades"):
            assert key in metrics

    def test_backtest_on_ma_rsi_signals(self, market_data):
        ma_result = ma_rsi_strategy(market_data)
        metrics = backtest_strategy(
            ma_result[["Close", "Signal"]],
            signal_col="Signal",
            close_col="Close",
        )
        for key in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "n_trades"):
            assert key in metrics

    def test_backtest_on_macd_signals(self, market_data):
        macd_result = macd_histogram_reversal_strategy(market_data)
        metrics = backtest_strategy(
            macd_result[["Close", "Signal"]],
            signal_col="Signal",
            close_col="Close",
        )
        assert "total_return_pct" in metrics
