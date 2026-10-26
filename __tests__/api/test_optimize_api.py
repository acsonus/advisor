"""API tests for the /api/strategies/optimize endpoint."""

from unittest.mock import patch
import pytest
from advisor.api import create_app


@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_optimize_endpoint_invalid_ticker(client):
    """Verify endpoint rejects invalid ticker with 400."""
    response = client.get("/api/strategies/optimize?ticker=INVALID$$$")
    assert response.status_code == 400


def test_optimize_endpoint_unknown_strategy(client):
    """Verify endpoint rejects unknown strategy with 400."""
    response = client.get("/api/strategies/optimize?strategy=non_existent_strategy")
    assert response.status_code == 400


def test_optimize_endpoint_invalid_iterations(client):
    """Verify endpoint rejects non-positive iterations with 400."""
    response = client.get("/api/strategies/optimize?iterations=-10")
    assert response.status_code == 400


def test_optimize_endpoint_invalid_cash(client):
    """Verify endpoint rejects non-positive initial cash with 400."""
    response = client.get("/api/strategies/optimize?initial_cash=0")
    assert response.status_code == 400


def test_optimize_endpoint_success_mocked(client):
    """Verify endpoint dispatches to service and returns 200 with result."""
    mock_report = {
        "ticker": "AAPL",
        "strategy_name": "ma_rsi",
        "best_strategy_params": {"short_ema_period": 10, "long_ema_period": 30},
        "best_risk_params": {"stop_loss_pct": 0.02, "take_profit_pct": 0.05},
        "best_fitness": 1.25,
        "baseline_in_sample": {"total_return_pct": 5.0},
        "baseline_out_of_sample": {"total_return_pct": 2.0},
        "optimized_in_sample": {"total_return_pct": 18.0},
        "optimized_out_of_sample": {"total_return_pct": 12.0},
        "out_of_sample_verified": True,
        "saved_to": "/path/to/optimal_params.json",
        "bars_total": 100,
        "bars_train": 70,
        "bars_test": 30,
        "top_trials": [],
    }

    with patch(
        "advisor.api.routes.optimize_strategy_for_ticker",
        return_value=mock_report,
    ) as mock_opt:
        response = client.get(
            "/api/strategies/optimize?ticker=AAPL&strategy=ma_rsi&iterations=20"
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert "optimization" in data
        assert data["optimization"]["best_strategy_params"]["short_ema_period"] == 10
        mock_opt.assert_called_once()


def test_simulate_with_use_optimal_flag(client):
    """Verify simulate endpoint passes use_optimal parameter properly."""
    mock_sim_result = {
        "ticker": "AAPL",
        "bars": 100,
        "strategies": {
            "ma_rsi": {
                "total_return_pct": 15.0,
                "profit": 1500.0,
                "n_trades": 6,
                "win_rate_pct": 66.7,
                "max_drawdown_pct": 4.5,
                "sharpe_ratio": 1.8,
                "skipped_buys_due_to_cash": 0,
                "used_optimal_params": True,
            }
        },
    }

    with patch(
        "advisor.api.routes.run_simulation_for_ticker",
        return_value=mock_sim_result,
    ) as mock_sim:
        response = client.get(
            "/api/strategies/simulate?tickers=AAPL&use_optimal=true"
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert data["assumptions"]["use_optimal"] is True
        mock_sim.assert_called_once_with(
            ticker="AAPL",
            period="6mo",
            interval="1d",
            initial_cash=10000.0,
            commission_bps=0.0,
            use_optimal_params=True,
        )
