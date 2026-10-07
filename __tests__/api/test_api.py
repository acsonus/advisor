"""Tests for Flask API endpoints."""

import pytest
from advisor.api import create_app


@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy"}


def test_run_strategies_invalid_ticker(client):
    response = client.get("/api/strategies/run?ticker=INVALID$$$")
    assert response.status_code == 400


def test_run_strategies_invalid_period(client):
    response = client.get("/api/strategies/run?period=bad_period")
    assert response.status_code == 400


def test_download_report_invalid_interval(client):
    response = client.get("/api/strategies/report?interval=bad_interval")
    assert response.status_code == 400


def test_simulate_invalid_cash(client):
    response = client.get("/api/strategies/simulate?initial_cash=-500")
    assert response.status_code == 400
