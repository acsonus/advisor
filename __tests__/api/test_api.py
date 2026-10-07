"""Tests for Flask API endpoints."""

import pytest
from advisor.api import create_app


@pytest.fixture
def client():
    """
    Goal:
        Provide a Flask test client configured in isolated test mode for exercising API route endpoints.

    Execution Principle:
        1. Initialize Flask application instance using `create_app({"TESTING": True})`.
        2. Contextually yield `test_client()` to caller test cases.
    """
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """
    Goal:
        Verify the liveness probe endpoint `/api/health` responds with HTTP 200 OK and healthy status.

    Execution Principle:
        1. Dispatch GET request to `/api/health`.
        2. Assert status code equals 200.
        3. Assert JSON response payload equals `{"status": "healthy"}`.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy"}


def test_run_strategies_invalid_ticker(client):
    """
    Goal:
        Verify strategy pipeline execution rejects invalid or malicious ticker format strings with HTTP 400.

    Execution Principle:
        1. Dispatch GET request to `/api/strategies/run?ticker=INVALID$$$`.
        2. Assert controller rejects the malformed ticker with status code 400.
    """
    response = client.get("/api/strategies/run?ticker=INVALID$$$")
    assert response.status_code == 400


def test_run_strategies_invalid_period(client):
    """
    Goal:
        Verify strategy execution rejects unsupported period arguments with HTTP 400.

    Execution Principle:
        1. Dispatch GET request to `/api/strategies/run?period=bad_period`.
        2. Assert response status code is 400.
    """
    response = client.get("/api/strategies/run?period=bad_period")
    assert response.status_code == 400


def test_download_report_invalid_interval(client):
    """
    Goal:
        Verify PDF report generation endpoint rejects unsupported interval parameters with HTTP 400.

    Execution Principle:
        1. Dispatch GET request to `/api/strategies/report?interval=bad_interval`.
        2. Assert validation guard intercepts the query and responds with status code 400.
    """
    response = client.get("/api/strategies/report?interval=bad_interval")
    assert response.status_code == 400


def test_simulate_invalid_cash(client):
    """
    Goal:
        Verify portfolio simulation endpoint rejects negative starting cash amounts with HTTP 400.

    Execution Principle:
        1. Dispatch GET request to `/api/strategies/simulate?initial_cash=-500`.
        2. Assert response status code is 400.
    """
    response = client.get("/api/strategies/simulate?initial_cash=-500")
    assert response.status_code == 400
