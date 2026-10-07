"""
app.py — Flask API server entry point.

Exposes strategy routes and orchestrator services via the application factory.
"""

import os
from advisor.api import create_app
from advisor.data.validator import validate_ticker as _validate_ticker
from advisor.services import (
    DEFAULT_SAMPLE_TICKERS,
    run_simulation_for_ticker as _run_simulation_for_ticker,
    run_strategies_pipeline as _run_strategies,
)

app = create_app()

__all__ = [
    "app",
    "create_app",
    "_run_strategies",
    "_run_simulation_for_ticker",
    "_validate_ticker",
    "DEFAULT_SAMPLE_TICKERS",
]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
