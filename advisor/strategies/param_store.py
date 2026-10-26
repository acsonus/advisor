"""Parameter store for saving and retrieving optimized strategy configurations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_PARAMS_FILE = Path(__file__).parent / "optimal_params.json"


def get_default_params_path() -> Path:
    """Return default path to optimal parameters JSON file."""
    return DEFAULT_PARAMS_FILE


def load_all_optimal_params(filepath: Path | str | None = None) -> dict[str, Any]:
    """Load entire parameter configuration dictionary."""
    path = Path(filepath) if filepath else DEFAULT_PARAMS_FILE
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_optimal_params(
    ticker: str,
    strategy_name: str,
    filepath: Path | str | None = None,
) -> dict[str, Any] | None:
    """Load optimized parameters for a specific ticker and strategy."""
    data = load_all_optimal_params(filepath)
    ticker_data = data.get(ticker.upper())
    if not ticker_data or not isinstance(ticker_data, dict):
        return None
    return ticker_data.get(strategy_name)


def save_optimal_params(
    ticker: str,
    strategy_name: str,
    strategy_params: dict[str, Any],
    risk_params: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    filepath: Path | str | None = None,
) -> Path:
    """
    Save winning parameters to JSON store.

    Structure:
    {
        "AAPL": {
            "ma_rsi": {
                "strategy_params": {...},
                "risk_params": {...},
                "metrics": {...},
                "updated_at": "..."
            }
        }
    }
    """
    path = Path(filepath) if filepath else DEFAULT_PARAMS_FILE
    data = load_all_optimal_params(path)

    ticker_key = ticker.upper()
    if ticker_key not in data:
        data[ticker_key] = {}

    data[ticker_key][strategy_name] = {
        "strategy_params": strategy_params,
        "risk_params": risk_params,
        "metrics": metrics or {},
        "updated_at": datetime.now().isoformat(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


def get_effective_params(
    ticker: str,
    strategy_name: str,
    filepath: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Retrieve stored parameters if present, else fallback to empty dictionaries.

    Returns:
    --------
    tuple[dict, dict]
        (strategy_params, risk_params)
    """
    entry = load_optimal_params(ticker, strategy_name, filepath)
    if not entry:
        return {}, {}
    return entry.get("strategy_params", {}), entry.get("risk_params", {})
