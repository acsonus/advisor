"""Opening Gap Fill Strategy."""

from dataclasses import dataclass
from typing import Any, ClassVar
import pandas as pd


@dataclass
class GapFillStrategy:
    """Opening gap fade / gap fill algorithm."""

    name: ClassVar[str] = "gap_fill"
    description: ClassVar[str] = "Opening gap fade algorithm"

    threshold_pct: float = -0.005

    def analyze(self, data: pd.DataFrame) -> dict[str, Any] | str:
        """
        Analyze gap condition on OHLCV data.

        data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        indexed by timestamp.
        """
        if len(data) < 2:
            return "No trade criteria met."

        # 1. Identify the Gap
        prev_close = data['Close'].iloc[-2]
        today_open = data['Open'].iloc[-1]

        gap_pct = (today_open - prev_close) / prev_close

        # Significant gap down
        if gap_pct < self.threshold_pct:
            print(f"Significant Gap Down Detected: {gap_pct:.2%}")

            # 2. Establish Opening Range
            or_high = data['High'].iloc[-1]
            or_low = data['Low'].iloc[-1]

            # 3. Strategy Logic (The Fade)
            current_price = 105.00  # Placeholder for live ticker

            if current_price > or_high:
                entry_price = current_price
                target_price = prev_close
                stop_loss = or_low

                return {
                    "Action": "BUY",
                    "Entry": entry_price,
                    "Target": target_price,
                    "Stop": stop_loss,
                    "Risk_Reward": (target_price - entry_price) / (entry_price - stop_loss),
                }

        return "No trade criteria met."


def gap_fill_algorithm(data: pd.DataFrame) -> dict[str, Any] | str:
    """Backward-compatible function wrapper for gap fill algorithm."""
    return GapFillStrategy().analyze(data)
