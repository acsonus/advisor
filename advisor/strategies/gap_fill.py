"""Opening Gap Fill Strategy."""

from dataclasses import dataclass
from typing import Any, ClassVar
import pandas as pd


@dataclass
class GapFillStrategy:
    """
    Opening gap fade / gap fill algorithm.

    Identifies significant opening gap-downs and looks for mean-reverting
    bounce opportunities targeting prior-day closing prices.
    """

    name: ClassVar[str] = "gap_fill"
    description: ClassVar[str] = "Opening gap fade algorithm"

    threshold_pct: float = -0.005

    def analyze(self, data: pd.DataFrame) -> dict[str, Any] | str:
        """
        Analyze opening price gaps and construct trade setup with target and risk/reward.

        Goal:
        -----
        Identify overnight gap-down anomalies exceeding a risk threshold (e.g. >0.5% drop),
        and evaluate whether intraday price action breaks above the opening range high
        to establish a gap-fill mean reversion long trade.

        Execution Principle:
        --------------------
        1. Length Check:
           Requires at least 2 bars to evaluate previous close vs. current open. If insufficient,
           returns "No trade criteria met.".
        2. Gap Calculation:
           - `prev_close = data['Close'].iloc[-2]`.
           - `today_open = data['Open'].iloc[-1]`.
           - `gap_pct = (today_open - prev_close) / prev_close`.
        3. Gap Threshold Filter:
           Checks if `gap_pct < threshold_pct` (e.g. -0.5% down).
        4. Setup Formulation:
           - Establishes Opening Range High (`or_high`) and Opening Range Low (`or_low`)
             from the current bar.
           - Checks if current price breaks above `or_high`.
           - Target price is set to `prev_close` (the gap fill level).
           - Stop loss is anchored at `or_low` (protecting below the day's low).
           - Calculates risk/reward ratio: `(target - entry) / (entry - stop)`.
        5. Returns trade parameters dict if criteria are met; else returns message string.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV DataFrame containing at least 2 bars with 'Open', 'High', 'Low', 'Close'.

        Returns:
        --------
        dict[str, Any] | str
            Dictionary with action, entry, target, stop, and risk/reward, or status string.
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
    """
    Functional wrapper for the opening gap fill algorithm.

    Goal:
    -----
    Maintain legacy functional interface compatibility.

    Execution Principle:
    --------------------
    Instantiates `GapFillStrategy` and delegates execution to `analyze(data)`.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV price series.

    Returns:
    --------
    dict[str, Any] | str
        Trade setup dictionary or status string.
    """
    return GapFillStrategy().analyze(data)
