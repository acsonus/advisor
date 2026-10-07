"""Bollinger Band Squeeze Breakout Strategy."""

from dataclasses import dataclass
from typing import ClassVar
import pandas as pd

from advisor.strategies.base import BaseStrategy


@dataclass
class BollingerSqueezeStrategy(BaseStrategy):
    """
    Bollinger Band volatility squeeze and breakout strategy.

    Detects periods of extreme historical volatility compression (the squeeze)
    followed by volatility expansion breakouts outside the bands.
    """

    name: ClassVar[str] = "bollinger_squeeze"
    description: ClassVar[str] = "Bollinger Band volatility squeeze breakout"

    window: int = 20
    num_std_dev: float = 2.0
    squeeze_window: int = 100
    squeeze_threshold: float = 0.2

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Bollinger Bands, identify non-anticipating squeezes, and detect breakouts.

        Goal:
        -----
        Capitalize on volatility cycles by identifying tight consolidation regimes (squeezes)
        and triggering entries as soon as price breaks out through the volatility envelope,
        guaranteeing strict avoidance of look-ahead bias.

        Execution Principle:
        --------------------
        1. Input Immutability:
           Copies input data with `df = data.copy()`.
        2. Classical Bollinger Bands:
           - Middle Band: 20-period simple moving average `df['Close'].rolling(window).mean()`.
           - Standard Deviation: 20-period rolling standard deviation of close prices.
           - Upper Band = `Middle_Band + (Std_Dev * num_std_dev)`.
           - Lower Band = `Middle_Band - (Std_Dev * num_std_dev)`.
           - Bollinger Band Width (`BB_Width`): `Upper_Band - Lower_Band`.
        3. Non-Anticipating Squeeze Detection:
           - Computes rolling minimum of width over `squeeze_window` (100 bars).
           - Squeeze threshold calculation:
             `rolling_std = df['BB_Width'].rolling(window=squeeze_window).std().shift(1)`.
             Crucially shifts rolling statistics by 1 bar and uses historical-only data
             rather than full-sample column standard deviation, preventing future information leakage.
           - `Is_Squeeze` flag is True when `BB_Width < (Min_BB_Width[i-1] + rolling_std * squeeze_threshold)`.
        4. Breakout Entry Triggers:
           - Default signal initialized to `'Hold'`.
           - Bullish Breakout ('Buy'):
             `Is_Squeeze[i-1] == True` (market was in a squeeze on prior bar)
             AND `Close[i] > Upper_Band[i-1]` (price breaks above the band).
           - Bearish Breakdown ('Sell'):
             `Is_Squeeze[i-1] == True` AND `Close[i] < Lower_Band[i-1]`.
        5. Returns DataFrame with:
           `['Close', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'BB_Width', 'Is_Squeeze', 'Signal']`.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV DataFrame with `'Close'` column.

        Returns:
        --------
        pd.DataFrame
            Calculated Bollinger bands, squeeze indicators, and signals.
        """
        df = data.copy()

        # Calculate Middle Band (Moving Average)
        df['Middle_Band'] = df['Close'].rolling(window=self.window).mean()

        # Calculate Standard Deviation
        df['Std_Dev'] = df['Close'].rolling(window=self.window).std()

        # Calculate Upper and Lower Bollinger Bands
        df['Upper_Band'] = df['Middle_Band'] + (df['Std_Dev'] * self.num_std_dev)
        df['Lower_Band'] = df['Middle_Band'] - (df['Std_Dev'] * self.num_std_dev)

        # Calculate Bollinger Band Width
        df['BB_Width'] = df['Upper_Band'] - df['Lower_Band']

        # Identify Squeeze Condition
        df['Min_BB_Width'] = df['BB_Width'].rolling(window=self.squeeze_window).min()
        rolling_std = df['BB_Width'].rolling(window=self.squeeze_window).std().shift(1)
        df['Is_Squeeze'] = df['BB_Width'] < (df['Min_BB_Width'].shift(1) + rolling_std * self.squeeze_threshold)

        # Generate Signals
        df['Signal'] = 'Hold'

        # Buy signal: Price breaks above upper band after a squeeze
        df.loc[(df['Is_Squeeze'].shift(1) == True) & (df['Close'] > df['Upper_Band'].shift(1)), 'Signal'] = 'Buy'

        # Sell signal: Price breaks below lower band after a squeeze
        df.loc[(df['Is_Squeeze'].shift(1) == True) & (df['Close'] < df['Lower_Band'].shift(1)), 'Signal'] = 'Sell'

        return df[['Close', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'BB_Width', 'Is_Squeeze', 'Signal']]


def bollinger_squeeze_strategy(
    data: pd.DataFrame,
    window: int = 20,
    num_std_dev: float = 2.0,
    squeeze_window: int = 100,
    squeeze_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Functional wrapper for the Bollinger Band Squeeze Breakout strategy.

    Goal:
    -----
    Preserve functional API parity for callers and test suites.

    Execution Principle:
    --------------------
    Instantiates `BollingerSqueezeStrategy` with the supplied configuration parameters
    and delegates execution to its `generate_signals` method.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV price series.
    window : int, default 20
        Moving average and volatility lookback window.
    num_std_dev : float, default 2.0
        Standard deviation band width multiplier.
    squeeze_window : int, default 100
        Lookback duration for computing minimum band width baseline.
    squeeze_threshold : float, default 0.2
        Sensitivity threshold above rolling minimum for qualifying as a squeeze.

    Returns:
    --------
    pd.DataFrame
        Strategy output DataFrame containing indicators and trade signals.
    """
    return BollingerSqueezeStrategy(
        window=window,
        num_std_dev=num_std_dev,
        squeeze_window=squeeze_window,
        squeeze_threshold=squeeze_threshold,
    ).generate_signals(data)
