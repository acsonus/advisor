from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
#in jypiter notebooks it is already avaliable by default
from IPython.display import display


def to_naive_s(series: pd.Series) -> pd.Series:
    """
    Normalise a datetime Series to tz-naive datetime64[s].

    yfinance returns tz-aware (UTC) timestamps for intraday intervals and
    tz-naive timestamps for daily/weekly intervals.  Calling .astype('datetime64[s]')
    directly on a tz-aware Series raises TypeError, so we strip the timezone first.
    """
    dt = pd.to_datetime(series)
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt.astype("datetime64[s]")


# ---------------------------------------------------------------------------
# Yahoo Finance supported values
# ---------------------------------------------------------------------------
VALID_PERIODS   = {'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'}
VALID_INTERVALS = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'}

# Maximum lookback in calendar days allowed per intraday interval by Yahoo Finance.
# Daily and weekly intervals have no enforced cap.
_INTERVAL_MAX_DAYS: dict[str, int] = {
    '1m':  7,
    '2m':  60,
    '5m':  60,
    '15m': 60,
    '30m': 60,
    '60m': 730,
    '90m': 60,
    '1h':  730,
}

# Approximate number of calendar days each period covers (used for validation).
_PERIOD_DAYS: dict[str, int] = {
    '1d':   1,
    '5d':   5,
    '1mo':  30,
    '3mo':  90,
    '6mo':  180,
    '1y':   365,
    '2y':   730,
    '5y':   1825,
    '10y':  3650,
    'ytd':  366,
    'max':  36500,
}


def news_sentiment_signal(df_news: pd.DataFrame,
                           bullish_threshold: float = 0.2,
                           bearish_threshold: float = -0.2) -> pd.DataFrame:
    """
    Aggregates per-article sentiment into a daily signal aligned with OHLCV dates.
    Expects df_news to have columns: ['Date', 'Signed_Score']
    Returns DataFrame indexed by Date with columns: ['daily_sentiment', 'Signal']
    """
    if df_news.empty or 'Signed_Score' not in df_news.columns:
        return pd.DataFrame(columns=['daily_sentiment', 'Signal'])

    daily = (
        df_news.groupby('Date')['Signed_Score']
               .mean()
               .rename('daily_sentiment')
               .to_frame()
    )
    daily['Signal'] = 'Hold'
    daily.loc[daily['daily_sentiment'] >  bullish_threshold, 'Signal'] = 'Buy'
    daily.loc[daily['daily_sentiment'] <  bearish_threshold, 'Signal'] = 'Sell'
    return daily

def downloadData(ticker_symbol: str = 'AAPL',
                 period: str = '1mo',
                 interval: str = '1d') -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker_symbol : str
        Ticker symbol, e.g. 'AAPL', 'BRK-B', '^GSPC'.
    period : str
        Lookback window.  One of: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max.
    interval : str
        Bar size.  One of: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo.
        Note: intraday intervals have Yahoo Finance limits on how far back
        data is available (e.g. 1m → max 7 days, 1h → max 730 days).
    """
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Choose from: {sorted(VALID_PERIODS)}")
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval '{interval}'. Choose from: {sorted(VALID_INTERVALS)}")
    if interval in _INTERVAL_MAX_DAYS:
        max_days = _INTERVAL_MAX_DAYS[interval]
        if _PERIOD_DAYS.get(period, 0) > max_days:
            raise ValueError(
                f"Interval '{interval}' supports at most {max_days} days of history, "
                f"but period '{period}' requests ~{_PERIOD_DAYS[period]} days."
            )

    data = yf.download(ticker_symbol, period=period, interval=interval, auto_adjust=True)

    # Flatten MultiIndex columns produced when a single ticker is downloaded.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    return data

def gap_fill_algorithm(data):
    """
    data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    indexed by timestamp.
    """
    # 1. Identify the Gap
    # Previous day's close vs Today's open
    prev_close = data['Close'].iloc[-2]
    today_open = data['Open'].iloc[-1]

    gap_pct = (today_open - prev_close) / prev_close

    # Define a significant gap (e.g., more than 0.5% down)
    if gap_pct < -0.005:
        print(f"Significant Gap Down Detected: {gap_pct:.2%}")

        # 2. Establish the 30-Minute Opening Range (OR)
        # In a real bot, you'd pull the high/low of the first 30 minsщ
        or_high = data['High'].iloc[-1] # Simplified for this example
        or_low = data['Low'].iloc[-1]

        # 3. Strategy Logic (The Fade)
        current_price = 105.00 # Placeholder for live ticker

        # Entry Trigger: Price breaks above the Opening Range High
        if current_price > or_high:
            entry_price = current_price
            target_price = prev_close  # The "Fill" target
            stop_loss = or_low         # Protect below the day's low

            return {
                "Action": "BUY",
                "Entry": entry_price,
                "Target": target_price,
                "Stop": stop_loss,
                "Risk_Reward": (target_price - entry_price) / (entry_price - stop_loss)
            }

    return "No trade criteria met."

def ma_rsi_strategy(data, short_ema_period=12, long_ema_period=26, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
    df = data.copy()

    # Calculate EMAs
    df['Short_EMA'] = df['Close'].ewm(span=short_ema_period, adjust=False).mean()
    df['Long_EMA'] = df['Close'].ewm(span=long_ema_period, adjust=False).mean()

    # Calculate RSI
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Wilder's smoothing: alpha = 1/rsi_period, equivalent to com = rsi_period - 1.
    # Using span=rsi_period (alpha=2/(rsi_period+1)) is a common mistake that
    # produces a faster-responding, non-standard RSI.
    avg_gain = gain.ewm(com=rsi_period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Generate Signals
    df['Signal'] = 'Hold'

    # Buy signal: Short EMA crosses above Long EMA (bullish trend change) AND
    # RSI has not yet reached overbought (there is still room to run).
    # Using rsi_overbought as the upper gate (instead of rsi_oversold) means the
    # signal fires on most valid bullish crossovers rather than only on the rare
    # coincidence of a crossover happening while the market is deeply oversold.
    df.loc[(df['Short_EMA'].shift(1) < df['Long_EMA'].shift(1)) &
           (df['Short_EMA'] > df['Long_EMA']) &
           (df['RSI'] < rsi_overbought), 'Signal'] = 'Buy'

    # Sell signal: Short EMA crosses below Long EMA (bearish trend change) AND
    # RSI is still above oversold (there is still room to fall).
    df.loc[(df['Short_EMA'].shift(1) > df['Long_EMA'].shift(1)) &
           (df['Short_EMA'] < df['Long_EMA']) &
           (df['RSI'] > rsi_oversold), 'Signal'] = 'Sell'

    return df[['Close', 'Short_EMA', 'Long_EMA', 'RSI', 'Signal']]

def bollinger_squeeze_strategy(data, window=20, num_std_dev=2, squeeze_window=100, squeeze_threshold=0.2):
    df = data.copy()

    # Calculate Middle Band (Moving Average)
    df['Middle_Band'] = df['Close'].rolling(window=window).mean()

    # Calculate Standard Deviation
    df['Std_Dev'] = df['Close'].rolling(window=window).std()

    # Calculate Upper and Lower Bollinger Bands
    df['Upper_Band'] = df['Middle_Band'] + (df['Std_Dev'] * num_std_dev)
    df['Lower_Band'] = df['Middle_Band'] - (df['Std_Dev'] * num_std_dev)

    # Calculate Bollinger Band Width
    df['BB_Width'] = df['Upper_Band'] - df['Lower_Band']

    # Identify Squeeze Condition (when BB_Width is at its lowest over a period)
    # A simple approach: compare current width to a rolling minimum or percentile
    df['Min_BB_Width'] = df['BB_Width'].rolling(window=squeeze_window).min()
    # Use a rolling std (shifted by 1) instead of the global column std so that
    # the squeeze threshold is computed from past data only — no look-ahead bias.
    rolling_std = df['BB_Width'].rolling(window=squeeze_window).std().shift(1)
    df['Is_Squeeze'] = df['BB_Width'] < (df['Min_BB_Width'].shift(1) + rolling_std * squeeze_threshold)

    # Generate Signals
    df['Signal'] = 'Hold'

    # Buy signal: Price breaks above upper band after a squeeze
    df.loc[(df['Is_Squeeze'].shift(1) == True) & (df['Close'] > df['Upper_Band'].shift(1)), 'Signal'] = 'Buy'

    # Sell signal: Price breaks below lower band after a squeeze
    df.loc[(df['Is_Squeeze'].shift(1) == True) & (df['Close'] < df['Lower_Band'].shift(1)), 'Signal'] = 'Sell'

    return df[['Close', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'BB_Width', 'Is_Squeeze', 'Signal']]

def macd_histogram_reversal_strategy(data, fast_period=12, slow_period=26, signal_period=9):
    df = data.copy()

    # Calculate the MACD line
    df['EMA_Fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']

    # Calculate the Signal line
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()

    # Calculate the MACD Histogram
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

    # Generate Signals
    df['Signal'] = 'Hold'

    # Buy signal: Histogram turns positive from negative
    df.loc[(df['MACD_Histogram'].shift(1) < 0) & (df['MACD_Histogram'] >= 0), 'Signal'] = 'Buy'

    # Sell signal: Histogram turns negative from positive
    df.loc[(df['MACD_Histogram'].shift(1) > 0) & (df['MACD_Histogram'] <= 0), 'Signal'] = 'Sell'

    return df[['Close', 'MACD', 'Signal_Line', 'MACD_Histogram', 'Signal']]

def calculate_vwap(data):
    df = data.copy()

    # Calculate Typical Price (TP)
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Calculate Cumulative TP * Volume
    df['TP_Volume'] = df['TP'] * df['Volume']
    df['Cumulative_TP_Volume'] = df['TP_Volume'].cumsum()

    # Calculate Cumulative Volume
    df['Cumulative_Volume'] = df['Volume'].cumsum()

    # Calculate VWAP
    df['VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']

    # Generate Signals
    df['Signal'] = 'Hold'
    # Buy signal: Close price crosses above VWAP
    df.loc[(df['Close'].shift(1) <= df['VWAP'].shift(1)) & (df['Close'] > df['VWAP']), 'Signal'] = 'Buy'
    # Sell signal: Close price crosses below VWAP
    df.loc[(df['Close'].shift(1) >= df['VWAP'].shift(1)) & (df['Close'] < df['VWAP']), 'Signal'] = 'Sell'

    return df[['Close', 'Volume', 'VWAP', 'Signal']]

def atr_trailing_stop(data, atr_period=14, atr_multiplier=2):
    df = data.copy()

    # Calculate True Range (TR)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    # Calculate Average True Range (ATR)
    df['ATR'] = df['TR'].ewm(span=atr_period, adjust=False).mean()

    # Initialize Trailing Stops and Signals
    df['Buy_Stop'] = np.nan
    df['Sell_Stop'] = np.nan
    df['Signal'] = 'Hold'

    long_position = False
    short_position = False

    for i in range(1, len(df)):
        current_atr = df['ATR'].iloc[i]
        current_close = df['Close'].iloc[i]
        prev_close = df['Close'].iloc[i-1]
        prev_buy_stop = df['Buy_Stop'].iloc[i-1]
        prev_sell_stop = df['Sell_Stop'].iloc[i-1]

        if not long_position and not short_position: # No active position
            # Potential Buy Signal (e.g., price moves up significantly)
            if current_close > prev_close and (current_close - prev_close) > (current_atr * atr_multiplier / 2):
                long_position = True
                df.loc[df.index[i], 'Signal'] = 'Buy'
                df.loc[df.index[i], 'Buy_Stop'] = current_close - (current_atr * atr_multiplier)
            # Potential Sell Signal (e.g., price moves down significantly)
            elif current_close < prev_close and (prev_close - current_close) > (current_atr * atr_multiplier / 2):
                short_position = True
                df.loc[df.index[i], 'Signal'] = 'Sell'
                df.loc[df.index[i], 'Sell_Stop'] = current_close + (current_atr * atr_multiplier)

        elif long_position: # Long position active
            new_buy_stop = current_close - (current_atr * atr_multiplier)
            # Guard against NaN in prev_buy_stop (first bar entering long position).
            df.loc[df.index[i], 'Buy_Stop'] = new_buy_stop if np.isnan(prev_buy_stop) else max(prev_buy_stop, new_buy_stop)
            if current_close < df['Buy_Stop'].iloc[i]: # Stop loss hit
                long_position = False
                df.loc[df.index[i], 'Signal'] = 'Sell'
            else:
                df.loc[df.index[i], 'Signal'] = 'Hold'

        elif short_position: # Short position active
            new_sell_stop = current_close + (current_atr * atr_multiplier)
            # Guard against NaN in prev_sell_stop (first bar entering short position).
            df.loc[df.index[i], 'Sell_Stop'] = new_sell_stop if np.isnan(prev_sell_stop) else min(prev_sell_stop, new_sell_stop)
            if current_close > df['Sell_Stop'].iloc[i]: # Stop loss hit
                short_position = False
                df.loc[df.index[i], 'Signal'] = 'Buy'
            else:
                df.loc[df.index[i], 'Signal'] = 'Hold'

    return df[['Close', 'ATR', 'Buy_Stop', 'Sell_Stop', 'Signal']]


def backtest_strategy(df: pd.DataFrame,
                      signal_col: str = 'Signal',
                      close_col: str = 'Close',
                      fee_bps: float = 0.0,
                      slippage_bps: float = 0.0,
                      initial_cash: float | None = None,
                      stop_loss_pct: float | None = None,
                      take_profit_pct: float | None = None,
                      max_hold_bars: int | None = None) -> dict:
    """
    Long-only backtest engine for any strategy that emits Buy / Sell / Hold signals.

    Execution model
    ---------------
    * A 'Buy'  signal at bar *i* → enter long at the close of bar *i*.
    * A 'Sell' signal at bar *i* → exit  long at the close of bar *i*.
    * One position at a time; no short-selling.
        * Any open position still active at the last bar is closed at that bar's close.
        * Optional execution frictions and risk constraints can be applied:
            - fee_bps + slippage_bps on every entry/exit
            - stop-loss and take-profit exits
            - maximum holding period in bars

    Parameters
    ----------
    df         : DataFrame that contains *signal_col* and *close_col*.
    signal_col : Name of the column with 'Buy' / 'Sell' / 'Hold' strings.
    close_col  : Name of the close-price column.
    fee_bps    : One-way transaction fee in basis points (0.0 = disabled).
    slippage_bps : One-way slippage in basis points (0.0 = disabled).
    initial_cash : Optional starting capital. If provided, orders are cash-constrained:
                   a Buy only executes when cash is sufficient to buy shares and pay
                   entry costs (fee + slippage). When omitted, legacy one-unit mode is used.
    stop_loss_pct : Exit if trade return falls below -stop_loss_pct.
    take_profit_pct : Exit if trade return rises above take_profit_pct.
    max_hold_bars : Exit after this many bars in a position.

    Returns
    -------
    dict with:
        total_return_pct      – total strategy return over the period (%).
        annualized_return_pct – CAGR, assuming 252 trading days per year (%).
        sharpe_ratio          – annualised Sharpe ratio (risk-free rate = 0).
        max_drawdown_pct      – worst peak-to-trough equity drawdown (≤ 0, %).
        win_rate_pct          – share of closed trades with positive return (%).
        profit_factor         – gross profit / gross loss (inf if no losing trades).
        expectancy_pct        – mean return per trade (%).
        n_trades              – number of completed round-trip trades.
        trade_returns         – list of individual trade returns as fractions.
    """
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("fee_bps and slippage_bps must be non-negative")
    if initial_cash is not None and initial_cash <= 0:
        raise ValueError("initial_cash must be positive when provided")
    if stop_loss_pct is not None and stop_loss_pct <= 0:
        raise ValueError("stop_loss_pct must be positive when provided")
    if take_profit_pct is not None and take_profit_pct <= 0:
        raise ValueError("take_profit_pct must be positive when provided")
    if max_hold_bars is not None and max_hold_bars <= 0:
        raise ValueError("max_hold_bars must be a positive integer when provided")

    prices  = df[close_col].to_numpy(dtype=float)
    signals = df[signal_col].to_numpy()
    n       = len(prices)

    one_way_cost_rate = (fee_bps + slippage_bps) / 10000.0

    if n < 2:
        return dict(total_return_pct=0.0, annualized_return_pct=0.0,
                    sharpe_ratio=0.0, max_drawdown_pct=0.0,
                    win_rate_pct=0.0, profit_factor=0.0,
                    expectancy_pct=0.0, n_trades=0, trade_returns=[])

    trade_returns: list = []
    skipped_buys_due_to_cash = 0

    if initial_cash is not None:
        # Cash-constrained mode: buy only if there is enough cash to cover
        # notional and transaction costs.
        cash = float(initial_cash)
        shares = 0.0
        in_position = False
        entry_total_cost = 0.0
        entry_price_raw = 0.0
        entry_index = -1
        equity = np.zeros(n, dtype=float)

        for i in range(n):
            price = prices[i]

            if in_position:
                bars_held = i - entry_index
                gross_ret = (price / entry_price_raw) - 1.0
                risk_exit = False

                if stop_loss_pct is not None and gross_ret <= -stop_loss_pct:
                    risk_exit = True
                elif take_profit_pct is not None and gross_ret >= take_profit_pct:
                    risk_exit = True
                elif max_hold_bars is not None and bars_held >= max_hold_bars:
                    risk_exit = True

                if risk_exit:
                    exit_value = shares * price * (1.0 - one_way_cost_rate)
                    trade_ret = (
                        (exit_value - entry_total_cost) / entry_total_cost
                        if entry_total_cost > 0 else 0.0
                    )
                    trade_returns.append(trade_ret)
                    cash += exit_value
                    shares = 0.0
                    in_position = False
                    entry_total_cost = 0.0
                    entry_price_raw = 0.0
                    entry_index = -1

            if signals[i] == 'Buy' and not in_position and price > 0:
                per_share_cost = price * (1.0 + one_way_cost_rate)
                max_shares = int(cash // per_share_cost)
                if max_shares >= 1:
                    shares = float(max_shares)
                    entry_total_cost = shares * per_share_cost
                    cash -= entry_total_cost
                    in_position = True
                    entry_price_raw = price
                    entry_index = i
                else:
                    skipped_buys_due_to_cash += 1
            elif signals[i] == 'Sell' and in_position and price > 0:
                exit_value = shares * price * (1.0 - one_way_cost_rate)
                trade_ret = (
                    (exit_value - entry_total_cost) / entry_total_cost
                    if entry_total_cost > 0 else 0.0
                )
                trade_returns.append(trade_ret)
                cash += exit_value
                shares = 0.0
                in_position = False
                entry_total_cost = 0.0
                entry_price_raw = 0.0
                entry_index = -1

            equity[i] = cash + shares * price

        if in_position:
            exit_value = shares * prices[-1] * (1.0 - one_way_cost_rate)
            trade_ret = (
                (exit_value - entry_total_cost) / entry_total_cost
                if entry_total_cost > 0 else 0.0
            )
            trade_returns.append(trade_ret)
            cash += exit_value
            equity[-1] = cash

        daily_rets = np.zeros(n, dtype=float)
        daily_rets[1:] = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1e-12)
        total_return = ((equity[-1] / initial_cash) - 1.0) * 100.0
    else:
        # Legacy one-unit mode (kept for backward compatibility with tests).
        in_position = False
        entry_price_raw = 0.0
        entry_price_exec = 0.0
        entry_index = -1
        positions = np.zeros(n)
        cost_rets = np.zeros(n)

        for i in range(n):
            if in_position:
                bars_held = i - entry_index
                gross_ret = (prices[i] / entry_price_raw) - 1.0
                risk_exit = False

                if stop_loss_pct is not None and gross_ret <= -stop_loss_pct:
                    risk_exit = True
                elif take_profit_pct is not None and gross_ret >= take_profit_pct:
                    risk_exit = True
                elif max_hold_bars is not None and bars_held >= max_hold_bars:
                    risk_exit = True

                if risk_exit:
                    exit_price_exec = prices[i] * (1.0 - one_way_cost_rate)
                    trade_returns.append((exit_price_exec - entry_price_exec) / entry_price_exec)
                    in_position = False
                    positions[i] = 1.0
                    cost_rets[i] += one_way_cost_rate
                    continue

            if signals[i] == 'Buy' and not in_position:
                in_position = True
                entry_price_raw = prices[i]
                entry_price_exec = prices[i] * (1.0 + one_way_cost_rate)
                entry_index = i
                positions[i] = 0.0
                cost_rets[i] += one_way_cost_rate
            elif signals[i] == 'Sell' and in_position:
                in_position = False
                exit_price_exec = prices[i] * (1.0 - one_way_cost_rate)
                trade_returns.append((exit_price_exec - entry_price_exec) / entry_price_exec)
                positions[i] = 1.0
                cost_rets[i] += one_way_cost_rate
            elif in_position:
                positions[i] = 1.0

        if in_position:
            exit_price_exec = prices[-1] * (1.0 - one_way_cost_rate)
            trade_returns.append((exit_price_exec - entry_price_exec) / entry_price_exec)
            cost_rets[-1] += one_way_cost_rate

        price_rets = np.concatenate(
            [[0.0], np.diff(prices) / np.where(prices[:-1] != 0, prices[:-1], 1e-12)]
        )
        daily_rets = positions * price_rets - cost_rets
        equity = np.cumprod(1.0 + daily_rets)
        total_return = (equity[-1] - 1.0) * 100.0

    # CAGR — annualised assuming 252 trading days per year
    n_years = n / 252.0
    annualized_return = (
        (equity[-1] ** (1.0 / n_years) - 1.0) * 100.0
        if equity[-1] > 0 and n_years > 0 else 0.0
    )

    # Annualised Sharpe ratio (risk-free rate = 0)
    std_dr = np.std(daily_rets)
    sharpe = (np.mean(daily_rets) / std_dr * np.sqrt(252)) if std_dr > 0 else 0.0
    
    # Maximum drawdown
    peak   = np.maximum.accumulate(equity)
    max_dd = float(((equity - peak) / peak).min()) * 100.0

    # Win rate
    n_trades = len(trade_returns)
    win_rate = (
        sum(1 for r in trade_returns if r > 0) / n_trades * 100.0
        if n_trades > 0 else 0.0
    )

    gross_profit = sum(r for r in trade_returns if r > 0)
    gross_loss = abs(sum(r for r in trade_returns if r < 0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float('inf')
    else:
        profit_factor = 0.0

    expectancy_pct = (sum(trade_returns) / n_trades * 100.0) if n_trades > 0 else 0.0

    return {
        'total_return_pct':      round(total_return,       2),
        'annualized_return_pct': round(annualized_return,  2),
        'sharpe_ratio':          round(sharpe,             3),
        'max_drawdown_pct':      round(max_dd,             2),
        'win_rate_pct':          round(win_rate,           1),
        'profit_factor':         (round(profit_factor, 3) if np.isfinite(profit_factor) else float('inf')),
        'expectancy_pct':        round(expectancy_pct,     2),
        'n_trades':              n_trades,
        'trade_returns':         trade_returns,
        'skipped_buys_due_to_cash': skipped_buys_due_to_cash,
    }

