import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
from zoneinfo import ZoneInfo


def convert_index_to_athens(df):
    """
    Goal:
        Convert DataFrame DatetimeIndex to the Europe/Athens timezone.

    Execution Principle:
        1. Create an isolated copy of the input DataFrame.
        2. Check if the index is a `pd.DatetimeIndex`; if not, return unmodified copy.
        3. If timezone is naive, localize index to UTC.
        4. Convert localized timestamps to `Europe/Athens` using `ZoneInfo`.
        5. Return DataFrame with updated DatetimeIndex.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        return out

    if out.index.tz is None:
        # yfinance intraday data is typically UTC when tz info is missing
        out.index = out.index.tz_localize("UTC")

    out.index = out.index.tz_convert(ZoneInfo("Europe/Athens"))
    return out

# Too much noise in signals; tests should proceed only on hour interval data
def identify_gap_fills(df, min_volume=None):
    """
    Goal:
        Detect opening price gaps between consecutive bars and evaluate whether the gap was filled during the bar.

    Execution Principle:
        1. Calculate previous bar close using `df['Close'].shift(1)`.
        2. Compute gap size: `Gap_Size = Open - Prev_Close`.
        3. Classify gap type: 'Up' if Gap_Size > 0, 'Down' if Gap_Size < 0, else 'None'.
        4. Detect fills:
           - Gap Up fills if bar `Low <= Prev_Close`.
           - Gap Down fills if bar `High >= Prev_Close`.
        5. Apply optional volume filter: if `min_volume` is provided, reset `Filled` to False for low-volume bars.
        6. Return annotated DataFrame with Gap_Size, Gap_Type, and Filled columns.
    """
    df = df.copy()
    
    # 1. Calculate Gap: Today's / Current Bar's Open vs. Previous Bar's Close
    df['Prev_Close'] = df['Close'].shift(1)
    df['Gap_Size'] = df['Open'] - df['Prev_Close']
    
    # 2. Categorize Gaps
    df['Gap_Type'] = np.where(df['Gap_Size'] > 0, 'Up', 
                             np.where(df['Gap_Size'] < 0, 'Down', 'None'))

    # 3. Same-Day / Same-Bar Fill Logic
    # Gap Up fills if Low drops to Previous Close
    # Gap Down fills if High rises to Previous Close
    df['Filled'] = False
    df.loc[(df['Gap_Type'] == 'Up') & (df['Low'] <= df['Prev_Close']), 'Filled'] = True
    df.loc[(df['Gap_Type'] == 'Down') & (df['High'] >= df['Prev_Close']), 'Filled'] = True

    # 4. Volume Filter
    if min_volume is not None and 'Volume' in df.columns:
        df.loc[df['Volume'] < min_volume, 'Filled'] = False

    return df

# Example Usage: Download real data from yfinance

# Download historical data (1h interval)
ticker = "AAPL"
start_date = "2026-05-01"
end_date = "2026-05-21"
interval = "1h"

df = yf.download(ticker, start=start_date, end=end_date, progress=False, interval=interval)

# Flatten MultiIndex columns returned by newer yfinance versions
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Ensure required columns are present
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df = convert_index_to_athens(df)

result = identify_gap_fills(df)
print(result)

# --- Chart ---
# Build addplot markers for filled gaps
filled_up   = result['Low'].where((result['Gap_Type'] == 'Up')   & result['Filled'])
filled_down = result['High'].where((result['Gap_Type'] == 'Down') & result['Filled'])

apds = []
if filled_up.notna().any():
    apds.append(mpf.make_addplot(filled_up, type='scatter', markersize=60, marker='^', color='green'))
if filled_down.notna().any():
    apds.append(mpf.make_addplot(filled_down, type='scatter', markersize=60, marker='v', color='red'))

mpf.plot(
    result[['Open', 'High', 'Low', 'Close', 'Volume']],
    type='candle',
    style='charles',
    title=f'{ticker} – Gap Fills {start_date} to {end_date} ({interval})',
    ylabel='Price (USD)',
    volume=True,
    addplot=apds if apds else None,
    figsize=(16, 8),
    show_nontrading=False,
)

