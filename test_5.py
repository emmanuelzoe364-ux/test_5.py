import os
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging 

# --- Configuration & Setup ---
TICKERS = ["BTC-USD", "ETH-USD"]
ASSET_NAME = "BTC/ETH Combined Index (50/50)"

# Configure basic logging to console (simulating an external API call log)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="BTC/ETH Index Signal Tracker", layout="wide")
st.title(f"🚀 {ASSET_NAME} Realignment Signal Tracker")

# Initialize session state for persistent signal tracking (for simulating alerts)
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = pd.Timestamp.min.tz_localize(None)

# -----------------------
# Helper function for external alert (SIMULATION)
# -----------------------

def send_external_alert(signal_type: str, message: str, email: str, phone: str):
    """Simulates sending an external alert via Email/SMS API."""
    if email or phone:
        logging.info(f"*** EXTERNAL ALERT SENT (Simulated) ***")
        if email:
            logging.info(f"EMAIL To: {email}")
        if phone:
            logging.info(f"SMS To: {phone}")
        logging.info(f"CONTENT: {message.replace('\n', ' | ')}")
    else:
        logging.info("External alert skipped: No email or phone recipient configured.")


# -----------------------
# Sidebar / user inputs
# -----------------------
st.sidebar.header("Settings")

st.sidebar.markdown(f"**Assets Tracked:** {', '.join(TICKERS)}")

# FIX 1: Default period reduced to 7 days for better intraday fetching success
period_days = st.sidebar.number_input("Fetch period (days)", min_value=7, max_value=365, value=7) 
interval = st.sidebar.selectbox("Interval", options=["15m","30m","1h"], index=2)
rsi_length = st.sidebar.number_input("RSI length", min_value=7, max_value=30, value=14)
ema_short = st.sidebar.number_input("Trend EMA short (for signal)", min_value=5, max_value=30, value=14)
ema_long = st.sidebar.number_input("Trend EMA long (for signal)", min_value=10, max_value=72, value=30)

# EMA span for the Cumulative Index itself
cumulative_ema_span = st.sidebar.number_input("Cumulative Index EMA Span", min_value=1, max_value=20, value=5)


cache_file = st.sidebar.text_input("Cache filename (Internal use only)", value="btc_eth_index_cache.csv") 
refresh = st.sidebar.number_input("Auto refresh (sec)", min_value=30, max_value=3600, value=300)
min_bars_after_cycle = st.sidebar.number_input("Max bars to look for re-alignment after cycle (0 = unlimited)", min_value=0, max_value=9999, value=0)
# New input for volume filtering
volume_length = st.sidebar.number_input("Combined Volume MA Length", min_value=1, max_value=50, value=14)
enable_volume_filter = st.sidebar.checkbox("Require Combined Volume Confirmation", value=False)


st.sidebar.markdown("---")
st.sidebar.header("External Notification Settings")
recipient_email = st.sidebar.text_input("Recipient Email (for simulation)", value="")
recipient_phone = st.sidebar.text_input("Recipient Phone (for simulation, e.g., +15551234)", value="")
st.sidebar.markdown("_The external alerts are simulated via logging. To make them real, you'd integrate a service like Twilio or SendGrid._")
st.markdown("---")

st.sidebar.markdown("RSI cycle rules: **rising** = cross up 29 → later cross up 71. **falling** = cross down 71 → later cross down 29.")
st.sidebar.markdown("Signals fire only after a completed cycle + Normalized Index dip/spike + EMA reclaim while EMA alignment holds.")


# -----------------------
# Helper functions for calculations
# -----------------------

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).
    Uses Exponential Moving Average (EMA) smoothing for responsiveness (EMA-RSI).
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    
    # *** CHANGE: Using EWM (EMA) instead of rolling().mean() (SMA) for faster smoothing ***
    # This makes the RSI more reactive to recent changes.
    ma_up = up.ewm(span=length, adjust=False, min_periods=1).mean()
    ma_down = down.ewm(span=length, adjust=False, min_periods=1).mean()
    
    rs = ma_up / (ma_down + 1e-10) 
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi

# -----------------------
# 🚀 Data Fetching and Processing (Optimized with st.cache_data)
# -----------------------

@st.cache_data(ttl=timedelta(seconds=refresh))
def fetch_and_process_data(period_days: int, interval: str):
    """
    Fetches BTC and ETH data, normalizes them, and creates a Combined Index.
    """
    
    status_text = st.empty()
    status_text.info(f"Fetching {TICKERS} {interval} data for {period_days} days (using yfinance). Cache TTL: {refresh} seconds.")
    
    # Enforce max period for intraday data
    current_period_days = period_days
    if interval in ["15m", "30m", "1h"]:
        MAX_INTRADAY_DAYS = 60 # Safe max for 1h and below
        if current_period_days > MAX_INTRADAY_DAYS:
            current_period_days = MAX_INTRADAY_DAYS
            status_text.warning(
                f"Intraday data ({interval}) from yfinance is limited to approximately {MAX_INTRADAY_DAYS} days. "
                f"Period reduced to {MAX_INTRADAY_DAYS} days for successful fetching."
            )

    period = f"{current_period_days}d"
    
    try:
        # Fetch data for multiple tickers
        raw = yf.download(TICKERS, start=pd.Timestamp.now() - timedelta(days=current_period_days), end=pd.Timestamp.now(), 
                          interval=interval, progress=False)
        
        if raw.empty or 'Close' not in raw.columns or len(raw['Close'].columns) != 2:
             error_msg = (
                 f"Fetched data is incomplete or empty for {TICKERS} at {interval} interval. "
                 "Ensure the tickers are correct and try reducing the 'Fetch period (days)'."
             )
             status_text.error(error_msg)
             return pd.DataFrame()

        df_close = raw['Close'].ffill().bfill()
        df_volume = raw['Volume'].ffill().bfill()
        
    except Exception as e:
        status_text.error(f"Failed during data fetch or initial access for {TICKERS}: {e}. Check network connection or try a manual refresh.")
        return pd.DataFrame()

    # Ensure index is timezone-naive
    if df_close.index.tz is not None:
        df_close.index = df_close.index.tz_localize(None)
    if df_volume.index.tz is not None:
        df_volume.index = df_volume.index.tz_localize(None)

    # --- CUMULATIVE TRACKER LOGIC ---
    base_prices = df_close.iloc[0]
    
    # 1. Normalize BTC and ETH individually
    normalized_btc = df_close['BTC-USD'] / base_prices['BTC-USD']
    normalized_eth = df_close['ETH-USD'] / base_prices['ETH-USD']
    
    # 2. Create the Combined Normalized Index (Equally Weighted)
    combined_index = (normalized_btc + normalized_eth) / 2
    
    # 3. Combined Volume (sum of trading activity)
    combined_volume = df_volume['BTC-USD'] + df_volume['ETH-USD']

    # Final DataFrame
    df = pd.DataFrame({
        'index_cum': combined_index,    # Raw combined index series
        'btc_cum': normalized_btc,      # BTC component
        'eth_cum': normalized_eth,      # ETH component
        'volume': combined_volume       # Combined Volume for confirmation
    }, index=combined_index.index)
    
    status_text.empty() # Clear the status message
    return df

# Fetch data using the cached function
df = fetch_and_process_data(period_days, interval)

if df.empty:
    st.stop()

# --- CUMULATIVE INDEX SMOOTHING (EMA) ---
# Apply EMA to the raw combined index.
df['index_cum_smooth'] = df['index_cum'].ewm(span=cumulative_ema_span, adjust=False).mean()

# -----------------------
# Indicators & Cycles (Calculated on the Combined Index EMA)
# -----------------------
df['EMA_short'] = df['index_cum_smooth'].ewm(span=ema_short, adjust=False).mean()
df['EMA_long'] = df['index_cum_smooth'].ewm(span=ema_long, adjust=False).mean() 
df['RSI'] = rsi(df['index_cum_smooth'], length=rsi_length)
# Volume Indicator: Simple Moving Average of Combined Volume
df['Volume_MA'] = df['volume'].rolling(volume_length, min_periods=1).mean()


cycle_id = 0
in_cycle = False
cycle_type = None
cycle_start_idx = None
cycles = [] 

df['cycle_id'] = np.nan
df['cycle_type'] = None
rsi_series = df['RSI']

df_index_list = df.index.to_list() 

if len(df_index_list) > 1:
    prev_rsi = rsi_series.iloc[0]
    for idx in df_index_list[1:]:
        cur_rsi = rsi_series.loc[idx]
        
        # Cycle start detection...
        if not in_cycle:
            if (prev_rsi <= 29) and (cur_rsi > 29):
                in_cycle = True; cycle_type = 'rising'; cycle_start_idx = idx; cycle_id += 1
            elif (prev_rsi >= 71) and (cur_rsi < 71):
                in_cycle = True; cycle_type = 'falling'; cycle_start_idx = idx; cycle_id += 1
        # Cycle end detection...
        else:
            if cycle_type == 'rising' and (prev_rsi < 71) and (cur_rsi >= 71):
                cycles.append({'id': cycle_id, 'type': 'rising', 'start': cycle_start_idx, 'end': idx})
                df.loc[cycle_start_idx:idx, 'cycle_id'] = cycle_id
                df.loc[cycle_start_idx:idx, 'cycle_type'] = 'rising'
                in_cycle = False; cycle_type = None; cycle_start_idx = None
            elif cycle_type == 'falling' and (prev_rsi > 29) and (cur_rsi <= 29):
                cycles.append({'id': cycle_id, 'type': 'falling', 'start': cycle_start_idx, 'end': idx})
                df.loc[cycle_start_idx:idx, 'cycle_id'] = cycle_id
                df.loc[cycle_start_idx:idx, 'cycle_type'] = 'falling'
                in_cycle = False; cycle_type = None; cycle_start_idx = None
        prev_rsi = cur_rsi

# -----------------------
# Realignment detection and signal setting
# -----------------------
df['signal'] = 0 # 1 buy, -1 sell
df['signal_reason'] = None

# Volume check function
def check_volume_confirmation(idx):
    if not enable_volume_filter:
        return True # Filter disabled, always allow signal
    
    # Check if current volume is greater than its average (Volume_MA)
    return df.at[idx, 'volume'] > df.at[idx, 'Volume_MA']

for c in cycles:
    end_idx = c['end']
    search_idx_list = df.loc[end_idx:].index.to_list()
    if len(search_idx_list) <= 1:
        continue
    
    if min_bars_after_cycle > 0:
        search_idx_list = search_idx_list[1:min_bars_after_cycle+2] 
    else:
        search_idx_list = search_idx_list[1:]

    dipped = False; spiked = False
    
    if c['type'] == 'rising':
        dip_idx = None; reclaim_idx = None
        for t in search_idx_list:
            # Look for dip below EMA long
            if (not dipped) and (df.at[t, 'index_cum_smooth'] < df.at[t, 'EMA_long']):
                dipped = True; dip_idx = t
            # Look for reclaim above EMA long
            if dipped and (df.at[t, 'index_cum_smooth'] > df.at[t, 'EMA_long']):
                reclaim_idx = t
                
                # FINAL BUY CONDITIONS: EMA alignment AND optional Volume confirmation
                if (df.at[reclaim_idx, 'EMA_short'] > df.at[reclaim_idx, 'EMA_long']) and check_volume_confirmation(reclaim_idx):
                    df.at[reclaim_idx, 'signal'] = 1
                    vol_note = " (Vol Confirmed)" if enable_volume_filter else ""
                    df.at[reclaim_idx, 'signal_reason'] = f"Buy: end rising cycle {c['id']} dip@{dip_idx.strftime('%H:%M')} reclaim@{reclaim_idx.strftime('%H:%M')}{vol_note}"
                    break
                else: break
                    
    elif c['type'] == 'falling':
        spike_idx = None; drop_idx = None
        for t in search_idx_list:
            # Look for spike above EMA long
            if (not spiked) and (df.at[t, 'index_cum_smooth'] > df.at[t, 'EMA_long']):
                spiked = True; spike_idx = t
            # Look for drop below EMA long
            if spiked and (df.at[t, 'index_cum_smooth'] < df.at[t, 'EMA_long']):
                drop_idx = t
                
                # FINAL SELL CONDITIONS: EMA alignment AND optional Volume confirmation
                if (df.at[drop_idx, 'EMA_short'] < df.at[drop_idx, 'EMA_long']) and check_volume_confirmation(drop_idx):
                    df.at[drop_idx, 'signal'] = -1
                    vol_note = " (Vol Confirmed)" if enable_volume_filter else ""
                    df.at[drop_idx, 'signal_reason'] = f"Sell: end falling cycle {c['id']} spike@{spike_idx.strftime('%H:%M')} drop@{drop_idx.strftime('%H:%M')}{vol_note}"
                    break
                else: break

# -----------------------
# Real-time Alerting (External + Internal)
# -----------------------
latest_signal = df[df['signal'] != 0].tail(1)

if not latest_signal.empty:
    latest_time = latest_signal.index[0] 
    signal_value = latest_signal['signal'].iloc[0]
    signal_type = "BUY" if signal_value == 1 else "SELL"
    
    if latest_time > st.session_state.last_signal_time:
        st.session_state.last_signal_time = latest_time
        
        # --- 1. Internal Alert Message (in the app) ---
        alert_message = (
            f"🔔 **NEW ALERT ({signal_type})**: Cycle Realignment Signal Fired for {ASSET_NAME}!\n\n"
            f"**Time**: {latest_time.strftime('%Y-%m-%d %H:%M:%S')} ({interval})\n"
            f"**Action**: {signal_type}\n"
            f"**Reason**: {latest_signal['signal_reason'].iloc[0]}"
        )
        st.error(alert_message, icon="🚨") 

        # --- 2. External Alert Generation (Simulated) ---
        external_message = f"{ASSET_NAME} RSI ALERT ({interval}): {signal_type} at {latest_time.strftime('%H:%M')}. Reason: {latest_signal['signal_reason'].iloc[0]}."
        send_external_alert(signal_type, external_message, recipient_email, recipient_phone)


# -----------------------
# Plotting: main chart + RSI subplot with cycle shading
# -----------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.25], vertical_spacing=0.05,
                    specs=[[{"secondary_y": False}], [{"secondary_y": False}]])
# top: Price + EMAs
# 1. Combined Index EMA
fig.add_trace(go.Scatter(x=df.index, y=df['index_cum_smooth'], mode='lines', 
                         name=f'{ASSET_NAME} (EMA {cumulative_ema_span})', line=dict(color='#0077b6', width=3)), row=1, col=1)
# 2. Individual Components for context
# FIX: Using RGBA color strings to handle opacity directly within the line dictionary.
fig.add_trace(go.Scatter(x=df.index, y=df['btc_cum'], mode='lines', name='BTC-USD Normalized (Raw)', 
                         line=dict(color='rgba(247, 147, 26, 0.5)', dash='dash')), row=1, col=1) 
fig.add_trace(go.Scatter(x=df.index, y=df['eth_cum'], mode='lines', name='ETH-USD Normalized (Raw)', 
                         line=dict(color='rgba(140, 140, 140, 0.5)', dash='dot')), row=1, col=1)

# EMAs on the Combined Index EMA
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_short'], mode='lines', name=f'Trend EMA {ema_short}', line=dict(color='orange', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_long'], mode='lines', name=f'Trend EMA {ema_long}', line=dict(color='red', width=2)), row=1, col=1)
fig.add_hline(y=1.0, line=dict(color='gray', dash='dash'), row=1, col=1) # Baseline for Normalized Price
fig.update_yaxes(title="Normalized Price (Base 1.0)", row=1, col=1)


# signals markers
buys = df[df['signal'] == 1]
sells = df[df['signal'] == -1]
if not buys.empty:
    # Use the Index EMA for marker placement
    fig.add_trace(go.Scatter(x=buys.index, y=buys['index_cum_smooth'], mode='markers', marker_symbol='triangle-up',
                             marker_color='green', marker_size=12, name='BUY', marker_line_width=1), row=1, col=1)
if not sells.empty:
    # Use the Index EMA for marker placement
    fig.add_trace(go.Scatter(x=sells.index, y=sells['index_cum_smooth'], mode='markers', marker_symbol='triangle-down',
                             marker_color='red', marker_size=12, name='SELL', marker_line_width=1), row=1, col=1)

# Cycle shading on price chart
for c in cycles:
    start = c['start']
    end = c['end']
    color = "rgba(0,255,0,0.06)" if c['type']=='rising' else "rgba(255,0,0,0.06)"
    fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.4, line_width=0, row=1, col=1)

# bottom: RSI
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name=f'RSI({rsi_length}) (Combined Index EMA)', line=dict(color='black')), row=2, col=1)
# add horizontal lines 29 & 71
fig.add_hline(y=29, line=dict(color='grey', dash='dash'), row=2, col=1)
fig.add_hline(y=71, line=dict(color='grey', dash='dash'), row=2, col=1)
fig.update_yaxes(range=[0, 100], row=2, col=1) # force RSI range

# Cycle shading on RSI subplot
for c in cycles:
    start = c['start']; end = c['end']
    color = "rgba(0,255,0,0.06)" if c['type']=='rising' else "rgba(255,0,0,0.06)"
    fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.3, line_width=0, row=2, col=1)

fig.update_layout(title=f"{ASSET_NAME} Realignment Signals Dashboard",
                  xaxis=dict(rangeslider=dict(visible=False)), height=800, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Show table of signals and diagnostics
# -----------------------
st.markdown("### Signals and Diagnostics (Based on Combined Index EMA)")
if df['signal'].abs().sum() == 0:
    st.info("No signals found in the selected period with current parameters. Try adjusting settings.")
else:
    # Diagnostic table includes the components and combined volume
    sig_df = df[df['signal'] != 0][['index_cum_smooth','btc_cum', 'eth_cum', 'EMA_short','EMA_long','RSI','volume', 'Volume_MA', 'cycle_id','cycle_type','signal_reason','signal']].copy()
    sig_df.index = sig_df.index.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(sig_df.tail(50))

# small metrics
st.markdown("### Summary")
st.write(f"Total cycles detected for {ASSET_NAME}: **{len(cycles)}**")
st.write(f"Total signals detected for {ASSET_NAME}: **{int(df['signal'].abs().sum())}**")
st.write(f"Last signal timestamp recorded: **{st.session_state.last_signal_time.strftime('%Y-%m-%d %H:%M:%S')}** (Used to prevent duplicate alerts.)")

# -----------------------
# Auto-Refresh / Manual Refresh
# -----------------------
st.markdown("---")
col_button, col_timer = st.columns([1, 4])

# Refresh button
if col_button.button(f"🔄 Refresh / Re-fetch {ASSET_NAME} Data"):
    fetch_and_process_data.clear()
    st.experimental_rerun()

# Auto-refresh timer logic
placeholder = col_timer.empty()
if refresh > 0:
    for i in range(refresh, 0, -1):
        with placeholder.container():
            st.markdown(f"Next auto-refresh in **{i}** seconds...")
        time.sleep(1)
    
    fetch_and_process_data.clear()
    st.experimental_rerun()
else:
    placeholder.markdown("Auto refresh is **disabled**.")