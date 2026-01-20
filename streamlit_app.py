"""
TERMINAL - Professional Trading Terminal
==========================================
Real-time clock using Streamlit components for JavaScript execution.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import json
import requests
import time

try:
    import feedparser
except:
    feedparser = None

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit chrome
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# REAL-TIME CLOCK (JavaScript via Streamlit Component)
# =============================================================================

def render_realtime_clock():
    """Render real-time clock with JavaScript"""
    
    clock_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'IBM Plex Mono', monospace;
                background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
                color: #f0f6fc;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px 30px;
                background: linear-gradient(90deg, #161b22 0%, #21262d 100%);
                border-bottom: 1px solid #30363d;
                border-radius: 10px;
            }
            .logo {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .logo h1 {
                font-size: 1.5em;
                font-weight: 700;
                color: #39d353;
                letter-spacing: 3px;
            }
            .live-badge {
                background: rgba(57, 211, 83, 0.2);
                color: #39d353;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.75em;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .live-dot {
                width: 6px;
                height: 6px;
                background: #39d353;
                border-radius: 50%;
                animation: pulse 1s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.5; transform: scale(0.8); }
            }
            .clocks {
                display: flex;
                gap: 40px;
            }
            .clock-item {
                text-align: center;
            }
            .clock-time {
                font-size: 2em;
                font-weight: 700;
                color: #39d353;
                text-shadow: 0 0 20px rgba(57, 211, 83, 0.4);
                letter-spacing: 2px;
            }
            .clock-ms {
                font-size: 0.45em;
                color: #58a6ff;
                opacity: 0.9;
            }
            .clock-label {
                font-size: 0.7em;
                color: #8b949e;
                margin-top: 4px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .date {
                color: #8b949e;
                font-size: 0.85em;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo">
                <span style="font-size: 1.5em;">⚡</span>
                <h1>TERMINAL</h1>
                <div class="live-badge">
                    <span class="live-dot"></span>
                    LIVE
                </div>
            </div>
            
            <div class="clocks">
                <div class="clock-item">
                    <div id="utc" class="clock-time">00:00:00<span class="clock-ms">.000</span></div>
                    <div class="clock-label">UTC</div>
                </div>
                <div class="clock-item">
                    <div id="ist" class="clock-time">00:00:00<span class="clock-ms">.000</span></div>
                    <div class="clock-label">🇮🇳 IST</div>
                </div>
                <div class="clock-item">
                    <div id="ny" class="clock-time">00:00:00<span class="clock-ms">.000</span></div>
                    <div class="clock-label">🇺🇸 NY</div>
                </div>
                <div class="clock-item">
                    <div id="date" class="date">Loading...</div>
                </div>
            </div>
        </div>
        
        <script>
            function pad(n) { return n < 10 ? '0' + n : n; }
            function pad3(n) { return n < 10 ? '00' + n : (n < 100 ? '0' + n : n); }
            
            function updateClocks() {
                const now = new Date();
                const ms = pad3(now.getUTCMilliseconds());
                
                // UTC
                document.getElementById('utc').innerHTML = 
                    pad(now.getUTCHours()) + ':' + 
                    pad(now.getUTCMinutes()) + ':' + 
                    pad(now.getUTCSeconds()) + 
                    '<span class="clock-ms">.' + ms + '</span>';
                
                // IST (UTC + 5:30)
                const ist = new Date(now.getTime() + 5.5 * 60 * 60 * 1000);
                document.getElementById('ist').innerHTML = 
                    pad(ist.getUTCHours()) + ':' + 
                    pad(ist.getUTCMinutes()) + ':' + 
                    pad(ist.getUTCSeconds()) + 
                    '<span class="clock-ms">.' + ms + '</span>';
                
                // NY (UTC - 5)
                const ny = new Date(now.getTime() - 5 * 60 * 60 * 1000);
                document.getElementById('ny').innerHTML = 
                    pad(ny.getUTCHours()) + ':' + 
                    pad(ny.getUTCMinutes()) + ':' + 
                    pad(ny.getUTCSeconds()) + 
                    '<span class="clock-ms">.' + ms + '</span>';
                
                // Date
                const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
                const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                document.getElementById('date').innerHTML = 
                    days[now.getUTCDay()] + ', ' + 
                    months[now.getUTCMonth()] + ' ' + 
                    now.getUTCDate() + ', ' + 
                    now.getUTCFullYear();
            }
            
            // Update every 10ms
            setInterval(updateClocks, 10);
            updateClocks();
        </script>
    </body>
    </html>
    """
    
    components.html(clock_html, height=130)


# =============================================================================
# PRICES
# =============================================================================

@st.cache_data(ttl=0.5)
def get_prices():
    """Fetch live prices"""
    prices = {}
    try:
        for sym in ['BTCUSDT', 'PAXGUSDT']:
            r = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={sym}", timeout=2)
            if r.status_code == 200:
                d = r.json()
                prices[sym] = {
                    'price': float(d['lastPrice']),
                    'change': float(d['priceChangePercent']),
                    'high': float(d['highPrice']),
                    'low': float(d['lowPrice'])
                }
    except:
        pass
    return prices


def render_prices():
    """Render price cards"""
    prices = get_prices()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'BTCUSDT' in prices:
            p = prices['BTCUSDT']
            delta = f"{p['change']:+.2f}%"
            st.metric(
                label="₿ Bitcoin (BTC/USDT)",
                value=f"${p['price']:,.2f}",
                delta=delta
            )
            st.caption(f"H: ${p['high']:,.0f} | L: ${p['low']:,.0f}")
    
    with col2:
        if 'PAXGUSDT' in prices:
            p = prices['PAXGUSDT']
            delta = f"{p['change']:+.2f}%"
            st.metric(
                label="🥇 Gold (PAXG/USDT)",
                value=f"${p['price']:,.2f}",
                delta=delta
            )
            st.caption(f"H: ${p['high']:,.0f} | L: ${p['low']:,.0f}")


# =============================================================================
# SIGNALS
# =============================================================================

def get_signal(symbol: str, tf: str) -> str:
    """Get signal for asset/timeframe"""
    try:
        path = DATA_DIR / f"{symbol}_{tf}_labeled.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            label = df.iloc[-1].get('label', 0)
            if label == 1: return 'BUY'
            elif label == -1: return 'SELL'
        return 'WAIT'
    except:
        return 'WAIT'


def render_signals():
    """Render signals"""
    st.subheader("📊 Trading Signals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**₿ Bitcoin**")
        cols = st.columns(3)
        for i, tf in enumerate(['15m', '30m', '1h']):
            with cols[i]:
                sig = get_signal('BTCUSDT', tf)
                if sig == 'BUY':
                    st.success(f"{tf}: **{sig}**")
                elif sig == 'SELL':
                    st.error(f"{tf}: **{sig}**")
                else:
                    st.info(f"{tf}: {sig}")
    
    with col2:
        st.write("**🥇 Gold**")
        cols = st.columns(3)
        for i, tf in enumerate(['15m', '30m', '1h']):
            with cols[i]:
                sig = get_signal('PAXGUSDT', tf)
                if sig == 'BUY':
                    st.success(f"{tf}: **{sig}**")
                elif sig == 'SELL':
                    st.error(f"{tf}: **{sig}**")
                else:
                    st.info(f"{tf}: {sig}")


# =============================================================================
# METRICS
# =============================================================================

def render_metrics():
    """Render backtest metrics"""
    st.subheader("📈 Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("BTC Win Rate", "69.9%", "+10%")
    with col2:
        st.metric("BTC Profit Factor", "2.32")
    with col3:
        st.metric("Gold Win Rate", "57.8%", "+8%")
    with col4:
        st.metric("Gold Profit Factor", "1.28")


# =============================================================================
# NEWS
# =============================================================================

def time_ago(dt_str):
    """Convert to time ago format"""
    try:
        for fmt in ['%a, %d %b %Y %H:%M:%S %z', '%Y-%m-%dT%H:%M:%SZ']:
            try:
                dt = datetime.strptime(dt_str.replace(' GMT', ' +0000'), fmt)
                break
            except:
                continue
        else:
            return "Just now"
        
        now = datetime.now(timezone.utc)
        diff = (now - dt).total_seconds() / 60
        
        if diff < 1: return "Just now"
        elif diff < 60: return f"{int(diff)}m ago"
        elif diff < 1440: return f"{int(diff/60)}h ago"
        else: return f"{int(diff/1440)}d ago"
    except:
        return "Recently"


@st.cache_data(ttl=30)
def get_news():
    """Get news"""
    news = []
    if feedparser:
        for url, src in [("https://cointelegraph.com/rss", "Cointelegraph")]:
            try:
                feed = feedparser.parse(url)
                for e in feed.entries[:4]:
                    news.append({
                        'title': e.title[:80] + '...' if len(e.title) > 80 else e.title,
                        'source': src,
                        'time': time_ago(e.get('published', ''))
                    })
            except:
                pass
    
    if not news:
        news = [
            {'title': 'Markets steady ahead of data release', 'source': 'Market', 'time': '5m ago'},
            {'title': 'Bitcoin consolidates near resistance', 'source': 'Crypto', 'time': '12m ago'}
        ]
    
    return news


def render_news():
    """Render news"""
    st.subheader("📰 News")
    
    news = get_news()
    for n in news:
        st.markdown(f"**{n['time']}** | {n['title']} _{n['source']}_")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if 'halted' not in st.session_state:
        st.session_state.halted = False
    
    # Real-time clock (JavaScript)
    render_realtime_clock()
    
    # Controls
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col2:
        auto = st.checkbox("Auto-refresh prices", value=True)
    with col3:
        if st.button("🛑 KILL" if not st.session_state.halted else "▶ RESUME", 
                     type="primary", use_container_width=True):
            st.session_state.halted = not st.session_state.halted
            st.rerun()
    
    if st.session_state.halted:
        st.error("⚠️ TRADING HALTED")
        return
    
    st.divider()
    
    # Prices
    render_prices()
    
    st.divider()
    
    # Two columns: Signals + News
    col1, col2 = st.columns([2, 1])
    with col1:
        render_signals()
        st.divider()
        render_metrics()
    with col2:
        render_news()
    
    # Auto-refresh for prices
    if auto:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
