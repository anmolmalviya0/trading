"""
FORGE DASHBOARD - WITH HONEST BACKTEST RESULTS
================================================
Shows per-asset results with TP/SL levels
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timezone
import yaml
import json
import time

st.set_page_config(page_title="Forge Trading System", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

@st.cache_data(ttl=300)
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

@st.cache_data(ttl=3)
def fetch_prices():
    prices = {}
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'PAXGUSDT']:
        try:
            r = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={sym}", timeout=3)
            if r.status_code == 200:
                d = r.json()
                prices[sym] = {'price': float(d['lastPrice']), 'change': float(d['priceChangePercent'])}
        except:
            pass
    return prices

@st.cache_data(ttl=30)
def get_backtest():
    path = DATA_DIR / "per_asset_backtest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def get_session():
    now = datetime.now(timezone.utc)
    hour = now.hour
    if 13 <= hour < 22:
        return "NY", "#00ff88"
    elif 8 <= hour < 16:
        return "London", "#00ff88"
    elif 0 <= hour < 8:
        return "Asia", "#ffa500"
    else:
        return "Off", "#888"

def render():
    config = load_config()
    prices = fetch_prices()
    bt = get_backtest()
    session, scol = get_session()
    
    assets_data = bt.get('assets', {}) if bt else {}
    confidence = bt.get('confidence_score', 0) if bt else 0
    
    # Price cards
    pc = ""
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'PAXGUSDT']:
        p = prices.get(sym, {})
        name = {'BTCUSDT': 'BTC', 'ETHUSDT': 'ETH', 'SOLUSDT': 'SOL', 'BNBUSDT': 'BNB', 'PAXGUSDT': 'GOLD'}.get(sym, sym)
        price = f"${p.get('price', 0):,.2f}" if p else "-"
        ch = p.get('change', 0)
        chg = f"{'▲' if ch >= 0 else '▼'}{ch:+.2f}%"
        cls = "up" if ch >= 0 else "dn"
        pc += f'<div class="pc"><div class="pn">{name}</div><div class="pv">{price}</div><div class="{cls}">{chg}</div></div>'
    
    # Per-asset backtest rows
    bt_rows = ""
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'PAXGUSDT']:
        a = assets_data.get(sym, {})
        name = a.get('name', sym[:3])
        trades = a.get('trades', 0)
        wr = a.get('win_rate', 0) * 100
        pf = a.get('profit_factor', 0)
        pnl = a.get('total_pnl_pct', 0)
        tp = a.get('sample_tp', 0)
        sl = a.get('sample_sl', 0)
        
        wr_cls = "grn" if wr >= 65 else ("ylw" if wr >= 50 else "red")
        pf_cls = "grn" if pf >= 1 else "red"
        pnl_cls = "grn" if pnl >= 0 else "red"
        
        bt_rows += f'''<tr>
            <td>{name}</td>
            <td>{trades:,}</td>
            <td class="{wr_cls}">{wr:.1f}%</td>
            <td class="{pf_cls}">{pf:.2f}</td>
            <td class="{pnl_cls}">{pnl:.1f}%</td>
            <td>${tp:,.2f}</td>
            <td>${sl:,.2f}</td>
        </tr>'''
    
    html = f'''
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',sans-serif;background:#050505;color:#e0e0e0}}
.hdr{{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;background:#0a0a0a;border-bottom:1px solid #1a1a1a}}
.logo{{display:flex;align-items:center;gap:8px}}
.logo h1{{font-size:1.1em;color:#00ff88}}
.mode{{background:#1a1a1a;color:#00ff88;padding:2px 6px;border-radius:3px;font-size:0.55em}}
.conf{{background:#111;padding:4px 10px;border-radius:4px;border:1px solid #222;font-size:1em;font-weight:700;color:#ffa500}}
.clocks{{display:flex;gap:12px;font-family:'JetBrains Mono',monospace;font-size:0.7em}}
.clk{{text-align:center}}.clk-l{{color:#555;font-size:0.6em}}.clk-v{{color:#00ff88}}
.sess{{background:{scol};color:#000;padding:3px 8px;border-radius:3px;font-size:0.65em;font-weight:600}}
.main{{padding:12px;display:flex;flex-direction:column;gap:12px}}
.section{{background:#0a0a0a;border:1px solid #1a1a1a;border-radius:6px;padding:12px}}
.stitle{{font-size:0.8em;color:#888;margin-bottom:8px;font-weight:600}}
.prices{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}}
.pc{{background:#111;padding:10px;border-radius:5px;text-align:center;border:1px solid #1a1a1a}}
.pn{{color:#666;font-size:0.6em}}.pv{{font-size:1.1em;font-weight:700;color:#00ff88;margin:2px 0}}
.up{{color:#00ff88;font-size:0.7em}}.dn{{color:#ff4444;font-size:0.7em}}
table{{width:100%;border-collapse:collapse;font-size:0.75em}}
th{{background:#111;padding:8px;color:#888;text-align:center}}
td{{padding:8px;border-top:1px solid #1a1a1a;text-align:center}}
.grn{{color:#00ff88}}.ylw{{color:#ffa500}}.red{{color:#ff4444}}
.warn{{background:#2a1a00;border:1px solid #3a2a00;padding:10px;border-radius:6px;margin-top:12px}}
.warn-title{{color:#ffa500;font-weight:600;margin-bottom:4px}}
.warn-text{{font-size:0.75em;color:#ccc}}
</style>
</head>
<body>
<div class="hdr">
    <div class="logo"><span style="font-size:1.2em">⚡</span><h1>FORGE</h1><span class="mode">5-YEAR BACKTEST</span></div>
    <div class="conf">WIN RATE: {confidence}%</div>
    <div class="clocks">
        <div class="clk"><div class="clk-l">UTC</div><div class="clk-v" id="utc">--:--</div></div>
        <div class="clk"><div class="clk-l">NY</div><div class="clk-v" id="ny">--:--</div></div>
    </div>
    <div class="sess">{session}</div>
</div>

<div class="main">
    <div class="section">
        <div class="stitle">💰 LIVE PRICES</div>
        <div class="prices">{pc}</div>
    </div>
    
    <div class="section">
        <div class="stitle">📊 PER-ASSET BACKTEST RESULTS (5 Years, 44K Candles/Asset)</div>
        <table>
            <thead>
                <tr>
                    <th>Asset</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Profit Factor</th>
                    <th>Total PnL</th>
                    <th>Sample TP</th>
                    <th>Sample SL</th>
                </tr>
            </thead>
            <tbody>{bt_rows}</tbody>
        </table>
        
        <div class="warn">
            <div class="warn-title">⚠️ HONEST ASSESSMENT</div>
            <div class="warn-text">
                The system achieves <strong>68-70% win rate</strong> using mean-reversion, but with tight TP and wide SL, 
                net PnL is negative. This is the mathematical reality: high win rate with poor Risk:Reward = losses.
                <br><br>
                <strong>To be profitable:</strong> Need either 65% WR with 1.2:1 RR, or 50% WR with 2:1 RR.
            </div>
        </div>
    </div>
</div>

<script>
function updateClocks(){{const now=new Date();const fmt=(d)=>d.toTimeString().slice(0,5);document.getElementById('utc').textContent=fmt(new Date(now.getTime()+now.getTimezoneOffset()*60000));try{{document.getElementById('ny').textContent=new Date(now.toLocaleString('en-US',{{timeZone:'America/New_York'}})).toTimeString().slice(0,5);}}catch(e){{}}}}
updateClocks();setInterval(updateClocks,1000);
</script>
</body>
</html>'''
    
    components.html(html, height=600, scrolling=True)

def main():
    st.markdown("<style>.stApp{background:#000}#MainMenu,footer,header{visibility:hidden}</style>", unsafe_allow_html=True)
    render()
    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main()
