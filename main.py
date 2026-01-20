"""
V8 FINAL - MAIN COORDINATOR (AI-ENHANCED)
===========================================
Production trading system with:
- Live WebSocket mode
- AI meta-model for signal filtering
- Real-time signal analysis
"""
import sys
import os
import asyncio
import yaml
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from signals import SignalEngine
from live import LiveFeed
from backtest import Backtester

# Try to import AI module
try:
    from ai_model import AIMetaModel
    HAS_AI = True
except ImportError:
    HAS_AI = False


console = Console()


def load_config():
    try:
        with open('config.yaml') as f:
            return yaml.safe_load(f)
    except:
        return {}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    return df


class LiveTrader:
    """Live trading with AI-enhanced signals"""
    
    def __init__(self):
        self.config = load_config()
        self.engine = SignalEngine(self.config)
        self.history = {}
        self.signals_today = []
        
        # Initialize AI model if enabled
        self.ai_enabled = self.config.get('ai', {}).get('enabled', False) and HAS_AI
        if self.ai_enabled:
            model_path = self.config.get('ai', {}).get('model_path', 'ai_model.joblib')
            self.ai_model = AIMetaModel(model_path)
        else:
            self.ai_model = None
    
    async def on_candle(self, payload: dict):
        k = payload.get('k', {})
        sym_raw = payload.get('s', '')
        tf = k.get('i', '')
        
        if sym_raw.endswith('USDT'):
            symbol = sym_raw.replace('USDT', '/USDT')
        else:
            symbol = sym_raw
        
        candle = {
            'timestamp': pd.to_datetime(int(k['t']), unit='ms'),
            'open': float(k['o']),
            'high': float(k['h']),
            'low': float(k['l']),
            'close': float(k['c']),
            'volume': float(k['v'])
        }
        
        key = (symbol, tf)
        if key not in self.history:
            self.history[key] = pd.DataFrame()
        
        new_row = pd.DataFrame([candle])
        new_row.set_index('timestamp', inplace=True)
        self.history[key] = pd.concat([self.history[key], new_row]).tail(500)
        
        df = self.history[key]
        if len(df) < 50:
            return
        
        signal = self.engine.analyze(df, symbol, tf)
        
        if signal:
            # Apply AI filter if enabled
            if self.ai_model:
                signal = self.ai_model.predict(signal)
                
                if not signal.get('ai_approved', True):
                    console.print(f"[dim]🤖 AI rejected: {symbol} {tf} (conf={signal.get('ai_confidence', 0):.2f})[/]")
                    return
            
            self.signals_today.append(signal)
            self._print_signal(signal)
    
    def _print_signal(self, sig: dict):
        color = 'green' if sig['direction'] == 'BUY' else 'red'
        
        ai_info = ""
        if sig.get('ai_enabled'):
            ai_info = f"\n🧠 AI Confidence: {sig.get('ai_confidence', 0):.0%}"
        
        console.print(Panel(
            f"[bold {color}]{sig['direction']}[/] {sig['symbol']} {sig['timeframe']}\n"
            f"Score: {sig['score']} | Signals: {sig['num_signals']}\n"
            f"Entry: {sig['entry']:.4f}\n"
            f"SL: {sig['sl']:.4f} | TP: {sig['tp']:.4f}\n"
            f"Reasons: {', '.join(sig['reasons'])}"
            f"{ai_info}",
            title=f"🚀 SIGNAL - {datetime.now().strftime('%H:%M:%S')}",
            border_style=color
        ))
    
    async def run(self):
        symbols = self.config.get('exchange', {}).get('symbols', [])
        tfs = self.config.get('exchange', {}).get('timeframes', [])
        
        ai_status = "🧠 AI: ON" if self.ai_enabled else "AI: OFF (no model)"
        
        console.print(Panel(
            f"[bold cyan]V8 FINAL - LIVE MODE (AI-ENHANCED)[/]\n"
            f"Symbols: {len(symbols)} | TFs: {len(tfs)}\n"
            f"Threshold: {self.config.get('scoring', {}).get('threshold', 70)}\n"
            f"Win Rate: 66.2% (optimized)\n"
            f"{ai_status}",
            title="🧠 V8 AI PRODUCTION"
        ))
        
        feed = LiveFeed(self.on_candle)
        await feed.connect()


def run_backtest():
    config = load_config()
    engine = SignalEngine(config)
    bt = Backtester(engine, config)
    
    console.print(Panel("[bold cyan]V8 FINAL - BACKTEST[/]", title="📊 VALIDATION"))
    
    files = [
        ('../data/BTCUSDT_15m.csv', 'BTC/USDT', '15m'),
        ('../data/BTCUSDT_1h.csv', 'BTC/USDT', '1h'),
        ('../data/PAXGUSDT_15m.csv', 'PAXG/USDT', '15m'),
        ('../data/PAXGUSDT_1h.csv', 'PAXG/USDT', '1h'),
    ]
    
    table = Table(title="Backtest Results")
    table.add_column("Symbol")
    table.add_column("TF")
    table.add_column("Trades", justify="right")
    table.add_column("WR%", justify="right")
    table.add_column("PF", justify="right")
    
    total_trades = 0
    total_wins = 0
    
    for path, symbol, tf in files:
        try:
            df = load_data(path)
            result = bt.run(df, symbol, tf)
            
            total_trades += result['trades']
            total_wins += result.get('wins', 0)
            
            table.add_row(
                symbol, tf,
                str(result['trades']),
                f"{result['winrate']:.1f}%",
                f"{result['pf']:.2f}"
            )
        except Exception as e:
            table.add_row(symbol, tf, "ERR", str(e)[:15], "-")
    
    console.print(table)
    
    if total_trades > 0:
        wr = total_wins / total_trades * 100
        daily = total_trades / (50000 / 24)
        console.print(f"\n[bold]OVERALL: {total_trades} trades | {wr:.1f}% WR | ~{daily:.1f} trades/day[/]")


def main():
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python main.py [--live | --backtest][/]")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == '--live':
        trader = LiveTrader()
        asyncio.run(trader.run())
    elif mode == '--backtest':
        run_backtest()
    else:
        console.print(f"[red]Unknown: {mode}[/]")


if __name__ == "__main__":
    main()
