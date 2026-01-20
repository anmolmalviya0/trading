"""
MARKETFORGE: Candle Builder
===========================
WebSocket tick-to-bar aggregator for live data.

Features:
- Binance WebSocket connection for BTC/USDT and PAXG/USDT
- Real-time tick-to-bar aggregation (5m/15m/30m/1h)
- Deduplication & timestamp alignment (UTC)
- Heartbeat & stale detection
- Auto-reconnect with exponential backoff

Usage:
    python candle_builder.py              # Run live
    python candle_builder.py --symbols BTCUSDT,PAXGUSDT
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Optional, Callable
import time
import ssl
import certifi
import argparse

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LIVE_DIR = DATA_DIR / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

# WebSocket endpoints
BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
BINANCE_WS_STREAM = "wss://stream.binance.com:9443/stream"

# Timeframes (in seconds)
TIMEFRAMES = {
    '5m': 5 * 60,
    '15m': 15 * 60,
    '30m': 30 * 60,
    '1h': 60 * 60,
}

# Stale thresholds
HEARTBEAT_INTERVAL = 30  # seconds
STALE_THRESHOLD = 60  # seconds


class CandleBar:
    """Represents a single aggregated candle"""
    
    def __init__(self, timestamp: int, timeframe: str):
        self.timestamp = timestamp  # Bar open time (ms)
        self.timeframe = timeframe
        self.o = None
        self.h = None
        self.l = None
        self.c = None
        self.v = 0.0
        self.trades = 0
        self.closed = False
    
    def update(self, price: float, quantity: float):
        """Update bar with new tick"""
        if self.o is None:
            self.o = price
            self.h = price
            self.l = price
        
        self.h = max(self.h, price)
        self.l = min(self.l, price)
        self.c = price
        self.v += quantity
        self.trades += 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'time': self.timestamp,
            'o': self.o or 0,
            'h': self.h or 0,
            'l': self.l or 0,
            'c': self.c or 0,
            'v': round(self.v, 8),
            'trades': self.trades,
            'closed': self.closed
        }
    
    def __repr__(self):
        return f"CandleBar({self.timeframe}, {datetime.fromtimestamp(self.timestamp/1000)}, OHLC={self.o}/{self.h}/{self.l}/{self.c})"


class TickBuffer:
    """Buffers and deduplicates incoming ticks"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.ticks: List[dict] = []
        self.seen_ids: set = set()
    
    def add(self, tick: dict) -> bool:
        """Add tick if not duplicate. Returns True if added."""
        tick_id = f"{tick['time']}_{tick['price']}_{tick['quantity']}"
        
        if tick_id in self.seen_ids:
            return False
        
        self.seen_ids.add(tick_id)
        self.ticks.append(tick)
        
        # Trim if too large
        if len(self.ticks) > self.max_size:
            removed = self.ticks[:1000]
            self.ticks = self.ticks[1000:]
            for t in removed:
                tid = f"{t['time']}_{t['price']}_{t['quantity']}"
                self.seen_ids.discard(tid)
        
        return True
    
    def clear(self):
        """Clear buffer"""
        self.ticks.clear()
        self.seen_ids.clear()


class CandleBuilder:
    """
    Aggregates ticks into multi-timeframe candles.
    
    Maintains rolling bars for each timeframe and emits
    completed bars when they close.
    """
    
    def __init__(self, symbols: List[str], 
                 on_candle_close: Optional[Callable] = None):
        self.symbols = [s.lower() for s in symbols]
        self.on_candle_close = on_candle_close
        
        # Current bars per symbol and timeframe
        self.bars: Dict[str, Dict[str, CandleBar]] = {
            sym: {tf: None for tf in TIMEFRAMES}
            for sym in self.symbols
        }
        
        # Completed bars history
        self.history: Dict[str, Dict[str, List[dict]]] = {
            sym: {tf: [] for tf in TIMEFRAMES}
            for sym in self.symbols
        }
        
        # Tick buffer per symbol
        self.buffers: Dict[str, TickBuffer] = {
            sym: TickBuffer() for sym in self.symbols
        }
        
        # Connection state
        self.ws = None
        self.connected = False
        self.reconnect_count = 0
        self.last_tick_time: Dict[str, float] = {sym: 0 for sym in self.symbols}
        self.heartbeat_task = None
    
    def _get_bar_open_time(self, timestamp_ms: int, tf_seconds: int) -> int:
        """Calculate bar open time from timestamp"""
        return (timestamp_ms // (tf_seconds * 1000)) * (tf_seconds * 1000)
    
    def _process_tick(self, symbol: str, tick: dict):
        """Process a single tick and update bars"""
        symbol = symbol.lower()
        if symbol not in self.symbols:
            return
        
        # Deduplicate
        if not self.buffers[symbol].add(tick):
            return
        
        price = tick['price']
        quantity = tick['quantity']
        timestamp = tick['time']
        
        self.last_tick_time[symbol] = time.time()
        
        # Update all timeframe bars
        for tf, tf_seconds in TIMEFRAMES.items():
            bar_open = self._get_bar_open_time(timestamp, tf_seconds)
            current_bar = self.bars[symbol][tf]
            
            if current_bar is None or current_bar.timestamp != bar_open:
                # Close previous bar
                if current_bar is not None and not current_bar.closed:
                    current_bar.closed = True
                    self.history[symbol][tf].append(current_bar.to_dict())
                    
                    # Trim history
                    if len(self.history[symbol][tf]) > 1000:
                        self.history[symbol][tf] = self.history[symbol][tf][-500:]
                    
                    # Callback
                    if self.on_candle_close:
                        asyncio.create_task(
                            self._emit_candle(symbol, tf, current_bar)
                        )
                
                # Start new bar
                self.bars[symbol][tf] = CandleBar(bar_open, tf)
            
            # Update current bar
            self.bars[symbol][tf].update(price, quantity)
    
    async def _emit_candle(self, symbol: str, timeframe: str, bar: CandleBar):
        """Emit completed candle via callback"""
        try:
            await self.on_candle_close(symbol, timeframe, bar.to_dict())
        except Exception as e:
            print(f"⚠️ Candle emit error: {e}")
    
    async def _heartbeat_loop(self):
        """Monitor connection health"""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            
            now = time.time()
            for symbol in self.symbols:
                elapsed = now - self.last_tick_time[symbol]
                if elapsed > STALE_THRESHOLD and self.last_tick_time[symbol] > 0:
                    print(f"⚠️ STALE FEED: {symbol.upper()} - no ticks for {elapsed:.0f}s")
    
    async def connect(self):
        """Connect to Binance WebSocket"""
        # Build stream URL for all symbols
        streams = [f"{sym}@aggTrade" for sym in self.symbols]
        url = f"{BINANCE_WS_STREAM}?streams={'/'.join(streams)}"
        
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        
        while True:
            try:
                print(f"🔌 Connecting to Binance WebSocket...")
                
                async with websockets.connect(
                    url,
                    ssl=ssl_ctx,
                    ping_interval=20,
                    ping_timeout=10
                ) as ws:
                    self.ws = ws
                    self.connected = True
                    self.reconnect_count = 0
                    
                    print(f"✅ Connected! Streaming: {[s.upper() for s in self.symbols]}")
                    
                    # Start heartbeat
                    self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            
                            if 'stream' in data and 'data' in data:
                                trade = data['data']
                                symbol = trade['s'].lower()
                                
                                tick = {
                                    'time': trade['T'],  # Trade time
                                    'price': float(trade['p']),
                                    'quantity': float(trade['q']),
                                    'is_buyer_maker': trade['m']
                                }
                                
                                self._process_tick(symbol, tick)
                        
                        except json.JSONDecodeError:
                            continue
            
            except websockets.ConnectionClosed as e:
                print(f"⚠️ Connection closed: {e}")
            except Exception as e:
                print(f"⚠️ WebSocket error: {e}")
            
            finally:
                self.connected = False
                if self.heartbeat_task:
                    self.heartbeat_task.cancel()
                
                # Reconnect with backoff
                self.reconnect_count += 1
                wait_time = min(2 ** self.reconnect_count, 60)
                print(f"🔄 Reconnecting in {wait_time}s (attempt {self.reconnect_count})...")
                await asyncio.sleep(wait_time)
    
    def get_current_bars(self) -> dict:
        """Get all current (incomplete) bars"""
        result = {}
        for symbol in self.symbols:
            result[symbol] = {}
            for tf in TIMEFRAMES:
                bar = self.bars[symbol][tf]
                if bar:
                    result[symbol][tf] = bar.to_dict()
        return result
    
    def get_history(self, symbol: str, timeframe: str, limit: int = 100) -> List[dict]:
        """Get historical completed bars"""
        symbol = symbol.lower()
        if symbol not in self.history:
            return []
        if timeframe not in self.history[symbol]:
            return []
        return self.history[symbol][timeframe][-limit:]
    
    def get_dataframe(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Get history as DataFrame"""
        bars = self.get_history(symbol, timeframe, limit)
        if not bars:
            return pd.DataFrame()
        return pd.DataFrame(bars)
    
    def is_stale(self, symbol: str, threshold: int = STALE_THRESHOLD) -> bool:
        """Check if feed is stale"""
        symbol = symbol.lower()
        if symbol not in self.last_tick_time:
            return True
        elapsed = time.time() - self.last_tick_time[symbol]
        return elapsed > threshold
    
    def save_history(self, symbol: str, timeframe: str):
        """Save history to CSV"""
        df = self.get_dataframe(symbol, timeframe)
        if not df.empty:
            path = LIVE_DIR / f"{symbol.upper()}_{timeframe}_live.csv"
            df.to_csv(path, index=False)
            print(f"💾 Saved {len(df)} bars to {path}")


async def on_candle_close_handler(symbol: str, timeframe: str, bar: dict):
    """Example candle close handler"""
    dt = datetime.fromtimestamp(bar['time'] / 1000, tz=timezone.utc)
    print(f"🕯️ {symbol.upper()} {timeframe} CLOSED: {dt.strftime('%H:%M')} | "
          f"O={bar['o']:.2f} H={bar['h']:.2f} L={bar['l']:.2f} C={bar['c']:.2f} V={bar['v']:.4f}")


async def main(symbols: List[str]):
    """Main entry point"""
    print("="*60)
    print("📊 MARKETFORGE: Candle Builder")
    print("="*60)
    print(f"Symbols: {symbols}")
    print(f"Timeframes: {list(TIMEFRAMES.keys())}")
    print("="*60)
    
    builder = CandleBuilder(
        symbols=symbols,
        on_candle_close=on_candle_close_handler
    )
    
    try:
        await builder.connect()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        for sym in symbols:
            for tf in TIMEFRAMES:
                builder.save_history(sym, tf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live candle builder')
    parser.add_argument('--symbols', type=str, default='BTCUSDT,PAXGUSDT',
                        help='Comma-separated symbols')
    
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    asyncio.run(main(symbols))
