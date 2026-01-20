"""
FORGE TRADING SYSTEM - DATA MODULE
===================================
Handles all data ingestion:
- WebSocket live prices (≤200ms)
- Historical data loading (Parquet/CSV)
- Candle-close detection
"""
import pandas as pd
import numpy as np
import requests
import asyncio
import websocket
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Callable, Optional
import yaml


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class HistoricalData:
    """Historical data loader and downloader"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.config = load_config()
    
    def download(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """Download klines from Binance"""
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        except Exception as e:
            print(f"Download error {symbol} {interval}: {e}")
            return pd.DataFrame()
    
    def download_all(self) -> Dict[str, pd.DataFrame]:
        """Download data for all configured assets and timeframes"""
        datasets = {}
        
        for symbol in self.config['assets']:
            for tf in self.config['timeframes']:
                print(f"Downloading {symbol} {tf}...")
                df = self.download(symbol, tf)
                
                if not df.empty:
                    # Save to parquet
                    path = self.data_dir / f"{symbol}_{tf}.parquet"
                    df.to_parquet(path)
                    datasets[f"{symbol}_{tf}"] = df
                    print(f"  ✓ {len(df)} candles")
        
        return datasets
    
    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        """Load data from local storage"""
        parquet_path = self.data_dir / f"{symbol}_{interval}.parquet"
        csv_path = self.data_dir / f"{symbol}_{interval}.csv"
        
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            return pd.DataFrame()
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all configured datasets"""
        datasets = {}
        
        for symbol in self.config['assets']:
            for tf in self.config['timeframes']:
                df = self.load(symbol, tf)
                if not df.empty:
                    datasets[f"{symbol}_{tf}"] = df
        
        return datasets


class LiveDataFeed:
    """WebSocket-based live data feed from Binance"""
    
    def __init__(self, on_price: Callable = None, on_candle: Callable = None):
        self.config = load_config()
        self.on_price = on_price      # Called on every tick
        self.on_candle = on_candle    # Called on candle close
        self.ws = None
        self.running = False
        self.prices = {}              # Latest prices
        self.candles = {}             # Current candles
    
    def _build_stream_url(self) -> str:
        """Build WebSocket stream URL for all assets"""
        streams = []
        
        for symbol in self.config['assets']:
            sym_lower = symbol.lower()
            # Ticker stream for live prices
            streams.append(f"{sym_lower}@ticker")
            # Kline streams for candle data
            for tf in self.config['timeframes']:
                streams.append(f"{sym_lower}@kline_{tf}")
        
        return f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            stream = data.get('stream', '')
            payload = data.get('data', {})
            
            if '@ticker' in stream:
                # Price update
                symbol = payload.get('s', '')
                self.prices[symbol] = {
                    'price': float(payload.get('c', 0)),
                    'change': float(payload.get('P', 0)),
                    'high': float(payload.get('h', 0)),
                    'low': float(payload.get('l', 0)),
                    'volume': float(payload.get('v', 0)),
                    'timestamp': datetime.now(timezone.utc)
                }
                
                if self.on_price:
                    self.on_price(symbol, self.prices[symbol])
            
            elif '@kline' in stream:
                # Candle update
                k = payload.get('k', {})
                symbol = payload.get('s', '')
                interval = k.get('i', '')
                is_closed = k.get('x', False)
                
                candle = {
                    'timestamp': pd.to_datetime(k.get('t', 0), unit='ms'),
                    'open': float(k.get('o', 0)),
                    'high': float(k.get('h', 0)),
                    'low': float(k.get('l', 0)),
                    'close': float(k.get('c', 0)),
                    'volume': float(k.get('v', 0)),
                    'is_closed': is_closed
                }
                
                key = f"{symbol}_{interval}"
                self.candles[key] = candle
                
                # Only call on_candle when candle closes
                if is_closed and self.on_candle:
                    self.on_candle(symbol, interval, candle)
        
        except Exception as e:
            print(f"WebSocket message error: {e}")
    
    def _on_error(self, ws, error):
        print(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status, close_msg):
        print(f"WebSocket closed: {close_status} {close_msg}")
        if self.running:
            # Reconnect after 5 seconds
            threading.Timer(5.0, self.start).start()
    
    def _on_open(self, ws):
        print("WebSocket connected")
    
    def start(self):
        """Start the WebSocket connection in a background thread"""
        self.running = True
        url = self._build_stream_url()
        
        self.ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        thread.start()
        return thread
    
    def stop(self):
        """Stop the WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
    
    def get_price(self, symbol: str) -> Optional[Dict]:
        """Get latest price for symbol"""
        return self.prices.get(symbol)
    
    def get_all_prices(self) -> Dict:
        """Get all current prices"""
        return self.prices.copy()


# Convenience function
def get_fresh_data():
    """Download fresh data for all assets"""
    loader = HistoricalData()
    return loader.download_all()
