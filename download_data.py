"""
MARKETFORGE: Download Data
==========================
Robust historical data downloader for Binance.

Features:
- Paging with retry and exponential backoff
- 5-10 years of 1-minute data support
- Output to Parquet (partitioned by date+symbol)
- Resume capability for interrupted downloads

Usage:
    python download_data.py BTCUSDT 1m --years 5
    python download_data.py PAXGUSDT 1h --start 2020-01-01
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import argparse
import ssl
import certifi
from typing import Optional, List
import pyarrow as pa
import pyarrow.parquet as pq

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
PARQUET_DIR.mkdir(exist_ok=True)

# Binance API
BINANCE_BASE = "https://api.binance.com"
BINANCE_KLINES = f"{BINANCE_BASE}/api/v3/klines"

# Rate limiting
MAX_REQUESTS_PER_MIN = 1200
REQUEST_WEIGHT_KLINES = 5
BATCH_SIZE = 1000  # Max candles per request


def get_interval_ms(interval: str) -> int:
    """Convert interval string to milliseconds"""
    multipliers = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
    }
    return multipliers.get(interval, 60 * 1000)


class BinanceDownloader:
    """Robust Binance historical data downloader"""
    
    def __init__(self, symbol: str, interval: str = "1m"):
        self.symbol = symbol.upper()
        self.interval = interval
        self.interval_ms = get_interval_ms(interval)
        
        self.session = None
        self.request_count = 0
        self.last_request_time = 0
        
        # Output paths
        self.csv_path = DATA_DIR / f"{self.symbol}_{self.interval}.csv"
        self.parquet_dir = PARQUET_DIR / self.symbol / self.interval
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = DATA_DIR / f".progress_{self.symbol}_{self.interval}.json"
    
    async def start_session(self):
        """Initialize HTTP session"""
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=10)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    def _load_progress(self) -> Optional[int]:
        """Load last downloaded timestamp"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)
                    return data.get('last_timestamp')
            except:
                pass
        return None
    
    def _save_progress(self, timestamp: int):
        """Save progress for resume"""
        with open(self.progress_file, 'w') as f:
            json.dump({
                'symbol': self.symbol,
                'interval': self.interval,
                'last_timestamp': timestamp,
                'updated_at': datetime.now().isoformat()
            }, f)
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        now = time.time()
        elapsed = now - self.last_request_time
        
        # Simple rate limit: ensure at least 50ms between requests
        if elapsed < 0.05:
            await asyncio.sleep(0.05 - elapsed)
        
        self.last_request_time = time.time()
    
    async def fetch_klines(self, start_time: int, end_time: int, 
                            retry: int = 3) -> List[list]:
        """Fetch klines with retry and backoff"""
        await self._rate_limit()
        
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': BATCH_SIZE
        }
        
        for attempt in range(retry):
            try:
                async with self.session.get(BINANCE_KLINES, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        # Rate limited - wait and retry
                        wait_time = 2 ** attempt * 10
                        print(f"   ⚠️ Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    elif resp.status == 418:
                        # IP banned - wait longer
                        print("   🛑 IP temporarily banned, waiting 60s...")
                        await asyncio.sleep(60)
                    else:
                        print(f"   ⚠️ HTTP {resp.status}, retrying...")
                        await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(f"   ⚠️ Request failed: {e}, retrying...")
                await asyncio.sleep(2 ** attempt)
        
        return []
    
    async def download(self, start_date: str = None, end_date: str = None,
                        years: int = None) -> pd.DataFrame:
        """Download historical data"""
        print(f"\n{'='*60}")
        print(f"📥 Downloading {self.symbol} {self.interval}")
        print(f"{'='*60}")
        
        # Determine time range
        end_ts = int(datetime.now().timestamp() * 1000)
        
        if end_date:
            end_ts = int(datetime.fromisoformat(end_date).timestamp() * 1000)
        
        if start_date:
            start_ts = int(datetime.fromisoformat(start_date).timestamp() * 1000)
        elif years:
            start_ts = end_ts - (years * 365 * 24 * 60 * 60 * 1000)
        else:
            start_ts = end_ts - (5 * 365 * 24 * 60 * 60 * 1000)  # Default 5 years
        
        # Check for resume
        last_progress = self._load_progress()
        if last_progress and last_progress > start_ts:
            print(f"   📎 Resuming from {datetime.fromtimestamp(last_progress/1000)}")
            start_ts = last_progress
        
        total_batches = (end_ts - start_ts) // (self.interval_ms * BATCH_SIZE) + 1
        print(f"   📅 Range: {datetime.fromtimestamp(start_ts/1000)} → {datetime.fromtimestamp(end_ts/1000)}")
        print(f"   📦 Estimated batches: {total_batches}")
        
        await self.start_session()
        
        all_data = []
        current_ts = start_ts
        batch_count = 0
        
        try:
            while current_ts < end_ts:
                batch_end = min(current_ts + self.interval_ms * BATCH_SIZE, end_ts)
                
                klines = await self.fetch_klines(current_ts, batch_end)
                
                if klines:
                    all_data.extend(klines)
                    current_ts = int(klines[-1][0]) + self.interval_ms
                    self._save_progress(current_ts)
                    batch_count += 1
                    
                    # Progress update every 50 batches
                    if batch_count % 50 == 0:
                        pct = (current_ts - start_ts) / (end_ts - start_ts) * 100
                        print(f"   ⏳ {pct:.1f}% | {len(all_data):,} candles | {datetime.fromtimestamp(current_ts/1000)}")
                else:
                    current_ts = batch_end
        
        finally:
            await self.close_session()
        
        if not all_data:
            print("   ❌ No data downloaded")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'time', 'o', 'h', 'l', 'c', 'v',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # Clean up
        df = df[['time', 'o', 'h', 'l', 'c', 'v']].copy()
        for col in ['o', 'h', 'l', 'c', 'v']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.drop_duplicates(subset='time')
        df = df.sort_values('time').reset_index(drop=True)
        
        print(f"\n   ✅ Downloaded {len(df):,} candles")
        
        # Save to CSV
        df.to_csv(self.csv_path, index=False)
        print(f"   💾 Saved to {self.csv_path}")
        
        # Save to Parquet (partitioned by date)
        self.save_parquet_partitioned(df)
        
        # Clean up progress file
        if self.progress_file.exists():
            self.progress_file.unlink()
        
        return df
    
    def save_parquet_partitioned(self, df: pd.DataFrame):
        """Save to Parquet partitioned by date"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['time'], unit='ms').dt.date.astype(str)
        
        # Save each date as separate file
        for date_str, group in df.groupby('date'):
            date_path = self.parquet_dir / f"{date_str}.parquet"
            group.drop(columns=['date']).to_parquet(date_path, index=False)
        
        print(f"   📂 Partitioned Parquet saved to {self.parquet_dir}")


async def download_all_assets():
    """Download all configured assets"""
    assets = [
        ('BTCUSDT', '1m', 5),
        ('BTCUSDT', '5m', 5),
        ('BTCUSDT', '15m', 5),
        ('BTCUSDT', '30m', 5),
        ('BTCUSDT', '1h', 5),
        ('PAXGUSDT', '1m', 3),
        ('PAXGUSDT', '5m', 3),
        ('PAXGUSDT', '15m', 3),
        ('PAXGUSDT', '30m', 3),
        ('PAXGUSDT', '1h', 3),
    ]
    
    for symbol, interval, years in assets:
        downloader = BinanceDownloader(symbol, interval)
        await downloader.download(years=years)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download historical data from Binance')
    parser.add_argument('symbol', nargs='?', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('interval', nargs='?', default='1m', help='Candle interval')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--years', type=int, default=5, help='Years of history')
    parser.add_argument('--all', action='store_true', help='Download all assets')
    
    args = parser.parse_args()
    
    if args.all:
        asyncio.run(download_all_assets())
    else:
        downloader = BinanceDownloader(args.symbol, args.interval)
        asyncio.run(downloader.download(
            start_date=args.start,
            end_date=args.end,
            years=args.years
        ))
