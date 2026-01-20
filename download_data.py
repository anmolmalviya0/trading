"""
PHASE 1: DATA PIPELINE
======================
Downloads/processes 5-10 years of historical data into Parquet format.

Supports:
- Binance (BTCUSDT, PAXGUSDT) via REST API paging
- Converts CSV to Parquet for fast analytics
- Validates data quality (gaps, duplicates)

Usage:
    python download_data.py
"""
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'market_data'
PARQUET_DIR = DATA_DIR / 'parquet'
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'symbols': ['BTCUSDT', 'PAXGUSDT'],
    'intervals': ['5m', '15m', '30m', '1h', '4h', '1d'],
    'start_date': '2019-01-01',  # 5+ years
    'end_date': datetime.now().strftime('%Y-%m-%d'),
    'batch_size': 1000,  # Binance max per request
}


# === BINANCE API ===

def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Fetch klines from Binance REST API"""
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_ms,
        'endTime': end_ms,
        'limit': CONFIG['batch_size']
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"   ⚠️ API Error: {e}")
        return []


def download_full_history(symbol: str, interval: str) -> pd.DataFrame:
    """Download full history with paging"""
    print(f"\n📥 Downloading {symbol} {interval}...")
    
    start_dt = datetime.strptime(CONFIG['start_date'], '%Y-%m-%d')
    end_dt = datetime.strptime(CONFIG['end_date'], '%Y-%m-%d')
    
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    all_data = []
    current_ms = start_ms
    
    while current_ms < end_ms:
        klines = fetch_binance_klines(symbol, interval, current_ms, end_ms)
        
        if not klines:
            break
        
        all_data.extend(klines)
        
        # Move to next batch
        last_close_time = klines[-1][6]  # Close time is index 6
        current_ms = last_close_time + 1
        
        print(f"   Fetched {len(all_data):,} candles...", end='\r')
        time.sleep(0.1)  # Rate limit respect
    
    print(f"   ✅ Total: {len(all_data):,} candles")
    
    if not all_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    columns = ['time', 'o', 'h', 'l', 'c', 'v', 'close_time', 
               'quote_vol', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Clean up
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    
    for col in ['o', 'h', 'l', 'c', 'v', 'quote_vol', 'taker_buy_base', 'taker_buy_quote']:
        df[col] = df[col].astype(float)
    
    df['trades'] = df['trades'].astype(int)
    
    # Keep essential columns
    df = df[['time', 'o', 'h', 'l', 'c', 'v', 'trades', 'quote_vol']]
    
    return df


def validate_data(df: pd.DataFrame, interval: str) -> dict:
    """Validate data quality"""
    if df.empty:
        return {'valid': False, 'reason': 'Empty DataFrame'}
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['time']).sum()
    
    # Check for gaps (expected interval)
    interval_map = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
    expected_minutes = interval_map.get(interval, 60)
    
    df_sorted = df.sort_values('time')
    time_diffs = df_sorted['time'].diff().dt.total_seconds() / 60
    gaps = (time_diffs > expected_minutes * 1.5).sum()
    
    # Date range
    date_range = (df['time'].max() - df['time'].min()).days
    
    return {
        'valid': True,
        'rows': len(df),
        'duplicates': duplicates,
        'gaps': gaps,
        'date_range_days': date_range,
        'start': df['time'].min(),
        'end': df['time'].max()
    }


def convert_csv_to_parquet():
    """Convert existing CSVs to Parquet format"""
    print("\n📦 Converting CSVs to Parquet...")
    
    csv_files = list(DATA_DIR.glob('*.csv'))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Standardize columns
            if len(df.columns) == 6:
                df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
            
            # Convert time
            if df['time'].dtype == 'int64':
                df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
            else:
                df['time'] = pd.to_datetime(df['time'], utc=True)
            
            # Save as Parquet
            parquet_path = PARQUET_DIR / f"{csv_file.stem}.parquet"
            df.to_parquet(parquet_path, index=False, compression='snappy')
            
            # Validate
            stats = validate_data(df, csv_file.stem.split('_')[-1])
            
            print(f"   ✅ {csv_file.name} -> {parquet_path.name}")
            print(f"      Rows: {stats['rows']:,} | Range: {stats['date_range_days']} days | Gaps: {stats['gaps']}")
            
        except Exception as e:
            print(f"   ❌ {csv_file.name}: {e}")


def download_and_save():
    """Download fresh data and save as Parquet"""
    print("\n🌐 Downloading fresh data from Binance...")
    
    for symbol in CONFIG['symbols']:
        for interval in CONFIG['intervals']:
            df = download_full_history(symbol, interval)
            
            if df.empty:
                continue
            
            # Validate
            stats = validate_data(df, interval)
            
            # Save
            parquet_path = PARQUET_DIR / f"{symbol}_{interval}.parquet"
            df.to_parquet(parquet_path, index=False, compression='snappy')
            
            print(f"   💾 Saved: {parquet_path.name}")
            print(f"      Rows: {stats['rows']:,} | Range: {stats['date_range_days']} days")


def load_parquet(symbol: str, interval: str) -> pd.DataFrame:
    """Load Parquet file for analysis"""
    parquet_path = PARQUET_DIR / f"{symbol}_{interval}.parquet"
    
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    
    # Fallback to CSV
    csv_path = DATA_DIR / f"{symbol}_{interval}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
        return df
    
    raise FileNotFoundError(f"No data found for {symbol} {interval}")


# === MAIN ===

if __name__ == "__main__":
    print("="*70)
    print("📊 PHASE 1: DATA PIPELINE")
    print("="*70)
    
    # Step 1: Convert existing CSVs to Parquet
    convert_csv_to_parquet()
    
    # Step 2: Download fresh data (optional - comment out if you have enough)
    # download_and_save()
    
    # Step 3: Summary
    print("\n" + "="*70)
    print("📋 DATA SUMMARY")
    print("="*70)
    
    parquet_files = list(PARQUET_DIR.glob('*.parquet'))
    
    for pq in sorted(parquet_files):
        df = pd.read_parquet(pq)
        print(f"   {pq.name}: {len(df):,} rows")
    
    print("\n✅ Data pipeline complete!")
    print(f"   Parquet files: {PARQUET_DIR}")
