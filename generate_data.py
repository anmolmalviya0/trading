"""
V8 DATA GENERATOR (FULL)
========================
Downloads ALL asset data and generates signals without session filter.
"""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import sys
sys.path.insert(0, '.')

from signals import SignalEngine

# Config
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'PAXGUSDT']
TIMEFRAMES = ['5m', '15m', '1h']
DATA_DIR = Path('../data')
DATA_DIR.mkdir(exist_ok=True)

def download_data(symbol, interval, limit=1000):
    """Download klines from Binance"""
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        print(f"  Downloaded {len(df)} candles")
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

class DataGenEngine(SignalEngine):
    """Engine with session filter disabled for backtesting"""
    def is_session_active(self):
        return True  # Always active for data generation

def generate_all():
    engine = DataGenEngine()
    
    print("=" * 50)
    print("V8 DATA GENERATOR - FULL RUN")
    print("=" * 50)
    
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"\n[{sym}] [{tf}]")
            
            # Download
            df = download_data(sym, tf)
            if df is None or len(df) < 100:
                print("  SKIP: Not enough data")
                continue
            
            # Save CSV
            csv_path = DATA_DIR / f"{sym}_{tf}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved CSV: {csv_path.name}")
            
            # Generate labels
            labels = []
            signals_found = 0
            
            for i in range(50, len(df)):
                subset = df.iloc[:i+1].copy()
                subset['timestamp'] = pd.to_datetime(subset['timestamp'])
                subset = subset.set_index('timestamp')
                
                sig = engine.analyze(subset, sym.replace('USDT', '/USDT'), tf)
                
                label = 0
                score = 0
                
                if sig:
                    score = sig['score']
                    label = 1 if sig['direction'] == 'BUY' else -1
                    signals_found += 1
                
                labels.append({
                    'timestamp': df.iloc[i]['timestamp'],
                    'open': df.iloc[i]['open'],
                    'high': df.iloc[i]['high'],
                    'low': df.iloc[i]['low'],
                    'close': df.iloc[i]['close'],
                    'label': label,
                    'score': score
                })
            
            # Save parquet
            lbl_df = pd.DataFrame(labels)
            parquet_path = DATA_DIR / f"{sym}_{tf}_labeled.parquet"
            lbl_df.to_parquet(parquet_path)
            print(f"  ✅ Parquet: {parquet_path.name} ({signals_found} signals)")
    
    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)

if __name__ == "__main__":
    generate_all()
