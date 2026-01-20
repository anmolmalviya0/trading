"""
DATA DOWNLOADER - Deep History Acquisition
==========================================
Downloads and standardizes historical data from multiple sources:
1. Kaggle: Binance Full History (1-min BTC 2017-2026)
2. Kaggle: Gold Price 1970-2024 (daily)
3. Alternative: QuantConnect tick data

All data is standardized to Parquet format for high-speed I/O.

Usage:
    python data_downloader.py
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import requests
import zipfile
import json

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'market_data'
KAGGLE_DIR = DATA_DIR / 'kaggle'
PARQUET_DIR = DATA_DIR / 'parquet'

# Create directories
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# KAGGLE DATASET DOWNLOADER
# =============================================================================

class KaggleDownloader:
    """
    Downloads datasets from Kaggle.
    
    Requirements:
    - kaggle.json API credentials in ~/.kaggle/
    - pip install kaggle
    """
    
    DATASETS = {
        'btc_full_history': 'jorijnsmit/binance-full-history',
        'gold_1970_2024': 'olegshpagin/gold-silver-prices-from-1985',
        'crypto_sentiment': 'kaushiksuresh147/cryptocurrency-sentiment-dataset',
    }
    
    def __init__(self):
        self.kaggle_available = self._check_kaggle()
    
    def _check_kaggle(self) -> bool:
        """Check if Kaggle API is available"""
        try:
            import kaggle
            return True
        except ImportError:
            print("⚠️ Kaggle not installed. Run: pip install kaggle")
            return False
        except Exception as e:
            print(f"⚠️ Kaggle API error: {e}")
            return False
    
    def download(self, dataset_key: str) -> Path:
        """Download a dataset from Kaggle"""
        if not self.kaggle_available:
            print("❌ Kaggle API not available")
            return None
        
        if dataset_key not in self.DATASETS:
            print(f"❌ Unknown dataset: {dataset_key}")
            return None
        
        dataset_name = self.DATASETS[dataset_key]
        output_dir = KAGGLE_DIR / dataset_key
        output_dir.mkdir(exist_ok=True)
        
        print(f"📥 Downloading {dataset_name}...")
        
        try:
            import kaggle
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(output_dir),
                unzip=True
            )
            print(f"✅ Downloaded to {output_dir}")
            return output_dir
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def list_available(self):
        """List available datasets"""
        print("\n📚 Available Datasets:")
        for key, name in self.DATASETS.items():
            print(f"  - {key}: {name}")


# =============================================================================
# DATA STANDARDIZER
# =============================================================================

class DataStandardizer:
    """
    Converts various data formats to standardized Parquet.
    
    Standard format:
    - Columns: time, o, h, l, c, v (OHLCV)
    - Time: UTC timestamp (milliseconds)
    - Sorted by time ascending
    """
    
    @staticmethod
    def standardize_binance(input_path: Path, symbol: str = 'BTCUSDT') -> Path:
        """Standardize Binance full history CSVs"""
        print(f"🔧 Standardizing Binance data for {symbol}...")
        
        # Find all CSV files
        csv_files = list(input_path.glob('*.csv'))
        if not csv_files:
            print("❌ No CSV files found")
            return None
        
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Binance format: open_time, open, high, low, close, volume, ...
                if 'open_time' in df.columns:
                    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
                    df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
                    all_data.append(df)
            except Exception as e:
                print(f"  ⚠️ Error reading {csv_file.name}: {e}")
        
        if not all_data:
            print("❌ No valid data found")
            return None
        
        # Combine and sort
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset='time')
        combined = combined.sort_values('time').reset_index(drop=True)
        
        # Save to Parquet
        output_path = PARQUET_DIR / f"{symbol}_1m.parquet"
        combined.to_parquet(output_path, index=False)
        
        print(f"✅ Saved {len(combined):,} rows to {output_path}")
        return output_path
    
    @staticmethod
    def standardize_gold(input_path: Path) -> Path:
        """Standardize Gold price history"""
        print("🔧 Standardizing Gold data...")
        
        # Find gold CSV
        csv_files = list(input_path.glob('*gold*.csv')) + list(input_path.glob('*Gold*.csv'))
        if not csv_files:
            csv_files = list(input_path.glob('*.csv'))
        
        if not csv_files:
            print("❌ No CSV files found")
            return None
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Common gold data formats
                if 'Date' in df.columns:
                    df['time'] = pd.to_datetime(df['Date']).astype(int) // 10**6
                elif 'date' in df.columns:
                    df['time'] = pd.to_datetime(df['date']).astype(int) // 10**6
                else:
                    continue
                
                # Map columns
                col_map = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'open' in col_lower:
                        col_map[col] = 'o'
                    elif 'high' in col_lower:
                        col_map[col] = 'h'
                    elif 'low' in col_lower:
                        col_map[col] = 'l'
                    elif 'close' in col_lower or 'price' in col_lower:
                        col_map[col] = 'c'
                    elif 'volume' in col_lower or 'vol' in col_lower:
                        col_map[col] = 'v'
                
                df = df.rename(columns=col_map)
                
                # Ensure required columns
                required = ['time', 'o', 'h', 'l', 'c']
                if not all(col in df.columns for col in required):
                    # Try to infer from single price column
                    if 'c' in df.columns:
                        df['o'] = df['c']
                        df['h'] = df['c']
                        df['l'] = df['c']
                    else:
                        continue
                
                if 'v' not in df.columns:
                    df['v'] = 0
                
                # Select and sort
                df = df[['time', 'o', 'h', 'l', 'c', 'v']]
                df = df.sort_values('time').reset_index(drop=True)
                
                # Save
                output_path = PARQUET_DIR / "GOLD_1d.parquet"
                df.to_parquet(output_path, index=False)
                
                print(f"✅ Saved {len(df):,} rows to {output_path}")
                return output_path
                
            except Exception as e:
                print(f"  ⚠️ Error processing {csv_file.name}: {e}")
        
        print("❌ Could not standardize Gold data")
        return None


# =============================================================================
# MACRO REGIME LABELER
# =============================================================================

class MacroRegimeLabeler:
    """
    Labels historical data with macro-economic regimes.
    
    Regimes:
    - INFLATION: High CPI periods
    - WAR: Major conflict periods
    - CRASH: Market crash periods
    - BULL: Strong uptrends
    - BEAR: Strong downtrends
    """
    
    # Historical regime periods (approximate)
    REGIMES = {
        'CRASH': [
            ('2000-03-01', '2002-10-01'),  # Dot-com crash
            ('2007-10-01', '2009-03-01'),  # Financial crisis
            ('2020-02-01', '2020-04-01'),  # COVID crash
            ('2022-01-01', '2022-12-01'),  # Crypto winter
        ],
        'INFLATION': [
            ('1979-01-01', '1982-12-01'),  # Volcker era
            ('2021-03-01', '2023-06-01'),  # Post-COVID inflation
        ],
        'WAR': [
            ('1990-08-01', '1991-02-01'),  # Gulf War
            ('2022-02-24', '2024-01-01'),  # Ukraine conflict
        ],
    }
    
    @staticmethod
    def label_data(df: pd.DataFrame) -> pd.DataFrame:
        """Add regime labels to dataframe"""
        df = df.copy()
        df['regime'] = 'NORMAL'
        
        # Convert time to datetime if needed
        if df['time'].dtype == np.int64:
            df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        else:
            df['datetime'] = pd.to_datetime(df['time'])
        
        for regime, periods in MacroRegimeLabeler.REGIMES.items():
            for start, end in periods:
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
                df.loc[mask, 'regime'] = regime
        
        df = df.drop(columns=['datetime'])
        return df
    
    @staticmethod
    def get_regime_stats(df: pd.DataFrame) -> dict:
        """Get statistics per regime"""
        stats = {}
        for regime in df['regime'].unique():
            regime_data = df[df['regime'] == regime]
            returns = regime_data['c'].pct_change().dropna()
            stats[regime] = {
                'count': len(regime_data),
                'mean_return': returns.mean() * 100,
                'volatility': returns.std() * 100,
                'max_drawdown': (regime_data['c'] / regime_data['c'].cummax() - 1).min() * 100
            }
        return stats


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def download_all():
    """Download all datasets"""
    downloader = KaggleDownloader()
    downloader.list_available()
    
    print("\n" + "="*70)
    print("📥 DOWNLOADING DATASETS")
    print("="*70)
    
    # Download each dataset
    for key in downloader.DATASETS.keys():
        print(f"\n--- {key} ---")
        path = downloader.download(key)
        if path:
            print(f"  ✅ Ready: {path}")


def standardize_all():
    """Standardize all downloaded data"""
    standardizer = DataStandardizer()
    
    print("\n" + "="*70)
    print("🔧 STANDARDIZING DATA")
    print("="*70)
    
    # Binance BTC
    btc_dir = KAGGLE_DIR / 'btc_full_history'
    if btc_dir.exists():
        standardizer.standardize_binance(btc_dir, 'BTCUSDT')
    
    # Gold
    gold_dir = KAGGLE_DIR / 'gold_1970_2024'
    if gold_dir.exists():
        standardizer.standardize_gold(gold_dir)


def label_regimes():
    """Add regime labels to all parquet files"""
    labeler = MacroRegimeLabeler()
    
    print("\n" + "="*70)
    print("🏷️ LABELING MACRO REGIMES")
    print("="*70)
    
    for parquet_file in PARQUET_DIR.glob('*.parquet'):
        print(f"\n--- {parquet_file.name} ---")
        df = pd.read_parquet(parquet_file)
        df = labeler.label_data(df)
        
        # Save with regime labels
        output_path = parquet_file.with_suffix('.regime.parquet')
        df.to_parquet(output_path, index=False)
        
        # Print stats
        stats = labeler.get_regime_stats(df)
        print(f"  Regimes found: {list(stats.keys())}")
        for regime, s in stats.items():
            print(f"    {regime}: {s['count']:,} rows, vol={s['volatility']:.2f}%")


if __name__ == "__main__":
    print("="*70)
    print("📊 DATA DOWNLOADER - Deep History Acquisition")
    print("="*70)
    print(f"Data Directory: {DATA_DIR}")
    print(f"Kaggle Directory: {KAGGLE_DIR}")
    print(f"Parquet Directory: {PARQUET_DIR}")
    
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'download':
            download_all()
        elif cmd == 'standardize':
            standardize_all()
        elif cmd == 'label':
            label_regimes()
        elif cmd == 'all':
            download_all()
            standardize_all()
            label_regimes()
    else:
        print("\nUsage:")
        print("  python data_downloader.py download   - Download from Kaggle")
        print("  python data_downloader.py standardize - Convert to Parquet")
        print("  python data_downloader.py label      - Add regime labels")
        print("  python data_downloader.py all        - Do all steps")
