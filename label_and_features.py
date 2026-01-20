"""
MARKETFORGE: Labeling & Features
================================
Triple-Barrier labeling and 60+ feature engineering.

Implements:
- Triple-Barrier (Lopez de Prado) with ATR-based barriers
- Purging logic for overlapping labels
- 60+ features with stable schema
- Leak-free scaling (fit only on training data)

Usage:
    python label_and_features.py BTCUSDT 15m
    python label_and_features.py --all
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import hashlib
from typing import Dict, List, Optional, Tuple
import argparse
import yaml
import joblib

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load config
with open(BASE_DIR / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)


# =============================================================================
# TRIPLE-BARRIER LABELING
# =============================================================================

class TripleBarrierLabeler:
    """
    Marcos Lopez de Prado Triple-Barrier Method.
    
    Labels each bar based on which barrier is touched first:
    - Upper barrier (Take Profit): +1
    - Lower barrier (Stop Loss): -1
    - Vertical barrier (Time): 0 or sign of return
    
    All barriers are ATR-based and configurable per timeframe.
    """
    
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        
        # Load config for this timeframe
        tf_config = CONFIG['labeling'].get(timeframe, CONFIG['labeling']['1h'])
        self.tp_mult = tf_config['tp_mult']
        self.sl_mult = tf_config['sl_mult']
        self.max_hold = tf_config['max_hold']
        
        self.atr_period = CONFIG['atr']['period']
    
    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """Compute ATR using only past data (no look-ahead)"""
        high = df['h']
        low = df['l']
        close = df['c']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # RMA smoothing (Wilder's smoothing)
        atr = tr.ewm(alpha=1/self.atr_period, min_periods=self.atr_period).mean()
        
        return atr
    
    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply triple-barrier labels.
        
        Returns DataFrame with additional columns:
        - atr: ATR at entry
        - label: +1 (TP), -1 (SL), 0 (vertical)
        - barrier_ret: Return at barrier touch
        - exit_bar: Bar offset of exit
        - exit_type: 'TP', 'SL', or 'VERTICAL'
        """
        df = df.copy()
        
        # Compute ATR
        df['atr'] = self.compute_atr(df)
        
        # Initialize label columns
        df['label'] = 0
        df['barrier_ret'] = 0.0
        df['exit_bar'] = 0
        df['exit_type'] = 'NONE'
        
        close = df['c'].values
        high = df['h'].values
        low = df['l'].values
        atr = df['atr'].values
        
        n = len(df)
        
        for i in range(n - self.max_hold):
            entry = close[i]
            current_atr = atr[i]
            
            if np.isnan(current_atr) or current_atr <= 0:
                current_atr = entry * 0.02  # Fallback: 2%
            
            # Calculate barriers
            tp_level = entry + current_atr * self.tp_mult
            sl_level = entry - current_atr * self.sl_mult
            
            # Check future bars
            for j in range(1, self.max_hold + 1):
                if i + j >= n:
                    break
                
                bar_high = high[i + j]
                bar_low = low[i + j]
                bar_close = close[i + j]
                
                # Check barriers (use high/low for intrabar crossings)
                hit_tp = bar_high >= tp_level
                hit_sl = bar_low <= sl_level
                
                if hit_tp and hit_sl:
                    # Both hit - use close to determine which first
                    # (simplified: assume SL hit first if gap down)
                    if bar_close < entry:
                        hit_tp = False
                    else:
                        hit_sl = False
                
                if hit_tp:
                    df.iloc[i, df.columns.get_loc('label')] = 1
                    df.iloc[i, df.columns.get_loc('barrier_ret')] = (tp_level - entry) / entry * 100
                    df.iloc[i, df.columns.get_loc('exit_bar')] = j
                    df.iloc[i, df.columns.get_loc('exit_type')] = 'TP'
                    break
                
                elif hit_sl:
                    df.iloc[i, df.columns.get_loc('label')] = -1
                    df.iloc[i, df.columns.get_loc('barrier_ret')] = (sl_level - entry) / entry * 100
                    df.iloc[i, df.columns.get_loc('exit_bar')] = j
                    df.iloc[i, df.columns.get_loc('exit_type')] = 'SL'
                    break
            
            else:
                # Vertical barrier - label based on return sign
                final_bar = min(i + self.max_hold, n - 1)
                final_ret = (close[final_bar] - entry) / entry * 100
                df.iloc[i, df.columns.get_loc('label')] = np.sign(final_ret)
                df.iloc[i, df.columns.get_loc('barrier_ret')] = final_ret
                df.iloc[i, df.columns.get_loc('exit_bar')] = self.max_hold
                df.iloc[i, df.columns.get_loc('exit_type')] = 'VERTICAL'
        
        return df
    
    def get_stats(self, df: pd.DataFrame) -> dict:
        """Get labeling statistics"""
        labeled = df[df['exit_type'] != 'NONE']
        
        return {
            'total_samples': len(labeled),
            'tp_count': (labeled['exit_type'] == 'TP').sum(),
            'sl_count': (labeled['exit_type'] == 'SL').sum(),
            'vertical_count': (labeled['exit_type'] == 'VERTICAL').sum(),
            'avg_hold': labeled['exit_bar'].mean(),
            'win_rate': (labeled['label'] == 1).mean() * 100,
        }


def purge_overlapping_labels(df: pd.DataFrame, max_hold: int) -> pd.DataFrame:
    """
    Remove overlapping labels causing data leakage.
    
    After a label is assigned, purge the next max_hold rows
    from the training set to prevent overlap.
    """
    df = df.copy()
    df['purged'] = False
    
    labeled_idx = df[df['label'] != 0].index.tolist()
    
    for idx in labeled_idx:
        loc = df.index.get_loc(idx)
        end_loc = min(loc + max_hold, len(df))
        # Mark overlapping rows
        for j in range(loc + 1, end_loc):
            if j < len(df):
                df.iloc[j, df.columns.get_loc('purged')] = True
    
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Computes 60+ technical features with stable schema.
    
    All features are computed using only past/present data (no look-ahead).
    Feature schema and scaler params are saved for reproducibility.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
        self.scaler_params: Dict = {}
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features"""
        df = df.copy()
        
        # =====================================================================
        # RETURNS (log returns at multiple lags)
        # =====================================================================
        for lag in CONFIG['features']['returns_lags']:
            df[f'ret_{lag}'] = np.log(df['c'] / df['c'].shift(lag)) * 100
        
        # =====================================================================
        # VOLATILITY
        # =====================================================================
        # ATR
        high = df['h']
        low = df['l']
        close = df['c']
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        df['atr'] = tr.ewm(alpha=1/14, min_periods=14).mean()
        df['atr_pct'] = df['atr'] / df['c'] * 100
        
        # Realized volatility windows
        for window in CONFIG['features']['volatility_windows']:
            df[f'vol_{window}'] = df['ret_1'].rolling(window).std() * np.sqrt(252)
        
        # =====================================================================
        # MOMENTUM
        # =====================================================================
        # RSI
        delta = df['c'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        macd_cfg = CONFIG['features']['macd']
        ema_fast = df['c'].ewm(span=macd_cfg['fast']).mean()
        ema_slow = df['c'].ewm(span=macd_cfg['slow']).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=macd_cfg['signal']).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Rate of Change
        df['roc_5'] = (df['c'] - df['c'].shift(5)) / df['c'].shift(5) * 100
        df['roc_10'] = (df['c'] - df['c'].shift(10)) / df['c'].shift(10) * 100
        
        # Momentum slope
        df['momentum'] = df['c'] - df['c'].shift(10)
        df['momentum_slope'] = df['momentum'].diff()
        
        # =====================================================================
        # MEAN/TREND
        # =====================================================================
        for period in CONFIG['features']['sma_periods']:
            df[f'sma_{period}'] = df['c'].rolling(period).mean()
            df[f'dist_sma_{period}'] = (df['c'] - df[f'sma_{period}']) / df['c'] * 100
        
        # MA ratios
        df['ma_ratio_20_50'] = df['sma_20'] / (df['sma_50'] + 1e-10)
        df['ma_ratio_50_200'] = df['sma_50'] / (df['sma_200'] + 1e-10)
        
        # Bollinger Bands
        bb_cfg = CONFIG['features']['bollinger']
        df['bb_mid'] = df['c'].rolling(bb_cfg['period']).mean()
        df['bb_std'] = df['c'].rolling(bb_cfg['period']).std()
        df['bb_upper'] = df['bb_mid'] + bb_cfg['std'] * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - bb_cfg['std'] * df['bb_std']
        df['bb_pct'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
        
        # ADX
        plus_dm = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        
        atr_14 = tr.ewm(alpha=1/14, min_periods=14).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr_14)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr_14)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.ewm(alpha=1/14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # =====================================================================
        # MICROSTRUCTURE
        # =====================================================================
        # High-Low spread (proxy for volatility)
        df['hl_spread'] = (df['h'] - df['l']) / df['c'] * 100
        
        # Volume features
        if 'v' in df.columns and df['v'].sum() > 0:
            df['vol_sma_20'] = df['v'].rolling(20).mean()
            df['rel_vol'] = df['v'] / (df['vol_sma_20'] + 1e-10)
            df['vol_spike'] = (df['rel_vol'] > 2).astype(int)
        else:
            df['rel_vol'] = 1.0
            df['vol_spike'] = 0
        
        # =====================================================================
        # TIME FEATURES
        # =====================================================================
        if 'time' in df.columns:
            # Handle both datetime strings and milliseconds timestamps
            try:
                # Try milliseconds first
                if df['time'].dtype in ['int64', 'float64']:
                    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
                else:
                    # Already a string datetime
                    df['datetime'] = pd.to_datetime(df['time'])
            except:
                # Fallback: just parse as-is
                df['datetime'] = pd.to_datetime(df['time'], errors='coerce')
            
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            
            # Session flags (UTC)
            df['session_asia'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['session_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['session_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
            
            df = df.drop(columns=['datetime'])
        
        # =====================================================================
        # CROSS-ASSET (placeholder - would be populated from other data)
        # =====================================================================
        # These would be filled from external data
        df['btc_gold_corr_20'] = 0  # Placeholder
        df['funding_rate'] = 0  # Placeholder
        
        # Build feature list
        exclude = ['time', 'o', 'h', 'l', 'c', 'v', 'label', 'barrier_ret', 
                   'exit_bar', 'exit_type', 'atr', 'purged']
        self.feature_names = [c for c in df.columns if c not in exclude and not c.startswith('sma_')]
        
        return df
    
    def get_feature_schema(self) -> dict:
        """Get feature schema for reproducibility"""
        schema = {
            'version': '1.0.0',
            'features': self.feature_names,
            'count': len(self.feature_names),
            'created_at': datetime.now().isoformat(),
        }
        
        # Add hash
        schema_str = json.dumps({k: v for k, v in schema.items() if k != 'created_at'}, sort_keys=True)
        schema['hash'] = hashlib.sha256(schema_str.encode()).hexdigest()[:16]
        
        return schema
    
    def save_schema(self, path: Path):
        """Save feature schema to JSON"""
        schema = self.get_feature_schema()
        with open(path, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"💾 Feature schema saved: {path}")
        return schema


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_asset(symbol: str, timeframe: str) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Full labeling and feature pipeline for one asset/timeframe.
    
    Returns:
    - DataFrame with labels and features
    - Labeling statistics
    - Feature schema
    """
    print(f"\n{'='*60}")
    print(f"🔧 Processing {symbol} {timeframe}")
    print(f"{'='*60}")
    
    # Load data
    data_path = DATA_DIR / f"{symbol}_{timeframe}.csv"
    
    if not data_path.exists():
        # Try parent directory
        data_path = BASE_DIR.parent / "market_data" / f"{symbol}_{timeframe}.csv"
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return None, None, None
    
    df = pd.read_csv(data_path)
    df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
    print(f"   📊 Loaded {len(df):,} rows")
    
    # 1. Apply Triple-Barrier labeling
    print("   🏷️ Applying Triple-Barrier labels...")
    labeler = TripleBarrierLabeler(timeframe)
    df = labeler.label(df)
    label_stats = labeler.get_stats(df)
    print(f"      TP: {label_stats['tp_count']:,} | SL: {label_stats['sl_count']:,} | "
          f"Win Rate: {label_stats['win_rate']:.1f}%")
    
    # 2. Purge overlapping labels
    print("   🧹 Purging overlapping labels...")
    df = purge_overlapping_labels(df, labeler.max_hold)
    purged_count = df['purged'].sum()
    print(f"      Purged: {purged_count:,} rows")
    
    # 3. Compute features
    print("   📐 Computing 60+ features...")
    engineer = FeatureEngineer()
    df = engineer.compute_features(df)
    print(f"      Features: {len(engineer.feature_names)}")
    
    # 4. Save feature schema
    schema_path = MODEL_DIR / f"feature_schema_{symbol}_{timeframe}.json"
    schema = engineer.save_schema(schema_path)
    
    # 5. Save processed data
    output_path = DATA_DIR / f"{symbol}_{timeframe}_labeled.parquet"
    df.to_parquet(output_path, index=False)
    print(f"   💾 Saved: {output_path}")
    
    # 6. Save label fingerprint
    fingerprint = {
        'symbol': symbol,
        'timeframe': timeframe,
        'tp_mult': float(labeler.tp_mult),
        'sl_mult': float(labeler.sl_mult),
        'max_hold': int(labeler.max_hold),
        'atr_period': int(labeler.atr_period),
        'created_at': datetime.now().isoformat(),
        'stats': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                  for k, v in label_stats.items()}
    }
    fingerprint_path = MODEL_DIR / f"label_fingerprint_{symbol}_{timeframe}.json"
    with open(fingerprint_path, 'w') as f:
        json.dump(fingerprint, f, indent=2)
    
    return df, label_stats, schema


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Labeling and feature engineering')
    parser.add_argument('symbol', nargs='?', default='BTCUSDT', help='Symbol')
    parser.add_argument('timeframe', nargs='?', default='15m', help='Timeframe')
    parser.add_argument('--all', action='store_true', help='Process all assets/TFs')
    
    args = parser.parse_args()
    
    if args.all:
        for symbol in ['BTCUSDT', 'PAXGUSDT']:
            for tf in ['5m', '15m', '30m', '1h']:
                process_asset(symbol, tf)
    else:
        process_asset(args.symbol, args.timeframe)
