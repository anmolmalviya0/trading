"""
PHASE 1: LABELING & FEATURE ENGINEERING
========================================
Triple-barrier labeling + multi-timeframe features.

Based on Marcos Lopez de Prado methodology:
- ATR-normalized barriers (not fixed %)
- First barrier hit determines label
- Walkforward-safe feature computation

Usage:
    python label_and_features.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'market_data'
PARQUET_DIR = DATA_DIR / 'parquet'
OUTPUT_DIR = BASE_DIR / 'datasets'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_CONFIG = {
    'tp_atr_mult': 2.0,      # Take profit = 2x ATR
    'sl_atr_mult': 1.0,      # Stop loss = 1x ATR
    'max_holding': 10,       # Max bars to hold
    'atr_period': 14,
}

FEATURE_CONFIG = {
    'return_periods': [1, 3, 5, 10, 20],
    'ma_periods': [10, 20, 50, 100, 200],
    'roc_periods': [5, 10, 20],
    'vol_windows': [10, 20, 50],
}


# === ATR CALCULATION ===

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['h']
    low = df['l']
    close = df['c']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr


# === TRIPLE BARRIER LABELING ===

def triple_barrier_label(df: pd.DataFrame, 
                         tp_mult: float = 2.0,
                         sl_mult: float = 1.0,
                         max_holding: int = 10) -> pd.DataFrame:
    """
    Apply Triple Barrier Method.
    
    Returns DataFrame with:
    - label: 1 (TP hit), 0 (SL hit or timeout loss), -1 (no trade)
    - barrier_ret: actual return at exit
    - barrier_touch: 'tp', 'sl', 'timeout'
    - holding_bars: bars held
    """
    df = df.copy()
    atr = calculate_atr(df, LABEL_CONFIG['atr_period'])
    
    labels = []
    returns = []
    touches = []
    holdings = []
    
    for i in range(len(df)):
        # Skip if not enough future data
        if i + max_holding >= len(df):
            labels.append(np.nan)
            returns.append(np.nan)
            touches.append(None)
            holdings.append(np.nan)
            continue
        
        entry = df['c'].iloc[i]
        current_atr = atr.iloc[i]
        
        # Skip if ATR invalid
        if pd.isna(current_atr) or current_atr <= 0:
            labels.append(np.nan)
            returns.append(np.nan)
            touches.append(None)
            holdings.append(np.nan)
            continue
        
        # Barriers
        upper = entry + current_atr * tp_mult
        lower = entry - current_atr * sl_mult
        
        label = 0
        ret = 0
        touch = 'timeout'
        hold = max_holding
        
        # Check each future bar
        for j in range(1, max_holding + 1):
            future_idx = i + j
            if future_idx >= len(df):
                break
            
            high = df['h'].iloc[future_idx]
            low = df['l'].iloc[future_idx]
            
            # Check TP first (optimistic)
            if high >= upper:
                label = 1
                ret = (upper - entry) / entry
                touch = 'tp'
                hold = j
                break
            
            # Check SL
            if low <= lower:
                label = 0
                ret = (lower - entry) / entry
                touch = 'sl'
                hold = j
                break
        
        # Timeout handling
        if touch == 'timeout':
            final = df['c'].iloc[min(i + max_holding, len(df) - 1)]
            ret = (final - entry) / entry
            label = 1 if ret > 0 else 0
        
        labels.append(label)
        returns.append(ret)
        touches.append(touch)
        holdings.append(hold)
    
    df['label'] = labels
    df['barrier_ret'] = returns
    df['barrier_touch'] = touches
    df['holding_bars'] = holdings
    
    return df


# === FEATURE ENGINEERING ===

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive feature set.
    All features use ONLY past data (no leakage).
    """
    df = df.copy()
    
    # === RETURNS ===
    for p in FEATURE_CONFIG['return_periods']:
        df[f'ret_{p}'] = df['c'].pct_change(p) * 100
    
    # Log returns for stationarity
    df['log_ret'] = np.log(df['c'] / df['c'].shift(1)) * 100
    
    # === RSI ===
    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # === MACD ===
    ema12 = df['c'].ewm(span=12).mean()
    ema26 = df['c'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # === MOVING AVERAGES ===
    for p in FEATURE_CONFIG['ma_periods']:
        df[f'sma{p}'] = df['c'].rolling(p).mean()
        df[f'ema{p}'] = df['c'].ewm(span=p).mean()
        df[f'dist_sma{p}'] = (df['c'] - df[f'sma{p}']) / df['c'] * 100
    
    # === BOLLINGER BANDS ===
    df['bb_mid'] = df['sma20']
    df['bb_std'] = df['c'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    df['bb_position'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # === ATR ===
    df['atr'] = calculate_atr(df)
    df['atr_pct'] = df['atr'] / df['c'] * 100
    
    # === VOLATILITY ===
    for w in FEATURE_CONFIG['vol_windows']:
        df[f'vol_{w}'] = df['log_ret'].rolling(w).std() * np.sqrt(252)
    
    df['vol_rank'] = df['vol_20'].rolling(100).rank(pct=True)
    
    # === ROC (Rate of Change) ===
    for p in FEATURE_CONFIG['roc_periods']:
        df[f'roc_{p}'] = (df['c'] / df['c'].shift(p) - 1) * 100
    
    # === VOLUME ===
    df['vol_sma'] = df['v'].rolling(20).mean()
    df['vol_ratio'] = df['v'] / (df['vol_sma'] + 1e-10)
    df['vol_zscore'] = (df['v'] - df['vol_sma']) / (df['v'].rolling(20).std() + 1e-10)
    
    # === TREND INDICATORS ===
    df['trend_10_20'] = (df['sma10'] - df['sma20']) / df['c'] * 100
    df['trend_20_50'] = (df['sma20'] - df['sma50']) / df['c'] * 100
    df['trend_50_100'] = (df['sma50'] - df['sma100']) / df['c'] * 100
    df['trend_50_200'] = (df['sma50'] - df['sma200']) / df['c'] * 100
    
    # === MOMENTUM ===
    df['momentum_10'] = df['c'] - df['c'].shift(10)
    df['momentum_20'] = df['c'] - df['c'].shift(20)
    
    # === REGIME DETECTION ===
    df['regime_volatility'] = pd.cut(
        df['vol_rank'], 
        bins=[0, 0.25, 0.75, 1.0], 
        labels=['low', 'normal', 'high']
    )
    
    # Trend regime based on MA alignment
    df['regime_trend'] = np.where(
        (df['sma20'] > df['sma50']) & (df['sma50'] > df['sma100']),
        'uptrend',
        np.where(
            (df['sma20'] < df['sma50']) & (df['sma50'] < df['sma100']),
            'downtrend',
            'choppy'
        )
    )
    
    return df


# === FEATURE LIST ===

def get_feature_columns() -> List[str]:
    """Return list of feature columns for ML"""
    return [
        # Returns
        'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20', 'log_ret',
        # Momentum
        'rsi', 'macd_hist', 'momentum_10', 'momentum_20',
        # ROC
        'roc_5', 'roc_10', 'roc_20',
        # Trend
        'dist_sma10', 'dist_sma20', 'dist_sma50', 'dist_sma100', 'dist_sma200',
        'trend_10_20', 'trend_20_50', 'trend_50_100',
        # Volatility
        'atr_pct', 'vol_10', 'vol_20', 'vol_50', 'vol_rank',
        # Bollinger
        'bb_width', 'bb_position',
        # Volume
        'vol_ratio', 'vol_zscore',
    ]


# === MULTI-TIMEFRAME MERGE ===

def merge_multi_timeframe(symbol: str, timeframes: List[str] = ['5m', '15m', '1h']) -> pd.DataFrame:
    """
    Merge features from multiple timeframes.
    Base timeframe is the lowest (e.g., 5m).
    Higher timeframes are forward-filled.
    """
    base_tf = timeframes[0]
    
    # Load base
    base_path = PARQUET_DIR / f"{symbol}_{base_tf}.parquet"
    if not base_path.exists():
        base_path = DATA_DIR / f"{symbol}_{base_tf}.csv"
        df_base = pd.read_csv(base_path)
        df_base.columns = ['time', 'o', 'h', 'l', 'c', 'v']
    else:
        df_base = pd.read_parquet(base_path)
    
    df_base['time'] = pd.to_datetime(df_base['time'], utc=True)
    df_base = df_base.set_index('time').sort_index()
    
    # Create base features
    df_base = create_features(df_base.reset_index()).set_index('time')
    
    # Merge higher timeframes
    for tf in timeframes[1:]:
        tf_path = PARQUET_DIR / f"{symbol}_{tf}.parquet"
        if not tf_path.exists():
            tf_path = DATA_DIR / f"{symbol}_{tf}.csv"
            df_tf = pd.read_csv(tf_path)
            df_tf.columns = ['time', 'o', 'h', 'l', 'c', 'v']
        else:
            df_tf = pd.read_parquet(tf_path)
        
        df_tf['time'] = pd.to_datetime(df_tf['time'], utc=True)
        df_tf = df_tf.set_index('time').sort_index()
        
        # Create features
        df_tf = create_features(df_tf.reset_index()).set_index('time')
        
        # Rename columns with timeframe suffix
        feature_cols = get_feature_columns()
        rename_dict = {col: f"{col}_{tf}" for col in feature_cols if col in df_tf.columns}
        df_tf = df_tf[list(rename_dict.keys())].rename(columns=rename_dict)
        
        # Merge (forward fill higher TF onto lower)
        df_base = df_base.join(df_tf, how='left')
        df_base = df_base.ffill()
    
    return df_base.reset_index()


# === PROCESS DATASET ===

def process_symbol(symbol: str, timeframe: str = '1h') -> pd.DataFrame:
    """Process single symbol/timeframe"""
    print(f"\n📊 Processing {symbol} {timeframe}...")
    
    # Load data
    parquet_path = PARQUET_DIR / f"{symbol}_{timeframe}.parquet"
    csv_path = DATA_DIR / f"{symbol}_{timeframe}.csv"
    
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
    else:
        raise FileNotFoundError(f"No data for {symbol} {timeframe}")
    
    print(f"   Loaded: {len(df):,} rows")
    
    # Features
    df = create_features(df)
    print(f"   Features: {len(get_feature_columns())} columns")
    
    # Labels
    df = triple_barrier_label(df, 
                              tp_mult=LABEL_CONFIG['tp_atr_mult'],
                              sl_mult=LABEL_CONFIG['sl_atr_mult'],
                              max_holding=LABEL_CONFIG['max_holding'])
    
    # Stats
    valid = df['label'].notna()
    label_dist = df.loc[valid, 'label'].value_counts(normalize=True)
    touch_dist = df.loc[valid, 'barrier_touch'].value_counts(normalize=True)
    
    print(f"   Labels: Win={label_dist.get(1, 0)*100:.1f}%, Loss={label_dist.get(0, 0)*100:.1f}%")
    print(f"   Exits: TP={touch_dist.get('tp', 0)*100:.1f}%, SL={touch_dist.get('sl', 0)*100:.1f}%, Timeout={touch_dist.get('timeout', 0)*100:.1f}%")
    
    return df


# === MAIN ===

if __name__ == "__main__":
    print("="*70)
    print("📐 PHASE 1: LABELING & FEATURE ENGINEERING")
    print("="*70)
    
    symbols = ['BTCUSDT', 'PAXGUSDT']
    timeframe = '1h'
    
    for symbol in symbols:
        try:
            # Process
            df = process_symbol(symbol, timeframe)
            
            # Save
            output_path = OUTPUT_DIR / f"{symbol}_{timeframe}_labeled.parquet"
            df.to_parquet(output_path, index=False)
            print(f"   💾 Saved: {output_path.name}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Export feature list
    feature_list = get_feature_columns()
    feature_path = OUTPUT_DIR / 'feature_columns.txt'
    with open(feature_path, 'w') as f:
        f.write('\n'.join(feature_list))
    
    print("\n" + "="*70)
    print("✅ LABELING COMPLETE")
    print("="*70)
    print(f"   Datasets: {OUTPUT_DIR}")
    print(f"   Features: {len(feature_list)} columns")
    print(f"   Label config: TP={LABEL_CONFIG['tp_atr_mult']}x ATR, SL={LABEL_CONFIG['sl_atr_mult']}x ATR")
