"""
MARKETFORGE: Dual Strategy System
===================================
BTC: Trend-Following (Triple Barrier)
GOLD: Mean-Reversion (RSI + Bollinger)
CORRELATION: Gold leads BTC signal

Implements:
- Asset-specific strategy selection
- Mean-reversion labels for Gold
- Gold-BTC correlation features
- Cross-asset signal confirmation

Usage:
    python dual_strategy.py label --all
    python dual_strategy.py train --all
    python dual_strategy.py backtest --all
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import yaml
import joblib
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load config
with open(BASE_DIR / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

ASSET_STRATEGY = {
    'BTCUSDT': 'TREND_FOLLOWING',    # Triple Barrier
    'PAXGUSDT': 'MEAN_REVERSION',    # RSI + Bollinger
}


# =============================================================================
# MEAN REVERSION LABELER (FOR GOLD)
# =============================================================================

class MeanReversionLabeler:
    """
    Mean-reversion strategy for Gold.
    
    Buy when:
    - RSI < 30 (oversold)
    - Price below lower Bollinger Band
    - Expecting price to revert to mean
    
    Sell when:
    - RSI > 70 (overbought)
    - Price above upper Bollinger Band
    - Expecting price to revert down
    """
    
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        
        # Timeframe-specific parameters
        self.params = {
            '5m': {'rsi_period': 14, 'rsi_oversold': 25, 'rsi_overbought': 75,
                   'bb_period': 20, 'bb_std': 2.0, 'max_hold': 8, 'target_pct': 0.3},
            '15m': {'rsi_period': 14, 'rsi_oversold': 28, 'rsi_overbought': 72,
                    'bb_period': 20, 'bb_std': 2.0, 'max_hold': 10, 'target_pct': 0.4},
            '30m': {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
                    'bb_period': 20, 'bb_std': 2.0, 'max_hold': 14, 'target_pct': 0.5},
            '1h': {'rsi_period': 14, 'rsi_oversold': 32, 'rsi_overbought': 68,
                   'bb_period': 20, 'bb_std': 2.0, 'max_hold': 18, 'target_pct': 0.6}
        }
        self.p = self.params.get(timeframe, self.params['1h'])
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute RSI and Bollinger Bands"""
        df = df.copy()
        
        # RSI
        delta = df['c'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.ewm(alpha=1/self.p['rsi_period'], min_periods=self.p['rsi_period']).mean()
        avg_loss = loss.ewm(alpha=1/self.p['rsi_period'], min_periods=self.p['rsi_period']).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_mid'] = df['c'].rolling(self.p['bb_period']).mean()
        df['bb_std'] = df['c'].rolling(self.p['bb_period']).std()
        df['bb_upper'] = df['bb_mid'] + self.p['bb_std'] * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - self.p['bb_std'] * df['bb_std']
        df['bb_pct'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        return df
    
    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean-reversion labels.
        
        +1 (BUY): RSI oversold AND price below lower BB
        -1 (SELL): RSI overbought AND price above upper BB
        0: No signal
        """
        df = self.compute_indicators(df)
        
        df['label'] = 0
        df['exit_type'] = 'NONE'
        df['barrier_ret'] = 0.0
        df['exit_bar'] = 0
        
        close = df['c'].values
        rsi = df['rsi'].values
        bb_lower = df['bb_lower'].values
        bb_upper = df['bb_upper'].values
        bb_mid = df['bb_mid'].values
        
        n = len(df)
        target_pct = self.p['target_pct'] / 100
        
        for i in range(self.p['bb_period'], n - self.p['max_hold']):
            # Skip if already labeled
            if df.iloc[i]['label'] != 0:
                continue
            
            current_rsi = rsi[i]
            current_close = close[i]
            current_bb_low = bb_lower[i]
            current_bb_up = bb_upper[i]
            current_bb_mid = bb_mid[i]
            
            # BUY signal: oversold + below lower BB
            if current_rsi < self.p['rsi_oversold'] and current_close < current_bb_low:
                # Check if price reverts to mean
                target_price = current_close * (1 + target_pct)
                stop_price = current_close * (1 - target_pct * 1.5)
                
                for j in range(1, self.p['max_hold'] + 1):
                    future_close = close[i + j]
                    
                    # Hit target (reversion to mean)
                    if future_close >= target_price:
                        df.iloc[i, df.columns.get_loc('label')] = 1
                        df.iloc[i, df.columns.get_loc('exit_type')] = 'TP'
                        df.iloc[i, df.columns.get_loc('barrier_ret')] = (future_close - current_close) / current_close * 100
                        df.iloc[i, df.columns.get_loc('exit_bar')] = j
                        break
                    
                    # Hit stop
                    if future_close <= stop_price:
                        df.iloc[i, df.columns.get_loc('label')] = -1
                        df.iloc[i, df.columns.get_loc('exit_type')] = 'SL'
                        df.iloc[i, df.columns.get_loc('barrier_ret')] = (future_close - current_close) / current_close * 100
                        df.iloc[i, df.columns.get_loc('exit_bar')] = j
                        break
            
            # SELL signal: overbought + above upper BB
            elif current_rsi > self.p['rsi_overbought'] and current_close > current_bb_up:
                target_price = current_close * (1 - target_pct)
                stop_price = current_close * (1 + target_pct * 1.5)
                
                for j in range(1, self.p['max_hold'] + 1):
                    future_close = close[i + j]
                    
                    # Hit target (reversion down)
                    if future_close <= target_price:
                        df.iloc[i, df.columns.get_loc('label')] = -1
                        df.iloc[i, df.columns.get_loc('exit_type')] = 'TP'
                        df.iloc[i, df.columns.get_loc('barrier_ret')] = (current_close - future_close) / current_close * 100
                        df.iloc[i, df.columns.get_loc('exit_bar')] = j
                        break
                    
                    # Hit stop
                    if future_close >= stop_price:
                        df.iloc[i, df.columns.get_loc('label')] = 1
                        df.iloc[i, df.columns.get_loc('exit_type')] = 'SL'
                        df.iloc[i, df.columns.get_loc('barrier_ret')] = (future_close - current_close) / current_close * 100
                        df.iloc[i, df.columns.get_loc('exit_bar')] = j
                        break
        
        return df
    
    def get_stats(self, df: pd.DataFrame) -> dict:
        """Get labeling statistics"""
        labeled = df[df['exit_type'] != 'NONE']
        
        return {
            'total_samples': len(labeled),
            'buy_signals': (labeled['label'] == 1).sum(),
            'sell_signals': (labeled['label'] == -1).sum(),
            'tp_count': (labeled['exit_type'] == 'TP').sum(),
            'sl_count': (labeled['exit_type'] == 'SL').sum(),
            'win_rate': (labeled['exit_type'] == 'TP').mean() * 100 if len(labeled) > 0 else 0,
            'avg_hold': labeled['exit_bar'].mean() if len(labeled) > 0 else 0,
        }


# =============================================================================
# GOLD-BTC CORRELATION FEATURES
# =============================================================================

def compute_correlation_features(btc_df: pd.DataFrame, gold_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cross-asset correlation features.
    
    Gold leads BTC: Use Gold's RSI/momentum to predict BTC
    BTC leads Gold: Use BTC's momentum to predict Gold reversion
    """
    # Align timestamps
    btc_df = btc_df.copy()
    gold_df = gold_df.copy()
    
    # Gold features for BTC prediction
    gold_ret = gold_df['c'].pct_change()
    gold_rsi = gold_df.get('rsi', pd.Series(index=gold_df.index))
    
    # Rolling correlation (Gold leads BTC by 1-3 bars)
    btc_ret = btc_df['c'].pct_change()
    
    # Lagged Gold returns as features for BTC
    for lag in [1, 2, 3, 5]:
        btc_df[f'gold_ret_lag{lag}'] = gold_ret.shift(lag).values[:len(btc_df)] if len(gold_ret) >= len(btc_df) else 0
    
    # Gold RSI for BTC (if available)
    if len(gold_rsi) >= len(btc_df):
        btc_df['gold_rsi'] = gold_rsi.values[:len(btc_df)]
        btc_df['gold_rsi_lag1'] = gold_rsi.shift(1).values[:len(btc_df)]
    
    # BTC momentum for Gold prediction
    btc_momentum = btc_df['c'].diff(5)
    
    for lag in [1, 2, 3]:
        gold_df[f'btc_momentum_lag{lag}'] = btc_momentum.shift(lag).values[:len(gold_df)] if len(btc_momentum) >= len(gold_df) else 0
    
    # Rolling 20-bar correlation
    if len(btc_ret) > 20 and len(gold_ret) > 20:
        min_len = min(len(btc_ret), len(gold_ret))
        corr = btc_ret.iloc[:min_len].rolling(20).corr(gold_ret.iloc[:min_len])
        btc_df['btc_gold_corr'] = corr.values[:len(btc_df)] if len(corr) >= len(btc_df) else 0
        gold_df['btc_gold_corr'] = corr.values[:len(gold_df)] if len(corr) >= len(gold_df) else 0
    
    return btc_df, gold_df


# =============================================================================
# FEATURE ENGINEERING (EXTENDED FOR MEAN REVERSION)
# =============================================================================

class DualStrategyFeatureEngineer:
    """
    Extended feature engineering for dual strategy.
    
    For TREND (BTC): Momentum, trend, volatility features
    For MEAN_REVERSION (Gold): RSI, BB, mean-reversion features
    """
    
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.feature_names: List[str] = []
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features based on strategy type"""
        df = df.copy()
        
        # Common features
        df = self._compute_common_features(df)
        
        if self.strategy == 'MEAN_REVERSION':
            df = self._compute_mean_reversion_features(df)
        else:  # TREND_FOLLOWING
            df = self._compute_trend_features(df)
        
        # Build feature list
        exclude = ['time', 'o', 'h', 'l', 'c', 'v', 'label', 'barrier_ret', 
                   'exit_bar', 'exit_type', 'atr', 'purged']
        self.feature_names = [c for c in df.columns if c not in exclude and not c.startswith('sma_') and not c.startswith('bb_mid') and not c.startswith('bb_std')]
        
        return df
    
    def _compute_common_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Common features for both strategies"""
        # Returns
        for lag in [1, 2, 3, 5, 10]:
            df[f'ret_{lag}'] = np.log(df['c'] / df['c'].shift(lag)) * 100
        
        # Volatility
        df['atr'] = self._compute_atr(df)
        df['atr_pct'] = df['atr'] / df['c'] * 100
        
        for window in [10, 20, 50]:
            df[f'vol_{window}'] = df['ret_1'].rolling(window).std() * np.sqrt(252)
        
        # RSI
        delta = df['c'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Time features
        if 'time' in df.columns:
            try:
                if df['time'].dtype in ['int64', 'float64']:
                    dt = pd.to_datetime(df['time'], unit='ms')
                else:
                    dt = pd.to_datetime(df['time'])
                df['hour'] = dt.dt.hour
                df['day_of_week'] = dt.dt.dayofweek
            except:
                df['hour'] = 12
                df['day_of_week'] = 3
        
        return df
    
    def _compute_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features for mean-reversion strategy (Gold)"""
        # Bollinger Bands
        df['bb_mid'] = df['c'].rolling(20).mean()
        df['bb_std'] = df['c'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pct'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
        
        # Distance from mean
        df['dist_from_mean'] = (df['c'] - df['bb_mid']) / df['bb_mid'] * 100
        df['dist_from_mean_std'] = df['dist_from_mean'] / (df['bb_std'] / df['bb_mid'] * 100 + 1e-10)
        
        # RSI extremes
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_extreme'] = df['rsi_oversold'] | df['rsi_overbought']
        
        # Mean reversion indicators
        df['price_below_bb'] = (df['c'] < df['bb_lower']).astype(int)
        df['price_above_bb'] = (df['c'] > df['bb_upper']).astype(int)
        
        # Reversion setup score (0-4)
        df['reversion_buy_score'] = (
            df['rsi_oversold'] + 
            df['price_below_bb'] + 
            (df['dist_from_mean'] < -2).astype(int) +
            (df['ret_1'] < 0).astype(int)
        )
        df['reversion_sell_score'] = (
            df['rsi_overbought'] + 
            df['price_above_bb'] + 
            (df['dist_from_mean'] > 2).astype(int) +
            (df['ret_1'] > 0).astype(int)
        )
        
        # Stochastic RSI
        rsi_min = df['rsi'].rolling(14).min()
        rsi_max = df['rsi'].rolling(14).max()
        df['stoch_rsi'] = (df['rsi'] - rsi_min) / (rsi_max - rsi_min + 1e-10)
        
        return df
    
    def _compute_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features for trend-following strategy (BTC)"""
        # MACD
        ema_12 = df['c'].ewm(span=12).mean()
        ema_26 = df['c'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # SMAs
        for period in [20, 50, 200]:
            df[f'sma_{period}'] = df['c'].rolling(period).mean()
            df[f'dist_sma_{period}'] = (df['c'] - df[f'sma_{period}']) / df['c'] * 100
        
        # MA crossovers
        df['ma_20_above_50'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['ma_50_above_200'] = (df['sma_50'] > df['sma_200']).astype(int)
        
        # ADX
        high = df['h']
        low = df['l']
        plus_dm = (high - high.shift()).where((high - high.shift()) > (low.shift() - low), 0)
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = (low.shift() - low).where((low.shift() - low) > (high - high.shift()), 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        
        tr = pd.concat([high - low, (high - df['c'].shift()).abs(), (low - df['c'].shift()).abs()], axis=1).max(axis=1)
        atr_14 = tr.ewm(alpha=1/14, min_periods=14).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr_14)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr_14)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.ewm(alpha=1/14).mean()
        
        # Trend strength
        df['trend_strength'] = df['adx'] * np.sign(df['macd'])
        
        # Momentum
        df['momentum_10'] = df['c'] - df['c'].shift(10)
        df['momentum_20'] = df['c'] - df['c'].shift(20)
        df['roc_10'] = (df['c'] - df['c'].shift(10)) / df['c'].shift(10) * 100
        
        return df
    
    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """Compute ATR"""
        tr = pd.concat([
            df['h'] - df['l'],
            (df['h'] - df['c'].shift()).abs(),
            (df['l'] - df['c'].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/14, min_periods=14).mean()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_asset_dual_strategy(symbol: str, timeframe: str) -> Tuple[pd.DataFrame, dict]:
    """Process asset with appropriate strategy"""
    print(f"\n{'='*60}")
    print(f"🔧 Processing {symbol} {timeframe}")
    print(f"{'='*60}")
    
    strategy = ASSET_STRATEGY.get(symbol, 'TREND_FOLLOWING')
    print(f"   📋 Strategy: {strategy}")
    
    # Load data
    data_path = DATA_DIR / f"{symbol}_{timeframe}.csv"
    if not data_path.exists():
        print(f"   ❌ Data not found: {data_path}")
        return None, None
    
    df = pd.read_csv(data_path)
    df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
    print(f"   📊 Loaded {len(df):,} rows")
    
    # Apply strategy-specific labeling
    if strategy == 'MEAN_REVERSION':
        print("   🔄 Applying Mean-Reversion labels (RSI + BB)...")
        labeler = MeanReversionLabeler(timeframe)
        df = labeler.label(df)
        stats = labeler.get_stats(df)
    else:
        print("   📈 Applying Trend-Following labels (Triple-Barrier)...")
        from label_and_features import TripleBarrierLabeler
        labeler = TripleBarrierLabeler(timeframe)
        df = labeler.label(df)
        stats = labeler.get_stats(df)
    
    print(f"      Signals: {stats.get('total_samples', 0):,} | Win Rate: {stats.get('win_rate', 0):.1f}%")
    
    # Compute features
    print(f"   📐 Computing {strategy} features...")
    engineer = DualStrategyFeatureEngineer(strategy)
    df = engineer.compute_features(df)
    print(f"      Features: {len(engineer.feature_names)}")
    
    # Save
    output_path = DATA_DIR / f"{symbol}_{timeframe}_labeled.parquet"
    df.to_parquet(output_path, index=False)
    print(f"   💾 Saved: {output_path}")
    
    # Save feature schema
    schema = {
        'version': '2.0.0',
        'strategy': strategy,
        'features': engineer.feature_names,
        'count': len(engineer.feature_names),
        'created_at': datetime.now().isoformat()
    }
    schema_path = MODEL_DIR / f"feature_schema_{symbol}_{timeframe}.json"
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    return df, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dual Strategy System')
    parser.add_argument('command', choices=['label', 'all'], help='Command')
    parser.add_argument('--symbol', default=None, help='Symbol')
    parser.add_argument('--timeframe', default=None, help='Timeframe')
    
    args = parser.parse_args()
    
    if args.command == 'all' or args.command == 'label':
        for symbol in ['BTCUSDT', 'PAXGUSDT']:
            for tf in ['15m', '30m', '1h']:  # Skip 5m
                process_asset_dual_strategy(symbol, tf)
