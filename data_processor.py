"""
INSTITUTIONAL DATA PROCESSOR
===========================
Implements De Prado standards:
- Dollar Bar construction from tick/trade data
- Fractional Differentiation (FracDiff)
- Stationarity testing

Reference: "Advances in Financial Machine Learning" by Marcos Lopez de Prado
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DOLLAR BARS
# =============================================================================

def construct_dollar_bars(trades_df: pd.DataFrame, 
                          dollar_threshold: float = 1_000_000) -> pd.DataFrame:
    """
    Construct Dollar Bars from tick/trade data.
    
    Dollar bars are sampled when a cumulative dollar amount is traded,
    rather than time-based sampling. This normalizes volatility.
    
    Parameters:
    -----------
    trades_df : DataFrame with columns ['timestamp', 'price', 'volume']
    dollar_threshold : Dollar value to trigger new bar (e.g., $1M)
    
    Returns:
    --------
    DataFrame with OHLCV dollar bars
    """
    trades_df = trades_df.copy()
    
    # Calculate dollar value per trade
    trades_df['dollar_value'] = trades_df['price'] * trades_df['volume']
    
    bars = []
    cumulative_dollar = 0
    open_price = None
    high_price = -np.inf
    low_price = np.inf
    close_price = None
    cumulative_volume = 0
    bar_start_time = None
    
    for idx, row in trades_df.iterrows():
        if open_price is None:
            open_price = row['price']
            bar_start_time = row['timestamp']
        
        high_price = max(high_price, row['price'])
        low_price = min(low_price, row['price'])
        close_price = row['price']
        cumulative_volume += row['volume']
        cumulative_dollar += row['dollar_value']
        
        if cumulative_dollar >= dollar_threshold:
            bars.append({
                'timestamp': bar_start_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': cumulative_volume,
                'dollar_volume': cumulative_dollar
            })
            
            # Reset
            cumulative_dollar = 0
            cumulative_volume = 0
            open_price = None
            high_price = -np.inf
            low_price = np.inf
    
    return pd.DataFrame(bars)


def simulate_dollar_bars_from_ohlcv(df: pd.DataFrame, 
                                     dollar_threshold: float = 1_000_000) -> pd.DataFrame:
    """
    Approximate dollar bars from OHLCV data (when tick data unavailable).
    Uses volume * close as proxy for dollar volume.
    
    Parameters:
    -----------
    df : DataFrame with columns ['time', 'o', 'h', 'l', 'c', 'v']
    dollar_threshold : Dollar value per bar
    
    Returns:
    --------
    DataFrame with simulated dollar bars
    """
    df = df.copy()
    df['dollar_vol'] = df['c'] * df['v']
    
    bars = []
    cumulative = 0
    open_price = None
    high_price = -np.inf
    low_price = np.inf
    close_price = None
    vol_sum = 0
    bar_start = None
    
    for idx, row in df.iterrows():
        if open_price is None:
            open_price = row['o']
            bar_start = row.get('time', idx)
        
        high_price = max(high_price, row['h'])
        low_price = min(low_price, row['l'])
        close_price = row['c']
        vol_sum += row['v']
        cumulative += row['dollar_vol']
        
        if cumulative >= dollar_threshold:
            bars.append({
                'time': bar_start,
                'o': open_price,
                'h': high_price,
                'l': low_price,
                'c': close_price,
                'v': vol_sum,
                'dollar_vol': cumulative
            })
            
            cumulative = 0
            vol_sum = 0
            open_price = None
            high_price = -np.inf
            low_price = np.inf
    
    return pd.DataFrame(bars)


# =============================================================================
# FRACTIONAL DIFFERENTIATION (FracDiff)
# =============================================================================

def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute weights for Fixed-Window Fractional Differentiation.
    
    Parameters:
    -----------
    d : Fractional differentiation order (0 < d < 1)
    threshold : Minimum weight to include
    
    Returns:
    --------
    Array of weights
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights[::-1])


def frac_diff_ffd(series: pd.Series, d: float = 0.4, threshold: float = 1e-5) -> pd.Series:
    """
    Apply Fixed-Window Fractional Differentiation.
    
    This preserves memory while achieving stationarity.
    d=0: Original series (full memory, non-stationary)
    d=1: First difference (stationary but memory-less)
    d~0.4: Good balance for most financial series
    
    Parameters:
    -----------
    series : Price series to differentiate
    d : Differentiation order (typically 0.3-0.5 for prices)
    threshold : Weight cutoff
    
    Returns:
    --------
    Fractionally differentiated series
    """
    weights = get_weights_ffd(d, threshold)
    width = len(weights)
    
    output = []
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1:i + 1].values
        output.append(np.dot(weights, window))
    
    result = pd.Series(output, index=series.index[width - 1:])
    return result


def find_optimal_d(series: pd.Series, 
                   target_adf_pvalue: float = 0.05,
                   d_range: Tuple[float, float] = (0.0, 1.0),
                   precision: float = 0.01) -> float:
    """
    Find minimum d that achieves stationarity while preserving maximum memory.
    
    Parameters:
    -----------
    series : Price series
    target_adf_pvalue : Target p-value for ADF test (default 0.05)
    d_range : Range to search
    precision : Step size for search
    
    Returns:
    --------
    Optimal d value
    """
    from statsmodels.tsa.stattools import adfuller
    
    for d in np.arange(d_range[0], d_range[1], precision):
        diff_series = frac_diff_ffd(series, d)
        if len(diff_series) < 100:
            continue
        
        try:
            adf_result = adfuller(diff_series.dropna(), maxlag=1)
            pvalue = adf_result[1]
            
            if pvalue < target_adf_pvalue:
                return d
        except:
            continue
    
    return d_range[1]  # Return 1.0 if nothing works


# =============================================================================
# STATIONARITY TESTING
# =============================================================================

def test_stationarity(series: pd.Series, name: str = "Series") -> dict:
    """
    Comprehensive stationarity testing using ADF test.
    
    Parameters:
    -----------
    series : Series to test
    name : Name for reporting
    
    Returns:
    --------
    Dict with test results
    """
    from statsmodels.tsa.stattools import adfuller
    
    series_clean = series.dropna()
    
    if len(series_clean) < 100:
        return {'error': 'Insufficient data'}
    
    result = adfuller(series_clean, maxlag=1)
    
    return {
        'name': name,
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05,
        'n_obs': len(series_clean)
    }


# =============================================================================
# FEATURE ENGINEERING (Institutional Grade)
# =============================================================================

def create_institutional_features(df: pd.DataFrame, 
                                   frac_d: float = 0.4) -> pd.DataFrame:
    """
    Create features following institutional quant standards.
    
    - Uses FracDiff prices (not raw prices)
    - Normalized features
    - No lookahead bias
    
    Parameters:
    -----------
    df : DataFrame with OHLCV data
    frac_d : Fractional differentiation order
    
    Returns:
    --------
    DataFrame with features
    """
    df = df.copy()
    
    # 1. Fractional Differentiation of Close
    df['close_frac'] = frac_diff_ffd(df['c'], d=frac_d)
    
    # 2. Returns (various horizons)
    for h in [1, 5, 10, 20]:
        df[f'ret_{h}'] = df['c'].pct_change(h)
    
    # 3. Volatility (realized)
    df['volatility'] = df['ret_1'].rolling(20).std() * np.sqrt(252)
    
    # 4. Volume features
    df['vol_ma_ratio'] = df['v'] / df['v'].rolling(20).mean()
    df['vol_zscore'] = (df['v'] - df['v'].rolling(50).mean()) / df['v'].rolling(50).std()
    
    # 5. ATR normalized
    tr = pd.concat([
        df['h'] - df['l'],
        (df['h'] - df['c'].shift()).abs(),
        (df['l'] - df['c'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_norm'] = df['atr'] / df['c']
    
    # 6. RSI (normalized)
    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['rsi_norm'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]
    
    # 7. MACD normalized
    ema12 = df['c'].ewm(span=12).mean()
    ema26 = df['c'].ewm(span=26).mean()
    macd = ema12 - ema26
    df['macd_norm'] = macd / df['c']  # Price-normalized
    
    # 8. Bollinger Band position
    sma20 = df['c'].rolling(20).mean()
    std20 = df['c'].rolling(20).std()
    df['bb_position'] = (df['c'] - sma20) / (2 * std20 + 1e-10)
    
    # 9. Trend strength
    df['trend_20_50'] = (df['c'].rolling(20).mean() / df['c'].rolling(50).mean()) - 1
    
    # 10. Dollar volume (if available)
    if 'dollar_vol' in df.columns:
        df['dollar_vol_norm'] = df['dollar_vol'] / df['dollar_vol'].rolling(20).mean()
    
    return df


# =============================================================================
# DATA PIPELINE
# =============================================================================

class InstitutionalDataPipeline:
    """
    Complete data processing pipeline for institutional trading.
    """
    
    def __init__(self, dollar_threshold: float = 1_000_000, frac_d: float = 0.4):
        self.dollar_threshold = dollar_threshold
        self.frac_d = frac_d
        self.optimal_d = None
    
    def process_tick_data(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """Process tick data into dollar bars with features."""
        # Construct dollar bars
        bars = construct_dollar_bars(ticks_df, self.dollar_threshold)
        
        # Find optimal d for stationarity
        self.optimal_d = find_optimal_d(bars['close'], precision=0.05)
        print(f"   Optimal d for stationarity: {self.optimal_d:.2f}")
        
        # Create features
        featured = create_institutional_features(bars, self.optimal_d)
        
        return featured
    
    def process_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process OHLCV data (simulated dollar bars)."""
        # Simulate dollar bars
        bars = simulate_dollar_bars_from_ohlcv(df, self.dollar_threshold)
        
        if len(bars) < 200:
            print("   ⚠️ Not enough bars, using original data")
            bars = df.copy()
        
        # Find optimal d
        self.optimal_d = find_optimal_d(bars['c'], precision=0.05)
        print(f"   Optimal d for stationarity: {self.optimal_d:.2f}")
        
        # Create features
        featured = create_institutional_features(bars, self.optimal_d)
        
        return featured
    
    def get_feature_columns(self) -> list:
        """Return list of feature columns for ML."""
        return [
            'close_frac', 'ret_1', 'ret_5', 'ret_10', 'ret_20',
            'volatility', 'vol_ma_ratio', 'vol_zscore',
            'atr_norm', 'rsi_norm', 'macd_norm', 
            'bb_position', 'trend_20_50'
        ]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("📊 INSTITUTIONAL DATA PROCESSOR - Testing")
    print("="*70)
    
    # Load sample data
    try:
        df = pd.read_csv('/Users/anmol/Desktop/gold/market_data/BTCUSDT_1h.csv')
        df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
        print(f"\n✅ Loaded {len(df)} candles")
    except:
        print("❌ Could not load data")
        exit()
    
    # Test pipeline
    pipeline = InstitutionalDataPipeline(dollar_threshold=10_000_000, frac_d=0.4)
    
    print("\n🔧 Processing data...")
    processed = pipeline.process_ohlcv_data(df)
    print(f"   Processed: {len(processed)} bars")
    
    # Test stationarity
    print("\n📈 Stationarity Tests:")
    
    # Raw price
    raw_test = test_stationarity(df['c'], "Raw Close Price")
    print(f"   Raw Close: p={raw_test['p_value']:.4f} | Stationary: {raw_test['is_stationary']}")
    
    # FracDiff price
    if 'close_frac' in processed.columns:
        frac_test = test_stationarity(processed['close_frac'].dropna(), "FracDiff Close")
        print(f"   FracDiff:  p={frac_test['p_value']:.4f} | Stationary: {frac_test['is_stationary']}")
    
    print(f"\n✅ Feature columns: {len(pipeline.get_feature_columns())}")
    print(f"   {pipeline.get_feature_columns()}")
    
    print("\n" + "="*70)
