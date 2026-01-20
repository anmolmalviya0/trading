"""
FORGE TRADING SYSTEM - FEATURE ENGINEERING
===========================================
All technical indicators:
- EMA (50, 200)
- RSI (Wilder method)
- ATR
- Bollinger Band Width
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (used for RSI, ATR)"""
    return series.ewm(alpha=1/period, adjust=False).mean()


def add_features(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.
    DataFrame must have: timestamp, open, high, low, close, volume
    """
    if config is None:
        config = load_config()
    
    df = df.copy()
    
    if len(df) < 50:
        return df
    
    feat = config.get('features', {})
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']
    
    # EMAs
    ema_fast = feat.get('ema_fast', 50)
    ema_slow = feat.get('ema_slow', 200)
    trend_period = feat.get('trend_ema', 4800)
    
    df['ema_fast'] = c.ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = c.ewm(span=ema_slow, adjust=False).mean()
    df['trend_ema'] = c.ewm(span=trend_period, adjust=False).mean()
    
    # RSI (Wilder method)
    rsi_period = feat.get('rsi_period', 14)
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = wilder_rma(gain, rsi_period)
    avg_loss = wilder_rma(loss, rsi_period)
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    atr_period = feat.get('atr_period', 14)
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = wilder_rma(tr, atr_period)
    
    # ADX (for regime detection)
    up = h - h.shift(1)
    down = l.shift(1) - l
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    atr_smooth = df['atr']
    plus_di = 100 * wilder_rma(pd.Series(plus_dm, index=df.index), 14) / (atr_smooth + 1e-10)
    minus_di = 100 * wilder_rma(pd.Series(minus_dm, index=df.index), 14) / (atr_smooth + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = wilder_rma(dx, 14)
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    # Bollinger Bands
    bb_period = feat.get('bb_period', 20)
    bb_std = feat.get('bb_std', 2)
    bb_mid = c.rolling(bb_period).mean()
    bb_std_val = c.rolling(bb_period).std()
    df['bb_upper'] = bb_mid + (bb_std_val * bb_std)
    df['bb_lower'] = bb_mid - (bb_std_val * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (bb_mid + 1e-10)
    df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(50).mean() * 0.5
    
    # Volume analysis
    df['volume_sma'] = v.rolling(20).mean()
    df['volume_ratio'] = v / (df['volume_sma'] + 1e-10)
    
    # VWAP (Volume Weighted Average Price)
    # Reset VWAP daily or rolling? Rolling 24h (24h * 60m / timeframe) is better for crypto
    # Approximating rolling VWAP for last 24 periods (e.g. 24h if 1h candles)
    vwap_period = feat.get('vwap_period', 24)
    tp = (h + l + c) / 3
    df['vwap'] = (tp * v).rolling(vwap_period).sum() / (v.rolling(vwap_period).sum() + 1e-10)
    df['vwap_dev'] = (c - df['vwap']) / (df['atr'] + 1e-10)
    
    # Order Flow Imbalance (OFI) Proxy
    # Buying pressure: volume * (close - low) / (high - low)
    # Selling pressure: volume * (high - close) / (high - low)
    range_len = h - l
    buy_press = v * ((c - l) / (range_len + 1e-10))
    sell_press = v * ((h - c) / (range_len + 1e-10))
    df['ofi'] = (buy_press - sell_press).rolling(3).mean() # Smoothed OFI
    df['ofi_signal'] = np.where(df['ofi'] > 0, 1, -1)
    
    # Volatility Ratio (Institutional Vol Filter)
    # Short-term ATR vs Long-term ATR
    df['vol_ratio'] = df['atr'] / (df['atr'].rolling(50).mean() + 1e-10)

    # ==========================================
    # HEIKIN ASHI CALCULATION
    # ==========================================
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = pd.Series(index=df.index, dtype='float64')
    ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    
    # Iterate to calculate HA Open (dependent on previous HA Open)
    # Using numpy for speed if possible, but loop is safer for correctness here
    # Since we have full history, we can use a loop or Numba if available.
    # For simplicity and robustness without extra deps, we use a loop but optimize.
    # Actually, we can use a shift trick if we accept slight inaccuracy at start, 
    # but for trading we need precision.
    
    # Vectorized approximation for HA Open is hard, standard loop is best for 1H data
    # Pre-calculate to avoid .iloc overhead
    opens = df['open'].values
    closes = df['close'].values
    ha_open_vals = np.zeros(len(df))
    ha_close_vals = ha_close.values
    ha_open_vals[0] = (opens[0] + closes[0]) / 2
    
    for i in range(1, len(df)):
        ha_open_vals[i] = (ha_open_vals[i-1] + ha_close_vals[i-1]) / 2
        
    df['ha_open'] = ha_open_vals
    df['ha_close'] = ha_close_vals
    df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
    df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
    
    # ==========================================
    # SUPERTREND CALCULATION
    # ==========================================
    st_period = 10
    st_multiplier = 3.0
    
    hl2 = (df['high'] + df['low']) / 2
    df['st_basic_upper'] = hl2 + (st_multiplier * df['atr'])
    df['st_basic_lower'] = hl2 - (st_multiplier * df['atr'])
    
    # SuperTrend Logic
    # We need to iterate to handle the 'stickiness' of the bands
    st_upper = np.zeros(len(df))
    st_lower = np.zeros(len(df))
    st_trend = np.zeros(len(df)) # 1 = Bull, -1 = Bear
    
    close_vals = df['close'].values
    basic_upper = df['st_basic_upper'].values
    basic_lower = df['st_basic_lower'].values
    
    for i in range(1, len(df)):
        # Upper Band
        if basic_upper[i] < st_upper[i-1] or close_vals[i-1] > st_upper[i-1]:
            st_upper[i] = basic_upper[i]
        else:
            st_upper[i] = st_upper[i-1]
            
        # Lower Band
        if basic_lower[i] > st_lower[i-1] or close_vals[i-1] < st_lower[i-1]:
            st_lower[i] = basic_lower[i]
        else:
            st_lower[i] = st_lower[i-1]
            
        # Trend
        if st_trend[i-1] == 1: # Was Bullish
            if close_vals[i] < st_lower[i]:
                st_trend[i] = -1 # Flip to Bearish
            else:
                st_trend[i] = 1 # Stay Bullish
        else: # Was Bearish
            if close_vals[i] > st_upper[i]:
                st_trend[i] = 1 # Flip to Bullish
            else:
                st_trend[i] = -1 # Stay Bearish
                
    df['supertrend'] = np.where(st_trend == 1, st_lower, st_upper)
    df['st_trend'] = st_trend
    
    # Price position
    df['price_vs_ema_fast'] = (c - df['ema_fast']) / (df['atr'] + 1e-10)
    df['price_vs_ema_slow'] = (c - df['ema_slow']) / (df['atr'] + 1e-10)
    
    # EMA cross state
    df['ema_bullish'] = df['ema_fast'] > df['ema_slow']
    
    return df


def get_market_regime(df: pd.DataFrame, config: dict = None) -> str:
    """
    Determine current market regime.
    Returns: 'trend_up', 'trend_down', 'range', 'volatile'
    """
    if config is None:
        config = load_config()
    
    if len(df) < 50 or 'adx' not in df.columns:
        return 'unknown'
    
    last = df.iloc[-1]
    conv = config.get('conviction', {})
    adx_min = conv.get('regime_adx_min', 20)
    adx_max = conv.get('regime_adx_max', 50)
    
    adx = last.get('adx', 0)
    plus_di = last.get('plus_di', 0)
    minus_di = last.get('minus_di', 0)
    
    if pd.isna(adx):
        return 'unknown'
    
    if adx > adx_max:
        return 'volatile'
    elif adx < adx_min:
        return 'range'
    elif plus_di > minus_di:
        return 'trend_up'
    else:
        return 'trend_down'
