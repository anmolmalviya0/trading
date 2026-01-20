"""
V8 FINAL - MULTI-SIGNAL ENGINE
================================
6 signal types for higher trade frequency:
1. Pivot Divergence (high quality)
2. RSI Reversal (medium frequency)
3. EMA Cross (momentum)
4. Volume Spike (accumulation/distribution)
5. Sweep Rejection (smart money)
6. S/R Break (structure)

Target: 5-10 trades/day with 55-65% WR
"""
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yaml


def load_config():
    try:
        with open('config.yaml') as f:
            return yaml.safe_load(f)
    except:
        return {}


def wilder_rma(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1/n, adjust=False).mean()


class SignalEngine:
    """
    Multi-signal engine with 6 signal types.
    Combines signals for confirmation (signal stacking).
    """
    
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.signals_cfg = self.config.get('signals', {})
        self.threshold = self.config.get('scoring', {}).get('threshold', 40)
        self.min_signals = self.config.get('scoring', {}).get('min_signals', 2)
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators"""
        df = df.copy()
        if len(df) < 50:
            return df
        
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']
        
        # RSI
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        rs = wilder_rma(gain, 14) / (wilder_rma(loss, 14) + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        df['atr'] = wilder_rma(tr, 14)
        
        # ADX
        up = h - h.shift(1)
        down = l.shift(1) - l
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        atr_s = df['atr']
        plus_di = 100 * wilder_rma(pd.Series(plus_dm, index=df.index), 14) / (atr_s + 1e-10)
        minus_di = 100 * wilder_rma(pd.Series(minus_dm, index=df.index), 14) / (atr_s + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = wilder_rma(dx, 14)
        
        # EMAs
        ema_cfg = self.signals_cfg.get('ema_cross', {})
        df['ema9'] = c.ewm(span=ema_cfg.get('fast', 9)).mean()
        df['ema21'] = c.ewm(span=ema_cfg.get('slow', 21)).mean()
        df['ema50'] = c.ewm(span=50).mean()
        df['ema200'] = c.ewm(span=200).mean()
        
        # BB Width
        mid = c.rolling(20).mean()
        std = c.rolling(20).std()
        df['bb_width'] = (std * 2) / (mid + 1e-10)
        
        # Volume
        df['vol_sma'] = v.rolling(20).mean()
        df['vol_spike'] = v / (df['vol_sma'] + 1e-10)
        
        # Pivots
        order = self.signals_cfg.get('divergence', {}).get('pivot_order', 5)
        low_idx = argrelextrema(l.values, np.less, order=order)[0]
        high_idx = argrelextrema(h.values, np.greater, order=order)[0]
        
        df['pivot_low'] = np.nan
        df['pivot_high'] = np.nan
        for i in low_idx:
            if i < len(df):
                df.iloc[i, df.columns.get_loc('pivot_low')] = l.iloc[i]
        for i in high_idx:
            if i < len(df):
                df.iloc[i, df.columns.get_loc('pivot_high')] = h.iloc[i]
        
        return df
    
    def check_divergence(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Check for pivot divergence"""
        if not self.signals_cfg.get('divergence', {}).get('enabled', True):
            return 0, ""
        
        piv_lows = df.dropna(subset=['pivot_low']).index.tolist()
        piv_highs = df.dropna(subset=['pivot_high']).index.tolist()
        
        weight = self.signals_cfg.get('divergence', {}).get('weight', 35)
        
        # Bullish divergence
        if len(piv_lows) >= 2:
            curr_idx = piv_lows[-1]
            prev_idx = piv_lows[-2]
            
            loc_curr = df.index.get_loc(curr_idx)
            loc_prev = df.index.get_loc(prev_idx)
            
            if loc_curr - loc_prev >= 5:
                curr_low = df.loc[curr_idx, 'low']
                prev_low = df.loc[prev_idx, 'low']
                curr_rsi = df.loc[curr_idx, 'rsi']
                prev_rsi = df.loc[prev_idx, 'rsi']
                
                if curr_low < prev_low and curr_rsi > prev_rsi and curr_rsi < 45:
                    return weight, "Bull Div"
        
        # Bearish divergence
        if len(piv_highs) >= 2:
            curr_idx = piv_highs[-1]
            prev_idx = piv_highs[-2]
            
            loc_curr = df.index.get_loc(curr_idx)
            loc_prev = df.index.get_loc(prev_idx)
            
            if loc_curr - loc_prev >= 5:
                curr_high = df.loc[curr_idx, 'high']
                prev_high = df.loc[prev_idx, 'high']
                curr_rsi = df.loc[curr_idx, 'rsi']
                prev_rsi = df.loc[prev_idx, 'rsi']
                
                if curr_high > prev_high and curr_rsi < prev_rsi and curr_rsi > 55:
                    return -weight, "Bear Div"
        
        return 0, ""
    
    def check_rsi(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Check RSI extremes"""
        if not self.signals_cfg.get('rsi', {}).get('enabled', True):
            return 0, ""
        
        last = df.iloc[-1]
        rsi_cfg = self.signals_cfg.get('rsi', {})
        weight = rsi_cfg.get('weight', 20)
        oversold = rsi_cfg.get('oversold', 35)
        overbought = rsi_cfg.get('overbought', 65)
        
        if last['rsi'] < oversold:
            return weight, f"RSI {last['rsi']:.0f}"
        elif last['rsi'] > overbought:
            return -weight, f"RSI {last['rsi']:.0f}"
        
        return 0, ""
    
    def check_ema_cross(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Check EMA crossover"""
        if not self.signals_cfg.get('ema_cross', {}).get('enabled', True):
            return 0, ""
        
        if len(df) < 3:
            return 0, ""
        
        weight = self.signals_cfg.get('ema_cross', {}).get('weight', 15)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish cross
        if prev['ema9'] < prev['ema21'] and curr['ema9'] > curr['ema21']:
            return weight, "EMA9x21 Bull"
        
        # Bearish cross
        if prev['ema9'] > prev['ema21'] and curr['ema9'] < curr['ema21']:
            return -weight, "EMA9x21 Bear"
        
        return 0, ""
    
    def check_volume_spike(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Check volume spike"""
        if not self.signals_cfg.get('volume_spike', {}).get('enabled', True):
            return 0, ""
        
        last = df.iloc[-1]
        mult = self.signals_cfg.get('volume_spike', {}).get('multiplier', 2.0)
        weight = self.signals_cfg.get('volume_spike', {}).get('weight', 10)
        
        if last['vol_spike'] > mult:
            # Direction based on candle
            if last['close'] > last['open']:
                return weight, f"Vol {last['vol_spike']:.1f}x"
            else:
                return -weight, f"Vol {last['vol_spike']:.1f}x"
        
        return 0, ""
    
    def check_sweep(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Check liquidity sweep"""
        if not self.signals_cfg.get('sweep', {}).get('enabled', True):
            return 0, ""
        
        if len(df) < 3:
            return 0, ""
        
        weight = self.signals_cfg.get('sweep', {}).get('weight', 15)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish sweep (wick below prev low, close above)
        if last['low'] < prev['low'] and last['close'] > prev['low']:
            return weight, "Bull Sweep"
        
        # Bearish sweep
        if last['high'] > prev['high'] and last['close'] < prev['high']:
            return -weight, "Bear Sweep"
        
        return 0, ""
    
    def is_session_active(self) -> bool:
        """Check if current time is in active session"""
        sessions = self.config.get('sessions', {})
        if not sessions.get('enabled', True):
            return True
        
        now = datetime.utcnow()
        hour = now.hour
        
        # London or NY session
        london_active = sessions.get('london_start', 8) <= hour < sessions.get('london_end', 17)
        ny_active = sessions.get('newyork_start', 13) <= hour < sessions.get('newyork_end', 22)
        
        return london_active or ny_active
    
    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[Dict]:
        """Analyze and generate signal"""
        # Add indicators
        df = self.add_indicators(df)
        
        if len(df) < 50:
            return None
        
        last = df.iloc[-1]
        
        # Safety checks
        if pd.isna(last.get('rsi')) or pd.isna(last.get('atr')):
            return None
        
        # Check session
        if not self.is_session_active():
            return None
        
        # Regime filter
        adx_thresh = self.config.get('regime', {}).get('adx_threshold', 15)
        if last['adx'] < adx_thresh:
            return None
        
        # Collect all signals
        signals = []
        total_score = 0
        reasons = []
        
        # Check each signal type
        for check_fn in [
            self.check_divergence,
            self.check_rsi,
            self.check_ema_cross,
            self.check_volume_spike,
            self.check_sweep
        ]:
            score, reason = check_fn(df)
            if score != 0:
                signals.append(score)
                total_score += score
                reasons.append(reason)
        
        # Need minimum signals
        if len(signals) < self.min_signals:
            return None
        
        # Determine direction from score
        if total_score > 0:
            direction = 'BUY'
        elif total_score < 0:
            direction = 'SELL'
            total_score = abs(total_score)
        else:
            return None
        
        # Check threshold
        if total_score < self.threshold:
            return None
        
        # Calculate SL/TP
        atr = last['atr']
        entry = last['close']
        sl_mult = self.config.get('risk', {}).get('sl_atr_mult', 1.2)
        tp_mult = self.config.get('risk', {}).get('tp_atr_mult', 2.0)
        
        if direction == 'BUY':
            sl = entry - (atr * sl_mult)
            tp = entry + (atr * tp_mult)
        else:
            sl = entry + (atr * sl_mult)
            tp = entry - (atr * tp_mult)
        
        return {
            'timestamp': str(last.name) if hasattr(last, 'name') else datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'score': total_score,
            'num_signals': len(signals),
            'entry': float(entry),
            'sl': float(sl),
            'tp': float(tp),
            'atr': float(atr),
            'rsi': float(last['rsi']),
            'adx': float(last['adx']),
            'reasons': reasons
        }
