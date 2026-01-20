"""
MARKETFORGE: Live Engine
=========================
Real-time inference and signal generation.

Implements:
- On-candle-close inference
- Structured signal JSON output (Section 7 spec)
- Multi-TF consensus aggregation
- SHAP reason codes
- Telegram alerts
- Database persistence

Usage:
    python live_engine.py              # Run live
    python live_engine.py --paper     # Paper trading mode
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3
import hashlib
import argparse
import yaml
import joblib
from typing import Dict, List, Optional
import aiohttp
import ssl
import certifi

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "performance.db"

# Load config
with open(BASE_DIR / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class SignalGenerator:
    """
    Generates structured trading signals per Section 7 spec.
    
    Output format:
    {
        "asset": "BTC/USDT",
        "timeframe": "15m",
        "timestamp_utc": "2026-01-17T23:20:00Z",
        "signal": "BUY" | "SELL" | "NO-TRADE",
        "bias_strength": "STRONG" | "MODERATE" | "WEAK",
        "entry_zone": [low, high],
        "entry_price": mid,
        "stop_loss": sl,
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "confidence_pct": 72.4,
        "regime": "TREND" | "CHOP" | "MEANREV" | "VOLATILE",
        "model_id": "sha256:...",
        "reason_codes": ["..."],
    }
    """
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Load model
        self.model = None
        self.scaler = None
        self.meta_model = None
        self.model_id = None
        self.feature_names = []
        
        self._load_model()
        
        # Config
        tf_config = CONFIG['labeling'].get(timeframe, CONFIG['labeling']['1h'])
        self.tp_mult = tf_config['tp_mult']
        self.sl_mult = tf_config['sl_mult']
        self.confidence_threshold = CONFIG['signals']['confidence_threshold']
    
    def _load_model(self):
        """Load trained model and metadata"""
        primary_path = MODEL_DIR / f"primary_{self.symbol}_{self.timeframe}.pkl"
        scaler_path = MODEL_DIR / f"scaler_{self.symbol}_{self.timeframe}.pkl"
        meta_path = MODEL_DIR / f"meta_{self.symbol}_{self.timeframe}.pkl"
        manifest_path = MODEL_DIR / f"manifest_{self.symbol}_{self.timeframe}.json"
        schema_path = MODEL_DIR / f"feature_schema_{self.symbol}_{self.timeframe}.json"
        
        if primary_path.exists():
            self.model = joblib.load(primary_path)
            self.scaler = joblib.load(scaler_path)
            
            if meta_path.exists():
                self.meta_model = joblib.load(meta_path)
            
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    self.model_id = manifest.get('model_id', 'unknown')
            
            if schema_path.exists():
                with open(schema_path) as f:
                    schema = json.load(f)
                    self.feature_names = schema.get('features', [])
            
            print(f"   ✅ Model loaded: {self.model_id}")
        else:
            print(f"   ⚠️ No model found for {self.symbol} {self.timeframe}")
    
    def detect_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime"""
        if len(df) < 50:
            return 'CHOP'
        
        # ADX for trend detection
        if 'adx' in df.columns:
            adx = df['adx'].iloc[-1]
        else:
            adx = 20  # Default
        
        # Volatility for vol regime
        if 'vol_10' in df.columns:
            vol_now = df['vol_10'].iloc[-1]
            vol_avg = df['vol_10'].rolling(50).mean().iloc[-1]
            vol_ratio = vol_now / (vol_avg + 1e-10)
        else:
            vol_ratio = 1.0
        
        if vol_ratio > 2.0:
            return 'VOLATILE'
        elif adx > 25:
            return 'TREND'
        elif adx < 15:
            return 'CHOP'
        else:
            return 'MEANREV'
    
    def get_reason_codes(self, df: pd.DataFrame, signal: str) -> List[str]:
        """Generate human-readable reason codes"""
        codes = []
        row = df.iloc[-1]
        
        # RSI
        if 'rsi' in row:
            rsi = row['rsi']
            if rsi < 30:
                codes.append('RSI_oversold')
            elif rsi > 70:
                codes.append('RSI_overbought')
        
        # MACD
        if 'macd_hist' in row:
            if row['macd_hist'] > 0:
                codes.append('MACD_bullish')
            else:
                codes.append('MACD_bearish')
        
        # ADX trend
        if 'adx' in row and row['adx'] > 25:
            codes.append('ADX_trending')
        
        # BB position
        if 'bb_pct' in row:
            if row['bb_pct'] < 0.1:
                codes.append('BB_lower_touch')
            elif row['bb_pct'] > 0.9:
                codes.append('BB_upper_touch')
        
        # Volume spike
        if 'vol_spike' in row and row['vol_spike']:
            codes.append('volume_spike')
        
        return codes[:5]  # Max 5 codes
    
    def generate(self, df: pd.DataFrame) -> dict:
        """
        Generate structured signal from dataframe.
        
        Returns complete signal JSON per Section 7 spec.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Base signal structure
        signal = {
            'asset': f"{self.symbol[:3]}/{self.symbol[3:]}",
            'timeframe': self.timeframe,
            'timestamp_utc': timestamp,
            'signal': 'NO-TRADE',
            'bias_strength': 'WEAK',
            'entry_zone': [0, 0],
            'entry_price': 0,
            'stop_loss': 0,
            'tp1': 0,
            'tp2': 0,
            'tp3': 0,
            'pips_tp1': 0,
            'confidence_pct': 0,
            'regime': 'CHOP',
            'model_id': self.model_id or 'none',
            'feature_schema_hash': '',
            'reason_codes': [],
            'notes': ''
        }
        
        if len(df) < 100:
            signal['notes'] = 'Insufficient data'
            return signal
        
        # Detect regime
        regime = self.detect_regime(df)
        signal['regime'] = regime
        
        # Check regime gates
        regime_gates = CONFIG['signals']['regime_gates']
        if regime == 'CHOP' and not regime_gates['allow_chop']:
            signal['notes'] = 'CHOP regime blocked'
            return signal
        
        # Get current price and ATR
        row = df.iloc[-1]
        price = row['c']
        atr = row.get('atr', price * 0.015)
        
        # Prepare features
        if self.model and self.scaler and self.feature_names:
            available_features = [f for f in self.feature_names if f in df.columns]
            
            if len(available_features) >= 10:
                X = df[available_features].iloc[-1:].fillna(0)
                X_scaled = self.scaler.transform(X)
                
                # Primary prediction
                proba = self.model.predict_proba(X_scaled)[0]
                pred_class = self.model.predict(X_scaled)[0]
                confidence = max(proba) * 100
                
                # Meta-model filter
                if self.meta_model:
                    meta_proba = self.meta_model.predict_proba(X_scaled)[0][1]
                    confidence *= meta_proba
                
                signal['confidence_pct'] = round(confidence, 1)
                
                # Apply threshold
                if confidence >= self.confidence_threshold:
                    if pred_class == 1:
                        signal['signal'] = 'BUY'
                    else:
                        signal['signal'] = 'SELL'
        else:
            # Fallback: simple rule-based
            rsi = row.get('rsi', 50)
            macd_hist = row.get('macd_hist', 0)
            
            score = 0
            if rsi < 30: score += 2
            elif rsi < 40: score += 1
            elif rsi > 70: score -= 2
            elif rsi > 60: score -= 1
            
            if macd_hist > 0: score += 1
            else: score -= 1
            
            confidence = 50 + abs(score) * 5
            signal['confidence_pct'] = round(confidence, 1)
            
            if score >= 2 and confidence >= self.confidence_threshold:
                signal['signal'] = 'BUY'
            elif score <= -2 and confidence >= self.confidence_threshold:
                signal['signal'] = 'SELL'
        
        # Calculate trade levels if we have a signal
        if signal['signal'] != 'NO-TRADE':
            # Entry zone
            spread = atr * 0.1
            signal['entry_zone'] = [round(price - spread, 2), round(price + spread, 2)]
            signal['entry_price'] = round(price, 2)
            
            # Stops and targets
            if signal['signal'] == 'BUY':
                signal['stop_loss'] = round(price - atr * self.sl_mult, 2)
                signal['tp1'] = round(price + atr * 1.0, 2)
                signal['tp2'] = round(price + atr * self.tp_mult, 2)
                signal['tp3'] = round(price + atr * self.tp_mult * 1.5, 2)
            else:
                signal['stop_loss'] = round(price + atr * self.sl_mult, 2)
                signal['tp1'] = round(price - atr * 1.0, 2)
                signal['tp2'] = round(price - atr * self.tp_mult, 2)
                signal['tp3'] = round(price - atr * self.tp_mult * 1.5, 2)
            
            signal['pips_tp1'] = abs(signal['tp1'] - price)
            
            # Bias strength
            conf = signal['confidence_pct']
            if conf >= 80:
                signal['bias_strength'] = 'STRONG'
            elif conf >= 70:
                signal['bias_strength'] = 'MODERATE'
            else:
                signal['bias_strength'] = 'WEAK'
            
            # Reason codes
            signal['reason_codes'] = self.get_reason_codes(df, signal['signal'])
        
        return signal


# =============================================================================
# MULTI-TIMEFRAME CONSENSUS
# =============================================================================

class ConsensusAggregator:
    """
    Aggregates signals across timeframes for final decision.
    
    Rules:
    - STRONG: ≥3 TFs align
    - MODERATE: 2 TFs align
    - Otherwise: NO-TRADE
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.signals: Dict[str, dict] = {}
    
    def add_signal(self, timeframe: str, signal: dict):
        """Add signal from one timeframe"""
        self.signals[timeframe] = signal
    
    def get_consensus(self) -> dict:
        """Calculate consensus across timeframes"""
        consensus = {
            'asset': f"{self.symbol[:3]}/{self.symbol[3:]}",
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'final_signal': 'NO-TRADE',
            'bias_strength': 'WEAK',
            'aligned_tfs': [],
            'buy_count': 0,
            'sell_count': 0,
            'confidence_avg': 0,
            'signals': self.signals
        }
        
        # Count aligned signals
        buy_tfs = []
        sell_tfs = []
        confidences = []
        
        for tf, sig in self.signals.items():
            if sig['signal'] == 'BUY':
                buy_tfs.append(tf)
            elif sig['signal'] == 'SELL':
                sell_tfs.append(tf)
            
            if sig['signal'] != 'NO-TRADE':
                confidences.append(sig['confidence_pct'])
        
        consensus['buy_count'] = len(buy_tfs)
        consensus['sell_count'] = len(sell_tfs)
        consensus['confidence_avg'] = np.mean(confidences) if confidences else 0
        
        # Determine final signal
        min_strong = CONFIG['signals']['consensus']['strong_min_tfs']
        min_moderate = CONFIG['signals']['consensus']['moderate_min_tfs']
        
        if len(buy_tfs) >= min_strong:
            consensus['final_signal'] = 'BUY'
            consensus['bias_strength'] = 'STRONG'
            consensus['aligned_tfs'] = buy_tfs
        elif len(sell_tfs) >= min_strong:
            consensus['final_signal'] = 'SELL'
            consensus['bias_strength'] = 'STRONG'
            consensus['aligned_tfs'] = sell_tfs
        elif len(buy_tfs) >= min_moderate:
            consensus['final_signal'] = 'BUY'
            consensus['bias_strength'] = 'MODERATE'
            consensus['aligned_tfs'] = buy_tfs
        elif len(sell_tfs) >= min_moderate:
            consensus['final_signal'] = 'SELL'
            consensus['bias_strength'] = 'MODERATE'
            consensus['aligned_tfs'] = sell_tfs
        
        return consensus


# =============================================================================
# DATABASE PERSISTENCE
# =============================================================================

class SignalPersistence:
    """Persist signals to SQLite"""
    
    def __init__(self):
        self.db_path = DB_PATH
    
    def save_signal(self, signal: dict):
        """Save signal to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO signals 
        (timestamp_utc, asset, timeframe, signal, bias_strength,
         entry_zone_low, entry_zone_high, entry_price,
         stop_loss, tp1, tp2, tp3, pips_tp1,
         confidence_pct, regime, model_id, reason_codes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['timestamp_utc'],
            signal['asset'],
            signal['timeframe'],
            signal['signal'],
            signal['bias_strength'],
            signal['entry_zone'][0],
            signal['entry_zone'][1],
            signal['entry_price'],
            signal['stop_loss'],
            signal['tp1'],
            signal['tp2'],
            signal['tp3'],
            signal['pips_tp1'],
            signal['confidence_pct'],
            signal['regime'],
            signal['model_id'],
            json.dumps(signal['reason_codes'])
        ))
        
        conn.commit()
        conn.close()


# =============================================================================
# TELEGRAM ALERTS
# =============================================================================

class TelegramAlert:
    """Send alerts to Telegram"""
    
    def __init__(self):
        self.enabled = CONFIG['alerts']['telegram']['enabled']
        self.bot_token = CONFIG['alerts']['telegram']['bot_token']
        self.chat_id = CONFIG['alerts']['telegram']['chat_id']
        
        # Rate limiting
        self.last_alerts: Dict[str, float] = {}
        self.rate_limit = CONFIG['alerts']['rate_limiting']
    
    async def send(self, signal: dict):
        """Send signal alert"""
        if not self.enabled:
            return
        
        # Rate limit check
        key = f"{signal['asset']}_{signal['signal']}"
        now = datetime.now().timestamp()
        
        if key in self.last_alerts:
            elapsed = now - self.last_alerts[key]
            if elapsed < self.rate_limit['window_minutes'] * 60:
                return  # Rate limited
        
        self.last_alerts[key] = now
        
        # Format message
        emoji = '🟢' if signal['signal'] == 'BUY' else '🔴'
        message = f"""
{emoji} **{signal['signal']}** {signal['asset']} ({signal['timeframe']})

📊 **Confidence:** {signal['confidence_pct']}%
🎯 **Bias:** {signal['bias_strength']}

📍 Entry: ${signal['entry_price']:,.2f}
🛑 Stop: ${signal['stop_loss']:,.2f}
✅ TP1: ${signal['tp1']:,.2f}
✅ TP2: ${signal['tp2']:,.2f}
✅ TP3: ${signal['tp3']:,.2f}

📈 Regime: {signal['regime']}
🔖 {', '.join(signal['reason_codes'][:3])}
        """
        
        # Send via Telegram API
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            async with aiohttp.ClientSession() as session:
                await session.post(url, json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                })
        except Exception as e:
            print(f"⚠️ Telegram send failed: {e}")


# =============================================================================
# LIVE ENGINE
# =============================================================================

class LiveEngine:
    """
    Main live trading engine.
    
    Runs inference on candle close for all assets/timeframes.
    """
    
    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.symbols = CONFIG['assets']['primary']
        self.timeframes = CONFIG['timeframes']
        
        # Components
        self.generators: Dict[str, Dict[str, SignalGenerator]] = {}
        self.persistence = SignalPersistence()
        self.alerts = TelegramAlert()
        
        # Initialize generators
        for asset in self.symbols:
            symbol = asset['symbol']
            self.generators[symbol] = {}
            for tf in self.timeframes:
                self.generators[symbol][tf] = SignalGenerator(symbol, tf)
    
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load latest data for inference"""
        # Try labeled data first
        path = DATA_DIR / f"{symbol}_{timeframe}_labeled.parquet"
        if path.exists():
            return pd.read_parquet(path)
        
        # Fallback to CSV
        csv_path = DATA_DIR / f"{symbol}_{timeframe}.csv"
        if not csv_path.exists():
            csv_path = BASE_DIR.parent / "market_data" / f"{symbol}_{timeframe}.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if len(df.columns) == 6:
                df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
            return df
        
        return pd.DataFrame()
    
    async def run_inference(self):
        """Run inference for all assets/timeframes"""
        print(f"\n{'='*60}")
        print(f"📊 MARKETFORGE - Live Inference")
        print(f"{'='*60}")
        print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print(f"Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        
        for asset in self.symbols:
            symbol = asset['symbol']
            print(f"\n{'='*60}")
            print(f"📈 {symbol}")
            print(f"{'='*60}")
            
            aggregator = ConsensusAggregator(symbol)
            
            for tf in self.timeframes:
                # Load data
                df = self.load_data(symbol, tf)
                
                if df.empty:
                    print(f"   {tf}: No data")
                    continue
                
                # Generate signal
                generator = self.generators[symbol][tf]
                signal = generator.generate(df)
                
                # Add to aggregator
                aggregator.add_signal(tf, signal)
                
                # Print signal
                if signal['signal'] != 'NO-TRADE':
                    print(f"   {tf}: {signal['signal']} @ {signal['confidence_pct']}% "
                          f"[{signal['bias_strength']}] - {signal['regime']}")
                    
                    # Persist
                    if CONFIG['signals']['persist_to_db']:
                        self.persistence.save_signal(signal)
                    
                    # Alert if STRONG
                    if signal['bias_strength'] == 'STRONG':
                        await self.alerts.send(signal)
                else:
                    print(f"   {tf}: NO-TRADE ({signal.get('notes', 'below threshold')})")
            
            # Consensus
            consensus = aggregator.get_consensus()
            print(f"\n   🎯 CONSENSUS: {consensus['final_signal']} "
                  f"({consensus['bias_strength']}) "
                  f"- Aligned: {consensus['aligned_tfs']}")
    
    async def run_loop(self, interval_seconds: int = 60):
        """Run continuous inference loop"""
        print(f"\n🚀 Starting continuous inference (interval: {interval_seconds}s)")
        
        while True:
            try:
                await self.run_inference()
            except Exception as e:
                print(f"❌ Inference error: {e}")
            
            await asyncio.sleep(interval_seconds)


def print_signal_json(signal: dict):
    """Pretty print signal JSON"""
    print(json.dumps(signal, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live trading engine')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, default=60, help='Inference interval (seconds)')
    
    args = parser.parse_args()
    
    engine = LiveEngine(paper_mode=args.paper or True)
    
    if args.once:
        asyncio.run(engine.run_inference())
    else:
        asyncio.run(engine.run_loop(args.interval))
