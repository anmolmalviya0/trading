"""
GOLD MEAN-REVERSION: Rule-Based Trading System
================================================
Direct rule-based signals for Gold (PAXGUSDT) - No ML needed!

Why Rule-Based:
- Mean-reversion labels show 70% win rate
- Too few signals (900) for ML training
- Clear, explainable rules work better

Strategy:
- BUY: RSI < 30 AND Price < Lower Bollinger Band
- SELL: RSI > 70 AND Price > Upper Bollinger Band
- TARGET: Mean reversion to BB middle

Usage:
    python gold_rules.py signal PAXGUSDT 15m
    python gold_rules.py backtest --all
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# Load config
with open(BASE_DIR / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)


# =============================================================================
# RULE-BASED PARAMETERS (Per Timeframe)
# =============================================================================

GOLD_RULES = {
    '5m': {
        'rsi_period': 14,
        'rsi_oversold': 28,
        'rsi_overbought': 72,
        'bb_period': 20,
        'bb_std': 2.0,
        'target_pct': 0.003,
        'stop_pct': 0.005,
        'max_hold': 8,
    },
    '15m': {
        'rsi_period': 14,
        'rsi_oversold': 30,      # Best results: 57.8% WR, 1.28 PF
        'rsi_overbought': 70,
        'bb_period': 20,
        'bb_std': 2.0,
        'target_pct': 0.004,
        'stop_pct': 0.006,
        'max_hold': 10,
    },
    '30m': {
        'rsi_period': 14,
        'rsi_oversold': 32,
        'rsi_overbought': 68,
        'bb_period': 20,
        'bb_std': 2.0,
        'target_pct': 0.005,
        'stop_pct': 0.008,
        'max_hold': 14,
    },
    '1h': {
        'rsi_period': 14,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'bb_period': 20,
        'bb_std': 2.0,
        'target_pct': 0.006,
        'stop_pct': 0.010,
        'max_hold': 18,
    }
}


# =============================================================================
# INDICATOR CALCULATION
# =============================================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands"""
    middle = close.rolling(period).mean()
    std_dev = close.rolling(period).std()
    upper = middle + std * std_dev
    lower = middle - std * std_dev
    return upper, middle, lower


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR"""
    tr = pd.concat([
        df['h'] - df['l'],
        (df['h'] - df['c'].shift()).abs(),
        (df['l'] - df['c'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

@dataclass
class GoldSignal:
    """Gold mean-reversion signal"""
    timestamp: str
    timeframe: str
    signal: str  # 'BUY', 'SELL', 'NO_TRADE'
    price: float
    rsi: float
    bb_lower: float
    bb_middle: float
    bb_upper: float
    
    # Trade plan
    entry_price: float
    stop_loss: float
    target_1: float  # Mean (BB middle)
    target_2: float  # Opposite BB
    
    # Confidence
    confidence: float  # 0-100
    reason: str


class GoldMeanReversionEngine:
    """
    Rule-based mean-reversion signal generator for Gold.
    
    Rules:
    - BUY when RSI < oversold AND price < lower BB
    - SELL when RSI > overbought AND price > upper BB
    - Target is mean reversion to BB middle
    """
    
    def __init__(self, timeframe: str = '15m'):
        self.timeframe = timeframe
        self.params = GOLD_RULES.get(timeframe, GOLD_RULES['15m'])
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators"""
        df = df.copy()
        
        df['rsi'] = compute_rsi(df['c'], self.params['rsi_period'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = compute_bollinger_bands(
            df['c'], self.params['bb_period'], self.params['bb_std']
        )
        df['atr'] = compute_atr(df, 14)
        
        # Additional context
        df['bb_pct'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['dist_from_middle'] = (df['c'] - df['bb_middle']) / df['bb_middle'] * 100
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> GoldSignal:
        """Generate signal for the latest bar"""
        df = self.compute_indicators(df)
        
        latest = df.iloc[-1]
        price = latest['c']
        rsi = latest['rsi']
        bb_lower = latest['bb_lower']
        bb_middle = latest['bb_middle']
        bb_upper = latest['bb_upper']
        
        # Default: no trade
        signal = 'NO_TRADE'
        confidence = 0
        reason = 'No setup'
        entry = price
        stop = price
        tp1 = price
        tp2 = price
        
        # BUY SETUP: Oversold + Below Lower BB
        if rsi < self.params['rsi_oversold'] and price < bb_lower:
            signal = 'BUY'
            entry = price
            stop = price * (1 - self.params['stop_pct'])
            tp1 = bb_middle  # Mean reversion target
            tp2 = bb_upper   # Extended target
            
            # Confidence based on how extreme the oversold condition is
            rsi_extreme = max(0, self.params['rsi_oversold'] - rsi) / 30 * 50  # 0-50
            price_extreme = max(0, (bb_lower - price) / price * 100) / 1 * 30  # 0-30
            confidence = min(100, 40 + rsi_extreme + price_extreme)
            reason = f"RSI oversold ({rsi:.1f}) + Price below lower BB"
        
        # SELL SETUP: Overbought + Above Upper BB
        elif rsi > self.params['rsi_overbought'] and price > bb_upper:
            signal = 'SELL'
            entry = price
            stop = price * (1 + self.params['stop_pct'])
            tp1 = bb_middle
            tp2 = bb_lower
            
            rsi_extreme = max(0, rsi - self.params['rsi_overbought']) / 30 * 50
            price_extreme = max(0, (price - bb_upper) / price * 100) / 1 * 30
            confidence = min(100, 40 + rsi_extreme + price_extreme)
            reason = f"RSI overbought ({rsi:.1f}) + Price above upper BB"
        
        return GoldSignal(
            timestamp=str(latest.get('time', datetime.now().isoformat())),
            timeframe=self.timeframe,
            signal=signal,
            price=price,
            rsi=rsi,
            bb_lower=bb_lower,
            bb_middle=bb_middle,
            bb_upper=bb_upper,
            entry_price=entry,
            stop_loss=stop,
            target_1=tp1,
            target_2=tp2,
            confidence=confidence,
            reason=reason
        )
    
    def generate_all_signals(self, df: pd.DataFrame) -> List[dict]:
        """Generate signals for all bars (for backtesting)"""
        df = self.compute_indicators(df)
        
        signals = []
        
        for i in range(self.params['bb_period'], len(df)):
            row = df.iloc[i]
            price = row['c']
            rsi = row['rsi']
            bb_lower = row['bb_lower']
            bb_upper = row['bb_upper']
            
            signal = None
            
            if rsi < self.params['rsi_oversold'] and price < bb_lower:
                signal = 'BUY'
            elif rsi > self.params['rsi_overbought'] and price > bb_upper:
                signal = 'SELL'
            
            if signal:
                signals.append({
                    'bar_idx': i,
                    'signal': signal,
                    'price': price,
                    'rsi': rsi,
                    'bb_lower': bb_lower,
                    'bb_upper': bb_upper,
                    'bb_middle': row['bb_middle']
                })
        
        return signals


# =============================================================================
# RULE-BASED BACKTEST
# =============================================================================

@dataclass
class GoldBacktestResult:
    """Gold backtest results"""
    symbol: str
    timeframe: str
    strategy: str
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    
    sharpe_ratio: float = 0.0
    
    avg_winner_pct: float = 0.0
    avg_loser_pct: float = 0.0


def run_gold_backtest(symbol: str = 'PAXGUSDT', timeframe: str = '15m') -> GoldBacktestResult:
    """Run rule-based backtest for Gold"""
    print(f"\n{'='*60}")
    print(f"📊 Gold Rule-Based Backtest: {symbol} {timeframe}")
    print(f"{'='*60}")
    
    # Load data
    data_path = DATA_DIR / f"{symbol}_{timeframe}.csv"
    if not data_path.exists():
        print(f"   ❌ Data not found: {data_path}")
        return GoldBacktestResult(symbol=symbol, timeframe=timeframe, strategy='MEAN_REVERSION')
    
    df = pd.read_csv(data_path)
    df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
    print(f"   📂 Loaded {len(df):,} bars")
    
    # Get params
    params = GOLD_RULES.get(timeframe, GOLD_RULES['15m'])
    
    # Generate signals
    engine = GoldMeanReversionEngine(timeframe)
    signals = engine.generate_all_signals(df)
    print(f"   📈 Raw signals: {len(signals):,}")
    
    # Simulate trades with minimum spacing
    trades = []
    last_trade_bar = -params['max_hold']
    capital = 10000
    equity = [capital]
    
    for sig in signals:
        bar_idx = sig['bar_idx']
        
        # Skip if too close to last trade
        if bar_idx - last_trade_bar < params['max_hold']:
            continue
        
        # Simulate trade
        entry_price = sig['price']
        entry_bar = bar_idx
        direction = sig['signal']
        
        target_price = sig['bb_middle']  # Mean reversion target
        stop_pct = params['stop_pct']
        
        if direction == 'BUY':
            stop_price = entry_price * (1 - stop_pct)
        else:
            stop_price = entry_price * (1 + stop_pct)
        
        # Simulate forward
        exit_price = None
        exit_reason = None
        
        for j in range(1, params['max_hold'] + 1):
            future_bar = entry_bar + j
            if future_bar >= len(df):
                break
            
            high = df.iloc[future_bar]['h']
            low = df.iloc[future_bar]['l']
            close = df.iloc[future_bar]['c']
            
            if direction == 'BUY':
                # Check stop
                if low <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'SL'
                    break
                # Check target (mean reversion)
                if high >= target_price:
                    exit_price = target_price
                    exit_reason = 'TP'
                    break
            else:  # SELL
                if high >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'SL'
                    break
                if low <= target_price:
                    exit_price = target_price
                    exit_reason = 'TP'
                    break
        
        # Time exit
        if exit_price is None:
            exit_bar = min(entry_bar + params['max_hold'], len(df) - 1)
            exit_price = df.iloc[exit_bar]['c']
            exit_reason = 'TIME'
        
        # Calculate P&L
        if direction == 'BUY':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        
        # Add trade
        trades.append({
            'direction': direction,
            'entry': entry_price,
            'exit': exit_price,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason
        })
        
        # Update capital
        position_size = capital * 0.02  # 2% risk
        pnl_dollar = position_size * (pnl_pct / 100)
        capital += pnl_dollar
        equity.append(capital)
        
        last_trade_bar = bar_idx
    
    print(f"   📊 Executed trades: {len(trades)}")
    
    # Calculate metrics
    result = GoldBacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        strategy='MEAN_REVERSION'
    )
    
    if not trades:
        return result
    
    result.total_trades = len(trades)
    
    winners = [t for t in trades if t['pnl_pct'] > 0]
    losers = [t for t in trades if t['pnl_pct'] <= 0]
    
    result.winning_trades = len(winners)
    result.losing_trades = len(losers)
    result.win_rate = len(winners) / len(trades) * 100
    
    if winners:
        result.avg_winner_pct = np.mean([t['pnl_pct'] for t in winners])
    if losers:
        result.avg_loser_pct = abs(np.mean([t['pnl_pct'] for t in losers]))
    
    gross_profit = sum(t['pnl_pct'] for t in winners) if winners else 0
    gross_loss = abs(sum(t['pnl_pct'] for t in losers)) if losers else 0.01
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    
    result.expectancy_r = np.mean([t['pnl_pct'] for t in trades])
    result.total_return_pct = (equity[-1] - 10000) / 10000 * 100
    
    # Max drawdown
    peak = 10000
    max_dd = 0
    for eq in equity:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)
    result.max_drawdown_pct = max_dd
    
    # Sharpe
    returns = pd.Series([t['pnl_pct'] for t in trades])
    if len(returns) > 1:
        result.sharpe_ratio = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
    
    print(f"\n   📈 Results:")
    print(f"      Trades: {result.total_trades}")
    print(f"      Win Rate: {result.win_rate:.1f}%")
    print(f"      Profit Factor: {result.profit_factor:.2f}")
    print(f"      Sharpe: {result.sharpe_ratio:.2f}")
    print(f"      Max DD: {result.max_drawdown_pct:.1f}%")
    print(f"      Total Return: {result.total_return_pct:.1f}%")
    
    return result


# =============================================================================
# LIVE SIGNAL CHECKER
# =============================================================================

def check_gold_signal(timeframe: str = '15m') -> GoldSignal:
    """Check current Gold signal"""
    symbol = 'PAXGUSDT'
    
    data_path = DATA_DIR / f"{symbol}_{timeframe}.csv"
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
    
    # Use last 100 bars
    df = df.tail(100)
    
    engine = GoldMeanReversionEngine(timeframe)
    signal = engine.generate_signal(df)
    
    print(f"\n{'='*60}")
    print(f"🥇 GOLD MEAN-REVERSION SIGNAL: {timeframe}")
    print(f"{'='*60}")
    print(f"   Signal: {signal.signal}")
    print(f"   Price: ${signal.price:,.2f}")
    print(f"   RSI: {signal.rsi:.1f}")
    print(f"   BB Lower: ${signal.bb_lower:,.2f}")
    print(f"   BB Middle: ${signal.bb_middle:,.2f}")
    print(f"   BB Upper: ${signal.bb_upper:,.2f}")
    
    if signal.signal != 'NO_TRADE':
        print(f"\n   📋 TRADE PLAN:")
        print(f"      Entry: ${signal.entry_price:,.2f}")
        print(f"      Stop Loss: ${signal.stop_loss:,.2f}")
        print(f"      Target 1 (Mean): ${signal.target_1:,.2f}")
        print(f"      Target 2: ${signal.target_2:,.2f}")
        print(f"      Confidence: {signal.confidence:.1f}%")
        print(f"      Reason: {signal.reason}")
    else:
        print(f"\n   ⏳ Waiting for setup...")
        print(f"      Need RSI < {GOLD_RULES[timeframe]['rsi_oversold']} for BUY")
        print(f"      Need RSI > {GOLD_RULES[timeframe]['rsi_overbought']} for SELL")
    
    return signal


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Gold Mean-Reversion System')
    parser.add_argument('command', choices=['signal', 'backtest'], help='Command')
    parser.add_argument('--symbol', default='PAXGUSDT', help='Symbol')
    parser.add_argument('--timeframe', default='15m', help='Timeframe')
    parser.add_argument('--all', action='store_true', help='Run all timeframes')
    
    args = parser.parse_args()
    
    if args.command == 'signal':
        check_gold_signal(args.timeframe)
    
    elif args.command == 'backtest':
        if args.all:
            all_results = []
            for tf in ['15m', '30m', '1h']:
                result = run_gold_backtest('PAXGUSDT', tf)
                all_results.append(result)
            
            print(f"\n{'='*60}")
            print("📊 GOLD MEAN-REVERSION SUMMARY")
            print(f"{'='*60}")
            for r in all_results:
                print(f"   {r.timeframe}: {r.win_rate:.1f}% WR | {r.profit_factor:.2f} PF | {r.total_return_pct:.1f}% Return")
        else:
            run_gold_backtest(args.symbol, args.timeframe)
