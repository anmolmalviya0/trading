"""
V8 FINAL - BACKTESTER
======================
Strict backtesting with multi-signal validation.
"""
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List


def load_config():
    try:
        with open('config.yaml') as f:
            return yaml.safe_load(f)
    except:
        return {}


class Backtester:
    def __init__(self, engine, config: dict = None):
        self.engine = engine
        self.config = config or load_config()
        
        risk = self.config.get('risk', {})
        self.fee = risk.get('fee_rate', 0.001)
        self.slip = risk.get('slippage_rate', 0.0005)
        self.max_hold = risk.get('max_holding_bars', 20)
    
    def run(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        trades = []
        
        for i in range(100, len(df) - 25):
            subset = df.iloc[:i+1].copy()
            signal = self.engine.analyze(subset, symbol, timeframe)
            
            if signal is None:
                continue
            
            result = self._simulate(df, i, signal)
            if result:
                trades.append(result)
        
        return self._stats(trades, symbol, timeframe)
    
    def _simulate(self, df: pd.DataFrame, idx: int, signal: Dict) -> Dict:
        direction = signal['direction']
        entry_raw = signal['entry']
        sl = signal['sl']
        tp = signal['tp']
        
        if direction == 'BUY':
            entry = entry_raw * (1 + self.slip)
        else:
            entry = entry_raw * (1 - self.slip)
        
        risk = abs(entry - sl)
        if risk <= 0:
            return None
        
        outcome = 0
        exit_price = entry
        
        for j in range(idx + 1, min(idx + 1 + self.max_hold, len(df))):
            bar = df.iloc[j]
            
            if direction == 'BUY':
                if bar['low'] <= sl:
                    outcome = -1
                    exit_price = sl
                    break
                if bar['high'] >= tp:
                    outcome = 1
                    exit_price = tp
                    break
            else:
                if bar['high'] >= sl:
                    outcome = -1
                    exit_price = sl
                    break
                if bar['low'] <= tp:
                    outcome = 1
                    exit_price = tp
                    break
        
        if outcome == 0:
            exit_price = df.iloc[min(idx + self.max_hold, len(df)-1)]['close']
            if direction == 'BUY':
                outcome = 1 if exit_price > entry else -1
            else:
                outcome = 1 if exit_price < entry else -1
        
        if direction == 'BUY':
            pnl = exit_price - entry
        else:
            pnl = entry - exit_price
        
        r_mult = pnl / risk - (entry + exit_price) * self.fee / risk
        
        return {
            'direction': direction,
            'outcome': outcome,
            'r_mult': r_mult,
            'num_signals': signal['num_signals']
        }
    
    def _stats(self, trades: List[Dict], symbol: str, timeframe: str) -> Dict:
        if not trades:
            return {'symbol': symbol, 'timeframe': timeframe, 'trades': 0, 'winrate': 0, 'pf': 0}
        
        wins = sum(1 for t in trades if t['r_mult'] > 0)
        total = len(trades)
        
        gross_profit = sum(t['r_mult'] for t in trades if t['r_mult'] > 0)
        gross_loss = abs(sum(t['r_mult'] for t in trades if t['r_mult'] <= 0))
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'trades': total,
            'wins': wins,
            'winrate': wins / total * 100 if total > 0 else 0,
            'pf': gross_profit / gross_loss if gross_loss > 0 else 0
        }
