"""
FORGE TRADING SYSTEM - DRIFT DETECTOR
======================================
Monitors live vs backtest consistency.
Detects: signal frequency, directional bias, win-rate, R-multiple drift.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from enum import Enum
import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class DriftState(Enum):
    STABLE = "STABLE"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


class DriftDetector:
    """
    Monitors live system for drift from expected backtest behavior.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.drift_cfg = self.config.get('drift', {})
        
        # Thresholds
        self.signal_freq_warn = self.drift_cfg.get('signal_freq_warn', 0.10)
        self.signal_freq_critical = self.drift_cfg.get('signal_freq_critical', 0.25)
        self.winrate_warn = self.drift_cfg.get('winrate_warn', 0.05)
        self.winrate_critical = self.drift_cfg.get('winrate_critical', 0.10)
        
        # Expected baselines (from backtest)
        self.baseline = {
            'signal_freq': 0.05,    # Expected signals per candle
            'buy_ratio': 0.5,       # Expected buy/sell ratio
            'win_rate': 0.65,       # Expected win rate
            'avg_r': 1.2            # Expected R-multiple
        }
        
        # Rolling windows
        self.signal_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        self.current_state = DriftState.STABLE
        self.drift_reasons: List[str] = []
    
    def set_baseline(self, backtest_results: Dict):
        """Set baseline from backtest results"""
        if backtest_results.get('total_trades', 0) > 0:
            self.baseline['win_rate'] = backtest_results.get('win_rate', 0.65)
            self.baseline['avg_r'] = backtest_results.get('avg_r', 1.2)
    
    def record_signal(self, signal: Dict):
        """Record a generated signal"""
        self.signal_history.append({
            'timestamp': datetime.now(timezone.utc),
            'direction': signal.get('direction', ''),
            'score': signal.get('score', 0),
            'symbol': signal.get('symbol', '')
        })
        
        # Keep last 500
        if len(self.signal_history) > 500:
            self.signal_history = self.signal_history[-500:]
    
    def record_trade(self, trade: Dict):
        """Record a completed trade"""
        self.trade_history.append({
            'timestamp': datetime.now(timezone.utc),
            'direction': trade.get('direction', ''),
            'pnl': trade.get('pnl', 0),
            'r_multiple': trade.get('r_multiple', 0),
            'won': trade.get('pnl', 0) > 0
        })
        
        # Keep last 100
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def _check_signal_frequency(self) -> tuple:
        """Check if signal frequency is drifting"""
        if len(self.signal_history) < 20:
            return DriftState.STABLE, None
        
        # Calculate recent signal rate
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        recent = [s for s in self.signal_history if s['timestamp'] > hour_ago]
        
        # Rough: expect ~12 candles per hour (5m), baseline is 5% = 0.6 signals/hour
        expected = self.baseline['signal_freq'] * 12
        actual = len(recent)
        
        if expected > 0:
            drift = abs(actual - expected) / max(expected, 1)
            
            if drift > self.signal_freq_critical:
                return DriftState.CRITICAL, f"Signal frequency drift: {drift:.0%}"
            elif drift > self.signal_freq_warn:
                return DriftState.DEGRADED, f"Signal frequency drift: {drift:.0%}"
        
        return DriftState.STABLE, None
    
    def _check_directional_bias(self) -> tuple:
        """Check for unexpected directional bias"""
        if len(self.signal_history) < 30:
            return DriftState.STABLE, None
        
        recent = self.signal_history[-30:]
        buys = sum(1 for s in recent if s['direction'] == 'BUY')
        
        buy_ratio = buys / len(recent)
        expected = self.baseline['buy_ratio']
        drift = abs(buy_ratio - expected)
        
        if drift > 0.3:
            return DriftState.CRITICAL, f"Directional bias: {buy_ratio:.0%} BUY (expected {expected:.0%})"
        elif drift > 0.15:
            return DriftState.DEGRADED, f"Directional bias: {buy_ratio:.0%} BUY"
        
        return DriftState.STABLE, None
    
    def _check_winrate(self) -> tuple:
        """Check rolling win rate"""
        if len(self.trade_history) < 10:
            return DriftState.STABLE, None
        
        recent = self.trade_history[-20:]
        wins = sum(1 for t in recent if t['won'])
        win_rate = wins / len(recent)
        
        expected = self.baseline['win_rate']
        drift = expected - win_rate
        
        if drift > self.winrate_critical:
            return DriftState.CRITICAL, f"Win rate degraded: {win_rate:.0%} (expected {expected:.0%})"
        elif drift > self.winrate_warn:
            return DriftState.DEGRADED, f"Win rate below target: {win_rate:.0%}"
        
        return DriftState.STABLE, None
    
    def _check_r_multiple(self) -> tuple:
        """Check average R-multiple"""
        if len(self.trade_history) < 10:
            return DriftState.STABLE, None
        
        recent = self.trade_history[-20:]
        avg_r = np.mean([t['r_multiple'] for t in recent])
        expected = self.baseline['avg_r']
        
        if avg_r < 0:
            return DriftState.CRITICAL, f"Negative avg R: {avg_r:.2f}"
        elif avg_r < expected * 0.5:
            return DriftState.DEGRADED, f"Low avg R: {avg_r:.2f} (expected {expected:.2f})"
        
        return DriftState.STABLE, None
    
    def evaluate(self) -> Dict:
        """
        Run all drift checks and return overall state.
        """
        checks = [
            ('Signal Freq', self._check_signal_frequency()),
            ('Direction', self._check_directional_bias()),
            ('Win Rate', self._check_winrate()),
            ('R-Multiple', self._check_r_multiple())
        ]
        
        self.drift_reasons = []
        worst_state = DriftState.STABLE
        
        results = []
        for name, (state, reason) in checks:
            results.append({
                'check': name,
                'state': state.value,
                'reason': reason or 'OK'
            })
            
            if reason:
                self.drift_reasons.append(f"{name}: {reason}")
            
            if state == DriftState.CRITICAL:
                worst_state = DriftState.CRITICAL
            elif state == DriftState.DEGRADED and worst_state != DriftState.CRITICAL:
                worst_state = DriftState.DEGRADED
        
        self.current_state = worst_state
        
        return {
            'state': worst_state.value,
            'checks': results,
            'reasons': self.drift_reasons,
            'confidence_multiplier': self._get_confidence_multiplier()
        }
    
    def _get_confidence_multiplier(self) -> float:
        """Get confidence scaling based on drift state"""
        if self.current_state == DriftState.STABLE:
            return 1.0
        elif self.current_state == DriftState.DEGRADED:
            return 0.7  # Reduce confidence by 30%
        else:
            return 0.0  # Freeze signals
    
    def should_freeze(self) -> bool:
        """Check if system should freeze new signals"""
        return self.current_state == DriftState.CRITICAL
