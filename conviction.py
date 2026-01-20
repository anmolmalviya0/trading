"""
FORGE TRADING SYSTEM - CONVICTION LAYER
========================================
Mandatory gatekeeper that filters all signals.
Every signal must pass ALL checks.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone
import yaml

from .features import get_market_regime


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class ConvictionFilter:
    """
    Mandatory gatekeeper for all signals.
    Default behavior: reject if any data is missing.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.conv_cfg = self.config.get('conviction', {})
        self.enabled = self.conv_cfg.get('enabled', True)
        self.reject_on_missing = self.conv_cfg.get('reject_on_missing', True)
    
    def check_spread(self, signal: Dict, current_price: float = None) -> Tuple[bool, str]:
        """
        Check if spread is within acceptable range.
        Uses estimated spread based on asset type.
        """
        max_bps = self.conv_cfg.get('spread_max_bps', 10)
        
        symbol = signal.get('symbol', '')
        
        # Estimated typical spreads (basis points)
        estimated_spreads = {
            'BTCUSDT': 2,
            'ETHUSDT': 3,
            'SOLUSDT': 5,
            'BNBUSDT': 4,
            'PAXGUSDT': 8
        }
        
        spread = estimated_spreads.get(symbol, 10)
        
        if spread > max_bps:
            return False, f"Spread too wide: {spread}bps > {max_bps}bps"
        
        return True, f"Spread OK: {spread}bps"
    
    def check_regime(self, signal: Dict, df: pd.DataFrame = None) -> Tuple[bool, str]:
        """
        Check if market regime is suitable for trading.
        Reject if range-bound or extremely volatile.
        """
        regime = signal.get('regime', 'unknown')
        direction = signal.get('direction', '')
        
        # Reject unknown regime
        if regime == 'unknown':
            if self.reject_on_missing:
                return False, "Regime unknown"
            return True, "Regime unknown (allowed)"
        
        # Reject range markets
        if regime == 'range':
            return False, "Market is range-bound"
        
        # Reject extreme volatility
        if regime == 'volatile':
            return False, "Market too volatile"
        
        # Check trend alignment
        if direction == 'BUY' and regime == 'trend_down':
            return False, "Buying against downtrend"
        
        if direction == 'SELL' and regime == 'trend_up':
            return False, "Selling against uptrend"
        
        return True, f"Regime aligned: {regime}"
    
    def check_slippage(self, signal: Dict) -> Tuple[bool, str]:
        """
        Estimate slippage impact.
        """
        atr = signal.get('atr', 0)
        entry = signal.get('entry', 0)
        
        if atr == 0 or entry == 0:
            if self.reject_on_missing:
                return False, "Missing ATR/entry data"
            return True, "Slippage check skipped"
        
        # Estimate slippage as % of ATR
        slippage_cfg = self.config.get('risk', {}).get('slippage_rate', 0.0005)
        estimated_slip = entry * slippage_cfg
        slip_atr_ratio = estimated_slip / atr
        
        # Reject if slippage > 20% of ATR
        if slip_atr_ratio > 0.2:
            return False, f"Slippage too high ({slip_atr_ratio:.1%} of ATR)"
        
        return True, f"Slippage OK ({slip_atr_ratio:.1%} of ATR)"
    
    def check_session(self, signal: Dict) -> Tuple[bool, str]:
        """
        Check if current time is in an active trading session.
        """
        sessions_cfg = self.config.get('sessions', {})
        
        if not sessions_cfg.get('enabled', True):
            return True, "Session filter disabled"
        
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        asia = sessions_cfg.get('asia', {})
        london = sessions_cfg.get('london', {})
        ny = sessions_cfg.get('newyork', {})
        
        in_asia = asia.get('start', 0) <= hour < asia.get('end', 8)
        in_london = london.get('start', 8) <= hour < london.get('end', 16)
        in_ny = ny.get('start', 13) <= hour < ny.get('end', 22)
        
        if in_london or in_ny:
            session = "London" if in_london and not in_ny else ("New York" if in_ny else "London/NY overlap")
            return True, f"Active session: {session}"
        
        if in_asia:
            return True, "Active session: Asia"
        
        return False, "Outside trading sessions"
    
    def check_score_quality(self, signal: Dict) -> Tuple[bool, str]:
        """
        Check signal quality metrics.
        """
        score = signal.get('score', 0)
        num_signals = signal.get('num_signals', 0)
        
        min_conf = self.config.get('signals', {}).get('min_confirmations', 2)
        threshold = self.config.get('signals', {}).get('threshold', 70)
        
        if score < threshold:
            return False, f"Score below threshold: {score} < {threshold}"
        
        if num_signals < min_conf:
            return False, f"Too few confirmations: {num_signals} < {min_conf}"
        
        return True, f"Score: {score}, Confirmations: {num_signals}"
    
    def evaluate(self, signal: Dict, df: pd.DataFrame = None, current_price: float = None) -> Dict:
        """
        Evaluate signal through all conviction filters.
        Returns conviction result with pass/fail and reasons.
        """
        if not self.enabled:
            return {
                'passed': True,
                'checks': [('Filter disabled', True, 'Conviction filter is disabled')],
                'reason': 'Conviction filter disabled'
            }
        
        checks = []
        
        # Run all checks
        spread_pass, spread_msg = self.check_spread(signal, current_price)
        checks.append(('Spread', spread_pass, spread_msg))
        
        regime_pass, regime_msg = self.check_regime(signal, df)
        checks.append(('Regime', regime_pass, regime_msg))
        
        slip_pass, slip_msg = self.check_slippage(signal)
        checks.append(('Slippage', slip_pass, slip_msg))
        
        session_pass, session_msg = self.check_session(signal)
        checks.append(('Session', session_pass, session_msg))
        
        quality_pass, quality_msg = self.check_score_quality(signal)
        checks.append(('Quality', quality_pass, quality_msg))
        
        # All must pass
        all_passed = all(c[1] for c in checks)
        
        # Build reason string
        if all_passed:
            reason = "All conviction checks passed"
        else:
            failed = [c[2] for c in checks if not c[1]]
            reason = "; ".join(failed)
        
        return {
            'passed': all_passed,
            'checks': checks,
            'reason': reason
        }


def create_trade_card(signal: Dict, conviction: Dict) -> str:
    """
    Create a formatted trade card for display.
    """
    if not conviction['passed']:
        status = "❌ REJECTED"
        color = "red"
    else:
        status = f"🟢 {signal['direction']}"
        color = "green" if signal['direction'] == 'BUY' else "red"
    
    symbol = signal.get('symbol', 'UNKNOWN')
    tf = signal.get('timeframe', '?')
    entry = signal.get('entry', 0)
    sl = signal.get('sl', 0)
    tp = signal.get('tp', 0)
    score = signal.get('score', 0)
    reasons = signal.get('reasons', [])
    
    card = f"""
┌─────────────────────────────────────────┐
│ {status}  {symbol}  {tf}
├─────────────────────────────────────────┤
│ Entry: ${entry:,.2f}
│ SL: ${sl:,.2f}  |  TP: ${tp:,.2f}
│ Score: {score}  |  Conviction: {'PASS' if conviction['passed'] else 'FAIL'}
├─────────────────────────────────────────┤
│ Reasons: {', '.join(reasons[:2]) if reasons else 'N/A'}
│ Filter: {conviction['reason'][:50]}
└─────────────────────────────────────────┘
"""
    return card
