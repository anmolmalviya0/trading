"""
FORGE TRADING SYSTEM - MUTATION LAB
====================================
Safe offline parameter optimizer.
NEVER modifies live system automatically.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
import json
import copy

from .backtest import Backtester
from .features import add_features


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@dataclass
class MutationResult:
    """Result of a mutation experiment"""
    mutation_id: str
    parameter: str
    original_value: float
    mutated_value: float
    backtest_trades: int
    backtest_winrate: float
    backtest_pf: float
    backtest_drawdown: float
    improvement: float  # vs baseline
    recommended: bool
    notes: str


class MutationLab:
    """
    Safe parameter optimizer that runs offline/shadow mode only.
    
    RULES:
    1. Never update live systems automatically
    2. Log all proposals for audit
    3. Evaluate on rolling historical data only
    4. Generate recommendations, not actions
    """
    
    # Parameters that can be mutated
    MUTABLE_PARAMS = [
        ('signals.threshold', 50, 90, 5),           # min, max, step
        ('signals.min_confirmations', 1, 4, 1),
        ('conviction.spread_max_bps', 5, 20, 5),
        ('conviction.regime_adx_min', 15, 30, 5),
        ('risk.sl_atr_mult', 1.0, 2.5, 0.25),
        ('risk.tp_atr_mult', 1.5, 4.0, 0.5),
    ]
    
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.results: List[MutationResult] = []
        self.baseline: Dict = {}
        self.recommendations: List[Dict] = []
        
        # Audit log
        self.audit_log: List[Dict] = []
    
    def _log_audit(self, action: str, details: Dict):
        """Log action for audit"""
        self.audit_log.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': action,
            'details': details
        })
    
    def _get_nested_value(self, config: dict, path: str):
        """Get nested config value"""
        keys = path.split('.')
        val = config
        for k in keys:
            val = val.get(k, {})
        return val
    
    def _set_nested_value(self, config: dict, path: str, value):
        """Set nested config value"""
        keys = path.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    
    def run_baseline(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Run backtest with current config as baseline"""
        bt = Backtester(self.config)
        result = bt.run(df, symbol, timeframe)
        
        self.baseline = {
            'trades': result['total_trades'],
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'max_drawdown': result['max_drawdown']
        }
        
        self._log_audit('baseline', self.baseline)
        return self.baseline
    
    def mutate_and_test(self, df: pd.DataFrame, symbol: str, timeframe: str,
                        param_path: str, values: List) -> List[MutationResult]:
        """Test a range of values for a parameter"""
        results = []
        original_value = self._get_nested_value(self.config, param_path)
        
        for val in values:
            # Create mutated config
            mutated_config = copy.deepcopy(self.config)
            self._set_nested_value(mutated_config, param_path, val)
            
            # Run backtest
            bt = Backtester(mutated_config)
            result = bt.run(df, symbol, timeframe)
            
            # Calculate improvement
            baseline_score = self.baseline.get('profit_factor', 1) * self.baseline.get('win_rate', 0.5)
            mutated_score = result['profit_factor'] * result['win_rate']
            improvement = (mutated_score - baseline_score) / max(baseline_score, 0.01)
            
            # Determine if recommended
            recommended = (
                improvement > 0.1 and  # 10% improvement
                result['max_drawdown'] <= self.baseline.get('max_drawdown', 1) * 1.2 and  # DD not too bad
                result['total_trades'] >= self.baseline.get('trades', 0) * 0.5  # Still has trades
            )
            
            mutation_result = MutationResult(
                mutation_id=f"{param_path}_{val}_{datetime.now().timestamp()}",
                parameter=param_path,
                original_value=original_value,
                mutated_value=val,
                backtest_trades=result['total_trades'],
                backtest_winrate=result['win_rate'],
                backtest_pf=result['profit_factor'],
                backtest_drawdown=result['max_drawdown'],
                improvement=improvement,
                recommended=recommended,
                notes=f"{'RECOMMEND' if recommended else 'REJECT'}: {improvement:+.1%} vs baseline"
            )
            
            results.append(mutation_result)
            self.results.append(mutation_result)
            
            self._log_audit('mutation_test', {
                'parameter': param_path,
                'value': val,
                'result': mutation_result.notes
            })
        
        return results
    
    def run_full_optimization(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Run optimization across all mutable parameters"""
        print(f"Running mutation lab for {symbol} {timeframe}...")
        
        # Establish baseline
        self.run_baseline(df, symbol, timeframe)
        print(f"  Baseline: WR={self.baseline['win_rate']:.1%}, PF={self.baseline['profit_factor']:.2f}")
        
        # Test each parameter
        for param_path, min_val, max_val, step in self.MUTABLE_PARAMS:
            values = list(np.arange(min_val, max_val + step, step))
            results = self.mutate_and_test(df, symbol, timeframe, param_path, values)
            
            # Find best
            recommended = [r for r in results if r.recommended]
            if recommended:
                best = max(recommended, key=lambda x: x.improvement)
                print(f"  {param_path}: RECOMMEND {best.mutated_value} ({best.improvement:+.1%})")
                
                self.recommendations.append({
                    'parameter': param_path,
                    'current': best.original_value,
                    'recommended': best.mutated_value,
                    'improvement': best.improvement,
                    'details': best.notes
                })
        
        return self.get_report()
    
    def get_report(self) -> Dict:
        """Get optimization report"""
        return {
            'baseline': self.baseline,
            'total_mutations_tested': len(self.results),
            'recommendations': self.recommendations,
            'summary': f"{len(self.recommendations)} parameters suggest improvement",
            'warning': 'NEVER APPLY AUTOMATICALLY - Review and test manually'
        }
    
    def export_recommendations(self, path: str = 'mutation_recommendations.json'):
        """Export recommendations to file for review"""
        report = self.get_report()
        report['audit_log'] = self.audit_log
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self._log_audit('export', {'path': path})
        return path
    
    def get_recommended_config(self) -> Dict:
        """
        Get config with recommended changes applied.
        
        WARNING: This is for REVIEW ONLY.
        Do NOT automatically apply to live system.
        """
        new_config = copy.deepcopy(self.config)
        
        for rec in self.recommendations:
            self._set_nested_value(new_config, rec['parameter'], rec['recommended'])
        
        return new_config
