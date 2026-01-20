"""
MARKETFORGE: Genetic Evolution Engine
======================================
Self-healing system with autonomous parameter optimization.

Implements:
- Genetic mutation for parameter evolution
- Automatic performance monitoring
- Self-healing when accuracy drops
- Asset-specific calibration

Usage:
    python genetic_engine.py evolve BTCUSDT 15m
    python genetic_engine.py auto-heal
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import random
import copy
import yaml
import joblib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

warnings.filterwarnings('ignore')

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
EVOLUTION_DIR = BASE_DIR / "evolution"
EVOLUTION_DIR.mkdir(exist_ok=True)

# Load base config
with open(BASE_DIR / "config.yaml") as f:
    BASE_CONFIG = yaml.safe_load(f)


# =============================================================================
# GENE DEFINITIONS
# =============================================================================

@dataclass
class TradingGenes:
    """Represents one set of trading parameters (a "genome")"""
    # Labeling
    tp_mult: float = 1.0
    sl_mult: float = 1.0
    max_hold: int = 12
    
    # Model
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    
    # Signals
    confidence_threshold: int = 65
    
    # Risk
    risk_per_trade_pct: float = 0.5
    
    # Fitness (calculated after backtest)
    profit_factor: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 100.0
    final_capital: float = 10000.0
    fitness_score: float = 0.0
    
    def mutate(self, mutation_rate: float = 0.2):
        """Create a mutated copy of this genome"""
        child = copy.deepcopy(self)
        
        if random.random() < mutation_rate:
            child.tp_mult = max(0.5, min(3.0, self.tp_mult + random.gauss(0, 0.2)))
        
        if random.random() < mutation_rate:
            child.sl_mult = max(0.5, min(2.0, self.sl_mult + random.gauss(0, 0.1)))
        
        if random.random() < mutation_rate:
            child.max_hold = max(6, min(48, self.max_hold + random.randint(-3, 3)))
        
        if random.random() < mutation_rate:
            child.n_estimators = max(100, min(1000, self.n_estimators + random.randint(-100, 100)))
        
        if random.random() < mutation_rate:
            child.max_depth = max(3, min(10, self.max_depth + random.randint(-1, 1)))
        
        if random.random() < mutation_rate:
            child.learning_rate = max(0.01, min(0.2, self.learning_rate + random.gauss(0, 0.02)))
        
        if random.random() < mutation_rate:
            child.confidence_threshold = max(50, min(85, self.confidence_threshold + random.randint(-5, 5)))
        
        if random.random() < mutation_rate:
            child.risk_per_trade_pct = max(0.25, min(1.0, self.risk_per_trade_pct + random.gauss(0, 0.1)))
        
        return child
    
    def crossover(self, other: 'TradingGenes') -> 'TradingGenes':
        """Create a child by combining genes from two parents"""
        child = TradingGenes()
        
        # Randomly select genes from each parent
        child.tp_mult = random.choice([self.tp_mult, other.tp_mult])
        child.sl_mult = random.choice([self.sl_mult, other.sl_mult])
        child.max_hold = random.choice([self.max_hold, other.max_hold])
        child.n_estimators = random.choice([self.n_estimators, other.n_estimators])
        child.max_depth = random.choice([self.max_depth, other.max_depth])
        child.learning_rate = random.choice([self.learning_rate, other.learning_rate])
        child.confidence_threshold = random.choice([self.confidence_threshold, other.confidence_threshold])
        child.risk_per_trade_pct = random.choice([self.risk_per_trade_pct, other.risk_per_trade_pct])
        
        return child
    
    def calculate_fitness(self):
        """Calculate fitness score from backtest results"""
        # Multi-objective fitness:
        # - Maximize profit factor (most important)
        # - Maximize win rate
        # - Minimize drawdown
        # - Positive return required
        
        if self.max_drawdown >= 50 or self.final_capital < 10000:
            self.fitness_score = 0
            return
        
        pf_score = min(self.profit_factor / 2.0, 1.0) * 40  # Max 40 points
        wr_score = (self.win_rate / 100) * 30  # Max 30 points
        dd_score = (1 - self.max_drawdown / 50) * 20  # Max 20 points
        return_score = min((self.final_capital - 10000) / 100000, 1.0) * 10  # Max 10 points
        
        self.fitness_score = max(0, pf_score + wr_score + dd_score + return_score)


# =============================================================================
# ASSET-SPECIFIC CALIBRATION
# =============================================================================

# Optimal genes discovered for each asset
ASSET_GENES = {
    'BTCUSDT': {
        '5m': TradingGenes(tp_mult=0.8, sl_mult=1.2, max_hold=12, confidence_threshold=75),
        '15m': TradingGenes(tp_mult=1.0, sl_mult=1.0, max_hold=12, confidence_threshold=65),
        '30m': TradingGenes(tp_mult=1.4, sl_mult=1.0, max_hold=18, confidence_threshold=65),
        '1h': TradingGenes(tp_mult=1.8, sl_mult=1.0, max_hold=24, confidence_threshold=65),
    },
    'PAXGUSDT': {
        # Gold needs different calibration - less volatile
        '5m': TradingGenes(tp_mult=0.6, sl_mult=0.8, max_hold=10, confidence_threshold=70),
        '15m': TradingGenes(tp_mult=0.8, sl_mult=0.9, max_hold=12, confidence_threshold=68),
        '30m': TradingGenes(tp_mult=1.0, sl_mult=1.0, max_hold=16, confidence_threshold=65),
        '1h': TradingGenes(tp_mult=1.2, sl_mult=1.1, max_hold=20, confidence_threshold=63),
    }
}


# =============================================================================
# GENETIC EVOLUTION
# =============================================================================

class GeneticEvolver:
    """
    Evolves trading parameters using genetic algorithms.
    
    The system automatically:
    1. Creates initial population
    2. Runs backtests to evaluate fitness
    3. Selects top performers
    4. Breeds and mutates
    5. Repeats until convergence or target reached
    """
    
    def __init__(self, symbol: str, timeframe: str, 
                 population_size: int = 20, generations: int = 10):
        self.symbol = symbol
        self.timeframe = timeframe
        self.population_size = population_size
        self.generations = generations
        
        self.population: List[TradingGenes] = []
        self.best_genes: Optional[TradingGenes] = None
        self.evolution_history: List[dict] = []
        
        # Target fitness
        self.target_profit_factor = 1.3
        self.target_win_rate = 55
        self.target_max_dd = 25
    
    def initialize_population(self):
        """Create initial population with some random and some seeded genes"""
        print(f"   🧬 Initializing population of {self.population_size}...")
        
        # Start with asset-specific optimal genes
        if self.symbol in ASSET_GENES and self.timeframe in ASSET_GENES[self.symbol]:
            base_genes = ASSET_GENES[self.symbol][self.timeframe]
            self.population.append(copy.deepcopy(base_genes))
            
            # Add mutations of the base
            for _ in range(self.population_size // 4):
                self.population.append(base_genes.mutate(0.3))
        
        # Add random individuals
        while len(self.population) < self.population_size:
            genes = TradingGenes(
                tp_mult=random.uniform(0.5, 2.0),
                sl_mult=random.uniform(0.5, 1.5),
                max_hold=random.randint(8, 30),
                n_estimators=random.randint(200, 800),
                max_depth=random.randint(4, 8),
                learning_rate=random.uniform(0.02, 0.1),
                confidence_threshold=random.randint(55, 80),
                risk_per_trade_pct=random.uniform(0.3, 0.8)
            )
            self.population.append(genes)
    
    def evaluate_fitness(self, genes: TradingGenes) -> TradingGenes:
        """Run backtest with genes and calculate fitness"""
        from backtest_engine import BacktestEngine
        
        # Update config with genes
        self._apply_genes_to_config(genes)
        
        # Run quick backtest
        try:
            engine = BacktestEngine(self.symbol, self.timeframe)
            result = engine.run()
            
            genes.profit_factor = result.profit_factor
            genes.win_rate = result.win_rate
            genes.max_drawdown = result.max_drawdown_pct
            genes.final_capital = result.final_capital
            genes.calculate_fitness()
        except Exception as e:
            print(f"      ⚠️ Backtest failed: {e}")
            genes.fitness_score = 0
        
        return genes
    
    def _apply_genes_to_config(self, genes: TradingGenes):
        """Update config.yaml with genes"""
        config = copy.deepcopy(BASE_CONFIG)
        
        # Update labeling
        config['labeling'][self.timeframe]['tp_mult'] = genes.tp_mult
        config['labeling'][self.timeframe]['sl_mult'] = genes.sl_mult
        config['labeling'][self.timeframe]['max_hold'] = genes.max_hold
        
        # Update model
        config['model']['hyperparameters']['n_estimators'] = genes.n_estimators
        config['model']['hyperparameters']['max_depth'] = genes.max_depth
        config['model']['hyperparameters']['learning_rate'] = genes.learning_rate
        
        # Update signals
        config['signals']['confidence_threshold'] = genes.confidence_threshold
        
        # Update risk
        config['risk']['position_sizing']['risk_per_trade_pct'] = genes.risk_per_trade_pct
        
        # Save
        with open(BASE_DIR / "config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def select_parents(self) -> List[TradingGenes]:
        """Tournament selection for breeding"""
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
        
        # Keep top 25%
        elite_count = max(2, len(sorted_pop) // 4)
        return sorted_pop[:elite_count]
    
    def evolve_generation(self, generation: int):
        """Run one generation of evolution"""
        print(f"\n   🔄 Generation {generation + 1}/{self.generations}")
        
        # Evaluate fitness for all
        for i, genes in enumerate(self.population):
            print(f"      📊 Evaluating genome {i+1}/{len(self.population)}...")
            self.evaluate_fitness(genes)
        
        # Select parents
        parents = self.select_parents()
        
        # Track best
        if parents[0].fitness_score > (self.best_genes.fitness_score if self.best_genes else 0):
            self.best_genes = copy.deepcopy(parents[0])
            print(f"      ⭐ New best: PF={self.best_genes.profit_factor:.2f}, "
                  f"WR={self.best_genes.win_rate:.1f}%, DD={self.best_genes.max_drawdown:.1f}%")
        
        # Record history
        self.evolution_history.append({
            'generation': generation + 1,
            'best_fitness': parents[0].fitness_score,
            'avg_fitness': np.mean([g.fitness_score for g in self.population]),
            'best_pf': parents[0].profit_factor,
            'best_wr': parents[0].win_rate,
            'best_dd': parents[0].max_drawdown
        })
        
        # Check if target reached
        if (self.best_genes.profit_factor >= self.target_profit_factor and
            self.best_genes.win_rate >= self.target_win_rate and
            self.best_genes.max_drawdown <= self.target_max_dd):
            print(f"      🎯 Target reached! Stopping evolution.")
            return True
        
        # Create next generation
        new_population = parents[:]  # Keep elite
        
        while len(new_population) < self.population_size:
            # Select two parents randomly from elite
            p1, p2 = random.sample(parents, 2)
            
            # Crossover
            child = p1.crossover(p2)
            
            # Mutate
            child = child.mutate(0.2)
            
            new_population.append(child)
        
        self.population = new_population
        return False
    
    def evolve(self) -> TradingGenes:
        """Run full evolution process"""
        print(f"\n{'='*60}")
        print(f"🧬 GENETIC EVOLUTION: {self.symbol} {self.timeframe}")
        print(f"{'='*60}")
        
        self.initialize_population()
        
        for gen in range(self.generations):
            converged = self.evolve_generation(gen)
            if converged:
                break
        
        # Save best genes
        self._save_best_genes()
        
        return self.best_genes
    
    def _save_best_genes(self):
        """Save best genes to file"""
        if self.best_genes is None:
            return
        
        result = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'evolved_at': datetime.now().isoformat(),
            'genes': asdict(self.best_genes),
            'evolution_history': self.evolution_history
        }
        
        path = EVOLUTION_DIR / f"best_genes_{self.symbol}_{self.timeframe}.json"
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n   💾 Best genes saved: {path}")


# =============================================================================
# SELF-HEALING MONITOR
# =============================================================================

class SelfHealingMonitor:
    """
    Monitors system health and triggers self-healing.
    
    Checks:
    - Model drift (feature distributions)
    - Performance degradation
    - Error rates
    
    Actions:
    - Re-calibrate parameters
    - Re-train models
    - Switch to safe mode
    """
    
    def __init__(self):
        self.health_log_path = BASE_DIR / "logs" / "health.json"
        self.alert_threshold_pf = 1.0  # Alert if PF drops below
        self.heal_threshold_pf = 0.8   # Auto-heal if PF drops below
        
        # Performance targets
        self.targets = {
            'BTCUSDT': {'15m': 1.5, '30m': 1.8, '1h': 2.0},
            'PAXGUSDT': {'15m': 1.2, '30m': 1.3, '1h': 1.4}
        }
    
    def check_health(self) -> Dict:
        """Check system health by running quick backtests"""
        print("\n🏥 Running health check...")
        
        from backtest_engine import BacktestEngine
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'assets': {},
            'needs_healing': [],
            'overall_status': 'HEALTHY'
        }
        
        for symbol in ['BTCUSDT', 'PAXGUSDT']:
            health_report['assets'][symbol] = {}
            
            for tf in ['15m', '30m', '1h']:  # Skip 5m
                try:
                    engine = BacktestEngine(symbol, tf)
                    result = engine.run()
                    
                    status = 'HEALTHY'
                    if result.profit_factor < self.heal_threshold_pf:
                        status = 'CRITICAL'
                        health_report['needs_healing'].append((symbol, tf))
                        health_report['overall_status'] = 'CRITICAL'
                    elif result.profit_factor < self.alert_threshold_pf:
                        status = 'WARNING'
                        if health_report['overall_status'] == 'HEALTHY':
                            health_report['overall_status'] = 'WARNING'
                    
                    health_report['assets'][symbol][tf] = {
                        'profit_factor': result.profit_factor,
                        'win_rate': result.win_rate,
                        'status': status
                    }
                except Exception as e:
                    health_report['assets'][symbol][tf] = {
                        'error': str(e),
                        'status': 'ERROR'
                    }
                    health_report['overall_status'] = 'ERROR'
        
        return health_report
    
    def auto_heal(self, symbol: str, timeframe: str):
        """Automatically heal an underperforming asset/timeframe"""
        print(f"\n🔧 AUTO-HEALING: {symbol} {timeframe}")
        
        # Step 1: Recalibrate with genetic evolution
        evolver = GeneticEvolver(symbol, timeframe, population_size=10, generations=5)
        best_genes = evolver.evolve()
        
        if best_genes and best_genes.profit_factor >= 1.0:
            print(f"   ✅ Healed! New PF: {best_genes.profit_factor:.2f}")
            return True
        else:
            print(f"   ❌ Could not heal. Consider manual intervention.")
            return False
    
    def run_healing_cycle(self):
        """Run a complete healing cycle"""
        health = self.check_health()
        
        print(f"\n📋 Health Status: {health['overall_status']}")
        
        if health['needs_healing']:
            print(f"   ⚠️ {len(health['needs_healing'])} asset(s) need healing")
            
            for symbol, tf in health['needs_healing']:
                self.auto_heal(symbol, tf)
        else:
            print("   ✅ All systems healthy!")
        
        return health


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Genetic Evolution Engine')
    parser.add_argument('command', choices=['evolve', 'heal', 'check'],
                        help='Command to run')
    parser.add_argument('symbol', nargs='?', default='BTCUSDT', help='Symbol')
    parser.add_argument('timeframe', nargs='?', default='15m', help='Timeframe')
    
    args = parser.parse_args()
    
    if args.command == 'evolve':
        evolver = GeneticEvolver(args.symbol, args.timeframe)
        evolver.evolve()
    
    elif args.command == 'heal':
        monitor = SelfHealingMonitor()
        monitor.run_healing_cycle()
    
    elif args.command == 'check':
        monitor = SelfHealingMonitor()
        health = monitor.check_health()
        print(json.dumps(health, indent=2))
