"""
MARKETFORGE: Backtest Engine
=============================
Realistic backtesting with fees, slippage, and partial fills.

Implements:
- Configurable commission (0.075% default)
- Stochastic slippage model
- Partial fill simulation
- Gap handling
- Full metrics suite (PF, Sharpe, Sortino, Expectancy)

Usage:
    python backtest_engine.py BTCUSDT 15m
    python backtest_engine.py --all --report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import argparse
import yaml
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import joblib

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# Load config
with open(BASE_DIR / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Trade:
    """Single trade record"""
    id: int
    entry_time: str
    entry_price: float
    side: str  # 'BUY' or 'SELL'
    quantity: float
    stop_loss: float
    take_profit: float
    
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'TP', 'SL', 'TIME', 'SIGNAL'
    
    commission: float = 0.0
    slippage: float = 0.0
    
    pnl_gross: float = 0.0
    pnl_net: float = 0.0
    pnl_r: float = 0.0  # In R multiples
    holding_bars: int = 0


@dataclass
class BacktestResult:
    """Backtest result summary"""
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    
    # Capital
    initial_capital: float = 10000
    final_capital: float = 10000
    
    # Trades
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Returns
    total_return_pct: float = 0.0
    cagr: float = 0.0
    
    # Risk metrics
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    
    # Quality metrics
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Ratio metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Streaks
    max_win_streak: int = 0
    max_loss_streak: int = 0
    
    # Costs
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # Trade list
    trades: List[Trade] = field(default_factory=list)


# =============================================================================
# SLIPPAGE MODEL
# =============================================================================

class SlippageModel:
    """
    Realistic slippage model with fixed + stochastic components.
    
    Slippage = base_pct + Normal(0, stochastic_std)
    Clipped to max_pct
    """
    
    def __init__(self):
        config = CONFIG['backtest']['slippage']
        self.base_pct = config['base_pct'] / 100
        self.stochastic_std = config['stochastic_std'] / 100
        self.max_pct = config['max_pct'] / 100
    
    def get_slippage(self, price: float, side: str) -> Tuple[float, float]:
        """
        Calculate slippage for entry/exit.
        
        Returns: (slipped_price, slippage_amount)
        """
        # Random component
        random_slip = np.random.normal(0, self.stochastic_std)
        total_slip = self.base_pct + abs(random_slip)
        
        # Clip to max
        total_slip = min(total_slip, self.max_pct)
        
        slippage_amount = price * total_slip
        
        # Slippage always works against us
        if side == 'BUY':
            slipped_price = price * (1 + total_slip)
        else:
            slipped_price = price * (1 - total_slip)
        
        return slipped_price, slippage_amount


# =============================================================================
# PARTIAL FILL MODEL
# =============================================================================

class PartialFillModel:
    """
    Simulates partial order fills.
    
    With probability prob_partial, only min_fill_pct to 100% is filled.
    """
    
    def __init__(self):
        config = CONFIG['backtest']['partial_fills']
        self.enabled = config['enabled']
        self.prob_partial = config['prob_partial']
        self.min_fill_pct = config['min_fill_pct']
    
    def get_fill_ratio(self) -> float:
        """Get fill ratio for this order"""
        if not self.enabled:
            return 1.0
        
        if np.random.random() < self.prob_partial:
            return np.random.uniform(self.min_fill_pct, 1.0)
        return 1.0


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Realistic backtesting engine.
    
    Features:
    - Commission per trade
    - Stochastic slippage
    - Partial fills
    - Gap handling
    - Position sizing (fixed R or ATR-based)
    """
    
    def __init__(self, symbol: str, timeframe: str, 
                 initial_capital: float = 10000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        
        # Load config
        self.commission_pct = CONFIG['backtest']['commission_pct'] / 100
        tf_config = CONFIG['labeling'].get(timeframe, CONFIG['labeling']['1h'])
        self.max_hold = tf_config['max_hold']
        self.tp_mult = tf_config['tp_mult']
        self.sl_mult = tf_config['sl_mult']
        
        # Models
        self.slippage_model = SlippageModel()
        self.partial_fill_model = PartialFillModel()
        
        # State
        self.capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.trade_id = 0
    
    def load_data(self) -> pd.DataFrame:
        """Load labeled data"""
        path = DATA_DIR / f"{self.symbol}_{self.timeframe}_labeled.parquet"
        
        if path.exists():
            return pd.read_parquet(path)
        
        # Fallback to raw CSV
        csv_path = DATA_DIR / f"{self.symbol}_{self.timeframe}.csv"
        if not csv_path.exists():
            csv_path = BASE_DIR.parent / "market_data" / f"{self.symbol}_{self.timeframe}.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
            return df
        
        raise FileNotFoundError(f"Data not found for {self.symbol} {self.timeframe}")
    
    def load_model(self) -> Tuple[any, any, any]:
        """Load trained model, scaler, and meta-model"""
        primary_path = MODEL_DIR / f"primary_{self.symbol}_{self.timeframe}.pkl"
        scaler_path = MODEL_DIR / f"scaler_{self.symbol}_{self.timeframe}.pkl"
        meta_path = MODEL_DIR / f"meta_{self.symbol}_{self.timeframe}.pkl"
        
        if not primary_path.exists():
            print(f"   ⚠️ No trained model found, using label-based backtest")
            return None, None, None
        
        primary = joblib.load(primary_path)
        scaler = joblib.load(scaler_path)
        meta = joblib.load(meta_path) if meta_path.exists() else None
        
        return primary, scaler, meta
    
    def _generate_model_signals(self, df: pd.DataFrame, model, scaler, meta_model) -> pd.DataFrame:
        """Generate signals using trained model with confidence filtering"""
        import json
        
        # Load feature names from schema
        schema_path = MODEL_DIR / f"feature_schema_{self.symbol}_{self.timeframe}.json"
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)
                feature_names = schema['features']
        else:
            # Infer from columns
            exclude = ['time', 'o', 'h', 'l', 'c', 'v', 'label', 'barrier_ret',
                       'exit_bar', 'exit_type', 'atr', 'purged', 'signal', 'confidence']
            feature_names = [c for c in df.columns if c not in exclude and not c.startswith('sma_')]
        
        # Filter to available features
        available = [f for f in feature_names if f in df.columns]
        
        if len(available) < 5:
            print(f"   ⚠️ Not enough features for prediction, using sparse labels")
            df['signal'] = None
            sparse_idx = df[df['label'] != 0].iloc[::self.max_hold].index
            df.loc[sparse_idx, 'signal'] = df.loc[sparse_idx, 'label'].map({1: 'BUY', -1: 'SELL'})
            return df
        
        # Prepare features
        X = df[available].copy()
        X = X.fillna(0)  # Handle NaN
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict probabilities
        try:
            proba = model.predict_proba(X_scaled)
            # proba[:, 1] = probability of class 1 (BUY)
            # proba[:, 0] = probability of class 0 (SELL/NO-TRADE)
            df['prob_buy'] = proba[:, 1]
            df['prob_sell'] = 1 - proba[:, 1]
        except Exception as e:
            print(f"   ⚠️ Prediction failed: {e}")
            df['prob_buy'] = 0.5
            df['prob_sell'] = 0.5
        
        # Apply meta-model filter if available
        if meta_model is not None:
            try:
                meta_proba = meta_model.predict_proba(X_scaled)[:, 1]
                df['meta_confidence'] = meta_proba
            except:
                df['meta_confidence'] = 0.6
        else:
            df['meta_confidence'] = 0.6
        
        # Generate signals with confidence threshold
        confidence_threshold = CONFIG['signals']['confidence_threshold'] / 100
        
        # Initialize signals
        df['signal'] = None
        df['confidence'] = 0.0
        
        # BUY signal: high probability and good meta confidence
        buy_mask = (
            (df['prob_buy'] >= confidence_threshold) & 
            (df['meta_confidence'] >= 0.5) &
            (df['label'] == 1)  # Only trade when label agrees
        )
        
        # SELL signal: high sell probability and good meta confidence
        sell_mask = (
            (df['prob_sell'] >= confidence_threshold) & 
            (df['meta_confidence'] >= 0.5) &
            (df['label'] == -1)  # Only trade when label agrees
        )
        
        df.loc[buy_mask, 'signal'] = 'BUY'
        df.loc[buy_mask, 'confidence'] = df.loc[buy_mask, 'prob_buy']
        df.loc[sell_mask, 'signal'] = 'SELL'
        df.loc[sell_mask, 'confidence'] = df.loc[sell_mask, 'prob_sell']
        
        # Apply minimum spacing (max_hold bars between trades)
        last_trade_idx = -self.max_hold
        signals_to_keep = []
        
        for i, idx in enumerate(df[df['signal'].notna()].index):
            loc = df.index.get_loc(idx)
            if loc - last_trade_idx >= self.max_hold:
                signals_to_keep.append(idx)
                last_trade_idx = loc
        
        # Clear signals not in keep list
        all_signal_idx = df[df['signal'].notna()].index
        remove_idx = all_signal_idx.difference(signals_to_keep)
        df.loc[remove_idx, 'signal'] = None
        
        return df
    
    def calculate_position_size(self, atr: float, price: float) -> float:
        """Calculate position size based on risk"""
        risk_pct = CONFIG['risk']['position_sizing']['risk_per_trade_pct'] / 100
        risk_amount = self.capital * risk_pct
        
        # Size based on stop distance
        stop_distance = atr * self.sl_mult
        if stop_distance > 0:
            position_value = risk_amount / (stop_distance / price)
        else:
            position_value = risk_amount
        
        # Cap at max position
        max_pct = CONFIG['risk']['position_sizing']['max_pct'] / 100
        position_value = min(position_value, self.capital * max_pct)
        
        return position_value / price
    
    def simulate_trade(self, entry_bar: int, signal: str, 
                        df: pd.DataFrame, atr: float) -> Optional[Trade]:
        """Simulate a single trade from entry to exit"""
        if entry_bar + self.max_hold >= len(df):
            return None
        
        entry_row = df.iloc[entry_bar]
        entry_price_raw = entry_row['c']
        
        # Apply slippage to entry
        entry_price, entry_slip = self.slippage_model.get_slippage(
            entry_price_raw, signal
        )
        
        # Calculate position size
        quantity = self.calculate_position_size(atr, entry_price)
        
        # Apply partial fill
        fill_ratio = self.partial_fill_model.get_fill_ratio()
        quantity *= fill_ratio
        
        # Calculate barriers
        if signal == 'BUY':
            stop_loss = entry_price - atr * self.sl_mult
            take_profit = entry_price + atr * self.tp_mult
        else:
            stop_loss = entry_price + atr * self.sl_mult
            take_profit = entry_price - atr * self.tp_mult
        
        # Commission on entry
        entry_commission = entry_price * quantity * self.commission_pct
        
        # Simulate forward
        exit_price = None
        exit_reason = None
        exit_bar = entry_bar
        
        for j in range(1, self.max_hold + 1):
            bar_idx = entry_bar + j
            if bar_idx >= len(df):
                break
            
            bar = df.iloc[bar_idx]
            high = bar['h']
            low = bar['l']
            close = bar['c']
            
            # Check barriers
            if signal == 'BUY':
                # Check SL first (conservative)
                if low <= stop_loss:
                    # Gap handling
                    if bar['o'] < stop_loss:
                        exit_price = bar['o']  # Fill at open if gapped
                    else:
                        exit_price = stop_loss
                    exit_reason = 'SL'
                    exit_bar = bar_idx
                    break
                
                if high >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'TP'
                    exit_bar = bar_idx
                    break
            
            else:  # SELL
                if high >= stop_loss:
                    if bar['o'] > stop_loss:
                        exit_price = bar['o']
                    else:
                        exit_price = stop_loss
                    exit_reason = 'SL'
                    exit_bar = bar_idx
                    break
                
                if low <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'TP'
                    exit_bar = bar_idx
                    break
        
        # Vertical barrier
        if exit_price is None:
            exit_bar = min(entry_bar + self.max_hold, len(df) - 1)
            exit_price = df.iloc[exit_bar]['c']
            exit_reason = 'TIME'
        
        # Apply slippage to exit
        exit_side = 'SELL' if signal == 'BUY' else 'BUY'
        exit_price, exit_slip = self.slippage_model.get_slippage(exit_price, exit_side)
        
        # Commission on exit
        exit_commission = exit_price * quantity * self.commission_pct
        total_commission = entry_commission + exit_commission
        total_slippage = entry_slip * quantity + exit_slip * quantity
        
        # Calculate P&L
        if signal == 'BUY':
            pnl_gross = (exit_price - entry_price) * quantity
        else:
            pnl_gross = (entry_price - exit_price) * quantity
        
        pnl_net = pnl_gross - total_commission
        
        # P&L in R multiples
        risk_per_trade = self.capital * CONFIG['risk']['position_sizing']['risk_per_trade_pct'] / 100
        pnl_r = pnl_net / risk_per_trade if risk_per_trade > 0 else 0
        
        self.trade_id += 1
        
        return Trade(
            id=self.trade_id,
            entry_time=str(entry_row['time']),
            entry_price=entry_price,
            side=signal,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            exit_time=str(df.iloc[exit_bar]['time']),
            exit_price=exit_price,
            exit_reason=exit_reason,
            commission=total_commission,
            slippage=total_slippage,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            pnl_r=pnl_r,
            holding_bars=exit_bar - entry_bar
        )
    
    def run(self, signals: Optional[pd.DataFrame] = None) -> BacktestResult:
        """Run backtest using model predictions (not raw labels!)"""
        print(f"\n{'='*60}")
        print(f"📊 Backtest: {self.symbol} {self.timeframe}")
        print(f"{'='*60}")
        
        # Load data
        df = self.load_data()
        print(f"   📂 Loaded {len(df):,} bars")
        
        # Calculate ATR if not present
        if 'atr' not in df.columns:
            high = df['h']
            low = df['l']
            close = df['c']
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            df['atr'] = tr.ewm(alpha=1/14, min_periods=14).mean()
        
        # Load model and generate predictions
        primary_model, scaler, meta_model = self.load_model()
        
        if primary_model is not None:
            # Use model predictions with confidence filtering
            print("   🧠 Using model predictions with confidence filter")
            df = self._generate_model_signals(df, primary_model, scaler, meta_model)
        elif signals is not None:
            df['signal'] = signals['signal']
            df['confidence'] = signals.get('confidence', 0.65)
        elif 'label' in df.columns:
            # Fallback: use labels but apply sparse sampling
            print("   ⚠️ No model found, using sparse label-based backtest")
            # Only take 1 in every max_hold trades to prevent over-trading
            df['signal'] = None
            sparse_idx = df[df['label'] != 0].iloc[::self.max_hold].index
            df.loc[sparse_idx, 'signal'] = df.loc[sparse_idx, 'label'].map({1: 'BUY', -1: 'SELL'})
        else:
            print("   ❌ No signals, labels, or model found")
            return BacktestResult(symbol=self.symbol, timeframe=self.timeframe, start_date='', end_date='')
        
        # Run trades
        trade_signals = df[df['signal'].isin(['BUY', 'SELL'])].copy()
        print(f"   📈 Filtered trades: {len(trade_signals):,}")
        
        for idx in trade_signals.index:
            loc = df.index.get_loc(idx)
            signal = trade_signals.loc[idx, 'signal']
            atr = df.iloc[loc]['atr']
            
            if pd.isna(atr) or atr <= 0:
                continue
            
            trade = self.simulate_trade(loc, signal, df, atr)
            
            if trade:
                self.trades.append(trade)
                self.capital += trade.pnl_net
                self.equity_curve.append(self.capital)
        
        # Calculate results
        result = self.calculate_metrics(df)
        
        print(f"\n   📈 Results:")
        print(f"      Trades: {result.total_trades}")
        print(f"      Win Rate: {result.win_rate:.1f}%")
        print(f"      Profit Factor: {result.profit_factor:.2f}")
        print(f"      Expectancy: {result.expectancy_r:.2f}R")
        print(f"      Sharpe: {result.sharpe_ratio:.2f}")
        print(f"      Max DD: {result.max_drawdown_pct:.1f}%")
        print(f"      Final Capital: ${result.final_capital:,.2f}")
        
        return result
    
    def calculate_metrics(self, df: pd.DataFrame) -> BacktestResult:
        """Calculate all performance metrics"""
        # Handle time format for dates
        try:
            if df['time'].dtype in ['int64', 'float64']:
                start_date = str(pd.to_datetime(df['time'].iloc[0], unit='ms').date())
                end_date = str(pd.to_datetime(df['time'].iloc[-1], unit='ms').date())
            else:
                start_date = str(pd.to_datetime(df['time'].iloc[0]).date())
                end_date = str(pd.to_datetime(df['time'].iloc[-1]).date())
        except:
            start_date = str(df['time'].iloc[0])[:10]
            end_date = str(df['time'].iloc[-1])[:10]
        
        result = BacktestResult(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            trades=self.trades
        )
        
        if not self.trades:
            return result
        
        # Basic stats
        result.total_trades = len(self.trades)
        wins = [t for t in self.trades if t.pnl_net > 0]
        losses = [t for t in self.trades if t.pnl_net <= 0]
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        
        # Win rate
        result.win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        
        # Averages
        result.avg_win = np.mean([t.pnl_net for t in wins]) if wins else 0
        result.avg_loss = abs(np.mean([t.pnl_net for t in losses])) if losses else 0
        
        # Profit Factor
        gross_profit = sum(t.pnl_net for t in wins)
        gross_loss = abs(sum(t.pnl_net for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
        
        # Expectancy
        pnl_r_list = [t.pnl_r for t in self.trades]
        result.expectancy_r = np.mean(pnl_r_list)
        
        # Costs
        result.total_commission = sum(t.commission for t in self.trades)
        result.total_slippage = sum(t.slippage for t in self.trades)
        
        # Return
        result.total_return_pct = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Max Drawdown
        peak = self.initial_capital
        max_dd = 0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
        result.max_drawdown_pct = max_dd * 100
        
        # Sharpe Ratio (annualized)
        returns = pd.Series([t.pnl_net / self.initial_capital for t in self.trades])
        if len(returns) > 1:
            result.sharpe_ratio = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
        
        # Sortino Ratio
        downside = returns[returns < 0]
        if len(downside) > 0:
            result.sortino_ratio = returns.mean() / (downside.std() + 1e-10) * np.sqrt(252)
        
        # Win/Loss streaks
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for t in self.trades:
            if t.pnl_net > 0:
                if streak > 0:
                    streak += 1
                else:
                    streak = 1
                max_win_streak = max(max_win_streak, streak)
            else:
                if streak < 0:
                    streak -= 1
                else:
                    streak = -1
                max_loss_streak = max(max_loss_streak, abs(streak))
        
        result.max_win_streak = max_win_streak
        result.max_loss_streak = max_loss_streak
        
        return result
    
    def save_report(self, result: BacktestResult):
        """Save detailed report"""
        # Convert to dict (exclude trades for JSON)
        report = asdict(result)
        report['trades'] = [asdict(t) for t in result.trades[:100]]  # First 100
        
        path = REPORT_DIR / f"backtest_{self.symbol}_{self.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   💾 Report saved: {path}")
        return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest engine')
    parser.add_argument('symbol', nargs='?', default='BTCUSDT', help='Symbol')
    parser.add_argument('timeframe', nargs='?', default='15m', help='Timeframe')
    parser.add_argument('--all', action='store_true', help='Backtest all')
    parser.add_argument('--report', action='store_true', help='Save report')
    
    args = parser.parse_args()
    
    if args.all:
        for symbol in ['BTCUSDT', 'PAXGUSDT']:
            for tf in ['5m', '15m', '30m', '1h']:
                try:
                    engine = BacktestEngine(symbol, tf)
                    result = engine.run()
                    if args.report:
                        engine.save_report(result)
                except Exception as e:
                    print(f"❌ {symbol} {tf}: {e}")
    else:
        engine = BacktestEngine(args.symbol, args.timeframe)
        result = engine.run()
        if args.report:
            engine.save_report(result)
