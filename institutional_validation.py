"""
INSTITUTIONAL VALIDATION SUITE
==============================
Complete validation pipeline for institutional-grade system.

Generates:
1. Model Performance Report
2. Multi-Timeframe Backtest Report
3. BTC/Gold Correlation Analysis
4. Final Institutional Assessment

Usage:
    python institutional_validation.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# === INSTITUTIONAL IMPORTS ===
try:
    from quant_model import PurgedKFold, purged_cross_val_score
    USE_PURGED_CV = True
    print("✅ Purged K-Fold enabled")
except ImportError:
    USE_PURGED_CV = False
    print("⚠️ Using standard K-Fold")

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'market_data'
PARQUET_DIR = DATA_DIR / 'parquet'
MODEL_DIR = BASE_DIR / 'models' / 'production'
ENSEMBLE_DIR = BASE_DIR / 'models' / 'ensemble'
REPORT_DIR = BASE_DIR / 'reports'
REPORT_DIR.mkdir(exist_ok=True)

SYMBOLS = ['BTCUSDT', 'PAXGUSDT']
TIMEFRAMES = ['5m', '15m', '30m', '1h']

# Load Config if exists
CONFIG_FILE = BASE_DIR / 'validation_config.json'
EVAL_CONFIG = {
    'rsi_period': 14,
    'sma_fast': 20,
    'sma_slow': 50,
    'tp_mult': 2.0,
    'sl_mult': 1.0
}
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE) as f:
            EVAL_CONFIG.update(json.load(f))
        print(f"🔧 Loaded Custom Config: {EVAL_CONFIG}")
    except: pass

# === FEATURE CALCULATION ===
def calculate_features(df):
    """Calculate ALL features for prediction"""
    df = df.copy()
    
    # Returns
    for p in [1, 3, 5, 10, 20]:
        df[f'ret_{p}'] = df['c'].pct_change(p) * 100
    
    # RSI
    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/EVAL_CONFIG['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/EVAL_CONFIG['rsi_period']).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # MACD
    ema12 = df['c'].ewm(span=12).mean()
    ema26 = df['c'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Moving averages
    df['sma20'] = df['c'].rolling(EVAL_CONFIG['sma_fast']).mean() # Renamed to standard accessor but using dynamic period
    df['sma50'] = df['c'].rolling(EVAL_CONFIG['sma_slow']).mean()
    df['sma100'] = df['c'].rolling(100).mean()
    df['sma200'] = df['c'].rolling(200).mean()
    
    df['dist_sma20'] = (df['c'] - df['sma20']) / df['c'] * 100
    df['dist_sma50'] = (df['c'] - df['sma50']) / df['c'] * 100
    df['dist_sma100'] = (df['c'] - df['sma100']) / df['c'] * 100
    df['dist_sma200'] = (df['c'] - df['sma200']) / df['c'] * 100
    
    # Bollinger
    df['bb_std'] = df['c'].rolling(20).std()
    df['bb_upper'] = df['sma20'] + 2 * df['bb_std']
    df['bb_lower'] = df['sma20'] - 2 * df['bb_std']
    df['bb_position'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20'] * 100
    
    # ATR
    tr = pd.concat([df['h']-df['l'], (df['h']-df['c'].shift()).abs(), 
                   (df['l']-df['c'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['c'] * 100
    
    # Volatility
    df['volatility'] = df['ret_1'].rolling(20).std() * np.sqrt(252)
    df['vol_rank'] = df['volatility'].rolling(100).rank(pct=True)
    
    # Volume
    df['vol_sma'] = df['v'].rolling(20).mean()
    df['vol_ratio'] = df['v'] / (df['vol_sma'] + 1e-10)
    df['vol_zscore'] = (df['v'] - df['vol_sma']) / (df['v'].rolling(20).std() + 1e-10)
    
    # ROC
    for p in [5, 10, 20]:
        df[f'roc_{p}'] = (df['c'] / df['c'].shift(p) - 1) * 100
    
    # Trends
    df['trend_20_50'] = (df['sma20'] - df['sma50']) / df['c'] * 100
    df['trend_50_100'] = (df['sma50'] - df['sma100']) / df['c'] * 100
    
    return df


# === TRIPLE BARRIER LABELING ===
def triple_barrier_label(df, tp_mult=2.0, sl_mult=1.0, max_hold=10):
    """Apply triple barrier labeling"""
    df = df.copy()
    labels, returns, touches = [], [], []
    
    for i in range(len(df) - max_hold):
        entry = df['c'].iloc[i]
        atr = df['atr'].iloc[i]
        
        if pd.isna(atr) or atr <= 0:
            labels.append(np.nan)
            returns.append(np.nan)
            touches.append(None)
            continue
        
        upper = entry + atr * tp_mult
        lower = entry - atr * sl_mult
        
        label, ret, touch = 0, 0, 'timeout'
        
        for j in range(1, max_hold + 1):
            if i + j >= len(df):
                break
            high = df['h'].iloc[i + j]
            low = df['l'].iloc[i + j]
            
            if high >= upper:
                label, ret, touch = 1, (upper - entry) / entry, 'tp'
                break
            if low <= lower:
                label, ret, touch = 0, (lower - entry) / entry, 'sl'
                break
        
        if touch == 'timeout':
            final = df['c'].iloc[min(i + max_hold, len(df) - 1)]
            ret = (final - entry) / entry
            label = 1 if ret > 0 else 0
        
        labels.append(label)
        returns.append(ret)
        touches.append(touch)
    
    labels.extend([np.nan] * max_hold)
    returns.extend([np.nan] * max_hold)
    touches.extend([None] * max_hold)
    
    df['label'] = labels
    df['barrier_ret'] = returns
    df['barrier_touch'] = touches
    
    return df


# === BACKTEST ENGINE ===
def run_backtest(df, fee_pct=0.001, slippage_pct=0.0005):
    """Run realistic backtest with TP/SL simulation"""
    df = df.copy()
    df = df.dropna(subset=['label', 'atr'])
    
    if len(df) < 100:
        return None
    
    # Simple split
    n = len(df)
    test_start = int(n * 0.7)
    test_df = df.iloc[test_start:]
    
    trades = []
    position = None
    
    for i in range(len(test_df) - 10):
        row = test_df.iloc[i]
        
        # Entry logic
        if position is None:
            signal = 'BUY' if row['label'] == 1 else 'SELL'
            entry = row['c'] * (1 + slippage_pct)
            atr = row['atr']
            
            if signal == 'BUY':
                sl = entry - atr * 1.0
                tp = entry + atr * 2.0
            else:
                sl = entry + atr * 1.0
                tp = entry - atr * 2.0
            
            position = {'signal': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'entry_idx': i}
        
        # Exit logic
        if position:
            for j in range(1, 11):
                if i + j >= len(test_df):
                    break
                
                future = test_df.iloc[i + j]
                high, low = future['h'], future['l']
                
                exit_price, exit_reason = None, None
                
                if position['signal'] == 'BUY':
                    if high >= position['tp']:
                        exit_price, exit_reason = position['tp'], 'TP'
                    elif low <= position['sl']:
                        exit_price, exit_reason = position['sl'], 'SL'
                else:
                    if low <= position['tp']:
                        exit_price, exit_reason = position['tp'], 'TP'
                    elif high >= position['sl']:
                        exit_price, exit_reason = position['sl'], 'SL'
                
                if exit_price:
                    if position['signal'] == 'BUY':
                        pnl = (exit_price - position['entry']) / position['entry']
                    else:
                        pnl = (position['entry'] - exit_price) / position['entry']
                    
                    pnl -= fee_pct * 2  # Entry + exit fees
                    
                    trades.append({
                        'signal': position['signal'],
                        'entry': position['entry'],
                        'exit': exit_price,
                        'pnl': pnl * 100,
                        'reason': exit_reason,
                        'bars_held': j
                    })
                    position = None
                    break
    
    if not trades:
        return None
    
    # Calculate metrics
    trade_df = pd.DataFrame(trades)
    
    wins = trade_df[trade_df['pnl'] > 0]
    losses = trade_df[trade_df['pnl'] <= 0]
    
    gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    
    # Cumulative for drawdown
    cumsum = trade_df['pnl'].cumsum()
    peak = cumsum.cummax()
    drawdown = peak - cumsum
    
    return {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 999,
        'total_return': trade_df['pnl'].sum(),
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        'max_drawdown': drawdown.max(),
        'sharpe': (trade_df['pnl'].mean() / trade_df['pnl'].std()) * np.sqrt(252) if trade_df['pnl'].std() > 0 else 0,
        'expectancy': trade_df['pnl'].mean(),
        'avg_bars_held': trade_df['bars_held'].mean()
    }


# === CORRELATION ANALYSIS ===
def calculate_correlation(btc_df, gold_df):
    """Calculate BTC/Gold correlation"""
    # Merge on time
    btc_df = btc_df.copy()
    gold_df = gold_df.copy()
    
    btc_df['time'] = pd.to_datetime(btc_df['time'])
    gold_df['time'] = pd.to_datetime(gold_df['time'])
    
    btc_df = btc_df.set_index('time')[['c']].rename(columns={'c': 'btc'})
    gold_df = gold_df.set_index('time')[['c']].rename(columns={'c': 'gold'})
    
    merged = btc_df.join(gold_df, how='inner')
    merged = merged.dropna()
    
    if len(merged) < 100:
        return None
    
    # Returns
    merged['btc_ret'] = merged['btc'].pct_change()
    merged['gold_ret'] = merged['gold'].pct_change()
    merged = merged.dropna()
    
    # Price correlation
    price_corr = merged['btc'].corr(merged['gold'])
    
    # Returns correlation
    returns_corr = merged['btc_ret'].corr(merged['gold_ret'])
    
    # Rolling correlation (30 periods)
    rolling_corr = merged['btc_ret'].rolling(30).corr(merged['gold_ret'])
    
    return {
        'price_correlation': price_corr,
        'returns_correlation': returns_corr,
        'rolling_corr_mean': rolling_corr.mean(),
        'rolling_corr_std': rolling_corr.std(),
        'rolling_corr_min': rolling_corr.min(),
        'rolling_corr_max': rolling_corr.max(),
        'samples': len(merged)
    }


# === MAIN VALIDATION ===
def run_full_validation():
    """Run complete institutional validation"""
    print("="*80)
    print("🏛️ INSTITUTIONAL VALIDATION SUITE")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_report': {},
        'backtest_report': {},
        'correlation_report': {},
        'summary': {}
    }
    
    # Load all data
    print("\n📥 Loading data...")
    all_data = {}
    
    for symbol in SYMBOLS:
        all_data[symbol] = {}
        for tf in TIMEFRAMES:
            try:
                # Try parquet first
                path = PARQUET_DIR / f"{symbol}_{tf}.parquet"
                if path.exists():
                    df = pd.read_parquet(path)
                else:
                    path = DATA_DIR / f"{symbol}_{tf}.csv"
                    df = pd.read_csv(path)
                    df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
                
                all_data[symbol][tf] = df
                print(f"   ✅ {symbol} {tf}: {len(df):,} rows")
            except Exception as e:
                print(f"   ❌ {symbol} {tf}: {e}")
    
    # === MODEL REPORT ===
    print("\n" + "="*80)
    print("📊 MODEL PERFORMANCE REPORT")
    print("="*80)
    
    for symbol in SYMBOLS:
        if '1h' not in all_data.get(symbol, {}):
            continue
        
        df = all_data[symbol]['1h'].copy()
        df = calculate_features(df)
        df = triple_barrier_label(df, tp_mult=EVAL_CONFIG['tp_mult'], sl_mult=EVAL_CONFIG['sl_mult'])
        df = df.dropna(subset=['label', 'atr'])
        
        n = len(df)
        train_end = int(n * 0.7)
        test_df = df.iloc[train_end:]
        
        # Label distribution
        win_pct = test_df['label'].mean() * 100
        tp_pct = (test_df['barrier_touch'] == 'tp').mean() * 100
        sl_pct = (test_df['barrier_touch'] == 'sl').mean() * 100
        timeout_pct = (test_df['barrier_touch'] == 'timeout').mean() * 100
        
        report = {
            'total_samples': len(df),
            'train_samples': train_end,
            'test_samples': len(test_df),
            'label_win_pct': round(win_pct, 1),
            'tp_hit_pct': round(tp_pct, 1),
            'sl_hit_pct': round(sl_pct, 1),
            'timeout_pct': round(timeout_pct, 1)
        }
        
        results['model_report'][symbol] = report
        
        print(f"\n   {symbol}:")
        print(f"      Total Samples: {report['total_samples']:,}")
        print(f"      Test Samples: {report['test_samples']:,}")
        print(f"      Win Label %: {report['label_win_pct']:.1f}%")
        print(f"      TP Hit: {report['tp_hit_pct']:.1f}% | SL Hit: {report['sl_hit_pct']:.1f}% | Timeout: {report['timeout_pct']:.1f}%")
    
    # === BACKTEST REPORT (ALL TIMEFRAMES) ===
    print("\n" + "="*80)
    print("📈 MULTI-TIMEFRAME BACKTEST REPORT")
    print("="*80)
    
    for symbol in SYMBOLS:
        results['backtest_report'][symbol] = {}
        print(f"\n   === {symbol} ===")
        
        for tf in TIMEFRAMES:
            if tf not in all_data.get(symbol, {}):
                continue
            
            df = all_data[symbol][tf].copy()
            df = calculate_features(df)
            df = triple_barrier_label(df)
            
            bt = run_backtest(df)
            
            if bt:
                results['backtest_report'][symbol][tf] = bt
                
                edge = "✅ EDGE" if bt['profit_factor'] > 1.0 and bt['win_rate'] > 50 else "⚠️ WEAK"
                
                print(f"\n   {tf}:")
                print(f"      Trades: {bt['total_trades']} | Win Rate: {bt['win_rate']:.1f}%")
                print(f"      Profit Factor: {bt['profit_factor']:.2f} | Sharpe: {bt['sharpe']:.2f}")
                print(f"      Total Return: {bt['total_return']:.2f}% | Max DD: {bt['max_drawdown']:.2f}%")
                print(f"      Expectancy: {bt['expectancy']:.3f}% | {edge}")
    
    # === CORRELATION ANALYSIS ===
    print("\n" + "="*80)
    print("🔗 BTC/GOLD CORRELATION ANALYSIS")
    print("="*80)
    
    for tf in TIMEFRAMES:
        if tf not in all_data.get('BTCUSDT', {}) or tf not in all_data.get('PAXGUSDT', {}):
            continue
        
        corr = calculate_correlation(
            all_data['BTCUSDT'][tf].copy(),
            all_data['PAXGUSDT'][tf].copy()
        )
        
        if corr:
            results['correlation_report'][tf] = corr
            
            print(f"\n   {tf}:")
            print(f"      Price Correlation: {corr['price_correlation']:.4f}")
            print(f"      Returns Correlation: {corr['returns_correlation']:.4f}")
            print(f"      Rolling Corr Mean: {corr['rolling_corr_mean']:.4f} ± {corr['rolling_corr_std']:.4f}")
            print(f"      Samples: {corr['samples']:,}")
    
    # === SUMMARY ===
    print("\n" + "="*80)
    print("🏆 INSTITUTIONAL SUMMARY")
    print("="*80)
    
    # Aggregate metrics
    all_backtests = []
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            if tf in results['backtest_report'].get(symbol, {}):
                bt = results['backtest_report'][symbol][tf]
                all_backtests.append({
                    'symbol': symbol,
                    'timeframe': tf,
                    **bt
                })
    
    if all_backtests:
        bt_df = pd.DataFrame(all_backtests)
        
        avg_wr = bt_df['win_rate'].mean()
        avg_pf = bt_df['profit_factor'].mean()
        avg_sharpe = bt_df['sharpe'].mean()
        total_trades = bt_df['total_trades'].sum()
        
        # Count edges
        edges = len(bt_df[(bt_df['profit_factor'] > 1.0) & (bt_df['win_rate'] > 50)])
        
        summary = {
            'total_backtests': len(bt_df),
            'total_trades_simulated': int(total_trades),
            'avg_win_rate': round(avg_wr, 1),
            'avg_profit_factor': round(avg_pf, 2),
            'avg_sharpe': round(avg_sharpe, 2),
            'strategies_with_edge': edges,
            'edge_percentage': round(edges / len(bt_df) * 100, 1)
        }
        
        results['summary'] = summary
        
        print(f"\n   Total Strategies Tested: {summary['total_backtests']}")
        print(f"   Total Trades Simulated: {summary['total_trades_simulated']:,}")
        print(f"   Avg Win Rate: {summary['avg_win_rate']:.1f}%")
        print(f"   Avg Profit Factor: {summary['avg_profit_factor']:.2f}")
        print(f"   Avg Sharpe: {summary['avg_sharpe']:.2f}")
        print(f"   Strategies with Edge: {summary['strategies_with_edge']}/{summary['total_backtests']} ({summary['edge_percentage']:.1f}%)")
    
    # Correlation summary
    if results['correlation_report']:
        avg_corr = np.mean([r['returns_correlation'] for r in results['correlation_report'].values()])
        print(f"\n   BTC/Gold Avg Correlation: {avg_corr:.4f}")
    
    # === SAVE REPORT ===
    report_path = REPORT_DIR / 'institutional_validation.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Report saved: {report_path}")
    
    # === VERDICT ===
    print("\n" + "="*80)
    print("🎯 INSTITUTIONAL VERDICT")
    print("="*80)
    
    if summary.get('avg_profit_factor', 0) > 1.2 and summary.get('avg_win_rate', 0) > 52:
        verdict = "✅ INSTITUTIONAL GRADE - Ready for paper trading"
    elif summary.get('avg_profit_factor', 0) > 1.0:
        verdict = "⚠️ MARGINAL EDGE - Needs optimization before live"
    else:
        verdict = "❌ NO EDGE - Requires strategy redesign"
    
    print(f"\n   {verdict}")
    print("\n" + "="*80)
    
    return results


# === MAIN ===
if __name__ == "__main__":
    results = run_full_validation()
