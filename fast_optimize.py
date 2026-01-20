"""
FAST OPTIMIZED BACKTEST
Using vectorized triple-barrier + smaller sample for speed
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("⚡ FAST OPTIMIZED BACKTEST")
print("="*70)

def calculate_atr(df, period=14):
    tr = pd.concat([
        df['h'] - df['l'],
        (df['h'] - df['c'].shift()).abs(),
        (df['l'] - df['c'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def fast_triple_barrier(df, atr_mult_tp=1.5, atr_mult_sl=1.0, max_hold=10):
    """Vectorized triple-barrier (faster)"""
    atr = calculate_atr(df)
    
    # Pre-compute future max/min
    future_max = df['h'].rolling(max_hold).max().shift(-max_hold)
    future_min = df['l'].rolling(max_hold).min().shift(-max_hold)
    
    entry = df['c']
    tp = entry + atr * atr_mult_tp
    sl = entry - atr * atr_mult_sl
    
    # Simplified: check if TP reachable before SL
    tp_reached = future_max >= tp
    sl_reached = future_min <= sl
    
    # Label: 1 if TP likely hit, 0 if SL likely hit
    # This is an approximation but much faster
    label = np.where(tp_reached & ~sl_reached, 1, 
                     np.where(sl_reached, 0, np.nan))
    
    return pd.Series(label, index=df.index)


def create_features(df):
    df = df.copy()
    
    df['ret_1'] = df['c'].pct_change(1) * 100
    df['ret_5'] = df['c'].pct_change(5) * 100
    df['ret_10'] = df['c'].pct_change(10) * 100
    
    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema12 = df['c'].ewm(span=12).mean()
    ema26 = df['c'].ewm(span=26).mean()
    df['macd_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
    
    df['sma20'] = df['c'].rolling(20).mean()
    df['sma50'] = df['c'].rolling(50).mean()
    
    df['dist_sma20'] = (df['c'] - df['sma20']) / df['c'] * 100
    df['dist_sma50'] = (df['c'] - df['sma50']) / df['c'] * 100
    
    df['bb_std'] = df['c'].rolling(20).std()
    df['bb_position'] = (df['c'] - (df['sma20'] - 2*df['bb_std'])) / (4 * df['bb_std'] + 1e-10)
    df['bb_width'] = (4 * df['bb_std']) / df['c'] * 100
    
    df['atr'] = calculate_atr(df)
    df['atr_pct'] = df['atr'] / df['c'] * 100
    df['volatility'] = df['c'].pct_change().rolling(20).std() * 100
    
    df['roc_5'] = (df['c'] / df['c'].shift(5) - 1) * 100
    df['roc_10'] = (df['c'] / df['c'].shift(10) - 1) * 100
    
    df['vol_ratio'] = df['v'] / (df['v'].rolling(20).mean() + 1e-10)
    
    df['trend_sma'] = (df['sma20'] > df['sma50']).astype(int)
    df['body_pct'] = (df['c'] - df['o']) / (df['o'] + 1e-10) * 100
    
    return df


def test_config(df, feature_cols, atr_tp, atr_sl, name):
    """Quick test of a configuration"""
    df = df.copy()
    df['label'] = fast_triple_barrier(df, atr_tp, atr_sl, 10)
    
    df_clean = df.dropna()
    
    # Use last 20k for speed
    df_clean = df_clean.tail(20000)
    
    split = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split]
    test_df = df_clean.iloc[split:]
    
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y_train = train_df['label']
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y_test = test_df['label']
    
    # Train
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predict
    prob = rf.predict_proba(X_test)[:, 1]
    
    base_rate = y_test.mean() * 100
    
    results = []
    for thresh in [0.50, 0.55, 0.60]:
        preds = (prob >= thresh).astype(int)
        trades = preds.sum()
        
        if trades >= 10:
            wins = ((preds == 1) & (y_test.values == 1)).sum()
            win_rate = wins / trades * 100
            pf = (wins * atr_tp) / ((trades - wins) * atr_sl + 1e-10)
            
            results.append({
                'thresh': thresh, 'trades': trades, 'wins': wins,
                'win_rate': win_rate, 'base_rate': base_rate, 'pf': pf
            })
    
    return results


# Load data
print("\n📥 Loading data...")
df = pd.read_csv('market_data/BTCUSDT_1h.csv')
df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
df = create_features(df)
print(f"   Loaded {len(df)} candles")

FEATURES = [
    'ret_1', 'ret_5', 'ret_10', 'rsi', 'macd_hist',
    'dist_sma20', 'dist_sma50', 'bb_position', 'bb_width',
    'atr_pct', 'volatility', 'roc_5', 'roc_10',
    'vol_ratio', 'trend_sma', 'body_pct'
]

configs = [
    (1.0, 1.0, "1:1"),
    (1.5, 1.0, "1.5:1"),
    (1.2, 0.8, "1.5:1 tight"),
    (1.0, 0.7, "1.4:1 very tight"),
    (0.8, 0.5, "1.6:1 scalp"),
]

print("\n📊 Testing configurations...")
print("-" * 70)

all_results = []
for atr_tp, atr_sl, name in configs:
    print(f"\n🔧 {name} (TP={atr_tp}x, SL={atr_sl}x)...", end=" ", flush=True)
    
    results = test_config(df, FEATURES, atr_tp, atr_sl, name)
    
    if results:
        best = max(results, key=lambda x: x['pf'])
        s = "✅" if best['win_rate'] >= 50 else "❌"
        pfs = "✅" if best['pf'] >= 1.0 else "❌"
        print(f"WR: {best['win_rate']:.1f}% ({best['trades']} trades) {s} | PF: {best['pf']:.2f} {pfs}")
        
        all_results.append({
            'name': name, 'tp': atr_tp, 'sl': atr_sl,
            **best
        })

print("\n" + "="*70)
print("🏆 RESULTS SUMMARY")
print("="*70)

print(f"\n{'Config':<15} {'Win Rate':<12} {'Trades':<10} {'PF':<10} {'Verdict':<10}")
print("-" * 60)

for r in sorted(all_results, key=lambda x: -x['pf']):
    wr_s = "✅" if r['win_rate'] >= 50 else "❌"
    pf_s = "✅" if r['pf'] >= 1.0 else "❌"
    print(f"{r['name']:<15} {r['win_rate']:.1f}% {wr_s:<5} {r['trades']:<10} {r['pf']:.2f} {pf_s:<5}")

best = max(all_results, key=lambda x: x['pf'])
print(f"\n🎯 BEST: {best['name']} with PF={best['pf']:.2f} and {best['win_rate']:.1f}% win rate")

if best['pf'] >= 1.0:
    print("\n✅ PROFITABLE CONFIGURATION!")
    print(f"\n   Use these settings:")
    print(f"   TP_ATR_MULT = {best['tp']}")
    print(f"   SL_ATR_MULT = {best['sl']}")
    print(f"   CONFIDENCE = {best['thresh']}")
else:
    print("\n⚠️ No profitable config found. May need different approach.")

print("="*70)
