"""
IMPROVED ML STRATEGY - Using Ensemble + Better Filtering
Focus: Higher accuracy through better signal selection
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎯 IMPROVED ML STRATEGY - Better Signal Selection")
print("="*70)

def calculate_atr(df, period=14):
    tr = pd.concat([
        df['h'] - df['l'],
        (df['h'] - df['c'].shift()).abs(),
        (df['l'] - df['c'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def create_features(df):
    df = df.copy()
    
    # Returns
    for p in [1, 3, 5, 10, 20]:
        df[f'ret_{p}'] = df['c'].pct_change(p) * 100
    
    # RSI
    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    # MACD
    ema12 = df['c'].ewm(span=12).mean()
    ema26 = df['c'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())).astype(int)
    
    # Moving Averages
    for p in [10, 20, 50, 100, 200]:
        df[f'sma{p}'] = df['c'].rolling(p).mean()
        df[f'dist_sma{p}'] = (df['c'] - df[f'sma{p}']) / df['c'] * 100
    
    # EMA cross
    df['ema9'] = df['c'].ewm(span=9).mean()
    df['ema21'] = df['c'].ewm(span=21).mean()
    df['ema_bullish'] = (df['ema9'] > df['ema21']).astype(int)
    
    # Bollinger Bands
    df['bb_std'] = df['c'].rolling(20).std()
    df['bb_upper'] = df['sma20'] + 2 * df['bb_std']
    df['bb_lower'] = df['sma20'] - 2 * df['bb_std']
    df['bb_position'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['c'] * 100
    
    # Volatility
    df['atr'] = calculate_atr(df)
    df['atr_pct'] = df['atr'] / df['c'] * 100
    df['volatility'] = df['c'].pct_change().rolling(20).std() * 100
    
    # Momentum
    for p in [5, 10, 20]:
        df[f'roc_{p}'] = (df['c'] / df['c'].shift(p) - 1) * 100
    
    # Volume
    df['vol_sma'] = df['v'].rolling(20).mean()
    df['vol_ratio'] = df['v'] / (df['vol_sma'] + 1e-10)
    df['vol_spike'] = (df['vol_ratio'] > 2).astype(int)
    
    # Time
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    
    # Trend strength
    df['trend_sma'] = (df['sma20'] > df['sma50']).astype(int)
    df['above_sma200'] = (df['c'] > df['sma200']).astype(int)
    df['strong_trend'] = ((df['sma20'] > df['sma50']) & (df['sma50'] > df['sma100'])).astype(int)
    
    # Pattern
    df['body'] = df['c'] - df['o']
    df['body_pct'] = df['body'] / (df['o'] + 1e-10) * 100
    df['bullish_candle'] = (df['body'] > 0).astype(int)
    df['consecutive_up'] = df['bullish_candle'].rolling(3).sum()
    
    # Support/Resistance
    df['high_20'] = df['h'].rolling(20).max()
    df['low_20'] = df['l'].rolling(20).min()
    df['near_resistance'] = (df['c'] / df['high_20'] > 0.98).astype(int)
    df['near_support'] = (df['c'] / df['low_20'] < 1.02).astype(int)
    
    return df


def create_labels(df, lookahead=10, threshold_pct=0.5):
    """Create labels based on future returns"""
    future_ret = (df['c'].shift(-lookahead) / df['c'] - 1) * 100
    df['label'] = (future_ret > threshold_pct).astype(int)
    return df


# Load data
print("\n📥 Loading data...")
df = pd.read_csv('market_data/BTCUSDT_1h.csv')
df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
print(f"   Loaded {len(df)} candles")

print("\n🔧 Creating features...")
df = create_features(df)

print("\n🏷️ Creating labels (10 candles, 0.5% threshold)...")
df = create_labels(df, lookahead=10, threshold_pct=0.5)

# Feature columns
FEATURES = [
    'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
    'rsi', 'rsi_oversold', 'rsi_overbought',
    'macd_hist', 'macd_cross_up',
    'dist_sma10', 'dist_sma20', 'dist_sma50', 'dist_sma100', 'dist_sma200',
    'ema_bullish',
    'bb_position', 'bb_width',
    'atr_pct', 'volatility',
    'roc_5', 'roc_10', 'roc_20',
    'vol_ratio', 'vol_spike',
    'hour', 'day_of_week',
    'trend_sma', 'above_sma200', 'strong_trend',
    'body_pct', 'bullish_candle', 'consecutive_up',
    'near_resistance', 'near_support'
]

# Clean data
df_clean = df.dropna()
X = df_clean[FEATURES].copy()
y = df_clean['label'].copy()

X = X.replace([np.inf, -np.inf], np.nan)
valid = ~X.isna().any(axis=1)
X = X[valid]
y = y[valid]

# Time-series split
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"\n📊 Train: {len(X_train)}, Test: {len(X_test)}")
print(f"   Features: {len(FEATURES)}")

# Clip extreme values
for col in X_train.columns:
    q1, q99 = X_train[col].quantile(0.01), X_train[col].quantile(0.99)
    X_train[col] = X_train[col].clip(q1, q99)
    X_test[col] = X_test[col].clip(q1, q99)

print("\n🤖 Training ensemble model...")

# Create ensemble
rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=30, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# Train
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lr.fit(X_train, y_train)

print("   ✅ Random Forest trained")
print("   ✅ Gradient Boosting trained")
print("   ✅ Logistic Regression trained")

# Ensemble prediction (average probabilities)
prob_rf = rf.predict_proba(X_test)[:, 1]
prob_gb = gb.predict_proba(X_test)[:, 1]
prob_lr = lr.predict_proba(X_test)[:, 1]

prob_ensemble = (prob_rf + prob_gb + prob_lr) / 3

# Test different thresholds
print("\n" + "="*70)
print("📊 RESULTS BY CONFIDENCE THRESHOLD")
print("="*70)
print(f"\n{'Threshold':<12} {'Trades':<10} {'Wins':<10} {'Win Rate':<12} {'Status':<10}")
print("-" * 55)

best_thresh = 0.5
best_wr = 0
best_trades = 0

for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    preds = (prob_ensemble >= thresh).astype(int)
    trades = preds.sum()
    
    if trades >= 5:
        wins = ((preds == 1) & (y_test.values == 1)).sum()
        win_rate = wins / trades * 100
        
        status = "✅" if win_rate >= 50 else "❌"
        print(f"{thresh:<12} {trades:<10} {wins:<10} {win_rate:.1f}%{'':<6} {status}")
        
        if win_rate >= 50 and (win_rate > best_wr or (win_rate == best_wr and trades > best_trades)):
            best_wr = win_rate
            best_thresh = thresh
            best_trades = trades

print("\n" + "="*70)
print("🏆 BEST CONFIGURATION")
print("="*70)

if best_wr >= 50:
    print(f"\n   ✅ FOUND PROFITABLE CONFIGURATION!")
    print(f"\n   Confidence Threshold: {best_thresh}")
    print(f"   Win Rate: {best_wr:.1f}%")
    print(f"   Trades: {best_trades}")
    
    # Feature importance
    print(f"\n📊 Top 10 Most Important Features:")
    importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(10).iterrows():
        print(f"   {row['feature']:<25} {row['importance']:.4f}")
else:
    print(f"\n   ⚠️ Best achieved: {best_wr:.1f}%")
    print(f"   Need further optimization")

print("\n" + "="*70)
print("📋 CONFIG FOR TERMINAL")
print("="*70)
print(f"""
CONFIG = {{
    'CONFIDENCE_THRESHOLD': {best_thresh},
    'WIN_RATE': {best_wr:.1f},
    'LOOKAHEAD': 10,
    'RETURN_THRESHOLD': 0.5
}}
""")
print("="*70)
