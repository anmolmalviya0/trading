"""
FAST TRAINING - Get Working System NOW
=======================================
Streamlined training on full data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
import joblib
import sqlite3

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'market_data'
MODEL_DIR = BASE_DIR / 'models' / 'production'
DB_PATH = BASE_DIR / 'trading.db'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("🚀 FAST TRAINING - Production Models")
print("="*70)

# Init DB
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS model_metadata (
    id INTEGER PRIMARY KEY, model_name TEXT, accuracy REAL, 
    train_samples INTEGER, trained_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()
conn.close()

# Train each symbol/TF
for symbol in ['BTCUSDT', 'PAXGUSDT']:
    for tf in ['1h']:  # Start with 1h only for speed
        print(f"\n📊 Training {symbol} {tf}...")
        
        # Load data
        file_path = DATA_DIR / f"{symbol}_{tf}.csv"
        if not file_path.exists():
            print(f"   ⚠️ File not found")
            continue
        
        df = pd.read_csv(file_path)
        df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
        print(f"   Loaded: {len(df):,} candles")
        
        # Features
        df['ret_1'] = df['c'].pct_change() * 100
        df['ret_5'] = df['c'].pct_change(5) * 100
        delta = df['c'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        df['sma20'] = df['c'].rolling(20).mean()
        df['sma50'] = df['c'].rolling(50).mean()
        df['dist_sma20'] = (df['c'] - df['sma20']) / df['c'] * 100
        df['dist_sma50'] = (df['c'] - df['sma50']) / df['c'] * 100
        tr = pd.concat([df['h']-df['l'], (df['h']-df['c'].shift()).abs(), 
                       (df['l']-df['c'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['vol'] = df['ret_1'].rolling(20).std()
        
        # Simple label: next 10 candles up or down
        df['label'] = (df['c'].shift(-10) > df['c']).astype(int)
        
        # Clean
        feature_cols = ['ret_1', 'ret_5', 'rsi', 'dist_sma20', 'dist_sma50', 'atr', 'vol']
        df_clean = df.dropna(subset=feature_cols + ['label'])
        
        print(f"   Clean samples: {len(df_clean):,}")
        
        # Split (time-based)
        split = int(len(df_clean) * 0.8)
        train = df_clean.iloc[:split]
        test = df_clean.iloc[split:]
        
        X_train = train[feature_cols]
        y_train = train['label']
        X_test = test[feature_cols]
        y_test = test['label']
        
        # Train with Pipeline
        rf_pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('model', RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42))
        ])
        
        gb_pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('model', GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42))
        ])
        
        print("   Training...")
        rf_pipe.fit(X_train, y_train)
        gb_pipe.fit(X_train, y_train)
        
        # Evaluate
        rf_pred = rf_pipe.predict(X_test)
        gb_pred = gb_pipe.predict(X_test)
        ensemble_proba = (rf_pipe.predict_proba(X_test)[:, 1] + gb_pipe.predict_proba(X_test)[:, 1]) / 2
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        acc = accuracy_score(y_test, ensemble_pred)
        print(f"   ✅ Accuracy: {acc*100:.1f}%")
        
        # Save
        model_path = MODEL_DIR / f"{symbol}_{tf}.pkl"
        joblib.dump({
            'rf_pipeline': rf_pipe,
            'gb_pipeline': gb_pipe,
            'feature_cols': feature_cols,
            'metadata': {'accuracy': acc, 'symbol': symbol, 'timeframe': tf}
        }, model_path)
        print(f"   💾 Saved: {model_path.name}")
        
        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO model_metadata (model_name, accuracy, train_samples) VALUES (?, ?, ?)',
                      (f"{symbol}_{tf}", acc, len(train)))
        conn.commit()
        conn.close()

print("\n" + "="*70)
print("✅ TRAINING COMPLETE")
print("="*70)
print(f"\n💾 Models saved to: {MODEL_DIR}")
print(f"🗄️ Database: {DB_PATH}")
print("\n🚀 Ready to start live terminal:")
print("   python3 live_terminal.py")
