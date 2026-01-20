
import pandas as pd
import numpy as np
from core.signals import SignalEngine
from core.features import add_features

# Create sample data
dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
df = pd.DataFrame({
    'timestamp': dates,
    'open': np.random.randn(100).cumsum() + 100,
    'high': np.random.randn(100).cumsum() + 102,
    'low': np.random.randn(100).cumsum() + 98,
    'close': np.random.randn(100).cumsum() + 100,
    'volume': np.random.randint(100, 1000, 100)
}, index=dates)

# Fix high/low
df['high'] = df[['open', 'close', 'high']].max(axis=1)
df['low'] = df[['open', 'close', 'low']].min(axis=1)

print("Adding features...")
df = add_features(df)
print("Columns:", df.columns.tolist())
print("\nSample Data (Last 5):")
print(df[['close', 'vwap', 'vwap_dev', 'ofi', 'vol_ratio']].tail())

engine = SignalEngine()
print("\nGenerating signal...")
sig = engine.generate_signal(df, 'TEST', '1h')
print("Signal Result:", sig)

if sig:
    print("Reasons:", sig['reasons'])
    print("Scores:", sig['score'])
else:
    print("No signal generated.")
