"""
KAGGLE-STYLE STRATEGY DISCOVERY
================================
Institutional-grade analysis with:
1. Correlation heatmaps (BTC/Gold/Features)
2. Feature importance visualization
3. Regime detection
4. Signal quality analysis
5. Strategy profitability heatmap

Based on top Kaggle crypto trading notebooks.

Usage:
    python kaggle_analysis.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'market_data'
REPORT_DIR = BASE_DIR / 'reports'
REPORT_DIR.mkdir(exist_ok=True)

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# === DATA LOADING ===
def load_data(symbol, timeframe):
    """Load and prepare data"""
    path = DATA_DIR / f"{symbol}_{timeframe}.csv"
    if not path.exists():
        return None
    
    df = pd.read_csv(path)
    df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    return df

# === FEATURE ENGINEERING ===
def create_all_features(df):
    """Create comprehensive feature set for analysis"""
    df = df.copy()
    
    # Returns
    for p in [1, 3, 5, 10, 20, 50]:
        df[f'ret_{p}'] = df['c'].pct_change(p) * 100
    
    # RSI
    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    
    # MACD
    ema12 = df['c'].ewm(span=12).mean()
    ema26 = df['c'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())).astype(int)
    df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())).astype(int)
    
    # Moving Averages
    for p in [10, 20, 50, 100, 200]:
        df[f'sma{p}'] = df['c'].rolling(p).mean()
        df[f'ema{p}'] = df['c'].ewm(span=p).mean()
        df[f'dist_sma{p}'] = (df['c'] - df[f'sma{p}']) / df['c'] * 100
    
    # Trend Strength
    df['trend_20_50'] = (df['sma20'] - df['sma50']) / df['c'] * 100
    df['trend_50_100'] = (df['sma50'] - df['sma100']) / df['c'] * 100
    df['trend_50_200'] = (df['sma50'] - df['sma200']) / df['c'] * 100
    
    # Bollinger Bands
    df['bb_mid'] = df['sma20']
    df['bb_std'] = df['c'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    df['bb_position'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(int)
    
    # ATR & Volatility
    tr = pd.concat([df['h']-df['l'], (df['h']-df['c'].shift()).abs(), (df['l']-df['c'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['c'] * 100
    df['volatility'] = df['ret_1'].rolling(20).std() * np.sqrt(252)
    df['vol_zscore'] = (df['volatility'] - df['volatility'].rolling(100).mean()) / (df['volatility'].rolling(100).std() + 1e-10)
    
    # Volume Analysis
    df['vol_sma'] = df['v'].rolling(20).mean()
    df['vol_ratio'] = df['v'] / (df['vol_sma'] + 1e-10)
    df['vol_zscore'] = (df['v'] - df['vol_sma']) / (df['v'].rolling(20).std() + 1e-10)
    df['high_volume'] = (df['vol_ratio'] > 1.5).astype(int)
    
    # Momentum
    for p in [5, 10, 20]:
        df[f'roc_{p}'] = (df['c'] / df['c'].shift(p) - 1) * 100
    
    # Stochastic
    low14 = df['l'].rolling(14).min()
    high14 = df['h'].rolling(14).max()
    df['stoch_k'] = 100 * (df['c'] - low14) / (high14 - low14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # Future returns (for analysis only)
    for p in [1, 3, 5, 10]:
        df[f'future_ret_{p}'] = df['c'].shift(-p) / df['c'] - 1
    
    return df

# === 1. CORRELATION HEATMAP ===
def plot_correlation_heatmap(df, title="Feature Correlation Heatmap"):
    """Create professional correlation heatmap"""
    feature_cols = [
        'ret_1', 'ret_5', 'ret_10', 'ret_20',
        'rsi', 'macd_hist', 
        'dist_sma20', 'dist_sma50', 'trend_20_50',
        'bb_position', 'bb_width',
        'atr_pct', 'volatility',
        'vol_ratio', 'roc_10',
        'stoch_k',
        'future_ret_1', 'future_ret_5'
    ]
    
    cols = [c for c in feature_cols if c in df.columns]
    corr_matrix = df[cols].corr()
    
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    save_path = REPORT_DIR / f"correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(save_path, dpi=150, facecolor='#0a0e14', edgecolor='none')
    plt.close()
    
    print(f"   💾 Saved: {save_path}")
    return corr_matrix

# === 2. BTC/GOLD CORRELATION ===
def analyze_btc_gold_correlation(btc_df, gold_df):
    """Analyze BTC vs Gold correlation"""
    # Merge on index
    merged = pd.DataFrame({
        'btc': btc_df['c'],
        'gold': gold_df['c']
    }).dropna()
    
    merged['btc_ret'] = merged['btc'].pct_change()
    merged['gold_ret'] = merged['gold'].pct_change()
    
    # Rolling correlation
    merged['rolling_corr_30'] = merged['btc_ret'].rolling(30).corr(merged['gold_ret'])
    merged['rolling_corr_100'] = merged['btc_ret'].rolling(100).corr(merged['gold_ret'])
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Price comparison (normalized)
    ax1 = axes[0]
    btc_norm = merged['btc'] / merged['btc'].iloc[0] * 100
    gold_norm = merged['gold'] / merged['gold'].iloc[0] * 100
    ax1.plot(btc_norm.index, btc_norm, label='BTC (normalized)', color='#f7931a', linewidth=1.5)
    ax1.plot(gold_norm.index, gold_norm, label='Gold (normalized)', color='#ffd700', linewidth=1.5)
    ax1.set_title('BTC vs Gold: Normalized Price Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Returns scatter
    ax2 = axes[1]
    ax2.scatter(merged['btc_ret']*100, merged['gold_ret']*100, alpha=0.3, s=10, c='#00ff88')
    z = np.polyfit(merged['btc_ret'].dropna(), merged['gold_ret'].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged['btc_ret'].min(), merged['btc_ret'].max(), 100)
    ax2.plot(x_line*100, p(x_line)*100, 'r--', linewidth=2, label=f'Trend: {z[0]:.3f}x')
    ax2.set_xlabel('BTC Returns (%)')
    ax2.set_ylabel('Gold Returns (%)')
    ax2.set_title('BTC vs Gold: Returns Scatter', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Rolling correlation
    ax3 = axes[2]
    ax3.plot(merged.index, merged['rolling_corr_30'], label='30-period', color='#00ff88', alpha=0.7)
    ax3.plot(merged.index, merged['rolling_corr_100'], label='100-period', color='#3b82f6', linewidth=2)
    ax3.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Correlation')
    ax3.set_title('BTC/Gold Rolling Correlation', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = REPORT_DIR / f"btc_gold_correlation_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(save_path, dpi=150, facecolor='#0a0e14', edgecolor='none')
    plt.close()
    
    # Stats
    overall_corr = merged['btc_ret'].corr(merged['gold_ret'])
    print(f"   📊 BTC/Gold Price Correlation: {merged['btc'].corr(merged['gold']):.4f}")
    print(f"   📊 BTC/Gold Returns Correlation: {overall_corr:.4f}")
    print(f"   💾 Saved: {save_path}")
    
    return overall_corr

# === 3. FEATURE IMPORTANCE ===
def analyze_feature_importance(df):
    """Analyze which features predict future returns"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import RobustScaler
    
    feature_cols = [
        'ret_1', 'ret_5', 'ret_10', 'ret_20',
        'rsi', 'macd_hist',
        'dist_sma20', 'dist_sma50', 'trend_20_50',
        'bb_position', 'bb_width', 'atr_pct',
        'volatility', 'vol_ratio', 'roc_10', 'stoch_k'
    ]
    
    cols = [c for c in feature_cols if c in df.columns]
    
    # Create target: profitable trade (5-period return > 0)
    df = df.copy()
    df['target'] = (df['future_ret_5'] > 0).astype(int)
    
    df_clean = df[cols + ['target']].dropna()
    
    X = df_clean[cols]
    y = df_clean['target']
    
    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train RF
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))
    plt.barh(importance['feature'], importance['importance'], color=colors)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Feature Importance for Predicting Profitable Trades', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, axis='x')
    
    for i, (feat, imp) in enumerate(zip(importance['feature'], importance['importance'])):
        plt.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    save_path = REPORT_DIR / f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(save_path, dpi=150, facecolor='#0a0e14', edgecolor='none')
    plt.close()
    
    print(f"   🏆 Top 5 Features:")
    for _, row in importance.tail(5).iloc[::-1].iterrows():
        print(f"      - {row['feature']}: {row['importance']:.4f}")
    print(f"   💾 Saved: {save_path}")
    
    return importance

# === 4. STRATEGY PROFITABILITY HEATMAP ===
def strategy_profitability_heatmap(df):
    """Analyze which RSI/Trend combinations are profitable"""
    df = df.copy()
    
    # Create bins
    df['rsi_bin'] = pd.cut(df['rsi'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 100], 
                          labels=['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '>80'])
    df['trend_bin'] = pd.cut(df['trend_20_50'], bins=[-100, -2, -1, 0, 1, 2, 100],
                            labels=['Strong Down', 'Down', 'Slight Down', 'Slight Up', 'Up', 'Strong Up'])
    
    # Calculate average future return for each combo
    heatmap_data = df.groupby(['rsi_bin', 'trend_bin'])['future_ret_5'].mean() * 100
    heatmap_data = heatmap_data.unstack()
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Avg 5-Period Return (%)'},
        linewidths=0.5
    )
    plt.title('Strategy Profitability: RSI vs Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Trend (SMA20 vs SMA50)')
    plt.ylabel('RSI Level')
    
    plt.tight_layout()
    save_path = REPORT_DIR / f"strategy_heatmap_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(save_path, dpi=150, facecolor='#0a0e14', edgecolor='none')
    plt.close()
    
    print(f"   💾 Saved: {save_path}")
    
    return heatmap_data

# === 5. REGIME DETECTION ===
def analyze_regimes(df):
    """Identify market regimes and their profitability"""
    df = df.copy()
    
    # Define regimes
    df['regime'] = 'Neutral'
    df.loc[(df['trend_20_50'] > 1) & (df['volatility'] < df['volatility'].quantile(0.7)), 'regime'] = 'Bull Trend'
    df.loc[(df['trend_20_50'] < -1) & (df['volatility'] < df['volatility'].quantile(0.7)), 'regime'] = 'Bear Trend'
    df.loc[df['volatility'] > df['volatility'].quantile(0.8), 'regime'] = 'High Volatility'
    df.loc[df['bb_squeeze'] == 1, 'regime'] = 'Squeeze'
    
    # Regime stats
    regime_stats = df.groupby('regime').agg({
        'future_ret_5': ['mean', 'std', 'count'],
        'volatility': 'mean'
    }).round(4)
    
    regime_stats.columns = ['Avg Return', 'Std Dev', 'Count', 'Avg Volatility']
    regime_stats['Sharpe'] = regime_stats['Avg Return'] / (regime_stats['Std Dev'] + 1e-10) * np.sqrt(252/5)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Regime distribution
    ax1 = axes[0]
    regime_counts = df['regime'].value_counts()
    colors = ['#00ff88', '#ff4444', '#ffaa00', '#3b82f6', '#9ca3af']
    ax1.pie(regime_counts, labels=regime_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Market Regime Distribution', fontsize=14, fontweight='bold')
    
    # Regime returns
    ax2 = axes[1]
    regime_returns = df.groupby('regime')['future_ret_5'].mean() * 100
    bars = ax2.bar(regime_returns.index, regime_returns.values, 
                   color=['#00ff88' if x > 0 else '#ff4444' for x in regime_returns.values])
    ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Average 5-Period Return (%)')
    ax2.set_title('Average Return by Regime', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, regime_returns.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = REPORT_DIR / f"regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(save_path, dpi=150, facecolor='#0a0e14', edgecolor='none')
    plt.close()
    
    print(f"   📊 Regime Statistics:")
    print(regime_stats.to_string())
    print(f"   💾 Saved: {save_path}")
    
    return regime_stats

# === MAIN ===
def run_kaggle_analysis():
    """Run complete Kaggle-style analysis"""
    print("="*70)
    print("📊 KAGGLE-STYLE STRATEGY DISCOVERY")
    print("="*70)
    
    # Load data
    print("\n📥 Loading data...")
    btc_1h = load_data('BTCUSDT', '1h')
    gold_1h = load_data('PAXGUSDT', '1h')
    
    if btc_1h is None or gold_1h is None:
        print("❌ Could not load data")
        return
    
    print(f"   BTC: {len(btc_1h):,} candles")
    print(f"   Gold: {len(gold_1h):,} candles")
    
    # Create features
    print("\n🔧 Creating features...")
    btc_feat = create_all_features(btc_1h)
    gold_feat = create_all_features(gold_1h)
    
    # 1. Correlation Heatmap
    print("\n1️⃣ CORRELATION HEATMAP")
    corr_btc = plot_correlation_heatmap(btc_feat, "BTC Feature Correlation Heatmap")
    
    # Find strong correlations with future returns
    if 'future_ret_5' in corr_btc.columns:
        future_corr = corr_btc['future_ret_5'].drop(['future_ret_1', 'future_ret_5'], errors='ignore').sort_values()
        print(f"   📈 Features most correlated with future returns:")
        for feat, corr in future_corr.tail(3).items():
            print(f"      + {feat}: {corr:.4f}")
        for feat, corr in future_corr.head(3).items():
            print(f"      - {feat}: {corr:.4f}")
    
    # 2. BTC/Gold Correlation
    print("\n2️⃣ BTC/GOLD CORRELATION ANALYSIS")
    btc_gold_corr = analyze_btc_gold_correlation(btc_1h, gold_1h)
    
    # 3. Feature Importance
    print("\n3️⃣ FEATURE IMPORTANCE")
    importance = analyze_feature_importance(btc_feat)
    
    # 4. Strategy Heatmap
    print("\n4️⃣ STRATEGY PROFITABILITY HEATMAP")
    strategy_heatmap = strategy_profitability_heatmap(btc_feat)
    
    # 5. Regime Analysis
    print("\n5️⃣ REGIME DETECTION")
    regimes = analyze_regimes(btc_feat)
    
    # Summary
    print("\n" + "="*70)
    print("📋 KEY FINDINGS")
    print("="*70)
    
    # Best strategy from heatmap
    max_ret = strategy_heatmap.max().max()
    max_idx = strategy_heatmap.stack().idxmax()
    print(f"\n   🏆 Best RSI/Trend Combo: RSI {max_idx[0]} + {max_idx[1]} Trend")
    print(f"      Average Return: {max_ret:.2f}%")
    
    # Best regime
    best_regime = regimes['Avg Return'].idxmax()
    print(f"\n   🏆 Best Regime: {best_regime}")
    print(f"      Average Return: {regimes.loc[best_regime, 'Avg Return']*100:.2f}%")
    
    # BTC/Gold insight
    if abs(btc_gold_corr) < 0.3:
        print(f"\n   💡 BTC/Gold Insight: LOW correlation ({btc_gold_corr:.4f})")
        print(f"      → Good for diversification!")
    else:
        print(f"\n   💡 BTC/Gold Insight: MODERATE correlation ({btc_gold_corr:.4f})")
    
    print(f"\n📁 Reports saved to: {REPORT_DIR}")
    print("="*70)

if __name__ == "__main__":
    run_kaggle_analysis()
