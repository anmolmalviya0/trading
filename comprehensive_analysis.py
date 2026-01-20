"""
COMPREHENSIVE SYSTEM ANALYSIS
==============================
Honest assessment of "The Ultimate Machine"

This script:
1. Tests ALL components for 600 seconds
2. Measures prediction accuracy across timeframes
3. Generates correlation heatmaps
4. Validates alternative data feeds
5. Logs all errors for debugging

Usage:
    python comprehensive_analysis.py
"""
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import traceback
import aiohttp
import ssl
import certifi

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'market_data'
REPORT_DIR = BASE_DIR / 'reports'
REPORT_DIR.mkdir(exist_ok=True)

TEST_DURATION = 600  # 10 minutes
SYMBOLS = ['BTCUSDT', 'PAXGUSDT']
TIMEFRAMES = ['5m', '15m', '30m', '1h']


# =============================================================================
# COMPONENT TESTS
# =============================================================================

class ComponentTester:
    """Test each component individually"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        
    async def test_binance_api(self, session):
        """Test Binance API connectivity"""
        print("\n📡 Testing Binance API...")
        try:
            url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            start = time.time()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                latency = (time.time() - start) * 1000
                if r.status == 200:
                    data = await r.json()
                    self.results['binance_api'] = {
                        'status': 'OK',
                        'latency_ms': round(latency, 2),
                        'price': float(data['price'])
                    }
                    print(f"   ✅ Connected | Latency: {latency:.0f}ms | BTC: ${data['price']}")
                    return True
        except Exception as e:
            self.errors.append(('binance_api', str(e)))
            self.results['binance_api'] = {'status': 'ERROR', 'error': str(e)}
            print(f"   ❌ Failed: {e}")
        return False
    
    async def test_order_book(self, session):
        """Test Order Book Imbalance feed"""
        print("\n📊 Testing Order Book Feed...")
        try:
            from order_book_feed import AdvancedMarketFeed
            feed = AdvancedMarketFeed(SYMBOLS)
            await feed.start()
            await feed.update()
            
            results = {}
            for sym in SYMBOLS:
                signal = feed.get_combined_signal(sym)
                results[sym] = {
                    'imbalance': signal['imbalance'],
                    'funding': signal['funding'],
                    'ob_bias': signal['ob_bias'],
                    'fr_bias': signal['fr_bias']
                }
                print(f"   ✅ {sym}: Imbalance={signal['imbalance']:.3f}, Bias={signal['ob_bias']}")
            
            await feed.stop()
            self.results['order_book'] = {'status': 'OK', 'data': results}
            return True
        except Exception as e:
            self.errors.append(('order_book', str(e), traceback.format_exc()))
            self.results['order_book'] = {'status': 'ERROR', 'error': str(e)}
            print(f"   ❌ Failed: {e}")
        return False
    
    async def test_alternative_data(self, session):
        """Test Alternative Data feeds"""
        print("\n📰 Testing Alternative Data Feed...")
        try:
            from alternative_data import AlternativeDataFeed
            feed = AlternativeDataFeed()
            await feed.start()
            await feed.update()
            
            results = {}
            for sym in SYMBOLS:
                alpha = feed.get_combined_alpha(sym)
                results[sym] = {
                    'alpha_score': alpha['alpha_score'],
                    'signal': alpha['signal'],
                    'confidence': alpha['confidence']
                }
                print(f"   ✅ {sym}: Alpha={alpha['alpha_score']:.3f}, Signal={alpha['signal']}")
            
            # Get headlines
            headlines = feed.news.get_headlines()[:3]
            for h in headlines:
                print(f"   📰 {h['title'][:50]}...")
            
            await feed.stop()
            self.results['alternative_data'] = {'status': 'OK', 'data': results}
            return True
        except Exception as e:
            self.errors.append(('alternative_data', str(e), traceback.format_exc()))
            self.results['alternative_data'] = {'status': 'ERROR', 'error': str(e)}
            print(f"   ❌ Failed: {e}")
        return False
    
    def test_regime_switcher(self, df):
        """Test Regime Switcher"""
        print("\n🔄 Testing Regime Switcher...")
        try:
            from regime_switcher import RegimeSwitcher
            switcher = RegimeSwitcher()
            signal = switcher.get_signal(df, 'BTCUSDT')
            
            self.results['regime_switcher'] = {
                'status': 'OK',
                'regime': signal['regime'],
                'engine': signal['engine'],
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'meta_filtered': signal['meta_filtered']
            }
            print(f"   ✅ Regime: {signal['regime']} | Engine: {signal['engine']}")
            print(f"   📊 Signal: {signal['signal']} ({signal['confidence']}%) | Filtered: {signal['meta_filtered']}")
            return True
        except Exception as e:
            self.errors.append(('regime_switcher', str(e), traceback.format_exc()))
            self.results['regime_switcher'] = {'status': 'ERROR', 'error': str(e)}
            print(f"   ❌ Failed: {e}")
        return False
    
    def test_profit_guard(self):
        """Test Profit Guard / Money Management"""
        print("\n💰 Testing Profit Guard...")
        try:
            from profit_guard import ProfitGuard
            guard = ProfitGuard(account_balance=10000)
            
            signal = {
                'signal': 'BUY',
                'confidence': 75,
                'stop_mult': 1.5,
                'tp_mult': 2.0,
                'atr_pct': 1.5
            }
            
            sizing = guard.calculate_position_size(signal)
            can_trade, reason = guard.can_trade()
            
            self.results['profit_guard'] = {
                'status': 'OK',
                'can_trade': can_trade,
                'reason': reason,
                'position_size': sizing['size'],
                'kelly_pct': sizing['kelly_pct']
            }
            print(f"   ✅ Can Trade: {can_trade} | Size: ${sizing['size']} | Kelly: {sizing['kelly_pct']}%")
            return True
        except Exception as e:
            self.errors.append(('profit_guard', str(e), traceback.format_exc()))
            self.results['profit_guard'] = {'status': 'ERROR', 'error': str(e)}
            print(f"   ❌ Failed: {e}")
        return False


# =============================================================================
# MULTI-TIMEFRAME ANALYSIS
# =============================================================================

class TimeframeAnalyzer:
    """Analyze predictions across multiple timeframes"""
    
    def __init__(self):
        self.predictions = {tf: [] for tf in TIMEFRAMES}
        self.actual_moves = {tf: [] for tf in TIMEFRAMES}
        self.correlations = {}
    
    def calculate_features(self, df):
        """Calculate features for prediction"""
        df = df.copy()
        
        # Returns
        df['ret_1'] = df['c'].pct_change() * 100
        df['ret_5'] = df['c'].pct_change(5) * 100
        
        # RSI
        delta = df['c'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        # MACD
        ema12 = df['c'].ewm(span=12).mean()
        ema26 = df['c'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9).mean()
        
        # Volatility
        df['volatility'] = df['ret_1'].rolling(20).std()
        
        return df
    
    def generate_signal(self, df):
        """Generate signal from features"""
        row = df.iloc[-1]
        score = 0
        
        # RSI
        if row['rsi'] < 30: score += 2
        elif row['rsi'] < 40: score += 1
        elif row['rsi'] > 70: score -= 2
        elif row['rsi'] > 60: score -= 1
        
        # MACD
        if row['macd_hist'] > 0: score += 1
        else: score -= 1
        
        # Momentum
        if row['ret_5'] > 0.5: score += 1
        elif row['ret_5'] < -0.5: score -= 1
        
        if score >= 2:
            return 'BUY', 65 + score * 5
        elif score <= -2:
            return 'SELL', 65 + abs(score) * 5
        else:
            return 'NEUTRAL', 50
    
    def analyze_timeframe(self, symbol, timeframe):
        """Analyze prediction accuracy for a timeframe"""
        path = DATA_DIR / f"{symbol}_{timeframe}.csv"
        if not path.exists():
            return None
        
        df = pd.read_csv(path)
        df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
        df = self.calculate_features(df)
        df = df.dropna()
        
        if len(df) < 100:
            return None
        
        # Walk-forward test on last 100 rows
        results = []
        for i in range(50, len(df) - 1):
            window = df.iloc[:i]
            signal, conf = self.generate_signal(window)
            
            # Actual next-bar move
            actual = 'UP' if df.iloc[i+1]['c'] > df.iloc[i]['c'] else 'DOWN'
            predicted = 'UP' if signal == 'BUY' else ('DOWN' if signal == 'SELL' else 'FLAT')
            
            correct = (predicted == actual) or (predicted == 'FLAT')
            results.append({
                'signal': signal,
                'confidence': conf,
                'predicted': predicted,
                'actual': actual,
                'correct': correct
            })
        
        # Calculate accuracy
        if results:
            trades = [r for r in results if r['signal'] != 'NEUTRAL']
            if trades:
                accuracy = sum(1 for t in trades if t['correct']) / len(trades) * 100
                return {
                    'total_signals': len(trades),
                    'accuracy': round(accuracy, 1),
                    'buy_signals': sum(1 for t in trades if t['signal'] == 'BUY'),
                    'sell_signals': sum(1 for t in trades if t['signal'] == 'SELL'),
                    'avg_confidence': round(np.mean([t['confidence'] for t in trades]), 1)
                }
        return None
    
    def generate_correlation_matrix(self, symbol):
        """Generate correlation matrix between timeframes"""
        data = {}
        for tf in TIMEFRAMES:
            path = DATA_DIR / f"{symbol}_{tf}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
                df = self.calculate_features(df).dropna()
                if len(df) > 50:
                    data[tf] = df['ret_1'].tail(50).reset_index(drop=True)
        
        if len(data) >= 2:
            df_corr = pd.DataFrame(data)
            return df_corr.corr().to_dict()
        return None


# =============================================================================
# STRESS TEST
# =============================================================================

class StressTester:
    """Run continuous stress test"""
    
    def __init__(self, duration=600):
        self.duration = duration
        self.results = {
            'start_time': None,
            'end_time': None,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_latency': 0,
            'latencies': [],
            'errors': [],
            'prices': {sym: [] for sym in SYMBOLS},
            'signals': {sym: [] for sym in SYMBOLS}
        }
    
    async def run(self):
        """Run the stress test"""
        print(f"\n⏱️ Starting {self.duration}s Stress Test...")
        self.results['start_time'] = datetime.now().isoformat()
        
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        conn = aiohttp.TCPConnector(ssl=ssl_ctx)
        
        async with aiohttp.ClientSession(connector=conn) as session:
            start = time.time()
            tick = 0
            
            while time.time() - start < self.duration:
                tick += 1
                elapsed = int(time.time() - start)
                
                # Progress indicator every 30 seconds
                if tick % 60 == 0:
                    print(f"   ⏱️ {elapsed}s / {self.duration}s | Requests: {self.results['total_requests']} | Errors: {self.results['failed_requests']}")
                
                for sym in SYMBOLS:
                    try:
                        req_start = time.time()
                        url = f"https://api.binance.com/api/v3/ticker/price?symbol={sym}"
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                            latency = (time.time() - req_start) * 1000
                            self.results['latencies'].append(latency)
                            self.results['total_requests'] += 1
                            
                            if r.status == 200:
                                data = await r.json()
                                price = float(data['price'])
                                self.results['prices'][sym].append({
                                    'time': elapsed,
                                    'price': price
                                })
                                self.results['successful_requests'] += 1
                            else:
                                self.results['failed_requests'] += 1
                    except Exception as e:
                        self.results['failed_requests'] += 1
                        self.results['errors'].append({
                            'time': elapsed,
                            'symbol': sym,
                            'error': str(e)
                        })
                
                await asyncio.sleep(0.5)  # 500ms between cycles
        
        self.results['end_time'] = datetime.now().isoformat()
        self.results['avg_latency'] = round(np.mean(self.results['latencies']), 2) if self.results['latencies'] else 0
        
        print(f"\n✅ Stress Test Complete!")
        print(f"   Total Requests: {self.results['total_requests']}")
        print(f"   Successful: {self.results['successful_requests']}")
        print(f"   Failed: {self.results['failed_requests']}")
        print(f"   Avg Latency: {self.results['avg_latency']}ms")
        print(f"   Unique Errors: {len(set(e['error'] for e in self.results['errors']))}")
        
        return self.results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

async def run_comprehensive_analysis():
    """Run the complete analysis"""
    print("="*70)
    print("🔬 COMPREHENSIVE SYSTEM ANALYSIS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {TEST_DURATION} seconds")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'duration': TEST_DURATION,
        'components': {},
        'timeframe_analysis': {},
        'correlations': {},
        'stress_test': {},
        'errors': [],
        'verdict': None
    }
    
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    conn = aiohttp.TCPConnector(ssl=ssl_ctx)
    
    async with aiohttp.ClientSession(connector=conn) as session:
        
        # === COMPONENT TESTS ===
        print("\n" + "="*70)
        print("1️⃣ COMPONENT TESTS")
        print("="*70)
        
        tester = ComponentTester()
        await tester.test_binance_api(session)
        await tester.test_order_book(session)
        await tester.test_alternative_data(session)
        
        # Load sample data for regime test
        sample_path = DATA_DIR / "BTCUSDT_1h.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)
            df.columns = ['time', 'o', 'h', 'l', 'c', 'v']
            tester.test_regime_switcher(df)
        
        tester.test_profit_guard()
        
        report['components'] = tester.results
        report['errors'].extend(tester.errors)
        
        # === TIMEFRAME ANALYSIS ===
        print("\n" + "="*70)
        print("2️⃣ MULTI-TIMEFRAME ANALYSIS")
        print("="*70)
        
        analyzer = TimeframeAnalyzer()
        
        for sym in SYMBOLS:
            print(f"\n📊 {sym}:")
            report['timeframe_analysis'][sym] = {}
            
            for tf in TIMEFRAMES:
                result = analyzer.analyze_timeframe(sym, tf)
                if result:
                    report['timeframe_analysis'][sym][tf] = result
                    print(f"   {tf}: Accuracy={result['accuracy']}% | Signals={result['total_signals']} | Conf={result['avg_confidence']}%")
                else:
                    print(f"   {tf}: No data available")
            
            # Correlations
            corr = analyzer.generate_correlation_matrix(sym)
            if corr:
                report['correlations'][sym] = corr
                print(f"   📈 Correlation matrix generated")
        
        # === STRESS TEST ===
        print("\n" + "="*70)
        print("3️⃣ STRESS TEST ({TEST_DURATION}s)")
        print("="*70)
        
        stress = StressTester(TEST_DURATION)
        stress_results = await stress.run()
        report['stress_test'] = {
            'total_requests': stress_results['total_requests'],
            'successful': stress_results['successful_requests'],
            'failed': stress_results['failed_requests'],
            'avg_latency_ms': stress_results['avg_latency'],
            'uptime_pct': round(stress_results['successful_requests'] / max(stress_results['total_requests'], 1) * 100, 2),
            'unique_errors': list(set(e['error'] for e in stress_results['errors']))
        }
        
        # === FINAL VERDICT ===
        print("\n" + "="*70)
        print("🏆 FINAL VERDICT")
        print("="*70)
        
        # Calculate scores
        component_score = sum(1 for c in report['components'].values() if c.get('status') == 'OK') / len(report['components']) * 100
        
        # Average accuracy across timeframes
        accuracies = []
        for sym_data in report['timeframe_analysis'].values():
            for tf_data in sym_data.values():
                if 'accuracy' in tf_data:
                    accuracies.append(tf_data['accuracy'])
        avg_accuracy = np.mean(accuracies) if accuracies else 0
        
        uptime = report['stress_test'].get('uptime_pct', 0)
        
        # Is it a toy or real?
        if component_score >= 80 and avg_accuracy >= 50 and uptime >= 95:
            verdict = "INSTITUTIONAL GRADE"
            emoji = "🏆"
        elif component_score >= 60 and avg_accuracy >= 45 and uptime >= 90:
            verdict = "PROFESSIONAL AMATEUR"
            emoji = "🥈"
        elif component_score >= 40 and uptime >= 80:
            verdict = "FUNCTIONAL PROTOTYPE"
            emoji = "🥉"
        else:
            verdict = "NEEDS WORK"
            emoji = "⚠️"
        
        report['verdict'] = {
            'rating': verdict,
            'component_score': round(component_score, 1),
            'avg_accuracy': round(avg_accuracy, 1),
            'uptime': uptime,
            'error_count': len(report['errors'])
        }
        
        print(f"\n{emoji} VERDICT: {verdict}")
        print(f"   Component Score: {component_score:.0f}%")
        print(f"   Average Accuracy: {avg_accuracy:.1f}%")
        print(f"   Uptime: {uptime:.1f}%")
        print(f"   Errors Found: {len(report['errors'])}")
        
        if report['errors']:
            print("\n⚠️ ERRORS TO FIX:")
            for i, err in enumerate(report['errors'][:5]):
                print(f"   {i+1}. {err[0]}: {err[1]}")
        
        # Save report
        report_path = REPORT_DIR / f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Report saved: {report_path}")
        
        return report


if __name__ == "__main__":
    asyncio.run(run_comprehensive_analysis())
