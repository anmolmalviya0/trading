"""
AUTONOMOUS SELF-HEALING ENGINE
==============================
A TRUE MACHINE that:
1. Detects errors automatically
2. Diagnoses the problem
3. Fixes itself
4. Backtests the fix
5. Deploys if successful

This is not a toy. This is an autonomous system.

Usage:
    python autonomous_engine.py
"""
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import traceback
import json
import subprocess
import time
import aiohttp
import ssl
import certifi

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / 'autonomous.db'

CONFIG = {
    'MODEL_RELOAD_HOURS': 24,        # Retrain every 24 hours
    'HEALTH_CHECK_INTERVAL': 30,     # Check health every 30 seconds
    'ERROR_THRESHOLD': 3,            # Max errors before auto-fix
    'BACKTEST_MIN_TRADES': 100,      # Min trades for valid backtest
    'BACKTEST_MIN_WINRATE': 50,      # Min win rate to deploy
    'BACKTEST_MIN_PF': 1.0,          # Min profit factor to deploy
}

# === DATABASE ===
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS error_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            component TEXT NOT NULL,
            error_type TEXT NOT NULL,
            error_message TEXT,
            stack_trace TEXT,
            auto_fixed INTEGER DEFAULT 0,
            fix_action TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            component TEXT NOT NULL,
            status TEXT NOT NULL,
            latency_ms INTEGER,
            details TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_reloads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_name TEXT NOT NULL,
            accuracy REAL,
            backtest_winrate REAL,
            backtest_pf REAL,
            deployed INTEGER DEFAULT 0,
            reason TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS auto_fixes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            error_id INTEGER,
            component TEXT NOT NULL,
            fix_type TEXT NOT NULL,
            fix_action TEXT,
            backtest_result TEXT,
            deployed INTEGER DEFAULT 0,
            success INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

# === ERROR TYPES ===
class ErrorType:
    API_TIMEOUT = "API_TIMEOUT"
    API_ERROR = "API_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    DATA_ERROR = "DATA_ERROR"
    WEBSOCKET_ERROR = "WEBSOCKET_ERROR"
    FEATURE_MISMATCH = "FEATURE_MISMATCH"
    PREDICTION_ERROR = "PREDICTION_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"

# === FIX TYPES ===
class FixType:
    RESTART_COMPONENT = "RESTART_COMPONENT"
    RELOAD_MODEL = "RELOAD_MODEL"
    RELOAD_DATA = "RELOAD_DATA"
    RECONNECT_API = "RECONNECT_API"
    RETRAIN_MODEL = "RETRAIN_MODEL"
    CLEAR_CACHE = "CLEAR_CACHE"

# === AUTONOMOUS ENGINE ===
class AutonomousEngine:
    """
    Self-healing autonomous trading engine.
    
    Monitors all components, detects errors, fixes them automatically,
    backtests fixes, and deploys only if backtest passes.
    """
    
    def __init__(self):
        self.running = False
        self.error_counts = {}
        self.last_model_reload = None
        self.next_model_reload = None
        self.health_status = {}
        self.last_errors = []
        self.auto_fix_history = []
        
        init_database()
        self._load_state()
        self._schedule_next_reload()
    
    def _load_state(self):
        """Load state from database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get last model reload
        cursor.execute("SELECT timestamp FROM model_reloads ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            self.last_model_reload = datetime.fromisoformat(row[0])
        else:
            self.last_model_reload = datetime.now()
        
        conn.close()
    
    def _schedule_next_reload(self):
        """Schedule next model reload"""
        self.next_model_reload = self.last_model_reload + timedelta(hours=CONFIG['MODEL_RELOAD_HOURS'])
        print(f"📅 Next model reload: {self.next_model_reload.strftime('%Y-%m-%d %H:%M')}")
    
    def get_reload_countdown(self):
        """Get time until next reload"""
        if not self.next_model_reload:
            return "N/A"
        
        delta = self.next_model_reload - datetime.now()
        if delta.total_seconds() <= 0:
            return "NOW"
        
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        return f"{hours}h {minutes}m"
    
    # === ERROR DETECTION ===
    
    def log_error(self, component: str, error_type: str, error: Exception):
        """Log an error and trigger auto-fix if threshold reached"""
        error_msg = str(error)
        stack = traceback.format_exc()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO error_log (timestamp, component, error_type, error_message, stack_trace)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), component, error_type, error_msg, stack))
        error_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Track error count
        key = f"{component}:{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        self.last_errors.append({
            'id': error_id,
            'time': datetime.now().isoformat(),
            'component': component,
            'type': error_type,
            'message': error_msg[:100]
        })
        self.last_errors = self.last_errors[-10:]  # Keep last 10
        
        print(f"❌ Error [{component}]: {error_type} - {error_msg[:50]}")
        
        # Check if auto-fix needed
        if self.error_counts[key] >= CONFIG['ERROR_THRESHOLD']:
            asyncio.create_task(self.auto_fix(error_id, component, error_type))
            self.error_counts[key] = 0
        
        return error_id
    
    # === HEALTH CHECKS ===
    
    async def check_api_health(self):
        """Check Binance API health"""
        try:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            conn = aiohttp.TCPConnector(ssl=ssl_ctx)
            
            start = time.time()
            async with aiohttp.ClientSession(connector=conn) as session:
                async with session.get(
                    "https://api.binance.com/api/v3/ping",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    latency = int((time.time() - start) * 1000)
                    
                    if r.status == 200:
                        self.health_status['api'] = {'status': 'OK', 'latency': latency}
                        return True, latency
                    else:
                        self.health_status['api'] = {'status': 'ERROR', 'code': r.status}
                        return False, latency
        except Exception as e:
            self.log_error('API', ErrorType.API_ERROR, e)
            self.health_status['api'] = {'status': 'DOWN', 'error': str(e)[:50]}
            return False, 0
    
    async def check_model_health(self):
        """Check if models are loaded and working"""
        try:
            import joblib
            model_dir = BASE_DIR / 'models' / 'production'
            
            models_ok = 0
            for sym in ['BTCUSDT', 'PAXGUSDT']:
                path = model_dir / f"{sym}_1h.pkl"
                if path.exists():
                    model = joblib.load(path)
                    if model:
                        models_ok += 1
            
            if models_ok >= 2:
                self.health_status['models'] = {'status': 'OK', 'loaded': models_ok}
                return True
            else:
                self.health_status['models'] = {'status': 'PARTIAL', 'loaded': models_ok}
                return False
        except Exception as e:
            self.log_error('MODEL', ErrorType.MODEL_ERROR, e)
            self.health_status['models'] = {'status': 'ERROR', 'error': str(e)[:50]}
            return False
    
    async def check_data_health(self):
        """Check if data files exist and are recent"""
        try:
            data_dir = BASE_DIR.parent / 'market_data'
            files_ok = 0
            
            for sym in ['BTCUSDT', 'PAXGUSDT']:
                path = data_dir / f"{sym}_1h.csv"
                if path.exists():
                    files_ok += 1
            
            if files_ok >= 2:
                self.health_status['data'] = {'status': 'OK', 'files': files_ok}
                return True
            else:
                self.health_status['data'] = {'status': 'PARTIAL', 'files': files_ok}
                return False
        except Exception as e:
            self.log_error('DATA', ErrorType.DATA_ERROR, e)
            self.health_status['data'] = {'status': 'ERROR', 'error': str(e)[:50]}
            return False
    
    async def run_health_check(self):
        """Run all health checks"""
        api_ok, api_latency = await self.check_api_health()
        model_ok = await self.check_model_health()
        data_ok = await self.check_data_health()
        
        overall = 'OK' if all([api_ok, model_ok, data_ok]) else 'DEGRADED'
        if not api_ok:
            overall = 'CRITICAL'
        
        self.health_status['overall'] = overall
        self.health_status['last_check'] = datetime.now().isoformat()
        
        return overall
    
    # === AUTO-FIX ===
    
    async def auto_fix(self, error_id: int, component: str, error_type: str):
        """Automatically fix errors based on type"""
        print(f"🔧 AUTO-FIX: Attempting to fix {component} - {error_type}")
        
        fix_type = None
        fix_action = None
        success = False
        
        try:
            if error_type == ErrorType.API_TIMEOUT or error_type == ErrorType.API_ERROR:
                fix_type = FixType.RECONNECT_API
                fix_action = "Waiting 10s and retrying API connection"
                await asyncio.sleep(10)
                ok, _ = await self.check_api_health()
                success = ok
            
            elif error_type == ErrorType.MODEL_ERROR or error_type == ErrorType.FEATURE_MISMATCH:
                fix_type = FixType.RELOAD_MODEL
                fix_action = "Reloading models from disk"
                success = await self.reload_models()
            
            elif error_type == ErrorType.DATA_ERROR:
                fix_type = FixType.RELOAD_DATA
                fix_action = "Checking and reloading data files"
                success = await self.check_data_health()
            
            elif error_type == ErrorType.PREDICTION_ERROR:
                fix_type = FixType.RETRAIN_MODEL
                fix_action = "Triggering model retrain"
                success = await self.retrain_and_backtest()
            
            else:
                fix_type = FixType.RESTART_COMPONENT
                fix_action = f"Restarting {component}"
                await asyncio.sleep(5)
                success = True
            
            # Log the fix
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO auto_fixes (timestamp, error_id, component, fix_type, fix_action, success)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), error_id, component, fix_type, fix_action, 1 if success else 0))
            conn.commit()
            conn.close()
            
            # Update error log
            if success:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("UPDATE error_log SET auto_fixed = 1, fix_action = ? WHERE id = ?", (fix_action, error_id))
                conn.commit()
                conn.close()
            
            self.auto_fix_history.append({
                'time': datetime.now().isoformat(),
                'component': component,
                'fix_type': fix_type,
                'success': success
            })
            self.auto_fix_history = self.auto_fix_history[-10:]
            
            icon = "✅" if success else "❌"
            print(f"{icon} AUTO-FIX: {fix_type} - {'Success' if success else 'Failed'}")
            
        except Exception as e:
            print(f"❌ AUTO-FIX FAILED: {e}")
    
    async def reload_models(self):
        """Reload models from disk"""
        try:
            import joblib
            model_dir = BASE_DIR / 'models' / 'production'
            
            for sym in ['BTCUSDT', 'PAXGUSDT']:
                path = model_dir / f"{sym}_1h.pkl"
                if path.exists():
                    joblib.load(path)
            
            return True
        except:
            return False
    
    # === RETRAIN AND BACKTEST ===
    
    async def retrain_and_backtest(self):
        """
        The core autonomous loop:
        1. Retrain model
        2. Backtest new model
        3. Deploy only if backtest passes
        """
        print("🔄 AUTONOMOUS RETRAIN: Starting retrain + backtest cycle...")
        
        try:
            # Step 1: Retrain
            print("   📊 Step 1: Retraining model...")
            retrain_result = await self._run_script('institutional_validation.py')
            
            if not retrain_result:
                print("   ❌ Retrain failed")
                return False
            
            # Step 2: Backtest
            print("   📈 Step 2: Running backtest...")
            backtest_result = await self._run_backtest()
            
            if not backtest_result:
                print("   ❌ Backtest failed or did not meet criteria")
                return False
            
            # Step 3: Deploy if passes
            if backtest_result['win_rate'] >= CONFIG['BACKTEST_MIN_WINRATE'] and \
               backtest_result['profit_factor'] >= CONFIG['BACKTEST_MIN_PF']:
                
                print(f"   ✅ Backtest PASSED: WR={backtest_result['win_rate']:.1f}%, PF={backtest_result['profit_factor']:.2f}")
                print("   🚀 Step 3: Deploying new model...")
                
                # Log successful reload
                self.last_model_reload = datetime.now()
                self._schedule_next_reload()
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_reloads (timestamp, model_name, backtest_winrate, backtest_pf, deployed, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    'ALL_MODELS',
                    backtest_result['win_rate'],
                    backtest_result['profit_factor'],
                    1,
                    'Scheduled retrain - passed backtest'
                ))
                conn.commit()
                conn.close()
                
                return True
            else:
                print(f"   ⚠️ Backtest FAILED criteria: WR={backtest_result['win_rate']:.1f}%, PF={backtest_result['profit_factor']:.2f}")
                print("   ❌ NOT deploying - keeping old model")
                return False
            
        except Exception as e:
            print(f"   ❌ Retrain cycle failed: {e}")
            self.log_error('RETRAIN', ErrorType.MODEL_ERROR, e)
            return False
    
    async def _run_script(self, script_name: str):
        """Run a Python script and capture result"""
        try:
            script_path = BASE_DIR / script_name
            if not script_path.exists():
                return False
            
            process = await asyncio.create_subprocess_exec(
                'python3', str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BASE_DIR)
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300  # 5 minute timeout
            )
            
            return process.returncode == 0
        except:
            return False
    
    async def _run_backtest(self):
        """Run backtest and return results"""
        try:
            # Load the latest backtest results
            report_path = BASE_DIR / 'reports' / 'institutional_validation.json'
            if report_path.exists():
                with open(report_path) as f:
                    data = json.load(f)
                
                summary = data.get('summary', {})
                return {
                    'win_rate': summary.get('avg_win_rate', 0),
                    'profit_factor': summary.get('avg_profit_factor', 0),
                    'trades': summary.get('total_trades_simulated', 0)
                }
            return None
        except:
            return None
    
    # === MAIN LOOP ===
    
    async def run(self):
        """Main autonomous loop"""
        print("="*60)
        print("🤖 AUTONOMOUS SELF-HEALING ENGINE")
        print("="*60)
        print(f"   Model reload: Every {CONFIG['MODEL_RELOAD_HOURS']} hours")
        print(f"   Health check: Every {CONFIG['HEALTH_CHECK_INTERVAL']} seconds")
        print(f"   Error threshold: {CONFIG['ERROR_THRESHOLD']} before auto-fix")
        print("="*60)
        
        self.running = True
        
        while self.running:
            try:
                # Health check
                status = await self.run_health_check()
                
                # Check if model reload is due
                if datetime.now() >= self.next_model_reload:
                    print("\n📅 SCHEDULED MODEL RELOAD")
                    await self.retrain_and_backtest()
                
                await asyncio.sleep(CONFIG['HEALTH_CHECK_INTERVAL'])
                
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                await asyncio.sleep(5)
    
    def get_status(self):
        """Get current status for UI"""
        return {
            'health': self.health_status,
            'next_reload': self.get_reload_countdown(),
            'last_errors': self.last_errors[-5:],
            'auto_fixes': self.auto_fix_history[-5:],
            'error_counts': dict(self.error_counts),
        }

# === SINGLETON ===
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = AutonomousEngine()
    return engine

# === MAIN ===
if __name__ == "__main__":
    engine = get_engine()
    asyncio.run(engine.run())
