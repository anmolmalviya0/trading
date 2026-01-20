"""
MARKETFORGE: Database Schema
============================
SQLite WAL schema for runtime persistence.

Tables:
- signals: All generated signals with full context
- trades: Trade outcomes for performance tracking
- model_versions: Model artifacts and manifests
- daily_stats: Daily P&L and guardrail state
- feature_drift: Feature distribution monitoring

Usage:
    python db_schema.py init    # Initialize database
    python db_schema.py migrate # Run migrations
"""

import sqlite3
from pathlib import Path
import json
from datetime import datetime
import hashlib
import sys

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "performance.db"


def get_connection():
    """Get database connection with WAL mode"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize complete database schema"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # =========================================================================
    # SIGNALS TABLE
    # =========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        asset TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        signal TEXT NOT NULL CHECK(signal IN ('BUY', 'SELL', 'NO-TRADE')),
        bias_strength TEXT CHECK(bias_strength IN ('STRONG', 'MODERATE', 'WEAK')),
        
        -- Entry Zone
        entry_zone_low REAL,
        entry_zone_high REAL,
        entry_price REAL,
        
        -- Targets
        stop_loss REAL,
        tp1 REAL,
        tp2 REAL,
        tp3 REAL,
        pips_tp1 REAL,
        
        -- Confidence & Regime
        confidence_pct REAL NOT NULL,
        regime TEXT CHECK(regime IN ('TREND', 'CHOP', 'MEANREV', 'VOLATILE')),
        
        -- Model Info
        model_id TEXT,
        feature_schema_hash TEXT,
        
        -- Explainability
        reason_codes TEXT,  -- JSON array
        shap_values TEXT,   -- JSON blob (compressed)
        feature_vector TEXT, -- JSON blob for reproducibility
        
        -- State
        executed INTEGER DEFAULT 0,
        operator_action TEXT,
        notes TEXT,
        
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Indexes for signals
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_asset_ts ON signals(asset, timestamp_utc)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_signal ON signals(signal)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_confidence ON signals(confidence_pct)')
    
    # =========================================================================
    # TRADES TABLE
    # =========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER,
        
        -- Trade Details
        timestamp_open TEXT NOT NULL,
        timestamp_close TEXT,
        asset TEXT NOT NULL,
        side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
        
        -- Prices
        entry_price REAL NOT NULL,
        exit_price REAL,
        quantity REAL NOT NULL,
        
        -- Costs
        commission REAL DEFAULT 0,
        slippage REAL DEFAULT 0,
        
        -- Outcome
        pnl_gross REAL,
        pnl_net REAL,
        pnl_r REAL,  -- P&L in R multiples
        
        -- Exit Info
        exit_reason TEXT,  -- TP1, TP2, TP3, SL, TIME, MANUAL
        holding_bars INTEGER,
        
        -- State
        status TEXT DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'CLOSED', 'CANCELLED')),
        
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT,
        
        FOREIGN KEY (signal_id) REFERENCES signals(id)
    )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)')
    
    # =========================================================================
    # MODEL VERSIONS TABLE
    # =========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id TEXT UNIQUE NOT NULL,  -- SHA256 hash
        
        -- Model Info
        model_type TEXT NOT NULL,
        asset TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        
        -- Training Window
        train_start TEXT,
        train_end TEXT,
        n_samples INTEGER,
        
        -- Hyperparameters
        hyperparams TEXT,  -- JSON
        
        -- Performance Metrics
        profit_factor REAL,
        win_rate REAL,
        expectancy_r REAL,
        sharpe_ratio REAL,
        sortino_ratio REAL,
        max_drawdown_pct REAL,
        total_trades INTEGER,
        
        -- Walk-Forward Results
        wfv_results TEXT,  -- JSON with per-fold metrics
        monte_carlo_results TEXT,  -- JSON with CI bounds
        
        -- Feature Schema
        feature_schema_hash TEXT,
        feature_schema TEXT,  -- JSON
        scaler_params TEXT,   -- JSON (fitted on train only)
        
        -- Artifact Path
        artifact_path TEXT,
        
        -- State
        is_active INTEGER DEFAULT 0,
        deployed_at TEXT,
        retired_at TEXT,
        
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_active ON model_versions(is_active)')
    cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_models_id ON model_versions(model_id)')
    
    # =========================================================================
    # DAILY STATS TABLE
    # =========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_stats (
        date TEXT PRIMARY KEY,
        
        -- Balance
        starting_balance REAL,
        ending_balance REAL,
        
        -- P&L
        pnl_gross REAL DEFAULT 0,
        pnl_net REAL DEFAULT 0,
        pnl_pct REAL DEFAULT 0,
        
        -- Trades
        trades_total INTEGER DEFAULT 0,
        trades_won INTEGER DEFAULT 0,
        trades_lost INTEGER DEFAULT 0,
        
        -- Risk
        max_intraday_drawdown REAL DEFAULT 0,
        
        -- Guardrails
        is_halted INTEGER DEFAULT 0,
        halt_reason TEXT,
        halt_time TEXT,
        
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT
    )
    ''')
    
    # =========================================================================
    # FEATURE DRIFT TABLE
    # =========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feature_drift (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        
        -- Drift Detection
        feature_name TEXT NOT NULL,
        kl_divergence REAL,
        population_zscore REAL,
        
        -- Reference vs Current
        reference_mean REAL,
        reference_std REAL,
        current_mean REAL,
        current_std REAL,
        
        -- Alert
        is_alert INTEGER DEFAULT 0,
        alert_sent INTEGER DEFAULT 0,
        
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_drift_ts ON feature_drift(timestamp_utc)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_drift_alert ON feature_drift(is_alert)')
    
    # =========================================================================
    # OPERATOR ACTIONS TABLE
    # =========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS operator_actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        
        action_type TEXT NOT NULL,  -- ACCEPT, REJECT, OVERRIDE, KILL_SWITCH
        signal_id INTEGER,
        
        -- Details
        details TEXT,  -- JSON
        
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        
        FOREIGN KEY (signal_id) REFERENCES signals(id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database initialized: {DB_PATH}")
    print("   Tables: signals, trades, model_versions, daily_stats, feature_drift, operator_actions")


def get_schema_hash():
    """Get SHA256 hash of current schema"""
    with open(__file__, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def run_migrations():
    """Run any pending migrations"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if migrations table exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS migrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT UNIQUE,
        applied_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Define migrations
    migrations = [
        ("v1.0.0", [
            # Initial schema - no changes needed
        ]),
        ("v1.0.1", [
            # Example future migration
            # "ALTER TABLE signals ADD COLUMN new_field TEXT"
        ]),
    ]
    
    for version, statements in migrations:
        cursor.execute("SELECT 1 FROM migrations WHERE version = ?", (version,))
        if cursor.fetchone() is None:
            print(f"   Applying migration {version}...")
            for stmt in statements:
                if stmt:
                    cursor.execute(stmt)
            cursor.execute("INSERT INTO migrations (version) VALUES (?)", (version,))
    
    conn.commit()
    conn.close()
    print("✅ Migrations complete")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "init"
    
    if cmd == "init":
        init_database()
    elif cmd == "migrate":
        run_migrations()
    else:
        print("Usage: python db_schema.py [init|migrate]")
