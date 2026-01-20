"""
FORGE TRADING SYSTEM - DATABASE
================================
SQLite with WAL mode for trade logging.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json
import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class Database:
    """SQLite database with WAL mode for trades and signals"""
    
    def __init__(self, db_path: str = None):
        config = load_config()
        db_cfg = config.get('database', {})
        
        self.db_path = db_path or db_cfg.get('path', 'trades.db')
        self.use_wal = db_cfg.get('wal_mode', True)
        
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        if self.use_wal:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
    
    def _create_tables(self):
        """Create required tables"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                score INTEGER,
                entry REAL,
                sl REAL,
                tp REAL,
                conviction_passed INTEGER,
                conviction_reason TEXT,
                reasons TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_time TEXT,
                entry_price REAL,
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                sl REAL,
                tp REAL,
                pnl REAL,
                pnl_pct REAL,
                r_multiple REAL,
                status TEXT DEFAULT 'open',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES signals(id)
            );
            
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL,
                change REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        """)
        self.conn.commit()
    
    def log_signal(self, signal: Dict, conviction: Dict) -> int:
        """Log a signal to database"""
        cursor = self.conn.execute("""
            INSERT INTO signals (
                timestamp, symbol, timeframe, direction, score, 
                entry, sl, tp, conviction_passed, conviction_reason, reasons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.get('timestamp', datetime.now(timezone.utc).isoformat()),
            signal.get('symbol', ''),
            signal.get('timeframe', ''),
            signal.get('direction', ''),
            signal.get('score', 0),
            signal.get('entry', 0),
            signal.get('sl', 0),
            signal.get('tp', 0),
            1 if conviction.get('passed', False) else 0,
            conviction.get('reason', ''),
            json.dumps(signal.get('reasons', []))
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def log_price(self, symbol: str, price: float, change: float):
        """Log price update"""
        self.conn.execute("""
            INSERT INTO prices (timestamp, symbol, price, change)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            symbol,
            price,
            change
        ))
        self.conn.commit()
    
    def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Get recent signals"""
        cursor = self.conn.execute("""
            SELECT * FROM signals
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_signals_by_symbol(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get signals for specific symbol"""
        cursor = self.conn.execute("""
            SELECT * FROM signals
            WHERE symbol = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (symbol, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_daily_trade_count(self) -> int:
        """Get number of trades today"""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM trades
            WHERE date(created_at) = ?
        """, (today,))
        
        return cursor.fetchone()[0]
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
