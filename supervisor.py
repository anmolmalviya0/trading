"""
FORGE TRADING SYSTEM - SUPERVISOR (SELF-HEALING)
=================================================
Jet-grade redundancy and self-healing monitor.
Monitors primary/secondary engines, handles failures.
"""
import threading
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Callable
from enum import Enum
import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class EngineState(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    RESTARTING = "RESTARTING"


class Supervisor:
    """
    Watchdog supervisor for redundancy and self-healing.
    
    Monitors:
    - Primary engine health
    - Secondary (standby) engine health
    - System integrity
    
    Actions:
    - Restart failing components
    - Switch to standby
    - Freeze trading if unsafe
    """
    
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.sup_cfg = self.config.get('supervisor', {})
        
        self.restart_attempts = self.sup_cfg.get('restart_attempts', 3)
        self.health_check_interval = self.sup_cfg.get('health_check_interval', 10)
        
        # Engine references
        self.primary_engine = None
        self.secondary_engine = None
        
        # State tracking
        self.primary_state = EngineState.HEALTHY
        self.secondary_state = EngineState.HEALTHY
        self.active_engine = 'primary'
        
        # Restart counters
        self.primary_restarts = 0
        self.secondary_restarts = 0
        
        # Health metrics
        self.last_primary_heartbeat = None
        self.last_secondary_heartbeat = None
        self.last_price_update = None
        self.last_signal = None
        
        # Callbacks
        self.on_failover: Optional[Callable] = None
        self.on_freeze: Optional[Callable] = None
        
        self.running = False
        self.thread = None
        self.frozen = False
        
        # Event log
        self.events: list = []
    
    def register_engines(self, primary, secondary=None):
        """Register engine instances for monitoring"""
        self.primary_engine = primary
        self.secondary_engine = secondary
    
    def heartbeat(self, engine: str):
        """Record heartbeat from engine"""
        now = datetime.now(timezone.utc)
        if engine == 'primary':
            self.last_primary_heartbeat = now
        else:
            self.last_secondary_heartbeat = now
    
    def record_price_update(self):
        """Record that price was updated"""
        self.last_price_update = datetime.now(timezone.utc)
    
    def record_signal(self):
        """Record that a signal was generated"""
        self.last_signal = datetime.now(timezone.utc)
    
    def _log_event(self, level: str, message: str):
        """Log supervisor event"""
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': level,
            'message': message
        }
        self.events.append(event)
        
        # Keep last 100
        if len(self.events) > 100:
            self.events = self.events[-100:]
        
        print(f"[SUPERVISOR] [{level}] {message}")
    
    def _check_engine_health(self, engine: str) -> EngineState:
        """Check health of an engine"""
        now = datetime.now(timezone.utc)
        
        if engine == 'primary':
            heartbeat = self.last_primary_heartbeat
            eng = self.primary_engine
        else:
            heartbeat = self.last_secondary_heartbeat
            eng = self.secondary_engine
        
        if eng is None:
            return EngineState.FAILED
        
        if heartbeat is None:
            return EngineState.DEGRADED
        
        # Check heartbeat age
        age = (now - heartbeat).total_seconds()
        
        if age > 60:
            return EngineState.FAILED
        elif age > 30:
            return EngineState.DEGRADED
        
        return EngineState.HEALTHY
    
    def _check_data_freshness(self) -> bool:
        """Check if data is still fresh"""
        if self.last_price_update is None:
            return False
        
        now = datetime.now(timezone.utc)
        age = (now - self.last_price_update).total_seconds()
        
        return age < 30  # Prices should update within 30s
    
    def _attempt_restart(self, engine: str) -> bool:
        """Attempt to restart an engine"""
        if engine == 'primary':
            if self.primary_restarts >= self.restart_attempts:
                return False
            
            self.primary_restarts += 1
            self.primary_state = EngineState.RESTARTING
            
            try:
                if self.primary_engine:
                    self.primary_engine.stop()
                    time.sleep(2)
                    self.primary_engine.start()
                    self._log_event('INFO', f'Primary engine restarted (attempt {self.primary_restarts})')
                    return True
            except Exception as e:
                self._log_event('ERROR', f'Primary restart failed: {e}')
                return False
        else:
            if self.secondary_restarts >= self.restart_attempts:
                return False
            
            self.secondary_restarts += 1
            self.secondary_state = EngineState.RESTARTING
            
            try:
                if self.secondary_engine:
                    self.secondary_engine.stop()
                    time.sleep(2)
                    self.secondary_engine.start()
                    self._log_event('INFO', f'Secondary engine restarted (attempt {self.secondary_restarts})')
                    return True
            except Exception as e:
                self._log_event('ERROR', f'Secondary restart failed: {e}')
                return False
        
        return False
    
    def _failover_to_secondary(self):
        """Switch to secondary engine"""
        if self.secondary_engine is None:
            self._log_event('ERROR', 'No secondary engine available for failover')
            return False
        
        if self.secondary_state == EngineState.HEALTHY:
            self.active_engine = 'secondary'
            self._log_event('WARN', 'Failover to secondary engine')
            
            if self.on_failover:
                self.on_failover('secondary')
            
            return True
        
        return False
    
    def _freeze_system(self, reason: str):
        """Freeze all trading"""
        if not self.frozen:
            self.frozen = True
            self._log_event('CRITICAL', f'System frozen: {reason}')
            
            if self.on_freeze:
                self.on_freeze(reason)
    
    def _unfreeze_system(self):
        """Unfreeze trading"""
        if self.frozen:
            self.frozen = False
            self._log_event('INFO', 'System unfrozen')
    
    def _health_check(self):
        """Run full health check"""
        # Check primary
        self.primary_state = self._check_engine_health('primary')
        
        # Check secondary
        if self.secondary_engine:
            self.secondary_state = self._check_engine_health('secondary')
        
        # Check data freshness
        data_fresh = self._check_data_freshness()
        
        # Decision logic
        if self.primary_state == EngineState.FAILED:
            if self.active_engine == 'primary':
                # Try restart first
                if not self._attempt_restart('primary'):
                    # Failover to secondary
                    if not self._failover_to_secondary():
                        self._freeze_system('Both engines failed')
        
        elif self.primary_state == EngineState.DEGRADED:
            self._log_event('WARN', 'Primary engine degraded')
        
        if not data_fresh and not self.frozen:
            self._log_event('WARN', 'Data feed stale')
        
        # Reset restart counters if healthy for a while
        if self.primary_state == EngineState.HEALTHY:
            self.primary_restarts = 0
        if self.secondary_state == EngineState.HEALTHY:
            self.secondary_restarts = 0
        
        # Unfreeze if recovered
        if self.frozen:
            if self.primary_state == EngineState.HEALTHY or \
               self.secondary_state == EngineState.HEALTHY:
                if data_fresh:
                    self._unfreeze_system()
    
    def _run_loop(self):
        """Supervisor main loop"""
        while self.running:
            self._health_check()
            time.sleep(self.health_check_interval)
    
    def start(self):
        """Start supervisor"""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self._log_event('INFO', 'Supervisor started')
        return self
    
    def stop(self):
        """Stop supervisor"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self._log_event('INFO', 'Supervisor stopped')
    
    def get_status(self) -> Dict:
        """Get current supervisor status"""
        return {
            'primary_state': self.primary_state.value,
            'secondary_state': self.secondary_state.value if self.secondary_engine else 'N/A',
            'active_engine': self.active_engine,
            'frozen': self.frozen,
            'primary_restarts': self.primary_restarts,
            'secondary_restarts': self.secondary_restarts,
            'data_fresh': self._check_data_freshness(),
            'recent_events': self.events[-5:]
        }
    
    def is_frozen(self) -> bool:
        """Check if system is frozen"""
        return self.frozen
