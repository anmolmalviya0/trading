"""
FORGE TRADING SYSTEM - LIVE ENGINE
===================================
- WebSocket candle subscription
- Signal deduplication per candle
- Reconnect logic + watchdog
"""
import asyncio
import threading
from datetime import datetime, timezone
from typing import Dict, Set, Callable, Optional
from pathlib import Path
import yaml
import time

from core.data import LiveDataFeed, HistoricalData
from core.features import add_features
from core.signals import SignalEngine
from core.conviction import ConvictionFilter, create_trade_card
from core.database import Database


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class LiveEngine:
    """
    Live trading engine.
    Produces Trade Cards, not auto-execution.
    """
    
    def __init__(self, on_signal: Callable = None, on_price: Callable = None):
        self.config = load_config()
        self.on_signal = on_signal
        self.on_price = on_price
        
        self.signal_engine = SignalEngine(self.config)
        self.conviction_filter = ConvictionFilter(self.config)
        self.database = Database()
        self.data_loader = HistoricalData()
        
        # Deduplication: track processed candles
        self.processed_candles: Set[str] = set()
        
        # Historical data for each symbol/tf
        self.candle_history: Dict[str, list] = {}
        
        # Initialize data feed
        self.feed = LiveDataFeed(
            on_price=self._handle_price,
            on_candle=self._handle_candle
        )
        
        # Trade card storage
        self.trade_cards: list = []
        
        # Circuit breaker
        self.risk = self.config.get('risk', {})
        self.max_trades_per_day = self.risk.get('max_trades_per_day', 10)
        
        # Running state
        self.running = False
    
    def _handle_price(self, symbol: str, data: Dict):
        """Handle live price update"""
        if self.on_price:
            self.on_price(symbol, data)
    
    def _handle_candle(self, symbol: str, interval: str, candle: Dict):
        """Handle candle close - generate signals"""
        key = f"{symbol}_{interval}"
        candle_id = f"{key}_{candle['timestamp']}"
        
        # Deduplication
        if candle_id in self.processed_candles:
            return
        self.processed_candles.add(candle_id)
        
        # Keep only last 1000 candle IDs
        if len(self.processed_candles) > 1000:
            oldest = list(self.processed_candles)[:500]
            for old in oldest:
                self.processed_candles.discard(old)
        
        # Build history
        if key not in self.candle_history:
            # Load from local storage
            local_df = self.data_loader.load(symbol, interval)
            if not local_df.empty:
                self.candle_history[key] = local_df.to_dict('records')
            else:
                self.candle_history[key] = []
        
        # Append new candle
        self.candle_history[key].append(candle)
        
        # Keep last 500 candles only
        if len(self.candle_history[key]) > 500:
            self.candle_history[key] = self.candle_history[key][-500:]
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(self.candle_history[key])
        
        if len(df) < 50:
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = add_features(df, self.config)
        
        # Generate signal
        signal = self.signal_engine.generate_signal(df.reset_index(), symbol, interval)
        
        if signal:
            # Check daily trade limit
            daily_trades = self.database.get_daily_trade_count()
            if daily_trades >= self.max_trades_per_day:
                return
            
            # Apply conviction filter
            conviction = self.conviction_filter.evaluate(signal, df)
            
            # Log to database
            self.database.log_signal(signal, conviction)
            
            # Create trade card
            card = create_trade_card(signal, conviction)
            
            card_data = {
                'card': card,
                'signal': signal,
                'conviction': conviction,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            self.trade_cards.append(card_data)
            
            # Keep only last 50 cards
            if len(self.trade_cards) > 50:
                self.trade_cards = self.trade_cards[-50:]
            
            if self.on_signal:
                self.on_signal(card_data)
    
    def start(self):
        """Start live engine"""
        self.running = True
        self.feed.start()
        print("Live engine started")
        return self
    
    def stop(self):
        """Stop live engine"""
        self.running = False
        self.feed.stop()
        self.database.close()
        print("Live engine stopped")
    
    def get_prices(self) -> Dict:
        """Get current prices"""
        return self.feed.get_all_prices()
    
    def get_trade_cards(self, limit: int = 10) -> list:
        """Get recent trade cards"""
        return self.trade_cards[-limit:]
    
    def get_asset_state(self, symbol: str) -> Dict:
        """Get current state for an asset across all timeframes"""
        state = {'symbol': symbol, 'timeframes': {}}
        
        for tf in self.config.get('timeframes', []):
            key = f"{symbol}_{tf}"
            
            if key in self.candle_history and len(self.candle_history[key]) > 0:
                last_candle = self.candle_history[key][-1]
                state['timeframes'][tf] = {
                    'close': last_candle.get('close', 0),
                    'timestamp': str(last_candle.get('timestamp', ''))
                }
            else:
                state['timeframes'][tf] = {'signal': 'WAIT', 'close': 0}
        
        return state


# Watchdog thread
def watchdog(engine: LiveEngine, check_interval: int = 60):
    """Watchdog to monitor and restart engine if needed"""
    while engine.running:
        time.sleep(check_interval)
        
        # Check if feed is still connected
        prices = engine.get_prices()
        if not prices:
            print("Watchdog: No prices - reconnecting...")
            engine.feed.stop()
            time.sleep(5)
            engine.feed.start()
