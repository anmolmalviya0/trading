"""
MULTI-BROKER INTEGRATION
========================
Unified interface for multiple brokers:
- Alpaca (Stocks + Crypto)
- Interactive Brokers (Futures, Options)
- Binance (Crypto)

Usage:
    from multi_broker import BrokerFactory
    
    broker = BrokerFactory.create('alpaca')
    order = await broker.place_order('AAPL', 'buy', 10)
"""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import os
from typing import Optional, Dict, List
import aiohttp
import ssl
import certifi

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    id: str
    broker: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    filled_qty: float = 0
    filled_price: float = 0
    created_at: str = ""

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    market_price: float
    pnl: float
    pnl_pct: float

@dataclass
class AccountInfo:
    broker: str
    buying_power: float
    equity: float
    cash: float
    positions_count: int

# === ABSTRACT BROKER ===
class BaseBroker(ABC):
    """Abstract base class for broker adapters"""
    
    def __init__(self, api_key: str = None, secret: str = None):
        self.api_key = api_key or os.getenv(f'{self.name.upper()}_API_KEY', '')
        self.secret = secret or os.getenv(f'{self.name.upper()}_SECRET', '')
        self.session = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def supported_assets(self) -> List[str]:
        pass
    
    @abstractmethod
    async def connect(self):
        pass
    
    @abstractmethod
    async def get_account(self) -> AccountInfo:
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        pass
    
    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = None,
        stop_price: float = None
    ) -> Order:
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict:
        pass

# === ALPACA BROKER ===
class AlpacaBroker(BaseBroker):
    """
    Alpaca Broker Adapter
    Supports: US Stocks + Crypto
    API: https://alpaca.markets/docs/api-references/
    """
    
    BASE_URL = "https://api.alpaca.markets"
    PAPER_URL = "https://paper-api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"
    
    def __init__(self, paper: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.paper = paper
        self.base_url = self.PAPER_URL if paper else self.BASE_URL
    
    @property
    def name(self) -> str:
        return "alpaca"
    
    @property
    def supported_assets(self) -> List[str]:
        return ["stocks", "crypto"]
    
    async def connect(self):
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        self.session = aiohttp.ClientSession(
            connector=connector,
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret,
            }
        )
        print(f"✅ Alpaca connected ({'Paper' if self.paper else 'Live'})")
    
    async def get_account(self) -> AccountInfo:
        async with self.session.get(f"{self.base_url}/v2/account") as r:
            if r.status == 200:
                data = await r.json()
                return AccountInfo(
                    broker="alpaca",
                    buying_power=float(data['buying_power']),
                    equity=float(data['equity']),
                    cash=float(data['cash']),
                    positions_count=0
                )
            raise Exception(f"Failed to get account: {r.status}")
    
    async def get_positions(self) -> List[Position]:
        async with self.session.get(f"{self.base_url}/v2/positions") as r:
            if r.status == 200:
                data = await r.json()
                return [
                    Position(
                        symbol=p['symbol'],
                        quantity=float(p['qty']),
                        avg_price=float(p['avg_entry_price']),
                        market_price=float(p['current_price']),
                        pnl=float(p['unrealized_pl']),
                        pnl_pct=float(p['unrealized_plpc']) * 100
                    )
                    for p in data
                ]
            return []
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = None,
        stop_price: float = None
    ) -> Order:
        payload = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": side.value,
            "type": order_type.value,
            "time_in_force": "day"
        }
        
        if order_type == OrderType.LIMIT and price:
            payload["limit_price"] = str(price)
        
        if stop_price:
            payload["stop_price"] = str(stop_price)
        
        async with self.session.post(f"{self.base_url}/v2/orders", json=payload) as r:
            if r.status in [200, 201]:
                data = await r.json()
                return Order(
                    id=data['id'],
                    broker="alpaca",
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.PENDING,
                    created_at=data['created_at']
                )
            error = await r.text()
            raise Exception(f"Order failed: {error}")
    
    async def cancel_order(self, order_id: str) -> bool:
        async with self.session.delete(f"{self.base_url}/v2/orders/{order_id}") as r:
            return r.status == 204
    
    async def get_quote(self, symbol: str) -> Dict:
        # Check if crypto
        if '/' in symbol or symbol.endswith('USD'):
            url = f"{self.DATA_URL}/v1beta3/crypto/us/latest/quotes?symbols={symbol}"
        else:
            url = f"{self.DATA_URL}/v2/stocks/{symbol}/quotes/latest"
        
        async with self.session.get(url) as r:
            if r.status == 200:
                data = await r.json()
                return data
            return {}

# === INTERACTIVE BROKERS ===
class IBKRBroker(BaseBroker):
    """
    Interactive Brokers Adapter
    Supports: Stocks, Futures, Options, Forex
    
    Note: Requires IB Gateway or TWS running locally
    """
    
    def __init__(self, gateway_port: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.gateway_url = f"https://localhost:{gateway_port}/v1/api"
        self.account_id = None
    
    @property
    def name(self) -> str:
        return "ibkr"
    
    @property
    def supported_assets(self) -> List[str]:
        return ["stocks", "futures", "options", "forex"]
    
    async def connect(self):
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE  # IB uses self-signed
        
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        self.session = aiohttp.ClientSession(connector=connector)
        
        # Get account ID
        async with self.session.get(f"{self.gateway_url}/portfolio/accounts") as r:
            if r.status == 200:
                data = await r.json()
                if data:
                    self.account_id = data[0]['id']
                    print(f"✅ IBKR connected: {self.account_id}")
                    return
        print("⚠️ IBKR connection requires Gateway/TWS running")
    
    async def get_account(self) -> AccountInfo:
        if not self.account_id:
            raise Exception("Not connected")
        
        async with self.session.get(
            f"{self.gateway_url}/portfolio/{self.account_id}/summary"
        ) as r:
            if r.status == 200:
                data = await r.json()
                return AccountInfo(
                    broker="ibkr",
                    buying_power=data.get('buyingPower', {}).get('amount', 0),
                    equity=data.get('equity', {}).get('amount', 0),
                    cash=data.get('cash', {}).get('amount', 0),
                    positions_count=0
                )
        return AccountInfo(broker="ibkr", buying_power=0, equity=0, cash=0, positions_count=0)
    
    async def get_positions(self) -> List[Position]:
        if not self.account_id:
            return []
        
        async with self.session.get(
            f"{self.gateway_url}/portfolio/{self.account_id}/positions"
        ) as r:
            if r.status == 200:
                data = await r.json()
                return [
                    Position(
                        symbol=p.get('contractDesc', 'UNKNOWN'),
                        quantity=p.get('position', 0),
                        avg_price=p.get('avgCost', 0),
                        market_price=p.get('mktPrice', 0),
                        pnl=p.get('unrealizedPnl', 0),
                        pnl_pct=0
                    )
                    for p in data
                ]
        return []
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = None,
        stop_price: float = None
    ) -> Order:
        # IBKR requires contract ID lookup first
        # This is a simplified version
        payload = {
            "orders": [{
                "acctId": self.account_id,
                "conid": 0,  # Would need contract lookup
                "orderType": order_type.value.upper(),
                "side": side.value.upper(),
                "quantity": quantity,
                "tif": "DAY"
            }]
        }
        
        if price:
            payload["orders"][0]["price"] = price
        
        # Note: Actual IBKR implementation requires more steps
        return Order(
            id="simulated",
            broker="ibkr",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        async with self.session.delete(
            f"{self.gateway_url}/iserver/account/{self.account_id}/order/{order_id}"
        ) as r:
            return r.status == 200
    
    async def get_quote(self, symbol: str) -> Dict:
        # IBKR requires contract ID
        return {"symbol": symbol, "price": 0, "note": "Requires contract lookup"}

# === BINANCE BROKER ===
class BinanceBroker(BaseBroker):
    """
    Binance Broker Adapter
    Supports: Crypto Spot + Futures
    """
    
    BASE_URL = "https://api.binance.com"
    
    @property
    def name(self) -> str:
        return "binance"
    
    @property
    def supported_assets(self) -> List[str]:
        return ["crypto"]
    
    async def connect(self):
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        self.session = aiohttp.ClientSession(connector=connector)
        print("✅ Binance connected")
    
    async def get_account(self) -> AccountInfo:
        # Requires signed request
        return AccountInfo(
            broker="binance",
            buying_power=0,
            equity=0,
            cash=0,
            positions_count=0
        )
    
    async def get_positions(self) -> List[Position]:
        # Requires signed request
        return []
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = None,
        stop_price: float = None
    ) -> Order:
        # Requires HMAC signature
        print(f"📝 Binance order: {side.value} {quantity} {symbol}")
        return Order(
            id="simulated",
            broker="binance",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        return True
    
    async def get_quote(self, symbol: str) -> Dict:
        async with self.session.get(
            f"{self.BASE_URL}/api/v3/ticker/price?symbol={symbol}"
        ) as r:
            if r.status == 200:
                data = await r.json()
                return {"symbol": symbol, "price": float(data['price'])}
        return {}

# === BROKER FACTORY ===
class BrokerFactory:
    """Factory for creating broker instances"""
    
    _brokers = {
        'alpaca': AlpacaBroker,
        'ibkr': IBKRBroker,
        'binance': BinanceBroker,
    }
    
    @classmethod
    def create(cls, broker_name: str, **kwargs) -> BaseBroker:
        """Create a broker instance"""
        if broker_name not in cls._brokers:
            raise ValueError(f"Unknown broker: {broker_name}")
        return cls._brokers[broker_name](**kwargs)
    
    @classmethod
    def list_brokers(cls) -> List[str]:
        """List available brokers"""
        return list(cls._brokers.keys())

# === UNIFIED TRADING INTERFACE ===
class UnifiedTrader:
    """
    Unified interface to trade across multiple brokers
    Automatically routes orders to the appropriate broker
    """
    
    def __init__(self):
        self.brokers: Dict[str, BaseBroker] = {}
        self.routing_rules = {
            'BTCUSDT': 'binance',
            'ETHUSDT': 'binance',
            'PAXGUSDT': 'binance',
            'AAPL': 'alpaca',
            'TSLA': 'alpaca',
            'SPY': 'alpaca',
            'ES': 'ibkr',  # E-mini S&P Futures
            'NQ': 'ibkr',  # Nasdaq Futures
        }
    
    async def add_broker(self, name: str, **kwargs):
        """Add and connect a broker"""
        broker = BrokerFactory.create(name, **kwargs)
        await broker.connect()
        self.brokers[name] = broker
    
    def get_broker_for_symbol(self, symbol: str) -> str:
        """Determine which broker to use for a symbol"""
        if symbol in self.routing_rules:
            return self.routing_rules[symbol]
        
        # Default logic
        if symbol.endswith('USDT') or symbol.endswith('USD'):
            return 'binance'
        return 'alpaca'
    
    async def trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float = None
    ) -> Order:
        """Place a trade through the appropriate broker"""
        broker_name = self.get_broker_for_symbol(symbol)
        
        if broker_name not in self.brokers:
            raise Exception(f"Broker {broker_name} not connected")
        
        broker = self.brokers[broker_name]
        side_enum = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        order_type = OrderType.LIMIT if price else OrderType.MARKET
        
        return await broker.place_order(
            symbol=symbol,
            side=side_enum,
            quantity=quantity,
            order_type=order_type,
            price=price
        )
    
    async def get_all_positions(self) -> Dict[str, List[Position]]:
        """Get positions from all connected brokers"""
        all_positions = {}
        for name, broker in self.brokers.items():
            positions = await broker.get_positions()
            all_positions[name] = positions
        return all_positions
    
    async def get_portfolio_summary(self) -> Dict:
        """Get unified portfolio summary"""
        total_equity = 0
        total_pnl = 0
        
        for name, broker in self.brokers.items():
            try:
                account = await broker.get_account()
                total_equity += account.equity
            except:
                pass
        
        positions = await self.get_all_positions()
        for broker_positions in positions.values():
            for pos in broker_positions:
                total_pnl += pos.pnl
        
        return {
            'total_equity': total_equity,
            'total_pnl': total_pnl,
            'brokers_connected': list(self.brokers.keys()),
            'positions': positions
        }

# === DEMO ===
async def demo():
    """Demo multi-broker functionality"""
    print("="*60)
    print("🏦 MULTI-BROKER INTEGRATION DEMO")
    print("="*60)
    
    trader = UnifiedTrader()
    
    # Connect Binance (no API key needed for quotes)
    await trader.add_broker('binance')
    
    # Get quote
    btc_quote = await trader.brokers['binance'].get_quote('BTCUSDT')
    print(f"\n📊 BTC Price: ${btc_quote.get('price', 'N/A')}")
    
    # Show routing
    print("\n📍 Order Routing:")
    for symbol in ['BTCUSDT', 'AAPL', 'ES', 'ETHUSDT']:
        broker = trader.get_broker_for_symbol(symbol)
        print(f"   {symbol} → {broker}")
    
    print("\n✅ Multi-broker system ready")

if __name__ == "__main__":
    asyncio.run(demo())
