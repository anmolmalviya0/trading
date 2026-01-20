"""
INSTITUTIONAL EXECUTION ENGINE
==============================
Implements professional execution:
- Limit Order Manager (no market orders)
- Dynamic order chasing with timeout
- Slippage model (2 basis points)
- Position sizing with Kelly criterion

Reference: Best practices from prop trading firms
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class OrderSide(Enum):
    BUY = 'BUY'
    SELL = 'SELL'


class OrderStatus(Enum):
    PENDING = 'PENDING'
    OPEN = 'OPEN'
    PARTIAL = 'PARTIAL'
    FILLED = 'FILLED'
    CANCELED = 'CANCELED'
    EXPIRED = 'EXPIRED'


class OrderType(Enum):
    LIMIT = 'LIMIT'
    LIMIT_IOC = 'LIMIT_IOC'  # Immediate or Cancel
    LIMIT_FOK = 'LIMIT_FOK'  # Fill or Kill


@dataclass
class Order:
    """Limit order representation"""
    id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    timeout_seconds: float = 30.0
    chase_enabled: bool = True
    chase_ticks: int = 2  # How many ticks to chase
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = self.created_at


@dataclass
class Fill:
    """Trade fill representation"""
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    timestamp: datetime
    fee: float
    slippage: float


@dataclass 
class Position:
    """Current position representation"""
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    entry_time: datetime = None


# =============================================================================
# SLIPPAGE MODEL
# =============================================================================

class SlippageModel:
    """
    Realistic slippage model for backtesting.
    
    Accounts for:
    - Base slippage (market impact)
    - Volume-dependent slippage
    - Volatility-dependent slippage
    """
    
    def __init__(self, 
                 base_bps: float = 2.0,      # Base 2 basis points
                 vol_factor: float = 0.5,    # Volatility multiplier
                 size_factor: float = 0.1):  # Size impact factor
        """
        Parameters:
        -----------
        base_bps : Base slippage in basis points (0.01% = 1bp)
        vol_factor : How much volatility amplifies slippage
        size_factor : How much order size amplifies slippage
        """
        self.base_bps = base_bps
        self.vol_factor = vol_factor
        self.size_factor = size_factor
    
    def calculate_slippage(self, 
                           price: float,
                           quantity: float,
                           side: OrderSide,
                           volatility: float = 0.02,
                           avg_volume: float = 1.0) -> Tuple[float, float]:
        """
        Calculate expected slippage for an order.
        
        Returns:
        --------
        Tuple of (fill_price, slippage_pct)
        """
        # Base slippage in percentage
        base_slip = self.base_bps / 10000
        
        # Volatility component
        vol_slip = base_slip * (volatility / 0.02) * self.vol_factor
        
        # Size impact (simplified)
        size_slip = base_slip * (quantity / avg_volume) * self.size_factor
        
        # Total slippage
        total_slip = base_slip + vol_slip + size_slip
        
        # Apply direction
        if side == OrderSide.BUY:
            fill_price = price * (1 + total_slip)  # Pay more
        else:
            fill_price = price * (1 - total_slip)  # Receive less
        
        return fill_price, total_slip * 100


# =============================================================================
# LIMIT ORDER MANAGER
# =============================================================================

class LimitOrderManager:
    """
    Institutional-grade limit order manager.
    
    Features:
    - No market orders (all limit)
    - Dynamic price chasing
    - Timeout logic
    - Order modification
    """
    
    def __init__(self,
                 slippage_model: SlippageModel = None,
                 default_timeout: float = 30.0,
                 chase_interval: float = 5.0,
                 max_chase_ticks: int = 3):
        """
        Parameters:
        -----------
        slippage_model : Model for slippage estimation
        default_timeout : Default order timeout in seconds
        chase_interval : Seconds between chase attempts
        max_chase_ticks : Maximum ticks to chase price
        """
        self.slippage_model = slippage_model or SlippageModel()
        self.default_timeout = default_timeout
        self.chase_interval = chase_interval
        self.max_chase_ticks = max_chase_ticks
        
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.order_counter = 0
    
    def _generate_order_id(self) -> str:
        self.order_counter += 1
        return f"ORD-{self.order_counter:06d}"
    
    def create_limit_order(self,
                           symbol: str,
                           side: OrderSide,
                           price: float,
                           quantity: float,
                           order_type: OrderType = OrderType.LIMIT,
                           timeout: float = None,
                           chase: bool = True) -> Order:
        """
        Create a new limit order.
        
        Parameters:
        -----------
        symbol : Trading pair
        side : BUY or SELL
        price : Limit price
        quantity : Order size
        order_type : LIMIT, LIMIT_IOC, LIMIT_FOK
        timeout : Override default timeout
        chase : Enable price chasing
        
        Returns:
        --------
        Order object
        """
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            order_type=order_type,
            status=OrderStatus.OPEN,
            timeout_seconds=timeout or self.default_timeout,
            chase_enabled=chase,
            chase_ticks=self.max_chase_ticks
        )
        
        self.orders[order.id] = order
        return order
    
    def modify_order_price(self, order_id: str, new_price: float) -> Order:
        """Modify order price (for chasing)"""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        if order.status not in [OrderStatus.OPEN, OrderStatus.PARTIAL]:
            raise ValueError(f"Cannot modify order in status {order.status}")
        
        order.price = new_price
        order.updated_at = datetime.now()
        
        return order
    
    def cancel_order(self, order_id: str) -> Order:
        """Cancel an open order"""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED]:
            return order
        
        order.status = OrderStatus.CANCELED
        order.updated_at = datetime.now()
        
        return order
    
    def simulate_fill(self,
                      order: Order,
                      market_price: float,
                      volatility: float = 0.02) -> Optional[Fill]:
        """
        Simulate order fill with slippage (for backtesting).
        
        Parameters:
        -----------
        order : Order to fill
        market_price : Current market price
        volatility : Current volatility
        
        Returns:
        --------
        Fill object if filled, None otherwise
        """
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED]:
            return None
        
        # Check if limit price is executable
        can_fill = False
        
        if order.side == OrderSide.BUY:
            can_fill = market_price <= order.price
        else:
            can_fill = market_price >= order.price
        
        if not can_fill:
            return None
        
        # Apply slippage
        fill_price, slippage_pct = self.slippage_model.calculate_slippage(
            order.price,
            order.quantity,
            order.side,
            volatility
        )
        
        # Calculate fee (0.1% = 10 bps)
        fee = fill_price * order.quantity * 0.001
        
        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            price=fill_price,
            quantity=order.quantity,
            timestamp=datetime.now(),
            fee=fee,
            slippage=slippage_pct
        )
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_qty = order.quantity
        order.avg_fill_price = fill_price
        order.updated_at = datetime.now()
        
        self.fills.append(fill)
        
        return fill
    
    def check_timeout(self, order: Order) -> bool:
        """Check if order has timed out"""
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED]:
            return False
        
        elapsed = (datetime.now() - order.created_at).total_seconds()
        
        if elapsed > order.timeout_seconds:
            order.status = OrderStatus.EXPIRED
            order.updated_at = datetime.now()
            return True
        
        return False
    
    def chase_price(self, order: Order, current_bid: float, current_ask: float) -> float:
        """
        Calculate new chase price.
        
        Returns:
        --------
        New limit price for chasing
        """
        if not order.chase_enabled or order.chase_ticks <= 0:
            return order.price
        
        # INSTITUTIONAL FIX: Use proper tick sizes from exchange
        # These are the actual tickSize values from Binance Exchange Info API
        TICK_SIZES = {
            'BTCUSDT': 0.01,    # BTC trades in $0.01 increments
            'PAXGUSDT': 0.01,   # PAXG trades in $0.01 increments
            'ETHUSDT': 0.01,
            'DEFAULT': 0.01
        }
        
        tick_size = TICK_SIZES.get(order.symbol.upper(), TICK_SIZES['DEFAULT'])
        
        if order.side == OrderSide.BUY:
            # Chase up toward ask
            new_price = min(order.price + tick_size, current_ask)
        else:
            # Chase down toward bid
            new_price = max(order.price - tick_size, current_bid)
        
        order.chase_ticks -= 1
        
        return new_price
    
    async def execute_with_chase(self,
                                  symbol: str,
                                  side: OrderSide,
                                  quantity: float,
                                  initial_price: float,
                                  get_market_price: callable) -> Optional[Fill]:
        """
        Execute order with dynamic price chasing.
        
        Parameters:
        -----------
        symbol : Trading pair
        side : BUY or SELL
        quantity : Order size
        initial_price : Starting limit price
        get_market_price : Function returning (bid, ask, last)
        
        Returns:
        --------
        Fill if successful, None otherwise
        """
        order = self.create_limit_order(
            symbol=symbol,
            side=side,
            price=initial_price,
            quantity=quantity,
            chase=True
        )
        
        print(f"   📝 Order created: {order.id} {side.value} {quantity} @ {initial_price:.2f}")
        
        start_time = datetime.now()
        chase_count = 0
        
        while order.status == OrderStatus.OPEN:
            # Check timeout
            if self.check_timeout(order):
                print(f"   ⏰ Order {order.id} expired")
                return None
            
            # Get current market
            bid, ask, last = get_market_price()
            
            # Try to fill
            fill = self.simulate_fill(order, last)
            
            if fill:
                print(f"   ✅ Filled: {fill.quantity} @ {fill.price:.2f} (slip: {fill.slippage:.3f}%)")
                return fill
            
            # Chase logic
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.chase_interval * (chase_count + 1):
                if order.chase_ticks > 0:
                    new_price = self.chase_price(order, bid, ask)
                    self.modify_order_price(order.id, new_price)
                    chase_count += 1
                    print(f"   🔄 Chasing to {new_price:.2f} ({order.chase_ticks} left)")
            
            await asyncio.sleep(0.5)
        
        return None


# =============================================================================
# POSITION MANAGER
# =============================================================================

class PositionManager:
    """
    Manages open positions with risk controls.
    """
    
    def __init__(self,
                 max_positions: int = 2,
                 max_position_size: float = 0.1,  # 10% of portfolio
                 default_stop_pct: float = 0.02,   # 2% stop loss
                 default_tp_pct: float = 0.04):    # 4% take profit
        
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.default_stop_pct = default_stop_pct
        self.default_tp_pct = default_tp_pct
        
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
    
    def can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position"""
        if symbol in self.positions:
            return False
        if len(self.positions) >= self.max_positions:
            return False
        return True
    
    def open_position(self,
                      symbol: str,
                      side: OrderSide,
                      entry_price: float,
                      quantity: float,
                      stop_loss: float = None,
                      take_profit: float = None) -> Position:
        """Open a new position"""
        if not self.can_open_position(symbol):
            raise ValueError(f"Cannot open position for {symbol}")
        
        # Default SL/TP
        if stop_loss is None:
            if side == OrderSide.BUY:
                stop_loss = entry_price * (1 - self.default_stop_pct)
            else:
                stop_loss = entry_price * (1 + self.default_stop_pct)
        
        if take_profit is None:
            if side == OrderSide.BUY:
                take_profit = entry_price * (1 + self.default_tp_pct)
            else:
                take_profit = entry_price * (1 - self.default_tp_pct)
        
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now()
        )
        
        self.positions[symbol] = position
        return position
    
    def update_pnl(self, symbol: str, current_price: float) -> float:
        """Update unrealized PnL for a position"""
        if symbol not in self.positions:
            return 0.0
        
        pos = self.positions[symbol]
        
        if pos.side == OrderSide.BUY:
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
        else:
            pos.unrealized_pnl = (pos.entry_price - current_price) * pos.quantity
        
        return pos.unrealized_pnl
    
    def check_exit_conditions(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if position should be exited"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        if pos.side == OrderSide.BUY:
            if current_price <= pos.stop_loss:
                return 'STOP_LOSS'
            if current_price >= pos.take_profit:
                return 'TAKE_PROFIT'
        else:
            if current_price >= pos.stop_loss:
                return 'STOP_LOSS'
            if current_price <= pos.take_profit:
                return 'TAKE_PROFIT'
        
        return None
    
    def close_position(self, symbol: str, exit_price: float, reason: str) -> Position:
        """Close a position"""
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")
        
        pos = self.positions[symbol]
        
        if pos.side == OrderSide.BUY:
            pos.realized_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pos.realized_pnl = (pos.entry_price - exit_price) * pos.quantity
        
        pos.unrealized_pnl = 0
        
        self.closed_positions.append(pos)
        del self.positions[symbol]
        
        return pos
    
    def get_stats(self) -> dict:
        """Get position statistics"""
        if not self.closed_positions:
            return {'total_pnl': 0, 'win_rate': 0, 'trades': 0}
        
        total_pnl = sum(p.realized_pnl for p in self.closed_positions)
        wins = sum(1 for p in self.closed_positions if p.realized_pnl > 0)
        win_rate = wins / len(self.closed_positions) * 100
        
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'trades': len(self.closed_positions),
            'wins': wins,
            'losses': len(self.closed_positions) - wins
        }


# =============================================================================
# KELLY POSITION SIZING
# =============================================================================

def kelly_criterion(win_rate: float, 
                    win_loss_ratio: float,
                    fraction: float = 0.5) -> float:
    """
    Calculate position size using Kelly Criterion.
    
    Parameters:
    -----------
    win_rate : Probability of winning (0-1)
    win_loss_ratio : Average win / Average loss
    fraction : Kelly fraction (0.5 = half Kelly, safer)
    
    Returns:
    --------
    Optimal position size as fraction of capital
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    # Kelly formula: f = (bp - q) / b
    # b = win/loss ratio
    # p = win probability
    # q = loss probability (1 - p)
    
    b = win_loss_ratio
    p = win_rate
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Apply fraction and cap
    position_size = max(0, min(kelly * fraction, 0.25))  # Max 25%
    
    return position_size


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("⚙️ INSTITUTIONAL EXECUTION ENGINE - Testing")
    print("="*70)
    
    # Test slippage model
    print("\n📊 Slippage Model Test:")
    slippage = SlippageModel(base_bps=2.0)
    
    fill_price, slip_pct = slippage.calculate_slippage(
        price=100000,
        quantity=1.0,
        side=OrderSide.BUY,
        volatility=0.02
    )
    print(f"   Buy at $100,000: Fill @ ${fill_price:.2f} (slip: {slip_pct:.4f}%)")
    
    fill_price, slip_pct = slippage.calculate_slippage(
        price=100000,
        quantity=1.0,
        side=OrderSide.SELL,
        volatility=0.02
    )
    print(f"   Sell at $100,000: Fill @ ${fill_price:.2f} (slip: {slip_pct:.4f}%)")
    
    # Test order manager
    print("\n📝 Order Manager Test:")
    manager = LimitOrderManager(slippage_model=slippage)
    
    order = manager.create_limit_order(
        symbol='BTCUSDT',
        side=OrderSide.BUY,
        price=99500,
        quantity=0.1
    )
    print(f"   Created: {order.id} @ {order.price}")
    
    # Simulate fill
    fill = manager.simulate_fill(order, market_price=99400, volatility=0.02)
    if fill:
        print(f"   Filled: {fill.price:.2f} (fee: ${fill.fee:.2f})")
    
    # Test position manager
    print("\n📈 Position Manager Test:")
    positions = PositionManager(default_stop_pct=0.02, default_tp_pct=0.04)
    
    pos = positions.open_position(
        symbol='BTCUSDT',
        side=OrderSide.BUY,
        entry_price=99500,
        quantity=0.1
    )
    print(f"   Opened: {pos.symbol} {pos.side.value} @ {pos.entry_price}")
    print(f"   SL: ${pos.stop_loss:.2f} | TP: ${pos.take_profit:.2f}")
    
    # Simulate price move
    exit_reason = positions.check_exit_conditions('BTCUSDT', 103500)
    if exit_reason:
        closed = positions.close_position('BTCUSDT', 103500, exit_reason)
        print(f"   Closed: {exit_reason} | PnL: ${closed.realized_pnl:.2f}")
    
    # Test Kelly
    print("\n🎲 Kelly Criterion Test:")
    size = kelly_criterion(win_rate=0.55, win_loss_ratio=2.0, fraction=0.5)
    print(f"   55% win rate, 2:1 R:R → Position: {size*100:.1f}% of capital")
    
    print("\n" + "="*70)
