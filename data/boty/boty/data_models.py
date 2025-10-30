# ============================================================================
# DATA MODELS
# ============================================================================

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any
from enum import Enum

class SignalAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Ticker:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: int


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    action: SignalAction
    confidence: float
    reason: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        return {
            'action': self.action.value,
            'confidence': self.confidence,
            'reason': self.reason,
            'metadata': self.metadata or {}
        }


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: Optional[float]
    opened_at: datetime
    unrealized_pnl: float = 0.0
    
    @property
    def notional(self):
        return self.entry_price * self.size


@dataclass
class Trade:
    symbol: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    duration: float  # hours
    exit_reason: str
    opened_at: datetime
    closed_at: datetime