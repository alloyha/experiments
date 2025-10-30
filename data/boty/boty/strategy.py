"""
Crypto Trading Bot - Strategy Engine & Risk Controller
Phase 2: Signal Generation and Risk Management
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, List
import logging


# ============================================================================
# STRATEGY ENGINE
# ============================================================================

class Strategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to dataframe"""
        pass
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, current_position: Optional['Position']) -> 'Signal':
        """Generate trading signal"""
        pass
    
    def get_indicators(self, df: pd.DataFrame) -> Dict:
        """Extract current indicator values for logging"""
        if len(df) == 0:
            return {}
        
        last = df.iloc[-1]
        return {col: float(last[col]) for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']}


class EMACrossoverStrategy(Strategy):
    """
    Classic EMA Crossover Strategy
    - Buy: Fast EMA crosses above Slow EMA + RSI confirmation
    - Sell: Fast EMA crosses below Slow EMA
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 rsi_period: int = 14, rsi_overbought: float = 70):
        super().__init__(name='EMA_Crossover')
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA and RSI"""
        df = df.copy()
        
        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # Crossover detection
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(self, df: pd.DataFrame, current_position: Optional['Position']) -> 'Signal':
        """Generate BUY/SELL/HOLD signal"""
        from trading_bot_core import Signal, SignalAction
        
        if len(df) < 2:
            return Signal(SignalAction.HOLD, 0.0, "Insufficient data")
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Golden Cross (bullish)
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            if last['rsi'] < self.rsi_overbought and current_position is None:
                confidence = min(0.9, (last['ema_fast'] - last['ema_slow']) / last['ema_slow'] * 100)
                return Signal(
                    action=SignalAction.BUY,
                    confidence=confidence,
                    reason=f"Golden cross: EMA{self.fast_period} crossed above EMA{self.slow_period}, RSI={last['rsi']:.1f}",
                    metadata={
                        'ema_fast': last['ema_fast'],
                        'ema_slow': last['ema_slow'],
                        'rsi': last['rsi']
                    }
                )
        
        # Death Cross (bearish)
        if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            if current_position is not None:
                return Signal(
                    action=SignalAction.SELL,
                    confidence=0.8,
                    reason=f"Death cross: EMA{self.fast_period} crossed below EMA{self.slow_period}",
                    metadata={
                        'ema_fast': last['ema_fast'],
                        'ema_slow': last['ema_slow']
                    }
                )
        
        return Signal(SignalAction.HOLD, 0.0, "No crossover detected")


class RSIMeanReversionStrategy(Strategy):
    """
    RSI Mean Reversion Strategy
    - Buy: RSI oversold (< 30)
    - Sell: RSI overbought (> 70)
    """
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(name='RSI_MeanReversion')
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and Bollinger Bands"""
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands for context
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, current_position: Optional['Position']) -> 'Signal':
        """Generate mean reversion signal"""
        from trading_bot_core import Signal, SignalAction
        
        if len(df) < 2:
            return Signal(SignalAction.HOLD, 0.0, "Insufficient data")
        
        last = df.iloc[-1]
        
        # Buy on oversold
        if last['rsi'] < self.oversold and current_position is None:
            # Extra confirmation: price near lower BB
            near_lower_bb = last['close'] < last['bb_lower'] * 1.02
            confidence = 0.8 if near_lower_bb else 0.6
            
            return Signal(
                action=SignalAction.BUY,
                confidence=confidence,
                reason=f"RSI oversold: {last['rsi']:.1f} < {self.oversold}",
                metadata={'rsi': last['rsi'], 'bb_position': (last['close'] - last['bb_lower']) / (last['bb_upper'] - last['bb_lower'])}
            )
        
        # Sell on overbought
        if last['rsi'] > self.overbought and current_position is not None:
            return Signal(
                action=SignalAction.SELL,
                confidence=0.8,
                reason=f"RSI overbought: {last['rsi']:.1f} > {self.overbought}",
                metadata={'rsi': last['rsi']}
            )
        
        return Signal(SignalAction.HOLD, 0.0, f"RSI neutral: {last['rsi']:.1f}")


# ============================================================================
# RISK CONTROLLER
# ============================================================================

@dataclass
class RiskParams:
    """Risk management parameters"""
    max_position_size: float = 1000  # USDT
    max_portfolio_exposure: float = 5000  # USDT
    max_daily_loss: float = 500  # USDT
    risk_per_trade: float = 0.02  # 2% of capital
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    use_atr_stops: bool = True
    atr_multiplier: float = 2.0


@dataclass
class ValidationResult:
    approved: bool
    reasons: List[str]
    approved_size: float = 0.0


class RiskController:
    """Validate trades and manage risk"""
    
    def __init__(self, params: RiskParams):
        self.params = params
        self.daily_pnl = 0.0
        self.daily_reset_time = pd.Timestamp.now().normalize()
        self.logger = logging.getLogger(__name__)
    
    def reset_daily_pnl(self):
        """Reset daily P&L counter"""
        now = pd.Timestamp.now()
        if now.normalize() > self.daily_reset_time:
            self.daily_pnl = 0.0
            self.daily_reset_time = now.normalize()
            self.logger.info("Daily P&L counter reset")
    
    def validate_trade(self, signal: 'Signal', current_price: float, 
                      portfolio_cash: float, current_position: Optional['Position']) -> ValidationResult:
        """
        Pre-trade validation
        Returns: ValidationResult with approved size
        """
        self.reset_daily_pnl()
        reasons = []
        
        # Check 1: Only one position at a time (for simplicity)
        if signal.action.value == 'BUY' and current_position is not None:
            reasons.append("Already have an open position")
            return ValidationResult(approved=False, reasons=reasons)
        
        if signal.action.value == 'SELL' and current_position is None:
            reasons.append("No position to sell")
            return ValidationResult(approved=False, reasons=reasons)
        
        # For BUY signals, calculate position size
        if signal.action.value == 'BUY':
            # Calculate position size based on risk
            stop_distance_pct = self.params.stop_loss_pct
            risk_amount = portfolio_cash * self.params.risk_per_trade
            
            # Position size = risk amount / stop distance
            position_value = risk_amount / stop_distance_pct
            position_value = min(position_value, self.params.max_position_size)
            position_value = min(position_value, portfolio_cash * 0.95)  # Max 95% of cash
            
            # Check 2: Position size limit
            if position_value > self.params.max_position_size:
                reasons.append(f"Position size ${position_value:.2f} exceeds limit ${self.params.max_position_size}")
                return ValidationResult(approved=False, reasons=reasons)
            
            # Check 3: Daily loss limit
            if self.daily_pnl < -self.params.max_daily_loss:
                reasons.append(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
                return ValidationResult(approved=False, reasons=reasons)
            
            # Check 4: Sufficient cash
            if position_value > portfolio_cash:
                reasons.append(f"Insufficient cash: ${portfolio_cash:.2f} < ${position_value:.2f}")
                return ValidationResult(approved=False, reasons=reasons)
            
            # Approved
            position_size = position_value / current_price
            self.logger.info(f"Trade approved: ${position_value:.2f} ({position_size:.6f} units)")
            return ValidationResult(
                approved=True,
                reasons=["All risk checks passed"],
                approved_size=position_size
            )
        
        # For SELL signals, always approve (closing position)
        return ValidationResult(approved=True, reasons=["Closing position approved"])
    
    def calculate_stop_loss(self, entry_price: float, df: pd.DataFrame) -> float:
        """Calculate stop loss price"""
        if self.params.use_atr_stops and len(df) >= 14:
            # ATR-based stop
            atr = self._calculate_atr(df, period=14)
            stop_price = entry_price - (self.params.atr_multiplier * atr)
        else:
            # Percentage-based stop
            stop_price = entry_price * (1 - self.params.stop_loss_pct)
        
        self.logger.info(f"Stop loss set at ${stop_price:.2f} ({((entry_price - stop_price) / entry_price * 100):.2f}% below entry)")
        return stop_price
    
    def calculate_take_profit(self, entry_price: float) -> float:
        """Calculate take profit price"""
        take_profit = entry_price * (1 + self.params.take_profit_pct)
        self.logger.info(f"Take profit set at ${take_profit:.2f} ({self.params.take_profit_pct*100:.1f}% above entry)")
        return take_profit
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracker"""
        self.reset_daily_pnl()
        self.daily_pnl += pnl
        self.logger.info(f"Daily P&L updated: ${self.daily_pnl:.2f}")


# ============================================================================
# POSITION MANAGER
# ============================================================================

class PositionManager:
    """Manage open positions and check stop/target levels"""
    
    def __init__(self, risk_controller: RiskController):
        self.risk_controller = risk_controller
        self.logger = logging.getLogger(__name__)
    
    def check_exit_conditions(self, position: 'Position', current_price: float) -> Optional[str]:
        """
        Check if position should be closed
        Returns: exit_reason or None
        """
        # Check stop loss
        if current_price <= position.stop_loss:
            self.logger.warning(f"Stop loss hit! Price ${current_price:.2f} <= Stop ${position.stop_loss:.2f}")
            return 'STOP_LOSS'
        
        # Check take profit
        if position.take_profit and current_price >= position.take_profit:
            self.logger.info(f"Take profit hit! Price ${current_price:.2f} >= Target ${position.take_profit:.2f}")
            return 'TAKE_PROFIT'
        
        return None
    
    def calculate_unrealized_pnl(self, position: 'Position', current_price: float) -> float:
        """Calculate current unrealized P&L"""
        if position.side == 'LONG':
            pnl = (current_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - current_price) * position.size
        
        return pnl
    
    def update_trailing_stop(self, position: 'Position', current_price: float) -> Optional[float]:
        """
        Update trailing stop if price moved favorably
        Returns: new stop price or None
        """
        if position.side == 'LONG':
            # Calculate new trailing stop (e.g., 2% below current price)
            new_stop = current_price * 0.98
            
            # Only move stop up, never down
            if new_stop > position.stop_loss:
                self.logger.info(f"Trailing stop updated: ${position.stop_loss:.2f} -> ${new_stop:.2f}")
                return new_stop
        
        return None


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

STRATEGIES = {
    'ema_crossover': EMACrossoverStrategy,
    'rsi_mean_reversion': RSIMeanReversionStrategy
}


def get_strategy(name: str, **kwargs) -> Strategy:
    """Factory function to create strategies"""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    
    return STRATEGIES[name](**kwargs)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example: Test strategy on sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='15T')
    prices = 40000 + np.cumsum(np.random.randn(100) * 100)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.random.rand(100) * 50,
        'low': prices - np.random.rand(100) * 50,
        'close': prices,
        'volume': np.random.rand(100) * 1000
    })
    
    # Test EMA Crossover Strategy
    strategy = EMACrossoverStrategy(fast_period=5, slow_period=10)
    df = strategy.calculate_indicators(df)
    signal = strategy.generate_signal(df, current_position=None)
    
    print(f"\nSignal: {signal.action.value}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Reason: {signal.reason}")
    
    # Test Risk Controller
    risk_params = RiskParams(
        max_position_size=1000,
        risk_per_trade=0.02,
        stop_loss_pct=0.02
    )
    
    risk_controller = RiskController(risk_params)
    validation = risk_controller.validate_trade(
        signal=signal,
        current_price=prices[-1],
        portfolio_cash=10000,
        current_position=None
    )
    
    print(f"\nValidation: {'APPROVED' if validation.approved else 'REJECTED'}")
    print(f"Reasons: {validation.reasons}")
    if validation.approved:
        print(f"Approved size: {validation.approved_size:.6f}")