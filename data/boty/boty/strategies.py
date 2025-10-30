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

from .data_models import Position, Signal, SignalAction

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
    def generate_signal(self, df: pd.DataFrame, current_position: Optional[Position]) -> 'Signal':
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
        from .data_models import Signal, SignalAction
        
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


class ImprovedEMACrossover(Strategy):
    """
    Improved EMA Crossover with:
    - Looser filters (more trades)
    - Volume confirmation
    - Trend strength filter
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 volume_threshold: float = 1.2):
        super().__init__(name='EMA_Crossover')
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_threshold = volume_threshold
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # RSI (but we'll use it more loosely)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume indicator
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Trend strength (ADX-like)
        df['price_change'] = df['close'].pct_change(periods=20)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, current_position: Optional[Position]) -> Signal:
        if len(df) < 2:
            return Signal(SignalAction.HOLD, 0.0, "Insufficient data")
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Golden Cross (BUY)
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            if current_position is None:
                # Confidence based on multiple factors
                confidence = 0.5
                reasons = ["Golden cross"]
                
                # Boost confidence if RSI not overbought
                if last['rsi'] < 75:
                    confidence += 0.2
                    reasons.append(f"RSI neutral ({last['rsi']:.0f})")
                
                # Boost if volume is high
                if last['volume_ratio'] > self.volume_threshold:
                    confidence += 0.2
                    reasons.append(f"High volume ({last['volume_ratio']:.1f}x)")
                
                # Boost if price trending up
                if last['price_change'] > 0:
                    confidence += 0.1
                    reasons.append("Uptrend")
                
                return Signal(
                    action=SignalAction.BUY,
                    confidence=min(confidence, 1.0),
                    reason=", ".join(reasons),
                    metadata={
                        'ema_fast': last['ema_fast'],
                        'ema_slow': last['ema_slow'],
                        'rsi': last['rsi'],
                        'volume_ratio': last['volume_ratio']
                    }
                )
        
        # Death Cross (SELL)
        if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            if current_position is not None:
                return Signal(
                    action=SignalAction.SELL,
                    confidence=0.8,
                    reason="Death cross - exit position",
                    metadata={'ema_fast': last['ema_fast'], 'ema_slow': last['ema_slow']}
                )
        
        # Also exit if RSI extremely overbought
        if current_position is not None and last['rsi'] > 85:
            return Signal(
                action=SignalAction.SELL,
                confidence=0.7,
                reason=f"RSI extremely overbought ({last['rsi']:.0f})",
                metadata={'rsi': last['rsi']}
            )
        
        return Signal(SignalAction.HOLD, 0.0, "No signal")


class BollingerBandStrategy(Strategy):
    """
    Bollinger Band Mean Reversion
    - Buy on lower band touch
    - Sell on upper band touch
    - More active than EMA crossover
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(name='Bollinger_Bands')
        self.period = period
        self.std_dev = std_dev
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.period).mean()
        df['bb_std'] = df['close'].rolling(window=self.period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.std_dev)
        
        # Band width (volatility indicator)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Distance from bands
        df['dist_from_lower'] = (df['close'] - df['bb_lower']) / df['bb_lower']
        df['dist_from_upper'] = (df['bb_upper'] - df['close']) / df['bb_upper']
        
        # RSI for confirmation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, current_position: Optional[Position]) -> Signal:
        if len(df) < self.period + 1:
            return Signal(SignalAction.HOLD, 0.0, "Insufficient data")
        
        last = df.iloc[-1]
        
        # BUY: Price touches or crosses below lower band
        if last['close'] <= last['bb_lower'] * 1.01:  # Within 1% of lower band
            if current_position is None:
                confidence = 0.6
                
                # Higher confidence if RSI oversold
                if last['rsi'] < 30:
                    confidence = 0.9
                elif last['rsi'] < 40:
                    confidence = 0.7
                
                return Signal(
                    action=SignalAction.BUY,
                    confidence=confidence,
                    reason=f"Price at lower band (RSI={last['rsi']:.0f})",
                    metadata={
                        'close': last['close'],
                        'bb_lower': last['bb_lower'],
                        'rsi': last['rsi']
                    }
                )
        
        # SELL: Price touches or crosses above upper band
        if last['close'] >= last['bb_upper'] * 0.99:  # Within 1% of upper band
            if current_position is not None:
                return Signal(
                    action=SignalAction.SELL,
                    confidence=0.8,
                    reason=f"Price at upper band (RSI={last['rsi']:.0f})",
                    metadata={
                        'close': last['close'],
                        'bb_upper': last['bb_upper'],
                        'rsi': last['rsi']
                    }
                )
        
        # Also sell if back to middle band with profit
        if current_position is not None:
            if last['close'] > current_position.entry_price * 1.01:  # At least 1% profit
                if abs(last['close'] - last['bb_middle']) < last['bb_std'] * 0.5:
                    return Signal(
                        action=SignalAction.SELL,
                        confidence=0.6,
                        reason="Back to middle band with profit",
                        metadata={'close': last['close'], 'bb_middle': last['bb_middle']}
                    )
        
        return Signal(SignalAction.HOLD, 0.0, "Waiting for band touch")


class MomentumStrategy(Strategy):
    """
    Momentum-based strategy
    - Buy on strong momentum + pullback
    - More active, designed for trending markets
    """
    
    def __init__(self, momentum_period: int = 10, pullback_period: int = 3):
        super().__init__(name='Momentum')
        self.momentum_period = momentum_period
        self.pullback_period = pullback_period
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Momentum
        df['momentum'] = df['close'].pct_change(periods=self.momentum_period) * 100
        
        # Short-term pullback
        df['pullback'] = df['close'].pct_change(periods=self.pullback_period) * 100
        
        # Moving average for trend
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['above_ma'] = df['close'] > df['ma50']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, current_position: Optional[Position]) -> Signal:
        if len(df) < 50:
            return Signal(SignalAction.HOLD, 0.0, "Insufficient data")
        
        last = df.iloc[-1]
        
        # BUY: Strong momentum + pullback + above MA
        if (last['momentum'] > 3 and              # Strong uptrend (>3% in momentum_period)
            last['pullback'] < -0.5 and          # Recent pullback (<-0.5% in pullback_period)
            last['above_ma'] and                 # Above 50 MA (uptrend)
            last['rsi'] < 70 and                 # Not overbought
            current_position is None):
            
            return Signal(
                action=SignalAction.BUY,
                confidence=0.8,
                reason=f"Momentum {last['momentum']:.1f}%, pullback {last['pullback']:.1f}%",
                metadata={
                    'momentum': last['momentum'],
                    'pullback': last['pullback'],
                    'rsi': last['rsi']
                }
            )
        
        # SELL: Momentum fading or RSI overbought
        if current_position is not None:
            if last['momentum'] < 1 or last['rsi'] > 80:
                return Signal(
                    action=SignalAction.SELL,
                    confidence=0.7,
                    reason=f"Momentum fading ({last['momentum']:.1f}%) or overbought (RSI={last['rsi']:.0f})",
                    metadata={'momentum': last['momentum'], 'rsi': last['rsi']}
                )
        
        return Signal(SignalAction.HOLD, 0.0, "Waiting for setup")


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
    
    def generate_signal(self, df: pd.DataFrame, current_position: Optional[Position]) -> 'Signal':
        """Generate mean reversion signal"""
        from .core import Signal, SignalAction
        
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


class BollingerBandStrategy(Strategy):
    """
    Bollinger Band Mean Reversion
    - Buy on lower band touch
    - Sell on upper band touch
    - More active than EMA crossover
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(name='Bollinger_Bands')
        self.period = period
        self.std_dev = std_dev
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.period).mean()
        df['bb_std'] = df['close'].rolling(window=self.period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.std_dev)
        
        # Band width (volatility indicator)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Distance from bands
        df['dist_from_lower'] = (df['close'] - df['bb_lower']) / df['bb_lower']
        df['dist_from_upper'] = (df['bb_upper'] - df['close']) / df['bb_upper']
        
        # RSI for confirmation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, current_position: Optional[Position]) -> Signal:
        if len(df) < self.period + 1:
            return Signal(SignalAction.HOLD, 0.0, "Insufficient data")
        
        last = df.iloc[-1]
        
        # BUY: Price touches or crosses below lower band
        if last['close'] <= last['bb_lower'] * 1.01:  # Within 1% of lower band
            if current_position is None:
                confidence = 0.6
                
                # Higher confidence if RSI oversold
                if last['rsi'] < 30:
                    confidence = 0.9
                elif last['rsi'] < 40:
                    confidence = 0.7
                
                return Signal(
                    action=SignalAction.BUY,
                    confidence=confidence,
                    reason=f"Price at lower band (RSI={last['rsi']:.0f})",
                    metadata={
                        'close': last['close'],
                        'bb_lower': last['bb_lower'],
                        'rsi': last['rsi']
                    }
                )
        
        # SELL: Price touches or crosses above upper band
        if last['close'] >= last['bb_upper'] * 0.99:  # Within 1% of upper band
            if current_position is not None:
                return Signal(
                    action=SignalAction.SELL,
                    confidence=0.8,
                    reason=f"Price at upper band (RSI={last['rsi']:.0f})",
                    metadata={
                        'close': last['close'],
                        'bb_upper': last['bb_upper'],
                        'rsi': last['rsi']
                    }
                )
        
        # Also sell if back to middle band with profit
        if current_position is not None:
            if last['close'] > current_position.entry_price * 1.01:  # At least 1% profit
                if abs(last['close'] - last['bb_middle']) < last['bb_std'] * 0.5:
                    return Signal(
                        action=SignalAction.SELL,
                        confidence=0.6,
                        reason="Back to middle band with profit",
                        metadata={'close': last['close'], 'bb_middle': last['bb_middle']}
                    )
        
        return Signal(SignalAction.HOLD, 0.0, "Waiting for band touch")


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
                      portfolio_cash: float, current_position: Optional[Position]) -> ValidationResult:
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
    
    def check_exit_conditions(self, position: Position, current_price: float) -> Optional[str]:
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
    
    def calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        """Calculate current unrealized P&L"""
        if position.side == 'LONG':
            pnl = (current_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - current_price) * position.size
        
        return pnl
    
    def update_trailing_stop(self, position: Position, current_price: float) -> Optional[float]:
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
    'improved_ema': ImprovedEMACrossover,
    'rsi_mean_reversion': RSIMeanReversionStrategy,
    'bollinger_bands': BollingerBandStrategy,
    'momentum': MomentumStrategy
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