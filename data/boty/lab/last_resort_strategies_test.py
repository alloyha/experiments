"""
Last Resort Strategy Ideas
When simple TA doesn't work, try these approaches
"""

import pandas as pd
import numpy as np
from typing import Optional

from boty import Strategy, Signal, SignalAction, Position


class LowFrequencyTrendFollower(Strategy):
    """
    MUCH lower frequency - only trade major trend changes
    Goal: 5-10 trades per year, not 200
    """
    
    def __init__(self):
        super().__init__(name='Low_Freq_Trend')
        self.ma_fast = 50   # Weekly
        self.ma_slow = 200  # Monthly
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Long-term moving averages
        df['ma_fast'] = df['close'].rolling(window=self.ma_fast).mean()
        df['ma_slow'] = df['close'].rolling(window=self.ma_slow).mean()
        
        # Trend strength
        df['trend_strength'] = (df['ma_fast'] - df['ma_slow']) / df['ma_slow'] * 100
        
        # Volatility filter (only trade when volatility is normal)
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, current_position: Optional[Position]) -> Signal:
        if len(df) < self.ma_slow + 1:
            return Signal(SignalAction.HOLD, 0.0, "Insufficient data")
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Only trade on MAJOR trend changes
        # Require 3% separation between MAs to avoid whipsaws
        
        if (prev['trend_strength'] <= -3 and 
            last['trend_strength'] > -2 and 
            current_position is None):
            # Major trend reversal from down to up
            return Signal(
                action=SignalAction.BUY,
                confidence=0.8,
                reason=f"Major bullish reversal (trend strength: {last['trend_strength']:.1f}%)",
                metadata={'trend_strength': last['trend_strength']}
            )
        
        if (current_position is not None and 
            last['trend_strength'] < -3):
            # Trend turned bearish
            return Signal(
                action=SignalAction.SELL,
                confidence=0.8,
                reason=f"Major bearish reversal (trend strength: {last['trend_strength']:.1f}%)",
                metadata={'trend_strength': last['trend_strength']}
            )
        
        return Signal(SignalAction.HOLD, 0.0, "Waiting for major trend change")


class BuyTheDipStrategy(Strategy):
    """
    Buy on significant drops, sell on recovery
    Goal: Capture panic selling / oversold bounces
    """
    
    def __init__(self, drop_threshold: float = -0.05, recovery_target: float = 0.03):
        super().__init__(name='Buy_The_Dip')
        self.drop_threshold = drop_threshold  # -5% drop triggers buy
        self.recovery_target = recovery_target  # +3% recovery triggers sell
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Short-term drops from recent high
        df['high_20'] = df['high'].rolling(window=20).max()
        df['drop_from_high'] = (df['close'] - df['high_20']) / df['high_20']
        
        # Recovery from low
        df['low_5'] = df['low'].rolling(window=5).min()
        df['recovery_from_low'] = (df['close'] - df['low_5']) / df['low_5']
        
        # Volume spike (panic selling confirmation)
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] / df['volume_ma']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, current_position: Optional[Position]) -> Signal:
        if len(df) < 20:
            return Signal(SignalAction.HOLD, 0.0, "Insufficient data")
        
        last = df.iloc[-1]
        
        # BUY: Significant drop + volume spike + oversold RSI
        if (last['drop_from_high'] < self.drop_threshold and
            last['volume_spike'] > 1.5 and
            last['rsi'] < 35 and
            current_position is None):
            
            confidence = 0.7
            # Boost confidence if really oversold
            if last['rsi'] < 25:
                confidence = 0.9
            
            return Signal(
                action=SignalAction.BUY,
                confidence=confidence,
                reason=f"Dip {last['drop_from_high']*100:.1f}% from high, RSI={last['rsi']:.0f}, Volume spike={last['volume_spike']:.1f}x",
                metadata={
                    'drop_pct': last['drop_from_high'] * 100,
                    'rsi': last['rsi'],
                    'volume_spike': last['volume_spike']
                }
            )
        
        # SELL: Price recovered from entry
        if current_position is not None:
            gain_from_entry = (last['close'] - current_position.entry_price) / current_position.entry_price
            
            if gain_from_entry > self.recovery_target:
                return Signal(
                    action=SignalAction.SELL,
                    confidence=0.8,
                    reason=f"Recovery target hit: {gain_from_entry*100:.1f}% gain",
                    metadata={'gain_pct': gain_from_entry * 100}
                )
        
        return Signal(SignalAction.HOLD, 0.0, "Waiting for significant dip")


class VolumeBreakoutStrategy(Strategy):
    """
    Trade on volume breakouts with price momentum
    Goal: Catch the start of big moves
    """
    
    def __init__(self, volume_threshold: float = 2.5, price_threshold: float = 0.02):
        super().__init__(name='Volume_Breakout')
        self.volume_threshold = volume_threshold  # 2.5x average volume
        self.price_threshold = price_threshold    # +2% price move
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Volume analysis
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        
        # Price momentum
        df['price_change'] = df['close'].pct_change(periods=1)
        df['price_change_5'] = df['close'].pct_change(periods=5)
        
        # Trend confirmation
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['above_ma'] = df['close'] > df['ma20']
        
        # RSI for overbought check
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, current_position: Optional[Position]) -> Signal:
        if len(df) < 20:
            return Signal(SignalAction.HOLD, 0.0, "Insufficient data")
        
        last = df.iloc[-1]
        
        # BUY: Volume spike + strong price move + uptrend
        if (last['volume_ratio'] > self.volume_threshold and
            last['price_change'] > self.price_threshold and
            last['above_ma'] and
            last['rsi'] < 75 and
            current_position is None):
            
            return Signal(
                action=SignalAction.BUY,
                confidence=0.8,
                reason=f"Volume breakout: {last['volume_ratio']:.1f}x avg, Price +{last['price_change']*100:.1f}%",
                metadata={
                    'volume_ratio': last['volume_ratio'],
                    'price_change': last['price_change'] * 100,
                    'rsi': last['rsi']
                }
            )
        
        # SELL: Momentum fading or RSI overbought
        if current_position is not None:
            if last['rsi'] > 80 or last['price_change_5'] < -0.02:
                return Signal(
                    action=SignalAction.SELL,
                    confidence=0.7,
                    reason=f"Momentum fading or overbought (RSI={last['rsi']:.0f})",
                    metadata={'rsi': last['rsi'], 'price_change_5': last['price_change_5'] * 100}
                )
        
        return Signal(SignalAction.HOLD, 0.0, "Waiting for volume breakout")


class PositionSizingStrategy(Strategy):
    """
    Simple EMA crossover but with dynamic position sizing
    Goal: Risk less on uncertain signals, more on confident ones
    """
    
    def __init__(self):
        super().__init__(name='Position_Sizing_EMA')
        self.fast_period = 12
        self.slow_period = 26
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Trend strength
        df['ema_distance'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow'] * 100
        
        # Volatility (for position sizing)
        df['volatility'] = df['close'].pct_change().rolling(20).std() * 100
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, current_position: Optional[Position]) -> Signal:
        if len(df) < 2:
            return Signal(SignalAction.HOLD, 0.0, "Insufficient data")
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Golden cross
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            if current_position is None:
                # Calculate confidence based on:
                # 1. Strength of crossover
                # 2. RSI level
                # 3. Volatility (inverse - prefer low vol)
                
                confidence = 0.3  # Base
                
                # Strong crossover (1%+ separation)
                if abs(last['ema_distance']) > 1:
                    confidence += 0.3
                elif abs(last['ema_distance']) > 0.5:
                    confidence += 0.2
                
                # Good RSI (not overbought)
                if last['rsi'] < 60:
                    confidence += 0.2
                elif last['rsi'] < 70:
                    confidence += 0.1
                
                # Low volatility (more predictable)
                if last['volatility'] < 2:
                    confidence += 0.2
                elif last['volatility'] < 3:
                    confidence += 0.1
                
                return Signal(
                    action=SignalAction.BUY,
                    confidence=min(confidence, 1.0),
                    reason=f"Golden cross (confidence={confidence:.0%}), EMA dist={last['ema_distance']:.2f}%, RSI={last['rsi']:.0f}",
                    metadata={
                        'ema_distance': last['ema_distance'],
                        'rsi': last['rsi'],
                        'volatility': last['volatility'],
                        'suggested_position_pct': confidence * 100  # Use this to scale position size
                    }
                )
        
        # Death cross
        if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            if current_position is not None:
                return Signal(
                    action=SignalAction.SELL,
                    confidence=0.8,
                    reason="Death cross - exit",
                    metadata={'ema_distance': last['ema_distance']}
                )
        
        return Signal(SignalAction.HOLD, 0.0, "No crossover")


# ============================================================================
# TEST THESE STRATEGIES
# ============================================================================

if __name__ == '__main__':
    """
    Quick test of last resort strategies
    Run with: python3 -m lab.last_resort_ideas
    """
    import asyncio
    from pathlib import Path
    from boty.backtest import Backtester, BacktestConfig
    
    print("\n" + "=" * 70)
    print("TESTING LAST RESORT STRATEGIES")
    print("=" * 70)
    print("\nThese are more sophisticated approaches:")
    print("  1. Low Frequency - Only major trend changes (5-10 trades/year)")
    print("  2. Buy The Dip - Panic selling bounces")
    print("  3. Volume Breakout - High volume + momentum")
    print("  4. Position Sizing - Dynamic allocation based on confidence")
    
    # Load data
    data_files = list(Path('./data').glob('*.csv'))
    if not data_files:
        print("\n❌ No data found. Run: python3 -m lab.fix_download")
        exit(1)
    
    df = pd.read_csv(data_files[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\n📊 Testing on {len(df)} bars from {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    bh_return = (df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close']
    print(f"💰 Buy & Hold Return: {bh_return*100:.2f}%\n")
    
    # Test strategies
    strategies = [
        ("Low Frequency Trend", LowFrequencyTrendFollower()),
        ("Buy The Dip", BuyTheDipStrategy(drop_threshold=-0.05, recovery_target=0.03)),
        ("Volume Breakout", VolumeBreakoutStrategy(volume_threshold=2.5, price_threshold=0.02)),
        ("Position Sizing EMA", PositionSizingStrategy()),
    ]
    
    config = BacktestConfig(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_pct=0.0005
    )
    
    results = []
    
    for name, strategy in strategies:
        print(f"Testing {name}...", end=" ")
        
        try:
            backtester = Backtester(strategy, config)
            metrics = backtester.run(df, symbol='BTC/USDT')
            
            if metrics.total_trades > 0:
                print(f"✅ {metrics.total_trades} trades, {metrics.total_return_pct*100:+.2f}% return")
            else:
                print(f"❌ NO TRADES")
            
            results.append({
                'Strategy': name,
                'Return %': metrics.total_return_pct * 100,
                'Trades': metrics.total_trades,
                'Win Rate': metrics.win_rate * 100,
                'Sharpe': metrics.sharpe_ratio,
                'Max DD %': metrics.max_drawdown_pct * 100,
                'vs B&H': (metrics.total_return_pct - bh_return) * 100
            })
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                'Strategy': name,
                'Return %': 0,
                'Trades': 0,
                'Win Rate': 0,
                'Sharpe': 0,
                'Max DD %': 0,
                'vs B&H': -bh_return * 100
            })
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    viable = df_results[
        (df_results['Trades'] >= 5) &
        (df_results['Sharpe'] > 0.5) &
        (df_results['Return %'] > 5)
    ]
    
    if len(viable) > 0:
        print("\n✅ VIABLE STRATEGIES FOUND:")
        for _, row in viable.iterrows():
            print(f"\n  {row['Strategy']}")
            print(f"    Return: {row['Return %']:.2f}%")
            print(f"    Trades: {row['Trades']:.0f}")
            print(f"    Sharpe: {row['Sharpe']:.2f}")
    else:
        print("\n❌ NO VIABLE STRATEGIES")
        print("\nEven these 'last resort' approaches don't beat buy-and-hold.")
        print("This confirms that simple algorithmic trading is not profitable.")
        print("\nRecommendation: Accept reality and buy-and-hold BTC/ETH.")
    
    print("\n" + "=" * 70)