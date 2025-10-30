"""
Phase 1 Validation Suite
Tests all foundation components before moving to Phase 2
"""

import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from functools import wraps

# Assuming your package structure is:
# trading_bot/
#   __init__.py
#   core.py
#   data_models.py
#   connectors.py
#   strategies.py
#   config.py

try:
    from boty import (
        StateManager, Ticker, Candle, Signal, Position, Trade, SignalAction,
        BinanceConnector, EMACrossoverStrategy, RiskController, RiskParams
    )
    from boty.config import TESTNET_CONFIG
except ImportError:
    print("⚠️  Import failed. Make sure you're running from project root:")
    print("   python -m boty.phase1_validation")
    exit(1)


class Phase1Validator:
    """Comprehensive validation for Phase 1 components"""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def test(display_name: str, name: str | None = None):
        """Decorator factory to track test results.

        This is a staticmethod so it can be used as a decorator at class
        definition time (e.g. @test("desc", name="func_name")). The
        returned decorator expects the wrapped method to be an instance
        method and will use args[0] as `self` when the wrapper runs.
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Expect the instance to be the first positional arg when
                # the decorated method is called.
                instance = args[0] if args else None
                test_name = name or display_name
                try:
                    if instance is not None and hasattr(instance, 'logger'):
                        instance.logger.info(f"Running: {display_name}")
                    result = await func(*args, **kwargs)
                    if instance is not None and hasattr(instance, 'passed'):
                        instance.passed.append(test_name)
                    if instance is not None and hasattr(instance, 'logger'):
                        instance.logger.info(f"✅ PASSED: {display_name}")
                    return result
                except Exception as e:
                    if instance is not None and hasattr(instance, 'failed'):
                        instance.failed.append((test_name, str(e)))
                    if instance is not None and hasattr(instance, 'logger'):
                        instance.logger.error(f"❌ FAILED: {test_name} - {e}")
                    import traceback
                    traceback.print_exc()
            return wrapper
        return decorator
    
    async def run_all_tests(self):
        """Execute all validation tests"""
        self.logger.info("=" * 70)
        self.logger.info("PHASE 1 VALIDATION SUITE")
        self.logger.info("=" * 70)
        
        # Test 1: Data Models
        await self.test_data_models()
        
        # Test 2: State Manager
        await self.test_state_manager()
        
        # Test 3: Exchange Connector (if API keys available)
        await self.test_exchange_connector()
        
        # Test 4: Strategy Engine
        await self.test_strategy_engine()
        
        # Test 5: Risk Controller
        await self.test_risk_controller()
        
        # Test 6: Integration Test
        await self.test_integration()
        
        # Print summary
        self.print_summary()
    
    @test("Data Models - Signal Creation", name="test_data_models")
    async def test_data_models(self):
        """Test data model instantiation"""
        # Test Signal
        signal = Signal(
            action=SignalAction.BUY,
            confidence=0.8,
            reason="Test signal",
            metadata={'test': True}
        )
        assert signal.action == SignalAction.BUY
        assert signal.confidence == 0.8
        assert signal.to_dict()['action'] == 'BUY'
        
        # Test Position
        position = Position(
            symbol='BTC/USDT',
            side='LONG',
            entry_price=42000,
            size=0.1,
            stop_loss=41000,
            take_profit=45000,
            opened_at=datetime.now()
        )
        assert position.notional == 4200  # 42000 * 0.1
        
        # Test Trade
        trade = Trade(
            symbol='BTC/USDT',
            entry_price=42000,
            exit_price=43000,
            size=0.1,
            pnl=100,
            pnl_pct=2.38,
            duration=2.5,
            exit_reason='TAKE_PROFIT',
            opened_at=datetime.now(),
            closed_at=datetime.now()
        )
        assert trade.pnl > 0
        
        self.logger.info("  → All data models validated")
    
    @test("State Manager - Database Operations", name="test_state_manager")
    async def test_state_manager(self):
        """Test state persistence"""
        state = StateManager('./test_phase1.db')
        
        # Test decision logging
        signal = Signal(SignalAction.HOLD, 0.5, "Test")
        state.log_decision(
            signal=signal,
            symbol='BTC/USDT',
            action_taken='TEST',
            indicators={'ema_fast': 42000},
            portfolio_state={'cash': 10000}
        )
        
        # Test position save/load
        position = Position(
            symbol='BTC/USDT',
            side='LONG',
            entry_price=42000,
            size=0.1,
            stop_loss=41000,
            take_profit=45000,
            opened_at=datetime.now()
        )
        state.save_position(position)
        loaded = state.load_position('BTC/USDT')
        assert loaded is not None
        assert loaded.entry_price == 42000
        
        # Test trade save
        trade = Trade(
            symbol='BTC/USDT',
            entry_price=42000,
            exit_price=43000,
            size=0.1,
            pnl=100,
            pnl_pct=2.38,
            duration=2.5,
            exit_reason='TEST',
            opened_at=datetime.now(),
            closed_at=datetime.now()
        )
        state.save_trade(trade)
        
        # Test performance stats
        stats = state.get_performance_stats()
        assert 'total_trades' in stats
        
        # Cleanup
        state.clear_position('BTC/USDT')
        
        self.logger.info("  → Database operations validated")

    @test("Exchange Connector - API Integration", name="test_exchange_connector")
    async def test_exchange_connector(self):
        """Test exchange connectivity"""
        config = TESTNET_CONFIG
        
        # Check if API keys are configured
        if not config['api_key'] or 'your_testnet_key' in config['api_key']:
            self.logger.warning("  → Skipping (no API keys configured)")
            return
        
        connector = BinanceConnector(
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            testnet=True
        )
        
        # Test ticker fetch
        ticker = await connector.get_ticker('BTC/USDT')
        assert ticker.last > 0
        assert ticker.symbol == 'BTC/USDT'
        self.logger.info(f"  → Current BTC price: ${ticker.last:,.2f}")
        
        # Test candle fetch
        df = await connector.get_candles('BTC/USDT', '15m', limit=50)
        assert len(df) > 0
        assert 'close' in df.columns
        self.logger.info(f"  → Fetched {len(df)} candles")
        
        # Test balance fetch
        try:
            balance = await connector.get_balance()
            self.logger.info(f"  → Testnet balance: {balance}")
        except Exception as e:
            self.logger.warning(f"  → Balance fetch failed (expected on new testnet): {e}")
        
        self.logger.info("  → Exchange connector validated")
    
    @test("Strategy Engine - Signal Generation", name="test_strategy_engine")
    async def test_strategy_engine(self):
        """Test strategy logic"""
        # Create synthetic data
        np.random.seed(42)
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
        
        # Verify indicators were added
        assert 'ema_fast' in df.columns
        assert 'ema_slow' in df.columns
        assert 'rsi' in df.columns
        
        # Generate signal
        signal = strategy.generate_signal(df, current_position=None)
        assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]
        
        self.logger.info(f"  → Signal: {signal.action.value} (confidence={signal.confidence:.2f})")
        self.logger.info(f"  → Reason: {signal.reason}")
    
    @test("Risk Controller - Trade Validation", name="test_risk_controller")
    async def test_risk_controller(self):
        """Test risk management"""
        risk_params = RiskParams(
            max_position_size=1000,
            risk_per_trade=0.02,
            stop_loss_pct=0.02
        )
        
        risk_controller = RiskController(risk_params)
        
        # Test BUY validation
        signal = Signal(SignalAction.BUY, 0.8, "Test buy")
        validation = risk_controller.validate_trade(
            signal=signal,
            current_price=42000,
            portfolio_cash=10000,
            current_position=None
        )
        
        assert validation.approved
        assert validation.approved_size > 0
        self.logger.info(f"  → Approved position size: {validation.approved_size:.6f}")
        
        # Test stop loss calculation
        df = pd.DataFrame({
            'high': [42500] * 20,
            'low': [41500] * 20,
            'close': [42000] * 20
        })
        stop = risk_controller.calculate_stop_loss(42000, df)
        assert stop < 42000
        self.logger.info(f"  → Stop loss: ${stop:.2f} ({((42000-stop)/42000*100):.2f}% below entry)")
        
        # Test take profit
        tp = risk_controller.calculate_take_profit(42000)
        assert tp > 42000
        self.logger.info(f"  → Take profit: ${tp:.2f}")
        
        # Test daily loss limit
        risk_controller.update_daily_pnl(-600)  # Exceed limit
        validation2 = risk_controller.validate_trade(
            signal=signal,
            current_price=42000,
            portfolio_cash=10000,
            current_position=None
        )
        assert not validation2.approved
        self.logger.info("  → Daily loss limit enforced correctly")
    
    @test("Integration Test - Full Trading Cycle", name="test_integration")
    async def test_integration(self):
        """Test components working together"""
        # Initialize components
        state = StateManager('./test_integration.db')
        strategy = EMACrossoverStrategy(fast_period=5, slow_period=10)
        risk_controller = RiskController(RiskParams())
        
        # Create test data with clear golden cross
        np.random.seed(42)
        prices = np.concatenate([
            np.linspace(40000, 39000, 50),  # Downtrend
            np.linspace(39000, 42000, 50)   # Uptrend (should trigger golden cross)
        ])
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='15T'),
            'open': prices,
            'high': prices + 50,
            'low': prices - 50,
            'close': prices,
            'volume': np.random.rand(100) * 1000
        })
        
        # Calculate indicators
        df = strategy.calculate_indicators(df)
        
        # Generate signal
        signal = strategy.generate_signal(df, current_position=None)
        self.logger.info(f"  → Signal: {signal.action.value}")
        
        # Validate trade
        if signal.action == SignalAction.BUY:
            validation = risk_controller.validate_trade(
                signal=signal,
                current_price=prices[-1],
                portfolio_cash=10000,
                current_position=None
            )
            
            if validation.approved:
                # Simulate position opening
                position = Position(
                    symbol='BTC/USDT',
                    side='LONG',
                    entry_price=prices[-1],
                    size=validation.approved_size,
                    stop_loss=risk_controller.calculate_stop_loss(prices[-1], df),
                    take_profit=risk_controller.calculate_take_profit(prices[-1]),
                    opened_at=datetime.now()
                )
                
                state.save_position(position)
                
                # Log decision
                state.log_decision(
                    signal=signal,
                    symbol='BTC/USDT',
                    action_taken='EXECUTED_BUY',
                    indicators=strategy.get_indicators(df),
                    portfolio_state={'cash': 10000 - position.notional}
                )
                
                self.logger.info(f"  → Position opened: {position.size:.6f} @ ${position.entry_price:.2f}")
                self.logger.info(f"  → Stop: ${position.stop_loss:.2f}, Target: ${position.take_profit:.2f}")
                
                # Cleanup
                state.clear_position('BTC/USDT')
        
        self.logger.info("  → Full cycle integration validated")
    
    def print_summary(self):
        """Print test results summary"""
        self.logger.info("=" * 70)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"✅ Passed: {len(self.passed)}")
        self.logger.info(f"❌ Failed: {len(self.failed)}")
        
        if self.failed:
            self.logger.error("\nFailed tests:")
            for name, error in self.failed:
                self.logger.error(f"  - {name}: {error}")
            self.logger.error("\n⚠️  PHASE 1 INCOMPLETE - Fix failures before proceeding")
        else:
            self.logger.info("\n🎉 PHASE 1 COMPLETE!")
            self.logger.info("\nNext steps:")
            self.logger.info("  1. Configure testnet API keys in config.py")
            self.logger.info("  2. Run manual connectivity test:")
            self.logger.info("     python -m boty.test_exchange")
            self.logger.info("  3. Proceed to Phase 2: Backtesting Framework")
        
        self.logger.info("=" * 70)


async def main():
    """Run validation suite"""
    validator = Phase1Validator()
    await validator.run_all_tests()


if __name__ == '__main__':
    asyncio.run(main())