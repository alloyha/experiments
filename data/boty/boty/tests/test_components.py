import asyncio
import logging

from boty.core import BinanceConnector, StateManager, setup_logging
from boty.strategies import EMACrossoverStrategy, RiskController, RiskParams

async def test_exchange_connection():
    """Test 1: Can we connect to exchange?"""
    print("\n=== Test 1: Exchange Connection ===")
    
    connector = BinanceConnector(
        api_key='your_key',
        api_secret='your_secret',
        testnet=True
    )
    
    # Test ticker
    ticker = await connector.get_ticker('BTC/USDT')
    print(f"✅ BTC Price: ${ticker.last:,.2f}")
    
    # Test balance
    balance = await connector.get_balance()
    print(f"✅ Balance: {balance}")
    
    # Test candles
    df = await connector.get_candles('BTC/USDT', '15m', 50)
    print(f"✅ Fetched {len(df)} candles")
    print(f"   Latest close: ${df.iloc[-1]['close']:,.2f}")
    # Close connector to release aiohttp resources
    try:
        await connector.close()
    except Exception:
        pass

async def test_strategy():
    """Test 2: Does strategy generate signals?"""
    print("\n=== Test 2: Strategy Testing ===")
    
    connector = BinanceConnector('key', 'secret', testnet=True)
    df = await connector.get_candles('BTC/USDT', '15m', 200)
    
    strategy = EMACrossoverStrategy()
    df = strategy.calculate_indicators(df)
    signal = strategy.generate_signal(df, None)
    
    print(f"✅ Signal: {signal.action.value}")
    print(f"   Reason: {signal.reason}")
    print(f"   Confidence: {signal.confidence:.2f}")
    try:
        await connector.close()
    except Exception:
        pass

async def test_risk_controller():
    """Test 3: Does risk controller validate properly?"""
    print("\n=== Test 3: Risk Controller ===")
    
    from boty.data_models import Signal, SignalAction
    
    risk_params = RiskParams(max_position_size=1000, risk_per_trade=0.02)
    risk = RiskController(risk_params)
    
    signal = Signal(SignalAction.BUY, 0.8, "Test")
    validation = risk.validate_trade(signal, 42000, 10000, None)
    
    print(f"✅ Validation: {'APPROVED' if validation.approved else 'REJECTED'}")
    if validation.approved:
        print(f"   Approved size: {validation.approved_size:.6f}")

async def test_state_manager():
    """Test 4: Can we save/load state?"""
    print("\n=== Test 4: State Manager ===")
    
    from boty.data_models import Signal, SignalAction
    
    state = StateManager('./test.db')
    
    signal = Signal(SignalAction.HOLD, 0.0, "Test log")
    state.log_decision(signal, 'BTC/USDT', 'TEST', {}, {})
    
    print("✅ Decision logged to database")
    
    stats = state.get_performance_stats()
    print(f"   Total trades: {stats['total_trades']}")

async def run_all_tests():
    """Run all component tests"""
    setup_logging()
    
    print("\n" + "="*60)
    print("RUNNING COMPONENT TESTS")
    print("="*60)
    
    await test_exchange_connection()
    await test_strategy()
    await test_risk_controller()
    await test_state_manager()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)

if __name__ == '__main__':
    asyncio.run(run_all_tests())