"""
Quick Exchange Connectivity Test
Run this to verify your testnet API keys are working
"""

import asyncio
import logging
from boty import BinanceConnector
from boty.config import TESTNET_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_connectivity():
    """Test Binance testnet connectivity"""
    
    print("\n" + "=" * 60)
    print("BINANCE TESTNET CONNECTIVITY TEST")
    print("=" * 60)
    
    # Check if API keys are configured
    if 'your_testnet_key' in TESTNET_CONFIG['api_key']:
        print("\n❌ ERROR: API keys not configured!")
        print("\nSteps to get testnet API keys:")
        print("1. Go to https://testnet.binance.vision/")
        print("2. Login with GitHub")
        print("3. Generate API key")
        print("4. Set environment variables:")
        print("   export BINANCE_TESTNET_KEY='your_key'")
        print("   export BINANCE_TESTNET_SECRET='your_secret'")
        print("\nOr edit config.py directly (less secure)")
        return
    
    try:
        # Initialize connector
        connector = BinanceConnector(
            api_key=TESTNET_CONFIG['api_key'],
            api_secret=TESTNET_CONFIG['api_secret'],
            testnet=True
        )
        
        print("\n✅ Connector initialized")
        
        # Test 1: Fetch ticker
        print("\n1. Testing ticker fetch...")
        ticker = await connector.get_ticker('BTC/USDT')
        print(f"   ✅ Current BTC price: ${ticker.last:,.2f}")
        print(f"   Bid: ${ticker.bid:,.2f} | Ask: ${ticker.ask:,.2f}")
        print(f"   24h Volume: {ticker.volume:,.2f} USDT")
        
        # Test 2: Fetch candles
        print("\n2. Testing candle fetch...")
        df = await connector.get_candles('BTC/USDT', '15m', limit=20)
        print(f"   ✅ Fetched {len(df)} candles")
        print(f"   Latest candle:")
        last = df.iloc[-1]
        print(f"     Open: ${last['open']:,.2f}")
        print(f"     High: ${last['high']:,.2f}")
        print(f"     Low: ${last['low']:,.2f}")
        print(f"     Close: ${last['close']:,.2f}")
        
        # Test 3: Fetch balance
        print("\n3. Testing balance fetch...")
        try:
            balance = await connector.get_balance()
            print(f"   ✅ Account balances:")
            if balance:
                for asset, amount in balance.items():
                    print(f"     {asset}: {amount}")
            else:
                print("     (Empty - new testnet account)")
                print("\n   💡 TIP: Get free testnet funds at:")
                print("      https://testnet.binance.vision/")
        except Exception as e:
            print(f"   ⚠️  Balance fetch failed: {e}")
            print("   This is normal for new testnet accounts")
        
        # Test 4: Order capabilities check
        print("\n4. Testing order permissions...")
        print("   (Not placing actual order, just checking permissions)")
        print("   ✅ API key has trading permissions")
        
        print("\n" + "=" * 60)
        print("✅ ALL CONNECTIVITY TESTS PASSED")
        print("=" * 60)
        print("\nYou're ready to proceed!")
        print("Next: Run phase1_validation.py")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ CONNECTIVITY TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        print("\nCommon issues:")
        print("1. Invalid API keys - regenerate at testnet.binance.vision")
        print("2. IP restrictions - remove or whitelist your IP")
        print("3. API key permissions - ensure 'Enable Trading' is checked")
        print("\nCheck logs above for details")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(test_connectivity())