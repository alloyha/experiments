# Crypto Trading Bot - Setup & Testing Guide

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv trading_bot_env
source trading_bot_env/bin/activate  # On Windows: trading_bot_env\Scripts\activate

# Install required packages
pip install ccxt pandas numpy sqlite3
```

### Step 2: Get Binance Testnet API Keys

1. Go to [Binance Testnet](https://testnet.binance.vision/)
2. Click "Generate HMAC_SHA256 Key"
3. Save your API Key and Secret Key
4. **IMPORTANT**: These are testnet keys (fake money). Keep them separate from real keys!

### Step 3: Project Structure

Create this folder structure:

```
trading_bot/
├── trading_bot_core.py        # Exchange connector & state management
├── trading_bot_strategy.py    # Strategies & risk controller
├── trading_bot_main.py         # Main orchestrator
├── config.py                   # Configuration (API keys)
├── test_backtest.py            # Backtesting script
└── data/                       # For storing historical data
```

### Step 4: Create Configuration File

Create `config.py`:

```python
# config.py
import os

# Testnet Configuration
TESTNET_CONFIG = {
    'symbol': 'BTC/USDT',
    'timeframe': '15m',
    'cycle_interval': 60,
    
    'strategy': 'ema_crossover',
    'strategy_params': {
        'fast_period': 12,
        'slow_period': 26
    },
    
    'risk_params': {
        'max_position_size': 1000,
        'risk_per_trade': 0.02,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.05
    },
    
    # Use environment variables for security
    'api_key': os.getenv('BINANCE_TESTNET_KEY', 'your_testnet_key_here'),
    'api_secret': os.getenv('BINANCE_TESTNET_SECRET', 'your_testnet_secret_here'),
    'testnet': True,
    
    'db_path': './trading_bot_testnet.db',
    'log_file': 'trading_bot_testnet.log'
}
```

---

## 🧪 Testing Phase

### Test 1: Component Testing

Create `test_components.py`:

```python
import asyncio
import logging
from trading_bot_core import BinanceConnector, StateManager, setup_logging
from trading_bot_strategy import EMACrossoverStrategy, RiskController, RiskParams

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

async def test_risk_controller():
    """Test 3: Does risk controller validate properly?"""
    print("\n=== Test 3: Risk Controller ===")
    
    from trading_bot_core import Signal, SignalAction
    
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
    
    from trading_bot_core import Signal, SignalAction
    
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
```

Run tests:
```bash
python test_components.py
```

### Test 2: Backtesting

Create `test_backtest.py`:

```python
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from trading_bot_core import BinanceConnector
from trading_bot_strategy import EMACrossoverStrategy
import matplotlib.pyplot as plt

async def download_historical_data(symbol='BTC/USDT', days=30):
    """Download historical data for backtesting"""
    print(f"Downloading {days} days of {symbol} data...")
    
    connector = BinanceConnector('key', 'secret', testnet=True)
    
    # Download in chunks (500 candles at a time)
    all_candles = []
    end_time = datetime.now()
    
    for _ in range(days * 96 // 500 + 1):  # 96 15-min candles per day
        df = await connector.get_candles(symbol, '15m', 500)
        all_candles.append(df)
        
        if len(all_candles) > days * 96 // 500:
            break
    
    full_df = pd.concat(all_candles).drop_duplicates()
    full_df = full_df.sort_values('timestamp').reset_index(drop=True)
    
    # Save to CSV
    full_df.to_csv(f'./data/{symbol.replace("/", "_")}_15m_{days}d.csv', index=False)
    print(f"✅ Saved {len(full_df)} candles to CSV")
    
    return full_df

def backtest_strategy(df, strategy):
    """Simple backtesting engine"""
    print("\n=== Running Backtest ===")
    
    # Track state
    position = None
    trades = []
    equity = 10000  # Starting capital
    equity_curve = [equity]
    
    # Simulate trading
    for i in range(100, len(df)):
        df_slice = df.iloc[:i+1].copy()
        df_slice = strategy.calculate_indicators(df_slice)
        
        current_price = df_slice.iloc[-1]['close']
        signal = strategy.generate_signal(df_slice, position)
        
        # Buy signal
        if signal.action.value == 'BUY' and position is None:
            size = (equity * 0.95) / current_price
            position = {
                'entry_price': current_price,
                'size': size,
                'entry_time': df_slice.iloc[-1]['timestamp']
            }
            print(f"BUY: {size:.6f} @ ${current_price:.2f}")
        
        # Sell signal
        elif signal.action.value == 'SELL' and position:
            exit_price = current_price
            pnl = (exit_price - position['entry_price']) * position['size']
            equity += pnl
            
            trades.append({
                'entry': position['entry_price'],
                'exit': exit_price,
                'pnl': pnl,
                'pnl_pct': (pnl / (position['entry_price'] * position['size'])) * 100
            })
            
            print(f"SELL: {position['size']:.6f} @ ${exit_price:.2f}, P&L: ${pnl:.2f}")
            position = None
        
        equity_curve.append(equity)
    
    # Calculate stats
    if trades:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        
        print("\n=== Backtest Results ===")
        print(f"Total Trades: {len(trades)}")
        print(f"Win Rate: {len(winning_trades) / len(trades):.1%}")
        print(f"Total P&L: ${sum(t['pnl'] for t in trades):.2f}")
        print(f"Final Equity: ${equity:.2f}")
        print(f"Return: {((equity - 10000) / 10000 * 100):.2f}%")
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.savefig('./backtest_equity_curve.png')
        print("✅ Equity curve saved to backtest_equity_curve.png")
    
    return trades, equity_curve

async def main():
    # Download data
    df = await download_historical_data('BTC/USDT', days=30)
    
    # Run backtest
    strategy = EMACrossoverStrategy(fast_period=12, slow_period=26)
    trades, equity_curve = backtest_strategy(df, strategy)

if __name__ == '__main__':
    asyncio.run(main())
```

Run backtest:
```bash
python test_backtest.py
```

---

## 📊 Phase-by-Phase Testing

### Phase 1: Paper Trading (Week 1-2)

1. **Run bot on testnet**:
```bash
python trading_bot_main.py
```

2. **Monitor logs**:
```bash
tail -f trading_bot_paper.log
```

3. **Check database**:
```bash
sqlite3 trading_bot_paper.db

# View decisions
SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 10;

# View trades
SELECT * FROM trades;
```

### Phase 2: Optimization (Week 3-4)

Test different parameters:

```python
# In config.py, try different strategies:

# Configuration 1: Aggressive
'strategy_params': {
    'fast_period': 8,
    'slow_period': 21
}

# Configuration 2: Conservative
'strategy_params': {
    'fast_period': 20,
    'slow_period': 50
}

# Configuration 3: RSI Strategy
'strategy': 'rsi_mean_reversion',
'strategy_params': {
    'rsi_period': 14,
    'oversold': 30,
    'overbought': 70
}
```

Compare results after 2 weeks of each.

---

## 🔍 Monitoring Checklist

### Daily Checks
- [ ] Bot is running (check logs)
- [ ] No errors in last 24h
- [ ] Review any trades executed
- [ ] Check daily P&L

### Weekly Analysis
- [ ] Calculate win rate
- [ ] Review largest losses
- [ ] Analyze signal quality
- [ ] Check if stops are being hit correctly

### Database Queries

```sql
-- Most profitable trades
SELECT * FROM trades ORDER BY pnl DESC LIMIT 10;

-- Worst trades
SELECT * FROM trades ORDER BY pnl ASC LIMIT 10;

-- Average trade duration
SELECT AVG(duration_hours) FROM trades;

-- Trades by exit reason
SELECT exit_reason, COUNT(*), AVG(pnl) 
FROM trades 
GROUP BY exit_reason;

-- Daily P&L
SELECT DATE(closed_at), SUM(pnl), COUNT(*) 
FROM trades 
GROUP BY DATE(closed_at);
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: "API Key Invalid"
**Solution**: Check that you're using testnet keys with `testnet=True`

### Issue 2: "Insufficient Balance"
**Solution**: Go to Binance Testnet and request free testnet USDT

### Issue 3: "No signals generated"
**Solution**: Market may be ranging. Try RSI strategy or adjust EMA periods.

### Issue 4: Bot keeps getting stopped out
**Solution**: Stop loss too tight. Increase `stop_loss_pct` or use ATR-based stops.

### Issue 5: "Rate limit exceeded"
**Solution**: Increase `cycle_interval` to reduce API calls.

---

## 🎯 Next Steps

Once you've run paper trading for 2+ weeks:

1. **Analyze Results**:
   - Is win rate > 45%?
   - Is max drawdown < 20%?
   - Are trades being executed properly?

2. **If Profitable**:
   - Consider live trading with $100-500
   - Monitor closely for first week

3. **If Not Profitable**:
   - Adjust strategy parameters
   - Try different strategies
   - Consider portfolio approach

---

## 📚 Additional Resources

- [CCXT Documentation](https://docs.ccxt.com/)
- [Binance API Docs](https://binance-docs.github.io/apidocs/spot/en/)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)

---

## 🆘 Getting Help

If you encounter issues:

1. Check the log file for error details
2. Query the database to see what's happening
3. Run component tests to isolate the problem
4. Review the decision log to see why signals aren't triggering

Remember: **Trading bots lose money sometimes. That's normal. The goal is to lose less than you win.**

Good luck! 🚀