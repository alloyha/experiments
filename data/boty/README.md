# Crypto Trading Bot - Setup & Testing Guide

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv trading_bot_env
source trading_bot_env/bin/activate  # On Windows: trading_bot_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Get Binance Testnet API Keys

1. Go to [Binance Testnet](https://testnet.binance.vision/)
2. Click "Generate HMAC_SHA256 Key"
3. Save your API Key and Secret Key
4. **IMPORTANT**: These are testnet keys (fake money). Keep them separate from real keys!

### Step 3: Project Structure

The project structure is already set up as follows:

```
boty/
├── __init__.py
├── config.py                   # Configuration (API keys)
├── connectors.py               # Exchange connector
├── core.py                     # Core components (state management, logging)
├── data_models.py              # Data models (Signal, Position, etc.)
├── main.py                     # Main orchestrator
├── phase1_validator.py         # Phase 1 validation suite
├── strategies.py               # Strategies & risk controller
├── tests/                      # Test scripts
│   ├── test_backtest.py        # Backtesting script
│   ├── test_components.py      # Component tests
│   └── ...
└── data/                       # For storing historical data
```

### Step 4: Create Configuration File

The `config.py` file is already included in the repository. Ensure your environment variables are set for API keys:

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

Run the component tests to validate individual modules:

```bash
python3 -m tests.test_components
```

### Test 2: Backtesting

Run the backtesting script to evaluate strategies:

```bash
python3 -m tests.test_backtest
```

---

## 📊 Phase-by-Phase Testing

### Phase 1: Paper Trading (Week 1-2)

1. **Run bot on testnet**:
```bash
python3 -m boty.main
```

2. **Monitor logs**:
```bash
tail -f trading_bot_testnet.log
```

3. **Check database**:
```bash
sqlite3 trading_bot_testnet.db

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