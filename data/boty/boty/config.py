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