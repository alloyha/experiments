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