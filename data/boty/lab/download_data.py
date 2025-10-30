"""
Historical Data Downloader
Downloads candle data from Binance for backtesting
"""


import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path


def download_binance_data(symbol='BTC/USDT', timeframe='15m', months=6):
    """
    Download historical data properly from Binance
    No API keys needed - public data
    """
    
    print(f"\n{'='*70}")
    print(f"DOWNLOADING {months} MONTHS OF {symbol} {timeframe} DATA")
    print('='*70)
    
    # Initialize exchange (no auth needed for public data)
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    # Calculate date range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=months * 30)
    
    print(f"\nPeriod: {start_time.date()} to {end_time.date()}")
    
    # Convert to milliseconds
    since = int(start_time.timestamp() * 1000)
    
    all_candles = []
    current_time = since
    
    # Download in chunks (1000 candles per request)
    chunk_count = 0
    
    while current_time < int(end_time.timestamp() * 1000):
        try:
            print(f"Fetching chunk {chunk_count + 1}...", end=" ")
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_time,
                limit=1000
            )
            
            if not ohlcv:
                print("No more data")
                break
            
            # Add to collection
            all_candles.extend(ohlcv)
            
            # Move to next chunk
            current_time = ohlcv[-1][0] + 1
            
            print(f"✅ Got {len(ohlcv)} candles (total: {len(all_candles)})")
            
            chunk_count += 1
            
            # Rate limiting
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
    
    # Convert to DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # Filter by date range
    df = df[
        (df['timestamp'] >= start_time) & 
        (df['timestamp'] <= end_time)
    ]
    
    print(f"\n{'='*70}")
    print(f"DOWNLOAD COMPLETE")
    print('='*70)
    print(f"Total candles: {len(df)}")
    print(f"First candle: {df['timestamp'].min()}")
    print(f"Last candle: {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    # Data quality checks
    print(f"\n📊 Data Quality:")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    print(f"  Average volume: {df['volume'].mean():.2f}")
    
    # Save to CSV
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    filename = f"{symbol.replace('/', '_')}_{timeframe}_{start_time.strftime('%Y-%m-%d')}_to_{end_time.strftime('%Y-%m-%d')}.csv"
    filepath = data_dir / filename
    
    df.to_csv(filepath, index=False)
    print(f"\n✅ Saved to: {filepath}")
    
    return df


def quick_analysis(df):
    """Quick analysis of downloaded data"""
    print(f"\n{'='*70}")
    print("QUICK MARKET ANALYSIS")
    print('='*70)
    
    # Overall return
    initial_price = df.iloc[0]['close']
    final_price = df.iloc[-1]['close']
    total_return = (final_price - initial_price) / initial_price * 100
    
    print(f"\n💰 Buy & Hold Performance:")
    print(f"  Initial price: ${initial_price:,.2f}")
    print(f"  Final price: ${final_price:,.2f}")
    print(f"  Return: {total_return:+.2f}%")
    
    # Volatility
    returns = df['close'].pct_change()
    volatility = returns.std() * 100
    
    print(f"\n📊 Market Statistics:")
    print(f"  Daily volatility: {volatility:.2f}%")
    print(f"  Max single-bar gain: {returns.max() * 100:.2f}%")
    print(f"  Max single-bar loss: {returns.min() * 100:.2f}%")
    
    # Trend analysis
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    
    above_ma50 = (df['close'] > df['ma50']).sum() / len(df) * 100
    above_ma200 = (df['close'] > df['ma200']).sum() / len(df) * 100
    
    print(f"\n📈 Trend Analysis:")
    print(f"  Time above 50-bar MA: {above_ma50:.1f}%")
    print(f"  Time above 200-bar MA: {above_ma200:.1f}%")
    
    if above_ma200 > 60:
        print(f"  → Strong uptrend period")
    elif above_ma200 < 40:
        print(f"  → Strong downtrend period")
    else:
        print(f"  → Sideways/mixed market")
    
    # Monthly breakdown
    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_returns = df.groupby('month').apply(
        lambda x: (x['close'].iloc[-1] - x['close'].iloc[0]) / x['close'].iloc[0] * 100
    )
    
    print(f"\n📅 Monthly Returns:")
    for month, ret in monthly_returns.items():
        emoji = "🟢" if ret > 0 else "🔴"
        print(f"  {emoji} {month}: {ret:+.2f}%")


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║          PROPER HISTORICAL DATA DOWNLOADER               ║
    ║                                                          ║
    ║  This will download 6 months of BTC/USDT 15m candles    ║
    ║  from Binance (no API keys needed - public data)        ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Download data
    df = download_binance_data(
        symbol='BTC/USDT',
        timeframe='15m',
        months=6
    )
    
    # Analyze it
    quick_analysis(df)
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print('='*70)
    print("""
    1. Run backtests again with this new data:
       python3 -m lab.quick_strategy_test
    
    2. Try different timeframes if results still poor:
       - 1h candles (less noise, fewer trades)
       - 4h candles (longer-term trends)
    
    3. Accept reality if all strategies lose:
       - Crypto might be too random for algo trading
       - Consider buy-and-hold or manual trading
       - Or focus on other markets (stocks, forex)
    """)