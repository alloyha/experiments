"""
Quick Strategy Test - Compare original vs improved strategies
"""

import asyncio
import pandas as pd
from pathlib import Path
import logging

from boty import EMACrossoverStrategy
from boty.backtest import Backtester, BacktestConfig
from boty.strategies import (
    EMACrossoverStrategy, 
    BollingerBandStrategy, 
    MomentumStrategy,
    RSIMeanReversionStrategy,
    ImprovedEMACrossover,
)

logging.basicConfig(level=logging.WARNING)  # Reduce noise


async def quick_comparison():
    """Compare all strategies quickly"""
    
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON - Finding What Actually Works")
    print("=" * 70)
    
    # Load data
    data_files = list(Path('./data').glob('*.csv'))
    if not data_files:
        print("❌ No data found. Run: python3 -m boty.download_data")
        return
    
    df = pd.read_csv(data_files[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\n📊 Data: {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Calculate buy-and-hold baseline
    bh_return = (df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close']
    print(f"💰 Buy & Hold Return: {bh_return*100:.2f}%")
    
    # Strategies to test
    strategies = [
        ("Original EMA (20/50)", EMACrossoverStrategy(fast_period=20, slow_period=50, rsi_overbought=65)),
        ("Improved EMA (12/26)", ImprovedEMACrossover(fast_period=12, slow_period=26)),
        ("Bollinger Bands", BollingerBandStrategy(period=20, std_dev=2.0)),
        ("Momentum", MomentumStrategy(momentum_period=10, pullback_period=3)),
        ("RSI Mean Reversion", RSIMeanReversionStrategy(rsi_period=14, oversold=30, overbought=70)),
    ]

    
    config = BacktestConfig(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_pct=0.0005
    )
    
    results = []
    
    print("\n" + "-" * 70)
    print("Testing strategies...")
    print("-" * 70)
    
    for name, strategy in strategies:
        print(f"\n{name}...", end=" ")
        
        backtester = Backtester(strategy, config)
        metrics = backtester.run(df, symbol='BTC/USDT')
        
        # Quick status
        if metrics.total_trades > 0:
            print(f"✅ {metrics.total_trades} trades, {metrics.total_return_pct*100:.2f}% return")
        else:
            print(f"❌ NO TRADES")
        
        results.append({
            'Strategy': name,
            'Return': metrics.total_return_pct * 100,
            'Annual': metrics.annual_return_pct * 100,
            'Trades': metrics.total_trades,
            'Win Rate': metrics.win_rate * 100,
            'Sharpe': metrics.sharpe_ratio,
            'Max DD': metrics.max_drawdown_pct * 100,
            'Profit Factor': metrics.profit_factor,
            'Avg Duration (h)': metrics.avg_trade_duration_hours,
            'vs B&H': (metrics.total_return_pct - bh_return) * 100
        })
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    df_results = pd.DataFrame(results)
    
    # Format for display
    df_display = df_results.copy()
    df_display['Return'] = df_display['Return'].apply(lambda x: f"{x:+.2f}%")
    df_display['Annual'] = df_display['Annual'].apply(lambda x: f"{x:+.2f}%")
    df_display['Win Rate'] = df_display['Win Rate'].apply(lambda x: f"{x:.1f}%")
    df_display['Sharpe'] = df_display['Sharpe'].apply(lambda x: f"{x:.2f}")
    df_display['Max DD'] = df_display['Max DD'].apply(lambda x: f"{x:.1f}%")
    df_display['Profit Factor'] = df_display['Profit Factor'].apply(lambda x: f"{x:.2f}")
    df_display['Avg Duration (h)'] = df_display['Avg Duration (h)'].apply(lambda x: f"{x:.1f}")
    df_display['vs B&H'] = df_display['vs B&H'].apply(lambda x: f"{x:+.2f}%")
    
    print("\n" + df_display.to_string(index=False))
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    # Find best by different metrics
    best_return = df_results.loc[df_results['Return'].idxmax()]
    best_sharpe = df_results.loc[df_results['Sharpe'].idxmax()]
    most_trades = df_results.loc[df_results['Trades'].idxmax()]
    
    print(f"\n🏆 Best Total Return: {best_return['Strategy']}")
    print(f"   Return: {best_return['Return']:.2f}%, Sharpe: {best_sharpe['Sharpe']:.2f}")
    
    print(f"\n📊 Best Risk-Adjusted: {best_sharpe['Strategy']}")
    print(f"   Sharpe: {best_sharpe['Sharpe']:.2f}, Return: {best_sharpe['Return']:.2f}%")
    
    print(f"\n⚡ Most Active: {most_trades['Strategy']}")
    print(f"   {most_trades['Trades']:.0f} trades, Win Rate: {most_trades['Win Rate']:.1f}%")
    
    # Reality check
    print("\n" + "=" * 70)
    print("REALITY CHECK")
    print("=" * 70)
    
    viable_strategies = df_results[
        (df_results['Trades'] >= 10) &  # At least 10 trades
        (df_results['Sharpe'] > 0.5) &  # Positive risk-adjusted returns
        (df_results['Max DD'] < 25)     # Max drawdown < 25%
    ]
    
    if len(viable_strategies) == 0:
        print("\n❌ NO STRATEGIES MEET MINIMUM CRITERIA")
        print("\nMinimum viable criteria:")
        print("  - At least 10 trades (statistical significance)")
        print("  - Sharpe ratio > 0.5 (positive risk-adjusted returns)")
        print("  - Max drawdown < 25% (manageable risk)")
        print("\nSuggestions:")
        print("  1. Try different timeframes (1h or 4h instead of 15m)")
        print("  2. Test on different time periods (2022, 2023, 2024)")
        print("  3. Consider that crypto is hard to predict algorithmically")
        print("  4. Buy-and-hold might be the better strategy")
    else:
        print(f"\n✅ {len(viable_strategies)} VIABLE STRATEGIES FOUND:")
        for _, strat in viable_strategies.iterrows():
            print(f"\n  {strat['Strategy']}")
            print(f"    Return: {strat['Return']:.2f}% | Sharpe: {strat['Sharpe']:.2f}")
            print(f"    Trades: {strat['Trades']:.0f} | Win Rate: {strat['Win Rate']:.1f}%")
            print(f"    Max DD: {strat['Max DD']:.1f}%")
        
        print("\n📌 NEXT STEPS:")
        print("  1. Pick the strategy with best Sharpe ratio")
        print("  2. Test on different time period (validation)")
        print("  3. If results hold → proceed to paper trading")
        print("  4. If results degrade → strategy is overfitted")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    asyncio.run(quick_comparison())