"""
Backtest Diagnostics - Find why strategy isn't trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from boty import EMACrossoverStrategy


class BacktestDiagnostics:
    """Debug why strategy generates so few signals"""
    
    @staticmethod
    def analyze_signals(df: pd.DataFrame, strategy: EMACrossoverStrategy):
        """Detailed signal analysis"""
        print("\n" + "=" * 70)
        print("SIGNAL GENERATION DIAGNOSTICS")
        print("=" * 70)
        
        # Calculate indicators
        df = strategy.calculate_indicators(df)
        
        # 1. Check crossover frequency
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['crossover'] = (
            (df['ema_diff'].shift(1) <= 0) & (df['ema_diff'] > 0)  # Golden cross
        ) | (
            (df['ema_diff'].shift(1) >= 0) & (df['ema_diff'] < 0)  # Death cross
        )
        
        crossover_count = df['crossover'].sum()
        
        print(f"\n📊 CROSSOVER ANALYSIS")
        print(f"  Total EMA crossovers: {crossover_count}")
        print(f"  Crossover frequency: {crossover_count / len(df) * 100:.2f}% of bars")
        print(f"  Average days between crossovers: {len(df) / crossover_count / 96:.1f} days" if crossover_count > 0 else "  N/A")
        
        # 2. Check RSI filter impact
        if 'rsi' in df.columns:
            golden_crosses = df[
                (df['ema_diff'].shift(1) <= 0) & (df['ema_diff'] > 0)
            ]
            
            if len(golden_crosses) > 0:
                rsi_pass = golden_crosses[golden_crosses['rsi'] < strategy.rsi_overbought]
                rsi_block = golden_crosses[golden_crosses['rsi'] >= strategy.rsi_overbought]
                
                print(f"\n🚦 RSI FILTER IMPACT")
                print(f"  Golden crosses detected: {len(golden_crosses)}")
                print(f"  Passed RSI filter (<{strategy.rsi_overbought}): {len(rsi_pass)}")
                print(f"  Blocked by RSI: {len(rsi_block)} ({len(rsi_block)/len(golden_crosses)*100:.1f}%)")
                
                if len(rsi_block) > 0:
                    print(f"  ⚠️  WARNING: RSI filter is blocking {len(rsi_block)} potential buy signals!")
                    print(f"     Consider raising threshold to 75-80 or removing filter")
        
        # 3. Check price trend
        df['price_change'] = df['close'].pct_change(periods=96)  # 24h change
        
        print(f"\n📈 MARKET REGIME")
        print(f"  Average 24h change: {df['price_change'].mean()*100:.2f}%")
        print(f"  Volatility (std): {df['price_change'].std()*100:.2f}%")
        
        trending_up = (df['close'] > df['close'].shift(96)).sum() / len(df)
        print(f"  Uptrend % of time: {trending_up*100:.1f}%")
        
        if trending_up < 0.4:
            print(f"  ⚠️  WARNING: Market was mostly sideways/down - EMA crossover struggles here")
        
        # 4. Visualize indicators
        BacktestDiagnostics._plot_indicators(df, strategy)
        
        return df
    
    @staticmethod
    def _plot_indicators(df: pd.DataFrame, strategy):
        """Plot price with indicators"""
        # Sample last 500 bars for readability
        df_plot = df.tail(500).copy()
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. Price + EMAs
        axes[0].plot(df_plot['timestamp'], df_plot['close'], label='Price', linewidth=1, alpha=0.7)
        axes[0].plot(df_plot['timestamp'], df_plot['ema_fast'], label=f'EMA {strategy.fast_period}', linewidth=1.5)
        axes[0].plot(df_plot['timestamp'], df_plot['ema_slow'], label=f'EMA {strategy.slow_period}', linewidth=1.5)
        
        # Mark crossovers
        golden = df_plot[
            (df_plot['ema_diff'].shift(1) <= 0) & (df_plot['ema_diff'] > 0)
        ]
        death = df_plot[
            (df_plot['ema_diff'].shift(1) >= 0) & (df_plot['ema_diff'] < 0)
        ]
        
        axes[0].scatter(golden['timestamp'], golden['close'], 
                       color='green', marker='^', s=100, label='Golden Cross', zorder=5)
        axes[0].scatter(death['timestamp'], death['close'], 
                       color='red', marker='v', s=100, label='Death Cross', zorder=5)
        
        axes[0].set_title('Price & EMA Crossovers', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. RSI
        axes[1].plot(df_plot['timestamp'], df_plot['rsi'], label='RSI', color='purple')
        axes[1].axhline(y=strategy.rsi_overbought, color='red', linestyle='--', label=f'Overbought ({strategy.rsi_overbought})')
        axes[1].axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
        axes[1].fill_between(df_plot['timestamp'], 30, strategy.rsi_overbought, alpha=0.1)
        axes[1].set_title('RSI Filter', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
        
        # 3. EMA Distance
        axes[2].plot(df_plot['timestamp'], df_plot['ema_diff'], label='Fast - Slow EMA', color='orange')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].fill_between(df_plot['timestamp'], 0, df_plot['ema_diff'], 
                           where=(df_plot['ema_diff'] > 0), alpha=0.3, color='green', label='Bullish')
        axes[2].fill_between(df_plot['timestamp'], 0, df_plot['ema_diff'], 
                           where=(df_plot['ema_diff'] <= 0), alpha=0.3, color='red', label='Bearish')
        axes[2].set_title('EMA Distance', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('strategy_diagnostics.png', dpi=150)
        print(f"\n✅ Diagnostic chart saved to strategy_diagnostics.png")
    
    @staticmethod
    def compare_to_buy_hold(df: pd.DataFrame, backtest_return: float):
        """Compare strategy to simple buy and hold"""
        print("\n" + "=" * 70)
        print("BUY & HOLD COMPARISON")
        print("=" * 70)
        
        initial_price = df.iloc[0]['close']
        final_price = df.iloc[-1]['close']
        
        bh_return = (final_price - initial_price) / initial_price
        
        print(f"\n💰 RETURNS")
        print(f"  Buy & Hold: {bh_return*100:.2f}%")
        print(f"  Your Strategy: {backtest_return*100:.2f}%")
        print(f"  Difference: {(backtest_return - bh_return)*100:.2f}%")
        
        if backtest_return < bh_return:
            print(f"\n  ❌ Strategy UNDERPERFORMED buy-and-hold by {(bh_return - backtest_return)*100:.2f}%")
            print(f"     This means you'd be better off just buying and holding!")
        else:
            print(f"\n  ✅ Strategy OUTPERFORMED buy-and-hold by {(backtest_return - bh_return)*100:.2f}%")
        
        print("\n" + "=" * 70)


# ============================================================================
# USAGE
# ============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    print("""
    BACKTEST DIAGNOSTICS TOOL
    
    This will analyze why your strategy generated so few trades.
    """)
    
    # Load your data
    data_files = list(Path('./data').glob('*.csv'))
    
    if not data_files:
        print("❌ No data files found in ./data/")
        print("Run: python3 -m boty.download_data first")
        sys.exit(1)
    
    data_file = data_files[0]
    print(f"Loading data from: {data_file}")
    
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Analyze with best parameters from your optimization
    print("\nAnalyzing with parameters: fast=20, slow=50, rsi=65")
    strategy = EMACrossoverStrategy(
        fast_period=20,
        slow_period=50,
        rsi_period=14,
        rsi_overbought=65
    )
    
    df_analyzed = BacktestDiagnostics.analyze_signals(df, strategy)
    
    # Compare to buy-and-hold
    BacktestDiagnostics.compare_to_buy_hold(df, 0.002399)  # Your best return
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
    Based on your results, here are the issues:
    
    1. ❌ TOO FEW TRADES (only 1-9 trades in 6 months)
       → Strategy is TOO CONSERVATIVE
       → Signals almost never trigger
    
    2. ❌ RETURNS NEAR ZERO (0.24% in 6 months)
       → Even if positive, this is essentially break-even
       → Not worth the risk/effort
    
    3. ❌ LIKELY UNDERPERFORMS BUY-AND-HOLD
       → BTC probably went up 20-50% in your test period
       → Your strategy made 0.24%
    
    SOLUTIONS:
    
    Option A: Use a more active strategy
       - Try shorter EMA periods (8/21 instead of 20/50)
       - Remove or loosen RSI filter (75-80 instead of 65)
       - Add more entry conditions (volume, momentum)
    
    Option B: Use a mean-reversion strategy instead
       - RSI oversold/overbought works better in ranging markets
       - See run_backtest.py option 2 to compare strategies
    
    Option C: Accept fewer trades but improve quality
       - Add trend filter (only trade in strong trends)
       - Use larger timeframes (1h or 4h instead of 15m)
       - Focus on high-conviction setups only
    
    NEXT STEP:
    Run: python3 -m boty.run_backtest
    Choose option 2 to compare multiple strategies
    """)