"""
Backtest Runner - Complete workflow
Download data → Run backtest → Analyze results → Optimize parameters
"""

import asyncio
import logging
import pandas as pd
from itertools import product
from typing import Dict, List

from boty import EMACrossoverStrategy, RSIMeanReversionStrategy
from boty.backtest import Backtester, BacktestConfig, BacktestMetrics
from .download_data import download_for_backtest


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestRunner:
    """Complete backtesting workflow"""
    
    @staticmethod
    async def quick_test():
        """Quick backtest with default parameters"""
        print("\n" + "=" * 70)
        print("QUICK BACKTEST - EMA Crossover Strategy")
        print("=" * 70)
        
        # 1. Download data (or load if exists)
        print("\n1. Downloading historical data...")
        df = await download_for_backtest(
            symbol='BTC/USDT',
            timeframe='15m',
            months=6
        )
        
        # 2. Create strategy
        print("\n2. Initializing strategy...")
        strategy = EMACrossoverStrategy(
            fast_period=12,
            slow_period=26,
            rsi_period=14,
            rsi_overbought=70
        )
        
        # 3. Configure backtest
        config = BacktestConfig(
            initial_capital=10000,
            commission_pct=0.001,   # 0.1% Binance fee
            slippage_pct=0.0005     # 0.05% slippage
        )
        
        # 4. Run backtest
        print("\n3. Running backtest...")
        backtester = Backtester(strategy, config)
        metrics = backtester.run(df, symbol='BTC/USDT')
        
        # 5. Save results
        print("\n4. Generating plots...")
        backtester.plot_results(metrics, save_path='backtest_ema_crossover.png')
        
        # 6. Print summary
        BacktestRunner._print_detailed_summary(metrics)
        
        return metrics
    
    @staticmethod
    async def compare_strategies():
        """Compare multiple strategies"""
        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON")
        print("=" * 70)
        
        # Download data
        df = await download_for_backtest('BTC/USDT', '15m', months=6)
        
        strategies = [
            ('EMA Crossover', EMACrossoverStrategy(fast_period=12, slow_period=26)),
            ('RSI Mean Reversion', RSIMeanReversionStrategy(rsi_period=14, oversold=30, overbought=70)),
            ('EMA Fast', EMACrossoverStrategy(fast_period=8, slow_period=21)),
        ]
        
        config = BacktestConfig(initial_capital=10000)
        results = []
        
        for name, strategy in strategies:
            print(f"\nTesting {name}...")
            backtester = Backtester(strategy, config)
            metrics = backtester.run(df, symbol='BTC/USDT')
            results.append((name, metrics))
        
        # Compare results
        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)
        
        comparison_df = pd.DataFrame([
            {
                'Strategy': name,
                'Return': f"{m.total_return_pct:.2%}",
                'Trades': m.total_trades,
                'Win Rate': f"{m.win_rate:.1%}",
                'Sharpe': f"{m.sharpe_ratio:.2f}",
                'Max DD': f"{m.max_drawdown_pct:.2%}",
                'Profit Factor': f"{m.profit_factor:.2f}"
            }
            for name, m in results
        ])
        
        print(comparison_df.to_string(index=False))
        
        return results
    
    @staticmethod
    async def optimize_parameters():
        """Find optimal strategy parameters"""
        print("\n" + "=" * 70)
        print("PARAMETER OPTIMIZATION")
        print("=" * 70)
        
        # Download data
        df = await download_for_backtest('BTC/USDT', '15m', months=6)
        
        # Parameter grid
        fast_periods = [8, 12, 20]
        slow_periods = [21, 26, 50]
        rsi_thresholds = [65, 70, 75]
        
        print(f"\nTesting {len(fast_periods) * len(slow_periods) * len(rsi_thresholds)} combinations...")
        
        results = []
        config = BacktestConfig(initial_capital=10000)
        
        for fast, slow, rsi in product(fast_periods, slow_periods, rsi_thresholds):
            if fast >= slow:
                continue  # Skip invalid combinations
            
            strategy = EMACrossoverStrategy(
                fast_period=fast,
                slow_period=slow,
                rsi_period=14,
                rsi_overbought=rsi
            )
            
            try:
                backtester = Backtester(strategy, config)
                metrics = backtester.run(df, symbol='BTC/USDT')
                
                results.append({
                    'fast_ema': fast,
                    'slow_ema': slow,
                    'rsi_threshold': rsi,
                    'return_pct': metrics.total_return_pct,
                    'sharpe': metrics.sharpe_ratio,
                    'win_rate': metrics.win_rate,
                    'max_dd': metrics.max_drawdown_pct,
                    'trades': metrics.total_trades
                })
                
                logger.info(f"EMA({fast},{slow}) RSI({rsi}): Return={metrics.total_return_pct:.2%}, Sharpe={metrics.sharpe_ratio:.2f}")
            
            except Exception as e:
                logger.error(f"Error with params ({fast},{slow},{rsi}): {e}")
        
        # Find best parameters
        results_df = pd.DataFrame(results)
        
        # Sort by Sharpe ratio (risk-adjusted returns)
        results_df = results_df.sort_values('sharpe', ascending=False)
        
        print("\n" + "=" * 70)
        print("TOP 5 PARAMETER COMBINATIONS (by Sharpe Ratio)")
        print("=" * 70)
        print(results_df.head(5).to_string(index=False))
        
        # Also check by raw returns
        print("\n" + "=" * 70)
        print("TOP 5 PARAMETER COMBINATIONS (by Total Return)")
        print("=" * 70)
        print(results_df.sort_values('return_pct', ascending=False).head(5).to_string(index=False))
        
        # Save full results
        results_df.to_csv('parameter_optimization_results.csv', index=False)
        print("\n✅ Full results saved to parameter_optimization_results.csv")
        
        return results_df
    
    @staticmethod
    def _print_detailed_summary(metrics: BacktestMetrics):
        """Print detailed backtest summary"""
        print("\n" + "=" * 70)
        print("DETAILED PERFORMANCE ANALYSIS")
        print("=" * 70)
        
        # Overall performance
        print("\n📊 OVERALL PERFORMANCE")
        print(f"  Total Return: {metrics.total_return_pct:.2%} (${metrics.total_pnl:,.2f})")
        print(f"  Annual Return: {metrics.annual_return_pct:.2%}")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
        
        # Trade statistics
        print("\n📈 TRADE STATISTICS")
        print(f"  Total Trades: {metrics.total_trades}")
        print(f"  Win Rate: {metrics.win_rate:.1%}")
        print(f"  Average Win: ${metrics.avg_win:.2f}")
        print(f"  Average Loss: ${metrics.avg_loss:.2f}")
        print(f"  Profit Factor: {metrics.profit_factor:.2f}")
        print(f"  Best Trade: ${max([t.pnl for t in metrics.trades]) if metrics.trades else 0:.2f}")
        print(f"  Worst Trade: ${min([t.pnl for t in metrics.trades]) if metrics.trades else 0:.2f}")
        
        # Risk metrics
        print("\n⚠️  RISK METRICS")
        print(f"  Max Drawdown: {metrics.max_drawdown_pct:.2%} (${metrics.max_drawdown:,.2f})")
        print(f"  Avg Trade Duration: {metrics.avg_trade_duration_hours:.1f} hours")
        print(f"  Max Trade Duration: {metrics.max_trade_duration_hours:.1f} hours")
        
        # Exit reasons breakdown
        if metrics.trades:
            exit_reasons = {}
            for trade in metrics.trades:
                exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1
            
            print("\n🚪 EXIT REASONS")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
                pct = count / len(metrics.trades) * 100
                print(f"  {reason}: {count} ({pct:.1f}%)")
        
        # Monthly returns
        if metrics.trades:
            monthly_pnl = {}
            for trade in metrics.trades:
                month_key = trade.closed_at.strftime('%Y-%m')
                monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + trade.pnl
            
            print("\n📅 MONTHLY P&L")
            for month in sorted(monthly_pnl.keys()):
                print(f"  {month}: ${monthly_pnl[month]:,.2f}")
        
        print("\n" + "=" * 70)
        
        # Assessment
        print("\n🎯 ASSESSMENT")
        
        assessments = []
        if metrics.sharpe_ratio > 1.5:
            assessments.append("✅ Excellent risk-adjusted returns (Sharpe > 1.5)")
        elif metrics.sharpe_ratio > 1.0:
            assessments.append("✅ Good risk-adjusted returns (Sharpe > 1.0)")
        else:
            assessments.append("⚠️  Poor risk-adjusted returns (Sharpe < 1.0)")
        
        if metrics.win_rate > 0.5:
            assessments.append("✅ Positive win rate (>50%)")
        else:
            assessments.append("⚠️  Low win rate (<50%)")
        
        if metrics.max_drawdown_pct < 0.15:
            assessments.append("✅ Acceptable drawdown (<15%)")
        else:
            assessments.append("⚠️  High drawdown (>15%)")
        
        if metrics.profit_factor > 1.5:
            assessments.append("✅ Strong profit factor (>1.5)")
        elif metrics.profit_factor > 1.0:
            assessments.append("⚠️  Weak profit factor (1.0-1.5)")
        else:
            assessments.append("❌ Losing strategy (profit factor <1.0)")
        
        for assessment in assessments:
            print(f"  {assessment}")
        
        print("\n" + "=" * 70)


# ============================================================================
# MAIN MENU
# ============================================================================

async def main():
    """Interactive backtest menu"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║             CRYPTO TRADING BOT - BACKTESTER              ║
    ║                       Phase 2                            ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    
    Select an option:
    
    1. Quick Backtest (EMA Crossover, 6 months)
    2. Compare Multiple Strategies
    3. Optimize Parameters (Grid Search)
    4. Exit
    """)
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        await BacktestRunner.quick_test()
    elif choice == '2':
        await BacktestRunner.compare_strategies()
    elif choice == '3':
        await BacktestRunner.optimize_parameters()
    elif choice == '4':
        print("Goodbye!")
    else:
        print("Invalid choice")


if __name__ == '__main__':
    asyncio.run(main())