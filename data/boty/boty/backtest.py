"""
Backtesting Engine - Phase 2
Test strategies on historical data before live trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging
import matplotlib.pyplot as plt

from boty import (
    Strategy, Signal, SignalAction, Position, Trade,
    RiskController, RiskParams, PositionManager
)


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 10000
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005   # 0.05% slippage
    
    # Risk parameters (same as live trading)
    risk_params: RiskParams = field(default_factory=lambda: RiskParams(
        max_position_size=1000,
        risk_per_trade=0.02,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        use_atr_stops=True
    ))


@dataclass
class BacktestMetrics:
    """Performance metrics from backtest"""
    # Returns
    total_return: float
    total_return_pct: float
    annual_return_pct: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # avg_win / |avg_loss|
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Duration
    avg_trade_duration_hours: float
    max_trade_duration_hours: float
    
    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    
    # Trade log
    trades: List[Trade] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'total_return': f"${self.total_pnl:.2f}",
            'total_return_pct': f"{self.total_return_pct:.2%}",
            'annual_return_pct': f"{self.annual_return_pct:.2%}",
            'total_trades': self.total_trades,
            'win_rate': f"{self.win_rate:.2%}",
            'avg_win': f"${self.avg_win:.2f}",
            'avg_loss': f"${self.avg_loss:.2f}",
            'profit_factor': f"{self.profit_factor:.2f}",
            'max_drawdown': f"${self.max_drawdown:.2f}",
            'max_drawdown_pct': f"{self.max_drawdown_pct:.2%}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'sortino_ratio': f"{self.sortino_ratio:.2f}",
            'avg_trade_duration': f"{self.avg_trade_duration_hours:.1f}h"
        }


class Backtester:
    """
    Backtest engine for strategy evaluation
    
    Simulates trading by replaying historical data bar-by-bar
    """
    
    def __init__(self, strategy: Strategy, config: BacktestConfig = None):
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.logger = logging.getLogger(__name__)
        
        # Risk management
        self.risk_controller = RiskController(self.config.risk_params)
        self.position_manager = PositionManager(self.risk_controller)
        
        # State
        self.cash = self.config.initial_capital
        self.equity_curve = []
        self.trades: List[Trade] = []
        self.current_position: Optional[Position] = None
    
    def run(self, df: pd.DataFrame, symbol: str = 'BTC/USDT') -> BacktestMetrics:
        """
        Run backtest on historical data
        
        Args:
            df: OHLCV DataFrame with columns [timestamp, open, high, low, close, volume]
            symbol: Trading symbol
        
        Returns:
            BacktestMetrics with performance statistics
        """
        self.logger.info(f"Starting backtest: {symbol}")
        self.logger.info(f"Data: {len(df)} bars from {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
        self.logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")
        
        # Reset state
        self.cash = self.config.initial_capital
        self.equity_curve = []
        self.trades = []
        self.current_position = None
        
        # Need enough data for indicators
        min_bars = max(self.strategy.slow_period if hasattr(self.strategy, 'slow_period') else 50, 100)
        
        # Main backtest loop
        for i in range(min_bars, len(df)):
            # Get data up to current bar (no lookahead bias!)
            historical_data = df.iloc[:i+1].copy()
            current_bar = df.iloc[i]
            current_time = current_bar['timestamp']
            current_price = current_bar['close']
            
            # Calculate indicators
            historical_data = self.strategy.calculate_indicators(historical_data)
            
            # Check existing position first
            if self.current_position:
                self._update_position(current_price, current_time, historical_data)
            
            # Generate new signal
            signal = self.strategy.generate_signal(historical_data, self.current_position)
            
            # Execute trade if signal generated
            if signal.action == SignalAction.BUY and self.current_position is None:
                self._open_position(signal, current_price, current_time, historical_data, symbol)
            
            elif signal.action == SignalAction.SELL and self.current_position is not None:
                self._close_position(current_price, current_time, 'SIGNAL')
            
            # Record equity
            equity = self._calculate_equity(current_price)
            self.equity_curve.append(equity)
        
        # Close any remaining position
        if self.current_position:
            final_price = df.iloc[-1]['close']
            final_time = df.iloc[-1]['timestamp']
            self._close_position(final_price, final_time, 'BACKTEST_END')
        
        # Calculate metrics
        metrics = self._calculate_metrics(df)
        
        self._print_summary(metrics)
        
        return metrics
    
    def _open_position(self, signal: Signal, price: float, timestamp: datetime, 
                       df: pd.DataFrame, symbol: str):
        """Simulate opening a position"""
        # Apply slippage
        execution_price = price * (1 + self.config.slippage_pct)
        
        # Validate trade
        validation = self.risk_controller.validate_trade(
            signal=signal,
            current_price=execution_price,
            portfolio_cash=self.cash,
            current_position=None
        )
        
        if not validation.approved:
            self.logger.debug(f"Trade rejected: {validation.reasons}")
            return
        
        # Calculate position size
        size = validation.approved_size
        position_value = size * execution_price
        commission = position_value * self.config.commission_pct
        
        # Check if we have enough cash
        total_cost = position_value + commission
        if total_cost > self.cash:
            self.logger.debug(f"Insufficient cash: ${self.cash:.2f} < ${total_cost:.2f}")
            return
        
        # Calculate stops
        stop_loss = self.risk_controller.calculate_stop_loss(execution_price, df)
        take_profit = self.risk_controller.calculate_take_profit(execution_price)
        
        # Create position
        self.current_position = Position(
            symbol=symbol,
            side='LONG',
            entry_price=execution_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=timestamp
        )
        
        # Update cash
        self.cash -= total_cost
        
        self.logger.info(
            f"[{timestamp}] OPEN: {size:.6f} @ ${execution_price:.2f}, "
            f"Stop: ${stop_loss:.2f}, Target: ${take_profit:.2f}"
        )
    
    def _update_position(self, price: float, timestamp: datetime, df: pd.DataFrame):
        """Check if position should be closed"""
        if not self.current_position:
            return
        
        # Check stop loss
        if price <= self.current_position.stop_loss:
            self._close_position(price, timestamp, 'STOP_LOSS')
            return
        
        # Check take profit
        if self.current_position.take_profit and price >= self.current_position.take_profit:
            self._close_position(price, timestamp, 'TAKE_PROFIT')
            return
    
    def _close_position(self, price: float, timestamp: datetime, reason: str):
        """Simulate closing a position"""
        if not self.current_position:
            return
        
        # Apply slippage
        execution_price = price * (1 - self.config.slippage_pct)
        
        # Calculate P&L
        position_value = self.current_position.size * execution_price
        commission = position_value * self.config.commission_pct
        proceeds = position_value - commission
        
        # Calculate realized P&L
        cost_basis = self.current_position.size * self.current_position.entry_price
        realized_pnl = proceeds - cost_basis - (cost_basis * self.config.commission_pct)
        pnl_pct = (realized_pnl / cost_basis) * 100
        
        # Duration
        duration = (timestamp - self.current_position.opened_at).total_seconds() / 3600
        
        # Create trade record
        trade = Trade(
            symbol=self.current_position.symbol,
            entry_price=self.current_position.entry_price,
            exit_price=execution_price,
            size=self.current_position.size,
            pnl=realized_pnl,
            pnl_pct=pnl_pct,
            duration=duration,
            exit_reason=reason,
            opened_at=self.current_position.opened_at,
            closed_at=timestamp
        )
        
        self.trades.append(trade)
        
        # Update cash
        self.cash += proceeds
        
        # Update daily P&L for risk controller
        self.risk_controller.update_daily_pnl(realized_pnl)
        
        self.logger.info(
            f"[{timestamp}] CLOSE: ${execution_price:.2f}, "
            f"P&L: ${realized_pnl:.2f} ({pnl_pct:+.2f}%), {reason}"
        )
        
        self.current_position = None
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current total equity"""
        equity = self.cash
        
        if self.current_position:
            position_value = self.current_position.size * current_price
            equity += position_value
        
        return equity
    
    def _calculate_metrics(self, df: pd.DataFrame) -> BacktestMetrics:
        """Calculate performance metrics"""
        final_equity = self.equity_curve[-1] if self.equity_curve else self.config.initial_capital
        
        # Returns
        total_pnl = final_equity - self.config.initial_capital
        total_return_pct = (total_pnl / self.config.initial_capital)
        
        # Annualized return
        days = (df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']).days
        years = days / 365.25
        annual_return_pct = (1 + total_return_pct) ** (1 / years) - 1 if years > 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Drawdown
        equity_array = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = peak - equity_array
        max_drawdown = np.max(drawdown)
        max_drawdown_pct = max_drawdown / np.max(peak) if np.max(peak) > 0 else 0
        
        # Sharpe ratio (simplified - using daily returns)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() > 0 else 0
        
        # Sortino ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        sortino = np.sqrt(252) * (returns.mean() / downside_returns.std()) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Duration
        avg_duration = np.mean([t.duration for t in self.trades]) if self.trades else 0
        max_duration = np.max([t.duration for t in self.trades]) if self.trades else 0
        
        return BacktestMetrics(
            total_return=total_pnl,
            total_return_pct=total_return_pct,
            annual_return_pct=annual_return_pct,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            avg_trade_duration_hours=avg_duration,
            max_trade_duration_hours=max_duration,
            equity_curve=self.equity_curve,
            timestamps=[df.iloc[i]['timestamp'] for i in range(len(self.equity_curve))],
            trades=self.trades
        )
    
    def _print_summary(self, metrics: BacktestMetrics):
        """Print backtest summary"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("BACKTEST RESULTS")
        self.logger.info("=" * 70)
        
        self.logger.info(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        self.logger.info(f"Final Equity: ${self.equity_curve[-1]:,.2f}")
        self.logger.info(f"Total Return: ${metrics.total_pnl:,.2f} ({metrics.total_return_pct:.2%})")
        self.logger.info(f"Annual Return: {metrics.annual_return_pct:.2%}")
        
        self.logger.info("\nTrade Statistics:")
        self.logger.info(f"  Total Trades: {metrics.total_trades}")
        self.logger.info(f"  Winning: {metrics.winning_trades} ({metrics.win_rate:.1%})")
        self.logger.info(f"  Losing: {metrics.losing_trades}")
        self.logger.info(f"  Avg Win: ${metrics.avg_win:.2f}")
        self.logger.info(f"  Avg Loss: ${metrics.avg_loss:.2f}")
        self.logger.info(f"  Profit Factor: {metrics.profit_factor:.2f}")
        
        self.logger.info("\nRisk Metrics:")
        self.logger.info(f"  Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2%})")
        self.logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        self.logger.info(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
        
        self.logger.info("\nTrade Duration:")
        self.logger.info(f"  Average: {metrics.avg_trade_duration_hours:.1f} hours")
        self.logger.info(f"  Maximum: {metrics.max_trade_duration_hours:.1f} hours")
        
        self.logger.info("=" * 70)
    
    def plot_results(self, metrics: BacktestMetrics, save_path: str = None):
        """Plot backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. Equity curve
        axes[0].plot(metrics.timestamps, metrics.equity_curve, linewidth=2)
        axes[0].axhline(y=self.config.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        axes[0].set_title(f'Equity Curve - {self.strategy.name}', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Equity ($)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        equity_array = np.array(metrics.equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak * 100
        axes[1].fill_between(metrics.timestamps, drawdown, 0, color='red', alpha=0.3)
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # 3. Trade P&L distribution
        if metrics.trades:
            pnls = [t.pnl for t in metrics.trades]
            colors = ['green' if p > 0 else 'red' for p in pnls]
            axes[2].bar(range(len(pnls)), pnls, color=colors, alpha=0.6)
            axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[2].set_title('Trade P&L', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Trade Number', fontsize=12)
            axes[2].set_ylabel('P&L ($)', fontsize=12)
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    import logging
    from boty import EMACrossoverStrategy
    
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data (replace with real data)
    print("Generating sample data...")
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=5000, freq='15min')
    
    # Simulate price with trend + noise
    trend = np.linspace(40000, 45000, 5000)
    noise = np.random.randn(5000).cumsum() * 50
    prices = trend + noise
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.random.rand(5000) * 100,
        'low': prices - np.random.rand(5000) * 100,
        'close': prices,
        'volume': np.random.rand(5000) * 1000
    })
    
    # Test strategy
    strategy = EMACrossoverStrategy(fast_period=12, slow_period=26)
    
    # Run backtest
    config = BacktestConfig(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_pct=0.0005
    )
    
    backtester = Backtester(strategy, config)
    metrics = backtester.run(df)
    
    # Plot results
    backtester.plot_results(metrics, save_path='backtest_results.png')