"""
Crypto Trading Bot - Main Orchestrator
Complete trading bot with all components integrated
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
import pandas as pd


# Import our components (in practice, these would be separate modules)
# from trading_bot_core import *
# from trading_bot_strategy import *


class TradingBot:
    """
    Main trading bot orchestrator
    Coordinates: data fetch → strategy → risk check → execution → monitoring
    """
    
    def __init__(self, config: Dict):
        """
        Initialize bot with configuration
        
        config = {
            'symbol': 'BTC/USDT',
            'timeframe': '15m',
            'strategy': 'ema_crossover',
            'strategy_params': {'fast_period': 12, 'slow_period': 26},
            'risk_params': {...},
            'api_key': '...',
            'api_secret': '...',
            'testnet': True,
            'db_path': './bot.db'
        }
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._init_components()
        
        # Runtime state
        self.is_running = False
        self.cycle_count = 0
        self.last_signal_time = None
    
    def _init_components(self):
        """Initialize all bot components"""
        from trading_bot_core import BinanceConnector, StateManager, setup_logging
        from trading_bot_strategy import get_strategy, RiskController, RiskParams, PositionManager
        
        # Logging
        setup_logging(self.config.get('log_file', 'trading_bot.log'))
        
        # Exchange connector
        self.connector = BinanceConnector(
            api_key=self.config['api_key'],
            api_secret=self.config['api_secret'],
            testnet=self.config.get('testnet', True)
        )
        
        # State manager
        self.state = StateManager(self.config.get('db_path', './trading_bot.db'))
        
        # Strategy
        self.strategy = get_strategy(
            self.config['strategy'],
            **self.config.get('strategy_params', {})
        )
        
        # Risk controller
        risk_params = RiskParams(**self.config.get('risk_params', {}))
        self.risk_controller = RiskController(risk_params)
        
        # Position manager
        self.position_manager = PositionManager(self.risk_controller)
        
        self.logger.info(f"Bot initialized: {self.config['strategy']} on {self.config['symbol']}")
    
    async def start(self):
        """Start the trading bot"""
        self.is_running = True
        self.logger.info("=" * 60)
        self.logger.info("TRADING BOT STARTED")
        self.logger.info("=" * 60)
        
        # Load any existing position
        current_position = self.state.load_position(self.config['symbol'])
        if current_position:
            self.logger.info(f"Loaded existing position: {current_position.side} {current_position.size:.6f} @ ${current_position.entry_price:.2f}")
        
        # Main event loop
        while self.is_running:
            try:
                await self._trading_cycle(current_position)
                self.cycle_count += 1
                
                # Wait before next cycle
                await asyncio.sleep(self.config.get('cycle_interval', 60))
            
            except KeyboardInterrupt:
                self.logger.info("Shutdown signal received...")
                await self.stop()
                break
            
            except Exception as e:
                self.logger.error(f"Error in trading cycle: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait a bit before retrying
    
    async def _trading_cycle(self, current_position: Optional['Position']):
        """
        Single trading cycle iteration
        
        Steps:
        1. Fetch latest market data
        2. Calculate indicators
        3. Generate signal
        4. Check existing position (stops/targets)
        5. Execute new trades if approved
        6. Update state and log decisions
        """
        from trading_bot_core import Signal, SignalAction, Trade, Position
        
        symbol = self.config['symbol']
        timeframe = self.config['timeframe']
        
        # ===== STEP 1: Fetch Market Data =====
        try:
            df = await self.connector.get_candles(symbol, timeframe, limit=200)
            ticker = await self.connector.get_ticker(symbol)
            current_price = ticker.last
            
            self.logger.debug(f"Cycle {self.cycle_count}: Price=${current_price:.2f}, Candles={len(df)}")
        
        except Exception as e:
            self.logger.error(f"Failed to fetch data: {e}")
            return
        
        # ===== STEP 2: Calculate Indicators =====
        df = self.strategy.calculate_indicators(df)
        indicators = self.strategy.get_indicators(df)
        
        # ===== STEP 3: Check Existing Position First =====
        if current_position:
            # Update unrealized P&L
            unrealized_pnl = self.position_manager.calculate_unrealized_pnl(
                current_position, current_price
            )
            current_position.unrealized_pnl = unrealized_pnl
            
            self.logger.info(
                f"Position: {current_position.size:.6f} @ ${current_position.entry_price:.2f}, "
                f"Current: ${current_price:.2f}, P&L: ${unrealized_pnl:.2f}"
            )
            
            # Check stop loss and take profit
            exit_reason = self.position_manager.check_exit_conditions(
                current_position, current_price
            )
            
            if exit_reason:
                # Close position
                await self._close_position(current_position, current_price, exit_reason)
                current_position = None  # Position closed
                return  # Skip to next cycle
            
            # Update trailing stop (optional enhancement)
            # new_stop = self.position_manager.update_trailing_stop(current_position, current_price)
            # if new_stop:
            #     current_position.stop_loss = new_stop
            #     self.state.save_position(current_position)
        
        # ===== STEP 4: Generate New Signal =====
        signal = self.strategy.generate_signal(df, current_position)
        
        self.logger.info(
            f"Signal: {signal.action.value} (confidence={signal.confidence:.2f}) - {signal.reason}"
        )
        
        # ===== STEP 5: Execute Trade if Signal Generated =====
        portfolio_cash = await self._get_portfolio_cash()
        
        action_taken = 'HELD'
        
        if signal.action == SignalAction.BUY and current_position is None:
            # Validate trade
            validation = self.risk_controller.validate_trade(
                signal, current_price, portfolio_cash, current_position
            )
            
            if validation.approved:
                # Execute BUY
                position = await self._open_position(
                    signal, current_price, validation.approved_size, df
                )
                if position:
                    current_position = position
                    action_taken = 'EXECUTED_BUY'
            else:
                self.logger.warning(f"Trade rejected: {validation.reasons}")
                action_taken = 'REJECTED'
        
        elif signal.action == SignalAction.SELL and current_position is not None:
            # Execute SELL
            await self._close_position(current_position, current_price, 'SIGNAL')
            current_position = None
            action_taken = 'EXECUTED_SELL'
        
        # ===== STEP 6: Log Decision =====
        portfolio_state = {
            'cash': portfolio_cash,
            'position': current_position.symbol if current_position else None,
            'unrealized_pnl': current_position.unrealized_pnl if current_position else 0
        }
        
        self.state.log_decision(
            signal=signal,
            symbol=symbol,
            action_taken=action_taken,
            indicators=indicators,
            portfolio_state=portfolio_state
        )
        
        # Print summary
        if self.cycle_count % 10 == 0:  # Every 10 cycles
            await self._print_performance_summary()
    
    async def _open_position(self, signal: 'Signal', entry_price: float, 
                            size: float, df: pd.DataFrame) -> Optional['Position']:
        """Open a new position"""
        from trading_bot_core import Position
        
        try:
            # Place market buy order
            order = await self.connector.place_market_order(
                symbol=self.config['symbol'],
                side='BUY',
                size=size
            )
            
            # Calculate stop loss and take profit
            stop_loss = self.risk_controller.calculate_stop_loss(entry_price, df)
            take_profit = self.risk_controller.calculate_take_profit(entry_price)
            
            # Create position object
            position = Position(
                symbol=self.config['symbol'],
                side='LONG',
                entry_price=order['avg_price'],
                size=order['filled'],
                stop_loss=stop_loss,
                take_profit=take_profit,
                opened_at=datetime.now()
            )
            
            # Save to state
            self.state.save_position(position)
            
            self.logger.info(
                f"✅ POSITION OPENED: {position.size:.6f} @ ${position.entry_price:.2f}, "
                f"Stop: ${stop_loss:.2f}, Target: ${take_profit:.2f}"
            )
            
            return position
        
        except Exception as e:
            self.logger.error(f"Failed to open position: {e}", exc_info=True)
            return None
    
    async def _close_position(self, position: 'Position', exit_price: float, 
                             exit_reason: str):
        """Close an existing position"""
        from trading_bot_core import Trade
        
        try:
            # Place market sell order
            order = await self.connector.place_market_order(
                symbol=position.symbol,
                side='SELL',
                size=position.size
            )
            
            # Calculate realized P&L
            realized_pnl = (order['avg_price'] - position.entry_price) * position.size
            pnl_pct = (realized_pnl / (position.entry_price * position.size)) * 100
            
            # Calculate duration
            duration_hours = (datetime.now() - position.opened_at).total_seconds() / 3600
            
            # Create trade record
            trade = Trade(
                symbol=position.symbol,
                entry_price=position.entry_price,
                exit_price=order['avg_price'],
                size=position.size,
                pnl=realized_pnl,
                pnl_pct=pnl_pct,
                duration=duration_hours,
                exit_reason=exit_reason,
                opened_at=position.opened_at,
                closed_at=datetime.now()
            )
            
            # Update state
            self.state.save_trade(trade)
            self.state.clear_position(position.symbol)
            self.risk_controller.update_daily_pnl(realized_pnl)
            
            # Log
            emoji = "🟢" if realized_pnl > 0 else "🔴"
            self.logger.info(
                f"{emoji} POSITION CLOSED: P&L=${realized_pnl:.2f} ({pnl_pct:+.2f}%), "
                f"Duration={duration_hours:.1f}h, Reason={exit_reason}"
            )
        
        except Exception as e:
            self.logger.error(f"Failed to close position: {e}", exc_info=True)
    
    async def _get_portfolio_cash(self) -> float:
        """Get available cash balance"""
        try:
            balances = await self.connector.get_balance()
            # For spot trading, look for USDT balance
            return balances.get('USDT', 0.0)
        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            return 0.0
    
    async def _print_performance_summary(self):
        """Print performance statistics"""
        stats = self.state.get_performance_stats()
        
        self.logger.info("=" * 60)
        self.logger.info("PERFORMANCE SUMMARY")
        self.logger.info("-" * 60)
        self.logger.info(f"Total Trades: {stats['total_trades']}")
        self.logger.info(f"Win Rate: {stats['win_rate']:.1%}")
        self.logger.info(f"Total P&L: ${stats['total_pnl']:.2f}")
        self.logger.info(f"Average P&L: ${stats['avg_pnl']:.2f}")
        if stats['total_trades'] > 0:
            self.logger.info(f"Average Win: ${stats['avg_win']:.2f}")
            self.logger.info(f"Average Loss: ${stats['avg_loss']:.2f}")
        self.logger.info(f"Daily P&L: ${self.risk_controller.daily_pnl:.2f}")
        self.logger.info("=" * 60)
    
    async def stop(self):
        """Stop the bot gracefully"""
        self.is_running = False
        self.logger.info("=" * 60)
        self.logger.info("TRADING BOT STOPPED")
        
        # Print final summary
        await self._print_performance_summary()
        
        # Close any open positions (optional - you may want to keep them)
        # position = self.state.load_position(self.config['symbol'])
        # if position:
        #     self.logger.warning("Open position exists. Close manually or bot will resume next time.")
        
        self.logger.info("=" * 60)


# ============================================================================
# CONFIGURATION & MAIN
# ============================================================================

def get_config(mode='paper_trading') -> Dict:
    """
    Get configuration for different modes
    
    Modes:
    - paper_trading: Testnet with sim mode
    - live_trading: Real money (use with caution!)
    - backtest: Historical data only
    """
    
    if mode == 'paper_trading':
        return {
            'symbol': 'BTC/USDT',
            'timeframe': '15m',
            'cycle_interval': 60,  # seconds
            
            # Strategy
            'strategy': 'ema_crossover',
            'strategy_params': {
                'fast_period': 12,
                'slow_period': 26,
                'rsi_period': 14,
                'rsi_overbought': 70
            },
            
            # Risk management
            'risk_params': {
                'max_position_size': 1000,  # USDT
                'max_portfolio_exposure': 5000,
                'max_daily_loss': 500,
                'risk_per_trade': 0.02,  # 2%
                'stop_loss_pct': 0.02,  # 2%
                'take_profit_pct': 0.05,  # 5%
                'use_atr_stops': True,
                'atr_multiplier': 2.0
            },
            
            # Exchange (IMPORTANT: Use your testnet keys!)
            'api_key': 'YOUR_TESTNET_API_KEY',
            'api_secret': 'YOUR_TESTNET_API_SECRET',
            'testnet': True,
            
            # Storage
            'db_path': './trading_bot_paper.db',
            'log_file': 'trading_bot_paper.log'
        }
    
    elif mode == 'live_trading':
        return {
            # Same as paper trading but:
            'api_key': 'YOUR_LIVE_API_KEY',  # REAL KEYS
            'api_secret': 'YOUR_LIVE_API_SECRET',
            'testnet': False,  # REAL TRADING
            
            # Reduce risk for live trading
            'risk_params': {
                'max_position_size': 100,  # Start small!
                'risk_per_trade': 0.01,  # 1%
                'stop_loss_pct': 0.015,  # 1.5%
            },
            
            'db_path': './trading_bot_live.db',
            'log_file': 'trading_bot_live.log'
        }
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


async def main():
    """Main entry point"""
    
    # Choose mode
    MODE = 'paper_trading'  # Change to 'live_trading' when ready
    
    # Get configuration
    config = get_config(MODE)
    
    # Create and start bot
    bot = TradingBot(config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║              CRYPTO TRADING BOT v1.0                     ║
    ║                                                          ║
    ║  WARNING: This bot trades real money in live mode!       ║
    ║  Always test thoroughly in paper trading first.          ║
    ║                                                          ║
    ║  Press Ctrl+C to stop the bot safely.                    ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Run bot
    asyncio.run(main())