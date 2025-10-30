"""
Crypto Trading Bot - Core Architecture
Phase 1: Foundation Components
"""

import ccxt
import pandas as pd
import sqlite3
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio

from .data_models import (
    Ticker, Candle, Signal, Position, Trade, SignalAction,
)
from .connectors import ExchangeConnector, BinanceConnector


# ============================================================================
# STATE MANAGER (Persistence)
# ============================================================================

class StateManager:
    """Manage bot state and decision logs"""
    
    def __init__(self, db_path: str = './trading_bot.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_tables()
        self.logger = logging.getLogger(__name__)
    
    def _init_tables(self):
        """Create database schema"""
        cursor = self.conn.cursor()
        
        # Candles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT,
                timeframe TEXT,
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)
        
        # Decision log (audit trail)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                signal_action TEXT,
                signal_reason TEXT,
                signal_confidence REAL,
                indicators TEXT,
                action_taken TEXT,
                portfolio_state TEXT
            )
        """)
        
        # Trade history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                entry_price REAL,
                exit_price REAL,
                size REAL,
                pnl REAL,
                pnl_pct REAL,
                duration_hours REAL,
                exit_reason TEXT,
                opened_at TEXT,
                closed_at TEXT
            )
        """)
        
        # Current position (single position for now)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS current_position (
                symbol TEXT PRIMARY KEY,
                side TEXT,
                entry_price REAL,
                size REAL,
                stop_loss REAL,
                take_profit REAL,
                opened_at TEXT
            )
        """)
        
        self.conn.commit()
    
    def log_decision(self, signal: Signal, symbol: str, action_taken: str, 
                    indicators: Dict, portfolio_state: Dict):
        """Log a trading decision for audit"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO decisions 
            (timestamp, symbol, signal_action, signal_reason, signal_confidence, 
             indicators, action_taken, portfolio_state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            symbol,
            signal.action.value,
            signal.reason,
            signal.confidence,
            json.dumps(indicators),
            action_taken,
            json.dumps(portfolio_state)
        ))
        self.conn.commit()
        self.logger.info(f"Logged decision: {signal.action.value} - {action_taken}")
    
    def save_position(self, position: Position):
        """Save current position"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO current_position 
            (symbol, side, entry_price, size, stop_loss, take_profit, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            position.symbol,
            position.side,
            position.entry_price,
            position.size,
            position.stop_loss,
            position.take_profit,
            position.opened_at.isoformat()
        ))
        self.conn.commit()
    
    def load_position(self, symbol: str) -> Optional[Position]:
        """Load current position"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM current_position WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        
        if row:
            return Position(
                symbol=row[0],
                side=row[1],
                entry_price=row[2],
                size=row[3],
                stop_loss=row[4],
                take_profit=row[5],
                opened_at=datetime.fromisoformat(row[6])
            )
        return None
    
    def clear_position(self, symbol: str):
        """Remove position after closing"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM current_position WHERE symbol = ?", (symbol,))
        self.conn.commit()
    
    def save_trade(self, trade: Trade):
        """Save completed trade"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trades 
            (symbol, entry_price, exit_price, size, pnl, pnl_pct, 
             duration_hours, exit_reason, opened_at, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.symbol,
            trade.entry_price,
            trade.exit_price,
            trade.size,
            trade.pnl,
            trade.pnl_pct,
            trade.duration,
            trade.exit_reason,
            trade.opened_at.isoformat(),
            trade.closed_at.isoformat()
        ))
        self.conn.commit()
        self.logger.info(f"Saved trade: PnL=${trade.pnl:.2f}")
    
    def get_trade_history(self, limit: int = 100) -> List[Trade]:
        """Get recent trades"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM trades ORDER BY closed_at DESC LIMIT ?
        """, (limit,))
        
        trades = []
        for row in cursor.fetchall():
            trades.append(Trade(
                symbol=row[1],
                entry_price=row[2],
                exit_price=row[3],
                size=row[4],
                pnl=row[5],
                pnl_pct=row[6],
                duration=row[7],
                exit_reason=row[8],
                opened_at=datetime.fromisoformat(row[9]),
                closed_at=datetime.fromisoformat(row[10])
            ))
        
        return trades
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance metrics"""
        trades = self.get_trade_history(limit=1000)
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }
        
        winning_trades = [t for t in trades if t.pnl > 0]
        
        return {
            'total_trades': len(trades),
            'win_rate': len(winning_trades) / len(trades),
            'total_pnl': sum(t.pnl for t in trades),
            'avg_pnl': sum(t.pnl for t in trades) / len(trades),
            'avg_win': sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t.pnl for t in trades if t.pnl < 0) / len([t for t in trades if t.pnl < 0]) if [t for t in trades if t.pnl < 0] else 0
        }


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str = 'trading_bot.log'):
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def test_components():
    """Test the core components"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize components
    logger.info("Initializing trading bot components...")
    
    # Note: Replace with actual API keys for testing
    connector = BinanceConnector(
        api_key='your_api_key_here',
        api_secret='your_api_secret_here',
        testnet=True
    )
    
    state_manager = StateManager('./test_bot.db')
    
    # Test ticker fetch
    logger.info("Testing ticker fetch...")
    ticker = await connector.get_ticker('BTC/USDT')
    logger.info(f"Current BTC price: ${ticker.last:,.2f}")
    
    # Test candle fetch
    logger.info("Testing candle fetch...")
    df = await connector.get_candles('BTC/USDT', '15m', limit=100)
    logger.info(f"Fetched {len(df)} candles")
    logger.info(f"Latest close: ${df.iloc[-1]['close']:,.2f}")
    
    # Test decision logging
    signal = Signal(
        action=SignalAction.HOLD,
        confidence=0.0,
        reason="Testing decision log",
        metadata={'test': True}
    )
    
    state_manager.log_decision(
        signal=signal,
        symbol='BTC/USDT',
        action_taken='LOGGED',
        indicators={'ema_fast': 42000, 'ema_slow': 41500},
        portfolio_state={'equity': 10000, 'cash': 10000}
    )
    
    logger.info("Component tests completed successfully!")


if __name__ == '__main__':
    # Run tests
    asyncio.run(test_components())