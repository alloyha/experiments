# Architectural Plan for Custom Trading Bot

Let me design a comprehensive, production-ready architecture. I'll structure this as a proper technical specification you can implement incrementally.

---

## **I. System Architecture Overview**

### **High-Level Design Principles**

1. **Modularity**: Each component is independently testable and replaceable
2. **Exchange Agnosticism**: Strategy code never touches exchange-specific APIs
3. **State Management**: All decisions are reproducible from logs
4. **Fail-Safe**: System degrades gracefully; never loses track of positions
5. **Observable**: Every decision point generates structured logs for audit

### **Component Diagram**

```
┌─────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                            │
│                    (Main Event Loop)                            │
└────────┬────────────────────────────────────────────────┬───────┘
         │                                                 │
         ▼                                                 ▼
┌─────────────────┐                              ┌─────────────────┐
│  DATA PIPELINE  │                              │  EXECUTION      │
│                 │                              │  ENGINE         │
│ • Market Data   │                              │                 │
│ • Order Book    │                              │ • Order Router  │
│ • Trades        │                              │ • Position Mgr  │
└────────┬────────┘                              └────────┬────────┘
         │                                                 │
         ▼                                                 ▼
┌─────────────────┐         ┌─────────────────┐  ┌─────────────────┐
│  STRATEGY       │         │  RISK           │  │  EXCHANGE       │
│  ENGINE         │────────▶│  CONTROLLER     │  │  CONNECTOR      │
│                 │         │                 │  │                 │
│ • Indicators    │         │ • Position Size │  │ • API Wrapper   │
│ • Signals       │         │ • Stop Loss     │  │ • Rate Limiter  │
│ • Entry/Exit    │         │ • Exposure      │  │ • Error Handler │
└─────────────────┘         └─────────────────┘  └────────┬────────┘
         │                           │                     │
         │                           │                     ▼
         ▼                           ▼            ┌─────────────────┐
┌─────────────────────────────────────────────┐  │  EXCHANGE API   │
│           STATE & PERSISTENCE                │  │  (Binance, etc) │
│                                              │  └─────────────────┘
│ • Portfolio State                            │
│ • Trade History                              │
│ • Decision Logs (for audit)                  │
└──────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│       MONITORING & ALERTING                  │
│                                              │
│ • Metrics Dashboard                          │
│ • Log Aggregation                            │
│ • Alert System (Telegram/Email)              │
└──────────────────────────────────────────────┘
```

---

## **II. Component Specifications**

### **1. Exchange Connector (Abstraction Layer)**

**Purpose**: Isolate exchange-specific API details from business logic

**Interface Definition**:
```python
class ExchangeConnector(ABC):
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Current price, bid/ask, volume"""
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int) -> OrderBook:
        """Bid/ask levels with sizes"""
        pass
    
    @abstractmethod
    async def get_candles(self, symbol: str, timeframe: str, 
                          limit: int) -> List[Candle]:
        """OHLCV data"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """Submit market/limit order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, Balance]:
        """Account balances"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """Active orders"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Current positions (for futures/margin)"""
        pass
```

**Key Features**:
- **Rate Limiting**: Built-in request throttling (exchange limits: ~1200/min for Binance)
- **Retry Logic**: Exponential backoff for transient errors
- **WebSocket Support**: Real-time data via separate connection pool
- **Error Normalization**: Map exchange-specific errors to standard exceptions

**Implementation Strategy**:
- Use **CCXT** as the underlying library (supports 100+ exchanges)
- Wrap CCXT with your interface for additional features (caching, logging)
- Store API credentials in environment variables or secure vault

**Example**:
```python
class BinanceConnector(ExchangeConnector):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future' if testnet else 'spot'}
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
    
    async def get_ticker(self, symbol: str) -> Ticker:
        raw = await self.exchange.fetch_ticker(symbol)
        return Ticker(
            symbol=symbol,
            bid=raw['bid'],
            ask=raw['ask'],
            last=raw['last'],
            volume=raw['volume'],
            timestamp=raw['timestamp']
        )
```

---

### **2. Data Pipeline**

**Purpose**: Normalize, cache, and distribute market data to strategy engine

**Components**:

#### **A. Market Data Manager**
```python
class MarketDataManager:
    def __init__(self, connector: ExchangeConnector):
        self.connector = connector
        self.cache = TTLCache(maxsize=1000, ttl=60)  # 1-min cache
        self.subscribers = []
    
    async def get_latest_candles(self, symbol: str, 
                                  timeframe: str, 
                                  limit: int = 100) -> pd.DataFrame:
        """Returns OHLCV DataFrame with caching"""
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        candles = await self.connector.get_candles(symbol, timeframe, limit)
        df = pd.DataFrame(candles)
        self.cache[cache_key] = df
        return df
    
    async def stream_tickers(self, symbols: List[str]):
        """WebSocket stream for real-time prices"""
        # Implementation uses exchange WebSocket API
        pass
```

#### **B. Data Storage**
- **Hot Data** (last 24h): In-memory cache (Redis optional)
- **Warm Data** (last 30 days): Local SQLite/PostgreSQL
- **Cold Data** (historical): Parquet files or time-series DB

**Schema**:
```sql
-- Candles table
CREATE TABLE candles (
    symbol TEXT,
    timeframe TEXT,
    timestamp BIGINT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    PRIMARY KEY (symbol, timeframe, timestamp)
);

-- Ticks table (for HFT strategies)
CREATE TABLE ticks (
    symbol TEXT,
    timestamp BIGINT,
    bid REAL,
    ask REAL,
    last REAL,
    volume REAL
);
```

---

### **3. Strategy Engine**

**Purpose**: Generate trading signals from market data

**Architecture**:

```python
class Strategy(ABC):
    """Base class for all strategies"""
    
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to candle data"""
        pass
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, 
                         current_position: Position) -> Signal:
        """
        Returns: Signal(action='BUY'|'SELL'|'HOLD', 
                       confidence=0.0-1.0, 
                       reason="EMA crossover")
        """
        pass
    
    @abstractmethod
    def get_config(self) -> StrategyConfig:
        """Strategy parameters for logging"""
        pass
```

**Example Strategy Implementation**:
```python
class EMACrossoverStrategy(Strategy):
    def __init__(self, fast_period=12, slow_period=26, rsi_threshold=70):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_threshold = rsi_threshold
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = df['close'].ewm(span=self.fast_period).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period).mean()
        df['rsi'] = self._calculate_rsi(df['close'], period=14)
        return df
    
    def generate_signals(self, df: pd.DataFrame, 
                         current_position: Position) -> Signal:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Golden cross
        if (prev['ema_fast'] <= prev['ema_slow'] and 
            last['ema_fast'] > last['ema_slow'] and 
            last['rsi'] < self.rsi_threshold):
            return Signal(
                action='BUY',
                confidence=0.8,
                reason=f"EMA golden cross, RSI={last['rsi']:.1f}",
                metadata={'ema_fast': last['ema_fast'], 
                         'ema_slow': last['ema_slow']}
            )
        
        # Death cross
        if (prev['ema_fast'] >= prev['ema_slow'] and 
            last['ema_fast'] < last['ema_slow']):
            return Signal(
                action='SELL',
                confidence=0.8,
                reason=f"EMA death cross",
                metadata={'ema_fast': last['ema_fast'], 
                         'ema_slow': last['ema_slow']}
            )
        
        return Signal(action='HOLD', confidence=0.0, reason="No signal")
```

**Strategy Registry**:
```python
STRATEGIES = {
    'ema_crossover': EMACrossoverStrategy,
    'rsi_mean_reversion': RSIMeanReversionStrategy,
    'breakout': BreakoutStrategy,
    # ... add more strategies
}
```

---

### **4. Risk Controller**

**Purpose**: Validate all trading decisions against risk limits

**Components**:

#### **A. Position Sizer**
```python
class PositionSizer:
    def calculate_size(self, signal: Signal, 
                       portfolio: Portfolio,
                       risk_params: RiskParams) -> float:
        """
        Returns: position size in base currency
        
        Methods:
        - Fixed fractional (e.g., 2% of capital per trade)
        - Kelly Criterion (optimal bet sizing)
        - Volatility-adjusted (ATR-based)
        """
        
        if risk_params.method == 'fixed_fractional':
            risk_per_trade = portfolio.equity * risk_params.risk_fraction
            # Size = risk / stop_distance
            stop_distance = self._calculate_stop_distance(signal)
            size = risk_per_trade / stop_distance
            return min(size, risk_params.max_position_size)
```

#### **B. Stop-Loss Manager**
```python
class StopLossManager:
    def calculate_stop(self, entry_price: float, 
                       signal: Signal,
                       df: pd.DataFrame) -> StopLoss:
        """
        Returns: StopLoss(price, type='fixed'|'trailing'|'atr')
        """
        
        # ATR-based stop (2x ATR below entry)
        atr = self._calculate_atr(df, period=14)
        stop_price = entry_price - (2 * atr)
        
        return StopLoss(
            price=stop_price,
            type='atr',
            distance=2 * atr,
            metadata={'atr': atr}
        )
    
    def update_trailing_stop(self, current_price: float, 
                             position: Position) -> Optional[float]:
        """Update trailing stop if price moved favorably"""
        if position.stop_type == 'trailing':
            # Move stop to lock in profits
            pass
```

#### **C. Risk Checks**
```python
class RiskController:
    def validate_trade(self, order: Order, portfolio: Portfolio) -> ValidationResult:
        """
        Pre-trade validation:
        - Max position size check
        - Max portfolio exposure check
        - Max daily loss check
        - Correlation check (don't over-concentrate)
        """
        
        checks = []
        
        # Check 1: Position size limit
        if order.size > self.config.max_position_size:
            checks.append(f"Position size {order.size} exceeds limit")
        
        # Check 2: Portfolio exposure
        total_exposure = portfolio.calculate_exposure()
        if total_exposure + order.notional > self.config.max_exposure:
            checks.append(f"Total exposure would exceed limit")
        
        # Check 3: Daily loss limit
        if portfolio.daily_pnl < -self.config.max_daily_loss:
            checks.append(f"Daily loss limit reached: {portfolio.daily_pnl}")
        
        if checks:
            return ValidationResult(approved=False, reasons=checks)
        
        return ValidationResult(approved=True, reasons=[])
```

---

### **5. Execution Engine**

**Purpose**: Convert signals into actual orders, manage order lifecycle

**Components**:

#### **A. Order Router**
```python
class OrderRouter:
    async def execute_signal(self, signal: Signal, 
                            portfolio: Portfolio) -> ExecutionResult:
        """
        1. Calculate position size (via RiskController)
        2. Determine order type (market vs limit)
        3. Submit order to exchange
        4. Monitor fill status
        5. Update portfolio state
        """
        
        # Step 1: Risk validation
        validation = self.risk_controller.validate_trade(signal, portfolio)
        if not validation.approved:
            return ExecutionResult(
                status='REJECTED',
                reason=validation.reasons
            )
        
        # Step 2: Create order
        order = Order(
            symbol=signal.symbol,
            side='BUY' if signal.action == 'BUY' else 'SELL',
            type='MARKET',  # or 'LIMIT' based on urgency
            size=validation.approved_size,
            price=None  # market order
        )
        
        # Step 3: Submit to exchange
        try:
            result = await self.connector.place_order(order)
            
            # Step 4: Log execution
            self.logger.log_trade(
                order=order,
                result=result,
                signal=signal,
                portfolio_state=portfolio.to_dict()
            )
            
            return ExecutionResult(
                status='FILLED',
                order_id=result.order_id,
                fill_price=result.avg_price,
                fill_size=result.filled
            )
        
        except ExchangeError as e:
            return ExecutionResult(
                status='ERROR',
                reason=str(e)
            )
```

#### **B. Position Manager**
```python
class PositionManager:
    def __init__(self):
        self.positions: Dict[str, Position] = {}
    
    def open_position(self, symbol: str, entry: ExecutionResult, 
                      stop: StopLoss, target: TakeProfit):
        """Record new position"""
        self.positions[symbol] = Position(
            symbol=symbol,
            side=entry.side,
            entry_price=entry.fill_price,
            size=entry.fill_size,
            stop_loss=stop,
            take_profit=target,
            opened_at=datetime.now()
        )
    
    def update_position(self, symbol: str, current_price: float):
        """Update unrealized PnL, check stop/target"""
        position = self.positions[symbol]
        position.unrealized_pnl = self._calculate_pnl(position, current_price)
        
        # Check stop-loss
        if current_price <= position.stop_loss.price:
            return CloseSignal(reason='STOP_LOSS_HIT')
        
        # Check take-profit
        if current_price >= position.take_profit.price:
            return CloseSignal(reason='TAKE_PROFIT_HIT')
        
        # Update trailing stop if applicable
        if position.stop_loss.type == 'trailing':
            new_stop = self.stop_manager.update_trailing_stop(
                current_price, position
            )
            if new_stop:
                position.stop_loss.price = new_stop
        
        return None
    
    def close_position(self, symbol: str, exit: ExecutionResult):
        """Finalize position, record trade"""
        position = self.positions.pop(symbol)
        trade = Trade(
            symbol=symbol,
            entry_price=position.entry_price,
            exit_price=exit.fill_price,
            size=position.size,
            pnl=self._calculate_realized_pnl(position, exit),
            duration=datetime.now() - position.opened_at,
            exit_reason=exit.reason
        )
        self.trade_history.append(trade)
        return trade
```

---

### **6. State & Persistence**

**Purpose**: Ensure system can recover from crashes without losing track of positions

**Critical State**:
```python
class SystemState:
    portfolio: Portfolio          # Current balances, equity
    positions: List[Position]     # Open positions
    orders: List[Order]          # Pending orders
    trade_history: List[Trade]   # Closed trades
    decision_log: List[Decision] # Audit trail
```

**Storage Strategy**:
```python
class StateManager:
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self._init_tables()
    
    def save_state(self, state: SystemState):
        """Atomic write of current state"""
        with self.db:
            self.db.execute("BEGIN TRANSACTION")
            self._save_portfolio(state.portfolio)
            self._save_positions(state.positions)
            self._save_orders(state.orders)
            self.db.execute("COMMIT")
    
    def load_state(self) -> SystemState:
        """Restore state on startup"""
        portfolio = self._load_portfolio()
        positions = self._load_positions()
        orders = self._load_orders()
        return SystemState(portfolio, positions, orders)
    
    def log_decision(self, decision: Decision):
        """
        Append-only decision log for audit
        
        Schema:
        - timestamp
        - signal (BUY/SELL/HOLD)
        - reason
        - indicators (JSON)
        - risk_checks (JSON)
        - action_taken (EXECUTED/REJECTED/HELD)
        """
        self.db.execute("""
            INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?)
        """, (
            decision.timestamp,
            decision.signal.action,
            decision.signal.reason,
            json.dumps(decision.signal.metadata),
            json.dumps(decision.risk_validation),
            decision.action_taken
        ))
```

**Crash Recovery**:
```python
async def recover_from_crash(connector: ExchangeConnector, 
                             state_manager: StateManager):
    """
    On startup:
    1. Load last known state from DB
    2. Query exchange for actual positions/orders
    3. Reconcile differences (exchange is source of truth)
    4. Resume operation
    """
    saved_state = state_manager.load_state()
    
    # Get ground truth from exchange
    actual_balances = await connector.get_balance()
    actual_positions = await connector.get_positions()
    actual_orders = await connector.get_open_orders()
    
    # Reconcile
    reconciled_state = reconcile(saved_state, {
        'balances': actual_balances,
        'positions': actual_positions,
        'orders': actual_orders
    })
    
    logger.info(f"Recovered state: {reconciled_state}")
    return reconciled_state
```

---

### **7. Orchestrator (Main Event Loop)**

**Purpose**: Coordinate all components in a coherent trading cycle

```python
class TradingBot:
    def __init__(self, config: BotConfig):
        self.connector = self._create_connector(config)
        self.data_manager = MarketDataManager(self.connector)
        self.strategy = STRATEGIES[config.strategy_name](**config.strategy_params)
        self.risk_controller = RiskController(config.risk_params)
        self.order_router = OrderRouter(self.connector, self.risk_controller)
        self.position_manager = PositionManager()
        self.state_manager = StateManager(config.db_path)
        
    async def run(self):
        """Main event loop"""
        
        # Recover from any previous crash
        state = await recover_from_crash(self.connector, self.state_manager)
        
        logger.info("Bot started")
        
        while True:
            try:
                await self._trading_cycle()
                await asyncio.sleep(self.config.cycle_interval)
            
            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")
                await self._shutdown()
                break
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                await self._handle_error(e)
    
    async def _trading_cycle(self):
        """
        Single iteration of trading logic:
        1. Fetch market data
        2. Calculate indicators
        3. Generate signal
        4. Risk validation
        5. Execute trade (if approved)
        6. Update positions (check stops/targets)
        7. Save state
        8. Log decision
        """
        
        # Step 1: Get latest data
        df = await self.data_manager.get_latest_candles(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            limit=200  # enough for indicators
        )
        
        # Step 2: Calculate indicators
        df = self.strategy.calculate_indicators(df)
        
        # Step 3: Generate signal
        current_position = self.position_manager.get_position(self.config.symbol)
        signal = self.strategy.generate_signals(df, current_position)
        
        # Step 4: Risk validation
        if signal.action in ['BUY', 'SELL']:
            validation = self.risk_controller.validate_trade(
                signal, self.state.portfolio
            )
        else:
            validation = None
        
        # Step 5: Execute if approved
        execution_result = None
        if validation and validation.approved:
            execution_result = await self.order_router.execute_signal(
                signal, self.state.portfolio
            )
            
            if execution_result.status == 'FILLED':
                # Open new position
                stop = self.risk_controller.calculate_stop(
                    execution_result.fill_price, signal, df
                )
                self.position_manager.open_position(
                    self.config.symbol, execution_result, stop
                )
        
        # Step 6: Update existing positions
        if current_position:
            current_price = df.iloc[-1]['close']
            close_signal = self.position_manager.update_position(
                self.config.symbol, current_price
            )
            
            if close_signal:
                # Close position (stop/target hit)
                close_result = await self.order_router.close_position(
                    current_position
                )
                trade = self.position_manager.close_position(
                    self.config.symbol, close_result
                )
                logger.info(f"Trade closed: {trade}")
        
        # Step 7: Save state
        self.state_manager.save_state(self.get_current_state())
        
        # Step 8: Log decision for audit
        decision = Decision(
            timestamp=datetime.now(),
            signal=signal,
            risk_validation=validation,
            action_taken='EXECUTED' if execution_result else 'HELD',
            execution_result=execution_result,
            portfolio_state=self.state.portfolio.to_dict()
        )
        self.state_manager.log_decision(decision)
```

---

### **8. Monitoring & Alerting**

**Purpose**: Observe system health and trading performance

**Components**:

#### **A. Metrics Collector**
```python
class MetricsCollector:
    def collect(self) -> Dict:
        return {
            'timestamp': datetime.now(),
            'portfolio': {
                'equity': portfolio.equity,
                'cash': portfolio.cash,
                'unrealized_pnl': portfolio.unrealized_pnl,
                'daily_pnl': portfolio.daily_pnl
            },
            'positions': {
                'count': len(position_manager.positions),
                'total_exposure': sum(p.notional for p in positions)
            },
            'trades': {
                'count_today': len([t for t in trades if t.date == today]),
                'win_rate': calculate_win_rate(trades),
                'avg_pnl': calculate_avg_pnl(trades)
            },
            'system': {
                'uptime': get_uptime(),
                'api_latency': measure_api_latency(),
                'error_count': error_counter.value
            }
        }
```

#### **B. Alert System**
```python
class AlertManager:
    def __init__(self, telegram_token: str, chat_id: str):
        self.bot = telegram.Bot(token=telegram_token)
        self.chat_id = chat_id
    
    async def send_alert(self, level: str, message: str):
        """
        Levels: INFO, WARNING, ERROR, CRITICAL
        """
        if level in ['ERROR', 'CRITICAL']:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=f"🚨 {level}: {message}"
            )
    
    async def send_trade_notification(self, trade: Trade):
        """Notify on every trade execution"""
        msg = f"""
        {'🟢' if trade.pnl > 0 else '🔴'} Trade Closed
        Symbol: {trade.symbol}
        PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.1f}%)
        Duration: {trade.duration}
        Reason: {trade.exit_reason}
        """
        await self.bot.send_message(chat_id=self.chat_id, text=msg)
```

#### **C. Dashboard (Optional)**
- Simple web UI using Flask/FastAPI
- Display current positions, recent trades, equity curve
- Real-time logs stream

---

## **III. Configuration Management**

**config.yaml**:
```yaml
# Exchange Configuration
exchange:
  name: binance
  testnet: true  # Use testnet for sim mode
  api_key: ${BINANCE_API_KEY}
  api_secret: ${BINANCE_API_SECRET}

# Trading Parameters
trading:
  symbol: BTC/USDT
  timeframe: 15m
  cycle_interval: 60  # seconds between strategy evaluations

# Strategy Configuration
strategy:
  name: ema_crossover
  params:
    fast_period: 12
    slow_period: 26
    rsi_threshold: 70

# Risk Management
risk:
  max_position_size: 1000  # USDT
  max_portfolio_exposure: 5000  # USDT (total across all positions)
  max_daily_loss: 500  # USDT
  position_sizing_method: fixed_fractional
  risk_per_trade: 0.02  # 2% of capital
  stop_loss_method: atr
  stop_loss_multiplier: 2.0

# Monitoring
monitoring:
  telegram_token: ${TELEGRAM_TOKEN}
  telegram_chat_id: ${TELEGRAM_CHAT_ID}
  log_level: INFO
  metrics_interval: 300  # seconds

# Persistence
database:
  path: ./data/trading_bot.db
```

---

## **IV. Implementation Roadmap**

### **Phase 1: Foundation (Week 1-2)**
**Goal**: Basic infrastructure without trading logic

Tasks:
1. Set up project structure, dependencies (CCXT, pandas, SQLite)
2. Implement `ExchangeConnector` with CCXT wrapper
3. Test API connectivity (get tickers, balances) on testnet
4. Implement `StateManager` with SQLite persistence
5. Build basic logging framework

**Deliverable**: Can connect to exchange, fetch data, store to DB

---

### **Phase 2: Data Pipeline (Week 3)**
**Goal**: Reliable market data ingestion

Tasks:
1. Implement `MarketDataManager` with caching
2. Build historical data downloader
3. Create candle database schema
4. Test data quality (handle missing candles, outliers)

**Deliverable**: Can fetch and store 6 months of historical data

---

### **Phase 3: Strategy Engine (Week 4-5)**
**Goal**: Generate trading signals from data

Tasks:
1. Implement `Strategy` base class
2. Build one simple strategy (EMA crossover)
3. Create backtesting framework (replay historical data)
4. Test strategy on historical data, measure signals

**Deliverable**: Strategy generates BUY/SELL signals with reasoning

---

### **Phase 4: Risk & Execution (Week 6-7)**
**Goal**: Convert signals into safe orders

Tasks:
1. Implement `RiskController` (position sizing, validation)
2. Build `StopLossManager` (ATR-based stops)
3. Implement `OrderRouter` (submit orders to exchange)
4. Test on testnet with TINY positions ($10)

**Deliverable**: Bot places real orders on testnet, respects risk limits

---

### **Phase 5: Position Management (Week 8)**
**Goal**: Track positions, handle stops/targets

Tasks:
1. Implement `PositionManager`
2. Build position update logic (check stops, trailing)
3. Test full trade lifecycle: open → monitor → close
4. Verify state persistence across restarts

**Deliverable**: Bot opens position, monitors it, closes on stop/target

---

### **Phase 6: Monitoring (Week 9)**
**Goal**: Observe what the bot is doing

Tasks:
1. Implement `MetricsCollector`
2. Build `AlertManager` (Telegram notifications)
3. Create decision logging for audit
4. Build simple dashboard (optional)

**Deliverable**: Receive notifications on trades, can audit decisions

---

### **Phase 7: Paper Trading (Week 10-12)**
**Goal**: Run continuously in sim mode, refine

Tasks:
1. Deploy bot on server/cloud (AWS/DigitalOcean)
2. Run 24/7 in paper trading mode
3. Monitor performance, fix bugs
4. Tune strategy parameters based on results

**Deliverable**: Bot runs for 2+ weeks without crashes, generates audit logs

---

### **Phase 8: Live Trading (Week 13+)**
**Goal**: Deploy with real money (small capital)

Tasks:
1. **Pre-launch checklist**:
   - Review all decision logs from paper trading
   - Analyze win rate, avg PnL, max drawdown
   - Verify risk limits are properly enforced
   - Test crash recovery (kill bot mid-trade, restart)
   - Audit code for security vulnerabilities
2. **Gradual rollout**:
   - Start with $100-500 capital
   - Monitor closely for first week
   - Compare live vs paper trading results
3. **Refinement**:
   - Adjust risk parameters based on live slippage
   - Fine-tune strategy if needed
   - Scale capital if consistently profitable

**Deliverable**: Bot trades live profitably (or fails fast with minimal loss)

---

## **V. Testing Strategy**

### **A. Unit Tests**

Test each component in isolation:

```python
# test_strategy.py
def test_ema_crossover_golden_cross():
    strategy = EMACrossoverStrategy(fast_period=5, slow_period=10)
    
    # Create synthetic data with known crossover
    df = pd.DataFrame({
        'close': [100, 101, 102, 105, 110, 115, 120]
    })
    df = strategy.calculate_indicators(df)
    
    signal = strategy.generate_signals(df, current_position=None)
    
    assert signal.action == 'BUY'
    assert 'golden cross' in signal.reason.lower()

# test_risk_controller.py
def test_position_size_limit():
    risk_controller = RiskController(max_position_size=1000)
    
    signal = Signal(action='BUY', size=2000)  # Over limit
    validation = risk_controller.validate_trade(signal, portfolio)
    
    assert validation.approved == False
    assert 'exceeds limit' in validation.reasons[0]
```

### **B. Integration Tests**

Test component interactions:

```python
# test_execution_flow.py
async def test_signal_to_execution():
    """Test full flow: signal → risk check → order → position"""
    
    # Setup
    bot = TradingBot(test_config)
    
    # Inject synthetic signal
    signal = Signal(action='BUY', symbol='BTC/USDT', confidence=0.8)
    
    # Execute
    result = await bot.order_router.execute_signal(signal, bot.portfolio)
    
    # Verify
    assert result.status == 'FILLED'
    assert bot.position_manager.has_position('BTC/USDT')
    assert len(bot.state_manager.get_decision_log()) == 1
```

### **C. Backtesting**

Test strategy on historical data:

```python
class Backtester:
    def __init__(self, strategy: Strategy, data: pd.DataFrame):
        self.strategy = strategy
        self.data = data
        self.trades = []
        self.equity_curve = []
    
    def run(self, initial_capital: float = 10000) -> BacktestResult:
        """
        Replay historical data bar-by-bar, simulating trading
        """
        portfolio = Portfolio(cash=initial_capital)
        position = None
        
        for i in range(100, len(self.data)):  # Need 100 bars for indicators
            # Get data up to current bar (no lookahead bias)
            df_slice = self.data.iloc[:i+1]
            
            # Calculate indicators
            df_slice = self.strategy.calculate_indicators(df_slice)
            
            # Generate signal
            signal = self.strategy.generate_signals(df_slice, position)
            
            # Simulate execution
            if signal.action == 'BUY' and position is None:
                entry_price = df_slice.iloc[-1]['close']
                size = portfolio.cash * 0.95 / entry_price  # 95% of capital
                position = Position(
                    entry_price=entry_price,
                    size=size,
                    stop=entry_price * 0.98  # 2% stop
                )
                portfolio.cash -= position.notional
            
            elif signal.action == 'SELL' and position:
                exit_price = df_slice.iloc[-1]['close']
                pnl = (exit_price - position.entry_price) * position.size
                portfolio.cash += position.notional + pnl
                
                self.trades.append(Trade(
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    pnl=pnl
                ))
                position = None
            
            # Update equity
            equity = portfolio.cash
            if position:
                current_price = df_slice.iloc[-1]['close']
                equity += position.size * current_price
            self.equity_curve.append(equity)
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            final_equity=self.equity_curve[-1],
            total_return=(self.equity_curve[-1] - initial_capital) / initial_capital,
            win_rate=sum(1 for t in self.trades if t.pnl > 0) / len(self.trades),
            max_drawdown=self._calculate_max_drawdown(),
            sharpe_ratio=self._calculate_sharpe()
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Maximum peak-to-trough decline"""
        peak = self.equity_curve[0]
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        return max_dd
```

**Usage**:
```python
# Load 1 year of BTC data
df = load_historical_data('BTC/USDT', '15m', start='2024-01-01')

# Backtest strategy
strategy = EMACrossoverStrategy(fast_period=12, slow_period=26)
backtester = Backtester(strategy, df)
result = backtester.run(initial_capital=10000)

print(f"Total Return: {result.total_return:.1%}")
print(f"Win Rate: {result.win_rate:.1%}")
print(f"Max Drawdown: {result.max_drawdown:.1%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

# Plot equity curve
plt.plot(result.equity_curve)
plt.title('Backtest Equity Curve')
plt.show()
```

### **D. Paper Trading Tests**

Before live trading:

1. **Consistency check**: Run paper and backtest simultaneously on same data. Results should match (within slippage tolerance).

2. **Stress tests**:
   - Kill bot during active trade → verify recovery
   - Simulate exchange API errors → verify retries
   - Inject extreme price moves → verify stops trigger

3. **Performance monitoring**:
   - Track API latency (should be <500ms)
   - Monitor memory usage (no leaks)
   - Verify decision logs are complete

---

## **VI. Security & Operational Considerations**

### **A. API Key Security**

```python
# NEVER hardcode keys
# ❌ BAD
api_key = "abc123..."

# ✅ GOOD: Environment variables
import os
api_key = os.getenv('BINANCE_API_KEY')

# ✅ BETTER: Secret management service (AWS Secrets Manager, HashiCorp Vault)
from aws_secretsmanager import get_secret
api_key = get_secret('trading_bot/binance_api_key')
```

**API Key Permissions**:
- Enable: "Read", "Trade" (spot)
- Disable: "Withdraw", "Transfer", "Margin"
- Whitelist your server IP if possible

### **B. Error Handling Patterns**

```python
class ExchangeError(Exception):
    """Base exception for exchange errors"""
    pass

class RateLimitError(ExchangeError):
    """Hit rate limit"""
    pass

class InsufficientBalanceError(ExchangeError):
    """Not enough funds"""
    pass

class OrderRejectedError(ExchangeError):
    """Order rejected by exchange"""
    pass


async def place_order_with_retry(order: Order, max_retries=3):
    """Retry transient errors, fail fast on fatal errors"""
    
    for attempt in range(max_retries):
        try:
            return await connector.place_order(order)
        
        except RateLimitError:
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
            continue
        
        except InsufficientBalanceError:
            # Fatal: don't retry
            logger.error("Insufficient balance, halting bot")
            await shutdown_bot()
            raise
        
        except ExchangeError as e:
            # Unknown error: retry with caution
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(5)
    
    raise Exception("Max retries exceeded")
```

### **C. Monitoring & Alerts**

**Critical alerts** (immediate notification):
- Order execution failed
- Position not found after order fill
- Daily loss limit breached
- Bot crashed
- Exchange API unreachable for >5 min

**Warning alerts** (check within 1 hour):
- Unusual slippage detected
- Win rate dropped below threshold
- API latency degraded

**Info notifications**:
- Trade executed
- Daily performance summary

```python
# Alert thresholds
ALERT_CONFIG = {
    'max_api_latency_ms': 1000,
    'min_win_rate': 0.40,  # Alert if win rate < 40%
    'max_drawdown_pct': 0.15,  # Alert if drawdown > 15%
    'max_position_age_hours': 72  # Alert if position open > 3 days
}
```

### **D. Disaster Recovery**

**Scenario 1: Bot crashes mid-trade**
- On restart: Query exchange for actual positions
- Compare with saved state
- If discrepancy: Use exchange as source of truth
- Log discrepancy for investigation

**Scenario 2: Exchange downtime**
```python
async def handle_exchange_downtime():
    """
    If exchange API is down:
    1. Cancel all pending orders (when API recovers)
    2. Close all positions at market (emergency exit)
    3. Halt bot until manual review
    """
    logger.critical("Exchange API down, initiating emergency protocol")
    
    # Wait for API to recover
    while not await connector.is_healthy():
        await asyncio.sleep(10)
    
    # Emergency exit
    positions = await connector.get_positions()
    for position in positions:
        await connector.place_order(Order(
            symbol=position.symbol,
            side='SELL' if position.side == 'LONG' else 'BUY',
            type='MARKET',
            size=position.size
        ))
    
    await shutdown_bot()
```

**Scenario 3: Flash crash**
- Stop-loss should trigger automatically
- If bot is frozen: Exchange-side stop-loss orders (if supported)
- Manual kill switch: Emergency shutdown command

### **E. Logging Strategy**

**Three log levels**:

1. **Decision Log** (audit trail):
```python
{
  "timestamp": "2025-10-30T14:23:45Z",
  "signal": {"action": "BUY", "reason": "EMA golden cross", "confidence": 0.8},
  "indicators": {"ema_fast": 42150, "ema_slow": 41980, "rsi": 58.3},
  "risk_validation": {"approved": true, "position_size": 0.05},
  "action_taken": "EXECUTED",
  "order_id": "abc123",
  "fill_price": 42180,
  "portfolio_state": {"equity": 10500, "positions": 1}
}
```

2. **Trade Log** (performance tracking):
```python
{
  "trade_id": "trade_001",
  "symbol": "BTC/USDT",
  "entry_time": "2025-10-30T14:23:45Z",
  "entry_price": 42180,
  "exit_time": "2025-10-30T18:15:22Z",
  "exit_price": 42650,
  "size": 0.05,
  "pnl": 23.5,
  "pnl_pct": 1.11,
  "duration_hours": 3.9,
  "exit_reason": "TAKE_PROFIT_HIT"
}
```

3. **System Log** (debugging):
```python
# Standard Python logging
logger.info("Bot started")
logger.debug(f"Fetched {len(df)} candles")
logger.warning(f"API latency high: {latency}ms")
logger.error("Order submission failed", exc_info=True)
```

**Log retention**:
- Decision logs: Keep forever (audit trail)
- Trade logs: Keep forever (performance analysis)
- System logs: Keep 90 days

---

## **VII. Advanced Features (Future Enhancements)**

Once the core system is stable, consider adding:

### **A. Multi-Symbol Trading**

```python
class MultiSymbolBot(TradingBot):
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.strategies = {sym: self._create_strategy(sym) for sym in symbols}
        self.position_managers = {sym: PositionManager() for sym in symbols}
    
    async def _trading_cycle(self):
        """Run strategy for each symbol"""
        for symbol in self.symbols:
            await self._run_symbol_strategy(symbol)
```

**Considerations**:
- Correlation risk: Don't open correlated positions (BTC + ETH are 0.8 correlated)
- Capital allocation: How to split capital across symbols?
- Portfolio-level stops: Max total exposure limit

### **B. Strategy Portfolio**

Run multiple strategies on same symbol:

```python
class StrategyPortfolio:
    def __init__(self):
        self.strategies = [
            EMACrossoverStrategy(weight=0.3),
            RSIMeanReversionStrategy(weight=0.3),
            BreakoutStrategy(weight=0.4)
        ]
    
    def generate_signals(self, df: pd.DataFrame) -> Signal:
        """
        Ensemble: Combine signals from multiple strategies
        
        Methods:
        - Voting: Buy if 2/3 strategies say buy
        - Weighted: Signal strength = weighted average of confidences
        - Conditional: Use strategy A in trending markets, B in ranging
        """
        signals = [s.generate_signals(df) for s in self.strategies]
        
        # Weighted voting
        buy_votes = sum(s.confidence * s.weight 
                        for s, strat in zip(signals, self.strategies) 
                        if s.action == 'BUY')
        
        if buy_votes > 0.5:
            return Signal(action='BUY', confidence=buy_votes)
        
        return Signal(action='HOLD', confidence=0)
```

### **C. Machine Learning Integration**

```python
class MLStrategy(Strategy):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)  # Pre-trained LSTM or RandomForest
    
    def generate_signals(self, df: pd.DataFrame, position) -> Signal:
        # Extract features
        features = self._extract_features(df)
        
        # Predict price direction
        prediction = self.model.predict(features)
        # prediction = [prob_down, prob_up]
        
        if prediction[1] > 0.65:  # 65% confidence threshold
            return Signal(
                action='BUY',
                confidence=prediction[1],
                reason=f"ML prediction: {prediction[1]:.1%} up"
            )
        
        return Signal(action='HOLD', confidence=0)
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Feature engineering:
        - Returns over multiple timeframes
        - Volatility measures
        - Volume patterns
        - Technical indicators
        """
        return np.array([
            df['close'].pct_change(periods=1).iloc[-1],
            df['close'].pct_change(periods=5).iloc[-1],
            df['close'].pct_change(periods=20).iloc[-1],
            df['volume'].rolling(20).mean().iloc[-1],
            df['rsi'].iloc[-1] / 100,
            # ... more features
        ])
```

**ML Model Training** (separate pipeline):
```python
# Train model offline, then deploy
from sklearn.ensemble import RandomForestClassifier

# Load historical data
df = load_historical_data('BTC/USDT', '1h', start='2023-01-01')

# Create labels (1 if price up in next 4 hours, 0 otherwise)
df['target'] = (df['close'].shift(-4) > df['close']).astype(int)

# Extract features
X = extract_features(df)
y = df['target']

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.1%}")

# Save
joblib.dump(model, 'models/btc_predictor_v1.pkl')
```

### **D. Adaptive Parameters**

Adjust strategy parameters based on market regime:

```python
class AdaptiveStrategy(Strategy):
    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.params = {
            'trending': {'fast_ema': 12, 'slow_ema': 26},
            'ranging': {'fast_ema': 20, 'slow_ema': 50}
        }
    
    def generate_signals(self, df: pd.DataFrame, position) -> Signal:
        # Detect current regime
        regime = self.regime_detector.detect(df)
        
        # Use regime-specific parameters
        params = self.params[regime]
        
        # Calculate indicators with adaptive parameters
        df['ema_fast'] = df['close'].ewm(span=params['fast_ema']).mean()
        df['ema_slow'] = df['close'].ewm(span=params['slow_ema']).mean()
        
        # Generate signal
        # ...
```

### **E. Distributed Architecture**

For high-frequency or multi-exchange trading:

```
┌─────────────┐
│   Redis     │  ← Shared state (positions, orders)
└──────┬──────┘
       │
       ├─────────┬─────────┬─────────
       │         │         │
┌──────▼─────┐ ┌▼─────┐ ┌─▼──────┐
│ Data Feed  │ │Signal│ │Execute │
│ Service    │ │Gen   │ │Service │
│            │ │Service│ │        │
│ (WS feeds) │ │      │ │(Orders)│
└────────────┘ └──────┘ └────────┘
```

**Benefits**:
- Scale components independently
- Fault isolation (one service crash doesn't kill bot)
- Multi-exchange support (one data service per exchange)

**Implementation**: Docker + Kubernetes or simple multiprocessing

---

## **VIII. Key Metrics to Track**

### **Strategy Metrics**

```python
class PerformanceMetrics:
    def calculate(self, trades: List[Trade]) -> Dict:
        return {
            # Profitability
            'total_pnl': sum(t.pnl for t in trades),
            'total_return_pct': self._calculate_return(trades),
            'win_rate': sum(1 for t in trades if t.pnl > 0) / len(trades),
            'avg_win': np.mean([t.pnl for t in trades if t.pnl > 0]),
            'avg_loss': np.mean([t.pnl for t in trades if t.pnl < 0]),
            'profit_factor': self._profit_factor(trades),  # avg_win / avg_loss
            
            # Risk
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe(trades),
            'sortino_ratio': self._calculate_sortino(trades),
            'max_consecutive_losses': self._max_consecutive_losses(trades),
            
            # Activity
            'total_trades': len(trades),
            'avg_trade_duration': np.mean([t.duration for t in trades]),
            'trades_per_day': len(trades) / self._days_elapsed(),
            
            # Execution
            'avg_slippage': np.mean([t.slippage for t in trades]),
            'total_fees': sum(t.fees for t in trades)
        }
```

**Target benchmarks** (realistic for crypto):
- Win rate: >45%
- Profit factor: >1.5
- Sharpe ratio: >1.0
- Max drawdown: <20%

### **System Health Metrics**

```python
HEALTH_METRICS = {
    'api_latency_p50': 200,  # ms
    'api_latency_p99': 800,  # ms
    'order_fill_rate': 0.98,  # 98% of orders fill
    'decision_cycle_time': 10,  # seconds
    'uptime': 0.999  # 99.9% uptime
}
```

---

## **IX. Example: Complete Minimal Bot**

Here's a simplified but complete bot to get started:

```python
# minimal_bot.py
import ccxt
import pandas as pd
import time
from datetime import datetime

class MinimalBot:
    def __init__(self, symbol='BTC/USDT', testnet=True):
        self.symbol = symbol
        self.exchange = ccxt.binance({
            'apiKey': 'YOUR_API_KEY',
            'secret': 'YOUR_SECRET',
            'enableRateLimit': True
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
        
        self.position = None
        self.trades = []
    
    def fetch_data(self, timeframe='15m', limit=100):
        """Fetch OHLCV candles"""
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def calculate_indicators(self, df):
        """Simple EMA crossover"""
        df['ema_fast'] = df['close'].ewm(span=12).mean()
        df['ema_slow'] = df['close'].ewm(span=26).mean()
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        return df
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_signal(self, df):
        """Simple signal logic"""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Golden cross
        if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
            if last['rsi'] < 70:
                return 'BUY'
        
        # Death cross
        if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
            return 'SELL'
        
        return 'HOLD'
    
    def execute_trade(self, signal, current_price):
        """Execute buy/sell"""
        if signal == 'BUY' and self.position is None:
            # Open position
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            size = (usdt_balance * 0.95) / current_price  # 95% of balance
            
            order = self.exchange.create_market_buy_order(self.symbol, size)
            self.position = {
                'entry_price': current_price,
                'size': size,
                'entry_time': datetime.now(),
                'stop_loss': current_price * 0.98  # 2% stop
            }
            print(f"✅ BUY {size:.4f} BTC at ${current_price}")
        
        elif signal == 'SELL' and self.position:
            # Close position
            order = self.exchange.create_market_sell_order(self.symbol, self.position['size'])
            pnl = (current_price - self.position['entry_price']) * self.position['size']
            
            self.trades.append({
                'entry': self.position['entry_price'],
                'exit': current_price,
                'pnl': pnl,
                'duration': datetime.now() - self.position['entry_time']
            })
            
            print(f"✅ SELL {self.position['size']:.4f} BTC at ${current_price}, PnL: ${pnl:.2f}")
            self.position = None
    
    def check_stop_loss(self, current_price):
        """Check if stop loss hit"""
        if self.position and current_price <= self.position['stop_loss']:
            print(f"🛑 STOP LOSS HIT at ${current_price}")
            self.execute_trade('SELL', current_price)
    
    def run(self):
        """Main loop"""
        print(f"🤖 Bot started, trading {self.symbol}")
        
        while True:
            try:
                # Fetch data
                df = self.fetch_data()
                df = self.calculate_indicators(df)
                
                # Current price
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
                
                # Check stop loss
                self.check_stop_loss(current_price)
                
                # Generate signal
                signal = self.generate_signal(df)
                
                # Execute
                if signal != 'HOLD':
                    self.execute_trade(signal, current_price)
                
                # Print status
                status = f"Position: {self.position is not None}, Trades: {len(self.trades)}"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")
                
                # Sleep
                time.sleep(60)  # Run every minute
            
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(60)

# Run bot
if __name__ == '__main__':
    bot = MinimalBot(testnet=True)
    bot.run()
```

**Usage**:
```bash
# Install dependencies
pip install ccxt pandas

# Set API keys in environment
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"

# Run
python minimal_bot.py
```

---

## **X. Final Recommendations**

### **Do's**
✅ Start with **paper trading** (testnet) for 2-4 weeks minimum  
✅ Keep **detailed logs** of every decision (your audit trail)  
✅ Implement **robust error handling** (exchanges are unreliable)  
✅ Use **stop losses** on every trade (no exceptions)  
✅ Start with **tiny capital** ($100-500) when going live  
✅ **Backtest** thoroughly on 6+ months of data  
✅ Monitor **slippage and fees** (they eat into profits)  
✅ Have a **kill switch** (manual shutdown capability)  

### **Don'ts**
❌ Don't trade without **stop losses**  
❌ Don't risk more than **2-5% per trade**  
❌ Don't **over-optimize** (curve fitting to historical data)  
❌ Don't trust **backtest returns** without live validation  
❌ Don't ignore **execution costs** (slippage, fees, spread)  
❌ Don't trade with money you **can't afford to lose**  
❌ Don't expect **consistent profits** (crypto is volatile)  
❌ Don't leave API keys in **code or git repos**  

### **Reality Check**
- Most retail algo traders **don't beat buy-and-hold**
- Your edge decays as more people use similar strategies
- Transaction costs are significant (0.1% per trade = 20% annual if trading 100x)
- You're competing against well-funded quant firms with superior infrastructure

**However**: Building a bot is valuable for:
- Learning systematic trading
- Understanding market microstructure
- Forcing discipline (no emotional trades)
- Automating repetitive tasks (rebalancing, DCA)

