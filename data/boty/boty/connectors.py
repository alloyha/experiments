


# ============================================================================
# EXCHANGE CONNECTOR (Abstraction Layer)
# ============================================================================

class ExchangeConnector(ABC):
    """Abstract interface for exchange operations"""
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        pass
    
    @abstractmethod
    async def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        pass
    
    @abstractmethod
    async def place_market_order(self, symbol: str, side: str, size: float) -> Dict:
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        pass


class BinanceConnector(ExchangeConnector):
    """Binance implementation using CCXT"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
        
        self.logger = logging.getLogger(__name__)
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Fetch current ticker"""
        try:
            raw = await self._async_fetch(self.exchange.fetch_ticker, symbol)
            return Ticker(
                symbol=symbol,
                bid=raw.get('bid', 0),
                ask=raw.get('ask', 0),
                last=raw['last'],
                volume=raw['quoteVolume'],
                timestamp=raw['timestamp']
            )
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {e}")
            raise
    
    async def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV candles"""
        try:
            ohlcv = await self._async_fetch(
                self.exchange.fetch_ohlcv,
                symbol, timeframe, limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        
        except Exception as e:
            self.logger.error(f"Error fetching candles: {e}")
            raise
    
    async def get_balance(self) -> Dict[str, float]:
        """Fetch account balances"""
        try:
            balance = await self._async_fetch(self.exchange.fetch_balance)
            return {k: v['free'] for k, v in balance.items() if v['free'] > 0}
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            raise
    
    async def place_market_order(self, symbol: str, side: str, size: float) -> Dict:
        """Place market order"""
        try:
            order_func = (self.exchange.create_market_buy_order if side == 'BUY' 
                         else self.exchange.create_market_sell_order)
            
            order = await self._async_fetch(order_func, symbol, size)
            
            return {
                'order_id': order['id'],
                'symbol': symbol,
                'side': side,
                'size': size,
                'filled': order.get('filled', size),
                'avg_price': order.get('average', order.get('price', 0)),
                'status': order['status']
            }
        
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """Get open positions (for spot, this is just non-zero balances)"""
        balances = await self.get_balance()
        # For spot trading, we'll track positions in our own state manager
        return []
    
    async def _async_fetch(self, func, *args, **kwargs):
        """Wrapper to run sync CCXT methods in async context"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
