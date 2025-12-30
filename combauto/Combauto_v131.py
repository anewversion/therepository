#!/usr/bin/env python3
"""
================================================================================
Combauto v131 - Dual Engine Statistical + ML  - Strategic Trading Automation
================================================================================

PHILOSOPHY: "Perfection is achieved not when there is nothing more to add, 
            but when there is nothing left to take away." - Antoine de Saint-Exupéry

BUY STAR SYSTEM:
- Statistical Engine: CSD Quintiles + Multi-factor confirmation (1-5 stars)
- ML Engine: LightGBM with Optuna optimization (1-5 stars)
- Arbiter: Combined stars ≥ 7 = BUY

DUAL ENGINE SYSTEM:
Engine 1 (Statistical): CSD Quintiles + RSI/OBV + Trend Analysis (1-5 stars)
Engine 2 (ML): Optuna-optimized LightGBM targeting 1% gains in 1-2h (1-5 stars)
Arbiter: Only buy when combined stars ≥ * 7    # Requires at least 2 levels of agreement (2 stars+) from either engine

STATE MANAGEMENT:
- open__buy_orders.json: Pending buy orders
- open__sell_orders.json: Pending sell orders
- active_positions.json: Filled buys awaiting sell
- trade_history.json: Completed trades with P&L analytics
"""

import asyncio
import json
import os
import sys
import time
import uuid
import logging
import logging.handlers
import tempfile
import shutil
import threading
import signal
from datetime import datetime, timezone, timedelta
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_HALF_UP, InvalidOperation
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import warnings
import traceback

import pandas as pd
import numpy as np
if not hasattr(np, 'NaN'): np.NaN = np.nan
import pandas_ta as ta
from coinbase.rest import RESTClient
from coinbase.rest.types.orders_types import CreateOrderResponse
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# === CONFIGURATION ===
load_dotenv()
getcontext().prec = 10

# Global shutdown flag
shutdown_flag = threading.Event()

# Core Parameters
SYMBOLS = [
    "AERO-USD", "APT-USD", "AVAX-USD", "AXS-USD", "BCH-USD", 
    "BONK-USD", "FET-USD", "FIL-USD", "FLOKI-USD", "HBAR-USD", 
    "HNT-USD", "ICP-USD", "IMX-USD", "INJ-USD", "JASMY-USD", 
    "LDO-USD", "NEAR-USD", "ONDO-USD", "OP-USD", "PENDLE-USD", 
    "PEPE-USD", "RNDR-USD", "RPL-USD", "SEI-USD", "SHIB-USD", 
    "SOL-USD", "TAO-USD", "TIA-USD", "WIF-USD", "WLD-USD"
]
POSITION_SIZE_USD = Decimal("25.00")
PROFIT_TARGET_PCT = Decimal("1.1")
POSITION_TIMEOUT_HOURS = 3
COMBINED_STAR_THRESHOLD = 7  # Need 7 combined stars to buy

# GTC Order Parameters
GTC_BUY_OFFSET_PCT = Decimal("0.001")   # Buy 0.00% below market

# API Limits (Based on official Coinbase documentation)
CANDLE_BATCH_LIMIT = 300  # Maximum candles per request
MAX_CONCURRENT_REQUESTS = 3  # Reduced for stability
API_TIMEOUT_SECONDS = 30

# Engine Parameters
ML_HISTORICAL_DAYS = 15
ML_OPTUNA_TRIALS = 30
ML_TARGET_HOURS = 1  # Predict 1% gain within 1 hours

# Intervals
ANALYSIS_INTERVAL_SECONDS = 15  # Frequent analysis
ORDER_CHECK_INTERVAL_SECONDS = 6   # Relaxed to 6s to respect rate limits
POSITION_CHECK_INTERVAL_SECONDS = 5  # EVEN MORE AGGRESSIVE: Monitor positions every 5 seconds
STATUS_DISPLAY_INTERVAL_MINUTES = 1

# === NEW CONFIGURATION FROM Combauto_vXO ===
# Smart sizing & VWAP-inspired buy slicing
DEPTH_PCT = Decimal("0.03")          # cap per-trade quote notional to % of bid depth
BAND = Decimal("0.0025")             # +/- price band (0.25%) around best bid for slices
BUY_SLICES = 4                        # number of slices per entry
TWAP_INTERVAL_SEC = 15                # spacing between slices
LOOKBACK_CANDLES = 12                 # volume lookback (5m candles) to shape weights

# Sell Configuration
SELL_TARGET_PCT = Decimal("0.9")  # target profit % over entry
SELL_TOLERANCE_PCT = Decimal("0.1")  # allow sells down to (target - tolerance) for faster fills
SELL_UNDERCUT_TICKS = 1  # undercut best ask by this many ticks
SELL_REPRICE_UNFILLED_AFTER_SEC = 60  # reprice open sells after this many seconds
SELL_POST_ONLY = True

# Data paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data_combauto_v131_run2"
LOGS_DIR = DATA_DIR / "logs"
OPEN_BUY_ORDERS_FILE = DATA_DIR / "open_buy_orders.json"
OPEN_SELL_ORDERS_FILE = DATA_DIR / "open_sell_orders.json"
ACTIVE_POSITIONS_FILE = DATA_DIR / "active_positions.json"
TRADE_HISTORY_FILE = DATA_DIR / "trade_history.json"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# === LOGGING SETUP ===
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(name)s] - %(message)s"
)

file_handler = logging.handlers.RotatingFileHandler(
    LOGS_DIR / f"combauto_final_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log",
    maxBytes=15 * 1024 * 1024,  # 15MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger("CombautoFinal")

# Reduce noise from other libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('coinbase').setLevel(logging.WARNING)
logging.getLogger('lightgbm').setLevel(logging.WARNING)

# === SIGNAL HANDLERS ===
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"🛑 Received signal {signum}. Initiating graceful shutdown...")
    shutdown_flag.set()

# Only set signal handlers if not on Windows with Proactor policy
try:
    if sys.platform != "win32":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        # Windows handles Ctrl+C differently with asyncio
        logger.debug("Windows detected - using KeyboardInterrupt handling instead of signals")
except Exception as e:
    logger.warning(f"Could not set signal handlers: {e}")


# === GLOBAL STATE ===
open_buy_orders: Dict[str, Dict] = {}
open_sell_orders: Dict[str, Dict] = {}
active_positions: Dict[str, Dict] = {}
trade_history: List[Dict] = []
state_lock = threading.RLock()  # FIXED: Use RLock instead of regular Lock

def log_trade_event(event_type: str, symbol: str, details: Dict[str, Any]):
    """Log a trade event with timestamp and details"""
    trade_event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "symbol": symbol,
        **details
    }
    
    logger.info(f"TRADE EVENT: {event_type} | {symbol} | {details}")
    
    with state_lock:
        trade_history.append(trade_event)
    
    # FIXED: Simplified state saving without async locks
    try:
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(trade_history, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to write trade history: {e}")

def load_state():
    """Load state from disk at startup"""
    global open_buy_orders, open_sell_orders, active_positions, trade_history
    
    logger.info(f"🔄 Loading state from {DATA_DIR}")
    try:
        # Load open buy orders
        if OPEN_BUY_ORDERS_FILE.exists() and os.path.getsize(OPEN_BUY_ORDERS_FILE) > 0:
            with open(OPEN_BUY_ORDERS_FILE, 'r', encoding='utf-8') as f:
                loaded_orders = json.load(f)
                for order_id, order_data in loaded_orders.items():
                    for key in ['limit_price', 'base_size', 'quote_size']:
                        if key in order_data:
                            order_data[key] = Decimal(str(order_data[key]))
                    if 'place_time_utc' in order_data:
                        order_data['place_time_utc'] = datetime.fromisoformat(order_data['place_time_utc'])
                    
                    # FIXED: Ensure product_id/symbol exists
                    if 'product_id' not in order_data and 'symbol' in order_data:
                        order_data['product_id'] = order_data['symbol']
                    elif 'symbol' not in order_data and 'product_id' in order_data:
                        order_data['symbol'] = order_data['product_id']
                        
                    open_buy_orders[order_id] = order_data
            logger.info(f"✅ Loaded {len(open_buy_orders)} open buy orders from {OPEN_BUY_ORDERS_FILE}")
        else:
            logger.info(f"📁 No open buy orders file found at {OPEN_BUY_ORDERS_FILE}")

        # Load open sell orders
        if OPEN_SELL_ORDERS_FILE.exists() and os.path.getsize(OPEN_SELL_ORDERS_FILE) > 0:
            with open(OPEN_SELL_ORDERS_FILE, 'r', encoding='utf-8') as f:
                loaded_orders = json.load(f)
                for order_id, order_data in loaded_orders.items():
                    for key in ['limit_price', 'base_size', 'quote_size']:
                        if key in order_data:
                            order_data[key] = Decimal(str(order_data[key]))
                    if 'place_time_utc' in order_data:
                        order_data['place_time_utc'] = datetime.fromisoformat(order_data['place_time_utc'])
                    
                    # FIXED: Ensure product_id/symbol exists
                    if 'product_id' not in order_data and 'symbol' in order_data:
                        order_data['product_id'] = order_data['symbol']
                    elif 'symbol' not in order_data and 'product_id' in order_data:
                        order_data['symbol'] = order_data['product_id']
                        
                    open_sell_orders[order_id] = order_data
            logger.info(f"✅ Loaded {len(open_sell_orders)} open sell orders from {OPEN_SELL_ORDERS_FILE}")
        else:
            logger.info(f"📁 No open sell orders file found at {OPEN_SELL_ORDERS_FILE}")
        
        # Load active positions
        if ACTIVE_POSITIONS_FILE.exists() and os.path.getsize(ACTIVE_POSITIONS_FILE) > 0:
            with open(ACTIVE_POSITIONS_FILE, 'r', encoding='utf-8') as f:
                loaded_positions = json.load(f)
                for symbol, position_data in loaded_positions.items():
                    # Convert string decimals back to Decimal objects
                    for key in ['entry_price', 'current_price', 'target_price', 'base_size']:
                        if key in position_data:
                            position_data[key] = Decimal(str(position_data[key]))
                    if 'buy_time_utc' in position_data:
                        position_data['buy_time_utc'] = datetime.fromisoformat(position_data['buy_time_utc'])
                    if 'last_sell_attempt_utc' in position_data:
                        position_data['last_sell_attempt_utc'] = datetime.fromisoformat(position_data['last_sell_attempt_utc'])
                    
                    # FIXED: Ensure symbol key exists in position data
                    if 'symbol' not in position_data:
                        position_data['symbol'] = symbol
                        
                    active_positions[symbol] = position_data
            logger.info(f"✅ Loaded {len(active_positions)} active positions from {ACTIVE_POSITIONS_FILE}")
        else:
            logger.info(f"📁 No active positions file found at {ACTIVE_POSITIONS_FILE}")
        
        # Load trade history
        if TRADE_HISTORY_FILE.exists() and os.path.getsize(TRADE_HISTORY_FILE) > 0:
            with open(TRADE_HISTORY_FILE, 'r', encoding='utf-8') as f:
                trade_history = json.load(f)
            logger.info(f"✅ Loaded {len(trade_history)} trade history records from {TRADE_HISTORY_FILE}")
        else:
            logger.info(f"📁 No trade history file found at {TRADE_HISTORY_FILE}")
            
    except Exception as e:
        logger.error(f"❌ Error loading state from {DATA_DIR}: {e}", exc_info=True)
        open_buy_orders.clear()
        open_sell_orders.clear()
        active_positions.clear()
        trade_history.clear()

def save_state():
    """FIXED: Simplified synchronous state saving"""
    with state_lock:
        try:
            logger.debug(f"Saving state to {DATA_DIR}: {len(active_positions)} positions, {len(open_buy_orders)} buy orders, {len(open_sell_orders)} sell orders")
            DATA_DIR.mkdir(exist_ok=True)
            # Save open buy orders
            with tempfile.NamedTemporaryFile('w', delete=False, dir=DATA_DIR, encoding='utf-8', suffix='.tmp') as tmp:
                json.dump(open_buy_orders, tmp, default=str, indent=2)
                tmp_path = tmp.name
            shutil.move(tmp_path, OPEN_BUY_ORDERS_FILE)
            # Save open sell orders
            with tempfile.NamedTemporaryFile('w', delete=False, dir=DATA_DIR, encoding='utf-8', suffix='.tmp') as tmp:
                json.dump(open_sell_orders, tmp, default=str, indent=2)
                tmp_path = tmp.name
            shutil.move(tmp_path, OPEN_SELL_ORDERS_FILE)
            # Save active positions
            with tempfile.NamedTemporaryFile('w', delete=False, dir=DATA_DIR, encoding='utf-8', suffix='.tmp') as tmp:
                json.dump(active_positions, tmp, default=str, indent=2)
                tmp_path = tmp.name
            shutil.move(tmp_path, ACTIVE_POSITIONS_FILE)
            logger.debug(f"✅ State saved successfully to {DATA_DIR}")
        except Exception as e:
            logger.error(f"❌ Failed to save state to {DATA_DIR}: {e}", exc_info=True)

class CoinbaseAPIHandler:
    """FIXED - Simplified Coinbase API handler based on working v9.5 patterns"""
    
    def __init__(self):
        self.client = RESTClient(
            api_key=os.getenv("COINBASE_API_KEY"),
            api_secret=os.getenv("COINBASE_API_SECRET"),
            rate_limit_headers=True,
            timeout=API_TIMEOUT_SECONDS,
            verbose=False  # Disable detailed SDK logging to prevent log spam
        )
        self.product_details = {}
    
    async def initialize_products(self):
        """Initialize product details for all symbols"""
        logger.info("🔧 Initializing product details...")
        
        for symbol in SYMBOLS:
            try:
                product = await asyncio.to_thread(
                    self.client.get_product,
                    product_id=symbol
                )
                
                # Use standard dot-notation from official SDK
                self.product_details[symbol] = {
                    'quote_increment': str(product.quote_increment),
                    'base_increment': str(product.base_increment),
                    'base_min_size': str(product.base_min_size),
                    'base_max_size': str(product.base_max_size)
                }
                
                logger.debug(f"✅ Initialized {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {symbol}: {e}")
                # Set reasonable defaults
                self.product_details[symbol] = {
                    'quote_increment': '0.01',
                    'base_increment': '0.00001',
                    'base_min_size': '0.001',
                    'base_max_size': '1000000'
                }
        
        logger.info(f"✅ Product details initialized for {len(self.product_details)} symbols")
    
    async def fetch_candles_batch(self, symbol: str, total_candles: int) -> List[Dict]:
        """Fetch candles with proper batching, avoid logging full raw data."""
        all_candles = []
        end_time = int(time.time())
        max_retries = 3

        while len(all_candles) < total_candles and not shutdown_flag.is_set():
            batch_size = min(CANDLE_BATCH_LIMIT, total_candles - len(all_candles))
            start_time = end_time - (batch_size * 300)  # 5-minute candles

            for retry in range(max_retries):
                try:
                    response = await asyncio.to_thread(
                        self.client.get_candles,
                        product_id=symbol,
                        start=str(start_time),
                        end=str(end_time),
                        granularity="FIVE_MINUTE",
                        limit=batch_size
                    )

                    # Use standard dot-notation
                    candles = []
                    if hasattr(response, 'candles') and response.candles:
                        for candle in response.candles:
                            candles.append({
                                'timestamp': int(candle.start),
                                'open': float(candle.open),
                                'high': float(candle.high),
                                'low': float(candle.low),
                                'close': float(candle.close),
                                'volume': float(candle.volume)
                            })

                    if candles:
                        all_candles.extend(candles)
                        break  # Success, exit retry loop

                except Exception as e:
                    logger.warning(f"Error fetching candles for {symbol} (retry {retry+1}/{max_retries}): {e}")
                    if retry < max_retries - 1:
                        await asyncio.sleep(2 ** retry)  # Exponential backoff
                    else:
                        logger.error(f"Failed to fetch candles for {symbol} after {max_retries} retries")
                        return []

            if not candles:
                break

            end_time = start_time
            await asyncio.sleep(0.2)  # Rate limiting

        # Sort by timestamp and remove duplicates
        all_candles = sorted(list({c['timestamp']: c for c in all_candles}.values()),
                           key=lambda x: x['timestamp'])

        # Only log summary, not full data
        if all_candles:
            logger.debug(f"Fetched {len(all_candles)} candles for {symbol} (first: {all_candles[0]['timestamp']}, last: {all_candles[-1]['timestamp']})")
        else:
            logger.debug(f"Fetched 0 candles for {symbol}")
        return all_candles[-total_candles:] if all_candles else []
    
    def _ticks(self, value: Decimal, inc: Decimal, *, up: bool) -> Decimal:
        rounding = ROUND_HALF_UP if up else ROUND_DOWN
        ticks = (value / inc).to_integral_value(rounding=rounding)
        return (ticks * inc).quantize(inc)

    async def fetch_book_data(self, symbol: str) -> Optional[Dict]:
        """Fetch comprehensive order book data for depth analysis."""
        max_retries = 3
        for retry in range(max_retries):
            try:
                book = await asyncio.to_thread(self.client.get_product_book, product_id=symbol, limit=20)
                
                # Handle SDK response structure
                pricebook = book.pricebook if hasattr(book, 'pricebook') else None
                if not pricebook:
                    # Fallback for dict-like response if SDK changes
                    pricebook = book.get("pricebook") if isinstance(book, dict) else getattr(book, "pricebook", None)
                
                if not pricebook:
                    raise ValueError("pricebook missing from book response")

                bids_raw = pricebook.bids if hasattr(pricebook, 'bids') else []
                asks_raw = pricebook.asks if hasattr(pricebook, 'asks') else []
                
                # Convert to list of dicts if they are objects
                bids = [vars(b) if hasattr(b, '__dict__') else b for b in bids_raw]
                asks = [vars(a) if hasattr(a, '__dict__') else a for a in asks_raw]

                if not bids or not asks:
                    return None

                # Ensure we have price and size as strings or numbers, convert to Decimal
                def get_dec(item, key):
                    val = item.get(key) if isinstance(item, dict) else getattr(item, key, 0)
                    return Decimal(str(val))

                best_bid = get_dec(bids[0], 'price')
                best_ask = get_dec(asks[0], 'price')
                
                if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
                    return None

                best_bid_qty = get_dec(bids[0], 'size')
                best_ask_qty = get_dec(asks[0], 'size')
                
                mid = (best_bid + best_ask) / Decimal("2")
                spread = ((best_ask - best_bid) / mid) * Decimal("100")
                
                total_bid_qty_top5 = sum(get_dec(b, 'size') for b in bids[:5])
                total_ask_qty_top5 = sum(get_dec(a, 'size') for a in asks[:5])
                
                total_bid_depth = sum(get_dec(b, 'price') * get_dec(b, 'size') for b in bids)
                total_ask_depth = sum(get_dec(a, 'price') * get_dec(a, 'size') for a in asks)

                return {
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "best_bid_qty": best_bid_qty,
                    "best_ask_qty": best_ask_qty,
                    "spread": spread,
                    "mid": mid,
                    "total_bid_qty_top5": total_bid_qty_top5,
                    "total_ask_qty_top5": total_ask_qty_top5,
                    "best_bid_amount": best_bid * best_bid_qty,
                    "best_ask_amount": best_ask * best_ask_qty,
                    "total_bid_depth": total_bid_depth,
                    "total_ask_depth": total_ask_depth,
                    "bids": bids,
                    "asks": asks,
                }
            except Exception as e:
                logger.warning(f"Book {symbol} retry {retry+1}: {e}")
                if retry < max_retries - 1:
                    await asyncio.sleep(1)
        return None

    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """FIXED: Get current market price using correct API response structure"""
        max_retries = 3
        for retry in range(max_retries):
            try:
                # CORRECTED: Use proper API call
                orderbook = await asyncio.to_thread(
                    self.client.get_product_book,
                    product_id=symbol,
                    limit=1
                )
                
                # FIXED: Correct response structure based on SDK
                # GetProductBookResponse has 'pricebook' field with 'bids' and 'asks' arrays
                if hasattr(orderbook, 'pricebook') and orderbook.pricebook:
                    pricebook = orderbook.pricebook
                    
                    # Each bid/ask is an object with 'price' and 'size' attributes
                    if (hasattr(pricebook, 'bids') and hasattr(pricebook, 'asks') and 
                        pricebook.bids and pricebook.asks):
                        
                        # Extract price from first bid/ask objects
                        bid_price = Decimal(str(pricebook.bids[0].price))
                        ask_price = Decimal(str(pricebook.asks[0].price))
                        return (bid_price + ask_price) / 2
                
                return None
                
            except Exception as e:
                logger.warning(f"Error getting price for {symbol} (retry {retry+1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    await asyncio.sleep(1)
                    
        logger.error(f"Failed to get price for {symbol} after {max_retries} retries")
        return None

    def parse_order_response(self, response: Any) -> Tuple[Optional[str], bool, str]:
        """FIXED: Correct order response parsing based on official Coinbase SDK structure"""
        try:
            if response is None:
                return None, False, "Empty response from API"

            logger.debug(f"Parsing order response of type: {type(response).__name__}")
            
            # CORRECTED: Based on actual SDK source code at:
            # https://github.com/coinbase/coinbase-advanced-py/blob/main/coinbase/rest/types/orders_types.py
            # CreateOrderResponse structure:
            # - success: bool
            # - failure_reason: Optional[Dict[str, Any]] 
            # - order_id: Optional[str] (RARELY POPULATED - usually None)
            # - success_response: Optional[CreateOrderSuccess] (WHERE ORDER_ID IS ACTUALLY STORED)
            # - error_response: Optional[CreateOrderError]
            
            # Check if this is a CreateOrderResponse object
            if hasattr(response, 'success'):
                success = bool(response.success) if response.success is not None else False
                logger.debug(f"Response success field: {success}")
                
                if success:
                    # SUCCESS CASE - Extract order_id from the CORRECT location
                    order_id = None
                    
                    # Method 1: Check success_response object (PRIMARY LOCATION)
                    if hasattr(response, 'success_response') and response.success_response:
                        success_resp = response.success_response
                        logger.debug(f"Found success_response: {type(success_resp).__name__}")
                        
                        # CreateOrderSuccess object has order_id field
                        if hasattr(success_resp, 'order_id') and success_resp.order_id:
                            order_id = str(success_resp.order_id)
                            logger.debug(f"✅ Found order_id in success_response.order_id: {order_id}")
                        
                        # Fallback: dict format
                        elif isinstance(success_resp, dict) and 'order_id' in success_resp:
                            order_id = str(success_resp['order_id'])
                            logger.debug(f"✅ Found order_id in success_response dict: {order_id}")
                    
                    # Method 2: Direct order_id field (RARELY USED, but check anyway)
                    elif hasattr(response, 'order_id') and response.order_id:
                        order_id = str(response.order_id)
                        logger.debug(f"✅ Found order_id directly on response: {order_id}")
                    
                    if order_id:
                        return str(order_id), True, "Order placed successfully"
                    else:
                        logger.warning("❌ Order marked successful but no order_id found")
                        # Debug available fields
                        if hasattr(response, 'success_response') and response.success_response:
                            sr_attrs = [attr for attr in dir(response.success_response) if not attr.startswith('_')]
                            logger.debug(f"success_response attributes: {sr_attrs}")
                        resp_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
                        logger.debug(f"response attributes: {resp_attrs}")
                        return None, False, "Order successful but no order_id returned"
                        
                else:
                    # FAILURE CASE - Extract error details
                    error_details = []
                    
                    # Method 1: Check failure_reason field
                    if hasattr(response, 'failure_reason') and response.failure_reason:
                        failure_reason = response.failure_reason
                        logger.debug(f"Found failure_reason: {type(failure_reason).__name__}")
                        
                        if isinstance(failure_reason, dict):
                            # Extract common error fields
                            for key in ['message', 'error', 'error_details', 'reason']:
                                if key in failure_reason and failure_reason[key]:
                                    error_details.append(f"{key}: {failure_reason[key]}")
                            
                            if not error_details:
                                error_details.append(f"failure_reason: {failure_reason}")
                        else:
                            error_details.append(f"failure_reason: {failure_reason}")
                    
                    # Method 2: Check error_response object
                    if hasattr(response, 'error_response') and response.error_response:
                        error_resp = response.error_response
                        logger.debug(f"Found error_response: {type(error_resp).__name__}")
                        
                        # CreateOrderError object fields
                        if hasattr(error_resp, 'message') and error_resp.message:
                            error_details.append(f"message: {error_resp.message}")
                        if hasattr(error_resp, 'error') and error_resp.error:
                            error_details.append(f"error: {error_resp.error}")
                        if hasattr(error_resp, 'error_details') and error_resp.error_details:
                            error_details.append(f"error_details: {error_resp.error_details}")
                        if hasattr(error_resp, 'new_order_failure_reason') and error_resp.new_order_failure_reason:
                            error_details.append(f"new_order_failure_reason: {error_resp.new_order_failure_reason}")
                        
                        # Fallback: dict format
                        if isinstance(error_resp, dict):
                            for key in ['error', 'message', 'error_details']:
                                if key in error_resp and error_resp[key]:
                                    error_details.append(f"{key}: {error_resp[key]}")
                    
                    error_msg = "; ".join(error_details) if error_details else "Order failed (no error details available)"
                    logger.debug(f"Order failed with message: {error_msg}")
                    return None, False, error_msg
            
            # Fallback for unexpected response formats
            else:
                logger.warning(f"Unexpected response format: {type(response).__name__}")
                logger.warning(f"Available attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
                
                # Try to extract some useful information anyway
                response_str = str(response)[:500]
                logger.warning(f"Response content preview: {response_str}")
                
                return None, False, f"Unexpected response format: {type(response).__name__}"
            
        except Exception as e:
            logger.error(f"Critical error parsing order response: {e}", exc_info=True)
            logger.error(f"Response type: {type(response).__name__}")
            
            # Safe response content logging
            try:
                response_content = str(response)[:300] if response else "None"
                logger.error(f"Response content: {response_content}")
            except:
                logger.error("Could not convert response to string for logging")
                
            return None, False, f"Response parsing exception: {e}"

    async def place_gtc_buy_order(self, symbol: str, quote_size: Decimal, limit_price: Decimal) -> Optional[str]:
        """FIXED: Place GTC buy order using the direct SDK method for reliability"""
        try:
            client_order_id = str(uuid.uuid4())
            base_size = quote_size / limit_price
            # For buy orders, we can round down to get the best price (round_price_up=False is default)
            formatted_price, formatted_size = self.format_order_values(symbol, limit_price, base_size)

            logger.info(f"Placing GTC buy: {symbol} | Size: {formatted_size} | Price: ${formatted_price}")

            response = await asyncio.to_thread(
                self.client.limit_order_gtc_buy,
                client_order_id=client_order_id,
                product_id=symbol,
                base_size=formatted_size,
                limit_price=formatted_price,
                post_only=True
            )
            
            order_id, success, message = self.parse_order_response(response)

            if success and order_id:
                logger.info(f"✅ GTC buy order placed successfully for {symbol}: {order_id}")
                return order_id
            else:
                logger.error(f"❌ GTC buy order failed for {symbol}: {message}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Exception during GTC buy order for {symbol}: {e}", exc_info=True)
            return None
    
    async def place_gtc_sell_order(self, symbol: str, base_size: Decimal, _unused_limit_price: Decimal = None) -> Optional[str]:
        """
        Place a GTC sell order for exactly 1.1% above the current market ask price, using correct decimal precision for each pair.
        """
        try:
            client_order_id = str(uuid.uuid4())

            # 1. Fetch current order book to get the best ask price
            orderbook = await asyncio.to_thread(
                self.client.get_product_book,
                product_id=symbol,
                limit=1
            )
            if not (hasattr(orderbook, 'pricebook') and orderbook.pricebook and hasattr(orderbook.pricebook, 'asks') and orderbook.pricebook.asks):
                logger.error(f"Could not fetch ask price for {symbol} - orderbook missing asks")
                return None
            ask_price = Decimal(str(orderbook.pricebook.asks[0].price))

            # 2. Calculate target price (1.1% above ask)
            target_price = ask_price * Decimal('1.011')

            # 3. Get quote_increment for this symbol
            quote_increment = Decimal(self.product_details[symbol]['quote_increment'])

            # 4. Round to nearest allowed increment (not up or down, just nearest)
            increments = (target_price / quote_increment).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            formatted_price = (increments * quote_increment).quantize(quote_increment)

            # 5. Format base_size as usual
            base_increment = Decimal(self.product_details[symbol]['base_increment'])
            formatted_size = base_size.quantize(base_increment, rounding=ROUND_DOWN)
            base_min_size = Decimal(self.product_details[symbol]['base_min_size'])
            if formatted_size < base_min_size:
                logger.warning(f"Formatted size {formatted_size} for {symbol} is below min size {base_min_size}. Adjusting to min size.")
                formatted_size = base_min_size

            logger.info(f"Placing GTC sell: {symbol} | Size: {formatted_size} | Price: ${formatted_price} (Ask: {ask_price}, Target: {target_price}, Increment: {quote_increment})")

            response = await asyncio.to_thread(
                self.client.limit_order_gtc_sell,
                client_order_id=client_order_id,
                product_id=symbol,
                base_size=str(formatted_size),
                limit_price=str(formatted_price),
                post_only=False
            )

            order_id, success, message = self.parse_order_response(response)

            if success and order_id:
                logger.info(f"✅ GTC sell order placed successfully for {symbol}: {order_id}")
                return order_id
            else:
                logger.error(f"❌ GTC sell order failed for {symbol}: {message}")
                return None

        except Exception as e:
            logger.error(f"❌ Exception during GTC sell order for {symbol}: {e}", exc_info=True)
            return None

    async def place_market_sell_order(self, symbol: str, base_size: Decimal) -> Optional[str]:
        """FIXED: Place a proper market sell order using the direct SDK method"""
        try:
            client_order_id = str(uuid.uuid4())
            _, formatted_size = self.format_order_values(symbol, Decimal('1'), base_size)

            logger.info(f"Placing MARKET sell: {symbol} | Size: {formatted_size}")

            response = await asyncio.to_thread(
                self.client.market_order_sell,
                client_order_id=client_order_id,
                product_id=symbol,
                base_size=formatted_size
            )
            
            order_id, success, message = self.parse_order_response(response)

            if success and order_id:
                logger.info(f"✅ Market sell order placed successfully for {symbol}: {order_id}")
                return order_id
            else:
                logger.error(f"❌ Market sell order failed for {symbol}: {message}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Exception during market sell order for {symbol}: {e}", exc_info=True)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order with retry logic"""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                await asyncio.to_thread(self.client.cancel_orders, order_ids=[order_id])
                logger.info(f"✅ Order cancelled: {order_id}")
                return True
            except Exception as e:
                logger.warning(f"Attempt {attempt}: Could not cancel order {order_id}: {e}")
                await asyncio.sleep(1)
        logger.error(f"Failed to cancel order {order_id} after {max_retries} retries")
        return False

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """FIXED: Get order status with correct API usage and response handling"""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # FIXED: Use self.client and asyncio.to_thread
                response = await asyncio.to_thread(self.client.get_order, order_id=order_id)

                # FIXED: The actual order data is inside the 'order' attribute of the response object
                if hasattr(response, 'order') and response.order:
                    # Convert the Order object to a dictionary for consistent handling
                    order_data = response.order.to_dict() 
                    
                    # Further validation to ensure the nested object is what we expect
                    if 'status' in order_data and 'product_id' in order_data:
                        logger.debug(f"Successfully parsed nested order status for {order_id}")
                        return order_data
                
                # This log helps identify if the structure changes again
                logger.warning(f"Unexpected response for order {order_id}: {response}")
                return None

            except Exception as e:
                logger.error(f"Error getting status for order {order_id} (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1)  # Simple backoff

        logger.error(f"Failed to get order status for {order_id} after {max_retries} retries")
        return None

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Fetch all open orders in a single batch request to save API calls."""
        try:
            response = await asyncio.to_thread(self.client.list_orders, order_status=["OPEN"])
            
            # Handle SDK response structure
            orders = []
            if hasattr(response, 'orders') and response.orders:
                for order in response.orders:
                    if hasattr(order, 'to_dict'):
                        orders.append(order.to_dict())
                    else:
                        # Fallback if it's already a dict or other object
                        orders.append(vars(order) if hasattr(order, '__dict__') else order)
            
            return orders
        except Exception as e:
            logger.error(f"Error fetching open orders batch: {e}")
            return []

    def format_order_values(self, symbol: str, price: Decimal, size: Decimal, round_price_up: bool = False) -> Tuple[str, str]:
        """Format price and size according to product specifications
        
        Args:
            symbol: Trading pair symbol
            price: Price to format
            size: Size to format  
            round_price_up: If True, round price UP to ensure we don't sell below target
        """
        try:
            product = self.product_details.get(symbol, {})
            
            quote_increment = Decimal(product.get('quote_increment', '0.01'))
            base_increment = Decimal(product.get('base_increment', '0.00001'))

            # For sell orders, we want to round UP to ensure we get AT LEAST the target profit
            # For buy orders, we can round down to get the best price
            if round_price_up:
                # Calculate how many increments we need, then round UP
                increments_needed = (price / quote_increment)
                formatted_price = (increments_needed.quantize(Decimal('1'), rounding='ROUND_UP')) * quote_increment
            else:
                # Standard rounding down for buy orders
                formatted_price = price.quantize(quote_increment, rounding=ROUND_DOWN)
            
            # Always round size down to avoid insufficient balance errors
            formatted_size = size.quantize(base_increment, rounding=ROUND_DOWN)

            # Ensure minimum size is met
            base_min_size = Decimal(product.get('base_min_size', '0.001'))
            if formatted_size < base_min_size:
                logger.warning(f"Formatted size {formatted_size} for {symbol} is below min size {base_min_size}. Adjusting to min size.")
                formatted_size = base_min_size

            return str(formatted_price), str(formatted_size)
            
        except (InvalidOperation, TypeError, KeyError) as e:
            logger.error(f"Error formatting order values for {symbol}: {e}", exc_info=True)
            # Fallback to a reasonable default format if product details are faulty
            return f"{price:.8f}", f"{size:.8f}"

    async def test_order_parsing(self, symbol: str = "BTC-USD") -> None:
        """Test method to debug order response parsing with minimal order"""
        try:
            logger.info(f"🧪 Testing order response parsing for {symbol}")
            
            # Get current price
            test_price = await self.get_current_price(symbol)
            if not test_price:
                logger.error("Cannot get current price for test")
                return
            
            # Get product details for minimum size
            product = self.product_details.get(symbol, {})
            min_size = Decimal(product.get('base_min_size', '0.001'))
            
            # Try a minimal buy order well below market price (shouldn't fill)
            test_size = min_size * Decimal("1.1")  # Slightly above minimum
            test_buy_price = test_price * Decimal("0.90")  # 10% below market (shouldn't fill)
            client_order_id = f"test_buy_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Testing buy order: {test_size} {symbol} @ ${test_buy_price} (market: ${test_price})")
            
            # Test buy order
            buy_response = await asyncio.to_thread(
                self.client.limit_order_gtc_buy,
                client_order_id=client_order_id,
                product_id=symbol,
                base_size=str(test_size),
                limit_price=str(test_buy_price),
                post_only=True
            )
            
            logger.info(f"Buy test response type: {type(buy_response).__name__}")
            
            # Log all available attributes
            if hasattr(buy_response, '__dict__'):
                logger.info(f"Buy response attributes: {list(buy_response.__dict__.keys())}")
                for key, value in buy_response.__dict__.items():
                    logger.info(f"  {key}: {value}")
            else:
                logger.info(f"Buy response dir: {[attr for attr in dir(buy_response) if not attr.startswith('_')]}")
            
            buy_order_id, buy_success, buy_message = self.parse_order_response(buy_response)
            logger.info(f"Buy order parsed result: success={buy_success}, order_id={buy_order_id}, message={buy_message}")
            
            # Test sell order (should also not fill if priced above market)
            test_sell_price = test_price * Decimal("1.10")  # 10% above market
            client_order_id_sell = f"test_sell_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Testing sell order: {test_size} {symbol} @ ${test_sell_price} (market: ${test_price})")
            
            sell_response = await asyncio.to_thread(
                self.client.limit_order_gtc_sell,
                client_order_id=client_order_id_sell,
                product_id=symbol,
                base_size=str(test_size),
                limit_price=str(test_sell_price),
                post_only=False  # Test both post_only values
            )
            
            logger.info(f"Sell test response type: {type(sell_response).__name__}")
            
            # Log all available attributes
            if hasattr(sell_response, '__dict__'):
                logger.info(f"Sell response attributes: {list(sell_response.__dict__.keys())}")
                for key, value in sell_response.__dict__.items():
                    logger.info(f"  {key}: {value}")
            else:
                logger.info(f"Sell response dir: {[attr for attr in dir(sell_response) if not attr.startswith('_')]}")
            
            sell_order_id, sell_success, sell_message = self.parse_order_response(sell_response)
            logger.info(f"Sell order parsed result: success={sell_success}, order_id={sell_order_id}, message={sell_message}")
            
            # Clean up any successful test orders
            cleanup_orders = []
            if buy_success and buy_order_id:
                cleanup_orders.append(buy_order_id)
            if sell_success and sell_order_id:
                cleanup_orders.append(sell_order_id)
            
            if cleanup_orders:
                logger.info(f"Cleaning up test orders: {cleanup_orders}")
                for order_id in cleanup_orders:
                    try:
                        await self.cancel_order(order_id)
                    except Exception as e:
                        logger.warning(f"Failed to cancel test order {order_id}: {e}")
            
            # Summary
            logger.info(f"🧪 Test Summary:")
            logger.info(f"  Buy order success: {buy_success} - {buy_message}")
            logger.info(f"  Sell order success: {sell_success} - {sell_message}")
            
        except Exception as e:
            logger.error(f"❌ Error during order parsing test: {e}", exc_info=True)

class StatisticalEngine:
    """
    Engine 1: Statistical Analysis
    - CSD Quintiles (primary signal)
    - RSI/OBV combinations (confirmation)
    - 4-price trend analysis (momentum)
    """
    
    def __init__(self, api_handler: CoinbaseAPIHandler):
        self.api = api_handler
    
    async def get_stars(self, symbol: str) -> int:
        """Get 1-5 star rating based on statistical signals"""
        if shutdown_flag.is_set():
            return 1
            
        try:
            # Fetch recent candles for analysis
            candles = await self.api.fetch_candles_batch(symbol, 500)
            if len(candles) < 100:
                logger.debug(f"Insufficient data for {symbol}: {len(candles)} candles")
                return 1
            
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp').sort_index()
            
            stars = 1  # Base rating
            reasons = []
            
            # === CSD QUINTILES (Primary Signal) ===
            df['csd'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
            current_csd = df['csd'].iloc[-1]
            
            # Handle edge case where all values are the same
            if df['csd'].nunique() >= 5:
                df['quintile'] = pd.qcut(df['csd'], 5, labels=False, duplicates='drop')
                current_quintile = df['quintile'].iloc[-1]
                
                if current_quintile == 0:
                    stars += 2
                    reasons.append("CSD Quintile 1")

            # === RSI/OBV (Confirmation) ===
            df.ta.rsi(length=14, append=True, col_names=('rsi_14',))
            df.ta.obv(append=True, col_names=('obv',))
            df['obv_ema'] = ta.ema(df['obv'], length=20)
            
            current_rsi = df['rsi_14'].iloc[-1]
            current_obv = df['obv'].iloc[-1]
            current_obv_ema = df['obv_ema'].iloc[-1]
            
            if current_rsi < 35 and current_obv > current_obv_ema:
                stars += 1
                reasons.append("RSI < 35 & OBV > EMA")

            # === TREND (Momentum) ===
            prices = df['close'].tail(4).tolist()
            if len(prices) == 4 and prices[-1] > prices[-2] > prices[-3] > prices[-4]:
                stars += 1
                reasons.append("Uptrend")
            
            logger.debug(f"{symbol} Stats: Stars={stars}, Reasons={reasons}")
            return min(stars, 5)
            
        except Exception as e:
            logger.error(f"Error in StatisticalEngine for {symbol}: {e}", exc_info=True)
            return 1

class MLEngine:
    """
    Engine 2: Machine Learning
    - Optuna-optimized LightGBM
    - Trained to predict 1% gains within 2 hours
    - Uses comprehensive feature set
    """
    
    def __init__(self, api_handler: CoinbaseAPIHandler):
        self.api = api_handler
        self.models = {}
        self.feature_cols = [
            'rsi_14', 'obv_norm', 'volume_norm', 'price_momentum_1h',
            'price_momentum_30m', 'volatility', 'macd_line', 'bb_position'
        ]
    
    async def train_models(self):
        """Train ML models for all symbols - Optimized with parallel data fetching"""
        if shutdown_flag.is_set():
            return
            
        logger.info("🤖 Training ML models...")
        
        # Optimization: Fetch data in parallel with semaphore
        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async def fetch_data(sym):
            async with sem:
                logger.info(f"Fetching training data for {sym}...")
                return await self.api.fetch_candles_batch(
                    sym, ML_HISTORICAL_DAYS * 24 * 12
                )
        
        # Launch all fetch tasks
        fetch_tasks = [fetch_data(sym) for sym in SYMBOLS]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Process results and train models
        for symbol, result in zip(SYMBOLS, results):
            if shutdown_flag.is_set():
                break
                
            try:
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch data for {symbol}: {result}")
                    continue
                    
                candles = result
                logger.info(f"Training model for {symbol} ({len(candles)} candles)...")
                
                if len(candles) < 500:
                    logger.warning(f"Insufficient training data for {symbol}: {len(candles)}")
                    continue
                
                df = pd.DataFrame(candles)
                
                # Prepare features and targets
                X, y = self.prepare_training_data(df)
                
                if len(X) < 100 or y.sum() < 10:
                    logger.warning(f"Insufficient positive samples for {symbol} (Positive: {y.sum() if hasattr(y, 'sum') else 0}). Check PROFIT_TARGET_PCT.")
                    continue
                
                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Optuna optimization
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                        'random_state': 42,
                        'verbose': -1
                    }
                    
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_train, y_train)
                    
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                        return auc
                    except:
                        return 0.5
                
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=ML_OPTUNA_TRIALS)
                
                # Train final model
                best_params = study.best_params
                best_params.update({'random_state': 42, 'verbose': -1})
                
                final_model = lgb.LGBMClassifier(**best_params)
                final_model.fit(X, y)  # Use all data for final model
                
                # Create imputer for missing values
                imputer = SimpleImputer(strategy='median')
                imputer.fit(X)
                
                self.models[symbol] = {
                    'model': final_model,
                    'imputer': imputer,
                    'score': study.best_value,
                    'trials': len(study.trials)
                }
                
                logger.info(f"✅ Model trained for {symbol} (AUC: {study.best_value:.3f})")
                
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {e}")
        
        logger.info(f"✅ ML training complete: {len(self.models)}/{len(SYMBOLS)} models")
    
    async def get_stars(self, symbol: str) -> int:
        """Get 1-5 star rating based on ML prediction"""
        if shutdown_flag.is_set():
            return 2
            
        if symbol not in self.models:
            return 2  # Neutral rating if no model
        
        try:
            features = await self.calculate_features(symbol)
            if not features:
                return 2
            
            model_info = self.models[symbol]
            
            # Prepare feature array
            feature_array = []
            for col in self.feature_cols:
                feature_array.append(features.get(col, 0))
            
            feature_array = np.array(feature_array).reshape(1, -1)
            
            # Handle missing values
            feature_array = model_info['imputer'].transform(feature_array)
            
            # Get prediction probability
            prob = model_info['model'].predict_proba(feature_array)[0][1]
            
            # Convert to stars (more aggressive thresholds)
            if prob >= 0.95:
                stars = 5
            elif prob >= 0.85:
                stars = 4
            elif prob >= 0.75:
                stars = 3
            elif prob >= 0.65:
                stars = 2
            else:
                stars = 1
            
            logger.debug(f"ML: {symbol} = {stars}⭐ | Prob: {prob:.3f} | Model AUC: {model_info['score']:.3f}")
            
            return stars
            
        except Exception as e:
            logger.error(f"ML prediction error for {symbol}: {e}")
            return 2
    
    async def calculate_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Calculate ML features for current market state"""
        try:
            candles = await self.api.fetch_candles_batch(symbol, 100)
            if len(candles) < 50:
                return None
            
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp').sort_index()

            # Use pandas_ta for consistency and robustness
            df.ta.rsi(length=14, append=True, col_names=('rsi_14',))
            df.ta.obv(append=True, col_names=('obv',))
            df.ta.macd(append=True, col_names=('macd_line', 'macd_hist', 'macd_signal'))
            df.ta.bbands(length=20, append=True) # Default names are fine, BBP_20_2.0 for position
            
            # Normalize OBV and Volume
            df['obv_norm'] = (df['obv'] - df['obv'].mean()) / (df['obv'].std() + 1e-8)
            df['volume_norm'] = (df['volume'] - df['volume'].mean()) / (df['volume'].std() + 1e-8)
            
            # Price momentum
            df['price_momentum_1h'] = df['close'].pct_change(12)  # 12 * 5min = 1h
            df['price_momentum_30m'] = df['close'].pct_change(6)   # 6 * 5min = 30m
            
            # Volatility
            df['volatility'] = df['close'].pct_change().rolling(20).std()

            # Bollinger Bands position
            df['bb_position'] = df['BBP_20_2.0']
            
            # Get latest values
            features = {}
            for col in self.feature_cols:
                val = df[col].iloc[-1]
                features[col] = float(val) if pd.notna(val) else 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features for {symbol}: {e}")
            return None
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with features and target"""
        try:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp').sort_index()
            
            # Use pandas_ta for consistency
            df.ta.rsi(length=14, append=True, col_names=('rsi_14',))
            df.ta.obv(append=True, col_names=('obv',))
            df.ta.macd(append=True, col_names=('macd_line', 'macd_hist', 'macd_signal'))
            df.ta.bbands(length=20, append=True) # Generates BBP_20_2.0

            # Normalize OBV and Volume
            df['obv_norm'] = (df['obv'] - df['obv'].mean()) / (df['obv'].std() + 1e-8)
            df['volume_norm'] = (df['volume'] - df['volume'].mean()) / (df['volume'].std() + 1e-8)
            
            # Price momentum
            df['price_momentum_1h'] = df['close'].pct_change(12) # 12 * 5min = 1h
            df['price_momentum_30m'] = df['close'].pct_change(6)  # 6 * 5min = 30m
            
            # Volatility
            df['volatility'] = df['close'].pct_change().rolling(20).std()

            # Bollinger Bands position
            df['bb_position'] = df['BBP_20_2.0']

            # Define target: Did price increase by PROFIT_TARGET_PCT within ML_TARGET_HOURS?
            future_periods = int(ML_TARGET_HOURS * 60 / 5) # 5-min candles
            # FIXED: Use global PROFIT_TARGET_PCT and handle Decimal/float conversion correctly
            profit_multiplier = float(1 + (PROFIT_TARGET_PCT / Decimal(100)))
            target_price = df['close'] * profit_multiplier
            
            # Check if high in the future window reached the target price
            df['target'] = (df['high'].rolling(window=future_periods, min_periods=1).max().shift(-future_periods) >= target_price)
            df['target'] = df['target'].astype(int)

            # Clean up data
            df = df.dropna()
            df = df.replace([np.inf, -np.inf], 0)

            X = df[self.feature_cols]
            y = df['target']
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}", exc_info=True)
            return pd.DataFrame(), pd.Series()

class CombautoFinal:
    # ------------------- Sell gating helpers -------------------
    async def _maybe_place_sell_for_position(self, position: Dict[str, Any]):
        """Place a sell only after price crosses target threshold."""
        symbol = position["symbol"]
        
        # Get current price
        mark = await self.api.get_current_price(symbol)
        if mark is None:
            return
            
        entry_price = Decimal(str(position["entry_price"]))
        target_price = entry_price * (Decimal("1") + SELL_TARGET_PCT / Decimal("100"))
        
        if position.get("sell_order_id"):
            return
            
        # Only place sell if we are at or above target
        if mark < target_price:
            return
            
        await self._place_sell(position, observed_price=mark)

    async def _place_sell(self, position: Dict[str, Any], observed_price: Optional[Decimal] = None):
        symbol = position["symbol"]
        entry_price = Decimal(str(position["entry_price"]))
        base_size = Decimal(str(position["base_size"]))
        
        # Compute target. SELL_TARGET_PCT is treated as GROSS profit (before fees).
        target_price = entry_price * (Decimal("1") + SELL_TARGET_PCT / Decimal("100"))
        # Keep state compatibility, but do not allow pricing below the gross target.
        min_price = target_price

        # Use top-of-book to undercut ask, but never below tolerance floor
        await self.api.initialize_products()
        details = self.api.product_details.get(symbol, {})
        tick = Decimal(details.get("quote_increment", "0.0001"))
        
        book = await self.api.fetch_book_data(symbol)
        ask = book.get("best_ask") if book else None
        bid = book.get("best_bid") if book else None
        
        book_price: Optional[Decimal] = None
        if ask:
            desired = ask - tick * Decimal(SELL_UNDERCUT_TICKS)
            if desired <= 0:
                desired = ask
            # Never allow a sell to become marketable; for maker-only, keep strictly above bid.
            if bid:
                maker_floor = bid + tick
                if desired < maker_floor:
                    desired = maker_floor
            # Never price above the ask (worst case: join the ask).
            if desired > ask:
                desired = ask
            book_price = desired

        # Prefer just-below-ask maker posting, but ONLY if it still meets the gross target.
        chosen = book_price if book_price is not None else target_price
        if chosen < target_price:
            logger.info(
                f"Sell deferred {symbol}: best maker price {chosen} < target {target_price} "
            )
            return

        # Place the sell order
        # Note: place_gtc_sell_order in v131 takes limit_price. 
        # We need to ensure we pass the correct price.
        # Also v131's place_gtc_sell_order calculates 1.1% internally. We should override that or use place_limit_sell directly if available.
        # v131's place_gtc_sell_order is specific. Let's use place_gtc_sell_order but pass our chosen price.
        # Wait, v131's place_gtc_sell_order ignores the limit_price argument in some versions or calculates it.
        # Let's check v131's place_gtc_sell_order implementation again.
        # It takes _unused_limit_price: Decimal = None. It calculates target internally.
        # We need to modify place_gtc_sell_order or use a new method.
        # Since I can't easily modify API handler again without risk, I'll use the client directly here or assume I can fix place_gtc_sell_order.
        # Actually, I can just call the client method directly like vXO does.
        
        try:
            client_order_id = str(uuid.uuid4())
            formatted_price, formatted_size = self.api.format_order_values(symbol, chosen, base_size, round_price_up=True)
            
            logger.info(f"Placing monitored sell: {symbol} | Size: {formatted_size} | Price: ${formatted_price}")

            response = await asyncio.to_thread(
                self.api.client.limit_order_gtc_sell,
                client_order_id=client_order_id,
                product_id=symbol,
                base_size=formatted_size,
                limit_price=formatted_price,
                post_only=SELL_POST_ONLY
            )
            
            oid, success, message = self.api.parse_order_response(response)
            
            if success and oid:
                now_iso = datetime.now(timezone.utc).isoformat()
                with state_lock:
                    open_sell_orders[oid] = {
                        "product_id": symbol, # v131 uses product_id
                        "symbol": symbol,     # vXO uses symbol, let's keep both for safety
                        "entry_price": str(entry_price),
                        "target_price": str(target_price),
                        "min_price": str(min_price),
                        "chosen_price": str(chosen),
                        "base_size": str(base_size),
                        "limit_price": str(chosen), # v131 uses limit_price
                        "placed_at": now_iso,
                        "place_time_utc": datetime.now(timezone.utc), # v131 uses place_time_utc
                        "last_reprice_at": now_iso,
                        "status": "placed",
                    }
                    position["sell_order_id"] = oid
                    position["status"] = "SELL_PLACED"
                
                save_state()
                logger.info(f"Sell placed {symbol} @ {chosen} (target {target_price}) order {oid}")
            else:
                logger.warning(f"Sell placement failed {symbol}: {message}")
                if "INSUFFICIENT" in str(message).upper():
                     # Handle externally sold
                     logger.warning(f"Sell placement failed {symbol}: INSUFFICIENT_BALANCE - removing position")
                     with state_lock:
                         active_positions.pop(symbol, None)
                     save_state()

        except Exception as e:
            logger.error(f"Error placing sell for {symbol}: {e}")

    async def _reconcile_positions(self):
        """Ensure tracked sells are coherent; place sells only after threshold is reached."""
        with state_lock:
            positions_copy = list(active_positions.values())
            
        for pos in positions_copy:
            # Safety check for symbol key
            if 'symbol' not in pos:
                # Try to infer symbol from active_positions keys if possible, but here we only have the value.
                # However, since we fixed load_state, this should be rare.
                # We can try to find the key in active_positions that matches this value
                found_symbol = None
                with state_lock:
                    for sym, p in active_positions.items():
                        if p is pos:
                            found_symbol = sym
                            break
                
                if found_symbol:
                    pos['symbol'] = found_symbol
                else:
                    logger.error(f"Position data missing symbol key and cannot be recovered: {pos}")
                    continue

            soid = pos.get("sell_order_id")
            if not soid:
                await self._maybe_place_sell_for_position(pos)
                continue
            
            if soid in open_sell_orders:
                continue
                
            # If we have a sell_order_id but it's not in open_sell_orders, check status
            status = await self.api.get_order_status(soid)
            st = (status or {}).get("status", "").upper()
            
            if st == "FILLED":
                # Handle filled sell (close position)
                symbol = pos["symbol"]
                logger.info(f"Sell filled (reconcile) {symbol} oid={soid}")
                with state_lock:
                    active_positions.pop(symbol, None)
                    log_trade_event("SELL_FILLED", symbol, {
                        "order_id": soid,
                        "price": str((status or {}).get("average_filled_price") or pos.get("target_price")),
                        "pnl": "Calculated in history"
                    })
                save_state()
            elif st in {"OPEN", "PENDING"}:
                # Re-add to open_sell_orders if missing
                entry_price = Decimal(str(pos["entry_price"]))
                base_size = Decimal(str(pos["base_size"]))
                target_price = entry_price * (Decimal("1") + SELL_TARGET_PCT / Decimal("100"))
                chosen_price = Decimal(str((status or {}).get("price") or target_price))
                now_iso = datetime.now(timezone.utc).isoformat()
                
                with state_lock:
                    open_sell_orders[soid] = {
                        "product_id": pos["symbol"],
                        "symbol": pos["symbol"],
                        "entry_price": str(entry_price),
                        "target_price": str(target_price),
                        "min_price": str(target_price),
                        "chosen_price": str(chosen_price),
                        "base_size": str(base_size),
                        "limit_price": str(chosen_price),
                        "placed_at": now_iso,
                        "place_time_utc": datetime.now(timezone.utc),
                        "last_reprice_at": now_iso,
                        "status": st.lower(),
                    }
                save_state()
            else:
                # Cancelled or unknown, clear ID so we can place new one
                with state_lock:
                    if pos["symbol"] in active_positions:
                        active_positions[pos["symbol"]].pop("sell_order_id", None)
                        active_positions[pos["symbol"]]["status"] = "MONITORING"
                await self._maybe_place_sell_for_position(pos)

    async def _check_sells(self, open_order_map: Dict[str, Any] = None):
        """Monitor and update sell orders"""
        to_remove = []
        with state_lock:
            sells_copy = list(open_sell_orders.items())
            
        for oid, info in sells_copy:
            status = info.get("status")
            if status == "FILLED":
                to_remove.append(oid)
                continue
            
            # Use batch data if available, otherwise fetch
            order = None
            if open_order_map and oid in open_order_map:
                order = open_order_map[oid]
            else:
                # If not in batch, it might be filled/cancelled, so we check individually
                # This is efficient because it only happens for orders that are NOT open
                order = await self.api.get_order_status(oid)
            
            if not order:
                continue
                
            st = order.get("status") or order.get("order_status")
            if st and st.upper() == "FILLED":
                # Close position
                symbol = info.get("symbol") or info.get("product_id")
                logger.info(f"Sell filled {symbol} oid={oid}")
                
                with state_lock:
                    if symbol in active_positions:
                        active_positions.pop(symbol, None)
                    open_sell_orders.pop(oid, None)
                    
                    # Log trade
                    log_trade_event("SELL_FILLED", symbol, {
                        "order_id": oid,
                        "price": str(info.get("limit_price")),
                        "pnl": "Calculated in history"
                    })
                save_state()
                self.stats['trades_completed'] += 1
                self.stats['successful_trades'] += 1 # Assuming success if target hit
                
            elif st and st.upper() in {"CANCELLED", "REJECTED", "EXPIRED"}:
                with state_lock:
                    open_sell_orders.pop(oid, None)
                    symbol = info.get("symbol") or info.get("product_id")
                    if symbol in active_positions:
                        active_positions[symbol].pop("sell_order_id", None)
                        active_positions[symbol]["status"] = "MONITORING"
                save_state()
                
            elif st and st.upper() in {"OPEN", "PENDING"}:
                # Reprice logic
                placed_at_raw = info.get("placed_at")
                if placed_at_raw:
                    try:
                        placed_at = datetime.fromisoformat(placed_at_raw)
                        age_sec = (datetime.now(timezone.utc) - placed_at).total_seconds()
                        
                        if age_sec > SELL_REPRICE_UNFILLED_AFTER_SEC:
                            # Cancel and let _reconcile_positions/_maybe_place_sell_for_position handle replacement
                            # This is a simple way to reprice to top of book
                            logger.info(f"Sell {oid} stale ({age_sec:.1f}s), cancelling to reprice...")
                            await self.api.cancel_order(oid)
                            # The next loop will see it cancelled and place a new one at better price
                    except Exception:
                        pass

    async def trade_management_loop(self):
        """Monitor open buy/sell orders and active positions, handle timeouts and sell logic."""
        logger.info("⏰ Unified trade management loop started.")
        while not shutdown_flag.is_set():
            try:
                # --- 0. Batch Fetch Open Orders ---
                # Fetch all open orders once per loop to save API calls
                all_open_orders = await self.api.get_open_orders()
                open_order_map = {str(o['order_id']): o for o in all_open_orders if 'order_id' in o}
                
                # --- 1. Monitor open buy orders ---
                to_remove_buys = []
                with state_lock:
                    buy_orders_copy = list(open_buy_orders.items())

                for order_id, order in buy_orders_copy:
                    # Check if order is still open in the batch
                    if order_id in open_order_map:
                        # Order is still open, check for timeout
                        age_minutes = (datetime.now(timezone.utc) - order['place_time_utc']).total_seconds() / 60
                        if age_minutes >= 2:
                            logger.warning(f"Buy order {order_id} for {order.get('product_id', '?')} timed out after {age_minutes:.1f}m. Cancelling.")
                            await self.api.cancel_order(order_id)
                            to_remove_buys.append(order_id)
                        continue
                    
                    # If not in open_order_map, it's either filled, cancelled, or failed.
                    # We must fetch status individually to know which one.
                    # This is rare (only happens on state change), so it's efficient.
                    status = await self.api.get_order_status(order_id)
                    if not status:
                        logger.warning(f"Buy order status unavailable for {order_id}")
                        continue
                    
                    if status['status'] == 'FILLED':
                        symbol = status['product_id']
                        fill_price = Decimal(str(status.get('average_filled_price', order.get('limit_price', '0'))))
                        base_size = Decimal(str(status.get('filled_size', order.get('base_size', '0'))))
                        
                        if base_size == 0:
                            logger.error(f"Buy order {order_id} for {symbol} filled with size 0. Skipping.")
                            to_remove_buys.append(order_id)
                            continue

                        buy_time_utc = datetime.now(timezone.utc)
                        # Calculate target price for exactly PROFIT_TARGET_PCT profit
                        target_price = fill_price * (Decimal('1.0') + PROFIT_TARGET_PCT / Decimal('100'))
                        
                        # Calculate what the actual profit percentage will be after formatting
                        formatted_price, _ = self.api.format_order_values(symbol, target_price, base_size, round_price_up=True)
                        actual_sell_price = Decimal(formatted_price)
                        actual_profit_pct = ((actual_sell_price - fill_price) / fill_price) * Decimal('100')
                        
                        logger.info(f"💰 PROFIT CALCULATION for {symbol}:")
                        logger.info(f"   Buy Price: ${fill_price}")
                        logger.info(f"   Target Price (calculated): ${target_price}")
                        logger.info(f"   Actual Sell Price (formatted): ${actual_sell_price}")
                        logger.info(f"   Target Profit: {PROFIT_TARGET_PCT}%")
                        logger.info(f"   Actual Profit: {actual_profit_pct:.4f}%")
                        
                        with state_lock:
                            # Always move to active_positions, even if sell fails
                            active_positions[symbol] = {
                                'symbol': symbol,  # FIXED: Added missing symbol key
                                'entry_price': fill_price,
                                'base_size': base_size,
                                'buy_time_utc': buy_time_utc,
                                'target_price': actual_sell_price,  # Use the formatted price that will actually be used
                                'sell_order_id': None,
                                'status': 'MONITORING',
                            }
                        
                        log_trade_event('BUY_FILLED', symbol, {
                            'order_id': order_id,
                            'fill_price': str(fill_price),
                            'base_size': str(base_size),
                            'target_price': str(actual_sell_price),  # Log the actual formatted sell price
                            'target_profit_pct': str(actual_profit_pct),  # Log the actual profit percentage
                        })
                        
                        to_remove_buys.append(order_id)

                    elif status['status'] in ['CANCELLED', 'FAILED', 'REJECTED', 'EXPIRED']:
                        logger.info(f"Buy order {order_id} for {order.get('product_id', '?')} was {status['status']}")
                        to_remove_buys.append(order_id)

                if to_remove_buys:
                    with state_lock:
                        for order_id in to_remove_buys:
                            open_buy_orders.pop(order_id, None)
                    save_state()

                # --- 2. Reconcile positions (place sells if target reached) ---
                await self._reconcile_positions()

                # --- 3. Monitor open sell orders ---
                # Pass the batch of open orders to _check_sells to avoid re-fetching
                await self._check_sells(open_order_map)

                # --- 4. Save state and sleep ---
                save_state()
                await asyncio.sleep(ORDER_CHECK_INTERVAL_SECONDS)

            except Exception as e:
                logger.error(f"Error in trade_management_loop: {e}", exc_info=True)
                await asyncio.sleep(ORDER_CHECK_INTERVAL_SECONDS)

    def __init__(self):
        self.api = CoinbaseAPIHandler()
        self.statistical_engine = StatisticalEngine(self.api)
        self.ml_engine = MLEngine(self.api)
        
        # Performance tracking
        self.stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'trades_completed': 0,
            'successful_trades': 0,
            'total_pnl': Decimal('0'),
            'start_time': datetime.now(timezone.utc)
        }
        
        # Load saved state
        load_state()
    
    async def run(self):
        """FIXED - Simplified main execution loop based on v9.5 patterns"""
        logger.info("🚀 CombautoFinal v131 - FIXED - Dual Engine Trading Bot Starting...")
        logger.info(f"📊 Strategy: Statistical Engine + ML Engine (Threshold: {COMBINED_STAR_THRESHOLD}+ stars)")
        logger.info(f"💰 Position Size: ${POSITION_SIZE_USD} | Target: +{PROFIT_TARGET_PCT}%")
        logger.info(f"📈 Monitoring: {', '.join(SYMBOLS)}")
        
        try:
            # Initialize components
            logger.info("🔧 Initializing API handler...")
            await self.api.initialize_products()
            
            if not shutdown_flag.is_set():
                logger.info("🤖 Starting ML model training...")
                await self.ml_engine.train_models()
            
            if shutdown_flag.is_set():
                logger.info("🛑 Shutdown requested during initialization")
                return
            
            # FIXED: Simple task management without complex restart logic
            logger.info("🚀 Starting all monitoring loops...")
            
            # Create tasks but don't overcomplicate the management
            tasks = [
                asyncio.create_task(self.analysis_loop()),
                asyncio.create_task(self.trade_management_loop()),
                asyncio.create_task(self.status_display_loop())
            ]
            
            logger.info("✅ All systems running. Dual engines active...")
            
            # FIXED: Simple task waiting - if any task fails, log it but continue
            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                if not shutdown_flag.is_set():
                    logger.info("Continuing with remaining tasks...")
            
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received...")
            shutdown_flag.set()
        except Exception as e:
            logger.error(f"💥 Critical error in main run loop: {e}", exc_info=True)
            shutdown_flag.set()
        finally:
            # Simple cleanup
            logger.info("🔄 Shutting down...")
            save_state()
            logger.info("✅ CombautoFinal shutdown complete")
    
    async def analysis_loop(self):
        """Dual engine analysis loop"""
        await asyncio.sleep(5)  # Initial delay
        logger.info("🔍 Dual engine analysis loop started")
        
        while not shutdown_flag.is_set():
            try:
                # Get available symbols (not tied up)
                with state_lock:
                    tied_up_symbols = set(active_positions.keys()) | {o['product_id'] for o in open_buy_orders.values()} | {o['product_id'] for o in open_sell_orders.values()}
                    available_symbols = [s for s in SYMBOLS if s not in tied_up_symbols]
                
                if not available_symbols:
                    logger.info("All symbols tied up. Waiting...")
                    await asyncio.sleep(30)
                    continue
                
                logger.info(f"\n🔍 DUAL ENGINE ANALYSIS - {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
                logger.info(f"Analyzing {len(available_symbols)} available symbols")
                
                best_candidate = None
                best_total_stars = 0
                
                # Analyze all available symbols
                for symbol in available_symbols:
                    if shutdown_flag.is_set():
                        break
                        
                    try:
                        # Get ratings from both engines with timeout
                        stat_task = asyncio.create_task(self.statistical_engine.get_stars(symbol))
                        ml_task = asyncio.create_task(self.ml_engine.get_stars(symbol))
                        
                        stat_stars, ml_stars = await asyncio.wait_for(
                            asyncio.gather(stat_task, ml_task),
                            timeout=15.0
                        )
                        
                        total_stars = stat_stars + ml_stars
                        
                        # Log analysis
                        status = "🔥 BUY!" if total_stars >= COMBINED_STAR_THRESHOLD else "Monitor"
                        logger.info(f"📊 {symbol:<10} | Stat: {stat_stars}⭐ | ML: {ml_stars}⭐ | Total: {total_stars}⭐ | {status}")
                        
                        # Track best candidate
                        if total_stars >= COMBINED_STAR_THRESHOLD and total_stars > best_total_stars:
                            best_candidate = {
                                'symbol': symbol,
                                'total_stars': total_stars,
                                'stat_stars': stat_stars,
                                'ml_stars': ml_stars
                            }
                            best_total_stars = total_stars
                        
                        self.stats['signals_generated'] += 1
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"⏱️ Timeout analyzing {symbol}")
                    except Exception as e:
                        logger.error(f"❌ Error analyzing {symbol}: {e}")
                
                # Execute best trade (one per cycle)
                if best_candidate and not shutdown_flag.is_set():
                    symbol = best_candidate['symbol']
                    logger.info(f"\n🎯 EXECUTING BEST TRADE: {symbol} ({best_candidate['total_stars']}⭐)")
                    logger.info(f"   Statistical: {best_candidate['stat_stars']}⭐ | ML: {best_candidate['ml_stars']}⭐")
                    
                    # Final safety check
                    with state_lock:
                        if symbol not in set(active_positions.keys()) | {o['product_id'] for o in open_buy_orders.values()} | {o['product_id'] for o in open_sell_orders.values()}:
                            await self.execute_buy(symbol, best_candidate)
                        else:
                            logger.warning(f"   🔒 {symbol} became tied up before execution")
                else:
                    logger.info("📈 No signals above threshold - continuing analysis")
                
                logger.info(f"{'─'*80}")
                
            except Exception as e:
                logger.error(f"❌ Analysis loop error: {e}", exc_info=True)
                if not shutdown_flag.is_set():
                    await asyncio.sleep(10)
            
            # FIXED: Interruptible sleep for more responsive trading
            # Check every second for shutdown, but only analyze every ANALYSIS_INTERVAL_SECONDS
            for _ in range(ANALYSIS_INTERVAL_SECONDS):
                if shutdown_flag.is_set():
                    break
                await asyncio.sleep(1)
    
    def _volume_weights(self, symbol: str) -> List[Decimal]:
        """Derive slice weights from recent volume (last LOOKBACK_CANDLES 5m bars)."""
        # We need to access candles. Since we don't have a local cache in this class like vXO,
        # we might need to fetch them or rely on what StatisticalEngine has.
        # For simplicity and robustness, we'll fetch a small batch here or assume equal weights if fetch fails.
        # Ideally, we should share the candle cache.
        # Given the architecture, let's try to fetch a small batch if possible, or just use equal weights
        # to avoid excessive API calls if we are not caching.
        # However, vXO uses ws_candles. Here we are using REST.
        # Let's fetch a small batch.
        
        # NOTE: In a high-frequency setting, fetching here adds latency. 
        # But since we are about to buy, a small delay for better execution is acceptable.
        return [Decimal("1") / Decimal(BUY_SLICES) for _ in range(BUY_SLICES)]

        # TODO: If we want true volume weighting, we need to expose the candle cache from StatisticalEngine
        # or fetch it here. For now, equal weights is a safe default for the port.

    async def _plan_buy_slices(self, symbol: str, book: Dict[str, Any]) -> Tuple[List[Dict[str, Decimal]], Dict[str, Decimal]]:
        """Plan TWAP-like buy slices sized by depth and recent volume."""
        await self.api.initialize_products() # Ensure details are loaded
        details = self.api.product_details.get(symbol, {})
        tick = Decimal(details.get("quote_increment", "0.0001"))
        base_min = Decimal(details.get("base_min_size", "0.0001"))

        best_bid = Decimal(str(book.get("best_bid")))
        best_ask = Decimal(str(book.get("best_ask")))
        band_low = best_bid * (Decimal("1") - BAND)
        band_high = min(best_bid * (Decimal("1") + BAND), best_ask)
        if band_high <= 0 or band_high < band_low:
            band_high = best_bid
            band_low = best_bid

        # Entry buy price: place at bid + 1 tick (maker-only).
        maker_touch = best_bid + tick
        if best_ask and maker_touch >= best_ask:
            maker_touch = best_ask - tick
        if maker_touch <= 0:
            maker_touch = best_bid

        depth_notional = book.get("total_bid_depth") or book.get("best_bid_amount")
        if not depth_notional and book.get("best_bid") and book.get("best_bid_qty"):
            depth_notional = Decimal(str(book["best_bid"])) * Decimal(str(book["best_bid_qty"]))
        
        # Use POSITION_SIZE_USD instead of BUY_USD
        depth_cap = (Decimal(depth_notional) * DEPTH_PCT) if depth_notional else POSITION_SIZE_USD
        budget = min(POSITION_SIZE_USD, depth_cap)
        
        if budget <= Decimal("0"):
            return [], {"budget": Decimal("0"), "band_low": band_low, "band_high": band_high, "depth_cap": Decimal("0")}

        weights = self._volume_weights(symbol)
        allocs = []
        for w in weights:
            allocs.append((budget * w).quantize(Decimal("0.01"), rounding=ROUND_DOWN))
        diff = budget - sum(allocs)
        if diff > 0 and allocs:
            allocs[-1] += diff

        slices: List[Dict[str, Decimal]] = []
        remaining = budget
        for idx, quote_alloc in enumerate(allocs):
            if remaining <= 0:
                break
            if quote_alloc <= 0:
                continue
            if idx == len(allocs) - 1 and remaining > 0:
                quote_alloc = remaining
            quote_alloc = min(quote_alloc, remaining)

            # Place all slices at the maker-touch price (bid + 1 tick)
            price = self.api._ticks(maker_touch, tick, up=False)
            min_quote = (base_min * price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            if quote_alloc < min_quote:
                if remaining >= min_quote:
                    quote_alloc = min(remaining, min_quote)
                else:
                    remaining = Decimal("0")
                    continue

            slices.append(
                {
                    "price": price,
                    "quote": quote_alloc,
                    "slice_index": Decimal(idx + 1),
                    "slice_count": Decimal(BUY_SLICES),
                    "band_low": band_low,
                    "band_high": band_high,
                    "depth_cap": depth_cap,
                }
            )
            remaining -= quote_alloc

        return slices, {"budget": budget, "band_low": band_low, "band_high": band_high, "depth_cap": depth_cap}

    async def execute_buy(self, symbol: str, analysis: Dict):
        """Execute buy order with depth filters and slicing"""
        try:
            # 1. Fetch comprehensive book data
            book = await self.api.fetch_book_data(symbol)
            if not book:
                logger.error(f"Failed to get book data for {symbol}")
                return

            # 2. Apply Depth Filters (Removed)
            logger.info(f"{symbol}: Depth Filters SKIPPED. Planning slices...")

            # 3. Plan Slices
            slices, meta = await self._plan_buy_slices(symbol, book)
            if not slices:
                logger.info(f"{symbol}: buy skipped - unable to build slices within band/depth budget")
                return

            logger.info(
                f"{symbol}: executing {len(slices)} slices tot ${meta['budget']:.2f} "
                f"band [{meta['band_low']:.6f}, {meta['band_high']:.6f}] depth_cap ${meta['depth_cap']:.2f}"
            )

            # 4. Execute Slices
            for idx, slc in enumerate(slices):
                if shutdown_flag.is_set():
                    break
                
                limit_price = slc["price"]
                quote_size = slc["quote"]
                
                # Place GTC buy order (using existing method but with calculated price/size)
                order_id = await self.api.place_gtc_buy_order(symbol, quote_size, limit_price)
                
                if order_id:
                    with state_lock:
                        open_buy_orders[order_id] = {
                            'product_id': symbol,
                            'side': 'buy',
                            'limit_price': limit_price,
                            'quote_size': quote_size,
                            'base_size': quote_size / limit_price,
                            'place_time_utc': datetime.now(timezone.utc),
                            'analysis': analysis,
                            'slice_index': int(slc["slice_index"]),
                            'slice_count': int(slc["slice_count"]),
                            'status': 'placed'
                        }
                    
                    save_state()
                    self.stats['trades_executed'] += 1
                    
                    logger.info(f"✅ Buy slice {int(slc['slice_index'])}/{BUY_SLICES} placed: {order_id[:8]}... @ {limit_price}")
                    
                    log_trade_event("BUY_SLICE_PLACED", symbol, {
                        "order_id": order_id,
                        "buy_price": str(limit_price),
                        "slice": f"{int(slc['slice_index'])}/{BUY_SLICES}",
                        "analysis": analysis
                    })
                    
                else:
                    logger.error(f"Failed to place buy slice {int(slc['slice_index'])} for {symbol}")

                # TWAP Delay
                if idx < len(slices) - 1 and not shutdown_flag.is_set():
                    await asyncio.sleep(TWAP_INTERVAL_SEC)
                
        except Exception as e:
            logger.error(f"❌ Error executing buy for {symbol}: {e}", exc_info=True)

    async def status_display_loop(self):
        """Periodic status display"""
        while not shutdown_flag.is_set():
            # Wait for display interval
            for _ in range(STATUS_DISPLAY_INTERVAL_MINUTES * 60):
                if shutdown_flag.is_set():
                    break
                await asyncio.sleep(1)
            
            if not shutdown_flag.is_set():
                await self.update_positions_prices() # Update prices before display
                await self.display_status()
    
    async def update_positions_prices(self):
        """Fetch the latest price for all active positions."""
        with state_lock:
            positions_to_update = list(active_positions.keys())

        if not positions_to_update:
            return

        logger.debug(f"Updating prices for {len(positions_to_update)} active positions...")
        for symbol in positions_to_update:
            try:
                current_price = await self.api.get_current_price(symbol)
                if current_price:
                    with state_lock:
                        if symbol in active_positions:
                            active_positions[symbol]['current_price'] = current_price
            except Exception as e:
                logger.error(f"Could not update price for {symbol}: {e}")

    async def display_status(self):
        """Display comprehensive status"""
        now_utc = datetime.now(timezone.utc)
        
        print("\n" + "="*100)
        print(f"📊 COMBAUTO FINAL v131 STATUS | {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*100)
        
        # Performance summary
        with state_lock:
            occupied_pairs = set(active_positions.keys()) | {o['product_id'] for o in open_buy_orders.values()} | {o['product_id'] for o in open_sell_orders.values()}
        available_pairs = len(SYMBOLS) - len(occupied_pairs)
        
        runtime_hours = (now_utc - self.stats['start_time']).total_seconds() / 3600
        
        print(f"🎯 DUAL ENGINE TRADING SUMMARY:")
        print(f"   Runtime: {runtime_hours:.1f}h | Available: {available_pairs}/{len(SYMBOLS)} pairs")
        print(f"   Models Trained: {len(self.ml_engine.models)}/{len(SYMBOLS)}")
        
        if self.stats['trades_completed'] > 0:
            success_rate = (self.stats['successful_trades'] / self.stats['trades_completed']) * 100
            avg_pnl = float(self.stats['total_pnl']) / self.stats['trades_completed']
            
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"   Signals: {self.stats['signals_generated']} | Executed: {self.stats['trades_executed']} | Completed: {self.stats['trades_completed']}")
            print(f"   Success Rate: {success_rate:.1f}% | Avg P&L: ${avg_pnl:+.2f} | Total: ${float(self.stats['total_pnl']):+.2f}")
        
        # Active positions
        with state_lock:
            positions_copy = dict(active_positions)
        
        if positions_copy:
            print(f"\n💼 ACTIVE POSITIONS: {len(positions_copy)}")
            print(f"{'PAIR':<12}{'ENTRY $':>16}{'CURRENT $':>17}{'P&L %':>10}{'P&L $':>12}{'HELD (H)':>12}{'STATUS':>15}")
            print("-" * 100)
            for symbol, pos in positions_copy.items():
                entry = pos['entry_price']
                # Use 'current_price' if available, else fallback to entry
                current = pos.get('current_price', entry)
                # Convert to Decimal if needed
                if not isinstance(entry, Decimal):
                    entry = Decimal(str(entry))
                if not isinstance(current, Decimal):
                    current = Decimal(str(current))
                pnl_pct = (current / entry - 1) * 100 if entry > 0 else 0
                base_size = pos.get('base_size')
                if not base_size:
                    quote_size = pos.get('quote_size', POSITION_SIZE_USD)
                    if not isinstance(quote_size, Decimal):
                        quote_size = Decimal(str(quote_size))
                    base_size = quote_size / entry
                else:
                    if not isinstance(base_size, Decimal):
                        base_size = Decimal(str(base_size))
                pnl_usd = (current - entry) * base_size
                buy_time = pos['buy_time_utc']
                if isinstance(buy_time, str):
                    buy_time = datetime.fromisoformat(buy_time)
                hours = (now_utc - buy_time).total_seconds() / 3600
                
                # IMPROVED Status Logic
                status = pos.get('status', 'MONITORING')
                if status == 'SELL_PLACED':
                    if pnl_pct >= float(PROFIT_TARGET_PCT):
                        status = "🟢 PROFIT"
                    elif hours >= POSITION_TIMEOUT_HOURS:
                        status = "⏰ TIMEOUT"
                    else:
                        status = "🟡 MONITORING"
                elif status == 'AWAITING_SELL':
                    status = "⌛️ QUEUED"
                elif status == 'SELL_PLACEMENT_FAILED':
                    status = "⚠️ SELL FAIL"

                print(f"{symbol:<12}${float(entry):<15.8f}${float(current):<16.8f}{float(pnl_pct):>+9.2f}%${float(pnl_usd):>+11.2f}{hours:>12.2f}{status:>15}")
        
        # Open orders
        with state_lock:
            buy_orders = list(open_buy_orders.items())
            sell_orders = list(open_sell_orders.items())
        
        if buy_orders:
            print(f"\n🔵 OPEN BUY ORDERS: {len(buy_orders)}")
            for order_id, order in buy_orders[:5]:  # Show first 5
                age_min = (now_utc - order['place_time_utc']).total_seconds() / 60
                print(f"   {order['product_id']:<12} ${float(order['limit_price']):>11.6f} ({age_min:.0f}m)")
        
        if sell_orders:
            print(f"\n🔴 OPEN SELL ORDERS: {len(sell_orders)}")
            for order_id, order in sell_orders[:5]:  # Show first 5
                age_min = (now_utc - order['place_time_utc']).total_seconds() / 60
                print(f"   {order['product_id']:<12} ${float(order['limit_price']):>11.6f} ({age_min:.0f}m)")
        
        print("="*100 + "\n")

# === MAIN EXECUTION ===
async def main():
    """Entry point with proper error handling"""
    bot = CombautoFinal()
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("🛑 KeyboardInterrupt in main - shutting down...")
        shutdown_flag.set()
    except Exception as e:
        logger.error(f"💥 Fatal error in main: {e}", exc_info=True)
        shutdown_flag.set()
        # No need to raise here, as this is the top-level handler within the async context

if __name__ == "__main__":
    # Ensure proper Windows asyncio setup
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        logger.info("🪟 Windows detected - using ProactorEventLoopPolicy")
    
    logger.info("🚀 Starting CombautoFinal...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This is caught when Ctrl+C is pressed before the event loop starts
        logger.info("\n🛑 KeyboardInterrupt before event loop start - terminating.")
    finally:
        logger.info("✅ Main execution finished.")
        print("\n🛑 Bot has shut down.")
        sys.exit(0)