# config.example.py
# Example configuration file for grid trading bot with ML-driven parameters.
# Copy this file to 'config.py' and replace the placeholder values with your actual credentials.
# All parameters are adjustable via long_term_ml.py WebSocket commands.
# Constraints ensure stability and compliance with exchange limits.

# API Credentials - REPLACE WITH YOUR ACTUAL CREDENTIALS
API_KEY = "your-api-key-here"
SECRET_KEY = """your-private-key-here"""

# Trading Pair and Position Settings
SYMBOL = "ETH-USD"
POSITION_SIZE = 0.0018  # Fixed: was 0.00020, outside valid range [0.0018, 0.003]
STRATEGY_POSITION_SIZE = 0.05
MIN_POSITION_SIZE = 0.0018
MAX_POSITION_SIZE = 0.0030
NUM_BUY_GRID_LINES = 20  # Number of base buy grid lines (base orders)
NUM_SELL_GRID_LINES = 20  # Number of base sell grid lines (base orders)
MIN_NUM_GRID_LINES = 20
MAX_NUM_GRID_LINES = 120  # Maximum total grid lines (base + feature)
FEATURE_ORDER_CAP = 60  # Maximum number of feature-based orders (MAX_TOTAL_ORDERS - NUM_BUY_GRID_LINES - NUM_SELL_GRID_LINES)

# Grid Configuration
GRID_SIZE = 12.0
MIN_GRID_SIZE = 6.0
MAX_GRID_SIZE = 19.0
MAX_ORDER_RANGE = 500.0  # Increased to match run_bot
MIN_ORDER_RANGE = 100.0
PRICE_DRIFT_THRESHOLD = 2.0

# Timing and Status
CHECK_ORDER_FREQUENCY = 30
MIN_CHECK_FREQUENCY = 2.0
MAX_CHECK_FREQUENCY = 10.0
CHECK_ORDER_STATUS = 'closed'
STAGNATION_TIMEOUT = 452642
MIN_STAGNATION_TIMEOUT = 7200
MAX_STAGNATION_TIMEOUT = 86400
MIN_RESET_INTERVAL = 18000
MAIN_LOOP_RECOVERY_DELAY = 5.0
HEARTBEAT_INTERVAL = 30
TIMESTAMP_VALIDATION = 1300

# Initial Balance Allocation
INITIAL_ETH_PERCENTAGE = 1.0
MIN_INITIAL_ETH_PERCENTAGE = 0.1
MAX_INITIAL_ETH_PERCENTAGE = 0.9
TARGET_ETH_BUFFER = 1.0
MIN_TARGET_ETH_BUFFER = 1.0
MAX_TARGET_ETH_BUFFER = 2.0
ETH_BALANCE_DIVISOR = 3.0
MIN_USD_BALANCE = 0.48

# ETH Replenishment Controls
REPLENISH_ETH_THRESHOLD = 0.5
MIN_REPLENISH_THRESHOLD = 0.02
MAX_REPLENISH_THRESHOLD = 0.95
MIN_REPLENISH_AMOUNT = 0.0005
MAX_REPLENISH_AMOUNT = 0.005

REBALANCE_ETH_THRESHOLD = 0.02
MIN_REBALANCE_THRESHOLD = 0.01
MAX_REBALANCE_THRESHOLD = 0.07
MIN_TOTAL_ORDERS = 40
MAX_TOTAL_ORDERS_LIMIT = 200  # For compatibility, matches MAX_TOTAL_ORDERS

# Order Placement
SELL_SIZE_MULTIPLIER = 0.95
MIN_SELL_SIZE_MULTIPLIER = 0.8
MAX_SELL_SIZE_MULTIPLIER = 1.0
BUY_SIZE_MULTIPLIER = 1.05
MIN_BUY_SIZE_MULTIPLIER = 0.9
MAX_BUY_SIZE_MULTIPLIER = 1.09

# Trailing Take-Profit Configuration
TRAILING_STOP_PERCENT = 0.6  # e.g., 2.0 for a 2% trail from the peak price
MIN_TRAILING_STOP_PERCENT = 0.5
MAX_TRAILING_STOP_PERCENT = 10.0

# Adaptive Trailing Stop Configuration (ATR-based)
TRAILING_STOP_ATR_MULTIPLIER = 2.5  # Multiplier for ATR-based trailing stop (e.g., 2.5 * ATR)
MIN_TRAILING_STOP_ATR_MULTIPLIER = 1.0
MAX_TRAILING_STOP_ATR_MULTIPLIER = 5.0

# --- ADDED: Breakout Trading Configuration ---
BREAKOUT_ENABLED = True
# Multiplier for ATR to confirm a breakout beyond the Bollinger Band
BREAKOUT_CONFIRMATION_ATR_MULTIPLIER = 1.0
# Multiplier for the standard position size for breakout trades
BREAKOUT_POSITION_SIZE_MULTIPLIER = 2.5
# Multiplier for ATR to set the trailing stop for a breakout position
BREAKOUT_TRAILING_STOP_ATR_MULTIPLIER = -2.0
# Cooldown in seconds between breakout trades to prevent rapid re-entry
BREAKOUT_COOLDOWN_SECONDS = 1800 # 30 minutes
# DCA step size as percent (e.g. 0.5 = 0.5% per step)
DCA_STEP_SIZE_PERCENT = 0.5

# --- Additional trading parameters continue here ---
# (Include all other parameters from your original config.py file)
