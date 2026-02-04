"""
Dashboard Configuration Constants
Extracted from dashboard.py - EXACT values, no changes
"""

# Simulation Configuration
HISTORY_WINDOW_SIZE = 30  # Send last 30 mins to API
CHART_WINDOW_SIZE = 50    # Display last 50 points on chart
DEFAULT_SPEED_MS = 1000   # Default simulation speed

# Default Simulation Parameters
DEFAULT_START_IDX = 42000  # Pick a busy time
DEFAULT_DURATION = 200     # Minutes

# Data Configuration
TEST_DATA_PATH = "data/raw/test.txt"
RESAMPLE_WINDOW = '5min'
FREQUENCY = '5m'

# Confidence Interval
SIGMA_MULTIPLIER = 2  # 2 Sigma (~95% confidence)
