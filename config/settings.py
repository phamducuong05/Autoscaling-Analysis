import re
from pathlib import Path

# ===========================
# Project Paths
# ===========================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ===========================
# Log Parsing Pattern
# ===========================
LOG_PATTERN = re.compile(
    r'(?P<host>\S+) - - \[(?P<timestamp>.*?)\] "(?P<request>.*?)" (?P<status>\d{3}) (?P<bytes>\S+)'
)

# ===========================
# System Outage Periods
# ===========================
# Tuple format: (start_datetime, end_datetime)
OUTAGES = [
    ('1995-08-01 14:52:02', '1995-08-03 04:36:12'),
    ('1995-07-28 13:32:26', '1995-08-01 00:00:00')
]

# ===========================
# Feature Engineering Parameters
# ===========================
# Frequency-specific parameters for feature engineering
FREQUENCY_PARAMS = {
    '1m': {
        'lag_1_steps': 1,
        'lag_1h_steps': 60,
        'lag_24h_steps': 1440,
        'lag_7d_steps': 10080,
        'window_1h': 60,
        'window_24h': 1440,
    },
    '5m': {
        'lag_1_steps': 1,
        'lag_1h_steps': 12,
        'lag_24h_steps': 288,
        'lag_7d_steps': 2016,
        'window_1h': 12,
        'window_24h': 288,
    },
    '15m': {
        'lag_1_steps': 1,
        'lag_1h_steps': 4,
        'lag_24h_steps': 96,
        'lag_7d_steps': 672,
        'window_1h': 4,
        'window_24h': 96,
    },
}

# ===========================
# Resampling Configuration
# ===========================
RESAMPLING_WINDOWS = {
    '1m': '1min',
    '5m': '5min',
    '15m': '15min',
}

# ===========================
# Data Column Names
# ===========================
TRAFFIC_COLUMNS = {
    'requests': 'requests',
    'bytes': 'bytes',
    'hosts': 'hosts',
    'errors': 'errors',
}

# ===========================
# Feature Column Names
# ===========================
FEATURE_COLUMNS = [
    'requests',
    'bytes',
    'hosts',
    'errors',
    'error_rate',
    'requests_target',
    'req_lag_1',
    'req_lag_12',
    'req_lag_288',
    'rolling_mean_1h',
    'rolling_std_1h',
    'rolling_mean_24h',
    'err_lag_1',
    'err_rolling_mean_1h',
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    'hour_sin',
    'hour_cos',
]

# ===========================
# Visualization Parameters
# ===========================
PLOT_CONFIG = {
    'figsize': (15, 6),
    'date_format': '%m-%d %H:%M',
}
