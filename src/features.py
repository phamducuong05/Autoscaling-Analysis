import numpy as np
import pandas as pd
from config.settings import OUTAGES, FREQUENCY_PARAMS


def add_features(df_input, frequency='5m'):
    """
    Thêm các features vào dataframe dựa vào tần suất sampling
    
    Features bao gồm:
    - Lag features: req_lag_1, req_lag_12, req_lag_288
    - Rolling features: rolling_mean_1h, rolling_std_1h, rolling_mean_24h
    - Error features: error_rate, err_lag_1, err_rolling_mean_1h
    - Time features: hour_of_day, day_of_week, is_weekend, hour_sin, hour_cos
    - Target: requests_target (imputed values cho outage periods)
    """
    df = df_input.copy()
    
    # Xác định các tham số dựa vào tần suất
    if frequency not in FREQUENCY_PARAMS:
        raise ValueError(
            f"Frequency '{frequency}' không được hỗ trợ. "
            f"Hãy chọn: {', '.join(FREQUENCY_PARAMS.keys())}"
        )
    
    params = FREQUENCY_PARAMS[frequency]
    lag_1_steps = params['lag_1_steps']
    lag_1h_steps = params['lag_1h_steps']
    lag_24h_steps = params['lag_24h_steps']
    lag_7d_steps = params['lag_7d_steps']
    window_1h = params['window_1h']
    window_24h = params['window_24h']

    # ===========================
    # 1. Calculate error rate
    # ===========================
    df['error_rate'] = df['errors'] / (df['requests'] + 1e-9)
    df['error_rate'] = df['error_rate'].fillna(0.0)

    # ===========================
    # 2. Impute values for outage periods
    # ===========================
    df['requests_target'] = df['requests']

    for start, end in OUTAGES:
        mask = (df.index >= start) & (df.index <= end)
        df.loc[mask, 'requests_target'] = df['requests'].shift(lag_7d_steps).loc[mask]
        df.loc[mask, 'error_rate'] = df['error_rate'].shift(lag_7d_steps).loc[mask]

    # Forward fill và backward fill cho missing values
    # Use simple forward fill for outages simulation
    # In pandas 2.0+, method='ffill' is deprecated.
    df['requests_target'] = df['requests_target'].ffill().fillna(0)
    df['error_rate'] = df['error_rate'].ffill().fillna(0)

    # ===========================
    # 3. Create lag features
    # ===========================
    target_for_lag = 'requests_target'

    df['req_lag_1'] = df[target_for_lag].shift(lag_1_steps)
    df['req_lag_12'] = df[target_for_lag].shift(lag_1h_steps)
    df['req_lag_288'] = df[target_for_lag].shift(lag_24h_steps)

    # ===========================
    # 4. Create rolling features
    # ===========================
    df['rolling_mean_1h'] = df[target_for_lag].rolling(window=window_1h).mean()
    df['rolling_std_1h'] = df[target_for_lag].rolling(window=window_1h).std()
    df['rolling_mean_24h'] = df[target_for_lag].rolling(window=window_24h).mean()

    # ===========================
    # 5. Create error rate features
    # ===========================
    df['err_lag_1'] = df['error_rate'].shift(lag_1_steps)
    df['err_rolling_mean_1h'] = df['error_rate'].rolling(window=window_1h).mean()

    # ===========================
    # 6. Extract time features (Cyclic encoding)
    # ===========================
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5

    # Cyclic encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

    # ===========================
    # 7. Clean and type cast
    # ===========================
    df = df.dropna()

    # Type Casting
    df['requests'] = df['requests'].astype(int)
    df['bytes'] = df['bytes'].astype(int)
    df['hosts'] = df['hosts'].astype(int)
    df['hour_of_day'] = df['hour_of_day'].astype(int)
    df['day_of_week'] = df['day_of_week'].astype(int)
    df['is_weekend'] = df['is_weekend'].astype(int)
    df['error_rate'] = df['error_rate'].astype(float)
    df['requests_target'] = df['requests_target'].astype(int)

    return df


def clean_data(df_input, quantile_threshold=0.99):
    """
    Làm sạch dữ liệu bằng cách:
    - Phát hiện và đánh dấu spikes (requests_target > threshold)
    - Clip giá trị spike về threshold
    - Interpolate các điểm có requests_target == 0
    """
    df = df_input.copy()
    
    threshold = df['requests_target'].quantile(quantile_threshold)
    
    df['is_spike'] = (df['requests_target'] > threshold).astype(int)
    
    df.loc[df['is_spike'] == 1, 'requests_target'] = threshold
    
    
    df.loc[df['requests_target'] == 0, 'requests_target'] = np.nan
    df['requests_target'] = df['requests_target'].interpolate(
        method='time'
    )
    
    # Fill remaining NaN (nếu có) bằng forward fill và backward fill
    df['requests_target'] = df['requests_target'].fillna(method='ffill')
    
    # Type cast
    df['requests_target'] = df['requests_target'].astype(int)
    
    return df
