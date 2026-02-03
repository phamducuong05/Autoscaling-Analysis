import pandas as pd
import os
from config.settings import LOG_PATTERN, PROJECT_ROOT, DATA_RAW_DIR


def parse_log_line(line):
    """
    Parse một dòng log raw thành dictionary
    """
    match = LOG_PATTERN.match(line)
    if match:
        data = match.groupdict()
        data['bytes'] = 0 if data['bytes'] == '-' else int(data['bytes'])
        data['status'] = int(data['status'])
        return data
    return None


def load_and_process_logs(file_paths):
    """
    Đọc file txt log và trả về DataFrame
    """
    parsed_data = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
            
        print(f"Reading file: {file_path}...")
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                parsed = parse_log_line(line)
                if parsed:
                    parsed_data.append(parsed)
    
    if not parsed_data:
        return pd.DataFrame()

    df = pd.DataFrame(parsed_data)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(
        df['timestamp'], 
        format='%d/%b/%Y:%H:%M:%S %z'
    )
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    return df


def resample_traffic(df, window='5min'):
    """
    Resample traffic data theo khoảng thời gian xác định
    """
    df_idx = df.set_index('timestamp')
    
    resampled = df_idx.resample(window).agg({
        'request': 'count',
        'bytes': 'sum',
        'host': 'nunique',
        'status': lambda x: (x >= 400).sum()
    })
    
    resampled.columns = ['requests', 'bytes', 'hosts', 'errors']
    
    # Fill 0 cho những khoảng trống 
    resampled = resampled.fillna(0)
    
    return resampled
