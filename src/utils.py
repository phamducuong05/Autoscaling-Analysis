import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config.settings import PLOT_CONFIG
import yaml
import os

def plot_imputation_check(df_input, outages):
    """
    Vẽ biểu đồ so sánh dữ liệu tuần bị crash và tuần trước đó.
    
    Hàm này giúp kiểm tra xem giá trị imputed (lấy từ 7 ngày trước) 
    có hợp lý không bằng cách so sánh với pattern tuần bình thường.
    """
    df = df_input.copy()
    df.index = df.index.tz_localize(None)
    
    # Cấu hình plot
    fig, axes = plt.subplots(
        nrows=len(outages),
        ncols=1,
        figsize=PLOT_CONFIG['figsize']
    )
    if len(outages) == 1:
        axes = [axes]
    
    # Khoảng offset 7 ngày
    offset_7d = pd.Timedelta(days=7)

    for i, (start, end) in enumerate(outages):
        ax = axes[i]
        
        # 1. Xác định khung nhìn
        # Lấy rộng ra mỗi bên 2 ngày để thấy context trước và sau khi sập
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        view_start = start_dt - pd.Timedelta(days=2)
        view_end = end_dt + pd.Timedelta(days=2)
        
        # 2. Lấy dữ liệu
        # Data tuần bị crash (Current)
        mask_current = (df.index >= view_start) & (df.index <= view_end)
        df_current = df.loc[mask_current]
        
        # Data tuần trước đó dùng để Impute
        # Lấy lùi về 7 ngày
        mask_past = (df.index >= (view_start - offset_7d)) & (df.index <= (view_end - offset_7d))
        df_past = df.loc[mask_past].copy()
        
        df_past.index = df_past.index + offset_7d

        # Plot dữ liệu
        ax.plot(
            df_past.index,
            df_past['requests'],
            color='orange',
            linestyle='--',
            alpha=0.7,
            label='Tuần trước (dùng để impute)',
            linewidth=2
        )
        
        ax.plot(
            df_current.index,
            df_current['requests'],
            color='blue',
            linewidth=1.5,
            label='Hiện tại',
            alpha=0.8
        )
        
        # Highlight outage period
        ax.axvspan(
            start_dt,
            end_dt,
            color='red',
            alpha=0.2,
            label='Khoảng Crash'
        )
        
        # Formatting
        ax.set_title(
            f'So sánh Pattern: Crash từ {start} đến {end}',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_ylabel('Số lượng Requests', fontsize=12)
        ax.set_xlabel('Thời gian', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(PLOT_CONFIG['date_format'])
        )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

def load_config(config_path="config/autoscaling_config.yaml"):
    if not os.path.exists(config_path):
        if os.path.exists(os.path.join("..", config_path)):
            config_path = os.path.join("..", config_path)
        elif os.path.exists(os.path.join("..", "..", config_path)):
            config_path = os.path.join("..", "..", config_path)
            
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return None