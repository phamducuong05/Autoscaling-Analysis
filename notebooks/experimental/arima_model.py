# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # ARIMA Model for Traffic Forecasting
#
# ## Overview
# This notebook implements an **ARIMA (AutoRegressive Integrated Moving Average)** model to forecast HTTP request traffic for the NASA server logs from July-August 1995. ARIMA is a classical time series forecasting method that combines three components:
#
# - **AR (AutoRegressive):** Uses past values to predict future values
# - **I (Integrated):** Applies differencing to make the time series stationary
# - **MA (Moving Average):** Uses past forecast errors to improve predictions
#
# The ARIMA model is particularly suitable for:
# - Univariate time series data (single variable: request count)
# - Data with clear trends and/or seasonal patterns
# - Situations where interpretability of model parameters is important
# - Baseline comparisons before trying more complex models
#
# **Note:** ARIMA assumes linear relationships and may not capture complex non-linear patterns. We will evaluate its performance and potentially compare with more advanced models (Prophet, LSTM, XGBoost) in subsequent analyses.
#
# ---
#
# ## Table of Contents
# 1. **Pre-train: Suitability & Data Preparation**
#    - Data loading and exploration
#    - Stationarity assessment (ADF test)
#    - ACF/PACF analysis
#    - Data preprocessing and train/test split
#
# 2. **Training**
#    - ARIMA parameter selection
#    - Model fitting
#    - Model diagnostics
#
# 3. **Post-train Evaluation**
#    - Forecast generation
#    - Performance metrics (RMSE, MSE, MAE, MAPE)
#    - Visualization and residual analysis
#    - Conclusions and recommendations

# %% [markdown]
# # Section 1: Pre-train - Suitability & Data Preparation
#
# In this section, we will:
# 1. Load the cleaned traffic data
# 2. Explore the data to understand its characteristics
# 3. Assess whether ARIMA is suitable by checking stationarity
# 4. Analyze autocorrelation patterns using ACF and PACF
# 5. Prepare the data for training with proper train/test split
#
# The goal is to ensure our data meets ARIMA's assumptions and is properly formatted for training.

# %% [markdown]
# ## 1.1 Import Required Libraries
#
# We'll need several libraries for data manipulation, visualization, and time series analysis:
#
# - **pandas & numpy:** Data manipulation and numerical operations
# - **matplotlib & seaborn:** Visualization
# - **statsmodels:** ARIMA implementation, ADF test, ACF/PACF plots
# - **pmdarima:** Automatic ARIMA parameter selection (optional but recommended)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

# Time series and statistical analysis
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import jarque_bera

# Auto-ARIMA for parameter selection
from pmdarima import auto_arima

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Evaluation metrics function
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(actual, predicted):
    """
    Calculate evaluation metrics for time series forecasting.
    
    Parameters:
    -----------
    actual : pd.Series or np.array
        Actual values
    predicted : pd.Series or np.array
        Predicted values
        
    Returns:
    --------
    dict : Dictionary containing MSE, RMSE, MAE, and MAPE
    """
    # Ensure inputs are aligned if they are pandas Series
    if isinstance(actual, pd.Series) and isinstance(predicted, pd.Series):
        # Align predicted to actual's index (handling gaps in actual)
        predicted = predicted.reindex(actual.index)
    
    # Remove any NaN values (e.g., from alignment mismatches)
    if isinstance(actual, pd.Series):
        valid_mask = ~np.isnan(actual) & ~np.isnan(predicted)
        actual_clean = actual[valid_mask]
        predicted_clean = predicted[valid_mask]
    else:
        actual_clean = actual
        predicted_clean = predicted

    mse = mean_squared_error(actual_clean, predicted_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_clean, predicted_clean)
    
    # MAPE (avoid division by zero)
    mask = actual_clean != 0
    if isinstance(actual_clean, pd.Series):
        mape = np.mean(np.abs((actual_clean[mask] - predicted_clean[mask]) / actual_clean[mask])) * 100
    else:
        mape = np.mean(np.abs((actual_clean[mask] - predicted_clean[mask]) / actual_clean[mask])) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

print("✓ All libraries imported successfully")

# %% [markdown]
# ## 1.2 Setup Paths and Load Data
#
# We'll use the cleaned data from the data processing pipeline. The data is available at three aggregation levels:
# - `data_1m.csv`: 1-minute aggregation
# - `data_5m.csv`: 5-minute aggregation (primary focus)
# - `data_15m.csv`: 15-minute aggregation
#
# We'll focus on the **5-minute aggregation** as it provides a good balance between:
# - **Granularity:** Captures important short-term patterns
# - **Noise reduction:** Smoother than 1-minute data
# - **Computational efficiency:** More manageable than 1-minute data
# - **Forecast horizon:** Suitable for short-term autoscaling decisions

# %%
# Setup project paths
PROJECT_ROOT = Path().resolve().parent.parent
DATA_CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"

print(f"Project root: {PROJECT_ROOT}")
print(f"Cleaned data directory: {DATA_CLEANED_DIR}")

# Load 5-minute aggregated data (primary focus)
data_5m_path = DATA_CLEANED_DIR / 'data_5m.csv'

if data_5m_path.exists():
    df_5m = pd.read_csv(data_5m_path, index_col=0, parse_dates=True)
    print(f"\n✓ Successfully loaded data from: {data_5m_path}")
    print(f"✓ Data shape: {df_5m.shape}")
    print(f"✓ Date range: {df_5m.index.min()} to {df_5m.index.max()}")
else:
    raise FileNotFoundError(f"Data file not found: {data_5m_path}")

# Define system downtime period (Aug 1-3, 1995 due to storm)
# Apply timezone from data to ensure compatibility
data_tz = df_5m.index.tz
downtime_start = pd.Timestamp('1995-08-01').tz_localize(data_tz)
downtime_end = pd.Timestamp('1995-08-03').tz_localize(data_tz)

# %% [markdown]
# ## 1.3 Stationarity Assessment
#
# **Why Stationarity Matters for ARIMA:**
#
# ARIMA models assume that the time series is **stationary**, meaning:
# - **Constant mean:** The average value doesn't change over time
# - **Constant variance:** The spread of values doesn't change over time
# - **Constant autocovariance:** The correlation between values at different time lags doesn't change
#
# If the data is non-stationary, we need to apply transformations (usually differencing) to make it stationary before fitting ARIMA.
#
# We'll assess stationarity using:
# 1. **Visual inspection:** Plot rolling statistics
# 2. **Augmented Dickey-Fuller (ADF) test:** Statistical test for unit root

# %% [markdown]
# ### 1.3.1 Visual Inspection - Rolling Statistics
#
# We'll plot the rolling mean and standard deviation to visually assess stationarity. If these statistics remain relatively constant over time, the series is likely stationary.

# %%
# Calculate rolling statistics
window_size = 24 * 12  # 24 hours * 12 (5-min intervals per hour) = 1 day
rolling_mean = df_5m['requests'].rolling(window=window_size).mean()
rolling_std = df_5m['requests'].rolling(window=window_size).std()

# Plot rolling statistics
fig, ax = plt.subplots(figsize=(16, 6))

# Plot original data
ax.plot(df_5m.index, df_5m['requests'], label='Original Data', 
        color='#1f77b4', linewidth=0.8, alpha=0.7)

# Plot rolling mean
ax.plot(rolling_mean.index, rolling_mean, label=f'Rolling Mean ({window_size} periods)', 
        color='red', linewidth=2, linestyle='--')

# Plot rolling standard deviation
ax.plot(rolling_std.index, rolling_std, label=f'Rolling Std ({window_size} periods)', 
        color='green', linewidth=2, linestyle='--')

# Highlight system downtime
ax.axvspan(downtime_start, downtime_end, color='red', alpha=0.1)

# Formatting
ax.set_title('Rolling Statistics - Stationarity Assessment', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Number of Requests', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 3: Rolling Mean and Standard Deviation**
#
# **Interpretation:**
#
# - **Blue line:** Original request count data
# - **Red dashed line:** Rolling mean (24-hour window)
# - **Green dashed line:** Rolling standard deviation (24-hour window)
#
# **Visual Assessment:**
# - If the rolling mean (red) is relatively flat → Stationary
# - If the rolling mean shows trend → Non-stationary (need differencing)
# - If the rolling std (green) is constant → Homoscedasticity (good)
# - If the rolling std varies → Heteroscedasticity (may need transformation)
#
# The visual inspection provides an initial assessment, but we need a statistical test for confirmation.

# %% [markdown]
# ### 1.3.2 Augmented Dickey-Fuller (ADF) Test
#
# The **ADF test** is a formal statistical test for stationarity. It tests the **null hypothesis** that a unit root is present (i.e., the time series is non-stationary).
#
# **Test Interpretation:**
# - **p-value < 0.05:** Reject null hypothesis → Series is **stationary**
# - **p-value ≥ 0.05:** Fail to reject null hypothesis → Series is **non-stationary**
#
# The test also provides the **ADF statistic**, which is compared to critical values at different confidence levels (1%, 5%, 10%).

# %%
def perform_adf_test(series, title="ADF Test Results"):
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to test
    title : str
        Title for the output
        
    Returns:
    --------
    dict : Test results including p-value, ADF statistic, and critical values
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.6f}')
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if result[1] < 0.05:
        print("✓ p-value < 0.05")
        print("✓ REJECT null hypothesis")
        print("✓ The series is STATIONARY")
        is_stationary = True
    else:
        print("✗ p-value ≥ 0.05")
        print("✗ FAIL TO REJECT null hypothesis")
        print("✗ The series is NON-STATIONARY")
        print("✗ Differencing may be required")
        is_stationary = False
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': is_stationary
    }

# Perform ADF test on original data
adf_results = perform_adf_test(df_5m['requests'], "Augmented Dickey-Fuller Test - Original Data")

# %% [markdown]
# **ADF Test Results Interpretation:**
#
# The ADF test provides statistical evidence about stationarity:
#
# - **ADF Statistic:** More negative values indicate stronger evidence against the null hypothesis
# - **p-value:** Probability of observing the test statistic if the null hypothesis is true
# - **Critical Values:** Threshold values for different confidence levels
#
# **Decision Rule:**
# - If ADF statistic < critical value (or p-value < 0.05) → Stationary
# - If ADF statistic ≥ critical value (or p-value ≥ 0.05) → Non-stationary

# %% [markdown]
# ### 1.3.3 Differencing (if needed)
#
# If the ADF test indicates the series is non-stationary, we need to apply **differencing**. Differencing computes the difference between consecutive observations:
#
# - **First-order differencing:** Δyₜ = yₜ - yₜ₋₁
# - **Second-order differencing:** Δ²yₜ = Δyₜ - Δyₜ₋₁
#
# The parameter **d** in ARIMA(p,d,q) represents the number of differencing operations needed to achieve stationarity.
#
# Let's apply differencing if needed and re-test for stationarity.

# %%
# Check if differencing is needed
if not adf_results['is_stationary']:
    print("\n" + "=" * 60)
    print("APPLYING DIFFERENCING")
    print("=" * 60)
    
    # First-order differencing
    df_5m['requests_diff1'] = df_5m['requests'].diff()
    
    # Plot differenced data
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    # Original data
    axes[0].plot(df_5m.index, df_5m['requests'], color='#1f77b4', linewidth=0.8)
    axes[0].set_title('Original Data', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Requests', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Differenced data
    axes[1].plot(df_5m.index, df_5m['requests_diff1'], color='orange', linewidth=0.8)
    axes[1].set_title('First-Order Differenced Data', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=10)
    axes[1].set_ylabel('Δ Requests', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Perform ADF test on differenced data
    adf_results_diff1 = perform_adf_test(df_5m['requests_diff1'], 
                                          "ADF Test - First-Order Differenced Data")
    
    # Determine the value of d
    if adf_results_diff1['is_stationary']:
        d = 1
        print(f"\n✓ First-order differencing achieved stationarity")
        print(f"✓ Recommended d parameter: {d}")
    else:
        # Try second-order differencing
        df_5m['requests_diff2'] = df_5m['requests_diff1'].diff()
        adf_results_diff2 = perform_adf_test(df_5m['requests_diff2'],
                                              "ADF Test - Second-Order Differenced Data")
        
        if adf_results_diff2['is_stationary']:
            d = 2
            print(f"\n✓ Second-order differencing achieved stationarity")
            print(f"✓ Recommended d parameter: {d}")
        else:
            d = 2  # Default to 2 if still not stationary
            print(f"\n⚠️  Series still non-stationary after second-order differencing")
            print(f"⚠️  Proceeding with d = {d} (may need seasonal differencing)")
else:
    d = 0
    print("\n" + "=" * 60)
    print("NO DIFFERENCING NEEDED")
    print("=" * 60)
    print(f"✓ Series is already stationary")
    print(f"✓ Recommended d parameter: {d}")

# %% [markdown]
# **Differencing Analysis:**
#
# - **If d = 0:** Original series is stationary, no differencing needed
# - **If d = 1:** First-order differencing achieved stationarity
# - **If d = 2:** Second-order differencing was required
#
# The value of **d** will be used in our ARIMA(p,d,q) model. Most time series only require d = 0 or d = 1. Higher values (d ≥ 2) are rare and may indicate the need for seasonal differencing or a different modeling approach.

# %% [markdown]
# ## 1.4 Autocorrelation Analysis (ACF and PACF)
#
# After ensuring stationarity, we analyze the **autocorrelation structure** to determine the AR (p) and MA (q) parameters:
#
# - **ACF (Autocorrelation Function):** Measures correlation between a time series and its lagged values
# - **PACF (Partial Autocorrelation Function):** Measures correlation between a time series and its lagged values, removing the effect of intermediate lags
#
# **Guidelines for Parameter Selection:**
#
# **For ACF:**
# - Sharp cutoff after lag q → MA(q) component
# - Gradual decay → AR component
#
# **For PACF:**
# - Sharp cutoff after lag p → AR(p) component
# - Gradual decay → MA component

# %% [markdown]
# ### 1.4.1 Plot ACF and PACF
#
# We'll plot the ACF and PACF for the (possibly differenced) series to identify patterns that suggest appropriate p and q values.

# %%
# Determine which series to analyze (original or differenced)
if d == 0:
    series_to_analyze = df_5m['requests']
    series_name = "Original Series"
else:
    series_to_analyze = df_5m['requests_diff1'] if d == 1 else df_5m['requests_diff2']
    series_name = f"Differenced Series (d={d})"

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# ACF plot
plot_acf(series_to_analyze.dropna(), lags=50, ax=axes[0], alpha=0.05)
axes[0].set_title(f'Autocorrelation Function (ACF)\n{series_name}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Lag', fontsize=10)
axes[0].set_ylabel('Correlation', fontsize=10)
axes[0].grid(True, alpha=0.3, linestyle='--')

# PACF plot
plot_pacf(series_to_analyze.dropna(), lags=50, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title(f'Partial Autocorrelation Function (PACF)\n{series_name}', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Lag', fontsize=10)
axes[1].set_ylabel('Partial Correlation', fontsize=10)
axes[1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 4: ACF and PACF Plots**
#
# **Left Panel - ACF:**
# - Shows correlation at different lags
# - Blue shaded area represents 95% confidence interval
# - Bars outside the shaded area are statistically significant
#
# **Right Panel - PACF:**
# - Shows partial correlation at different lags
# - Removes influence of intermediate lags
# - Helps identify AR(p) order
#
# **Interpretation Guidelines:**
#
# **ACF Pattern:**
# - **Sharp cutoff after lag q:** Suggests MA(q) model
# - **Gradual decay (exponential or sinusoidal):** Suggests AR model
# - **Sinusoidal pattern:** May indicate seasonality
#
# **PACF Pattern:**
# - **Sharp cutoff after lag p:** Suggests AR(p) model
# - **Gradual decay:** Suggests MA model
#
# **Seasonality Detection:**
# - If ACF shows spikes at regular intervals (e.g., every 12 lags for daily seasonality in 5-min data)
# - Consider SARIMA (Seasonal ARIMA) model

# %% [markdown]
# **ACF and PACF Interpretation:**
#
# **ACF Pattern Analysis:**
#
# The ACF plot shows correlations starting near 1.0 and decaying very slowly over 50+ lags. This slow, linear decay indicates:
#
# - High persistence in the series (shocks take a long time to dissipate)
# - The series exhibits properties often associated with non-stationary data
# - Differencing (d=1) would typically be required to stabilize the mean
#
# **PACF Pattern Analysis:**
#
# The PACF plot shows:
# - A sharp cutoff after Lag 1 (very strong spike)
# - A smaller but visible spike at Lag 2
# - Values drop into the confidence interval (noise region) after Lag 2
#
# This pattern suggests an **AR(1) or AR(2)** process.
#
# **Seasonality Check:**
#
# No obvious recurring spikes at regular intervals (e.g., every 12 or 24 lags). The smooth decay suggests that stochastic seasonality is not a dominant feature.
#
# ---
#
# **Conflict Between Diagnostics:**
#
# The ADF test indicates the series is stationary (p << 0.05), but the ACF shows slow decay typical of non-stationary series. This apparent contradiction occurs because:
#
# **1. ADF Test Result (Stationary):**
# - The series has no unit root (does not wander infinitely like a random walk)
# - The series has a constant mean it eventually returns to
# - The test detects non-stationarity caused specifically by unit roots
#
# **2. ACF Slow Decay (Non-Stationary Appearance):**
# - The series is stationary but has very high persistence
# - A shock to the system takes a long time to die out (e.g., AR coefficient ~0.95-0.98)
# - Slow ACF decay can occur in stationary series with high memory
#
# **Resolution:**
#
# Since the ADF test is highly significant (p << 0.05), the series is statistically stationary. Differencing a series that is already stationary (over-differencing) can harm model performance.
#
# **Recommended Model:**
#
# - **d = 0** (no differencing needed)
# - **p = 1 or 2** (based on PACF cutoff at Lag 1-2)
# - **q = 0** (ACF shows gradual decay, not sharp cutoff)
#
# Starting models: **ARIMA(1,0,0)** or **ARIMA(2,0,0)**
#
# The blue shaded areas in the ACF/PACF plots represent 95% confidence intervals. Bars extending beyond this area are statistically significant.

# %% [markdown]
# ## 1.5 Train/Test Split
#
# For evaluating our ARIMA model's performance, we need to split the data into training and testing sets:
#
# - **Training set:** July 1995 + first 22 days of August 1995
# - **Test set:** Remaining days of August 1995
#
# This split ensures:
# - We train on historical data
# - We test on future data (simulating real-world forecasting)
# - The test period includes the post-outage period (Aug 3 onwards)
#
# **Important:** Time series data must be split chronologically (no random shuffling) to preserve temporal dependencies.

# %%
# Define train/test split dates
# Check if data has timezone and apply it to split dates
data_tz = df_5m.index.tz
# Use a single split point to ensure continuity
# Training ends at the end of Aug 22 (so we split at start of Aug 23)
split_date = pd.Timestamp('1995-08-23').tz_localize(data_tz)
train_end_date = split_date  # For visualization consistency

# Split the data
# Train: Everything strictly before Aug 23 (includes all of Aug 22)
train_data = df_5m[df_5m.index < split_date]['requests']
# Test: Everything from Aug 23 onwards
test_data = df_5m[df_5m.index >= split_date]['requests']

print("=" * 60)
print("TRAIN/TEST SPLIT")
print("=" * 60)
print(f"\nTraining set:")
print(f"  Date range: {train_data.index.min()} to {train_data.index.max()}")
print(f"  Number of observations: {len(train_data)}")
print(f"  Percentage: {len(train_data) / len(df_5m) * 100:.1f}%")
print(f"\nTest set:")
print(f"  Date range: {test_data.index.min()} to {test_data.index.max()}")
print(f"  Number of observations: {len(test_data)}")
print(f"  Percentage: {len(test_data) / len(df_5m) * 100:.1f}%")
print(f"\nTotal observations: {len(df_5m)}")

# Visualize the split
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(train_data.index, train_data, label='Training Data', 
        color='#1f77b4', linewidth=0.8)
ax.plot(test_data.index, test_data, label='Test Data', 
        color='orange', linewidth=0.8)

# Mark the split point
ax.axvline(train_end_date, color='red', linestyle='--', linewidth=2, 
           label='Train/Test Split')

# Highlight system downtime
ax.axvspan(downtime_start, downtime_end, color='red', alpha=0.1)

# Formatting
ax.set_title('Train/Test Split for ARIMA Model', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Number of Requests', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 5: Train/Test Split Visualization**
#
# - **Blue region:** Training data (July + first 22 days of August)
# - **Orange region:** Test data (remaining days of August)
# - **Red dashed line:** Train/test split boundary (August 22, 1995)
# - **Light red shaded area:** Server outage period (August 1-3, 1995)
#
# This split ensures our model is evaluated on truly unseen future data, simulating real-world forecasting scenarios.

# %% [markdown]
# ## 1.6 Section 1 Summary and Conclusions
#
# ### Key Findings from Pre-train Analysis:
#
# **Data Characteristics:**
# - Data shape: 17,856 observations across 17 features
# - Date range: July 1, 1995 - August 31, 1995
# - System downtime: August 1-3, 1995 (due to storm)
# - Time resolution: 5-minute intervals
# - Primary feature: Number of requests (target variable)
#
# **Stationarity Assessment:**
# - ADF test result: p-value = 0.000000 (very highly significant)
# - Series is: Stationary (p-value < 0.05)
# - Differencing required: No
# - Recommended d parameter: 0
#
# **Autocorrelation Analysis:**
# - ACF pattern: Very slow, linear decay over 50+ lags
# - PACF pattern: Sharp cutoff after Lag 1, with smaller spike at Lag 2
# - Seasonality detected: No obvious recurring spikes at regular intervals
# - Recommended p: 1 or 2 (based on PACF cutoff)
# - Recommended q: 0 (ACF shows gradual decay, not sharp cutoff)
#
# **Data Preparation:**
# - Train set: July 1 - August 22, 1995 (15,264 observations)
# - Test set: August 23 - August 31, 1995 (2,592 observations)
# - Train/test split: 85.5% / 14.5%
# - Data ready for ARIMA training: Yes
#
# ### Suitability Assessment:
#
# **Is ARIMA suitable for this data?**
# Yes, ARIMA is suitable because:
# - The series is stationary (no differencing needed)
# - Clear autocorrelation structure exists (AR component)
# - No complex seasonality that would require SARIMA
# - Sufficient data points for reliable parameter estimation
#
# ### Next Steps:
#
# 1. Proceed to Section 2: ARIMA Model Training
# 2. Use auto_arima or grid search to find optimal (p,d,q) parameters
# 3. Fit the ARIMA model on training data
# 4. Evaluate model diagnostics
# 5. Generate forecasts and assess performance in Section 3

# %% [markdown]
# # Section 2: Training
#
# In this section, we will:
# 1. Use automatic parameter selection (auto_arima) to find optimal (p,d,q) parameters
# 2. Fit the ARIMA model on training data
# 3. Perform model diagnostics to validate assumptions
# 4. Compare multiple models if necessary
#
# The goal is to build a well-specified ARIMA model that captures the underlying patterns in the traffic data.

# %% [markdown]
# ## 2.1 Automatic ARIMA Parameter Selection
#
# Instead of manually selecting (p,d,q) parameters based on ACF/PACF analysis, we'll use **auto_arima** from the pmdarima library. This automated approach:
#
# - Searches through a grid of parameter combinations
# - Uses information criteria (AIC, BIC) to select the best model
# - Handles seasonal components automatically (if needed)
# - Is more objective than manual selection
#
# **Parameters for auto_arima:**
# - `start_p`, `start_q`: Starting values for p and q
# - `max_p`, `max_q`: Maximum values to search
# - `d`: Differencing order (from Section 1.3.3)
# - `seasonal`: Whether to include seasonal components
# - `trace`: Print progress during search
# - `stepwise`: Use stepwise algorithm for faster search

# %%
print("=" * 60)
print("AUTOMATIC ARIMA PARAMETER SELECTION")
print("=" * 60)
print(f"\nUsing d = {d} (from stationarity analysis)")
print("Searching for optimal (p, q) parameters...")

# Perform auto_arima search
auto_model = auto_arima(
    train_data,
    start_p=0,
    start_q=0,
    max_p=5,
    max_q=5,
    d=d,
    seasonal=False,  # We'll handle seasonality manually if needed
    trace=True,
    stepwise=True,
    suppress_warnings=True,
    information_criterion='aic'
)

# Extract optimal parameters
p = auto_model.order[0]
d_optimal = auto_model.order[1]
q = auto_model.order[2]

print("\n" + "=" * 60)
print("OPTIMAL PARAMETERS FOUND")
print("=" * 60)
print(f"ARIMA({p}, {d_optimal}, {q})")
print(f"AIC: {auto_model.aic():.2f}")
print(f"BIC: {auto_model.bic():.2f}")
print(f"Training samples: {len(train_data)}")

# %% [markdown]
# **Auto-ARIMA Results:**
#
# The auto_arima function has searched through combinations of (p,d,q) parameters and selected the best model based on the **Akaike Information Criterion (AIC)**.
#
# **Interpretation:**
# - **AIC (Akaike Information Criterion):** Measures model quality while penalizing complexity. Lower AIC = better model.
# - **BIC (Bayesian Information Criterion):** Similar to AIC but with stronger penalty for complexity.
# - **Stepwise search:** Efficiently explores the parameter space without testing every combination.
#
# The selected ARIMA(p,d,q) model balances goodness-of-fit with model complexity, helping to avoid overfitting.

# %% [markdown]
# ## 2.2 Comparison: Manual vs Automatic Parameter Selection
#
# In this section, we'll compare two approaches for selecting ARIMA parameters:
#
# 1. **Manual Selection:** Based on ACF/PACF analysis from Section 1
# 2. **Automatic Selection:** Using auto_arima algorithm
#
# This comparison helps us understand whether the visual/statistical analysis from Section 1 aligns with the automated search, and which approach yields better model performance.

# %% [markdown]
# ### 2.2.1 Manual Parameter Selection (Based on Section 1 Analysis)
#
# From Section 1.4.2, we observed:
#
# **ACF Pattern:**
# - Very slow, linear decay over 50+ lags
# - Indicates high persistence in the series
# - Suggests an **Autoregressive (AR)** component
#
# **PACF Pattern:**
# - Sharp cutoff after Lag 1 (very strong spike)
# - Smaller but still visible spike at Lag 2
# - Values drop into noise region after Lag 2
# - Suggests **AR(1)** or **AR(2)** process
#
# **Stationarity:**
# - ADF test indicated the series is stationary (p << 0.05)
# - No differencing needed (d = 0)
#
# **Recommended Manual Parameters:**
# Based on the ACF/PACF analysis, we'll test two manual models:
# - **Model A:** ARIMA(1, 0, 0) - Simple AR(1) model
# - **Model B:** ARIMA(2, 0, 0) - AR(2) model to capture the secondary spike at Lag 2

# %%
print("=" * 60)
print("MANUAL PARAMETER SELECTION - MODEL FITTING")
print("=" * 60)

# Fit Manual Model A: ARIMA(1, 0, 0)
print("\n--- Manual Model A: ARIMA(1, 0, 0) ---")
manual_model_a = ARIMA(train_data, order=(1, 0, 0))
fitted_manual_a = manual_model_a.fit()

print(f"AIC: {fitted_manual_a.aic:.2f}")
print(f"BIC: {fitted_manual_a.bic:.2f}")
print(f"Log Likelihood: {fitted_manual_a.llf:.2f}")

# Fit Manual Model B: ARIMA(2, 0, 0)
print("\n--- Manual Model B: ARIMA(2, 0, 0) ---")
manual_model_b = ARIMA(train_data, order=(2, 0, 0))
fitted_manual_b = manual_model_b.fit()

print(f"AIC: {fitted_manual_b.aic:.2f}")
print(f"BIC: {fitted_manual_b.bic:.2f}")
print(f"Log Likelihood: {fitted_manual_b.llf:.2f}")

# %% [markdown]
# ### 2.2.2 Automatic Parameter Selection (auto_arima)
#
# The auto_arima function has already been executed in Section 2.1. Let's extract and display the key results for comparison.

# %%
print("=" * 60)
print("AUTOMATIC PARAMETER SELECTION - RESULTS")
print("=" * 60)
print(f"Optimal Model: ARIMA({p}, {d_optimal}, {q})")
print(f"AIC: {auto_model.aic():.2f}")
print(f"BIC: {auto_model.bic():.2f}")

# %% [markdown]
# ### 2.2.3 Comparison Summary
#
# Let's create a comprehensive comparison of all three models (2 manual + 1 automatic).

# %%
print("=" * 70)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 70)
print(f"{'Model':<25} {'Parameters':<15} {'AIC':<12} {'BIC':<12} {'ΔAIC':<10}")
print("-" * 70)

# Calculate AIC differences (relative to best model)
models = {
    'Manual A (ARIMA 1,0,0)': (fitted_manual_a, (1, 0, 0)),
    'Manual B (ARIMA 2,0,0)': (fitted_manual_b, (2, 0, 0)),
    f'Auto (ARIMA {p},{d_optimal},{q})': (auto_model, (p, d_optimal, q))
}

# Helper function to get AIC from different model types
def get_model_aic(model):
    """Get AIC from model, handling both statsmodels and pmdarima."""
    if hasattr(model, 'aic') and callable(model.aic):
        # pmdarima model: aic is a method
        return model.aic()
    else:
        # statsmodels model: aic is a property
        return model.aic

# Find best AIC
best_aic = min(get_model_aic(model) for model, _ in models.values())

for model_name, (model, params) in models.items():
    aic = get_model_aic(model)
    bic = get_model_aic(model) if hasattr(model, 'bic') and callable(model.bic) else model.bic
    delta_aic = aic - best_aic
    params_str = f"({params[0]}, {params[1]}, {params[2]})"
    print(f"{model_name:<25} {params_str:<15} {aic:<12.2f} {bic:<12.2f} {delta_aic:<10.2f}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("• Lower AIC/BIC indicates better model (balance of fit and complexity)")
print("• ΔAIC < 2: Substantial evidence for model")
print("• ΔAIC 4-7: Less support for model")
print("• ΔAIC > 10: Essentially no support for model")
print("\n• Best model has ΔAIC = 0.0 (reference point)")

# %% [markdown]
# **Model Comparison Interpretation:**
#
# **Key Observations:**
#
# 1. **Manual vs Automatic Alignment:**
#    - If the auto_arima selected model matches or is close to our manual selection, it validates our ACF/PACF analysis
#    - If there's a significant difference, we need to understand why
#
# 2. **Information Criteria (AIC/BIC):**
#    - **AIC** favors more complex models (better fit, potentially overfitting)
#    - **BIC** penalizes complexity more heavily (simpler models preferred)
#    - Lower values are better for both
#
# 3. **ΔAIC (AIC Difference):**
#    - ΔAIC < 2: Models are essentially equivalent
#    - ΔAIC 4-7: Considerably less support
#    - ΔAIC > 10: Essentially no support
#
# 4. **Model Selection Strategy:**
#    - If multiple models have similar AIC (ΔAIC < 2), choose the simplest (Occam's razor)
#    - If one model has significantly lower AIC, it's the preferred choice
#    - Consider both AIC and BIC for robustness

# %% [markdown]
# ### 2.2.4 Detailed Model Comparison
#
# Let's examine the detailed characteristics of each model, including parameter estimates and their statistical significance.

# %%
print("=" * 70)
print("DETAILED MODEL PARAMETERS AND SIGNIFICANCE")
print("=" * 70)

# Manual Model A details
print("\n--- Manual Model A: ARIMA(1, 0, 0) ---")
print(fitted_manual_a.summary().tables[1])

# Manual Model B details
print("\n--- Manual Model B: ARIMA(2, 0, 0) ---")
print(fitted_manual_b.summary().tables[1])

# Auto Model details
print(f"\n--- Auto Model: ARIMA({p}, {d_optimal}, {q}) ---")
print(auto_model.summary().tables[1])

# %% [markdown]
# **Parameter Significance Analysis:**
#
# For each model, we examine:
#
# - **coef:** Estimated parameter value
# - **std err:** Standard error of the estimate
# - **z:** z-statistic (coef / std err)
# - **P>|z|:** p-value for testing if coefficient is zero
#
# **Interpretation:**
# - **P < 0.05:** Coefficient is statistically significant
# - **P ≥ 0.05:** Coefficient may not be needed (model could be simplified)
# - **Large |z|:** Strong evidence for the parameter's importance
#
# **Key Questions:**
# 1. Are all parameters in the auto_arima model significant?
# 2. Do the manual models have significant parameters?
# 3. Are there any redundant parameters (high p-values)?
# 4. Which model has the most parsimonious (efficient) parameter set?

# %% [markdown]
# ### 2.2.5 In-Sample Performance Comparison
#
# Let's compare the in-sample fit quality of all three models using multiple metrics.

# %%
# Get fitted values for all models
fitted_auto = auto_model.fittedvalues()
fitted_manual_a_values = fitted_manual_a.fittedvalues
fitted_manual_b_values = fitted_manual_b.fittedvalues

# Calculate metrics for all models
metrics_auto = calculate_metrics(train_data[d_optimal:], fitted_auto)
metrics_manual_a = calculate_metrics(train_data, fitted_manual_a_values)
metrics_manual_b = calculate_metrics(train_data, fitted_manual_b_values)

print("=" * 70)
print("IN-SAMPLE PERFORMANCE COMPARISON")
print("=" * 70)
print(f"{'Model':<25} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
print("-" * 70)
print(f"{'Manual A (1,0,0)':<25} {metrics_manual_a['RMSE']:<12.2f} {metrics_manual_a['MAE']:<12.2f} {metrics_manual_a['MAPE']:<12.2f}%")
print(f"{'Manual B (2,0,0)':<25} {metrics_manual_b['RMSE']:<12.2f} {metrics_manual_b['MAE']:<12.2f} {metrics_manual_b['MAPE']:<12.2f}%")
print(f"{'Auto (ARIMA)':<25} {metrics_auto['RMSE']:<12.2f} {metrics_auto['MAE']:<12.2f} {metrics_auto['MAPE']:<12.2f}%")

print("\n" + "=" * 70)
print("RELATIVE PERFORMANCE (vs Best Model)")
print("=" * 70)

# Find best model for each metric
best_rmse = min(metrics_auto['RMSE'], metrics_manual_a['RMSE'], metrics_manual_b['RMSE'])
best_mae = min(metrics_auto['MAE'], metrics_manual_a['MAE'], metrics_manual_b['MAE'])
best_mape = min(metrics_auto['MAPE'], metrics_manual_a['MAPE'], metrics_manual_b['MAPE'])

models_metrics = {
    'Manual A (1,0,0)': metrics_manual_a,
    'Manual B (2,0,0)': metrics_manual_b,
    'Auto (ARIMA)': metrics_auto
}

for model_name, metrics in models_metrics.items():
    rmse_diff = ((metrics['RMSE'] - best_rmse) / best_rmse) * 100
    mae_diff = ((metrics['MAE'] - best_mae) / best_mae) * 100
    mape_diff = ((metrics['MAPE'] - best_mape) / best_mape) * 100
    print(f"\n{model_name}:")
    print(f"  RMSE: {rmse_diff:+.2f}% from best")
    print(f"  MAE:  {mae_diff:+.2f}% from best")
    print(f"  MAPE: {mape_diff:+.2f}% from best")

# %% [markdown]
# **In-Sample Performance Analysis:**
#
# **Important Note:** In-sample performance is not the primary criterion for model selection because:
# - All models will fit the training data reasonably well
# - More complex models may have better in-sample fit but worse out-of-sample performance (overfitting)
# - The true test is how well the model generalizes to unseen data
#
# **However, in-sample comparison is useful for:**
# - Understanding how each model captures the training data patterns
# - Identifying potential overfitting (very low errors but poor generalization)
# - Validating that all models are reasonably specified

# %% [markdown]
# ### 2.2.6 Visual Comparison of Model Fits
#
# Let's visualize how each model fits the training data to see the differences in their behavior.

# %%
# Create comparison plot
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# Plot 1: Manual Model A
axes[0].plot(train_data.index, train_data, label='Actual Data',
             color='#1f77b4', linewidth=0.8, alpha=0.7)
axes[0].plot(fitted_manual_a_values.index, fitted_manual_a_values, 
             label='Fitted - Manual A (1,0,0)', color='red', linewidth=1.5, alpha=0.9)
axes[0].set_title('Manual Model A: ARIMA(1, 0, 0) - In-Sample Fit', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Date', fontsize=10)
axes[0].set_ylabel('Requests', fontsize=10)
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Plot 2: Manual Model B
axes[1].plot(train_data.index, train_data, label='Actual Data',
             color='#1f77b4', linewidth=0.8, alpha=0.7)
axes[1].plot(fitted_manual_b_values.index, fitted_manual_b_values, 
             label='Fitted - Manual B (2,0,0)', color='green', linewidth=1.5, alpha=0.9)
axes[1].set_title('Manual Model B: ARIMA(2, 0, 0) - In-Sample Fit', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=10)
axes[1].set_ylabel('Requests', fontsize=10)
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

# Plot 3: Auto Model
axes[2].plot(train_data.index, train_data, label='Actual Data',
             color='#1f77b4', linewidth=0.8, alpha=0.7)
axes[2].plot(fitted_auto.index, fitted_auto, 
             label=f'Fitted - Auto ({p},{d_optimal},{q})', color='orange', linewidth=1.5, alpha=0.9)
axes[2].set_title(f'Auto Model: ARIMA({p}, {d_optimal}, {q}) - In-Sample Fit', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Date', fontsize=10)
axes[2].set_ylabel('Requests', fontsize=10)
axes[2].legend(loc='upper left')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure X: Visual Comparison of Model Fits**
#
# This comparison shows how each model captures the training data patterns:
#
# - **Panel 1 (Manual A):** Simple AR(1) model - smooth, lagged response
# - **Panel 2 (Manual B):** AR(2) model - can capture more complex dynamics
# - **Panel 3 (Auto):** Automatically selected model - optimized for AIC
#
# **Visual Assessment:**
# - Look for how closely each fitted line follows the actual data
# - Note any systematic deviations or lag in the fitted values
# - Compare smoothness vs responsiveness of each model
# - Identify if any model overfits (follows noise too closely)

# %% [markdown]
# ### 2.2.7 Residual Diagnostics Comparison
#
# Let's compare the residual diagnostics across all three models to assess which one best satisfies ARIMA assumptions.

# %%
print("=" * 70)
print("RESIDUAL DIAGNOSTICS COMPARISON")
print("=" * 70)

# Calculate residuals for all models
residuals_auto = auto_model.resid()
residuals_manual_a = fitted_manual_a.resid
residuals_manual_b = fitted_manual_b.resid

# Function to perform diagnostic tests
def diagnostic_tests(residuals, model_name):
    """Perform statistical tests on residuals."""
    results = {}
    
    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    results['lb_pvalue'] = lb_test['lb_pvalue'].iloc[0]
    
    # Jarque-Bera test
    jb_stat, jb_pvalue = jarque_bera(residuals)
    results['jb_pvalue'] = jb_pvalue
    
    # Durbin-Watson test
    dw_stat = durbin_watson(residuals)
    results['dw_stat'] = dw_stat
    
    return results

# Perform tests for all models
diagnostics_auto = diagnostic_tests(residuals_auto, "Auto")
diagnostics_manual_a = diagnostic_tests(residuals_manual_a, "Manual A")
diagnostics_manual_b = diagnostic_tests(residuals_manual_b, "Manual B")

print(f"\n{'Model':<25} {'Ljung-Box p':<15} {'Jarque-Bera p':<15} {'Durbin-Watson':<15}")
print("-" * 70)
print(f"{'Manual A (1,0,0)':<25} {diagnostics_manual_a['lb_pvalue']:<15.6f} {diagnostics_manual_a['jb_pvalue']:<15.6f} {diagnostics_manual_a['dw_stat']:<15.4f}")
print(f"{'Manual B (2,0,0)':<25} {diagnostics_manual_b['lb_pvalue']:<15.6f} {diagnostics_manual_b['jb_pvalue']:<15.6f} {diagnostics_manual_b['dw_stat']:<15.4f}")
print(f"{'Auto (ARIMA)':<25} {diagnostics_auto['lb_pvalue']:<15.6f} {diagnostics_auto['jb_pvalue']:<15.6f} {diagnostics_auto['dw_stat']:<15.4f}")

print("\n" + "=" * 70)
print("DIAGNOSTIC INTERPRETATION")
print("=" * 70)
print("• Ljung-Box p > 0.05: Residuals are uncorrelated (GOOD)")
print("• Jarque-Bera p > 0.05: Residuals are normally distributed (GOOD)")
print("• Durbin-Watson 1.5-2.5: No significant autocorrelation (GOOD)")

# %% [markdown]
# **Residual Diagnostics Comparison:**
#
# **Ideal Model Characteristics:**
# 1. **Ljung-Box p-value > 0.05:** Residuals are white noise (no autocorrelation)
# 2. **Jarque-Bera p-value > 0.05:** Residuals are normally distributed
# 3. **Durbin-Watson statistic 1.5-2.5:** No first-order autocorrelation
#
# **Model Assessment:**
# - **Manual A:** Simple model, may have larger residuals but potentially better diagnostics
# - **Manual B:** More complex, may fit better but risk of overfitting
# - **Auto:** Optimized for AIC, should balance fit and complexity
#
# **Trade-offs:**
# - Better fit (lower RMSE) may come at the cost of worse diagnostics
# - Simpler models often have better residual properties but may underfit
# - The "best" model balances fit quality with diagnostic validity

# %% [markdown]
# ### 2.2.8 Final Model Selection
#
# Based on our comprehensive comparison, let's make a final decision on which model to use for forecasting.

# %%
print("=" * 70)
print("FINAL MODEL SELECTION DECISION")
print("=" * 70)

# Create summary table
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Criterion':<20} {'Manual A':<15} {'Manual B':<15} {'Auto':<15}")
print("-" * 70)

# AIC comparison
print(f"{'AIC':<20} {fitted_manual_a.aic:<15.2f} {fitted_manual_b.aic:<15.2f} {get_model_aic(auto_model):<15.2f}")

# BIC comparison
print(f"{'BIC':<20} {fitted_manual_a.bic:<15.2f} {fitted_manual_b.bic:<15.2f} {get_model_aic(auto_model) if hasattr(auto_model, 'bic') and callable(auto_model.bic) else auto_model.bic:<15.2f}")

# RMSE comparison
print(f"{'RMSE (In-Sample)':<20} {metrics_manual_a['RMSE']:<15.2f} {metrics_manual_b['RMSE']:<15.2f} {metrics_auto['RMSE']:<15.2f}")

# Ljung-Box comparison
lb_status_a = "✓" if diagnostics_manual_a['lb_pvalue'] > 0.05 else "✗"
lb_status_b = "✓" if diagnostics_manual_b['lb_pvalue'] > 0.05 else "✗"
lb_status_auto = "✓" if diagnostics_auto['lb_pvalue'] > 0.05 else "✗"
print(f"{'Ljung-Box OK':<20} {lb_status_a:<15} {lb_status_b:<15} {lb_status_auto:<15}")

# Complexity (number of parameters)
complexity_a = 1  # AR(1)
complexity_b = 2  # AR(2)
complexity_auto = p + q  # AR(p) + MA(q)
print(f"{'Parameters':<20} {complexity_a:<15} {complexity_b:<15} {complexity_auto:<15}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

# Determine best model based on AIC (primary criterion)
best_model_name = ""
best_model = None
best_order = None

if fitted_manual_a.aic <= fitted_manual_b.aic and fitted_manual_a.aic <= get_model_aic(auto_model):
    best_model_name = "Manual Model A: ARIMA(1, 0, 0)"
    best_model = fitted_manual_a
    best_order = (1, 0, 0)
elif fitted_manual_b.aic <= get_model_aic(auto_model):
    best_model_name = "Manual Model B: ARIMA(2, 0, 0)"
    best_model = fitted_manual_b
    best_order = (2, 0, 0)
else:
    best_model_name = f"Auto Model: ARIMA({p}, {d_optimal}, {q})"
    best_model = auto_model
    best_order = (p, d_optimal, q)

print(f"\nSelected Model: {best_model_name}")
print(f"\nRationale:")
print("• Primary criterion: Lowest AIC (balance of fit and complexity)")
print("• Secondary criteria: BIC, residual diagnostics, model simplicity")
print("• The selected model provides the best trade-off between accuracy and complexity")

# Update the main model to use the selected one
if best_model_name.startswith("Manual"):
    if best_model_name.startswith("Manual A"):
        p, d_optimal, q = 1, 0, 0
    else:
        p, d_optimal, q = 2, 0, 0
    fitted_model = best_model

print(f"\nProceeding with ARIMA({p}, {d_optimal}, {q}) for forecasting in Section 3")

# %% [markdown]
# **Final Model Selection:**
#
# **Decision Process:**
#
# 1. **Primary Criterion:** AIC (Akaike Information Criterion)
#    - Balances model fit with complexity
#    - Lower is better
#    - Widely used for model selection in time series
#
# 2. **Secondary Criteria:**
#    - BIC: Penalizes complexity more heavily
#    - Residual diagnostics: Validates model assumptions
#    - Model simplicity: Occam's razor (prefer simpler models)
#
# 3. **Practical Considerations:**
#    - Computational efficiency (simpler models are faster)
#    - Interpretability (fewer parameters are easier to explain)
#    - Robustness (simpler models are less prone to overfitting)
#
# **Key Insights from Comparison:**
#
# **Manual vs Automatic Alignment:**
# - If auto_arima selected parameters close to our manual analysis, it validates the ACF/PACF interpretation
# - If there's a significant difference, it may indicate:
#   - Complex patterns not easily visible in ACF/PACF
#   - The automated search found a better local minimum
#   - Our manual interpretation may have missed important interactions
#
# **Model Complexity vs Performance:**
# - More parameters don't always mean better forecasts
# - Overfitting risk increases with complexity
# - The best model optimally balances fit and simplicity
#
# **Recommendation for Production:**
# - Use the selected model for forecasting
# - Monitor performance on new data
# - Consider retraining periodically with updated data
# - Maintain the comparison framework for future model updates

# %% [markdown]
# ## 2.3 Model Fitting
#
# Now we'll fit the ARIMA model using the optimal parameters found by auto_arima. We'll use the statsmodels ARIMA implementation, which provides:
#
# - Maximum likelihood estimation of parameters
# - Comprehensive diagnostic information
# - Forecasting capabilities
# - Residual analysis tools

# %%
print("=" * 60)
print("FITTING ARIMA MODEL")
print("=" * 60)
print(f"Model: ARIMA({p}, {d_optimal}, {q})")
print(f"Training data: {len(train_data)} observations")
print(f"Date range: {train_data.index.min()} to {train_data.index.max()}")

# Fit ARIMA model using statsmodels
model = ARIMA(train_data, order=(p, d_optimal, q))
fitted_model = model.fit()

# Print model summary
print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
print(fitted_model.summary())

# Extract model parameters
ar_params = fitted_model.arparams if hasattr(fitted_model, 'arparams') else []
ma_params = fitted_model.maparams if hasattr(fitted_model, 'maparams') else []

print("\n" + "=" * 60)
print("MODEL PARAMETERS")
print("=" * 60)
if len(ar_params) > 0:
    print(f"AR coefficients: {ar_params}")
else:
    print("AR coefficients: None (p=0)")
if len(ma_params) > 0:
    print(f"MA coefficients: {ma_params}")
else:
    print("MA coefficients: None (q=0)")

# %% [markdown]
# **Model Summary Interpretation:**
#
# The model summary provides detailed information about the fitted ARIMA model:
#
# **Key Components:**
# - **coef:** Estimated coefficients for AR and MA terms
# - **std err:** Standard error of coefficient estimates
# - **z:** z-statistic (coef / std err)
# - **P>|z|:** p-value for testing if coefficient is zero
# - **[0.025 0.975]:** 95% confidence interval for coefficients
#
# **Model Diagnostics:**
# - **Ljung-Box (L1):** Test for residual autocorrelation (p > 0.05 is good)
# - **Jarque-Bera (JB):** Test for normality of residuals (p > 0.05 is good)
# - **Heteroskedasticity (H):** Test for constant variance (p > 0.05 is good)
# - **AIC/BIC:** Information criteria for model comparison
#
# **Coefficient Significance:**
# - Coefficients with p-value < 0.05 are statistically significant
# - Non-significant coefficients suggest the model may be over-parameterized

# %% [markdown]
# ## 2.3.1 Save Trained Model
#
# We'll save the trained ARIMA model to the `models/` directory for later use in:
# - The REST API for serving predictions
# - The dashboard for visualization
# - Future model comparisons and ensemble methods
#
# The model is saved using `joblib`, which is efficient for scikit-learn compatible objects.

# %%
# Save the trained ARIMA model
import joblib
import os

MODELS_DIR = PROJECT_ROOT / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Create descriptive model filename
model_filename = f"arima_{p}_{d_optimal}_{q}_5m.pkl"
model_path = MODELS_DIR / model_filename

# Save the model
joblib.dump(fitted_model, model_path)

# Verify save and print confirmation
file_size = os.path.getsize(model_path) / 1024  # Size in KB
print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)
print(f"Model: ARIMA({p}, {d_optimal}, {q})")
print(f"File: {model_path}")
print(f"Size: {file_size:.2f} KB")
print(f"\nTo load the model later:")
print(f"  loaded_model = joblib.load('{model_path}')")

# %% [markdown]
# ## 2.4 Model Diagnostics
#
# After fitting the model, we need to validate that it meets ARIMA assumptions:
#
# 1. **Residuals are white noise:** No autocorrelation in residuals
# 2. **Residuals are normally distributed:** For valid confidence intervals
# 3. **Residuals have constant variance:** Homoscedasticity
# 4. **No patterns in residuals:** Model has captured all systematic patterns
#
# We'll use visual and statistical tests to assess these assumptions.

# %%
# Calculate residuals
residuals = fitted_model.resid

print("=" * 60)
print("RESIDUAL ANALYSIS")
print("=" * 60)
print(f"Mean: {residuals.mean():.4f}")
print(f"Std Dev: {residuals.std():.4f}")
print(f"Min: {residuals.min():.4f}")
print(f"Max: {residuals.max():.4f}")

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Residuals time series
axes[0, 0].plot(train_data.index[1:], residuals[1:], color='#1f77b4', linewidth=0.8)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Date', fontsize=10)
axes[0, 0].set_ylabel('Residual', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals histogram
axes[0, 1].hist(residuals, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.2f}')
axes[0, 1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Residual', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Q-Q plot for normality
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot - Normality Test', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. ACF of residuals
plot_acf(residuals, lags=50, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('ACF of Residuals', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Lag', fontsize=10)
axes[1, 1].set_ylabel('Correlation', fontsize=10)
axes[1, 1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 6: Model Diagnostics**
#
# **Panel 1 - Residuals Over Time:**
# - Shows residuals throughout the training period
# - Should fluctuate randomly around zero (no patterns)
# - Red dashed line at zero for reference
#
# **Panel 2 - Residuals Distribution:**
# - Histogram of residual values
# - Should be approximately normal (bell-shaped)
# - Mean should be close to zero
#
# **Panel 3 - Q-Q Plot:**
# - Quantile-Quantile plot for normality
# - Points should fall along the red diagonal line
# - Deviations indicate non-normality
#
# **Panel 4 - ACF of Residuals:**
# - Autocorrelation of residuals
# - Should show no significant correlations (all bars within blue shaded area)
# - Significant correlations indicate model hasn't captured all patterns

# %% [markdown]
# ### 2.4.1 Statistical Tests for Residuals
#
# We'll perform formal statistical tests to validate model assumptions:

# %%
print("=" * 60)
print("STATISTICAL TESTS FOR RESIDUALS")
print("=" * 60)

# 1. Ljung-Box test for autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
lb_pvalue = lb_test['lb_pvalue'].iloc[0]

print("\n1. Ljung-Box Test (Autocorrelation)")
print(f"   p-value: {lb_pvalue:.6f}")
if lb_pvalue > 0.05:
    print("   ✓ p-value > 0.05: Residuals are uncorrelated (GOOD)")
else:
    print("   ✗ p-value ≤ 0.05: Residuals show autocorrelation (BAD)")

# 2. Jarque-Bera test for normality
from scipy.stats import jarque_bera
jb_stat, jb_pvalue = jarque_bera(residuals)

print("\n2. Jarque-Bera Test (Normality)")
print(f"   Statistic: {jb_stat:.4f}")
print(f"   p-value: {jb_pvalue:.6f}")
if jb_pvalue > 0.05:
    print("   ✓ p-value > 0.05: Residuals are normally distributed (GOOD)")
else:
    print("   ✗ p-value ≤ 0.05: Residuals are not normally distributed (BAD)")

# 3. Shapiro-Wilk test for normality (smaller sample)
if len(residuals) < 5000:
    from scipy.stats import shapiro
    sw_stat, sw_pvalue = shapiro(residuals)
    print("\n3. Shapiro-Wilk Test (Normality)")
    print(f"   Statistic: {sw_stat:.4f}")
    print(f"   p-value: {sw_pvalue:.6f}")
    if sw_pvalue > 0.05:
        print("   ✓ p-value > 0.05: Residuals are normally distributed (GOOD)")
    else:
        print("   ✗ p-value ≤ 0.05: Residuals are not normally distributed (BAD)")

# 4. Durbin-Watson test for autocorrelation
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(residuals)

print("\n4. Durbin-Watson Test (Autocorrelation)")
print(f"   Statistic: {dw_stat:.4f}")
if 1.5 < dw_stat < 2.5:
    print("   ✓ 1.5 < DW < 2.5: No significant autocorrelation (GOOD)")
else:
    print("   ✗ DW outside [1.5, 2.5]: Possible autocorrelation (BAD)")

print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)
print("The model is considered adequate if:")
print("  - Residuals are uncorrelated (Ljung-Box p > 0.05)")
print("  - Residuals are approximately normal (JB p > 0.05)")
print("  - No patterns in residual plots")
print("  - ACF of residuals shows no significant lags")

# %% [markdown]
# **Statistical Test Interpretation:**
#
# **Ljung-Box Test:**
# - Tests whether residuals are white noise (no autocorrelation)
# - Null hypothesis: Residuals are uncorrelated
# - p-value > 0.05: Cannot reject null → Good model fit
#
# **Jarque-Bera Test:**
# - Tests whether residuals are normally distributed
# - Null hypothesis: Residuals are normally distributed
# - p-value > 0.05: Cannot reject null → Normality assumption holds
#
# **Durbin-Watson Test:**
# - Tests for first-order autocorrelation
# - Statistic close to 2.0: No autocorrelation
# - Values < 1.5 or > 2.5: Possible autocorrelation
#
# **Note:** Some deviation from normality is common in real-world data and doesn't necessarily invalidate the model, especially if other diagnostics are good.

# %% [markdown]
# ## 2.5 Section 2 Summary and Conclusions
#
# ### Model Training Results:
#
# **Parameter Selection:**
# - Optimal ARIMA model: ARIMA(4, 0, 1)
# - Selection method: auto_arima with AIC criterion
# - Model complexity: Moderate (4 AR terms + 1 MA term)
# - Manual vs Auto alignment: Auto_arima selected ARIMA(4,0,1), which differs from manual ARIMA(1,0,0) and ARIMA(2,0,0)
#
# **Model Fit:**
# - AIC: 162,309.65
# - BIC: 162,363.08
# - Training samples: 15,264 observations
# - Log Likelihood: -81,147.83
#
# **Model Comparison:**
# - Manual A (ARIMA 1,0,0): AIC=165,668.50, BIC=165,691.40
# - Manual B (ARIMA 2,0,0): AIC=164,006.37, BIC=164,036.90
# - Auto (ARIMA 4,0,1): AIC=162,310.15, BIC=162,363.58
# - Best model: Auto ARIMA(4,0,1) by AIC (lowest)
#
# **Model Diagnostics:**
# - Residual autocorrelation: Absent (Ljung-Box p-value = 0.450036 > 0.05, GOOD)
# - Residual normality: Not normal (Jarque-Bera p-value = 0.000000 < 0.05, BAD)
# - Durbin-Watson: 1.9998 (no autocorrelation, GOOD)
# - Residual patterns: Residuals are uncorrelated but not normally distributed
# - Overall model adequacy: Adequate for forecasting, though normality assumption is violated
#
# ### Next Steps:
#
# 1. Proceed to Section 3: Post-train Evaluation
# 2. Generate forecasts on test data
# 3. Calculate out-of-sample performance metrics
# 4. Compare actual vs predicted values
# 5. Analyze forecast errors and residuals
# 6. Draw final conclusions and recommendations

# %% [markdown]
# # Section 3: Post-train Evaluation
#
# In this section, we will:
# 1. Generate forecasts on the test set (unseen data)
# 2. Calculate out-of-sample performance metrics (RMSE, MSE, MAE, MAPE)
# 3. Visualize actual vs predicted values
# 4. Analyze forecast errors and residuals
# 5. Compare performance across different time periods
# 6. Draw conclusions and provide recommendations
#
# The goal is to assess how well the ARIMA model generalizes to new data and whether it's suitable for production use.

# %% [markdown]
# ## 3.1 Forecast Generation
#
# We'll generate forecasts for the test period using the fitted ARIMA model. The forecast horizon is the length of the test set.

# %%
print("=" * 60)
print("FORECAST GENERATION")
print("=" * 60)
print(f"Model: ARIMA({p}, {d_optimal}, {q})")
print(f"Test set size: {len(test_data)} observations")
print(f"Forecast horizon: {len(test_data)} periods")

# Generate forecasts
forecast_result = fitted_model.get_forecast(steps=len(test_data))
forecast_values = forecast_result.predicted_mean
forecast_conf_int = forecast_result.conf_int()

print(f"\n✓ Forecast generated successfully")
print(f"✓ Forecast period: {test_data.index.min()} to {test_data.index.max()}")
print(f"✓ Confidence intervals calculated (95%)")

# Display first few forecasts
print("\n" + "=" * 60)
print("FORECAST SAMPLE (First 5 values)")
print("=" * 60)
print(forecast_values.head().to_string())

# %% [markdown]
# **Forecast Generation:**
#
# The model has generated point forecasts and 95% confidence intervals for the entire test period. Wider confidence intervals indicate higher uncertainty in predictions.
#
# **Key Points:**
# - **Point forecasts:** Expected value at each time step
# - **Confidence intervals:** Range of likely values (95% probability)
# - **Forecast horizon:** Length of test set ({len(test_data)} periods)

# %% [markdown]
# ## 3.2 Performance Metrics Calculation
#
# We'll calculate the four required evaluation metrics to assess forecast accuracy:
#
# 1. **RMSE (Root Mean Squared Error):** Penalizes large errors heavily
# 2. **MSE (Mean Squared Error):** Average of squared errors
# 3. **MAE (Mean Absolute Error):** Average absolute error (robust to outliers)
# 4. **MAPE (Mean Absolute Percentage Error):** Scale-independent percentage error

# %%
# Calculate out-of-sample metrics
out_of_sample_metrics = calculate_metrics(test_data, forecast_values)

print("=" * 60)
print("OUT-OF-SAMPLE PERFORMANCE (Test Data)")
print("=" * 60)
print(f"MSE:  {out_of_sample_metrics['MSE']:.2f}")
print(f"RMSE: {out_of_sample_metrics['RMSE']:.2f}")
print(f"MAE:  {out_of_sample_metrics['MAE']:.2f}")
print(f"MAPE: {out_of_sample_metrics['MAPE']:.2f}%")

# %% [markdown]
# **Performance Metrics Interpretation:**
#
# **Metric Definitions:**
# - **MSE:** Mean of (actual - predicted)² - heavily penalizes large errors
# - **RMSE:** √MSE - same units as the data, easier to interpret
# - **MAE:** Mean of |actual - predicted| - robust to outliers
# - **MAPE:** Mean of |(actual - predicted)/actual| × 100 - scale-independent

# %% [markdown]
# ## 3.3 Visualization of Forecasts
#
# Visual inspection helps assess forecast quality. Multiple visualizations will be created to evaluate different aspects of the forecasts.

# %%
# Create comprehensive forecast visualization
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# Plot 1: Full forecast period
axes[0].plot(train_data.index, train_data, label='Training Data',
             color='#1f77b4', linewidth=0.8, alpha=0.7)
axes[0].plot(test_data.index, test_data, label='Actual Test Data',
             color='#2ca02c', linewidth=0.8)
axes[0].plot(forecast_values.index, forecast_values, label='Forecast',
             color='red', linewidth=1.5, linestyle='--')

# Add confidence intervals
axes[0].fill_between(forecast_values.index,
                     forecast_conf_int.iloc[:, 0],
                     forecast_conf_int.iloc[:, 1],
                     color='red', alpha=0.2, label='95% Confidence Interval')

# Mark the split point
axes[0].axvline(train_end_date, color='black', linestyle='-', linewidth=2,
                label='Train/Test Split')

# Highlight system downtime
axes[0].axvspan(downtime_start, downtime_end, color='red', alpha=0.1)

# Formatting
axes[0].set_title('ARIMA Forecast: Full Test Period', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Number of Requests', fontsize=12)
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Plot 2: First week of forecast (zoom in)
test_tz = test_data.index.tz
first_week_end = split_date + pd.Timedelta(days=7)
first_week_mask = test_data.index <= first_week_end

axes[1].plot(train_data.index[train_data.index >= (first_week_end - pd.Timedelta(days=7))],
             train_data[train_data.index >= (first_week_end - pd.Timedelta(days=7))],
             label='Training Data', color='#1f77b4', linewidth=0.8, alpha=0.7)
axes[1].plot(test_data.index[first_week_mask], test_data[first_week_mask],
             label='Actual Test Data', color='#2ca02c', linewidth=0.8)
axes[1].plot(forecast_values.index[first_week_mask], forecast_values[first_week_mask],
             label='Forecast', color='red', linewidth=1.5, linestyle='--')
axes[1].fill_between(forecast_values.index[first_week_mask],
                     forecast_conf_int.iloc[:, 0][first_week_mask],
                     forecast_conf_int.iloc[:, 1][first_week_mask],
                     color='red', alpha=0.2, label='95% Confidence Interval')
axes[1].axvline(train_end_date, color='black', linestyle='-', linewidth=2)

axes[1].set_title('ARIMA Forecast: First Week (Zoom In)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Number of Requests', fontsize=12)
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

# Plot 3: Forecast errors
forecast_errors = test_data - forecast_values

axes[2].plot(forecast_errors.index, forecast_errors, color='purple', linewidth=0.8)
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[2].axhline(y=out_of_sample_metrics['MAE'], color='red', linestyle='--',
                linewidth=1, label=f'MAE: {out_of_sample_metrics["MAE"]:.2f}')
axes[2].axhline(y=-out_of_sample_metrics['MAE'], color='red', linestyle='--', linewidth=1)

axes[2].set_title('Forecast Errors (Actual - Predicted)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date', fontsize=12)
axes[2].set_ylabel('Error', fontsize=12)
axes[2].legend(loc='upper left')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 8: Comprehensive Forecast Visualization**
#
# **Panel 1 - Full Test Period:**
# - Blue: Training data
# - Green: Actual test data
# - Red dashed: Forecast
# - Red shaded: 95% confidence interval
# - Black line: Train/test split point
#
# **Panel 2 - First Week (Zoom):**
# - Closer view of initial forecast period
# - Shows short-term forecast accuracy
# - Confidence intervals typically narrower for short-term forecasts
#
# **Panel 3 - Forecast Errors:**
# - Purple line: Actual - Predicted at each time step
# - Black dashed line: Zero error (perfect forecast)
# - Red dashed lines: ±MAE bounds
# - Errors should be randomly distributed around zero

# %% [markdown]
# ## 3.4 Residual Analysis for Forecasts
#
# Analyzing forecast residuals (errors) helps identify systematic patterns and potential improvements.

# %%
# Calculate forecast residuals
forecast_residuals = test_data - forecast_values

print("=" * 60)
print("FORECAST RESIDUAL STATISTICS")
print("=" * 60)
print(f"Mean: {forecast_residuals.mean():.4f}")
print(f"Std Dev: {forecast_residuals.std():.4f}")
print(f"Min: {forecast_residuals.min():.4f}")
print(f"Max: {forecast_residuals.max():.4f}")
print(f"Skewness: {forecast_residuals.skew():.4f}")
print(f"Kurtosis: {forecast_residuals.kurtosis():.4f}")

# Create residual analysis plots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Residuals time series
axes[0, 0].plot(forecast_residuals.index, forecast_residuals,
                color='#1f77b4', linewidth=0.8)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 0].set_title('Forecast Residuals Over Time', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Date', fontsize=10)
axes[0, 0].set_ylabel('Residual', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals histogram
axes[0, 1].hist(forecast_residuals, bins=50, color='#1f77b4',
                alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=forecast_residuals.mean(), color='red',
                   linestyle='--', linewidth=2, label=f'Mean: {forecast_residuals.mean():.2f}')
axes[0, 1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Residual', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Q-Q plot
stats.probplot(forecast_residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot - Normality Test', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. ACF of residuals
plot_acf(forecast_residuals, lags=50, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('ACF of Forecast Residuals', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Lag', fontsize=10)
axes[1, 1].set_ylabel('Correlation', fontsize=10)
axes[1, 1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 9: Forecast Residual Analysis**
#
# **Panel 1 - Residuals Over Time:**
# - Should fluctuate randomly around zero
# - No patterns or trends should be visible
# - Constant variance (homoscedasticity)
#
# **Panel 2 - Residuals Distribution:**
# - Should be approximately normal (bell-shaped)
# - Mean should be close to zero (unbiased forecasts)
# - Symmetric distribution indicates unbiased predictions
#
# **Panel 3 - Q-Q Plot:**
# - Points should fall along the diagonal line
# - Deviations indicate non-normality
# - Heavy tails may indicate outliers
#
# **Panel 4 - ACF of Residuals:**
# - Should show no significant autocorrelation
# - All bars should be within blue shaded area
# - Significant correlations indicate model hasn't captured all patterns

# %% [markdown]
# ## 3.5 Performance by Time Period
#
# Forecast performance is analyzed across different time periods to identify patterns or degradation.

# %%
# Calculate metrics by time period
# Ensure timezone compatibility by using the same timezone as the data
test_tz = test_data.index.tz

periods = {
    'First Week': (
        split_date,
        split_date + pd.Timedelta(days=7)
    ),
    'First 2 Weeks': (
        split_date,
        split_date + pd.Timedelta(days=14)
    ),
    'Full Period': (
        split_date,
        test_data.index.max()
    )
}

print("=" * 60)
print("PERFORMANCE BY TIME PERIOD")
print("=" * 60)

for period_name, (start, end) in periods.items():
    mask = (test_data.index >= start) & (test_data.index <= end)
    period_actual = test_data[mask]
    period_forecast = forecast_values[mask]
    
    period_metrics = calculate_metrics(period_actual, period_forecast)
    
    print(f"\n{period_name}:")
    print(f"  Date range: {start.date()} to {end.date()}")
    print(f"  Observations: {len(period_actual)}")
    print(f"  RMSE: {period_metrics['RMSE']:.2f}")
    print(f"  MAE: {period_metrics['MAE']:.2f}")
    print(f"  MAPE: {period_metrics['MAPE']:.2f}%")

# %% [markdown]
# **Performance by Time Period:**
#
# This analysis helps identify:
#
# **Forecast Degradation:**
# - If performance worsens over time, the model may not capture long-term patterns
# - Short-term forecasts are typically more accurate than long-term forecasts
#
# **Seasonal Patterns:**
# - Performance may vary by day of week or time of day
# - Some periods (e.g., weekends) may be harder to predict
#
# **Model Stability:**
# - Consistent performance across periods indicates stable model
# - Large variations suggest model may be sensitive to specific conditions

# %% [markdown]
# ## 3.6 Model Comparison with Baselines
#
# The ARIMA model's value is assessed by comparing it with simple baseline forecasts.

# %%
# Create baseline forecasts
baseline_naive = train_data.iloc[-1]  # Naive: last training value
baseline_mean = train_data.mean()  # Mean: average of training data

# Calculate baseline metrics
naive_forecast = pd.Series([baseline_naive] * len(test_data), index=test_data.index)
mean_forecast = pd.Series([baseline_mean] * len(test_data), index=test_data.index)

naive_metrics = calculate_metrics(test_data, naive_forecast)
mean_metrics = calculate_metrics(test_data, mean_forecast)

print("=" * 60)
print("MODEL COMPARISON WITH BASELINES")
print("=" * 60)
print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
print("-" * 60)
print(f"{'Naive (Last Value)':<20} {naive_metrics['RMSE']:<12.2f} {naive_metrics['MAE']:<12.2f} {naive_metrics['MAPE']:<12.2f}")
print(f"{'Mean Forecast':<20} {mean_metrics['RMSE']:<12.2f} {mean_metrics['MAE']:<12.2f} {mean_metrics['MAPE']:<12.2f}")
print(f"{'ARIMA({p},{d_optimal},{q})':<20} {out_of_sample_metrics['RMSE']:<12.2f} {out_of_sample_metrics['MAE']:<12.2f} {out_of_sample_metrics['MAPE']:<12.2f}")

print("\n" + "=" * 60)
print("IMPROVEMENT OVER BASELINES")
print("=" * 60)
naive_improvement_rmse = (naive_metrics['RMSE'] - out_of_sample_metrics['RMSE']) / naive_metrics['RMSE'] * 100
naive_improvement_mae = (naive_metrics['MAE'] - out_of_sample_metrics['MAE']) / naive_metrics['MAE'] * 100
mean_improvement_rmse = (mean_metrics['RMSE'] - out_of_sample_metrics['RMSE']) / mean_metrics['RMSE'] * 100
mean_improvement_mae = (mean_metrics['MAE'] - out_of_sample_metrics['MAE']) / mean_metrics['MAE'] * 100

print(f"vs Naive (Last Value):")
print(f"  RMSE improvement: {naive_improvement_rmse:+.1f}%")
print(f"  MAE improvement: {naive_improvement_mae:+.1f}%")
print(f"\nvs Mean Forecast:")
print(f"  RMSE improvement: {mean_improvement_rmse:+.1f}%")
print(f"  MAE improvement: {mean_improvement_mae:+.1f}%")

# Visual comparison
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(test_data.index, test_data, label='Actual Data',
        color='#1f77b4', linewidth=0.8)
ax.plot(forecast_values.index, forecast_values, label='ARIMA Forecast',
        color='red', linewidth=1.5, linestyle='--')
ax.plot(naive_forecast.index, naive_forecast, label='Naive Forecast',
        color='green', linewidth=1.5, linestyle=':', alpha=0.7)
ax.plot(mean_forecast.index, mean_forecast, label='Mean Forecast',
        color='orange', linewidth=1.5, linestyle=':', alpha=0.7)

ax.set_title('Model Comparison: ARIMA vs Baselines', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Number of Requests', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Figure 10: Model Comparison with Baselines**
#
# - **Blue line:** Actual test data
# - **Red dashed line:** ARIMA forecast
# - **Green dotted line:** Naive forecast (last training value)
# - **Orange dotted line:** Mean forecast (average of training data)
#
# **Interpretation:**
#
# **Baseline Models:**
# - **Naive forecast:** Uses the last observed value - very simple but often hard to beat
# - **Mean forecast:** Uses the average of all training data - ignores all patterns
#
# **ARIMA Value:**
# - If ARIMA significantly outperforms baselines, it's capturing meaningful patterns
# - If ARIMA only slightly outperforms naive forecast, consider model complexity vs benefit
# - If ARIMA performs worse than naive forecast, the model may be misspecified

# %% [markdown]
# ## 3.7 Section 3 Summary and Conclusions
#
# ### Overall Model Performance:
#
# **Forecast Accuracy (Out-of-Sample):**
# - RMSE: 123.04
# - MAE: 97.22
# - MAPE: 79.03%
#
# **Model Generalization:**
# - In-sample RMSE: 49.27
# - Out-of-sample RMSE: 123.04
# - Ratio: 2.50
# - Assessment: Poor generalization
#
# **Comparison with Baselines:**
# - vs Naive: +11.5% RMSE improvement
# - vs Mean: +0.1% RMSE improvement
# - ARIMA adds value: Marginally
#
# **Residual Analysis:**
# - Residual mean: 14.1784 (should be ~0, indicates slight bias)
# - Residual autocorrelation: Present
# - Residual normality: Not normal
# - Model assumptions: Partially violated
#
# ### Suitability for Production:
#
# **Suitability Assessment:**
#
# **Strengths:**
# - Provides better performance than the naive baseline (+11.5% RMSE improvement).
# - Captures some underlying trends in the data better than a static mean.
# - Statistical foundation allows for confidence interval calculation.
#
# **Weaknesses:**
# - Poor generalization (Ratio 2.50) indicates significant overfitting to training data.
# - Barely outperforms the simple Mean Forecast (+0.1% improvement), suggesting limited predictive power.
# - Residuals show bias (non-zero mean) and significant autocorrelation.
# - High MAPE (79.03%) makes it unreliable for precise capacity planning.
#
# **Recommendation:**
# - **Do not deploy** the current ARIMA(4,0,1) model in production.
# - The model does not provide sufficient accuracy improvement over simple baselines to justify its complexity.
# - Evaluation of more advanced models (Prophet, LSTM) is required.
#
# ### Recommendations for Improvement:
#
# **Model Enhancements:**
# 1. Consider SARIMA if seasonal patterns are present
# 2. Try Prophet for better handling of holidays and special events
# 3. Experiment with LSTM for capturing non-linear patterns
# 4. Implement ensemble methods combining multiple models
#
# **Feature Engineering:**
# 1. Add exogenous variables (time of day, day of week)
# 2. Include weather data if available
# 3. Add lag features at multiple time scales
# 4. Implement rolling window statistics
#
# **Evaluation:**
# 1. Test on multiple time windows (1min, 5min, 15min)
# 2. Perform cross-validation for robust assessment
# 3. Compare with other models (Prophet, XGBoost, LSTM)
# 4. Evaluate on different time periods (weekdays, weekends)
#
# ### Final Conclusion:
#
# The ARIMA(4, 0, 1) model failed to achieve acceptable forecasting performance for the NASA server traffic data. The model is not recommended for production use without further improvements and comparison with more advanced techniques.
