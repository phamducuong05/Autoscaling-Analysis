# ARIMA Model Structure & Results

## Overview
This document outlines the structure, process, and results of the ARIMA model implementation for the Autoscaling Analysis project. The notebook `notebooks/experimental/arima_model.py` serves as the initial experiment for time series forecasting using the ARIMA methodology on the 1995 NASA HTTP log dataset.

## Table of Contents
1. [Section 1: Pre-train - Suitability & Data Preparation](#section-1-pre-train---suitability--data-preparation)
2. [Section 2: Training](#section-2-training)
3. [Section 3: Post-train Evaluation](#section-3-post-train-evaluation)
4. [Key Findings Summary](#key-findings-summary)
5. [Recommendations](#recommendations)

## Section 1: Pre-train - Suitability & Data Preparation

### 1.1 Import Libraries
Standard libraries including `pandas`, `numpy`, `matplotlib`, `statsmodels`, and `pmdarima` are imported.

### 1.2 Data Loading
- **Dataset:** 1995 NASA HTTP access logs (Cleaned)
- **Observations:** 17,856 data points
- **Features:** 17 columns
- **Time Interval:** 5-minute aggregation windows
- **Date Range:** July 1, 1995 to August 31, 1995

### 1.3 Stationarity Assessment
Augmented Dickey-Fuller (ADF) test was performed to check for stationarity.
- **Result:** p-value = 0.000000
- **Conclusion:** The series is Stationary. Differencing parameter (`d`) = 0.

### 1.4 ACF/PACF Analysis
Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots were generated to estimate model parameters.
- **ACF:** Slow decay pattern observed.
- **PACF:** Sharp cutoff at lag 1-2.
- **Suggestion:** Indicates an Autoregressive (AR) model of order 1 or 2.

### 1.5 Train/Test Split
Data was split chronologically to respect time series order.
- **Training Set:** 15,264 samples (85.5%) - July 1 to Aug 22
- **Test Set:** 2,592 samples (14.5%) - Remaining Aug days

### 1.6 Summary
Data is prepared, stationary, and suitable for ARIMA modeling without differencing ($d=0$).

## Section 2: Training

### 2.1 Auto-ARIMA Parameter Selection
`pmdarima.auto_arima` was used to automatically discover optimal parameters based on AIC.
- **Selected Model:** ARIMA(4, 0, 1) with intercept

### 2.2 Manual vs Automatic Comparison
Comparative analysis between auto-selected and manual candidate models.
- **Auto-ARIMA AIC:** 162,310.15
- **Auto-ARIMA BIC:** 162,363.58
- **Manual Models Tested:** ARIMA(1,0,0) and ARIMA(2,0,0)

### 2.3 Model Fitting
The best performing model (ARIMA 4,0,1) was fitted to the full training dataset.

### 2.3.1 Model Saving
Model artifacts were serialized for potential reuse.

### 2.4 Model Diagnostics
Residual analysis was conducted to validate model assumptions.
- **Ljung-Box Test:** p=0.450 (Residuals are uncorrelated - Good)
- **Jarque-Bera Test:** p=0.000 (Residuals are not normally distributed)

### 2.5 Summary
The model captures the correlation structure well but residuals exhibit non-normal characteristics, likely due to traffic spikes.

## Section 3: Post-train Evaluation

### 3.1 Forecast Generation
Forecasts were generated for the test period (last week of August).

### 3.2 Performance Metrics
Quantitative assessment of model accuracy on unseen data.
- **RMSE:** 123.04
- **MAE:** 97.22
- **MAPE:** 79.03%

### 3.3 Visualization
Visual comparison plots of Actual vs Predicted traffic were generated.

### 3.4 Residual Analysis
Analysis of prediction errors over time.

### 3.5 Performance by Time Period
Metric breakdown (In-sample vs Out-of-sample).
- **In-sample RMSE:** 49.27
- **Out-of-sample RMSE:** 123.04
- **Generalization Ratio:** 2.50 (High overfitting indication)

### 3.6 Baseline Comparison
Comparison against simple baseline models.
- **vs Naive Approach:** +11.5% RMSE improvement
- **vs Mean Baseline:** +0.1% RMSE improvement

### 3.7 Summary and Conclusions
The model shows significant overfitting and fails to generalize well to the test set. It barely outperforms a simple mean baseline.

## Key Findings Summary
*   **Stationarity:** Data is stationary without differencing ($d=0$).
*   **Optimal Model:** ARIMA(4, 0, 1) was identified as the best fit by AIC.
*   **Overfitting:** Significant gap between training error (49.27) and test error (123.04).
*   **Performance:** High MAPE (79.03%) indicates poor predictive accuracy for scaling decisions.
*   **Diagnostics:** Residuals are uncorrelated but not normally distributed.

## Recommendations
1.  **Do Not Deploy:** The current ARIMA model is not accurate enough for production autoscaling.
2.  **Try Advanced Models:** Investigate Prophet (for seasonality) or LSTM (for complex patterns).
3.  **Feature Engineering:** Incorporate exogenous variables (time of day, day of week) which standard ARIMA lacks.
4.  **Seasonality:** Explore SARIMA to explicitly model the daily/weekly cycles observed in ACF.
