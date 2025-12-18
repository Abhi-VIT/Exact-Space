# Cyclone Operation Analysis

This analysis explores the operational patterns, anomalies, and performance of a cyclone system using time series data. The analysis is broken down into several key components:

## 1. Shutdown Detection

- Identified periods where the cyclone was not operational
- Method: Used temperature thresholds and rate of change
- Results stored in `shutdown_periods.csv`

## 2. Operational States Analysis

- Applied K-means clustering to identify distinct operational states
- Features: Temperature and draft measurements
- Results summarized in `clusters_summary.csv`
- Visualization available in `plots/clusters.png`

## 3. Anomaly Detection

- Implemented Isolation Forest algorithm to detect anomalous behavior
- Analyzed deviations using z-scores
- Results stored in `anomalous_periods.csv`

## 4. Time Series Forecasting

- Developed one-hour ahead forecasts using:
  - Persistence model (baseline)
  - Moving average model
- Results stored in `forecasts.csv`
- Performance visualization in `plots/forecasting.png`

## Key Findings

1. The cyclone exhibits distinct operational states characterized by different temperature and draft profiles
2. Several anomalous periods were detected, mainly during transition states
3. Simple forecasting models provide reasonable short-term predictions

## Files Description

- `task1_analysis.ipynb`: Main analysis notebook
- `shutdown_periods.csv`: Start and end times of detected shutdowns
- `anomalous_periods.csv`: Detected anomalies with scores and affected variables
- `clusters_summary.csv`: Statistical summary of each operational state
- `forecasts.csv`: Actual and predicted values from forecasting models

### Plots
- `time_series_patterns.png`: Visualization of temperature and draft patterns
- `shutdown_periods.png`: Timeline with highlighted shutdown periods
- `clusters.png`: Visualization of operational states
- `forecasting.png`: Comparison of forecasting models' performance
