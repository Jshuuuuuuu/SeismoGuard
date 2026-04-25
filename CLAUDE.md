# Main Study Blueprint (Region XI) — Locked Plan For Chapter 3

## Study Title (Final)
Event-Based Regression Modeling Of Earthquake Magnitudes In The Davao Region Using Machine Learning

---

## Core Goal (What The Model Will Do)
For Each Province In Region XI, On Each Forecast Date **T**, The Model Predicts:

**Ŷ(T) = The Maximum Earthquake Magnitude Expected In The Next 7 Days (T+1 To T+7)**

This Produces **Five Separate Forecasts (One Per Province)**:
- Davao de Oro
- Davao del Norte
- Davao del Sur
- Davao Occidental
- Davao Oriental

---

## Data Requirement (What Dataset Is Needed)
### Spatial Scope
Region XI (Davao Region) Only.

### Temporal Scope
- Development (Training + Backtesting): **01/01/2017 To 12/31/2025**
- Final Holdout Test Year: **2026**

### Minimum Fields Needed Per Earthquake Event
- Event Date/Time000
- Latitude
- Longitude
- Depth
- Magnitude
- Province (Or Coordinates That Can Be Mapped To Province)

---

## Forecast Horizon (Fixed For This Plan)
- Forecast Horizon: **7 Days**
- Meaning: The Model Uses Past Data Up To Date **T** And Predicts The **Next 7 Days** (T+1 To T+7).

---

## Target Variable (Regression Label)
For Each Province, For Each Forecast Date **T**:

**Y(T) = Max Magnitude Observed In That Province Time series are divided into two: linear and non-linear methods. Linear time series methods estimate by assuming that the series is stationary. Non-linear time series methods predict based on the raw version of the series in the real world.During (T+1 To T+7 Days)**

### No-Event Rule (Must Be Stated In Chapter 3)
If No Earthquake Occurs In That Province During (T+1 To T+7 Days), Then:
- **Y(T) = 0.0** As A “No-Event Indicator”

(Disclaimer: This Is A Modeling Convention, Not A Physical Magnitude.)

---

## Prediction Frequency (Recommended Choice)
- Recommended: **Weekly Forecast Dates** (Example: Every1–20 Monday)
- Alternative: Daily Forecast Dates (More Data Points, More Compute)

---

## Feature Engineering (Baseline Feature Set)
All Features Must Use Only Events **At Or Before T**.

### Rolling Window Features (Per Province)
Compute These For Each Province Separately:

#### A) Seismicity Rate
- Count Of Events In Last 1 Day
- Count Of Events In Last 7 Days
- Count Of Events In Last 30 Days

#### B) Magnitude Activity
- Maximum Magnitude In Last 7 Days
- Maximum Magnitude In Last 30 Days
- Mean Magnitude In Last 30 Days (Optional)

#### C) Depth Summary (Optional But Easy)
- Mean Depth In Last 30 Days
- Maximum Depth In Last 30 Days

#### D) Recency Signal (Optional But Useful)
- Days Since Last Event With M ≥ 4.0

---

## Modeling Approach (What Models To Use)
Because The Output Is A Numeric Magnitude, The Models Must Be **Regressors**.

### Required Baseline Regressor
- Linear Regression (Or Ridge Regression)

### Main ML Regressors
- Random Forest Regressor
- Gradient Boosting Regressor

### Important Note (Consistency Check)
- Logistic Regression Is Not Used Here Because Logistic Regression Is For Classification (Yes/No), Not Numeric Magnitude Forecasting.

---

## Validation Framework (Walk-Forward Static Retraining)
This Simulates Real Deployment With Strict Future Testing.

### Expanding Window (Yearly Rolling-Origin)
Example Pattern:
- Train: 2017-2019 → Test: 2020
- Train: 2019-2020 → Test: 2021
- Train: 2020–2021 → Test: 2022

### Region XI Version (Using 2000–2026 Scope)
- Backtest: Train Using Past Years, Test One Future Year At A Time (2017–2025 Recommended After A Burn-In)
- Final Holdout: Train 2000–2025 → Test 2026

### What “Test Year” Means
Testing In Year Y Means:
- Generate Forecasts On Dates Inside Year Y
- Each Forecast Predicts The Next 7 Days
- Compare Predicted Max Magnitude Vs Actual Max Magnitude Over That Horizon

---

## Evaluation Metrics (Regression)
Report Per Province And Summarize Across Years:
- MAE (Primary)
- RMSE
- R² (Optional)


## Outputs (What You Will Present)
### Per Province
- Forecasted Maximum Magnitude For Next 7 Days (Numeric Output)
- Backtest Performance (MAE/RMSE) For 2017–2025
- Final Holdout Performance For 2026

### Optional Province Comparison
- Ranking Of Provinces By Forecasted Risk (Higher Predicted Max Magnitude = Higher Short-Term Risk)

---

## Feature Upgrade Slots (Add Later If Approved)
These Are Optional Enhancements You Can Add After Approval:
- Fault Distance Features (If Fault Data Is Available)
- Spatial Clustering Metrics (Epicenter Spread, Hotspot Density)
- Declustering (Mainshock/Aftershock Separation)
- Two-Stage Modeling (Event Occurrence Then Magnitude Conditional On Occurrence)
- Calibration-Style Analysis For Derived Threshold Risk (Example: Predicted Max M > 4)

---

## One-Sentence Study Summary (For Chapter 1–3)
This Study Develops Province-Level Machine Learning Regression Models To Forecast The Maximum Earthquake Magnitude Expected Within A 7-Day Horizon In Region XI Using Historical Seismic Catalog Data And Walk-Forward Static Retraining Validation, With 2000–2025 For Backtesting And 2026 As A Final Holdout Test Year.