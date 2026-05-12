# Forecast Evaluation Plan

Archived forecast:

- Forecast file: `forecast_archive/forecast_2026-05-12_enhanced_linear_regression.csv`
- Model: enhanced Linear Regression pipeline
- Feature set: 17 engineered features + 17 missing indicators + 5 province one-hot indicators
- Forecast date: 2026-05-12 00:00:00 UTC
- Forecast window: 2026-05-13 00:00:00 UTC to 2026-05-19 00:00:00 UTC
- Target definition: highest observed earthquake magnitude per province during the 7-day window

Evaluation workflow after the window closes:

1. Download or append the latest PHIVOLCS earthquake events covering 2026-05-13 through 2026-05-19.
2. Filter the actual events to the five Davao provinces.
3. Compute the observed maximum magnitude per province during the forecast window.
4. Fill `actual_max_magnitude_7d` in `forecast_archive/forecast_2026-05-12_evaluation_template.csv`.
5. Compute errors:
   - `error = actual_max_magnitude_7d - predicted_max_magnitude_7d`
   - `absolute_error = abs(error)`
6. Use the results to discuss model behavior and decide whether to add features, tune models, or retrain with updated data.

Important note:

This forecast should remain frozen. Do not overwrite the archived forecast after seeing the actual PHIVOLCS outcomes, because it represents the out-of-sample prediction made before the target window was observed.
