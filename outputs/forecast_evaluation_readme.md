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

Scraped-record workflow:

```bash
python webscraping/scrape_davao_earthquakes.py --start-date 2026-05-13 --end-date 2026-05-26 --output data/raw/earthquakes/davao_region_2026_02_01_to_present.csv
python src/evaluate_forecast_archive.py --actuals-checked-date 2026-05-20
python src/update_davao_processed_features.py
```

The primary 7-day evaluation must remain 2026-05-13 to 2026-05-19. If a province has zero events in that primary window, the evaluation script also records an optional follow-up window in `extended_*` columns. That extended window is useful for discussion, but it should not replace the original 7-day target score.

Important note:

This forecast should remain frozen. Do not overwrite the archived forecast after seeing the actual PHIVOLCS outcomes, because it represents the out-of-sample prediction made before the target window was observed.
