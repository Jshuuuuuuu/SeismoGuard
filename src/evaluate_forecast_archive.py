from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORECAST = ROOT / "outputs" / "forecast_archive" / "forecast_2026-05-12_enhanced_linear_regression.csv"
DEFAULT_EVENTS = ROOT / "data" / "raw" / "earthquakes" / "davao_region_2026_02_01_to_present.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "forecast_archive" / "forecast_2026-05-12_evaluation_results.csv"
DEFAULT_EVENTS_USED = ROOT / "outputs" / "forecast_archive" / "forecast_2026-05-12_actual_events_used.csv"

PROVINCE_MAP = {
    "davao de oro": "Davao De Oro",
    "davao del norte": "Davao Del Norte",
    "davao del sur": "Davao Del Sur",
    "davao occidental": "Davao Occidental",
    "davao oriental": "Davao Oriental",
}


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def extract_province(location: object) -> str | None:
    parts = re.findall(r"\(([^)]+)\)", normalize_text(location))
    for part in reversed(parts):
        province = PROVINCE_MAP.get(normalize_text(part).casefold())
        if province is not None:
            return province
    return None


def load_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path)
    events["datetime"] = pd.to_datetime(events["datetime"], errors="coerce")
    events["magnitude"] = pd.to_numeric(events["magnitude"], errors="coerce")
    events["province"] = events["location"].map(extract_province)
    return events.dropna(subset=["datetime", "magnitude", "province"])


def evaluate_forecast(
    forecast_path: Path,
    events_path: Path,
    output_path: Path,
    events_used_path: Path,
    actuals_checked_date: str,
    extended_zero_days: int,
) -> pd.DataFrame:
    forecast = pd.read_csv(forecast_path)
    events = load_events(events_path)

    forecast["forecast_window_start"] = pd.to_datetime(forecast["forecast_window_start"], errors="coerce")
    forecast["forecast_window_end"] = pd.to_datetime(forecast["forecast_window_end"], errors="coerce")
    forecast["predicted_max_magnitude_7d"] = pd.to_numeric(
        forecast["predicted_max_magnitude_7d"],
        errors="coerce",
    )

    rows = []
    events_used = []
    for _, row in forecast.iterrows():
        province = row["province"]
        start = row["forecast_window_start"]
        end = row["forecast_window_end"]
        province_events = events[
            (events["province"] == province)
            & (events["datetime"] >= start.tz_localize(None))
            & (events["datetime"] <= end.tz_localize(None))
        ].copy()

        actual = float(province_events["magnitude"].max()) if not province_events.empty else 0.0
        predicted = float(row["predicted_max_magnitude_7d"])
        result = row.to_dict()
        result.update(
            {
                "actual_max_magnitude_7d": actual,
                "error": actual - predicted,
                "absolute_error": abs(actual - predicted),
                "actual_data_source": "PHIVOLCS scraped bulletins",
                "actuals_checked_date": actuals_checked_date,
                "status": "evaluated",
                "event_count_7d": int(len(province_events)),
            }
        )

        if province_events.empty and extended_zero_days > 0:
            extended_start = end + pd.Timedelta(seconds=1)
            extended_end = end + pd.Timedelta(days=extended_zero_days)
            extended_events = events[
                (events["province"] == province)
                & (events["datetime"] >= extended_start.tz_localize(None))
                & (events["datetime"] <= extended_end.tz_localize(None))
            ].copy()
            result["extended_window_start"] = extended_start
            result["extended_window_end"] = extended_end
            result["extended_event_count"] = int(len(extended_events))
            result["extended_max_magnitude"] = (
                float(extended_events["magnitude"].max()) if not extended_events.empty else 0.0
            )
        else:
            result["extended_window_start"] = pd.NaT
            result["extended_window_end"] = pd.NaT
            result["extended_event_count"] = pd.NA
            result["extended_max_magnitude"] = pd.NA

        rows.append(result)
        if not province_events.empty:
            events_used.append(province_events.assign(forecast_province=province, window="primary_7d"))

    evaluated = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(output_path, index=False)

    if events_used:
        pd.concat(events_used, ignore_index=True).to_csv(events_used_path, index=False)
    else:
        pd.DataFrame(columns=list(events.columns) + ["forecast_province", "window"]).to_csv(
            events_used_path,
            index=False,
        )

    return evaluated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an archived 7-day province forecast against scraped PHIVOLCS records."
    )
    parser.add_argument("--forecast", default=str(DEFAULT_FORECAST), help="Archived forecast CSV.")
    parser.add_argument("--events", default=str(DEFAULT_EVENTS), help="Scraped PHIVOLCS events CSV.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Evaluation output CSV.")
    parser.add_argument("--events-used", default=str(DEFAULT_EVENTS_USED), help="Actual events used output CSV.")
    parser.add_argument("--actuals-checked-date", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument(
        "--extended-zero-days",
        type=int,
        default=7,
        help="For provinces with zero events in the primary window, also summarize this many follow-up days.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluated = evaluate_forecast(
        forecast_path=Path(args.forecast),
        events_path=Path(args.events),
        output_path=Path(args.output),
        events_used_path=Path(args.events_used),
        actuals_checked_date=args.actuals_checked_date,
        extended_zero_days=args.extended_zero_days,
    )

    print(
        evaluated[
            [
                "province",
                "predicted_max_magnitude_7d",
                "actual_max_magnitude_7d",
                "error",
                "absolute_error",
                "event_count_7d",
                "extended_event_count",
                "extended_max_magnitude",
            ]
        ].to_string(index=False)
    )
    print(f"MAE: {evaluated['absolute_error'].mean():.4f}")
    print(f"RMSE: {(evaluated['error'].pow(2).mean() ** 0.5):.4f}")


if __name__ == "__main__":
    main()
