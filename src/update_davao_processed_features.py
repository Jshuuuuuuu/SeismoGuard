from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from feature_engineering import _min_dist_to_fault_km


ROOT = Path(__file__).resolve().parents[1]
RAW_SCRAPED_PATH = ROOT / "data" / "raw" / "earthquakes" / "davao_region_2026_02_01_to_present.csv"
PROCESSED_EVENTS_PATH = ROOT / "data" / "processed" / "Davao_Earthquakes_with_Dist.csv"
FEATURE_MATRIX_PATH = ROOT / "data" / "processed" / "Davao_ML_Feature_Matrix.csv"

VALID_PROVINCES = {
    "davao de oro": "Davao de Oro",
    "davao del norte": "Davao del Norte",
    "davao del sur": "Davao del Sur",
    "davao occidental": "Davao Occidental",
    "davao oriental": "Davao Oriental",
}

FEATURE_COLUMNS = [
    "count_7d",
    "count_30d",
    "max_mag_7d",
    "max_mag_30d",
    "mean_depth_30d",
    "days_since_m5",
    "min_fault_dist_km",
    "mean_fault_dist_7d",
    "pct_within_10km_30d",
    "num_clusters_30d",
    "pct_clustered_30d",
    "b_value_180d",
    "a_value_180d",
    "delta_b_90d",
    "n_above_mc_180d",
    "province",
    "forecast_date",
    "target_max_mag_7d",
]


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def extract_province(location: object) -> str | None:
    parts = re.findall(r"\(([^)]+)\)", normalize_text(location))
    for part in reversed(parts):
        normalized = normalize_text(part).casefold()
        province = VALID_PROVINCES.get(normalized)
        if province is not None:
            return province
    return None


def event_key(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_datetime(df["datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
        + "|"
        + pd.to_numeric(df["latitude"], errors="coerce").round(5).astype(str)
        + "|"
        + pd.to_numeric(df["longitude"], errors="coerce").round(5).astype(str)
        + "|"
        + pd.to_numeric(df["depth"], errors="coerce").round(3).astype(str)
        + "|"
        + pd.to_numeric(df["magnitude"], errors="coerce").round(3).astype(str)
        + "|"
        + df["location"].map(normalize_text)
    )


def load_new_events() -> pd.DataFrame:
    raw = pd.read_csv(RAW_SCRAPED_PATH)
    events = raw[["datetime", "latitude", "longitude", "depth", "magnitude", "location"]].copy()
    events["datetime"] = pd.to_datetime(events["datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ["latitude", "longitude", "depth", "magnitude"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    events["location"] = events["location"].map(normalize_text)
    events = events.dropna(subset=["datetime", "latitude", "longitude", "depth", "magnitude", "location"])
    events["dist_km"] = events.apply(
        lambda row: _min_dist_to_fault_km(float(row["latitude"]), float(row["longitude"])),
        axis=1,
    )
    return events[["datetime", "latitude", "longitude", "depth", "magnitude", "location", "dist_km"]]


def update_processed_events() -> pd.DataFrame:
    existing = pd.read_csv(PROCESSED_EVENTS_PATH)
    new_events = load_new_events()

    existing_keys = set(event_key(existing))
    new_events = new_events[~event_key(new_events).isin(existing_keys)].copy()

    combined = pd.concat([existing, new_events], ignore_index=True)
    if not new_events.empty:
        new_events.to_csv(PROCESSED_EVENTS_PATH, mode="a", header=False, index=False)
    return combined


def b_value_features(events: pd.DataFrame, t: pd.Timestamp) -> dict[str, float]:
    def compute(window: pd.DataFrame) -> tuple[float, float, int]:
        mags = pd.to_numeric(window["magnitude"], errors="coerce").dropna()
        above = mags[mags >= 2.7]
        n_above = int(len(above))
        if n_above < 30:
            return np.nan, np.nan, n_above

        # Aki maximum-likelihood b-value with 0.1 magnitude-bin correction.
        denominator = float(above.mean() - (2.7 - 0.05))
        if denominator <= 0:
            return np.nan, np.nan, n_above
        b_value = math.log10(math.e) / denominator
        a_value = math.log10(n_above) + b_value * 2.7
        return float(b_value), float(a_value), n_above

    past_180d = events[(events["datetime"] >= t - pd.Timedelta(days=180)) & (events["datetime"] < t)]
    cur_90d = events[(events["datetime"] >= t - pd.Timedelta(days=90)) & (events["datetime"] < t)]
    prev_90d = events[
        (events["datetime"] >= t - pd.Timedelta(days=180))
        & (events["datetime"] < t - pd.Timedelta(days=90))
    ]

    b_180, a_180, n_above = compute(past_180d)
    b_cur, _, _ = compute(cur_90d)
    b_prev, _, _ = compute(prev_90d)
    delta_b = float(b_cur - b_prev) if not (np.isnan(b_cur) or np.isnan(b_prev)) else np.nan

    return {
        "b_value_180d": b_180,
        "a_value_180d": a_180,
        "delta_b_90d": delta_b,
        "n_above_mc_180d": n_above,
    }


def clustering_features(events_30d: pd.DataFrame) -> dict[str, float]:
    coords = events_30d[["latitude", "longitude"]].dropna()
    if coords.empty:
        return {"num_clusters_30d": 0, "pct_clustered_30d": 0.0}

    labels = DBSCAN(eps=0.1, min_samples=2, metric="euclidean").fit_predict(coords.to_numpy())
    return {
        "num_clusters_30d": int(len(set(labels) - {-1})),
        "pct_clustered_30d": float((labels != -1).mean()),
    }


def build_feature_row(events: pd.DataFrame, province: str, t: pd.Timestamp) -> dict[str, object]:
    province_events = events[events["province"] == province].sort_values("datetime")
    past = province_events[province_events["datetime"] < t]
    past_7d = past[past["datetime"] >= t - pd.Timedelta(days=7)]
    past_30d = past[past["datetime"] >= t - pd.Timedelta(days=30)]

    sig = past[past["magnitude"] >= 5.0]
    days_since_m5 = 999 if sig.empty else int((t - sig["datetime"].max()).days)

    horizon = province_events[
        (province_events["datetime"] >= t + pd.Timedelta(days=1))
        & (province_events["datetime"] <= t + pd.Timedelta(days=7))
    ]

    row: dict[str, object] = {
        "count_7d": int(len(past_7d)),
        "count_30d": int(len(past_30d)),
        "max_mag_7d": float(past_7d["magnitude"].max()) if not past_7d.empty else 0.0,
        "max_mag_30d": float(past_30d["magnitude"].max()) if not past_30d.empty else 0.0,
        "mean_depth_30d": float(past_30d["depth"].mean()) if not past_30d.empty else np.nan,
        "days_since_m5": days_since_m5,
        "min_fault_dist_km": float(past_30d["dist_km"].min()) if not past_30d.empty else 0.0,
        "mean_fault_dist_7d": float(past_7d["dist_km"].mean()) if not past_7d.empty else 0.0,
        "pct_within_10km_30d": float((past_30d["dist_km"] <= 10).mean()) if not past_30d.empty else 0.0,
        "province": province,
        "forecast_date": t.strftime("%Y-%m-%d"),
        "target_max_mag_7d": float(horizon["magnitude"].max()) if not horizon.empty else 0.0,
    }
    row.update(clustering_features(past_30d))
    row.update(b_value_features(province_events, t))
    return row


def append_feature_matrix(events: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.read_csv(FEATURE_MATRIX_PATH)
    matrix["forecast_date"] = pd.to_datetime(matrix["forecast_date"], errors="coerce")

    events = events.copy()
    events["datetime"] = pd.to_datetime(events["datetime"], errors="coerce")
    for col in ["latitude", "longitude", "depth", "magnitude", "dist_km"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    events["province"] = events["location"].map(extract_province)
    events = events.dropna(subset=["datetime", "province", "latitude", "longitude", "depth", "magnitude"])

    start = matrix["forecast_date"].max() + pd.Timedelta(days=7)
    end = events["datetime"].max().normalize() - pd.Timedelta(days=7)
    if end < start:
        matrix["forecast_date"] = matrix["forecast_date"].dt.strftime("%Y-%m-%d")
        return matrix

    rows: list[dict[str, object]] = []
    for t in pd.date_range(start=start, end=end, freq="W-MON"):
        for province in sorted(VALID_PROVINCES.values()):
            rows.append(build_feature_row(events, province, t))

    appended = pd.DataFrame(rows)
    appended = appended[FEATURE_COLUMNS]
    appended.to_csv(FEATURE_MATRIX_PATH, mode="a", header=False, index=False)

    combined = pd.concat([matrix, appended], ignore_index=True).reset_index(drop=True)
    combined["forecast_date"] = pd.to_datetime(combined["forecast_date"]).dt.strftime("%Y-%m-%d")
    return combined


def main() -> None:
    events = update_processed_events()
    matrix = append_feature_matrix(events)

    event_dates = pd.to_datetime(events["datetime"], errors="coerce")
    print(f"Processed events: {len(events)} rows ({event_dates.min()} to {event_dates.max()})")
    print(f"Feature matrix: {len(matrix)} rows ({matrix['forecast_date'].min()} to {matrix['forecast_date'].max()})")


if __name__ == "__main__":
    main()
