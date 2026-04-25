from __future__ import annotations

import math
from typing import Any

import pandas as pd


def build_category_1_3_features(province_df: pd.DataFrame, forecast_date_T: Any) -> dict[str, int | float]:
    """Build Category 1-3 time-window features up to forecast date T.

    Required columns in ``province_df``:
    - ``datetime``: event timestamp
    - ``magnitude``: earthquake magnitude
    - ``depth``: earthquake depth

    Past-only semantics (no leakage):
    - Use only events strictly before T: datetime < T
    - 1 day window: [T - 1 day, T)
    - 7 day window: [T - 7 days, T)
    - 30 day window: [T - 30 days, T)
    """
    required_cols = {"datetime", "magnitude", "depth"}
    missing_cols = required_cols - set(province_df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"province_df is missing required columns: {missing}")

    t = pd.to_datetime(forecast_date_T, utc=True)
    if pd.isna(t):
        raise ValueError("forecast_date_T could not be parsed into a valid datetime")

    df = province_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")

    # Strictly past-only: events at T are excluded to prevent leakage.
    df = df[(df["datetime"].notna()) & (df["datetime"] < t)]

    def _window_slice(days: int) -> pd.DataFrame:
        start = t - pd.Timedelta(days=days)
        return df[df["datetime"] >= start]

    df_1d = _window_slice(1)
    df_7d = _window_slice(7)
    df_30d = _window_slice(30)

    return {
        "count_1d": int(len(df_1d)),
        "count_7d": int(len(df_7d)),
        "count_30d": int(len(df_30d)),
        "max_mag_7d": float(df_7d["magnitude"].max()) if not df_7d.empty else float("nan"),
        "max_mag_30d": float(df_30d["magnitude"].max()) if not df_30d.empty else float("nan"),
        "mean_depth_30d": float(df_30d["depth"].mean()) if not df_30d.empty else float("nan"),
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two latitude/longitude points."""
    r_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_km * c


def nearest_fault_distance_km(
    event_lat: float,
    event_lon: float,
    fault_points_df: pd.DataFrame,
    fault_lat_col: str = "latitude",
    fault_lon_col: str = "longitude",
) -> float:
    """Return nearest fault-point distance (km) from an event location.

    ``fault_points_df`` is expected to come from a PHIVOLCS/DOST fault map export
    with at least latitude/longitude columns.
    """
    required_cols = {fault_lat_col, fault_lon_col}
    missing_cols = required_cols - set(fault_points_df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"fault_points_df is missing required columns: {missing}")

    fault_df = fault_points_df.copy()
    fault_df[fault_lat_col] = pd.to_numeric(fault_df[fault_lat_col], errors="coerce")
    fault_df[fault_lon_col] = pd.to_numeric(fault_df[fault_lon_col], errors="coerce")
    fault_df = fault_df.dropna(subset=[fault_lat_col, fault_lon_col])

    if fault_df.empty:
        return float("nan")

    distances = [
        haversine_km(event_lat, event_lon, float(lat), float(lon))
        for lat, lon in zip(fault_df[fault_lat_col], fault_df[fault_lon_col])
    ]
    return float(min(distances)) if distances else float("nan")


def build_category_4_5_features(
    province_df: pd.DataFrame,
    forecast_date_T: Any,
    fault_points_df: pd.DataFrame,
    event_lat_col: str = "latitude",
    event_lon_col: str = "longitude",
    fault_lat_col: str = "latitude",
    fault_lon_col: str = "longitude",
) -> dict[str, float]:
    """Build Category 4 and 5 features using strict past-only data.

    Category 4 (recency):
    - days_since_m5: days from T to most recent event before T with magnitude >= 5.0

    Category 5 (fault proximity):
    - nearest_fault_km: nearest fault-point distance for latest event before T
    """
    required_cols = {"datetime", "magnitude", event_lat_col, event_lon_col}
    missing_cols = required_cols - set(province_df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"province_df is missing required columns: {missing}")

    t = pd.to_datetime(forecast_date_T, utc=True)
    if pd.isna(t):
        raise ValueError("forecast_date_T could not be parsed into a valid datetime")

    events = province_df.copy()
    events["datetime"] = pd.to_datetime(events["datetime"], errors="coerce", utc=True)
    events["magnitude"] = pd.to_numeric(events["magnitude"], errors="coerce")
    events[event_lat_col] = pd.to_numeric(events[event_lat_col], errors="coerce")
    events[event_lon_col] = pd.to_numeric(events[event_lon_col], errors="coerce")

    # Strict no-leakage cutoff.
    events = events[(events["datetime"].notna()) & (events["datetime"] < t)].copy()

    if events.empty:
        return {"days_since_m5": float("nan"), "nearest_fault_km": float("nan")}

    # Recency signal for latest M5+ event.
    m5_events = events[events["magnitude"] >= 5.0]
    if m5_events.empty:
        days_since_m5 = float("nan")
    else:
        last_m5_time = m5_events["datetime"].max()
        days_since_m5 = float((t - last_m5_time) / pd.Timedelta(days=1))

    # Fault proximity from the latest known event location before T.
    latest_event = events.sort_values("datetime").iloc[-1]
    lat = latest_event[event_lat_col]
    lon = latest_event[event_lon_col]
    if pd.isna(lat) or pd.isna(lon):
        nearest_fault_km = float("nan")
    else:
        nearest_fault_km = nearest_fault_distance_km(
            event_lat=float(lat),
            event_lon=float(lon),
            fault_points_df=fault_points_df,
            fault_lat_col=fault_lat_col,
            fault_lon_col=fault_lon_col,
        )

    return {
        "days_since_m5": days_since_m5,
        "nearest_fault_km": float(nearest_fault_km),
    }
