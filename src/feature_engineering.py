from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


def _build_category_1_3_base(province_df: pd.DataFrame, forecast_date_T: Any) -> dict[str, int | float]:
    """Base builder for Category 1-3 time-window features up to forecast date T.

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


def cat1(province_df: pd.DataFrame, forecast_date_T: Any) -> dict[str, int]:
    """Category 1 features (seismicity rate)."""
    features = _build_category_1_3_base(province_df, forecast_date_T)
    return {
        "count_1d": int(features["count_1d"]),
        "count_7d": int(features["count_7d"]),
        "count_30d": int(features["count_30d"]),
    }


def cat2(province_df: pd.DataFrame, forecast_date_T: Any) -> dict[str, float]:
    """Category 2 features (magnitude activity)."""
    features = _build_category_1_3_base(province_df, forecast_date_T)
    return {
        "max_mag_7d": float(features["max_mag_7d"]),
        "max_mag_30d": float(features["max_mag_30d"]),
    }


def cat3(province_df: pd.DataFrame, forecast_date_T: Any) -> dict[str, float]:
    """Category 3 features (depth summary)."""
    features = _build_category_1_3_base(province_df, forecast_date_T)
    return {
        "mean_depth_30d": float(features["mean_depth_30d"]),
    }


def build_category_1_3_features(province_df: pd.DataFrame, forecast_date_T: Any) -> dict[str, int | float]:
    """Legacy name for Category 1-3 features. Prefer cat1/cat2/cat3."""
    return _build_category_1_3_base(province_df, forecast_date_T)


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


def _build_category_4_5_base(
    province_df: pd.DataFrame,
    forecast_date_T: Any,
    fault_points_df: pd.DataFrame,
    event_lat_col: str = "latitude",
    event_lon_col: str = "longitude",
    fault_lat_col: str = "latitude",
    fault_lon_col: str = "longitude",
) -> dict[str, float]:
    """Base builder for Category 4 and 5 features using strict past-only data."""
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


def cat4(
    province_df: pd.DataFrame,
    forecast_date_T: Any,
    fault_points_df: pd.DataFrame,
    event_lat_col: str = "latitude",
    event_lon_col: str = "longitude",
    fault_lat_col: str = "latitude",
    fault_lon_col: str = "longitude",
) -> dict[str, float]:
    """Category 4 features (recency)."""
    features = _build_category_4_5_base(
        province_df=province_df,
        forecast_date_T=forecast_date_T,
        fault_points_df=fault_points_df,
        event_lat_col=event_lat_col,
        event_lon_col=event_lon_col,
        fault_lat_col=fault_lat_col,
        fault_lon_col=fault_lon_col,
    )
    return {"days_since_m5": float(features["days_since_m5"])}


def cat5(
    province_df: pd.DataFrame,
    forecast_date_T: Any,
    fault_points_df: pd.DataFrame,
    event_lat_col: str = "latitude",
    event_lon_col: str = "longitude",
    fault_lat_col: str = "latitude",
    fault_lon_col: str = "longitude",
) -> dict[str, float]:
    """Category 5 features (fault proximity)."""
    features = _build_category_4_5_base(
        province_df=province_df,
        forecast_date_T=forecast_date_T,
        fault_points_df=fault_points_df,
        event_lat_col=event_lat_col,
        event_lon_col=event_lon_col,
        fault_lat_col=fault_lat_col,
        fault_lon_col=fault_lon_col,
    )
    return {"nearest_fault_km": float(features["nearest_fault_km"])}


def build_category_4_5_features(
    province_df: pd.DataFrame,
    forecast_date_T: Any,
    fault_points_df: pd.DataFrame,
    event_lat_col: str = "latitude",
    event_lon_col: str = "longitude",
    fault_lat_col: str = "latitude",
    fault_lon_col: str = "longitude",
) -> dict[str, float]:
    """Legacy name for Category 4-5 features. Prefer cat4/cat5."""
    return _build_category_4_5_base(
        province_df=province_df,
        forecast_date_T=forecast_date_T,
        fault_points_df=fault_points_df,
        event_lat_col=event_lat_col,
        event_lon_col=event_lon_col,
        fault_lat_col=fault_lat_col,
        fault_lon_col=fault_lon_col,
    )


def cat6(
    province_df: pd.DataFrame,
    forecast_date_T: Any,
    event_lat_col: str = "latitude",
    event_lon_col: str = "longitude",
    dbscan_eps_degrees: float = 0.1,
    dbscan_min_samples: int = 2,
    grid_cell_size_degrees: float = 0.1,
) -> dict[str, int | float]:
    """Category 6 clustering features on the last 30 days before T.
    Returns:
    - num_clusters_30d: number of DBSCAN clusters excluding noise
    - largest_cluster_size_30d: size of the largest cluster
    - pct_clustered_30d: share of 30-day events assigned to a cluster
    - max_grid_cell_count_30d: max number of events in any 0.1-degree grid cell
    """
    required_cols = {"datetime", event_lat_col, event_lon_col}
    missing_cols = required_cols - set(province_df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"province_df is missing required columns: {missing}")

    t = pd.to_datetime(forecast_date_T, utc=True)
    if pd.isna(t):
        raise ValueError("forecast_date_T could not be parsed into a valid datetime")

    events = province_df.copy()
    events["datetime"] = pd.to_datetime(events["datetime"], errors="coerce", utc=True)
    events[event_lat_col] = pd.to_numeric(events[event_lat_col], errors="coerce")
    events[event_lon_col] = pd.to_numeric(events[event_lon_col], errors="coerce")

    # Strict no-leakage cutoff with a 30-day lookback window.
    start = t - pd.Timedelta(days=30)
    events = events[(events["datetime"].notna()) & (events["datetime"] >= start) & (events["datetime"] < t)].copy()

    coords = events[[event_lat_col, event_lon_col]].dropna()
    if coords.empty:
        return {
            "num_clusters_30d": 0,
            "largest_cluster_size_30d": 0,
            "pct_clustered_30d": 0.0,
            "max_grid_cell_count_30d": 0,
        }

    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "scikit-learn is required for cat6; install scikit-learn to use DBSCAN clustering"
        ) from exc

    clustering = DBSCAN(eps=dbscan_eps_degrees, min_samples=dbscan_min_samples, metric="euclidean")
    labels = clustering.fit_predict(coords[[event_lat_col, event_lon_col]].to_numpy())

    clustered_mask = labels != -1
    clustered_points = int(clustered_mask.sum())
    cluster_labels = sorted(set(labels) - {-1})
    cluster_sizes = [int((labels == label).sum()) for label in cluster_labels]
    num_clusters = int(len(cluster_labels))
    largest_cluster_size = int(max(cluster_sizes)) if cluster_sizes else 0
    pct_clustered = float(clustered_points / len(labels)) if len(labels) else 0.0

    grid = coords.copy()
    grid["grid_lat"] = (grid[event_lat_col] // grid_cell_size_degrees).astype(int)
    grid["grid_lon"] = (grid[event_lon_col] // grid_cell_size_degrees).astype(int)
    grid_cell_counts = grid.groupby(["grid_lat", "grid_lon"]).size()
    max_grid_cell_count = int(grid_cell_counts.max()) if not grid_cell_counts.empty else 0

    return {
        "num_clusters_30d": int(num_clusters),
        "largest_cluster_size_30d": int(largest_cluster_size),
        "pct_clustered_30d": pct_clustered,
        "max_grid_cell_count_30d": int(max_grid_cell_count),
    }


def build_category_6_features(
    province_df: pd.DataFrame,
    forecast_date_T: Any,
    event_lat_col: str = "latitude",
    event_lon_col: str = "longitude",
    dbscan_eps_degrees: float = 0.1,
    dbscan_min_samples: int = 2,
    grid_cell_size_degrees: float = 0.1,
) -> dict[str, int | float]:
    """Legacy name for Category 6 features. Prefer cat6."""
    return cat6(
        province_df=province_df,
        forecast_date_T=forecast_date_T,
        event_lat_col=event_lat_col,
        event_lon_col=event_lon_col,
        dbscan_eps_degrees=dbscan_eps_degrees,
        dbscan_min_samples=dbscan_min_samples,
        grid_cell_size_degrees=grid_cell_size_degrees,
    )


def event_based_generator(
    events_df: pd.DataFrame,
    feature_fn,
    province_col: str = "province",
    datetime_col: str = "datetime",
    magnitude_col: str = "magnitude",
    forecast_freq: str = "W-MON",
    forecast_start: Any | None = None,
    forecast_end: Any | None = None,
    feature_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate event-based samples per province using weekly forecast dates.
    Returns:
    - X: feature matrix (DataFrame) indexed by (province, forecast_date)
    - y: target vector (Series) indexed by (province, forecast_date)
    """
    required_cols = {province_col, datetime_col, magnitude_col}
    missing_cols = required_cols - set(events_df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"events_df is missing required columns: {missing}")

    feature_kwargs = feature_kwargs or {}

    events = events_df.copy()
    events[datetime_col] = pd.to_datetime(events[datetime_col], errors="coerce", utc=True)
    events[magnitude_col] = pd.to_numeric(events[magnitude_col], errors="coerce")
    events = events.dropna(subset=[datetime_col])

    rows: list[dict[str, Any]] = []
    targets: list[float] = []
    index_keys: list[tuple[str, pd.Timestamp]] = []

    for province, province_df in events.groupby(province_col):
        province_events = province_df.sort_values(datetime_col)
        if province_events.empty:
            continue

        min_date = province_events[datetime_col].min()
        max_date = province_events[datetime_col].max()

        start = pd.to_datetime(forecast_start, utc=True) if forecast_start is not None else min_date.normalize()
        end = pd.to_datetime(forecast_end, utc=True) if forecast_end is not None else max_date.normalize()

        # Ensure target window (T+1 to T+7) is within available data.
        end = min(end, max_date - pd.Timedelta(days=7))
        if end < start:
            continue

        for t in pd.date_range(start=start, end=end, freq=forecast_freq, tz="UTC"):
            features = feature_fn(province_events, t, **feature_kwargs)
            if not isinstance(features, dict):
                raise ValueError("feature_fn must return a dict of feature_name -> value")

            horizon_start = t + pd.Timedelta(days=1)
            horizon_end = t + pd.Timedelta(days=7)
            target_events = province_events[
                (province_events[datetime_col] >= horizon_start)
                & (province_events[datetime_col] <= horizon_end)
            ]
            target = float(target_events[magnitude_col].max()) if not target_events.empty else 0.0

            rows.append(features)
            targets.append(target)
            index_keys.append((str(province), pd.Timestamp(t)))

    index = pd.MultiIndex.from_tuples(index_keys, names=[province_col, "forecast_date"])
    X = pd.DataFrame(rows, index=index)
    y = pd.Series(targets, index=index, name="target")
    return X, y


def leakage_audit(
    X: pd.DataFrame,
    events_df: pd.DataFrame,
    feature_fn,
    province_col: str = "province",
    datetime_col: str = "datetime",
    sample_size: int = 10,
    random_state: int = 42,
    feature_kwargs: dict[str, Any] | None = None,
    atol: float = 1e-9,
) -> pd.DataFrame:
    """Audit leakage by recomputing sampled feature rows using past-only data.

    Returns a long-form DataFrame with observed values (from X) and
    recomputed values (full vs past-only). Any mismatches indicate leakage
    or drift between X and the feature function.
    """
    if not isinstance(X.index, pd.MultiIndex) or province_col not in X.index.names:
        raise ValueError("X must be indexed by a MultiIndex including province and forecast_date")

    if "forecast_date" not in X.index.names:
        raise ValueError("X index must include 'forecast_date'")

    feature_kwargs = feature_kwargs or {}

    events = events_df.copy()
    events[datetime_col] = pd.to_datetime(events[datetime_col], errors="coerce", utc=True)
    events = events.dropna(subset=[datetime_col])

    sample = X.sample(n=min(sample_size, len(X)), random_state=random_state)
    records: list[dict[str, Any]] = []

    def _values_match(a: Any, b: Any) -> bool:
        if pd.isna(a) and pd.isna(b):
            return True
        try:
            return math.isclose(float(a), float(b), abs_tol=atol)
        except (TypeError, ValueError):
            return a == b

    for (province, forecast_date), row in sample.iterrows():
        province_events = events[events[province_col] == province].copy()
        forecast_date = pd.to_datetime(forecast_date, utc=True)
        past_only = province_events[province_events[datetime_col] < forecast_date]

        recomputed_full = feature_fn(province_events, forecast_date, **feature_kwargs)
        recomputed_past = feature_fn(past_only, forecast_date, **feature_kwargs)

        for feature_name, observed_value in row.to_dict().items():
            full_value = recomputed_full.get(feature_name)
            past_value = recomputed_past.get(feature_name)
            matches_full = _values_match(observed_value, full_value)
            matches_past = _values_match(observed_value, past_value)
            records.append(
                {
                    "province": province,
                    "forecast_date": forecast_date,
                    "feature": feature_name,
                    "observed": observed_value,
                    "recomputed_full": full_value,
                    "recomputed_past": past_value,
                    "match_full": matches_full,
                    "match_past": matches_past,
                }
            )

    return pd.DataFrame.from_records(records)


def plot_feature_diagnostics(
    X: pd.DataFrame,
    output_dir: str | Path = "outputs",
    bins: int = 30,
    figsize: tuple[int, int] = (12, 8),
) -> dict[str, Path]:
    """Plot feature distributions and correlation heatmap.

    Saves two files:
    - feature_distributions.png
    - feature_correlation_heatmap.png
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("matplotlib is required for plotting diagnostics") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    numeric_X = X.select_dtypes(include="number")
    if numeric_X.empty:
        raise ValueError("X has no numeric columns to plot")

    ax = numeric_X.hist(bins=bins, figsize=figsize)
    for axes in ax:
        for a in axes:
            a.set_ylabel("count")
    dist_path = output_path / "feature_distributions.png"
    plt.tight_layout()
    plt.savefig(dist_path, dpi=150)
    plt.close()

    corr = numeric_X.corr()
    plt.figure(figsize=figsize)
    plt.imshow(corr, cmap="viridis", aspect="auto")
    plt.colorbar(label="correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    corr_path = output_path / "feature_correlation_heatmap.png"
    plt.savefig(corr_path, dpi=150)
    plt.close()

    return {
        "feature_distributions": dist_path,
        "feature_correlation_heatmap": corr_path,
    }
