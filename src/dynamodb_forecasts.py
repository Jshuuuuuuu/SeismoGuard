from __future__ import annotations

import logging
import os
import re
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

try:
    import boto3
    from boto3.dynamodb.conditions import Attr
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover - exercised only when boto3 is not installed locally.
    boto3 = None
    Attr = None
    BotoCoreError = ClientError = Exception


AWS_REGION = "ap-southeast-1"
TABLE_NAME = "SeismoGuardForecasts"
MODEL_NAME = "sklearn_random_forest_pipeline"
PROVINCES = [
    "Davao De Oro",
    "Davao Del Norte",
    "Davao Del Sur",
    "Davao Occidental",
    "Davao Oriental",
]

logger = logging.getLogger(__name__)


def _require_boto3() -> None:
    """Raise a clear setup error if boto3 is not available in the local environment."""
    if boto3 is None or Attr is None:
        raise ImportError("boto3 is required for DynamoDB access. Install it with: pip install boto3")


def _dynamodb_table(table_name: str = TABLE_NAME, region_name: str = AWS_REGION) -> Any:
    """Return a DynamoDB table resource using boto3's resource interface."""
    _require_boto3()
    return boto3.resource("dynamodb", region_name=region_name).Table(table_name)


def _to_decimal(value: float | int | str | Decimal) -> Decimal:
    """Convert numeric values to Decimal without passing Python floats directly to DynamoDB."""
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _province_normalized(province: str) -> str:
    """Normalize a province name for use in the forecast_id partition key."""
    return re.sub(r"[^A-Za-z0-9]+", "", province.title())


def _forecast_id(province: str, forecast_date: date) -> str:
    """Build the ProvinceNormalized_YYYY-MM-DD DynamoDB partition key."""
    return f"{_province_normalized(province)}_{forecast_date.isoformat()}"


def _window_dates(forecast_date: date) -> tuple[date, date]:
    """Return the inclusive T plus 1 through T plus 7 forecast window dates."""
    return forecast_date + timedelta(days=1), forecast_date + timedelta(days=7)


def _prepare_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize event datetime, magnitude, and province columns."""
    required = {"datetime", "magnitude", "province"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"events_df is missing required columns: {sorted(missing)}")

    events = events_df.copy()
    events["datetime"] = pd.to_datetime(events["datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    events["magnitude"] = pd.to_numeric(events["magnitude"], errors="coerce")
    return events.dropna(subset=["datetime", "magnitude", "province"])


def create_table(table_name: str = TABLE_NAME, region_name: str = AWS_REGION) -> Any:
    """Create the SeismoGuard forecast DynamoDB table if needed and wait until it is active."""
    _require_boto3()
    dynamodb = boto3.resource("dynamodb", region_name=region_name)

    try:
        table = dynamodb.Table(table_name)
        table.load()
        logger.info("DynamoDB table already exists: %s", table_name)
        return table
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code != "ResourceNotFoundException":
            logger.exception("Failed to check DynamoDB table %s", table_name)
            raise
    except BotoCoreError:
        logger.exception("Failed to check DynamoDB table %s", table_name)
        raise

    try:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "forecast_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "forecast_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
        table.reload()
        logger.info("Created DynamoDB table: %s", table_name)
        return table
    except (BotoCoreError, ClientError):
        logger.exception("Failed to create DynamoDB table %s", table_name)
        raise


def write_forecast(
    forecast_date: date,
    forecast: dict[str, float],
    model: str = MODEL_NAME,
    table_name: str = TABLE_NAME,
    region_name: str = AWS_REGION,
) -> None:
    """Write one pending DynamoDB forecast record per province for a forecast date."""
    table = _dynamodb_table(table_name=table_name, region_name=region_name)
    window_start, window_end = _window_dates(forecast_date)

    for province, predicted_mag in forecast.items():
        item = {
            "forecast_id": _forecast_id(province, forecast_date),
            "province": province,
            "forecast_date": forecast_date.isoformat(),
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "predicted_mag": _to_decimal(predicted_mag),
            "actual_mag": None,
            "absolute_error": None,
            "event_count_7d": None,
            "model": model,
            "status": "pending",
        }

        try:
            table.put_item(Item=item)
            logger.info("Wrote pending forecast record: %s", item["forecast_id"])
        except (BotoCoreError, ClientError):
            logger.exception("Failed to write forecast record for %s on %s", province, forecast_date)


def update_actuals(
    forecast_date: date,
    events_df: pd.DataFrame,
    provinces: list[str] | tuple[str, ...] = tuple(PROVINCES),
    table_name: str = TABLE_NAME,
    region_name: str = AWS_REGION,
) -> None:
    """Evaluate pending forecast records against PHIVOLCS events and update actual values."""
    table = _dynamodb_table(table_name=table_name, region_name=region_name)
    events = _prepare_events(events_df)

    for province in provinces:
        forecast_id = _forecast_id(province, forecast_date)

        try:
            response = table.get_item(Key={"forecast_id": forecast_id})
        except (BotoCoreError, ClientError):
            logger.exception("Failed to read forecast record %s", forecast_id)
            continue

        item = response.get("Item")
        if item is None:
            logger.warning("No DynamoDB forecast record found for %s; skipping actual update", forecast_id)
            continue

        window_start = pd.Timestamp(item["window_start"])
        window_end_exclusive = pd.Timestamp(item["window_end"]) + pd.Timedelta(days=1)
        province_events = events[
            (events["province"] == province)
            & (events["datetime"] >= window_start)
            & (events["datetime"] < window_end_exclusive)
        ]

        event_count = int(len(province_events))
        actual_mag = Decimal("0") if province_events.empty else _to_decimal(province_events["magnitude"].max())
        predicted_mag = _to_decimal(item["predicted_mag"])
        absolute_error = abs(actual_mag - predicted_mag)

        try:
            table.update_item(
                Key={"forecast_id": forecast_id},
                UpdateExpression=(
                    "SET actual_mag = :actual_mag, "
                    "absolute_error = :absolute_error, "
                    "event_count_7d = :event_count_7d, "
                    "#status = :status"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":actual_mag": actual_mag,
                    ":absolute_error": absolute_error,
                    ":event_count_7d": event_count,
                    ":status": "evaluated",
                },
            )
            logger.info("Updated evaluated actuals for forecast record: %s", forecast_id)
        except (BotoCoreError, ClientError):
            logger.exception("Failed to update actuals for forecast record %s", forecast_id)


def get_validation_summary(table_name: str = TABLE_NAME, region_name: str = AWS_REGION) -> dict[str, Any]:
    """Return overall MAE, overall RMSE, and per-province MAE from evaluated DynamoDB records."""
    _require_boto3()
    table = _dynamodb_table(table_name=table_name, region_name=region_name)
    records: list[dict[str, Any]] = []
    scan_kwargs: dict[str, Any] = {"FilterExpression": Attr("status").eq("evaluated")}

    try:
        while True:
            response = table.scan(**scan_kwargs)
            records.extend(response.get("Items", []))

            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key
    except (BotoCoreError, ClientError):
        logger.exception("Failed to scan evaluated forecast records from %s", table_name)
        raise

    if not records:
        return {"overall_mae": None, "overall_rmse": None, "per_province_mae": {}}

    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, Decimal):
                record[key] = float(value)

    df = pd.DataFrame(records)
    errors = pd.to_numeric(df["absolute_error"], errors="coerce").dropna()
    overall_mae = float(errors.mean()) if not errors.empty else None
    overall_rmse = float((errors.pow(2).mean()) ** 0.5) if not errors.empty else None
    per_province_mae = (
        df.dropna(subset=["absolute_error"])
        .groupby("province")["absolute_error"]
        .mean()
        .astype(float)
        .to_dict()
    )

    return {
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "per_province_mae": per_province_mae,
    }


def run_weekly(
    forecast_date: date,
    events_df: pd.DataFrame,
    forecast: dict[str, float],
    model: str = MODEL_NAME,
    table_name: str = TABLE_NAME,
    region_name: str = AWS_REGION,
) -> None:
    """Evaluate last Monday's forecast first, then write today's new weekly forecast."""
    days_since_monday = forecast_date.weekday()
    days_back = 7 if days_since_monday == 0 else days_since_monday
    last_monday = forecast_date - timedelta(days=days_back)
    update_actuals(
        forecast_date=last_monday,
        events_df=events_df,
        table_name=table_name,
        region_name=region_name,
    )
    write_forecast(
        forecast_date=forecast_date,
        forecast=forecast,
        model=model,
        table_name=table_name,
        region_name=region_name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    dummy_forecast_date = date(2026, 5, 25)
    dummy_forecast = {
        "Davao De Oro": 3.2,
        "Davao Del Norte": 2.8,
        "Davao Del Sur": 3.6,
        "Davao Occidental": 2.5,
        "Davao Oriental": 4.1,
    }
    dummy_events = pd.DataFrame(
        [
            {
                "datetime": "2026-05-19 08:30:00",
                "magnitude": 3.4,
                "depth": 22,
                "latitude": 7.2,
                "longitude": 126.1,
                "province": "Davao De Oro",
                "location": "Dummy event (Davao De Oro)",
            },
            {
                "datetime": "2026-05-20 14:10:00",
                "magnitude": 4.0,
                "depth": 31,
                "latitude": 7.0,
                "longitude": 126.4,
                "province": "Davao Oriental",
                "location": "Dummy event (Davao Oriental)",
            },
        ]
    )

    if os.getenv("SEISMOGUARD_RUN_DYNAMODB_EXAMPLE") == "1":
        create_table()
        write_forecast(dummy_forecast_date - timedelta(days=7), dummy_forecast)
        update_actuals(dummy_forecast_date - timedelta(days=7), dummy_events)
        print(get_validation_summary())
        run_weekly(dummy_forecast_date, dummy_events, dummy_forecast)
    else:
        print("Example calls are ready. Set SEISMOGUARD_RUN_DYNAMODB_EXAMPLE=1 to run them against AWS.")
