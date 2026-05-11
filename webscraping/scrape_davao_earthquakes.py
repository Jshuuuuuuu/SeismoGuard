from __future__ import annotations

import argparse
import csv
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
import urllib3
from bs4 import BeautifulSoup
from requests.exceptions import SSLError


START_DATE = date(2026, 2, 1)
PHIVOLCS_LATEST_URL = "https://earthquake.phivolcs.dost.gov.ph/"
PHIVOLCS_MONTHLY_URL = "https://earthquake.phivolcs.dost.gov.ph/EQLatest-Monthly/{year}/{year}_{month}.html"
DEFAULT_OUTPUT = Path("data/raw/earthquakes/davao_region_2026_02_01_to_present.csv")

DAVAO_REGION_KEYWORDS = (
    "davao de oro",
    "davao del norte",
    "davao del sur",
    "davao occidental",
    "davao oriental",
    "city of davao",
    "davao city",
)

PHIVOLCS_FIELD_LABELS = (
    "Date/Time",
    "Location",
    "Depth of Focus",
    "Origin",
    "Magnitude",
    "Reported Intensities",
    "Expecting Damage",
    "Expecting Aftershocks",
    "Issued On",
    "Prepared by",
)

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


@dataclass(frozen=True)
class EarthquakeEvent:
    datetime: str
    latitude: float
    longitude: float
    depth: float
    magnitude: float
    magnitude_type: str
    location: str
    source_url: str


def normalize_space(value: str) -> str:
    return (
        re.sub(r"\s+", " ", value.replace("\xa0", " "))
        .replace("Â°", "°")
        .strip()
    )


def parse_phivolcs_datetime(value: str) -> datetime:
    match = re.search(
        r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\s*-\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AP]M)",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Could not parse PHIVOLCS date/time: {value!r}")

    day, month_name, year, hour, minute, second, am_pm = match.groups()
    month = MONTHS[month_name.lower()]
    hour_i = int(hour)
    if am_pm.upper() == "PM" and hour_i != 12:
        hour_i += 12
    elif am_pm.upper() == "AM" and hour_i == 12:
        hour_i = 0

    return datetime(
        int(year),
        month,
        int(day),
        hour_i,
        int(minute),
        int(second or "0"),
    )


def parse_location(value: str) -> tuple[float, float, str]:
    match = re.search(
        r"([+-]?\d+(?:\.\d+)?)\s*°?\s*N\s*,\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*E\s*-\s*(.+)",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Could not parse PHIVOLCS location: {value!r}")

    latitude, longitude, location_text = match.groups()
    return float(latitude), float(longitude), normalize_space(location_text)


def parse_depth(value: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", value)
    if not match:
        raise ValueError(f"Could not parse depth: {value!r}")
    return float(match.group(1))


def parse_magnitude(value: str) -> tuple[float, str]:
    match = re.search(r"\b([A-Za-z]+)?\s*(\d+(?:\.\d+)?)\b", value)
    if not match:
        raise ValueError(f"Could not parse magnitude: {value!r}")
    mag_type, magnitude = match.groups()
    return float(magnitude), (mag_type or "").strip()


def text_after_label(page_text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}(?:\s*\(Km\))?\s*:\s*", page_text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not find label: {label}")

    value_start = match.end()
    next_label_start = len(page_text)
    for next_label in PHIVOLCS_FIELD_LABELS:
        if next_label.casefold() == label.casefold():
            continue
        next_match = re.search(
            rf"{re.escape(next_label)}(?:\s*\(Km\))?\s*:",
            page_text[value_start:],
            flags=re.IGNORECASE,
        )
        if next_match:
            next_label_start = min(next_label_start, value_start + next_match.start())

    return normalize_space(page_text[value_start:next_label_start])


def parse_event_page(html: str, source_url: str) -> EarthquakeEvent:
    soup = BeautifulSoup(html, "html.parser")
    page_text = normalize_space(soup.get_text(" "))

    event_dt = parse_phivolcs_datetime(text_after_label(page_text, "Date/Time"))
    latitude, longitude, location = parse_location(text_after_label(page_text, "Location"))
    depth = parse_depth(text_after_label(page_text, "Depth of Focus"))
    magnitude, magnitude_type = parse_magnitude(text_after_label(page_text, "Magnitude"))

    return EarthquakeEvent(
        datetime=event_dt.strftime("%Y-%m-%d %H:%M:%S"),
        latitude=latitude,
        longitude=longitude,
        depth=depth,
        magnitude=magnitude,
        magnitude_type=magnitude_type,
        location=location,
        source_url=source_url,
    )


def is_davao_region_event(event: EarthquakeEvent) -> bool:
    searchable = event.location.casefold()
    return any(keyword in searchable for keyword in DAVAO_REGION_KEYWORDS)


def get_url(session: requests.Session, url: str, **kwargs: object) -> requests.Response:
    try:
        response = session.get(url, **kwargs)
        response.raise_for_status()
    except SSLError as exc:
        raise RuntimeError(
            "SSL certificate verification failed while connecting to PHIVOLCS. "
            "Run again with --no-verify-ssl if you trust this PHIVOLCS connection."
        ) from exc
    return response


def month_starts(start_date: date, end_date: date) -> Iterable[date]:
    current = date(start_date.year, start_date.month, 1)
    final = date(end_date.year, end_date.month, 1)
    while current <= final:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def source_urls_for_range(start_date: date, end_date: date) -> list[str]:
    today = date.today()
    urls: list[str] = []
    for month_start in month_starts(start_date, end_date):
        if month_start.year == today.year and month_start.month == today.month:
            urls.append(PHIVOLCS_LATEST_URL)
            continue

        month_name = MONTH_NAMES[month_start.month - 1]
        urls.append(PHIVOLCS_MONTHLY_URL.format(year=month_start.year, month=month_name))
    return urls


def parse_table_event(row: BeautifulSoup, page_url: str) -> EarthquakeEvent | None:
    cells = [normalize_space(cell.get_text(" ", strip=True)) for cell in row.find_all(["td", "th"])]
    if len(cells) != 6 or cells[0].casefold().startswith("date - time"):
        return None

    try:
        event_dt = parse_phivolcs_datetime(cells[0])
        latitude = float(cells[1])
        longitude = float(cells[2])
        depth = float(cells[3])
        magnitude = float(cells[4])
    except ValueError:
        return None

    source_url = page_url
    anchor = row.find("a", href=True)
    if anchor:
        href = anchor["href"].replace("\\", "/")
        source_url = urljoin(page_url, href)

    return EarthquakeEvent(
        datetime=event_dt.strftime("%Y-%m-%d %H:%M:%S"),
        latitude=latitude,
        longitude=longitude,
        depth=depth,
        magnitude=magnitude,
        magnitude_type="",
        location=cells[5],
        source_url=source_url,
    )


def parse_table_events(html: str, page_url: str) -> list[EarthquakeEvent]:
    soup = BeautifulSoup(html, "html.parser")
    events: list[EarthquakeEvent] = []
    seen: set[tuple[str, float, float, float, float, str]] = set()
    for row in soup.find_all("tr"):
        event = parse_table_event(row, page_url)
        if event is None:
            continue

        key = (
            event.datetime,
            event.latitude,
            event.longitude,
            event.depth,
            event.magnitude,
            event.location,
        )
        if key in seen:
            continue
        seen.add(key)
        events.append(event)

    return events


def scrape_events(start_date: date, end_date: date, verify_ssl: bool) -> list[EarthquakeEvent]:
    session = requests.Session()
    session.verify = verify_ssl
    session.headers.update(
        {
            "User-Agent": (
                "SeismoGuard Davao Region earthquake scraper "
                "(research use; contact project maintainer)"
            )
        }
    )
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    events: list[EarthquakeEvent] = []
    seen: set[tuple[str, float, float, float, float, str]] = set()
    for url in source_urls_for_range(start_date, end_date):
        response = get_url(session, url, timeout=30)
        for event in parse_table_events(response.text, response.url):
            event_date = datetime.strptime(event.datetime, "%Y-%m-%d %H:%M:%S").date()
            key = (
                event.datetime,
                event.latitude,
                event.longitude,
                event.depth,
                event.magnitude,
                event.location,
            )
            if key in seen:
                continue
            if start_date <= event_date <= end_date and is_davao_region_event(event):
                seen.add(key)
                events.append(event)

    return sorted(events, key=lambda item: item.datetime)


def write_csv(events: Iterable[EarthquakeEvent], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(event) for event in events]
    fieldnames = [
        "datetime",
        "latitude",
        "longitude",
        "depth",
        "magnitude",
        "magnitude_type",
        "location",
        "source_url",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape PHIVOLCS earthquake bulletins for Davao Region events."
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE.isoformat(),
        help="Inclusive start date in YYYY-MM-DD format. Default: 2026-02-01.",
    )
    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
        help="Inclusive end date in YYYY-MM-DD format. Default: today's local date.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"CSV output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=75,
        help="Ignored; kept for compatibility with the older PHIVOLCS menu scraper.",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification for PHIVOLCS requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    output_path = Path(args.output)

    if end_date < start_date:
        raise ValueError("end-date must be on or after start-date")

    events = scrape_events(
        start_date=start_date,
        end_date=end_date,
        verify_ssl=not args.no_verify_ssl,
    )
    write_csv(events, output_path)
    print(f"Saved {len(events)} Davao Region earthquake events to {output_path}")


if __name__ == "__main__":
    main()
