"""Open-Meteo archive fetch, cached to disk. Failures never invent numbers."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from src.paths import CACHE, CONFIG

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = 30


class WeatherFetchError(Exception):
    pass


def load_centroids() -> dict:
    path = CONFIG / "state_centroids.json"
    return json.loads(path.read_text(encoding="utf-8"))["states"]


def _annual_from_daily(daily: dict, state: str) -> pd.DataFrame:
    dates = pd.to_datetime(daily["time"])
    temp = pd.to_numeric(daily.get("temperature_2m_mean"), errors="coerce")
    rain = pd.to_numeric(daily.get("precipitation_sum"), errors="coerce")
    df = pd.DataFrame({"date": dates, "temp": temp, "rain": rain})
    df["Year"] = df["date"].dt.year
    g = df.groupby("Year", as_index=False).agg(temp_mean=("temp", "mean"), rain_sum=("rain", "sum"))
    g["State Name"] = state
    return g


def fetch_state_annual(state: str, lat: float, lon: float, start: str, end: str, cache_dir: Path | None = None) -> pd.DataFrame:
    cache_dir = cache_dir or CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in state)
    path = cache_dir / f"openmeteo_{safe}_{start}_{end}.csv"
    if path.exists():
        return pd.read_csv(path)

    last_err = None
    for attempt in range(4):
        try:
            r = requests.get(
                ARCHIVE,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start,
                    "end_date": end,
                    "daily": "temperature_2m_mean,precipitation_sum",
                    "timezone": "auto",
                },
                timeout=TIMEOUT,
            )
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 20))
                wait = max(retry_after, 15 + attempt * 10)
                print(f"Open-Meteo 429 rate limit for {state}; waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            daily = r.json().get("daily") or {}
            if "time" not in daily:
                raise WeatherFetchError(f"unexpected archive payload for {state}")
            out = _annual_from_daily(daily, state)
            out.to_csv(path, index=False)
            return out
        except requests.HTTPError as e:
            last_err = e
            if e.response is not None and e.response.status_code == 429:
                time.sleep(20)
            else:
                time.sleep(3 * (attempt + 1))
        except (requests.RequestException, WeatherFetchError, ValueError, KeyError) as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise WeatherFetchError(f"Open-Meteo archive failed for {state}: {last_err}")


def fetch_all_states(years: tuple[int, int], states: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    centroids = load_centroids()
    wanted = states or list(centroids)
    start, end = f"{years[0]}-01-01", f"{years[1]}-12-31"
    frames, failures = [], []
    for state in wanted:
        loc = centroids.get(state)
        if not loc:
            failures.append(state)
            print(f"NO_CENTROID {state}")
            continue
        try:
            frames.append(fetch_state_annual(state, loc["lat"], loc["lon"], start, end))
            print(f"WEATHER_OK {state}")
            time.sleep(2.0)  # Gentle spacing between API queries
        except WeatherFetchError as e:
            failures.append(state)
            print(f"WEATHER_FAIL {state}: {e}")
    if not frames:
        return pd.DataFrame(columns=["Year", "temp_mean", "rain_sum", "State Name"]), failures
    return pd.concat(frames, ignore_index=True), failures
