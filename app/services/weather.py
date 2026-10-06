"""Live weather via Open-Meteo (free, no key, non-commercial use). Results are cached.
If the API fails we raise WeatherUnavailable: the API returns 503 and NEVER invents default numbers."""

import datetime as dt
import logging

import requests

log = logging.getLogger("weather")
TIMEOUT = 10
WINDOW_DAYS = 30


class WeatherUnavailable(Exception):
    pass


class LocationNotFound(Exception):
    pass


def _get(url, params):
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("upstream error %s: %s", url, e)
        raise WeatherUnavailable("weather provider is unavailable") from e


def geocode(cache, district: str, state: str | None, ttl: int = 86400) -> dict:
    key = cache.make_key("geo", district.lower(), (state or "").lower())

    def produce():
        j = _get("https://geocoding-api.open-meteo.com/v1/search", {"name": district, "count": 10, "country_code": "IN"})
        results = j.get("results") or []
        if not results:
            raise LocationNotFound(f"could not locate '{district}'")
        match = [x for x in results if state and state.lower() in (x.get("admin1") or "").lower()]
        x = (match or results)[0]
        return {"lat": x["latitude"], "lon": x["longitude"], "name": x["name"], "state": x.get("admin1")}

    return cache.get_or_set(key, ttl, produce)


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def weather_features(cache, lat: float, lon: float, ttl: int = 1800) -> dict:
    """Mean temperature, mean humidity and total rainfall for the last 30 days (+ today)."""
    key = cache.make_key("wx", round(lat, 2), round(lon, 2), dt.date.today().isoformat())

    def produce():
        j = _get(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": lat,
                "longitude": lon,
                "timezone": "auto",
                "past_days": WINDOW_DAYS,
                "forecast_days": 1,
                "hourly": "temperature_2m,relative_humidity_2m",
                "daily": "precipitation_sum",
            },
        )
        try:
            temp = _mean(j["hourly"]["temperature_2m"])
            hum = _mean(j["hourly"]["relative_humidity_2m"])
            rain = sum(v for v in j["daily"]["precipitation_sum"] if v is not None)
        except (KeyError, TypeError) as e:
            raise WeatherUnavailable("unexpected weather response") from e
        if temp is None or hum is None:
            raise WeatherUnavailable("weather response had no data")
        return {
            "temperature": round(temp, 2),
            "humidity": round(hum, 2),
            "rainfall": round(rain, 2),
            "source": "open-meteo.com",
            "window_days": WINDOW_DAYS,
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }

    return cache.get_or_set(key, ttl, produce)
