"""7-day Open-Meteo forecast (ET0, rain, tmax). Failures raise; callers show a fallback message."""

from __future__ import annotations

import pandas as pd
import requests

FORECAST = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10


class ForecastUnavailable(Exception):
    pass


def seven_day(lat: float, lon: float) -> pd.DataFrame:
    try:
        r = requests.get(
            FORECAST,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "et0_fao_evapotranspiration,precipitation_sum,temperature_2m_max",
                "forecast_days": 7,
                "timezone": "auto",
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        d = r.json().get("daily") or {}
        return pd.DataFrame(
            {
                "date": d["time"],
                "et0_mm": d["et0_fao_evapotranspiration"],
                "rain_mm": d["precipitation_sum"],
                "tmax_c": d["temperature_2m_max"],
            }
        )
    except (requests.RequestException, KeyError, TypeError) as e:
        raise ForecastUnavailable("7-day forecast is temporarily unavailable") from e


def irrigation_plan(lat: float, lon: float, crop: str) -> tuple[pd.DataFrame, dict]:
    from src.planner import load_kc, water_balance

    cfg = load_kc()
    spec = cfg["crops"].get(crop.lower(), {"kc_mid": cfg["default_kc"], "root_m": 0.8})
    soil = cfg["soil"]
    taw = soil["taw_mm_per_m"] * spec.get("root_m", 0.8)
    daily = seven_day(lat, lon)
    bal = water_balance(
        daily["et0_mm"].tolist(),
        daily["rain_mm"].tolist(),
        kc=float(spec.get("kc_mid", cfg["default_kc"])),
        taw_mm=taw,
        p=float(soil["p_depletion"]),
    )
    bal.insert(0, "date", daily["date"].tolist())
    meta = {
        "source": cfg["source"],
        "kc_mid": spec.get("kc_mid", cfg["default_kc"]),
        "label": "agronomic calculation",
        "crop": crop,
    }
    return bal, meta
