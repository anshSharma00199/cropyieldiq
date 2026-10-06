"""7-day forecast risk flags from documented thresholds."""

from __future__ import annotations

import json

import pandas as pd

from src.paths import CONFIG


def load_thresholds() -> dict:
    return json.loads((CONFIG / "risk_thresholds.json").read_text(encoding="utf-8"))


def flag_forecast(daily: pd.DataFrame, thr: dict | None = None) -> list[dict]:
    """daily columns: date, tmax_c, rain_mm."""
    thr = thr or load_thresholds()
    flags = []
    tmax = daily["tmax_c"].to_numpy()
    rain = daily["rain_mm"].to_numpy()
    hot_days = int((tmax >= thr["heat_stress_tmax_c"]).sum())
    if hot_days >= thr["heat_stress_days"]:
        flags.append(
            {
                "flag": "heat_stress",
                "detail": f"{hot_days} day(s) with tmax >= {thr['heat_stress_tmax_c']} C",
            }
        )
    heavy = int((rain >= thr["heavy_rain_daily_mm"]).sum())
    if heavy:
        flags.append(
            {
                "flag": "heavy_rain",
                "detail": f"{heavy} day(s) with rainfall >= {thr['heavy_rain_daily_mm']} mm",
            }
        )
    run = 0
    max_run = 0
    for r in rain:
        if r <= thr["dry_spell_max_daily_mm"]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    if max_run >= thr["dry_spell_days"]:
        flags.append(
            {
                "flag": "dry_spell",
                "detail": f"{max_run} consecutive day(s) with rainfall <= {thr['dry_spell_max_daily_mm']} mm",
            }
        )
    return flags
