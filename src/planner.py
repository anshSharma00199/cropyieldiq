"""FAO-56 style 7-day water balance and a 1-D search over an observed model input."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.paths import CONFIG


def load_kc() -> dict:
    return json.loads((CONFIG / "crop_coefficients.json").read_text(encoding="utf-8"))


def water_balance(
    et0_mm: list[float],
    rain_mm: list[float],
    kc: float,
    taw_mm: float,
    p: float = 0.5,
    start_depletion_mm: float = 0.0,
) -> pd.DataFrame:
    """Daily irrigation need so depletion does not exceed RAW = p * TAW.

    depletes by ETc, refilled by rain (capped at depletion), then irrigation
    brings depletion back to 0 when RAW would be exceeded (refill to field capacity).
    """
    raw = p * taw_mm
    dep = float(start_depletion_mm)
    rows = []
    for i, (et0, rain) in enumerate(zip(et0_mm, rain_mm), start=1):
        etc = kc * float(et0)
        rain = max(0.0, float(rain))
        dep = dep + etc - rain
        dep = min(max(dep, 0.0), taw_mm)
        irrig = 0.0
        if dep > raw:
            irrig = dep  # refill to field capacity
            dep = 0.0
        rows.append(
            {
                "day": i,
                "et0_mm": float(et0),
                "etc_mm": etc,
                "rain_mm": rain,
                "irrigation_mm": irrig,
                "depletion_end_mm": dep,
                "raw_mm": raw,
            }
        )
    return pd.DataFrame(rows)


def search_feature(
    predict_fn,
    base_row: dict,
    feature: str,
    grid: np.ndarray,
    cost_per_unit: float,
) -> pd.DataFrame:
    """Maximize interval lower bound minus cost * feature. Association only."""
    rows = []
    for x in grid:
        row = dict(base_row)
        row[feature] = float(x)
        yhat, lo, hi = predict_fn(row)
        score = float(lo) - float(cost_per_unit) * float(x)
        rows.append(
            {
                feature: float(x),
                "yhat": float(yhat),
                "lo": float(lo),
                "hi": float(hi),
                "score": score,
            }
        )
    return pd.DataFrame(rows)
