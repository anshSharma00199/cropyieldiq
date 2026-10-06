"""Yield model inference helpers used by the API (no FastAPI imports)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.conformal import add_intervals
from src.paths import YIELD_BUNDLE
from src.planner import search_feature
from src.validation import shap_top3_sentences


class YieldNotLoaded(RuntimeError):
    pass


class YieldService:
    def __init__(self):
        self.bundle = None

    def load(self, path: str | Path | None = None):
        p = Path(path) if path else YIELD_BUNDLE
        if not p.exists():
            raise YieldNotLoaded(f"missing {p}")
        self.bundle = joblib.load(p)
        return self.bundle["version"]

    def _need(self):
        if self.bundle is None:
            raise YieldNotLoaded("yield model is not loaded")
        return self.bundle

    def _frame(self, inputs: dict) -> pd.DataFrame:
        b = self._need()
        row = {
            "State Name": inputs["state"],
            "Crop": str(inputs["crop"]).strip().lower(),
            "Year": int(inputs.get("year", 2017)),
            "Area_ha": float(inputs["area_ha"]),
            "lag_yield": float(inputs.get("lag_yield", b["lag_median"])),
        }
        num = b["features_num"]
        if "temp_mean" in num:
            row["temp_mean"] = float(inputs.get("temp_mean", b.get("weather_medians", {}).get("temp_mean", np.nan)))
        if "rain_sum" in num:
            row["rain_sum"] = float(
                inputs.get(
                    "rainfall",
                    inputs.get("rain_sum", b.get("weather_medians", {}).get("rain_sum", np.nan)),
                )
            )
        return pd.DataFrame([row])

    def predict(self, inputs: dict) -> dict:
        b = self._need()
        crop = str(inputs["crop"]).strip().lower()
        warnings = []
        if crop not in b["crops"]:
            raise ValueError(f"crop '{crop}' is not in the yield model ({b['crops']})")
        if inputs["state"] not in b["states"]:
            warnings.append(f"State {inputs['state']} was not in training; one-hot will be unknown.")
        unused = []
        if inputs.get("fertilizer") is not None:
            unused.append("fertilizer")
        if inputs.get("pesticide") is not None:
            unused.append("pesticide")
        if unused:
            warnings.append(
                f"{', '.join(unused)} not used: same-year N/P/K requirement columns leaked yield "
                "(correlation ≈ 0.99) and pesticide was absent from the table."
            )
        row = self._frame(inputs)

        # Warn when inputs are outside the observed training distribution.
        feature_ranges = b.get("feature_ranges", {})
        input_values = {
            "Area_ha": row.iloc[0]["Area_ha"],
            "Year": row.iloc[0]["Year"],
            "lag_yield": row.iloc[0]["lag_yield"],
        }

        if "temp_mean" in row.columns:
            input_values["temp_mean"] = row.iloc[0]["temp_mean"]
        if "rain_sum" in row.columns:
            input_values["rain_sum"] = row.iloc[0]["rain_sum"]

        labels = {
            "Area_ha": "Area",
            "Year": "Year",
            "lag_yield": "Previous yield",
            "temp_mean": "Mean temperature",
            "rain_sum": "Rainfall",
        }

        for feature, value in input_values.items():
            limits = feature_ranges.get(feature)
            if limits is None or not np.isfinite(value):
                continue

            label = labels.get(feature, feature)
            if value < limits["min"] or value > limits["max"]:
                warnings.append(
                    f"{label} ({value:.2f}) is outside the full range "
                    f"seen in training data ({limits['min']:.2f} to "
                    f"{limits['max']:.2f}); this prediction may be unreliable."
                )
            elif value < limits["p01"] or value > limits["p99"]:
                warnings.append(
                    f"{label} ({value:.2f}) is outside the central 98% "
                    f"of training values ({limits['p01']:.2f} to "
                    f"{limits['p99']:.2f}); use caution interpreting this prediction."
                )

        num = b["features_num"]
        yhat = float(b["pipe"].predict(row[b["features_cat"] + num])[0])
        lo, hi = add_intervals(np.array([yhat]), b["qhat"])
        bg = b["background"]
        try:
            shap_s = shap_top3_sentences(b["pipe"], row, num, bg)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).exception("SHAP explanation failed: %s", exc)
            shap_s = [f"SHAP unavailable: {type(exc).__name__}: {exc}"]
        return {
            "yield_kg_per_ha": yhat,
            "interval_lo": float(lo[0]),
            "interval_hi": float(hi[0]),
            "alpha": b["alpha"],
            "shap_sentences": shap_s,
            "model_version": b["version"],
            "warnings": warnings,
            "disclaimer": b["disclaimer"],
        }

    def plan_rainfall(self, inputs: dict, cost_per_mm: float, points: int = 25) -> dict:
        b = self._need()
        if "rain_sum" not in b["features_num"]:
            raise ValueError("rainfall is not a model feature (weather fetch may have failed)")
        lo_r, hi_r = b["rain_range"]

        def pfn(row_inputs):
            pred = self.predict({**inputs, "rainfall": row_inputs["rain_sum"]})
            return pred["yield_kg_per_ha"], pred["interval_lo"], pred["interval_hi"]

        grid = np.linspace(lo_r, hi_r, points)
        curve = search_feature(pfn, {"rain_sum": inputs.get("rainfall", lo_r)}, "rain_sum", grid, cost_per_mm)
        best = curve.sort_values("score", ascending=False).iloc[0]
        return {
            "feature": "rain_sum",
            "label": "model-based association",
            "curve": curve.to_dict(orient="records"),
            "suggested": float(best["rain_sum"]),
            "suggested_lo": float(best["lo"]),
            "note": (
                "Same-year fertiliser columns were dropped as target leakage. "
                "This search varies annual rainfall inside the observed training range "
                "and maximises the conformal lower bound minus your cost term. "
                "Not a causal irrigation or fertiliser prescription."
            ),
        }
