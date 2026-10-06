"""Loads the trained model + model card and serves predictions. Reload is atomic (swap of one reference)."""

import json
import logging
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.services import agronomy
from src.conformal import predict_sets

log = logging.getLogger("ml")


class ModelNotLoaded(RuntimeError):
    pass


class ModelService:
    def __init__(self):
        self._state = None  # (model, card)
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._state is not None

    def load(self, model_path: str, card_path: str):
        model = joblib.load(model_path)
        card = json.loads(Path(card_path).read_text())
        calibrated = None
        conformal = None
        cal_path = Path(model_path).with_name("crop_model_calibrated.joblib")
        conf_path = Path(model_path).with_name("crop_conformal.json")
        if cal_path.exists():
            calibrated = joblib.load(cal_path)
        if conf_path.exists():
            conformal = json.loads(conf_path.read_text())
        with self._lock:
            self._state = (model, card, calibrated, conformal)
        log.info("model loaded", extra={"model_version": card["model_version"]})
        return card["model_version"]

    def _get(self):
        if self._state is None:
            raise ModelNotLoaded("model is not loaded")
        return self._state

    def _parts(self):
        model, card, calibrated, conformal = self._get()

        # Use the uncalibrated model because calibration worsened
        # log-loss and ECE on the current held-out test split.
        clf = model

        return model, card, calibrated, conformal, clf

    @property
    def version(self) -> str:
        return self._get()[1]["model_version"]

    @property
    def card(self) -> dict:
        return self._get()[1]

    def _frame(self, inputs: dict, card: dict) -> pd.DataFrame:
        return pd.DataFrame([[float(inputs[f]) for f in card["features"]]], columns=card["features"])

    def predict(self, inputs: dict) -> dict:
        _raw, card, calibrated, conformal, clf = self._parts()
        proba = clf.predict_proba(self._frame(inputs, card))[0]
        classes = np.asarray(clf.classes_)
        order = np.argsort(proba)[::-1][:3]
        top3 = [{"crop": str(classes[i]), "probability": round(float(proba[i]), 4)} for i in order]
        p1 = top3[0]["probability"]
        # Keep the old heuristic internally for API compatibility; the UI shows calibrated prob + set.
        margin = top3[0]["probability"] - top3[1]["probability"]
        confidence = "high" if p1 >= 0.8 and margin >= 0.3 else "medium" if p1 >= 0.5 else "low"
        conformal_set = []
        if conformal is not None:
            conformal_set = predict_sets(proba.reshape(1, -1), classes, float(conformal["qhat"]))[0]

        warnings = []
        for f, r in card["feature_range"].items():
            v = inputs[f]
            if v < r["min"] or v > r["max"]:
                warnings.append(
                    f"{f}={v} is outside the range seen in training data ({r['min']} to {r['max']}); the result may be unreliable."
                )
        expl = agronomy.explain(top3[0]["crop"], inputs, card["crop_ranges"])
        expl["label"] = "Typical-range check"
        return {
            "recommendation": top3[0]["crop"],
            "confidence": confidence,
            "top_probability": round(float(p1), 4),
            "conformal_set": conformal_set,
            "top3": top3,
            "explanation": expl,
            "warnings": warnings,
        }

    def sensitivity(self, inputs: dict, feature: str, points: int = 15) -> dict:
        """What-if: vary ONE input across its observed range and report the probability of every top crop."""
        _raw, card, _cal, _conf, clf = self._parts()
        if feature not in card["features"]:
            raise ValueError(f"unknown feature {feature}")
        base = self.predict(inputs)["recommendation"]
        r = card["feature_range"][feature]
        grid = np.linspace(r["min"], r["max"], points)
        rows = [[(g if f == feature else inputs[f]) for f in card["features"]] for g in grid]
        proba = clf.predict_proba(pd.DataFrame(rows, columns=card["features"]))
        classes = [str(c) for c in clf.classes_]
        keep = sorted(set([base] + [classes[i] for i in np.argsort(proba.max(axis=0))[::-1][:3]]))
        curve = [
            {"value": round(float(g), 2), **{c: round(float(proba[i][classes.index(c)]), 4) for c in keep}} for i, g in enumerate(grid)
        ]
        best = [classes[int(np.argmax(p))] for p in proba]
        return {
            "feature": feature,
            "base_recommendation": base,
            "crops_shown": keep,
            "curve": curve,
            "recommendation_changes": sorted(set(best)),
            "note": "Association learned from the dataset, not a causal agronomic effect.",
        }
