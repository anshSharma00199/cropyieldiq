"""Model tests. They also guard against the 'static output' bug: different inputs MUST give different results."""

import json
from pathlib import Path

import pandas as pd

from app.services.ml_service import ModelService

ROOT = Path(__file__).resolve().parents[1]
_svc = None


def svc() -> ModelService:
    global _svc
    if _svc is None:
        _svc = ModelService()
        _svc.load(str(ROOT / "models/crop_model.joblib"), str(ROOT / "models/model_card.json"))
    return _svc


def profile(crop):
    r = svc().card["crop_ranges"][crop]
    return {f: r[f]["p50"] for f in svc().card["features"]}


def test_model_card_metrics_are_reasonable():
    m = svc().card["metrics"]
    assert m["test_accuracy"] >= 0.95 and m["accuracy_with_10pct_input_noise"] >= 0.90
    assert "label" not in svc().card["features"]  # no target leakage


def test_known_training_row_is_recommended_correctly():
    csv_old = ROOT / "data/Crop_recommendation.csv"
    if csv_old.exists():
        df = pd.read_csv(csv_old)
        row = df[df.label == "rice"].iloc[0]
        inputs = {f: float(row[f]) for f in svc().card["features"]}
    else:
        # Uses the median training profile of rice from the model card
        inputs = profile("rice")
    assert svc().predict(inputs)["recommendation"] == "rice"


def test_outputs_are_dynamic_across_crops():
    tops = {svc().predict(profile(c))["recommendation"] for c in svc().card["classes"]}
    assert len(tops) >= 20  # 22 crops -> (almost) 22 different answers


def test_changing_one_input_changes_the_output():
    base = profile("rice")
    dry = dict(base, rainfall=30.0, humidity=35.0)
    a, b = svc().predict(base), svc().predict(dry)
    assert a["recommendation"] != b["recommendation"] or a["top3"] != b["top3"]
    assert a["explanation"]["advice"] != b["explanation"]["advice"]


def test_out_of_range_input_warns():
    assert svc().predict(dict(profile("rice"), N=900.0))["warnings"]


def test_top3_probabilities_valid_and_sorted():
    top3 = svc().predict(profile("maize"))["top3"]
    probs = [t["probability"] for t in top3]
    assert probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1.0001


def test_sensitivity_curve_shape():
    s = svc().sensitivity(profile("rice"), "rainfall", points=10)
    assert len(s["curve"]) == 10 and s["base_recommendation"] in s["crops_shown"]
    assert json.dumps(s)  # JSON serialisable
