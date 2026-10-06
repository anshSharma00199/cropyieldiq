"""Train the crop-recommendation model from data/Crop_recommendation.csv.

Usage:  python ml/train.py [--data data/Crop_recommendation.csv] [--out models] [--upload]
Outputs: models/crop_model.joblib and models/model_card.json (metrics, ranges, limitations).
Deterministic (fixed seed) so CI and Docker builds reproduce the same model.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
SEED = 42
PARAMS = dict(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1)


COLUMN_MAP = {
    "Crop": "label",
    "crop": "label",
    "N_req_kg_per_ha": "N",
    "P_req_kg_per_ha": "P",
    "K_req_kg_per_ha": "K",
    "Temperature_C": "temperature",
    "Humidity_%": "humidity",
    "pH": "ph",
    "Rainfall_mm": "rainfall",
    "Yield_kg_per_ha": "yield_kg_per_ha",
}


def load_and_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.rename(columns=COLUMN_MAP, inplace=True)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        sys.exit(f"Dataset is missing columns: {missing}")
    if df[FEATURES + [TARGET]].isna().any().any():
        sys.exit("Dataset contains missing values; clean it first.")
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Crop_recommendation.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--upload", action="store_true", help="also upload artifacts to S3 (needs boto3 + env)")
    a = ap.parse_args()

    data_path, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_and_validate(data_path)
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()

    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    model = RandomForestClassifier(**PARAMS).fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)

    cv = cross_val_score(RandomForestClassifier(**PARAMS), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))

    # Robustness: add Gaussian noise (10% of each feature's std) to mimic imprecise weather/soil inputs
    rng = np.random.default_rng(SEED)
    noisy = Xte + rng.normal(0, 0.10 * Xte.std().values, Xte.shape)
    noise_acc = accuracy_score(yte, model.predict(noisy))

    metrics = {
        "test_accuracy": round(accuracy_score(yte, pred), 4),
        "test_macro_f1": round(f1_score(yte, pred, average="macro"), 4),
        "test_top3_accuracy": round(top_k_accuracy_score(yte, proba, k=3, labels=model.classes_), 4),
        "cv5_accuracy_mean": round(float(cv.mean()), 4),
        "cv5_accuracy_std": round(float(cv.std()), 4),
        "accuracy_with_10pct_input_noise": round(float(noise_acc), 4),
        "n_train": len(Xtr),
        "n_test": len(Xte),
    }

    crop_ranges = {
        crop: {
            f: {
                "p10": round(float(g[f].quantile(0.10)), 2),
                "p50": round(float(g[f].median()), 2),
                "p90": round(float(g[f].quantile(0.90)), 2),
            }
            for f in FEATURES
        }
        for crop, g in df.groupby(TARGET)
    }
    feature_range = {f: {"min": round(float(X[f].min()), 2), "max": round(float(X[f].max()), 2)} for f in FEATURES}
    importance = dict(sorted(zip(FEATURES, (round(float(v), 4) for v in model.feature_importances_)), key=lambda kv: -kv[1]))

    version = "rf-{:%Y%m%d}-{}".format(
        datetime.now(timezone.utc), hashlib.sha256((data_sha + json.dumps(PARAMS, default=str)).encode()).hexdigest()[:8]
    )
    card = {
        "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "RandomForestClassifier",
        "params": {k: v for k, v in PARAMS.items() if k != "n_jobs"},
        "sklearn_version": sklearn.__version__,
        "data_file": data_path.name,
        "data_sha256": data_sha,
        "n_rows": len(df),
        "features": FEATURES,
        "classes": list(model.classes_),
        "metrics": metrics,
        "feature_importance": importance,
        "feature_range": feature_range,
        "crop_ranges": crop_ranges,
        "limitations": [
            "Trained on the Kaggle Crop Recommendation dataset: 2,200 rows, 100 per crop, no location or date columns.",
            "Classes are perfectly balanced and well separated, so accuracy here is optimistic and is NOT validated across regions or years.",
            "It recommends WHICH crop suits soil and weather inputs. It does not predict yield (the dataset has no yield column).",
            "Rainfall units/time scale are not documented in the dataset; live mode uses a 30-day total as an assumption.",
            "Advice is rule-based decision support, not a certified agronomic prescription.",
        ],
    }
    joblib.dump(model, out / "crop_model.joblib", compress=3)
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps({"version": version, **metrics}, indent=2))
    print("importance:", importance)
    print("size MB:", round((out / "crop_model.joblib").stat().st_size / 1e6, 2))

    if a.upload:
        from app.services.storage import upload_artifacts

        upload_artifacts(out)


if __name__ == "__main__":
    main()
