"""T1: crop classifier audit, probability calibration, split-conformal sets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src import CROP_ALPHA, SEED
from src.conformal import classification_scores, conformal_quantile, predict_sets, set_metrics
from src.paths import CROP_CALIBRATED, CROP_CARD, CROP_CONFORMAL, CROP_CSV, CROP_MODEL, RESULTS

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
PARAMS = dict(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1)


def expected_calibration_error(y_true, proba, classes, n_bins: int = 10) -> float:
    """Max-probability ECE (Naeini et al.): |acc(bin) - conf(bin)| weighted by bin size."""
    classes = np.asarray(classes)
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    y_true = np.asarray(y_true)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (conf >= bins[i]) & (conf <= bins[i + 1])
        else:
            mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def load_crop_csv(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else CROP_CSV
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Crop dataset is missing columns: {missing}")
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    return df


def _reliability_plot(y_true, proba_raw, proba_cal, classes, out: Path):
    y_true = np.asarray(y_true)
    pred_r = classes[proba_raw.argmax(axis=1)]
    pred_c = classes[proba_cal.argmax(axis=1)]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, conf, correct in [
        ("uncalibrated", proba_raw.max(axis=1), (pred_r == y_true).astype(float)),
        ("calibrated", proba_cal.max(axis=1), (pred_c == y_true).astype(float)),
    ]:
        frac, mean_pred = calibration_curve(correct, conf, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Crop model reliability (max-probability)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def train_crop(data_path: Path | None = None, out_dir: Path | None = None) -> dict:
    df = load_crop_csv(data_path)
    n_raw = len(df)
    n_dups = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    balance = df[TARGET].value_counts().sort_index().to_dict()

    X = df[FEATURES]
    y = df[TARGET].values
    # Separate sets: train / probability-calibration / conformal-calibration / test.
    # Using the same rows for Platt scaling and conformal scores would make coverage optimistic.
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.50, stratify=y, random_state=SEED)
    X_prob, X_rest2, y_prob, y_rest2 = train_test_split(X_rest, y_rest, test_size=0.60, stratify=y_rest, random_state=SEED)
    X_conf, X_te, y_conf, y_te = train_test_split(X_rest2, y_rest2, test_size=0.50, stratify=y_rest2, random_state=SEED)

    rf = RandomForestClassifier(**PARAMS).fit(X_tr, y_tr)
    calibrated = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv="prefit").fit(X_prob, y_prob)

    proba_raw = rf.predict_proba(X_te)
    proba_cal = calibrated.predict_proba(X_te)
    classes = calibrated.classes_
    pred = calibrated.predict(X_te)

    rng = np.random.default_rng(SEED)
    noisy = X_te + rng.normal(0, 0.10 * X_te.std().values, X_te.shape)
    noise_acc = float(accuracy_score(y_te, calibrated.predict(noisy)))
    cv = cross_val_score(
        RandomForestClassifier(**PARAMS),
        X_tr,
        y_tr,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    )

    scores = classification_scores(calibrated.predict_proba(X_conf), y_conf, classes)
    qhat = conformal_quantile(scores, CROP_ALPHA)
    sets = predict_sets(proba_cal, classes, qhat)
    sm = set_metrics(sets, y_te)

    metrics = {
        "n_raw": n_raw,
        "n_duplicates_removed": n_dups,
        "n_after_dedup": len(df),
        "n_train": int(len(X_tr)),
        "n_prob_cal": int(len(X_prob)),
        "n_conf_cal": int(len(X_conf)),
        "n_test": int(len(X_te)),
        "class_counts": {k: int(v) for k, v in balance.items()},
        "seed": SEED,
        "uncalibrated": {
            "top1_accuracy": float(accuracy_score(y_te, rf.predict(X_te))),
            "top3_accuracy": float(top_k_accuracy_score(y_te, proba_raw, k=3, labels=classes)),
            "log_loss": float(log_loss(y_te, proba_raw, labels=classes)),
            "ece": expected_calibration_error(y_te, proba_raw, classes),
        },
        "calibrated": {
            "top1_accuracy": float(accuracy_score(y_te, pred)),
            "top3_accuracy": float(top_k_accuracy_score(y_te, proba_cal, k=3, labels=classes)),
            "log_loss": float(log_loss(y_te, proba_cal, labels=classes)),
            "ece": expected_calibration_error(y_te, proba_cal, classes),
        },
        "conformal": {
            "alpha": CROP_ALPHA,
            "target_coverage": 1.0 - CROP_ALPHA,
            "qhat": qhat,
            **sm,
        },
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
    importance = dict(
        sorted(
            zip(FEATURES, (round(float(v), 4) for v in rf.feature_importances_)),
            key=lambda kv: -kv[1],
        )
    )
    data_path = Path(data_path) if data_path else CROP_CSV
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    version = "rf-cal-{:%Y%m%d}-{}".format(
        datetime.now(timezone.utc),
        hashlib.sha256((data_sha + json.dumps(PARAMS, default=str)).encode()).hexdigest()[:8],
    )

    out = Path(out_dir) if out_dir else CROP_MODEL.parent
    out.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _reliability_plot(y_te, proba_raw, proba_cal, classes, RESULTS / "crop_calibration.png")

    joblib.dump(rf, out / CROP_MODEL.name, compress=3)
    joblib.dump(calibrated, out / CROP_CALIBRATED.name, compress=3)
    conformal_blob = {
        "qhat": qhat,
        "alpha": CROP_ALPHA,
        "classes": list(map(str, classes)),
        "empirical_coverage": sm["empirical_coverage"],
        "average_set_size": sm["average_set_size"],
    }
    (out / CROP_CONFORMAL.name).write_text(json.dumps(conformal_blob, indent=2))

    # Keep the original card fields so the API /meta contract stays stable;
    # calibrated + conformal numbers live under extra keys.
    card = {
        "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "CalibratedClassifierCV(RandomForestClassifier, sigmoid, prefit)",
        "params": {k: v for k, v in PARAMS.items() if k != "n_jobs"},
        "sklearn_version": sklearn.__version__,
        "data_file": data_path.name,
        "data_sha256": data_sha,
        "n_rows": len(df),
        "features": FEATURES,
        "classes": list(map(str, classes)),
        "metrics": {
            "test_accuracy": round(metrics["calibrated"]["top1_accuracy"], 4),
            "test_macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
            "test_top3_accuracy": round(metrics["calibrated"]["top3_accuracy"], 4),
            "cv5_accuracy_mean": round(float(cv.mean()), 4),
            "cv5_accuracy_std": round(float(cv.std()), 4),
            "accuracy_with_10pct_input_noise": round(noise_acc, 4),
            "n_train": metrics["n_train"],
            "n_test": metrics["n_test"],
            "log_loss": metrics["calibrated"]["log_loss"],
            "ece": metrics["calibrated"]["ece"],
            "conformal_coverage": sm["empirical_coverage"],
            "conformal_avg_set_size": sm["average_set_size"],
        },
        "audit": metrics,
        "feature_importance": importance,
        "feature_range": feature_range,
        "crop_ranges": crop_ranges,
        "limitations": [
            "Trained on the Kaggle Crop Recommendation dataset (no location or date columns).",
            "Classes are nearly perfectly separated, so accuracy is optimistic and is NOT validated across regions or years.",
            "It recommends WHICH crop suits soil and weather inputs. Yield is a separate model.",
            "Conformal sets assume exchangeable calibration and test points; that assumption is not geographically testable here.",
            "Advice is rule-based decision support, not a certified agronomic prescription.",
        ],
    }
    (out / CROP_CARD.name).write_text(json.dumps(card, indent=2))
    print(
        json.dumps(
            {
                "version": version,
                **metrics["calibrated"],
                "conformal": metrics["conformal"],
                "n_duplicates_removed": n_dups,
                "class_counts_min": min(balance.values()),
                "class_counts_max": max(balance.values()),
            },
            indent=2,
        )
    )
    return metrics


if __name__ == "__main__":
    train_crop()
