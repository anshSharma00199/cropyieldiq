"""Crop model audit, probability calibration, and split-conformal classification sets.

Tasks covered (T1):
1. Audit: duplicates check, class balance, stratified split with fixed seed (42).
2. Metrics: top-1 accuracy, top-3 accuracy, log-loss, expected calibration error (ECE).
3. Calibration: CalibratedClassifierCV (Platt scaling / sigmoid) on calibration split.
4. Split-conformal sets: target coverage 90% (alpha=0.10) using calibration scores.
5. Evaluation: empirical coverage and average set size on test set.
6. Persistence: saves crop_model_calibrated.joblib, crop_conformal.json, and calibration curve plot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score
from sklearn.model_selection import train_test_split

from src import CROP_ALPHA, SEED
from src.conformal import classification_scores, conformal_quantile, predict_sets, set_metrics
from src.paths import CALIBRATION_PNG, CROP_CALIBRATED, CROP_CONFORMAL, CROP_CSV, MODELS, RESULTS

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
PARAMS = dict(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1)


def compute_ece(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected Calibration Error (ECE) for multi-class classification.

    Groups predictions into n_bins based on confidence (max predicted probability)
    and computes the weighted difference between confidence and accuracy.
    """
    classes = np.asarray(classes)
    y_true = np.asarray(y_true)
    y_pred_idx = np.argmax(proba, axis=1)
    y_pred = classes[y_pred_idx]
    confidences = np.max(proba, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        b_lo, b_hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences >= b_lo) & (confidences <= b_hi if i == n_bins - 1 else confidences < b_hi)
        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            diff = abs(bin_acc - bin_conf)
            ece += (count / len(y_true)) * diff
            bin_stats.append(
                {
                    "bin": f"[{b_lo:.1f}, {b_hi:.1f}]",
                    "count": count,
                    "accuracy": round(bin_acc, 4),
                    "confidence": round(bin_conf, 4),
                    "diff": round(diff, 4),
                }
            )
    return float(ece), bin_stats


def plot_calibration_comparison(
    yte: np.ndarray,
    raw_proba: np.ndarray,
    cal_proba: np.ndarray,
    classes: np.ndarray,
    out_path: Path,
    n_bins: int = 10,
) -> None:
    """Plots reliability diagram comparing uncalibrated vs calibrated probabilities."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    classes = np.asarray(classes)
    yte = np.asarray(yte)

    raw_conf = np.max(raw_proba, axis=1)
    raw_acc = (classes[np.argmax(raw_proba, axis=1)] == yte).astype(float)

    cal_conf = np.max(cal_proba, axis=1)
    cal_acc = (classes[np.argmax(cal_proba, axis=1)] == yte).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    raw_bin_accs, cal_bin_accs = [], []
    for i in range(n_bins):
        blo, bhi = bins[i], bins[i + 1]
        m_raw = (raw_conf >= blo) & (raw_conf <= bhi if i == n_bins - 1 else raw_conf < bhi)
        m_cal = (cal_conf >= blo) & (cal_conf <= bhi if i == n_bins - 1 else cal_conf < bhi)
        raw_bin_accs.append(np.mean(raw_acc[m_raw]) if np.sum(m_raw) > 0 else np.nan)
        cal_bin_accs.append(np.mean(cal_acc[m_cal]) if np.sum(m_cal) > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(bin_centers, raw_bin_accs, "s-", color="red", label="Uncalibrated RF")
    ax.plot(bin_centers, cal_bin_accs, "o-", color="green", label="Calibrated RF (Platt)")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Crop Recommendation Reliability Diagram")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved calibration comparison figure to {out_path}")


def run_crop_audit_and_calibration() -> dict:
    print("=" * 60)
    print("T1: Crop Model Audit, Calibration & Conformal Sets")
    print("=" * 60)

    # 1. Load dataset & audit
    df = pd.read_csv(CROP_CSV)
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    initial_rows = len(df)
    n_dupes = int(df.duplicated().sum())
    print(f"Loaded dataset: {CROP_CSV.name} ({initial_rows} rows)")
    print(f"Duplicate rows detected: {n_dupes}")
    if n_dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Rows after dropping duplicates: {len(df)}")

    # Class balance check
    class_counts = df[TARGET].value_counts()
    print("\nClass balance summary:")
    print(f"Number of classes: {len(class_counts)}")
    print(f"Min samples per class: {class_counts.min()}, Max samples: {class_counts.max()}")
    print("Samples per crop (first 5):")
    print(class_counts.head().to_dict())

    # 2. Stratified Split (Fixed seed)
    X = df[FEATURES]
    y = df[TARGET]
    # 80% development, 20% untouched test

    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

    # Split development into 75% fit and 25% holdout
    X_fit, X_holdout, y_fit, y_holdout = train_test_split(X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED)

    # Split holdout equally: probability calibration and conformal calibration
    X_cal, X_conf, y_cal, y_conf = train_test_split(X_holdout, y_holdout, test_size=0.50, stratify=y_holdout, random_state=SEED)

    print(
        f"\nStratified splits: Fit={len(X_fit)}, "
        f"Probability calibration={len(X_cal)}, "
        f"Conformal calibration={len(X_conf)}, Test={len(X_test)}"
    )
    # 3. Fit base Random Forest
    base_rf = RandomForestClassifier(**PARAMS).fit(X_fit, y_fit)
    classes = base_rf.classes_

    # Raw model evaluation on Test
    raw_test_pred = base_rf.predict(X_test)
    raw_test_proba = base_rf.predict_proba(X_test)
    raw_top1 = float(accuracy_score(y_test, raw_test_pred))
    raw_top3 = float(top_k_accuracy_score(y_test, raw_test_proba, k=3, labels=classes))
    raw_loss = float(log_loss(y_test, raw_test_proba, labels=classes))
    raw_ece, raw_bin_stats = compute_ece(y_test.to_numpy(), raw_test_proba, classes)

    print("\n--- Uncalibrated Base Model (Test Set) ---")
    print(f"Top-1 Accuracy : {raw_top1:.4f}")
    print(f"Top-3 Accuracy : {raw_top3:.4f}")
    print(f"Log-loss       : {raw_loss:.4f}")
    print(f"ECE            : {raw_ece:.4f}")

    # 4. Probability Calibration (Platt scaling / sigmoid on calibration set)
    try:
        from sklearn.frozen import FrozenEstimator

        cal_clf = CalibratedClassifierCV(estimator=FrozenEstimator(base_rf), method="sigmoid")
    except ImportError:
        cal_clf = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv="prefit")
    cal_clf.fit(X_cal, y_cal)

    cal_test_pred = cal_clf.predict(X_test)
    cal_test_proba = cal_clf.predict_proba(X_test)
    cal_top1 = float(accuracy_score(y_test, cal_test_pred))
    cal_top3 = float(top_k_accuracy_score(y_test, cal_test_proba, k=3, labels=classes))
    cal_loss = float(log_loss(y_test, cal_test_proba, labels=classes))
    cal_ece, cal_bin_stats = compute_ece(y_test.to_numpy(), cal_test_proba, classes)

    print("\n--- Calibrated Model (Test Set) ---")
    print(f"Top-1 Accuracy : {cal_top1:.4f}")
    print(f"Top-3 Accuracy : {cal_top3:.4f}")
    print(f"Log-loss       : {cal_loss:.4f}")
    print(f"ECE            : {cal_ece:.4f}")

    # 5. Split-conformal Prediction Sets with target coverage 90%
    # Use an independent conformal split, not the probability-calibration split.
    conf_scores = classification_scores(base_rf.predict_proba(X_conf), y_conf.to_numpy(), classes)

    qhat = conformal_quantile(conf_scores, CROP_ALPHA)

    print(f"\nConformal Calibration on {len(X_conf)} independent samples:")
    print(f"Target Coverage : {(1.0 - CROP_ALPHA) * 100:.0f}%")
    print(f"Conformal q_hat : {qhat:.4f}")

    # Evaluate prediction sets on the untouched test set
    test_sets = predict_sets(raw_test_proba, classes, qhat)
    conf_metrics = set_metrics(test_sets, y_test.to_numpy())

    print("\n--- Conformal Set Evaluation (Test Set) ---")
    print(f"Empirical Coverage : {conf_metrics['empirical_coverage']:.4f} (target: {1.0 - CROP_ALPHA:.2f})")
    print(f"Average Set Size   : {conf_metrics['average_set_size']:.2f}")

    # 6. Save artifacts
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    joblib.dump(cal_clf, CROP_CALIBRATED, compress=3)
    print(f"\nSaved calibrated model to {CROP_CALIBRATED}")

    conformal_meta = {
        "alpha": CROP_ALPHA,
        "target_coverage": 1.0 - CROP_ALPHA,
        "qhat": float(qhat),
        "empirical_test_coverage": float(conf_metrics["empirical_coverage"]),
        "average_test_set_size": float(conf_metrics["average_set_size"]),
        "n_fit": len(X_fit),
        "n_probability_calibration": len(X_cal),
        "n_test": len(X_test),
        "calibration_method": "CalibratedClassifierCV(sigmoid, prefit)",
        "classes": list(classes),
    }
    CROP_CONFORMAL.write_text(json.dumps(conformal_meta, indent=2))
    print(f"Saved conformal metadata to {CROP_CONFORMAL}")

    plot_calibration_comparison(y_test.to_numpy(), raw_test_proba, cal_test_proba, classes, CALIBRATION_PNG)

    results_summary = {
        "uncalibrated": {
            "top1_accuracy": raw_top1,
            "top3_accuracy": raw_top3,
            "log_loss": raw_loss,
            "ece": raw_ece,
        },
        "calibrated": {
            "top1_accuracy": cal_top1,
            "top3_accuracy": cal_top3,
            "log_loss": cal_loss,
            "ece": cal_ece,
        },
        "conformal": {
            "target_coverage": 1.0 - CROP_ALPHA,
            "qhat": qhat,
            "empirical_coverage": conf_metrics["empirical_coverage"],
            "average_set_size": conf_metrics["average_set_size"],
        },
    }
    (RESULTS / "crop_audit_metrics.json").write_text(json.dumps(results_summary, indent=2))
    return results_summary


if __name__ == "__main__":
    run_crop_audit_and_calibration()
