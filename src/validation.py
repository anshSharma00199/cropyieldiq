"""Honest validation splits, baselines, tree models, conformal intervals, SHAP dump."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import SEED, YIELD_ALPHA
from src.conformal import add_intervals, interval_coverage, interval_quantile, mean_interval_width
from src.paths import MODELS, RESULTS, YIELD_BUNDLE, YIELD_SHAP_SUMMARY
from src.yield_pipeline import LEAKAGE_COLS, build_table

CAT = ["State Name", "Crop"]
NUM_BASE = ["Year", "Area_ha", "lag_yield"]


def numeric_features(df: pd.DataFrame) -> list[str]:
    extra = [c for c in ("temp_mean", "rain_sum") if c in df.columns and df[c].notna().any()]
    return NUM_BASE + extra


def assert_no_leakage(df: pd.DataFrame) -> None:
    present = [c for c in LEAKAGE_COLS if c in df.columns]
    if present:
        raise AssertionError(f"leakage columns still in modeling table: {present}")


def _xy(df: pd.DataFrame, num: list[str]):
    cols = CAT + num
    return df[cols], df["Yield_kg_per_ha"].to_numpy(dtype=float)


def make_preprocessor(num: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
            ("num", "passthrough", num),
        ]
    )


def make_splits(df: pd.DataFrame, protocol: str, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (train, test). Test years/states never appear in train for (b)/(c)."""
    n = len(df)
    if protocol == "random":
        rng = np.random.default_rng(kw.get("seed", SEED))
        perm = rng.permutation(n)
        cut = int(0.8 * n)
        tr, te = np.zeros(n, bool), np.zeros(n, bool)
        tr[perm[:cut]] = True
        te[perm[cut:]] = True
        return tr, te
    if protocol == "temporal":
        ycut = int(kw["year_cut"])
        tr = df["Year"].to_numpy() <= ycut
        te = df["Year"].to_numpy() > ycut
        return tr, te
    if protocol == "leave_states_out":
        held = set(kw["holdout_states"])
        te = df["State Name"].isin(held).to_numpy()
        tr = ~te
        return tr, te
    raise ValueError(f"unknown protocol {protocol}")


def temporal_cal_test(df: pd.DataFrame, train_end: int, cal_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = df["Year"].to_numpy()
    return y <= train_end, (y > train_end) & (y <= cal_end), y > cal_end


def metrics_block(y_true, y_hat, drought_mask=None) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
    out = {
        "r2": float(r2_score(y_true, y_hat)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_hat)),
        "n_test": int(len(y_true)),
    }
    if drought_mask is not None and drought_mask.any() and (~drought_mask).any():
        out["rmse_drought"] = float(np.sqrt(mean_squared_error(y_true[drought_mask], y_hat[drought_mask])))
        out["rmse_normal"] = float(np.sqrt(mean_squared_error(y_true[~drought_mask], y_hat[~drought_mask])))
        out["n_drought"] = int(drought_mask.sum())
        out["n_normal"] = int((~drought_mask).sum())
    else:
        out["rmse_drought"] = np.nan
        out["rmse_normal"] = np.nan
        out["n_drought"] = 0
        out["n_normal"] = int(len(y_true))
    return out


def _baseline_last_year(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    pred = test["lag_yield"].to_numpy(dtype=float)
    return pred


def _baseline_state_crop_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    means = train.groupby(["State Name", "Crop"])["Yield_kg_per_ha"].mean()
    global_mean = float(train["Yield_kg_per_ha"].mean())
    return np.array([means.get((r["State Name"], r["Crop"]), global_mean) for _, r in test.iterrows()], dtype=float)


def _sk_model(name: str, num: list[str]):
    pre = make_preprocessor(num)
    if name == "linreg":
        est = LinearRegression()
    elif name == "rf":
        est = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    elif name == "xgb":
        from xgboost import XGBRegressor

        est = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            objective="reg:squarederror",
        )
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("est", est)])


def fit_predict(name: str, num: list[str], train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if name == "last_year":
        return _baseline_last_year(train, test)
    if name == "state_crop_mean":
        return _baseline_state_crop_mean(train, test)
    model = _sk_model(name, num)
    Xtr, ytr = _xy(train, num)
    Xte, _ = _xy(test, num)
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def evaluate_protocol(df: pd.DataFrame, protocol: str, num: list[str], **split_kw) -> list[dict]:
    tr, te = make_splits(df, protocol, **split_kw)
    train, test = df.loc[tr], df.loc[te]
    if protocol == "temporal":
        assert test["Year"].min() > train["Year"].max()
    if protocol == "leave_states_out":
        assert set(test["State Name"]).isdisjoint(set(train["State Name"]))
    drought = test["is_drought"].to_numpy() if "is_drought" in test.columns else None
    y = test["Yield_kg_per_ha"].to_numpy(dtype=float)
    rows = []
    for name in ("last_year", "state_crop_mean", "linreg", "rf", "xgb"):
        yhat = fit_predict(name, num, train, test)
        row = {"protocol": protocol, "model": name, **split_kw, **metrics_block(y, yhat, drought)}
        rows.append(row)
        print(f"{protocol:16} {name:16} RMSE={row['rmse']:.3f} MAE={row['mae']:.3f} R2={row['r2']:.4f}")
    return rows


def rolling_origin(df: pd.DataFrame, num: list[str], cuts: list[int]) -> list[dict]:
    rows = []
    for cut in cuts:
        chunk = evaluate_protocol(df, "temporal", num, year_cut=cut)
        for r in chunk:
            r["protocol"] = f"rolling_origin_{cut}"
        rows.extend(chunk)
    return rows


def conformal_for_split(
    df: pd.DataFrame,
    num: list[str],
    model_name: str,
    train_mask,
    cal_mask,
    test_mask,
) -> dict:
    train, cal, test = df.loc[train_mask], df.loc[cal_mask], df.loc[test_mask]
    pipe = _sk_model(model_name, num)
    Xtr, ytr = _xy(train, num)
    pipe.fit(Xtr, ytr)
    yhat_cal = pipe.predict(_xy(cal, num)[0])
    qhat = interval_quantile(cal["Yield_kg_per_ha"].to_numpy() - yhat_cal, YIELD_ALPHA)
    yhat = pipe.predict(_xy(test, num)[0])
    lo, hi = add_intervals(yhat, qhat)
    y = test["Yield_kg_per_ha"].to_numpy(dtype=float)
    return {
        "qhat": qhat,
        "coverage": interval_coverage(y, lo, hi),
        "mean_width": mean_interval_width(lo, hi),
        "n_train": int(len(train)),
        "n_cal": int(len(cal)),
        "n_test": int(len(test)),
        "model": model_name,
        "pipe": pipe,
        "yhat": yhat,
        "lo": lo,
        "hi": hi,
        "y": y,
        "test": test,
    }


def shap_summary(pipe: Pipeline, df_bg: pd.DataFrame, num: list[str], out: Path) -> list[str]:
    import shap

    X, _ = _xy(df_bg, num)
    Xt = pipe.named_steps["pre"].transform(X)
    names = pipe.named_steps["pre"].get_feature_names_out()
    model = pipe.named_steps["est"]

    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, Xt)
        sv = explainer.shap_values(Xt)
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt)
    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=15)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [str(names[i]) for i in order[:10]]


def shap_top3_sentences(pipe: Pipeline, row: pd.DataFrame, num: list[str], bg: pd.DataFrame) -> list[str]:
    import shap

    # SHAP needs input features only; prediction rows have no target column.
    cols = CAT + num
    X_row = row[cols]
    X_bg = bg[cols]

    Xt = pipe.named_steps["pre"].transform(X_row)
    Xb = pipe.named_steps["pre"].transform(X_bg)

    names = list(pipe.named_steps["pre"].get_feature_names_out())

    estimator = pipe.named_steps["est"]

    if isinstance(estimator, (LinearRegression,)):
        explainer = shap.LinearExplainer(estimator, Xb)
    else:
        explainer = shap.TreeExplainer(estimator)

    sv = np.asarray(explainer.shap_values(Xt)).reshape(-1)
    order = np.argsort(np.abs(sv))[::-1][:3]

    med = np.median(Xb, axis=0)
    sentences = []

    for i in order:
        name = names[i]
        val = float(Xt[0, i])
        ref = float(med[i])
        direction = "raised" if sv[i] > 0 else "lowered"
        pretty = name.replace("num__", "").replace("cat__", "").replace("_", " ")

        sentences.append(
            f"{pretty} (value {val:.3g} vs typical {ref:.3g}) "
            f"{direction} the predicted yield. "
            "This is an association in the model, not a proven cause."
        )

    return sentences


def _plot_validation(tbl: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = tbl.pivot_table(index="model", columns="protocol", values="rmse", aggfunc="first")
    # Keep only the three named protocols if present; rolling-origin rows have longer names.
    keep = [c for c in pivot.columns if c in ("random", "temporal", "leave_states_out")]
    if keep:
        pivot[keep].plot(kind="bar", ax=ax)
    else:
        pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("RMSE (kg/ha)")
    ax.set_title("Yield RMSE by model and validation protocol")
    ax.legend(title="protocol")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run_validation(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        table_path = DATA / "yield_model_table.csv"
        if table_path.exists():
            df = pd.read_csv(table_path)
            print(f"Loaded existing table from {table_path} (n={len(df)})")
        else:
            df = build_table()
    assert_no_leakage(df)
    df = df.dropna(subset=numeric_features(df) + ["Yield_kg_per_ha"] + CAT).reset_index(drop=True)
    num = numeric_features(df)
    print("numeric features", num, "n=", len(df))

    year_cut = int(df["Year"].quantile(0.70))
    cal_cut = int(df["Year"].quantile(0.85))
    holdout = ["Punjab", "Tamil Nadu", "Kerala"]
    holdout = [s for s in holdout if s in set(df["State Name"])]
    print("temporal year_cut", year_cut, "cal_cut", cal_cut, "holdout", holdout)

    rows = []
    rows += evaluate_protocol(df, "random", num, seed=SEED)
    rows += evaluate_protocol(df, "temporal", num, year_cut=year_cut)
    rows += evaluate_protocol(df, "leave_states_out", num, holdout_states=holdout)
    cuts = sorted({int(df["Year"].quantile(q)) for q in (0.55, 0.65, 0.75)})
    rows += rolling_origin(df, num, cuts)

    tbl = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(RESULTS / "validation_results.csv", index=False)
    _plot_validation(tbl, RESULTS / "validation.png")
    print(tbl[["protocol", "model", "r2", "rmse", "mae", "rmse_drought", "rmse_normal", "n_test"]].to_string(index=False))

    # Best model by temporal RMSE (honest protocol, not the random split).
    temporal = tbl[tbl["protocol"] == "temporal"]
    best = temporal.sort_values("rmse").iloc[0]["model"]
    if best in ("last_year", "state_crop_mean"):
        best = "xgb"
    print("best_by_temporal_rmse", best)

    conf_rows = []
    # (a) random: split train further into fit/cal
    tr, te = make_splits(df, "random", seed=SEED)
    rng = np.random.default_rng(SEED)
    tr_idx = np.where(tr)[0]
    rng.shuffle(tr_idx)
    n_cal = max(20, int(0.25 * len(tr_idx)))
    cal_idx, fit_idx = tr_idx[:n_cal], tr_idx[n_cal:]
    fit_m = np.zeros(len(df), bool)
    fit_m[fit_idx] = True
    cal_m = np.zeros(len(df), bool)
    cal_m[cal_idx] = True
    c_rand = conformal_for_split(df, num, best, fit_m, cal_m, te)
    conf_rows.append(
        {
            "protocol": "random",
            "coverage": c_rand["coverage"],
            "mean_width": c_rand["mean_width"],
            "qhat": c_rand["qhat"],
            **{k: c_rand[k] for k in ("n_train", "n_cal", "n_test")},
        }
    )

    tr_m, cal_m, te_m = temporal_cal_test(df, year_cut, cal_cut)
    c_tmp = conformal_for_split(df, num, best, tr_m, cal_m, te_m)
    conf_rows.append(
        {
            "protocol": "temporal",
            "coverage": c_tmp["coverage"],
            "mean_width": c_tmp["mean_width"],
            "qhat": c_tmp["qhat"],
            **{k: c_tmp[k] for k in ("n_train", "n_cal", "n_test")},
        }
    )

    tr, te = make_splits(df, "leave_states_out", holdout_states=holdout)
    tr_idx = np.where(tr)[0]
    years = np.sort(df.loc[tr, "Year"].unique())
    cut_y = years[int(0.8 * len(years))]
    fit_m = tr & (df["Year"].to_numpy() <= cut_y)
    cal_m = tr & (df["Year"].to_numpy() > cut_y)
    c_lso = conformal_for_split(df, num, best, fit_m, cal_m, te)
    conf_rows.append(
        {
            "protocol": "leave_states_out",
            "coverage": c_lso["coverage"],
            "mean_width": c_lso["mean_width"],
            "qhat": c_lso["qhat"],
            **{k: c_lso[k] for k in ("n_train", "n_cal", "n_test")},
        }
    )

    conf_tbl = pd.DataFrame(conf_rows)
    conf_tbl.to_csv(RESULTS / "conformal_coverage.csv", index=False)
    print(conf_tbl.to_string(index=False))

    # Production bundle: temporal fit+cal (honest), conformal qhat from cal years.
    pipe = c_tmp["pipe"]

    # Use only temporal training-period observations as SHAP background.
    shap_train_mask = df["Year"].to_numpy() <= year_cut

    shap_train = df.loc[shap_train_mask]

    if shap_train.empty:
        raise ValueError("No training samples available for SHAP background.")

    bg = shap_train.sample(n=min(200, len(shap_train)), random_state=SEED)
    try:
        top_feats = shap_summary(pipe, bg, num, YIELD_SHAP_SUMMARY)
        print("shap top", top_feats)
    except Exception as e:
        top_feats = []
        print("SHAP summary failed:", e)

    rain_col = "rain_sum" if "rain_sum" in num else None
    # Store observed numeric ranges and central 98% ranges for inference warnings.
    range_columns = ["Area_ha", "lag_yield", "Year"]
    range_columns += [col for col in ("temp_mean", "rain_sum") if col in df.columns]

    feature_ranges = {}
    for col in range_columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if not values.empty:
            feature_ranges[col] = {
                "min": float(values.min()),
                "p01": float(values.quantile(0.01)),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
            }

    weather_medians = {}
    for col in ("temp_mean", "rain_sum"):
        if col in num and col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if values.empty:
                raise ValueError(f"No valid training values available for {col}")
            weather_medians[col] = float(values.median())

    bundle = {
        "model_name": best,
        "pipe": pipe,
        "qhat": float(c_tmp["qhat"]),
        "alpha": YIELD_ALPHA,
        "features_num": num,
        "features_cat": CAT,
        "year_cut": year_cut,
        "cal_cut": cal_cut,
        "holdout_states": holdout,
        "crops": sorted(df["Crop"].unique()),
        "states": sorted(df["State Name"].unique()),
        "rain_range": (
            float(df["rain_sum"].min()) if rain_col else None,
            float(df["rain_sum"].max()) if rain_col else None,
        ),
        "area_range": (float(df["Area_ha"].min()), float(df["Area_ha"].max())),
        "feature_ranges": feature_ranges,
        "lag_median": float(df["lag_yield"].median()),
        "weather_medians": weather_medians,
        "background": bg,
        "shap_top_global": top_feats,
        "conformal_table": conf_tbl,
        "version": "yld-{:%Y%m%d}-{}".format(datetime.now(timezone.utc), best),
        "disclaimer": "Decision support only. Not a certified agronomic prescription.",
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, YIELD_BUNDLE, compress=3)
    (RESULTS / "conformal_coverage.json").write_text(json.dumps(conf_rows, indent=2, default=float))
    print("wrote", YIELD_BUNDLE)
    return {"validation": tbl, "conformal": conf_tbl, "best": best, "year_cut": year_cut}


if __name__ == "__main__":
    run_validation()
