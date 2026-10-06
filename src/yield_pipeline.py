"""Clean the historical yield table without target leakage.

Dropped as features (same-row target leakage or deterministic functions of it):
- N_req_kg_per_ha, P_req_kg_per_ha, K_req_kg_per_ha  (corr with yield ≈ 0.98–1.00)
- Total_N_kg, Total_P_kg, Total_K_kg  (= req_per_ha * Area_ha exactly)
There is no Production column; Yield_kg_per_ha is already the target.
Dataset Temperature/Rainfall/etc. are crop-level constants (4 unique values = 4 crops),
so they are not used. Year-varying climate comes from Open-Meteo (optional, cached).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.paths import DATA, YIELD_CSV
from src.weather_hist import fetch_all_states

LEAKAGE_COLS = [
    "N_req_kg_per_ha",
    "P_req_kg_per_ha",
    "K_req_kg_per_ha",
    "Total_N_kg",
    "Total_P_kg",
    "Total_K_kg",
]
FAKE_WEATHER_COLS = [
    "Temperature_C",
    "Humidity_%",
    "pH",
    "Rainfall_mm",
    "Wind_Speed_m_s",
    "Solar_Radiation_MJ_m2_day",
]
TOP_CROPS = 3
MIN_YEARS_PER_STATE_CROP = 25


def load_raw(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or YIELD_CSV)


def aggregate_state_crop_year(raw: pd.DataFrame) -> pd.DataFrame:
    """Area-weighted yield so lag features match state+crop as specified."""
    df = raw.copy()
    df = df[(df["Area_ha"] > 0) & (df["Yield_kg_per_ha"] > 0)]
    df["Crop"] = df["Crop"].astype(str).str.strip().str.lower()
    df["yw"] = df["Yield_kg_per_ha"] * df["Area_ha"]
    g = df.groupby(["State Name", "Crop", "Year"], as_index=False).agg(
        Area_ha=("Area_ha", "sum"),
        yw=("yw", "sum"),
        n_districts=("Dist Name", "nunique"),
    )
    g["Yield_kg_per_ha"] = g["yw"] / g["Area_ha"]
    return g.drop(columns=["yw"])


def choose_crops_and_states(g: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    crop_counts = g.groupby("Crop").agg(rows=("Year", "size"), years=("Year", "nunique"), states=("State Name", "nunique"))
    crop_counts = crop_counts.sort_values(["rows", "years"], ascending=False)
    print("=== crop coverage (state-crop-year rows) ===")
    print(crop_counts.to_string())
    crops = list(crop_counts.head(TOP_CROPS).index)
    print(f"Selected crops: {crops}")

    sub = g[g["Crop"].isin(crops)].copy()
    sc = sub.groupby(["State Name", "Crop"]).agg(n_years=("Year", "nunique"), n_rows=("Year", "size")).reset_index()
    print("=== state x crop year counts ===")
    print(sc.sort_values("n_years", ascending=False).to_string(index=False))

    keep_states = sc.groupby("State Name")["n_years"].max().loc[lambda s: s >= MIN_YEARS_PER_STATE_CROP].index.tolist()
    print(f"States with >= {MIN_YEARS_PER_STATE_CROP} years for at least one selected crop: {len(keep_states)}")
    print(sorted(keep_states))
    out = sub[sub["State Name"].isin(keep_states)].copy()
    return crops, keep_states, out


def add_lag_yield(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["State Name", "Crop", "Year"]).copy()
    df["lag_yield"] = df.groupby(["State Name", "Crop"])["Yield_kg_per_ha"].shift(1)
    # Only keep rows where the lag is from the previous calendar year (no peeking across gaps).
    prev_year = df.groupby(["State Name", "Crop"])["Year"].shift(1)
    df.loc[prev_year != df["Year"] - 1, "lag_yield"] = pd.NA
    return df.dropna(subset=["lag_yield"]).reset_index(drop=True)


def add_weather(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, list[str]]:
    years = (int(df["Year"].min()), int(df["Year"].max()))
    wx, failures = fetch_all_states(years, sorted(df["State Name"].unique()))
    if wx.empty:
        print("Weather fetch produced no rows; continuing without climate features.")
        df["temp_mean"] = pd.NA
        df["rain_sum"] = pd.NA
        return df, False, failures
    merged = df.merge(wx, on=["State Name", "Year"], how="left")
    n_miss = int(merged["rain_sum"].isna().sum())
    print(f"Weather merge missing rain on {n_miss}/{len(merged)} rows; failures={failures}")
    ok = n_miss < len(merged)
    return merged, ok, failures


def drought_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rain_sum" not in df.columns or df["rain_sum"].isna().all():
        df["is_drought"] = False
        return df
    q20 = df.groupby("State Name")["rain_sum"].transform(lambda s: s.quantile(0.20))
    df["is_drought"] = df["rain_sum"] <= q20
    return df


def build_table(path: Path | None = None, use_weather: bool = True) -> pd.DataFrame:
    raw = load_raw(path)
    print("raw rows", len(raw), "columns", list(raw.columns))
    g = aggregate_state_crop_year(raw)
    print("aggregated state-crop-year rows", len(g))
    crops, states, g = choose_crops_and_states(g)
    g = add_lag_yield(g)
    print("after lag (dropped first year / gaps)", len(g))
    if use_weather:
        g, _ok, _fail = add_weather(g)
    g = drought_flag(g)
    out_path = DATA / "yield_model_table.csv"
    g.to_csv(out_path, index=False)
    print("wrote", out_path, "n=", len(g))
    print("year range", int(g["Year"].min()), int(g["Year"].max()))
    print(g.groupby("Crop").size().to_string())
    return g


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build clean historical yield modeling table.")
    parser.add_argument("--skip-weather", action="store_true", help="Skip Open-Meteo weather fetch and use lag/area features only.")
    parser.add_argument("--data", type=str, default=None, help="Path to raw yield CSV.")
    args = parser.parse_args()
    build_table(path=Path(args.data) if args.data else None, use_weather=not args.skip_weather)
