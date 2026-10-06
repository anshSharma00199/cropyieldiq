from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
CACHE = DATA / "cache"

CROP_CSV = DATA / "Crop_recommendation.csv"
YIELD_CSV = DATA / "Custom_Crops_yield_Historical_Dataset.csv"

CROP_MODEL = MODELS / "crop_model.joblib"
CROP_CALIBRATED = MODELS / "crop_model_calibrated.joblib"
CROP_CONFORMAL = MODELS / "crop_conformal.json"
CROP_CARD = MODELS / "model_card.json"

YIELD_BUNDLE = MODELS / "yield_bundle.joblib"
YIELD_SHAP_SUMMARY = RESULTS / "shap_summary.png"
VALIDATION_CSV = RESULTS / "validation_results.csv"
VALIDATION_PNG = RESULTS / "validation.png"
CALIBRATION_PNG = RESULTS / "crop_calibration.png"
