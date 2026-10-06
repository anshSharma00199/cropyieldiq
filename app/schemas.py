from datetime import date
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
FEATURES = Literal["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


class RegisterIn(BaseModel):
    email: str = Field(max_length=254, pattern=EMAIL_RE)
    password: str = Field(min_length=10, max_length=128)
    full_name: str = Field(default="", max_length=100)

    @field_validator("email")
    @classmethod
    def lower(cls, v):
        return v.strip().lower()


class LoginIn(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)

    @field_validator("email")
    @classmethod
    def lower(cls, v):
        return v.strip().lower()


class RefreshIn(BaseModel):
    refresh_token: str


class LogoutIn(BaseModel):
    refresh_token: str | None = None


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool


class SoilWeatherIn(BaseModel):
    N: float = Field(ge=0, le=300, description="Nitrogen (kg/ha, dataset scale)")
    P: float = Field(ge=0, le=300, description="Phosphorus")
    K: float = Field(ge=0, le=400, description="Potassium")
    temperature: float = Field(ge=-20, le=60, description="Mean temperature, deg C")
    humidity: float = Field(ge=0, le=100, description="Relative humidity, %")
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0, le=2000, description="Rainfall, mm")


class CropPlannerIn(BaseModel):
    crop: str = Field(min_length=1, max_length=80)
    inputs: SoilWeatherIn


class LiveIn(BaseModel):
    district: str | None = Field(default=None, max_length=80)
    state: str | None = Field(default=None, max_length=80)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    N: float = Field(ge=0, le=300)
    P: float = Field(ge=0, le=300)
    K: float = Field(ge=0, le=400)
    ph: float = Field(ge=0, le=14)

    @model_validator(mode="after")
    def need_location(self):
        if not self.district and (self.lat is None or self.lon is None):
            raise ValueError("provide a district, or both lat and lon")
        return self


class SensitivityIn(BaseModel):
    inputs: SoilWeatherIn
    feature: FEATURES
    points: int = Field(default=15, ge=5, le=40)


class FeedbackIn(BaseModel):
    prediction_id: int
    helpful: bool
    comment: str = Field(default="", max_length=500)


class RoleIn(BaseModel):
    role: Literal["farmer", "expert", "admin"]


class ActiveIn(BaseModel):
    is_active: bool


class PredictionOut(BaseModel):
    id: int
    kind: str
    inputs: dict
    recommendation: str
    confidence: str
    top_probability: float | None = None
    conformal_set: list[str] = []
    top3: list[dict]
    explanation: dict
    warnings: list[str]
    model_version: str
    weather: dict | None = None
    assumptions: list[str] = []


class YieldPredictionIn(BaseModel):
    state: str = Field(min_length=1, max_length=80)
    crop: str = Field(min_length=1, max_length=80)
    area_ha: float = Field(gt=0, le=10_000_000)
    year: int = Field(default=2017, ge=1900, le=2100)
    lag_yield: float | None = Field(default=None, ge=0)
    temp_mean: float | None = Field(default=None, ge=-20, le=60)
    rainfall: float | None = Field(default=None, ge=0, le=10000)


class YieldPredictionOut(BaseModel):
    prediction_id: int
    yield_kg_per_ha: float
    interval_lo: float
    interval_hi: float
    alpha: float
    shap_sentences: list[str]
    model_version: str
    warnings: list[str]
    disclaimer: str


MAX_ACTUAL_YIELD_KG_PER_HA = 100_000.0  # 100 tonnes/ha is a safe upper bound exceeding global record biomass for crops


class YieldOutcomeIn(BaseModel):
    yield_prediction_id: int = Field(gt=0, description="Positive ID of the yield prediction")
    actual_yield_kg_per_ha: float = Field(
        gt=0,
        le=MAX_ACTUAL_YIELD_KG_PER_HA,
        description="Actual measured yield in kg/ha (strictly positive and <= 100,000 kg/ha)",
    )
    harvest_date: date | None = Field(default=None, description="Optional date when harvest occurred")
    notes: str = Field(default="", max_length=500, description="Optional notes (max 500 chars)")

    @field_validator("actual_yield_kg_per_ha")
    @classmethod
    def validate_actual_yield(cls, v: float) -> float:
        if math.isnan(v) or not math.isfinite(v):
            raise ValueError("actual_yield_kg_per_ha must be a finite number")
        if v <= 0:
            raise ValueError("actual_yield_kg_per_ha must be strictly positive")
        if v > MAX_ACTUAL_YIELD_KG_PER_HA:
            raise ValueError(f"actual_yield_kg_per_ha must not exceed {MAX_ACTUAL_YIELD_KG_PER_HA} kg/ha")
        return v


class YieldOutcomeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    yield_prediction_id: int
    actual_yield_kg_per_ha: float
    harvest_date: date | None = None
    notes: str = ""
    status: str = "recorded"


class YieldEvaluationOut(BaseModel):
    sample_count: int
    count: int
    mae: float | None = None
    rmse: float | None = None
    mean_error: float | None = None
    bias: float | None = None
    mape: float | None = None
    inside_interval_count: int = 0
    inside_intervals: int = 0
    coverage_percentage: float | None = None
    coverage_ratio: float | None = None
    disclaimer: str
