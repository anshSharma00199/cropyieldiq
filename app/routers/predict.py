from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import metrics, state
from app.config import settings
from app.database import get_db
from app.deps import require_roles, user_limit
from app.models import ROLES, Feedback, Prediction, User, YieldOutcome, YieldPrediction
from app.schemas import (
    CropPlannerIn,
    FeedbackIn,
    LiveIn,
    PredictionOut,
    SensitivityIn,
    SoilWeatherIn,
    YieldOutcomeIn,
    YieldOutcomeOut,
    YieldPredictionIn,
    YieldPredictionOut,
)
from app.services import agronomy, weather
from app.services.ml_service import ModelNotLoaded
from src.yield_service import YieldNotLoaded

router = APIRouter(tags=["predict"])
any_user = require_roles(*ROLES)


def _cached_predict(inputs: dict) -> dict:
    key = state.cache.make_key("pred", state.model_service.version, {k: round(v, 2) for k, v in inputs.items()})
    return state.cache.get_or_set(key, settings.predict_ttl, lambda: state.model_service.predict(inputs))


def _save(db: Session, user: User, kind: str, inputs: dict, result: dict, extra: dict) -> PredictionOut:
    row = Prediction(
        user_id=user.id,
        kind=kind,
        inputs=inputs,
        result={**result, **extra},
        top_crop=result["recommendation"],
        probability=result["top3"][0]["probability"],
        model_version=state.model_service.version,
    )
    db.add(row)
    db.commit()
    metrics.PREDICTIONS.labels(result["recommendation"], kind).inc()
    return PredictionOut(id=row.id, kind=kind, inputs=inputs, model_version=row.model_version, **result, **extra)


@router.post("/predict", response_model=PredictionOut, dependencies=[Depends(user_limit)])
def predict(body: SoilWeatherIn, db: Session = Depends(get_db), user: User = Depends(any_user)):
    inputs = body.model_dump()
    try:
        result = _cached_predict(inputs)
    except ModelNotLoaded:
        raise HTTPException(503, "Model is not available yet")
    return _save(db, user, "manual", inputs, result, {})


@router.post("/planner", dependencies=[Depends(user_limit)])
def crop_planner(body: CropPlannerIn, user: User = Depends(any_user)):
    try:
        card = state.model_service.card
        crop = body.crop.strip().lower()
        crop_ranges = card.get("crop_ranges", {})

        if crop not in crop_ranges:
            raise HTTPException(
                404,
                f"Unknown crop '{body.crop}'. Choose one of the supported crops from /api/v1/meta.",
            )

        result = agronomy.explain(
            crop,
            body.inputs.model_dump(),
            crop_ranges,
        )

        return {
            "crop": crop,
            "model_version": card["model_version"],
            "typical_ranges": crop_ranges[crop],
            "feature_status": result["feature_status"],
            "advice": result["advice"],
            "disclaimer": (
                "Dataset-based guidance only. Use a soil test and local agricultural "
                "expert advice before making fertilizer or irrigation decisions."
            ),
        }
    except ModelNotLoaded:
        raise HTTPException(503, "Model is not available yet")


@router.post(
    "/yield/predict",
    response_model=YieldPredictionOut,
    dependencies=[Depends(user_limit)],
)
def predict_yield(
    body: YieldPredictionIn,
    db: Session = Depends(get_db),
    user: User = Depends(any_user),
):
    # Omit None values because YieldService converts supplied values to float.
    inputs = {key: value for key, value in body.model_dump().items() if value is not None}

    try:
        pred = state.yield_service.predict(inputs)
    except YieldNotLoaded:
        raise HTTPException(503, "Yield model is not available yet")
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    row = YieldPrediction(
        user_id=user.id,
        inputs=inputs,
        yield_kg_per_ha=pred["yield_kg_per_ha"],
        interval_lo=pred["interval_lo"],
        interval_hi=pred["interval_hi"],
        model_version=pred["model_version"],
    )
    db.add(row)
    db.commit()

    return {**pred, "prediction_id": row.id}


@router.post(
    "/yield/outcome",
    status_code=201,
    response_model=YieldOutcomeOut,
    dependencies=[Depends(user_limit)],
)
def record_yield_outcome(
    body: YieldOutcomeIn,
    db: Session = Depends(get_db),
    user: User = Depends(any_user),
):
    pred = db.get(YieldPrediction, body.yield_prediction_id)
    if pred is None or pred.user_id != user.id:
        raise HTTPException(404, "Yield prediction not found")

    existing = db.scalar(select(YieldOutcome).where(YieldOutcome.yield_prediction_id == body.yield_prediction_id))
    if existing:
        raise HTTPException(409, "Outcome already submitted for this yield prediction")

    outcome = YieldOutcome(
        yield_prediction_id=pred.id,
        user_id=user.id,
        actual_yield_kg_per_ha=body.actual_yield_kg_per_ha,
        harvest_date=body.harvest_date,
        notes=body.notes,
    )
    try:
        db.add(outcome)
        db.commit()
        db.refresh(outcome)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Outcome already submitted for this yield prediction")

    return outcome


@router.post("/predict/live", response_model=PredictionOut, dependencies=[Depends(user_limit)])
def predict_live(body: LiveIn, db: Session = Depends(get_db), user: User = Depends(any_user)):
    try:
        if body.lat is not None and body.lon is not None:
            loc = {"lat": body.lat, "lon": body.lon, "name": body.district or "custom point", "state": body.state}
        else:
            loc = weather.geocode(state.cache, body.district, body.state)
        wx = weather.weather_features(state.cache, loc["lat"], loc["lon"], settings.weather_ttl)
    except weather.LocationNotFound as e:
        raise HTTPException(404, str(e))
    except weather.WeatherUnavailable:
        metrics.WEATHER_ERRORS.inc()
        raise HTTPException(503, "Live weather is temporarily unavailable. Use manual mode or try again shortly.")
    inputs = {
        "N": body.N,
        "P": body.P,
        "K": body.K,
        "ph": body.ph,
        "temperature": wx["temperature"],
        "humidity": wx["humidity"],
        "rainfall": wx["rainfall"],
    }
    try:
        result = _cached_predict(inputs)
    except ModelNotLoaded:
        raise HTTPException(503, "Model is not available yet")
    extra = {
        "weather": {**wx, "location": loc},
        "assumptions": [
            f"Temperature and humidity are {wx['window_days']}-day means and rainfall is the "
            f"{wx['window_days']}-day total, because the training dataset does not document its time scale."
        ],
    }
    return _save(db, user, "live", inputs, result, extra)


@router.post("/predict/sensitivity", dependencies=[Depends(user_limit)])
def sensitivity(body: SensitivityIn, user: User = Depends(any_user)):
    inputs = body.inputs.model_dump()
    key = state.cache.make_key("sens", state.model_service.version, body.feature, body.points, {k: round(v, 2) for k, v in inputs.items()})
    try:
        return state.cache.get_or_set(key, settings.predict_ttl, lambda: state.model_service.sensitivity(inputs, body.feature, body.points))
    except ModelNotLoaded:
        raise HTTPException(503, "Model is not available yet")


@router.get("/history", dependencies=[Depends(user_limit)])
def history(limit: int = 20, offset: int = 0, db: Session = Depends(get_db), user: User = Depends(any_user)):
    limit = max(1, min(limit, 100))
    rows = db.scalars(
        select(Prediction).where(Prediction.user_id == user.id).order_by(Prediction.created_at.desc()).limit(limit).offset(max(offset, 0))
    ).all()
    return [
        {
            "id": r.id,
            "kind": r.kind,
            "crop": r.top_crop,
            "probability": r.probability,
            "inputs": r.inputs,
            "model_version": r.model_version,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@router.post("/feedback", status_code=201, dependencies=[Depends(user_limit)])
def feedback(body: FeedbackIn, request: Request, db: Session = Depends(get_db), user: User = Depends(any_user)):
    pred = db.get(Prediction, body.prediction_id)
    if pred is None or pred.user_id != user.id:  # users can only rate their own predictions
        raise HTTPException(404, "Prediction not found")
    db.add(Feedback(prediction_id=pred.id, user_id=user.id, helpful=body.helpful, comment=body.comment))
    db.commit()
    return {"status": "recorded"}


@router.get("/meta")
def meta(response: Response):
    """Public and identical for everyone, so a CDN / nginx may cache it for an hour."""
    response.headers["Cache-Control"] = "public, max-age=3600"
    try:
        card = state.model_service.card
    except ModelNotLoaded:
        raise HTTPException(503, "Model is not available yet")
    return {
        "model_version": card["model_version"],
        "crops": card["classes"],
        "features": card["features"],
        "feature_range": card["feature_range"],
        "metrics": card["metrics"],
        "feature_importance": card["feature_importance"],
        "limitations": card["limitations"],
    }
