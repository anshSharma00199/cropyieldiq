import math
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import state
from app.audit import audit
from app.config import settings
from app.database import get_db
from app.deps import client_ip, require_roles, user_limit
from app.models import (
    ROLE_ADMIN,
    ROLE_EXPERT,
    AuditLog,
    Feedback,
    Prediction,
    User,
    YieldOutcome,
    YieldPrediction,
    utcnow,
)
from app.notifications import notify
from app.schemas import ActiveIn, RoleIn, UserOut, YieldEvaluationOut
from app.services.storage import fetch_artifacts

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(user_limit)])
admin_only = require_roles(ROLE_ADMIN)
staff = require_roles(ROLE_EXPERT, ROLE_ADMIN)


@router.get("/stats")
def stats(db: Session = Depends(get_db), user: User = Depends(staff)):
    since = utcnow() - timedelta(days=14)
    by_crop = db.execute(
        select(Prediction.top_crop, func.count()).group_by(Prediction.top_crop).order_by(func.count().desc()).limit(10)
    ).all()
    daily = db.execute(
        select(func.date(Prediction.created_at), func.count())
        .where(Prediction.created_at >= since)
        .group_by(func.date(Prediction.created_at))
        .order_by(func.date(Prediction.created_at))
    ).all()
    fb_total = db.scalar(select(func.count(Feedback.id))) or 0
    fb_helpful = db.scalar(select(func.count(Feedback.id)).where(Feedback.helpful.is_(True))) or 0
    return {
        "total_predictions": db.scalar(select(func.count(Prediction.id))) or 0,
        "total_users": db.scalar(select(func.count(User.id))) or 0,
        "avg_top_probability": round(float(db.scalar(select(func.avg(Prediction.probability))) or 0), 4),
        "top_crops": [{"crop": c, "count": n} for c, n in by_crop],
        "daily_last_14_days": [{"date": str(d), "count": n} for d, n in daily],
        "feedback": {"total": fb_total, "helpful_rate": round(fb_helpful / fb_total, 3) if fb_total else None},
        "cache": {"hits": state.cache.hits, "misses": state.cache.misses, "errors": state.cache.errors},
    }


@router.get("/yield/evaluation", response_model=YieldEvaluationOut)
def yield_evaluation(db: Session = Depends(get_db), user: User = Depends(staff)):
    query = select(
        YieldOutcome.actual_yield_kg_per_ha,
        YieldPrediction.yield_kg_per_ha,
        YieldPrediction.interval_lo,
        YieldPrediction.interval_hi,
    ).join(YieldPrediction, YieldOutcome.yield_prediction_id == YieldPrediction.id)
    rows = db.execute(query).all()

    if not rows:
        return YieldEvaluationOut(
            sample_count=0,
            count=0,
            mae=None,
            rmse=None,
            mean_error=None,
            bias=None,
            mape=None,
            inside_interval_count=0,
            inside_intervals=0,
            coverage_percentage=None,
            coverage_ratio=None,
            disclaimer=(
                "Retrospective evaluation on farmer-reported harvest outcomes. "
                "The sample may be small, self-selected, or subject to reporting bias; "
                "this does not provide proof of general model generalization."
            ),
        )

    n = len(rows)
    errors = []
    abs_errors = []
    sq_errors = []
    pct_errors = []
    inside_count = 0

    for actual, pred, lo, hi in rows:
        err = pred - actual
        errors.append(err)
        abs_errors.append(abs(err))
        sq_errors.append(err**2)

        if actual > 0:
            pct_errors.append(abs(err) / actual)

        if lo <= actual <= hi:
            inside_count += 1

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)
    mean_err = sum(errors) / n
    mape = (sum(pct_errors) / len(pct_errors) * 100.0) if pct_errors else None
    coverage_pct = (inside_count / n) * 100.0
    coverage_ratio = inside_count / n

    return YieldEvaluationOut(
        sample_count=n,
        count=n,
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        mean_error=round(mean_err, 4),
        bias=round(mean_err, 4),
        mape=round(mape, 4) if mape is not None else None,
        inside_interval_count=inside_count,
        inside_intervals=inside_count,
        coverage_percentage=round(coverage_pct, 2),
        coverage_ratio=round(coverage_ratio, 4),
        disclaimer=(
            "Retrospective evaluation on farmer-reported harvest outcomes. "
            "The sample may be small, self-selected, or subject to reporting bias; "
            "this does not provide proof of general model generalization."
        ),
    )


@router.get("/users", response_model=list[UserOut])
def users(limit: int = 50, offset: int = 0, db: Session = Depends(get_db), user: User = Depends(admin_only)):
    return db.scalars(select(User).order_by(User.id).limit(max(1, min(limit, 200))).offset(max(offset, 0))).all()


def _guard_last_admin(db: Session, target: User):
    admins = db.scalar(select(func.count(User.id)).where(User.role == ROLE_ADMIN, User.is_active.is_(True))) or 0
    if target.role == ROLE_ADMIN and admins <= 1:
        raise HTTPException(409, "Cannot remove the last active admin")


@router.patch("/users/{user_id}/role", response_model=UserOut)
def set_role(user_id: int, body: RoleIn, request: Request, db: Session = Depends(get_db), actor: User = Depends(admin_only)):
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(404, "User not found")
    if body.role != ROLE_ADMIN:
        _guard_last_admin(db, target)
    old, target.role = target.role, body.role
    db.commit()
    audit(db, actor.id, "role_changed", {"target": target.id, "from": old, "to": body.role}, client_ip(request))
    return target


@router.patch("/users/{user_id}/active", response_model=UserOut)
def set_active(user_id: int, body: ActiveIn, request: Request, db: Session = Depends(get_db), actor: User = Depends(admin_only)):
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(404, "User not found")
    if not body.is_active:
        _guard_last_admin(db, target)
    target.is_active = body.is_active
    db.commit()
    audit(db, actor.id, "user_active_changed", {"target": target.id, "is_active": body.is_active}, client_ip(request))
    return target


@router.post("/model/reload")
def reload_model(request: Request, db: Session = Depends(get_db), actor: User = Depends(admin_only)):
    try:
        fetch_artifacts()
        version = state.model_service.load(settings.model_path, settings.model_card_path)
    except Exception as e:
        notify("error", "Model reload failed", str(e)[:300])
        raise HTTPException(500, "Model reload failed; the previous model is still serving")
    audit(db, actor.id, "model_reloaded", {"version": version}, client_ip(request))
    notify("info", "Model reloaded", f"version {version} by user {actor.id}")
    return {"model_version": version}


@router.get("/audit")
def audit_log(limit: int = 100, db: Session = Depends(get_db), actor: User = Depends(admin_only)):
    rows = db.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(max(1, min(limit, 500)))).all()
    return [
        {"id": r.id, "actor_id": r.actor_id, "action": r.action, "detail": r.detail, "ip": r.ip, "created_at": r.created_at.isoformat()}
        for r in rows
    ]
