from fastapi import APIRouter, Depends, Header, HTTPException, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import metrics, state
from app.config import settings
from app.database import get_db

router = APIRouter(tags=["ops"])


@router.get("/health/live")
def live():
    """Liveness: the process is up. Used by Docker/Kubernetes to decide on restarts."""
    return {"status": "ok"}


@router.get("/health/ready")
def ready(response: Response, db: Session = Depends(get_db)):
    """Readiness: dependencies OK. Used by the load balancer to decide whether to send traffic."""
    checks = {"model": state.model_service.loaded}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        checks["database"] = False
    cache_ok = state.cache.ping()  # cache down = degraded, not unready (we fall back to computing)
    ok = checks["model"] and checks["database"]
    response.status_code = 200 if ok else 503
    return {"status": "ready" if ok else "not_ready", "cache": "ok" if cache_ok else "degraded", **checks}


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics(x_metrics_token: str | None = Header(default=None)):
    if settings.metrics_token and x_metrics_token != settings.metrics_token:
        raise HTTPException(403, "Forbidden")
    body, content_type = metrics.render()
    return Response(body, media_type=content_type)
