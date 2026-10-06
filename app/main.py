"""CropYieldIQ API entry point: middleware (request id, logging, metrics, security headers, size limit),
error tracking (Sentry), start-up (DB, model, admin bootstrap) and routers."""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app import metrics, state
from app.config import settings
from app.core.security import hash_password, password_problems
from app.database import Base, SessionLocal, engine
from app.deps import ip_limit
from app.logging_conf import request_id_ctx, setup_logging
from app.models import ROLE_ADMIN, User
from app.notifications import notify
from app.routers import admin, auth, health, predict
from app.services.storage import fetch_artifacts

log = logging.getLogger("app")


def _bootstrap_admin():
    if not (settings.admin_email and settings.admin_password):
        return
    if password_problems(settings.admin_password):
        log.error("ADMIN_PASSWORD is too weak; admin bootstrap skipped")
        return
    with SessionLocal() as db:
        if not db.scalar(select(User).where(User.email == settings.admin_email.lower())):
            db.add(
                User(
                    email=settings.admin_email.lower(),
                    password_hash=hash_password(settings.admin_password),
                    full_name="Administrator",
                    role=ROLE_ADMIN,
                )
            )
            db.commit()
            log.info("bootstrap admin created")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    Base.metadata.create_all(engine)

    # Load crop recommendation model.
    try:
        fetch_artifacts()
        state.model_service.load(settings.model_path, settings.model_card_path)
    except Exception:
        log.exception("crop model failed to load; /health/ready will report not_ready")
        notify("critical", "Crop model failed to load at startup", dedupe_key="model-load")

    # Load yield forecasting model.
    try:
        state.yield_service.load()
        log.info("yield model loaded", extra={"model_version": state.yield_service.bundle["version"]})
    except Exception:
        log.exception("yield model failed to load")
        notify("critical", "Yield model failed to load at startup", dedupe_key="yield-model-load")

    _bootstrap_admin()
    yield


def create_app() -> FastAPI:
    if settings.sentry_dsn:
        import sentry_sdk

        sentry_sdk.init(dsn=settings.sentry_dsn, environment=settings.env, traces_sample_rate=0.1, send_default_pii=False)

    docs = None if settings.is_prod else "/docs"
    app = FastAPI(
        title="CropYieldIQ API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=docs,
        redoc_url=None,
        openapi_url=None if settings.is_prod else "/openapi.json",
    )

    @app.middleware("http")
    async def observability(request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        length = request.headers.get("content-length")
        try:
            if length and length.isdigit() and int(length) > settings.max_body_bytes:
                response = JSONResponse({"detail": "Request body too large"}, status_code=413)
            else:
                response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"method": request.method, "path": request.url.path})
            if settings.sentry_dsn:
                import sentry_sdk

                sentry_sdk.capture_exception()
            notify("error", "Unhandled server error", f"{request.method} {request.url.path} request_id={rid}", dedupe_key="http-500")
            response = JSONResponse({"detail": "Internal server error", "request_id": rid}, status_code=500)
        duration = time.perf_counter() - start
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        metrics.REQUESTS.labels(request.method, route_path, str(response.status_code)).inc()
        metrics.LATENCY.labels(request.method, route_path).observe(duration)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if settings.is_prod:
            response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        if request.url.path not in ("/health/live", "/health/ready", "/metrics"):
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": route_path,
                    "status": response.status_code,
                    "ms": round(duration * 1000, 1),
                    "user_id": getattr(request.state, "user_id", None),
                },
            )
        request_id_ctx.reset(token)
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    app.include_router(health.router)
    for r in (auth.router, predict.router, admin.router):
        app.include_router(r, prefix="/api/v1", dependencies=[Depends(ip_limit)])
    return app


app = create_app()
