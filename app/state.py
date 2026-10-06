"""Process-wide singletons (one per worker). Shared state across replicas lives in Redis/Postgres, not here."""

from app import metrics
from app.config import settings
from app.core.cache import build_cache
from app.core.ratelimit import build_limiter
from app.services.ml_service import ModelService
from src.yield_service import YieldService

cache = build_cache(settings.redis_url, on_event=lambda kind: metrics.CACHE_EVENTS.labels(kind).inc())
limiter = build_limiter(settings.redis_url)
model_service = ModelService()
yield_service = YieldService()
