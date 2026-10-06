"""Prometheus metrics. Route labels use the route TEMPLATE (e.g. /users/{id}) to keep cardinality bounded."""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency", ["method", "route"], buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)
PREDICTIONS = Counter("predictions_total", "Predictions served", ["crop", "kind"])
CACHE_EVENTS = Counter("cache_events_total", "Cache events", ["result"])
RATE_LIMITED = Counter("rate_limited_total", "Requests rejected by rate limiting", ["bucket"])
AUTH_EVENTS = Counter("auth_events_total", "Authentication events", ["event"])
WEATHER_ERRORS = Counter("weather_errors_total", "Upstream weather failures")


def render():
    return generate_latest(), CONTENT_TYPE_LATEST
