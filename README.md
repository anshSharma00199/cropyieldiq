# CropYieldIQ: Smart Crop Advisory Platform

A production-style web platform that **recommends the best crop for given soil and weather conditions**, explains
why, supports what-if analysis, and can pull **live weather** for any Indian district. Every output is computed
from the user's inputs: change a value and the answer changes.

> **Important scope note.** The dataset is the Kaggle *Crop Recommendation* dataset (N, P, K, temperature, humidity,
> pH, rainfall → crop label). It contains **no yield column**, so this system recommends *which crop* suits the
> conditions; it does **not** predict yield in kg/ha. Yield forecasting needs a yield dataset (see "Roadmap").

## Quick start (local, no Docker)
```
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
python ml/train.py                                   # trains the model from data/Crop_recommendation.csv
ADMIN_EMAIL=you@example.com ADMIN_PASSWORD='Str0ng-Passw0rd' uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
pip install -r requirements-frontend.txt
streamlit run frontend/streamlit_app.py              # UI on http://localhost:8501
pytest                                               # run all tests
```
Uses SQLite and an in-process cache by default. Set `DATABASE_URL` and `REDIS_URL` for Postgres and Redis.

## Full stack with Docker
```
cp .env.example .env     # set SECRET_KEY, POSTGRES_PASSWORD, REDIS_PASSWORD, ADMIN_*
docker compose up -d --build        # nginx + 2 API replicas + UI + Postgres + Redis + Prometheus + Alertmanager + Grafana
docker compose up -d --scale api=4  # scale out
```

## Where each engineering requirement lives
| Requirement | Implementation |
|---|---|
| System architecture / design | `docs/SYSTEM_DESIGN.md` |
| APIs and backend logic | `app/routers/*`, `app/services/*` (FastAPI, versioned `/api/v1`) |
| Database | SQLAlchemy models in `app/models.py` (SQLite dev, PostgreSQL prod), audit log, feedback |
| Authentication | `app/routers/auth.py`, `app/core/security.py` (scrypt hashing, JWT access + rotating refresh, logout revocation, lockout) |
| Permissions (RBAC) | `app/deps.py` `require_roles`: farmer / expert / admin |
| Caching | `app/core/cache.py` (Redis or in-memory; weather, geocoding, predictions); nginx cache for `/meta` |
| CDN | `docs/DEPLOYMENT.md` (Cloudflare/CloudFront in front of nginx; `Cache-Control` headers already set) |
| Rate limiting | `app/core/ratelimit.py` + `app/deps.py` (per IP, per user, stricter on auth) and nginx `limit_req` |
| Logging | `app/logging_conf.py` (JSON, request id), nginx JSON access log |
| Error tracking | Sentry (`SENTRY_DSN`), global error handler returns a request id |
| Monitoring | `app/metrics.py` (`/metrics`), `monitoring/*` (Prometheus, alert rules, Alertmanager, Grafana) |
| Notifications | `app/notifications.py` (Slack/email) + Alertmanager routes |
| Security | validation ranges, security headers, CORS allow-list, body-size limit, non-root read-only containers, secret scanning, dependency + image scanning |
| Storage | `app/services/storage.py` (local or S3-compatible model artifacts), Postgres volume |
| Testing | `tests/` (unit, ML behaviour, API end-to-end), coverage in CI |
| CI/CD | `.github/workflows/ci.yml`, `cd.yml` |
| Version control | `CONTRIBUTING.md`, PR template, pre-commit, conventional commits, protected `main` |
| Cloud hosting and scaling | `docker-compose.yml`, `docs/DEPLOYMENT.md` |

## API summary (`/api/v1`)
| Method and path | Who | Purpose |
|---|---|---|
| POST `/auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout` | public / user | account and tokens |
| GET `/auth/me` | any user | current profile |
| POST `/predict` | any user | recommend a crop from N, P, K, temperature, humidity, pH, rainfall |
| POST `/predict/live` | any user | same, but temperature/humidity/rainfall fetched live for a district or lat/lon |
| POST `/predict/sensitivity` | any user | what-if curve: vary one input across its observed range |
| POST `/yield/predict` | any user | forecast crop yield (kg/ha) with conformal intervals; persists prediction & returns `prediction_id` |
| POST `/yield/outcome` | any user | record actual harvest outcome for user's own yield prediction |
| GET `/history` | any user | your past predictions |
| POST `/feedback` | any user | rate a prediction |
| GET `/meta` | public, CDN-cacheable | crops, ranges, model version and metrics |
| GET `/admin/stats` | expert, admin | usage and feedback statistics |
| GET `/admin/yield/evaluation` | expert, admin | retrospective yield evaluation metrics (MAE, RMSE, bias, MAPE, coverage) |
| GET `/admin/users`, PATCH `/admin/users/{id}/role`, `/active` | admin | user management (audited) |
| POST `/admin/model/reload`, GET `/admin/audit` | admin | hot-reload model, view audit log |
| GET `/health/live`, `/health/ready`, `/metrics` | ops | probes and Prometheus |

See `docs/YIELD_FEEDBACK_AND_EVALUATION.md` for in-depth documentation on yield feedback and retrospective evaluation.

## Verification status (be honest in your report)
- Model training, prediction logic, cache, rate limiter, password/JWT code and agronomy rules: **tests ran and passed** (18 tests).
- The FastAPI layer, `tests/test_api.py`, Streamlit UI, Docker, nginx, CI/CD and monitoring configs were written but
  **not executed** in the environment that generated this project (no framework packages or network were available).
  Run `pytest` and `docker compose up` yourself and fix any small issues you hit; CI will repeat the checks on every push.
- Live weather calls (Open-Meteo) were never made from the build environment; they are covered by mocked tests only.

## Model facts (from `models/model_card.json`)
Random Forest, 200 trees. Hold-out accuracy about 99%, 5-fold CV about 99%, about 97% with 10% input noise.
These numbers are **optimistic**: the dataset is perfectly balanced, has well-separated classes, and no location or
time column, so regional and temporal generalisation is unknown. Say so in your report.

## Roadmap
1. Add a real yield dataset (district-year yields from ICRISAT / data.gov.in) and the validated yield model from the
   earlier plan (temporal and region-held-out splits, conformal intervals).
2. Replace the crop-range rules with agronomist-reviewed ICAR recommendations.
3. Add Alembic migrations, a React/PWA front end (true per-user IP limiting), SMS/IVR delivery, and Hindi/regional UI strings.
