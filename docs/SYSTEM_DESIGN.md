# System Design

## 1. Goals and non-goals
**Goals:** accurate, explainable crop recommendations; live weather input; safe multi-user access with roles;
operable in production (observable, scalable, recoverable); cheap enough for a student/NGO budget.
**Non-goals (v1):** yield forecasting (no yield data), IoT sensors, native mobile apps, payments.

## 2. Architecture
```
Browser / mobile
      |
   [CDN: Cloudflare / CloudFront]   caches static assets and GET /api/v1/meta, absorbs DDoS, terminates TLS
      |
   [nginx]  TLS, gzip, edge rate limit, security headers, JSON access log, /meta cache, round-robin to replicas
      |------------------------------.
 [Streamlit UI]                [FastAPI API  x N replicas, stateless]
                                 |  middleware: request-id, logging, metrics, headers, size limit
                                 |  deps: IP limit -> auth (JWT) -> RBAC -> user limit
                                 |  services: ml_service, agronomy, weather, storage, notifications
        .------------------------+--------------------------.-------------------.
   [PostgreSQL]            [Redis]                     [Open-Meteo API]     [Object storage (S3/R2)]
   users, predictions,     cache, rate-limit counters, live weather +       model artifacts,
   feedback, audit log     token revocation list       geocoding            DB backups
        
 Observability: JSON logs -> Docker/Loki/CloudWatch; /metrics -> Prometheus -> Grafana + Alertmanager -> Slack/email;
 exceptions -> Sentry.
```
**Why this shape:** the API holds no per-user state in memory (state lives in Postgres/Redis), so replicas can be added
or killed freely. nginx and the CDN absorb cheap traffic before it reaches Python. The model is small (about 1 MB) and
loaded into each worker, so predictions need no network hop.

## 3. Request flow (POST /api/v1/predict)
1. CDN/nginx: TLS, `limit_req` (20 req/s per IP), body limit 64 KB, forwards `X-Request-ID` and client IP.
2. Middleware: assigns/propagates request id, rejects oversized bodies, starts timer.
3. Router dependency `ip_limit` (120 req/min per IP, Redis counter).
4. `get_current_user`: verifies JWT signature, expiry, token type, revocation list; loads user; rejects inactive users.
5. `require_roles` (RBAC) then `user_limit` (60 req/min per user).
6. Pydantic validates ranges (pH 0-14, humidity 0-100, ...); invalid -> 422.
7. Cache lookup keyed by `model_version + rounded inputs`; on miss run the model, store with TTL.
8. Save the prediction (audit trail + history), increment Prometheus counters, return JSON.
9. Middleware logs one JSON line (route template, status, ms, user id, request id) and records latency.

## 4. Data model
`users(id, email*, password_hash, role, is_active, failed_attempts, locked_until)` ·
`predictions(id, user_id*, kind, inputs JSON, result JSON, top_crop*, probability, model_version, created_at*)` ·
`feedback(prediction_id*, user_id, helpful, comment)` · `audit_log(actor_id*, action*, detail JSON, ip, created_at*)` ·
`yield_predictions(id, user_id*, inputs JSON, yield_kg_per_ha, interval_lo, interval_hi, model_version, created_at*)` ·
`yield_outcomes(id, yield_prediction_id* [unique], user_id*, actual_yield_kg_per_ha, harvest_date, notes, created_at)`.
(* = indexed.) JSON columns keep the schema stable when model features change. Add Alembic migrations before the
first schema change in production. Retention: purge `predictions` older than your policy; never log passwords or tokens.

## 5. Caching strategy
| Data | Where | TTL | Key | Invalidation |
|---|---|---|---|---|
| Predictions and what-if curves | Redis (app) | 1 h | model_version + rounded inputs | automatic: new model version = new keys |
| Live weather (30-day window) | Redis | 30 min | lat/lon (2 dp) + date | TTL |
| Geocoding | Redis | 24 h | district + state | TTL |
| `/api/v1/meta` | CDN + nginx | 1 h | URL | TTL or purge after model reload |
| Revoked token ids | Redis | until token expiry | jti | automatic |
Pattern: cache-aside. Cache errors never fail a request (treated as a miss). **With more than one replica you must run
Redis**; the in-memory fallback is per process, so revocation and limits would not be shared.

## 6. Authentication and authorisation
- Passwords: scrypt (N=2^14, r=8, p=1) with per-user salt; min 10 chars with letters and digits; constant-time compare;
  dummy hash for unknown emails to equalise timing.
- Tokens: HS256 JWT, 30-minute access token, 7-day refresh token that **rotates** (one use). Logout adds the token id to a
  Redis revocation list. Use RS256 and a key service if other services must verify tokens.
- Lockout: 5 failed logins -> 15-minute lock, audit entry and an alert.
- Roles:

| Capability | farmer | expert | admin |
|---|:-:|:-:|:-:|
| Predict, what-if, history, feedback (own data) | yes | yes | yes |
| Yield forecast and harvest outcome submission (own data) | yes | yes | yes |
| View aggregate stats and yield evaluation metrics | no | yes | yes |
| Manage users, roles, view audit log, reload model | no | no | yes |
- Rules: users only see their own predictions; the last active admin cannot be demoted or disabled; every role change is audited.

## 7. Rate limiting (defence in depth)
| Layer | Rule | Purpose |
|---|---|---|
| CDN / WAF | managed rules, bot protection | volumetric abuse |
| nginx | 20 req/s per IP (burst 40); auth paths 10 req/min | cheap rejection before Python |
| App, per IP | 120 per 60 s | scraping |
| App, auth | 10 per 60 s per IP | credential stuffing |
| App, per user | 60 per 60 s | fair use, protects the Open-Meteo quota |
Responses use HTTP 429 with `Retry-After` and `X-RateLimit-*` headers. The limiter fails **open** if Redis is down.
Known limitation: with the Streamlit UI, calls come from the Streamlit server, so per-IP limits see one IP. Per-user
limits and account lockout still apply; a browser-based front end (React) would restore true per-client IP limiting.

## 8. Security checklist (mapped to OWASP Top 10)
- Broken access control: RBAC dependency on every protected route; object-level checks on history/feedback; tests in `test_api.py`.
- Cryptographic failures: scrypt, JWT with required claims, secrets from environment, production refuses a weak `SECRET_KEY`, TLS at the edge, HSTS.
- Injection: SQLAlchemy parameterised queries, Pydantic validation, no shell calls, no dynamic SQL.
- Insecure design: lockout, rotation, revocation, audit log, fail-safe defaults (503 when weather fails, never made-up numbers).
- Misconfiguration: docs/openapi disabled in production, strict CORS allow-list, security headers, non-root read-only containers, only nginx exposes a port, Postgres/Redis not published, `/metrics` blocked at the edge.
- Vulnerable components: `pip-audit`, Trivy image scan, Dependabot (enable in GitHub).
- Auth failures: rate limit + lockout + strong password policy. Logging: request ids, audit trail, alerts on login-failure spikes.
- SSRF: the server only calls fixed Open-Meteo hosts; user input is never used as a URL.
- Privacy: store the minimum (email, name, inputs); document retention; honour deletion requests (delete user and their rows).

## 9. Observability
**Logs:** JSON to stdout with `request_id`, route, status, latency, user id. **Metrics:** request rate/latency/status per
route template, predictions per crop, cache hit/miss/error, rate-limit rejections, auth events, weather errors.
**Alerts:** API down, 5xx rate above 5%, p95 above 1 s, login-failure spike, heavy rate limiting, weather provider failing.
**Error tracking:** Sentry captures exceptions with the request id. **Notifications:** Alertmanager -> Slack/email, plus
application alerts (account lockouts, model reload, startup failure, unhandled errors) with de-duplication.
**Probes:** `/health/live` (restart decisions) and `/health/ready` (traffic decisions: DB and model must be OK).

## 10. Scaling plan
| Stage | Users | Setup |
|---|---|---|
| 1 | up to a few hundred | one VM, `docker compose`, 2 API replicas, Postgres and Redis in containers |
| 2 | thousands | managed Postgres + Redis, 3-6 replicas behind a load balancer, CDN caching, nightly backups |
| 3 | tens of thousands | container platform (ECS/Kubernetes) with autoscaling on CPU and p95 latency, Postgres read replica for `/admin/stats`, Redis cluster |
Bottlenecks to watch: scrypt login cost (CPU), Open-Meteo quota (mitigated by caching and per-user limits), DB connections
(pool size x replicas must stay below Postgres `max_connections`; add PgBouncer). Model inference takes milliseconds; if the
model grows, move it behind its own service and scale it separately.

## 11. Storage and backups
Postgres on a persistent volume with a nightly `pg_dump` to object storage (7 daily + 4 weekly copies; test a restore
quarterly). Model artifacts versioned in S3/R2 under `models/<version>/` so any replica can load the same model and a bad
model can be rolled back. Data in `data/` is tracked with DVC or git (check the Kaggle licence).

## 12. ML lifecycle
`ml/train.py` validates the schema, trains with a fixed seed, evaluates (hold-out, 5-fold CV, noise robustness, top-3),
writes `model_card.json` (metrics, feature ranges, per-crop ranges, **limitations**) and a version string. CI trains and
tests; Docker bakes the model; `POST /admin/model/reload` hot-swaps a new artifact atomically. Tests assert that different
inputs produce different outputs. Monitor drift later by comparing live input distributions with `feature_range`.

## 13. Failure modes
| Failure | Behaviour |
|---|---|
| Redis down | cache treated as miss, limiter fails open, readiness shows `cache: degraded`; revocation not enforced until it returns |
| Open-Meteo down | live mode returns 503 with a clear message; manual mode keeps working |
| DB down | `/health/ready` returns 503, load balancer stops routing to the replica |
| Model file missing/corrupt | startup alert, readiness 503, previous model keeps serving after a failed reload |
| Replica crash | Docker restarts it; nginx re-resolves replicas every 10 s |
| Traffic spike | CDN/nginx/app limits return 429; scale out replicas |

## 14. CI/CD and release
PR -> CI (ruff, tests with coverage, bandit, pip-audit, gitleaks, Docker build, Trivy scan, container smoke test) ->
review -> merge to `main` -> CD builds and pushes the image to GHCR, deploys over SSH with `docker compose up -d`
(rolling replacement), runs a smoke test on `/health/ready`. Rollback = redeploy the previous image tag.

## 15. Known limitations
No yield prediction; model validated only on a clean public dataset (no regional/temporal test); rainfall time scale in the
dataset is undocumented (live mode assumes a 30-day total, shown to the user); advice is rule-based and not agronomist-reviewed;
token revocation requires Redis when running multiple replicas; Streamlit hides client IPs from the API.
