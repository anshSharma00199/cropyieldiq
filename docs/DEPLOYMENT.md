# Deployment Guide

## Choose a hosting option
| Option | Best for | Rough monthly cost* | Effort |
|---|---|---|---|
| A. Single VM + docker compose (Hetzner, DigitalOcean, AWS Lightsail, Oracle free tier) | semester project, pilot | USD 5-20 | low |
| B. Managed PaaS (Render, Railway, Fly.io) with managed Postgres/Redis | no server admin | USD 15-60 | low |
| C. AWS: ECS Fargate + RDS Postgres + ElastiCache + S3 + ALB + CloudFront + WAF + CloudWatch | production, autoscaling | USD 100+ | high |
*Indicative only; check current pricing.

## Option A: single VM (recommended for your project)
1. Create an Ubuntu 22.04/24.04 VM (2 vCPU, 2-4 GB RAM). Add a DNS `A` record for your domain.
2. Harden: SSH keys only, `ufw allow 22,80,443`, `unattended-upgrades`, a non-root deploy user.
3. Install Docker + Compose plugin. `sudo mkdir /opt/cropyieldiq && sudo chown deploy /opt/cropyieldiq`.
4. Copy the repo (or only `docker-compose.yml`, `nginx/`, `monitoring/`, `.env`) to `/opt/cropyieldiq`.
5. `cp .env.example .env`; generate `SECRET_KEY`, passwords; set `CORS_ORIGINS=https://your-domain`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`.
6. `docker compose up -d --build`; check `curl localhost/health/ready`; log in as the admin; **remove `ADMIN_PASSWORD` from `.env`**.
7. TLS: either put Cloudflare in front (below), or use Let's Encrypt: run certbot (`certbot certonly --webroot`), mount
   `/etc/letsencrypt` into the nginx container, enable the 443 server block, redirect 80 -> 443, add HSTS, renew with a cron/systemd timer.
8. Reach Grafana through an SSH tunnel: `ssh -L 3000:localhost:3000 deploy@server`; add Prometheus as a data source (`http://prometheus:9090`).

## CDN (Cloudflare free plan, works in front of nginx)
1. Add your domain to Cloudflare, switch nameservers, proxy (orange cloud) the `A` record.
2. SSL/TLS mode **Full (strict)** with a Cloudflare Origin Certificate on nginx (or Let's Encrypt).
3. Caching: respect origin headers. `/api/v1/meta` is already `Cache-Control: public, max-age=3600`. Add a rule to **bypass cache**
   for `/api/v1/*` except `/api/v1/meta`, and for `/health/*`.
4. Security: enable WAF managed rules, Bot Fight Mode, and a rate-limiting rule for `/api/v1/auth/*` (for example 20 requests per minute).
5. Real client IPs: in nginx add `set_real_ip_from` for Cloudflare's published ranges and `real_ip_header CF-Connecting-IP;`.
6. Restrict the origin: allow ports 80/443 only from Cloudflare IP ranges (firewall) so attackers cannot bypass the CDN.
(CloudFront equivalent: origin = ALB/nginx, cache policy "CachingDisabled" for `/api/*`, "CachingOptimized" for `/api/v1/meta` and static files, AWS WAF attached.)

## Error tracking, alerts and notifications
- **Sentry:** create a project (Python/FastAPI), set `SENTRY_DSN`. PII sending is disabled in code.
- **Slack:** create an Incoming Webhook; set `SLACK_WEBHOOK_URL` (app alerts) and paste it into `monitoring/alertmanager.yml` (infra alerts).
- **Email:** set `SMTP_*` and `ALERT_EMAIL_*`, or uncomment `email_configs` in Alertmanager.
- Test: stop one API container (`docker stop <name>`); `ApiDown` should fire after about 2 minutes.

## Storage and backups
- Model artifacts in S3/R2: set `STORAGE_BACKEND=s3`, `S3_BUCKET`, AWS credentials (prefer an IAM role), `pip install boto3` in the image,
  then `python ml/train.py --upload` and `POST /api/v1/admin/model/reload`.
- Nightly DB backup (cron on the VM):
  `docker compose exec -T postgres pg_dump -U cyiq cyiq | gzip > /backups/cyiq-$(date +%F).sql.gz` then sync to object storage; restore with `gunzip -c file | docker compose exec -T postgres psql -U cyiq cyiq`.
  Test a restore before you rely on it.

## CI/CD setup (GitHub)
1. Push the repo; enable branch protection on `main` (PR + review + required checks: `test`, `security`, `docker`).
2. Enable Dependabot alerts/updates and secret scanning.
3. Create an environment `production` (optional: required reviewers = manual approval gate).
4. Repository secrets: `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY` (a dedicated key), `DEPLOY_DOMAIN`.
5. On the server: `docker login ghcr.io` with a read-only token so `docker compose pull` works.
6. Merge to `main` -> CI -> CD builds, pushes, deploys, smoke-tests. Rollback: re-run CD on the previous commit or set `API_IMAGE` to the old tag.

## Scaling
- Vertical: bigger VM, raise `WEB_CONCURRENCY` (about 2 x cores).
- Horizontal on one host: `docker compose up -d --scale api=4`. nginx re-resolves replicas every 10 s.
- Multi-host: move Postgres and Redis to managed services, run API replicas on several hosts behind a load balancer, keep sessions out of memory (already true).
- Autoscaling: ECS/Kubernetes HPA on CPU 60% or p95 latency. Keep `pool_size x replicas` below Postgres `max_connections` (add PgBouncer).

## Production checklist
- [ ] `APP_ENV=production`, strong `SECRET_KEY`, unique passwords, `.env` not in git
- [ ] TLS + HSTS, CDN/WAF, origin locked to the CDN
- [ ] Postgres/Redis not exposed publicly; firewall allows only 22 (your IP), 80, 443
- [ ] Redis enabled (needed for shared rate limits and token revocation across replicas)
- [ ] Admin bootstrap password removed from `.env`; MFA on GitHub, cloud and Sentry accounts
- [ ] Backups tested; alerts tested; a named person receives the alerts
- [ ] Privacy notice and data-retention policy written; model limitations visible in the UI
- [ ] Agronomist reviewed the advice rules before real farmers use them
