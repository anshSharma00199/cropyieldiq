"""Central configuration, read from environment variables (12-factor). No secrets in code."""

import os
from dataclasses import dataclass


def _b(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _limit(name: str, default: str) -> tuple:
    """'60/60' -> (60 requests, 60 seconds)."""
    n, w = os.getenv(name, default).split("/")
    return int(n), int(w)


@dataclass(frozen=True)
class Settings:
    env: str
    secret_key: str
    database_url: str
    redis_url: str
    access_token_minutes: int
    refresh_token_days: int
    cors_origins: list
    rl_ip: tuple
    rl_user: tuple
    rl_auth: tuple
    max_body_bytes: int
    trust_proxy: bool
    model_path: str
    model_card_path: str
    storage_backend: str
    s3_bucket: str
    s3_prefix: str
    sentry_dsn: str
    slack_webhook_url: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    alert_email_from: str
    alert_email_to: str
    metrics_token: str
    admin_email: str
    admin_password: str
    weather_ttl: int
    predict_ttl: int
    log_level: str

    @property
    def is_prod(self) -> bool:
        return self.env == "production"


def load_settings() -> Settings:
    env = os.getenv("APP_ENV", "development")
    secret = os.getenv("SECRET_KEY", "")
    if env == "production" and (len(secret) < 32 or secret.startswith("change-me")):
        raise RuntimeError("SECRET_KEY must be a random string of 32+ characters in production")
    return Settings(
        env=env,
        secret_key=secret or "dev-only-insecure-key-do-not-use-in-production",
        database_url=os.getenv("DATABASE_URL", "sqlite:///./cropyieldiq.db"),
        redis_url=os.getenv("REDIS_URL", ""),
        access_token_minutes=int(os.getenv("ACCESS_TOKEN_MINUTES", "30")),
        refresh_token_days=int(os.getenv("REFRESH_TOKEN_DAYS", "7")),
        cors_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",") if o.strip()],
        rl_ip=_limit("RL_IP", "120/60"),
        rl_user=_limit("RL_USER", "60/60"),
        rl_auth=_limit("RL_AUTH", "10/60"),
        max_body_bytes=int(os.getenv("MAX_BODY_BYTES", "65536")),
        trust_proxy=_b("TRUST_PROXY", False),
        model_path=os.getenv("MODEL_PATH", "models/crop_model.joblib"),
        model_card_path=os.getenv("MODEL_CARD_PATH", "models/model_card.json"),
        storage_backend=os.getenv("STORAGE_BACKEND", "local"),
        s3_bucket=os.getenv("S3_BUCKET", ""),
        s3_prefix=os.getenv("S3_PREFIX", "models/"),
        sentry_dsn=os.getenv("SENTRY_DSN", ""),
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
        smtp_host=os.getenv("SMTP_HOST", ""),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        smtp_user=os.getenv("SMTP_USER", ""),
        smtp_password=os.getenv("SMTP_PASSWORD", ""),
        alert_email_from=os.getenv("ALERT_EMAIL_FROM", ""),
        alert_email_to=os.getenv("ALERT_EMAIL_TO", ""),
        metrics_token=os.getenv("METRICS_TOKEN", ""),
        admin_email=os.getenv("ADMIN_EMAIL", ""),
        admin_password=os.getenv("ADMIN_PASSWORD", ""),
        weather_ttl=int(os.getenv("WEATHER_TTL", "1800")),
        predict_ttl=int(os.getenv("PREDICT_TTL", "3600")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


settings = load_settings()
