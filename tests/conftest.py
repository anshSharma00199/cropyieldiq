"""Test environment. Env vars are set BEFORE the app is imported so it uses an in-memory DB and no Redis."""

import os
import subprocess
import sys
from pathlib import Path

os.environ.update(
    {
        "APP_ENV": "test",
        "DATABASE_URL": "sqlite:///:memory:",
        "REDIS_URL": "",
        "SECRET_KEY": "test-secret-key-test-secret-key-0123456789",
        "TRUST_PROXY": "true",
        "RL_IP": "100000/60",
        "RL_USER": "100000/60",
        "RL_AUTH": "100000/60",
        "SLACK_WEBHOOK_URL": "",
        "SMTP_HOST": "",
        "SENTRY_DSN": "",
    }
)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

if not (ROOT / "models" / "crop_model.joblib").exists():  # CI: build the (deterministic) model first
    subprocess.run([sys.executable, "ml/train.py"], check=True, cwd=ROOT)
