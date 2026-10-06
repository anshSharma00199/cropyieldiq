"""Best-effort alerts to Slack (webhook) and/or email (SMTP). Never raises; de-duplicates repeats."""

import logging
import smtplib
import threading
import time
from email.message import EmailMessage

import requests

from app.config import settings

log = logging.getLogger("notify")
_last = {}
_lock = threading.Lock()


def notify(level: str, title: str, message: str = "", dedupe_key: str | None = None, dedupe_seconds: int = 300):
    key = dedupe_key or title
    with _lock:
        now = time.time()
        if now - _last.get(key, 0) < dedupe_seconds:
            return False
        _last[key] = now
    text = f"[{level.upper()}] {title}\n{message}".strip()
    sent = False
    if settings.slack_webhook_url:
        try:
            requests.post(settings.slack_webhook_url, json={"text": text}, timeout=5).raise_for_status()
            sent = True
        except Exception as e:
            log.warning("slack notification failed: %s", e)
    if settings.smtp_host and settings.alert_email_to:
        try:
            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"] = f"[CropYieldIQ {level}] {title}", settings.alert_email_from, settings.alert_email_to
            msg.set_content(text)
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as s:
                s.starttls()
                if settings.smtp_user:
                    s.login(settings.smtp_user, settings.smtp_password)
                s.send_message(msg)
            sent = True
        except Exception as e:
            log.warning("email notification failed: %s", e)
    log.info("notification %s", "sent" if sent else "logged only (no channel configured)", extra={"title": title})
    return sent
