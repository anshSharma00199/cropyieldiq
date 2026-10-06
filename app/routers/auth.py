import time
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import metrics, state
from app.audit import audit
from app.config import settings
from app.core.security import TokenError, create_token, decode_token, hash_password, password_problems, verify_password
from app.database import get_db
from app.deps import auth_limit, bearer, client_ip, get_current_user
from app.models import ROLE_FARMER, User, utcnow
from app.notifications import notify
from app.schemas import LoginIn, LogoutIn, RefreshIn, RegisterIn, TokenOut, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])
MAX_FAILED, LOCK_MINUTES = 5, 15
_DUMMY_HASH = hash_password("dummy-password-for-constant-time-checks")  # equalises timing for unknown emails


def _issue(user: User) -> TokenOut:
    access, _, _ = create_token(str(user.id), user.role, "access", settings.access_token_minutes * 60, settings.secret_key)
    refresh, _, _ = create_token(str(user.id), user.role, "refresh", settings.refresh_token_days * 86400, settings.secret_key)
    return TokenOut(access_token=access, refresh_token=refresh, expires_in=settings.access_token_minutes * 60)


def _revoke(payload: dict):
    ttl = int(payload["exp"] - time.time())
    if ttl > 0:
        state.cache.set(f"revoked:{payload['jti']}", 1, ttl)


@router.post("/register", response_model=UserOut, status_code=201, dependencies=[Depends(auth_limit)])
def register(body: RegisterIn, request: Request, db: Session = Depends(get_db)):
    problems = password_problems(body.password)
    if problems:
        raise HTTPException(422, "Password " + "; ".join(problems))
    if db.scalar(select(User).where(User.email == body.email)):
        raise HTTPException(409, "Email already registered")
    user = User(email=body.email, password_hash=hash_password(body.password), full_name=body.full_name, role=ROLE_FARMER)
    db.add(user)
    db.commit()
    metrics.AUTH_EVENTS.labels("register").inc()
    audit(db, user.id, "register", {}, client_ip(request))
    return user


@router.post("/login", response_model=TokenOut, dependencies=[Depends(auth_limit)])
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    now, ip = utcnow(), client_ip(request)
    user = db.scalar(select(User).where(User.email == body.email))
    if user and user.locked_until and user.locked_until > now:
        raise HTTPException(
            429,
            "Account temporarily locked after repeated failures. Try again later.",
            headers={"Retry-After": str(int((user.locked_until - now).total_seconds()) + 1)},
        )
    ok = verify_password(body.password, user.password_hash if user else _DUMMY_HASH)
    if not user or not ok or not user.is_active:
        metrics.AUTH_EVENTS.labels("login_failed").inc()
        if user:
            user.failed_attempts += 1
            if user.failed_attempts >= MAX_FAILED:
                user.locked_until, user.failed_attempts = now + timedelta(minutes=LOCK_MINUTES), 0
                db.commit()
                audit(db, user.id, "account_locked", {}, ip)
                notify("warning", "Account locked", f"user_id={user.id} ip={ip}", dedupe_key=f"lock:{user.id}")
            else:
                db.commit()
        raise HTTPException(401, "Invalid email or password")
    user.failed_attempts, user.locked_until = 0, None
    db.commit()
    metrics.AUTH_EVENTS.labels("login_ok").inc()
    return _issue(user)


@router.post("/refresh", response_model=TokenOut, dependencies=[Depends(auth_limit)])
def refresh(body: RefreshIn, db: Session = Depends(get_db)):
    bad = HTTPException(401, "Invalid refresh token")
    try:
        payload = decode_token(body.refresh_token, settings.secret_key, "refresh")
    except TokenError:
        raise bad
    if state.cache.get(f"revoked:{payload['jti']}"):
        raise bad
    user = db.get(User, int(payload["sub"]))
    if not user or not user.is_active:
        raise bad
    _revoke(payload)  # rotation: a refresh token works once
    return _issue(user)


@router.post("/logout", status_code=204)
def logout(body: LogoutIn | None = None, creds: HTTPAuthorizationCredentials = Depends(bearer), user: User = Depends(get_current_user)):
    _revoke(decode_token(creds.credentials, settings.secret_key, "access"))
    if body and body.refresh_token:
        try:
            _revoke(decode_token(body.refresh_token, settings.secret_key, "refresh"))
        except TokenError:
            pass


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user
