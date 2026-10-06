"""FastAPI dependencies: client IP, rate limiting, authentication, role-based access control."""

from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app import metrics, state
from app.config import settings
from app.core.security import TokenError, decode_token
from app.database import get_db
from app.models import User

bearer = HTTPBearer(auto_error=False)


def client_ip(request: Request) -> str:
    # Only trust the proxy header when we really sit behind our own nginx / load balancer.
    if settings.trust_proxy:
        ip = request.headers.get("x-real-ip")
        if ip:
            return ip.strip()
    return request.client.host if request.client else "unknown"


def _enforce(bucket: str, ident: str, rule: tuple, response: Response):
    limit, window = rule
    allowed, remaining, retry_after = state.limiter.check(bucket, ident, limit, window)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    if not allowed:
        metrics.RATE_LIMITED.labels(bucket).inc()
        raise HTTPException(429, "Too many requests. Please slow down.", headers={"Retry-After": str(retry_after)})


def ip_limit(request: Request, response: Response):
    _enforce("ip", client_ip(request), settings.rl_ip, response)


def auth_limit(request: Request, response: Response):
    _enforce("auth", client_ip(request), settings.rl_auth, response)


def get_current_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer), db: Session = Depends(get_db)) -> User:
    unauthorized = HTTPException(401, "Invalid or missing credentials", headers={"WWW-Authenticate": "Bearer"})
    if creds is None:
        raise unauthorized
    try:
        payload = decode_token(creds.credentials, settings.secret_key, "access")
    except TokenError:
        raise unauthorized
    if state.cache.get(f"revoked:{payload['jti']}"):
        raise unauthorized
    user = db.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise unauthorized
    request.state.user_id = user.id
    return user


def user_limit(response: Response, user: User = Depends(get_current_user)):
    _enforce("user", str(user.id), settings.rl_user, response)


def require_roles(*roles: str):
    def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(403, "You do not have permission to do this")
        return user

    return checker
