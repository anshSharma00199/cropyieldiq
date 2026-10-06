"""Framework-free security primitives: password hashing (scrypt), JWT tokens, password policy."""

import base64
import hashlib
import hmac
import os
import re
import time
import uuid

import jwt  # PyJWT

_N, _R, _P = 2**14, 8, 1


class TokenError(Exception):
    pass


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(_N, _R, _P, base64.b64encode(salt).decode(), base64.b64encode(dk).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt_b64, dk_b64 = stored.split("$")
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(dk_b64)
        dk = hashlib.scrypt(password.encode(), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def password_problems(password: str) -> list:
    problems = []
    if len(password) < 10:
        problems.append("must be at least 10 characters")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        problems.append("must contain letters and digits")
    if password.lower() in {"password123", "1234567890", "qwertyuiop"}:
        problems.append("is too common")
    return problems


def create_token(subject: str, role: str, token_type: str, lifetime_seconds: int, secret: str) -> tuple:
    now = int(time.time())
    jti = uuid.uuid4().hex
    payload = {"sub": subject, "role": role, "type": token_type, "iat": now, "exp": now + lifetime_seconds, "jti": jti}
    return jwt.encode(payload, secret, algorithm="HS256"), jti, payload["exp"]


def decode_token(token: str, secret: str, expected_type: str) -> dict:
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["exp", "sub", "jti"]})
    except jwt.PyJWTError as e:
        raise TokenError(str(e))
    if payload.get("type") != expected_type:
        raise TokenError("wrong token type")
    return payload
