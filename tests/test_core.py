"""Framework-free unit tests: cache, rate limiter, security primitives, agronomy rules."""

import time

from app.core.cache import Cache, MemoryBackend
from app.core.ratelimit import MemoryWindow, RateLimiter
from app.core.security import TokenError, create_token, decode_token, hash_password, password_problems, verify_password
from app.services.agronomy import explain


class BrokenBackend:
    def get(self, k):
        raise ConnectionError("down")

    def set(self, k, v, ttl):
        raise ConnectionError("down")

    def hit(self, k, w):
        raise ConnectionError("down")

    def delete(self, k):
        raise ConnectionError("down")

    def ping(self):
        raise ConnectionError("down")


def test_cache_get_set_and_expiry():
    c = Cache(MemoryBackend())
    c.set("a", {"x": 1}, ttl=60)
    assert c.get("a") == {"x": 1}
    c.set("b", 1, ttl=0.01)
    time.sleep(0.05)
    assert c.get("b") is None
    assert c.hits == 1 and c.misses == 1


def test_cache_get_or_set_runs_producer_once():
    c, calls = Cache(MemoryBackend()), []
    producer = lambda: calls.append(1) or {"v": 42}
    assert c.get_or_set("k", 60, producer) == {"v": 42}
    assert c.get_or_set("k", 60, producer) == {"v": 42}
    assert len(calls) == 1


def test_cache_failure_never_breaks_requests():
    c = Cache(BrokenBackend())
    assert c.get_or_set("k", 60, lambda: "computed") == "computed"
    assert c.errors >= 1 and c.ping() is False


def test_cache_keys_are_stable_and_input_sensitive():
    assert Cache.make_key("p", {"a": 1, "b": 2}) == Cache.make_key("p", {"b": 2, "a": 1})
    assert Cache.make_key("p", {"a": 1}) != Cache.make_key("p", {"a": 2})


def test_rate_limiter_blocks_after_limit():
    rl = RateLimiter(MemoryWindow())
    results = [rl.check("t", "1.1.1.1", 3, 60) for _ in range(4)]
    assert [r[0] for r in results] == [True, True, True, False]
    assert results[2][1] == 0 and results[3][2] > 0  # remaining 0, retry_after > 0
    assert rl.check("t", "2.2.2.2", 3, 60)[0] is True  # other clients unaffected


def test_rate_limiter_window_resets():
    rl = RateLimiter(MemoryWindow())
    assert rl.check("t", "x", 1, 1)[0] and not rl.check("t", "x", 1, 1)[0]
    time.sleep(1.1)
    assert rl.check("t", "x", 1, 1)[0]


def test_rate_limiter_fails_open_when_backend_down():
    assert RateLimiter(BrokenBackend()).check("t", "x", 1, 60)[0] is True


def test_password_hash_and_verify():
    h = hash_password("S3cure-pass-1")
    assert h != hash_password("S3cure-pass-1")  # unique salt
    assert verify_password("S3cure-pass-1", h) and not verify_password("wrong", h)
    assert not verify_password("x", "garbage")


def test_password_policy():
    assert password_problems("short1")
    assert password_problems("onlyletterslong")
    assert password_problems("GoodPass12345") == []


def test_jwt_roundtrip_and_rejections():
    secret = "k" * 40
    tok, jti, _ = create_token("7", "farmer", "access", 60, secret)
    p = decode_token(tok, secret, "access")
    assert p["sub"] == "7" and p["role"] == "farmer" and p["jti"] == jti
    for bad in (
        lambda: decode_token(tok, secret, "refresh"),  # wrong type
        lambda: decode_token(tok, "other" * 10, "access"),  # wrong key
        lambda: decode_token(tok + "x", secret, "access"),  # tampered
        lambda: decode_token(create_token("7", "f", "access", -5, secret)[0], secret, "access"),
    ):  # expired
        try:
            bad()
            raise AssertionError("expected TokenError")
        except TokenError:
            pass


def test_agronomy_flags_low_and_high_values():
    ranges = {
        "rice": {
            "N": {"p10": 60, "p50": 80, "p90": 100},
            "ph": {"p10": 5.5, "p50": 6.4, "p90": 7.2},
        }
    }

    out = explain("rice", {"N": 20, "ph": 6.4}, ranges)
    status = {s["feature"]: s["status"] for s in out["feature_status"]}

    assert status == {"N": "low", "ph": "ok"}
    assert "Nitrogen" in out["advice"][0]
    assert "training data" in out["advice"][0]

    within = explain("rice", {"N": 80, "ph": 6.4}, ranges)
    assert within["feature_status"][0]["status"] == "ok"
    assert "within the range observed" in within["advice"][0]


def test_agronomy_low_rainfall_does_not_prescribe_irrigation():
    ranges = {"rice": {"rainfall": {"p10": 190, "p50": 230, "p90": 280}}}

    out = explain("rice", {"rainfall": 100}, ranges)

    assert out["feature_status"][0]["status"] == "low"
    assert any("below the range observed" in msg for msg in out["advice"])
    assert not any("irrigation" in msg.lower() for msg in out["advice"])


def test_agronomy_high_nutrients_does_not_recommend_fertilizer():
    ranges = {
        "rice": {
            "N": {"p10": 60, "p50": 80, "p90": 100},
            "P": {"p10": 35, "p50": 45, "p90": 60},
            "K": {"p10": 35, "p50": 40, "p90": 45},
        }
    }

    out = explain("rice", {"N": 120, "P": 70, "K": 50}, ranges)

    assert all(s["status"] == "high" for s in out["feature_status"])

    nutrient_advice = [msg for msg in out["advice"] if "above the range observed" in msg]
    assert len(nutrient_advice) == 3

    assert not any(word in msg.lower() for msg in out["advice"] for word in ("fertilizer", "potash", "top-up", "skip additional"))


def test_agronomy_unknown_crop_returns_no_feature_status():
    out = explain("unknown_crop", {"N": 50}, {})

    assert out["feature_status"] == []
    assert any("do not establish" in msg for msg in out["advice"])
    assert not any("inside the typical range" in msg for msg in out["advice"])
