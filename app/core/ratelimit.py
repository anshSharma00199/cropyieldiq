"""Fixed-window rate limiter with Redis (shared across replicas) or in-memory backends.
Fails OPEN if Redis is down (availability over strictness); nginx still limits at the edge."""

import logging
import threading
import time

log = logging.getLogger("ratelimit")


class MemoryWindow:
    def __init__(self):
        self._d, self._lock = {}, threading.Lock()

    def hit(self, key: str, window: int):
        now = time.time()
        with self._lock:
            start, count = self._d.get(key, (now, 0))
            if now - start >= window:
                start, count = now, 0
            count += 1
            self._d[key] = (start, count)
            if len(self._d) > 20000:  # purge expired keys so memory stays bounded
                self._d = {k: v for k, v in self._d.items() if now - v[0] < window}
            return count, max(window - (now - start), 0)


class RedisWindow:
    def __init__(self, url: str):
        import redis

        self.r = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)

    def hit(self, key: str, window: int):
        pipe = self.r.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        count, ttl = pipe.execute()
        if ttl is None or ttl < 0:
            self.r.expire(key, window)
            ttl = window
        return int(count), int(ttl)


class RateLimiter:
    def __init__(self, backend):
        self.backend = backend

    def check(self, bucket: str, ident: str, limit: int, window: int):
        """Returns (allowed, remaining, retry_after_seconds)."""
        try:
            count, ttl = self.backend.hit(f"rl:{bucket}:{ident}", window)
        except Exception as e:
            log.warning("rate limiter backend error, failing open: %s", e)
            return True, limit, 0
        allowed = count <= limit
        return allowed, max(limit - count, 0), 0 if allowed else int(ttl) + 1


def build_limiter(redis_url: str = "") -> RateLimiter:
    return RateLimiter(RedisWindow(redis_url) if redis_url else MemoryWindow())
