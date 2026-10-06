"""Cache-aside layer. Redis when REDIS_URL is set (shared by all API replicas), otherwise an in-process
LRU+TTL cache. Cache failures never break a request: they are logged and treated as a miss."""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict

log = logging.getLogger("cache")


class MemoryBackend:
    def __init__(self, max_items: int = 5000):
        self._d, self._lock, self._max = OrderedDict(), threading.Lock(), max_items

    def get(self, key):
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < time.time():
                del self._d[key]
                return None
            self._d.move_to_end(key)
            return value

    def set(self, key, value, ttl):
        with self._lock:
            self._d[key] = (time.time() + ttl, value)
            self._d.move_to_end(key)
            while len(self._d) > self._max:
                self._d.popitem(last=False)

    def delete(self, key):
        with self._lock:
            self._d.pop(key, None)

    def ping(self):
        return True


class RedisBackend:
    def __init__(self, url: str):
        import redis

        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)

    def get(self, key):
        return self.r.get(key)

    def set(self, key, value, ttl):
        self.r.set(key, value, ex=max(int(ttl), 1))

    def delete(self, key):
        self.r.delete(key)

    def ping(self):
        return bool(self.r.ping())


class Cache:
    def __init__(self, backend, prefix: str = "cyiq:", on_event=None):
        self.backend, self.prefix, self.on_event = backend, prefix, on_event
        self.hits = self.misses = self.errors = 0

    @staticmethod
    def make_key(*parts) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _emit(self, kind):
        if self.on_event:
            try:
                self.on_event(kind)
            except Exception:
                pass

    def get(self, key):
        try:
            raw = self.backend.get(self.prefix + key)
        except Exception as e:
            self.errors += 1
            self._emit("error")
            log.warning("cache get failed: %s", e)
            return None
        if raw is None:
            self.misses += 1
            self._emit("miss")
            return None
        self.hits += 1
        self._emit("hit")
        return json.loads(raw)

    def set(self, key, value, ttl):
        try:
            self.backend.set(self.prefix + key, json.dumps(value), ttl)
        except Exception as e:
            self.errors += 1
            self._emit("error")
            log.warning("cache set failed: %s", e)

    def delete(self, key):
        try:
            self.backend.delete(self.prefix + key)
        except Exception as e:
            log.warning("cache delete failed: %s", e)

    def get_or_set(self, key, ttl, producer):
        """Return cached value, else compute with producer(), store it and return it."""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = producer()
        self.set(key, value, ttl)
        return value

    def ping(self) -> bool:
        try:
            return self.backend.ping()
        except Exception:
            return False


def build_cache(redis_url: str = "", on_event=None) -> Cache:
    if redis_url:
        return Cache(RedisBackend(redis_url), on_event=on_event)
    return Cache(MemoryBackend(), on_event=on_event)
