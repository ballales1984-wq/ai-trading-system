"""Tests for cache management API routes."""

from fastapi.testclient import TestClient


class FakeRedisCacheManager:
    """Small fake async Redis manager for route tests."""

    def __init__(self, keys=None, connected=True):
        self.is_connected = connected
        self._keys = list(keys or [])
        self.scan_calls = 0
        self.unlinked = []

    async def scan(self, cursor=0, match=None, count=500):
        self.scan_calls += 1
        if cursor == 0:
            if match and match.endswith("*"):
                prefix = match[:-1]
                keys = [k for k in self._keys if k.startswith(prefix)]
            elif match:
                keys = [k for k in self._keys if k == match]
            else:
                keys = list(self._keys)
            return 1, keys
        return 0, []

    async def unlink(self, *keys):
        self.unlinked.extend(keys)
        return len(keys)


class FakeInMemoryCache:
    """Small fake in-memory cache for route tests."""

    def __init__(self, size=3):
        self._size = size
        self.cleared = False

    def get_stats(self):
        return {"size": self._size, "hits": 0, "misses": 0, "hit_rate": 0.0}

    def clear(self):
        self.cleared = True
        self._size = 0


def _build_client(monkeypatch, admin_key="secret123", prefix="cache:"):
    monkeypatch.setenv("ADMIN_SECRET_KEY", admin_key)
    monkeypatch.setenv("REDIS_CACHE_PREFIX", prefix)

    import app.main as main_module
    cache_routes = main_module.cache
    monkeypatch.setattr(cache_routes, "CACHE_NAMESPACE_PREFIX", prefix)
    app = main_module.app
    return app, cache_routes, TestClient(app)


def test_destructive_endpoint_requires_admin_key(monkeypatch):
    _, _, client = _build_client(monkeypatch)

    response = client.delete("/api/v1/cache/in-memory")

    assert response.status_code == 403


def test_missing_admin_secret_denies_access_fail_secure(monkeypatch):
    monkeypatch.delenv("ADMIN_SECRET_KEY", raising=False)
    from app.main import app
    client = TestClient(app)
    response = client.delete("/api/v1/cache/in-memory", headers={"X-Admin-Key": "anything"})

    assert response.status_code == 503


def test_clear_redis_rejects_pattern_outside_namespace(monkeypatch):
    _, cache_routes, client = _build_client(monkeypatch)
    fake_manager = FakeRedisCacheManager(keys=["cache:a", "other:b"])

    async def fake_get_cache_manager():
        return fake_manager

    monkeypatch.setattr(cache_routes, "get_cache_manager", fake_get_cache_manager)
    response = client.delete(
        "/api/v1/cache/redis",
        params={"pattern": "other:*"},
        headers={"X-Admin-Key": "secret123"},
    )

    assert response.status_code == 400
    assert "Pattern must start with" in response.json()["detail"]


def test_clear_redis_uses_scan_and_unlink(monkeypatch):
    _, cache_routes, client = _build_client(monkeypatch)
    fake_manager = FakeRedisCacheManager(keys=["cache:a", "cache:b", "other:x"])

    async def fake_get_cache_manager():
        return fake_manager

    monkeypatch.setattr(cache_routes, "get_cache_manager", fake_get_cache_manager)
    response = client.delete(
        "/api/v1/cache/redis",
        params={"pattern": "cache:*"},
        headers={"X-Admin-Key": "secret123"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["cleared_count"] == 2
    assert fake_manager.scan_calls >= 2
    assert fake_manager.unlinked == ["cache:a", "cache:b"]


def test_clear_all_clears_in_memory_and_namespaced_redis(monkeypatch):
    _, cache_routes, client = _build_client(monkeypatch)
    fake_manager = FakeRedisCacheManager(keys=["cache:k1", "cache:k2", "other:k3"])
    fake_in_memory = FakeInMemoryCache(size=4)

    async def fake_get_cache_manager():
        return fake_manager

    def fake_get_cache():
        return fake_in_memory

    monkeypatch.setattr(cache_routes, "get_cache_manager", fake_get_cache_manager)
    monkeypatch.setattr(cache_routes, "IN_MEMORY_CACHE_AVAILABLE", True)
    monkeypatch.setattr(cache_routes, "get_cache", fake_get_cache)
    response = client.delete(
        "/api/v1/cache/",
        headers={"X-Admin-Key": "secret123"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["cleared_count"] == 6
    assert fake_in_memory.cleared is True
    assert fake_manager.unlinked == ["cache:k1", "cache:k2"]
