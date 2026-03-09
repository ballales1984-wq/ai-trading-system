# TODO: Cache Restart & Clear Implementation
<!-- markdownlint-disable MD012 MD022 MD026 MD032 -->

## Steps:
1. [x] Analyze existing cache implementations
2. [x] Create cache management API route (app/api/routes/cache.py)
3. [x] Update app/main.py to include cache router
4. [x] Add admin authentication for destructive endpoints (`X-Admin-Key` + `ADMIN_SECRET_KEY`)
5. [x] Harden Redis clear operations (`SCAN` + `UNLINK`, no blocking `KEYS`)
6. [x] Restrict Redis clear to cache namespace (`REDIS_CACHE_PREFIX`, default `cache:`)
7. [x] Add API tests for cache routes (`tests/test_cache_routes.py`)
8. [x] Run cache route tests (`pytest -q tests/test_cache_routes.py` -> 5 passed)

