# Project Finalization Tracker

Last updated: 2026-02-23
Scope: close project end-to-end with working data/API/news/forms and stable deploy

## Final Goal

Deliver a stable release where:
- frontend shows live prices, charts, news feed, orders, portfolio
- marketing waitlist form works reliably
- backend APIs are reachable in deployed environment
- core tests and regression checks pass
- docs reflect real status (no false "100% complete")

## Done Criteria (Release Gate)

- [x] `API_GATE` `/api/v1/market/prices` returns `200` in local runtime
- [x] `API_GATE` `/api/v1/market/candles/{symbol}` endpoint available in backend route map
- [x] `API_GATE` `/api/v1/market/news` returns JSON payload on in-process test client
- [x] `API_GATE` `/api/v1/waitlist` count path verified; create/duplicate flow covered in route logic
- [x] `UI_GATE` dashboard renders prices/charts without hard dependency on fallback-only mode
- [x] `UI_GATE` dashboard news list path implemented and wired to API client
- [x] `UI_GATE` orders create/execute/cancel paths show actionable errors
- [x] `UI_GATE` emergency stop modal works (activate/deactivate wiring completed)
- [x] `OPS_GATE` deploy routing (Vercel rewrite or direct API) documented and repeatable
- [x] `TEST_GATE` targeted frontend/backend checks pass
- [x] `DOC_GATE` checklist/docs updated with real status

## Workstreams

### WS1 - Runtime Connectivity
- [x] Stabilize backend exposure method (local+ngrok script available and tested)
- [ ] Remove fragile hardcoded endpoints from deploy path
- [x] Verify CORS + headers for ngrok/Vercel proxy (added ngrok skip-warning headers)
- [x] Add quick health diagnostics command block

### WS2 - Data + News Pipeline
- [x] Confirm provider keys loaded (`.env` inspection captured)
- [x] Validate `market/news` with provider fallback behavior
- [x] Ensure frontend consumes and displays provider payload safely
- [x] Add clear UI message when provider unavailable

### WS3 - Frontend Completion
- [x] Final pass on dashboard states (loading/error/empty)
- [x] Final pass on marketing form behavior
- [x] Final pass on responsive tables/charts and critical actions
- [x] Freeze UI for release candidate

### WS4 - Testing + Regression
- [x] Run targeted API smoke checks
- [x] Run frontend type checks
- [x] Run selected backend syntax/test checks for changed modules
- [x] Capture pass/fail evidence in this tracker

### WS5 - Release + Documentation
- [x] Align branch and deployment flow
- [x] Update all status docs to factual state
- [x] Create final release checklist with exact commands
- [ ] Mark release candidate + go/no-go decision

## Execution Log

- 2026-02-23: tracker created and linked from `TODO_CHECKLIST.md`
- 2026-02-23: local API smoke passed (`/health`, `/api/v1/market/prices`, `/api/v1/market/news` => 200).
- 2026-02-23: provider keys missing in current shell env (`NEWSAPI_KEY`, `COINMARKETCAP_API_KEY`, `BENZINGA_API_KEY`, `TWITTER_BEARER_TOKEN`, `ADMIN_SECRET_KEY`).
- 2026-02-23: `.env` check: `NEWSAPI_KEY` and `COINMARKETCAP_API_KEY` set, `BENZINGA_API_KEY`/`TWITTER_BEARER_TOKEN` empty, `ADMIN_SECRET_KEY` absent.
- 2026-02-23: Added `market/news` timeout guard to prevent provider hang from blocking dashboard.
- 2026-02-23: Added frontend telemetry sender and backend ingestion endpoint for runtime errors.
- 2026-02-23: Completed frontend state-standardization pass (skeleton/error/empty) on portfolio/market/orders.
- 2026-02-23: `npx tsc --noEmit` passed after final changes.
- 2026-02-23: Removed hardcoded ngrok rewrite from `vercel.json`; API path now deploy-stable.
- 2026-02-23: Aligned `api/index.py` with frontend dependencies (`/api/v1/market/news`, `/api/v1/health/client-events`).
- 2026-02-23: Closed RC freeze for frontend/dashboard scope; ready for focused debug pass.

## Commands (Operational)

```bash
# backend local
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# ngrok tunnel
powershell -ExecutionPolicy Bypass -File scripts/dev_start_ngrok.ps1

# frontend type check
cd frontend && npx tsc --noEmit

# API smoke
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/v1/market/prices
curl "http://127.0.0.1:8000/api/v1/market/news?query=bitcoin&limit=3"
```
