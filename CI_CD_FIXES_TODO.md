# CI/CD Fixes TODO

## Task: Fix 6 Failing CI/CD Checks

### Issues Fixed:
1. [x] Add missing dependencies to requirements.txt
2. [x] Add missing event_log table to StateManager
3. [x] Fix Python version consistency in CI workflows
4. [x] Verify all fixes

### Summary of Changes Made:

#### 1. requirements.txt
- Added missing CI/CD dependencies:
  - Code Quality: ruff, isort, mypy, radon
  - Testing: pytest-xdist, pytest-timeout, pytest-html
  - Security: bandit, pip-audit, safety
  - Coverage: codecov
- Fixed pytest-asyncio version (1.3.0 -> 0.23.8)

#### 2. src/core/state_manager.py
- Added missing `event_log` table to database initialization
- Added indexes for event_log table

#### 3. .github/workflows/python-app.yml
- Changed Python version from 3.12 to 3.11 for consistency

#### 4. .github/workflows/ci-cd-production.yml
- Already uses Python 3.11 (confirmed consistency)

### Files Modified:
- requirements.txt
- src/core/state_manager.py
- .github/workflows/python-app.yml
- .github/workflows/ci-cd-production.yml

