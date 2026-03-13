# AI Trading System - 100 Days Piano + Master TODO
## 🎯 APPROVED PLAN EXECUTION - Following 100 Giorni Piano

**Status**: [PLAN APPROVED ✅ | EDITS PENDING] Lint cleanup plan confirmed. Next: execute file fixes.

### 📅 100 Days Piano Mapping
- **Days 1-25**: Study codebase + local run → Complete.
- **Days 26-50**: Indicators + paper trading → Pending.
- **Days 51-75**: Flows + tests → Pending.
- **Days 76-100**: Polish + demo → Pending.

### 📋 Granular Steps from Approved Plan

#### Phase 1: Cleanup (Days 1-10) [5/5 → 6/6]
- [x] Update .gitignore (node_modules, .history, pycs, temps).
- [x] git rm --cached junk + commit \"Cleanup repo\".
- [x] Rebase dates (git rebase --root --exec date fix) or new commits.
- [x] PyInstaller slim exe (<100MB): desktop_app/ --exclude torch/matplotlib.
- [x] pytest tests/ --cov (126+ tests collecting, test_backtest.py: 6 solid tests - RUNNING ✅).
- [ ] **Lint cleanup: Fix MD013/MD012/MD022 etc. in 100_GIORNI_PIANO.md/README.md/TODO.md** [IN PROGRESS]

#### Phase 2: Core Polish (Days 11-30) [0/4]
- [ ] Backtest real data: Hook Binance/Bybit loader (fix CMC 401).
- [ ] Integrate scheduler.py + risk_engine.py tests.
- [ ] HMM full impl (remove try/except stub).
- [ ] Validate metrics not hardcoded (Sharpe/DD realistic).

#### Phase 3: Tests/Validate (Days 31-50) [0/3]
- [ ] pytest full coverage >80%.
- [ ] docker-compose up → localhost:8000 paper trading.
- [ ] Frontend dev: npm run dev → TS clean.

#### Phase 4: Demo/SaaS (Days 51-75) [0/4]
- [ ] Render/Vercel deploy.
- [ ] README badges/GIF demo.
- [ ] v2.1.1 release exe attach.
- [ ] Discord/GA4 waitlist.

#### Phase 5: Final (Days 76-100) [0/2]
- [ ] Video demo.
- [ ] Star hunt + production tests.

### 🔧 Commands to Run
```
# Lint verify after fixes
npx markdownlint-cli2 "**/*.md"
# Phase 1 complete → Phase 2
pytest tests/ -v
docker-compose up -d
```

**Next**: Execute lint fixes on 3 MD files. Update status post-edits.
**Progress**: 25% (Cleanup near-complete). Edit after Phase 1 ✅.

