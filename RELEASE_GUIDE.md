# AI Trading System - Release Guide

## Guida Passo-Passo per Release Professionale

---

## 1. Pulizia Repository

### Rimuovi file temporanei locali (NON tracciati da git)

```bash
# Windows - PowerShell
Get-ChildItem -Path . -Recurse -Include "__pycache__","*.pyc","*.pyo","node_modules","dist","build" | Remove-Item -Recurse -Force
Remove-Item -Path "auto_trader.log" -Force -ErrorAction SilentlyContinue

# Oppure usa cleanup script esistente
python CLEANUP_PLAN.py
```

### Verifica file tracciati da rimuovere

```bash
git status --short
```

---

## 2. Aggiorna .gitignore (già configurato)

Il `.gitignore` attuale include:
- `__pycache__/`, `*.pyc`
- `node_modules/`, `dist/`, `build/`
- `.env` (chiavi API)
- `logs/`, `*.log`

---

## 3. Aggiorna Documentazione

### README.md (verifica esistente)

Il file attuale include:
- Descrizione progetto
- Stack tecnologico
- Quick start

### CHANGELOG.md

Aggiungi le novità recenti:

```markdown
# Changelog

## [2.1.0] - 2026-03-22

### Added
- Backtest engine con commissioni reali e slippage
- Filtro NO_TRADE_ZONE (0.45-0.55)
- Filtro MIN_CONFIDENCE < 0.6
- VaR risk control (>5% → BLOCK)
- Concept Engine con FAISS
- START_SILENT.bat per avvio silenzioso

### Improved
- Confidence calibration ML (margine + entropia)
- Walk-forward validation
- Protezione Kill Switch (max drawdown -15%)

### Fixed
- Empty DataFrame issue
- TechnicalAnalysis.to_dict() keys mancanti
- Unicode error nei log

## [2.0.0] - Previous Releases
- See CHANGELOG.md history
```

---

## 4. Test e Quality Check

### Esegui test Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=. --cov-report=html

# Coverage report
start htmlcov/index.html
```

### Linting Python

```bash
# Install linting tools
pip install flake8 black isort

# Format code
black . --line-length=88
isort . --profile=black

# Check lint
flake8 . --max-line-length=88 --ignore=E501,W503
```

### Frontend Build

```bash
cd frontend
npm install
npm run build
```

---

## 5. Version Bump

### Trova la versione attuale

```bash
# Search for version
find . -name "version*" -o -name "__version__"
grep -r "VERSION" config.py | head -5
```

### Aggiorna versione

Modifica `config.py` o crea `version.py`:

```python
# version.py
VERSION = "2.1.0"
RELEASE_DATE = "2026-03-22"
```

---

## 6. Commit e Tag

### Commit finale pre-release

```bash
git add -A
git commit -m "Release 2.1.0: backtest engine, ML confidence, risk controls"

# Push
git push origin main
```

### Crea Tag

```bash
# Tag annotato
git tag -a v2.1.0 -m "Release 2.1.0: Professional trading system with ML and risk management"

# Push tag
git push origin v2.1.0
```

---

## 7. GitHub Release

### Crea Release su GitHub

1. Vai su: https://github.com/ballales1984-wq/ai-trading-system/releases
2. Click "Draft new release"
3. Seleziona tag: `v2.1.0`
4. Titolo: `Release 2.1.0`
5. Descrizione:

```markdown
## Novità Principali

### 🧠 ML e AI
- Ensemble ML (XGBoost, RandomForest, GradientBoosting, ExtraTrees)
- 28 features per predizione
- Confidence calibration con margine + entropia
- Walk-forward validation

### 🛡️ Gestione Rischio
- NO_TRADE_ZONE (score 0.45-0.55)
- MIN_CONFIDENCE < 0.6
- VaR risk control (>5% blocca)
- Kill Switch (max drawdown -15%)

### 📊 Backtest
- Commissioni reali (0.1%)
- Slippage simulato (0.1-0.3%)
- Stop loss / Take profit

### 🚀 Performance
- 36+ asset supportati
- HMM regime detection
- Concept Engine FAISS

## Quick Start

```bash
# Backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm run dev

# AutoTrader
python main_auto_trader.py --mode live --dry-run
```

## Download

- Source code (ZIP)
- Built executable (se disponibile)
```

6. Click "Publish release"

---

## 8. Checklist Pre-Release

- [ ] Tutti i test passano
- [ ] Nessuna chiave API nel repository
- [ ] .gitignore aggiornato
- [ ] README.md completo
- [ ] CHANGELOG.md aggiornato
- [ ] Version bump fatto
- [ ] Tag creato
- [ ] Release pubblicata su GitHub
- [ ] Frontend build funziona

---

## Comandi Rapidi (Copy-Paste)

```bash
# 1. Pulizia
git clean -fd
git clean -fdx

# 2. Test
pytest tests/ -x -v

# 3. Format
black . --check
isort . --check-only

# 4. Commit
git add -A
git commit -m "Release 2.1.0"

# 5. Tag
git tag -a v2.1.0 -m "Release 2.1.0"

# 6. Push
git push origin main --follow-tags
```

---

*Ultimo aggiornamento: 2026-03-22*
