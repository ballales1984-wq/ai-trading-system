# 🧹 PIANO DI PULIZIA - AI Trading System

## 📋 FILE DA ELIMINARE

### 1. File Log (vecchi/obsoleti)
- [ ] `auto_trader.log`
- [ ] `backend.log`
- [ ] `execution_log.log`
- [ ] `build_kivy_latest.log` (957MB!)

### 2. File Temporanei
- [ ] `temp.html`
- [ ] `temp_orders.json`
- [ ] `city_signals_summary.csv`
- [ ] `city_signals_timeline.csv`
- [ ] `health_tmp.pyc.2083572273392`

### 3. File Obsoleti/Doppi (tenere solo versione principale)
- [ ] `api/index.py` (usa `app/main.py`)
- [ ] `api_server.py` (usa `app/main.py`)
- [ ] `start_backend.py` (usa `start_fastapi.py`)
- [ ] `start_dashboard_api.py` (usa `dashboard.py`)
- [ ] `config_fallback.py` (usa `config.py`)
- [ ] `main_old.py` (obsoleto)
- [ ] `restore.py` (tenere solo restore_dashboard.py)

### 4. Script Build Obsoleti
- [ ] `build_exe.bat`
- [ ] `build_exe.ps1`
- [ ] `build_exe.py`
- [ ] `ai_trading_kivy_desktop.spec`
- [ ] `ai_trading_system.spec`
- [ ] `main.spec`

### 5. File Documentazione Ridondanti (tenere solo README.md)
- [ ] `ACCESSIBILITY_OPTIMIZATION.md`
- [ ] `API_FLOW_DIAGRAM.md`
- [ ] `API_INTEGRATION_ARCHITECTURE.md`
- [ ] `ARCHITECTURE.md`
- [ ] `ARCHITECTURE_INTEGRATION.md`
- [ ] `CHECKLIST_MANC`
- [ ] `CHECKLIST_MANCANZE.md`
- [ ] `CI_CD_FIXES_TODO.md`
- [ ] `COMPONENT_DIAGRAM.md`
- [ ] `DASHBOARD_README.md`
- [ ] `DASHBOARD_UX_IMPROVEMENTS.md`
- [ ] `DEPLOYMENT.md`
- [ ] `DEPLOYMENT_DECISION.md`
- [ ] `DEPLOYMENT_STATUS.md`
- [ ] `DEPLOYMENT_SUMMARY.md`
- [ ] `FRONTEND_OPTIMIZATION.md`
- [ ] `HARDENING_PLAN.md`
- [ ] `IMPROVEMENT_PLAN.md`
- [ ] `LEGAL.md`
- [ ] `MAINTENANCE_PLAN.md`
- [ ] `MARKETING_PLAN.md`
- [ ] `PROJECT_ANALYSIS.md`
- [ ] `PROJECT_DOCUMENTATION_COMPLETE.md`
- [ ] `PROJECT_SUMMARY.md`
- [ ] `PULIZIA_ORDINE.md`
- [ ] `ROADMAP.md`
- [ ] `ROADMAP_SAAS.md`
- [ ] `ROADMAP_VISIVA.md`
- [ ] `SPIEGAZIONE.md`
- [ ] `STABILIZATION_PLAN.md`
- [ ] `STATO_PROGETTO.md`
- [ ] `STABLE_RELEASE.md`
- [ ] `SYSTEM_CHECKLIST.md`
- [ ] `TEST_MAP.txt`
- [ ] `TEST_DEBUG_PLAN.md`
- [ ] `TEST_INTEGRATION_REPORT.md`

### 6. File Dockerfile Backup
- [ ] `Dockerfile.backup`
- [ ] `Dockerfile.render` (se non usato)
- [ ] `Dockerfile.render.optimized` (se non usato)

### 7. File Docker Ridondanti
- [ ] `docker-compose.hedgefund.yml` (tenere solo production e stable)
- [ ] `docker-compose.stable.yml` (verificare se necessario)
- [ ] `DOCKER_CONTAINERS.txt`
- [ ] `DOCKER_IMAGES.txt`

### 8. File Vari
- [ ] `.coverage` (cache coverage)
- [ ] `coverage.json` (cache coverage)
- [ ] `ChatGPT Image 18 feb 2026, 01_18_39.png`

---

## ✅ FILE DA MANTENERE

### Core Application
- `app/` - Backend principale
- `frontend/` - Frontend React
- `src/` - Trading engine legacy

### Test (NON TOCCARE)
- `tests/` - Tutti i file di test

### Script di Avvio Principali
- `start_backend.py`
- `start_fastapi.py`
- `dashboard.py`
- `start_frontend_and_backend.ps1`
- `start_frontend_online.ps1`
- `run_tests.bat`

### Configurazione
- `config.py`
- `.env`
- `.env.example`
- `requirements.txt`
- `docker-compose.yml`
- `docker-compose.production.yml`

### Documentazione Principale
- `README.md`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `LICENSE`
- `DISCORD_INVITE.md`

---

## 🚀 ESECUZIONE PULIZIA

Dopo aver creato questo piano, eseguire:

```powershell
# Entrare nella cartella del progetto
cd c:/ai-trading-system

# Eliminare file log
Remove-Item *.log -Force

# Eliminare file temporanei
Remove-Item temp.html, temp_orders.json, city_signals*.csv, health_tmp.pyc.* -Force

# Eliminare file obsoleti (esempi)
Remove-Item api_server.py, start_backend.py, main_old.py -Force
```

---

## 📊 RIEPILOGO POST-PULIZIA

Dopo la pulizia, la struttura dovrebbe essere:

```
ai-trading-system/
├── app/                    ← Backend FastAPI (principale)
├── frontend/               ← Frontend React
├── src/                    ← Trading engine
├── tests/                  ← Test (800+)
├── docker/                 ← Docker configs
├── scripts/                ← Script ausiliari
├── logs/                   ← Log attivi
├── data/                   ← Dati
├── public/                 ← File pubblici
├── .env                    ← Config
├── config.py               ← Config
├── requirements.txt        ← Dipendenze
├── docker-compose.yml      ← Docker
├── README.md               ← Docs
└── start_*.py / start_*.ps1 ← Avvio
```

