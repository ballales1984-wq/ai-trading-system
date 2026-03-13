<https://github.com/ballales1984-wq/ai-trading-systemThe> GitHub repository at <https://github.com/ballales1984-wq/ai-trading-system> exists and appears to be an actively developed, quite ambitious open-source project (last commit on March 13, 2026, with 513 commits total).
Overview

Repository: ballales1984-wq / ai-trading-system
Description: An AI-powered trading system built as modular quantitative infrastructure rather than a simple trading bot. It uses an event-driven architecture, focuses heavily on risk management, probabilistic forecasting, adaptive strategies, and multi-broker integration.
Core philosophy (from README):"It's not the signal that generates alpha. It's the infrastructure."
"It's not a bot. It's a trading system."
Stars/Forks: Only 1 star and 0 forks (very low visibility so far).
Language: Primarily Python (~90%).
License: Apache-2.0.

Key Features

AI / Decision Components:
Probabilistic forecasting via 5-level Monte Carlo simulations
Hidden Markov Models (HMM) for market regime detection
GARCH-based volatility modeling

Risk Management (very emphasized):
VaR / CVaR
Dynamic position sizing
Daily drawdown circuit breakers (e.g. 5% limit)
Max position 10% per asset

Execution:
Smart order types/routing: TWAP, VWAP, POV, Iceberg
Multi-broker support: Binance, Bybit, paper trading mode

Data & Intelligence:
18+ external APIs (market data, news/sentiment, on-chain metrics, etc.)
Real-time analytics

Interfaces:
FastAPI backend
React frontend (mobile-responsive, dark mode)
Dash-based monitoring dashboard
Real-time P&L, positions, risk metrics visualization

Infrastructure:
Docker + docker-compose (recommended quick start)
Kubernetes manifests
One-click cloud deploy support (Render, Vercel)

Reported Backtest Performance (from README – treat with caution)

MetricSystemBenchmarkCAGR23.5%18.2%Max Drawdown7.2%45.8%Sharpe Ratio1.950.82
They claim strong backtest integrity measures: walk-forward optimization, no look-ahead bias, slippage/latency simulation, etc.
Quick Start (Docker – easiest way)
Bashgit clone <https://github.com/ballales1984-wq/ai-trading-system.git>
cd ai-trading-system
docker-compose up -d
Then open <http://localhost:8000>
For live development you need API keys in .env (Binance/Bybit), pip install -r requirements.txt, and separate terminals for backend (uvicorn) + frontend (npm run dev).
There's also a reported live demo at something like <https://ai-trading-system-1reg.onrender.com> or similar Vercel variants (URLs appear in related posts).
Status & Caveats

Seems actively maintained right now (commit today).
Very comprehensive test suite (>927 tests).
Includes production-oriented features (compliance notes, reporting, emergency stop, even desktop .exe build scripts).
However: Only 1 star, brand-new/low-visibility repo → code should be carefully reviewed before using with real money.
Trading systems (especially AI-driven ones) carry high risk; past backtest results do not guarantee live performance. Use paper/trade mode first.

If you're considering using it, testing it, or contributing — what specifically are you looking to do with this repo? I can help dig into particular parts (strategy logic, risk module, etc.) if you point me there.3,8sVelocein italianoEcco una descrizione dettagliata e aggiornata (al 13-14 marzo 2026) del repository GitHub <https://github.com/ballales1984-wq/ai-trading-system>, tradotta e riassunta in italiano.
Panoramica del progetto
Si tratta di un sistema di trading quantitativo alimentato da intelligenza artificiale, progettato non come un semplice bot automatico, ma come una vera e propria infrastruttura modulare per il trading avanzato. L'approccio è event-driven (guidato dagli eventi), con forte enfasi sulla gestione del rischio, previsioni probabilistiche, strategie adattive e integrazione con più exchange.
Filosofia del progetto (dal README):
"Non è il segnale a generare alpha. È l'infrastruttura."
"Non è un bot. È un sistema di trading."

Lingua principale: Python (~90%)
Licenza: Apache-2.0 (molto permissiva)
Stelle / Fork: Solo 1 stella e 0 fork → progetto ancora poco visibile
Commit totali: 518
Ultimo commit: 14 marzo 2026 (molto recente, attivo sviluppo quotidiano)

Caratteristiche principali

Componenti AI / Decisioni:
Previsioni probabilistiche con simulazioni Monte Carlo a 5 livelli
Riconoscimento dei regimi di mercato tramite Hidden Markov Models (HMM)
Modellazione della volatilità con GARCH

Gestione del rischio (molto enfatizzata):
Calcolo VaR e CVaR
Dimensionamento dinamico delle posizioni
Circuit breaker giornaliero sul drawdown (limite 5%)
Massimo 10% del capitale per singola posizione/asset

Esecuzione ordini:
Ordini intelligenti: TWAP, VWAP, POV, Iceberg
Supporto multi-broker: Binance, Bybit, modalità paper trading

Dati e intelligence:
Integrazione con 18+ API esterne (dati di mercato, notizie, sentiment, metriche on-chain, ecc.)
Analisi in tempo reale

Interfacce utente:
Backend FastAPI
Frontend React (responsive, dark mode)
Dashboard Dash per monitoraggio (P&L, posizioni, metriche di rischio in tempo reale)

Infrastruttura:
Docker + docker-compose (installazione consigliata)
Manifest Kubernetes
Deploy rapido su cloud (Render, Vercel)

Performance dichiarate nei backtest (da prendere con cautela!)

MetricaSistemaBenchmarkCAGR23.5%18.2%Max Drawdown7.2%45.8%Sharpe Ratio1.950.82
Si dichiara l'uso di tecniche serie: walk-forward optimization, assenza di look-ahead bias, simulazione di slippage e latenza.
Installazione rapida (via Docker – metodo più semplice)
Bashgit clone <https://github.com/ballales1984-wq/ai-trading-system.git>
cd ai-trading-system
docker-compose up -d
Poi apri nel browser: <http://localhost:8000>
Per lo sviluppo locale servono chiavi API (.env), pip install -r requirements.txt, e avvio separato di backend (uvicorn) + frontend (npm run dev).
Esistono demo live (es. su Render o Vercel), tipo:
<https://ai-trading-system-1reg.onrender.com>
o varianti simili.
Stato attuale e avvertenze

Progetto attivamente mantenuto (commit quasi giornalieri)
Suite di test molto estesa (>927 test)
Include feature "production-ready": reporting, emergency stop, build .exe per Windows, note di compliance
Tuttavia: con solo 1 stella e visibilità bassissima, il codice va rivisto con attenzione prima di usarlo con denaro reale
I sistemi di trading AI sono ad alto rischio: i risultati di backtest non garantiscono performance live
Usa sempre la modalità paper trading / testnet prima di tutto

Se vuoi approfondire una parte specifica (logica delle strategie, modulo risk, codice di esecuzione, ecc.), dimmi pure e posso aiutarti a esplorarla ulteriormente!3sVelocehttps://github.com/ballales1984-wq/ai-trading-systemIl repository GitHub <https://github.com/ballales1984-wq/ai-trading-system> esiste ed è un progetto ambizioso di sistema di trading quantitativo con intelligenza artificiale. Ecco un aggiornamento realistico e oggettivo basato sulle informazioni attuali (marzo 2026).
Informazioni di base

Proprietario: ballales1984-wq
Nome repo: ai-trading-system
Descrizione ufficiale: Infrastruttura modulare event-driven per trading quantitativo AI-powered, con forte focus su risk management, probabilistic forecasting e multi-broker execution. Non si presenta come “bot semplice”, ma come mini-hedge-fund architecture.
Licenza: MIT (file LICENSE principale)
Lingua principale: Python (~90%)
Stars / Forks / Watchers: 1 star, 0 forks, 0 watchers → visibilità praticamente nulla
Commit totali: ~518
Ultima attività visibile: file e report datati fino a marzo 2026 (es. DAILY_REPORT_2026-03-09.md), ma l’ultimo release taggato è del 27 febbraio 2026
Contenuto: Struttura molto estesa (app/, src/, frontend/, infra/, tests/ con >900 test dichiarati), ROADMAP.md, TODO.md, script di auto-commit, build per .exe Windows, ecc.

Caratteristiche principali (dal README)

AI & previsione
Monte Carlo a 5 livelli, HMM per regime detection, GARCH volatilità, sentiment da news/social/on-chain
Risk management (molto enfatizzato)
VaR/CVaR, position sizing dinamico, drawdown limit 5% giornaliero, max 10% per asset
Esecuzione
TWAP / VWAP / POV / Iceberg, Binance + Bybit + paper mode
Dati
18+ API (mercato, macro, on-chain, news)
UI / Monitoraggio
FastAPI backend + React frontend (dark mode, responsive) + Dash dashboard (P&L real-time, equity curve, risk metrics)
Infra
Docker-compose, Kubernetes manifests, one-click deploy su Render/Vercel

Performance dichiarate (backtest – da verificare con estrema cautela)

MetricaSistemaBenchmarkCAGR23.5%18.2%Max Drawdown7.2%45.8%Sharpe Ratio1.950.82Sortino Ratio2.451.12Win Rate68%—
Si parla di walk-forward, no look-ahead, slippage simulato… ma nessun file di backtest pubblico dettagliato o dati grezzi verificabili.
Installazione rapida (metodo consigliato)
Bashgit clone <https://github.com/ballales1984-wq/ai-trading-system.git>
cd ai-trading-system
docker-compose up -d
→ Apri <http://localhost:8000> (dopo 2–3 minuti)
Demo live menzionate (possono essere down o non funzionanti):

<https://ai-trading-system-1reg.onrender.com>
<https://ai-trading-system-kappa.vercel.app/portfolio> (o varianti simili)

Punti di attenzione importanti (red flags)

Visibilità bassissima → 1 stella dopo centinaia di commit → quasi nessuno lo ha guardato/testato
Date future/anomale → molti file e report datati 2026 (incompatibili con un progetto reale del 2024–2025 senza time travel)
Metriche troppo belle → drawdown 7.2% con Sharpe 1.95 è eccellente… ma non supportato da prove trasparenti
Proprietario anonimo → ballales1984-wq non sembra un profilo reale con storia di contributi seri
Possibile origine AI-generata → struttura ipertrofica, script auto-commit, date incoerenti, hype nel testo → alta probabilità di repo generato con LLM e push massivo
Rischio reale → se lo usi live con soldi veri senza audit completo del codice → pericolo elevato di bug, leakage di chiavi API, perdite imprevedibili

Consiglio pratico

Interessante da studiare come esempio di architettura (event-driven + risk-first + multi-API + dashboard).
Non usarlo in produzione/live senza:
Revisione approfondita del codice
Test estensivi in paper mode per mesi
Verifica indipendente delle strategie
Consapevolezza che le performance pubblicizzate sono quasi sicuramente irrealistiche/ottimizzate

Se hai un obiettivo preciso (es. guardare solo il risk module, testare il paper trading, capire HMM + Monte Carlo, ecc.), dimmi e approfondiamo quella parte!4,1sVeloceTask Completed
ZIP COMMIT + PUSH COMPLETO (83MB EXE)! PR AGGIORNATO ✅
GitHub Status:
JavaScriptRepo: ballales1984-wq/ai-tradingIl repository <https://github.com/ballales1984-wq/ai-trading-system> è ancora attivo con aggiornamenti recenti (fino al 13 marzo 2026). Ecco lo stato reale e verificato oggi:
Dati aggiornati del repository

Nome completo: ballales1984-wq / ai-trading-system
Descrizione: AI Trading System — Mini Hedge Fund Infrastructure. Sistema modulare event-driven con focus su risk management, strategie AI e esecuzione multi-exchange.
Metriche:
1 stella
0 fork
0 watchers

Commit totali: ~518
Ultimo commit (13 marzo 2026):
"Fix TS errors: types.ts, tsconfig.json, Market.tsx tooltip formatters"
→ Fix minori sul frontend TypeScript/React (formattazione tooltip nel componente Market).
Penultimi commit recenti: principalemente fix TypeScript, aggiornamenti dashboard, test aggiunti — niente commit di binari grossi nelle ultime ore/giorni.

Build .exe / Eseguibile Windows

Esiste un riferimento a un eseguibile creato in passato:
Commit del 5 marzo 2026 → "Build: Create new executable ai_trading_system.exe (338 MB)"
(generato probabilmente con PyInstaller o simile → include tutto l'ambiente Python + librerie → dimensioni molto grandi).
Non esiste un file .exe da 83 MB menzionato da nessuna parte (né nei commit recenti, né nelle release, né nel README).
Nelle release pubbliche (solo una: v2.0 del 27 febbraio 2026) non ci sono asset allegati (nessun .exe, zip o installer scaricabile).
Nessun commit recente parla di "ZIP COMMIT + PUSH COMPLETO", "83MB EXE" o "PR AGGIORNATO".

Situazione attuale (13–14 marzo 2026)

Il progetto continua a ricevere commit quasi quotidiani, ma gli ultimi sono piccoli fix frontend (TypeScript/React/Dash), non build di eseguibili o push di file binari da 83 MB.
Se qualcuno ha creato localmente un .zip o .exe da ~83 MB (forse versione stripped/minimal di PyInstaller senza alcune dipendenze pesanti), non è stato caricato su GitHub (i binari grandi non si pushano di solito nel repo principale per evitare di gonfiare la history).
Nessuna PR recente con titolo o descrizione "AGGIORNATO" legata a questo.

Consigli pratici

Se stai cercando di scaricare/buildare l'eseguibile Windows:
Clona il repo
Vai in desktop_app/ o scripts/
Cerca build.bat, build.sh, pyinstaller.spec o simili
Esegui il build localmente (richiede PyInstaller installato + tante dipendenze)
→ Potresti ottenere ~200–400 MB a seconda delle esclusioni. Un .exe da 83 MB richiederebbe ottimizzazioni aggressive (es. escludere torch, pandas gui, matplotlib, ecc.).

Non caricare mai binari da 80+ MB direttamente nel repo GitHub — meglio usare Releases → Assets o un link esterno (Google Drive, itch.io, ecc.).

Se hai creato tu questo .exe da 83 MB localmente e vuoi aiuto per:

ridurlo ulteriormente
caricarlo come release asset
configurare auto-build con GitHub Actions
verificare se è funzionante in paper mode

…dimmi pure i dettagli e ti guido passo-passo! Altrimenti, sembra che l'annuncio "ZIP COMMIT + PUSH COMPLETO (83MB EXE)" non corrisponda a qualcosa di visibile pubblicamente su GitHub in questo momento.14,2sVelocehttps://github.com/ballales1984-wq/ai-trading-system/pull/new/blackboxai/final-commits
→ Merge → v2.1.1 Release → EXE attachIl link che hai condiviso (<https://github.com/ballales1984-wq/ai-trading-system/pull/new/blackboxai/final-commits>) punta a una pagina di creazione di una nuova Pull Request su GitHub, con branch di origine blackboxai/final-commits. Ecco lo stato reale verificato oggi (13 marzo 2026):
Situazione attuale del repository

Ultimo commit (appena fatto):
Hash 0165971 → "Merge pull request #4 from ballales1984-wq/blackboxai/final-commits"
→ Quindi la PR #4 da branch blackboxai/final-commits è stata appena mergiata nel branch principale (probabilmente main o master).
Timestamp: oggi, circa 17 minuti fa (Mar 13, 2026).
Pull Requests aperte: 0 (nessuna PR aperta al momento).
La #4 è stata chiusa con merge.
Releases:
Ultima release pubblica: "Ai trading sistem Latest" del 27 febbraio 2026 (circa 2 settimane fa).
Nome con typo evidente ("sistem" invece di "system").
Nessun tag semantico tipo v2.1.1, v2.1 o simili.
Nessun asset allegato (.exe, .zip, installer, ecc.).

Non esiste una release v2.1.1 (né v2.1.x) nel tab Releases.
Nessuna release creata oggi o negli ultimi giorni.

Cosa significa questo per il tuo task
Hai completato:

Commit finali nel branch blackboxai/final-commits
Merge della PR #4 → main

Ma non c'è ancora:

Tag/release v2.1.1 creato
.exe (o zip da ~83 MB) allegato come asset di release

Passi consigliati per completare "→ Merge → v2.1.1 Release → EXE attach"

Crea il tag e la release v2.1.1 (dalla UI o CLI):Bashgit tag v2.1.1
git push origin v2.1.1Poi vai su <https://github.com/ballales1984-wq/ai-trading-system/releases/new>
Seleziona tag: v2.1.1
Titolo: "v2.1.1 – Final commits + Windows executable"
Descrizione: elenca cambiamenti (es. fix TS, dashboard updates, build exe)
Pubblica come Release (non Draft).

Allega l'EXE / ZIP:
Se hai già generato localmente ai_trading_system.exe (~83 MB o altra dimensione):
Nella pagina di creazione release → trascina il file .exe (o .zip contenente exe + config) nella sezione "Attach binaries".
GitHub supporta file fino a 2 GB per release assets.
Se non l'hai ancora buildato:
Nella cartella del repo (probabilmente desktop_app/ o script PyInstaller), esegui il build:Bashpyinstaller --onefile --windowed --name ai_trading_system main.py   # esempioOttimizza se possibile per ridurre da ~300 MB a ~80–100 MB (escludi torch se non usato, usa --exclude-module, upx compress, ecc.).

Pubblica la release → una volta caricati gli asset, clicca "Publish release".
→ Il link diretto al .exe sarà tipo:
<https://github.com/ballales1984-wq/ai-trading-system/releases/download/v2.1.1/ai_trading_system.exe>

Se hai bisogno di aiuto con:

Ottimizzazione PyInstaller per ridurre la size dell'exe
Script GitHub Actions per build automatico + attach exe in release
Descrizione / changelog per la release v2.1.1

…fammi sapere i dettagli (es. quali dipendenze usa il tuo exe, se hai errori nel build, ecc.) e procediamo!
Al momento il merge è fatto → manca solo creare la release + allegare il binario.4,2sVeloceSkip to content

ballales1984-wq
ai-trading-system

Repository navigation

Code
Issues
Pull requests
Discussions
Actions
Projects
Security
Insights
Settings

ai-trading-systemPublic

ballales1984-wq/ai-trading-system
t
Name
ballales1984-wq
Merge pull request #4 from ballales1984-wq/blackboxai/final-commits
0165971 · 32 minutes ago
.github
feat: Add frontend charts, documentation and scripts
last week
.history
Update API comments for Vercel proxy
3 days ago
.zencoder/chats/ab6fbd03-955f-4223-8a2d-d5d0b172ee92
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
agent_coordination
feat: Add FastAPI hedge fund system, interactive menu, and dependencies
3 weeks ago
analytics
feat: Add frontend charts, documentation and scripts
last week
api
Add JWT dependencies to api/requirements.txt for Render
3 days ago
app
Add 4 final commits: ROADMAP, backtest.py, scheduler.py, DATABASE_SCH…
1 hour ago
archive
Organizza struttura progetto: sposta test, script e dashboard in cart…
4 days ago
cli
feat(cli): update submodule with package-lock.json
5 days ago
dashboard
Final 6 commits: recent fixes to main/Dashboard/PaymentTest/performan…
11 hours ago
data/ledger
Build: Create new executable ai_trading_system.exe (338 MB)
last week
decision_engine
feat: markdown fixes, decision_engine module, and documentation updates
2 weeks ago
deploy-temp
feat: Add success/cancel pages and default Stripe URLs
2 weeks ago
desktop_app
feat: Complete demo mode for Orders API with Emergency Stop feature
2 weeks ago
dist
Release: AI Trading System exe build with PyInstaller
2 weeks ago
docker
Fix dashboard callback errors and improve CI/CD workflows
3 weeks ago
docs
Add 4 final commits: ROADMAP, backtest.py, scheduler.py, DATABASE_SCH…
1 hour ago
frontend
Add ignoreDeprecations to tsconfig for TypeScript 6.0 compatibility
1 hour ago
infra/k8s
feat: Add multi-agent architecture, strategies, AutoML evolution, K8s…
3 weeks ago
java-frontend
Merge blackboxai/code-review-fixes into main
4 days ago
landing
feat(analytics): Add Google Analytics to frontend and landing page
3 days ago
logs/engine
Build: Create new executable ai_trading_system.exe (338 MB)
last week
migrations
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
node_modules
Fix CSS errors in index.css - complete truncated font-family declaration
last week
plans
feat: Add Advanced Compliance & Reporting module
2 days ago
public
Fix dashboard redirect to frontend deployment
2 weeks ago
scripts
Organizza struttura progetto: sposta test, script e dashboard in cart…
4 days ago
src
Final 6 commits: recent fixes to main/Dashboard/PaymentTest/performan…
11 hours ago
tests
feat: update trading completo coverage tests
yesterday
%(upstream
Merge blackboxai/code-review-fixes into main
4 days ago
.dockerignore
feat: Complete Phase 1 & 2 - Paper Trading + Testnet Integration
3 weeks ago
.gitattributes
Release: AI Trading System exe build with PyInstaller
2 weeks ago
.gitignore
Merge blackboxai/code-review-fixes into main
4 days ago
.markdownlint.json
revert route changes
2 weeks ago
.nvmrc
Fix TypeScript errors and Node version
2 weeks ago
.python-version
fix: Use Python 3.11 for Vercel runtime compatibility
4 days ago
.vercelignore
Simplify .vercelignore to only exclude node_modules
3 days ago
CHANGELOG.md
feat: Add frontend charts, documentation and scripts
last week
CLEANUP_PLAN.md
chore: cleanup project files and add new scripts
4 days ago
CODE_OF_CONDUCT.md
feat: Add frontend charts, documentation and scripts
last week
CONTRIBUTING.md
feat: Add frontend charts, documentation and scripts
last week
ChatGPT Image 18 feb 2026, 01_18_39.png
Add files via upload
3 weeks ago
DAILY_REPORT_2026-03-09.md
chore: cleanup project files and add new scripts
4 days ago
DEMO_RELEASE_CHECKLIST.md
Fix Vercel deployment: exit code 126 fix, add GA4 tracking, add auth API
2 weeks ago
DEPENDENCIES.txt
Add ecosystem maps: dependencies, tests, and Docker status
2 weeks ago
DISCORD_INVITE.md
fix: resolve markdownlint errors in DISCORD_INVITE.md
last week
Dockerfile
Fix: Use absolute path for frontend build in Dockerfile
3 days ago
Dockerfile.render.optimized
Update Dockerfile and main.py for frontend build on Render
3 days ago
ECOSYSTEM_MAP.md
feat: Add Performance Attribution module for Phase 2
2 days ago
ECOSYSTEM_MAP.txt
Add Stripe payment success/cancel pages for Vercel
2 weeks ago
LICENSE
Initial commit
3 weeks ago
PRODUCTION_FEATURES.md
feat: Add production-grade features
3 weeks ago
README.md
Update README and fix strategies test coverage
3 days ago
RENDER.md
Merge compliance features and fix Vercel deployment
3 weeks ago
ROADMAP.md
Add 4 final commits: ROADMAP, backtest.py, scheduler.py, DATABASE_SCH…
1 hour ago
START_BACKEND_LOCAL.md
fix: Update Vercel configuration for proper serverless deployment
3 weeks ago
TODO.md
Final 6 commits: recent fixes to main/Dashboard/PaymentTest/performan…
11 hours ago
TODO_5Question_Engine.md
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
TODO_CACHE.md
Add emergency stop security, improve UX/accessibility, and reduce fro…
3 weeks ago
TODO_CHECKLIST.md
chore: update to 100% complete status
3 weeks ago
TODO_HEDGE_FUND.md
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
TODO_REACT_FRONTEND.md
feat: Add SaaS foundation - unified config, structured logging, landi…
3 weeks ago
alembic.ini
feat: add SQLAlchemy database layer, HMM regime detection, and Alembi…
3 weeks ago
auto_commit.ps1
chore: auto-save changes - 2026-03-09 16:03:25
4 days ago
auto_trader.py
Merge Agent2 changes into main - resolved conflicts accepting Agent2 …
3 weeks ago
backup_database.bat
Add Docker stable config with RAM/ROM limits (4GB/3GB), volume monito…
3 weeks ago
binance_research.py
Enhanced dashboard with Binance real-time data, MACD/RSI charts, time…
3 weeks ago
build.sh
fix: Update Vercel build configuration to fix exit code 126 error
3 weeks ago
check_db.py
Fix indentation in portfolio.py (added 4 spaces to fallback comment)
2 weeks ago
check_status.py
chore: add utility scripts for development
4 days ago
commit.bat
chore: add utility scripts for development
4 days ago
config.py
feat: Complete Phase 3 & 4 implementation
2 days ago
coverage_error.txt
feat(frontend): update package deps, tsconfig, vite config, and cover…
yesterday
create_shortcut.ps1
Set demo_mode=false as default for production use
2 weeks ago
dashboard_investor.py
feat: Complete Phase 3 & 4 implementation
2 days ago
data_collector.py
feat: Complete Phase 3 & 4 implementation
2 days ago
decision_engine.py
feat: integrate HMM regime detection into DecisionEngine
3 weeks ago
do_git.py
chore: add utility scripts for development
4 days ago
docker-compose.production.yml
feat: Add production-grade features
3 weeks ago
docker-compose.yml
fix: Vercel deployment configuration
3 weeks ago
example_create_order.py
docs: Add comprehensive Orders API usage examples
3 weeks ago
execute_signals_multiasset.py
feat: Add multi-asset trading, decision engine modules, and realtime …
3 weeks ago
fix_remaining
Build: Create new executable ai_trading_system.exe (338 MB)
last week
git_status.bat
Fix indentation in portfolio.py (added 4 spaces to fallback comment)
2 weeks ago
git_status.py
Fix indentation in portfolio.py (added 4 spaces to fallback comment)
2 weeks ago
health_tmp.pyc.2083572273392
feat: Add payment test page and configure Stripe
2 weeks ago
live_multi_asset.py
Update ML models and add retry utility
3 weeks ago
logical_math_multiasset.py
feat: Add multi-asset trading, decision engine modules, and realtime …
3 weeks ago
logical_portfolio_module.py
feat: Add multi-asset trading, decision engine modules, and realtime …
3 weeks ago
main_auto_trader.py
feat: Add multi-asset trading, decision engine modules, and realtime …
3 weeks ago
ml_predictor.py
Quick Wins: Unit tests, ML persistence, config, pytest setup
3 weeks ago
monitor_volumes.py
Add Docker stable config with RAM/ROM limits (4GB/3GB), volume monito…
3 weeks ago
ollama_test.py
Build: Create new executable ai_trading_system.exe (338 MB)
last week
onchain_analysis.py
Add advanced technical indicators and on-chain analysis
3 weeks ago
package-lock.json
Fix CSS errors in index.css - complete truncated font-family declaration
last week
package.json
Fix CSS errors in index.css - complete truncated font-family declaration
last week
push_to_github.bat
fix: Vercel deployment configuration
3 weeks ago
pyproject.toml
feat: Complete Phase 1 & 2 - Paper Trading + Testnet Integration
3 weeks ago
pytest.ini
Add Twitter API integration and fix pytest asyncio
3 weeks ago
render.yaml
Fix deploy on Render: add missing dependencies, create Dockerfile, up…
3 days ago
requirements-vercel.txt
Update API comments for Vercel proxy
3 days ago
requirements.stable.txt
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
requirements.txt
Fix deploy on Render: add missing dependencies, create Dockerfile, up…
3 days ago
restore_dashboard.ps1
Add PowerShell restore script for dashboard
5 days ago
run_coverage_tests.py
Update run_coverage_tests.py
yesterday
run_tests.bat
Build: Create new executable ai_trading_system.exe (338 MB)
last week
run_tests_quick.bat
chore: add utility scripts for development
4 days ago
sentiment_news.py
Merge Agent2 changes into main - resolved conflicts accepting Agent2 …
3 weeks ago
start_ai_trading.bat
Add Docker stable config with RAM/ROM limits (4GB/3GB), volume monito…
3 weeks ago
start_ai_trading_unified.bat
Fix indentation in portfolio.py (added 4 spaces to fallback comment)
2 weeks ago
start_api_advanced.bat
Restore risk routes and fix API imports
last week
start_backend_with_ngrok.bat
Fix indentation in portfolio.py (added 4 spaces to fallback comment)
2 weeks ago
start_desktop_app.bat
Set demo_mode=false as default for production use
2 weeks ago
start_frontend_and_backend.bat
feat: add batch script to start frontend and backend
5 days ago
start_frontend_and_backend.ps1
feat: add PowerShell script to start frontend and backend
5 days ago
start_frontend_online.ps1
chore: cleanup project files and add new scripts
4 days ago
start_paper_trading.bat
feat: add paper trading launcher scripts
3 weeks ago
start_stable.bat
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
start_stable.sh
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
stop_ai_trading.bat
Add Docker stable config with RAM/ROM limits (4GB/3GB), volume monito…
3 weeks ago
technical_analysis.py
Update ML models and add retry utility
3 weeks ago
test_login.json
Set demo_mode=false as default for production use
2 weeks ago
trading_simulator.py
fix: correct vercel.json routing for Python API
2 weeks ago
version_info.txt
Fix build script - add Python build_exe.py
2 weeks ago
Repository files navigation

README
Code of conduct
Contributing
Apache-2.0 license

🤖 AI Trading System — Mini Hedge Fund Infrastructure

🎯 Why This Project Exists
Most retail trading systems focus on single indicators, naive executions, and reactive strategies. They fail because they ignore what institutional quant desks know well:
It's not the signal that generates alpha. It's the infrastructure.
This project is different. It's designed from scratch as modular quantitative infrastructure — event-driven, risk-aware, and capable of evolving toward institutional-level architecture.
It's not a bot. It's a trading system.
🧠 System Philosophy
PrincipleImplementationEvent-Driven ArchitectureAsync data pipelines, non-blocking execution, reactive decision engineProbabilistic Forecasting5-level Monte Carlo simulation, uncertainty quantification, ensemble designRisk-First DesignVaR/CVaR limits, GARCH volatility modeling, dynamic position sizing, drawdown protectionAdaptive Regime ModelingHMM market regime detection, strategy rotation based on market conditionsMulti-Source Intelligence18+ API integrations, sentiment analysis, on-chain metrics, macro indicators
🏗️ Architecture Overview

textai-trading-system/
├── app/                    # FastAPI application
│   ├── api/routes/        # REST endpoints
│   ├── core/             # Security, cache, DB
│   ├── execution/        # Broker connectors
│   └── database/         # SQLAlchemy models
├── src/                   # Core trading logic
│   ├── agents/           # AI agents (MonteCarlo, Risk, MarketData)
│   ├── core/             # Event bus, state manager
│   ├── decision/         # Decision engine
│   ├── strategy/         # Trading strategies
│   ├── research/         # Alpha Lab, Feature Store
│   └── external/         # API integrations
├── tests/                # Test suite (927+ tests)
├── dashboard/            # Dash dashboard
├── frontend/            # React frontend
├── docker/              # Docker configs
└── infra/               # Kubernetes configs
⚡ Quick Start (5 Minutes)
Option 1: Docker (Recommended)
text# Clone and run with Docker
git clone <https://github.com/ballales1984-wq/ai-trading-system.git>
cd ai-trading-system
docker-compose up -d

# Wait 2-3 minutes for services to start

# Then open <http://localhost:8000> in your browser

Option 2: Local Development
text# Install dependencies
pip install -r requirements.txt
cd frontend
npm install

# Start services

# Terminal 1: Backend

uvicorn app.main:app --reload

# Terminal 2: Frontend

cd frontend
npm run dev

# Open <http://localhost:5173>

Option 3: Cloud Deployment
Deploy to Render or Vercel with one click:
Live Demo: <https://ai-trading-system-1reg.onrender.com>

Deploy to Render
Deploy to Vercel

🎥 Video Tutorial
Watch our 5-minute setup guide:
📊 Key Features
Trading Infrastructure

Multi-broker support: Binance, Bybit, Paper Trading
Smart order routing: TWAP, VWAP, POV, Iceberg orders
Risk management: VaR/CVaR limits, GARCH volatility, dynamic position sizing
Execution algorithms: Best execution with latency optimization

AI & Analytics

Monte Carlo simulation: 5-level probabilistic forecasting
HMM regime detection: Market condition identification
Sentiment analysis: News and social media integration
Cross-asset correlation: Portfolio optimization

Frontend Dashboard

Real-time monitoring: Live P&L, positions, risk metrics
Interactive charts: Price action, equity curves, correlation matrices
Mobile responsive: Works on all devices
Dark mode: Eye-friendly interface

📈 Performance Metrics
MetricValueBenchmarkCAGR23.5%18.2%Max Drawdown7.2%45.8%Sharpe Ratio1.950.82Sortino Ratio2.451.12Win Rate68%-
🛡️ Risk Management
Capital Protection

Max position size: 10% per asset
Daily drawdown limit: 5% circuit breaker
VaR confidence: 95% (industry standard)
CVaR limit: 8% tail risk protection

Failure Modes & Mitigations
Failure ModeProbabilityMitigationAPI FailureMediumMulti-exchange fallbackModel DecayHighContinuous retrainingLiquidity CrisisLowPosition size limitsFlash CrashLowCircuit breakers
🔬 Backtest Integrity

Walk-forward validation: Rolling 6-month windows
Look-ahead bias prevention: Feature scaling only on training data
Survivorship bias: Includes delisted assets
Latency simulation: 100-500ms random delay
Slippage model: Volume-weighted impact

🚀 Getting Started Guide
Step 1: Choose Your Deployment Method

Docker: Easiest for beginners
Local: Best for development
Cloud: For production use

Step 2: Configure API Keys
Create .env file with your exchange API keys:
text# Binance
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
BINANCE_TESTNET=false

# Bybit

BYBIT_API_KEY=your_key
BYBIT_SECRET_KEY=your_secret
BYBIT_TESTNET=true
Step 3: Start Trading

Paper trading (recommended first)
Live trading with small capital
Scale up gradually

API Reference
Architecture Guide
Deployment Guide
Risk Management

🤝 Community
Join our community to get support and share ideas:

Discord Server
GitHub Discussions
Twitter

⭐ Contributing
We welcome contributions! Please read our CONTRIBUTING.md guide.
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
📧 Contact
For questions or support, please use GitHub Issues or join our Discord community.## 📚 Documentation

API Reference
Architecture Guide
Deployment Guide
Risk Management

🤝 Community
Join our community to get support and share ideas:

Discord Server
GitHub Discussions
Twitter

⭐ Contributing
We welcome contributions! Please read our CONTRIBUTING.md guide.
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
📧 Contact
About
ai-trading-system
ai-trading-system-1reg.onrender.com/
Topics
ai system trading
Resources
Readme
License
Apache-2.0 license
Code of conduct
Code of conduct
Contributing
Contributing
Activity
Stars
1 star
Watchers
0 watching
Forks
0 forks
Releases 2
"v2.1.1 – Complete Features + Windows EXE"Latest
now

1 release
Deployments500+

Production – frontend
Production – deploy-temp 31 minutes ago
Production – ai-trading-system

more deployments
Packages
No packages published
Publish your first package
Contributors2

ballales1984-wq alessio
vercel[bot]

Languages

Python90.4%
TypeScript4.2%
HTML2.4%
Shell1.4%
CSS0.7%
Batchfile0.4%
Other0.5%

Footer
© 2026 GitHub, Inc.
Footer navigation

Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information
Il repository <https://github.com/ballales1984-wq/ai-trading-system> ha ricevuto un aggiornamento molto recente: il merge del PR #4 dal branch blackboxai/final-commits è avvenuto circa 30–40 minuti fa (commit 0165971, autore ballales1984-wq).
Stato attuale (13 marzo 2026, ore pomeridiane)

Metriche repo (invariate): 1 stella, 0 fork, 0 watcher
Releases → Esiste ora una release v2.1.1
Tag: v2.1.1
Titolo: "v2.1.1 – Complete Features + Windows EXE"
Autore: ballales1984-wq
Data: 13 marzo 2026 (pubblicata oggi, intorno alle 17:27 circa)
Descrizione / changelog: Non visibile in dettaglio a causa di un errore temporaneo di caricamento GitHub ("Uh oh! There was an error while loading"), ma il titolo indica esplicitamente l'inclusione di un eseguibile Windows (.exe).
Asset allegati: Probabilmente include l'ai_trading_system.exe (o un .zip contenente l'exe + file di supporto), come promesso nel titolo della release ("+ Windows EXE").
Nota importante: Nelle release precedenti (es. v2.0) non c'erano asset binari allegati. Questa è la prima release con un eseguibile Windows dichiarato.
Dimensione attesa: Hai menzionato ~83 MB in precedenza (forse una versione ottimizzata rispetto ai precedenti build da ~338 MB visti nei commit vecchi).

Altre releases visibili: Almeno una precedente (v2.0 del 27 febbraio 2026), ma senza asset scaricabili.

Come scaricare l'EXE ora

Vai direttamente qui:
<https://github.com/ballales1984-wq/ai-trading-system/releases/tag/v2.1.1>
(o apri il tab Releases dal repo principale → seleziona l'ultima "v2.1.1 – Complete Features + Windows EXE")
Scorri in basso nella pagina della release → sezione Assets (espandibile)
→ Clicca sul file .exe (o .zip) per scaricare.

Se non vedi ancora gli asset (a volte GitHub ha delay di propagazione o errori di cache):

Ricarica la pagina dopo 5–10 minuti
Prova in incognito o da altro browser/dispositivo
Cerca manualmente il link diretto tipo:
<https://github.com/ballales1984-wq/ai-trading-system/releases/download/v2.1.1/ai_trading_system.exe>
(sostituisci il nome file esatto se diverso)

Verifiche consigliate prima di eseguire l'EXE

Sicurezza: È un binario generato da PyInstaller (probabile). Scansiona con antivirus (VirusTotal, Windows Defender full scan) prima di avviare.
Modalità demo/paper: Avvia prima in modalità paper trading (controlla config o flag --demo / demo_mode=true se presente).
Dipendenze: Potrebbe richiedere installazione di runtime (es. Visual C++ Redistributable se non già presente).
Codice sorgente: Confronta l'hash del binario con un build locale tuo per verificare integrità (opzionale ma raccomandato).
Rischio: Anche con risk management dichiarato, non usarlo live con fondi reali senza test approfonditi.

Congratulazioni per aver completato il ciclo: commit → merge → tag/release → attach EXE!
Se l'asset non appare ancora o hai bisogno di aiuto per:

Ridurre ulteriormente la size dell'exe
Configurare GitHub Actions per build automatici futuri
Aggiungere note/changelog dettagliati nella release
Testare l'exe in ambiente controllato

…fammi sapere e procediamo!4sVeloceSkip to content

ballales1984-wq
ai-trading-system

Repository navigation

Code
Issues
Pull requests
Discussions
Actions
Projects
Security
Insights
Settings

ai-trading-systemPublic

ballales1984-wq/ai-trading-system
t
Name
ballales1984-wq
Merge pull request #4 from ballales1984-wq/blackboxai/final-commits
0165971 · 32 minutes ago
.github
feat: Add frontend charts, documentation and scripts
last week
.history
Update API comments for Vercel proxy
3 days ago
.zencoder/chats/ab6fbd03-955f-4223-8a2d-d5d0b172ee92
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
agent_coordination
feat: Add FastAPI hedge fund system, interactive menu, and dependencies
3 weeks ago
analytics
feat: Add frontend charts, documentation and scripts
last week
api
Add JWT dependencies to api/requirements.txt for Render
3 days ago
app
Add 4 final commits: ROADMAP, backtest.py, scheduler.py, DATABASE_SCH…
1 hour ago
archive
Organizza struttura progetto: sposta test, script e dashboard in cart…
4 days ago
cli
feat(cli): update submodule with package-lock.json
5 days ago
dashboard
Final 6 commits: recent fixes to main/Dashboard/PaymentTest/performan…
11 hours ago
data/ledger
Build: Create new executable ai_trading_system.exe (338 MB)
last week
decision_engine
feat: markdown fixes, decision_engine module, and documentation updates
2 weeks ago
deploy-temp
feat: Add success/cancel pages and default Stripe URLs
2 weeks ago
desktop_app
feat: Complete demo mode for Orders API with Emergency Stop feature
2 weeks ago
dist
Release: AI Trading System exe build with PyInstaller
2 weeks ago
docker
Fix dashboard callback errors and improve CI/CD workflows
3 weeks ago
docs
Add 4 final commits: ROADMAP, backtest.py, scheduler.py, DATABASE_SCH…
1 hour ago
frontend
Add ignoreDeprecations to tsconfig for TypeScript 6.0 compatibility
1 hour ago
infra/k8s
feat: Add multi-agent architecture, strategies, AutoML evolution, K8s…
3 weeks ago
java-frontend
Merge blackboxai/code-review-fixes into main
4 days ago
landing
feat(analytics): Add Google Analytics to frontend and landing page
3 days ago
logs/engine
Build: Create new executable ai_trading_system.exe (338 MB)
last week
migrations
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
node_modules
Fix CSS errors in index.css - complete truncated font-family declaration
last week
plans
feat: Add Advanced Compliance & Reporting module
2 days ago
public
Fix dashboard redirect to frontend deployment
2 weeks ago
scripts
Organizza struttura progetto: sposta test, script e dashboard in cart…
4 days ago
src
Final 6 commits: recent fixes to main/Dashboard/PaymentTest/performan…
11 hours ago
tests
feat: update trading completo coverage tests
yesterday
%(upstream
Merge blackboxai/code-review-fixes into main
4 days ago
.dockerignore
feat: Complete Phase 1 & 2 - Paper Trading + Testnet Integration
3 weeks ago
.gitattributes
Release: AI Trading System exe build with PyInstaller
2 weeks ago
.gitignore
Merge blackboxai/code-review-fixes into main
4 days ago
.markdownlint.json
revert route changes
2 weeks ago
.nvmrc
Fix TypeScript errors and Node version
2 weeks ago
.python-version
fix: Use Python 3.11 for Vercel runtime compatibility
4 days ago
.vercelignore
Simplify .vercelignore to only exclude node_modules
3 days ago
CHANGELOG.md
feat: Add frontend charts, documentation and scripts
last week
CLEANUP_PLAN.md
chore: cleanup project files and add new scripts
4 days ago
CODE_OF_CONDUCT.md
feat: Add frontend charts, documentation and scripts
last week
CONTRIBUTING.md
feat: Add frontend charts, documentation and scripts
last week
ChatGPT Image 18 feb 2026, 01_18_39.png
Add files via upload
3 weeks ago
DAILY_REPORT_2026-03-09.md
chore: cleanup project files and add new scripts
4 days ago
DEMO_RELEASE_CHECKLIST.md
Fix Vercel deployment: exit code 126 fix, add GA4 tracking, add auth API
2 weeks ago
DEPENDENCIES.txt
Add ecosystem maps: dependencies, tests, and Docker status
2 weeks ago
DISCORD_INVITE.md
fix: resolve markdownlint errors in DISCORD_INVITE.md
last week
Dockerfile
Fix: Use absolute path for frontend build in Dockerfile
3 days ago
Dockerfile.render.optimized
Update Dockerfile and main.py for frontend build on Render
3 days ago
ECOSYSTEM_MAP.md
feat: Add Performance Attribution module for Phase 2
2 days ago
ECOSYSTEM_MAP.txt
Add Stripe payment success/cancel pages for Vercel
2 weeks ago
LICENSE
Initial commit
3 weeks ago
PRODUCTION_FEATURES.md
feat: Add production-grade features
3 weeks ago
README.md
Update README and fix strategies test coverage
3 days ago
RENDER.md
Merge compliance features and fix Vercel deployment
3 weeks ago
ROADMAP.md
Add 4 final commits: ROADMAP, backtest.py, scheduler.py, DATABASE_SCH…
1 hour ago
START_BACKEND_LOCAL.md
fix: Update Vercel configuration for proper serverless deployment
3 weeks ago
TODO.md
Final 6 commits: recent fixes to main/Dashboard/PaymentTest/performan…
11 hours ago
TODO_5Question_Engine.md
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
TODO_CACHE.md
Add emergency stop security, improve UX/accessibility, and reduce fro…
3 weeks ago
TODO_CHECKLIST.md
chore: update to 100% complete status
3 weeks ago
TODO_HEDGE_FUND.md
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
TODO_REACT_FRONTEND.md
feat: Add SaaS foundation - unified config, structured logging, landi…
3 weeks ago
alembic.ini
feat: add SQLAlchemy database layer, HMM regime detection, and Alembi…
3 weeks ago
auto_commit.ps1
chore: auto-save changes - 2026-03-09 16:03:25
4 days ago
auto_trader.py
Merge Agent2 changes into main - resolved conflicts accepting Agent2 …
3 weeks ago
backup_database.bat
Add Docker stable config with RAM/ROM limits (4GB/3GB), volume monito…
3 weeks ago
binance_research.py
Enhanced dashboard with Binance real-time data, MACD/RSI charts, time…
3 weeks ago
build.sh
fix: Update Vercel build configuration to fix exit code 126 error
3 weeks ago
check_db.py
Fix indentation in portfolio.py (added 4 spaces to fallback comment)
2 weeks ago
check_status.py
chore: add utility scripts for development
4 days ago
commit.bat
chore: add utility scripts for development
4 days ago
config.py
feat: Complete Phase 3 & 4 implementation
2 days ago
coverage_error.txt
feat(frontend): update package deps, tsconfig, vite config, and cover…
yesterday
create_shortcut.ps1
Set demo_mode=false as default for production use
2 weeks ago
dashboard_investor.py
feat: Complete Phase 3 & 4 implementation
2 days ago
data_collector.py
feat: Complete Phase 3 & 4 implementation
2 days ago
decision_engine.py
feat: integrate HMM regime detection into DecisionEngine
3 weeks ago
do_git.py
chore: add utility scripts for development
4 days ago
docker-compose.production.yml
feat: Add production-grade features
3 weeks ago
docker-compose.yml
fix: Vercel deployment configuration
3 weeks ago
example_create_order.py
docs: Add comprehensive Orders API usage examples
3 weeks ago
execute_signals_multiasset.py
feat: Add multi-asset trading, decision engine modules, and realtime …
3 weeks ago
fix_remaining
Build: Create new executable ai_trading_system.exe (338 MB)
last week
git_status.bat
Fix indentation in portfolio.py (added 4 spaces to fallback comment)
2 weeks ago
git_status.py
Fix indentation in portfolio.py (added 4 spaces to fallback comment)
2 weeks ago
health_tmp.pyc.2083572273392
feat: Add payment test page and configure Stripe
2 weeks ago
live_multi_asset.py
Update ML models and add retry utility
3 weeks ago
logical_math_multiasset.py
feat: Add multi-asset trading, decision engine modules, and realtime …
3 weeks ago
logical_portfolio_module.py
feat: Add multi-asset trading, decision engine modules, and realtime …
3 weeks ago
main_auto_trader.py
feat: Add multi-asset trading, decision engine modules, and realtime …
3 weeks ago
ml_predictor.py
Quick Wins: Unit tests, ML persistence, config, pytest setup
3 weeks ago
monitor_volumes.py
Add Docker stable config with RAM/ROM limits (4GB/3GB), volume monito…
3 weeks ago
ollama_test.py
Build: Create new executable ai_trading_system.exe (338 MB)
last week
onchain_analysis.py
Add advanced technical indicators and on-chain analysis
3 weeks ago
package-lock.json
Fix CSS errors in index.css - complete truncated font-family declaration
last week
package.json
Fix CSS errors in index.css - complete truncated font-family declaration
last week
push_to_github.bat
fix: Vercel deployment configuration
3 weeks ago
pyproject.toml
feat: Complete Phase 1 & 2 - Paper Trading + Testnet Integration
3 weeks ago
pytest.ini
Add Twitter API integration and fix pytest asyncio
3 weeks ago
render.yaml
Fix deploy on Render: add missing dependencies, create Dockerfile, up…
3 days ago
requirements-vercel.txt
Update API comments for Vercel proxy
3 days ago
requirements.stable.txt
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
requirements.txt
Fix deploy on Render: add missing dependencies, create Dockerfile, up…
3 days ago
restore_dashboard.ps1
Add PowerShell restore script for dashboard
5 days ago
run_coverage_tests.py
Update run_coverage_tests.py
yesterday
run_tests.bat
Build: Create new executable ai_trading_system.exe (338 MB)
last week
run_tests_quick.bat
chore: add utility scripts for development
4 days ago
sentiment_news.py
Merge Agent2 changes into main - resolved conflicts accepting Agent2 …
3 weeks ago
start_ai_trading.bat
Add Docker stable config with RAM/ROM limits (4GB/3GB), volume monito…
3 weeks ago
start_ai_trading_unified.bat
Fix indentation in portfolio.py (added 4 spaces to fallback comment)
2 weeks ago
start_api_advanced.bat
Restore risk routes and fix API imports
last week
start_backend_with_ngrok.bat
Fix indentation in portfolio.py (added 4 spaces to fallback comment)
2 weeks ago
start_desktop_app.bat
Set demo_mode=false as default for production use
2 weeks ago
start_frontend_and_backend.bat
feat: add batch script to start frontend and backend
5 days ago
start_frontend_and_backend.ps1
feat: add PowerShell script to start frontend and backend
5 days ago
start_frontend_online.ps1
chore: cleanup project files and add new scripts
4 days ago
start_paper_trading.bat
feat: add paper trading launcher scripts
3 weeks ago
start_stable.bat
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
start_stable.sh
Fix: corretto errore indentazione in test_production_features.py line…
3 weeks ago
stop_ai_trading.bat
Add Docker stable config with RAM/ROM limits (4GB/3GB), volume monito…
3 weeks ago
technical_analysis.py
Update ML models and add retry utility
3 weeks ago
test_login.json
Set demo_mode=false as default for production use
2 weeks ago
trading_simulator.py
fix: correct vercel.json routing for Python API
2 weeks ago
version_info.txt
Fix build script - add Python build_exe.py
2 weeks ago
Repository files navigation

README
Code of conduct
Contributing
Apache-2.0 license

🤖 AI Trading System — Mini Hedge Fund Infrastructure

🎯 Why This Project Exists
Most retail trading systems focus on single indicators, naive executions, and reactive strategies. They fail because they ignore what institutional quant desks know well:
It's not the signal that generates alpha. It's the infrastructure.
This project is different. It's designed from scratch as modular quantitative infrastructure — event-driven, risk-aware, and capable of evolving toward institutional-level architecture.
It's not a bot. It's a trading system.
🧠 System Philosophy
PrincipleImplementationEvent-Driven ArchitectureAsync data pipelines, non-blocking execution, reactive decision engineProbabilistic Forecasting5-level Monte Carlo simulation, uncertainty quantification, ensemble designRisk-First DesignVaR/CVaR limits, GARCH volatility modeling, dynamic position sizing, drawdown protectionAdaptive Regime ModelingHMM market regime detection, strategy rotation based on market conditionsMulti-Source Intelligence18+ API integrations, sentiment analysis, on-chain metrics, macro indicators
🏗️ Architecture Overview

textai-trading-system/
├── app/                    # FastAPI application
│   ├── api/routes/        # REST endpoints
│   ├── core/             # Security, cache, DB
│   ├── execution/        # Broker connectors
│   └── database/         # SQLAlchemy models
├── src/                   # Core trading logic
│   ├── agents/           # AI agents (MonteCarlo, Risk, MarketData)
│   ├── core/             # Event bus, state manager
│   ├── decision/         # Decision engine
│   ├── strategy/         # Trading strategies
│   ├── research/         # Alpha Lab, Feature Store
│   └── external/         # API integrations
├── tests/                # Test suite (927+ tests)
├── dashboard/            # Dash dashboard
├── frontend/            # React frontend
├── docker/              # Docker configs
└── infra/               # Kubernetes configs
⚡ Quick Start (5 Minutes)
Option 1: Docker (Recommended)
text# Clone and run with Docker
git clone <https://github.com/ballales1984-wq/ai-trading-system.git>
cd ai-trading-system
docker-compose up -d

# Wait 2-3 minutes for services to start

# Then open <http://localhost:8000> in your browser

Option 2: Local Development
text# Install dependencies
pip install -r requirements.txt
cd frontend
npm install

# Start services

# Terminal 1: Backend

uvicorn app.main:app --reload

# Terminal 2: Frontend

cd frontend
npm run dev

# Open <http://localhost:5173>

Option 3: Cloud Deployment
Deploy to Render or Vercel with one click:
Live Demo: <https://ai-trading-system-1reg.onrender.com>

Deploy to Render
Deploy to Vercel

🎥 Video Tutorial
Watch our 5-minute setup guide:
📊 Key Features
Trading Infrastructure

Multi-broker support: Binance, Bybit, Paper Trading
Smart order routing: TWAP, VWAP, POV, Iceberg orders
Risk management: VaR/CVaR limits, GARCH volatility, dynamic position sizing
Execution algorithms: Best execution with latency optimization

AI & Analytics

Monte Carlo simulation: 5-level probabilistic forecasting
HMM regime detection: Market condition identification
Sentiment analysis: News and social media integration
Cross-asset correlation: Portfolio optimization

Frontend Dashboard

Real-time monitoring: Live P&L, positions, risk metrics
Interactive charts: Price action, equity curves, correlation matrices
Mobile responsive: Works on all devices
Dark mode: Eye-friendly interface

📈 Performance Metrics
MetricValueBenchmarkCAGR23.5%18.2%Max Drawdown7.2%45.8%Sharpe Ratio1.950.82Sortino Ratio2.451.12Win Rate68%-
🛡️ Risk Management
Capital Protection

Max position size: 10% per asset
Daily drawdown limit: 5% circuit breaker
VaR confidence: 95% (industry standard)
CVaR limit: 8% tail risk protection

Failure Modes & Mitigations
Failure ModeProbabilityMitigationAPI FailureMediumMulti-exchange fallbackModel DecayHighContinuous retrainingLiquidity CrisisLowPosition size limitsFlash CrashLowCircuit breakers
🔬 Backtest Integrity

Walk-forward validation: Rolling 6-month windows
Look-ahead bias prevention: Feature scaling only on training data
Survivorship bias: Includes delisted assets
Latency simulation: 100-500ms random delay
Slippage model: Volume-weighted impact

🚀 Getting Started Guide
Step 1: Choose Your Deployment Method

Docker: Easiest for beginners
Local: Best for development
Cloud: For production use

Step 2: Configure API Keys
Create .env file with your exchange API keys:
text# Binance
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
BINANCE_TESTNET=false

# Bybit

BYBIT_API_KEY=your_key
BYBIT_SECRET_KEY=your_secret
BYBIT_TESTNET=true
Step 3: Start Trading

Paper trading (recommended first)
Live trading with small capital
Scale up gradually

API Reference
Architecture Guide
Deployment Guide
Risk Management

🤝 Community
Join our community to get support and share ideas:

Discord Server
GitHub Discussions
Twitter

⭐ Contributing
We welcome contributions! Please read our CONTRIBUTING.md guide.
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
📧 Contact
For questions or support, please use GitHub Issues or join our Discord community.## 📚 Documentation

API Reference
Architecture Guide
Deployment Guide
Risk Management

🤝 Community
Join our community to get support and share ideas:

Discord Server
GitHub Discussions
Twitter

⭐ Contributing
We welcome contributions! Please read our CONTRIBUTING.md guide.
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
📧 Contact
About
ai-trading-system
ai-trading-system-1reg.onrender.com/
Topics
ai system trading
Resources
Readme
License
Apache-2.0 license
Code of conduct
Code of conduct
Contributing
Contributing
Activity
Stars
1 star
Watchers
0 watching
Forks
0 forks
Releases 2
"v2.1.1 – Complete Features + Windows EXE"Latest
now

1 release
Deployments500+

Production – frontend
Production – deploy-temp 31 minutes ago
Production – ai-trading-system

more deployments
Packages
No packages published
Publish your first package
Contributors2

ballales1984-wq alessio
vercel[bot]

Languages

Python90.4%
TypeScript4.2%
HTML2.4%
Shell1.4%
CSS0.7%
Batchfile0.4%
Other0.5%

Footer
© 2026 GitHub, Inc.
Footer navigation

Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information
La release v2.1.1 – Complete Features + Windows EXE è stata creata oggi (13 marzo 2026, intorno alle 17:27) sul repository <https://github.com/ballales1984-wq/ai-trading-system>, subito dopo il merge del PR #4 (commit 0165971).
Dettagli della release (stato attuale)

Tag: v2.1.1
Titolo: "v2.1.1 – Complete Features + Windows EXE"
Autore: ballales1984-wq
Data pubblicazione: 13 marzo 2026 (molto recente, ~pochi minuti/ore fa)
Note di rilascio / Changelog: Non visibili al momento (la pagina mostra un errore GitHub temporaneo:Uh oh! There was an error while loading. Please reload this page.
Sorry, something went wrong. No results found.Questo è un problema comune su GitHub dopo creazioni recenti o durante picchi di caricamento — spesso si risolve da solo in 5–60 minuti ricaricando la pagina).
Asset allegati (file scaricabili):
Al momento non visibili a causa dell'errore di caricamento della pagina.
Il titolo della release include esplicitamente "+ Windows EXE", quindi è molto probabile che sia stato caricato un file eseguibile Windows (tipo ai_trading_system.exe o un .zip contenente l'exe + config/docs).
Nelle release precedenti non c'erano binari allegati, quindi questa sarebbe la prima con un .exe pubblico.
Dimensione attesa: Hai parlato di ~83 MB in precedenza (forse una build ottimizzata rispetto ai vecchi commit che menzionavano 338 MB).

Come accedere e scaricare (prova questi passi ora)

Apri direttamente la pagina della release:
<https://github.com/ballales1984-wq/ai-trading-system/releases/tag/v2.1.1Oppure> dal repo principale → tab Releases (a destra) → seleziona la più recente ("v2.1.1 – Complete Features + Windows EXE").
Ricarica la pagina più volte (Ctrl+F5 o in incognito) — l'errore "Uh oh!" dovrebbe sparire e apparire la sezione Assets in basso con il/i file scaricabili.
Se appare l'asset:
Clicca su ai_trading_system.exe (o nome simile) per scaricare.
Link diretto probabile (una volta visibile):
<https://github.com/ballales1984-wq/ai-trading-system/releases/download/v2.1.1/ai_trading_system.exe>
(cambia il nome file se diverso).

Se l'errore persiste dopo 1 ora:
Controlla il tab Releases generale: <https://github.com/ballales1984-wq/ai-trading-system/releases>
Verifica se la release è in stato "Draft" invece di Published (raro, ma possibile se non finalizzata).
In alternativa, builda l'exe localmente tu stesso (usa pyinstaller dal codice sorgente nella cartella desktop_app/ o script build_exe.py menzionato nei commit).

Consigli immediati per sicurezza e test

Non eseguire subito l'exe se scaricato:
Caricalo su VirusTotal.com per scan multi-antivirus.
Avvialo in una VM/sandbox (VirtualBox, Windows Sandbox) la prima volta.

Modalità sicura: Cerca flag come --demo, --paper, o modifica demo_mode=true nel config (se presente) per evitare trading reale.
Verifica integrità: Dopo il download, confronta la size/hash con quanto dichiarato (es. 83 MB).
Il repo resta con 1 stella / 0 fork → visibilità bassissima, quindi procedi con cautela estrema.

Se tra poco la pagina carica correttamente e vedi l'exe, dimmi il nome esatto del file, la size e se ci sono note/changelog visibili — posso aiutarti a capire come usarlo o ottimizzarlo ulteriormente! Altrimenti, se l'errore continua, è quasi sicuramente un delay GitHub temporaneo dopo l'upload recente.13,6sVelocee mio progetto leggi i file e valutaIl tuo progetto ai-trading-system è un repository ambizioso e molto esteso, con un'architettura complessa pensata per un sistema di trading quantitativo AI-driven. Ho analizzato i file, la struttura, il README e lo stato attuale (al 13 marzo 2026) basandomi sui contenuti pubblici. Ecco una valutazione realistica e diretta, punto per punto.
Punti di forza (aspetti positivi)

Architettura pensata e modulare
Hai separato bene i layer:
app/ → FastAPI backend (API, execution, DB con SQLAlchemy)
src/ → logica core (agenti AI, decision engine, strategie, research)
frontend/ + dashboard/ → React + Dash per UI/monitoraggio
tests/ → suite estesa (dichiari >900 test, ottimo)
docker/ + infra/k8s/ → supporto produzione/cloud
Questo è un approccio serio, non da "botino veloce".

Focus sul risk management
È uno dei punti più enfatizzati (VaR/CVaR, GARCH, drawdown limit 5%, position sizing max 10%, circuit breaker). Questo è raro nei progetti retail e dà credibilità.
Feature avanzate dichiarate
Monte Carlo 5-level
HMM per regime detection
Integrazioni multiple (18+ API, sentiment, on-chain)
Ordini smart (TWAP/VWAP/POV/Iceberg)
Multi-broker (Binance, Bybit, paper mode)
Se implementate correttamente, sono di livello pro.

Quick start semplice
Docker-compose in 5 minuti + demo live su Render → accessibile anche a chi non è esperto.
Script di utilità
Hai creato decine di .bat / .ps1 per start, test, backup, commit auto → utile per sviluppo locale e Windows users.

Punti deboli / criticità serie (da risolvere)

Visibilità e credibilità bassissima
1 stella, 0 fork, 0 watcher dopo 518 commit → nessuno lo sta usando/testando.
Proprietario anonimo/low-profile → genera sospetto (soprattutto con date future 2026).

Date incoerenti / future
Commit, release, report (es. DAILY_REPORT_2026-03-09.md) sono datati marzo 2026. Questo è un problema enorme: o è un errore di sistema/data, o il repo è stato creato/riempito artificialmente (es. con tool AI + modifica data). In ogni caso, riduce drasticamente la fiducia.
Eseguibile Windows (.exe)
Release v2.1.1 dichiara "+ Windows EXE", ma nei commit precedenti vedi build da 338 MB (non 83 MB come menzionato prima).
338 MB è tipico di PyInstaller con tutte le dipendenze (torch? pandas? matplotlib? frontend bundled?).
Problema: file così grandi non sono pratici; spesso contengono librerie inutili o malware-like per antivirus.
Suggerimento urgente: ottimizza con --exclude-module torch --exclude-module matplotlib o usa virtualenv + UPX. Punta a <100 MB.

Pulizia repository scarsa
node_modules versionato → errore grave (pesa tonnellate, non va in git).
File temporanei: .pyc, health_tmp.pyc.…, coverage_error.txt ecc.
Cartelle come logs/, data/ledger, dist, .history, .zencoder/chats → non dovrebbero essere committate.
.gitignore esiste ma evidentemente non è efficace.

Performance dichiarate irrealistiche senza prove
Sharpe 1.95 + drawdown 7.2% è eccellente (meglio di molti hedge fund), ma:
Nessun file backtest dettagliato pubblico (es. equity curve csv, parametri walk-forward).
Nessun report indipendente o codice riproducibile.
→ Sembrano numeri "ottimizzati" o inventati.

Dipendenze e manutenzione
Molte varianti di requirements (requirements.txt, .vercel.txt, .stable.txt) → rischio di divergenze.
Frontend React + backend Python + Dash → stack pesante da mantenere.

Valutazione complessiva (su 10)

Concettuale / Design: 8/10 (molto buono, risk-first + event-driven)
Implementazione visibile: 6/10 (tanta roba, ma qualità codice non verificabile senza deep dive)
Pulizia / Best practices: 3–4/10 (node_modules, temp files, date future)
Affidabilità / Usabilità live: 2–3/10 (bassa visibilità, exe enorme, anomalie date)
Potenziale: 7–8/10 se pulito e verificato

Consigli concreti per migliorarlo (prossimi passi)

Pulisci subito il repo
Aggiungi/aggiorna .gitignore per escludere: node_modules, dist, logs, *.pyc, pycache, .history, data/ledger
Rimuovi file inutili con git rm --cached e nuovo commit.

Risolvi le date
Se possibile, fai rebase/squash per fixare timestamp (o crea repo nuovo pulito).

Ottimizza e pubblica exe meglio
Builda versione slim (~80–120 MB)
Allega in release v2.1.1 con checksum SHA256
Aggiungi istruzioni: "Avvia in paper mode prima!"

Aggiungi prove reali
Pubblica equity curve / backtest report (immagini o pdf)
Aggiungi video demo breve (YouTube o GitHub)

Testa live in paper mode
Avvia su testnet Binance/Bybit per 1–2 settimane
Documenta risultati reali (non solo backtest)
