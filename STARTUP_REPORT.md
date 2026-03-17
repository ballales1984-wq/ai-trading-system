# AI Trading System - Rapporto di Avvio

## Data: 17 Marzo 2026

---

## 1. Panoramica del Sistema

L'**AI Trading System** è una piattaforma di trading algoritmica completa che include:

- **Backend API** (FastAPI) - Server principale per la logica di trading
- **Frontend React** - Interfaccia utente modern
- **Dashboard Python** (Dash) - Dashboard analitico avanzato
- **AI Assistant** (Streamlit) - Chatbot per analisi in linguaggio naturale

---

## 2. Servizi Attivi

| # | Servizio | URL | Porta | Status |
|---|----------|-----|-------|--------|
| 1 | Backend API | http://localhost:8000 | 8000 | ✅ Attivo |
| 2 | Frontend React | http://localhost:5173 | 5173 | ✅ Attivo |
| 3 | Dashboard React | http://localhost:5173/dashboard | 5173 | ✅ Attivo |
| 4 | Python Dash | http://localhost:8050 | 8050 | ✅ Attivo |
| 5 | AI Chat Assistant | http://localhost:8501 | 8501 | ✅ Attivo |

---

## 3. Pagine Disponibili

### Frontend (Porta 5173)
- `/` - Pagina Marketing
- `/login` - Login utente
- `/dashboard` - Dashboard principale con dati real-time
- `/portfolio` - Gestione portafoglio
- `/market` - Grafici e prezzi mercato
- `/orders` - Storico ordini
- `/news` - News AI con sentiment analysis
- `/strategy` - Strategie di trading
- `/risk` - Metriche di rischio
- `/settings` - Impostazioni

### Dashboard Python (Porta 8050)
- Dashboard completo con:
  - Grafici prezzi in tempo reale
  - Indicatori tecnici (RSI, MACD, Bollinger Bands)
  - Analisi Monte Carlo
  - Regime detection (HMM)
  - Meta-labeling

### AI Assistant (Porta 8501)
- Chatbot per domande in linguaggio naturale
- Esempi di domande:
  - "Qual è il mio rischio?"
  - "Come va il portafoglio?"
  - "Spiegami cosa è il VaR"
  - "Mostrami le performance"

---

## 4. Componenti Principali

### Backend (`app/`)
- `app/api/routes/` - API endpoints
  - `orders.py` - Gestione ordini
  - `portfolio.py` - Portfolio management
  - `market.py` - Dati mercato
  - `news.py` - News e sentiment
- `app/core/` - Configurazione
- `app/risk/` - Moduli di rischio

### Moduli Trading (`src/`)
- `technical_analysis.py` - Analisi tecnica
- `decision_engine.py` - Motore decisionale
- `ml_predictor.py` - Modelli ML
- `risk_engine.py` - Gestione rischi
- `hedgefund_ml.py` - Pipeline ML avanzate

### OpenClaw Integration (`openclaw_skills/`)
- HMM Regime Detection
- GARCH Volatility
- Monte Carlo Simulation
- Portfolio Optimization

---

## 5. Fix Applicati

### Dashboard Python (porta 8050)
- Corretto errore `DataFrame.to_dict()` con parametro `date_format` non supportato
- Aggiunto error handling ai callback
- Sostituito `orient='records'` con chiamate compatibili

### Note
Il dashboard Python (8050) mostra ancora alcuni errori nei log:
- `Series.to_dict()` - compatibilità pandas
- `HiddenMarkovRegimeDetector` - parametri diversi dalla versione della libreria

Questi sono warning non critici che non impediscono il funzionamento base.

---

## 6. Come Avviare il Sistema

### Metodo 1: Script Automatizzato
```bash
start_frontend_and_backend.bat
```

### Metodo 2: Avvio Manuale

```bash
# Terminal 1: Backend
cd c:/ai-trading-system
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd c:/ai-trading-system/frontend
npm run dev

# Terminal 3: Dashboard Python (opzionale)
cd c:/ai-trading-system/dashboard
python app.py

# Terminal 4: AI Assistant (opzionale)
cd c:/ai-trading-system
streamlit run ai_financial_dashboard.py --server.port 8501
```

---

## 7. Problemi Noti

1. **Dashboard Python (8050)**: Errori non critici nei log
2. **Streamlit**: Deprecation warning per `use_container_width`

---

## 8. Prossimi Sviluppi

- [ ] Integrare funzionalità avanzate nel React Dashboard
- [ ] Migliorare compatibilità HMM con libreria aggiornata
- [ ] Aggiungere更多 metriche in tempo reale
- [ ] Implementare trading live

---

## 9. Contatti e Risorse

- **GitHub**: https://github.com/ballales1984-wq/ai-trading-system
- **Documentazione API**: `/docs` su porta 8000

---

*Rapporto generato il 17 Marzo 2026*
