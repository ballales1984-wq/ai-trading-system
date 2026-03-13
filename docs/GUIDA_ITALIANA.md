# Guida Italiana - AI Trading System

## Introduzione

Questo è un sistema di trading algoritmico avanzato che utilizza l'intelligenza artificiale per analizzare i mercati finanziari ed eseguire operazioni di trading in modo automatico.

---

## Modalità di Trading

### 1. Paper Trading (Consigliato per iniziare)

Il paper trading permette di simulare operazioni di trading senza usare soldi reali. È il modo più sicuro per testare il sistema.

**Come avviarlo:**

```bash
# Windows
start_paper_trading.bat

# Oppure direttamente
python start_paper_trading.py
```

**A cosa serve:**

- Testare le strategie di trading senza rischiare denaro reale
- Vedere come il sistema performa in condizioni di mercato reali
- Imparare come funziona il sistema
- Fare debugging e modifiche in sicurezza

---

### 2. Live Trading (Trading con denaro reale)

Il live trading esegue operazioni reali sui mercati. **Usare con estrema cautela!**

**Come avviarlo:**

```bash
# Windows
start_ai_trading.bat

# Oppure direttamente
python run_live.py
```

**A cosa serve:**

- Trading automatico con denaro reale
- Richiede API keys degli exchange (Binance, Bybit, ecc.)

**⚠️ AVVERTENZE IMPORTANTI:**

- Solo per utenti esperti
- Iniziare sempre con il paper trading
- Configurare correttamente le API keys
- Impostare limiti di perdita massima

---

### 3. Backend API

Il backend fornisce tutte le API per il trading system.

**Come avviarlo:**

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Endpoint disponibili:**

- `http://localhost:8000` - Documentazione API
- `http://localhost:8000/docs` - Swagger UI
- `http://localhost:8000/api/v1/market/prices` - Prezzi mercato
- `http://localhost:8000/api/v1/portfolio/*` - Portfolio

---

### 4. Frontend (Dashboard)

L'interfaccia web per visualizzare e controllare il sistema.

**Come avviarlo:**

```bash
cd frontend
npm run dev
```

**Poi aprire:** <http://localhost:5173>

**Pagine disponibili:**

- **Dashboard**: Panoramica del sistema e statistiche
- **Portfolio**: Le tue posizioni e performance
- **Market**: Dati di mercato in tempo reale
- **Strategies**: Gestione delle strategie di trading

---

## Configurazione

### API Keys

Per fare trading reale, devi configurare le API keys:

1. **Binance**: Creare API key da binance.com
2. **Bybit**: Creare API key da bybit.com
3. **Altre exchange**: Configurare nel file `.env`

Le variabili di ambiente da impostare:

```
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret
BYBIT_API_KEY=your_bybit_key
BYBIT_SECRET_KEY=your_bybit_secret
```

### Database

Il sistema usa PostgreSQL. Assicurati che il database sia avviato:

```bash
# Con Docker
docker-compose up -d postgres
```

---

## Strategie di Trading

Il sistema include diverse strategie:

### Mean Reversion

Cerca situazioni dove il prezzo si allontana troppo dalla media e torna indietro.

### Monte Carlo

Utilizza simulazioni probabilistiche per decidere le operazioni.

### Montblanck

Strategia proprietaria basata su analisi multi-timeframe.

### ML Enhanced

Utilizza machine learning (XGBoost) per predire i movimenti di prezzo.

---

## Gestione del Rischio

Il sistema include protezioni importanti:

- **Stop Loss**: Limita le perdite per ogni operazione
- **Take Profit**: Chiude automaticamente in profitto
- **Position Sizing**: Dimensione corretta delle posizioni
- **Max Drawdown**: Limite massimo di perdita totale
- **VaR (Value at Risk)**: Calcolo del rischio di portafoglio

---

## Comandi Utili

### Avvio rapido (tutto insieme)

```bash
# Windows - avvia tutto
start_stable.bat
```

### Test delle API

```bash
python test_api.py
```

### Test del trading

```bash
python test_paper_trading.py
```

### Monitoraggio

```bash
python dashboard_realtime.py
```

---

## Risoluzione Problemi

### Il frontend non si connette all'API

1. Verifica che il backend sia avviato sulla porta 8000
2. Controlla che non ci siano errori nel terminale del backend

### Errori di connessione al database

1. Verifica che PostgreSQL sia avviato
2. Controlla le credenziali nel file `.env`

### Operazioni non eseguite

1. Controlla di avere soldi nel portafoglio (paper o reale)
2. Verifica che le API keys siano configurate
3. Controlla i log nella cartella `logs/`

---

## Struttura del Progetto

```
ai-trading-system/
├── app/                    # Backend FastAPI
│   ├── api/              # API routes
│   └── core/             # Core utilities
├── frontend/              # React frontend
│   └── src/
│       ├── pages/       # Pagine web
│       └── services/    # API client
├── src/                   # Strategie e trading
│   ├── strategy/        # Strategie di trading
│   ├── decision/        # Motore decisionale
│   └── execution/      # Esecuzione ordini
├── tests/                # Test
└── docs/                 # Documentazione
```

---

## Supporto

Per problemi o domande:

1. Controllare i file di log nella cartella `logs/`
2. Consultare la documentazione in `docs/`
3. Testare con `python test_api.py`

---

## Note Finali

- **Inizia sempre con paper trading**
- **Non usare mai più di quanto puoi permetterti di perdere**
- **Monitora sempre il sistema**
- **Fai backup regolari del database**

Buon trading! 🚀
