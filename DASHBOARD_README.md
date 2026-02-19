# 📊 Modern Trading Dashboard

Dashboard moderna integrata con l'API FastAPI per il monitoraggio in tempo reale del portfolio e gestione del rischio.

## 🚀 Caratteristiche

- **Integrazione API REST**: Si connette direttamente all'API FastAPI
- **Aggiornamento Real-time**: Aggiorna automaticamente ogni 5 secondi
- **Visualizzazioni Moderne**: Grafici interattivi con Plotly
- **Metriche di Rischio**: VaR, CVaR, Sharpe Ratio
- **Gestione Ordini**: Visualizza ordini recenti e posizioni aperte
- **Market Data**: Prezzi di mercato in tempo reale
- **UI Moderna**: Design dark theme professionale

## 📋 Requisiti

- Python 3.10+
- FastAPI server in esecuzione su `http://localhost:8000`
- Dipendenze: `dash`, `plotly`, `pandas`, `requests`

## 🎯 Avvio Rapido

### Opzione 1: Script dedicato (Consigliato)

```bash
python start_dashboard_api.py
```

### Opzione 2: Tramite main.py

```bash
# Imposta variabile d'ambiente per usare la dashboard API
set USE_API_DASHBOARD=true
python main.py --mode dashboard
```

### Opzione 3: Direttamente

```bash
python dashboard_api.py
```

## 🌐 Accesso

- **Dashboard**: http://localhost:8050
- **API Server**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📊 Sezioni Dashboard

### 1. Stat Cards
- Total Value: Valore totale del portfolio
- Total P&L: Profitto/Perdita totale
- Cash Balance: Saldo disponibile
- Open Positions: Numero di posizioni aperte
- VaR (1-day): Value at Risk giornaliero
- Sharpe Ratio: Rapporto rischio/rendimento

### 2. Portfolio Chart
Grafico storico del valore del portfolio negli ultimi 30 giorni

### 3. Risk Metrics Chart
Visualizzazione delle metriche di rischio (VaR, CVaR)

### 4. Positions Table
Tabella delle posizioni aperte con:
- Symbol
- Side (LONG/SHORT)
- Quantity
- Entry Price
- Current Price
- Unrealized P&L

### 5. Orders Table
Tabella degli ordini recenti con:
- Order ID
- Symbol
- Side
- Type
- Quantity
- Status
- Created timestamp

### 6. Market Prices
Ticker dei prezzi di mercato principali con variazione 24h

## 🔧 Configurazione

Modifica `dashboard_api.py` per cambiare:

```python
API_BASE_URL = "http://localhost:8000"  # URL API server
API_PREFIX = "/api/v1"                   # Prefisso API
```

E l'intervallo di aggiornamento:

```python
dcc.Interval(
    id='interval-component',
    interval=5*1000,  # Millisecondi (5 secondi)
    n_intervals=0
)
```

## 🎨 Personalizzazione Theme

Modifica il dizionario `THEME` in `dashboard_api.py`:

```python
THEME = {
    'background': '#0d1117',
    'card': '#161b22',
    'border': '#30363d',
    'text': '#c9d1d9',
    'green': '#3fb950',
    'red': '#f85149',
    # ... altri colori
}
```

## 🐛 Troubleshooting

### Dashboard non si connette all'API

1. Verifica che l'API server sia in esecuzione:
   ```bash
   curl http://localhost:8000/health
   ```

2. Controlla l'URL nell'API_BASE_URL

3. Verifica i log della console per errori

### Dati non aggiornati

- Controlla che l'intervallo di refresh sia configurato correttamente
- Verifica che l'API restituisca dati validi
- Controlla la console del browser per errori JavaScript

## 📝 Note

- La dashboard è read-only (sola lettura)
- Per creare/modificare ordini usa l'API direttamente o Swagger UI
- I dati vengono aggiornati automaticamente ogni 5 secondi
- La dashboard funziona anche se alcuni endpoint API non sono disponibili (mostra messaggi di errore)

## 🔄 Sviluppi Futuri

- [ ] Creazione ordini direttamente dalla dashboard
- [ ] Filtri avanzati per posizioni e ordini
- [ ] Grafici tecnici avanzati
- [ ] Notifiche in tempo reale
- [ ] Export dati in CSV/PDF
- [ ] Multi-account support
- [ ] Dark/Light theme toggle
