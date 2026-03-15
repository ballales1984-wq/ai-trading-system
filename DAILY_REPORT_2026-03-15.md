# Daily Report - 15 Marzo 2026

## Problemi Risolti

### 1. Python 3.14 Compatibility Issue
- **Problema**: numba non supporta Python 3.14.0 (solo >=3.10,<3.14)
- **Soluzione**: Switchato a Python 3.13 per tutti i servizi backend

### 2. ml_predictor.py - Class Name Error
- **Problema**: `SimpleMovingAveragePredictor` non esisteva
- **Soluzione**: Cambiato a `PricePredictor` in `get_sma_predictor()`

### 3. live_multi_asset.py - Numpy Array Error
- **Problema**: "truth value of array with more than one element is ambiguous"
- **Soluzione**: Aggiunto conversione esplicita da numpy array a scalare prima del confronto

### 4. Missing Dependencies
- Installato: `email-validator`, `prometheus_client`

### 5. Frontend Proxy
- Aggiornato `vite.config.ts` per puntare alla porta corretta (8000)

## Commit GitHub

1. **c9f1ef9** - Fix: Replace SimpleMovingAveragePredictor with PricePredictor
2. **01c8487** - Fix: Handle numpy array outputs in signal generation

## Stato Applicazione

### Servizi Attivi
- **Frontend**: http://localhost:5173 ✅ (HTTP 200)
- **Backend API**: http://localhost:8000 ✅ (healthy)
- **Live Trading Bot**: Running ✅

### Database
- **Ordini salvati**: 20 ordini (Feb 28 - Mar 15, 2026)
- **Portfolio history**: Salvato correttamente
- **Commissioni**: Incluse nei calcoli

## Commissioni Trading

### Paper Trading (attuale)
- **Tasso**: 0.1% fisso (simulato)
- **Note**: Valori fittizi

### Live Trading
- **Tasso**: Commissioni reali di Binance
- **Fonte**: Prelevate direttamente dall'API Binance
- **Tipico**: 0.1% taker, 0.1% maker

## Prossimi Passi (Opzionali)
1. Testare live trading con account Binance testnet
2. Aggiungere più asset (stocks, futures)
3. Implementare AI explainability per segnali
4. Aggiungere stress test Monte Carlo

---
*Report generato automaticamente*
