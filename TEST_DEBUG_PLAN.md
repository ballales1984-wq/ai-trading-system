# Piano di Test e Debug - AI Trading System

## 1. Panoramica della Strategia di Testing

### 1.1 Livelli di Testing
- **Unit Tests**: Test singoli componenti/moduli
- **Integration Tests**: Test interfacce tra moduli
- **System Tests**: Test end-to-end del sistema
- **Performance Tests**: Test di carico e performance

### 1.2 Framework di Testing
- **unittest**: Framework standard Python
- **pytest**: Per test avanzati e coverage
- **pytest-cov**: Per coverage analysis
- **mock**: Per mocking dipendenze esterne

---

## 2. Unit Tests per Modulo

### 2.1 Test Data Collector (`data_collector.py`)

| Test Case | Descrizione | Expected Result |
|-----------|-------------|-----------------|
| `test_exchange_initialization` | Test inizializzazione Binance | Exchange creato correttamente |
| `test_fetch_ohlcv` | Test fetch OHLCV data | DataFrame con dati OHLCV |
| `test_fetch_ticker` | Test fetch ticker price | Prezzo corrente restituito |
| `test_calculate_correlation` | Test calcolo correlazione | Valore tra -1 e 1 |
| `test_new_exchanges` | Test nuovi exchange (Coinbase, Kraken, OKX) | Exchange inizializzato |

### 2.2 Test ML Predictor (`ml_predictor.py`)

| Test Case | Descrizione | Expected Result |
|-----------|-------------|-----------------|
| `test_prepare_features` | Test preparazione features | Features estratte correttamente |
| `test_train_model` | Test training modello | Modello addestrato |
| `test_predict` | Test prediction | Predizione restituita |
| `test_prediction_range` | Test range predizioni | Valori in range valido |

### 2.3 Test Decision Engine (`decision_engine.py`)

| Test Case | Descrizione | Expected Result |
|-----------|-------------|-----------------|
| `test_generate_signal` | Test generazione segnale | Signal generato |
| `test_ml_integration` | Test integrazione ML | ML score incluso nel segnale |
| `test_signal_confidence` | Test calcolo confidence | Confidence 0-1 |
| `test_stop_loss_calculation` | Test calcolo stop loss | Stop loss calcolato |
| `test_position_sizing` | Test calcolo position size | Size calcolato |

### 2.4 Test Technical Analysis (`technical_analysis.py`)

| Test Case | Descrizione | Expected Result |
|-----------|-------------|-----------------|
| `test_rsi_calculation` | Test RSI | RSI tra 0-100 |
| `test_macd_calculation` | Test MACD | Valori MACD validi |
| `test_bollinger_bands` | Test Bollinger Bands | Bande calcolate |
| `test_volume_analysis` | Test analisi volume | Indicatori volume validi |

### 2.5 Test Sentiment Analyzer (`sentiment_news.py`)

| Test Case | Descrizione | Expected Result |
|-----------|-------------|-----------------|
| `test_fear_greed_index` | Test Fear & Greed Index | Indice 0-100 |
| `test_cryptopanic_news` | Test fetch news | News list returned |
| `test_coingecko_data` | Test CoinGecko API | Dati market restituiti |
| `test_sentiment_score` | Test calcolo sentiment | Score normalizzato |

---

## 3. Integration Tests

### 3.1 Test ML + Decision Engine
```python
def test_ml_decision_integration():
    """Test integrazione ML predictor con DecisionEngine"""
    # 1. Crea DecisionEngine
    # 2. Addestra ML model
    # 3. Genera segnale
    # 4. Verifica ml_score nel segnale
```

### 3.2 Test Data + Analysis Pipeline
```python
def test_data_analysis_pipeline():
    """Test pipeline completo data -> analysis -> signal"""
    # 1. Fetch data
    # 2. Technical analysis
    # 3. Sentiment analysis
    # 4. Generate signal
    # 5. Verify all components present
```

### 3.3 Test Multi-Exchange
```python
def test_multi_exchange_support():
    """Test supporto multi-exchange"""
    exchanges = ['binance', 'coinbase', 'kraken', 'okx']
    for ex in exchanges:
        collector = DataCollector(exchange_name=ex)
        assert collector.exchange is not None
```

---

## 4. Debug Strategy

### 4.1 Logging Structure
```python
# Livelli di log:
# DEBUG: Dettagli esecuzione
# INFO: Operazioni normali
# WARNING: Situazioni anomale ma gestite
# ERROR: Errori che non bloccano il sistema
# CRITICAL: Errori critici
```

### 4.2 Breakpoints Strategici
- **Data Collection**: Dopo fetch dati, verifica qualità
- **ML Prediction**: Log input features e output
- **Signal Generation**: Log tutti i component scores
- **Order Execution**: Log dettagli ordine

### 4.3 Error Handling
```python
# Pattern per error handling:
try:
    # Operazione
except SpecificException as e:
    logger.error(f"Details: {e}")
    # Fallback action
finally:
    # Cleanup
```

---

## 5. Test Execution Plan

### 5.1 Ordine di Esecuzione
1. **Unit Tests** (ogni modulo separatamente)
2. **Integration Tests** (moduli collegati)
3. **System Tests** (app completo)
4. **Performance Tests** (sotto carico)

### 5.2 Comandi di Test
```bash
# Esegui tutti i test
pytest tests/ -v

# Esegui con coverage
pytest tests/ --cov=src --cov-report=html

# Esegui test specifico modulo
pytest tests/test_decision_engine.py -v

# Esegui test con verbose output
pytest tests/ -vv --tb=long
```

### 5.3 CI/CD Integration
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/ --cov
```

---

## 6. Known Issues e Debug Points

### 6.1 Problemi Comuni
| Problema | Causa | Soluzione |
|----------|-------|-----------|
| API Rate Limit | Troppe richieste | Implementare rate limiting |
| ML Model Not Trained | Model non addestrato | Call train_ml_model() prima |
| Empty DataFrame | Nessun dato disponibile | Verificare connettività exchange |
| Prediction Out of Range | Features non normalizzate | Ver Normalization |

### 6.2 Debug Checklist
- [ ] Verificare connessione internet
- [ ] Verificare API keys
- [ ] Verificare formato data
- [ ] Verificare dipendenze installate
- [ ] Verificare permessi file
- [ ] Verificare variabili ambiente

---

## 7. Monitoraggio e Alert

### 7.1 Metriche da Monitorare
- **Success Rate**: % segnali generati con successo
- **Latency**: Tempo risposta API
- **Error Rate**: % errori per modulo
- **ML Accuracy**: Accuratezza predizioni ML

### 7.2 Alert Thresholds
```python
ALERT_THRESHOLDS = {
    'error_rate': 0.05,  # 5% error rate
    'latency_ms': 5000,   # 5 secondi
    'success_rate': 0.90  # 90% successo
}
```
