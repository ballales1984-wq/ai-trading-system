# Piano di Test - AI Trading System
## Periodo di Test: 3 Mesi

---

## 1. Panoramica del Sistema

L'AI Trading System è un ecosistema completo di trading che include:

### 1.1 Componenti Principali
- **Frontend** (Porta 5173): React + Vite
- **Backend API** (Porta 8000): FastAPI + Python
- **Dashboard** (Porta 8050): Dash (Plotly) + Python
- **AI Assistant** (Porta 8501): Streamlit
- **Auto-trader**: Trading automatizzato

### 1.2 Mercati Supportati
- Criptovalute (BTC, ETH, SOL, etc.)
- Forex (EUR/USD, GBP/USD, etc.)
- Materie Prime (Oro, Petrolio, etc.)
- Indici (US30, US500, etc.)

---

## 2. Piano di Test Settimanale

### Settimana 1-2: Test Funzionali Base

#### Giorno 1-3: Autenticazione e Accesso
- [ ] Test login con credenziali corrette
- [ ] Test login con credenziali errate
- [ ] Test logout
- [ ] Test sessione timeout
- [ ] Test accesso con token JWT

#### Giorno 4-7: Dashboard Principale
- [ ] Verifica caricamento dashboard
- [ ] Test refresh dati in tempo reale
- [ ] Test visualizzazione portfolio
- [ ] Test metriche di rischio
- [ ] Test grafici performance

#### Giorno 8-14: Trading e Ordini
- [ ] Test creazione ordine market
- [ ] Test creazione ordine limit
- [ ] Test creazione ordine stop-loss
- [ ] Test cancellazione ordine
- [ ] Test modifica ordine
- [ ] Test storico ordini

### Settimana 3-4: Test di Integrazione

#### Giorno 15-21: API e Servizi
- [ ] Test tutti gli endpoint API
- [ ] Test WebSocket connessioni
- [ ] Test real-time price updates
- [ ] Test notifiche push
- [ ] Test rate limiting

#### Giorno 22-28: Auto-Trader
- [ ] Test avvio auto-trader
- [ ] Test generazione segnali
- [ ] Test esecuzione ordini automatica
- [ ] Test stop-loss automatico
- [ ] Test take-profit automatico

### Settimana 5-8: Test di Performance

#### Giorno 29-56: Load Testing
- [ ] Test carico utenti concurrenti
- [ ] Test API response time
- [ ] Test database performance
- [ ] Test WebSocket throughput
- [ ] Test memory leaks
- [ ] Test CPU usage

### Settimana 9-12: Test di Stabilità

#### Giorno 57-84: Long-Running Tests
- [ ] Test 24h sistema continuo
- [ ] Test 7 giorni sistema continuo
- [ ] Test recover da errori
- [ ] Test backup e restore
- [ ] Test logging e monitoring

---

## 3. Checklist Giornaliera

### Test Manuali Quotidiani
```
[x] Verifica tutti i servizi attivi
[x] Controlla errori nei log
[x] Verifica dati mercato aggiornati
[x] Controlla esecuzioni ordini
[x] Verifica metriche portfolio
[x] Test funzionalità frontend
[x] Test funzionalità dashboard
[x] Verifica notifiche
```

---

## 4. Metriche da Monitorare

### 4.1 Performance
- Tempo di risposta API < 200ms
- Tempo di caricamento pagina < 3s
- Utilizzo CPU < 80%
- Utilizzo memoria < 85%

### 4.2 Trading
- Numero ordini eseguiti/giorno
- Win rate strategie
- Drawdown massimo
- Sharpe ratio

### 4.3 Sistema
- Uptime servizi > 99.5%
- Errori critici = 0
- Timeout API = 0

---

## 5. Test Automatizzati

### 5.1 Unit Tests
```bash
# Esegui test unitari
pytest tests/unit/ -v

# Coverage target > 80%
pytest --cov=app tests/
```

### 5.2 Integration Tests
```bash
# Test API
pytest tests/api/ -v

# Test database
pytest tests/database/ -v
```

### 5.3 E2E Tests
```bash
# Test frontend
cd frontend && npm test

# Test end-to-end
playwright test
```

---

## 6. Bug Reporting

### Template Segnalazione Bug
```
Titolo: [Componente] Descrizione breve
Priority: [P1/P2/P3/P4]
Componente: [Frontend/Backend/Dashboard/AutoTrader]
Passi per riprodurre:
1. 
2. 
3. 
Risultato atteso:
Risultato attuale:
Screenshot/Log:
```

---

## 7. Risorse

### Accessi
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- Dashboard: http://localhost:8050
- AI Assistant: http://localhost:8501
- API Docs: http://localhost:8000/docs

### Credenziali Demo
- Email: admin@ai-trading.com
- Password: admin123

---

## 8. Contatti e Supporto

Per problemi o domande durante il test:
- Controllare i log nei terminali attivi
- Verificare lo stato dei servizi con `/health` endpoint
- Consultare la documentazione in `docs/`

---

*Ultimo aggiornamento: 2026-03-18*
*Versione: 2.1.0*
