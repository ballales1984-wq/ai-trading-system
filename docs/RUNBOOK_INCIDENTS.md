# Runbook Incidentale — AI Trading System

Questo documento fornisce le procedure operative per la gestione degli incidenti nel sistema di trading AI.

---

## 1. API Exchange Down

### Sintomi
- Errori 5xx dalle API dell'exchange
- Timeout frequenti nelle chiamate
- Mancanza di aggiornamento prezzi in tempo reale
- Alert da Prometheus/Grafana

### Azioni

#### Step 1: Verifica stato exchange
```bash
# Controlla status page dell'exchange
curl -s https://status.binance.com | grep -i operational

# Verifica connettività
ping -c 5 api.binance.com
```

#### Step 2: Attiva fallback
```bash
# Se configurato, passa a exchange secondario
export ACTIVE_EXCHANGE=kraken  # o paper trading

# Riavvia il sistema con nuovo exchange
python start_ai_trading.bat --exchange kraken
```

#### Step 3: Se nessun fallback disponibile
```bash
# Attiva Emergency Stop
python -m app.emergency_stop --mode=read_only

# Passa in modalità sola lettura
# (nessun nuovo ordine, solo chiusure)
```

#### Step 4: Documenta l'incidente
```bash
# Crea entry nel log incidenti
echo "$(date -Iseconds) - API Exchange Down - Exchange: binance" >> INCIDENT_LOG.md
```

---

## 2. Breach di rischio (drawdown > limite)

### Sintomi
- `RiskBook.daily_drawdown_ok()` restituisce `False`
- Dashboard Grafana mostra drawdown > 5%
- Alert da sistema di risk management

### Azioni

#### Step 1: Blocca immediatamente nuove aperture
```python
# Il codice dovrebbe già farlo automaticamente
# Verifica che non ci siano posizioni nuove
from src.risk.risk_book import risk_book
assert risk_book.daily_drawdown_ok()  # Fallisce se drawdown > 5%
```

#### Step 2: Consenti solo chiusure
```bash
# Abilita modalità close-only
export TRADING_MODE=close_only
```

#### Step 3: Notifica il team
```bash
# Invia alert (es. Discord webhook)
curl -X POST -H "Content-Type: application/json" \
  -d '{"content":"⚠️ RISK ALERT: Drawdown > 5% - Close-only mode attivato"}' \
  $DISCORD_WEBHOOK_URL
```

#### Step 4: Analisi post-incidente
```bash
# Genera report
python -m app.risk.generate_report --period=24h

# Analizza:
# - Strategie attive
# - Esposizioni per asset
# - Log del Decision Engine
```

---

## 3. Latenza elevata

### Sintomi
- Grafana: aumento p95/p99 latency API > 500ms
- OpenClaw risponde lentamente (> 5s)
- Timeout nelle chiamate al decision engine

### Azioni

#### Step 1: Controlla carico sistema
```bash
# Controlla utilizzo risorse
top -b -n 1 | head -20

# Controlla memoria
free -h

# Controlla I/O disco
iostat -x 1 5
```

#### Step 2: Scala orizzontalmente
```bash
# Se usando Kubernetes
kubectl scale deployment ai-trading-backend --replicas=3

# O aumenta worker
export WORKER_THREADS=8
```

#### Step 3: Riduci carico computazionale
```bash
# Riduci simulazioni Monte Carlo
export MC_SIMULATIONS=1000  # invece di 10000

# Disabilita skill pesanti temporaneamente
# Modifica skill_registry.yaml: setta enabled: false per MC
```

#### Step 4: Verifica database
```bash
# Controlla query lente
psql -c "SELECT * FROM pg_stat_activity WHERE state != 'idle';"

# Verifica lock
psql -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

---

## 4. Modello difettoso in produzione

### Sintomi
- Performance anomale (drawdown elevato, basso Sharpe)
- Drift evidente tra previsioni e realtà
- Alert da monitoraggio modelli
- Metriche fuori dai range attesi

### Azioni

#### Step 1: Identifica il modello
```bash
# Usa ModelRegistry per trovare il champion
python -c "
from src.research.model_registry import get_model_registry
registry = get_model_registry()
champion = registry.get_champion('hmm_regime_detector')
print(f'Champion: {champion.version}')
"
```

#### Step 2: Switcha su challenger stabile
```bash
# Promuovi challenger a champion
python -c "
from src.research.model_registry import get_model_registry
registry = get_model_registry()
challengers = registry.get_challengers('hmm_regime_detector')
if challengers:
    # Promuovi il più recente
    registry.promote_to_champion('hmm_regime_detector', challengers[0].version)
"
```

#### Step 3: Riduci sizing temporaneo
```bash
# Riduci size posizioni del 50%
export POSITION_SCALING=0.5
```

#### Step 4: Avvia retraining
```bash
# Genera nuovo training set
python -m src.research.generate_dataset --symbol=BTCUSDT --days=90

# Avvia training
python -m src.research.train_model --model=hmm_regime_detector --dataset=latest
```

---

## 5. Perdita di connettività database

### Sintomi
- Errori di connessione al database
- Timeout nelle query
- Dati non persistenti

### Azioni

#### Step 1: Verifica stato database
```bash
# Check PostgreSQL
pg_isready -h localhost -p 5432

# Check TimescaleDB
psql -c "SELECT 1;" trading_data
```

#### Step 2: Failover a replica
```bash
# Se configurato replica
export DATABASE_URL=postgresql://user:pass@replica:5432/trading
```

#### Step 3: Modalità degradata
```bash
# Abilita modalità offline
export OFFLINE_MODE=true
# Usa dati in cache
```

---

## 6. Allarme di sicurezza

### Sintomi
- Tentativi di accesso non autorizzati
- Attività anomale sul network
- Alert da WAF/AppSec

### Azioni

#### Step 1: Isola il sistema
```bash
# Blocca traffico in entrata
iptables -A INPUT -j DROP

# Solo traffico interno
iptables -A INPUT -s 10.0.0.0/8 -j ACCEPT
```

#### Step 2: Analizza log
```bash
# Cerca attività sospette
grep -i "unauthorized\|failed\|error" app/logs/*.log | tail -100

# Controlla accessi
last -50
```

#### Step 3: Ripristina da backup
```bash
# Se necessario, ripristina a snapshot precedente
python -m app.backup.restore --timestamp=latest
```

---

## Quick Reference

| Incidente | Comando rapido |
|-----------|----------------|
| Exchange down | `python -m app.emergency_stop --mode=read_only` |
| Drawdown eccessivo | `export TRADING_MODE=close_only` |
| Latenza alta | `export MC_SIMULATIONS=1000` |
| Modello difettoso | `registry.promote_to_champion()` |
| DB down | `export OFFLINE_MODE=true` |

---

## Contatti

- **Team DevOps**: #devops-internal
- **Team Trading**: #trading-desk
- **On-call**: Rotazione settimanale (vedi PAGERDUTY_SCHEDULE)

---

*Ultimo aggiornamento: 2024*
*Versione: 1.0*
