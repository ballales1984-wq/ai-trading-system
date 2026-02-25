# AI Trading System - Spiegazione Semplice

## Cosa fa?

Sistema di trading automatico con AI che decide quando comprare/vendere criptovalute.

## Le 5 Domande

| # | Domanda | Funzione |
|---|---------|----------|
| 1 | COSA? | ML + Tech Analysis → Buy/Sell |
| 2 | PERCHE? | Macro + Sentiment |
| 3 | QUANTO? | Position sizing |
| 4 | QUANDO? | Monte Carlo timing |
| 5 | RISCHIO? | VaR/CVaR limits |

## Architettura Sicura

```
Browser → Frontend → Backend (tuo server) → Exchange
                     ↑
              Le TUE API keys (qui!)
```

Le chiavi API sono sul TUO server, MAI sul frontend!

## Struttura

- decision_engine/ - Cervello AI
- app/ - Backend FastAPI
- frontend/ - Dashboard React
- src/ - Componenti core

## Avvio

```
bash
# Backend
python app/main.py

# Frontend
cd frontend && npm run dev

# Docker
docker-compose up
```

## API Keys

File .env (NON committare!):

```
bash
BINANCE_API_KEY=tuakey
BINANCE_SECRET_KEY=segreto
