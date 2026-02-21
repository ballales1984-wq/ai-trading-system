# Refactor Plan - AI Trading System → Micro SaaS

## Overview
Trasformazione del progetto AI Trading System in un micro SaaS pronto per 100 utenti paganti.

## Fase 1: Stabilità Tecnica & Base SaaS (Passi 1-25)

### 1. Refactor Codice Core FastAPI + Dash

#### Stato Attuale
- **Backend FastAPI**: `app/` - Struttura modulare con routes, core, database
- **Dashboard Dash**: `dashboard/` - Applicazione separata
- **Core Logic**: `src/` - Moduli di business logic

#### Obiettivi Refactor
1. **Unificare configurazione** - Singolo file config per tutto il sistema
2. **Standardizzare logging** - Sistema di logging coerente
3. **Preparare multi-tenancy** - Isolamento dati per utente
4. **Ottimizzare import** - Ridurre dipendenze circolari
5. **Type hints** - Aggiungere type hints completi

#### Struttura Target
```
ai-trading-system/
├── app/                    # FastAPI Backend
│   ├── main.py            # Entry point
│   ├── api/routes/        # API endpoints
│   ├── core/              # Config, logging, security
│   ├── database/          # Models, repositories
│   └── services/          # Business logic
├── dashboard/              # Dash Frontend
│   ├── app.py             # Main dashboard
│   └── components/        # Riutilizzabili
├── src/                    # Core trading logic
│   ├── core/              # Engine, event bus
│   ├── strategies/        # Trading strategies
│   ├── risk/              # Risk management
│   └── ml/                # ML models
├── tests/                  # Test suite
└── docs/                   # Documentazione
```

### 2. Logging Dashboard
- [ ] Implementare logging strutturato JSON
- [ ] Aggiungere request ID tracking
- [ ] Log rotation e retention
- [ ] Dashboard error tracking

### 3. Logging Execution Engine
- [ ] Trade execution logging
- [ ] Order lifecycle tracking
- [ ] Performance metrics logging
- [ ] Error handling standardizzato

### 4. Backup Giornaliero Database
- [ ] Script backup automatico
- [ ] Retention policy (30 giorni)
- [ ] Backup verification
- [ ] Restore procedure

### 5. Test Coverage 70→80%
- [ ] Core modules tests
- [ ] API endpoint tests
- [ ] Integration tests
- [ ] Coverage reporting

## Priorità Immediate

### Alta Priorità (Questa settimana)
1. ✅ Backend avviato
2. 🔄 Analisi struttura codice
3. ⬜ Creare config unificato
4. ⬜ Standardizzare logging
5. ⬜ Setup backup database

### Media Priorità (Prossima settimana)
- Dashboard demo read-only
- Screenshot professionali
- Landing page minimale
- Modulo raccolta email

### Bassa Priorità (Settimana 3-4)
- Content marketing
- Social media presence
- Community building

## Metriche di Successo

### Tecniche
- Test coverage > 80%
- Zero errori critici in produzione
- Response time < 200ms API
- Uptime > 99.5%

### Business
- 50 email raccolte (settimana 1)
- 10 beta users (settimana 2)
- 5 paganti (settimana 3)
- 20 paganti (settimana 4)

## Note
- Mantenere backward compatibility
- Documentare ogni cambiamento
- Testare prima di deployare
- Backup prima di modifiche critiche
