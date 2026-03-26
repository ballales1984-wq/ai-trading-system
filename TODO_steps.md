# TODO Steps — Execution Plan (Pragmatic, Test-Driven)

Ultimo aggiornamento: 2026-03-26 (UTC)
Scopo: trasformare il progetto in "production-ready" con passi piccoli, verificabili e con evidenze.

---

## 0) Stato reale verificato

### Test eseguiti
- `pytest --maxfail=1 -q` ➜ **FAIL** su `tests/test_bybit.py` per chiamata rete esterna (Bybit non raggiungibile nell'ambiente).
- `pytest -q tests/test_app_core_modules.py` ➜ **PASS** (34 passed, warning presenti).

### Problemi osservati (da risolvere prima di allargare la suite)
1. Test che dipendono da rete esterna non isolati.
2. Warning pytest config (`asyncio_mode`, `asyncio_default_fixture_loop_scope`).
3. Deprecazioni Pydantic (`Field(..., env=...)`) e `datetime.utcnow()`.

---

## 1) Piano operativo per priorità

## P0 — Stabilizzare pipeline test (subito)

### P0.1 Isolare test esterni (Bybit/network)
- [ ] Marcare i test che richiedono rete/API esterne con marker `integration` o `external`.
- [ ] Escluderli di default nella suite CI locale (`-m "not external"`).
- [ ] Aggiungere mock/fake client per i casi principali di Bybit.

**Definition of Done**
- [ ] `pytest --maxfail=1 -q -m "not external"` passa.
- [ ] Nessun test unitario dipende da internet.

### P0.2 Ridurre warning framework
- [ ] Allineare `pytest.ini` ai plugin installati oppure installare plugin mancanti.
- [ ] Migrare config Pydantic da `env=` al pattern compatibile v2.
- [ ] Sostituire `datetime.utcnow()` con datetime timezone-aware.

**Definition of Done**
- [ ] Warning pytest config azzerati.
- [ ] Warning deprecazione principali ridotti/azzerati nella suite core.

### P0.3 Definire baseline CI-safe
- [ ] Creare comando standard locale/CI: `pytest -q -m "not external"`.
- [ ] Aggiungere comando coverage minimo: `pytest -q -m "not external" --cov=app --cov-report=term-missing`.

**Definition of Done**
- [ ] Pipeline base ripetibile in ambiente senza rete esterna.

---

## P1 — Data & Risk integrity (dopo stabilizzazione test)

### P1.1 Demo vs Live esplicito
- [ ] Introdurre `data_mode = demo|live` esplicito nelle route.
- [ ] In `live`, vietare fallback mock silenziosi: ritornare errore typed + `data_quality`.

### P1.2 Risk metrics affidabili
- [ ] Evitare fallback sintetico non dichiarato per metriche live.
- [ ] Se storico insufficiente: stato `insufficient_data` (non numero artificiale).

### P1.3 Coerenza documentazione
- [ ] Allineare README/TODO/ROADMAP con comportamento runtime reale.

**Definition of Done P1**
- [ ] Nessun dato mock nascosto in endpoint live.
- [ ] Risk API esplicita sulla qualità/sorgente dati.

---

## P2 — Execution reliability & operability

### P2.1 Broker hardening
- [ ] Completare gestione errori/retry/timeouts/idempotenza connettori.
- [ ] Aggiungere reconciliation ordini/posizioni.

### P2.2 Osservabilità
- [ ] Metriche minime: order failure rate, data freshness, reconciliation mismatch.
- [ ] Alerting su soglie critiche.

**Definition of Done P2**
- [ ] Flusso ordini monitorabile end-to-end con alert utili.

---

## 2) Sequenza di esecuzione consigliata (settimana corrente)

### Giorno 1
- [ ] Marcare test esterni + aggiornare selezione pytest.
- [ ] Introdurre target `test-ci-safe` nel Makefile (se assente).

### Giorno 2
- [ ] Fix warning pytest/plugin.
- [ ] Fix `datetime.utcnow()` e deprecazioni più rumorose.

### Giorno 3
- [ ] Eseguire baseline completa CI-safe + coverage.
- [ ] Pubblicare report con numeri (pass/fail/warnings/coverage).

---

## 3) Comandi standard

### Suite CI-safe (senza rete esterna)
```bash
pytest -q -m "not external"
```

### Coverage CI-safe
```bash
pytest -q -m "not external" --cov=app --cov-report=term-missing
```

### Full suite (solo in ambiente con accesso rete/credenziali)
```bash
pytest --maxfail=1 -q
```

---

## 4) Evidenze da allegare a ogni PR
- [ ] Comando eseguito
- [ ] Output sintetico (pass/fail)
- [ ] Warning nuovi/risolti
- [ ] File toccati
- [ ] Rischio regressione + rollback plan
