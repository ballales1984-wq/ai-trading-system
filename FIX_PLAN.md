# Piano di Correzione - AI Trading System

## 🚨 PROBLEMI CRITICI DA RISOLVERE

### 1. Monte Carlo - Random Seeds Fix
**File**: `decision_engine/monte_carlo.py`
**Problema**: `np.random.seed(42)` hardcoded su linee 68, 108, 143, 172, 223
**Impatto**: Simulazioni deterministiche = useless
**Soluzione**: Rimuovere tutti i seed fissi

### 2. Risk Engine - Mock Data
**File**: `app/risk/risk_engine.py`
**Problema**: Linea 347-349 usa `np.random.normal()` per generare dati finti
**Impatto**: Risk metrics completamente falsi
**Soluzione**: Usare dati storici reali per calcoli

### 3. HMM - Rolling Window Bug
**File**: `decision_engine/core.py`
**Problema**: Linea 430 - `.rolling()` chiamato su numpy array
**Impatto**: HMM detection non funziona
**Soluzione**: Convertire a pandas Series

### 4. GitIgnore - node_modules
**File**: `.gitignore`
**Problema**: node_modules è già nel .gitignore (linee 28, 77, 100) ma è già tracciato da git
**Impatto**: Repo gonfiato inutilmente
**Soluzione**: Eseguire: `git rm -r --cached node_modules`

---

## 📋 PIANO DI LAVORO

### Fase 1: Fix Critici (Giorno 1-2)
- [x] Rimuovere np.random.seed() da monte_carlo.py
- [x] Fix risk_engine.py con dati reali
- [x] Fix HMM rolling window bug
- [x] Aggiornare .gitignore (già corretto, ma serve git rm --cached)

### Fase 2: Miglioramenti Core (Giorno 3-5)
- [ ] Implementare data collector per historical returns
- [ ] Aggiungere real-time price feed al paper connector
- [ ] Migliorare error handling
- [ ] Aggiungere logging migliore

### Fase 3: Testing (Giorno 6-7)
- [ ] Verificare Monte Carlo output
- [ ] Test risk calculations
- [ ] Validare HMM detection
- [ ] Test paper trading flow

---

## 🔧 COMMANDI UTILI

```bash
# Check current issues
grep -rn "np.random.seed" decision_engine/
grep -rn "np.random.normal" app/risk/

# Fix node_modules (già in gitignore ma tracciato)
git rm -r --cached node_modules
git commit -m "Remove node_modules from git tracking"

# Run tests
python -m pytest tests/ -v

# Check imports
python -c "from decision_engine.core import DecisionEngine"
```
