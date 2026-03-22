# 🖥️ Servizi Attivi - Analisi e Unificazione

## Servizi Attualmente in Esecuzione

| Porta | Servizio | Tecnologia | File |
|-------|----------|------------|------|
| **5173** | Frontend React | React + Vite | `frontend/` |
| **8050** | ML Monitoring | Python Dash | `dashboard/app.py` |
| **8051** | Investor Portal | Python Dash | `dashboard_investor.py` |
| **8502** | AI Assistant | Streamlit | `ai_financial_dashboard.py` |

---

## Analisi di Ciascun Servizio

### 1. Frontend React (Porta 5173)
**Tecnologia**: React 18 + TypeScript + Vite + Tailwind CSS

**Funzionalità**:
- Dashboard principale con tab (Overview, Portfolio, Market, Orders, Risk)
- Autenticazione utente
- Grafici interattivi (Recharts)
- WebSocket per dati real-time

**Pagine**:
- `/dashboard` - Dashboard principale
- `/portfolio` - Gestione portafoglio
- `/market` - Visualizzazione mercati
- `/orders` - Storico ordini
- `/news` - Feed notizie
- `/strategy` - Gestione strategie
- `/risk` - Analisi rischi
- `/settings` - Impostazioni
- `/login` - Login utente

### 2. ML Monitoring Dashboard (Porta 8050)
**Tecnologia**: Python Dash

**Funzionalità**:
- Monitoraggio modelli ML
- Metriche di performance
- Grafici real-time
- Strategy comparison

**File principali**:
- `dashboard/app.py` - Main app
- `dashboard/dashboard_realtime.py` - Grafici real-time
- `dashboard/strategy_comparison_tab.py` - Confronto strategie

### 3. Investor Portal (Porta 8051)
**Tecnologia**: Python Dash

**Funzionalità**:
- Portale per investitori
- Reporting automatizzato
- KPI e metriche

**File**: `dashboard_investor.py`

### 4. AI Financial Assistant (Porta 8502)
**Tecnologia**: Streamlit

**Funzionalità**:
- Assistente AI conversazionale
- Analisi finanziaria
- Query sui dati

**File**: `ai_financial_dashboard.py`

---

## Problemi Attuali

1. **Frammentazione**: 4 interfacce separate
2. **Tecnologie diverse**: React + 2Dash + Streamlit
3. **Dati non sincronizzati**: Ogni dashboard ha i propri dati
4. **UX confusionaria**: L'utente deve navigare tra 4 URL diverse
5. **Manutenzione complessa**: 4 codebase separate

---

## Soluzioni Proposte

### Opzione A: Unificazione nel Frontend React (Raccomandata)

Spostare tutto nel frontend React sulla porta 5173:

```
http://localhost:5173
├── /dashboard          # Dashboard principale (già esiste)
│   ├── Overview        # ML Metrics (da dashboard:8050)
│   ├── Portfolio       # (già esiste)
│   ├── Market          # (già esiste)
│   ├── Orders          # (già esiste)
│   └── Risk            # (già esiste)
├── /investor-portal   # Da dashboard_investor.py
├── /ai-assistant      # Da ai_financial_dashboard.py (Streamlit)
└── /ml-monitoring     # Grafici ML (da dashboard:8050)
```

**Vantaggi**:
- Unica tecnologia (React)
- Unico URL
- Dati condivisi
- UX coerente

**Svantaggi**:
- Migrarre codice Dash/Streamlit in React

### Opzione B: Unificazione con Proxy Nginx

Mantenere i servizi separati ma unificare con reverse proxy:

```
nginx reverse proxy
├── /           → 5173 (React)
├── /ml         → 8050 (Dash)
├── /investor   → 8051 (Dash)
└── /ai         → 8502 (Streamlit)
```

**Vantaggi**:
- Nessuna riscrittura codice
- Sviluppo indipendente

**Svantaggi**:
- 4 servizi ancora attivi
- Dati ancora separati
- Complessità nginx

### Opzione C: Frontend + Embedded Streamlit

Usare componenti React che embedding Streamlit:

```
http://localhost:5173
├── /dashboard          # Già esiste
├── /ai                 # React component + Streamlit via iframe
└── /ml                 # Grafici React (da Dash)
```

**Vantaggi**:
- Mantiene funzionalità esistenti
- Unico punto di accesso

---

## Piano di Implementazione (Opzione A)

### Step 1: Estendere Frontend React
- Aggiungere nuove pagine/routes:
  - `/investor-portal`
  - `/ai-assistant`
  - `/ml-monitoring`

### Step 2: Creare Componenti ML
```typescript
// frontend/src/components/ml/
- MLMetricsCard.tsx      // Metriche modelli
- ModelPerformance.tsx   // Performance modelli
- FeatureImportance.tsx  // Importanza feature
```

### Step 3: Creare Componenti Investor Portal
```typescript
// frontend/src/components/investor/
- InvestorReport.tsx    // Report mensile
- KPICards.tsx          // KPI principali
- PerformanceTable.tsx  // Tabella performance
```

### Step 4: Integrare AI Assistant
```typescript
// frontend/src/components/ai/
- AIChatWidget.tsx     // Chat component
- FinancialQuery.tsx   // Query finanziarie
```

### Step 5: Shutdown Servizi
Dopo migration completata:
- Spegnere porta 8050, 8051, 8502
- Usare solo porta 5173

---

## Tabella Comparativa

| Aspetto | Attuale | Proposto (Opz A) |
|---------|---------|-------------------|
| URL | 4 diversi | 1 solo |
| Tech | React+Dash+Streamlit | Solo React |
| Dati | Separati | Condivisi |
| Manutenzione | 4 codebase | 1 codebase |
| UX | Frammentata | Unificata |

---

## Prossimi Passi

1. **Confermare opzione preferita**: A, B o C?
2. **Prioritizzare funzionalità**: Quale servizio migrare prima?
3. **Pianificare timeline**: Sprint di lavoro

---

*Documento generato per AI Trading System v2.3*
