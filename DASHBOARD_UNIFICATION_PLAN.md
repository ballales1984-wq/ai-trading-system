# 🎯 Dashboard Unification Plan - Page Mapping

## Current Pages & Their Integration

The system currently has **10 pages** accessible via the sidebar navigation. Here's how they map to a unified dashboard:

### Current Route Structure

| Route | Page | Size | Funzionalità | Integrazione Dashboard |
|-------|------|------|--------------|----------------------|
| `/dashboard` | Dashboard.tsx | 27KB | Overview + Tabs | ✅ Mantieni come home |
| `/portfolio` | Portfolio.tsx | 21KB | Portafoglio + allocazione | ➡️ Tab nel dashboard |
| `/market` | Market.tsx | 16KB | Prezzi + grafici | ➡️ Tab nel dashboard |
| `/orders` | Orders.tsx | 12KB | Ordini + creazione | ➡️ Tab nel dashboard |
| `/news` | News.tsx | 5KB | Feed notizie | ➡️ Widget laterale |
| `/strategy` | Strategy.tsx | 5KB | Gestione strategie | ➡️ Tab nel dashboard |
| `/risk` | Risk.tsx | 7KB | Metriche rischio | ➡️ Tab nel dashboard |
| `/settings` | Settings.tsx | 7KB | Impostazioni | ✅ Pagina separata |
| `/login` | Login.tsx | 9KB | Autenticazione | ✅ Pagina pubblica |
| `/marketing` | Marketing.tsx | 14KB | Landing page | ✅ Pagina pubblica |

---

## 🚀 Proposed Unified Dashboard Layout

### Struttura a Tab (come attuale)

```
┌─────────────────────────────────────────────────────────────────┐
│  SIDEBAR (fissa)                                                │
│  ┌─────────┐                                                    │
│  │ Logo    │   ┌──────────────────────────────────────────┐    │
│  │─────────│   │  TAB BAR: [Overview] [Portfolio]      │    │
│  │ Dashboard│   │           [Market] [Orders] [Risk]   │    │
│  │ Portfolio│   ├──────────────────────────────────────────┤    │
│  │ Market  │   │                                          │    │
│  │ Orders  │   │         CONTENUTO TAB                   │    │
│  │ News    │   │                                          │    │
│  │ Strategy│   │                                          │    │
│  │ Risk    │   │                                          │    │
│  │─────────│   │                                          │    │
│  │ Settings│   └──────────────────────────────────────────┘    │
│  └─────────┘                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Tab 1: Overview (Dashboard attuale)
- Riepilogo portafoglio (totale, P/L giornaliero)
- Grafico performance (area chart)
- Posizioni aperte (top 5)
- Ordini recenti (top 5)
- Prezzi live (BTC, ETH, SOL)
- Status WebSocket

### Tab 2: Portfolio
- Grafico allocazione (pie chart)
- Storico portafoglio (line chart)
- Tutte le posizioni con dettagli
- Performance metrics (Sharpe, Drawdown, Win Rate)
- Pulsante per rebalancing

### Tab 3: Market
- Lista prezzi con variazioni
- Grafico candele (selezionabile per simbolo)
- Order book
- Sentiment mercato
- News feed laterale

### Tab 4: Orders
- Lista ordini con filtri
- Form creazione ordine
- Storico ordini
- Stato emergenza (trading halted)

### Tab 5: Risk
- VaR, CVaR, Volatilità
- Limiti di rischio
- Posizioni con rischio
- Matrice correlazione

---

## 📦 Implementazione: Nuovi Componenti

### 1. UnifiedDashboard Container

```typescript
// frontend/src/pages/UnifiedDashboard.tsx
import { useState } from 'react';
import { DashboardOverview } from '../components/dashboard/OverviewTab';
import { DashboardPortfolio } from '../components/dashboard/PortfolioTab';
import { DashboardMarket } from '../components/dashboard/MarketTab';
import { DashboardOrders } from '../components/dashboard/OrdersTab';
import { DashboardRisk } from '../components/dashboard/RiskTab';

type TabId = 'overview' | 'portfolio' | 'market' | 'orders' | 'risk';

const tabs = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'portfolio', label: 'Portfolio', icon: PieChart },
  { id: 'market', label: 'Market', icon: TrendingUp },
  { id: 'orders', label: 'Orders', icon: ClipboardList },
  { id: 'risk', label: 'Risk', icon: Shield },
];

export default function UnifiedDashboard() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');

  return (
    <div className="p-6">
      {/* Tab Navigation */}
      <div className="flex gap-2 mb-6">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as TabId)}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              activeTab === tab.id ? 'bg-primary text-white' : 'bg-gray-800'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'overview' && <DashboardOverview />}
        {activeTab === 'portfolio' && <DashboardPortfolio />}
        {activeTab === 'market' && <DashboardMarket />}
        {activeTab === 'orders' && <DashboardOrders />}
        {activeTab === 'risk' && <DashboardRisk />}
      </div>
    </div>
  );
}
```

### 2. Creazione Componenti Tab

```
frontend/src/components/dashboard/
├── OverviewTab.tsx      # Estrai da Dashboard.tsx attuale
├── PortfolioTab.tsx     # Estrai da Portfolio.tsx
├── MarketTab.tsx        # Estrai da Market.tsx
├── OrdersTab.tsx       # Estrai da Orders.tsx
├── RiskTab.tsx         # Estrai da Risk.tsx
└── index.ts            # Esporta tutto
```

### 3. Estrazione Step-by-Step

#### Step 1: Estrai OverviewTab
```bash
# Copia sezione "Overview" da Dashboard.tsx
# Crea: components/dashboard/OverviewTab.tsx
```

#### Step 2: Estrai PortfolioTab  
```bash
# Sposta tutto Portfolio.tsx in components/dashboard/
# Aggiusta import paths
```

#### Step 3: Estrai MarketTab
```bash
# Sposta Market.tsx in components/dashboard/
# Integra NewsFeed come widget laterale
```

#### Step 4: Estrai OrdersTab
```bash
# Sposta Orders.tsx in components/dashboard/
# Aggiungi form creazione ordine
```

#### Step 5: Estrai RiskTab
```bash
# Sposta Risk.tsx in components/dashboard/
# Aggiungi grafici correlazione
```

---

## 🔧 Modifiche a App.tsx

```typescript
// frontend/src/App.tsx
import UnifiedDashboard from './components/dashboard/UnifiedDashboard';

// ... altre import

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/" element={<Marketing />} />
        <Route path="/marketing" element={<Marketing />} />
        <Route path="/login" element={<Login />} />
        
        {/* Protected routes - Single unified dashboard */}
        <Route element={<Layout />}>
          <Route path="dashboard" element={<UnifiedDashboard />} />
          
          {/* Legacy routes - redirect to dashboard tabs */}
          <Route path="portfolio" element={<Navigate to="/dashboard?tab=portfolio" />} />
          <Route path="market" element={<Navigate to="/dashboard?tab=market" />} />
          <Route path="orders" element={<Navigate to="/dashboard?tab=orders" />} />
          <Route path="risk" element={<Navigate to="/dashboard?tab=risk" />} />
          
          <Route path="settings" element={<Settings />} />
          <Route path="news" element={<News />} />      // Keep as separate
          <Route path="strategy" element={<Strategy />} /> // Keep as separate
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
```

---

## 📋 Lista Task Implementazione

### Fase 1: Preparazione
- [ ] Creare directory `components/dashboard/`
- [ ] Creare file index.ts per esportazione
- [ ] Definire interfacce condivise

### Fase 2: Estrazione Componenti
- [ ] Estrarre OverviewTab da Dashboard.tsx
- [ ] Estrarre PortfolioTab da Portfolio.tsx
- [ ] Estrarre MarketTab da Market.tsx
- [ ] Estrarre OrdersTab da Orders.tsx
- [ ] Estrarre RiskTab da Risk.tsx

### Fase 3: Integrazione
- [ ] Creare UnifiedDashboard.tsx
- [ ] Implementare navigazione tab
- [ ] Condividere dati tra tab (context o props)
- [ ] Testare transizioni tab

### Fase 4: Refactoring Routes
- [ ] Aggiornare App.tsx
- [ ] Implementare redirect per vecchie route
- [ ] Aggiornare sidebar links

---

## ✅ Vantaggi dell'Unificazione

| Aspetto | Prima | Dopo |
|---------|-------|------|
| Navigazione | 7 click per cambiare vista | 1 click (tab) |
| Caricamento dati | Fetch separato per ogni pagina | Cache condivisa |
| Stato utente | Perso tra pagine | Mantenuto nel contesto |
| UX | Frammentata | Fluida e coerente |
| Manutenzione | 7 file separati | 1 container + 5 componenti |

---

## 📞 Prossimi Passi

1. **Confermare il piano** - Vuoi procedere con l'implementazione?
2. **Prioritizzare** - Quale tab implementare prima?
3. **Test** - Vogliamo aggiungere test unitari?

---

*Plan generato per AI Trading System v2.3*
