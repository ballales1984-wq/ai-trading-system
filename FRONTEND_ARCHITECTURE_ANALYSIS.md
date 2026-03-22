# 📊 Frontend Architecture Analysis & Unification Guide

> **Data**: 2026-03-22  
> **Sistema**: AI Trading System Frontend  
> **Version**: React 18 + TypeScript + Vite

---

## 📋 Executive Summary

This document analyzes the current frontend architecture and provides a step-by-step guide to unify services into a single, scalable dashboard interface.

---

## 🏗️ Current Architecture Overview

### Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18+ | UI Framework |
| TypeScript | 5.x | Type Safety |
| Vite | 5.x | Build Tool |
| Tailwind CSS | 3.x | Styling |
| TanStack Query | 5.x | Data Fetching |
| React Router | 6.x | Navigation |
| Recharts | - | Charts |
| Axios | - | HTTP Client |
| Lucide React | - | Icons |

### Directory Structure

```
frontend/src/
├── App.tsx                    # Main app with routing
├── main.tsx                   # Entry point
├── index.css                  # Global styles
├── types.ts                   # Legacy types
├── components/
│   ├── layout/
│   │   └── Layout.tsx         # Main layout with sidebar
│   ├── ui/                    # Reusable UI components
│   │   ├── ErrorBoundary.tsx
│   │   ├── Toast.tsx
│   │   ├── LoadingSpinner.tsx
│   │   ├── Skeleton.tsx
│   │   └── StatusBadge.tsx
│   ├── charts/                # Chart components
│   │   ├── CandlestickChart.tsx
│   │   ├── MonteCarloChart.tsx
│   │   ├── DrawdownChart.tsx
│   │   └── RiskReturnScatter.tsx
│   └── trading/
│       ├── CandlestickChart.tsx
│       └── OrderBook.tsx
├── pages/                     # Page components
│   ├── Dashboard.tsx          # Main dashboard (27KB)
│   ├── Portfolio.tsx          # Portfolio management
│   ├── Market.tsx             # Market view
│   ├── Orders.tsx             # Order history
│   ├── News.tsx               # News feed
│   ├── Strategy.tsx           # Strategy management
│   ├── Risk.tsx               # Risk dashboard
│   ├── Settings.tsx           # Settings page
│   ├── Login.tsx              # Login page
│   └── Marketing.tsx         # Marketing page
├── services/
│   └── api.ts                # Unified API service (10KB)
├── hooks/
│   ├── useMarketData.ts       # WebSocket market data
│   └── useWebSocket.ts        # WebSocket connection
└── types/
    └── index.ts              # TypeScript definitions
```

---

## 🔌 Current API Services

### Unified API Service (`services/api.ts`)

The frontend uses a centralized API service with the following modules:

| API Module | Endpoints | Usage |
|------------|-----------|-------|
| `portfolioApi` | summary, positions, performance, allocation, history | Portfolio management |
| `marketApi` | prices, price, candles, orderbook, sentiment | Market data |
| `ordersApi` | list, get, create, cancel, execute | Order management |
| `emergencyApi` | getStatus, activate, deactivate | Emergency controls |
| `riskApi` | metrics, limits, positionRisks, correlation | Risk analytics |
| `strategyApi` | list, get, getPerformance, run | Strategy management |
| `newsApi` | getNews, getNewsBySymbol | News feed |
| `paymentApi` | isConfigured, redirectToPaymentLink | Payments |

### Base URL Configuration

```typescript
const defaultApiBase =
  typeof window !== 'undefined' && ['5173', '5174', '3000'].includes(window.location.port)
    ? 'http://127.0.0.1:8000/api/v1'
    : '/api/v1';

const API_BASE = import.meta.env.VITE_API_BASE_URL || defaultApiBase;
```

---

## 🔗 Backend API Connectivity

### Connected Endpoints

The frontend connects to these backend routes:

| Backend Route | File | Frontend Service |
|---------------|------|------------------|
| `/api/v1/portfolio/*` | `app/api/routes/portfolio.py` | `portfolioApi` |
| `/api/v1/market/*` | `app/api/routes/market.py` | `marketApi` |
| `/api/v1/orders/*` | `app/api/routes/orders.py` | `ordersApi` |
| `/api/v1/risk/*` | `app/api/routes/risk.py` | `riskApi` |
| `/api/v1/strategy/*` | `app/api/routes/strategy.py` | `strategyApi` |
| `/api/v1/news/*` | `app/api/routes/news.py` | `newsApi` |
| `/ws/prices` | `app/api/routes/ws.py` | `useMarketData` |

---

## 🎯 Components Analysis

### Critical Components

| Component | Size | Purpose | Status |
|-----------|------|---------|--------|
| Dashboard.tsx | 27KB | Main dashboard with tabs | ✅ Active |
| Portfolio.tsx | 21KB | Portfolio management | ✅ Active |
| Market.tsx | 16KB | Market view + charts | ✅ Active |
| Orders.tsx | 12KB | Order history | ✅ Active |
| CandlestickChart.tsx | 8KB | Trading charts | ✅ Active |
| OrderBook.tsx | 5KB | Order book display | ✅ Active |
| Layout.tsx | 7KB | Navigation + sidebar | ✅ Active |

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Components                          │
├─────────────────────────────────────────────────────────────────┤
│  Dashboard ◄──► Portfolio ◄──► Market ◄──► Orders ◄──► Risk  │
│       │            │            │           │           │      │
│       └────────────┴────────────┴───────────┴───────────┘      │
│                            │                                    │
│                            ▼                                    │
│              ┌─────────────────────────┐                       │
│              │   TanStack Query Cache  │                       │
│              └───────────┬─────────────┘                       │
│                          │                                      │
│              ┌───────────┴─────────────┐                       │
│              │     services/api.ts     │                       │
│              │   (Axios + Interceptor)│                       │
│              └───────────┬─────────────┘                       │
│                          │                                      │
│         ┌────────────────┼────────────────┐                   │
│         │                │                │                    │
│         ▼                ▼                ▼                    │
│   ┌──────────┐   ┌───────────┐   ┌────────────┐              │
│   │ REST API │   │  WebSocket │   │  Backend   │              │
│   │ /api/v1 │   │  /ws/prices│   │  (FastAPI) │              │
│   └──────────┘   └───────────┘   └────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Current Data Aggregation

### Dashboard Data Sources

The main Dashboard aggregates data from multiple APIs:

```typescript
// From Dashboard.tsx
const { data: dualSummary } = useQuery({
  queryKey: ['portfolio-dual-summary'],
  queryFn: portfolioApi.getDualSummary,
  refetchInterval: 30000,
});

const { data: history } = useQuery({
  queryKey: ['portfolio-history'],
  queryFn: () => portfolioApi.getHistory(30),
  refetchInterval: 60000,
});

const { data: markets } = useQuery({
  queryKey: ['market-prices'],
  queryFn: marketApi.getAllPrices,
  refetchInterval: 15000,
});

const { data: performance } = useQuery({
  queryKey: ['portfolio-performance'],
  queryFn: portfolioApi.getPerformance,
  refetchInterval: 60000,
});

const { data: orders } = useQuery({
  queryKey: ['orders'],
  queryFn: () => ordersApi.list(),
  refetchInterval: 10000,
});

const { data: positions } = useQuery({
  queryKey: ['portfolio-positions'],
  queryFn: () => portfolioApi.getPositions(),
  refetchInterval: 30000,
});
```

---

## 🎯 Issues Identified

### 1. **Monolithic Dashboard**
- Dashboard.tsx is 27KB with multiple tabs
- All data fetching happens in a single component
- Hard to maintain and test

### 2. **No Central State Management**
- Using TanStack Query for server state
- No client-side global state (Context API unused)
- Each component fetches its own data

### 3. **Duplicated Types**
- `types.ts` and `types/index.ts` contain overlapping definitions
- Some types defined in both api.ts and types/

### 4. **Limited WebSocket Usage**
- Only useMarketData uses WebSocket
- Other real-time features (orders, positions) use polling

### 5. **No Error Handling Centralization**
- Error boundaries exist but not consistently used
- Each component handles errors independently

---

## 🛠️ Unification Plan

### Step 1: Create Unified Service Interface

Create a new unified service layer that combines all APIs:

```typescript
// frontend/src/services/unifiedApi.ts

import { portfolioApi, marketApi, ordersApi, riskApi, strategyApi, newsApi } from './api';

export class UnifiedTradingService {
  // Portfolio operations
  async getPortfolioSummary() {
    const [summary, positions, performance] = await Promise.all([
      portfolioApi.getSummary(),
      portfolioApi.getPositions(),
      portfolioApi.getPerformance()
    ]);
    return { summary, positions, performance };
  }

  // Market data
  async getMarketOverview() {
    const [prices, sentiment] = await Promise.all([
      marketApi.getAllPrices(),
      marketApi.getSentiment()
    ]);
    return { prices, sentiment };
  }

  // Orders and positions
  async getTradingState() {
    const [orders, positions] = await Promise.all([
      ordersApi.list(),
      portfolioApi.getPositions()
    ]);
    return { orders, positions };
  }

  // Risk metrics
  async getRiskAnalysis() {
    const [metrics, limits, positionRisks] = await Promise.all([
      riskApi.getMetrics(),
      riskApi.getLimits(),
      riskApi.getPositionRisks()
    ]);
    return { metrics, limits, positionRisks };
  }
}

export const tradingService = new UnifiedTradingService();
```

### Step 2: Create Dashboard Context

```typescript
// frontend/src/context/DashboardContext.tsx
import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { tradingService } from '../services/unifiedApi';

interface DashboardState {
  portfolio: any;
  market: any;
  orders: any;
  risk: any;
  isLoading: boolean;
  error: Error | null;
  lastUpdated: Date | null;
}

const DashboardContext = createContext<DashboardState | undefined>(undefined);

export function DashboardProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<DashboardState>({
    portfolio: null,
    market: null,
    orders: null,
    risk: null,
    isLoading: true,
    error: null,
    lastUpdated: null
  });

  // Fetch all data in parallel
  const { data, isLoading, error } = useQuery({
    queryKey: ['dashboard-unified'],
    queryFn: async () => {
      const [portfolio, market, orders, risk] = await Promise.all([
        tradingService.getPortfolioSummary(),
        tradingService.getMarketOverview(),
        tradingService.getTradingState(),
        tradingService.getRiskAnalysis()
      ]);
      return { portfolio, market, orders, risk };
    },
    refetchInterval: 30000 // Refresh every 30 seconds
  });

  useEffect(() => {
    if (data) {
      setState({
        portfolio: data.portfolio,
        market: data.market,
        orders: data.orders,
        risk: data.risk,
        isLoading: false,
        error: null,
        lastUpdated: new Date()
      });
    }
  }, [data]);

  return (
    <DashboardContext.Provider value={state}>
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard() {
  const context = useContext(DashboardContext);
  if (!context) {
    throw new Error('useDashboard must be used within DashboardProvider');
  }
  return context;
}
```

### Step 3: Modular Dashboard Components

Create a modular dashboard structure:

```
frontend/src/
├── components/
│   └── dashboard/
│       ├── DashboardHeader.tsx      # Summary cards
│       ├── PortfolioWidget.tsx      # Portfolio overview
│       ├── MarketWidget.tsx         # Market prices
│       ├── OrdersWidget.tsx         # Recent orders
│       ├── RiskWidget.tsx           # Risk metrics
│       ├── PerformanceChart.tsx     # Charts
│       └── UnifiedDashboard.tsx    # Main container
├── context/
│   └── DashboardContext.tsx         # Global state
└── services/
    └── unifiedApi.ts                # Unified service
```

### Step 4: Implement Widget Components

```typescript
// frontend/src/components/dashboard/DashboardHeader.tsx
import { useDashboard } from '../../context/DashboardContext';
import { TrendingUp, TrendingDown, DollarSign, Wallet } from 'lucide-react';

export function DashboardHeader() {
  const { portfolio, isLoading, error } = useDashboard();

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  const { summary } = portfolio;
  const isPositive = summary.daily_return_pct >= 0;

  return (
    <div className="grid grid-cols-4 gap-4">
      <div className="card">
        <div className="flex items-center gap-2">
          <DollarSign className="w-5 h-5" />
          <span>Total Value</span>
        </div>
        <div className="text-2xl font-bold">
          ${summary.total_value.toLocaleString()}
        </div>
      </div>
      
      <div className="card">
        <div className="flex items-center gap-2">
          {isPositive ? <TrendingUp /> : <TrendingDown />}
          <span>Daily P/L</span>
        </div>
        <div className={`text-2xl font-bold ${isPositive ? 'text-green' : 'text-red'}`}>
          {isPositive ? '+' : ''}{summary.daily_pnl.toLocaleString()}
        </div>
      </div>
      {/* More cards... */}
    </div>
  );
}
```

---

## 🧪 Testing Checklist

| Module | Test | Status |
|--------|------|--------|
| API Service | Unit tests for each API method | ⏳ Pending |
| Unified Service | Integration tests | ⏳ Pending |
| Context Provider | Context rendering test | ⏳ Pending |
| Widget Components | Component tests | ⏳ Pending |
| Dashboard Integration | E2E test | ⏳ Pending |
| WebSocket Connection | Connection test | ⏳ Pending |
| Error Handling | Error boundary test | ⏳ Pending |

---

## 📈 Implementation Roadmap

### Phase 1: Foundation (Day 1-2)
- [ ] Create unified API service
- [ ] Define TypeScript interfaces
- [ ] Set up Dashboard Context

### Phase 2: Component Development (Day 3-5)
- [ ] Create widget components
- [ ] Implement data flow
- [ ] Add error handling

### Phase 3: Integration (Day 6-7)
- [ ] Integrate with existing pages
- [ ] Test WebSocket connection
- [ ] Verify all API endpoints

### Phase 4: Optimization (Day 8-10)
- [ ] Performance optimization
- [ ] Caching strategy
- [ ] Documentation

---

## 🔗 Backend Connection Verification

To verify backend connectivity:

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Check market data
curl http://localhost:8000/api/v1/market/prices

# Check portfolio
curl http://localhost:8000/api/v1/portfolio/summary
```

---

## ✅ Conclusion

The current frontend architecture is well-structured with a unified API service. The main improvement opportunity is:

1. **Modularize the Dashboard** - Break down the 27KB Dashboard.tsx into smaller widgets
2. **Add global state** - Use React Context for shared data
3. **Enhance real-time** - Extend WebSocket usage beyond market data
4. **Centralize types** - Remove duplication between type files

The step-by-step guide above provides a clear path to achieve a more scalable and maintainable frontend.

---

*Document generated for AI Trading System v2.3*
