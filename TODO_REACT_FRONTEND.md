# React Frontend Implementation Plan

## Information Gathered

### Current Backend Structure
- FastAPI application in `app/main.py`
- API routes at `app/api/routes/` (orders, portfolio, market, strategy, risk, health)
- FastAPI runs on port 8000 with `/api/v1` prefix
- Uses SQLAlchemy for database

### Current Dashboard
- Uses Dash (Python) in `dashboard/app.py`
- Has charts for prices, equity curves, RSI, MACD
- Shows portfolio metrics, ML results, hedge fund analytics
- Runs on port 8050

### API Endpoints Available
- `/api/v1/portfolio/summary` - Portfolio summary
- `/api/v1/portfolio/positions` - Open positions
- `/api/v1/market/price/{symbol}` - Price data
- `/api/v1/market/candles/{symbol}` - OHLCV data
- `/api/v1/orders` - Order management
- `/api/v1/market/prices` - All market prices
- `/api/v1/portfolio/performance` - Performance metrics

## Plan

### Phase 1: Project Setup
1. Create `frontend/` directory with Vite + React + TypeScript
2. Set up Tailwind CSS
3. Configure API client and environment variables

### Phase 2: Core Components
1. Create reusable UI components (Card, Button, Table, etc.)
2. Create layout components (Sidebar, Header)
3. Create charting components

### Phase 3: Pages
1. **Dashboard Page** - Main overview with metrics and charts
2. **Portfolio Page** - Positions, allocation, performance
3. **Market Page** - Price lists, individual charts
4. **Orders Page** - Order management interface
5. **Settings Page** - Configuration

### Phase 4: Integration
1. Connect to existing FastAPI backend
2. Add real-time updates
3. Add error handling and loading states

### Phase 5: Polish
1. Dark theme styling
2. Responsive design
3. Animations and transitions

## Technology Stack
- **Framework:** React 18 with TypeScript
- **Build Tool:** Vite
- **Styling:** Tailwind CSS
- **Data Fetching:** React Query (TanStack Query)
- **Charts:** Recharts
- **Routing:** React Router v6
- **Icons:** Lucide React

## Files to Create
```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/
│   │   ├── charts/
│   │   └── layout/
│   ├── pages/
│   ├── hooks/
│   ├── services/
│   ├── types/
│   ├── App.tsx
│   └── main.tsx
├── index.html
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── vite.config.ts
```

