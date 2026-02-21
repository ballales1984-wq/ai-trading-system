# Guida al Sistema di Routing - AI Trading System

Questo documento spiega come funziona il sistema di indirizzamento delle pagine e delle API nel progetto.

---

## Architettura Generale

L'applicazione utilizza un'architettura **client-server** con:

- **Frontend**: React + Vite (porta 5173)
- **Backend**: FastAPI (porta 8000)
- **Comunicazione**: Il frontend comunica con il backend tramite proxy

```
┌─────────────────────┐          ┌─────────────────────┐
│   Frontend (React)  │─────────▶│   Backend (FastAPI) │
│   Porta: 5173       │  proxy    │   Porta: 8000       │
└─────────────────────┘           └─────────────────────┘
```

---

## Routing Frontend (React Router)

### Configurazione Principale

Il routing frontend si trova in: `frontend/src/App.tsx`

```
tsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import Portfolio from './pages/Portfolio';
import Market from './pages/Market';
import Orders from './pages/Orders';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="portfolio" element={<Portfolio />} />
          <Route path="market" element={<Market />} />
          <Route path="orders" element={<Orders />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
```

### Tabella delle Pagine Frontend

| Path URL | Pagina | Descrizione |
|----------|--------|-------------|
| `/` | Layout (redirect a `/dashboard`) | Pagina principale con sidebar |
| `/dashboard` | Dashboard | Panoramica del sistema e statistiche |
| `/portfolio` | Portfolio | Posizioni e performance del portafoglio |
| `/market` | Market | Dati di mercato in tempo reale |
| `/orders` | Orders | Storico e gestione ordini |

### Navigazione

La navigazione è gestita dal componente `Layout` (`frontend/src/components/layout/Layout.tsx`):

```
tsx
import { Outlet, NavLink } from 'react-router-dom';

const navItems = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/portfolio', icon: PieChart, label: 'Portfolio' },
  { to: '/market', icon: TrendingUp, label: 'Market' },
  { to: '/orders', icon: ClipboardList, label: 'Orders' },
];
```

### Configurazione Vite (Proxy)

File: `frontend/vite.config.ts`

```
ts
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

**Come funziona il proxy**:
- Quando il frontend chiama `/api/v1/...`, Vite intercetta la richiesta
- La inoltra al backend su `http://localhost:8000`
- Risponde al frontend come se fosse la stessa origine

---

## Routing Backend (FastAPI)

### Configurazione Principale

Il routing backend si trova in: `app/main.py`

```
python
from app.api.routes import health, orders, portfolio, strategy, risk, market, waitlist

# Include routers
app.include_router(
    health.router,
    prefix=settings.api_prefix,  # = "/api/v1"
    tags=["Health"]
)

app.include_router(
    orders.router,
    prefix=f"{settings.api_prefix}/orders",  # = "/api/v1/orders"
    tags=["Orders"]
)

app.include_router(
    portfolio.router,
    prefix=f"{settings.api_prefix}/portfolio",  # = "/api/v1/portfolio"
    tags=["Portfolio"]
)

# ... altri router
```

### Tabella delle API Backend

| Endpoint | Descrizione |
|----------|-------------|
| `GET /` | Pagina principale (landing) |
| `GET /health` | Health check del sistema |
| `GET /docs` | Documentazione Swagger UI |
| `GET /redoc` | Documentazione ReDoc |
| `GET /openapi.json` | Schema OpenAPI JSON |
| `GET /api/v1/orders/*` | API per la gestione ordini |
| `GET /api/v1/portfolio/*` | API per il portafoglio |
| `GET /api/v1/market/*` | API per i dati di mercato |
| `GET /api/v1/strategy/*` | API per le strategie |
| `GET /api/v1/risk/*` | API per la gestione del rischio |
| `GET /landing/*` | Pagine statiche landing |

### Struttura delle API Routes

Le route API sono organizzate in moduli:

```
app/api/routes/
├── __init__.py
├── health.py      # Endpoint salute sistema
├── orders.py      # Gestione ordini
├── portfolio.py   # Portfolio e posizioni
├── market.py      # Dati di mercato
├── strategy.py    # Strategie di trading
├── risk.py        # Gestione rischi
└── waitlist.py    # Waitlist utenti
```

### Esempio di Route

Esempio da `app/api/routes/orders.py`:

```
python
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_orders():
    """Ottieni tutti gli ordini"""
    return {"orders": []}

@router.post("/")
async def create_order(order: OrderCreate):
    """Crea un nuovo ordine"""
    return {"order_id": "123", "status": "pending"}
```

---

## Flusso di Comunicazione

### 1. richiesta API dal Frontend

```
1. Frontend chiama: fetch('/api/v1/portfolio')
                    ↓
2. Vite proxy intercetta: /api/v1/portfolio
                    ↓
3. Inoltra a: http://localhost:8000/api/v1/portfolio
                    ↓
4. FastAPI riceve la richiesta
                    ↓
5. Elabora e restituisce JSON
                    ↓
6. Frontend riceve la risposta
```

### 2. Configurazione URL Base

Il frontend configura l'URL base in `frontend/src/services/api.ts`:

```
ts
const API_BASE_URL = '/api/v1';

// Esempio di chiamata
export const fetchPortfolio = () => 
  fetch(`${API_BASE_URL}/portfolio`);
```

---

## Come Aggiungere una Nuova Pagina

### 1. Creare la pagina React

Creare file in `frontend/src/pages/NuovaPagina.tsx`:

```
tsx
export default function NuovaPagina() {
  return <div>Nuova Pagina</div>;
}
```

### 2. Aggiungere la route in App.tsx

```
tsx
import NuovaPagina from './pages/NuovaPagina';

// In App():
<Route path="nuovapagina" element={<NuovaPagina />} />
```

### 3. Aggiungere al menu di navigazione

In `frontend/src/components/layout/Layout.tsx`:

```
tsx
const navItems = [
  // ... items esistenti
  { to: '/nuovapagina', icon: NuovoIcon, label: 'Nuova Pagina' },
];
```

### 4. (Opzionale) Creare l'endpoint API

Creare file in `app/api/routes/nuovapagina.py`:

```
python
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_nuovapagina():
    return {"message": "Dati della nuova pagina"}
```

Registrare in `app/main.py`:

```python
app.include_router(
    nuovapagina.router,
    prefix=f"{settings.api_prefix}/nuovapagina",
    tags=["NuovaPagina"]
)
```

---

## Come Aggiungere una Nuova API

### 1. Creare il file route

In `app/api/routes/nome_route.py`:

```
python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class ItemCreate(BaseModel):
    name: str
    value: float

@router.get("/")
async def get_items():
    """Ottieni lista items"""
    return {"items": []}

@router.post("/")
async def create_item(item: ItemCreate):
    """Crea nuovo item"""
    return {"id": 1, **item.dict()}
```

### 2. Registrare la route

In `app/main.py`:

```
python
from app.api.routes import nome_route

app.include_router(
    nome_route.router,
    prefix=f"{settings.api_prefix}/nome_route",
    tags=["Nome Route"]
)
```

### 3. Testare l'endpoint

Avviare il server:
```
bash
python -m uvicorn app.main:app --reload
```

Visitare: http://localhost:8000/docs

---

## Variabili d'Ambiente

Le configurazioni delle route sono gestite in:

- **Frontend**: `frontend/vite.config.ts` (porta)
- **Backend**: `app/core/config.py` (prefix API)

Configurazione tipica del backend (`app/core/config.py`):

```
python
class Settings(BaseSettings):
    api_prefix: str = "/api/v1"
    cors_origins: list = ["http://localhost:5173"]
    # ...
```

---

## Risoluzione Problemi

### Frontend non si connette al backend

1. Verificare che il backend sia avviato: `python -m uvicorn app.main:app`
2. Controllare che il proxy in `vite.config.ts` punti a `http://localhost:8000`
3. Verificare che CORS sia configurato correttamente

### Errore 404 sulle API

1. Controllare che l'endpoint esista in `app/api/routes/`
2. Verificare che il router sia registrato in `app/main.py`
3. Controllare il prefix: deve essere `/api/v1/...`

### La pagina non viene trovata

1. Verificare che la route sia definita in `App.tsx`
2. Controllare che il componente sia importato correttamente
3. Verificare che il path sia corretto (senza slash iniziale)

---

## Link Utili

- Documentazione React Router: https://reactrouter.com/
- Documentazione FastAPI: https://fastapi.tiangolo.com/
- Swagger UI (API docs): http://localhost:8000/docs
- ReDoc (API docs alternative): http://localhost:8000/redoc

---

## Summary

| Componente | File | Porta |
|------------|------|-------|
| Frontend Dev Server | `vite.config.ts` | 5173 |
| Backend API | `app/main.py` | 8000 |
| API Prefix | `app/core/config.py` | `/api/v1` |
| Landing Page | `landing/index.html` | `/` |
