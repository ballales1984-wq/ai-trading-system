# AI Trading System - Documentazione

## Indice Documentazione

Questa cartella contiene la documentazione completa del sistema di trading.

---

## 📚 Documenti Disponibili

### API Reference
- **[API_REFERENCE.md](API_REFERENCE.md)** - Guida completa alle API esterne integrate
  - 22 API documentate
  - Configurazione environment
  - Rate limits e usage

### Architettura
- **[../ARCHITECTURE.md](../ARCHITECTURE.md)** - Architettura generale del sistema
- **[../ARCHITECTURE_INTEGRATION.md](../ARCHITECTURE_INTEGRATION.md)** - Integrazione componenti
- **[../API_INTEGRATION_ARCHITECTURE.md](../API_INTEGRATION_ARCHITECTURE.md)** - Architettura integrazione API

### Diagrammi
- **[../COMPONENT_DIAGRAM.md](../COMPONENT_DIAGRAM.md)** - Diagramma componenti
- **[../ECOSYSTEM_MAP.md](../ECOSYSTEM_MAP.md)** - Mappa ecosistema
- **[../API_FLOW_DIAGRAM.md](../API_FLOW_DIAGRAM.md)** - Diagramma flusso API

### Roadmap
- **[../ROADMAP.md](../ROADMAP.md)** - Roadmap progetto 100%

---

## 🚀 Quick Start

1. **Configura le API Key**
   ```bash
   cp .env.example .env
   # Modifica .env con le tue chiavi
   ```

2. **Avvia il Dashboard**
   ```bash
   python main.py --mode dashboard
   ```

3. **Esegui i Test**
   ```bash
   pytest tests/ -v
   ```

---

## 📁 Struttura Progetto

```
ai-trading-system/
├── docs/                    # Documentazione
│   ├── README.md           # Questo file
│   └── API_REFERENCE.md    # Riferimento API
├── src/                    # Codice sorgente
│   ├── external/           # Client API esterni
│   ├── core/               # Core engine
│   ├── live/               # Trading live
│   └── ml_*.py             # ML modules
├── app/                    # FastAPI backend
├── dashboard/              # Dashboard frontend
├── tests/                  # Test suite
└── main.py                 # Entry point
```

---

## 🔗 Link Utili

| Risorsa | Link |
|---------|------|
| README Principale | [../README.md](../README.md) |
| Requirements | [../requirements.txt](../requirements.txt) |
| Docker Compose | [../docker-compose.yml](../docker-compose.yml) |
| Pyproject | [../pyproject.toml](../pyproject.toml) |

---

*Ultimo aggiornamento: 2026-02-19*
