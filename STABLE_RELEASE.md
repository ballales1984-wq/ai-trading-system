# AI Trading System - Stable Release v1.0.0

## 📦 Versione Stabile per Produzione

Questa è la **versione stabile** del sistema AI Trading, configurata per rispettare i limiti di risorse:

| Risorsa | Limite | Allocazione |
|---------|--------|-------------|
| **RAM** | 4 GB | Trading (2GB) + DB (1GB) + Redis (512MB) + Dashboard (512MB) |
| **ROM** | 3 GB | DB (1.5GB) + Logs (500MB) + ML Temp (300MB) + Redis (300MB) + Models (200MB) + Cache (200MB) |

---

## 🚀 Avvio Rapido

### Windows
```batch
# Avvia tutti i servizi
start_stable.bat

# Oppure con rebuild
start_stable.bat build
```

### Linux/macOS
```bash
# Avvia tutti i servizi
./start_stable.sh

# Oppure con rebuild
./start_stable.sh --build
```

---

## 📋 Prerequisiti

1. **Docker Engine** 20.10+
2. **Docker Compose** 2.0+
3. **File .env** con le API keys configurate

### Creare il file .env
```bash
cp .env.example .env
# Modifica .env con le tue API keys
```

---

## 🏗️ Struttura dei File

```
ai-trading-system/
├── docker-compose.stable.yml    # Configurazione Docker principale
├── docker/Dockerfile.stable     # Dockerfile ottimizzato per produzione
├── requirements.stable.txt      # Dipendenze Python pinned
├── scripts/resource_monitor.py  # Monitor RAM/ROM
├── start_stable.bat             # Script avvio Windows
├── start_stable.sh              # Script avvio Linux/macOS
└── data/                        # Directory dati (creata automaticamente)
    ├── pgdata/                  # Database PostgreSQL
    ├── redisdata/               # Cache Redis
    ├── ml_temp/                 # File temporanei ML
    ├── models/                  # Modelli ML salvati
    ├── logs/                    # Log applicazione
    └── cache/                   # Cache applicazione
```

---

## 🔧 Servizi

| Servizio | Porta | RAM | ROM | Descrizione |
|----------|-------|-----|-----|-------------|
| **trading-system** | 8000 | 2 GB | 500 MB | Engine principale + ML |
| **dashboard** | 8050 | 512 MB | 200 MB | Dashboard web |
| **postgres** | 5432 | 1 GB | 1.5 GB | Database TimescaleDB |
| **redis** | 6379 | 512 MB | 300 MB | Cache |

---

## 📊 Monitoraggio Risorse

### Controllo Manuale
```bash
# All'interno del container
python /app/scripts/resource_monitor.py

# Output JSON
python /app/scripts/resource_monitor.py --json

# Monitoraggio continuo
python /app/scripts/resource_monitor.py --watch --interval 60
```

### Con Docker
```bash
# Status dei container
docker compose -f docker-compose.stable.ps

# Utilizzo risorse
docker stats --no-stream
```

---

## 🔒 Versioni Bloccate

### Python
- **Versione**: 3.11.5 (slim-bookworm)
- Non cambiare questa versione per garantire riproducibilità

### Dipendenze Principali
```
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
torch==2.1.2+cpu
dash==2.14.2
fastapi==0.108.0
redis==5.0.1
SQLAlchemy==2.0.23
```

Vedi [`requirements.stable.txt`](requirements.stable.txt) per la lista completa.

---

## 🛠️ Comandi Utili

### Gestione Servizi
```bash
# Avvia servizi
docker compose -f docker-compose.stable.yml up -d

# Ferma servizi
docker compose -f docker-compose.stable.yml down

# Riavvia un servizio
docker compose -f docker-compose.stable.yml restart trading-system

# Vedi log
docker compose -f docker-compose.stable.yml logs -f trading-system
```

### Build
```bash
# Build immagine
docker compose -f docker-compose.stable.yml build

# Build senza cache
docker compose -f docker-compose.stable.yml build --no-cache
```

### Pulizia
```bash
# Rimuovi container e volumi
docker compose -f docker-compose.stable.yml down -v

# Rimuovi anche immagini
docker compose -f docker-compose.stable.yml down -v --rmi all
```

---

## 📈 Endpoint

Una volta avviati i servizi:

| Endpoint | URL | Descrizione |
|----------|-----|-------------|
| Dashboard | http://localhost:8050 | Interfaccia web |
| API Health | http://localhost:8000/health | Health check |
| API Docs | http://localhost:8000/docs | Documentazione API |
| Prometheus | http://localhost:9090 | Metriche (se abilitato) |

---

## ⚠️ Troubleshooting

### Container non parte
```bash
# Controlla log
docker compose -f docker-compose.stable.yml logs trading-system

# Verifica risorse disponibili
docker system df
```

### Errore memoria
```bash
# Aumenta memoria Docker Desktop
# Settings > Resources > Memory: almeno 6GB
```

### Database non connette
```bash
# Verifica che postgres sia healthy
docker compose -f docker-compose.stable.yml ps

# Controlla log postgres
docker compose -f docker-compose.stable.yml logs postgres
```

---

## 🔄 Aggiornamenti

Per aggiornare alla nuova versione stabile:

1. **Backup dati**
   ```bash
   # Backup database
   docker exec ai_trading_db pg_dump -U trading trading_db > backup.sql
   ```

2. **Pull nuova versione**
   ```bash
   git pull origin main
   ```

3. **Rebuild e restart**
   ```bash
   docker compose -f docker-compose.stable.yml build --no-cache
   docker compose -f docker-compose.stable.yml up -d
   ```

---

## 📝 Note Importanti

1. **Non cambiare le versioni** delle dipendenze senza test approfonditi
2. **Monitora sempre** l'uso di RAM/ROM prima di aumentare batch size per ML
3. **Backup regolari** del database (almeno giornalieri)
4. **Log rotation** è configurata automaticamente (max 50MB x 3 file)

---

## 📞 Supporto

Per problemi o domande:
- Controlla i log: `docker compose -f docker-compose.stable.yml logs -f`
- Verifica risorse: `python scripts/resource_monitor.py`
- Consulta la documentazione: `docs/`

---

**Versione**: 1.0.0-stable  
**Data rilascio**: 2026-02-20  
**Python**: 3.11.5  
**Docker Compose**: 3.9
