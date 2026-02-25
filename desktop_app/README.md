# AI Trading System - Desktop App

Applicazione desktop completa per utenti paganti. Include tutto il sistema in un unico file eseguibile.

## Caratteristiche

- Setup guidato: Inserimento API keys al primo avvio
- Backend integrato: FastAPI avviato automaticamente
- Interfaccia grafica: Tkinter (incluso in Python)
- Sicurezza: API keys salvate solo sul PC locale (mai online)
- Testnet/Mainnet: Toggle per passare da test a produzione

## Requisiti

Tkinter e gia incluso in Python. Per creare l'EXE:

```
bash
pip install pyinstaller
```

## Come usare

### 1. Avvio in modalita sviluppo
```
bash
cd desktop_app
python main_tkinter.py
```

### 2. Creare l'EXE (per distribuzione)
```
bash
cd desktop_app
python build_exe.py
```

L'EXE viene creato in `dist/AI_Trading_System_Pro.exe`

## Architettura

```
AI Trading System Pro (Desktop App Tkinter)
    |
    +-- SetupScreen (inserimento API keys)
    +-- MainScreen (controllo backend)
    +-- ConfigManager (salvataggio config locale)
```

## Sicurezza

- Le API keys sono salvate in `~/.ai_trading_config.json`
- File accessibile solo dall'utente
- Nessuna connessione a server esterni per le chiavi
- Il backend gira solo in localhost (127.0.0.1:8000)

## Flusso utente

1. Primo avvio: Inserisce API keys Binance
2. Scelta testnet: Toggle per testnet (consigliato) o mainnet
3. Avvio sistema: Clicca "Avvia Sistema" per far partire il backend
4. Uso: Apre il browser per la dashboard
5. Chiusura: "Ferma Sistema" o chiudi l'app

## Build per distribuzione

```
bash
# Installa dipendenze
pip install pyinstaller

# Crea EXE
python desktop_app/build_exe.py

# Trova l'EXE in dist/
```

## Troubleshooting

Problema: Errore import tkinter
Soluzione: Reinstalla Python e assicurati di includere tkinter

Problema: Backend non parte
Soluzione: Verifica che Python sia nel PATH e le dipendenze siano installate

## File creati

- `main_tkinter.py` - App Tkinter completa
- `build_exe.py` - Script per creare l'EXE
- `README.md` - Questo file
