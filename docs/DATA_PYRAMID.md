# Piramide dei Dati e Previsioni

## Livello 1 — Dati Grezzi & Open (Base)

### Fonti
- API open (Open-Meteo, CoinGecko, etc.)
- Dati storici CSV
- Webcam pubbliche
- Sensori meteo
- Dati traffico
- Eventi locali

### Tipologia Dati
- Temperatura
- Precipitazioni
- Traffico veicolare
- Affluenza
- Indicatori economici generali

### Funzione
Raccolta grezza dei dati senza elaborazioni

### Output
Dati pronti per aggregazione

---

## Livello 2 — Moduli Macro Aggregati

### Tipologia
- Dati aggregati giornalieri
- Dati aggregati settimanali
- Dati aggregati mensili

### Parametri
- Stagioni (inverno, primavera, estate, autunno)
- Giorno/Notte
- Festività
- Weekend

### Funzione
Trasformazione dei dati grezzi in segnali compositi e pattern

### Output
- Segnali mediati
- Indicatori macro

---

## Livello 3 — Dati Professionali / API Premium

### Fonti
- API a pagamento
- Dati finanziari ad alta frequenza
- Indicatori macroeconomici dettagliati

### Tipologia
- Prezzi materie prime
- Consumo energetico
- Flussi di capitale
- Criptovalute

### Funzione
Integrazione dati precisi e aggiornati per raffinare previsioni

### Output
Segnali ad alta accuratezza e frequenza

---

## Livello 4 — Singolarità / Previsioni / Guadagno (Apex)

### Funzione
Combinazione dei segnali mediati e dei dati professionali

### Algoritmi
- Monte Carlo (5 livelli)
- Machine Learning (XGBoost, LightGBM)
- Hidden Markov Models (HMM)
- Regressioni
- Pattern Recognition

### Output
- Decisioni strategiche BUY/SELL/HOLD
- Previsione consumi
- Movimenti di mercato
- Piani di guadagno

### Feedback
Aggiornamento pesi dei segnali e parametri dei moduli inferiori

---

## Flusso Logico

```
1. Raccolta dati grezzi
        ↓
2. Aggregazione macro e segnali mediati
        ↓
3. Integrazione dati professionali
        ↓
4. Elaborazione previsioni e strategie
        ↓
5. Azione / output decisionale
        ↓
6. Feedback per affinamento continuo
        ↓
        (torna al punto 1)
```

---

## Mappa Operativa Dettagliata

### Livello 1: Dati Grezzi

| Sotto-modulo | Fonte | Frequenza | Normalizzazione |
|-------------|-------|------------|----------------|
| Meteo | Open-Meteo | Oraria | -30°C to +50°C → 0-1 |
| Cripto | CoinGecko | Minutaria | Volume 0-1 |
| Traffico | TomTom/Google | Oraria | Indice 0-1 |
| Eventi | Eventbrite | Giornaliera | Count 0-1 |
| Criminalità | FBI/OpenData | Mensile | Indice 0-1 |
| Energia | EIA | Giornaliera | $/MWh → 0-1 |

### Livello 2: Moduli Aggregati

| Modulo | Input | Output | Peso Default |
|--------|-------|--------|--------------|
| Giorno/Notte | Ora | Factor 0.8-1.2 | 15% |
| Settimana | Giorno | Factor 0.9-1.4 | 10% |
| Stagione | Mese | Factor 0.9-1.2 | 10% |
| Anno | Data | Factor 1.0-1.5 | 5% |
| Meteo Aggregato | Dati meteo | Index 0-1 | 20% |
| Economico | PIL + Energia | Index 0-1 | 15% |
| Sociale | Traffico + Eventi | Index 0-1 | 10% |
| Criminalità | Indice crime | Index 0-1 | 5% |
| Cripto | Volume crypto | Index 0-1 | 10% |

### Livello 3: Integrazione Professionale

| Tipo Dato | Fonte | Uso | Impatto |
|-----------|-------|-----|---------|
| Prezzi Commodities | Quandl/Nasdaq | Previsione inflazione | Alto |
| Flussi Capitale | Bloomberg | Analisi macro | Alto |
| Dati On-chain | Blockchain | Sentiment crypto | Medio |
| News Sentiment | NewsAPI/GDELT | Bias emotivo | Medio |

### Livello 4: Motore Predittivo

| Algoritmo | Input | Output | Frequenza |
|-----------|-------|--------|-----------|
| Monte Carlo 5-Livelli | Segnali aggregati | Distribuzione probabilistica | On-demand |
| HMM Regime Detection | Serie storiche | Regime corrente | Giornaliera |
| ML Ensemble | Feature vector | Probabilità prezzo | Oraria |
| Sentiment Analysis | News/Social | Bias -1 to +1 | Oraria |

---

## Esempio di Calcolo Segnale Composito

```python
# Pseudocode per calcolo finale

segnale_finale = (
    time_module * 0.30 +        # 30% fattore temporale
    weather_module * 0.20 +      # 20% meteo
    economic_module * 0.15 +     # 15% economico
    crypto_module * 0.15 +      # 15% crypto
    social_module * 0.10 +       # 10% sociale
    crime_module * 0.10           # 10% criminalità
)

# Output: indice 0-2 (dove 1 = normale)
# > 1.2 = alta attività / BUY
# < 0.8 = bassa attività / SELL
# 0.8-1.2 = neutrale / HOLD
```

---

## Aggiornamento e Feedback Loop

```python
# Ogni fine giornata:
1. Calcola accuratezza previsione vs risultato reale
2. Aggiusta pesi dei moduli (gradient descent)
3. Aggiorna pesi algoritmi ML
4. Ricalcola distribuzioni Monte Carlo
5. Nuovo ciclo con parametri affinati
```

---

## Scalabilità

- **Orizzontale**: Aggiungere nuove città senza modificare codice
- **Verticale**: Aggiungere nuovi moduli (es. maree, immagini satellite)
- **Temporale**: Backup storico per training ML
- **Modulare**: Ogni modulo indipendente dagli altri

---

*Questo documento definisce l'architettura completa del sistema di dati e previsioni.*
*Ultimo aggiornamento: Febbraio 2026*

