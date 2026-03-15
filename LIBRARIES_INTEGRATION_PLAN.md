# Piano di Integrazione Librerie Finance/Trading - VERSIONE SICURA

## ⚠️ PROBLEMA CONFLITTI

Installare nuove librerie può causare conflitti con le dipendenze esistenti:
- numpy, pandas, scipy già installati
- TA-Lib già in requirements.stable.txt
- Potenziali conflitti di versione

## ✅ SOLUZIONE: Estendere il codice esistente

Invece di installare nuove librerie, possiamo **estendere** i moduli già presenti:

---

## 📋 FUNZIONALITÀ AGGIUNTIVE SENZA NUOVE DIPENDENZE

### 1. Estendere [`technical_analysis.py`](technical_analysis.py)

Aggiungere indicatori avanzati al codice esistente:

```python
# NUOVI METODI DA AGGIUNGERE:

def calculate_keltner_channels(df, period=20, multiplier=2):
    """Keltner Channels"""
    ema = df['close'].ewm(span=period).mean()
    atr = calculate_atr(df, period)
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    return upper, ema, lower

def calculate_donchian_channels(df, period=20):
    """Donchian Channels"""
    upper = df['high'].rolling(period).max()
    lower = df['low'].rolling(period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower

def calculate_vwap(df):
    """Volume Weighted Average Price"""
    df = df.copy()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
    return df['vwap']

def calculate_ichimoku(df):
    """Ichimoku Cloud - full implementation"""
    nine_period = 9
    twenty_six_period = 26
    fifty_two_period = 52
    
    # Tenkan-sen (Conversion Line)
    tenkan = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    
    # Kijun-sen (Base Line)
    kijun = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B)
    senkou_b = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    
    # Chikou Span (Lagging Span)
    chikou = df['close'].shift(-26)
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def calculate_obv(df):
    """On Balance Volume"""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv

def calculate_mfi(df, period=14):
    """Money Flow Index"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi
```

---

### 2. Estendere [`app/portfolio/performance.py`](app/portfolio/performance.py)

Aggiungere metriche avanzate senza nuove dipendenze:

```python
# NUOVE FUNZIONI:

def calculate_sortino_ratio(returns, target_return=0):
    """Calculate Sortino Ratio"""
    downside_returns = returns[returns < target_return]
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0
    return (returns.mean() - target_return) / downside_std

def calculate_calmar_ratio(returns, max_drawdown):
    """Calculate Calmar Ratio"""
    annual_return = returns.mean() * 252
    if max_drawdown == 0:
        return 0
    return annual_return / abs(max_drawdown)

def calculate_ulcer_index(prices):
    """Calculate Ulcer Index (risk metric)"""
    daily_returns = prices.pct_change()
    drawdown = (prices / prices.cummax() - 1) * 100
    ulcer = np.sqrt((drawdown ** 2).mean())
    return ulcer

def calculate_value_at_risk(returns, confidence=0.95):
    """Calculate VaR using historical method"""
    return np.percentile(returns, (1 - confidence) * 100)

def calculate_conditional_var(returns, confidence=0.95):
    """Calculate CVaR / Expected Shortfall"""
    var = calculate_value_at_risk(returns, confidence)
    return returns[returns <= var].mean()

def calculate_information_ratio(returns, benchmark_returns):
    """Calculate Information Ratio"""
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std()
    if tracking_error == 0:
        return 0
    return active_returns.mean() / tracking_error
```

---

### 3. Estendere [`src/allocation.py`](src/allocation.py)

Aggiungere strategie di allocazione avanzate:

```python
# NUOVE CLASSI:

class RiskParityAllocator:
    """Risk Parity allocation - equal risk contribution"""
    
    def allocate(self, returns_df, target_vol=0.15):
        cov_matrix = returns_df.cov()
        inv_cov = np.linalg.inv(cov_matrix.values)
        ones = np.ones(len(returns_df.columns))
        
        # Calculate risk parity weights
        risk_contrib = inv_cov @ ones
        weights = risk_contrib / risk_contrib.sum()
        
        return dict(zip(returns_df.columns, weights))

class MinimumVarianceAllocator:
    """Minimum Variance portfolio"""
    
    def allocate(self, returns_df):
        cov_matrix = returns_df.cov()
        
        # Optimize for minimum variance
        n_assets = len(returns_df.columns)
        result = minimize(
            lambda w: w @ cov_matrix.values @ w,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=[(0, 1) for _ in range(n_assets)],
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        
        return dict(zip(returns_df.columns, result.x))

class BlackLittermanAllocator:
    """Black-Litterman model implementation"""
    
    def __init__(self, market_cap_weights, views=None):
        self.market_cap_weights = market_cap_weights
        self.views = views or {}
    
    def allocate(self, returns_df, confidence=0.5):
        # Simplified Black-Litterman
        # In production, use full implementation
        cov_matrix = returns_df.cov()
        
        # Equilibrium market implied returns
        risk_aversion = 2.5  # Typical value
        implied_returns = risk_aversion * cov_matrix.values @ np.array(list(self.market_cap_weights.values()))
        
        # Blend with historical returns
        hist_returns = returns_df.mean()
        adjusted_returns = confidence * implied_returns + (1 - confidence) * hist_returns.values
        
        # Simple mean-variance optimization
        return self._optimize(adjusted_returns, cov_matrix.values)
    
    def _optimize(self, expected_returns, cov_matrix):
        n = len(expected_returns)
        result = minimize(
            lambda w: -w @ expected_returns + 0.5 * w @ cov_matrix @ w,
            np.ones(n) / n,
            method='SLSQP',
            bounds=[(0, 1) for _ in range(n)],
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        return dict(zip([f'Asset_{i}' for i in range(n)], result.x))
```

---

## 🎯 PIANO ALTERNATIVO SENZA CONFLITTI

### Opzione A: Estendere codice esistente (Consigliato)
- 0 nuove dipendenze
- 100% compatibile
- Richiede tempo di sviluppo

### Opzione B: Virtual Environment separato
```bash
# Creare ambiente virtuale dedicato
python -m venv trading_venv
source trading_venv/Scripts/activate
pip install pandas-ta pyportfolioopt
```

### Opzione C: Docker separato
```dockerfile
FROM python:3.11
RUN pip install pandas-ta pyportfolioopt
# Import from main app
```

---

## 📊 RIEPILOGO

| Approccio | Pro | Contro |
|-----------|-----|--------|
| **Estendere codice** | No conflitti, controllo totale | Più tempo |
| **Virtual env** | Isolato, funziona | Più complessità |
| **Nuove libs subito** | Subito pronto | Possibili conflitti |

**Raccomandazione: Iniziare con l'estensione del codice esistente**
