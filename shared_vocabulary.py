"""
Shared Vocabulary Module
Consolidates financial concepts, definitions, and keywords for the AI Assistant and Concept Engine.
Optimized for Italian language support while maintaining English technical terms.
"""

from typing import Dict, List

# Unified Financial Concepts
SHARED_FINANCIAL_CONCEPTS: Dict[str, Dict] = {
    # Trading Basics
    "long_position": {
        "term": "Long Position",
        "definition": "Posizione rialzista che guadagna quando il prezzo sale (Acquisto)",
        "category": "trading",
        "keywords": ["buy", "long", "acquisto", "rialzo", "rialzista", "purchase"]
    },
    "short_position": {
        "term": "Short Position",
        "definition": "Posizione ribassista che guadagna quando il prezzo scende (Vendita)",
        "category": "trading",
        "keywords": ["sell", "short", "ribasso", "ribassista", "shortare", "shorting"]
    },
    "stop_loss": {
        "term": "Stop Loss",
        "definition": "Ordine automatico per limitare le perdite a un livello predefinito",
        "category": "risk",
        "keywords": ["stop", "stop loss", "protezione", "limite", "exit"]
    },
    "take_profit": {
        "term": "Take Profit",
        "definition": "Ordine automatico per chiudere in profitto a un target specifico",
        "category": "risk",
        "keywords": ["target", "profit", "take profit", "obiettivo", "chiusura"]
    },
    "leverage": {
        "term": "Leva (Leverage)",
        "definition": "Moltiplicatore che amplifica l'esposizione con meno capitale",
        "category": "trading",
        "keywords": ["leverage", "leva", "margine", "x10", "x100", "leveraged"]
    },
    "margin": {
        "term": "Margine",
        "definition": "Capitale richiesto come garanzia per mantenere una posizione aperta",
        "category": "trading",
        "keywords": ["margin", "collateral", "collaterale"]
    },
    
    # Risk Metrics
    "var": {
        "term": "VaR (Value at Risk)",
        "definition": "Massima perdita attesa in un intervallo di confidenza (es. 95%)",
        "category": "risk",
        "keywords": ["var", "value at risk", "rischio", "confidenza", "maximum loss"]
    },
    "cvar": {
        "term": "CVaR (Conditional VaR)",
        "definition": "Perdita attesa media quando si supera la soglia del VaR (Expected Shortfall)",
        "category": "risk",
        "keywords": ["cvar", "expected shortfall", "tail risk", "average loss", "tail"]
    },
    "drawdown": {
        "term": "Drawdown",
        "definition": "Calo percentuale dal picco al minimo locale del portafoglio",
        "category": "risk",
        "keywords": ["drawdown", "dd", "perdita", "calo", "peak to trough"]
    },
    "max_dd": {
        "term": "Max Drawdown",
        "definition": "Il massimo calo storico registrato dal portafoglio dal suo picco",
        "category": "risk",
        "keywords": ["maxdd", "max drawdown", "peak", "worst case"]
    },
    "sharpe_ratio": {
        "term": "Sharpe Ratio",
        "definition": "Ritorno aggiustato per il rischio (extra-rendimento / deviazione standard)",
        "category": "metrics",
        "keywords": ["sharpe", "risk-adjusted", "performance", "ratio"]
    },
    "sortino_ratio": {
        "term": "Sortino Ratio",
        "definition": "Variante dello Sharpe che considera solo la volatilità negativa",
        "category": "metrics",
        "keywords": ["sortino", "downside risk", "downside volatility"]
    },
    
    # Technical Analysis
    "rsi": {
        "term": "RSI (Relative Strength Index)",
        "definition": "Indicatore di momentum che misura velocità e cambiamento dei movimenti dei prezzi",
        "category": "technical",
        "keywords": ["rsi", "overbought", "oversold", "ipercomprato", "ipervenduto"]
    },
    "macd": {
        "term": "MACD",
        "definition": "Moving Average Convergence Divergence - indicatore di trend e momentum",
        "category": "technical",
        "keywords": ["macd", "trend", "moving average", "convergence", "divergence"]
    },
    "ema": {
        "term": "EMA (Media Mobile Esponenziale)",
        "definition": "Media mobile che dà più peso ai prezzi più recenti",
        "category": "technical",
        "keywords": ["ema", "media mobile", "exponential average"]
    },
    "sma": {
        "term": "SMA (Media Mobile Semplice)",
        "definition": "Media aritmetica dei prezzi su un periodo specifico",
        "category": "technical",
        "keywords": ["sma", "ma", "media mobile", "simple average"]
    },
    "bollinger": {
        "term": "Bande di Bollinger",
        "definition": "Bande di volatilità basate su media mobile e deviazione standard",
        "category": "technical",
        "keywords": ["bollinger", "bands", "volatility", "bands", "volatilità"]
    },
    "support": {
        "term": "Supporto",
        "definition": "Livello di prezzo dove la domanda tende ad assorbire l'offerta",
        "category": "technical",
        "keywords": ["support", "floor", "supporto", "minimo", "demand"]
    },
    "resistance": {
        "term": "Resistenza",
        "definition": "Livello di prezzo dove l'offerta tende a superare la domanda",
        "category": "technical",
        "keywords": ["resistance", "ceiling", "resistenza", "massimo", "supply"]
    },
    "breakout": {
        "term": "Breakout",
        "definition": "Rottura decisa di un livello di supporto o resistenza",
        "category": "technical",
        "keywords": ["breakout", "rottura", "rupture", "volatilità"]
    },
    "pullback": {
        "term": "Pullback",
        "definition": "Ritorno temporaneo del prezzo verso un livello appena rotto",
        "category": "technical",
        "keywords": ["pullback", "retrace", "ritracciamento"]
    },
    
    # Market Structure
    "volatility": {
        "term": "Volatilità",
        "definition": "Misura della dispersione dei rendimenti e delle oscillazioni dei prezzi",
        "category": "market",
        "keywords": ["volatility", "volatile", "oscillazione", "instabilita", "variance"]
    },
    "liquidity": {
        "term": "Liquidità",
        "definition": "Facilità con cui un asset può essere scambiato senza impattare il prezzo",
        "category": "market",
        "keywords": ["liquidity", "liquidità", "depth", "volume", "slippage"]
    },
    "spread": {
        "term": "Spread",
        "definition": "Differenza tra il prezzo di acquisto (bid) e di vendita (ask)",
        "category": "market",
        "keywords": ["spread", "bid", "ask", "denaro", "lettera", "differenziale"]
    },
    "volume": {
        "term": "Volume",
        "definition": "Quantità totale di un asset scambiata in un dato periodo",
        "category": "market",
        "keywords": ["volume", "traded", "scambi", "turnover"]
    },
    "market_cap": {
        "term": "Market Cap",
        "definition": "Valore totale di mercato (Prezzo * Circolante)",
        "category": "market",
        "keywords": ["market cap", "capitalizzazione", "valuation"]
    },
    
    # DeFi & Blockchain
    "staking": {
        "term": "Staking",
        "definition": "Blocco di asset per supportare la sicurezza della rete in cambio di premi",
        "category": "defi",
        "keywords": ["staking", "stake", "reward", "validator", "delegation"]
    },
    "liquidity_pool": {
        "term": "Liquidity Pool",
        "definition": "Riserva di token bloccata in uno smart contract per facilitare il trading",
        "category": "defi",
        "keywords": ["lp", "pool", "dex", "swap", "amm"]
    },
    "impermanent_loss": {
        "term": "Impermanent Loss",
        "definition": "Perdita temporanea subita dai fornitori di liquidità rispetto all'holding",
        "category": "defi",
        "keywords": ["il", "impermanent loss", "divergence loss"]
    },
    "yield_farming": {
        "term": "Yield Farming",
        "definition": "Strategia per massimizzare i rendimenti spostando asset tra protocolli DeFi",
        "category": "defi",
        "keywords": ["yield", "farming", "rendimento", "apy", "apr"]
    },
    "smart_contract": {
        "term": "Smart Contract",
        "definition": "Contratto auto-eseguibile con i termini scritti direttamente nel codice",
        "category": "crypto",
        "keywords": ["smart", "contract", "contratto", "automa", "solidity"]
    },
    "layer2": {
        "term": "Layer 2",
        "definition": "Protocolli costruiti sopra una blockchain (L1) per migliorarne la scalabilità",
        "category": "crypto",
        "keywords": ["l2", "scalability", "arbitrum", "optimism", "rollup", "zksync"]
    },
    
    # Economic
    "inflation": {
        "term": "Inflazione",
        "definition": "Aumento generalizzato del livello dei prezzi e perdita di potere d'acquisto",
        "category": "economic",
        "keywords": ["inflation", "inflazione", "cpi", "prezzi", "purchasing power"]
    },
    "interest_rate": {
        "term": "Tasso di Interesse",
        "definition": "Costo del denaro stabilito dalle banche centrali (FED, BCE)",
        "category": "economic",
        "keywords": ["interest", "fed", "rate", "tasso", "monetary policy"]
    },
    "recession": {
        "term": "Recessione",
        "definition": "Periodo di declino economico significativo, spesso definito da due trimestri di PIL negativo",
        "category": "economic",
        "keywords": ["recession", "recessione", "downturn", "contrazione"]
    },
    "quantitative_easing": {
        "term": "Quantitative Easing (QE)",
        "definition": "Politica monetaria espansiva in cui la banca centrale acquista titoli",
        "category": "economic",
        "keywords": ["qe", "printing", "liquidity injection", "monetary base"]
    },
    
    # Sentiment
    "bullish": {
        "term": "Bullish",
        "definition": "Aspettativa di mercato che i prezzi salgano (Rialzista)",
        "category": "sentiment",
        "keywords": ["bullish", "rialzista", "buy", "optimistic", "bull"]
    },
    "bearish": {
        "term": "Bearish",
        "definition": "Aspettativa di mercato che i prezzi scendano (Ribassista)",
        "category": "sentiment",
        "keywords": ["bearish", "ribassista", "sell", "pessimistic", "bear"]
    },
    "fomo": {
        "term": "FOMO",
        "definition": "Fear Of Missing Out - Paura di essere lasciati fuori da un rialzo",
        "category": "sentiment",
        "keywords": ["fomo", "miss", "hype", "rush"]
    },
    "fomo_index": {
        "term": "Indice FOMO",
        "definition": "Misura quantitativa della paura di perdere opportunità",
        "category": "sentiment",
        "keywords": ["fomo", "hype", "greed"]
    },
    "fud": {
        "term": "FUD",
        "definition": "Fear, Uncertainty and Doubt - Diffusione di notizie negative per generare paura",
        "category": "sentiment",
        "keywords": ["fud", "fear", "uncertainty", "panic"]
    },
    "fear_greed_index": {
        "term": "Fear & Greed Index",
        "definition": "Indice che misura le emozioni dominanti nel mercato (0-100)",
        "category": "sentiment",
        "keywords": ["fear", "greed", "sentiment index", "emotion"]
    },
    
    # Strategy
    "mean_reversion": {
        "term": "Mean Reversion",
        "definition": "Teoria secondo cui i prezzi degli asset tendono a tornare alla loro media storica",
        "category": "strategy",
        "keywords": ["mean", "reversion", "media", "ritorno"]
    },
    "momentum": {
        "term": "Momentum Trading",
        "definition": "Strategia che cavalca la forza e la velocità di un trend esistente",
        "category": "strategy",
        "keywords": ["momentum", "trend", "forza", "continuation"]
    },
    "dca": {
        "term": "DCA (Dollar Cost Averaging)",
        "definition": "Investimento costante di una somma fissa a intervalli regolari",
        "category": "strategy",
        "keywords": ["dca", "average", "accumulo", "cost averaging"]
    },
    "hedging": {
        "term": "Hedging",
        "definition": "Strategia di copertura per ridurre il rischio di variazioni avverse dei prezzi",
        "category": "strategy",
        "keywords": ["hedge", "coverage", "copertura"]
    },
    
    # Advanced Institutional Risk (Hedge Fund Edition)
    "emergency_liquidation": {
        "term": "Emergency Liquidation",
        "definition": "Chiusura istantanea di tutte le posizioni in caso di superamento dei limiti critici di drawdown",
        "category": "risk_control",
        "keywords": ["emergency", "liquidation", "close all", "panic button", "liquidazione"]
    },
    "correlated_exposure_guard": {
        "term": "Correlated Exposure Guard",
        "definition": "Sistema preventivo che impedisce l'apertura automatica di posizioni correlate per evitare concentrazione di rischio",
        "category": "risk_control",
        "keywords": ["correlation", "correlated", "exposure", "guard", "protezione correlazione"]
    },
    "volatility_circuit_breaker": {
        "term": "Volatility Circuit Breaker",
        "definition": "Meccanismo di interruzione automatica del trading durante periodi di estrema volatilità di mercato",
        "category": "risk_control",
        "keywords": ["circuit breaker", "halt", "pause", "volatility", "stop trading", "interruzione"]
    },
    "concentration_risk": {
        "term": "Concentration Risk",
        "definition": "Rischio derivante dall'avere un'esposizione eccessiva su un singolo asset o settore",
        "category": "risk",
        "keywords": ["concentration", "diversification", "portfolio weight", "concentrazione"]
    },
    "beta_exposure": {
        "term": "Beta Exposure",
        "definition": "Misura della sensibilità del portafoglio o di un asset rispetto ai movimenti del mercato generale",
        "category": "risk",
        "keywords": ["beta", "exposure", "market risk", "sensitività"]
    },
}

def get_concept(concept_id: str) -> Dict:
    """Helper to get a safe copy of a concept"""
    return SHARED_FINANCIAL_CONCEPTS.get(concept_id, {}).copy()

def get_all_concepts() -> List[Dict]:
    """Helper to get all concepts as a list"""
    return list(SHARED_FINANCIAL_CONCEPTS.values())

def get_categories() -> List[str]:
    """Helper to get all categories"""
    return list(set(c["category"] for c in SHARED_FINANCIAL_CONCEPTS.values()))
