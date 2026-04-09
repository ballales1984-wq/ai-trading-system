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
        "keywords": ["buy", "long", "acquisto", "rialzo", "rialzista", "purchase"],
    },
    "short_position": {
        "term": "Short Position",
        "definition": "Posizione ribassista che guadagna quando il prezzo scende (Vendita)",
        "category": "trading",
        "keywords": ["sell", "short", "ribasso", "ribassista", "shortare", "shorting"],
    },
    "stop_loss": {
        "term": "Stop Loss",
        "definition": "Ordine automatico per limitare le perdite a un livello predefinito",
        "category": "risk",
        "keywords": ["stop", "stop loss", "protezione", "limite", "exit"],
    },
    "take_profit": {
        "term": "Take Profit",
        "definition": "Ordine automatico per chiudere in profitto a un target specifico",
        "category": "risk",
        "keywords": ["target", "profit", "take profit", "obiettivo", "chiusura"],
    },
    "leverage": {
        "term": "Leva (Leverage)",
        "definition": "Moltiplicatore che amplifica l'esposizione con meno capitale",
        "category": "trading",
        "keywords": ["leverage", "leva", "margine", "x10", "x100", "leveraged"],
    },
    "margin": {
        "term": "Margine",
        "definition": "Capitale richiesto come garanzia per mantenere una posizione aperta",
        "category": "trading",
        "keywords": ["margin", "collateral", "collaterale"],
    },
    # Risk Metrics
    "var": {
        "term": "VaR (Value at Risk)",
        "definition": "Massima perdita attesa in un intervallo di confidenza (es. 95%)",
        "category": "risk",
        "keywords": ["var", "value at risk", "rischio", "confidenza", "maximum loss"],
    },
    "cvar": {
        "term": "CVaR (Conditional VaR)",
        "definition": "Perdita attesa media quando si supera la soglia del VaR (Expected Shortfall)",
        "category": "risk",
        "keywords": ["cvar", "expected shortfall", "tail risk", "average loss", "tail"],
    },
    "drawdown": {
        "term": "Drawdown",
        "definition": "Calo percentuale dal picco al minimo locale del portafoglio",
        "category": "risk",
        "keywords": ["drawdown", "dd", "perdita", "calo", "peak to trough"],
    },
    "max_dd": {
        "term": "Max Drawdown",
        "definition": "Il massimo calo storico registrato dal portafoglio dal suo picco",
        "category": "risk",
        "keywords": ["maxdd", "max drawdown", "peak", "worst case"],
    },
    "sharpe_ratio": {
        "term": "Sharpe Ratio",
        "definition": "Ritorno aggiustato per il rischio (extra-rendimento / deviazione standard)",
        "category": "metrics",
        "keywords": ["sharpe", "risk-adjusted", "performance", "ratio"],
    },
    "sortino_ratio": {
        "term": "Sortino Ratio",
        "definition": "Variante dello Sharpe che considera solo la volatilità negativa",
        "category": "metrics",
        "keywords": ["sortino", "downside risk", "downside volatility"],
    },
    # Technical Analysis
    "rsi": {
        "term": "RSI (Relative Strength Index)",
        "definition": "Indicatore di momentum che misura velocità e cambiamento dei movimenti dei prezzi",
        "category": "technical",
        "keywords": ["rsi", "overbought", "oversold", "ipercomprato", "ipervenduto"],
    },
    "macd": {
        "term": "MACD",
        "definition": "Moving Average Convergence Divergence - indicatore di trend e momentum",
        "category": "technical",
        "keywords": ["macd", "trend", "moving average", "convergence", "divergence"],
    },
    "ema": {
        "term": "EMA (Media Mobile Esponenziale)",
        "definition": "Media mobile che dà più peso ai prezzi più recenti",
        "category": "technical",
        "keywords": ["ema", "media mobile", "exponential average"],
    },
    "sma": {
        "term": "SMA (Media Mobile Semplice)",
        "definition": "Media aritmetica dei prezzi su un periodo specifico",
        "category": "technical",
        "keywords": ["sma", "ma", "media mobile", "simple average"],
    },
    "bollinger": {
        "term": "Bande di Bollinger",
        "definition": "Bande di volatilità basate su media mobile e deviazione standard",
        "category": "technical",
        "keywords": ["bollinger", "bands", "volatility", "bands", "volatilità"],
    },
    "support": {
        "term": "Supporto",
        "definition": "Livello di prezzo dove la domanda tende ad assorbire l'offerta",
        "category": "technical",
        "keywords": ["support", "floor", "supporto", "minimo", "demand"],
    },
    "resistance": {
        "term": "Resistenza",
        "definition": "Livello di prezzo dove l'offerta tende a superare la domanda",
        "category": "technical",
        "keywords": ["resistance", "ceiling", "resistenza", "massimo", "supply"],
    },
    "breakout": {
        "term": "Breakout",
        "definition": "Rottura decisa di un livello di supporto o resistenza",
        "category": "technical",
        "keywords": ["breakout", "rottura", "rupture", "volatilità"],
    },
    "pullback": {
        "term": "Pullback",
        "definition": "Ritorno temporaneo del prezzo verso un livello appena rotto",
        "category": "technical",
        "keywords": ["pullback", "retrace", "ritracciamento"],
    },
    # Market Structure
    "volatility": {
        "term": "Volatilità",
        "definition": "Misura della dispersione dei rendimenti e delle oscillazioni dei prezzi",
        "category": "market",
        "keywords": ["volatility", "volatile", "oscillazione", "instabilita", "variance"],
    },
    "liquidity": {
        "term": "Liquidità",
        "definition": "Facilità con cui un asset può essere scambiato senza impattare il prezzo",
        "category": "market",
        "keywords": ["liquidity", "liquidità", "depth", "volume", "slippage"],
    },
    "spread": {
        "term": "Spread",
        "definition": "Differenza tra il prezzo di acquisto (bid) e di vendita (ask)",
        "category": "market",
        "keywords": ["spread", "bid", "ask", "denaro", "lettera", "differenziale"],
    },
    "volume": {
        "term": "Volume",
        "definition": "Quantità totale di un asset scambiata in un dato periodo",
        "category": "market",
        "keywords": ["volume", "traded", "scambi", "turnover"],
    },
    "market_cap": {
        "term": "Market Cap",
        "definition": "Valore totale di mercato (Prezzo * Circolante)",
        "category": "market",
        "keywords": ["market cap", "capitalizzazione", "valuation"],
    },
    # DeFi & Blockchain
    "staking": {
        "term": "Staking",
        "definition": "Blocco di asset per supportare la sicurezza della rete in cambio di premi",
        "category": "defi",
        "keywords": ["staking", "stake", "reward", "validator", "delegation"],
    },
    "liquidity_pool": {
        "term": "Liquidity Pool",
        "definition": "Riserva di token bloccata in uno smart contract per facilitare il trading",
        "category": "defi",
        "keywords": ["lp", "pool", "dex", "swap", "amm"],
    },
    "impermanent_loss": {
        "term": "Impermanent Loss",
        "definition": "Perdita temporanea subita dai fornitori di liquidità rispetto all'holding",
        "category": "defi",
        "keywords": ["il", "impermanent loss", "divergence loss"],
    },
    "yield_farming": {
        "term": "Yield Farming",
        "definition": "Strategia per massimizzare i rendimenti spostando asset tra protocolli DeFi",
        "category": "defi",
        "keywords": ["yield", "farming", "rendimento", "apy", "apr"],
    },
    "smart_contract": {
        "term": "Smart Contract",
        "definition": "Contratto auto-eseguibile con i termini scritti direttamente nel codice",
        "category": "crypto",
        "keywords": ["smart", "contract", "contratto", "automa", "solidity"],
    },
    "layer2": {
        "term": "Layer 2",
        "definition": "Protocolli costruiti sopra una blockchain (L1) per migliorarne la scalabilità",
        "category": "crypto",
        "keywords": ["l2", "scalability", "arbitrum", "optimism", "rollup", "zksync"],
    },
    # Economic
    "inflation": {
        "term": "Inflazione",
        "definition": "Aumento generalizzato del livello dei prezzi e perdita di potere d'acquisto",
        "category": "economic",
        "keywords": ["inflation", "inflazione", "cpi", "prezzi", "purchasing power"],
    },
    "interest_rate": {
        "term": "Tasso di Interesse",
        "definition": "Costo del denaro stabilito dalle banche centrali (FED, BCE)",
        "category": "economic",
        "keywords": ["interest", "fed", "rate", "tasso", "monetary policy"],
    },
    "recession": {
        "term": "Recessione",
        "definition": "Periodo di declino economico significativo, spesso definito da due trimestri di PIL negativo",
        "category": "economic",
        "keywords": ["recession", "recessione", "downturn", "contrazione"],
    },
    "quantitative_easing": {
        "term": "Quantitative Easing (QE)",
        "definition": "Politica monetaria espansiva in cui la banca centrale acquista titoli",
        "category": "economic",
        "keywords": ["qe", "printing", "liquidity injection", "monetary base"],
    },
    # Sentiment
    "bullish": {
        "term": "Bullish",
        "definition": "Aspettativa di mercato che i prezzi salgano (Rialzista)",
        "category": "sentiment",
        "keywords": ["bullish", "rialzista", "buy", "optimistic", "bull"],
    },
    "bearish": {
        "term": "Bearish",
        "definition": "Aspettativa di mercato che i prezzi scendano (Ribassista)",
        "category": "sentiment",
        "keywords": ["bearish", "ribassista", "sell", "pessimistic", "bear"],
    },
    "fomo": {
        "term": "FOMO",
        "definition": "Fear Of Missing Out - Paura di essere lasciati fuori da un rialzo",
        "category": "sentiment",
        "keywords": ["fomo", "miss", "hype", "rush"],
    },
    "fomo_index": {
        "term": "Indice FOMO",
        "definition": "Misura quantitativa della paura di perdere opportunità",
        "category": "sentiment",
        "keywords": ["fomo", "hype", "greed"],
    },
    "fud": {
        "term": "FUD",
        "definition": "Fear, Uncertainty and Doubt - Diffusione di notizie negative per generare paura",
        "category": "sentiment",
        "keywords": ["fud", "fear", "uncertainty", "panic"],
    },
    "fear_greed_index": {
        "term": "Fear & Greed Index",
        "definition": "Indice che misura le emozioni dominanti nel mercato (0-100)",
        "category": "sentiment",
        "keywords": ["fear", "greed", "sentiment index", "emotion"],
    },
    # Strategy
    "mean_reversion": {
        "term": "Mean Reversion",
        "definition": "Teoria secondo cui i prezzi degli asset tendono a tornare alla loro media storica",
        "category": "strategy",
        "keywords": ["mean", "reversion", "media", "ritorno"],
    },
    "momentum": {
        "term": "Momentum Trading",
        "definition": "Strategia che cavalca la forza e la velocità di un trend esistente",
        "category": "strategy",
        "keywords": ["momentum", "trend", "forza", "continuation"],
    },
    "dca": {
        "term": "DCA (Dollar Cost Averaging)",
        "definition": "Investimento costante di una somma fissa a intervalli regolari",
        "category": "strategy",
        "keywords": ["dca", "average", "accumulo", "cost averaging"],
    },
    "hedging": {
        "term": "Hedging",
        "definition": "Strategia di copertura per ridurre il rischio di variazioni avverse dei prezzi",
        "category": "strategy",
        "keywords": ["hedge", "coverage", "copertura"],
    },
    # Advanced Institutional Risk (Hedge Fund Edition)
    "emergency_liquidation": {
        "term": "Emergency Liquidation",
        "definition": "Chiusura istantanea di tutte le posizioni in caso di superamento dei limiti critici di drawdown",
        "category": "risk_control",
        "keywords": ["emergency", "liquidation", "close all", "panic button", "liquidazione"],
    },
    "correlated_exposure_guard": {
        "term": "Correlated Exposure Guard",
        "definition": "Sistema preventivo che impedisce l'apertura automatica di posizioni correlate per evitare concentrazione di rischio",
        "category": "risk_control",
        "keywords": ["correlation", "correlated", "exposure", "guard", "protezione correlazione"],
    },
    "volatility_circuit_breaker": {
        "term": "Volatility Circuit Breaker",
        "definition": "Meccanismo di interruzione automatica del trading durante periodi di estrema volatilità di mercato",
        "category": "risk_control",
        "keywords": [
            "circuit breaker",
            "halt",
            "pause",
            "volatility",
            "stop trading",
            "interruzione",
        ],
    },
    "concentration_risk": {
        "term": "Concentration Risk",
        "definition": "Rischio derivante dall'avere un'esposizione eccessiva su un singolo asset o settore",
        "category": "risk",
        "keywords": ["concentration", "diversification", "portfolio weight", "concentrazione"],
    },
    "beta_exposure": {
        "term": "Beta Exposure",
        "definition": "Misura della sensibilità del portafoglio o di un asset rispetto ai movimenti del mercato generale",
        "category": "risk",
        "keywords": ["beta", "exposure", "market risk", "sensitività"],
    },
    # Technical Analysis Indicators
    "rsi": {
        "term": "RSI (Relative Strength Index)",
        "definition": "Indicatore di momentum che misura la velocità e il cambiamento dei movimenti di prezzo (0-100)",
        "category": "technical",
        "keywords": ["rsi", "overbought", "oversold", "momentum", "strength"],
    },
    "macd": {
        "term": "MACD (Moving Average Convergence Divergence)",
        "definition": "Indicatore di trend che mostra la relazione tra due medie mobili del prezzo",
        "category": "technical",
        "keywords": ["macd", "convergence", "divergence", "signal line", "histogram"],
    },
    "bollinger_bands": {
        "term": "Bollinger Bands",
        "definition": "Bande di volatilità che si adattano alla volatilità del mercato",
        "category": "technical",
        "keywords": ["bollinger", "bands", "volatility", " squeeze", "envelope"],
    },
    "moving_average": {
        "term": "Moving Average",
        "definition": "Media mobile che smootha i dati di prezzo su un periodo specifico",
        "category": "technical",
        "keywords": ["ma", "sma", "ema", "wma", "average", "trend"],
    },
    "fibonacci_retracement": {
        "term": "Fibonacci Retracement",
        "definition": "Livelli di supporto/resistenza basati sulla sequenza di Fibonacci",
        "category": "technical",
        "keywords": ["fibonacci", "retracement", "support", "resistance", "golden ratio"],
    },
    "atr": {
        "term": "ATR (Average True Range)",
        "definition": "Misura della volatilità del mercato calcolando la media dei range veri",
        "category": "technical",
        "keywords": ["atr", "volatility", "range", "true range"],
    },
    "stochastic_oscillator": {
        "term": "Stochastic Oscillator",
        "definition": "Indicatore di momentum che confronta il prezzo di chiusura con il range",
        "category": "technical",
        "keywords": ["stochastic", "oscillator", "k", "d", "momentum"],
    },
    "adx": {
        "term": "ADX (Average Directional Index)",
        "definition": "Misura la forza di un trend senza considerare la direzione",
        "category": "technical",
        "keywords": ["adx", "dmi", "trend strength", "directional"],
    },
    # Portfolio Management
    "diversification": {
        "term": "Diversification",
        "definition": "Strategia di distribuzione del capitale su asset multiple per ridurre il rischio",
        "category": "portfolio",
        "keywords": ["diversification", "spread", "allocation", "portfolio"],
    },
    "asset_allocation": {
        "term": "Asset Allocation",
        "definition": "Distribuzione del capitale tra diverse classi di asset",
        "category": "portfolio",
        "keywords": ["allocation", "asset", "weight", "distribution", "portfolio"],
    },
    "rebalancing": {
        "term": "Rebalancing",
        "definition": "Ridistribuzione periodica del portafoglio per mantenere l'allocazione target",
        "category": "portfolio",
        "keywords": ["rebalancing", "adjust", "target", "threshold"],
    },
    "sharpe_ratio": {
        "term": "Sharpe Ratio",
        "definition": "Misura del rendimento aggiustato per il rischio",
        "category": "portfolio",
        "keywords": ["sharpe", "risk adjusted", "return", "performance"],
    },
    "sortino_ratio": {
        "term": "Sortino Ratio",
        "definition": "Sharpe ratio che considera solo la volatilità negativa",
        "category": "portfolio",
        "keywords": ["sortino", "downside risk", "return", "performance"],
    },
    "max_drawdown": {
        "term": "Maximum Drawdown",
        "definition": "La più grande perdita registrata da un picco precedente",
        "category": "portfolio",
        "keywords": ["drawdown", "max drawdown", "peak", "trough", "loss"],
    },
    "compound_annual_growth": {
        "term": "CAGR (Compound Annual Growth Rate)",
        "definition": "Tasso di crescita annuale composto che smootha la volatilità",
        "category": "portfolio",
        "keywords": ["cagr", "growth", "annual", "compound", "return"],
    },
    # DeFi/Crypto Specific
    "liquidity_pool": {
        "term": "Liquidity Pool",
        "definition": "Pool di fondi bloccati in smart contract per facilitare scambi",
        "category": "defi",
        "keywords": ["liquidity pool", "lp", "amm", "swap", "yield"],
    },
    "staking": {
        "term": "Staking",
        "definition": "Blocco di token per partecipare alla validazione della rete e ricevere reward",
        "category": "defi",
        "keywords": ["staking", "lock", "reward", "validator", "delegation"],
    },
    "yield_farming": {
        "term": "Yield Farming",
        "definition": "Strategia di massimizzazione del rendimento spostando capitale tra protocolli",
        "category": "defi",
        "keywords": ["yield farming", "farm", "apy", "reward", "harvest"],
    },
    "impermanent_loss": {
        "term": "Impermanent Loss",
        "definition": "Perdita temporanea che si verifica quando il prezzo degli asset in LP cambia",
        "category": "defi",
        "keywords": ["impermanent loss", "il", "loss", " lp", "无常损失"],
    },
    "smart_contract": {
        "term": "Smart Contract",
        "definition": "Codice auto-eseguibile su blockchain che enforcea accordi",
        "category": "defi",
        "keywords": ["smart contract", "contract", "code", "automate", "execute"],
    },
    "oracle": {
        "term": "Oracle",
        "definition": "Servizio che fornisce dati esterni alla blockchain",
        "category": "defi",
        "keywords": ["oracle", "data feed", "external", "chainlink", "price feed"],
    },
    "gas_fee": {
        "term": "Gas Fee",
        "definition": "Fee di transazione sulla blockchain ( Ethereum)",
        "category": "defi",
        "keywords": ["gas", "fee", "transaction cost", "network fee"],
    },
    "slippage": {
        "term": "Slippage",
        "definition": "Differenza tra prezzo atteso e prezzo eseguito",
        "category": "defi",
        "keywords": ["slippage", "price impact", "execution", "spread"],
    },
    # Macro Economics
    "inflation_rate": {
        "term": "Inflation Rate",
        "definition": "Tasso di crescita dei prezzi nel tempo",
        "category": "macro",
        "keywords": ["inflation", "cpi", "ppi", "price index", "deflation"],
    },
    "interest_rate": {
        "term": "Interest Rate",
        "definition": "Tasso applicato al capitale prestato",
        "category": "macro",
        "keywords": ["interest rate", "fed rate", "policy rate", "benchmark"],
    },
    "gdp_growth": {
        "term": "GDP Growth",
        "definition": "Tasso di crescita del prodotto interno lordo",
        "category": "macro",
        "keywords": ["gdp", "growth", "economic", "quarterly", "output"],
    },
    "unemployment_rate": {
        "term": "Unemployment Rate",
        "definition": "Percentuale della forza lavoro senza impiego",
        "category": "macro",
        "keywords": ["unemployment", "jobs", "labor", "employment"],
    },
    "consumer_sentiment": {
        "term": "Consumer Sentiment",
        "definition": "Misura della fiducia dei consumatori nell'economia",
        "category": "macro",
        "keywords": ["consumer confidence", "sentiment", "survey", "michigan"],
    },
    # Derivatives/Options
    "call_option": {
        "term": "Call Option",
        "definition": "Contratto che dà il diritto di acquistare un asset a un prezzo specifico",
        "category": "derivatives",
        "keywords": ["call", "option", "buy right", "strike", "premium"],
    },
    "put_option": {
        "term": "Put Option",
        "definition": "Contratto che dà il diritto di vendere un asset a un prezzo specifico",
        "category": "derivatives",
        "keywords": ["put", "option", "sell right", "strike", "premium"],
    },
    "futures_contract": {
        "term": "Futures Contract",
        "definition": "Contratto standardizzato per acquistare/vendere un asset a una data futura",
        "category": "derivatives",
        "keywords": ["futures", "contract", "delivery", "settlement", "margin"],
    },
    "option_greeks": {
        "term": "Option Greeks",
        "definition": "Metriche che misurano la sensitività del prezzo delle opzioni",
        "category": "derivatives",
        "keywords": ["greeks", "delta", "gamma", "theta", "vega", "rho"],
    },
    "strike_price": {
        "term": "Strike Price",
        "definition": "Prezzo al quale l'opzione può essere esercitata",
        "category": "derivatives",
        "keywords": ["strike", "exercise price", "target price"],
    },
    "implied_volatility": {
        "term": "Implied Volatility",
        "definition": "Volatilità futura attesa derivata dal prezzo delle opzioni",
        "category": "derivatives",
        "keywords": ["iv", "implied", "volatility", "option pricing"],
    },
    # Trading Strategies
    "mean_reversion": {
        "term": "Mean Reversion",
        "definition": "Strategia che assume che i prezzi ritornino alla media storica",
        "category": "strategy",
        "keywords": ["mean reversion", "revert", "average", "oscillation"],
    },
    "momentum_trading": {
        "term": "Momentum Trading",
        "definition": "Strategia che sfrutta trend in corso",
        "category": "strategy",
        "keywords": ["momentum", "trend", "follow", "strength"],
    },
    "breakout_trading": {
        "term": "Breakout Trading",
        "definition": "Strategia che entra quando il prezzo rompe livelli di supporto/resistenza",
        "category": "strategy",
        "keywords": ["breakout", "break", "resistance", "support", "突破"],
    },
    "scalping": {
        "term": "Scalping",
        "definition": "Strategia che cerca piccoli profitti da micro-movimenti",
        "category": "strategy",
        "keywords": ["scalping", "quick", "small", "frequent", "micro"],
    },
    "swing_trading": {
        "term": "Swing Trading",
        "definition": "Strategia che tiene posizioni da giorni a settimane",
        "category": "strategy",
        "keywords": ["swing", "medium term", "hold", "days"],
    },
    "position_trading": {
        "term": "Position Trading",
        "definition": "Strategia a lungo termine basata su trend primari",
        "category": "strategy",
        "keywords": ["position", "long term", "trend", "holding"],
    },
    # Risk Management
    "position_sizing": {
        "term": "Position Sizing",
        "definition": "Determinazione della quantità di capitale per ogni trade",
        "category": "risk",
        "keywords": ["position sizing", "size", "risk per trade", "capital"],
    },
    "risk_reward_ratio": {
        "term": "Risk-Reward Ratio",
        "definition": "Rapporto tra potenziale perdita e potenziale guadagno",
        "category": "risk",
        "keywords": ["risk reward", "ratio", "rr", "reward"],
    },
    "kelly_criterion": {
        "term": "Kelly Criterion",
        "definition": "Formula per determinare la dimensione ottimale della posizione",
        "category": "risk",
        "keywords": ["kelly", "optimal", "bet size", "edge"],
    },
    "correlation_matrix": {
        "term": "Correlation Matrix",
        "definition": "Tabella che mostra la correlazione tra asset",
        "category": "risk",
        "keywords": ["correlation", "matrix", "correlation", "asset"],
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
