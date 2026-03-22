"""
Decision Engine V2 - Versione PRO con Adaptive Thresholds
======================================================
Versione avanzata del motore decisionale con:
- Confidence dinamica basata sulla volatilità
- Regime detection (bull/bear/sideways)
- Adaptive thresholds
- Logging dettagliato del breakdown
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Regimi di mercato."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class DecisionEngineV2:
    """
    Motore decisionale V2 con adaptive thresholds e regime detection.
    
    Caratteristiche:
    - Confidence dinamica
    - Regime detection
    - Thresholds adattivi
    - Logging dettagliato
    """
    
    # Parametri base
    BASE_MIN_CONFIDENCE = 0.45
    BASE_NO_TRADE_LOW = 0.40
    BASE_NO_TRADE_HIGH = 0.60
    BASE_THRESHOLD = 0.40
    
    def __init__(
        self,
        initial_balance: float = 100000,
        base_risk_per_trade: float = 0.02,
        max_positions: int = 5
    ):
        self.initial_balance = initial_balance
        self.base_risk_per_trade = base_risk_per_trade
        self.max_positions = max_positions
        
        # Stato corrente
        self.current_regime = MarketRegime.SIDEWAYS
        self.market_volatility = 0.5
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        
        # Parametri dinamici (si aggiustano automaticamente)
        self.min_confidence = self.BASE_MIN_CONFIDENCE
        self.no_trade_low = self.BASE_NO_TRADE_LOW
        self.no_trade_high = self.BASE_NO_TRADE_HIGH
        self.threshold = self.BASE_THRESHOLD
        
        logger.info("🎯 Decision Engine V2 inizializzato")
    
    def detect_regime(
        self, 
        returns: List[float], 
        volatility: float,
        trend_indicator: float
    ) -> MarketRegime:
        """
        Rileva il regime di mercato corrente.
        
        Args:
            returns: Lista dei rendimenti recenti
            volatility: Volatilità corrente (0-1)
            trend_indicator: Indicatore di trend (-1 a 1)
        
        Returns:
            Regime di mercato rilevato
        """
        # Calcola media rendimenti
        avg_return = np.mean(returns) if returns else 0
        
        # Determina regime
        if volatility > 0.7:
            self.current_regime = MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.3:
            self.current_regime = MarketRegime.LOW_VOLATILITY
        elif avg_return > 0.01 and trend_indicator > 0.3:
            self.current_regime = MarketRegime.BULL
        elif avg_return < -0.01 and trend_indicator < -0.3:
            self.current_regime = MarketRegime.BEAR
        else:
            self.current_regime = MarketRegime.SIDEWAYS
        
        self.market_volatility = volatility
        
        return self.current_regime
    
    def calculate_dynamic_thresholds(self) -> Dict[str, float]:
        """
        Calcola thresholds dinamici basati sul regime.
        
        Returns:
            Dict con i parametri adattati
        """
        # Moltiplicatore basato sul regime
        regime_multipliers = {
            MarketRegime.BULL: 0.95,           # Più permissivo in bull
            MarketRegime.BEAR: 1.10,           # Più conservativo in bear
            MarketRegime.SIDEWAYS: 1.0,        # Normale
            MarketRegime.HIGH_VOLATILITY: 1.20, # Molto conservativo
            MarketRegime.LOW_VOLATILITY: 0.90  # Più permissivo
        }
        
        mult = regime_multipliers.get(self.current_regime, 1.0)
        
        # Calcola thresholds adattati
        self.min_confidence = self.BASE_MIN_CONFIDENCE * mult
        self.no_trade_low = self.BASE_NO_TRADE_LOW * mult
        self.no_trade_high = min(0.95, self.BASE_NO_TRADE_HIGH * mult)
        self.threshold = self.BASE_THRESHOLD * mult
        
        # Log dei parametri
        logger.info(
            f"📊 Regime: {self.current_regime.value} | "
            f"Vol: {self.market_volatility:.2f} | "
            f"MinConf: {self.min_confidence:.2f} | "
            f"NoTrade: {self.no_trade_low:.2f}-{self.no_trade_high:.2f}"
        )
        
        return {
            "min_confidence": self.min_confidence,
            "no_trade_low": self.no_trade_low,
            "no_trade_high": self.no_trade_high,
            "threshold": self.threshold
        }
    
    def calculate_confidence_with_breakdown(
        self,
        technical_score: float,
        sentiment_score: float,
        momentum_score: float,
        volatility_score: float,
        volume_score: float = 0.0
    ) -> Tuple[float, Dict]:
        """
        Calcola la confidence con breakdown dettagliato.
        
        Args:
            technical_score: Score tecnico (-1 a 1)
            sentiment_score: Score sentiment (-1 a 1)
            momentum_score: Score momentum (-1 a 1)
            volatility_score: Score volatilità (0-1, più alto = più volatile)
            volume_score: Score volume (-1 a 1)
        
        Returns:
            Tuple (confidence, breakdown_dict)
        """
        # Pesi per ogni componente
        weights = {
            "technical": 0.30,
            "sentiment": 0.20,
            "momentum": 0.25,
            "volume": 0.10,
            "volatility_penalty": -0.15
        }
        
        # Calcolo base della confidence
        base_confidence = (
            weights["technical"] * abs(technical_score) +
            weights["sentiment"] * abs(sentiment_score) +
            weights["momentum"] * abs(momentum_score) +
            weights["volume"] * abs(volume_score)
        )
        
        # Penalità per volatilità eccessiva
        # Volatilità alta = meno confidence
        volatility_penalty = weights["volatility_penalty"] * volatility_score
        
        # Accordo tra segnali (se tutti vanno nella stessa direzione = più confidence)
        signals = [technical_score, sentiment_score, momentum_score]
        agreement = 1 - np.std(signals) / 1.5  # Normalizzato
        agreement_bonus = 0.1 * max(0, agreement)
        
        # Confidence finale
        confidence = np.clip(
            base_confidence + volatility_penalty + agreement_bonus,
            0, 1
        )
        
        # Breakdown per logging
        breakdown = {
            "technical_score": technical_score,
            "technical_contribution": weights["technical"] * abs(technical_score),
            "sentiment_score": sentiment_score,
            "sentiment_contribution": weights["sentiment"] * abs(sentiment_score),
            "momentum_score": momentum_score,
            "momentum_contribution": weights["momentum"] * abs(momentum_score),
            "volatility_score": volatility_score,
            "volatility_penalty": volatility_penalty,
            "agreement": agreement,
            "agreement_bonus": agreement_bonus,
            "base_confidence": base_confidence,
            "final_confidence": confidence
        }
        
        return confidence, breakdown
    
    def log_confidence_breakdown(self, asset_name: str, breakdown: Dict):
        """Logga il breakdown della confidence."""
        logger.info(f"""
┌─────────────────────────────────────────────────────────┐
│ CONFIDENCE BREAKDOWN - {asset_name:8}
├─────────────────────────────────────────────────────────┤
│ technical:    {breakdown['technical_score']:+.2f} → {breakdown['technical_contribution']:.3f}
│ sentiment:    {breakdown['sentiment_score']:+.2f} → {breakdown['sentiment_contribution']:.3f}
│ momentum:     {breakdown['momentum_score']:+.2f} → {breakdown['momentum_contribution']:.3f}
│ volatility:   {breakdown['volatility_score']:.2f} → penalty {breakdown['volatility_penalty']:.3f}
│ agreement:    {breakdown['agreement']:.2f} → bonus {breakdown['agreement_bonus']:.3f}
├─────────────────────────────────────────────────────────┤
│ FINAL CONFIDENCE: {breakdown['final_confidence']:.2f}
└─────────────────────────────────────────────────────────┘
        """.strip())
    
    def evaluate_signal(
        self,
        asset_name: str,
        technical_score: float,
        sentiment_score: float,
        momentum_score: float,
        volatility_score: float,
        volume_score: float = 0.0,
        returns: List[float] = None,
        trend_indicator: float = 0.0
    ) -> Dict:
        """
        Valuta un segnale e ritorna la decisione.
        
        Args:
            asset_name: Nome dell'asset
            technical_score: Score tecnico
            sentiment_score: Score sentiment
            momentum_score: Score momentum
            volatility_score: Score volatilità
            volume_score: Score volume
            returns: Lista rendimenti recenti
            trend_indicator: Indicatore trend
        
        Returns:
            Dict con decisione e dettagli
        """
        # 1. Rileva regime
        if returns is not None:
            self.detect_regime(returns, volatility_score, trend_indicator)
        
        # 2. Calcola thresholds dinamici
        self.calculate_dynamic_thresholds()
        
        # 3. Calcola confidence con breakdown
        confidence, breakdown = self.calculate_confidence_with_breakdown(
            technical_score,
            sentiment_score,
            momentum_score,
            volatility_score,
            volume_score
        )
        
        # 4. Log breakdown
        self.log_confidence_breakdown(asset_name, breakdown)
        
        # 5. Normalizza score combinato
        combined_score = (
            0.4 * technical_score +
            0.3 * sentiment_score +
            0.3 * momentum_score
        )
        normalized_score = (combined_score + 1) / 2  # 0-1
        
        # 6. Check filtri
        in_no_trade_zone = self.no_trade_low < normalized_score < self.no_trade_high
        below_min_confidence = confidence < self.min_confidence
        below_threshold = abs(combined_score) < self.threshold
        
        # 7. Decisione
        if in_no_trade_zone:
            decision = "HOLD"
            reason = f"NO_TRADE_ZONE ({normalized_score:.2f})"
        elif below_min_confidence:
            decision = "HOLD"
            reason = f"LOW_CONFIDENCE ({confidence:.2f} < {self.min_confidence:.2f})"
        elif below_threshold:
            decision = "HOLD"
            reason = f"BELOW_THRESHOLD ({abs(combined_score):.2f} < {self.threshold:.2f})"
        elif combined_score > self.threshold:
            decision = "BUY"
            reason = f"STRONG_BUY (score={combined_score:.2f}, conf={confidence:.2f})"
        elif combined_score < -self.threshold:
            decision = "SELL"
            reason = f"STRONG_SELL (score={combined_score:.2f}, conf={confidence:.2f})"
        else:
            decision = "HOLD"
            reason = "NEUTRAL"
        
        # Log decisione
        logger.info(f"🎯 {asset_name}: {decision} - {reason}")
        
        return {
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "combined_score": combined_score,
            "normalized_score": normalized_score,
            "regime": self.current_regime.value,
            "volatility": self.market_volatility,
            "breakdown": breakdown,
            "thresholds": {
                "min_confidence": self.min_confidence,
                "no_trade_low": self.no_trade_low,
                "no_trade_high": self.no_trade_high,
                "threshold": self.threshold
            }
        }
    
    def get_stats(self) -> Dict:
        """Ritorna statistiche del motore."""
        total = self.win_count + self.loss_count
        win_rate = self.win_count / total if total > 0 else 0
        
        return {
            "total_trades": self.trade_count,
            "wins": self.win_count,
            "losses": self.loss_count,
            "win_rate": win_rate,
            "current_regime": self.current_regime.value,
            "current_volatility": self.market_volatility,
            "current_thresholds": {
                "min_confidence": self.min_confidence,
                "no_trade_zone": f"{self.no_trade_low:.2f}-{self.no_trade_high:.2f}",
                "threshold": self.threshold
            }
        }


# === ESEMPIO DI UTILIZZO ===
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    print("\n" + "=" * 60)
    print("TEST DECISION ENGINE V2")
    print("=" * 60)
    
    # Inizializza motore
    engine = DecisionEngineV2(initial_balance=100000)
    
    # Test con dati realistici
    test_cases = [
        {
            "name": "BTC",
            "technical": 0.6,
            "sentiment": 0.4,
            "momentum": 0.5,
            "volatility": 0.4,
            "volume": 0.3,
            "returns": [0.01, 0.02, -0.01, 0.03, 0.01],
            "trend": 0.4
        },
        {
            "name": "ETH",
            "technical": 0.2,
            "sentiment": 0.1,
            "momentum": 0.15,
            "volatility": 0.7,  # Alta volatilità
            "volume": 0.1,
            "returns": [0.05, -0.08, 0.06, -0.10, 0.07],
            "trend": 0.1
        },
        {
            "name": "SOL",
            "technical": -0.3,
            "sentiment": -0.2,
            "momentum": -0.25,
            "volatility": 0.5,
            "volume": -0.2,
            "returns": [-0.01, -0.02, -0.01, -0.03, -0.02],
            "trend": -0.3
        }
    ]
    
    for test in test_cases:
        print(f"\n📊 Analisi {test['name']}:")
        result = engine.evaluate_signal(
            asset_name=test["name"],
            technical_score=test["technical"],
            sentiment_score=test["sentiment"],
            momentum_score=test["momentum"],
            volatility_score=test["volatility"],
            volume_score=test["volume"],
            returns=test["returns"],
            trend_indicator=test["trend"]
        )
        print(f"   Decision: {result['decision']}")
    
    # Statistiche finali
    print("\n" + "=" * 60)
    print("STATISTICHE MOTORE")
    print("=" * 60)
    stats = engine.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
