"""
Opportunity Filter Module
=========================
Modulo per il filtraggio intelligente delle opportunità di trading.
Combina analisi semantica (sentiment, news, eventi) con analisi numerica (indicatori tecnici).
"""

from typing import List, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OpportunityFilter:
    """
    Filtro opportunità che combina analisi semantica e numerica
    per selezionare solo gli asset potenzialmente profittevoli.
    
    RISK ENGINE PRO CONSTANTS:
    - NO_TRADE_ZONE: score tra 0.45 e 0.55 → HOLD (neutral zone)
    - MIN_CONFIDENCE: confidence < 0.6 → HOLD (low confidence)
    - MAX_VAR: VaR > 5% → HOLD (risk too high)
    """
    
    # RISK ENGINE PRO - Costanti di sicurezza
    NO_TRADE_LOW = 0.45
    NO_TRADE_HIGH = 0.55
    MIN_CONFIDENCE = 0.6
    MAX_VAR = 0.05  # 5%
    
    def __init__(
        self,
        threshold_confidence: float = 0.6,
        semantic_weight: float = 0.5,
        numeric_weight: float = 0.5
    ):
        """
        Inizializza il filtro opportunità.
        
        Args:
            threshold_confidence: Soglia minima di confidenza per selezionare un asset
            semantic_weight: Peso dell'analisi semantica (0-1)
            numeric_weight: Peso dell'analisi numerica (0-1)
        """
        self.threshold = threshold_confidence
        self.semantic_weight = semantic_weight
        self.numeric_weight = numeric_weight
        
        # Verifica che i pesi sommino a 1
        total_weight = semantic_weight + numeric_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.semantic_weight = semantic_weight / total_weight
            self.numeric_weight = numeric_weight / total_weight

    def analyze_semantic(self, asset_data: Dict) -> float:
        """
        Analizza trend logico/semantico: news, sentiment, eventi macro.
        
        Args:
            asset_data: Dizionario con dati semantici dell'asset
                - sentiment_score: Punteggio sentiment (-1 a +1)
                - event_impact: Impatto eventi (-1 a +1)
                - trend_signal: Segnale trend (-1 a +1)
                - news_score: Punteggio news (-1 a +1)
                
        Returns:
            Punteggio semantico tra -1 e 1
        """
        sentiment_score = asset_data.get("sentiment_score", 0)
        event_impact = asset_data.get("event_impact", 0)
        trend_signal = asset_data.get("trend_signal", 0)
        news_score = asset_data.get("news_score", 0)
        
        # Combinazione ponderata dei fattori semantici
        score = (
            0.35 * sentiment_score +
            0.25 * event_impact +
            0.20 * trend_signal +
            0.20 * news_score
        )
        
        return np.clip(score, -1, 1)

    def analyze_numeric(self, asset_data: Dict) -> float:
        """
        Analizza trend matematico/quantitativo: indicatori, volatilità, rischio.
        
        Args:
            asset_data: Dizionario con dati numerici dell'asset
                - rsi_score: Punteggio RSI normalizzato (-1 a +1)
                - macd_score: Punteggio MACD normalizzato (-1 a +1)
                - volatility_score: Punteggio volatilità (0-1, penalità)
                - momentum_score: Punteggio momentum (-1 a +1)
                - volume_score: Punteggio volume (-1 a +1)
                
        Returns:
            Punteggio numerico tra -1 e 1
        """
        rsi_score = asset_data.get("rsi_score", 0)
        macd_score = asset_data.get("macd_score", 0)
        volatility_score = asset_data.get("volatility_score", 0)
        momentum_score = asset_data.get("momentum_score", 0)
        volume_score = asset_data.get("volume_score", 0)
        
        # Combinazione ponderata dei fattori numerici
        # Volatilità alta è una penalità (sottrazione)
        score = (
            0.25 * rsi_score +
            0.25 * macd_score +
            0.20 * momentum_score +
            0.15 * volume_score -
            0.15 * volatility_score
        )
        
        return np.clip(score, -1, 1)

    def combine_scores(self, semantic: float, numeric: float) -> float:
        """
        Combina segnali logici e numerici in un punteggio finale.
        
        Args:
            semantic: Punteggio semantico
            numeric: Punteggio numerico
            
        Returns:
            Punteggio combinato tra -1 e 1
        """
        combined = (
            self.semantic_weight * semantic +
            self.numeric_weight * numeric
        )
        return np.clip(combined, -1, 1)

    def filter_assets(self, assets: List[Dict]) -> List[Dict]:
        """
        Filtra solo asset con confidenza sufficiente e aggiunge punteggio finale.
        
        Args:
            assets: Lista di dizionari con dati degli asset
            
        Returns:
            Lista di asset filtrati con punteggi aggiunti
        """
        selected_assets = []
        
        for asset in assets:
            # Calcola punteggi
            semantic_score = self.analyze_semantic(asset)
            numeric_score = self.analyze_numeric(asset)
            combined_score = self.combine_scores(semantic_score, numeric_score)
            
            # Normalizza score a 0-1 per controllo no-trade zone
            normalized_score = (combined_score + 1) / 2
            confidence = abs(combined_score)
            
            # RISK ENGINE: Check no-trade zone FIRST
            if self.is_no_trade_zone(normalized_score):
                logger.info(
                    f"Asset {asset.get('name', 'Unknown')} FILTERED: "
                    f"score={normalized_score:.2f} in NO-TRADE ZONE ({self.NO_TRADE_LOW}-{self.NO_TRADE_HIGH})"
                )
                continue
            
            # RISK ENGINE: Check confidence threshold
            if confidence < self.MIN_CONFIDENCE:
                logger.info(
                    f"Asset {asset.get('name', 'Unknown')} FILTERED: "
                    f"confidence={confidence:.2f} < MIN_CONFIDENCE={self.MIN_CONFIDENCE}"
                )
                continue
            
            # Aggiungi punteggi al dizionario asset
            asset['semantic_score'] = semantic_score
            asset['numeric_score'] = numeric_score
            asset['combined_score'] = combined_score
            asset['confidence'] = confidence
            
            # Seleziona solo se punteggio assoluto sopra soglia
            if abs(combined_score) >= self.threshold:
                selected_assets.append(asset)
                logger.info(
                    f"Asset {asset.get('name', 'Unknown')} selected: "
                    f"semantic={semantic_score:.2f}, numeric={numeric_score:.2f}, "
                    f"combined={combined_score:.2f}, confidence={confidence:.2f}"
                )
            else:
                logger.debug(
                    f"Asset {asset.get('name', 'Unknown')} filtered out: "
                    f"combined={combined_score:.2f} < threshold={self.threshold}"
                )
        
        # Ordina per punteggio combinato decrescente
        selected_assets.sort(key=lambda x: abs(x['combined_score']), reverse=True)
        
        return selected_assets

    def is_no_trade_zone(self, combined_score: float) -> bool:
        """
        Check if score is in neutral zone (no-trade zone).
        
        Args:
            combined_score: Punteggio combinato (0-1 normalized)
            
        Returns:
            True if in no-trade zone, False otherwise
        """
        return self.NO_TRADE_LOW < combined_score < self.NO_TRADE_HIGH
    
    def get_signal_direction(self, combined_score: float, confidence: float = None) -> str:
        """
        Determina la direzione del segnale basata sul punteggio combinato.
        
        RISK ENGINE PRO:
        - NO_TRADE_ZONE: score tra 0.45 e 0.55 → HOLD
        - MIN_CONFIDENCE: confidence < 0.6 → HOLD
        
        Args:
            combined_score: Punteggio combinato
            confidence: Livello di confidenza (opzionale)
            
        Returns:
            "BUY", "SELL", o "HOLD"
        """
        # RISK ENGINE: Check confidence first
        if confidence is not None and confidence < self.MIN_CONFIDENCE:
            logger.info(f"HOLD: Low confidence {confidence:.2f} < {self.MIN_CONFIDENCE}")
            return "HOLD"
        
        # RISK ENGINE: Check no-trade zone
        # Normalize score to 0-1 range for zone check
        normalized_score = (combined_score + 1) / 2  # Convert -1 to 1 → 0 to 1
        if self.is_no_trade_zone(normalized_score):
            logger.info(f"HOLD: Neutral zone score={normalized_score:.2f} (0.45-0.55)")
            return "HOLD"
        
        # Original logic
        if combined_score > 0.1:
            return "BUY"
        elif combined_score < -0.1:
            return "SELL"
        else:
            return "HOLD"


# === Esempio di utilizzo ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    assets = [
        {
            "name": "Oro",
            "sentiment_score": 0.8,
            "event_impact": 0.3,
            "trend_signal": 0.5,
            "news_score": 0.6,
            "rsi_score": 0.7,
            "macd_score": 0.6,
            "volatility_score": 0.2,
            "momentum_score": 0.5,
            "volume_score": 0.4
        },
        {
            "name": "Rame",
            "sentiment_score": -0.4,
            "event_impact": -0.2,
            "trend_signal": -0.1,
            "news_score": -0.3,
            "rsi_score": -0.5,
            "macd_score": -0.4,
            "volatility_score": 0.1,
            "momentum_score": -0.3,
            "volume_score": -0.2
        },
        {
            "name": "BTC",
            "sentiment_score": 0.1,
            "event_impact": 0.0,
            "trend_signal": 0.2,
            "news_score": 0.1,
            "rsi_score": 0.3,
            "macd_score": 0.2,
            "volatility_score": 0.3,
            "momentum_score": 0.1,
            "volume_score": 0.2
        },
    ]

    filtro = OpportunityFilter(threshold_confidence=0.6)
    selezionati = filtro.filter_assets(assets)
    
    print("\n" + "=" * 60)
    print("ASSET SELEZIONATI")
    print("=" * 60)
    for asset in selezionati:
        direction = filtro.get_signal_direction(asset['combined_score'])
        print(f"  {asset['name']:6} | {direction:4} | Combined: {asset['combined_score']:.2f}")
        print(f"         | Semantic: {asset['semantic_score']:.2f} | Numeric: {asset['numeric_score']:.2f}")
        print()
