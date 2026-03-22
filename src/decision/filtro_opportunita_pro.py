"""
Opportunity Filter Module - VERSIONE OPTIMIZZATA
==============================================
Versione corretta del filtro con parametri adattivi per evitare il blocco trading.

Problemi della versione originale:
1. NO_TRADE_ZONE troppo stretta (0.45-0.55 = 10% del range)
2. MIN_CONFIDENCE troppo alta (0.6 = 60%)
3. Triple filtering = zero segnali

Soluzione:
1. NO_TRADE_ZONE più stretta o rimovibile
2. MIN_CONFIDENCE adattiva
3. Logging dettagliato per debug
"""

from typing import List, Dict, Optional
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OpportunityFilterPro:
    """
    Filtro opportunità OPTIMIZZATO per generare segnali di trading.
    
    Problema originale: triple filtering = zero trades
    Soluzione: parametri bilanciati + adaptive confidence
    """
    
    # VERSIONE CONSERVATIVA (originale - genera 0 trades)
    NO_TRADE_LOW = 0.45
    NO_TRADE_HIGH = 0.55
    MIN_CONFIDENCE = 0.6
    
    # VERSIONE BILANCIATA (raccomandata)
    NO_TRADE_LOW_BALANCED = 0.42
    NO_TRADE_HIGH_BALANCED = 0.58
    MIN_CONFIDENCE_BALANCED = 0.45
    
    # VERSIONE AGGRESSIVA
    NO_TRADE_LOW_AGGRESSIVE = 0.35
    NO_TRADE_HIGH_AGGRESSIVE = 0.65
    MIN_CONFIDENCE_AGGRESSIVE = 0.35
    
    def __init__(
        self,
        threshold_confidence: float = 0.45,  # Default abbassato da 0.6
        semantic_weight: float = 0.5,
        numeric_weight: float = 0.5,
        mode: str = "balanced"  # "conservative", "balanced", "aggressive"
    ):
        """
        Inizializza il filtro opportunità optimizzato.
        
        Args:
            threshold_confidence: Soglia minima di confidenza (default 0.45)
            semantic_weight: Peso dell'analisi semantica (0-1)
            numeric_weight: Peso dell'analisi numerica (0-1)
            mode: Modalità di funzionamento
        """
        self.threshold = threshold_confidence
        self.semantic_weight = semantic_weight
        self.numeric_weight = numeric_weight
        self.mode = mode
        
        # Applica le costanti in base alla modalità
        if mode == "conservative":
            self.NO_TRADE_LOW = self.NO_TRADE_LOW
            self.NO_TRADE_HIGH = self.NO_TRADE_HIGH
            self.MIN_CONFIDENCE = self.MIN_CONFIDENCE
        elif mode == "aggressive":
            self.NO_TRADE_LOW = self.NO_TRADE_LOW_AGGRESSIVE
            self.NO_TRADE_HIGH = self.NO_TRADE_HIGH_AGGRESSIVE
            self.MIN_CONFIDENCE = self.MIN_CONFIDENCE_AGGRESSIVE
        else:  # balanced
            self.NO_TRADE_LOW = self.NO_TRADE_LOW_BALANCED
            self.NO_TRADE_HIGH = self.NO_TRADE_HIGH_BALANCED
            self.MIN_CONFIDENCE = self.MIN_CONFIDENCE_BALANCED
            
        logger.info(f"🎯 OpportunityFilterPro inizializzato in modalità: {mode}")
        logger.info(f"   NO_TRADE_ZONE: {self.NO_TRADE_LOW}-{self.NO_TRADE_HIGH}")
        logger.info(f"   MIN_CONFIDENCE: {self.MIN_CONFIDENCE}")
        logger.info(f"   THRESHOLD: {self.threshold}")
        
        # Verifica che i pesi sommino a 1
        total_weight = semantic_weight + numeric_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.semantic_weight = semantic_weight / total_weight
            self.numeric_weight = numeric_weight / total_weight

    def analyze_semantic(self, asset_data: Dict) -> float:
        """
        Analizza trend logico/semantico: news, sentiment, eventi macro.
        """
        sentiment_score = asset_data.get("sentiment_score", 0)
        event_impact = asset_data.get("event_impact", 0)
        trend_signal = asset_data.get("trend_signal", 0)
        news_score = asset_data.get("news_score", 0)
        
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
        """
        rsi_score = asset_data.get("rsi_score", 0)
        macd_score = asset_data.get("macd_score", 0)
        volatility_score = asset_data.get("volatility_score", 0)
        momentum_score = asset_data.get("momentum_score", 0)
        volume_score = asset_data.get("volume_score", 0)
        
        score = (
            0.25 * rsi_score +
            0.25 * macd_score +
            0.20 * momentum_score +
            0.15 * volume_score -
            0.15 * volatility_score
        )
        
        return np.clip(score, -1, 1)

    def combine_scores(self, semantic: float, numeric: float) -> float:
        """Combina segnali semantici e numerici."""
        combined = (
            self.semantic_weight * semantic +
            self.numeric_weight * numeric
        )
        return np.clip(combined, -1, 1)
    
    def calculate_confidence(self, semantic: float, numeric: float) -> float:
        """
        Calcola la confidenza in modo più granulare.
        
        Problema originale: confidence = abs(combined_score) è troppo semplice
        Soluzione: considera l'accordo tra segnali semantic e numeric
        """
        combined = self.combine_scores(semantic, numeric)
        
        # Calcola l'accordo tra semantic e numeric
        # Se entrambi vanno nella stessa direzione → alta confidence
        agreement = 1 - np.abs(semantic - numeric) / 2  # 0 a 1
        
        # Confidence base
        base_confidence = np.abs(combined)
        
        # Adjust per accordo
        # Se semantic e numeric sono d'accordo, aumenta confidence
        final_confidence = base_confidence * 0.7 + agreement * 0.3
        
        return np.clip(final_confidence, 0, 1)

    def filter_assets(self, assets: List[Dict]) -> List[Dict]:
        """
        Filtra asset con logica OTTIMIZZATA.
        """
        selected_assets = []
        
        # Statistiche per debug
        stats = {
            "total": len(assets),
            "no_trade_zone": 0,
            "low_confidence": 0,
            "below_threshold": 0,
            "selected": 0
        }
        
        for asset in assets:
            # Calcola punteggi
            semantic_score = self.analyze_semantic(asset)
            numeric_score = self.analyze_numeric(asset)
            combined_score = self.combine_scores(semantic_score, numeric_score)
            
            # Normalizza score a 0-1
            normalized_score = (combined_score + 1) / 2
            
            # Calcola confidence migliorata
            confidence = self.calculate_confidence(semantic_score, numeric_score)
            
            # Check 1: NO_TRADE_ZONE
            if self.is_no_trade_zone(normalized_score):
                stats["no_trade_zone"] += 1
                logger.debug(
                    f"❌ {asset.get('name', 'Unknown')}: NO_TRADE_ZONE "
                    f"(score={normalized_score:.2f})"
                )
                continue
            
            # Check 2: MIN_CONFIDENCE
            if confidence < self.MIN_CONFIDENCE:
                stats["low_confidence"] += 1
                logger.debug(
                    f"❌ {asset.get('name', 'Unknown')}: LOW_CONF "
                    f"(conf={confidence:.2f} < {self.MIN_CONFIDENCE})"
                )
                continue
            
            # Check 3: THRESHOLD
            if abs(combined_score) < self.threshold:
                stats["below_threshold"] += 1
                logger.debug(
                    f"❌ {asset.get('name', 'Unknown')}: BELOW_THRESHOLD "
                    f"(score={combined_score:.2f} < {self.threshold})"
                )
                continue
            
            # ✅ ASSET SELEZIONATO
            stats["selected"] += 1
            asset['semantic_score'] = semantic_score
            asset['numeric_score'] = numeric_score
            asset['combined_score'] = combined_score
            asset['normalized_score'] = normalized_score
            asset['confidence'] = confidence
            
            selected_assets.append(asset)
            logger.info(
                f"✅ {asset.get('name', 'Unknown')}: SELECTED | "
                f"sem={semantic_score:.2f} | num={numeric_score:.2f} | "
                f"combined={combined_score:.2f} | conf={confidence:.2f}"
            )
        
        # Log finale statistico
        logger.info("=" * 60)
        logger.info("📊 FILTRO OPPORTUNITÀ - STATISTICHE")
        logger.info(f"   Totali: {stats['total']}")
        logger.info(f"   ❌ No-Trade Zone: {stats['no_trade_zone']}")
        logger.info(f"   ❌ Low Confidence: {stats['low_confidence']}")
        logger.info(f"   ❌ Below Threshold: {stats['below_threshold']}")
        logger.info(f"   ✅ Selezionati: {stats['selected']}")
        logger.info("=" * 60)
        
        # Ordina per confidenza decrescente
        selected_assets.sort(key=lambda x: x['confidence'], reverse=True)
        
        return selected_assets

    def is_no_trade_zone(self, normalized_score: float) -> bool:
        """Check if score is in neutral zone."""
        return self.NO_TRADE_LOW < normalized_score < self.NO_TRADE_HIGH
    
    def get_signal_direction(self, combined_score: float, confidence: float = None) -> str:
        """Determina la direzione del segnale."""
        if confidence is not None and confidence < self.MIN_CONFIDENCE:
            return "HOLD"
        
        normalized_score = (combined_score + 1) / 2
        if self.is_no_trade_zone(normalized_score):
            return "HOLD"
        
        if combined_score > 0.1:
            return "BUY"
        elif combined_score < -0.1:
            return "SELL"
        else:
            return "HOLD"


# === CONFIGURAZIONI PREDEFINE ===
def get_filter_config(mode: str = "balanced") -> dict:
    """
    Restituisce configurazione predefinita per il filtro.
    
    Args:
        mode: "conservative", "balanced", "aggressive"
    
    Returns:
        Dict con parametri
    """
    configs = {
        "conservative": {
            "threshold_confidence": 0.50,
            "no_trade_low": 0.45,
            "no_trade_high": 0.55,
            "min_confidence": 0.55,
            "description": "Molto sicuro, pochi trade"
        },
        "balanced": {
            "threshold_confidence": 0.40,
            "no_trade_low": 0.40,
            "no_trade_high": 0.60,
            "min_confidence": 0.40,
            "description": "Bilanciato - consigliato per iniziare"
        },
        "aggressive": {
            "threshold_confidence": 0.30,
            "no_trade_low": 0.30,
            "no_trade_high": 0.70,
            "min_confidence": 0.30,
            "description": "Molti trade, più rischio"
        }
    }
    
    return configs.get(mode, configs["balanced"])


# === TEST ===
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    print("\n" + "=" * 60)
    print("TEST FILTRO OPPORTUNITÀ PRO")
    print("=" * 60)
    
    # Test con dati realistici
    assets = [
        {
            "name": "BTC",
            "sentiment_score": 0.3,
            "event_impact": 0.2,
            "trend_signal": 0.4,
            "news_score": 0.3,
            "rsi_score": 0.5,
            "macd_score": 0.4,
            "volatility_score": 0.3,
            "momentum_score": 0.4,
            "volume_score": 0.3
        },
        {
            "name": "ETH",
            "sentiment_score": 0.1,
            "event_impact": 0.0,
            "trend_signal": 0.2,
            "news_score": 0.1,
            "rsi_score": 0.3,
            "macd_score": 0.2,
            "volatility_score": 0.4,
            "momentum_score": 0.2,
            "volume_score": 0.2
        },
        {
            "name": "SOL",
            "sentiment_score": -0.2,
            "event_impact": -0.1,
            "trend_signal": -0.1,
            "news_score": -0.2,
            "rsi_score": -0.3,
            "macd_score": -0.2,
            "volatility_score": 0.5,
            "momentum_score": -0.2,
            "volume_score": -0.1
        }
    ]
    
    # Test modalità balanced
    print("\n🟡 MODALITÀ BALANCED")
    filtro = OpportunityFilterPro(mode="balanced")
    selezionati = filtro.filter_assets(assets)
    
    print("\n📋 RISULTATI:")
    for asset in selezionati:
        direction = filtro.get_signal_direction(asset['combined_score'], asset['confidence'])
        print(f"   {asset['name']:6} | {direction:4} | score={asset['combined_score']:.2f} | conf={asset['confidence']:.2f}")
