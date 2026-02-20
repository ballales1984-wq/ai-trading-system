"""
Automatic Decision Engine
=========================
Modulo per la generazione automatica di ordini basata su analisi duale
(semantica + numerica) con integrazione Monte Carlo per scenario analysis.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
from datetime import datetime

from .filtro_opportunita import OpportunityFilter

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Simulatore Monte Carlo per analisi di scenario.
    """
    
    def __init__(
        self,
        n_simulations: int = 1000,
        confidence_level: float = 0.95
    ):
        """
        Inizializza il simulatore Monte Carlo.
        
        Args:
            n_simulations: Numero di simulazioni da eseguire
            confidence_level: Livello di confidenza per VaR
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
    
    def simulate_price_path(
        self,
        current_price: float,
        volatility: float,
        drift: float = 0.0,
        time_horizon: int = 30
    ) -> np.ndarray:
        """
        Simula percorsi di prezzo usando moto browniano geometrico.
        
        Args:
            current_price: Prezzo corrente
            volatility: Volatilità annualizzata
            drift: Tasso di drift (rendimento atteso)
            time_horizon: Orizzonte temporale in giorni
            
        Returns:
            Array di prezzi simulati
        """
        dt = 1 / 365  # Time step giornaliero
        
        # Genera random shocks
        random_shocks = np.random.normal(0, 1, (self.n_simulations, time_horizon))
        
        # Calcola percorsi
        price_paths = np.zeros((self.n_simulations, time_horizon + 1))
        price_paths[:, 0] = current_price
        
        for t in range(time_horizon):
            price_paths[:, t + 1] = price_paths[:, t] * np.exp(
                (drift - 0.5 * volatility ** 2) * dt +
                volatility * np.sqrt(dt) * random_shocks[:, t]
            )
        
        return price_paths
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: Optional[float] = None
    ) -> float:
        """
        Calcola il Value at Risk.
        
        Args:
            returns: Array di rendimenti simulati
            confidence_level: Livello di confidenza (opzionale)
            
        Returns:
            VaR come percentuale
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_expected_shortfall(
        self,
        returns: np.ndarray,
        confidence_level: Optional[float] = None
    ) -> float:
        """
        Calcola l'Expected Shortfall (CVaR).
        
        Args:
            returns: Array di rendimenti simulati
            confidence_level: Livello di confidenza (opzionale)
            
        Returns:
            ES come percentuale
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        var = self.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def assess_trade_risk(
        self,
        asset_data: Dict,
        position_size: float
    ) -> Dict:
        """
        Valuta il rischio di un trade usando Monte Carlo.
        
        Args:
            asset_data: Dati dell'asset
            position_size: Dimensione della posizione in USDT
            
        Returns:
            Dizionario con metriche di rischio
        """
        current_price = asset_data.get("price", 100)
        volatility = asset_data.get("volatility_annual", 0.5)  # Default 50% annual
        
        # Simula percorsi di prezzo
        price_paths = self.simulate_price_path(
            current_price=current_price,
            volatility=volatility,
            drift=asset_data.get("expected_return", 0.0),
            time_horizon=30
        )
        
        # Calcola rendimenti finali
        final_prices = price_paths[:, -1]
        returns = (final_prices - current_price) / current_price
        
        # Calcola metriche di rischio
        var = self.calculate_var(returns)
        es = self.calculate_expected_shortfall(returns)
        
        # Probabilità di profitto
        prob_profit = np.mean(returns > 0)
        
        # Rendimento atteso
        expected_return = np.mean(returns)
        
        # Perdita massima attesa (in USDT)
        max_loss_usdt = position_size * abs(es)
        
        return {
            "var": var,
            "expected_shortfall": es,
            "prob_profit": prob_profit,
            "expected_return": expected_return,
            "max_loss_usdt": max_loss_usdt,
            "n_simulations": self.n_simulations
        }


class DecisionEngine:
    """
    Engine decisionale automatico che combina:
    - Filtro opportunità (semantico + numerico)
    - Simulazione Monte Carlo per risk assessment
    - Generazione automatica di ordini
    """
    
    def __init__(
        self,
        portfolio_balance: float = 100000,
        threshold_confidence: float = 0.6,
        max_risk_per_trade: float = 0.02,
        semantic_weight: float = 0.5,
        numeric_weight: float = 0.5,
        monte_carlo_sims: int = 1000
    ):
        """
        Inizializza il Decision Engine.
        
        Args:
            portfolio_balance: Bilancio iniziale del portafoglio
            threshold_confidence: Soglia minima di confidenza
            max_risk_per_trade: Rischio massimo per trade (% del portafoglio)
            semantic_weight: Peso analisi semantica
            numeric_weight: Peso analisi numerica
            monte_carlo_sims: Numero simulazioni Monte Carlo
        """
        self.portfolio = {
            "cash": portfolio_balance,
            "positions": {},
            "history": []
        }
        
        self.filter = OpportunityFilter(
            threshold_confidence=threshold_confidence,
            semantic_weight=semantic_weight,
            numeric_weight=numeric_weight
        )
        
        self.monte_carlo = MonteCarloSimulator(n_simulations=monte_carlo_sims)
        self.max_risk_per_trade = max_risk_per_trade
        
        # Statistiche
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0
        }

    def calculate_position_size(
        self,
        asset_data: Dict,
        combined_score: float
    ) -> float:
        """
        Calcola la dimensione della posizione basata su confidenza e rischio.
        
        Args:
            asset_data: Dati dell'asset
            combined_score: Punteggio combinato
            
        Returns:
            Dimensione posizione in USDT
        """
        # Base: percentuale del portafoglio proporzionale alla confidenza
        base_size = self.portfolio["cash"] * abs(combined_score) * 0.1
        
        # Limita al rischio massimo per trade
        max_risk_amount = self.portfolio["cash"] * self.max_risk_per_trade
        
        # Considera volatilità per aggiustare size
        volatility = asset_data.get("volatility_score", 0.5)
        volatility_adjustment = max(0.5, 1 - volatility)
        
        final_size = min(base_size * volatility_adjustment, max_risk_amount * 5)
        
        return max(0, min(final_size, self.portfolio["cash"]))

    def generate_orders(self, assets: List[Dict]) -> List[Dict]:
        """
        Genera ordini BUY/SELL/HOLD basati su analisi completa.
        
        Args:
            assets: Lista di asset con dati semantici e numerici
            
        Returns:
            Lista di ordini generati
        """
        orders = []
        
        # Filtra asset con potenziale
        filtered_assets = self.filter.filter_assets(assets)
        
        for asset in filtered_assets:
            score = asset['combined_score']
            direction = self.filter.get_signal_direction(score)
            
            if direction == "HOLD":
                continue
            
            # Calcola dimensione posizione
            position_size = self.calculate_position_size(asset, score)
            
            if position_size <= 0:
                continue
            
            # Valuta rischio con Monte Carlo
            risk_assessment = self.monte_carlo.assess_trade_risk(asset, position_size)
            
            # Verifica se il rischio è accettabile
            max_acceptable_loss = self.portfolio["cash"] * self.max_risk_per_trade
            if risk_assessment["max_loss_usdt"] > max_acceptable_loss:
                # Riduci dimensione posizione
                position_size *= max_acceptable_loss / risk_assessment["max_loss_usdt"]
                risk_assessment = self.monte_carlo.assess_trade_risk(asset, position_size)
            
            # Genera ordine
            order = {
                "asset": asset.get("name", "Unknown"),
                "action": direction,
                "amount": round(position_size, 2),
                "confidence": abs(score),
                "semantic_score": asset.get("semantic_score", 0),
                "numeric_score": asset.get("numeric_score", 0),
                "monte_carlo": risk_assessment,
                "timestamp": datetime.now().isoformat()
            }
            
            # Aggiorna portafoglio (simulato)
            self._update_portfolio(order)
            
            orders.append(order)
            
            logger.info(
                f"Order generated: {direction} {position_size:.2f} USDT of {asset.get('name')} "
                f"(confidence: {abs(score):.2f}, VaR: {risk_assessment['var']:.2%})"
            )
        
        return orders

    def _update_portfolio(self, order: Dict):
        """
        Aggiorna lo stato del portafoglio dopo un ordine.
        
        Args:
            order: Ordine eseguito
        """
        asset = order["asset"]
        action = order["action"]
        amount = order["amount"]
        
        if action == "BUY":
            if amount > self.portfolio["cash"]:
                logger.warning(f"Insufficient cash for BUY order: {amount} > {self.portfolio['cash']}")
                return
            
            self.portfolio["cash"] -= amount
            self.portfolio["positions"][asset] = self.portfolio["positions"].get(asset, 0) + amount
            
        elif action == "SELL":
            current_position = self.portfolio["positions"].get(asset, 0)
            sell_amount = min(amount, current_position)
            
            self.portfolio["cash"] += sell_amount
            self.portfolio["positions"][asset] = current_position - sell_amount
            
            if self.portfolio["positions"][asset] <= 0:
                del self.portfolio["positions"][asset]
        
        # Registra in storico
        self.portfolio["history"].append({
            "timestamp": order["timestamp"],
            "asset": asset,
            "action": action,
            "amount": amount,
            "cash_after": self.portfolio["cash"]
        })
        
        self.stats["total_trades"] += 1

    def get_portfolio_summary(self) -> Dict:
        """
        Ritorna un riepilogo del portafoglio.
        
        Returns:
            Dizionario con riepilogo
        """
        total_position_value = sum(self.portfolio["positions"].values())
        
        return {
            "cash": round(self.portfolio["cash"], 2),
            "positions": dict(self.portfolio["positions"]),
            "total_position_value": round(total_position_value, 2),
            "total_value": round(self.portfolio["cash"] + total_position_value, 2),
            "n_positions": len(self.portfolio["positions"]),
            "stats": self.stats.copy()
        }

    def run_trading_cycle(
        self,
        assets: List[Dict],
        execute: bool = True
    ) -> Dict:
        """
        Esegue un ciclo completo di trading.
        
        Args:
            assets: Lista di asset da analizzare
            execute: Se True, aggiorna il portafoglio
            
        Returns:
            Dizionario con risultati del ciclo
        """
        logger.info("=" * 60)
        logger.info("TRADING CYCLE STARTED")
        logger.info("=" * 60)
        
        # Genera ordini
        orders = self.generate_orders(assets)
        
        # Riepilogo
        summary = self.get_portfolio_summary()
        
        result = {
            "orders": orders,
            "portfolio": summary,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Trading cycle completed: {len(orders)} orders generated")
        logger.info(f"Portfolio value: {summary['total_value']} USDT")
        
        return result


# === Esempio di utilizzo ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Dati di esempio
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
            "volume_score": 0.4,
            "price": 2000,
            "volatility_annual": 0.15,
            "expected_return": 0.08
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
            "volume_score": -0.2,
            "price": 4.5,
            "volatility_annual": 0.25,
            "expected_return": -0.05
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
            "volume_score": 0.2,
            "price": 95000,
            "volatility_annual": 0.6,
            "expected_return": 0.15
        },
    ]

    # Crea engine
    engine = DecisionEngine(
        portfolio_balance=100000,
        threshold_confidence=0.6,
        max_risk_per_trade=0.02,
        monte_carlo_sims=1000
    )
    
    # Esegui ciclo di trading
    result = engine.run_trading_cycle(assets)
    
    # Stampa risultati
    print("\n" + "=" * 60)
    print("ORDERS GENERATED")
    print("=" * 60)
    for order in result["orders"]:
        mc = order["monte_carlo"]
        print(f"  {order['action']:4} {order['amount']:>10.2f} USDT of {order['asset']}")
        print(f"       Confidence: {order['confidence']:.2f}")
        print(f"       VaR (95%): {mc['var']:.2%}")
        print(f"       Prob Profit: {mc['prob_profit']:.1%}")
        print()
    
    print("=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)
    summary = result["portfolio"]
    print(f"  Cash: {summary['cash']:,.2f} USDT")
    print(f"  Positions: {summary['n_positions']}")
    print(f"  Total Value: {summary['total_value']:,.2f} USDT")
    print("=" * 60)
