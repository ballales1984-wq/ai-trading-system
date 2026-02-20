from typing import List, Dict
import numpy as np

class MonteCarloSimulator:
    """
    Valuta il rischio e la probabilità di profitto usando simulazioni Monte Carlo
    """
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations

    def simulate_asset(self, asset_data: Dict) -> float:
        """
        Genera un punteggio basato sulla simulazione Monte Carlo dei possibili ritorni
        """
        expected_return = asset_data.get("expected_return", 0.0)
        volatility = asset_data.get("volatility", 0.1)
        simulations = np.random.normal(loc=expected_return, scale=volatility, size=self.num_simulations)
        prob_positive = np.mean(simulations > 0)
        monte_score = 2*prob_positive - 1  # normalizza da -1 a 1
        return monte_score

class OpportunityFilter:
    def __init__(self, threshold_confidence=0.6):
        self.threshold = threshold_confidence
        self.mc = MonteCarloSimulator()

    def analyze_semantic(self, asset_data: Dict) -> float:
        sentiment_score = asset_data.get("sentiment_score", 0)
        event_impact = asset_data.get("event_impact", 0)
        trend_signal = asset_data.get("trend_signal", 0)
        score = 0.5*sentiment_score + 0.3*event_impact + 0.2*trend_signal
        return np.clip(score, -1, 1)

    def analyze_numeric(self, asset_data: Dict) -> float:
        rsi_score = asset_data.get("rsi_score", 0)
        macd_score = asset_data.get("macd_score", 0)
        volatility_score = asset_data.get("volatility_score", 0)
        score = 0.4*rsi_score + 0.4*macd_score - 0.2*volatility_score
        return np.clip(score, -1, 1)

    def analyze_montecarlo(self, asset_data: Dict) -> float:
        return self.mc.simulate_asset(asset_data)

    def combine_scores(self, semantic: float, numeric: float, monte: float) -> float:
        # Peso uguale tra logica, matematica e simulazioni
        return np.clip((semantic + numeric + monte)/3, -1, 1)

    def filter_assets(self, assets: List[Dict]) -> List[Dict]:
        selected_assets = []
        for asset in assets:
            semantic_score = self.analyze_semantic(asset)
            numeric_score = self.analyze_numeric(asset)
            monte_score = self.analyze_montecarlo(asset)
            combined_score = self.combine_scores(semantic_score, numeric_score, monte_score)
            
            asset['semantic_score'] = semantic_score
            asset['numeric_score'] = numeric_score
            asset['monte_score'] = monte_score
            asset['combined_score'] = combined_score

            if abs(combined_score) >= self.threshold:
                selected_assets.append(asset)
        return selected_assets

class DecisionEngine:
    def __init__(self, portfolio_balance: float = 100000):
        self.portfolio = {"cash": portfolio_balance, "positions": {}}
        self.filter = OpportunityFilter(threshold_confidence=0.6)

    def generate_orders(self, assets: List[Dict]) -> List[Dict]:
        orders = []
        filtered_assets = self.filter.filter_assets(assets)
        for asset in filtered_assets:
            score = asset['combined_score']
            if score > 0:
                action = "BUY"
                amount = self.portfolio['cash'] * score
                self.portfolio['cash'] -= amount
                self.portfolio['positions'][asset['name']] = self.portfolio['positions'].get(asset['name'], 0) + amount
            elif score < 0:
                action = "SELL"
                amount = self.portfolio['positions'].get(asset['name'], 0) * abs(score)
                self.portfolio['positions'][asset['name']] = self.portfolio['positions'].get(asset['name'], 0) - amount
                self.portfolio['cash'] += amount
            else:
                action = "HOLD"
                amount = 0
            orders.append({
                "asset": asset['name'],
                "action": action,
                "amount": amount,
                "confidence": abs(score)
            })
        return orders

# ------------------------------
# Esempio di utilizzo
if __name__ == "__main__":
    assets = [
        {"name": "Oro", "sentiment_score": 0.8, "event_impact": 0.3, "trend_signal": 0.5,
         "rsi_score": 0.7, "macd_score": 0.6, "volatility_score": 0.2,
         "expected_return": 0.05, "volatility": 0.1},
        {"name": "Rame", "sentiment_score": -0.4, "event_impact": -0.2, "trend_signal": -0.1,
         "rsi_score": -0.5, "macd_score": -0.4, "volatility_score": 0.1,
         "expected_return": -0.02, "volatility": 0.08},
        {"name": "BTC", "sentiment_score": 0.1, "event_impact": 0.0, "trend_signal": 0.2,
         "rsi_score": 0.3, "macd_score": 0.2, "volatility_score": 0.3,
         "expected_return": 0.03, "volatility": 0.25},
    ]

    engine = DecisionEngine(portfolio_balance=100000)
    orders = engine.generate_orders(assets)

    for order in orders:
        print(f"{order['action']} {order['asset']} -> Amount: {order['amount']:.2f}, Confidence: {order['confidence']:.2f}")

    print("\nPortafoglio finale:", engine.portfolio)
