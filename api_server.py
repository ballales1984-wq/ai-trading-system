"""
REST API Server for Frontend Integration
Provides JSON endpoints for the Java Spring Boot frontend
"""

import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
import config
from data_collector import DataCollector
from decision_engine import DecisionEngine
from trading_simulator import TradingSimulator
from sentiment_news import SentimentAnalyzer
from technical_analysis import TechnicalAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
data_collector = DataCollector(simulation=True)
decision_engine = DecisionEngine(data_collector=data_collector)
trading_simulator = TradingSimulator(initial_balance=500000)
sentiment_analyzer = SentimentAnalyzer()
technical_analyzer = TechnicalAnalyzer()


# ==================== API ENDPOINTS ====================

@app.route('/api/market-data', methods=['GET'])
def get_market_data():
    """Get market data for a symbol"""
    symbol = request.args.get('symbol', 'BTC/USDT')
    
    try:
        market_data = data_collector.fetch_market_data(symbol)
        
        return jsonify({
            'symbol': symbol,
            'price': market_data.current_price,
            'bid': market_data.bid,
            'ask': market_data.ask,
            'volume': market_data.volume_24h,
            'change_24h': getattr(market_data, 'change_24h', 0),
            'timestamp': market_data.timestamp.isoformat() if hasattr(market_data, 'timestamp') else None
        })
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/signals', methods=['GET'])
def get_signals():
    """Get trading signals"""
    try:
        symbols = data_collector.get_supported_symbols()
        signals = decision_engine.generate_signals(symbols)
        
        signal_list = []
        for signal in signals:
            signal_list.append({
                'symbol': signal.symbol,
                'action': signal.action,
                'confidence': signal.confidence,
                'strength': signal.strength,
                'current_price': signal.current_price,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'technical_score': signal.technical_score,
                'sentiment_score': signal.sentiment_score,
                'ml_score': signal.ml_score,
                'reason': signal.reason,
                'timestamp': signal.timestamp.isoformat() if hasattr(signal, 'timestamp') else None
            })
        
        return jsonify(signal_list)
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Get current portfolio"""
    try:
        portfolio = trading_simulator.check_portfolio()
        
        return jsonify({
            'balance': portfolio.get('balance', 0),
            'total_value': portfolio.get('total_value', 0),
            'initial_balance': portfolio.get('initial_balance', 0),
            'total_pnl': portfolio.get('total_pnl', 0),
            'total_pnl_percent': portfolio.get('pnl_percent', 0),
            'positions': [],
            'trades': []
        })
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/execute-trade', methods=['POST'])
def execute_trade():
    """Execute a trade"""
    data = request.get_json()
    
    symbol = data.get('symbol')
    side = data.get('side')  # BUY or SELL
    quantity = data.get('quantity', 0.01)
    
    if not symbol or not side:
        return jsonify({'error': 'Missing symbol or side'}), 400
    
    try:
        # Get current price
        market_data = data_collector.fetch_market_data(symbol)
        price = market_data.current_price
        
        # Execute trade
        if side.upper() == 'BUY':
            result = trading_simulator.buy(symbol, quantity, price)
        else:
            result = trading_simulator.sell(symbol, quantity, price)
        
        return jsonify({
            'success': result.get('success', False),
            'message': result.get('message', ''),
            'trade_id': result.get('trade_id', ''),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price
        })
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    """Get supported symbols"""
    try:
        symbols = data_collector.get_supported_symbols()
        return jsonify(symbols)
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'version': '1.0.0',
        'mode': 'simulation' if config.SIMULATION_MODE else 'live'
    })


# ==================== MAIN ====================

if __name__ == '__main__':
    logger.info("Starting API Server for Frontend Integration")
    logger.info(f"Mode: {'Simulation' if config.SIMULATION_MODE else 'Live'}")
    logger.info(f"Supported symbols: {data_collector.get_supported_symbols()}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
