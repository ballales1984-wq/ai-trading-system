"""
Real-Time Multi-Asset Execution Dashboard
=========================================
Dashboard connected to execution module showing:
- Generated signals (BUY/SELL) with confidence
- Assets involved and calculated quantity
- Selected exchange
- Order status (sent, executed, failed)
- Real-time portfolio updates

Author: AI Trading System
Version: 1.0.0
"""

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import logging
import random
import threading
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# Import our existing modules
from logical_portfolio_module import Portfolio, LogicalPortfolioEngine, NewsItem
from logical_math_multiasset import MathDecisionEngineMultiAsset, IntegratedDecisionSystemMultiAsset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ==================== ORDER STATUS TRACKING ====================

class OrderStatus:
    """Track order execution status"""
    PENDING = "pending"
    SENT = "sent"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    asset: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    exchange: str
    status: str = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    executed_price: float = 0.0
    error_message: str = ""

    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'asset': self.asset,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'exchange': self.exchange,
            'status': self.status,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'executed_price': self.executed_price,
            'error_message': self.error_message
        }


class OrderManager:
    """Manages order tracking and execution simulation"""
    
    def __init__(self, exchange: str = "Binance"):
        self.exchange = exchange
        self.orders: List[Order] = []
        self._order_counter = 0
        self._lock = threading.Lock()
    
    def create_order(self, asset: str, side: str, quantity: float, price: float) -> Order:
        """Create a new order"""
        with self._lock:
            self._order_counter += 1
            order = Order(
                order_id=f"ORD-{self._order_counter:06d}",
                asset=asset,
                side=side,
                quantity=quantity,
                price=price,
                exchange=self.exchange,
                status=OrderStatus.PENDING
            )
            self.orders.append(order)
            return order
    
    def send_order(self, order: Order) -> Order:
        """Simulate sending order to exchange"""
        order.status = OrderStatus.SENT
        logger.info(f"Order {order.order_id} sent to {self.exchange}")
        return order
    
    def execute_order(self, order: Order, executed_price: float) -> Order:
        """Simulate order execution"""
        order.status = OrderStatus.EXECUTED
        order.executed_price = executed_price
        logger.info(f"Order {order.order_id} executed at {executed_price}")
        return order
    
    def fail_order(self, order: Order, error: str) -> Order:
        """Simulate order failure"""
        order.status = OrderStatus.FAILED
        order.error_message = error
        logger.warning(f"Order {order.order_id} failed: {error}")
        return order
    
    def get_orders(self, limit: int = 50) -> List[Dict]:
        """Get recent orders"""
        with self._lock:
            return [o.to_dict() for o in self.orders[-limit:]]
    
    def get_order_stats(self) -> Dict:
        """Get order statistics"""
        with self._lock:
            total = len(self.orders)
            if total == 0:
                return {'total': 0, 'executed': 0, 'failed': 0, 'pending': 0, 'success_rate': 0}
            
            executed = sum(1 for o in self.orders if o.status == OrderStatus.EXECUTED)
            failed = sum(1 for o in self.orders if o.status == OrderStatus.FAILED)
            pending = sum(1 for o in self.orders if o.status in [OrderStatus.PENDING, OrderStatus.SENT])
            
            return {
                'total': total,
                'executed': executed,
                'failed': failed,
                'pending': pending,
                'success_rate': (executed / total * 100) if total > 0 else 0
            }


# ==================== EXECUTION ENGINE ====================

class MultiAssetExecutionEngine:
    """
    Multi-Asset Execution Engine
    Connects the decision system with order execution
    """
    
    def __init__(self, portfolio: Portfolio, use_testnet: bool = True):
        self.portfolio = portfolio
        self.use_testnet = use_testnet
        self.exchange = "Binance Testnet" if use_testnet else "Binance"
        
        # Initialize decision systems
        self.logical_engine = LogicalPortfolioEngine(portfolio)
        self.math_engine = MathDecisionEngineMultiAsset(portfolio)
        
        # Order manager
        self.order_manager = OrderManager(exchange=self.exchange)
        
        # Signal history
        self.signal_history: List[Dict] = []
        
        logger.info(f"MultiAssetExecutionEngine initialized (Exchange: {self.exchange})")
    
    def process_news_feed(self, news_feed: List[NewsItem]) -> List[Dict]:
        """Process news and generate signals"""
        # Step 1: Logical analysis
        logical_signals = self.logical_engine.analyze_news(news_feed)
        
        # Step 2: Mathematical evaluation
        final_signals = []
        for sig in logical_signals:
            final = self.math_engine.evaluate_signal(sig)
            final["source"] = sig.get("source", "Unknown")
            final["title"] = sig.get("title", "")
            final["sentiment"] = sig.get("sentiment", 0)
            final_signals.append(final)
            
            # Add to history
            self.signal_history.append({
                **final,
                'timestamp': datetime.now().isoformat()
            })
        
        return final_signals
    
    def execute_signal(self, signal: Dict) -> Optional[Order]:
        """Execute a trading signal"""
        if signal.get('final_signal') == 'HOLD' or not signal.get('can_execute'):
            return None
        
        asset = signal['asset']
        side = signal['final_signal']
        
        # Get current price (simulated)
        current_price = self._get_simulated_price(asset)
        
        # Calculate quantity based on confidence and portfolio value
        portfolio_value = self.portfolio.total_value()
        confidence = signal.get('adjusted_confidence', 0.5)
        
        # Risk-based position sizing
        position_pct = min(confidence * 0.1, 0.3)  # Max 30% of portfolio
        position_value = portfolio_value * position_pct
        quantity = position_value / current_price if current_price > 0 else 0
        
        # Create order
        order = self.order_manager.create_order(asset, side, quantity, current_price)
        
        # Simulate order execution
        order = self.order_manager.send_order(order)
        
        # Simulate 80% success rate
        if random.random() < 0.8:
            # Simulate price slippage
            slippage = random.uniform(-0.001, 0.001)
            executed_price = current_price * (1 + slippage)
            order = self.order_manager.execute_order(order, executed_price)
            
            # Update portfolio balance
            if side == "BUY":
                self.portfolio.update_balance(asset, quantity)
            else:
                self.portfolio.update_balance(asset, -quantity)
        else:
            order = self.order_manager.fail_order(order, "Insufficient liquidity")
        
        return order
    
    def _get_simulated_price(self, asset: str) -> float:
        """Get simulated price for asset"""
        prices = {
            'BTC': 95000,
            'ETH': 3500,
            'SOL': 180,
            'ADA': 0.8,
            'XRP': 2.5,
            'DOT': 7.5,
            'AVAX': 38,
            'MATIC': 0.45,
        }
        # Add some randomness
        base = prices.get(asset, 100)
        return base * random.uniform(0.98, 1.02)
    
    def get_signals(self, limit: int = 20) -> List[Dict]:
        """Get recent signals"""
        return self.signal_history[-limit:]
    
    def get_portfolio_state(self) -> Dict:
        """Get current portfolio state"""
        return {
            'balances': self.portfolio.balances.copy(),
            'total_value': self.portfolio.total_value(),
            'prices': self.portfolio._prices.copy() if hasattr(self.portfolio, '_prices') else {}
        }


# ==================== SAMPLE NEWS FEED ====================

def get_sample_news_feed() -> List[NewsItem]:
    """Generate sample news feed for demonstration"""
    news_items = [
        NewsItem("Bitcoin Surges Past $95K on ETF Inflows", "CoinDesk", asset="BTC"),
        NewsItem("Ethereum Upgrade Boosts Network Activity", "CoinTelegraph", asset="ETH"),
        NewsItem("Solana DeFi TVL Reaches New High", "The Block", asset="SOL"),
        NewsItem("Fed Signals Potential Rate Cut in March", "Reuters", asset="BTC"),
        NewsItem("Cardano Smart Contract Adoption Increases", "CoinTelegraph", asset="ADA"),
        NewsItem("Ripple Partners with Major Bank for Cross-Border Payments", "Bloomberg", asset="XRP"),
        NewsItem("Polkadot Parachain Auctions Attract Record Interest", "CoinDesk", asset="DOT"),
        NewsItem("Avalanche Foundation Launches $50M Gaming Fund", "The Block", asset="AVAX"),
    ]
    return news_items


# ==================== DASHBOARD APPLICATION ====================

# Initialize global components
portfolio = Portfolio(balances={
    "BTC": 1.5,
    "ETH": 15.0,
    "SOL": 100.0,
    "ADA": 5000.0,
    "USDT": 50000.0
})

# Set initial prices
portfolio.set_price("BTC", 95000)
portfolio.set_price("ETH", 3500)
portfolio.set_price("SOL", 180)
portfolio.set_price("ADA", 0.8)
portfolio.set_price("USDT", 1.0)

# Initialize execution engine
engine = MultiAssetExecutionEngine(portfolio, use_testnet=True)

# Process initial news feed
initial_news = get_sample_news_feed()
initial_signals = engine.process_news_feed(initial_news)

# Execute initial signals
for signal in initial_signals:
    if signal.get('can_execute') and signal.get('final_signal') != 'HOLD':
        engine.execute_signal(signal)

# Create Dash app
app = dash.Dash(
    __name__,
    title="🚀 Multi-Asset Execution Dashboard",
    update_title=None
)

# Theme colors
THEME = {
    'background': '#0d1117',
    'card': '#161b22',
    'border': '#30363d',
    'text': '#c9d1d9',
    'text_muted': '#8b949e',
    'green': '#3fb950',
    'red': '#f85149',
    'blue': '#58a6ff',
    'purple': '#a371f7',
    'orange': '#f0883e',
    'yellow': '#d29922',
}

# App Layout with Enhanced UX
app.layout = html.Div([
    # Loading Overlay
    html.Div([
        html.Div([
            html.Div(className="loading-spinner-large"),
            html.Div("Loading dashboard...", className="loading-text"),
        ], className="loading-content")
    ], id="loading-overlay", className="loading-overlay"),
    
    # Toast Container
    html.Div(id="toast-container", className="toast-container"),
    
    # Header
    html.Div([
        html.Div([
            html.H1("🚀 Multi-Asset Execution Dashboard", 
                   style={'margin': 0, 'color': THEME['blue'], 'font-size': '28px'}),
            html.P("Real-time signals, order execution & portfolio tracking",
                  style={'margin': '5px 0 0 0', 'color': THEME['text_muted']})
        ], style={'flex': '1'}),
        
        # Exchange Status + Last Updated
        html.Div([
            html.Div([
                html.Span("●", style={'color': THEME['green'], 'margin-right': '8px'}),
                html.Span(f"Exchange: {engine.exchange}", style={'color': THEME['text']})
            ], style={'margin-bottom': '5px'}),
            html.Div([
                html.Span("Mode: ", style={'color': THEME['text_muted']}),
                html.Span("Paper Trading", style={'color': THEME['green'], 'font-weight': 'bold'})
            ]),
            html.Div(id="last-updated", className="last-updated", style={'margin-top': '8px', 'font-size': '11px'})
        ], style={'text-align': 'right', 'padding': '10px'})
    ], style={
        'background': 'linear-gradient(135deg, #161b22 0%, #1f2937 100%)',
        'padding': '20px',
        'border-bottom': f'1px solid {THEME["border"]}',
        'display': 'flex',
        'align-items': 'center'
    }),
    
    # Auto-refresh interval
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0),
    
    # Main Content
    html.Div([
        # Stats Row
        html.Div([
            html.Div([
                html.Div("Total Portfolio Value", style={'color': THEME['text_muted'], 'font-size': '14px'}),
                html.Div(id='total-value', style={'color': THEME['green'], 'font-size': '28px', 'font-weight': 'bold'})
            ], className="stat-card", style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px', 
                     'border': f'1px solid {THEME["border"]}', 'text-align': 'center', 'flex': '1'}),
            
            html.Div([
                html.Div("Open Orders", style={'color': THEME['text_muted'], 'font-size': '14px'}),
                html.Div(id='open-orders', style={'color': THEME['blue'], 'font-size': '28px', 'font-weight': 'bold'})
            ], className="stat-card", style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'text-align': 'center', 'flex': '1'}),
            
            html.Div([
                html.Div("Executed Orders", style={'color': THEME['text_muted'], 'font-size': '14px'}),
                html.Div(id='executed-orders', style={'color': THEME['green'], 'font-size': '28px', 'font-weight': 'bold'})
            ], className="stat-card", style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'text-align': 'center', 'flex': '1'}),
            
            html.Div([
                html.Div("Failed Orders", style={'color': THEME['text_muted'], 'font-size': '14px'}),
                html.Div(id='failed-orders', style={'color': THEME['red'], 'font-size': '28px', 'font-weight': 'bold'})
            ], className="stat-card", style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'text-align': 'center', 'flex': '1'}),
            
            html.Div([
                html.Div("Success Rate", style={'color': THEME['text_muted'], 'font-size': '14px'}),
                html.Div(id='success-rate', style={'color': THEME['purple'], 'font-size': '28px', 'font-weight': 'bold'})
            ], className="stat-card", style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'text-align': 'center', 'flex': '1'}),
        ], className="stats-row", style={'display': 'flex', 'gap': '15px', 'margin-bottom': '20px'}),
        
        # Charts Row
        html.Div([
            # Signals Chart
            html.Div([
                html.H3("📊 Signal Confidence by Asset", style={'color': THEME['text'], 'margin-bottom': '10px'}),
                dcc.Graph(id='signals-chart', style={'height': '300px'})
            ], style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'flex': '1'}),
            
            # Portfolio Distribution
            html.Div([
                html.H3("💰 Portfolio Distribution", style={'color': THEME['text'], 'margin-bottom': '10px'}),
                dcc.Graph(id='portfolio-chart', style={'height': '300px'})
            ], style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'margin-bottom': '20px'}),
        
        # Tables Row
        html.Div([
            # Signals Table
            html.Div([
                html.H3("📈 Trading Signals", style={'color': THEME['text'], 'margin-bottom': '10px'}),
                html.Div(id='signals-table', style={'max-height': '350px', 'overflow-y': 'auto'})
            ], style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'flex': '1'}),
            
            # Orders Table
            html.Div([
                html.H3("📋 Order Status", style={'color': THEME['text'], 'margin-bottom': '10px'}),
                html.Div(id='orders-table', style={'max-height': '350px', 'overflow-y': 'auto'})
            ], style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'margin-bottom': '20px'}),
        
        # Portfolio Holdings
        html.Div([
            html.H3("💼 Portfolio Holdings", style={'color': THEME['text'], 'margin-bottom': '10px'}),
            html.Div(id='portfolio-holdings', style={'max-height': '250px', 'overflow-y': 'auto'})
        ], style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                 'border': f'1px solid {THEME["border"]}'}),
        
    ], style={'padding': '20px', 'background': THEME['background'], 'min-height': '100vh'}),
    
    # Hidden store for data
    dcc.Store(id='signals-data'),
    dcc.Store(id='orders-data'),
    dcc.Store(id='portfolio-data'),
    
], style={'background': THEME['background'], 'font-family': 'Arial, sans-serif'})


# ==================== CALLBACKS ====================

@app.callback(
    [Output('total-value', 'children'),
     Output('open-orders', 'children'),
     Output('executed-orders', 'children'),
     Output('failed-orders', 'children'),
     Output('success-rate', 'children'),
     Output('signals-data', 'data'),
     Output('orders-data', 'data'),
     Output('portfolio-data', 'data')],
    [Input('interval-component', 'n_intervals')],
    [State('signals-data', 'data'),
     State('orders-data', 'data'),
     State('portfolio-data', 'data')]
)
def update_dashboard(n, current_signals, current_orders, current_portfolio):
    try:
        # Simulate new signal processing periodically
        if n > 0 and n % 3 == 0:  # Every 3 intervals
            news = get_sample_news_feed()
            signals = engine.process_news_feed(news)
            
            # Execute signals
            for sig in signals:
                if sig.get('can_execute') and sig.get('final_signal') != 'HOLD':
                    engine.execute_signal(sig)
        
        # Get data
        order_stats = engine.order_manager.get_order_stats()
        signals = engine.get_signals(20)
        portfolio_state = engine.get_portfolio_state()
        orders = engine.order_manager.get_orders(20)
        
        # Format values
        total_value = f"${portfolio_state['total_value']:,.2f}"
        open_orders = str(order_stats['pending'])
        executed_orders = str(order_stats['executed'])
        failed_orders = str(order_stats['failed'])
        success_rate = f"{order_stats['success_rate']:.1f}%"
        
        # Ensure we always return valid data (not None)
        # Clean signals data to ensure it's JSON serializable
        clean_signals = []
        for sig in (signals if signals else []):
            if isinstance(sig, dict):
                clean_sig = {}
                for k, v in sig.items():
                    # Handle NaN/Inf values that can't be serialized
                    if isinstance(v, float):
                        import math
                        if math.isnan(v) or math.isinf(v):
                            clean_sig[k] = 0.0
                        else:
                            clean_sig[k] = v
                    else:
                        clean_sig[k] = v
                clean_signals.append(clean_sig)
        
        return (
            total_value,
            open_orders,
            executed_orders,
            failed_orders,
            success_rate,
            clean_signals,
            orders if orders else [],
            portfolio_state
        )
    except Exception as e:
        logger.error(f"Error in update_dashboard: {e}")
        # Return safe defaults on error
        return (
            "$0.00",
            "0",
            "0",
            "0",
            "0.0%",
            current_signals if current_signals else [],
            current_orders if current_orders else [],
            current_portfolio if current_portfolio else {'balances': {}, 'total_value': 0, 'prices': {}}
        )


@app.callback(
    Output('signals-chart', 'figure'),
    [Input('signals-data', 'data')]
)
def update_signals_chart(signals):
    try:
        # Handle None or empty data
        if not signals:
            # Return empty figure with message
            fig = go.Figure()
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title='No signals available - waiting for data...',
                xaxis_title='Asset',
                yaxis_title='Confidence'
            )
            return fig
        
        # Ensure signals is a list
        if not isinstance(signals, list):
            logger.warning(f"Unexpected signals data type: {type(signals)}")
            signals = [signals] if signals else []
        
        # Filter out any invalid signal entries
        valid_signals = []
        for s in signals:
            if isinstance(s, dict) and 'asset' in s and 'adjusted_confidence' in s:
                # Ensure final_signal exists and is valid
                signal_copy = s.copy()
                if 'final_signal' not in signal_copy or signal_copy['final_signal'] is None:
                    signal_copy['final_signal'] = 'HOLD'
                # Normalize signal to uppercase
                signal_copy['final_signal'] = str(signal_copy['final_signal']).upper()
                valid_signals.append(signal_copy)
        
        if not valid_signals:
            fig = go.Figure()
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title='No valid signals available',
                xaxis_title='Asset',
                yaxis_title='Confidence'
            )
            return fig
        
        df = pd.DataFrame(valid_signals)
        
        # Ensure required columns exist
        if 'asset' not in df.columns or 'adjusted_confidence' not in df.columns:
            fig = go.Figure()
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title='Invalid signal data structure'
            )
            return fig
        
        # Create bar chart with safe color mapping
        color_map = {'BUY': '#3fb950', 'SELL': '#f85149', 'HOLD': '#8b949e'}
        fig = px.bar(
            df, 
            x='asset', 
            y='adjusted_confidence',
            color='final_signal',
            color_discrete_map=color_map,
            title='Signal Confidence by Asset'
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title='Confidence',
            xaxis_title='Asset',
            legend_title='Signal',
            font=dict(color=THEME['text']),
            height=280
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in update_signals_chart: {e}")
        # Return safe empty figure on error
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title='Error loading signals',
            xaxis_title='Asset',
            yaxis_title='Confidence'
        )
        return fig


@app.callback(
    Output('portfolio-chart', 'figure'),
    [Input('portfolio-data', 'data')]
)
def update_portfolio_chart(portfolio_data):
    if not portfolio_data or 'balances' not in portfolio_data:
        fig = go.Figure()
        fig.update_layout(template='plotly_dark', title='No portfolio data')
        return fig
    
    balances = portfolio_data.get('balances', {})
    prices = portfolio_data.get('prices', {})
    
    # Calculate values
    assets = []
    values = []
    for asset, qty in balances.items():
        price = prices.get(asset, 1.0) if asset != 'USDT' else 1.0
        value = qty * price
        if value > 0:
            assets.append(asset)
            values.append(value)
    
    if not assets:
        fig = go.Figure()
        fig.update_layout(template='plotly_dark', title='No holdings')
        return fig
    
    # Create pie chart
    fig = px.pie(
        values=values,
        names=assets,
        title='Portfolio Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=THEME['text']),
        height=280
    )
    
    return fig


@app.callback(
    Output('signals-table', 'children'),
    [Input('signals-data', 'data')]
)
def update_signals_table(signals):
    if not signals:
        return html.P("No signals available", style={'color': THEME['text_muted']})
    
    df = pd.DataFrame(signals[-10:])  # Last 10 signals
    
    # Create table
    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Time", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("Asset", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("Signal", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("Confidence", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("Can Execute", style={'color': THEME['text_muted'], 'padding': '10px'}),
            ], style={'background': '#21262d'})
        ]),
        html.Tbody([
            html.Tr([
                html.Td(s.get('timestamp', '')[:19], style={'color': THEME['text'], 'padding': '8px', 'font-size': '12px'}),
                html.Td(s.get('asset', ''), style={'color': THEME['text'], 'padding': '8px', 'font-weight': 'bold'}),
                html.Td(
                    s.get('final_signal', ''),
                    style={
                        'color': THEME['green'] if s.get('final_signal') == 'BUY' 
                                 else THEME['red'] if s.get('final_signal') == 'SELL' 
                                 else THEME['text_muted'],
                        'padding': '8px',
                        'font-weight': 'bold'
                    }
                ),
                html.Td(
                    f"{s.get('adjusted_confidence', 0):.2f}",
                    style={'color': THEME['blue'], 'padding': '8px'}
                ),
                html.Td(
                    "✅" if s.get('can_execute') else "❌",
                    style={'padding': '8px', 'text-align': 'center'}
                ),
            ]) for s in df.to_dict('records')
        ])
    ], style={'width': '100%', 'border-collapse': 'collapse', 'font-size': '13px'})


@app.callback(
    Output('orders-table', 'children'),
    [Input('orders-data', 'data')]
)
def update_orders_table(orders):
    if not orders:
        return html.P("No orders yet", style={'color': THEME['text_muted']})
    
    df = pd.DataFrame(orders[-10:])  # Last 10 orders
    
    # Create table
    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Order ID", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("Asset", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("Side", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("Quantity", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("Price", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("Status", style={'color': THEME['text_muted'], 'padding': '10px'}),
            ], style={'background': '#21262d'})
        ]),
        html.Tbody([
            html.Tr([
                html.Td(o.get('order_id', ''), style={'color': THEME['text_muted'], 'padding': '8px', 'font-size': '11px', 'font-family': 'monospace'}),
                html.Td(o.get('asset', ''), style={'color': THEME['text'], 'padding': '8px', 'font-weight': 'bold'}),
                html.Td(
                    o.get('side', ''),
                    style={
                        'color': THEME['green'] if o.get('side') == 'BUY' else THEME['red'],
                        'padding': '8px',
                        'font-weight': 'bold'
                    }
                ),
                html.Td(f"{o.get('quantity', 0):.4f}", style={'color': THEME['text'], 'padding': '8px'}),
                html.Td(f"${o.get('price', 0):,.2f}", style={'color': THEME['text'], 'padding': '8px'}),
                html.Td(
                    o.get('status', '').upper(),
                    style={
                        'color': THEME['green'] if o.get('status') == 'executed' 
                                 else THEME['red'] if o.get('status') == 'failed'
                                 else THEME['orange'] if o.get('status') == 'sent'
                                 else THEME['text_muted'],
                        'padding': '8px',
                        'font-weight': 'bold',
                        'font-size': '11px'
                    }
                ),
            ]) for o in df.to_dict('records')
        ])
    ], style={'width': '100%', 'border-collapse': 'collapse', 'font-size': '12px'})


@app.callback(
    Output('portfolio-holdings', 'children'),
    [Input('portfolio-data', 'data')]
)
def update_portfolio_holdings(portfolio_data):
    if not portfolio_data or 'balances' not in portfolio_data:
        return html.P("No holdings", style={'color': THEME['text_muted']})
    
    balances = portfolio_data.get('balances', {})
    prices = portfolio_data.get('prices', {})
    
    holdings = []
    for asset, qty in balances.items():
        price = prices.get(asset, 1.0) if asset != 'USDT' else 1.0
        value = qty * price
        
        holdings.append({
            'asset': asset,
            'quantity': qty,
            'price': price,
            'value': value,
            'pct': (value / portfolio_data['total_value'] * 100) if portfolio_data['total_value'] > 0 else 0
        })
    
    df = pd.DataFrame(holdings)
    
    # Create table
    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Asset", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("Quantity", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("Price", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("Value", style={'color': THEME['text_muted'], 'padding': '10px'}),
                html.Th("% of Portfolio", style={'color': THEME['text_muted'], 'padding': '10px'}),
            ], style={'background': '#21262d'})
        ]),
        html.Tbody([
            html.Tr([
                html.Td(h['asset'], style={'color': THEME['text'], 'padding': '8px', 'font-weight': 'bold'}),
                html.Td(f"{h['quantity']:.4f}", style={'color': THEME['text'], 'padding': '8px'}),
                html.Td(f"${h['price']:,.2f}", style={'color': THEME['text'], 'padding': '8px'}),
                html.Td(f"${h['value']:,.2f}", style={'color': THEME['green'], 'padding': '8px'}),
                html.Td(f"{h['pct']:.1f}%", style={'color': THEME['blue'], 'padding': '8px'}),
            ]) for h in df.to_dict('records')
        ])
    ], style={'width': '100%', 'border-collapse': 'collapse', 'font-size': '13px'})


# ==================== RUN SERVER ====================

# Add enhanced CSS and JS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Loading Overlay */
            .loading-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(13, 17, 23, 0.95);
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                z-index: 9999;
            }
            
            .loading-overlay.hidden {
                display: none;
            }
            
            .loading-spinner-large {
                width: 50px;
                height: 50px;
                border: 4px solid rgba(88, 166, 255, 0.2);
                border-top-color: #58a6ff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            .loading-content {
                text-align: center;
            }
            
            .loading-text {
                color: #8b949e;
                margin-top: 20px;
                font-size: 16px;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            /* Toast Notifications */
            .toast-container {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            .toast {
                padding: 14px 20px;
                border-radius: 8px;
                color: white;
                font-weight: 500;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
                animation: slideInRight 0.3s ease;
                min-width: 250px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .toast-success {
                background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
                border-left: 4px solid #3fb950;
            }
            
            .toast-error {
                background: linear-gradient(135deg, #da3633 0%, #f85149 100%);
            }
            
            .toast-info {
                background: linear-gradient(135deg, #1f6feb 0%, #58a6ff 100%);
            }
            
            .toast-close {
                margin-left: auto;
                cursor: pointer;
                opacity: 0.7;
                font-size: 18px;
            }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            /* Last Updated */
            .last-updated {
                color: #6e7681;
                font-size: 11px;
            }
            
            /* Enhanced Stat Cards */
            .stat-card {
                transition: all 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            }
        </style>
    </head>
    <body>
        <div id="toast-container" class="toast-container"></div>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            // Toast notification system
            function showToast(message, type = 'info') {
                const container = document.getElementById('toast-container');
                const toast = document.createElement('div');
                toast.className = 'toast toast-' + type;
                
                const icons = {
                    success: '✓',
                    error: '✕',
                    info: 'ℹ'
                };
                
                toast.innerHTML = `
                    <span>${icons[type] || 'ℹ'}</span>
                    <span>${message}</span>
                    <span class="toast-close" onclick="this.parentElement.remove()">×</span>
                `;
                
                container.appendChild(toast);
                
                setTimeout(() => {
                    toast.style.opacity = '0';
                    setTimeout(() => toast.remove(), 300);
                }, 4000);
            }
            
            // Hide loading overlay when page loads
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(() => {
                    const overlay = document.getElementById('loading-overlay');
                    if (overlay) {
                        overlay.classList.add('hidden');
                    }
                }, 500);
            });
        </script>
    </body>
</html>
'''

if __name__ == "__main__":
    logger.info("Starting Multi-Asset Execution Dashboard...")
    app.run(debug=True, host="127.0.0.1", port=8050)

