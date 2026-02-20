"""
Real-Time Multi-Asset Execution Dashboard with Graphs
=================================================
Advanced dashboard with:
- Signal confidence bar charts
- Portfolio performance line charts
- Order status timeline
- Real-time updates every 5 seconds

Author: AI Trading System
Version: 1.0.0
"""

import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
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
    side: str
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
            'timestamp': self.timestamp.isoformat(),
            'executed_price': self.executed_price,
            'error_message': self.error_message
        }


class OrderManager:
    """Manages order tracking and execution simulation"""
    
    def __init__(self, exchange: str = "Binance"):
        self.exchange = exchange
        self.orders: List[Order] = []
        self._order_counter = 0
        self._portfolio_history: List[Dict] = []
        self._lock = threading.Lock()
    
    def create_order(self, asset: str, side: str, quantity: float, price: float) -> Order:
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
        order.status = OrderStatus.SENT
        return order
    
    def execute_order(self, order: Order, executed_price: float) -> Order:
        order.status = OrderStatus.EXECUTED
        order.executed_price = executed_price
        return order
    
    def fail_order(self, order: Order, error: str) -> Order:
        order.status = OrderStatus.FAILED
        order.error_message = error
        return order
    
    def get_orders(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            return [o.to_dict() for o in self.orders[-limit:]]
    
    def get_order_stats(self) -> Dict:
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
    
    def add_portfolio_snapshot(self, total_value: float, positions: Dict):
        """Add portfolio value snapshot for history"""
        with self._lock:
            self._portfolio_history.append({
                'timestamp': datetime.now(),
                'total_value': total_value,
                'positions': positions.copy()
            })
            # Keep last 100 snapshots
            if len(self._portfolio_history) > 100:
                self._portfolio_history = self._portfolio_history[-100:]
    
    def get_portfolio_history(self) -> List[Dict]:
        with self._lock:
            return [
                {'timestamp': p['timestamp'], 'total_value': p['total_value']}
                for p in self._portfolio_history
            ]


# ==================== EXECUTION ENGINE ====================

class MultiAssetExecutionEngine:
    """Multi-Asset Execution Engine"""
    
    def __init__(self, portfolio: Portfolio, use_testnet: bool = True):
        self.portfolio = portfolio
        self.use_testnet = use_testnet
        self.exchange = "Binance Testnet" if use_testnet else "Binance"
        
        # Initialize decision systems
        self.logical_engine = LogicalPortfolioEngine(portfolio)
        self.math_engine = MathDecisionEngineMultiAsset(portfolio)
        
        # Order manager with portfolio history
        self.order_manager = OrderManager(exchange=self.exchange)
        
        # Signal history
        self.signal_history: List[Dict] = []
        
        # Track portfolio value over time
        self._portfolio_values: List[float] = [portfolio.total_value()]
        
        logger.info(f"MultiAssetExecutionEngine initialized (Exchange: {self.exchange})")
    
    def process_news_feed(self, news_feed: List[NewsItem]) -> List[Dict]:
        logical_signals = self.logical_engine.analyze_news(news_feed)
        
        final_signals = []
        for sig in logical_signals:
            final = self.math_engine.evaluate_signal(sig)
            final["source"] = sig.get("source", "Unknown")
            final["title"] = sig.get("title", "")
            final["sentiment"] = sig.get("sentiment", 0)
            final_signals.append(final)
            
            self.signal_history.append({
                **final,
                'timestamp': datetime.now().isoformat()
            })
        
        return final_signals
    
    def execute_signal(self, signal: Dict) -> Optional[Order]:
        if signal.get('final_signal') == 'HOLD' or not signal.get('can_execute'):
            return None
        
        asset = signal['asset']
        side = signal['final_signal']
        
        current_price = self._get_simulated_price(asset)
        portfolio_value = self.portfolio.total_value()
        confidence = signal.get('adjusted_confidence', 0.5)
        
        position_pct = min(confidence * 0.1, 0.3)
        position_value = portfolio_value * position_pct
        quantity = position_value / current_price if current_price > 0 else 0
        
        order = self.order_manager.create_order(asset, side, quantity, current_price)
        order = self.order_manager.send_order(order)
        
        if random.random() < 0.8:
            slippage = random.uniform(-0.001, 0.001)
            executed_price = current_price * (1 + slippage)
            order = self.order_manager.execute_order(order, executed_price)
            
            if side == "BUY":
                self.portfolio.update_balance(asset, quantity)
            else:
                self.portfolio.update_balance(asset, -quantity)
        else:
            order = self.order_manager.fail_order(order, "Insufficient liquidity")
        
        # Track portfolio value
        self._portfolio_values.append(self.portfolio.total_value())
        self.order_manager.add_portfolio_snapshot(
            self.portfolio.total_value(),
            self.portfolio.balances
        )
        
        return order
    
    def _get_simulated_price(self, asset: str) -> float:
        prices = {
            'BTC': 95000, 'ETH': 3500, 'SOL': 180, 'ADA': 0.8,
            'XRP': 2.5, 'DOT': 7.5, 'AVAX': 38, 'MATIC': 0.45,
        }
        base = prices.get(asset, 100)
        return base * random.uniform(0.98, 1.02)
    
    def get_signals(self, limit: int = 20) -> List[Dict]:
        return self.signal_history[-limit:]
    
    def get_portfolio_state(self) -> Dict:
        return {
            'balances': self.portfolio.balances.copy(),
            'total_value': self.portfolio.total_value(),
            'prices': self.portfolio._prices.copy() if hasattr(self.portfolio, '_prices') else {},
            'history': self._portfolio_values.copy()
        }
    
    def get_performance_metrics(self) -> Dict:
        values = self._portfolio_values
        if len(values) < 2:
            return {'total_return': 0, 'daily_return': 0, 'volatility': 0}
        
        total_return = (values[-1] - values[0]) / values[0] * 100
        daily_return = (values[-1] - values[-2]) / values[-2] * 100 if len(values) > 1 else 0
        
        # Calculate volatility
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        volatility = (sum(returns) / len(returns) * 100) if returns else 0
        
        return {
            'total_return': total_return,
            'daily_return': daily_return,
            'volatility': volatility,
            'peak_value': max(values),
            'current_value': values[-1]
        }


def get_sample_news_feed() -> List[NewsItem]:
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

# Initialize components
portfolio = Portfolio(balances={
    "BTC": 1.5, "ETH": 15.0, "SOL": 100.0, "ADA": 5000.0, "USDT": 50000.0
})

portfolio.set_price("BTC", 95000)
portfolio.set_price("ETH", 3500)
portfolio.set_price("SOL", 180)
portfolio.set_price("ADA", 0.8)
portfolio.set_price("USDT", 1.0)

engine = MultiAssetExecutionEngine(portfolio, use_testnet=True)

# Initial processing
initial_news = get_sample_news_feed()
initial_signals = engine.process_news_feed(initial_news)

for signal in initial_signals:
    if signal.get('can_execute') and signal.get('final_signal') != 'HOLD':
        engine.execute_signal(signal)

# Create Dash app
app = dash.Dash(__name__, title="🚀 Multi-Asset Execution Dashboard")

# Theme
THEME = {
    'bg': '#0d1117', 'card': '#161b22', 'border': '#30363d',
    'text': '#c9d1d9', 'muted': '#8b949e',
    'green': '#3fb950', 'red': '#f85149', 'blue': '#58a6ff',
    'purple': '#a371f7', 'orange': '#f0883e', 'yellow': '#d29922',
}

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("🚀 Multi-Asset Execution Dashboard", 
                   style={'margin': 0, 'color': THEME['blue'], 'font-size': '28px'}),
            html.P("Real-time signals, order execution & portfolio tracking",
                  style={'margin': '5px 0 0 0', 'color': THEME['muted']})
        ], style={'flex': '1'}),
        
        html.Div([
            html.Div([
                html.Span("●", style={'color': THEME['green'], 'margin-right': '8px'}),
                html.Span(f"Exchange: {engine.exchange}", style={'color': THEME['text']})
            ], style={'margin-bottom': '5px'}),
            html.Div([
                html.Span("Mode: ", style={'color': THEME['muted']}),
                html.Span("Paper Trading", style={'color': THEME['green'], 'font-weight': 'bold'})
            ])
        ], style={'text-align': 'right', 'padding': '10px'})
    ], style={
        'background': 'linear-gradient(135deg, #161b22 0%, #1f2937 100%)',
        'padding': '20px', 'border-bottom': f'1px solid {THEME["border"]}',
        'display': 'flex', 'align-items': 'center'
    }),
    
    dcc.Interval(id='interval', interval=5000, n_intervals=0),
    
    html.Div([
        # Stats Row
        html.Div([
            html.Div([
                html.Div("Total Value", style={'color': THEME['muted'], 'font-size': '14px'}),
                html.Div(id='total-value', style={'color': THEME['green'], 'font-size': '24px', 'font-weight': 'bold'})
            ], style={'background': THEME['card'], 'padding': '15px', 'border-radius': '10px', 
                     'border': f'1px solid {THEME["border"]}', 'text-align': 'center', 'flex': '1'}),
            
            html.Div([
                html.Div("Total Return", style={'color': THEME['muted'], 'font-size': '14px'}),
                html.Div(id='total-return', style={'color': THEME['blue'], 'font-size': '24px', 'font-weight': 'bold'})
            ], style={'background': THEME['card'], 'padding': '15px', 'border-radius': '10px',
                     'border': f'1px solid {THEME["border"]}', 'text-align': 'center', 'flex': '1'}),
            
            html.Div([
                html.Div("Orders", style={'color': THEME['muted'], 'font-size': '14px'}),
                html.Div(id='order-count', style={'color': THEME['purple'], 'font-size': '24px', 'font-weight': 'bold'})
            ], style={'background': THEME['card'], 'padding': '15px', 'border-radius': '10px',
                     'border': f'1px solid {THEME["border"]}', 'text-align': 'center', 'flex': '1'}),
            
            html.Div([
                html.Div("Success Rate", style={'color': THEME['muted'], 'font-size': '14px'}),
                html.Div(id='success-rate', style={'color': THEME['green'], 'font-size': '24px', 'font-weight': 'bold'})
            ], style={'background': THEME['card'], 'padding': '15px', 'border-radius': '10px',
                     'border': f'1px solid {THEME["border"]}', 'text-align': 'center', 'flex': '1'}),
        ], style={'display': 'flex', 'gap': '15px', 'margin-bottom': '20px'}),
        
        # Charts Row 1: Signal Confidence + Portfolio Performance
        html.Div([
            html.Div([
                html.H3("📊 Signal Confidence by Asset", style={'color': THEME['text'], 'margin-bottom': '10px'}),
                dcc.Graph(id='signals-chart')
            ], style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'flex': '1'}),
            
            html.Div([
                html.H3("📈 Portfolio Cumulative Value", style={'color': THEME['text'], 'margin-bottom': '10px'}),
                dcc.Graph(id='portfolio-chart')
            ], style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'margin-bottom': '20px'}),
        
        # Charts Row 2: Order Status + Asset Distribution
        html.Div([
            html.Div([
                html.H3("📋 Order Status Distribution", style={'color': THEME['text'], 'margin-bottom': '10px'}),
                dcc.Graph(id='order-chart')
            ], style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'flex': '1'}),
            
            html.Div([
                html.H3("💰 Portfolio Distribution", style={'color': THEME['text'], 'margin-bottom': '10px'}),
                dcc.Graph(id='distribution-chart')
            ], style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'margin-bottom': '20px'}),
        
        # Tables Row
        html.Div([
            html.Div([
                html.H3("📈 Recent Signals", style={'color': THEME['text'], 'margin-bottom': '10px'}),
                html.Div(id='signals-table', style={'max-height': '300px', 'overflow-y': 'auto'})
            ], style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'flex': '1'}),
            
            html.Div([
                html.H3("📋 Recent Orders", style={'color': THEME['text'], 'margin-bottom': '10px'}),
                html.Div(id='orders-table', style={'max-height': '300px', 'overflow-y': 'auto'})
            ], style={'background': THEME['card'], 'padding': '20px', 'border-radius': '12px',
                     'border': f'1px solid {THEME["border"]}', 'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px'}),
        
    ], style={'padding': '20px', 'background': THEME['bg'], 'min-height': '100vh'}),
    
    dcc.Store(id='signals-data'),
    dcc.Store(id='orders-data'),
    dcc.Store(id='portfolio-data'),
    
], style={'background': THEME['bg'], 'font-family': 'Arial, sans-serif'})


@app.callback(
    [Output('total-value', 'children'),
     Output('total-return', 'children'),
     Output('order-count', 'children'),
     Output('success-rate', 'children'),
     Output('signals-data', 'data'),
     Output('orders-data', 'data'),
     Output('portfolio-data', 'data')],
    [Input('interval', 'n_intervals')]
)
def update_dashboard(n):
    if n > 0 and n % 3 == 0:
        news = get_sample_news_feed()
        signals = engine.process_news_feed(news)
        
        for sig in signals:
            if sig.get('can_execute') and sig.get('final_signal') != 'HOLD':
                engine.execute_signal(sig)
    
    order_stats = engine.order_manager.get_order_stats()
    signals = engine.get_signals(20)
    portfolio_state = engine.get_portfolio_state()
    orders = engine.order_manager.get_orders(20)
    metrics = engine.get_performance_metrics()
    
    total_value = f"${portfolio_state['total_value']:,.2f}"
    total_return = f"{metrics['total_return']:+.2f}%"
    order_count = str(order_stats['total'])
    success_rate = f"{order_stats['success_rate']:.1f}%"
    
    return total_value, total_return, order_count, success_rate, signals, orders, portfolio_state


@app.callback(Output('signals-chart', 'figure'), [Input('signals-data', 'data')])
def update_signals_chart(signals):
    if not signals:
        return go.Figure()
    
    df = pd.DataFrame(signals)
    
    fig = px.bar(
        df, x='asset', y='adjusted_confidence',
        color='final_signal',
        color_discrete_map={'BUY': '#3fb950', 'SELL': '#f85149', 'HOLD': '#8b949e'},
        title='Signal Confidence by Asset'
    )
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='transparent',
        plot_bgcolor='transparent', font=dict(color=THEME['text']),
        height=300, yaxis_title='Confidence'
    )
    return fig


@app.callback(Output('portfolio-chart', 'figure'), [Input('portfolio-data', 'data')])
def update_portfolio_chart(portfolio_data):
    if not portfolio_data or 'history' not in portfolio_data:
        return go.Figure()
    
    history = portfolio_data.get('history', [])
    if not history:
        return go.Figure()
    
    df = pd.DataFrame({
        'time': range(len(history)),
        'value': history
    })
    
    fig = px.line(
        df, x='time', y='value',
        title='Portfolio Value Over Time'
    )
    
    fig.update_traces(
        fill='tozeroy',
        fillcolor='rgba(63, 185, 128, 0.2)',
        line=dict(color='#3fb950', width=2)
    )
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='transparent',
        plot_bgcolor='transparent', font=dict(color=THEME['text']),
        height=300, yaxis_title='Value ($)'
    )
    return fig


@app.callback(Output('order-chart', 'figure'), [Input('orders-data', 'data')])
def update_order_chart(orders):
    if not orders:
        return go.Figure()
    
    df = pd.DataFrame(orders)
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['status', 'count']
    
    fig = px.pie(
        status_counts, values='count', names='status',
        title='Order Status Distribution',
        color='status',
        color_discrete_map={
            'executed': '#3fb950',
            'failed': '#f85149',
            'sent': '#f0883e',
            'pending': '#8b949e'
        }
    )
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='transparent',
        plot_bgcolor='transparent', font=dict(color=THEME['text']),
        height=300
    )
    return fig


@app.callback(Output('distribution-chart', 'figure'), [Input('portfolio-data', 'data')])
def update_distribution_chart(portfolio_data):
    if not portfolio_data or 'balances' not in portfolio_data:
        return go.Figure()
    
    balances = portfolio_data.get('balances', {})
    prices = portfolio_data.get('prices', {})
    
    assets = []
    values = []
    for asset, qty in balances.items():
        price = prices.get(asset, 1.0) if asset != 'USDT' else 1.0
        value = qty * price
        if value > 0:
            assets.append(asset)
            values.append(value)
    
    if not assets:
        return go.Figure()
    
    fig = px.pie(
        values=values, names=assets,
        title='Portfolio Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='transparent',
        plot_bgcolor='transparent', font=dict(color=THEME['text']),
        height=300
    )
    return fig


@app.callback(Output('signals-table', 'children'), [Input('signals-data', 'data')])
def update_signals_table(signals):
    if not signals:
        return html.P("No signals available", style={'color': THEME['muted']})
    
    df = pd.DataFrame(signals[-10:])
    
    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Asset", style={'color': THEME['muted'], 'padding': '8px'}),
                html.Th("Signal", style={'color': THEME['muted'], 'padding': '8px'}),
                html.Th("Confidence", style={'color': THEME['muted'], 'padding': '8px'}),
                html.Th("Can Exec", style={'color': THEME['muted'], 'padding': '8px'}),
            ], style={'background': '#21262d'})
        ]),
        html.Tbody([
            html.Tr([
                html.Td(s['asset'], style={'color': THEME['text'], 'padding': '6px', 'font-weight': 'bold'}),
                html.Td(
                    s['final_signal'],
                    style={'color': THEME['green'] if s['final_signal'] == 'BUY' 
                                 else THEME['red'] if s['final_signal'] == 'SELL' 
                                 else THEME['muted'], 'padding': '6px', 'font-weight': 'bold'}
                ),
                html.Td(f"{s.get('adjusted_confidence', 0):.2f}", style={'color': THEME['blue'], 'padding': '6px'}),
                html.Td("✅" if s.get('can_execute') else "❌", style={'padding': '6px', 'text-align': 'center'}),
            ]) for s in df.to_dict('records')
        ])
    ], style={'width': '100%', 'border-collapse': 'collapse', 'font-size': '13px'})


@app.callback(Output('orders-table', 'children'), [Input('orders-data', 'data')])
def update_orders_table(orders):
    if not orders:
        return html.P("No orders yet", style={'color': THEME['muted']})
    
    df = pd.DataFrame(orders[-10:])
    
    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Order ID", style={'color': THEME['muted'], 'padding': '8px'}),
                html.Th("Asset", style={'color': THEME['muted'], 'padding': '8px'}),
                html.Th("Side", style={'color': THEME['muted'], 'padding': '8px'}),
                html.Th("Qty", style={'color': THEME['muted'], 'padding': '8px'}),
                html.Th("Status", style={'color': THEME['muted'], 'padding': '8px'}),
            ], style={'background': '#21262d'})
        ]),
        html.Tbody([
            html.Tr([
                html.Td(o['order_id'], style={'color': THEME['muted'], 'padding': '6px', 'font-size': '11px', 'font-family': 'monospace'}),
                html.Td(o['asset'], style={'color': THEME['text'], 'padding': '6px', 'font-weight': 'bold'}),
                html.Td(
                    o['side'],
                    style={'color': THEME['green'] if o['side'] == 'BUY' else THEME['red'], 'padding': '6px', 'font-weight': 'bold'}
                ),
                html.Td(f"{o['quantity']:.4f}", style={'color': THEME['text'], 'padding': '6px'}),
                html.Td(
                    o['status'].upper(),
                    style={'color': THEME['green'] if o['status'] == 'executed' 
                                 else THEME['red'] if o['status'] == 'failed'
                                 else THEME['orange'] if o['status'] == 'sent'
                                 else THEME['muted'], 'padding': '6px', 'font-size': '11px', 'font-weight': 'bold'}
                ),
            ]) for o in df.to_dict('records')
        ])
    ], style={'width': '100%', 'border-collapse': 'collapse', 'font-size': '12px'})


if __name__ == "__main__":
    logger.info("Starting Advanced Multi-Asset Execution Dashboard with Graphs...")
    app.run_server(debug=True, host="127.0.0.1", port=8051)

