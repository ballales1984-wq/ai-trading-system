"""
Dashboard Module
Interactive visualization dashboard for the trading system
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

import pandas as pd
import numpy as np

# Dash imports
try:
    import dash
    from dash import dcc, html, callback, Input, Output
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Warning: Dash not available. Install with: pip install dash plotly")

import config
from data_collector import DataCollector
from technical_analysis import TechnicalAnalyzer
from decision_engine import DecisionEngine
from sentiment_news import SentimentAnalyzer
from trading_simulator import TradingSimulator

# Configure logging
logger = logging.getLogger(__name__)


# ==================== DASHBOARD CLASS ====================

class TradingDashboard:
    """
    Interactive dashboard for trading signals and market analysis.
    Uses Dash and Plotly for visualization.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the dashboard.
        
        Args:
            debug: Enable debug mode
        """
        if not DASH_AVAILABLE:
            raise ImportError("Dash and Plotly are required. Install: pip install dash plotly")
        
        self.debug = debug
        self.data_collector = DataCollector(simulation=True)
        self.decision_engine = DecisionEngine(self.data_collector)
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.simulator = TradingSimulator(initial_balance=10000.0)
        
        # Cache for data
        self.cached_signals = []
        self.cached_prices = {}
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            title="Crypto Commodity Trading Dashboard",
            update_title=None
        )
        
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info("TradingDashboard initialized")
    
    # ==================== LAYOUT ====================
    
    def _setup_layout(self):
        """Setup the dashboard layout"""
        
        # Custom dark theme
        dark_theme = {
            'background': '#0d1117',
            'card': '#161b22',
            'border': '#30363d',
            'text': '#c9d1d9',
            'text_muted': '#8b949e',
            'green': '#3fb950',
            'red': '#f85149',
            'yellow': '#d29922',
            'blue': '#58a6ff',
        }
        
        self.theme = dark_theme
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🧠 Crypto + Commodity Trading System",
                       style={'margin': '0', 'color': self.theme['text']}),
                html.P("Experimental AI Trading Signals with Cross-Market Analysis",
                      style={'margin': '5px 0 0 0', 'color': self.theme['text_muted']}),
            ], style={
                'background': self.theme['card'],
                'padding': '20px',
                'border-bottom': f"1px solid {self.theme['border']}"
            }),
            
            # Refresh interval
            dcc.Interval(
                id='refresh-interval',
                interval=config.DASHBOARD_CONFIG['refresh_interval'] * 1000,
                n_intervals=0
            ),
            
            # Main content
            html.Div([
                # Top row - Signals summary
                html.Div([
                    self._create_signals_summary(),
                ], style={'margin-bottom': '20px'}),
                
                # Portfolio Section
                html.Div([
                    self._create_portfolio_section(),
                ], style={'margin-bottom': '20px'}),
                
                # Second row - Charts
                html.Div([
                    # Price chart
                    html.Div([
                        html.H3("📈 Price Chart", style={'color': self.theme['text']}),
                        dcc.Dropdown(
                            id='symbol-selector',
                            options=[
                                {'label': s, 'value': s} 
                                for s in self.data_collector.get_supported_symbols()
                            ],
                            value='BTC/USDT',
                            style={'color': '#000'}
                        ),
                        dcc.Graph(id='price-chart')
                    ], style={
                        'background': self.theme['card'],
                        'padding': '15px',
                        'border-radius': '8px',
                        'flex': '2'
                    }),
                    
                    # Correlation heatmap
                    html.Div([
                        html.H3("🔗 Correlation Matrix", style={'color': self.theme['text']}),
                        dcc.Graph(id='correlation-heatmap')
                    ], style={
                        'background': self.theme['card'],
                        'padding': '15px',
                        'border-radius': '8px',
                        'flex': '1'
                    }),
                ], style={
                    'display': 'flex',
                    'gap': '20px',
                    'margin-bottom': '20px'
                }),
                
                # Third row - Signals table and sentiment
                html.Div([
                    # Signals table
                    html.Div([
                        html.H3("🎯 Trading Signals", style={'color': self.theme['text']}),
                        html.Div(id='signals-table')
                    ], style={
                        'background': self.theme['card'],
                        'padding': '15px',
                        'border-radius': '8px',
                        'flex': '2'
                    }),
                    
                    # Sentiment panel
                    html.Div([
                        html.H3("📰 Market Sentiment", style={'color': self.theme['text']}),
                        html.Div(id='sentiment-panel')
                    ], style={
                        'background': self.theme['card'],
                        'padding': '15px',
                        'border-radius': '8px',
                        'flex': '1'
                    }),
                ], style={
                    'display': 'flex',
                    'gap': '20px'
                }),
                
            ], style={
                'padding': '20px',
                'background': self.theme['background'],
                'min-height': '100vh'
            }),
            
            # Hidden store for signals data
            dcc.Store(id='signals-data'),
        ], style={'fontFamily': 'Arial, sans-serif'})
    
    # ==================== COMPONENTS ====================
    
    def _create_signals_summary(self) -> html.Div:
        """Create the signals summary cards"""
        return html.Div([
            # Total Signals
            html.Div([
                html.H4("Total Signals", style={'margin': '0', 'color': self.theme['text_muted']}),
                html.H2(id='total-signals', style={'margin': '5px 0', 'color': self.theme['text']}),
            ], style={
                'background': self.theme['card'],
                'padding': '15px',
                'border-radius': '8px',
                'text-align': 'center',
                'flex': '1'
            }),
            
            # Buy Signals
            html.Div([
                html.H4("Buy Signals", style={'margin': '0', 'color': self.theme['text_muted']}),
                html.H2(id='buy-signals', style={'margin': '5px 0', 'color': self.theme['green']}),
            ], style={
                'background': self.theme['card'],
                'padding': '15px',
                'border-radius': '8px',
                'text-align': 'center',
                'flex': '1'
            }),
            
            # Sell Signals
            html.Div([
                html.H4("Sell Signals", style={'margin': '0', 'color': self.theme['text_muted']}),
                html.H2(id='sell-signals', style={'margin': '5px 0', 'color': self.theme['red']}),
            ], style={
                'background': self.theme['card'],
                'padding': '15px',
                'border-radius': '8px',
                'text-align': 'center',
                'flex': '1'
            }),
            
            # Hold Signals
            html.Div([
                html.H4("Hold", style={'margin': '0', 'color': self.theme['text_muted']}),
                html.H2(id='hold-signals', style={'margin': '5px 0', 'color': self.theme['yellow']}),
            ], style={
                'background': self.theme['card'],
                'padding': '15px',
                'border-radius': '8px',
                'text-align': 'center',
                'flex': '1'
            }),
            
            # Avg Confidence
            html.Div([
                html.H4("Avg Confidence", style={'margin': '0', 'color': self.theme['text_muted']}),
                html.H2(id='avg-confidence', style={'margin': '5px 0', 'color': self.theme['blue']}),
            ], style={
                'background': self.theme['card'],
                'padding': '15px',
                'border-radius': '8px',
                'text-align': 'center',
                'flex': '1'
            }),
            
        ], style={
            'display': 'flex',
            'gap': '15px'
        })

    def _create_portfolio_section(self) -> html.Div:
        """Create portfolio/balance section"""
        return html.Div([
            html.H3("Portfolio Balance", style={'color': self.theme['text']}),
            
            # Balance Row
            html.Div([
                # Total Balance
                html.Div([
                    html.H4("Total Balance", style={'margin': '0', 'color': self.theme['text_muted']}),
                    html.H2(id='total-balance', style={'margin': '5px 0', 'color': self.theme['text']}),
                ], style={'flex': '1', 'text-align': 'center'}),
                
                # Available Balance
                html.Div([
                    html.H4("Available", style={'margin': '0', 'color': self.theme['text_muted']}),
                    html.H2(id='available-balance', style={'margin': '5px 0', 'color': self.theme['green']}),
                ], style={'flex': '1', 'text-align': 'center'}),
                
                # PnL
                html.Div([
                    html.H4("Total P&L", style={'margin': '0', 'color': self.theme['text_muted']}),
                    html.H2(id='total-pnl', style={'margin': '5px 0', 'color': self.theme['green']}),
                ], style={'flex': '1', 'text-align': 'center'}),
                
                # Win Rate
                html.Div([
                    html.H4("Win Rate", style={'margin': '0', 'color': self.theme['text_muted']}),
                    html.H2(id='win-rate', style={'margin': '5px 0', 'color': self.theme['blue']}),
                ], style={'flex': '1', 'text-align': 'center'}),
                
            ], style={'display': 'flex', 'gap': '15px', 'margin-top': '15px'}),
            
            # Positions
            html.Div(id='positions-list', style={'margin-top': '15px'}),
            
        ], style={
            'background': self.theme['card'],
            'padding': '15px',
            'border-radius': '8px',
            'margin-top': '20px'
        })
    
    # ==================== CALLBACKS ====================
    
    def _setup_callbacks(self):
        """Setup Dash callbacks"""
        
        @self.app.callback(
            [Output('signals-data', 'data'),
             Output('total-signals', 'children'),
             Output('buy-signals', 'children'),
             Output('sell-signals', 'children'),
             Output('hold-signals', 'children'),
             Output('avg-confidence', 'children')],
            [Input('refresh-interval', 'n_intervals')]
        )
        def update_signals(n):
            """Update signals data"""
            signals = self.decision_engine.generate_signals()
            self.cached_signals = signals
            
            total = len(signals)
            buy = len([s for s in signals if s.action == 'BUY'])
            sell = len([s for s in signals if s.action == 'SELL'])
            hold = len([s for s in signals if s.action == 'HOLD'])
            
            avg_conf = np.mean([s.confidence for s in signals]) * 100 if signals else 0
            
            # Serialize signals for storage
            signals_json = json.dumps([s.to_dict() for s in signals], default=str)
            
            return signals_json, str(total), str(buy), str(sell), str(hold), f"{avg_conf:.0f}%"
        
        @self.app.callback(
            Output('price-chart', 'figure'),
            [Input('symbol-selector', 'value'),
             Input('signals-data', 'data')]
        )
        def update_price_chart(symbol, signals_data):
            """Update price chart"""
            # Get price data
            df = self.data_collector.fetch_ohlcv(symbol, '1h', 100)
            
            if df is None or df.empty:
                return go.Figure()
            
            # Create candlestick chart
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color=self.theme['green'],
                decreasing_line_color=self.theme['red']
            ))
            
            # Add technical indicators
            analysis = self.technical_analyzer.analyze(df, symbol)
            
            # EMA lines
            if analysis.ema_short > 0:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=[analysis.ema_short] * len(df),
                    mode='lines',
                    name='EMA Short',
                    line=dict(color='#58a6ff', width=1)
                ))
            
            if analysis.ema_medium > 0:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=[analysis.ema_medium] * len(df),
                    mode='lines',
                    name='EMA Medium',
                    line=dict(color='#d29922', width=1)
                ))
            
            # Bollinger Bands
            if analysis.bb_upper > 0:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=[analysis.bb_upper] * len(df),
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(88, 166, 255, 0.3)', width=1),
                    showlegend=True
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=[analysis.bb_lower] * len(df),
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(88, 166, 255, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(88, 166, 255, 0.1)',
                    showlegend=True
                ))
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
        
        @self.app.callback(
            Output('correlation-heatmap', 'figure'),
            [Input('signals-data', 'data')]
        )
        def update_correlation_heatmap(signals_data):
            """Update correlation heatmap"""
            # Get symbols
            symbols = list(config.CRYPTO_SYMBOLS.values())[:6]
            
            # Calculate correlation matrix
            corr_matrix = self.data_collector.calculate_correlation_matrix(symbols, 24)
            
            if corr_matrix.empty:
                return go.Figure()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.index,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                showscale=True
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                height=350,
                margin=dict(l=50, r=50, t=30, b=50)
            )
            
            return fig
        
        @self.app.callback(
            Output('signals-table', 'children'),
            [Input('signals-data', 'data')]
        )
        def update_signals_table(signals_data):
            """Update signals table"""
            if not signals_data:
                return html.P("No signals available", style={'color': self.theme['text_muted']})
            
            signals = json.loads(signals_data)
            
            if not signals:
                return html.P("No signals available", style={'color': self.theme['text_muted']})
            
            # Create table
            rows = []
            
            for signal in signals[:10]:  # Top 10
                action_color = {
                    'BUY': self.theme['green'],
                    'SELL': self.theme['red'],
                    'HOLD': self.theme['yellow']
                }.get(signal['action'], self.theme['text'])
                
                rows.append(html.Tr([
                    html.Td(signal['symbol'], style={'color': self.theme['text']}),
                    html.Td(
                        html.Span(signal['action'], 
                                 style={'color': action_color, 'font-weight': 'bold'}),
                        style={'text-align': 'center'}
                    ),
                    html.Td(f"${signal['current_price']:,.2f}", 
                           style={'color': self.theme['text'], 'text-align': 'right'}),
                    html.Td(signal['confidence'], 
                           style={'color': self.theme['text'], 'text-align': 'center'}),
                    html.Td(f"{signal['technical_score']}", 
                           style={'color': self.theme['text'], 'text-align': 'center'}),
                ]))
            
            return html.Table([
                html.Thead(html.Tr([
                    html.Th("Symbol", style={'color': self.theme['text_muted']}),
                    html.Th("Action", style={'color': self.theme['text_muted'], 'text-align': 'center'}),
                    html.Th("Price", style={'color': self.theme['text_muted'], 'text-align': 'right'}),
                    html.Th("Confidence", style={'color': self.theme['text_muted'], 'text-align': 'center'}),
                    html.Th("Technical", style={'color': self.theme['text_muted'], 'text-align': 'center'}),
                ])),
                html.Tbody(rows)
            ], style={'width': '100%', 'border-collapse': 'collapse'})
        
        @self.app.callback(
            Output('sentiment-panel', 'children'),
            [Input('signals-data', 'data')]
        )
        def update_sentiment_panel(signals_data):
            """Update sentiment panel"""
            # Get sentiment for major assets
            assets = ['Bitcoin', 'Ethereum', 'Gold']
            sentiments = []
            
            for asset in assets:
                sentiment = self.sentiment_analyzer.get_combined_sentiment(asset)
                sentiments.append(sentiment)
            
            # Create sentiment cards
            cards = []
            
            for sent in sentiments:
                score = sent['combined_score']
                color = self.theme['green'] if score > 0.2 else self.theme['red'] if score < -0.2 else self.theme['yellow']
                
                cards.append(html.Div([
                    html.H5(sent['asset'], style={'margin': '0', 'color': self.theme['text_muted']}),
                    html.H3(f"{score:+.2f}", style={'margin': '5px 0', 'color': color}),
                    html.P(f"F/G: {sent['social_sentiment']['fear_greed_index']}",
                          style={'margin': '0', 'color': self.theme['text_muted'], 'font-size': '12px'}),
                ], style={
                    'background': self.theme['background'],
                    'padding': '10px',
                    'border-radius': '5px',
                    'margin-bottom': '10px'
                }))
            
            return cards
    
    # ==================== PORTFOLIO CALLBACKS ====================
    
    @self.app.callback(
        [Output('total-balance', 'children'),
         Output('available-balance', 'children'),
         Output('total-pnl', 'children'),
         Output('win-rate', 'children'),
         Output('positions-list', 'children')],
        [Input('refresh-interval', 'n_intervals')]
    )
    def update_portfolio(n):
        """Update portfolio data"""
        # Run one iteration of trading
        try:
            signals = self.decision_engine.generate_signals()
            self.simulator._process_signals(signals)
            self.simulator._manage_positions()
            self.simulator._update_portfolio()
        except:
            pass
        
        # Get portfolio state
        state = self.simulator.get_portfolio_state()
        
        total = f"${state['total_value']:,.2f}"
        available = f"${state['balance']:,.2f}"
        pnl = state['total_pnl']
        pnl_color = self.theme['green'] if pnl >= 0 else self.theme['red']
        pnl_text = f"${pnl:+,.2f}"
        win_rate = f"{state['win_rate']:.1f}%"
        
        # Positions
        positions = []
        for symbol, pos in state['positions'].items():
            pnl_pct = ((pos['current_price'] - pos['entry_price']) / pos['entry_price']) * 100
            pos_color = self.theme['green'] if pnl_pct >= 0 else self.theme['red']
            
            positions.append(html.Div([
                html.Span(symbol, style={'color': self.theme['text'], 'font-weight': 'bold'}),
                html.Span(f" ${pos['current_price']:,.2f}", style={'color': self.theme['text_muted']}),
                html.Span(f" ({pnl_pct:+,.2f}%)", style={'color': pos_color}),
            ], style={'margin': '5px 0', 'display': 'block'}))
        
        positions_display = positions if positions else [html.P("No open positions", style={'color': self.theme['text_muted']})]
        
        return total, available, pnl_text, win_rate, positions_display

    # ==================== RUN ====================
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """
        Run the dashboard server.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        host = host or config.DASHBOARD_CONFIG['host']
        port = port or config.DASHBOARD_CONFIG['port']
        debug = debug if debug is not None else self.debug
        
        logger.info(f"Starting dashboard on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


# ==================== SIMPLE CONSOLE OUTPUT ====================

def print_dashboard_summary():
    """Print a simple console-based dashboard summary"""
    print("\n" + "="*70)
    print("CRYPTO + COMMODITY TRADING SYSTEM - DASHBOARD SUMMARY")
    print("="*70)
    
    # Initialize components
    data_collector = DataCollector(simulation=True)
    decision_engine = DecisionEngine(data_collector)
    
    # Get signals
    print("\n🎯 Generating trading signals...")
    signals = decision_engine.generate_signals()
    
    # Print summary
    print(decision_engine.generate_signal_report(signals))
    
    # Print correlations
    print("\n🔗 Asset Correlations:")
    symbols = ['BTC/USDT', 'ETH/USDT', 'PAXG/USDT', 'XRP/USDT']
    
    corr_matrix = data_collector.calculate_correlation_matrix(symbols, 24)
    
    if not corr_matrix.empty:
        print(corr_matrix.round(3).to_string())
    
    # Print sentiment
    print("\n📰 Market Sentiment:")
    
    for asset in ['Bitcoin', 'Ethereum', 'Gold', 'Oil']:
        sentiment = SentimentAnalyzer().get_combined_sentiment(asset)
        print(f"  {asset}: {sentiment['combined_score']:+.2f}")
        print(f"    Fear/Greed: {sentiment['social_sentiment']['fear_greed_index']}")
    
    print("\n" + "="*70)
    print("To start the interactive dashboard, run: dashboard.run()")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run simple console summary
    print_dashboard_summary()

