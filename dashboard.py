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
from binance_research import BinanceResearch

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
        self.binance_research = BinanceResearch()
        
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
                        html.Div([
                            html.H3("📈 Price Chart", style={'color': self.theme['text'], 'display': 'inline-block', 'margin-right': '15px'}),
                            dcc.Dropdown(
                                id='symbol-selector',
                                options=[
                                    {'label': s, 'value': s} 
                                    for s in self.data_collector.get_supported_symbols()
                                ],
                                value='BTC/USDT',
                                style={'color': '#000', 'width': '150px', 'display': 'inline-block', 'vertical-align': 'middle'},
                                clearable=False
                            ),
                            dcc.Dropdown(
                                id='timeframe-selector',
                                options=[
                                    {'label': '1 Hour', 'value': '1h'},
                                    {'label': '4 Hours', 'value': '4h'},
                                    {'label': '1 Day', 'value': '1d'},
                                ],
                                value='1h',
                                style={'color': '#000', 'width': '120px', 'display': 'inline-block', 'vertical-align': 'middle', 'margin-left': '10px'},
                                clearable=False
                            ),
                        ], style={'margin-bottom': '10px'}),
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
                
                # Third row - Signals table, sentiment, and news
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
                    'gap': '20px',
                    'margin-bottom': '20px'
                }),
                
                # Fourth row - Portfolio equity curve and Binance market data
                html.Div([
                    # Portfolio performance chart
                    html.Div([
                        html.H3("📊 Portfolio Performance", style={'color': self.theme['text']}),
                        dcc.Graph(id='portfolio-chart')
                    ], style={
                        'background': self.theme['card'],
                        'padding': '15px',
                        'border-radius': '8px',
                        'flex': '2'
                    }),
                    
                    # Binance Market Data
                    html.Div([
                        html.H3("Binance Market Data", style={'color': self.theme['text']}),
                        html.Div(id='binance-market-data')
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
             Input('timeframe-selector', 'value'),
             Input('signals-data', 'data')]
        )
        def update_price_chart(symbol, timeframe, signals_data):
            """Update price chart with technical indicators"""
            # Get price data with selected timeframe
            df = self.data_collector.fetch_ohlcv(symbol, timeframe, 100)
            
            if df is None or df.empty:
                return go.Figure()
            
            # Analyze data
            analysis = self.technical_analyzer.analyze(df, symbol)
            
            # Create candlestick chart with subplots for MACD, RSI, Volume
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.15, 0.15, 0.2],
                subplot_titles=('Price + EMA + Bollinger Bands + VWAP', 'MACD + Signal', 'RSI + Stoch', 'Volume + ATR')
            )
            
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
            ), row=1, col=1)
            
            # EMA lines
            if analysis.ema_short > 0:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['close'].ewm(span=9, adjust=False).mean(),
                    mode='lines',
                    name='EMA 9',
                    line=dict(color='#58a6ff', width=1.5)
                ), row=1, col=1)
            
            if analysis.ema_medium > 0:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['close'].ewm(span=21, adjust=False).mean(),
                    mode='lines',
                    name='EMA 21',
                    line=dict(color='#d29922', width=1.5)
                ), row=1, col=1)
            
            # SMA 50
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['close'].rolling(window=50).mean(),
                mode='lines',
                name='SMA 50',
                line=dict(color='#a371f7', width=1.5, dash='dot')
            ), row=1, col=1)
            
            # VWAP (Volume Weighted Average Price)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=vwap,
                mode='lines',
                name='VWAP',
                line=dict(color='#f778ba', width=2)
            ), row=1, col=1)
            
            # Bollinger Bands
            if analysis.bb_upper > 0:
                # Calculate BB
                sma = df['close'].rolling(window=20).mean()
                std = df['close'].rolling(window=20).std()
                bb_upper = sma + (std * 2)
                bb_lower = sma - (std * 2)
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=bb_upper,
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(88, 166, 255, 0.5)', width=1),
                    showlegend=False
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=bb_lower,
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(88, 166, 255, 0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(88, 166, 255, 0.1)',
                    showlegend=False
                ), row=1, col=1)
            
            # MACD (row 2)
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - signal
            
            # MACD Histogram colors
            colors = [self.theme['green'] if val >= 0 else self.theme['red'] for val in macd_hist]
            
            fig.add_trace(go.Bar(
                x=df.index,
                y=macd_hist,
                name='MACD Hist',
                marker_color=colors,
                showlegend=False
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=macd,
                mode='lines',
                name='MACD',
                line=dict(color='#58a6ff', width=1.5)
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=signal,
                mode='lines',
                name='Signal',
                line=dict(color='#d29922', width=1.5)
            ), row=2, col=1)
            
            # RSI (row 3)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color='#a371f7', width=1.5),
                showlegend=False
            ), row=3, col=1)
            
            # Stochastic Oscillator
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            stoch = 100 * (df['close'] - low_14) / (high_14 - low_14)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=stoch,
                mode='lines',
                name='Stochastic',
                line=dict(color='#f0883e', width=1.5),
                showlegend=False
            ), row=3, col=1)
            
            # RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(248, 81, 73, 0.5)", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(63, 185, 80, 0.5)", row=3, col=1)
            fig.add_hline(y=80, line_dash="dot", line_color="rgba(240, 136, 62, 0.5)", row=3, col=1)
            fig.add_hline(y=20, line_dash="dot", line_color="rgba(240, 136, 62, 0.5)", row=3, col=1)
            
            # Volume (row 4)
            colors_vol = [self.theme['green'] if df['close'].iloc[i] >= df['open'].iloc[i] else self.theme['red'] 
                         for i in range(len(df))]
            
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors_vol,
                showlegend=False
            ), row=4, col=1)
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=atr,
                mode='lines',
                name='ATR',
                line=dict(color='#79c0ff', width=1.5),
                showlegend=False
            ), row=4, col=1)
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=750,
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=30, b=50)
            )
            
            # Update y-axis ranges
            fig.update_yaxes(title_text="", row=1, col=1)
            fig.update_yaxes(title_text="", row=2, col=1)
            fig.update_yaxes(title_text="", range=[0, 100], row=3, col=1)
            fig.update_yaxes(title_text="", row=4, col=1)
            
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
        
        # Pairs comparison callback
        @self.app.callback(
            Output('pairs-comparison', 'children'),
            [Input('refresh-interval', 'n_intervals')]
        )
        def update_pairs_comparison(n):
            """Update trading pairs comparison from Binance"""
            try:
                # Get top trading pairs from Binance
                tickers = self.binance_research.get_24hr_tickers()
                
                if not tickers:
                    return [html.P("Loading...", style={'color': self.theme['text_muted']})]
                
                # Sort by volume and get top 10
                sorted_tickers = sorted(tickers, key=lambda x: x.get('quote_volume', 0), reverse=True)[:10]
                
                rows = []
                for t in sorted_tickers:
                    symbol = t.get('symbol', '')
                    price = t.get('price', 0)
                    change = t.get('price_change_percent', 0)
                    volume = t.get('quote_volume', 0)
                    
                    color = self.theme['green'] if change >= 0 else self.theme['red']
                    
                    rows.append(html.Div([
                        html.Span(symbol, style={'color': self.theme['text'], 'font-weight': 'bold', 'width': '80px', 'display': 'inline-block'}),
                        html.Span(f"${price:,.0f}" if price > 1 else f"${price:.4f}", 
                                 style={'color': self.theme['text_muted'], 'width': '70px', 'display': 'inline-block', 'text-align': 'right'}),
                        html.Span(f"{change:+.1f}%", style={'color': color, 'width': '60px', 'display': 'inline-block', 'text-align': 'right'}),
                    ], style={'display': 'flex', 'justify-content': 'space-between', 'padding': '5px', 'border-bottom': '1px solid #30363d'}))
                
                return rows
            except Exception as e:
                return [html.P(f"Error: {str(e)[:50]}", style={'color': self.theme['red'], 'font-size': '11px'})]
        
        # Portfolio callback
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
        
        # Portfolio chart callback
        @self.app.callback(
            Output('portfolio-chart', 'figure'),
            [Input('refresh-interval', 'n_intervals')]
        )
        def update_portfolio_chart(n):
            """Update portfolio equity curve"""
            # Get equity history
            equity_history = self.simulator.get_equity_history()
            
            if not equity_history:
                # Generate sample data for demo
                import random
                timestamps = pd.date_range(end=datetime.now(), periods=50, freq='h')
                values = [10000 + random.uniform(-500, 800) for _ in range(50)]
                equity_history = [{'timestamp': t, 'value': v} for t, v in zip(timestamps, values)]
            
            df = pd.DataFrame(equity_history)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['value'],
                mode='lines',
                name='Portfolio Value',
                fill='tozeroy',
                line=dict(color=self.theme['blue'], width=2),
                fillcolor='rgba(88, 166, 255, 0.2)'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=250,
                xaxis_title="",
                yaxis_title="",
                margin=dict(l=50, r=50, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
        
        # News feed callback
        @self.app.callback(
            Output('news-feed', 'children'),
            [Input('refresh-interval', 'n_intervals')]
        )
        def update_news_feed(n):
            """Update news feed"""
            # Get news from sentiment analyzer
            news_items = []
            
            try:
                # Try to get news from sentiment analyzer
                sentiment = self.sentiment_analyzer.get_combined_sentiment('Bitcoin')
                if 'news' in sentiment:
                    news_items = sentiment['news'][:5]
            except:
                pass
            
            # If no news, show sample messages
            if not news_items:
                news_items = [
                    {'title': 'BTC maintains support above $95K', 'source': 'Market Watch', 'sentiment': 'positive'},
                    {'title': 'ETH gas fees decrease', 'source': 'CryptoNews', 'sentiment': 'neutral'},
                    {'title': 'Gold reaches new highs', 'source': 'Commodities', 'sentiment': 'positive'},
                    {'title': 'Market volatility expected', 'source': 'Analysis', 'sentiment': 'negative'},
                ]
            
            news_elements = []
            for item in news_items[:5]:
                sentiment_color = {
                    'positive': self.theme['green'],
                    'negative': self.theme['red'],
                    'neutral': self.theme['yellow']
                }.get(item.get('sentiment', 'neutral'), self.theme['text_muted'])
                
                news_elements.append(html.Div([
                    html.P(item.get('title', 'No title'), 
                          style={'margin': '0', 'color': self.theme['text'], 'font-size': '12px'}),
                    html.P(item.get('source', 'Unknown'), 
                          style={'margin': '2px 0', 'color': self.theme['text_muted'], 'font-size': '10px'}),
                ], style={
                    'padding': '8px',
                    'border-left': f"3px solid {sentiment_color}",
                    'margin-bottom': '8px',
                    'background': self.theme['background'],
                    'border-radius': '4px'
                }))
            
            return news_elements if news_elements else [html.P("No news available", style={'color': self.theme['text_muted']})]
        
        # Binance Market Data callback
        @self.app.callback(
            Output('binance-market-data', 'children'),
            [Input('refresh-interval', 'n_intervals')]
        )
        def update_binance_market(n):
            """Update Binance market data from real API"""
            try:
                # Get market data
                market = self.binance_research.get_market_cap()
                btc = self.binance_research.get_market_summary('BTCUSDT')
                eth = self.binance_research.get_market_summary('ETHUSDT')
                
                elements = []
                
                # Market overview
                if market.get('status') == 'OK':
                    elements.append(html.Div([
                        html.H5("Market Overview", style={'margin': '0', 'color': self.theme['text_muted']}),
                        html.P(f"BTC Dominance: {market.get('btc_dominance', 0):.1f}%", 
                              style={'margin': '5px 0', 'color': self.theme['text']}),
                        html.P(f"ETH Dominance: {market.get('eth_dominance', 0):.1f}%", 
                              style={'margin': '0', 'color': self.theme['text']}),
                        html.P(f"Trading Pairs: {market.get('num_pairs', 0)}", 
                              style={'margin': '0', 'color': self.theme['text_muted'], 'font-size': '11px'}),
                    ], style={'margin-bottom': '15px'}))
                
                # BTC Data
                if btc.get('status') == 'OK':
                    btc_color = self.theme['green'] if btc.get('price_change_percent', 0) >= 0 else self.theme['red']
                    elements.append(html.Div([
                        html.H5("BTC/USDT", style={'margin': '0', 'color': self.theme['text_muted']}),
                        html.H3(f"${btc.get('price', 0):,.0f}", style={'margin': '5px 0', 'color': self.theme['text']}),
                        html.P(f"{btc.get('price_change_percent', 0):+.2f}%", 
                              style={'margin': '0', 'color': btc_color, 'font-weight': 'bold'}),
                        html.P(f"Vol: ${btc.get('quote_volume_24h', 0)/1e9:.1f}B", 
                              style={'margin': '0', 'color': self.theme['text_muted'], 'font-size': '11px'}),
                    ], style={'margin-bottom': '15px', 'padding': '10px', 'background': self.theme['background'], 'border-radius': '5px'}))
                
                # ETH Data
                if eth.get('status') == 'OK':
                    eth_color = self.theme['green'] if eth.get('price_change_percent', 0) >= 0 else self.theme['red']
                    elements.append(html.Div([
                        html.H5("ETH/USDT", style={'margin': '0', 'color': self.theme['text_muted']}),
                        html.H3(f"${eth.get('price', 0):,.0f}", style={'margin': '5px 0', 'color': self.theme['text']}),
                        html.P(f"{eth.get('price_change_percent', 0):+.2f}%", 
                              style={'margin': '0', 'color': eth_color, 'font-weight': 'bold'}),
                        html.P(f"Vol: ${eth.get('quote_volume_24h', 0)/1e9:.1f}B", 
                              style={'margin': '0', 'color': self.theme['text_muted'], 'font-size': '11px'}),
                    ], style={'padding': '10px', 'background': self.theme['background'], 'border-radius': '5px'}))
                
                return elements if elements else [html.P("Loading market data...", style={'color': self.theme['text_muted']})]
                
            except Exception as e:
                return [html.P(f"Error: {str(e)}", style={'color': self.theme['red']})]
    
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

