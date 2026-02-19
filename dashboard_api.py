#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Trading Dashboard - FastAPI Integration
==============================================
Modern dashboard that integrates with the FastAPI REST API.
Features:
- Real-time portfolio monitoring
- Order management
- Risk metrics visualization
- Market data charts
- Strategy performance
- Modern, responsive UI
"""

import sys
import io
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

# Modern dark theme
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

# Initialize Dash app
app = dash.Dash(
    __name__,
    title="Hedge Fund Trading Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
    ]
)

# ============================================================================
# API CLIENT FUNCTIONS
# ============================================================================

def get_portfolio_summary() -> Dict:
    """Get portfolio summary from API."""
    try:
        response = requests.get(f"{API_BASE_URL}{API_PREFIX}/portfolio/summary", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}

def get_positions() -> List[Dict]:
    """Get positions from API."""
    try:
        response = requests.get(f"{API_BASE_URL}{API_PREFIX}/portfolio/positions", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

def get_orders(limit: int = 10) -> List[Dict]:
    """Get orders from API."""
    try:
        response = requests.get(f"{API_BASE_URL}{API_PREFIX}/orders/", params={"limit": limit}, timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

def get_risk_metrics() -> Dict:
    """Get risk metrics from API."""
    try:
        response = requests.get(f"{API_BASE_URL}{API_PREFIX}/risk/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}

def get_market_prices() -> List[Dict]:
    """Get market prices from API."""
    try:
        response = requests.get(f"{API_BASE_URL}{API_PREFIX}/market/prices", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('markets', [])
    except:
        pass
    return []

def get_portfolio_history(days: int = 30) -> List[Dict]:
    """Get portfolio history from API."""
    try:
        response = requests.get(f"{API_BASE_URL}{API_PREFIX}/portfolio/history", params={"days": days}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('history', [])
    except:
        pass
    return []

def get_strategies() -> List[Dict]:
    """Get strategies from API."""
    try:
        response = requests.get(f"{API_BASE_URL}{API_PREFIX}/strategy/", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

def create_stat_card(title: str, value: str, change: Optional[str] = None, 
                     color: str = THEME['text'], icon: str = "") -> html.Div:
    """Create a stat card component."""
    change_color = THEME['green'] if change and change.startswith('+') else THEME['red'] if change else THEME['text_muted']
    
    return html.Div([
        html.Div([
            html.Div(icon, style={'font-size': '24px', 'margin-bottom': '8px'}),
            html.Div(title, style={
                'color': THEME['text_muted'],
                'font-size': '12px',
                'text-transform': 'uppercase',
                'letter-spacing': '0.5px',
                'margin-bottom': '4px'
            }),
            html.Div(value, style={
                'color': color,
                'font-size': '24px',
                'font-weight': '600',
                'margin-bottom': '4px'
            }),
            html.Div(change or '', style={
                'color': change_color,
                'font-size': '12px'
            }) if change else None,
        ], style={'padding': '20px'})
    ], style={
        'background': THEME['card'],
        'border': f"1px solid {THEME['border']}",
        'border-radius': '8px',
        'min-height': '120px',
        'display': 'flex',
        'flex-direction': 'column',
        'justify-content': 'space-between'
    })

def create_layout():
    """Create the main dashboard layout."""
    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.H1("Hedge Fund Trading System", style={
                    'margin': '0',
                    'color': THEME['text'],
                    'font-size': '28px',
                    'font-weight': '700',
                    'font-family': 'Inter, sans-serif'
                }),
                html.P("Real-time Portfolio & Risk Management Dashboard", style={
                    'margin': '4px 0 0 0',
                    'color': THEME['text_muted'],
                    'font-size': '14px'
                }),
            ], style={'flex': '1'}),
            
            html.Div([
                html.Div(id='connection-status', style={
                    'display': 'inline-block',
                    'padding': '8px 16px',
                    'background': THEME['green'],
                    'color': '#fff',
                    'border-radius': '6px',
                    'font-size': '12px',
                    'font-weight': '500',
                    'margin-right': '16px'
                }),
                html.Div(id='last-update', style={
                    'color': THEME['text_muted'],
                    'font-size': '12px'
                }),
            ], style={'display': 'flex', 'align-items': 'center'}),
        ], style={
            'background': 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
            'padding': '24px',
            'border-bottom': f"1px solid {THEME['border']}",
            'display': 'flex',
            'justify-content': 'space-between',
            'align-items': 'center'
        }),
        
        # Auto-refresh interval
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # Update every 5 seconds
            n_intervals=0
        ),
        
        # Main content
        html.Div([
            # Stats Row
            html.Div(id='stats-cards', style={
                'display': 'grid',
                'grid-template-columns': 'repeat(auto-fit, minmax(200px, 1fr))',
                'gap': '20px',
                'margin': '20px',
                'margin-bottom': '24px'
            }),
            
            # Charts Row 1: Portfolio & Performance
            html.Div([
                html.Div([
                    html.H3("Portfolio Value History", style={
                        'color': THEME['text'],
                        'margin-bottom': '16px',
                        'font-size': '18px',
                        'font-weight': '600'
                    }),
                    dcc.Graph(id='portfolio-chart', style={'height': '350px'}),
                ], style={
                    'background': THEME['card'],
                    'padding': '24px',
                    'border-radius': '8px',
                    'border': f"1px solid {THEME['border']}",
                    'flex': '1'
                }),
                
                html.Div([
                    html.H3("Risk Metrics", style={
                        'color': THEME['text'],
                        'margin-bottom': '16px',
                        'font-size': '18px',
                        'font-weight': '600'
                    }),
                    dcc.Graph(id='risk-chart', style={'height': '350px'}),
                ], style={
                    'background': THEME['card'],
                    'padding': '24px',
                    'border-radius': '8px',
                    'border': f"1px solid {THEME['border']}",
                    'flex': '1'
                }),
            ], style={'display': 'flex', 'gap': '20px', 'margin': '20px', 'margin-bottom': '24px'}),
            
            # Data Tables Row
            html.Div([
                html.Div([
                    html.H3("Open Positions", style={
                        'color': THEME['text'],
                        'margin-bottom': '16px',
                        'font-size': '18px',
                        'font-weight': '600'
                    }),
                    html.Div(id='positions-table'),
                ], style={
                    'background': THEME['card'],
                    'padding': '24px',
                    'border-radius': '8px',
                    'border': f"1px solid {THEME['border']}",
                    'flex': '1'
                }),
                
                html.Div([
                    html.H3("Recent Orders", style={
                        'color': THEME['text'],
                        'margin-bottom': '16px',
                        'font-size': '18px',
                        'font-weight': '600'
                    }),
                    html.Div(id='orders-table'),
                ], style={
                    'background': THEME['card'],
                    'padding': '24px',
                    'border-radius': '8px',
                    'border': f"1px solid {THEME['border']}",
                    'flex': '1'
                }),
            ], style={'display': 'flex', 'gap': '20px', 'margin': '20px', 'margin-bottom': '24px'}),
            
            # Market Prices Row
            html.Div([
                html.H3("Market Prices", style={
                    'color': THEME['text'],
                    'margin-bottom': '16px',
                    'font-size': '18px',
                    'font-weight': '600'
                }),
                html.Div(id='market-prices', style={
                    'display': 'flex',
                    'gap': '16px',
                    'flex-wrap': 'wrap'
                }),
            ], style={
                'background': THEME['card'],
                'padding': '24px',
                'border-radius': '8px',
                'border': f"1px solid {THEME['border']}",
                'margin': '20px'
            }),
        ], style={
            'background': THEME['background'],
            'min-height': '100vh'
        }),
    ], style={
        'font-family': 'Inter, sans-serif',
        'background': THEME['background'],
        'color': THEME['text']
    })

app.layout = create_layout()

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('stats-cards', 'children'),
     Output('connection-status', 'children'),
     Output('last-update', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_stats(n):
    """Update stats cards."""
    try:
        portfolio = get_portfolio_summary()
        risk = get_risk_metrics()
        
        total_value = portfolio.get('total_value', 0)
        total_pnl = portfolio.get('total_pnl', 0)
        total_pnl_pct = portfolio.get('total_return_pct', 0)
        cash = portfolio.get('cash_balance', 0)
        num_positions = portfolio.get('num_positions', 0)
        
        var_1d = risk.get('var_1d', 0)
        sharpe = risk.get('sharpe_ratio', 0)
        
        cards = [
            create_stat_card(
                "Total Value",
                f"${total_value:,.2f}",
                f"{total_pnl_pct:+.2f}%",
                THEME['text'],
                "💰"
            ),
            create_stat_card(
                "Total P&L",
                f"${total_pnl:+,.2f}",
                f"{total_pnl_pct:+.2f}%",
                THEME['green'] if total_pnl >= 0 else THEME['red'],
                "📈"
            ),
            create_stat_card(
                "Cash Balance",
                f"${cash:,.2f}",
                None,
                THEME['blue'],
                "💵"
            ),
            create_stat_card(
                "Open Positions",
                str(num_positions),
                None,
                THEME['purple'],
                "📊"
            ),
            create_stat_card(
                "VaR (1-day)",
                f"${var_1d:,.2f}",
                None,
                THEME['orange'],
                "⚠️"
            ),
            create_stat_card(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                None,
                THEME['yellow'],
                "📉"
            ),
        ]
        
        status = "🟢 Connected"
        update_time = datetime.now().strftime("%H:%M:%S")
        
        return cards, status, f"Last update: {update_time}"
    except Exception as e:
        return [], "🔴 Disconnected", f"Error: {str(e)[:30]}"

@app.callback(
    Output('portfolio-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_portfolio_chart(n):
    """Update portfolio value chart."""
    try:
        history = get_portfolio_history(days=30)
        
        if not history:
            # Generate sample data if API unavailable
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            values = [1000000 + np.random.normal(0, 10000) for _ in range(30)]
            df = pd.DataFrame({'date': dates, 'value': values})
        else:
            df = pd.DataFrame(history)
            df['date'] = pd.to_datetime(df['date'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color=THEME['blue'], width=2),
            fill='tonexty',
            fillcolor=f"rgba(88, 166, 255, 0.1)"
        ))
        
        fig.update_layout(
            plot_bgcolor=THEME['card'],
            paper_bgcolor=THEME['card'],
            font_color=THEME['text'],
            xaxis=dict(gridcolor=THEME['border']),
            yaxis=dict(gridcolor=THEME['border']),
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode='x unified'
        )
        
        return fig
    except:
        return go.Figure()

@app.callback(
    Output('risk-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_risk_chart(n):
    """Update risk metrics chart."""
    try:
        risk = get_risk_metrics()
        
        metrics = ['VaR 1d', 'VaR 5d', 'CVaR 1d', 'CVaR 5d']
        values = [
            risk.get('var_1d', 0),
            risk.get('var_5d', 0),
            risk.get('cvar_1d', 0),
            risk.get('cvar_5d', 0)
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=values,
                marker_color=[THEME['red'], THEME['orange'], THEME['yellow'], THEME['purple']],
                text=[f"${v:,.0f}" for v in values],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            plot_bgcolor=THEME['card'],
            paper_bgcolor=THEME['card'],
            font_color=THEME['text'],
            xaxis=dict(gridcolor=THEME['border']),
            yaxis=dict(gridcolor=THEME['border']),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        return fig
    except:
        return go.Figure()

@app.callback(
    Output('positions-table', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_positions_table(n):
    """Update positions table."""
    try:
        positions = get_positions()
        
        if not positions:
            return html.P("No open positions", style={'color': THEME['text_muted']})
        
        df = pd.DataFrame(positions)
        
        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[
                {'name': 'Symbol', 'id': 'symbol'},
                {'name': 'Side', 'id': 'side'},
                {'name': 'Quantity', 'id': 'quantity', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                {'name': 'Entry Price', 'id': 'entry_price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                {'name': 'Current Price', 'id': 'current_price', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                {'name': 'P&L', 'id': 'unrealized_pnl', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            ],
            style_cell={
                'backgroundColor': THEME['card'],
                'color': THEME['text'],
                'border': f"1px solid {THEME['border']}",
                'textAlign': 'left',
                'fontFamily': 'Inter, sans-serif',
                'fontSize': '13px'
            },
            style_header={
                'backgroundColor': THEME['background'],
                'color': THEME['text'],
                'fontWeight': '600',
                'border': f"1px solid {THEME['border']}"
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{unrealized_pnl} > 0'},
                    'backgroundColor': f"rgba(63, 185, 80, 0.1)",
                    'color': THEME['green']
                },
                {
                    'if': {'filter_query': '{unrealized_pnl} < 0'},
                    'backgroundColor': f"rgba(248, 81, 73, 0.1)",
                    'color': THEME['red']
                }
            ]
        )
        
        return table
    except Exception as e:
        return html.P(f"Error loading positions: {str(e)}", style={'color': THEME['red']})

@app.callback(
    Output('orders-table', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_orders_table(n):
    """Update orders table."""
    try:
        orders = get_orders(limit=10)
        
        if not orders:
            return html.P("No recent orders", style={'color': THEME['text_muted']})
        
        df = pd.DataFrame(orders)
        
        # Format datetime columns
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[
                {'name': 'Order ID', 'id': 'order_id'},
                {'name': 'Symbol', 'id': 'symbol'},
                {'name': 'Side', 'id': 'side'},
                {'name': 'Type', 'id': 'order_type'},
                {'name': 'Quantity', 'id': 'quantity', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                {'name': 'Status', 'id': 'status'},
                {'name': 'Created', 'id': 'created_at'},
            ],
            style_cell={
                'backgroundColor': THEME['card'],
                'color': THEME['text'],
                'border': f"1px solid {THEME['border']}",
                'textAlign': 'left',
                'fontFamily': 'Inter, sans-serif',
                'fontSize': '13px',
                'maxWidth': '150px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'
            },
            style_header={
                'backgroundColor': THEME['background'],
                'color': THEME['text'],
                'fontWeight': '600',
                'border': f"1px solid {THEME['border']}"
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{status} = PENDING'},
                    'backgroundColor': f"rgba(216, 153, 34, 0.1)",
                    'color': THEME['yellow']
                },
                {
                    'if': {'filter_query': '{status} = FILLED'},
                    'backgroundColor': f"rgba(63, 185, 80, 0.1)",
                    'color': THEME['green']
                }
            ]
        )
        
        return table
    except Exception as e:
        return html.P(f"Error loading orders: {str(e)}", style={'color': THEME['red']})

@app.callback(
    Output('market-prices', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_market_prices(n):
    """Update market prices display."""
    try:
        markets = get_market_prices()
        
        if not markets:
            return html.P("No market data available", style={'color': THEME['text_muted']})
        
        price_cards = []
        for market in markets[:10]:  # Show first 10
            symbol = market.get('symbol', 'N/A')
            price = market.get('price', 0)
            change_pct = market.get('change_pct_24h', 0)
            change_color = THEME['green'] if change_pct >= 0 else THEME['red']
            
            card = html.Div([
                html.Div(symbol, style={
                    'font-weight': '600',
                    'font-size': '14px',
                    'margin-bottom': '4px',
                    'color': THEME['text']
                }),
                html.Div(f"${price:,.2f}", style={
                    'font-size': '18px',
                    'font-weight': '600',
                    'margin-bottom': '4px',
                    'color': THEME['text']
                }),
                html.Div(f"{change_pct:+.2f}%", style={
                    'font-size': '12px',
                    'color': change_color
                }),
            ], style={
                'background': THEME['background'],
                'padding': '16px',
                'border-radius': '6px',
                'border': f"1px solid {THEME['border']}",
                'min-width': '140px',
                'text-align': 'center'
            })
            
            price_cards.append(card)
        
        return price_cards
    except Exception as e:
        return html.P(f"Error loading market data: {str(e)}", style={'color': THEME['red']})

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("HEDGE FUND TRADING DASHBOARD")
    print("="*70)
    print(f"\nDashboard URL: http://localhost:8050")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"\nStarting dashboard...")
    print("Press Ctrl+C to stop\n")
    
    app.run_server(debug=True, host='0.0.0.0', port=8050)
