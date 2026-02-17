"""
Professional Dashboard Application
=================================
Advanced trading dashboard with:
- Real-time price charts
- Technical indicators
- Equity curve visualization
- Risk metrics display
- Multi-symbol analysis

Author: AI Trading System
Version: 1.0.0
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.indicators import calculate_all_indicators, rsi, macd, bollinger_bands
from src.signal_engine import generate_composite_signal, detect_trend
from src.backtest import run_backtest
from src.risk import calculate_all_risk_metrics, max_drawdown


# Initialize app
app = dash.Dash(
    __name__,
    title="AI Trading System - Professional Dashboard",
    update_title=None
)

# Default configuration
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1d"

# Sample data generator for demo
def generate_sample_data(days: int = 365) -> pd.DataFrame:
    """Generate sample OHLCV data for demonstration."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate random walk prices
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)
    prices = 100 * (1 + returns).cumprod()
    
    # Generate OHLC
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'high': prices * (1 + np.random.uniform(0, 0.02, days)),
        'low': prices * (1 - np.random.uniform(0, 0.02, days)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, days)
    })
    
    df.set_index('date', inplace=True)
    return df


# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🤖 AI Trading System", className="header-title"),
        html.Div([
            html.Span("Professional Trading Dashboard", className="header-subtitle")
        ])
    ], className="header"),
    
    # Controls
    html.Div([
        html.Div([
            html.Label("Symbol:", className="control-label"),
            dcc.Dropdown(
                id='symbol-selector',
                options=[
                    {'label': 'BTC/USDT', 'value': 'BTC/USDT'},
                    {'label': 'ETH/USDT', 'value': 'ETH/USDT'},
                    {'label': 'BNB/USDT', 'value': 'BNB/USDT'},
                    {'label': 'SOL/USDT', 'value': 'SOL/USDT'},
                    {'label': 'XRP/USDT', 'value': 'XRP/USDT'},
                ],
                value=DEFAULT_SYMBOL,
                className="control-dropdown"
            )
        ], className="control-group"),
        
        html.Div([
            html.Label("Timeframe:", className="control-label"),
            dcc.Dropdown(
                id='timeframe-selector',
                options=[
                    {'label': '1 Hour', 'value': '1h'},
                    {'label': '4 Hours', 'value': '4h'},
                    {'label': '1 Day', 'value': '1d'},
                    {'label': '1 Week', 'value': '1w'},
                ],
                value=DEFAULT_TIMEFRAME,
                className="control-dropdown"
            )
        ], className="control-group"),
        
        html.Div([
            html.Button('🔄 Refresh', id='refresh-btn', className="btn-refresh")
        ], className="control-group")
    ], className="controls"),
    
    # Metrics Row
    html.Div([
        html.Div([
            html.Div("Total Return", className="metric-label"),
            html.Div(id='total-return', className="metric-value")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Sharpe Ratio", className="metric-label"),
            html.Div(id='sharpe-ratio', className="metric-value")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Max Drawdown", className="metric-label"),
            html.Div(id='max-drawdown', className="metric-value")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Win Rate", className="metric-label"),
            html.Div(id='win-rate', className="metric-value")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Current Signal", className="metric-label"),
            html.Div(id='current-signal', className="metric-value")
        ], className="metric-card"),
    ], className="metrics-row"),
    
    # Main Charts
    html.Div([
        # Price Chart with Indicators
        html.Div([
            dcc.Graph(id='price-chart', className="chart")
        ], className="chart-container"),
        
        # Equity Curve
        html.Div([
            dcc.Graph(id='equity-curve', className="chart")
        ], className="chart-container"),
    ], className="charts-row"),
    
    # Secondary Charts
    html.Div([
        # RSI Chart
        html.Div([
            dcc.Graph(id='rsi-chart', className="chart")
        ], className="chart-container-small"),
        
        # MACD Chart
        html.Div([
            dcc.Graph(id='macd-chart', className="chart")
        ], className="chart-container-small"),
        
        # Signal Distribution
        html.Div([
            dcc.Graph(id='signal-dist', className="chart")
        ], className="chart-container-small"),
    ], className="charts-row"),
    
    # Hidden store for data
    dcc.Store(id='market-data'),
    dcc.Store(id='backtest-results'),
    
    # Auto-refresh interval
    dcc.Interval(
        id='auto-refresh',
        interval=60*1000,  # 1 minute
        n_intervals=0
    )
    
], className="dashboard")


# Callbacks
@callback(
    Output('market-data', 'data'),
    [Input('symbol-selector', 'value'),
     Input('timeframe-selector', 'value'),
     Input('refresh-btn', 'n_clicks'),
     Input('auto-refresh', 'n_intervals')]
)
def load_market_data(symbol, timeframe, n_clicks, n_intervals):
    """Load market data (using sample data for demo)."""
    # In production, use: loader = DataLoader(); df = loader.fetch_ohlcv(symbol, timeframe)
    df = generate_sample_data(365)
    
    # Calculate indicators
    df_with_indicators = calculate_all_indicators(df)
    
    return df_with_indicators.to_dict()


@callback(
    Output('backtest-results', 'data'),
    Input('market-data', 'data')
)
def run_backtest_analysis(data):
    """Run backtest on loaded data."""
    df = pd.DataFrame.from_dict(data)
    
    # Generate signals
    signals = generate_composite_signal(df)
    
    # Run backtest
    result = run_backtest(df, signals, initial_capital=10000)
    
    # Calculate risk metrics
    risk_metrics = calculate_all_risk_metrics(
        result.strategy_returns,
        result.equity_curve
    )
    
    return {
        'metrics': result.metrics,
        'risk_metrics': risk_metrics,
        'signals': signals.to_dict(),
        'equity': result.equity_curve.to_dict()
    }


@callback(
    [Output('total-return', 'children'),
     Output('sharpe-ratio', 'children'),
     Output('max-drawdown', 'children'),
     Output('win-rate', 'children'),
     Output('current-signal', 'children')],
    Input('backtest-results', 'data')
)
def update_metrics(results):
    """Update metric displays."""
    if not results:
        return "0.00%", "0.00", "0.00%", "0.00%", "HOLD"
    
    metrics = results.get('metrics', {})
    risk = results.get('risk_metrics', {})
    
    total_return = metrics.get('total_return', 0) * 100
    sharpe = risk.get('sharpe_ratio', 0)
    max_dd = risk.get('max_drawdown', 0) * 100
    win_rate = metrics.get('win_rate', 0) * 100
    
    # Get current signal
    signals = results.get('signals', {})
    current_signal = 'HOLD'
    if signals:
        signal_values = list(signals.values())
        if signal_values:
            current_signal = signal_values[-1]
    
    return (
        f"{total_return:+.2f}%",
        f"{sharpe:.2f}",
        f"{max_dd:.2f}%",
        f"{win_rate:.2f}%",
        current_signal
    )


@callback(
    Output('price-chart', 'figure'),
    Input('market-data', 'data')
)
def update_price_chart(data):
    """Update main price chart."""
    df = pd.DataFrame.from_dict(data)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                       line=dict(color='gray', width=1), mode='lines'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                       line=dict(color='gray', width=1), mode='lines'),
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Price Chart with Bollinger Bands',
        template='plotly_dark',
        height=500,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig


@callback(
    Output('equity-curve', 'figure'),
    Input('backtest-results', 'data')
)
def update_equity_curve(data):
    """Update equity curve chart."""
    if not data or 'equity' not in data:
        fig = go.Figure()
        fig.update_layout(
            title='Equity Curve',
            template='plotly_dark',
            height=300
        )
        return fig
    
    equity = pd.Series(data['equity'])
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            name='Equity',
            fill='tozeroy',
            line=dict(color='#00ff88', width=2)
        )
    )
    
    # Add benchmark
    initial = equity.iloc[0] if len(equity) > 0 else 10000
    benchmark = pd.Series(initial, index=equity.index)
    fig.add_trace(
        go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='gray', width=1, dash='dash')
        )
    )
    
    fig.update_layout(
        title='Equity Curve vs Benchmark',
        template='plotly_dark',
        height=300,
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)'
    )
    
    return fig


@callback(
    Output('rsi-chart', 'figure'),
    Input('market-data', 'data')
)
def update_rsi_chart(data):
    """Update RSI chart."""
    df = pd.DataFrame.from_dict(data)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['rsi'],
            mode='lines',
            name='RSI',
            line=dict(color='#ff9900', width=2)
        )
    )
    
    # Overbought/Oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    fig.update_layout(
        title='RSI (14)',
        template='plotly_dark',
        height=250,
        yaxis_range=[0, 100]
    )
    
    return fig


@callback(
    Output('macd-chart', 'figure'),
    Input('market-data', 'data')
)
def update_macd_chart(data):
    """Update MACD chart."""
    df = pd.DataFrame.from_dict(data)
    
    fig = go.Figure()
    
    # MACD Line
    fig.add_trace(
        go.Scatter(x=df.index, y=df['macd'], name='MACD',
                   line=dict(color='blue', width=2))
    )
    
    # Signal Line
    fig.add_trace(
        go.Scatter(x=df.index, y=df['macd_signal'], name='Signal',
                   line=dict(color='orange', width=2))
    )
    
    # Histogram
    colors = ['green' if v >= 0 else 'red' for v in df['macd_histogram']]
    fig.add_trace(
        go.Bar(x=df.index, y=df['macd_histogram'], name='Histogram',
               marker_color=colors)
    )
    
    fig.update_layout(
        title='MACD (12, 26, 9)',
        template='plotly_dark',
        height=250
    )
    
    return fig


@callback(
    Output('signal-dist', 'figure'),
    Input('backtest-results', 'data')
)
def update_signal_dist(data):
    """Update signal distribution chart."""
    if not data or 'signals' not in data:
        fig = go.Figure()
        fig.update_layout(
            title='Signal Distribution',
            template='plotly_dark',
            height=250
        )
        return fig
    
    signals = pd.Series(data['signals'])
    signal_counts = signals.value_counts()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=signal_counts.index,
            values=signal_counts.values,
            hole=0.4,
            marker=dict(colors=['#00ff88', '#ff4444', '#888888'])
        )
    ])
    
    fig.update_layout(
        title='Signal Distribution',
        template='plotly_dark',
        height=250
    )
    
    return fig


# Add CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #0d1117;
                color: #c9d1d9;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 0;
            }
            .dashboard {
                padding: 20px;
                max-width: 1600px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                padding: 20px;
                margin-bottom: 20px;
                border-bottom: 1px solid #30363d;
            }
            .header-title {
                color: #58a6ff;
                font-size: 2.5em;
                margin: 0;
            }
            .header-subtitle {
                color: #8b949e;
                font-size: 1.2em;
            }
            .controls {
                display: flex;
                gap: 20px;
                justify-content: center;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            .control-group {
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            .control-label {
                color: #8b949e;
                font-size: 0.9em;
            }
            .control-dropdown {
                min-width: 150px;
            }
            .btn-refresh {
                background-color: #238636;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 1em;
                height: 40px;
                margin-top: auto;
            }
            .btn-refresh:hover {
                background-color: #2ea043;
            }
            .metrics-row {
                display: flex;
                gap: 15px;
                justify-content: center;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            .metric-card {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 15px 25px;
                text-align: center;
                min-width: 150px;
            }
            .metric-label {
                color: #8b949e;
                font-size: 0.9em;
                margin-bottom: 5px;
            }
            .metric-value {
                color: #58a6ff;
                font-size: 1.5em;
                font-weight: bold;
            }
            .charts-row {
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            .chart-container {
                flex: 1;
                min-width: 500px;
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 10px;
            }
            .chart-container-small {
                flex: 1;
                min-width: 300px;
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 10px;
            }
            .chart {
                width: 100%;
            }
            /* Dash component overrides */
            .DashDropdown {
                background-color: #0d1117 !important;
            }
            .Select-control {
                background-color: #0d1117 !important;
                border-color: #30363d !important;
            }
            .Select-menu-outer {
                background-color: #0d1117 !important;
                border-color: #30363d !important;
            }
            .Select-option {
                background-color: #0d1117 !important;
            }
            .Select-option:hover {
                background-color: #21262d !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
