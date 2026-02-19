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
from src.risk import calculate_all_risk_metrics, max_drawdown, rolling_drawdown
from src.ml_model import MLSignalModel
from src.performance import generate_performance_report, calculate_all_performance_metrics
from src.fund_simulator import FundSimulator, generate_fund_report
from src.hedgefund_ml import (
    HiddenMarkovRegimeDetector,
    MarketRegime,
    MetaLabelGenerator,
    MetaLabelConfig,
    HedgeFundMLPipeline
)
from src.portfolio_optimizer import PortfolioOptimizer
from src.performance import calculate_risk_adjusted_returns


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
        html.H1("🤖 AI Trading System v2.1 - Hedge Fund Edition", className="header-title"),
        html.Div([
            html.Span("Professional Trading Dashboard with Regime Detection & Meta-Labeling", className="header-subtitle")
        ])
    ], className="header"),
    
    # Controls
    html.Div([
        html.Div([
            html.Label("🪙 Symbol:", className="control-label"),
            dcc.Dropdown(
                id='symbol-selector',
                options=[
                    {'label': '🟠 BTC/USDT - Bitcoin', 'value': 'BTC/USDT'},
                    {'label': '◈ ETH/USDT - Ethereum', 'value': 'ETH/USDT'},
                    {'label': '🟡 BNB/USDT - Binance', 'value': 'BNB/USDT'},
                    {'label': '🔷 SOL/USDT - Solana', 'value': 'SOL/USDT'},
                    {'label': '❄️ XRP/USDT - Ripple', 'value': 'XRP/USDT'},
                    {'label': '🐕 DOGE/USDT - Dogecoin', 'value': 'DOGE/USDT'},
                    {'label': '🔗 LINK/USDT - Chainlink', 'value': 'LINK/USDT'},
                    {'label': '📊 ADA/USDT - Cardano', 'value': 'ADA/USDT'},
                    {'label': '🌐 DOT/USDT - Polkadot', 'value': 'DOT/USDT'},
                    {'label': '🌀 AVAX/USDT - Avalanche', 'value': 'AVAX/USDT'},
                    {'label': '🎯 MATIC/USDT - Polygon', 'value': 'MATIC/USDT'},
                    {'label': '⚡ ATOM/USDT - Cosmos', 'value': 'ATOM/USDT'},
                ],
                value=DEFAULT_SYMBOL,
                className="control-dropdown",
                searchable=True,
                clearable=False
            )
        ], className="control-group"),
        
        html.Div([
            html.Label("⏱️ Timeframe:", className="control-label"),
            dcc.Dropdown(
                id='timeframe-selector',
                options=[
                    {'label': '📈 15 Minutes', 'value': '15m'},
                    {'label': '📊 1 Hour', 'value': '1h'},
                    {'label': '📈 4 Hours', 'value': '4h'},
                    {'label': '📅 1 Day', 'value': '1d'},
                    {'label': '📆 1 Week', 'value': '1w'},
                    {'label': '🗓️ 1 Month', 'value': '1M'},
                ],
                value=DEFAULT_TIMEFRAME,
                className="control-dropdown",
                clearable=False
            )
        ], className="control-group"),
        
        html.Div([
            html.Button([
                html.Span("🔄 ", className="btn-icon"),
                "Refresh"
            ], id='refresh-btn', className="btn-refresh btn-animated")
        ], className="control-group"),
        
        # Live indicator
        html.Div([
            html.Div([
                html.Span(className="live-indicator"),
                " LIVE"
            ], className="live-badge")
        ], className="control-group")
    ], className="controls"),
    
    # Core v2.0 Live Status
    html.Div([
        html.Div([
            html.Div([
                html.Span(className="live-indicator"),
                " Core Status"
            ], className="metric-label"),
            html.Div("Ready", className="metric-value metric-value-live")
        ], className="metric-card metric-card-highlight"),
        
        html.Div([
            html.Div("Mode", className="metric-label"),
            html.Div("Paper", className="metric-value")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Risk Level", className="metric-label"),
            html.Div("LOW", className="metric-value metric-value-risk")
        ], className="metric-card"),
    ], className="metrics-row"),
    
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
    
    # ML & Fund Metrics Row
    html.Div([
        html.Div([
            html.Div("ML Accuracy", className="metric-label"),
            html.Div(id='ml-accuracy', className="metric-value metric-value-ml")
        ], className="metric-card"),
        
        html.Div([
            html.Div("ML Confidence", className="metric-label"),
            html.Div(id='ml-confidence', className="metric-value metric-value-ml")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Sortino", className="metric-label"),
            html.Div(id='sortino-ratio', className="metric-value")
        ], className="metric-card"),
        
        html.Div([
            html.Div("VaR 95%", className="metric-label"),
            html.Div(id='var-metric', className="metric-value")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Fund Net Return", className="metric-label"),
            html.Div(id='fund-net-return', className="metric-value metric-value-fund")
        ], className="metric-card"),
    ], className="metrics-row"),
    
    # Hedge Fund Metrics Row
    html.Div([
        html.Div([
            html.Div("Market Regime", className="metric-label"),
            html.Div(id='regime-display', className="metric-value metric-value-live")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Regime Confidence", className="metric-label"),
            html.Div(id='regime-confidence', className="metric-value")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Meta-Label Acc", className="metric-label"),
            html.Div(id='meta-label-acc', className="metric-value metric-value-ml")
        ], className="metric-card"),
        
        html.Div([
            html.Div("CVaR 95%", className="metric-label"),
            html.Div(id='cvar-metric', className="metric-value")
        ], className="metric-card"),
        
        html.Div([
            html.Div("Best Strategy", className="metric-label"),
            html.Div(id='best-strategy', className="metric-value metric-value-fund")
        ], className="metric-card"),
    ], className="metrics-row"),
    
    # Drawdown Chart Row
    html.Div([
        html.Div([
            dcc.Graph(id='drawdown-chart', className="chart")
        ], className="chart-container"),
        
        # Regime Detection Chart
        html.Div([
            dcc.Graph(id='regime-chart', className="chart")
        ], className="chart-container"),
    ], className="charts-row"),
    
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
    dcc.Store(id='ml-results'),
    dcc.Store(id='fund-results'),
    dcc.Store(id='hedgefund-results'),
    
    # Auto-refresh interval
    dcc.Interval(
        id='auto-refresh',
        interval=60*1000,  # 1 minute
        n_intervals=0
    ),
    
    # Footer
    html.Div([
        html.Div("🤖 AI Trading System v2.1 | Powered by Machine Learning", className="footer-text"),
        html.Div("© 2026 Hedge Fund Edition | Regime Detection & Meta-Labeling", className="footer-subtext")
    ], className="footer")
    
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


# ML Model Callback
@callback(
    Output('ml-results', 'data'),
    Input('market-data', 'data')
)
def run_ml_analysis(data):
    """Run ML model analysis."""
    df = pd.DataFrame.from_dict(data)
    
    try:
        model = MLSignalModel('random_forest', n_estimators=50)
        metrics = model.train(df)
        signals = model.predict_signals(df)
        
        # Get latest prediction confidence
        result = model.predict(df)
        latest_prob = result.probability[-1] if len(result.probability) > 0 else 0.5
        
        return {
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0),
            'confidence': latest_prob,
            'signals': signals.to_dict()
        }
    except Exception as e:
        print(f"ML Error: {e}")
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'confidence': 0.5,
            'signals': {}
        }


# Fund Simulator Callback
@callback(
    Output('fund-results', 'data'),
    Input('backtest-results', 'data')
)
def run_fund_simulation(data):
    """Run fund simulation with fees."""
    if not data or 'equity' not in data:
        return {'net_return': 0, 'total_fees': 0}
    
    try:
        equity = pd.Series(data['equity'])
        
        fund = FundSimulator(initial_capital=1000000)
        adjusted, metrics = fund.apply_fees(equity)
        
        return {
            'net_return': metrics.net_return * 100,
            'gross_return': metrics.gross_return * 100,
            'total_fees': metrics.total_fees,
            'management_fee': metrics.management_fee,
            'performance_fee': metrics.performance_fee,
            'final_aum': metrics.aum_final
        }
    except Exception as e:
        print(f"Fund Error: {e}")
        return {'net_return': 0, 'total_fees': 0}


# ML & Fund Metrics Callback
@callback(
    [Output('ml-accuracy', 'children'),
     Output('ml-confidence', 'children'),
     Output('sortino-ratio', 'children'),
     Output('var-metric', 'children'),
     Output('fund-net-return', 'children')],
    [Input('ml-results', 'data'),
     Input('backtest-results', 'data'),
     Input('fund-results', 'data')]
)
def update_ml_fund_metrics(ml_data, backtest_data, fund_data):
    """Update ML and Fund metrics."""
    # ML metrics
    ml_acc = f"{ml_data.get('accuracy', 0)*100:.1f}%" if ml_data else "N/A"
    ml_conf = f"{ml_data.get('confidence', 0.5)*100:.1f}%" if ml_data else "N/A"
    
    # Risk metrics from backtest
    sortino = "N/A"
    var_val = "N/A"
    if backtest_data and 'risk_metrics' in backtest_data:
        risk = backtest_data['risk_metrics']
        sortino = f"{risk.get('sortino_ratio', 0):.2f}"
        var_val = f"{risk.get('var_95', 0)*100:.2f}%"
    
    # Fund metrics
    fund_net = "N/A"
    if fund_data:
        fund_net = f"{fund_data.get('net_return', 0):+.2f}%"
    
    return ml_acc, ml_conf, sortino, var_val, fund_net


# Drawdown Chart Callback
@callback(
    Output('drawdown-chart', 'figure'),
    Input('backtest-results', 'data')
)
def update_drawdown_chart(data):
    """Update drawdown chart."""
    if not data or 'equity' not in data:
        fig = go.Figure()
        fig.update_layout(
            title='Drawdown',
            template='plotly_dark',
            height=250
        )
        return fig
    
    equity = pd.Series(data['equity'])
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak * 100
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#ff4444', width=2)
        )
    )
    
    fig.update_layout(
        title='Drawdown (%)',
        template='plotly_dark',
        height=250,
        yaxis_title='Drawdown %'
    )
    
    return fig


# Hedge Fund Analysis Callback
@callback(
    Output('hedgefund-results', 'data'),
    [Input('market-data', 'data'),
     Input('backtest-results', 'data')]
)
def run_hedgefund_analysis(market_data, backtest_data):
    """Run hedge fund analysis including regime detection and meta-labeling."""
    if not market_data:
        return {'regime': 'unknown', 'confidence': 0, 'meta_acc': 0, 'cvar': 0, 'best_strategy': 'N/A'}
    
    try:
        df = pd.DataFrame.from_dict(market_data)
        
        # Run regime detection
        regime_detector = HiddenMarkovRegimeDetector(n_states=4)
        regime_result = regime_detector.fit_predict(df)
        
        # Get current regime
        current_regime = regime_result.current_regime.regime.value if regime_result.current_regime else 'unknown'
        regime_confidence = regime_result.current_regime.probability if regime_result.current_regime else 0
        
        # Run meta-labeling if we have backtest signals
        meta_acc = 0
        if backtest_data and 'signals' in backtest_data:
            try:
                meta_gen = MetaLabelGenerator()
                signals = pd.Series(backtest_data['signals'])
                # Generate meta-labels
                meta_labels = meta_gen.generate_labels(df, signals)
                # Calculate meta-label accuracy (comparing to actual returns)
                if len(meta_labels) > 0:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) == len(meta_labels):
                        meta_acc = (meta_labels == (returns > 0)).mean()
            except Exception as e:
                print(f"Meta-labeling error: {e}")
        
        # Calculate CVaR from backtest equity
        cvar = 0
        if backtest_data and 'equity' in backtest_data:
            try:
                equity = pd.Series(backtest_data['equity'])
                returns = equity.pct_change().dropna()
                # CVaR 95%
                var_threshold = returns.quantile(0.05)
                cvar = returns[returns <= var_threshold].mean() * 100
            except:
                pass
        
        # Determine best strategy based on regime
        best_strategy = 'momentum'  # default
        if current_regime == 'trending_up':
            best_strategy = 'momentum'
        elif current_regime == 'trending_down':
            best_strategy = 'mean_reversion'
        elif current_regime == 'mean_reverting':
            best_strategy = 'mean_reversion'
        elif current_regime == 'high_volatility':
            best_strategy = 'risk_parity'
        elif current_regime == 'low_volatility':
            best_strategy = 'momentum'
        
        return {
            'regime': current_regime,
            'confidence': regime_confidence,
            'meta_acc': meta_acc,
            'cvar': cvar,
            'best_strategy': best_strategy,
            'regime_history': [r.regime.value for r in regime_result.regime_history[-50:]] if regime_result.regime_history else []
        }
    except Exception as e:
        print(f"Hedge fund analysis error: {e}")
        return {'regime': 'unknown', 'confidence': 0, 'meta_acc': 0, 'cvar': 0, 'best_strategy': 'N/A'}


# Hedge Fund Metrics Callback
@callback(
    [Output('regime-display', 'children'),
     Output('regime-confidence', 'children'),
     Output('meta-label-acc', 'children'),
     Output('cvar-metric', 'children'),
     Output('best-strategy', 'children')],
    Input('hedgefund-results', 'data')
)
def update_hedgefund_metrics(data):
    """Update hedge fund metric displays."""
    if not data:
        return "UNKNOWN", "0%", "0%", "0%", "N/A"
    
    regime = data.get('regime', 'unknown').replace('_', ' ').title()
    confidence = f"{data.get('confidence', 0)*100:.1f}%"
    meta_acc = f"{data.get('meta_acc', 0)*100:.1f}%"
    cvar = f"{data.get('cvar', 0):.2f}%"
    best_strategy = data.get('best_strategy', 'N/A').title()
    
    return regime, confidence, meta_acc, cvar, best_strategy


# Regime Chart Callback
@callback(
    Output('regime-chart', 'figure'),
    Input('hedgefund-results', 'data')
)
def update_regime_chart(data):
    """Update regime detection chart."""
    if not data or 'regime_history' not in data:
        fig = go.Figure()
        fig.update_layout(
            title='Market Regime Detection',
            template='plotly_dark',
            height=250
        )
        return fig
    
    regime_history = data.get('regime_history', [])
    
    # Map regimes to colors
    regime_colors = {
        'trending_up': '#00ff88',
        'trending_down': '#ff4444',
        'mean_reverting': '#ffaa00',
        'high_volatility': '#ff00ff',
        'low_volatility': '#00aaff',
        'consolidation': '#888888',
        'unknown': '#666666'
    }
    
    # Create numeric values for y-axis
    regime_numeric = []
    for r in regime_history:
        if r == 'trending_up':
            regime_numeric.append(5)
        elif r == 'trending_down':
            regime_numeric.append(4)
        elif r == 'mean_reverting':
            regime_numeric.append(3)
        elif r == 'high_volatility':
            regime_numeric.append(2)
        elif r == 'low_volatility':
            regime_numeric.append(1)
        else:
            regime_numeric.append(0)
    
    fig = go.Figure()
    
    if regime_numeric:
        # Create step line for regime
        x_vals = list(range(len(regime_numeric)))
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=regime_numeric,
            mode='lines',
            name='Regime',
            fill='tozeroy',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Market Regime History',
        template='plotly_dark',
        height=250,
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=['Unknown', 'Low Vol', 'High Vol', 'Mean Rev', 'Trend Down', 'Trend Up']
        ),
        xaxis_title='Time Period',
        yaxis_title='Regime'
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
                text-shadow: 0 0 20px rgba(88, 166, 255, 0.3);
                animation: glow 2s ease-in-out infinite alternate;
            }
            @keyframes glow {
                from { text-shadow: 0 0 10px rgba(88, 166, 255, 0.3); }
                to { text-shadow: 0 0 25px rgba(88, 166, 255, 0.6); }
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
                transition: all 0.3s ease;
            }
            .metric-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
                border-color: #58a6ff;
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
            .metric-value-ml {
                color: #a371f7;
            }
            .metric-value-fund {
                color: #3fb950;
            }
            .metric-value-live {
                color: #f0883e;
            }
            .metric-value-risk {
                color: #f85149;
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
                transition: all 0.3s ease;
            }
            .chart-container:hover {
                border-color: #58a6ff;
                box-shadow: 0 4px 20px rgba(88, 166, 255, 0.1);
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
            /* Live indicator animation */
            .live-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: #3fb950;
                margin-right: 8px;
                animation: pulse 1.5s ease-in-out infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.7); }
                70% { opacity: 0.7; box-shadow: 0 0 0 10px rgba(63, 185, 80, 0); }
                100% { opacity: 1; box-shadow: 0 0 0 0 rgba(63, 185, 80, 0); }
            }
            .btn-animated {
                transition: all 0.3s ease;
            }
            .btn-animated:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 15px rgba(35, 134, 54, 0.4);
            }
            .metric-card-highlight {
                border-left: 3px solid #3fb950;
            }
            /* Footer styling */
            .dashboard-footer {
                text-align: center;
                padding: 20px;
                margin-top: 30px;
                border-top: 1px solid #30363d;
                color: #8b949e;
                font-size: 0.85em;
            }
            .dashboard-footer a {
                color: #58a6ff;
                text-decoration: none;
            }
            .dashboard-footer a:hover {
                text-decoration: underline;
            }
            /* Responsive adjustments */
            @media (max-width: 768px) {
                .header-title { font-size: 1.8em; }
                .metrics-row { justify-content: flex-start; }
                .chart-container { min-width: 100%; }
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
            /* Live Indicator */
            .live-badge {
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(63, 185, 80, 0.1);
                border: 1px solid #3fb950;
                border-radius: 20px;
                padding: 8px 16px;
                color: #3fb950;
                font-weight: bold;
                font-size: 0.9em;
            }
            .live-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                background-color: #3fb950;
                border-radius: 50%;
                animation: pulse 1.5s ease-in-out infinite;
                margin-right: 5px;
            }
            @keyframes pulse {
                0% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.5; transform: scale(1.2); }
                100% { opacity: 1; transform: scale(1); }
            }
            /* Footer */
            .footer {
                text-align: center;
                padding: 25px;
                margin-top: 30px;
                border-top: 1px solid #30363d;
                background: linear-gradient(180deg, transparent 0%, rgba(22, 27, 34, 0.8) 100%);
            }
            .footer-text {
                color: #8b949e;
                font-size: 1em;
                margin-bottom: 5px;
            }
            .footer-subtext {
                color: #6e7681;
                font-size: 0.85em;
            }
            /* Button Animation */
            .btn-animated {
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
            }
            .btn-animated::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s ease;
            }
            .btn-animated:hover::before {
                left: 100%;
            }
            .btn-icon {
                margin-right: 5px;
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
