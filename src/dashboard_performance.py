"""
Dashboard Performance Extension
===============================
Adds equity curve and drawdown visualization to the trading dashboard.

Features:
- Real-time equity curve
- Drawdown visualization
- Performance metrics display
- Risk status indicators
- Monte Carlo level indicators

Author: AI Trading System
Version: 1.0.0
"""

import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DashboardPerformanceExtension:
    """
    Extension for adding performance visualization to dashboard.
    """
    
    def __init__(self, app: dash.Dash, performance_monitor: Any, risk_guard: Any):
        """
        Initialize the extension.
        
        Args:
            app: Dash application
            performance_monitor: PerformanceMonitor instance
            risk_guard: RiskGuard instance
        """
        self.app = app
        self.performance_monitor = performance_monitor
        self.risk_guard = risk_guard
        
        self._register_callbacks()
        logger.info("DashboardPerformanceExtension initialized")
    
    def _register_callbacks(self):
        """Register dashboard callbacks."""
        
        @self.app.callback(
            [
                Output('equity-curve-graph', 'figure'),
                Output('drawdown-graph', 'figure'),
                Output('performance-metrics', 'children'),
                Output('risk-status', 'children'),
                Output('monte-carlo-indicator', 'children')
            ],
            [
                Input('performance-interval', 'n_intervals'),
                Input('refresh-button', 'n_clicks')
            ]
        )
        def update_performance_charts(n_intervals, n_clicks):
            """Update performance charts and metrics."""
            # Get equity curve
            equity_df = self.performance_monitor.get_equity_curve()
            
            # Create equity curve figure
            equity_fig = self._create_equity_curve(equity_df)
            
            # Create drawdown figure
            drawdown_fig = self._create_drawdown_chart(equity_df)
            
            # Get performance metrics
            metrics = self.performance_monitor.get_summary()
            metrics_div = self._create_metrics_div(metrics)
            
            # Get risk status
            risk_status = self._create_risk_status()
            
            # Get Monte Carlo indicator
            mc_indicator = self._create_mc_indicator()
            
            return equity_fig, drawdown_fig, metrics_div, risk_status, mc_indicator
    
    def _create_equity_curve(self, equity_df: pd.DataFrame) -> go.Figure:
        """Create equity curve figure."""
        fig = go.Figure()
        
        if equity_df.empty:
            # Create empty figure with placeholder
            fig.add_annotation(
                text="No equity data yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color="gray")
            )
            fig.update_layout(
                title="Equity Curve",
                template="plotly_dark",
                height=300
            )
            return fig
        
        # Add equity line
        fig.add_trace(go.Scatter(
            x=equity_df.index,
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#00D4AA', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)'
        ))
        
        # Add initial capital line
        initial_capital = self.performance_monitor.initial_capital
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital"
        )
        
        # Add peak line
        peak_capital = self.performance_monitor.peak_capital
        if peak_capital > initial_capital:
            fig.add_hline(
                y=peak_capital,
                line_dash="dot",
                line_color="#FFD700",
                annotation_text="Peak"
            )
        
        fig.update_layout(
            title="Equity Curve",
            template="plotly_dark",
            height=300,
            showlegend=True,
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            hovermode='x unified'
        )
        
        return fig
    
    def _create_drawdown_chart(self, equity_df: pd.DataFrame) -> go.Figure:
        """Create drawdown chart."""
        fig = go.Figure()
        
        if equity_df.empty:
            fig.add_annotation(
                text="No drawdown data yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color="gray")
            )
            fig.update_layout(
                title="Drawdown",
                template="plotly_dark",
                height=200
            )
            return fig
        
        # Calculate drawdown
        equity = equity_df['equity']
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        # Add drawdown area
        fig.add_trace(go.Scatter(
            x=equity_df.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='#FF6B6B', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.3)'
        ))
        
        # Add warning threshold line
        fig.add_hline(
            y=-10,
            line_dash="dash",
            line_color="#FFA500",
            annotation_text="Warning (-10%)"
        )
        
        # Add critical threshold line
        fig.add_hline(
            y=-20,
            line_dash="dash",
            line_color="#FF0000",
            annotation_text="Critical (-20%)"
        )
        
        fig.update_layout(
            title="Drawdown (%)",
            template="plotly_dark",
            height=200,
            showlegend=False,
            xaxis_title="Time",
            yaxis_title="Drawdown (%)",
            hovermode='x unified'
        )
        
        return fig
    
    def _create_metrics_div(self, metrics: Dict) -> html.Div:
        """Create performance metrics display."""
        capital = metrics.get('capital', {})
        trades = metrics.get('trades', {})
        risk = metrics.get('risk', {})
        streaks = metrics.get('streaks', {})
        
        # Determine P&L color
        pnl_percent = capital.get('total_pnl_percent', '0%')
        pnl_color = '#00D4AA' if '+' in pnl_percent or float(pnl_percent.replace('%', '')) > 0 else '#FF6B6B'
        
        return html.Div([
            # Capital Row
            html.Div([
                html.Div([
                    html.H4("Capital", className="metric-label"),
                    html.P(f"${capital.get('current', 0):,.0f}", className="metric-value")
                ], className="metric-box"),
                html.Div([
                    html.H4("Total P&L", className="metric-label"),
                    html.P(pnl_percent, className="metric-value", style={'color': pnl_color})
                ], className="metric-box"),
                html.Div([
                    html.H4("Peak", className="metric-label"),
                    html.P(f"${capital.get('peak', 0):,.0f}", className="metric-value")
                ], className="metric-box"),
            ], className="metrics-row"),
            
            # Trades Row
            html.Div([
                html.Div([
                    html.H4("Total Trades", className="metric-label"),
                    html.P(str(trades.get('total', 0)), className="metric-value")
                ], className="metric-box"),
                html.Div([
                    html.H4("Win Rate", className="metric-label"),
                    html.P(trades.get('win_rate', '0%'), className="metric-value")
                ], className="metric-box"),
                html.Div([
                    html.H4("Open Positions", className="metric-label"),
                    html.P(str(trades.get('open_positions', 0)), className="metric-value")
                ], className="metric-box"),
            ], className="metrics-row"),
            
            # Risk Row
            html.Div([
                html.Div([
                    html.H4("Max Drawdown", className="metric-label"),
                    html.P(risk.get('max_drawdown_percent', '0%'), className="metric-value")
                ], className="metric-box"),
                html.Div([
                    html.H4("Sharpe Ratio", className="metric-label"),
                    html.P(risk.get('sharpe_ratio', '0.00'), className="metric-value")
                ], className="metric-box"),
                html.Div([
                    html.H4("Sortino Ratio", className="metric-label"),
                    html.P(risk.get('sortino_ratio', '0.00'), className="metric-value")
                ], className="metric-box"),
            ], className="metrics-row"),
            
            # Streaks Row
            html.Div([
                html.Div([
                    html.H4("Max Consecutive Wins", className="metric-label"),
                    html.P(str(streaks.get('max_consecutive_wins', 0)), className="metric-value", style={'color': '#00D4AA'})
                ], className="metric-box"),
                html.Div([
                    html.H4("Max Consecutive Losses", className="metric-label"),
                    html.P(str(streaks.get('max_consecutive_losses', 0)), className="metric-value", style={'color': '#FF6B6B'})
                ], className="metric-box"),
                html.Div([
                    html.H4("Current Streak", className="metric-label"),
                    html.P(self._format_streak(streaks.get('current_streak', 0)), className="metric-value")
                ], className="metric-box"),
            ], className="metrics-row"),
        ])
    
    def _format_streak(self, streak: int) -> str:
        """Format streak value."""
        if streak > 0:
            return f"+{streak} W"
        elif streak < 0:
            return f"{streak} L"
        else:
            return "0"
    
    def _create_risk_status(self) -> html.Div:
        """Create risk status indicator."""
        status = self.risk_guard.get_status()
        
        risk_level = status.get('risk_level', 'NORMAL')
        trading_status = status.get('status', 'ACTIVE')
        can_trade = status.get('can_trade', True)
        halt_reason = status.get('halt_reason', '')
        
        # Determine colors
        level_colors = {
            'NORMAL': '#00D4AA',
            'WARNING': '#FFA500',
            'CRITICAL': '#FF6B6B',
            'EMERGENCY': '#FF0000'
        }
        
        status_colors = {
            'ACTIVE': '#00D4AA',
            'PAUSED': '#FFA500',
            'HALTED': '#FF0000',
            'LOCKED': '#FF0000'
        }
        
        level_color = level_colors.get(risk_level, 'gray')
        status_color = status_colors.get(trading_status, 'gray')
        
        return html.Div([
            html.Div([
                html.H4("Risk Level", className="status-label"),
                html.Div([
                    html.Span("●", style={'color': level_color, 'font-size': '24px'}),
                    html.Span(risk_level, style={'color': level_color, 'font-weight': 'bold', 'margin-left': '8px'})
                ])
            ], className="status-box"),
            html.Div([
                html.H4("Trading Status", className="status-label"),
                html.Div([
                    html.Span("●", style={'color': status_color, 'font-size': '24px'}),
                    html.Span(trading_status, style={'color': status_color, 'font-weight': 'bold', 'margin-left': '8px'})
                ])
            ], className="status-box"),
            html.Div([
                html.H4("Can Trade", className="status-label"),
                html.Span("✓ YES" if can_trade else "✗ NO", 
                         style={'color': '#00D4AA' if can_trade else '#FF6B6B', 'font-weight': 'bold'})
            ], className="status-box"),
            *([html.Div([
                html.H4("Halt Reason", className="status-label"),
                html.Span(halt_reason, style={'color': '#FF6B6B'})
            ], className="status-box")] if halt_reason else [])
        ], className="risk-status-container")
    
    def _create_mc_indicator(self) -> html.Div:
        """Create Monte Carlo level indicator."""
        # This would be connected to actual MC results in production
        return html.Div([
            html.H4("Monte Carlo Analysis", className="mc-title"),
            html.Div([
                html.Div([
                    html.Span("Level 1", className="mc-level-label"),
                    html.Div(className="mc-level-bar mc-level-active")
                ], className="mc-level-row"),
                html.Div([
                    html.Span("Level 2", className="mc-level-label"),
                    html.Div(className="mc-level-bar mc-level-active")
                ], className="mc-level-row"),
                html.Div([
                    html.Span("Level 3", className="mc-level-label"),
                    html.Div(className="mc-level-bar mc-level-active")
                ], className="mc-level-row"),
                html.Div([
                    html.Span("Level 4", className="mc-level-label"),
                    html.Div(className="mc-level-bar mc-level-active")
                ], className="mc-level-row"),
                html.Div([
                    html.Span("Level 5", className="mc-level-label"),
                    html.Div(className="mc-level-bar mc-level-active")
                ], className="mc-level-row"),
            ], className="mc-levels-container"),
            html.Div([
                html.Span("Confidence: ", style={'color': 'gray'}),
                html.Span("75%", style={'color': '#00D4AA', 'font-weight': 'bold'})
            ], className="mc-confidence")
        ], className="mc-indicator-container")


def get_performance_layout() -> html.Div:
    """Get the performance section layout for dashboard."""
    return html.Div([
        html.H2("📊 Performance Monitor", className="section-title"),
        
        # Interval for updates
        dcc.Interval(
            id='performance-interval',
            interval=5000,  # 5 seconds
            n_intervals=0
        ),
        
        # Risk Status Row
        html.Div([
            html.H3("Risk Status", className="subsection-title"),
            html.Div(id='risk-status', className="risk-status")
        ], className="section"),
        
        # Equity Curve
        html.Div([
            html.H3("Equity Curve", className="subsection-title"),
            dcc.Graph(id='equity-curve-graph')
        ], className="section"),
        
        # Drawdown
        html.Div([
            html.H3("Drawdown", className="subsection-title"),
            dcc.Graph(id='drawdown-graph')
        ], className="section"),
        
        # Performance Metrics
        html.Div([
            html.H3("Performance Metrics", className="subsection-title"),
            html.Div(id='performance-metrics', className="metrics-container")
        ], className="section"),
        
        # Monte Carlo Indicator
        html.Div([
            html.H3("Monte Carlo Analysis", className="subsection-title"),
            html.Div(id='monte-carlo-indicator', className="mc-container")
        ], className="section"),
        
        # Refresh Button
        html.Button("Refresh", id='refresh-button', n_clicks=0, className="refresh-btn")
    ], className="performance-section")


# CSS Styles for the performance section
PERFORMANCE_STYLES = """
.section-title {
    color: #00D4AA;
    margin-bottom: 20px;
}

.subsection-title {
    color: #888;
    font-size: 16px;
    margin-bottom: 10px;
}

.section {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}

.metrics-container {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.metrics-row {
    display: flex;
    justify-content: space-around;
    gap: 10px;
}

.metric-box {
    background: #16213e;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    flex: 1;
}

.metric-label {
    color: #888;
    font-size: 12px;
    margin: 0 0 5px 0;
}

.metric-value {
    color: #fff;
    font-size: 18px;
    font-weight: bold;
    margin: 0;
}

.risk-status-container {
    display: flex;
    justify-content: space-around;
    gap: 15px;
}

.status-box {
    background: #16213e;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    flex: 1;
}

.status-label {
    color: #888;
    font-size: 12px;
    margin: 0 0 5px 0;
}

.mc-indicator-container {
    background: #16213e;
    border-radius: 8px;
    padding: 15px;
}

.mc-title {
    color: #888;
    font-size: 14px;
    margin: 0 0 15px 0;
}

.mc-levels-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.mc-level-row {
    display: flex;
    align-items: center;
    gap: 10px;
}

.mc-level-label {
    color: #888;
    font-size: 12px;
    width: 60px;
}

.mc-level-bar {
    flex: 1;
    height: 8px;
    background: #333;
    border-radius: 4px;
}

.mc-level-active {
    background: linear-gradient(90deg, #00D4AA, #00FF88);
}

.mc-confidence {
    margin-top: 15px;
    text-align: center;
}

.refresh-btn {
    background: #00D4AA;
    color: #000;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    font-weight: bold;
}

.refresh-btn:hover {
    background: #00FF88;
}
"""
