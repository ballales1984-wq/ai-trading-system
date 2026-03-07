import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import json
import os

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "AI Trading System - Analytics Dashboard"

# Sample data (would be replaced with real data from GitHub API, etc.)
data = {
    "github": {
        "stars": 1,
        "forks": 0,
        "issues": 5,
        "pull_requests": 2,
        "contributors": 2,
        "last_updated": "2026-03-07"
    },
    "community": {
        "discord_members": 0,
        "active_users": 0,
        "paper_trading_users": 0,
        "live_trading_users": 0
    },
    "performance": {
        "cagr": 23.5,
        "max_drawdown": 7.2,
        "sharpe_ratio": 1.95,
        "win_rate": 68
    }
}

# Layout
app.layout = html.Div([
    # Header
    html.Header([
        html.Div([
            html.H1("AI Trading System - Analytics Dashboard"),
            html.P("Track project metrics and performance")
        ], className="header-content")
    ], className="header"),

    # Metrics Cards
    html.Section([
        html.Div([
            html.H3("GitHub Metrics"),
            html.Div([
                html.Div([
                    html.H4("Stars"),
                    html.P(f"{data['github']['stars']}"),
                    html.Small("Target: 100+")
                ], className="metric-card"),
                html.Div([
                    html.H4("Forks"),
                    html.P(f"{data['github']['forks']}"),
                    html.Small("Target: 20+")
                ], className="metric-card"),
                html.Div([
                    html.H4("Issues"),
                    html.P(f"{data['github']['issues']}"),
                    html.Small("Active: 5+")
                ], className="metric-card"),
                html.Div([
                    html.H4("Contributors"),
                    html.P(f"{data['github']['contributors']}"),
                    html.Small("Target: 10+")
                ], className="metric-card")
            ], className="metrics-grid")
        ], className="metrics-section"),

        html.Div([
            html.H3("Community Metrics"),
            html.Div([
                html.Div([
                    html.H4("Discord Members"),
                    html.P(f"{data['community']['discord_members']}"),
                    html.Small("Target: 100+")
                ], className="metric-card"),
                html.Div([
                    html.H4("Active Users"),
                    html.P(f"{data['community']['active_users']}"),
                    html.Small("Target: 50+")
                ], className="metric-card"),
                html.Div([
                    html.H4("Paper Trading"),
                    html.P(f"{data['community']['paper_trading_users']}"),
                    html.Small("Target: 50+")
                ], className="metric-card"),
                html.Div([
                    html.H4("Live Trading"),
                    html.P(f"{data['community']['live_trading_users']}"),
                    html.Small("Target: 10+")
                ], className="metric-card")
            ], className="metrics-grid")
        ], className="metrics-section"),

        html.Div([
            html.H3("Performance Metrics"),
            html.Div([
                html.Div([
                    html.H4("CAGR"),
                    html.P(f"{data['performance']['cagr']}%"),
                    html.Small("Target: >20%")
                ], className="metric-card"),
                html.Div([
                    html.H4("Max Drawdown"),
                    html.P(f"{data['performance']['max_drawdown']}%"),
                    html.Small("Target: <10%")
                ], className="metric-card"),
                html.H4("Sharpe Ratio"),
                html.P(f"{data['performance']['sharpe_ratio']}"),
                html.Small("Target: >1.5")
            ], className="metrics-grid")
        ], className="metrics-section")
    ], className="main-content"),

    # Charts Section
    html.Section([
        html.H2("Growth Trends"),
        html.Div([
            dcc.Graph(
                id='stars-growth',
                figure={
                    'data': [
                        {'x': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                         'y': [1, 5, 15, 30, 60, 100],
                         'type': 'line', 'name': 'Stars',
                         'marker': {'color': '#007bff'}}
                    ],
                    'layout': {
                        'title': 'GitHub Stars Growth',
                        'xaxis': {'title': 'Month'},
                        'yaxis': {'title': 'Number of Stars'}
                    }
                }
            )
        ], className="chart-container")
    ], className="charts-section"),

    # Footer
    html.Footer([
        html.P("AI Trading System Analytics Dashboard - Tracking our journey to 100+ stars and active community")
    ], className="footer")
], className="dashboard")

# CSS Styling
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

# Run app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)