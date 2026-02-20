#!/usr/bin/env python3
"""
Strategy Comparison Dashboard Component
=======================================

Dashboard tab per confrontare Monte Carlo vs Mont Blanck.
Mostra grafici, metriche e performance in tempo reale.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Import strategy comparison
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.strategy.strategy_comparison import StrategyComparisonEngine, StrategyType

logger = logging.getLogger("StrategyComparisonDashboard")


# Layout del tab
def get_strategy_comparison_layout() -> html.Div:
    """Restituisce il layout per il tab di confronto strategie."""
    return html.Div([
        # Header
        html.Div([
            html.H2("⚔️ Strategy Comparison: Monte Carlo vs Mont Blanck", 
                   className="text-2xl font-bold mb-4"),
            html.P("Confronto in tempo reale delle performance delle due strategie",
                  className="text-gray-600 mb-6"),
        ], className="mb-6"),
        
        # Control Panel
        html.Div([
            # Controlli
            html.Div([
                html.Label("Simbolo:", className="font-semibold mr-2"),
                dcc.Dropdown(
                    id="comparison-symbol",
                    options=[
                        {"label": "BTC/USDT", "value": "BTCUSDT"},
                        {"label": "ETH/USDT", "value": "ETHUSDT"},
                        {"label": "SOL/USDT", "value": "SOLUSDT"},
                    ],
                    value="BTCUSDT",
                    clearable=False,
                    style={"width": "150px"}
                ),
            ], className="flex items-center mr-6"),
            
            html.Div([
                html.Label("Saldo Iniziale:", className="font-semibold mr-2"),
                dcc.Input(
                    id="initial-balance",
                    type="number",
                    value=10000,
                    min=1000,
                    max=100000,
                    step=1000,
                    style={"width": "120px"}
                ),
            ], className="flex items-center mr-6"),
            
            html.Div([
                html.Label("Trade Size %:", className="font-semibold mr-2"),
                dcc.Input(
                    id="trade-size-pct",
                    type="number",
                    value=10,
                    min=1,
                    max=50,
                    step=1,
                    style={"width": "80px"}
                ),
            ], className="flex items-center mr-6"),
            
            html.Button("▶️ Avvia Simulazione", id="start-comparison", 
                       className="bg-green-500 text-white px-4 py-2 rounded mr-2"),
            html.Button("⏹️ Stop", id="stop-comparison",
                       className="bg-red-500 text-white px-4 py-2 rounded mr-2"),
            html.Button("🔄 Reset", id="reset-comparison",
                       className="bg-gray-500 text-white px-4 py-2 rounded"),
        ], className="flex flex-wrap items-center mb-6 p-4 bg-gray-100 rounded-lg"),
        
        # Metric Cards
        html.Div([
            # Monte Carlo Card
            html.Div([
                html.H3("🎲 Monte Carlo", className="text-lg font-bold text-blue-600 mb-3"),
                html.Div([
                    html.Div([
                        html.Span("Balance:", className="text-gray-600"),
                        html.Span(id="mc-balance", className="font-bold ml-2"),
                    ], className="mb-2"),
                    html.Div([
                        html.Span("PnL:", className="text-gray-600"),
                        html.Span(id="mc-pnl", className="font-bold ml-2"),
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Trades:", className="text-gray-600"),
                        html.Span(id="mc-trades", className="font-bold ml-2"),
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Win Rate:", className="text-gray-600"),
                        html.Span(id="mc-winrate", className="font-bold ml-2"),
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Max DD:", className="text-gray-600"),
                        html.Span(id="mc-drawdown", className="font-bold ml-2"),
                    ], className="mb-2"),
                ]),
            ], className="bg-white p-4 rounded-lg shadow-md w-64"),
            
            # Mont Blanck Card
            html.Div([
                html.H3("🏔️ Mont Blanck", className="text-lg font-bold text-orange-600 mb-3"),
                html.Div([
                    html.Div([
                        html.Span("Balance:", className="text-gray-600"),
                        html.Span(id="mb-balance", className="font-bold ml-2"),
                    ], className="mb-2"),
                    html.Div([
                        html.Span("PnL:", className="text-gray-600"),
                        html.Span(id="mb-pnl", className="font-bold ml-2"),
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Trades:", className="text-gray-600"),
                        html.Span(id="mb-trades", className="font-bold ml-2"),
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Win Rate:", className="text-gray-600"),
                        html.Span(id="mb-winrate", className="font-bold ml-2"),
                    ], className="mb-2"),
                    html.Div([
                        html.Span("Max DD:", className="text-gray-600"),
                        html.Span(id="mb-drawdown", className="font-bold ml-2"),
                    ], className="mb-2"),
                ]),
            ], className="bg-white p-4 rounded-lg shadow-md w-64"),
            
            # Winner Card
            html.Div([
                html.H3("🏆 Winner", className="text-lg font-bold text-green-600 mb-3"),
                html.Div([
                    html.Div(id="winner-name", className="text-2xl font-bold mb-2"),
                    html.Div([
                        html.Span("Diff:", className="text-gray-600"),
                        html.Span(id="winner-diff", className="font-bold ml-2"),
                    ], className="mb-2"),
                ]),
            ], className="bg-white p-4 rounded-lg shadow-md w-64"),
            
        ], className="flex flex-wrap gap-4 mb-6"),
        
        # Grafici
        html.Div([
            # Grafico Balance Comparison
            html.Div([
                html.H4("📈 Balance Comparison", className="font-bold mb-2"),
                dcc.Graph(id="balance-comparison-graph", style={"height": "400px"}),
            ], className="bg-white p-4 rounded-lg shadow-md mb-4"),
            
            # Grafico PnL Comparison
            html.Div([
                html.H4("💰 PnL Comparison", className="font-bold mb-2"),
                dcc.Graph(id="pnl-comparison-graph", style={"height": "300px"}),
            ], className="bg-white p-4 rounded-lg shadow-md mb-4"),
            
            # Grafico Drawdown
            html.Div([
                html.H4("📉 Drawdown Comparison", className="font-bold mb-2"),
                dcc.Graph(id="drawdown-comparison-graph", style={"height": "300px"}),
            ], className="bg-white p-4 rounded-lg shadow-md"),
        ], className="grid grid-cols-1 gap-4 mb-6"),
        
        # Trade History
        html.Div([
            html.H4("📋 Trade History", className="font-bold mb-2"),
            html.Div(id="trade-history-table", className="overflow-x-auto"),
        ], className="bg-white p-4 rounded-lg shadow-md"),
        
        # Interval per aggiornamento
        dcc.Interval(
            id="comparison-interval",
            interval=2000,  # 2 secondi
            n_intervals=0,
            disabled=True
        ),
        
        # Store per i dati
        dcc.Store(id="comparison-engine-store"),
        dcc.Store(id="comparison-running", data=False),
        dcc.Store(id="price-data-store", data=[]),
        
    ], className="p-6")


# Callbacks
def register_strategy_comparison_callbacks(app):
    """Registra i callback per il confronto strategie."""
    
    @app.callback(
        [Output("comparison-interval", "disabled"),
         Output("comparison-engine-store", "data"),
         Output("comparison-running", "data")],
        [Input("start-comparison", "n_clicks"),
         Input("stop-comparison", "n_clicks"),
         Input("reset-comparison", "n_clicks")],
        [State("initial-balance", "value"),
         State("trade-size-pct", "value"),
         State("comparison-engine-store", "data"),
         State("comparison-running", "data")]
    )
    def control_comparison(start_clicks, stop_clicks, reset_clicks, 
                          initial_balance, trade_size_pct, engine_data, is_running):
        """Controlla avvio/stop/reset della simulazione."""
        ctx = dash.callback_context
        
        if not ctx.triggered:
            return True, engine_data, False
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "start-comparison":
            # Avvia simulazione
            engine = StrategyComparisonEngine(
                initial_balance=initial_balance or 10000,
                trade_size_pct=(trade_size_pct or 10) / 100
            )
            return False, {"initialized": True, "balance": initial_balance}, True
        
        elif button_id == "stop-comparison":
            return True, engine_data, False
        
        elif button_id == "reset-comparison":
            return True, None, False
        
        return True, engine_data, False
    
    @app.callback(
        [Output("mc-balance", "children"),
         Output("mc-pnl", "children"),
         Output("mc-trades", "children"),
         Output("mc-winrate", "children"),
         Output("mc-drawdown", "children"),
         Output("mb-balance", "children"),
         Output("mb-pnl", "children"),
         Output("mb-trades", "children"),
         Output("mb-winrate", "children"),
         Output("mb-drawdown", "children"),
         Output("winner-name", "children"),
         Output("winner-diff", "children"),
         Output("balance-comparison-graph", "figure"),
         Output("pnl-comparison-graph", "figure"),
         Output("drawdown-comparison-graph", "figure"),
         Output("price-data-store", "data")],
        [Input("comparison-interval", "n_intervals")],
        [State("comparison-engine-store", "data"),
         State("comparison-running", "data"),
         State("price-data-store", "data"),
         State("comparison-symbol", "value")]
    )
    def update_comparison(n_intervals, engine_data, is_running, price_data, symbol):
        """Aggiorna i dati del confronto."""
        if not is_running or not engine_data:
            # Dati vuoti
            empty_fig = go.Figure()
            return (
                "$10,000.00", "$0.00", "0", "0%", "0%",
                "$10,000.00", "$0.00", "0", "0%", "0%",
                "-", "0%", empty_fig, empty_fig, empty_fig, []
            )
        
        # Simula nuovo prezzo (in produzione, questo verrebbe da API reale)
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        if not price_data:
            price_data = [50000]  # Prezzo iniziale
        
        # Genera nuovo prezzo con random walk
        last_price = price_data[-1]
        change = np.random.normal(0, 0.001) * last_price
        new_price = last_price + change
        price_data.append(new_price)
        
        # Mantieni solo ultimi 500 prezzi
        if len(price_data) > 500:
            price_data = price_data[-500:]
        
        # Crea engine e aggiorna
        engine = StrategyComparisonEngine(initial_balance=engine_data.get("balance", 10000))
        
        # Riproduci storico
        for price in price_data[:-1]:
            engine.update(symbol, price)
        
        # Ultimo aggiornamento
        result = engine.update(symbol, price_data[-1])
        
        # Ottieni summary
        summary = engine.get_performance_summary()
        
        mc = summary["MonteCarlo"]
        mb = summary["MontBlanck"]
        
        # Formatta metriche
        def format_currency(val):
            return f"${val:,.2f}"
        
        def format_pct(val):
            return f"{val:.2f}%"
        
        # Crea grafici
        df = engine.get_comparison_dataframe()
        
        # Grafico Balance
        balance_fig = go.Figure()
        if len(df) > 0:
            balance_fig.add_trace(go.Scatter(
                y=df["mc_balance"],
                mode="lines",
                name="Monte Carlo",
                line=dict(color="blue", width=2)
            ))
            balance_fig.add_trace(go.Scatter(
                y=df["mb_balance"],
                mode="lines",
                name="Mont Blanck",
                line=dict(color="orange", width=2)
            ))
        balance_fig.update_layout(
            title="Balance Over Time",
            xaxis_title="Time",
            yaxis_title="Balance ($)",
            template="plotly_white",
            showlegend=True
        )
        
        # Grafico PnL
        pnl_fig = go.Figure()
        if len(df) > 0:
            pnl_fig.add_trace(go.Bar(
                name="Monte Carlo",
                x=["PnL"],
                y=[mc["total_pnl"]],
                marker_color="blue"
            ))
            pnl_fig.add_trace(go.Bar(
                name="Mont Blanck",
                x=["PnL"],
                y=[mb["total_pnl"]],
                marker_color="orange"
            ))
        pnl_fig.update_layout(
            title="Total PnL",
            yaxis_title="PnL ($)",
            template="plotly_white",
            barmode="group"
        )
        
        # Grafico Drawdown
        dd_fig = go.Figure()
        if len(df) > 0:
            dd_fig.add_trace(go.Scatter(
                y=df["mc_drawdown"] * 100,
                mode="lines",
                name="Monte Carlo",
                line=dict(color="blue", width=2),
                fill="tozeroy"
            ))
            dd_fig.add_trace(go.Scatter(
                y=df["mb_drawdown"] * 100,
                mode="lines",
                name="Mont Blanck",
                line=dict(color="orange", width=2),
                fill="tozeroy"
            ))
        dd_fig.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Time",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            showlegend=True
        )
        
        return (
            format_currency(mc["current_balance"]),
            format_currency(mc["total_pnl"]) + f" ({format_pct(mc['total_return_pct'])})",
            str(mc["total_trades"]),
            format_pct(mc["win_rate"] * 100),
            format_pct(mc["max_drawdown_pct"]),
            format_currency(mb["current_balance"]),
            format_currency(mb["total_pnl"]) + f" ({format_pct(mb['total_return_pct'])})",
            str(mb["total_trades"]),
            format_pct(mb["win_rate"] * 100),
            format_pct(mb["max_drawdown_pct"]),
            summary["winner"],
            format_pct(summary["return_difference"]),
            balance_fig,
            pnl_fig,
            dd_fig,
            price_data
        )


# Per test standalone
if __name__ == "__main__":
    import dash_bootstrap_components as dbc
    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = html.Div([
        get_strategy_comparison_layout()
    ])
    
    register_strategy_comparison_callbacks(app)
    
    print("Strategy Comparison Dashboard running on http://localhost:8051")
    app.run_server(debug=True, port=8051)
