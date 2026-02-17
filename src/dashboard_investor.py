"""
Dashboard Investitore
====================
Dashboard per gli investitori per visualizzare:
- Equity
- PnL
- Fee
- Posizioni
- Storico

Run con:
    python -m src.dashboard_investor
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import logging

from src.account_manager import AccountManager, EquityTracker, PerformanceFeeCalculator

logger = logging.getLogger(__name__)

# Inizializza
account_manager = AccountManager()
equity_tracker = EquityTracker()
fee_calculator = PerformanceFeeCalculator()

# Crea app Dash
app = dash.Dash(
    __name__,
    title="Investor Dashboard - AI Trading",
    update_title=None,
)

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("📊 Investor Dashboard", style={"margin": 0}),
        html.P("Il tuo conto trading in tempo reale", style={"margin": 0}),
    ], style={
        "background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
        "color": "white",
        "padding": "20px",
        "borderRadius": "10px",
        "marginBottom": "20px",
    }),
    
    # Selezione utente
    html.Div([
        html.Label("Seleziona il tuo account:", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="user-selector",
            options=[],
            value=None,
            placeholder="Scegli un account...",
            style={"marginBottom": "20px"},
        ),
    ]),
    
    # KPI Cards
    html.Div(id="kpi-cards", style={"display": "none"}, children=[
        html.Div([
            # Equity Card
            html.Div([
                html.H3("💰 Equity", style={"margin": 0, "fontSize": "14px", "color": "#888"}),
                html.H2(id="equity-value", style={"margin": "5px 0", "color": "#00d4aa"}),
            ], className="kpi-card"),
            
            # Daily PnL Card
            html.Div([
                html.H3("📈 Daily PnL", style={"margin": 0, "fontSize": "14px", "color": "#888"}),
                html.H2(id="daily-pnl", style={"margin": "5px 0"}),
            ], className="kpi-card"),
            
            # Total PnL Card
            html.Div([
                html.H3("📊 Total PnL", style={"margin": 0, "fontSize": "14px", "color": "#888"}),
                html.H2(id="total-pnl", style={"margin": "5px 0"}),
            ], className="kpi-card"),
            
            # Fee Card
            html.Div([
                html.H3("💸 Fee Mat Gate", style={"margin": 0, "fontSize": "14px", "color": "#888"}),
                html.H2(id="fee-value", style={"margin": "5px 0", "color": "#ff6b6b"}),
            ], className="kpi-card"),
            
        ], className="kpi-row"),
    ]),
    
    # Grafici
    html.Div(id="charts", style={"display": "none"}, children=[
        # Equity Curve
        html.Div([
            dcc.Graph(id="equity-chart"),
        ], className="chart-container"),
        
        # Positions Table
        html.Div([
            html.H3("📋 Posizioni Aperte", style={"marginBottom": "10px"}),
            html.Div(id="positions-table"),
        ], className="chart-container", style={"marginTop": "20px"}),
        
        # Fee Breakdown
        html.Div([
            dcc.Graph(id="fee-chart"),
        ], className="chart-container"),
    ]),
    
    # Intervallo aggiornamento
    dcc.Interval(
        id="refresh-interval",
        interval=60 * 1000,  # 1 minute
        n_intervals=0,
    ),
    
], style={"padding": "20px", "background": "#0f0f1a", "minHeight": "100vh"})


# CSS styles
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
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #0f0f1a;
                color: white;
                margin: 0;
                padding: 0;
            }
            .kpi-row {
                display: flex;
                gap: 15px;
                marginBottom: 20px;
            }
            .kpi-card {
                flex: 1;
                background: #1a1a2e;
                borderRadius: 10px;
                padding: 20px;
                boxShadow: 0 4px 6px rgba(0,0,0,0.3);
            }
            .chart-container {
                background: #1a1a2e;
                borderRadius: 10px;
                padding: 20px;
                marginBottom: 20px;
                boxShadow: 0 4px 6px rgba(0,0,0,0.3);
            }
            .positive { color: #00d4aa !important; }
            .negative { color: #ff6b6b !important; }
            table {
                width: 100%;
                borderCollapse: collapse;
            }
            th, td {
                padding: 12px;
                textAlign: left;
                borderBottom: 1px solid #333;
            }
            th {
                background: #16213e;
                color: #888;
            }
            .dash-dropdown {
                background: #1a1a2e !important;
                color: white !important;
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


@callback(
    Output("user-selector", "options"),
    Output("user-selector", "value"),
    Input("refresh-interval", "n_intervals"),
)
def load_users(n):
    users = account_manager.list_users()
    options = [{"label": u.username, "value": u.user_id} for u in users]
    value = options[0]["value"] if options else None
    return options, value


@callback(
    Output("kpi-cards", "style"),
    Output("equity-value", "children"),
    Output("equity-value", "className"),
    Output("daily-pnl", "children"),
    Output("daily-pnl", "className"),
    Output("total-pnl", "children"),
    Output("total-pnl", "className"),
    Output("fee-value", "children"),
    Input("user-selector", "value"),
    Input("refresh-interval", "n_intervals"),
)
def update_kpis(user_id, n):
    if not user_id:
        return {"display": "none"}, "$0", "", "$0", "", "$0", "", "$0"
    
    snapshots = equity_tracker.get_snapshots(user_id, days=30)
    
    if not snapshots:
        return {"display": "none"}, "$0", "", "$0", "", "$0", "", "$0"
    
    latest = snapshots[-1]
    first = snapshots[0]
    
    # Equity
    equity = latest["equity"]
    equity_str = f"${equity:,.2f}"
    
    # Daily PnL
    daily_pnl = latest["daily_pnl"]
    daily_pnl_str = f"{'+' if daily_pnl >= 0 else ''}${daily_pnl:,.2f} ({latest['daily_pnl_pct']:.2f}%)"
    daily_class = "positive" if daily_pnl >= 0 else "negative"
    
    # Total PnL
    total_pnl = equity - first["equity"]
    total_pnl_str = f"{'+' if total_pnl >= 0 else ''}${total_pnl:,.2f} ({((total_pnl/first['equity'])*100):.2f}%)"
    total_class = "positive" if total_pnl >= 0 else "negative"
    
    # Fees (approximate)
    user = account_manager.get_user(user_id)
    fees = fee_calculator.calculate_fees(
        current_equity=equity,
        high_water_mark=first["equity"],
        days_elapsed=30,
    )
    fee_str = f"${fees['total_fee']:.2f}"
    
    return {"display": "block"}, equity_str, "", daily_pnl_str, daily_class, total_pnl_str, total_class, fee_str


@callback(
    Output("charts", "style"),
    Output("equity-chart", "figure"),
    Output("fee-chart", "figure"),
    Input("user-selector", "value"),
    Input("refresh-interval", "n_intervals"),
)
def update_charts(user_id, n):
    if not user_id:
        return {"display": "none"}, go.Figure(), go.Figure()
    
    snapshots = equity_tracker.get_snapshots(user_id, days=90)
    
    if not snapshots:
        return {"display": "none"}, go.Figure(), go.Figure()
    
    # Prepare data
    df = pd.DataFrame(snapshots)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Equity chart
    equity_fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    equity_fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["equity"],
        mode="lines",
        name="Equity",
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 170, 0.1)",
    ))
    
    equity_fig.update_layout(
        title="📈 Equity Curve",
        paper_bgcolor="transparent",
        plot_bgcolor="transparent",
        font=dict(color="white"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    # Fee chart
    df["fee"] = df["equity"].diff().clip(lower=0) * 0.20  # Approx 20% of profits
    
    fee_fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    fee_fig.add_trace(go.Bar(
        x=df["timestamp"],
        y=df["fee"],
        name="Fee",
        marker_color="#ff6b6b",
    ))
    
    fee_fig.update_layout(
        title="💸 Fee Mat Gate (Stimate)",
        paper_bgcolor="transparent",
        plot_bgcolor="transparent",
        font=dict(color="white"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
    )
    
    return {"display": "block"}, equity_fig, fee_fig


@callback(
    Output("positions-table", "children"),
    Input("user-selector", "value"),
    Input("refresh-interval", "n_intervals"),
)
def update_positions(user_id, n):
    if not user_id:
        return "N/A"
    
    user = account_manager.get_user(user_id)
    if not user:
        return "Utente non trovato"
    
    try:
        from src.execution import ExchangeClient
        client = ExchangeClient(user.api_key, user.api_secret, testnet=user.testnet)
        positions = client.get_positions()
        
        if not positions:
            return "Nessuna posizione aperta"
        
        rows = []
        for pos in positions:
            rows.append(html.Tr([
                html.Td(pos["symbol"]),
                html.Td(f"{pos['size']:.4f}"),
                html.Td(f"${pos['entry_price']:.2f}"),
                html.Td(f"${pos.get('unrealized_pnl', 0):.2f}", 
                       style={"color": "#00d4aa" if pos.get('unrealized_pnl', 0) > 0 else "#ff6b6b"}),
            ]))
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Symbol"),
                html.Th("Size"),
                html.Th("Entry Price"),
                html.Th("Unrealized PnL"),
            ])),
            html.Tbody(rows),
        ])
        
    except Exception as e:
        return f"Errore: {str(e)}"


if __name__ == "__main__":
    print("🚀 Avvio Investor Dashboard...")
    print("📊 Apri http://127.0.0.1:8051 nel browser")
    app.run(debug=False, port=8051)
