"""
Enhanced Investor Dashboard - AI Trading System
Dashboard avanzata per investitori del fondo
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
import json

# Import fund modules
import sys
sys.path.insert(0, '.')
from src.fund.fund_manager import FundManager, InvestorStatus
from src.fund.performance import PerformanceAnalyzer


class InvestorDashboard:
    """
    Dashboard per gli investitori del fondo
    """
    
    def __init__(self, fund_manager: FundManager):
        self.fund = fund_manager
        self.app = dash.Dash(
            __name__,
            title="Investor Portal",
            updates_title="AI Trading Fund - Investor Portal"
        )
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Configura il layout della dashboard"""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Investor Portal", className="header-title"),
                html.Div([
                    html.Span(id="fund-name-display", className="fund-name"),
                    html.Span(id="nav-display", className="nav-value"),
                ], className="header-info")
            ], className="dashboard-header"),
            
            # Main content
            html.Div([
                # Left sidebar - Investor info
                html.Div([
                    html.Div([
                        html.H3("My Portfolio"),
                        html.Div(id="investor-summary", className="info-box")
                    ], className="sidebar-section"),
                    
                    html.Div([
                        html.H3("Quick Actions"),
                        html.Button("Request Subscription", id="btn-subscribe", className="action-btn"),
                        html.Button("Request Redemption", id="btn-redeem", className="action-btn secondary"),
                        html.Button("Download Report", id="btn-report", className="action-btn outline"),
                    ], className="sidebar-section"),
                    
                    html.Div([
                        html.H3("Notifications"),
                        html.Div(id="notifications-list", className="notifications")
                    ], className="sidebar-section")
                ], className="sidebar"),
                
                # Main area
                html.Div([
                    # Performance metrics row
                    html.Div([
                        html.Div([
                            html.Div("Total Value", className="metric-label"),
                            html.Div(id="total-value", className="metric-value")
                        ], className="metric-card"),
                        html.Div([
                            html.Div("Total Return", className="metric-label"),
                            html.Div(id="total-return", className="metric-value")
                        ], className="metric-card"),
                        html.Div([
                            html.Div("YTD Return", className="metric-label"),
                            html.Div(id="ytd-return", className="metric-value")
                        ], className="metric-card"),
                        html.Div([
                            html.Div("Shares Owned", className="metric-label"),
                            html.Div(id="shares-owned", className="metric-value")
                        ], className="metric-card"),
                    ], className="metrics-row"),
                    
                    # Charts row
                    html.Div([
                        html.Div([
                            dcc.Graph(id="nav-chart")
                        ], className="chart-container"),
                        
                        html.Div([
                            dcc.Graph(id="performance-chart")
                        ], className="chart-container"),
                    ], className="charts-row"),
                    
                    # Holdings and transactions
                    html.Div([
                        html.Div([
                            html.H3("Portfolio Allocation"),
                            dcc.Graph(id="allocation-chart")
                        ], className="half-width"),
                        
                        html.Div([
                            html.H3("Recent Transactions"),
                            html.Div(id="transactions-table", className="table-container")
                        ], className="half-width"),
                    ], className="bottom-row")
                ], className="main-content")
            ], className="dashboard-content"),
            
            # Hidden stores
            dcc.Store(id="selected-investor-store"),
            
            # Modal for subscription
            html.Div([
                html.Div([
                    html.H2("Request Subscription"),
                    html.Label("Amount ($)"),
                    dcc.Input(id="sub-amount", type="number", placeholder="Enter amount"),
                    html.Button("Submit", id="btn-submit-sub", className="modal-btn"),
                    html.Button("Cancel", id="btn-cancel-sub", className="modal-btn secondary")
                ], className="modal-content")
            ], id="subscription-modal", className="modal", style={"display": "none"}),
            
            # Modal for redemption
            html.Div([
                html.Div([
                    html.H2("Request Redemption"),
                    html.Label("Shares"),
                    dcc.Input(id="redeem-shares", type="number", placeholder="Enter shares"),
                    html.Button("Submit", id="btn-submit-redeem", className="modal-btn"),
                    html.Button("Cancel", id="btn-cancel-redeem", className="modal-btn secondary")
                ], className="modal-content")
            ], id="redemption-modal", className="modal", style={"display": "none"})
            
        ], className="dashboard-container")
        
        # Add CSS
        self._add_css()
    
    def _add_css(self):
        """Aggiunge gli stili CSS"""
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: #0a0e17;
                        color: #e0e0e0;
                    }
                    
                    .dashboard-container {
                        min-height: 100vh;
                    }
                    
                    .dashboard-header {
                        background: linear-gradient(135deg, #1a1f35 0%, #0d1421 100%);
                        padding: 20px 40px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        border-bottom: 2px solid #2d3748;
                    }
                    
                    .header-title {
                        color: #4fd1c5;
                        font-size: 28px;
                        font-weight: 600;
                    }
                    
                    .header-info {
                        display: flex;
                        gap: 30px;
                    }
                    
                    .fund-name {
                        color: #a0aec0;
                        font-size: 16px;
                    }
                    
                    .nav-value {
                        color: #48bb78;
                        font-size: 18px;
                        font-weight: 600;
                    }
                    
                    .dashboard-content {
                        display: flex;
                        padding: 20px;
                        gap: 20px;
                    }
                    
                    .sidebar {
                        width: 280px;
                        flex-shrink: 0;
                    }
                    
                    .sidebar-section {
                        background: #1a202c;
                        border-radius: 12px;
                        padding: 20px;
                        margin-bottom: 20px;
                    }
                    
                    .sidebar-section h3 {
                        color: #4fd1c5;
                        font-size: 16px;
                        margin-bottom: 15px;
                    }
                    
                    .info-box {
                        background: #2d3748;
                        border-radius: 8px;
                        padding: 15px;
                    }
                    
                    .action-btn {
                        display: block;
                        width: 100%;
                        padding: 12px;
                        margin-bottom: 10px;
                        background: #4fd1c5;
                        color: #0a0e17;
                        border: none;
                        border-radius: 8px;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.2s;
                    }
                    
                    .action-btn:hover {
                        background: #38b2ac;
                        transform: translateY(-2px);
                    }
                    
                    .action-btn.secondary {
                        background: #ed8936;
                    }
                    
                    .action-btn.secondary:hover {
                        background: #dd6b20;
                    }
                    
                    .action-btn.outline {
                        background: transparent;
                        border: 2px solid #4fd1c5;
                        color: #4fd1c5;
                    }
                    
                    .main-content {
                        flex: 1;
                    }
                    
                    .metrics-row {
                        display: grid;
                        grid-template-columns: repeat(4, 1fr);
                        gap: 15px;
                        margin-bottom: 20px;
                    }
                    
                    .metric-card {
                        background: #1a202c;
                        border-radius: 12px;
                        padding: 20px;
                        border: 1px solid #2d3748;
                    }
                    
                    .metric-label {
                        color: #a0aec0;
                        font-size: 14px;
                        margin-bottom: 8px;
                    }
                    
                    .metric-value {
                        color: #fff;
                        font-size: 24px;
                        font-weight: 700;
                    }
                    
                    .charts-row {
                        display: grid;
                        grid-template-columns: repeat(2, 1fr);
                        gap: 15px;
                        margin-bottom: 20px;
                    }
                    
                    .chart-container {
                        background: #1a202c;
                        border-radius: 12px;
                        padding: 15px;
                        border: 1px solid #2d3748;
                    }
                    
                    .bottom-row {
                        display: grid;
                        grid-template-columns: repeat(2, 1fr);
                        gap: 15px;
                    }
                    
                    .half-width {
                        background: #1a202c;
                        border-radius: 12px;
                        padding: 15px;
                        border: 1px solid #2d3748;
                    }
                    
                    .half-width h3 {
                        color: #4fd1c5;
                        margin-bottom: 15px;
                    }
                    
                    .table-container {
                        max-height: 300px;
                        overflow-y: auto;
                    }
                    
                    .modal {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(0,0,0,0.7);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        z-index: 1000;
                    }
                    
                    .modal-content {
                        background: #1a202c;
                        padding: 30px;
                        border-radius: 12px;
                        width: 400px;
                    }
                    
                    .modal-content h2 {
                        color: #4fd1c5;
                        margin-bottom: 20px;
                    }
                    
                    .modal-content label {
                        display: block;
                        color: #a0aec0;
                        margin-bottom: 8px;
                    }
                    
                    .modal-content input {
                        width: 100%;
                        padding: 12px;
                        background: #2d3748;
                        border: 1px solid #4a5568;
                        border-radius: 8px;
                        color: #fff;
                        margin-bottom: 20px;
                    }
                    
                    .modal-btn {
                        padding: 12px 24px;
                        background: #4fd1c5;
                        color: #0a0e17;
                        border: none;
                        border-radius: 8px;
                        font-weight: 600;
                        cursor: pointer;
                    }
                    
                    .modal-btn.secondary {
                        background: #718096;
                        margin-left: 10px;
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
    
    def _setup_callbacks(self):
        """Configura i callback"""
        
        @self.app.callback(
            Output("fund-name-display", "children"),
            Output("nav-display", "children"),
            Input("selected-investor-store", "data")
        )
        def update_header(investor_id):
            fund_name = self.fund.name
            nav = f"NAV: ${float(self.fund.current_nav):.2f}"
            return fund_name, nav
        
        @self.app.callback(
            Output("total-value", "children"),
            Output("total-return", "children"),
            Output("ytd-return", "children"),
            Output("shares-owned", "children"),
            Input("selected-investor-store", "data")
        )
        def update_metrics(investor_id):
            if not investor_id or investor_id not in self.fund.investors:
                return "$0.00", "0.00%", "0.00%", "0"
            
            investor = self.fund.investors[investor_id]
            shares = investor.current_value / self.fund.current_nav if self.fund.current_nav > 0 else Decimal("0")
            
            return (
                f"${float(investor.current_value):,.2f}",
                f"{investor.return_percentage:.2f}%",
                f"{self.fund.cumulative_return:.2f}%",
                f"{float(shares):,.2f}"
            )
        
        @self.app.callback(
            Output("nav-chart", "figure"),
            Input("selected-investor-store", "data")
        )
        def update_nav_chart(investor_id):
            if not self.fund.nav_history:
                return go.Figure()
            
            dates = [nav.date.strftime("%Y-%m-%d") for nav in self.fund.nav_history]
            navs = [float(nav.nav_per_share) for nav in self.fund.nav_history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=navs,
                mode="lines",
                fill="tozeroy",
                line=dict(color="#4fd1c5", width=2),
                fillcolor="rgba(79, 209, 197, 0.1)"
            ))
            
            fig.update_layout(
                title="NAV History",
                paper_bgcolor="transparent",
                plot_bgcolor="transparent",
                font=dict(color="#a0aec0"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#2d3748"),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
        
        @self.app.callback(
            Output("performance-chart", "figure"),
            Input("selected-investor-store", "data")
        )
        def update_performance_chart(investor_id):
            if len(self.fund.nav_history) < 2:
                return go.Figure()
            
            returns = []
            dates = []
            cumulative = 0
            
            for i, nav in enumerate(self.fund.nav_history[1:], 1):
                if i > 0:
                    prev = self.fund.nav_history[i-1]
                    daily_return = (float(nav.nav_per_share) - float(prev.nav_per_share)) / float(prev.nav_per_share) * 100
                    cumulative += daily_return
                else:
                    daily_return = 0
                    cumulative = 0
                
                returns.append(cumulative)
                dates.append(nav.date.strftime("%Y-%m-%d"))
            
            colors = ["#48bb78" if r >= 0 else "#f56565" for r in returns]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dates,
                y=returns,
                marker_color=colors
            ))
            
            fig.update_layout(
                title="Cumulative Returns",
                paper_bgcolor="transparent",
                plot_bgcolor="transparent",
                font=dict(color="#a0aec0"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#2d3748"),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
        
        @self.app.callback(
            Output("allocation-chart", "figure"),
            Input("selected-investor-store", "data")
        )
        def update_allocation_chart(investor_id):
            # Simplified allocation - in real app would come from portfolio
            labels = ["Crypto", "Stablecoins", "Cash", "Options"]
            values = [60, 25, 10, 5]
            colors = ["#4fd1c5", "#48bb78", "#ed8936", "#9f7aea"]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hole=0.4
            )])
            
            fig.update_layout(
                title="Portfolio Allocation",
                paper_bgcolor="transparent",
                plot_bgcolor="transparent",
                font=dict(color="#a0aec0"),
                showlegend=False,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
        
        @self.app.callback(
            Output("transactions-table", "children"),
            Input("selected-investor-store", "data")
        )
        def update_transactions(investor_id):
            if not investor_id or investor_id not in self.fund.investors:
                return html.P("No transactions")
            
            # Combine subscriptions and redemptions
            investor_subs = [s for s in self.fund.subscriptions if s.investor_id == investor_id]
            investor_reds = [r for r in self.fund.redemptions if r.investor_id == investor_id]
            
            transactions = []
            
            for sub in investor_subs:
                transactions.append({
                    "date": sub.created_at.strftime("%Y-%m-%d"),
                    "type": "Subscription",
                    "amount": f"${float(sub.amount):,.2f}",
                    "status": sub.status
                })
            
            for red in investor_reds:
                transactions.append({
                    "date": red.created_at.strftime("%Y-%m-%d"),
                    "type": "Redemption",
                    "amount": f"${float(red.amount):,.2f}",
                    "status": red.status
                })
            
            if not transactions:
                return html.P("No transactions")
            
            # Create table
            return html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Date"),
                        html.Th("Type"),
                        html.Th("Amount"),
                        html.Th("Status")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(t["date"]),
                        html.Td(t["type"]),
                        html.Td(t["amount"]),
                        html.Td(t["status"])
                    ]) for t in sorted(transactions, key=lambda x: x["date"], reverse=True)[:10]
                ])
            ], className="transactions-table")
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """Avvia il server"""
        self.app.run_server(debug=debug, port=port)


def create_sample_fund() -> FundManager:
    """Crea un fondo di esempio"""
    fund = FundManager("AI Trading Fund", Decimal("10000000"))
    fund.open_fund()
    fund.fee_structure.management_fee_annual = 0.02
    fund.fee_structure.performance_fee_annual = 0.20
    
    # Simula NAV history
    import random
    base_nav = 100.0
    for i in range(30):
        change = random.uniform(-0.02, 0.03)
        base_nav *= (1 + change)
        fund.update_nav(
            portfolio_value=Decimal(str(base_nav * 100000)),
            cash=Decimal("500000")
        )
    
    return fund


if __name__ == "__main__":
    fund = create_sample_fund()
    dashboard = InvestorDashboard(fund)
    print("Starting Investor Dashboard on http://localhost:8050")
    dashboard.run_server()
