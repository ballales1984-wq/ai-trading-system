"""
Dashboard Memory Monitor
======================
Real-time RAM and ROM monitoring dashboard with colored alerts and bar charts.
Integrates with Dash for live visualization.

Usage:
    python dashboard_memory_monitor.py
    
    # Or integrate with existing dashboard:
    from dashboard_memory_monitor import create_memory_layout, register_memory_callbacks
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Try to import required packages
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Output, Input, State
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Warning: Dash not available")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available")

try:
    import shutil
    SHUTIL_AVAILABLE = True
except ImportError:
    SHUTIL_AVAILABLE = False


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# RAM limits in GB
RAM_LIMITS = {
    "Trading Engine": 1.5,
    "Database (PostgreSQL)": 1.0,
    "Cache (Redis)": 0.5,
    "Dashboard": 0.5,
    "Prometheus": 0.25,
    "Nginx": 0.125,
    "Total System": 4.0
}

# ROM limits in GB
ROM_LIMITS = {
    "Database Data": 1.0,
    "Redis Data": 0.5,
    "ML Temp": 0.5,
    "Logs": 0.2,
    "Prometheus": 0.3,
    "Dashboard": 0.1,
    "Total": 3.0
}

# Volume paths
VOLUME_PATHS = {
    "Database Data": "/var/lib/postgresql/data",
    "Redis Data": "/data",
    "ML Temp": "/app/ml_temp",
    "Logs": "/app/logs",
    "Prometheus": "/prometheus",
    "Dashboard": "/app/data"
}

# Colors for status
COLOR_OK = "#28a745"      # Green
COLOR_WARNING = "#ffc107"  # Yellow
COLOR_CRITICAL = "#dc3545" # Red
COLOR_INFO = "#17a2b8"     # Blue


# =============================================================================
# Data Collection
# =============================================================================

def get_system_ram() -> Dict[str, float]:
    """Get RAM usage for all components."""
    if not PSUTIL_AVAILABLE:
        return {k: 0.0 for k in RAM_LIMITS}
    
    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        used_gb = mem.used / (1024 ** 3)
        
        # Estimate per-component usage
        # In Docker, we use the limits as estimates
        result = {}
        remaining = used_gb
        
        components = list(RAM_LIMITS.keys())[:-1]  # Exclude Total
        for i, comp in enumerate(components):
            limit = RAM_LIMITS[comp]
            # Distribute proportionally
            if i < len(components) - 1:
                estimated = min(limit * 0.7, remaining * 0.3)
            else:
                estimated = remaining
            result[comp] = max(0.1, estimated)
            remaining -= estimated
        
        result["Total System"] = used_gb
        return result
        
    except Exception as e:
        logger.error(f"Error getting RAM: {e}")
        return {k: 0.0 for k in RAM_LIMITS}


def get_system_rom() -> Dict[str, float]:
    """Get ROM usage for all volumes."""
    if not SHUTIL_AVAILABLE:
        return {k: 0.0 for k in ROM_LIMITS}
    
    result = {}
    total = 0.0
    
    for name, path in VOLUME_PATHS.items():
        try:
            # Try different path variations
            paths_to_try = [
                path,
                os.path.join(os.getcwd(), path.lstrip("/")),
                f".{path}",
                f"/app/{path.lstrip('/')}"
            ]
            
            actual_path = None
            for p in paths_to_try:
                if os.path.exists(p):
                    actual_path = p
                    break
            
            if actual_path:
                usage = shutil.disk_usage(actual_path)
                used_gb = usage.used / (1024 ** 3)
            else:
                used_gb = 0.0
                
        except Exception as e:
            logger.debug(f"Could not get ROM for {name}: {e}")
            used_gb = 0.0
        
        result[name] = used_gb
        total += used_gb
    
    result["Total"] = total
    return result


def get_status_color(current: float, limit: float) -> str:
    """Get color based on usage percentage."""
    if limit <= 0:
        return COLOR_INFO
    
    percent = (current / limit) * 100
    
    if percent >= 90:
        return COLOR_CRITICAL
    elif percent >= 70:
        return COLOR_WARNING
    else:
        return COLOR_OK


# =============================================================================
# Dash Components
# =============================================================================

if DASH_AVAILABLE:
    # Create Dash app
    app = dash.Dash(__name__)
    app.title = "🖥️ Resource Monitor - AI Trading System"
    
    # Custom CSS
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
                    background-color: #1a1a2e;
                    color: #eee;
                    margin: 0;
                    padding: 20px;
                }
                .card {
                    background-color: #16213e;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                }
                .metric-value {
                    font-size: 2em;
                    font-weight: bold;
                }
                .metric-limit {
                    font-size: 0.9em;
                    color: #888;
                }
                .status-ok { color: #28a745; }
                .status-warning { color: #ffc107; }
                .status-critical { color: #dc3545; }
                h1, h2, h3 { color: #fff; }
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
    
    # Layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("🖥️ Resource Monitor"),
            html.P("AI Trading System - Real-time RAM & ROM Monitoring"),
        ], className="card", style={"textAlign": "center"}),
        
        # Summary cards
        html.Div([
            html.Div([
                html.H3("💾 RAM Usage"),
                html.Div(id="ram-summary"),
            ], className="card"),
            
            html.Div([
                html.H3("💿 ROM Usage"),
                html.Div(id="rom-summary"),
            ], className="card"),
        ], style={"display": "flex", "gap": "20px"}),
        
        # Bar charts
        html.Div([
            html.Div([
                html.H2("📊 RAM Distribution"),
                dcc.Graph(id="ram-chart"),
            ], className="card"),
            
            html.Div([
                html.H2("📊 ROM Distribution"),
                dcc.Graph(id="rom-chart"),
            ], className="card"),
        ]),
        
        # Detailed tables
        html.Div([
            html.Div([
                html.H2("📋 RAM Details"),
                html.Table(id="ram-table"),
            ], className="card"),
            
            html.Div([
                html.H2("📋 ROM Details"),
                html.Table(id="rom-table"),
            ], className="card"),
        ]),
        
        # Auto-refresh
        dcc.Interval(
            id="interval-component",
            interval=5000,  # 5 seconds
            n_intervals=0
        ),
        
        # Hidden store for data
        dcc.Store(id="resource-data"),
        
    ], style={"maxWidth": "1400px", "margin": "0 auto"})
    
    
    # Callback to update all components
    @app.callback(
        [Output("ram-summary", "children"),
         Output("rom-summary", "children"),
         Output("ram-chart", "figure"),
         Output("rom-chart", "figure"),
         Output("ram-table", "children"),
         Output("rom-table", "children"),
         Output("resource-data", "data")],
        [Input("interval-component", "n_intervals")]
    )
    def update_dashboard(n):
        # Get current data
        ram_data = get_system_ram()
        rom_data = get_system_rom()
        
        # Calculate totals
        total_ram = ram_data.get("Total System", 0)
        total_rom = rom_data.get("Total", 0)
        limit_ram = RAM_LIMITS["Total System"]
        limit_rom = ROM_LIMITS["Total"]
        
        # Summary cards
        ram_summary = html.Div([
            html.Div(f"{total_ram:.2f} GB", className="metric-value", 
                    style={"color": get_status_color(total_ram, limit_ram)}),
            html.Div(f"of {limit_ram} GB limit", className="metric-limit"),
            html.Div(f"{total_ram/limit_ram*100:.1f}%", className="metric-limit"),
        ])
        
        rom_summary = html.Div([
            html.Div(f"{total_rom:.2f} GB", className="metric-value",
                    style={"color": get_status_color(total_rom, limit_rom)}),
            html.Div(f"of {limit_rom} GB limit", className="metric-limit"),
            html.Div(f"{total_rom/limit_rom*100:.1f}%", className="metric-limit"),
        ])
        
        # RAM bar chart
        ram_fig = go.Figure()
        components = [k for k in RAM_LIMITS.keys() if k != "Total System"]
        values = [ram_data.get(c, 0) for c in components]
        limits = [RAM_LIMITS[c] for c in components]
        colors = [get_status_color(v, l) for v, l in zip(values, limits)]
        
        ram_fig.add_trace(go.Bar(
            x=components,
            y=values,
            marker_color=colors,
            name="Used"
        ))
        
        ram_fig.add_trace(go.Bar(
            x=components,
            y=[l - v for l, v in zip(limits, values)],
            marker_color="rgba(128,128,128,0.3)",
            name="Available"
        ))
        
        ram_fig.update_layout(
            barmode="stack",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#eee"},
            height=300,
            margin={"l": 40, "r": 40, "t": 20, "b": 40}
        )
        
        # ROM bar chart
        rom_fig = go.Figure()
        rom_components = [k for k in ROM_LIMITS.keys() if k != "Total"]
        rom_values = [rom_data.get(c, 0) for c in rom_components]
        rom_limits = [ROM_LIMITS[c] for c in rom_components]
        rom_colors = [get_status_color(v, l) for v, l in zip(rom_values, rom_limits)]
        
        rom_fig.add_trace(go.Bar(
            x=rom_components,
            y=rom_values,
            marker_color=rom_colors,
            name="Used"
        ))
        
        rom_fig.add_trace(go.Bar(
            x=rom_components,
            y=[l - v for l, v in zip(rom_limits, rom_values)],
            marker_color="rgba(128,128,128,0.3)",
            name="Available"
        ))
        
        rom_fig.update_layout(
            barmode="stack",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#eee"},
            height=300,
            margin={"l": 40, "r": 40, "t": 20, "b": 40}
        )
        
        # RAM table
        ram_table = []
        for comp in components:
            used = ram_data.get(comp, 0)
            limit = RAM_LIMITS[comp]
            percent = (used / limit * 100) if limit > 0 else 0
            color = get_status_color(used, limit)
            
            ram_table.append(html.Tr([
                html.Td(comp),
                html.Td(f"{used:.2f} GB"),
                html.Td(f"{limit} GB"),
                html.Td(f"{percent:.1f}%", style={"color": color, "fontWeight": "bold"})
            ]))
        
        # ROM table
        rom_table = []
        for comp in rom_components:
            used = rom_data.get(comp, 0)
            limit = ROM_LIMITS[comp]
            percent = (used / limit * 100) if limit > 0 else 0
            color = get_status_color(used, limit)
            
            rom_table.append(html.Tr([
                html.Td(comp),
                html.Td(f"{used:.2f} GB"),
                html.Td(f"{limit} GB"),
                html.Td(f"{percent:.1f}%", style={"color": color, "fontWeight": "bold"})
            ]))
        
        # Store data for other components
        data = {
            "ram": ram_data,
            "rom": rom_data,
            "timestamp": datetime.now().isoformat()
        }
        
        return (ram_summary, rom_summary, ram_fig, rom_fig, 
                ram_table, rom_table, data)


# =============================================================================
# Standalone Run
# =============================================================================

def run_dashboard(host: str = "0.0.0.0", port: int = 8051):
    """Run the dashboard server."""
    if not DASH_AVAILABLE:
        print("Error: Dash is not installed. Install with: pip install dash plotly")
        return
    
    print(f"\n{'='*60}")
    print("🖥️  Resource Monitor Dashboard")
    print(f"{'='*60}")
    print(f"Starting server at http://{host}:{port}")
    print("Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
    app.run_server(host=host, port=port, debug=False)


# =============================================================================
# Integration Functions
# =============================================================================

def create_memory_tab():
    """Create a memory monitoring tab for integration."""
    if not DASH_AVAILABLE:
        return html.Div("Dash not available")
    
    return html.Div([
        html.H2("🖥️ Resource Monitor"),
        
        # Summary
        html.Div([
            html.Div(id="mem-ram-summary"),
            html.Div(id="mem-rom-summary"),
        ], style={"display": "flex", "gap": "20px"}),
        
        # Charts
        dcc.Graph(id="mem-ram-chart"),
        dcc.Graph(id="mem-rom-chart"),
        
        # Refresh
        dcc.Interval(id="mem-interval", interval=5000, n_intervals=0),
    ])


def register_memory_callbacks(app):
    """Register callbacks for memory tab."""
    if not DASH_AVAILABLE:
        return
    
    @app.callback(
        [Output("mem-ram-summary", "children"),
         Output("mem-rom-summary", "children"),
         Output("mem-ram-chart", "figure"),
         Output("mem-rom-chart", "figure")],
        [Input("mem-interval", "n_intervals")]
    )
    def update_memory_tab(n):
        # Reuse the same logic
        ram_data = get_system_ram()
        rom_data = get_system_rom()
        
        # Summary
        total_ram = ram_data.get("Total System", 0)
        total_rom = rom_data.get("Total", 0)
        
        ram_summary = html.Div([
            html.H4("RAM"),
            html.Div(f"{total_ram:.2f} GB", style={"fontSize": "2em"})
        ])
        
        rom_summary = html.Div([
            html.H4("ROM"),
            html.Div(f"{total_rom:.2f} GB", style={"fontSize": "2em"})
        ])
        
        # Charts (simplified)
        ram_fig = go.Figure(go.Bar(
            x=list(RAM_LIMITS.keys())[:-1],
            y=[ram_data.get(c, 0) for c in list(RAM_LIMITS.keys())[:-1]]
        ))
        
        rom_fig = go.Figure(go.Bar(
            x=list(ROM_LIMITS.keys())[:-1],
            y=[rom_data.get(c, 0) for c in list(ROM_LIMITS.keys())[:-1]]
        ))
        
        return ram_summary, rom_summary, ram_fig, rom_fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_dashboard()

