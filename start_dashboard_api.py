#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick start script for API-integrated dashboard
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard_api import app

if __name__ == '__main__':
    print("="*70)
    print("HEDGE FUND TRADING DASHBOARD - API INTEGRATED")
    print("="*70)
    print("\nStarting dashboard...")
    print("Dashboard URL: http://localhost:8050")
    print("API URL: http://localhost:8000")
    print("\nMake sure the FastAPI server is running!")
    print("Press Ctrl+C to stop\n")
    
    app.run_server(debug=True, host='0.0.0.0', port=8050)
