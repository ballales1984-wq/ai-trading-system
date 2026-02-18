"""
App Analysis Tests - Verify functionality while app is running
These tests verify that all components work correctly in real-time
"""
import pytest
import requests
import time
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDashboardRunning:
    """Test that dashboard is running and responding"""
    
    def test_dashboard_is_accessible(self):
        """Verify dashboard is accessible at localhost:8050"""
        try:
            response = requests.get("http://127.0.0.1:8050", timeout=10)
            assert response.status_code == 200, f"Dashboard returned status {response.status_code}"
            assert "Quantum AI" in response.text or "Trading" in response.text
            print(f"✅ Dashboard accessible - Status: {response.status
