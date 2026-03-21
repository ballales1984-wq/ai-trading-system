"""
Validation Dashboard
===================
Streamlit dashboard to visualize validation reports.

Usage:
    streamlit run tests/validation_dashboard.py
"""

import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd

# Page config
st.set_page_config(
    page_title="AI Trading System - Validation Dashboard",
    page_icon="📊",
    layout="wide"
)

def load_report(filepath: str = "test_report_full_validation.json"):
    """Load validation report from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def main():
    st.title("AI Trading System - Validation Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Settings")
    report_file = st.sidebar.text_input("Report File", "test_report_full_validation.json")
    
    # Load report
    report = load_report(report_file)
    
    if report is None:
        st.warning(f"No report found: {report_file}")
        st.info("Run validation first: `python -m tests.automated_testing_framework`")
        return
    
    # Timestamp
    st.sidebar.markdown("---")
    st.sidebar.info(f"Report timestamp: {report.get('timestamp', 'N/A')}")
    
    # Recommendation header
    recommendation = report.get('recommendation', 'UNKNOWN')
    if "APPROVED" in recommendation:
        st.success(f"## {recommendation}")
    else:
        st.error(f"## {recommendation}")
    
    st.markdown("---")
    
    # Level 1: Walk-Forward
    st.header("Level 1: Walk-Forward Backtest")
    
    wf = report.get('walk_forward', {})
    if wf:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Periods", wf.get('periods', 'N/A'))
        col2.metric("Mean Return", f"{wf.get('mean_return', 0):.2%}")
        col3.metric("Sharpe (mean)", f"{wf.get('sharpe_mean', 0):.2f}")
        
        ci = wf.get('bootstrap_ci', [0, 0])
        col4.metric("Bootstrap CI", f"[{ci[0]:.2%}, {ci[1]:.2%}]")
        
        # Status
        wf_passed = report.get('levels_passed', {}).get('level1_walkforward', False)
        st.progress(1.0 if wf_passed else 0.0, text="PASS" if wf_passed else "FAIL")
    else:
        st.info("No walk-forward data available")
    
    st.markdown("---")
    
    # Level 2: Paper Trading
    st.header("Level 2: Paper Trading")
    
    paper = report.get('paper_trading', {})
    if paper:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Sharpe", f"{paper.get('sharpe_ratio', 0):.2f}")
        col2.metric("Sortino", f"{paper.get('sortino_ratio', 0):.2f}")
        col3.metric("Max Drawdown", f"{paper.get('max_drawdown_percent', 0):.1f}%")
        col4.metric("Win Rate", f"{paper.get('win_rate', 0):.1f}%")
        col5.metric("Profit Factor", f"{paper.get('profit_factor', 0):.2f}")
        
        paper_passed = report.get('levels_passed', {}).get('level2_paper', False)
        st.progress(1.0 if paper_passed else 0.0, text="PASS" if paper_passed else "FAIL")
    else:
        st.info("No paper trading data available")
    
    st.markdown("---")
    
    # Level 3: Small Capital
    st.header("Level 3: Small Capital Simulation")
    
    small = report.get('small_capital', {})
    if small:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Sharpe", f"{small.get('sharpe_ratio', 0):.2f}")
        col2.metric("Sortino", f"{small.get('sortino_ratio', 0):.2f}")
        col3.metric("Max Drawdown", f"{small.get('max_drawdown_percent', 0):.1f}%")
        col4.metric("Win Rate", f"{small.get('win_rate', 0):.1f}%")
        col5.metric("Profit Factor", f"{small.get('profit_factor', 0):.2f}")
        
        small_passed = report.get('levels_passed', {}).get('level3_small_capital', False)
        st.progress(1.0 if small_passed else 0.0, text="PASS" if small_passed else "FAIL")
    else:
        st.info("No small capital data available")
    
    st.markdown("---")
    
    # Level 4: Stress Tests
    st.header("Level 4: Black Swan Stress Tests")
    
    stress = report.get('stress_tests', [])
    if stress:
        # Create dataframe for display
        stress_df = pd.DataFrame([
            {
                "Scenario": s.get('scenario', 'N/A'),
                "Max Drawdown": f"{s.get('max_drawdown', 0):.1%}",
                "Survived": "✅" if s.get('survived', False) else "❌"
            }
            for s in stress
        ])
        
        st.table(stress_df)
        
        survived = sum(1 for s in stress if s.get('survived', False))
        total = len(stress)
        st.progress(survived/total, text=f"{survived}/{total} scenarios survived")
    else:
        st.info("No stress test data available")
    
    st.markdown("---")
    
    # Summary metrics
    st.header("Summary")
    
    levels = report.get('levels_passed', {})
    
    summary_data = {
        "Level": ["Walk-Forward", "Paper Trading", "Small Capital", "Stress Tests"],
        "Status": [
            "✅ PASS" if levels.get('level1_walkforward', False) else "❌ FAIL",
            "✅ PASS" if levels.get('level2_paper', False) else "❌ FAIL",
            "✅ PASS" if levels.get('level3_small_capital', False) else "❌ FAIL",
            "✅ PASS" if levels.get('level4_stress', False) else "❌ FAIL",
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

if __name__ == "__main__":
    main()
