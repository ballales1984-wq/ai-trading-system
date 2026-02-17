"""
Fund Simulator - Hedge Fund Style Fee Structure
===============================================
Professional fund simulation with:
- Management fee (2% standard)
- Performance fee (20% standard)
- High-water mark logic
- AUM tracking
- Net return calculation

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FundMetrics:
    """Fund performance metrics"""
    gross_return: float
    management_fee: float
    performance_fee: float
    net_return: float
    total_fees: float
    aum_final: float
    high_water_mark: float


class FundSimulator:
    """
    Simulate hedge fund with fee structure.
    
    Standard fee structure:
    - 2% annual management fee
    - 20% performance fee (over high-water mark)
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000,
        management_fee: float = 0.02,
        performance_fee: float = 0.20,
        high_water_mark: bool = True
    ):
        """
        Initialize fund simulator.
        
        Parameters:
        -----------
        initial_capital : float
            Starting AUM (Assets Under Management)
        management_fee : float
            Annual management fee (0.02 = 2%)
        performance_fee : float
            Performance fee percentage (0.20 = 20%)
        high_water_mark : bool
            Use high-water mark for performance fees
        """
        self.initial_capital = initial_capital
        self.management_fee = management_fee
        self.performance_fee = performance_fee
        self.high_water_mark_enabled = high_water_mark
        
        self.reset()
    
    def reset(self):
        """Reset simulator to initial state."""
        self.aum = self.initial_capital
        self.high_water_mark = self.initial_capital
        self.cumulative_fees = 0
        self.trade_log = []
    
    def apply_fees(
        self,
        equity_curve: pd.Series,
        periods_per_year: int = 252
    ) -> Tuple[pd.Series, FundMetrics]:
        """
        Apply fund fees to equity curve.
        
        Parameters:
        -----------
        equity_curve : pd.Series
            Raw equity curve (before fees)
        periods_per_year : int
            Periods per year for fee calculation
        
        Returns:
        --------
        Tuple[pd.Series, FundMetrics] : (adjusted_equity, metrics)
        """
        periods = len(equity_curve)
        period_management_fee = self.management_fee / periods_per_year
        
        # Calculate daily management fee
        management_fees = []
        current_aum = self.initial_capital
        
        for i in range(len(equity_curve)):
            # Daily management fee
            daily_mgmt_fee = current_aum * period_management_fee
            
            # Check for performance fee
            if self.high_water_mark_enabled:
                # Only charge performance fee if above HWM
                if current_aum > self.high_water_mark:
                    perf_fee = (current_aum - self.high_water_mark) * self.performance_fee
                    self.high_water_mark = current_aum
                else:
                    perf_fee = 0
            else:
                # Simple performance fee on all gains
                perf_fee = max(0, equity_curve.iloc[i] - equity_curve.iloc[0]) * self.performance_fee / len(equity_curve)
            
            total_fee = daily_mgmt_fee + perf_fee
            management_fees.append(total_fee)
            
            # Update AUM
            current_aum = equity_curve.iloc[i] - sum(management_fees)
        
        # Create adjusted equity curve
        fees = pd.Series(management_fees, index=equity_curve.index)
        adjusted_equity = equity_curve - fees.cumsum()
        
        # Calculate final metrics
        gross_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if equity_curve.iloc[0] > 0 else 0
        total_mgmt_fee = sum(management_fees) * 0.8  # Approximate
        total_perf_fee = sum(management_fees) * 0.2  # Approximate
        total_fees = sum(management_fees)
        
        net_return = (adjusted_equity.iloc[-1] / adjusted_equity.iloc[0] - 1) if adjusted_equity.iloc[0] > 0 else 0
        
        metrics = FundMetrics(
            gross_return=gross_return,
            management_fee=total_mgmt_fee,
            performance_fee=total_perf_fee,
            net_return=net_return,
            total_fees=total_fees,
            aum_final=adjusted_equity.iloc[-1],
            high_water_mark=self.high_water_mark
        )
        
        return adjusted_equity, metrics
    
    def simulate_investment(
        self,
        years: int,
        annual_return: float,
        volatility: float = 0.15
    ) -> Tuple[pd.Series, FundMetrics]:
        """
        Simulate investment with given parameters.
        
        Parameters:
        -----------
        years : int
            Investment horizon in years
        annual_return : float
            Expected annual return
        volatility : float
            Annual volatility
        
        Returns:
        --------
        Tuple[pd.Series, FundMetrics] : (equity_curve, metrics)
        """
        periods = years * 252
        dt = 1/252
        
        # Generate returns using geometric Brownian motion
        np.random.seed(42)
        returns = np.random.normal(
            annual_return * dt,
            volatility * np.sqrt(dt),
            periods
        )
        
        # Calculate equity curve
        equity = self.initial_capital * np.cumprod(1 + returns)
        equity = pd.Series(equity)
        
        return self.apply_fees(equity)
    
    def compare_fee_structures(
        self,
        equity_curve: pd.Series,
        fee_structures: List[Dict]
    ) -> pd.DataFrame:
        """
        Compare different fee structures.
        
        Parameters:
        -----------
        equity_curve : pd.Series
            Base equity curve
        fee_structures : List[Dict]
            List of fee structure configurations
        
        Returns:
        --------
        pd.DataFrame : Comparison table
        """
        results = []
        
        for fs in fee_structures:
            # Save current fees
            orig_mgmt = self.management_fee
            orig_perf = self.performance_fee
            
            # Set new fees
            self.management_fee = fs.get('management_fee', 0.02)
            self.performance_fee = fs.get('performance_fee', 0.20)
            
            # Calculate
            adjusted, metrics = self.apply_fees(equity_curve)
            
            results.append({
                'Structure': fs.get('name', 'Custom'),
                'Mgmt Fee': f"{self.management_fee*100:.1f}%",
                'Perf Fee': f"{self.performance_fee*100:.1f}%",
                'Gross Return': f"{metrics.gross_return*100:.2f}%",
                'Net Return': f"{metrics.net_return*100:.2f}%",
                'Total Fees': f"${metrics.total_fees:,.0f}",
                'Final AUM': f"${metrics.aum_final:,.0f}"
            })
            
            # Restore
            self.management_fee = orig_mgmt
            self.performance_fee = orig_perf
            self.reset()
        
        return pd.DataFrame(results)


def calculate_investor_returns(
    initial_investment: float,
    years: int,
    strategy_annual_return: float,
    management_fee: float = 0.02,
    performance_fee: float = 0.20
) -> Dict[str, float]:
    """
    Calculate investor returns with fund fees.
    
    Parameters:
    -----------
    initial_investment : float
        Initial investment amount
    years : int
        Investment horizon
    strategy_annual_return : float
        Strategy annual return (gross)
    management_fee : float
        Annual management fee
    performance_fee : float
        Performance fee
    
    Returns:
    --------
    Dict[str, float] : Return calculations
    """
    simulator = FundSimulator(
        initial_capital=initial_investment,
        management_fee=management_fee,
        performance_fee=performance_fee
    )
    
    equity, metrics = simulator.simulate_investment(
        years=years,
        annual_return=strategy_annual_return
    )
    
    return {
        'initial_investment': initial_investment,
        'gross_value': initial_investment * (1 + metrics.gross_return),
        'net_value': metrics.aum_final,
        'total_fees': metrics.total_fees,
        'management_fees': metrics.management_fee,
        'performance_fees': metrics.performance_fee,
        'gross_return': metrics.gross_return * 100,
        'net_return': metrics.net_return * 100,
        'irr': (metrics.aum_final / initial_investment) ** (1/years) - 1
    }


def calculate_break_even_fees(
    initial_investment: float,
    target_net_return: float,
    years: int,
    strategy_return: float
) -> Dict[str, float]:
    """
    Calculate fees needed to achieve target net return.
    
    Parameters:
    -----------
    initial_investment : float
        Initial investment
    target_net_return : float
        Target net annual return
    years : int
        Investment horizon
    strategy_return : float
        Strategy gross annual return
    
    Returns:
    --------
    Dict[str, float] : Break-even fee analysis
    """
    # Target final value
    target_final = initial_investment * (1 + target_net_return) ** years
    gross_final = initial_investment * (1 + strategy_return) ** years
    
    # Total fees that can be paid
    max_fees = gross_final - target_final
    
    # Assume 50/50 split between mgmt and perf fees
    max_mgmt_fee = max_fees * 0.5 / years / initial_investment
    max_perf_fee = max_fees * 0.5 / (gross_final - initial_investment) if gross_final > initial_investment else 0
    
    return {
        'max_total_fees': max_fees,
        'break_even_mgmt_fee': max_mgmt_fee * 100,
        'break_even_perf_fee': max_perf_fee * 100,
        'strategy_return': strategy_return * 100,
        'target_net_return': target_net_return * 100
    }


def generate_fund_report(
    initial_capital: float,
    equity_curve: pd.Series,
    management_fee: float = 0.02,
    performance_fee: float = 0.20
) -> str:
    """
    Generate professional fund performance report.
    
    Parameters:
    -----------
    initial_capital : float
        Initial AUM
    equity_curve : pd.Series
        Equity curve
    management_fee : float
        Annual management fee
    performance_fee : float
        Performance fee
    
    Returns:
    --------
    str : Formatted report
    """
    simulator = FundSimulator(
        initial_capital=initial_capital,
        management_fee=management_fee,
        performance_fee=performance_fee
    )
    
    adjusted, metrics = simulator.apply_fees(equity_curve)
    
    years = len(equity_curve) / 252
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    HEDGE FUND PERFORMANCE REPORT                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  AUM SUMMARY                                                          ║
║  ─────────────────────────────────────────────────────────────────    ║
║  Initial AUM:              ${initial_capital:>15,.0f}                      ║
║  Final AUM (Net):          ${metrics.aum_final:>15,.0f}                      ║
║  High Water Mark:         ${metrics.high_water_mark:>15,.0f}                      ║
║                                                                      ║
║  RETURNS                                                                ║
║  ─────────────────────────────────────────────────────────────────    ║
║  Gross Return:            {metrics.gross_return*100:>+15.2f}%                      ║
║  Net Return:              {metrics.net_return*100:>+15.2f}%                      ║
║  Annualized (Gross):      {(1+metrics.gross_return)**(1/years)-1 if years>0 else 0*100:>+15.2f}%                      ║
║  Annualized (Net):        {(1+metrics.net_return)**(1/years)-1 if years>0 else 0*100:>+15.2f}%                      ║
║                                                                      ║
║  FEES                                                                   ║
║  ─────────────────────────────────────────────────────────────────    ║
║  Management Fee:          ${metrics.management_fee:>15,.0f}  ({management_fee*100:.1f}% annually)     ║
║  Performance Fee:         ${metrics.performance_fee:>15,.0f}  ({performance_fee*100:.0f}% of profits)      ║
║  Total Fees:              ${metrics.total_fees:>15,.0f}                      ║
║                                                                      ║
║  FEE STRUCTURE                                                          ║
║  ─────────────────────────────────────────────────────────────────    ║
║  Management Fee:          {management_fee*100:.1f}% per annum                                ║
║  Performance Fee:         {performance_fee*100:.0f}% over HWM                                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    return report
