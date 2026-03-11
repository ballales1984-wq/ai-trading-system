"""
Tests for Fund Management Module
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta

from src.fund.fund_manager import (
    FundManager,
    FundStatus,
    InvestorStatus,
    FeeStructure,
    Investor
)


class TestFundManager:
    """Test FundManager class"""
    
    def test_fund_creation(self):
        """Test fund creation with initial capital"""
        fund = FundManager("Test Fund", Decimal("1000000"))
        
        assert fund.name == "Test Fund"
        assert fund.status == FundStatus.SETUP
        assert fund.initial_capital == Decimal("1000000")
        assert fund.current_nav == Decimal("100.00")
        assert fund.total_shares == Decimal("10000")
    
    def test_open_fund(self):
        """Test opening the fund"""
        fund = FundManager("Test Fund", Decimal("1000000"))
        fund.open_fund()
        
        assert fund.status == FundStatus.OPEN
    
    def test_add_investor(self):
        """Test adding investor"""
        fund = FundManager("Test Fund", Decimal("1000000"))
        investor = fund.add_investor("John Doe", "john@example.com")
        
        assert investor.name == "John Doe"
        assert investor.email == "john@example.com"
        assert investor.investor_type == "individual"
        assert investor.status == InvestorStatus.PENDING
        assert investor.id in fund.investors
    
    def test_approve_investor(self):
        """Test approving investor"""
        fund = FundManager("Test Fund", Decimal("1000000"))
        investor = fund.add_investor("John Doe", "john@example.com")
        
        result = fund.approve_investor(investor.id)
        
        assert result is True
        assert fund.investors[investor.id].status == InvestorStatus.APPROVED
    
    def test_process_subscription(self):
        """Test processing subscription"""
        fund = FundManager("Test Fund", Decimal("1000000"))
        fund.open_fund()
        
        investor = fund.add_investor("John Doe", "john@example.com")
        investor.kyc_approved = True
        
        subscription = fund.process_subscription(
            investor.id, 
            Decimal("100000"),
            check_kyc=False
        )
        
        assert subscription is not None
        assert subscription.status == "completed"
        assert investor.total_contributed == Decimal("100000")
    
    def test_process_subscription_with_entry_fee(self):
        """Test subscription with entry fee"""
        fund = FundManager("Test Fund", Decimal("1000000"))
        fund.fee_structure.entry_fee = 0.01  # 1%
        fund.open_fund()
        
        investor = fund.add_investor("John Doe", "john@example.com")
        investor.kyc_approved = True
        
        subscription = fund.process_subscription(
            investor.id,
            Decimal("100000"),
            check_kyc=False
        )
        
        assert subscription is not None
        # Entry fee was charged
        assert investor.total_fees_paid > 0
    
    def test_process_redemption(self):
        """Test processing redemption"""
        fund = FundManager("Test Fund", Decimal("1000000"))
        fund.open_fund()
        
        investor = fund.add_investor("John Doe", "john@example.com")
        investor.kyc_approved = True
        investor.current_value = Decimal("100000")
        
        # Prima aggiorno NAV e total_shares per avere un valore consistente
        fund.total_shares += Decimal("1000")  # shares per l'investitore
        fund.total_aum += Decimal("100000")
        
        shares_to_redeem = Decimal("500")
        
        redemption = fund.process_redemption(
            investor.id,
            shares_to_redeem,
            check_min_balance=False
        )
        
        assert redemption is not None
        assert redemption.status == "completed"
    
    def test_update_nav(self):
        """Test updating NAV"""
        fund = FundManager("Test Fund", Decimal("1000000"))
        
        fund.update_nav(
            portfolio_value=Decimal("1100000"),
            cash=Decimal("100000")
        )
        
        assert fund.current_nav > Decimal("100.00")
        assert len(fund.nav_history) == 2
    
    def test_calculate_fees(self):
        """Test calculating fees"""
        fund = FundManager("Test Fund", Decimal("1000000"))
        
        fees = fund.calculate_fees()
        
        assert "management_fees" in fees
        assert "performance_fees" in fees
        assert "total_fees" in fees
    
    def test_get_fund_performance(self):
        """Test getting fund performance"""
        fund = FundManager("Test Fund", Decimal("1000000"))
        fund.open_fund()
        
        perf = fund.get_fund_performance()
        
        assert "fund_name" in perf
        assert "current_nav" in perf
        assert "total_aum" in perf
        assert perf["fund_name"] == "Test Fund"
    
    def test_investor_gain_loss(self):
        """Test investor gain/loss calculation"""
        investor = Investor(
            name="Test Investor",
            email="test@example.com",
            current_value=Decimal("110000"),
            total_contributed=Decimal("100000"),
            total_redeemed=Decimal("5000")
        )
        
        assert investor.gain_loss == Decimal("15000")
    
    def test_investor_return_percentage(self):
        """Test investor return percentage"""
        investor = Investor(
            name="Test Investor",
            email="test@example.com",
            current_value=Decimal("120000"),
            total_contributed=Decimal("100000"),
            total_redeemed=Decimal("0")
        )
        
        assert investor.return_percentage == 20.0
    
    def test_fund_serialization(self):
        """Test fund serialization"""
        fund = FundManager("Test Fund", Decimal("1000000"))
        
        data = fund.to_dict()
        
        assert data["name"] == "Test Fund"
        assert data["status"] == "setup"
        assert "current_nav" in data
        assert "total_aum" in data


class TestFeeStructure:
    """Test FeeStructure class"""
    
    def test_fee_structure_defaults(self):
        """Test default fee structure"""
        fees = FeeStructure()
        
        assert fees.management_fee_annual == 0.02
        assert fees.performance_fee_annual == 0.20
        assert fees.high_water_mark is True
        assert fees.hurdle_rate == 0.0
    
    def test_fee_structure_to_dict(self):
        """Test fee structure serialization"""
        fees = FeeStructure(
            management_fee_annual=0.015,
            performance_fee_annual=0.20,
            entry_fee=0.005
        )
        
        data = fees.to_dict()
        
        assert data["management_fee_annual"] == 0.015
        assert data["performance_fee_annual"] == 0.20
        assert data["entry_fee"] == 0.005


class TestPerformance:
    """Test performance metrics"""
    
    def test_fund_performance_with_returns(self):
        """Test fund with sample returns"""
        fund = FundManager("Test Fund", Decimal("1000000"))
        
        # Simula alcuni giorni di trading
        for i in range(10):
            portfolio = Decimal("1000000") + Decimal(str(i * 1000))
            cash = Decimal("100000")
            fund.update_nav(portfolio, cash)
        
        perf = fund.get_fund_performance()
        
        assert "cumulative_return" in perf
        assert perf["num_investors"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
