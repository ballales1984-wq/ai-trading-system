"""
Fund Manager Module - AI Trading System
Gestisce la struttura e le operazioni di un fondo di investimento
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
from enum import Enum
import json
import uuid


class FundStatus(Enum):
    """Stati del fondo"""
    SETUP = "setup"
    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATING = "liquidating"
    LIQUIDATED = "liquidated"


class FeeType(Enum):
    """Tipologie di fee"""
    MANAGEMENT = "management"      # Fee di gestione (AUM)
    PERFORMANCE = "performance"    # Fee di performance (profit)
    ENTRY = "entry"               # Fee di ingresso
    EXIT = "exit"                 # Fee di uscita


class InvestorStatus(Enum):
    """Stati investitore"""
    PENDING = "pending"
    APPROVED = "approved"
    ACTIVE = "active"
    REDEEMING = "redeeming"
    REDEEMED = "redeemed"


@dataclass
class FeeStructure:
    """Struttura delle commissioni del fondo"""
    management_fee_annual: float = 0.02        # 2% annuale
    performance_fee_annual: float = 0.20       # 20% sui profitti
    high_water_mark: bool = True               # High water mark
    hurdle_rate: float = 0.0                  # Soglia minima per performance fee
    entry_fee: float = 0.0                    # Fee ingresso
    exit_fee: float = 0.0                     # Fee uscita
    
    def to_dict(self) -> Dict:
        return {
            "management_fee_annual": self.management_fee_annual,
            "performance_fee_annual": self.performance_fee_annual,
            "high_water_mark": self.high_water_mark,
            "hurdle_rate": self.hurdle_rate,
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee
        }


@dataclass
class Investor:
    """Rappresenta un investitore nel fondo"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    email: str = ""
    investor_type: str = "individual"  # individual, institutional, accredited
    status: InvestorStatus = InvestorStatus.PENDING
    initial_investment: Decimal = Decimal("0")
    current_value: Decimal = Decimal("0")
    total_contributed: Decimal = Decimal("0")
    total_redeemed: Decimal = Decimal("0")
    total_fees_paid: Decimal = Decimal("0")
    high_water_mark: Decimal = Decimal("0")
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    notes: str = ""
    kyc_approved: bool = False
    accreditation_level: str = "none"
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "investor_type": self.investor_type,
            "status": self.status.value,
            "initial_investment": str(self.initial_investment),
            "current_value": str(self.current_value),
            "total_contributed": str(self.total_contributed),
            "total_redeemed": str(self.total_redeemed),
            "total_fees_paid": str(self.total_fees_paid),
            "high_water_mark": str(self.high_water_mark),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "notes": self.notes,
            "kyc_approved": self.kyc_approved,
            "accreditation_level": self.accreditation_level
        }
    
    @property
    def gain_loss(self) -> Decimal:
        """Guadagno/perdita totale"""
        return self.current_value + self.total_redeemed - self.total_contributed
    
    @property
    def return_percentage(self) -> float:
        """Ritorno percentuale"""
        if self.total_contributed == 0:
            return 0.0
        return float(self.gain_loss / self.total_contributed * 100)


@dataclass
class NAV:
    """Net Asset Value del fondo"""
    date: datetime
    nav_per_share: Decimal
    total_shares: Decimal
    total_aum: Decimal
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "nav_per_share": str(self.nav_per_share),
            "total_shares": str(self.total_shares),
            "total_aum": str(self.total_aum),
            "daily_return": self.daily_return,
            "cumulative_return": self.cumulative_return
        }


@dataclass
class Subscription:
    """Richiesta di sottoscrizione"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    investor_id: str = ""
    amount: Decimal = Decimal("0")
    status: str = "pending"  # pending, processing, completed, rejected
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    shares_issued: Decimal = Decimal("0")
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "investor_id": self.investor_id,
            "amount": str(self.amount),
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "shares_issued": str(self.shares_issued)
        }


@dataclass
class Redemption:
    """Richiesta di rimborso"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    investor_id: str = ""
    shares: Decimal = Decimal("0")
    amount: Decimal = Decimal("0")
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "investor_id": self.investor_id,
            "shares": str(self.shares),
            "amount": str(self.amount),
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None
        }


class FundManager:
    """
    Gestisce le operazioni del fondo di investimento
    """
    
    def __init__(self, name: str, initial_capital: Decimal = Decimal("1000000")):
        self.id = str(uuid.uuid4())
        self.name = name
        self.status = FundStatus.SETUP
        self.initial_capital = initial_capital
        self.fee_structure = FeeStructure()
        
        # Struttura NAV
        self.nav_history: List[NAV] = []
        self.current_nav = Decimal("100.00")  # NAV iniziale per share
        self.total_shares = initial_capital / self.current_nav
        
        # Investitori
        self.investors: Dict[str, Investor] = {}
        self.subscriptions: List[Subscription] = []
        self.redemptions: List[Redemption] = []
        
        # Performance tracking
        self.high_water_mark = Decimal("100.00")
        self.cumulative_return = 0.0
        self.daily_returns: List[float] = []
        
        # Contabilità
        self.total_aum = initial_capital
        self.cash_balance = initial_capital
        self.invested_capital = Decimal("0")
        self.pendingSubscriptions = Decimal("0")
        self.pendingRedemptions = Decimal("0")
        
        # Timestamps
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.last_nav_date: Optional[datetime] = None
        
        # Crea NAV iniziale
        self._create_initial_nav()
    
    def _create_initial_nav(self):
        """Crea il NAV iniziale"""
        nav = NAV(
            date=datetime.now(),
            nav_per_share=self.current_nav,
            total_shares=self.total_shares,
            total_aum=self.total_aum,
            daily_return=0.0,
            cumulative_return=0.0
        )
        self.nav_history.append(nav)
        self.last_nav_date = datetime.now()
    
    def open_fund(self):
        """Apre il fondo agli investitori"""
        if self.status == FundStatus.SETUP:
            self.status = FundStatus.OPEN
            self.updated_at = datetime.now()
    
    def close_fund(self):
        """Chiude il fondo nuove sottoscrizioni"""
        if self.status == FundStatus.OPEN:
            self.status = FundStatus.CLOSED
            self.updated_at = datetime.now()
    
    def add_investor(self, name: str, email: str, investor_type: str = "individual") -> Investor:
        """Aggiunge un nuovo investitore"""
        investor = Investor(
            name=name,
            email=email,
            investor_type=investor_type,
            status=InvestorStatus.PENDING
        )
        self.investors[investor.id] = investor
        return investor
    
    def approve_investor(self, investor_id: str) -> bool:
        """Approva un investitore"""
        if investor_id in self.investors:
            investor = self.investors[investor_id]
            investor.status = InvestorStatus.APPROVED
            investor.updated_at = datetime.now()
            return True
        return False
    
    def process_subscription(self, investor_id: str, amount: Decimal, 
                           check_kyc: bool = True) -> Optional[Subscription]:
        """
        Processa una richiesta di sottoscrizione
        """
        if investor_id not in self.investors:
            return None
        
        investor = self.investors[investor_id]
        
        # Check KYC se richiesto
        if check_kyc and not investor.kyc_approved:
            return None
        
        # Check stato fondo
        if self.status != FundStatus.OPEN:
            return None
        
        # Calcola shares da emettere
        shares = amount / self.current_nav
        
        # Crea sottoscrizione
        sub = Subscription(
            investor_id=investor_id,
            amount=amount,
            shares_issued=shares
        )
        
        # Applica entry fee
        net_amount = amount
        if self.fee_structure.entry_fee > 0:
            entry_fee_amount = amount * Decimal(str(self.fee_structure.entry_fee))
            net_amount = amount - entry_fee_amount
            investor.total_fees_paid += entry_fee_amount
        
        # Aggiorna investitore
        investor.total_contributed += amount
        investor.current_value += net_amount
        investor.high_water_mark = self.current_nav
        investor.status = InvestorStatus.ACTIVE
        
        # Aggiorna fondo
        self.pendingSubscriptions += amount
        self.total_shares += shares
        self.total_aum += amount
        
        sub.status = "completed"
        sub.processed_at = datetime.now()
        
        self.subscriptions.append(sub)
        self.updated_at = datetime.now()
        
        return sub
    
    def process_redemption(self, investor_id: str, shares: Decimal,
                          check_min_balance: bool = True) -> Optional[Redemption]:
        """
        Processa una richiesta di rimborso
        """
        if investor_id not in self.investors:
            return None
        
        investor = self.investors[investor_id]
        
        # Check balance minimo
        if check_min_balance:
            remaining_shares = (investor.current_value / self.current_nav) - shares
            if remaining_shares * self.current_nav < Decimal("1000"):  # Min 1000
                return None
        
        # Calcola valore rimborso
        amount = shares * self.current_nav
        
        # Applica exit fee
        net_amount = amount
        if self.fee_structure.exit_fee > 0:
            exit_fee_amount = amount * Decimal(str(self.fee_structure.exit_fee))
            net_amount = amount - exit_fee_amount
            investor.total_fees_paid += exit_fee_amount
        
        # Crea richiesta rimborso
        red = Redemption(
            investor_id=investor_id,
            shares=shares,
            amount=net_amount
        )
        
        # Aggiorna investitore
        investor.current_value -= amount
        investor.total_redeemed += net_amount
        
        if investor.current_value <= Decimal("100"):
            investor.status = InvestorStatus.REDEEMED
        
        # Aggiorna fondo
        self.pendingRedemptions += amount
        self.total_shares -= shares
        self.total_aum -= amount
        
        red.status = "completed"
        red.processed_at = datetime.now()
        
        self.redemptions.append(red)
        self.updated_at = datetime.now()
        
        return red
    
    def calculate_fees(self) -> Dict[str, Decimal]:
        """
 fee        Calcola le da addebitare agli investitori
        """
        fees = {
            "management_fees": Decimal("0"),
            "performance_fees": Decimal("0"),
            "total_fees": Decimal("0")
        }
        
        # Management fee (annualizzata, pro-rata per giorno)
        if self.last_nav_date:
            days = (datetime.now() - self.last_nav_date).days
            mgmt_fee_daily = self.fee_structure.management_fee_annual / 365
            mgmt_fee_decimal = Decimal(str(mgmt_fee_daily))
            fees["management_fees"] = self.total_aum * mgmt_fee_decimal * Decimal(str(days))
        
        # Performance fee
        if self.fee_structure.performance_fee_annual > 0:
            if self.current_nav > self.high_water_mark:
                # Calcolo excess return
                excess = self.current_nav - self.high_water_mark
                perf_fee = float(excess) * float(self.total_shares) * self.fee_structure.performance_fee_annual
                fees["performance_fees"] = Decimal(str(perf_fee))
        
        fees["total_fees"] = fees["management_fees"] + fees["performance_fees"]
        
        return fees
    
    def update_nav(self, portfolio_value: Decimal, cash: Decimal):
        """
        Aggiorna il NAV del fondo
        """
        # Calcola nuovo NAV
        new_aum = portfolio_value + cash - self.pendingSubscriptions + self.pendingRedemptions
        new_nav = new_aum / self.total_shares if self.total_shares > 0 else Decimal("0")
        
        # Calcola ritorno giornaliero
        daily_return = 0.0
        if self.current_nav > 0:
            daily_return = (float(new_nav) / float(self.current_nav) - 1) * 100
        
        self.cumulative_return += daily_return
        
        # Crea nuovo record NAV
        nav = NAV(
            date=datetime.now(),
            nav_per_share=new_nav,
            total_shares=self.total_shares,
            total_aum=new_aum,
            daily_return=daily_return,
            cumulative_return=self.cumulative_return
        )
        
        self.nav_history.append(nav)
        self.current_nav = new_nav
        self.total_aum = new_aum
        self.cash_balance = cash
        self.invested_capital = portfolio_value
        self.pendingSubscriptions = Decimal("0")
        self.pendingRedemptions = Decimal("0")
        
        # Update high water mark
        if new_nav > self.high_water_mark:
            self.high_water_mark = new_nav
        
        self.last_nav_date = datetime.now()
        self.updated_at = datetime.now()
        
        # Track daily returns
        self.daily_returns.append(daily_return)
    
    def get_investor_portfolio(self, investor_id: str) -> Optional[Dict]:
        """
        Restituisce il portfolio di un investitore
        """
        if investor_id not in self.investors:
            return None
        
        investor = self.investors[investor_id]
        
        return {
            "investor": investor.to_dict(),
            "shares": investor.current_value / self.current_nav if self.current_nav > 0 else Decimal("0"),
            "nav": self.current_nav,
            "current_value": investor.current_value,
            "cost_basis": investor.total_contributed,
            "gain_loss": investor.gain_loss,
            "return_pct": investor.return_percentage,
            "allocation_pct": float(investor.current_value / self.total_aum * 100) if self.total_aum > 0 else 0
        }
    
    def get_fund_performance(self) -> Dict:
        """
        Restituisce le performance del fondo
        """
        return {
            "fund_name": self.name,
            "status": self.status.value,
            "current_nav": float(self.current_nav),
            "total_aum": float(self.total_aum),
            "total_shares": float(self.total_shares),
            "cumulative_return": self.cumulative_return,
            "volatility": self._calculate_volatility(),
            "sharpe_ratio": self._calculate_sharpe(),
            "max_drawdown": self._calculate_max_drawdown(),
            "num_investors": len([i for i in self.investors.values() 
                                 if i.status in [InvestorStatus.ACTIVE, InvestorStatus.APPROVED]]),
            "fee_structure": self.fee_structure.to_dict()
        }
    
    def _calculate_volatility(self) -> float:
        """Calcola la volatilità dei rendimenti"""
        if len(self.daily_returns) < 2:
            return 0.0
        import statistics
        return statistics.stdev(self.daily_returns) if len(self.daily_returns) > 1 else 0.0
    
    def _calculate_sharpe(self) -> float:
        """Calcola lo Sharpe Ratio (assumendo risk-free = 2%)"""
        if len(self.daily_returns) < 2:
            return 0.0
        avg_return = sum(self.daily_returns) / len(self.daily_returns)
        volatility = self._calculate_volatility()
        if volatility == 0:
            return 0.0
        return (avg_return - 0.02/252) / volatility * (252 ** 0.5)
    
    def _calculate_max_drawdown(self) -> float:
        """Calcola il max drawdown"""
        if not self.nav_history:
            return 0.0
        
        peak = self.nav_history[0].nav_per_share
        max_dd = 0.0
        
        for nav in self.nav_history:
            if nav.nav_per_share > peak:
                peak = nav.nav_per_share
            
            dd = (float(peak) - float(nav.nav_per_share)) / float(peak) * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def to_dict(self) -> Dict:
        """Serializza il fondo"""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "initial_capital": str(self.initial_capital),
            "current_nav": str(self.current_nav),
            "total_aum": str(self.total_aum),
            "total_shares": str(self.total_shares),
            "cumulative_return": self.cumulative_return,
            "fee_structure": self.fee_structure.to_dict(),
            "num_investors": len(self.investors),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def save_to_file(self, filepath: str):
        """Salva il fondo su file JSON"""
        data = {
            "fund": self.to_dict(),
            "investors": [i.to_dict() for i in self.investors.values()],
            "nav_history": [n.to_dict() for n in self.nav_history],
            "subscriptions": [s.to_dict() for s in self.subscriptions],
            "redemptions": [r.to_dict() for r in self.redemptions]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'FundManager':
        """Carica il fondo da file JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        fund_data = data["fund"]
        manager = cls(
            name=fund_data["name"],
            initial_capital=Decimal(fund_data["initial_capital"])
        )
        
        # Rebuild investors
        for inv_data in data.get("investors", []):
            inv = Investor(
                id=inv_data["id"],
                name=inv_data["name"],
                email=inv_data["email"],
                investor_type=inv_data["investor_type"],
                status=InvestorStatus(inv_data["status"]),
                initial_investment=Decimal(inv_data["initial_investment"]),
                current_value=Decimal(inv_data["current_value"]),
                total_contributed=Decimal(inv_data["total_contributed"]),
                total_redeemed=Decimal(inv_data["total_redeemed"]),
                total_fees_paid=Decimal(inv_data["total_fees_paid"]),
                high_water_mark=Decimal(inv_data["high_water_mark"]),
                kyc_approved=inv_data.get("kyc_approved", False)
            )
            manager.investors[inv.id] = inv
        
        return manager
