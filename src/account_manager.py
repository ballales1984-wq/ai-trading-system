"""
Account Manager - Gestione Utenti e API Keys
============================================
Gestisce gli utenti, le loro API keys, e traccia equity/PnL.

Modello: Managed Account dove l'utente collega il proprio conto exchange.
"""

import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import uuid

logger = logging.getLogger(__name__)


@dataclass
class User:
    """Rappresenta un utente nel sistema."""
    user_id: str
    username: str
    email: str
    api_key: str
    api_secret: str
    exchange: str = "binance"
    testnet: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = True
    
    # Configurazione trading
    max_risk_per_trade: float = 0.02  # 2% per trade
    max_daily_loss: float = 0.05  # 5% max daily loss
    
    # Fee configuration
    performance_fee_pct: float = 20.0  # 20% performance fee
    management_fee_pct: float = 2.0  # 2% annual management fee
    
    # High water mark per fee calculation
    high_water_mark: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        return cls(**data)


@dataclass 
class AccountSnapshot:
    """Snapshot giornaliero dell'account dell'utente."""
    user_id: str
    timestamp: str
    equity: float
    balance: float
    unrealized_pnl: float
    daily_pnl: float
    daily_pnl_pct: float
    open_positions: int
    fee_accrued: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AccountManager:
    """
    Gestisce gli utenti e i loro account.
    
    Responsabilità:
    - Registrazione nuovi utenti
    - Validazione API keys
    - Tracking equity giornaliero
    - Calcolo PnL
    """
    
    def __init__(self, data_dir: str = "data/accounts"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.users_file = self.data_dir / "users.json"
        self.snapshots_dir = self.data_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        
        self.users: Dict[str, User] = {}
        self._load_users()
    
    def _load_users(self):
        """Carica gli utenti da file."""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    for user_id, user_data in data.items():
                        self.users[user_id] = User.from_dict(user_data)
                logger.info(f"✅ Caricati {len(self.users)} utenti")
            except Exception as e:
                logger.error(f"Errore caricamento utenti: {e}")
    
    def _save_users(self):
        """Salva gli utenti su file."""
        data = {user_id: user.to_dict() for user_id, user in self.users.items()}
        with open(self.users_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_user_id(self, username: str) -> str:
        """Genera un user ID univoco."""
        raw = f"{username}_{datetime.now().isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
    
    def register_user(
        self,
        username: str,
        email: str,
        api_key: str,
        api_secret: str,
        exchange: str = "binance",
        testnet: bool = True,
    ) -> User:
        """
        Registra un nuovo utente.
        
        Args:
            username: Nome utente
            email: Email
            api_key: API Key dell'exchange
            api_secret: API Secret dell'exchange
            exchange: Exchange (binance, bybit, okx)
            testnet: Usa testnet
            
        Returns:
            User creato
        """
        user_id = self._generate_user_id(username)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            api_key=api_key,
            api_secret=api_secret,
            exchange=exchange,
            testnet=testnet,
        )
        
        self.users[user_id] = user
        self._save_users()
        
        logger.info(f"✅ Utente registrato: {username} ({user_id})")
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Ritorna un utente per ID."""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Ritorna un utente per username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def update_user(self, user_id: str, **kwargs) -> bool:
        """Aggiorna i dati di un utente."""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        user.last_active = datetime.now().isoformat()
        self._save_users()
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """Elimina un utente."""
        if user_id in self.users:
            del self.users[user_id]
            self._save_users()
            return True
        return False
    
    def list_users(self) -> List[User]:
        """Lista tutti gli utenti."""
        return list(self.users.values())
    
    def list_active_users(self) -> List[User]:
        """Lista solo gli utenti attivi."""
        return [u for u in self.users.values() if u.is_active]


class EquityTracker:
    """
    Traccia l'equity degli account nel tempo.
    Calcola PnL e salvataggio snapshots.
    """
    
    def __init__(self, data_dir: str = "data/accounts"):
        self.data_dir = Path(data_dir)
        self.snapshots_dir = self.data_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_snapshot_file(self, user_id: str) -> Path:
        return self.snapshots_dir / f"{user_id}_snapshots.json"
    
    def save_snapshot(self, snapshot: AccountSnapshot):
        """Salva uno snapshot dell'account."""
        file_path = self._get_snapshot_file(snapshot.user_id)
        
        snapshots = []
        if file_path.exists():
            with open(file_path, 'r') as f:
                snapshots = json.load(f)
        
        snapshots.append(snapshot.to_dict())
        
        # Mantieni solo gli ultimi 365 giorni
        if len(snapshots) > 365:
            snapshots = snapshots[-365:]
        
        with open(file_path, 'w') as f:
            json.dump(snapshots, f, indent=2)
    
    def get_snapshots(self, user_id: str, days: int = 30) -> List[AccountSnapshot]:
        """Ritorna gli snapshot degli ultimi N giorni."""
        file_path = self._get_snapshot_file(user_id)
        
        if not file_path.exists():
            return []
        
        with open(file_path, 'r') as f:
            snapshots = json.load(f)
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        return [AccountSnapshot(**s) for s in snapshots if s["timestamp"] >= cutoff]
    
    def get_latest_snapshot(self, user_id: str) -> Optional[AccountSnapshot]:
        """Ritorna l'ultimo snapshot."""
        snapshots = self.get_snapshots(user_id, days=1)
        return snapshots[-1] if snapshots else None
    
    def calculate_pnl(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Calcola il PnL su N giorni."""
        snapshots = self.get_snapshots(user_id, days=days)
        
        if len(snapshots) < 2:
            return {"error": "Non abbastanza dati"}
        
        first = snapshots[0]
        last = snapshots[-1]
        
        total_pnl = last["equity"] - first["equity"]
        total_pnl_pct = (total_pnl / first["equity"]) * 100 if first["equity"] > 0 else 0
        
        return {
            "start_equity": first["equity"],
            "end_equity": last["equity"],
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "days": days,
            "start_date": first["timestamp"],
            "end_date": last["timestamp"],
        }


class PerformanceFeeCalculator:
    """
    Calcola la performance fee basata su high-water mark.
    
    Fee structure:
    - Management fee: 2% annuale
    - Performance fee: 20% sopra l'high-water mark
    """
    
    def __init__(self):
        self.management_fee_annual = 0.02  # 2%
        self.performance_fee = 0.20  # 20%
    
    def calculate_fees(
        self,
        current_equity: float,
        high_water_mark: float,
        days_elapsed: int = 30,
    ) -> Dict[str, float]:
        """
        Calcola le fee per il periodo.
        
        Args:
            current_equity: Equity attuale
            high_water_mark: High water mark precedente
            days_elapsed: Giorni dall'ultimo calcolo
            
        Returns:
            Dizionario con fee breakdown
        """
        # Management fee (pro-rata giornaliero)
        daily_mgmt_fee = self.management_fee_annual / 365
        mgmt_fee = current_equity * daily_mgmt_fee * days_elapsed
        
        # Performance fee (solo se sopra HWM)
        if current_equity > high_water_mark:
            profit = current_equity - high_water_mark
            perf_fee = profit * self.performance_fee
            new_hwm = current_equity
        else:
            perf_fee = 0.0
            new_hwm = high_water_mark
        
        total_fee = mgmt_fee + perf_fee
        net_equity = current_equity - total_fee
        
        return {
            "gross_equity": current_equity,
            "high_water_mark": high_water_mark,
            "management_fee": mgmt_fee,
            "performance_fee": perf_fee,
            "total_fee": total_fee,
            "net_equity": net_equity,
            "new_high_water_mark": new_hwm,
        }
    
    def generate_monthly_report(
        self,
        user_id: str,
        snapshots: List[AccountSnapshot],
    ) -> Dict[str, Any]:
        """Genera un report mensile per l'utente."""
        if len(snapshots) < 2:
            return {"error": "Dati insufficienti"}
        
        first = snapshots[0]
        last = snapshots[-1]
        
        gross_pnl = last["equity"] - first["equity"]
        fees = self.calculate_fees(
            current_equity=last["equity"],
            high_water_mark=first["equity"],  # Simplified
            days_elapsed=30,
        )
        
        return {
            "user_id": user_id,
            "period": {
                "start": first["timestamp"],
                "end": last["timestamp"],
            },
            "equity": {
                "start": first["equity"],
                "end": last["equity"],
            },
            "pnl": {
                "gross": gross_pnl,
                "gross_pct": (gross_pnl / first["equity"]) * 100,
                "net": gross_pnl - fees["total_fee"],
                "net_pct": ((gross_pnl - fees["total_fee"]) / first["equity"]) * 100,
            },
            "fees": {
                "management": fees["management_fee"],
                "performance": fees["performance_fee"],
                "total": fees["total_fee"],
            },
            "metrics": {
                "max_drawdown": self._calculate_max_drawdown(snapshots),
                "win_rate": self._calculate_win_rate(snapshots),
            },
        }
    
    def _calculate_max_drawdown(self, snapshots: List[AccountSnapshot]) -> float:
        """Calcola il max drawdown."""
        if not snapshots:
            return 0.0
        
        equity_curve = [s["equity"] for s in snapshots]
        peak = equity_curve[0]
        max_dd = 0.0
        
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd * 100
    
    def _calculate_win_rate(self, snapshots: List[AccountSnapshot]) -> float:
        """Calcola il win rate (giorni positivi)."""
        if len(snapshots) < 2:
            return 0.0
        
        positive_days = sum(1 for i in range(1, len(snapshots)) 
                           if snapshots[i]["daily_pnl"] > 0)
        
        return (positive_days / (len(snapshots) - 1)) * 100


# ======================
# API KEY VALIDATOR
# ======================

def validate_exchange_api(
    exchange: str,
    api_key: str,
    api_secret: str,
    testnet: bool = True,
) -> Dict[str, Any]:
    """
    Valida le credenziali dell'exchange.
    
    Returns:
        Dizionario con is_valid e message
    """
    try:
        if exchange.lower() == "binance":
            from binance.client import Client
            from binance.exceptions import BinanceAPIException
            
            client = Client(api_key, api_secret, testnet=testnet)
            account = client.get_account()
            
            return {
                "is_valid": True,
                "message": "API key valida",
                "balances": {
                    b["asset"]: float(b["free"]) 
                    for b in account.get("balances", []) 
                    if float(b["free"]) > 0
                },
            }
        
        else:
            return {
                "is_valid": False,
                "message": f"Exchange {exchange} non supportato",
            }
    
    except BinanceAPIException as e:
        return {
            "is_valid": False,
            "message": f"Errore API: {e.error_code} - {e.message}",
        }
    except Exception as e:
        return {
            "is_valid": False,
            "message": f"Errore: {str(e)}",
        }


# ======================
# ESEMPIO DI UTILIZZO
# ======================

if __name__ == "__main__":
    # Test del sistema
    
    # 1. Crea account manager
    am = AccountManager()
    
    # 2. Registra un utente di test
    user = am.register_user(
        username="test_user",
        email="test@example.com",
        api_key="test_api_key",
        api_secret="test_api_secret",
        testnet=True,
    )
    
    print(f"✅ Utente creato: {user.username} ({user.user_id})")
    
    # 3. Test equity tracker
    et = EquityTracker()
    
    # Simula alcuni snapshot
    for i in range(5):
        snapshot = AccountSnapshot(
            user_id=user.user_id,
            timestamp=datetime.now().isoformat(),
            equity=10000 + i * 100,
            balance=10000,
            unrealized_pnl=i * 100,
            daily_pnl=100,
            daily_pnl_pct=1.0,
            open_positions=1,
        )
        et.save_snapshot(snapshot)
    
    # 4. Calcola PnL
    pnl = et.calculate_pnl(user.user_id, days=30)
    print(f"📊 PnL: {pnl}")
    
    # 5. Calcola fee
    pfc = PerformanceFeeCalculator()
    fees = pfc.calculate_fees(
        current_equity=10500,
        high_water_mark=10000,
        days_elapsed=30,
    )
    print(f"💰 Fee: {fees}")
