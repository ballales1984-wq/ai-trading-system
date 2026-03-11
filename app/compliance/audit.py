"""
Advanced Audit Trail Module - AI Trading System
Sistema di audit trail avanzato per compliance
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import uuid
import hashlib


class AuditEventType(Enum):
    """Tipi di eventi di audit"""
    # Autenticazione
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    
    # Trading
    ORDER_CREATED = "order_created"
    ORDER_EXECUTED = "order_executed"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    
    # Account
    ACCOUNT_CREATED = "account_created"
    ACCOUNT_MODIFIED = "account_modified"
    ACCOUNT_SUSPENDED = "account_suspended"
    KYC_APPROVED = "kyc_approved"
    KYC_REJECTED = "kyc_rejected"
    
    # Fund
    SUBSCRIPTION_CREATED = "subscription_created"
    SUBSCRIPTION_PROCESSED = "subscription_processed"
    REDEMPTION_CREATED = "redemption_created"
    REDEMPTION_PROCESSED = "redemption_processed"
    FEE_CHARGED = "fee_charged"
    
    # Sistema
    CONFIG_CHANGED = "config_changed"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    TRADING_HALTED = "trading_halted"
    EMERGENCY_STOP = "emergency_stop"
    
    # Compliance
    COMPLIANCE_VIOLATION = "compliance_violation"
    REPORT_GENERATED = "report_generATED"
    ALERT_TRIGGERED = "alert_triggered"


class RiskLevel(Enum):
    """Livelli di rischio"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Stato compliance"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    PENDING_REVIEW = "pending_review"


@dataclass
class AuditEvent:
    """Evento di audit"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: AuditEventType = AuditEventType.LOGIN
    user_id: Optional[str] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    compliance_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    previous_value: Optional[str] = None
    new_value: Optional[str] = None
    session_id: Optional[str] = None
    api_key_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "username": self.username,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "details": self.details,
            "risk_level": self.risk_level.value,
            "compliance_status": self.compliance_status.value,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "session_id": self.session_id,
            "api_key_id": self.api_key_id
        }
    
    def get_hash(self) -> str:
        """Calcola hash dell'evento per integrità"""
        data = f"{self.id}{self.timestamp.isoformat()}{self.event_type.value}{self.user_id}{self.action}"
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class AuditFilter:
    """Filtri per la ricerca di audit trail"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    event_types: List[AuditEventType] = field(default_factory=list)
    user_ids: List[str] = field(default_factory=list)
    risk_levels: List[RiskLevel] = field(default_factory=list)
    resource_types: List[str] = field(default_factory=list)
    search_text: Optional[str] = None
    limit: int = 1000


class AuditLogger:
    """
    Sistema avanzato di audit trail
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.events: List[AuditEvent] = []
        self.storage_path = storage_path
        self.user_sessions: Dict[str, Dict] = {}
        
        # Statistiche
        self.stats = {
            "total_events": 0,
            "by_type": {},
            "by_user": {},
            "by_risk_level": {},
            "violations": 0
        }
    
    def log_event(self, event: AuditEvent) -> str:
        """
        Registra un evento di audit
        """
        # Aggiungi hash per integrità
        event_hash = event.get_hash()
        
        # Aggiungi agli eventi
        self.events.append(event)
        self.stats["total_events"] += 1
        
        # Aggiorna statistiche
        event_type = event.event_type.value
        self.stats["by_type"][event_type] = self.stats["by_type"].get(event_type, 0) + 1
        
        if event.user_id:
            self.stats["by_user"][event.user_id] = self.stats["by_user"].get(event.user_id, 0) + 1
        
        risk_level = event.risk_level.value
        self.stats["by_risk_level"][risk_level] = self.stats["by_risk_level"].get(risk_level, 0) + 1
        
        if event.compliance_status == ComplianceStatus.VIOLATION:
            self.stats["violations"] += 1
        
        # Salva su storage se configurato
        if self.storage_path:
            self._save_to_storage(event)
        
        return event_hash
    
    def log_login(self, user_id: str, username: str, ip_address: str,
                  success: bool, user_agent: Optional[str] = None) -> str:
        """Logga un tentativo di login"""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN if success else AuditEventType.LOGIN_FAILED,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            action="User login" if success else "Failed login attempt",
            risk_level=RiskLevel.LOW if success else RiskLevel.MEDIUM,
            compliance_status=ComplianceStatus.COMPLIANT if success else ComplianceStatus.WARNING,
            details={"success": success}
        )
        return self.log_event(event)
    
    def log_order(self, user_id: str, order_id: str, order_type: str,
                  action: str, amount: float, price: float,
                  risk_level: RiskLevel = RiskLevel.LOW) -> str:
        """Logga operazioni su ordini"""
        event = AuditEvent(
            user_id=user_id,
            resource_type="order",
            resource_id=order_id,
            action=action,
            risk_level=risk_level,
            details={
                "order_type": order_type,
                "amount": amount,
                "price": price
            }
        )
        return self.log_event(event)
    
    def log_compliance_violation(self, user_id: str, violation_type: str,
                                 details: Dict, severity: RiskLevel = RiskLevel.HIGH) -> str:
        """Logga una violazione di compliance"""
        event = AuditEvent(
            event_type=AuditEventType.COMPLIANCE_VIOLATION,
            user_id=user_id,
            action=f"Compliance violation: {violation_type}",
            risk_level=severity,
            compliance_status=ComplianceStatus.VIOLATION,
            details=details
        )
        return self.log_event(event)
    
    def log_risk_event(self, user_id: str, risk_type: str, limit: float,
                       current_value: float, severity: RiskLevel) -> str:
        """Logga un evento di rischio"""
        event = AuditEvent(
            event_type=AuditEventType.RISK_LIMIT_EXCEEDED,
            user_id=user_id,
            action=f"Risk limit exceeded: {risk_type}",
            risk_level=severity,
            compliance_status=ComplianceStatus.WARNING if severity != RiskLevel.CRITICAL else ComplianceStatus.VIOLATION,
            details={
                "risk_type": risk_type,
                "limit": limit,
                "current_value": current_value,
                "exceeded_by": current_value - limit
            }
        )
        return self.log_event(event)
    
    def query_events(self, audit_filter: AuditFilter) -> List[AuditEvent]:
        """
        Query events with filters
        """
        results = []
        
        for event in self.events:
            # Date filter
            if audit_filter.start_date and event.timestamp < audit_filter.start_date:
                continue
            if audit_filter.end_date and event.timestamp > audit_filter.end_date:
                continue
            
            # Event type filter
            if audit_filter.event_types and event.event_type not in audit_filter.event_types:
                continue
            
            # User filter
            if audit_filter.user_ids and event.user_id not in audit_filter.user_ids:
                continue
            
            # Risk level filter
            if audit_filter.risk_levels and event.risk_level not in audit_filter.risk_levels:
                continue
            
            # Resource type filter
            if audit_filter.resource_types and event.resource_type not in audit_filter.resource_types:
                continue
            
            # Text search
            if audit_filter.search_text:
                search = audit_filter.search_text.lower()
                if search not in event.action.lower():
                    continue
            
            results.append(event)
        
        # Limit results
        return results[:audit_filter.limit]
    
    def get_user_activity(self, user_id: str, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[AuditEvent]:
        """Get all activity for a user"""
        filter = AuditFilter(
            start_date=start_date,
            end_date=end_date,
            user_ids=[user_id]
        )
        return self.query_events(filter)
    
    def get_failed_logins(self, hours: int = 24) -> List[AuditEvent]:
        """Get failed login attempts in last N hours"""
        start_date = datetime.now() - timedelta(hours=hours)
        filter = AuditFilter(
            start_date=start_date,
            event_types=[AuditEventType.LOGIN_FAILED]
        )
        return self.query_events(filter)
    
    def get_compliance_violations(self, 
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> List[AuditEvent]:
        """Get all compliance violations"""
        filter = AuditFilter(
            start_date=start_date,
            end_date=end_date,
            event_types=[AuditEventType.COMPLIANCE_VIOLATION],
            compliance_statuses=[ComplianceStatus.VIOLATION]
        )
        return self.query_events(filter)
    
    def get_stats(self) -> Dict:
        """Get audit statistics"""
        return self.stats.copy()
    
    def _save_to_storage(self, event: AuditEvent):
        """Salva evento su storage"""
        # Simplified - in production would use database
        pass
    
    def export_to_json(self, filepath: str, 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None):
        """Esporta audit trail in JSON"""
        events = self.events
        
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]
        
        data = {
            "export_date": datetime.now().isoformat(),
            "total_events": len(events),
            "events": [e.to_dict() for e in events]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def verify_integrity(self) -> bool:
        """
        Verifica integrità del log (hash chain)
        """
        if len(self.events) < 2:
            return True
        
        for i in range(1, len(self.events)):
            prev_hash = self.events[i-1].get_hash()
            # In production, would verify chain
            # This is simplified
            pass
        
        return True


class ComplianceChecker:
    """
    Verifica compliance in tempo reale
    """
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.violation_rules: Dict[str, Dict] = {}
    
    def add_rule(self, rule_id: str, rule: Dict):
        """Aggiunge una regola di compliance"""
        self.violation_rules[rule_id] = rule
    
    def check_trading_limits(self, user_id: str, order_value: float,
                           daily_volume: float, max_daily_volume: float) -> bool:
        """Verifica limiti di trading"""
        if daily_volume + order_value > max_daily_volume:
            self.audit_logger.log_compliance_violation(
                user_id=user_id,
                violation_type="daily_volume_exceeded",
                details={
                    "order_value": order_value,
                    "daily_volume": daily_volume,
                    "max_allowed": max_daily_volume
                },
                severity=RiskLevel.HIGH
            )
            return False
        return True
    
    def check_position_limits(self, user_id: str, position_size: float,
                            max_position: float) -> bool:
        """Verifica limiti di posizione"""
        if position_size > max_position:
            self.audit_logger.log_compliance_violation(
                user_id=user_id,
                violation_type="position_size_exceeded",
                details={
                    "position_size": position_size,
                    "max_allowed": max_position
                },
                severity=RiskLevel.HIGH
            )
            return False
        return True
    
    def check_withdrawal_limits(self, user_id: str, amount: float,
                               balance: float, max_withdrawal_pct: float = 0.9) -> bool:
        """Verifica limiti di prelievo"""
        max_withdrawal = balance * max_withdrawal_pct
        if amount > max_withdrawal:
            self.audit_logger.log_compliance_violation(
                user_id=user_id,
                violation_type="withdrawal_limit_exceeded",
                details={
                    "amount": amount,
                    "balance": balance,
                    "max_allowed": max_withdrawal
                },
                severity=RiskLevel.MEDIUM
            )
            return False
        return True
    
    def check_kyc_status(self, user_id: str, kyc_approved: bool,
                        requires_kyc: bool = True) -> bool:
        """Verifica stato KYC"""
        if requires_kyc and not kyc_approved:
            self.audit_logger.log_compliance_violation(
                user_id=user_id,
                violation_type="kyc_required",
                details={"kyc_approved": kyc_approved},
                severity=RiskLevel.HIGH
            )
            return False
        return True
    
    def check_trading_hours(self) -> bool:
        """Verifica se il trading è permesso (orario)"""
        # Simplified - in production would check more rules
        return True
