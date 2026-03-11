"""
Compliance Alerts Module - AI Trading System
Sistema di alert automatici per compliance
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json


class AlertSeverity(Enum):
    """Severità alert"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Categoria alert"""
    TRADING = "trading"
    RISK = "risk"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    OPERATIONAL = "operational"
    KYC = "kyc"
    AML = "aml"


class AlertStatus(Enum):
    """Stato alert"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


@dataclass
class AlertRule:
    """Regola per generazione alert"""
    id: str
    name: str
    category: AlertCategory
    condition: str
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 60
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """Alert di compliance"""
    id: str
    timestamp: datetime = field(default_factory=datetime.now)
    rule_id: Optional[str] = None
    category: AlertCategory = AlertCategory.COMPLIANCE
    severity: AlertSeverity = AlertSeverity.WARNING
    status: AlertStatus = AlertStatus.ACTIVE
    title: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "rule_id": self.rule_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes
        }


class AlertManager:
    """
    Sistema di gestione alert
    """
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.rules: Dict[str, AlertRule] = {}
        self.notification_handlers: Dict[str, Callable] = {}
        self.alert_stats = {
            "total": 0,
            "active": 0,
            "acknowledged": 0,
            "resolved": 0,
            "by_severity": {},
            "by_category": {}
        }
    
    def add_rule(self, rule: AlertRule):
        """Aggiunge una regola di alert"""
        self.rules[rule.id] = rule
    
    def remove_rule(self, rule_id: str):
        """Rimuove una regola"""
        if rule_id in self.rules:
            del self.rules[rule_id]
    
    def enable_rule(self, rule_id: str):
        """Abilita una regola"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
    
    def disable_rule(self, rule_id: str):
        """Disabilita una regola"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
    
    def create_alert(self,
                    category: AlertCategory,
                    severity: AlertSeverity,
                    title: str,
                    message: str,
                    details: Optional[Dict] = None,
                    user_id: Optional[str] = None,
                    rule_id: Optional[str] = None) -> Alert:
        """Crea un nuovo alert"""
        alert = Alert(
            id=f"alert_{datetime.now().timestamp()}",
            category=category,
            severity=severity,
            title=title,
            message=message,
            details=details or {},
            user_id=user_id,
            rule_id=rule_id
        )
        
        self.alerts.append(alert)
        self._update_stats(alert)
        
        # Invia notifiche
        self._send_notifications(alert)
        
        return alert
    
    def create_trading_alert(self,
                           user_id: str,
                           alert_type: str,
                           details: Dict) -> Alert:
        """Crea alert trading"""
        severity = self._determine_severity(alert_type, details)
        
        messages = {
            "volume_spike": f"Unusual trading volume detected: {details.get('volume', 0)}",
            "price_move": f"Large price movement: {details.get('change_pct', 0)}%",
            "order_ratelimit": f"Rate limit exceeded: {details.get('count', 0)} orders",
            "position_limit": f"Position limit approaching: {details.get('utilization', 0)}%"
        }
        
        return self.create_alert(
            category=AlertCategory.TRADING,
            severity=severity,
            title=f"Trading Alert: {alert_type}",
            message=messages.get(alert_type, f"Trading alert: {alert_type}"),
            details=details,
            user_id=user_id
        )
    
    def create_risk_alert(self,
                         user_id: str,
                         alert_type: str,
                         details: Dict) -> Alert:
        """Crea alert rischio"""
        severity = AlertSeverity.CRITICAL if details.get("exceeded", False) else AlertSeverity.WARNING
        
        messages = {
            "var_breach": f"VaR breach: {details.get('var', 0)} exceeds limit",
            "drawdown": f"Drawdown alert: {details.get('drawdown', 0)}%",
            "leverage": f"Leverage warning: {details.get('leverage', 0)}x",
            "margin_call": "Margin call triggered"
        }
        
        return self.create_alert(
            category=AlertCategory.RISK,
            severity=severity,
            title=f"Risk Alert: {alert_type}",
            message=messages.get(alert_type, f"Risk alert: {alert_type}"),
            details=details,
            user_id=user_id
        )
    
    def create_security_alert(self,
                            alert_type: str,
                            details: Dict,
                            user_id: Optional[str] = None) -> Alert:
        """Crea alert sicurezza"""
        messages = {
            "failed_login": f"Failed login attempt from {details.get('ip', 'unknown')}",
            "suspicious_ip": f"Suspicious IP detected: {details.get('ip', 'unknown')}",
            "api_key_compromised": "API key may be compromised",
            "unusual_activity": "Unusual account activity detected"
        }
        
        return self.create_alert(
            category=AlertCategory.SECURITY,
            severity=AlertSeverity.ERROR,
            title=f"Security Alert: {alert_type}",
            message=messages.get(alert_type, f"Security alert: {alert_type}"),
            details=details,
            user_id=user_id
        )
    
    def create_compliance_alert(self,
                               alert_type: str,
                               details: Dict,
                               user_id: Optional[str] = None) -> Alert:
        """Crea alert compliance"""
        messages = {
            "kyc_expiring": f"KYC document expiring in {details.get('days', 0)} days",
            "trading_limit": f"Trading limit exceeded: {details.get('utilization', 0)}%",
            "report_missing": f"Required report missing: {details.get('report_type', 'unknown')}",
            "audit_failure": "Audit trail verification failed"
        }
        
        return self.create_alert(
            category=AlertCategory.COMPLIANCE,
            severity=AlertSeverity.WARNING,
            title=f"Compliance Alert: {alert_type}",
            message=messages.get(alert_type, f"Compliance alert: {alert_type}"),
            details=details,
            user_id=user_id
        )
    
    def _determine_severity(self, alert_type: str, details: Dict) -> AlertSeverity:
        """Determina severità basata su tipo e dettagli"""
        if details.get("exceeded"):
            return AlertSeverity.ERROR
        
        thresholds = {
            "volume_spike": 2.0,  # 2x normal
            "price_move": 10.0,   # 10%
            "order_ratelimit": 1.0
        }
        
        threshold = thresholds.get(alert_type, 1.0)
        value = abs(details.get("value", 0))
        
        if value > threshold * 2:
            return AlertSeverity.CRITICAL
        elif value > threshold:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def acknowledge_alert(self, alert_id: str, user_id: str, notes: Optional[str] = None) -> bool:
        """Ackowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = user_id
                alert.acknowledged_at = datetime.now()
                self.alert_stats["acknowledged"] += 1
                self.alert_stats["active"] -= 1
                return True
        return False
    
    def resolve_alert(self, alert_id: str, user_id: str, resolution_notes: str) -> bool:
        """Risolvi un alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_by = user_id
                alert.resolved_at = datetime.now()
                alert.resolution_notes = resolution_notes
                self.alert_stats["resolved"] += 1
                if alert.status == AlertStatus.ACTIVE:
                    self.alert_stats["active"] -= 1
                return True
        return False
    
    def dismiss_alert(self, alert_id: str, user_id: str) -> bool:
        """Dismiss an alert"""
        for alert in self.alerts:
            if alert.id == alert_id and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.DISMISSED
                self.alert_stats["active"] -= 1
                return True
        return False
    
    def get_active_alerts(self,
                         category: Optional[AlertCategory] = None,
                         severity: Optional[AlertSeverity] = None,
                         user_id: Optional[str] = None) -> List[Alert]:
        """Get active alerts with filters"""
        results = []
        
        for alert in self.alerts:
            if alert.status != AlertStatus.ACTIVE:
                continue
            
            if category and alert.category != category:
                continue
            
            if severity and alert.severity != severity:
                continue
            
            if user_id and alert.user_id != user_id:
                continue
            
            results.append(alert)
        
        return sorted(results, key=lambda a: (a.severity.value, a.timestamp), 
                     reverse=True)
    
    def get_alert_summary(self) -> Dict:
        """Get alert summary"""
        return {
            "total": self.alert_stats["total"],
            "active": len([a for a in self.alerts if a.status == AlertStatus.ACTIVE]),
            "acknowledged": len([a for a in self.alerts if a.status == AlertStatus.ACKNOWLEDGED]),
            "resolved": len([a for a in self.alerts if a.status == AlertStatus.RESOLVED]),
            "critical_count": len([a for a in self.alerts 
                                  if a.severity == AlertSeverity.CRITICAL 
                                  and a.status == AlertStatus.ACTIVE]),
            "by_category": self._count_by_field("category"),
            "by_severity": self._count_by_field("severity")
        }
    
    def _count_by_field(self, field: str) -> Dict:
        """Count alerts by field"""
        counts = {}
        for alert in self.alerts:
            if field == "category":
                key = alert.category.value
            elif field == "severity":
                key = alert.severity.value
            else:
                continue
            
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    def _update_stats(self, alert: Alert):
        """Update alert statistics"""
        self.alert_stats["total"] += 1
        
        if alert.status == AlertStatus.ACTIVE:
            self.alert_stats["active"] += 1
        
        cat = alert.category.value
        self.alert_stats["by_category"][cat] = self.alert_stats["by_category"].get(cat, 0) + 1
        
        sev = alert.severity.value
        self.alert_stats["by_severity"][sev] = self.alert_stats["by_severity"].get(sev, 0) + 1
    
    def register_notification_handler(self, channel: str, handler: Callable):
        """Register a notification handler"""
        self.notification_handlers[channel] = handler
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for alert"""
        for channel, handler in self.notification_handlers.items():
            try:
                handler(alert)
            except Exception as e:
                print(f"Failed to send notification to {channel}: {e}")
    
    def export_alerts(self, filepath: str, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None):
        """Export alerts to JSON"""
        alerts = self.alerts
        
        if start_date:
            alerts = [a for a in alerts if a.timestamp >= start_date]
        if end_date:
            alerts = [a for a in alerts if a.timestamp <= end_date]
        
        data = {
            "export_date": datetime.now().isoformat(),
            "total_alerts": len(alerts),
            "alerts": [a.to_dict() for a in alerts]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Predefined alert rules
def get_default_rules() -> List[AlertRule]:
    """Get default compliance alert rules"""
    return [
        AlertRule(
            id="daily_volume_limit",
            name="Daily Trading Volume Limit",
            category=AlertCategory.TRADING,
            condition="daily_volume > threshold",
            threshold=1000000,
            severity=AlertSeverity.WARNING,
            notification_channels=["email", "dashboard"]
        ),
        AlertRule(
            id="position_size_limit",
            name="Position Size Limit",
            category=AlertCategory.RISK,
            condition="position_size > threshold",
            threshold=100000,
            severity=AlertSeverity.ERROR,
            notification_channels=["email", "sms"]
        ),
        AlertRule(
            id="kyc_expiry",
            name="KYC Document Expiry",
            category=AlertCategory.KYC,
            condition="days_to_expiry <= 30",
            threshold=30,
            severity=AlertSeverity.WARNING,
            notification_channels=["email"]
        ),
        AlertRule(
            id="failed_logins",
            name="Failed Login Attempts",
            category=AlertCategory.SECURITY,
            condition="failed_count > 5",
            threshold=5,
            severity=AlertSeverity.ERROR,
            notification_channels=["email", "sms"]
        ),
        AlertRule(
            id="var_breach",
            name="Value at Risk Breach",
            category=AlertCategory.RISK,
            condition="var > limit",
            threshold=0.05,
            severity=AlertSeverity.CRITICAL,
            notification_channels=["email", "sms", "dashboard"]
        )
    ]
