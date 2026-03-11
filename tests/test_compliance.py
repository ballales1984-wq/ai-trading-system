"""
Tests for Compliance Module
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from app.compliance.audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditFilter,
    ComplianceChecker,
    RiskLevel,
    ComplianceStatus
)

from app.compliance.reporting import (
    ComplianceReporter,
    ReportType,
    ReportFormat,
    RegulationType
)

from app.compliance.alerts import (
    AlertManager,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertCategory,
    AlertStatus,
    get_default_rules
)


class TestAuditLogger:
    """Test AuditLogger"""
    
    def test_audit_logger_creation(self):
        """Test audit logger creation"""
        logger = AuditLogger()
        
        assert logger.events == []
        assert logger.stats["total_events"] == 0
    
    def test_log_event(self):
        """Test logging an event"""
        logger = AuditLogger()
        
        event = AuditEvent(
            event_type=AuditEventType.LOGIN,
            user_id="user1",
            action="User logged in"
        )
        
        event_id = logger.log_event(event)
        
        assert event_id is not None
        assert len(logger.events) == 1
        assert logger.stats["total_events"] == 1
    
    def test_log_login_success(self):
        """Test logging successful login"""
        logger = AuditLogger()
        
        event_id = logger.log_login(
            user_id="user1",
            username="john",
            ip_address="192.168.1.1",
            success=True
        )
        
        assert event_id is not None
        assert len(logger.events) == 1
        assert logger.events[0].event_type == AuditEventType.LOGIN
    
    def test_log_login_failure(self):
        """Test logging failed login"""
        logger = AuditLogger()
        
        event_id = logger.log_login(
            user_id="user1",
            username="john",
            ip_address="192.168.1.1",
            success=False
        )
        
        assert event_id is not None
        assert logger.events[0].event_type == AuditEventType.LOGIN_FAILED
        assert logger.events[0].compliance_status == ComplianceStatus.WARNING
    
    def test_log_compliance_violation(self):
        """Test logging compliance violation"""
        logger = AuditLogger()
        
        event_id = logger.log_compliance_violation(
            user_id="user1",
            violation_type="kyc_missing",
            details={"required": True},
            severity=RiskLevel.HIGH
        )
        
        assert event_id is not None
        assert logger.stats["violations"] == 1
        assert logger.events[0].compliance_status == ComplianceStatus.VIOLATION
    
    def test_query_events_by_type(self):
        """Test querying events by type"""
        logger = AuditLogger()
        
        # Add some events
        logger.log_event(AuditEvent(event_type=AuditEventType.LOGIN, user_id="u1"))
        logger.log_event(AuditEvent(event_type=AuditEventType.ORDER_CREATED, user_id="u1"))
        logger.log_event(AuditEvent(event_type=AuditEventType.LOGIN, user_id="u2"))
        
        # Query
        filter = AuditFilter(event_types=[AuditEventType.LOGIN])
        results = logger.query_events(filter)
        
        assert len(results) == 2
    
    def test_query_events_by_user(self):
        """Test querying events by user"""
        logger = AuditLogger()
        
        logger.log_event(AuditEvent(event_type=AuditEventType.LOGIN, user_id="u1"))
        logger.log_event(AuditEvent(event_type=AuditEventType.ORDER_CREATED, user_id="u1"))
        logger.log_event(AuditEvent(event_type=AuditEventType.LOGIN, user_id="u2"))
        
        filter = AuditFilter(user_ids=["u1"])
        results = logger.query_events(filter)
        
        assert len(results) == 2
    
    def test_get_user_activity(self):
        """Test getting user activity"""
        logger = AuditLogger()
        
        logger.log_event(AuditEvent(event_type=AuditEventType.LOGIN, user_id="u1"))
        logger.log_event(AuditEvent(event_type=AuditEventType.ORDER_CREATED, user_id="u1"))
        
        activity = logger.get_user_activity("u1")
        
        assert len(activity) == 2
    
    def test_get_stats(self):
        """Test getting statistics"""
        logger = AuditLogger()
        
        logger.log_event(AuditEvent(event_type=AuditEventType.LOGIN, user_id="u1"))
        
        stats = logger.get_stats()
        
        assert "total_events" in stats
        assert stats["total_events"] == 1


class TestComplianceChecker:
    """Test ComplianceChecker"""
    
    def test_check_trading_limits_pass(self):
        """Test trading limits check passes"""
        logger = AuditLogger()
        checker = ComplianceChecker(logger)
        
        result = checker.check_trading_limits(
            user_id="user1",
            order_value=1000,
            daily_volume=1000,
            max_daily_volume=10000
        )
        
        assert result is True
    
    def test_check_trading_limits_fail(self):
        """Test trading limits check fails"""
        logger = AuditLogger()
        checker = ComplianceChecker(logger)
        
        result = checker.check_trading_limits(
            user_id="user1",
            order_value=5000,
            daily_volume=6000,
            max_daily_volume=10000
        )
        
        assert result is False
        assert logger.stats["violations"] == 1
    
    def test_check_position_limits(self):
        """Test position limits check"""
        logger = AuditLogger()
        checker = ComplianceChecker(logger)
        
        result = checker.check_position_limits(
            user_id="user1",
            position_size=150000,
            max_position=100000
        )
        
        assert result is False
    
    def test_check_kyc_status(self):
        """Test KYC status check"""
        logger = AuditLogger()
        checker = ComplianceChecker(logger)
        
        # Without KYC
        result = checker.check_kyc_status(
            user_id="user1",
            kyc_approved=False,
            requires_kyc=True
        )
        
        assert result is False
        
        # With KYC
        result = checker.check_kyc_status(
            user_id="user1",
            kyc_approved=True,
            requires_kyc=True
        )
        
        assert result is True


class TestComplianceReporter:
    """Test ComplianceReporter"""
    
    def test_generate_trade_report(self):
        """Test generating trade report"""
        reporter = ComplianceReporter()
        
        trades = [
            {"timestamp": datetime.now(), "side": "buy", "volume": 1.0, "value": 50000, "fee": 10, "asset_type": "crypto"},
            {"timestamp": datetime.now(), "side": "sell", "volume": 0.5, "value": 25000, "fee": 5, "asset_type": "crypto"}
        ]
        
        report = reporter.generate_trade_report(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now(),
            trades=trades
        )
        
        assert report.total_trades == 2
        assert report.buy_trades == 1
        assert report.sell_trades == 1
        assert report.total_volume == 1.5
    
    def test_generate_position_report(self):
        """Test generating position report"""
        reporter = ComplianceReporter()
        
        positions = [
            {"side": "long", "value": 50000, "unrealized_pnl": 1000, "asset_type": "crypto"},
            {"side": "short", "value": 25000, "unrealized_pnl": -500, "asset_type": "crypto"}
        ]
        
        report = reporter.generate_position_report(positions)
        
        assert report.total_positions == 2
        assert report.long_positions == 1
        assert report.short_positions == 1
    
    def test_generate_risk_report(self):
        """Test generating risk report"""
        reporter = ComplianceReporter()
        
        positions = [
            {"value": 50000},
            {"value": 25000}
        ]
        
        report = reporter.generate_risk_report(
            positions=positions,
            violations=2,
            alerts=5
        )
        
        assert report.violations == 2
        assert report.alerts_triggered == 5
    
    def test_export_json(self):
        """Test exporting report as JSON"""
        reporter = ComplianceReporter()
        
        data = {"test": "data"}
        result = reporter.export_report(data, ReportFormat.JSON)
        
        assert "test" in result
        assert "data" in result


class TestAlertManager:
    """Test AlertManager"""
    
    def test_alert_manager_creation(self):
        """Test alert manager creation"""
        manager = AlertManager()
        
        assert manager.alerts == []
        assert manager.rules == {}
    
    def test_create_alert(self):
        """Test creating an alert"""
        manager = AlertManager()
        
        alert = manager.create_alert(
            category=AlertCategory.TRADING,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test"
        )
        
        assert alert.id is not None
        assert alert.status == AlertStatus.ACTIVE
        assert len(manager.alerts) == 1
    
    def test_create_trading_alert(self):
        """Test creating trading alert"""
        manager = AlertManager()
        
        alert = manager.create_trading_alert(
            user_id="user1",
            alert_type="volume_spike",
            details={"volume": 2.5}
        )
        
        assert alert.category == AlertCategory.TRADING
        assert alert.user_id == "user1"
    
    def test_create_risk_alert(self):
        """Test creating risk alert"""
        manager = AlertManager()
        
        alert = manager.create_risk_alert(
            user_id="user1",
            alert_type="var_breach",
            details={"var": 0.06, "exceeded": True}
        )
        
        assert alert.category == AlertCategory.RISK
        assert alert.severity == AlertSeverity.CRITICAL
    
    def test_create_security_alert(self):
        """Test creating security alert"""
        manager = AlertManager()
        
        alert = manager.create_security_alert(
            alert_type="failed_login",
            details={"ip": "192.168.1.100"}
        )
        
        assert alert.category == AlertCategory.SECURITY
        assert alert.severity == AlertSeverity.ERROR
    
    def test_acknowledge_alert(self):
        """Test acknowledging an alert"""
        manager = AlertManager()
        
        alert = manager.create_alert(
            category=AlertCategory.TRADING,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test"
        )
        
        result = manager.acknowledge_alert(alert.id, "admin")
        
        assert result is True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "admin"
    
    def test_resolve_alert(self):
        """Test resolving an alert"""
        manager = AlertManager()
        
        alert = manager.create_alert(
            category=AlertCategory.TRADING,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test"
        )
        
        result = manager.resolve_alert(alert.id, "admin", "Issue resolved")
        
        assert result is True
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolution_notes == "Issue resolved"
    
    def test_get_active_alerts(self):
        """Test getting active alerts"""
        manager = AlertManager()
        
        manager.create_alert(AlertCategory.TRADING, AlertSeverity.WARNING, "Alert 1", "Msg")
        manager.create_alert(AlertCategory.RISK, AlertSeverity.ERROR, "Alert 2", "Msg")
        
        # Resolve one
        manager.alerts[0].status = AlertStatus.RESOLVED
        
        active = manager.get_active_alerts()
        
        assert len(active) == 1
        assert active[0].category == AlertCategory.RISK
    
    def test_get_alert_summary(self):
        """Test getting alert summary"""
        manager = AlertManager()
        
        manager.create_alert(AlertCategory.TRADING, AlertSeverity.WARNING, "Alert 1", "Msg")
        manager.create_alert(AlertCategory.RISK, AlertSeverity.ERROR, "Alert 2", "Msg")
        
        summary = manager.get_alert_summary()
        
        assert "active" in summary
        assert summary["active"] == 2
    
    def test_default_rules(self):
        """Test getting default rules"""
        rules = get_default_rules()
        
        assert len(rules) > 0
        assert all(isinstance(r, AlertRule) for r in rules)


class TestAlert:
    """Test Alert dataclass"""
    
    def test_alert_to_dict(self):
        """Test alert serialization"""
        alert = Alert(
            id="test123",
            category=AlertCategory.TRADING,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message"
        )
        
        data = alert.to_dict()
        
        assert data["id"] == "test123"
        assert data["category"] == "trading"
        assert data["severity"] == "warning"
        assert data["title"] == "Test Alert"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
