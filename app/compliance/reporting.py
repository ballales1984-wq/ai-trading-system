"""
Compliance Reporting Module - AI Trading System
Report automatici per regulatory compliance
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import csv
from io import StringIO


class ReportType(Enum):
    """Tipi di report"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    AD_HOC = "ad_hoc"


class ReportFormat(Enum):
    """Formati di report"""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    HTML = "html"


class RegulationType(Enum):
    """Tipi di regolamentazione"""
    SEC = "sec"           # SEC (US)
    MIFID_II = "mifid_ii" # MiFID II (EU)
    GDPR = "gdpr"         # GDPR (EU)
    AML = "aml"           # Anti-Money Laundering
    KYC = "kyc"           # Know Your Customer


@dataclass
class ReportConfig:
    """Configurazione report"""
    report_type: ReportType = ReportType.DAILY
    format: ReportFormat = ReportFormat.JSON
    regulations: List[RegulationType] = field(default_factory=list)
    include_trades: bool = True
    include_positions: bool = True
    include_fees: bool = True
    include_risk: bool = True
    include_audit: bool = True


@dataclass
class TradeReport:
    """Report di trading"""
    date: datetime
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    total_volume: float = 0.0
    total_value: float = 0.0
    fees_paid: float = 0.0
    by_asset_type: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "total_trades": self.total_trades,
            "buy_trades": self.buy_trades,
            "sell_trades": self.sell_trades,
            "total_volume": self.total_volume,
            "total_value": self.total_value,
            "fees_paid": self.fees_paid,
            "by_asset_type": self.by_asset_type
        }


@dataclass
class PositionReport:
    """Report posizioni"""
    date: datetime
    total_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    total_exposure: float = 0.0
    total_value: float = 0.0
    unrealized_pnl: float = 0.0
    by_asset_type: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "total_positions": self.total_positions,
            "long_positions": self.long_positions,
            "short_positions": self.short_positions,
            "total_exposure": self.total_exposure,
            "total_value": self.total_value,
            "unrealized_pnl": self.unrealized_pnl,
            "by_asset_type": self.by_asset_type
        }


@dataclass
class RiskReport:
    """Report rischi"""
    date: datetime
    var_95: float = 0.0
    var_99: float = 0.0
    max_drawdown: float = 0.0
    leverage_used: float = 0.0
    risk_score: float = 0.0
    violations: int = 0
    alerts_triggered: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "var_95": self.var_95,
            "var_99": self.var_99,
            "max_drawdown": self.max_drawdown,
            "leverage_used": self.leverage_used,
            "risk_score": self.risk_score,
            "violations": self.violations,
            "alerts_triggered": self.alerts_triggered
        }


class ComplianceReporter:
    """
    Genera report di compliance automatici
    """
    
    def __init__(self):
        self.report_templates: Dict[str, Dict] = {}
        self.scheduled_reports: List[Dict] = []
    
    def generate_trade_report(self, 
                            start_date: datetime,
                            end_date: datetime,
                            trades: List[Dict]) -> TradeReport:
        """Genera report di trading"""
        report = TradeReport(date=end_date)
        
        for trade in trades:
            trade_date = trade.get("timestamp")
            if isinstance(trade_date, str):
                trade_date = datetime.fromisoformat(trade_date)
            
            if start_date <= trade_date <= end_date:
                report.total_trades += 1
                report.total_volume += trade.get("volume", 0)
                report.total_value += trade.get("value", 0)
                report.fees_paid += trade.get("fee", 0)
                
                if trade.get("side") == "buy":
                    report.buy_trades += 1
                else:
                    report.sell_trades += 1
                
                asset_type = trade.get("asset_type", "unknown")
                report.by_asset_type[asset_type] = report.by_asset_type.get(asset_type, 0) + 1
        
        return report
    
    def generate_position_report(self, 
                               positions: List[Dict]) -> PositionReport:
        """Genera report posizioni"""
        report = PositionReport(date=datetime.now())
        
        for pos in positions:
            report.total_positions += 1
            if pos.get("side") == "long":
                report.long_positions += 1
            else:
                report.short_positions += 1
            
            report.total_exposure += abs(pos.get("value", 0))
            report.total_value += pos.get("value", 0)
            report.unrealized_pnl += pos.get("unrealized_pnl", 0)
            
            asset_type = pos.get("asset_type", "unknown")
            report.by_asset_type[asset_type] = report.by_asset_type.get(asset_type, 0) + pos.get("value", 0)
        
        return report
    
    def generate_risk_report(self, 
                           positions: List[Dict],
                           violations: int,
                           alerts: int) -> RiskReport:
        """Genera report rischi"""
        report = RiskReport(
            date=datetime.now(),
            violations=violations,
            alerts_triggered=alerts
        )
        
        # Calcola exposure totale
        total_exposure = sum(abs(p.get("value", 0)) for p in positions)
        total_value = sum(p.get("value", 0) for p in positions)
        
        if total_value > 0:
            report.leverage_used = abs(total_exposure / total_value)
        
        # Simplified VaR calculation
        report.var_95 = total_exposure * 0.02  # Simplified
        report.var_99 = total_exposure * 0.05  # Simplified
        
        # Risk score (simplified)
        report.risk_score = min(100, (report.leverage_used * 20) + (violations * 10))
        
        return report
    
    def generate_sec_form_cf(self, 
                           trades: List[Dict],
                           positions: List[Dict]) -> Dict:
        """
        Genera SEC Form CF (Combined Fee Report)
        Simplified per trading crypto
        """
        return {
            "form_type": "SEC Form CF",
            "filing_date": datetime.now().isoformat(),
            "period_start": (datetime.now() - timedelta(days=30)).isoformat(),
            "period_end": datetime.now().isoformat(),
            "transaction_summary": {
                "total_transactions": len(trades),
                "total_volume": sum(t.get("volume", 0) for t in trades),
                "total_fees": sum(t.get("fee", 0) for t in trades)
            },
            "current_holdings": {
                "total_positions": len(positions),
                "total_value": sum(p.get("value", 0) for p in positions)
            },
            "compliance_statement": "This report has been prepared in accordance with SEC requirements."
        }
    
    def generate_mifid_ii_report(self,
                                trades: List[Dict],
                                positions: List[Dict],
                                risk_metrics: Dict) -> Dict:
        """
        Genera MiFID II Transaction Report
        Simplified per trading crypto
        """
        return {
            "report_type": "MiFID II Transaction Report",
            "reporting_entity": "AI Trading System",
            "report_date": datetime.now().isoformat(),
            "reporting_period": {
                "start": (datetime.now() - timedelta(days=30)).isoformat(),
                "end": datetime.now().isoformat()
            },
            "transactions": [
                {
                    "transaction_id": t.get("id"),
                    "trading_date": t.get("timestamp"),
                    "instrument": t.get("symbol"),
                    "venue": t.get("exchange", "CRYPTO_EXCHANGE"),
                    "transaction_type": t.get("side"),
                    "quantity": t.get("volume"),
                    "price": t.get("price"),
                    "value": t.get("value")
                } for t in trades
            ],
            "positions_summary": {
                "total_long": sum(1 for p in positions if p.get("side") == "long"),
                "total_short": sum(1 for p in positions if p.get("side") == "short")
            },
            "risk_metrics": risk_metrics
        }
    
    def generate_aml_report(self,
                          transactions: List[Dict],
                          alerts: List[Dict]) -> Dict:
        """
        Genera Anti-Money Laundering Report
        """
        return {
            "report_type": "AML Suspicious Activity Report",
            "report_date": datetime.now().isoformat(),
            "period": {
                "start": (datetime.now() - timedelta(days=30)).isoformat(),
                "end": datetime.now().isoformat()
            },
            "summary": {
                "total_transactions": len(transactions),
                "total_volume": sum(t.get("value", 0) for t in transactions),
                "suspicious_activity_flags": len(alerts),
                "transactions_over_10k": len([t for t in transactions if t.get("value", 0) >= 10000])
            },
            "suspicious_activities": alerts,
            "recommendations": self._generate_aml_recommendations(alerts)
        }
    
    def _generate_aml_recommendations(self, alerts: List[Dict]) -> List[str]:
        """Genera raccomandazioni AML"""
        recommendations = []
        
        if len(alerts) > 10:
            recommendations.append("Review all flagged transactions immediately")
        if len(alerts) > 5:
            recommendations.append("Consider filing Suspicious Activity Report (SAR)")
        
        recommendations.append("Continue monitoring high-volume accounts")
        recommendations.append("Verify source of funds for large transactions")
        
        return recommendations
    
    def generate_kyc_compliance_report(self,
                                     users: List[Dict]) -> Dict:
        """
        Genera KYC Compliance Report
        """
        approved = sum(1 for u in users if u.get("kyc_approved", False))
        pending = sum(1 for u in users if u.get("kyc_status") == "pending")
        rejected = sum(1 for u in users if u.get("kyc_status") == "rejected")
        
        return {
            "report_type": "KYC Compliance Report",
            "report_date": datetime.now().isoformat(),
            "summary": {
                "total_users": len(users),
                "approved": approved,
                "pending": pending,
                "rejected": rejected,
                "compliance_rate": (approved / len(users) * 100) if users else 0
            },
            "expiration_alerts": self._get_kyc_expirations(users),
            "recommendations": self._get_kyc_recommendations(approved, pending, rejected)
        }
    
    def _get_kyc_expirations(self, users: List[Dict]) -> List[Dict]:
        """Get KYC documents expiring soon"""
        expirations = []
        for user in users:
            exp_date = user.get("kyc_expiration")
            if exp_date:
                if isinstance(exp_date, str):
                    exp_date = datetime.fromisoformat(exp_date)
                if exp_date <= datetime.now() + timedelta(days=30):
                    expirations.append({
                        "user_id": user.get("id"),
                        "expiration_date": exp_date.isoformat()
                    })
        return expirations
    
    def _get_kyc_recommendations(self, approved: int, pending: int, rejected: int) -> List[str]:
        """Genera raccomandazioni KYC"""
        recommendations = []
        
        if pending > approved * 0.1:
            recommendations.append("Review pending KYC applications")
        
        if rejected > approved * 0.05:
            recommendations.append("Review rejected applications for patterns")
        
        recommendations.append("Continue periodic re-verification")
        
        return recommendations
    
    def export_report(self, report_data: Dict, format: ReportFormat) -> str:
        """Esporta report nel formato specificato"""
        if format == ReportFormat.JSON:
            return json.dumps(report_data, indent=2)
        elif format == ReportFormat.CSV:
            return self._to_csv(report_data)
        elif format == ReportFormat.HTML:
            return self._to_html(report_data)
        else:
            return str(report_data)
    
    def _to_csv(self, data: Dict) -> str:
        """Converti a CSV"""
        output = StringIO()
        
        # Simplified - flatten dict to CSV
        if "transactions" in data:
            writer = csv.DictWriter(output, fieldnames=data["transactions"][0].keys() if data["transactions"] else [])
            writer.writeheader()
            writer.writerows(data["transactions"])
        
        return output.getvalue()
    
    def _to_html(self, data: Dict) -> str:
        """Converti a HTML"""
        html = f"""
        <html>
        <head><title>Compliance Report</title></head>
        <body>
            <h1>Compliance Report</h1>
            <pre>{json.dumps(data, indent=2)}</pre>
        </body>
        </html>
        """
        return html
    
    def schedule_report(self, config: ReportConfig, cron_expression: str):
        """Schedula un report ricorrente"""
        self.scheduled_reports.append({
            "config": config,
            "cron": cron_expression,
            "next_run": self._calculate_next_run(cron_expression)
        })
    
    def _calculate_next_run(self, cron: str) -> datetime:
        """Calcola prossima esecuzione (simplified)"""
        # Simplified - in production would parse cron
        return datetime.now() + timedelta(days=1)
