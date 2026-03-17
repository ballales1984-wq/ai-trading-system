"""
Account Performance Report Module
=================================
Genera report testuali dettagliati sull'andamento del conto di trading.
Analizza metriche e produce spiegazioni in linguaggio naturale.

Autore: AI Trading System
Data: 2026-03-17
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Metriche di performance del conto"""
    equity: float
    initial_equity: float
    pnl: float
    pnl_percent: float
    winrate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    volatility: float
    risk_score: float  # 0-100
    open_positions: int
    positions_value: float
    cash_balance: float


@dataclass
class ReportSection:
    """Sezione del report"""
    title: str
    content: str
    severity: str = "normal"  # normal, warning, critical, success


class AccountReportGenerator:
    """
    Generatore di report testuali per l'andamento del conto.
    
    Analizza le metriche di trading e produce spiegazioni in linguaggio naturale
    con identificazione delle criticità e raccomandazioni.
    """
    
    def __init__(self):
        self.sections: List[ReportSection] = []
    
    def generate_report(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """
        Genera un report completo dell'andamento del conto.
        
        Args:
            metrics: Oggetto PerformanceMetrics con i dati del conto
            
        Returns:
            Dict contenente il report strutturato
        """
        self.sections = []
        
        # Genera tutte le sezioni del report
        self._add_executive_summary(metrics)
        self._add_performance_analysis(metrics)
        self._add_risk_analysis(metrics)
        self._add_trading_analysis(metrics)
        self._add_positions_analysis(metrics)
        self._add_criticities(metrics)
        self._add_recommendations(metrics)
        
        # Compila il report finale
        report = {
            "generated_at": datetime.now().isoformat(),
            "title": f"📊 Report Andamento Conto - {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            "metrics": {
                "equity": f"{metrics.equity:,.2f} USDT",
                "pnl": f"{metrics.pnl:+,.2f} USDT ({metrics.pnl_percent:+.2f}%)",
                "winrate": f"{metrics.winrate:.1f}%",
                "total_trades": metrics.total_trades,
                "risk_score": f"{metrics.risk_score:.0f}/100"
            },
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "severity": s.severity
                }
                for s in self.sections
            ]
        }
        
        return report
    
    def _add_executive_summary(self, m: PerformanceMetrics):
        """Genera il sommario esecutivo"""
        # Determina lo stato generale
        if m.pnl_percent > 10:
            status = "🟢 ECCELLENTE"
            status_text = "Il conto sta performando in modo eccezionale"
        elif m.pnl_percent > 5:
            status = "🟢 OTTIMO"
            status_text = "Il conto ha una performance positiva solida"
        elif m.pnl_percent > 0:
            status = "🟡 POSITIVO"
            status_text = "Il conto è in profitto ma con margini ridotti"
        elif m.pnl_percent > -5:
            status = "🟠 ATTENZIONE"
            status_text = "Il conto è in leggera perdita"
        elif m.pnl_percent > -15:
            status = "🔴 CRITICO"
            status_text = "Il conto ha subito perdite significative"
        else:
            status = "🔴 EMERGENZA"
            status_text = "Il conto richiede intervento immediato"
        
        content = f"""
**Stato Generale: {status}**

{status_text}

• Equità attuale: **{m.equity:,.2f} USDT**
• Profit/Perdita: **{m.pnl:+,.2f} USDT** ({m.pnl_percent:+.2f}%)
• Capitale iniziale: {m.initial_equity:,.2f} USDT

Il conto ha {'guadagnato' if m.pnl > 0 else 'perso'} {abs(m.pnl):,.2f} USDT 
dall'inizio delle operazioni, con un ritorno del {abs(m.pnl_percent):.2f}%.
"""
        self.sections.append(ReportSection(
            title="📋 Sommario Esecutivo",
            content=content.strip(),
            severity="success" if m.pnl > 0 else "warning" if m.pnl > -5 else "critical"
        ))
    
    def _add_performance_analysis(self, m: PerformanceMetrics):
        """Analizza le performance di trading"""
        # Calcolo profit factor
        total_wins = m.winning_trades * m.avg_win
        total_losses = abs(m.losing_trades * m.avg_loss) if m.losing_trades > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Analisi trend
        if m.pnl_percent > 0:
            trend = "📈 UPTREND"
            trend_desc = "Il conto sta registrando profitti consistenti"
        elif m.pnl_percent < -5:
            trend = "📉 DOWNTREND"
            trend_desc = "Il conto sta attraversando una fase negativa"
        else:
            trend = "➡️ LATERALE"
            trend_desc = "Il conto è in una fase di consolidamento"
        
        content = f"""
**Analisi Performance**

**Trend: {trend}**
{trend_desc}

**Metriche Chiave:**
• Profit Factor: {profit_factor:.2f} (obiettivo > 1.5)
• Sharpe Ratio: {m.sharpe_ratio:.2f} (obiettivo > 1.0)
• Volatilità: {m.volatility:.2f}%

**Distribuzione Trade:**
• Trade totali: {m.total_trades}
• Trade vincenti: {m.winning_trades} ({m.winrate:.1f}%)
• Trade perdenti: {m.losing_trades} ({100-m.winrate:.1f}%)

**Media Trade:**
• Vincita media: +{m.avg_win:.2f} USDT
• Perdita media: {m.avg_loss:.2f} USDT
• Rapporto rischio/rendimento: {abs(m.avg_win/m.avg_loss):.2f}:1

**Estremi:**
• Vincita più grande: +{m.largest_win:.2f} USDT
• Perdita più grande: {m.largest_loss:.2f} USDT
"""
        
        # Determina severità
        if profit_factor > 1.5 and m.sharpe_ratio > 1.0:
            severity = "success"
        elif profit_factor > 1.0:
            severity = "normal"
        else:
            severity = "warning"
        
        self.sections.append(ReportSection(
            title="📈 Analisi Performance",
            content=content.strip(),
            severity=severity
        ))
    
    def _add_risk_analysis(self, m: PerformanceMetrics):
        """Analisi dei rischi"""
        # Interpretazione del rischio
        if m.risk_score < 30:
            risk_level = "🟢 BASSO"
            risk_desc = "Il profilo di rischio è conservativo"
        elif m.risk_score < 60:
            risk_level = "🟡 MEDIO"
            risk_desc = "Il profilo di rischio è moderato"
        elif m.risk_score < 80:
            risk_level = "🟠 ALTO"
            risk_desc = "Il profilo di rischio è aggressivo"
        else:
            risk_level = "🔴 MOLTO ALTO"
            risk_desc = "Il profilo di rischio è estremo - attenzione!"
        
        content = f"""
**Analisi Rischio**

**Livello Rischio: {risk_level}** ({m.risk_score:.0f}/100)
{risk_desc}

**Drawdown:**
• Drawdown massimo: {m.max_drawdown:.2f} USDT
• Drawdown percentuale: {m.max_drawdown_percent:.2f}%

**Valutazione Rischio:**
"""
        
        # Aggiungi valutazioni specifiche
        if m.max_drawdown_percent > 20:
            content += "⚠️ Il drawdown supera il 20% - rischio elevato\n"
        if m.volatility > 30:
            content += "⚠️ Alta volatilità - possibili oscillazioni forti\n"
        if m.sharpe_ratio < 1:
            content += "⚠️ Sharpe ratio basso - rendimenti adjusted per rischio insufficienti\n"
        
        content += f"""
**Raccomandazione:**
Il rischio attuale è {'appropriato' if m.risk_score < 60 else 'elevato'} per il capitale investito.
"""
        
        self.sections.append(ReportSection(
            title="⚠️ Analisi Rischio",
            content=content.strip(),
            severity="success" if m.risk_score < 30 else "warning" if m.risk_score < 70 else "critical"
        ))
    
    def _add_trading_analysis(self, m: PerformanceMetrics):
        """Analisi dell'attività di trading"""
        content = f"""
**Analisi Trading**

**Attività:**
• Posizioni aperte: {m.open_positions}
• Valore posizioni: {m.positions_value:,.2f} USDT
• Liquidità disponibile: {m.cash_balance:,.2f} USDT

**Efficienza:**
"""
        
        # Calcolo taxa di successo recente
        if m.total_trades > 0:
            trade_density = m.total_trades / 30  # Trade al giorno stimato
            content += f"• Densità trading: {trade_density:.1f} trade/giorno\n"
        
        if m.winrate >= 55:
            content += "✅ Tasso di vincita superiore alla media\n"
        elif m.winrate >= 50:
            content += "➖ Tasso di vincita nella media\n"
        else:
            content += "❌ Tasso di vincita sotto la media\n"
        
        if abs(m.avg_win/m.avg_loss) >= 2 if m.avg_loss != 0 else False:
            content += "✅ Ottimo rapporto rischio/rendimento\n"
        else:
            content += "⚠️ Migliorare il rapporto rischio/rendimento\n"
        
        self.sections.append(ReportSection(
            title="🎯 Analisi Trading",
            content=content.strip(),
            severity="normal"
        ))
    
    def _add_positions_analysis(self, m: PerformanceMetrics):
        """Analisi delle posizioni correnti"""
        content = f"""
**Posizioni Correnti**

• Numero posizioni aperte: {m.open_positions}
• Valore totale posizioni: {m.positions_value:,.2f} USDT
• Percentuale del portafoglio in posizioni: {(m.positions_value/m.equity*100):.1f}%

**Raccomandazione posizionamento:**
"""
        
        position_ratio = m.positions_value / m.equity if m.equity > 0 else 0
        
        if position_ratio < 0.3:
            content += "💡 Basso utilizzo del capitale - considerare nuove opportunità\n"
        elif position_ratio < 0.6:
            content += "✅ Allocazione bilanciata del capitale\n"
        elif position_ratio < 0.8:
            content += "⚠️ Alta esposizione - valutare riduzione posizioni\n"
        else:
            content += "🔴 Esposizione molto alta - rischio di liquidità!\n"
        
        self.sections.append(ReportSection(
            title="💰 Analisi Posizioni",
            content=content.strip(),
            severity="normal"
        ))
    
    def _add_criticities(self, m: PerformanceMetrics):
        """Identifica e elenca le criticità"""
        criticities = []
        
        # Check criticità
        if m.pnl_percent < -10:
            criticities.append("🔴 Perdita superiore al 10% - richiede azione correttiva")
        
        if m.max_drawdown_percent > 25:
            criticities.append("🔴 Drawdown superiore al 25% - stop loss troppo larghi o nessuna gestione rischio")
        
        if m.winrate < 40:
            criticities.append("🔴 Win rate inferiore al 40% - strategia non efficace")
        
        if m.risk_score > 80:
            criticities.append("🔴 Rischio troppo elevato - ridurre esposizione")
        
        if m.total_trades > 0 and m.avg_loss > m.avg_win * 3:
            criticities.append("🟠 Perdita media troppo grande rispetto alle vincite")
        
        if m.volatility > 50:
            criticities.append("⚠️ Volatilità molto alta - mercato instabile")
        
        content = "**Criticità Identificate:**\n\n"
        
        if criticities:
            content += "\n".join(f"• {c}" for c in criticities)
        else:
            content += "✅ Nessuna criticità critica rilevata\n"
            content += "• Il conto sta performando in modo stabile\n"
            content += "• Le metriche di rischio sono sotto controllo\n"
        
        # Determina severità
        severity = "critical" if any("🔴" in c for c in criticities) else "success" if not criticities else "warning"
        
        self.sections.append(ReportSection(
            title="🚨 Criticità",
            content=content,
            severity=severity
        ))
    
    def _add_recommendations(self, m: PerformanceMetrics):
        """Genera raccomandazioni actionable"""
        recommendations = []
        
        # Raccomandazioni basate sulle metriche
        if m.pnl_percent < 0:
            recommendations.append("1. **Rivedere la strategia**: Il conto è in perdita, analizzare i pattern dei trade perdenti")
        
        if m.winrate < 50:
            recommendations.append("2. **Migliorare entry point**: Il win rate è basso, aspettare conferme più solide")
        
        if m.avg_loss > m.avg_win:
            recommendations.append("3. **Strettere stop loss**: Le perdite superano le vincite medie")
        
        if m.risk_score > 60:
            recommendations.append("4. **Ridurre esposizione**: Il rischio è elevato, ridurre la size delle posizioni")
        
        if m.open_positions > 10:
            recommendations.append("5. **Diversificare meglio**: Troppe posizioni aperte, concentrare su setup migliori")
        
        if m.volatility > 40:
            recommendations.append("6. **Attenzione al mercato**: Alta volatilità, considerare posizioni più conservative")
        
        # Aggiungi raccomandazioni positive
        if m.pnl_percent > 10:
            recommendations.append("✅ Performance eccellente - continuare con la strategia attuale")
        
        if m.winrate > 60:
            recommendations.append("✅ Alto win rate - la strategia funziona, ottimizzare la size")
        
        if not recommendations:
            recommendations.append("✅ Il conto è in salute - continuare con la strategia attuale")
        
        content = "**Raccomandazioni Operative:**\n\n"
        content += "\n\n".join(recommendations)
        
        self.sections.append(ReportSection(
            title="💡 Raccomandazioni",
            content=content,
            severity="normal"
        ))


def generate_text_report(metrics: PerformanceMetrics) -> str:
    """
    Funzione principale per generare un report testuale.
    
    Args:
        metrics: Metriche di performance del conto
        
    Returns:
        Report formattato in testo
    """
    generator = AccountReportGenerator()
    report = generator.generate_report(metrics)
    
    # Formatta come testo
    text = f"""
{'='*60}
{report['title']}
{'='*60}

📊 METRICHE RAPIDE:
• Equità: {report['metrics']['equity']}
• P/L: {report['metrics']['pnl']}
• Win Rate: {report['metrics']['winrate']}
• Risk Score: {report['metrics']['risk_score']}

"""
    
    for section in report['sections']:
        text += f"\n{'─'*50}\n"
        text += f"{section['title']}\n"
        text += f"{'─'*50}\n"
        text += f"{section['content']}\n"
    
    text += f"\n{'='*60}\n"
    text += f"Report generato il {datetime.now().strftime('%d/%m/%Y alle ore %H:%M')}\n"
    text += f"{'='*60}\n"
    
    return text


# Esempio di utilizzo
if __name__ == "__main__":
    # Set UTF-8 encoding for Windows
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    # Esempio con dati mock
    metrics = PerformanceMetrics(
        equity=12500.0,
        initial_equity=10000.0,
        pnl=2500.0,
        pnl_percent=25.0,
        winrate=62.5,
        total_trades=40,
        winning_trades=25,
        losing_trades=15,
        avg_win=150.0,
        avg_loss=83.0,
        largest_win=500.0,
        largest_loss=200.0,
        max_drawdown=800.0,
        max_drawdown_percent=6.4,
        sharpe_ratio=1.8,
        volatility=18.5,
        risk_score=35,
        open_positions=3,
        positions_value=4500.0,
        cash_balance=8000.0
    )
    
    report = generate_text_report(metrics)
    print(report)
