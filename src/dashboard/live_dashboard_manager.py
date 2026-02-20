"""
Live Dashboard Manager
======================
Integra Dashboard con Telegram Alerts e Candlestick Charts.

Day 4 Checklist:
- [x] Candlestick + indicatori su dashboard
- [x] PnL, drawdown, metriche multi-asset live
- [x] Telegram alerts per trade/rischi/errori
- [x] Grafici e refresh live
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.live.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Tipo di alert."""
    TRADE = "trade"
    RISK = "risk"
    ERROR = "error"
    INFO = "info"
    PROFIT = "profit"
    LOSS = "loss"


@dataclass
class DashboardMetrics:
    """Metriche per la dashboard."""
    timestamp: datetime
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    open_positions: int
    portfolio_value: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'open_positions': self.open_positions,
            'portfolio_value': self.portfolio_value
        }


@dataclass
class Position:
    """Posizione aperta."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class Alert:
    """Alert da inviare."""
    alert_type: AlertType
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict = field(default_factory=dict)
    
    def to_telegram_message(self) -> str:
        """Converte in messaggio Telegram formattato."""
        emoji_map = {
            AlertType.TRADE: "📊",
            AlertType.RISK: "⚠️",
            AlertType.ERROR: "❌",
            AlertType.INFO: "ℹ️",
            AlertType.PROFIT: "💰",
            AlertType.LOSS: "📉"
        }
        
        emoji = emoji_map.get(self.alert_type, "📌")
        
        return f"""
{emoji} *{self.title}*

{self.message}

🕐 {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""


class CandlestickChart:
    """
    Genera grafici candlestick con indicatori.
    """
    
    def __init__(self, title: str = "Price Chart"):
        self.title = title
    
    def create_chart(
        self,
        df: pd.DataFrame,
        indicators: Optional[Dict] = None,
        show_volume: bool = True,
        show_signals: bool = True
    ) -> go.Figure:
        """
        Crea grafico candlestick con indicatori.
        
        Args:
            df: DataFrame con OHLCV data
            indicators: Dict con indicatori da mostrare
            show_volume: Mostra volume
            show_signals: Mostra segnali buy/sell
            
        Returns:
            Plotly Figure
        """
        # Determina numero di subplot
        rows = 2 if show_volume else 1
        row_heights = [0.7, 0.3] if show_volume else [1.0]
        
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('timestamp', df.index),
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="OHLC",
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Indicatori
        if indicators:
            # Moving Averages
            if 'sma_20' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('timestamp', df.index),
                        y=indicators['sma_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='#ffd700', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'ema_50' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('timestamp', df.index),
                        y=indicators['ema_50'],
                        mode='lines',
                        name='EMA 50',
                        line=dict(color='#2196f3', width=1)
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                x = df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('timestamp', df.index)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=indicators['bb_upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='rgba(139, 195, 74, 0.5)', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=indicators['bb_lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='rgba(139, 195, 74, 0.5)', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(139, 195, 74, 0.1)'
                    ),
                    row=1, col=1
                )
        
        # Volume
        if show_volume and 'volume' in df.columns:
            colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] 
                     else '#ef5350' for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(
                    x=df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('timestamp', df.index),
                    y=df['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Layout
        fig.update_layout(
            title=self.title,
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=600 if show_volume else 400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="Time", row=rows, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        if show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def create_pnl_chart(
        self,
        pnl_history: List[Dict],
        title: str = "Portfolio PnL"
    ) -> go.Figure:
        """
        Crea grafico PnL nel tempo.
        
        Args:
            pnl_history: Lista di dict con timestamp e pnl
            title: Titolo del grafico
            
        Returns:
            Plotly Figure
        """
        if not pnl_history:
            return go.Figure()
        
        df = pd.DataFrame(pnl_history)
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        # PnL Line
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['pnl'],
                mode='lines',
                name='PnL',
                line=dict(color='#2196f3', width=2),
                fill='tozeroy',
                fillcolor='rgba(33, 150, 243, 0.1)'
            ),
            row=1, col=1
        )
        
        # Drawdown
        if 'drawdown' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['drawdown'],
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#ef5350', width=1)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig


class LiveDashboardManager:
    """
    Manager per la dashboard live con Telegram alerts.
    
    Integra:
    - Candlestick charts con indicatori
    - Metriche multi-asset in tempo reale
    - Telegram alerts per trade/rischi/errori
    - Refresh automatico
    """
    
    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        telegram_enabled: bool = False,
        refresh_interval: int = 5,
        alert_thresholds: Optional[Dict] = None
    ):
        """
        Inizializza il manager.
        
        Args:
            telegram_token: Token del bot Telegram
            telegram_chat_id: Chat ID Telegram
            telegram_enabled: Se inviare alert Telegram
            refresh_interval: Intervallo refresh in secondi
            alert_thresholds: Soglie per alert automatici
        """
        self.refresh_interval = refresh_interval
        self.telegram_enabled = telegram_enabled
        
        # Telegram notifier
        if telegram_enabled and telegram_token and telegram_chat_id:
            self.telegram = TelegramNotifier(
                bot_token=telegram_token,
                chat_id=telegram_chat_id,
                enabled=True
            )
        else:
            self.telegram = None
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'pnl_loss_pct': -0.05,  # Alert se perdita > 5%
            'drawdown_pct': -0.10,  # Alert se drawdown > 10%
            'win_rate_min': 0.4,    # Alert se win rate < 40%
        }
        
        # Stato
        self.metrics: Optional[DashboardMetrics] = None
        self.positions: List[Position] = []
        self.pnl_history: List[Dict] = []
        self.alerts: List[Alert] = []
        
        # Chart generator
        self.chart_generator = CandlestickChart()
        
        # Thread
        self._running = False
        self._refresh_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_metrics_update: Optional[Callable[[DashboardMetrics], None]] = None
        self._on_position_update: Optional[Callable[[List[Position]], None]] = None
        
        logger.info(f"📊 LiveDashboardManager initialized (telegram={telegram_enabled})")
    
    def set_callbacks(
        self,
        on_metrics_update: Optional[Callable[[DashboardMetrics], None]] = None,
        on_position_update: Optional[Callable[[List[Position]], None]] = None
    ):
        """Imposta callback per aggiornamenti."""
        self._on_metrics_update = on_metrics_update
        self._on_position_update = on_position_update
    
    def start(self):
        """Avvia il refresh automatico."""
        if self._running:
            return
        
        self._running = True
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()
        
        logger.info("✅ LiveDashboardManager started")
    
    def stop(self):
        """Ferma il refresh automatico."""
        self._running = False
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5)
        
        logger.info("🛑 LiveDashboardManager stopped")
    
    def _refresh_loop(self):
        """Loop di refresh automatico."""
        while self._running:
            try:
                self._update_metrics()
                self._check_alerts()
                time.sleep(self.refresh_interval)
            except Exception as e:
                logger.error(f"Error in refresh loop: {e}")
                time.sleep(1)
    
    def _update_metrics(self):
        """Aggiorna le metriche."""
        # Placeholder - in produzione questi dati arriverebbero dal portfolio manager
        if self.metrics is None:
            self.metrics = DashboardMetrics(
                timestamp=datetime.now(),
                total_pnl=0.0,
                daily_pnl=0.0,
                unrealized_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                total_trades=0,
                open_positions=0,
                portfolio_value=10000.0
            )
        
        # Callback
        if self._on_metrics_update:
            self._on_metrics_update(self.metrics)
    
    def _check_alerts(self):
        """Controlla se inviare alert automatici."""
        if not self.metrics or not self.telegram:
            return
        
        # Check drawdown
        if self.metrics.max_drawdown < self.alert_thresholds['drawdown_pct']:
            self.send_alert(Alert(
                alert_type=AlertType.RISK,
                title="⚠️ High Drawdown Alert",
                message=f"Drawdown has reached {self.metrics.max_drawdown*100:.2f}%",
                data={'drawdown': self.metrics.max_drawdown}
            ))
        
        # Check win rate
        if self.metrics.win_rate < self.alert_thresholds['win_rate_min'] and self.metrics.total_trades > 10:
            self.send_alert(Alert(
                alert_type=AlertType.RISK,
                title="⚠️ Low Win Rate Alert",
                message=f"Win rate has dropped to {self.metrics.win_rate*100:.1f}%",
                data={'win_rate': self.metrics.win_rate}
            ))
    
    def update_metrics(self, **kwargs):
        """Aggiorna le metriche manualmente."""
        if self.metrics is None:
            self.metrics = DashboardMetrics(
                timestamp=datetime.now(),
                total_pnl=0.0,
                daily_pnl=0.0,
                unrealized_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                total_trades=0,
                open_positions=0,
                portfolio_value=10000.0
            )
        
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
        
        self.metrics.timestamp = datetime.now()
        
        # Aggiungi a history
        self.pnl_history.append({
            'timestamp': self.metrics.timestamp,
            'pnl': self.metrics.total_pnl,
            'drawdown': self.metrics.max_drawdown
        })
        
        # Callback
        if self._on_metrics_update:
            self._on_metrics_update(self.metrics)
    
    def update_positions(self, positions: List[Position]):
        """Aggiorna le posizioni."""
        self.positions = positions
        
        # Callback
        if self._on_position_update:
            self._on_position_update(positions)
    
    def send_alert(self, alert: Alert):
        """
        Invia un alert.
        
        Args:
            alert: Alert da inviare
        """
        self.alerts.append(alert)
        
        # Log
        logger.info(f"Alert: {alert.title} - {alert.message}")
        
        # Telegram
        if self.telegram:
            message = alert.to_telegram_message()
            self.telegram.send(message)
    
    def send_trade_alert(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None
    ):
        """Invia alert per trade eseguito."""
        if pnl and pnl > 0:
            alert_type = AlertType.PROFIT
            title = f"💰 Profit: {symbol}"
        elif pnl and pnl < 0:
            alert_type = AlertType.LOSS
            title = f"📉 Loss: {symbol}"
        else:
            alert_type = AlertType.TRADE
            title = f"📊 Trade: {symbol}"
        
        message = f"""
Side: {side.upper()}
Quantity: {quantity}
Price: ${price:.2f}
"""
        if pnl:
            message += f"PnL: ${pnl:.2f}\n"
        
        self.send_alert(Alert(
            alert_type=alert_type,
            title=title,
            message=message,
            data={'symbol': symbol, 'side': side, 'quantity': quantity, 'price': price, 'pnl': pnl}
        ))
    
    def send_risk_alert(self, title: str, message: str, data: Optional[Dict] = None):
        """Invia alert di rischio."""
        self.send_alert(Alert(
            alert_type=AlertType.RISK,
            title=title,
            message=message,
            data=data or {}
        ))
    
    def send_error_alert(self, title: str, message: str, error: Optional[str] = None):
        """Invia alert di errore."""
        full_message = message
        if error:
            full_message += f"\n\nError: {error}"
        
        self.send_alert(Alert(
            alert_type=AlertType.ERROR,
            title=title,
            message=full_message
        ))
    
    def get_candlestick_chart(
        self,
        df: pd.DataFrame,
        indicators: Optional[Dict] = None
    ) -> go.Figure:
        """Genera grafico candlestick."""
        return self.chart_generator.create_chart(df, indicators)
    
    def get_pnl_chart(self) -> go.Figure:
        """Genera grafico PnL."""
        return self.chart_generator.create_pnl_chart(self.pnl_history)
    
    def get_metrics_summary(self) -> Dict:
        """Ottiene riassunto metriche."""
        if not self.metrics:
            return {}
        
        return {
            'metrics': self.metrics.to_dict(),
            'positions': [p.to_dict() for p in self.positions],
            'recent_alerts': [
                {
                    'type': a.alert_type.value,
                    'title': a.title,
                    'message': a.message,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in self.alerts[-10:]
            ]
        }
    
    def get_dashboard_data(self) -> Dict:
        """Ottiene tutti i dati per la dashboard."""
        return {
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'positions': [p.to_dict() for p in self.positions],
            'pnl_history': self.pnl_history[-100:],  # Ultimi 100 punti
            'alerts_count': len(self.alerts),
            'running': self._running
        }


def create_dashboard_manager(
    telegram_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    telegram_enabled: bool = False
) -> LiveDashboardManager:
    """
    Factory function per creare LiveDashboardManager.
    
    Args:
        telegram_token: Token del bot Telegram
        telegram_chat_id: Chat ID Telegram
        telegram_enabled: Se inviare alert Telegram
        
    Returns:
        LiveDashboardManager configurato
    """
    return LiveDashboardManager(
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        telegram_enabled=telegram_enabled
    )
