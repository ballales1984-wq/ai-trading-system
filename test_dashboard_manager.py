"""
Test Live Dashboard Manager
===========================
Test per Day 4: Dashboard & Telegram Alerts

Verifica:
- Candlestick + indicatori su dashboard
- PnL, drawdown, metriche multi-asset live
- Telegram alerts per trade/rischi/errori
- Grafici e refresh live
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Import moduli da testare
from src.dashboard.live_dashboard_manager import (
    LiveDashboardManager, CandlestickChart, DashboardMetrics,
    Position, Alert, AlertType, create_dashboard_manager
)


class TestAlertType:
    """Test per AlertType."""
    
    def test_alert_type_values(self):
        """Test valori alert type."""
        assert AlertType.TRADE.value == "trade"
        assert AlertType.RISK.value == "risk"
        assert AlertType.ERROR.value == "error"
        assert AlertType.INFO.value == "info"
        assert AlertType.PROFIT.value == "profit"
        assert AlertType.LOSS.value == "loss"


class TestDashboardMetrics:
    """Test per DashboardMetrics."""
    
    def test_metrics_creation(self):
        """Test creazione metriche."""
        metrics = DashboardMetrics(
            timestamp=datetime.now(),
            total_pnl=100.0,
            daily_pnl=50.0,
            unrealized_pnl=25.0,
            max_drawdown=-0.05,
            sharpe_ratio=1.5,
            win_rate=0.6,
            total_trades=100,
            open_positions=3,
            portfolio_value=10500.0
        )
        
        assert metrics.total_pnl == 100.0
        assert metrics.sharpe_ratio == 1.5
        assert metrics.win_rate == 0.6
    
    def test_metrics_to_dict(self):
        """Test conversione a dict."""
        metrics = DashboardMetrics(
            timestamp=datetime.now(),
            total_pnl=100.0,
            daily_pnl=50.0,
            unrealized_pnl=25.0,
            max_drawdown=-0.05,
            sharpe_ratio=1.5,
            win_rate=0.6,
            total_trades=100,
            open_positions=3,
            portfolio_value=10500.0
        )
        
        d = metrics.to_dict()
        
        assert 'total_pnl' in d
        assert d['total_pnl'] == 100.0
        assert 'sharpe_ratio' in d


class TestPosition:
    """Test per Position."""
    
    def test_position_creation(self):
        """Test creazione posizione."""
        pos = Position(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            pnl=100.0,
            pnl_pct=0.02
        )
        
        assert pos.symbol == "BTCUSDT"
        assert pos.pnl == 100.0
        assert pos.pnl_pct == 0.02
    
    def test_position_to_dict(self):
        """Test conversione a dict."""
        pos = Position(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            pnl=100.0,
            pnl_pct=0.02
        )
        
        d = pos.to_dict()
        
        assert d['symbol'] == "BTCUSDT"
        assert d['pnl'] == 100.0


class TestAlert:
    """Test per Alert."""
    
    def test_alert_creation(self):
        """Test creazione alert."""
        alert = Alert(
            alert_type=AlertType.TRADE,
            title="Test Alert",
            message="This is a test"
        )
        
        assert alert.alert_type == AlertType.TRADE
        assert alert.title == "Test Alert"
    
    def test_alert_to_telegram_message(self):
        """Test conversione a messaggio Telegram."""
        alert = Alert(
            alert_type=AlertType.PROFIT,
            title="Profit Alert",
            message="You made $100"
        )
        
        msg = alert.to_telegram_message()
        
        assert "💰" in msg  # Profit emoji
        assert "*Profit Alert*" in msg
        assert "You made $100" in msg
    
    def test_alert_to_telegram_message_risk(self):
        """Test messaggio Telegram per risk."""
        alert = Alert(
            alert_type=AlertType.RISK,
            title="Risk Alert",
            message="Drawdown too high"
        )
        
        msg = alert.to_telegram_message()
        
        assert "⚠️" in msg  # Risk emoji


class TestCandlestickChart:
    """Test per CandlestickChart."""
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Crea dati OHLCV di test."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1h')
        np.random.seed(42)
        
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.randn(100) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(100) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(100) * 0.01)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        })
    
    def test_chart_creation(self, sample_ohlcv):
        """Test creazione grafico."""
        chart = CandlestickChart(title="Test Chart")
        
        fig = chart.create_chart(sample_ohlcv)
        
        assert fig is not None
        assert fig.layout.title.text == "Test Chart"
    
    def test_chart_with_indicators(self, sample_ohlcv):
        """Test grafico con indicatori."""
        chart = CandlestickChart()
        
        indicators = {
            'sma_20': sample_ohlcv['close'].rolling(20).mean(),
            'ema_50': sample_ohlcv['close'].ewm(span=50).mean()
        }
        
        fig = chart.create_chart(sample_ohlcv, indicators=indicators)
        
        assert fig is not None
    
    def test_chart_with_volume(self, sample_ohlcv):
        """Test grafico con volume."""
        chart = CandlestickChart()
        
        fig = chart.create_chart(sample_ohlcv, show_volume=True)
        
        assert fig is not None
    
    def test_pnl_chart(self):
        """Test grafico PnL."""
        chart = CandlestickChart()
        
        pnl_history = [
            {'timestamp': datetime.now() - timedelta(hours=i), 'pnl': i * 10, 'drawdown': -i * 0.01}
            for i in range(50)
        ]
        
        fig = chart.create_pnl_chart(pnl_history)
        
        assert fig is not None


class TestLiveDashboardManager:
    """Test per LiveDashboardManager."""
    
    @pytest.fixture
    def manager(self):
        """Crea manager per test."""
        return LiveDashboardManager(
            telegram_enabled=False  # Disable per test
        )
    
    def test_manager_creation(self, manager):
        """Test creazione manager."""
        assert manager is not None
        assert manager.telegram_enabled == False
        assert manager.telegram is None
    
    def test_update_metrics(self, manager):
        """Test aggiornamento metriche."""
        manager.update_metrics(
            total_pnl=100.0,
            daily_pnl=50.0,
            sharpe_ratio=1.5
        )
        
        assert manager.metrics is not None
        assert manager.metrics.total_pnl == 100.0
        assert manager.metrics.sharpe_ratio == 1.5
    
    def test_update_positions(self, manager):
        """Test aggiornamento posizioni."""
        positions = [
            Position("BTCUSDT", "LONG", 0.1, 50000, 51000, 100, 0.02),
            Position("ETHUSDT", "LONG", 1.0, 3000, 3100, 100, 0.033)
        ]
        
        manager.update_positions(positions)
        
        assert len(manager.positions) == 2
        assert manager.positions[0].symbol == "BTCUSDT"
    
    def test_send_alert(self, manager):
        """Test invio alert."""
        alert = Alert(
            alert_type=AlertType.INFO,
            title="Test",
            message="Test message"
        )
        
        manager.send_alert(alert)
        
        assert len(manager.alerts) == 1
    
    def test_send_trade_alert(self, manager):
        """Test alert trade."""
        manager.send_trade_alert(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
            price=50000,
            pnl=100
        )
        
        assert len(manager.alerts) == 1
        assert manager.alerts[0].alert_type == AlertType.PROFIT
    
    def test_send_trade_alert_loss(self, manager):
        """Test alert trade con perdita."""
        manager.send_trade_alert(
            symbol="BTCUSDT",
            side="SELL",
            quantity=0.1,
            price=50000,
            pnl=-50
        )
        
        assert len(manager.alerts) == 1
        assert manager.alerts[0].alert_type == AlertType.LOSS
    
    def test_send_risk_alert(self, manager):
        """Test alert rischio."""
        manager.send_risk_alert(
            title="High Drawdown",
            message="Drawdown at 15%"
        )
        
        assert len(manager.alerts) == 1
        assert manager.alerts[0].alert_type == AlertType.RISK
    
    def test_send_error_alert(self, manager):
        """Test alert errore."""
        manager.send_error_alert(
            title="Connection Error",
            message="Failed to connect",
            error="Timeout"
        )
        
        assert len(manager.alerts) == 1
        assert manager.alerts[0].alert_type == AlertType.ERROR
    
    def test_get_metrics_summary(self, manager):
        """Test riepilogo metriche."""
        manager.update_metrics(total_pnl=100.0)
        
        summary = manager.get_metrics_summary()
        
        assert 'metrics' in summary
        assert summary['metrics']['total_pnl'] == 100.0
    
    def test_get_dashboard_data(self, manager):
        """Test dati dashboard."""
        manager.update_metrics(total_pnl=100.0)
        
        data = manager.get_dashboard_data()
        
        assert 'metrics' in data
        assert 'positions' in data
        assert 'pnl_history' in data
    
    def test_callbacks(self, manager):
        """Test callbacks."""
        metrics_received = []
        positions_received = []
        
        def on_metrics(m):
            metrics_received.append(m)
        
        def on_positions(p):
            positions_received.append(p)
        
        manager.set_callbacks(
            on_metrics_update=on_metrics,
            on_position_update=on_positions
        )
        
        manager.update_metrics(total_pnl=100.0)
        manager.update_positions([Position("BTCUSDT", "LONG", 0.1, 50000, 51000, 100, 0.02)])
        
        assert len(metrics_received) == 1
        assert len(positions_received) == 1


class TestLiveDashboardManagerThreaded:
    """Test con thread."""
    
    def test_start_stop(self):
        """Test avvio e stop."""
        manager = LiveDashboardManager(telegram_enabled=False, refresh_interval=1)
        
        manager.start()
        
        time.sleep(0.5)
        
        assert manager._running == True
        
        manager.stop()
        
        assert manager._running == False
    
    def test_automatic_metrics_update(self):
        """Test aggiornamento automatico metriche."""
        manager = LiveDashboardManager(telegram_enabled=False, refresh_interval=1)
        
        # Imposta metriche iniziali
        manager.update_metrics(total_pnl=0)
        
        manager.start()
        
        time.sleep(1.5)
        
        manager.stop()
        
        # Dovrebbe aver aggiornato le metriche
        assert manager.metrics is not None


class TestLiveDashboardManagerWithTelegram:
    """Test con Telegram mock."""
    
    @patch('src.dashboard.live_dashboard_manager.TelegramNotifier')
    def test_telegram_alert(self, mock_telegram):
        """Test alert con Telegram."""
        mock_instance = MagicMock()
        mock_instance.send.return_value = True
        mock_telegram.return_value = mock_instance
        
        manager = LiveDashboardManager(
            telegram_token="test_token",
            telegram_chat_id="test_chat",
            telegram_enabled=True
        )
        
        manager.send_trade_alert(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
            price=50000,
            pnl=100
        )
        
        # Telegram send dovrebbe essere stato chiamato
        mock_instance.send.assert_called_once()


class TestCreateDashboardManager:
    """Test factory function."""
    
    def test_create_manager(self):
        """Test creazione tramite factory."""
        manager = create_dashboard_manager(
            telegram_enabled=False
        )
        
        assert manager is not None
        assert manager.telegram_enabled == False


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])