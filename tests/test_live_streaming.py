"""
Test Live Streaming Manager
===========================
Test per Day 1: Live Multi-Asset Streaming

Verifica:
- WebSocket Binance per tutti gli asset
- PortfolioManager.update_prices() a ogni tick
- PaperBroker per trading live
- Log posizioni aperte e PnL
- Stop-loss in tempo reale
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import moduli da testare
from src.live.live_streaming_manager import (
    LiveStreamingManager,
    StopLossOrder,
    PositionLog,
    create_live_manager
)
from src.core.portfolio.portfolio_manager import Position, PositionSide


class TestStopLossOrder:
    """Test per StopLossOrder."""
    
    def test_stop_loss_creation(self):
        """Test creazione stop-loss."""
        order = StopLossOrder(
            symbol="BTCUSDT",
            stop_price=50000,
            quantity=0.1
        )
        
        assert order.symbol == "BTCUSDT"
        assert order.stop_price == 50000
        assert order.quantity == 0.1
        assert order.is_trailing == False
        assert order.triggered == False
    
    def test_stop_loss_trigger(self):
        """Test trigger stop-loss."""
        order = StopLossOrder(
            symbol="BTCUSDT",
            stop_price=50000,
            quantity=0.1
        )
        
        # Prezzo sopra stop - non triggera
        assert order.check_trigger(51000) == False
        assert order.triggered == False
        
        # Prezzo sotto stop - triggera
        assert order.check_trigger(49000) == True
        assert order.triggered == True
        
        # Già triggerato - non triggera di nuovo
        assert order.check_trigger(48000) == False
    
    def test_trailing_stop_update(self):
        """Test trailing stop si aggiorna quando il prezzo sale."""
        order = StopLossOrder(
            symbol="BTCUSDT",
            stop_price=50000,
            quantity=0.1,
            is_trailing=True,
            trail_pct=0.02  # 2%
        )
        
        # Prezzo sale - stop si aggiorna
        current_price = 52000
        order.check_trigger(current_price)
        
        # Nuovo stop dovrebbe essere 52000 * 0.98 = 50960
        expected_stop = 52000 * (1 - 0.02)
        assert order.stop_price == expected_stop
    
    def test_trailing_stop_not_update_on_drop(self):
        """Test trailing stop non si aggiorna quando il prezzo scende."""
        order = StopLossOrder(
            symbol="BTCUSDT",
            stop_price=50000,
            quantity=0.1,
            is_trailing=True,
            trail_pct=0.02
        )
        
        # Prezzo scende - stop non si aggiorna
        order.check_trigger(49000)
        assert order.stop_price == 50000  # Invariato


class TestLiveStreamingManager:
    """Test per LiveStreamingManager."""
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock per BinanceMultiWebSocket."""
        with patch('src.live.live_streaming_manager.BinanceMultiWebSocket') as mock:
            ws_instance = MagicMock()
            ws_instance.get_all_prices.return_value = {
                'BTCUSDT': 50000,
                'ETHUSDT': 3000,
                'SOLUSDT': 100
            }
            ws_instance.get_price.return_value = 50000
            ws_instance.is_ready.return_value = True
            mock.return_value = ws_instance
            yield mock
    
    @pytest.fixture
    def manager(self, mock_websocket):
        """Crea un manager per i test."""
        return LiveStreamingManager(
            symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
            initial_balance=100000,
            testnet=True
        )
    
    def test_manager_creation(self, manager):
        """Test creazione manager."""
        assert manager is not None
        assert len(manager.symbols) == 3
        assert manager.portfolio is not None
        assert manager.default_stop_loss_pct == 0.02
    
    def test_open_position_with_stop_loss(self, manager):
        """Test apertura posizione con stop-loss automatico."""
        position = manager.open_position(
            symbol='BTCUSDT',
            side='long',
            quantity=0.1,
            price=50000
        )
        
        assert position is not None
        assert position.symbol == 'BTCUSDT'
        assert position.side == PositionSide.LONG
        assert position.quantity == 0.1
        
        # Verifica stop-loss creato
        assert 'BTCUSDT' in manager.stop_loss_orders
        stop_order = manager.stop_loss_orders['BTCUSDT']
        assert stop_order.stop_price == 50000 * 0.98  # 2% sotto
    
    def test_close_position(self, manager):
        """Test chiusura posizione."""
        # Apri posizione
        manager.open_position(
            symbol='BTCUSDT',
            side='long',
            quantity=0.1,
            price=50000
        )
        
        # Chiudi posizione
        result = manager.close_position(symbol='BTCUSDT', price=51000)
        
        assert result is not None
        assert 'pnl' in result
        # PnL = (51000 - 50000) * 0.1 = 100
        assert result['pnl'] == pytest.approx(100, rel=0.01)
        
        # Stop-loss rimosso
        assert 'BTCUSDT' not in manager.stop_loss_orders
    
    def test_get_current_prices(self, manager):
        """Test ottenimento prezzi correnti."""
        prices = manager.get_current_prices()
        
        assert isinstance(prices, dict)
        assert 'BTCUSDT' in prices
        assert 'ETHUSDT' in prices
        assert 'SOLUSDT' in prices
    
    def test_get_portfolio_state(self, manager):
        """Test stato portfolio."""
        state = manager.get_portfolio_state()
        
        assert 'balance' in state
        assert 'positions' in state
        assert 'metrics' in state
    
    def test_position_logging(self, manager):
        """Test logging posizioni."""
        # Apri posizione
        manager.open_position(
            symbol='BTCUSDT',
            side='long',
            quantity=0.1,
            price=50000
        )
        
        # Esegui log
        manager._log_positions()
        
        # Verifica log creato
        logs = manager.get_position_logs()
        assert len(logs) > 0
        
        last_log = logs[-1]
        assert last_log.symbol == 'BTCUSDT'
        assert last_log.side == 'long'
    
    def test_stop_loss_execution(self, manager):
        """Test esecuzione stop-loss."""
        # Apri posizione
        manager.open_position(
            symbol='BTCUSDT',
            side='long',
            quantity=0.1,
            price=50000
        )
        
        # Simula trigger stop-loss
        manager._check_stop_losses({'BTCUSDT': 48000})  # Sotto stop
        
        # Posizione dovrebbe essere chiusa
        positions = manager.get_open_positions()
        btc_positions = [p for p in positions if p.symbol == 'BTCUSDT']
        assert len(btc_positions) == 0
    
    def test_callbacks(self, manager):
        """Test callback per eventi."""
        price_update_called = []
        position_change_called = []
        stop_loss_called = []
        
        def on_price_update(symbol, price):
            price_update_called.append((symbol, price))
        
        def on_position_change(position):
            position_change_called.append(position)
        
        def on_stop_loss(order, price):
            stop_loss_called.append((order, price))
        
        manager.set_callbacks(
            on_price_update=on_price_update,
            on_position_change=on_position_change,
            on_stop_loss_triggered=on_stop_loss
        )
        
        # Apri posizione - dovrebbe chiamare position_change
        manager.open_position(
            symbol='BTCUSDT',
            side='long',
            quantity=0.1,
            price=50000
        )
        
        assert len(position_change_called) == 1
    
    def test_update_stop_loss(self, manager):
        """Test aggiornamento stop-loss."""
        manager.open_position(
            symbol='BTCUSDT',
            side='long',
            quantity=0.1,
            price=50000
        )
        
        # Aggiorna stop-loss
        manager.update_stop_loss('BTCUSDT', 49500)
        
        assert manager.stop_loss_orders['BTCUSDT'].stop_price == 49500


class TestPaperBrokerIntegration:
    """Test integrazione con PaperBroker."""
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock per WebSocket."""
        with patch('src.live.live_streaming_manager.BinanceMultiWebSocket') as mock:
            ws_instance = MagicMock()
            ws_instance.get_all_prices.return_value = {
                'BTCUSDT': 50000,
                'ETHUSDT': 3000
            }
            ws_instance.get_price.return_value = 50000
            ws_instance.is_ready.return_value = True
            mock.return_value = ws_instance
            yield mock
    
    @pytest.fixture
    def manager(self, mock_websocket):
        """Manager con paper trading."""
        return LiveStreamingManager(
            symbols=['BTCUSDT', 'ETHUSDT'],
            initial_balance=10000,  # Bilancio più piccolo per test
            testnet=True
        )
    
    def test_paper_trading_workflow(self, manager):
        """Test workflow completo paper trading."""
        # 1. Verifica stato iniziale
        state = manager.get_portfolio_state()
        assert state['balance'] == 10000
        
        # 2. Apri posizione
        position = manager.open_position(
            symbol='BTCUSDT',
            side='long',
            quantity=0.01,
            price=50000
        )
        assert position is not None
        
        # 3. Verifica posizione aperta
        positions = manager.get_open_positions()
        assert len(positions) == 1
        
        # 4. Aggiorna prezzi (simula tick)
        manager.portfolio.update_prices({'BTCUSDT': 51000})
        
        # 5. Verifica PnL aggiornato
        pos = manager.portfolio.get_position('BTCUSDT')
        expected_pnl = (51000 - 50000) * 0.01  # 10
        assert pos.unrealized_pnl == pytest.approx(expected_pnl, rel=0.01)
        
        # 6. Chiudi posizione
        result = manager.close_position('BTCUSDT', price=51000)
        assert result['pnl'] == pytest.approx(expected_pnl, rel=0.01)
        
        # 7. Verifica posizioni chiuse
        positions = manager.get_open_positions()
        assert len(positions) == 0
    
    def test_multiple_positions(self, manager):
        """Test gestione multiple posizioni."""
        # Apri più posizioni
        manager.open_position('BTCUSDT', 'long', 0.01, 50000)
        manager.open_position('ETHUSDT', 'long', 0.1, 3000)
        
        positions = manager.get_open_positions()
        assert len(positions) == 2
        
        # Verifica stop-loss per entrambe
        assert 'BTCUSDT' in manager.stop_loss_orders
        assert 'ETHUSDT' in manager.stop_loss_orders
    
    def test_position_sizing_auto(self, manager):
        """Test calcolo automatico dimensione posizione."""
        # Senza specificare quantità
        position = manager.open_position(
            symbol='BTCUSDT',
            side='long',
            price=50000,
            stop_loss_pct=0.02
        )
        
        # Quantità dovrebbe essere calcolata
        assert position.quantity > 0
        
        # Verifica che non superi il limite
        max_position_value = 10000 * 0.3  # 30% del portfolio
        position_value = position.quantity * 50000
        assert position_value <= max_position_value


class TestLiveStreamingManagerThreaded:
    """Test con thread attivi."""
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock per WebSocket con dati dinamici."""
        with patch('src.live.live_streaming_manager.BinanceMultiWebSocket') as mock:
            ws_instance = MagicMock()
            
            # Prezzi che cambiano
            price_counter = [50000]
            
            def get_prices():
                return {'BTCUSDT': price_counter[0]}
            
            def get_price(symbol):
                return price_counter[0]
            
            ws_instance.get_all_prices = get_prices
            ws_instance.get_price = get_price
            ws_instance.is_ready.return_value = True
            ws_instance.start = MagicMock()
            ws_instance.stop = MagicMock()
            mock.return_value = ws_instance
            yield mock, price_counter
    
    def test_update_loop_runs(self, mock_websocket):
        """Test che il loop di update gira correttamente."""
        mock_ws, price_counter = mock_websocket
        
        manager = LiveStreamingManager(
            symbols=['BTCUSDT'],
            initial_balance=10000,
            testnet=True
        )
        
        # Apri posizione
        manager.open_position('BTCUSDT', 'long', 0.01, 50000)
        
        # Avvia manager
        manager.start()
        
        # Aspetta un po'
        time.sleep(0.5)
        
        # Ferma
        manager.stop()
        
        # Verifica che start/stop siano stati chiamati
        mock_ws.return_value.start.assert_called_once()
        mock_ws.return_value.stop.assert_called_once()


class TestCreateLiveManager:
    """Test factory function."""
    
    def test_create_live_manager(self):
        """Test creazione manager tramite factory."""
        with patch('src.live.live_streaming_manager.BinanceMultiWebSocket'):
            manager = create_live_manager(
                symbols=['BTCUSDT', 'ETHUSDT'],
                initial_balance=50000,
                testnet=True
            )
            
            assert manager is not None
            assert len(manager.symbols) == 2
            assert manager.default_stop_loss_pct == 0.02
            assert manager.enable_trailing_stop == True


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
