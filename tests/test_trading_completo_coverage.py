"""
Test coverage for trading_completo module
"""
import os
import tempfile
import shutil
from src.trading_completo import (
    inizializza_registro,
    registra_trade,
    report_giornaliero,
    report_premi,
    get_balance,
    set_balance,
    update_balance,
    reset_balance,
    apri_posizione,
    chiudi_posizione,
    get_posizione,
    get_all_posizioni,
    assegna_premio,
    reset_awards,
    reset_registro,
    set_data_dir,
    get_awards,
    get_total_awards,
    Trade,
    Position,
    AWARD_CONFIG
)



class TestTradingCompleto:
    """Test trading_completo module"""
    
    def setup_method(self):
        """Set up a temporary directory for each test"""
        self.test_dir = tempfile.mkdtemp()
        set_data_dir(self.test_dir)
        inizializza_registro()
        
    def teardown_method(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)
        
    def test_initialize_registry(self):
        """Test that initializing the registry creates the necessary files"""
        registro_path = os.path.join(self.test_dir, "registro_trading.csv")
        saldo_path = os.path.join(self.test_dir, "saldo.txt")
        posizioni_path = os.path.join(self.test_dir, "posizioni.json")
        premi_path = os.path.join(self.test_dir, "premi.json")
        
        assert os.path.exists(registro_path)
        assert os.path.exists(saldo_path)
        assert os.path.exists(posizioni_path)
        assert os.path.exists(premi_path)
        
        # Check CSV header
        with open(registro_path, 'r') as f:
            header = f.readline().strip()
            expected_header = "Data/Ora,Asset,Tipo,Quantità,Prezzo,Commissione,Profit/Loss,Saldo Totale,API Usata,Punti Premio,Strategy,Prezzo Acquisto"
            assert header == expected_header
            
    def test_get_set_balance(self):
        """Test getting and setting balance"""
        # Initial balance should be 0.0
        assert get_balance() == 0.0
        
        # Set balance to 100.0
        set_balance(100.0)
        assert get_balance() == 100.0
        
        # Set balance to -50.0
        set_balance(-50.0)
        assert get_balance() == -50.0
        
    def test_update_balance(self):
        """Test updating balance by profit/loss"""
        set_balance(100.0)
        assert get_balance() == 100.0
        
        # Update with +50.0
        new_balance = update_balance(50.0)
        assert new_balance == 150.0
        assert get_balance() == 150.0
        
        # Update with -30.0
        new_balance = update_balance(-30.0)
        assert new_balance == 120.0
        assert get_balance() == 120.0
        
    def test_reset_balance(self):
        """Test resetting balance"""
        set_balance(200.0)
        assert get_balance() == 200.0
        
        reset_balance()
        assert get_balance() == 0.0
        
        reset_balance(500.0)
        assert get_balance() == 500.0
        
    def test_apri_posizione(self):
        """Test opening a position"""
        posizione = apri_posizione("BTC", 0.1, 25000.0, "TestAPI", "TestStrategy")
        
        assert posizione.asset == "BTC"
        assert posizione.quantita == 0.1
        assert posizione.prezzo_acquisto == 25000.0
        assert posizione.api_usata == "TestAPI"
        assert posizione.strategy == "TestStrategy"
        assert posizione.data_acquisto is not None
        
        # Check that position is stored
        pos = get_posizione("BTC")
        assert pos is not None
        assert pos.asset == "BTC"
        assert pos.quantita == 0.1
        
    def test_chiudi_posizione(self):
        """Test closing a position"""
        # Open a position first
        apri_posizione("ETH", 1.0, 1800.0, "TestAPI", "TestStrategy")
        
        # Close the position
        posizione = chiudi_posizione("ETH")
        
        assert posizione is not None
        assert posizione.asset == "ETH"
        assert posizione.quantita == 1.0
        assert posizione.prezzo_acquisto == 1800.0
        
        # Check that position is removed
        pos = get_posizione("ETH")
        assert pos is None
        
        # Try to close non-existent position
        posizione = chiudi_posizione("NONEXISTENT")
        assert posizione is None
        
    def test_get_all_posizioni(self):
        """Test getting all positions"""
        # Initially empty
        assert get_all_posizioni() == {}
        
        # Add two positions
        apri_posizione("BTC", 0.1, 25000.0, "TestAPI", "TestStrategy")
        apri_posizione("ETH", 1.0, 1800.0, "TestAPI", "TestStrategy")
        
        posizioni = get_all_posizioni()
        assert len(posizioni) == 2
        assert "BTC" in posizioni
        assert "ETH" in posizioni
        
    def test_assegna_premio(self):
        """Test award assignment"""
        # Test BUY trade with loss (only commission)
        trade = {
            "asset": "BTC",
            "tipo": "BUY",
            "quantita": 0.1,
            "prezzo": 25000,
            "commissione": 5,
            "api_usata": "TestAPI",
            "strategy": "TestStrategy"
        }
        punteggio = assegna_premio(trade)
        # BUY with loss: profit_loss = -commissione = -5
        # Base: profit_loss * abs(penalty_loss) = -5 * 2 = -10 (since penalty_loss is -2)
        # Penalty for wrong direction: BUY and loss -> +penalty_wrong_direction = -5
        # Total: -10 + (-5) = -15? Wait, let's check the code:
        # Actually, in the code:
        #   if profit_loss > 0:
        #       punteggio = profit_loss * config["base_points_per_profit"]
        #   else:
        #       punteggio = profit_loss * abs(config["penalty_loss"])
        # Then add/subtract bonuses/penalties.
        # For BUY with loss: profit_loss = -5
        #   punteggio = -5 * abs(-2) = -5 * 2 = -10
        #   Then: tipo == "BUY" and profit_loss < 0 -> punteggio += config["penalty_wrong_direction"] = -5
        #   So total = -10 + (-5) = -15
        # But also check for API bonus and strategy bonus: only if profit_loss > 0, so none.
        assert punteggio == -15.0
        
        # Test SELL trade with profit
        trade = {
            "asset": "BTC",
            "tipo": "SELL",
            "quantita": 0.1,
            "prezzo": 26000,
            "commissione": 5,
            "prezzo_acquisto": 25000,
            "api_usata": "TestAPI",
            "strategy": "TestStrategy"
        }
        punteggio = assegna_premio(trade)
        # profit_loss = (26000 - 25000) * 0.1 - 5 = 1000 * 0.1 - 5 = 100 - 5 = 95
        # Base: 95 * 1.0 = 95
        # Bonus: SELL with profit -> +10
        # Bonus: API contribution -> +5
        # Bonus: Strategy correct -> +15
        # No penalty (wrong direction only for BUY with loss)
        # Total: 95 + 10 + 5 + 15 = 125
        assert punteggio == 125.0
        
    def test_reset_awards(self):
        """Test resetting awards"""
        # Assign some awards
        trade = {
            "asset": "BTC",
            "tipo": "SELL",
            "quantita": 0.1,
            "prezzo": 26000,
            "commissione": 5,
            "prezzo_acquisto": 25000,
            "api_usata": "TestAPI",
            "strategy": "TestStrategy"
        }
        assegna_premio(trade)
        assert get_total_awards() > 0
        
        reset_awards()
        assert get_total_awards() == 0
        assert get_awards() == {}
        
    def test_reset_registro(self):
        """Test resetting the registry"""
        # Add a trade
        trade = {
            "asset": "BTC",
            "tipo": "BUY",
            "quantita": 0.1,
            "prezzo": 25000,
            "commissione": 5,
            "api_usata": "TestAPI",
            "strategy": "TestStrategy"
        }
        registra_trade(trade)
        
        # Check that balance is not zero (due to commission)
        assert get_balance() == -5.0
        
        # Reset registry
        reset_registro()
        
        # Check that balance is reset to zero
        assert get_balance() == 0.0
        
        # Check that positions are cleared
        assert get_posizione("BTC") is None
        
        # Check that awards are reset
        assert get_total_awards() == 0
        
    def test_report_functions(self):
        """Test report functions (they just log, so we check they don't crash)"""
        # These functions should not raise exceptions
        report_giornaliero()
        report_premi()
        
        # Add some data and test again
        trade = {
            "asset": "BTC",
            "tipo": "BUY",
            "quantita": 0.1,
            "prezzo": 25000,
            "commissione": 5,
            "api_usata": "TestAPI",
            "strategy": "TestStrategy"
        }
        registra_trade(trade)
        
        report_giornaliero()
        report_premi()
        
    def test_trade_and_position_dataclasses(self):
        """Test Trade and Position dataclasses"""
        # Test Trade
        trade = Trade(
            timestamp="2023-01-01T12:00:00",
            asset="BTC",
            tipo="BUY",
            quantita=0.1,
            prezzo=25000.0,
            commissione=5.0,
            profit_loss=-5.0,
            saldo_totale=100.0,
            api_usata="TestAPI",
            punteggio_premio=-15.0,
            strategy="TestStrategy",
            prezzo_acquisto=25000.0
        )
        
        assert trade.asset == "BTC"
        assert trade.tipo == "BUY"
        assert trade.quantita == 0.1
        assert trade.prezzo == 25000.0
        assert trade.commissione == 5.0
        assert trade.profit_loss == -5.0
        assert trade.saldo_totale == 100.0
        assert trade.api_usata == "TestAPI"
        assert trade.punteggio_premio == -15.0
        assert trade.strategy == "TestStrategy"
        assert trade.prezzo_acquisto == 25000.0
        
        # Test Position
        position = Position(
            asset="ETH",
            quantita=1.0,
            prezzo_acquisto=1800.0,
            data_acquisto="2023-01-01T12:00:00",
            api_usata="TestAPI",
            strategy="TestStrategy"
        )
        
        assert position.asset == "ETH"
        assert position.quantita == 1.0
        assert position.prezzo_acquisto == 1800.0
        assert position.data_acquisto == "2023-01-01T12:00:00"
        assert position.api_usata == "TestAPI"
        assert position.strategy == "TestStrategy"
        
    def test_set_data_dir(self):
        """Test setting custom data directory"""
        custom_dir = os.path.join(self.test_dir, "custom")
        set_data_dir(custom_dir)
        
        # Check that the directory was created
        assert os.path.exists(custom_dir)
        
        # Initialize registry in the new directory
        inizializza_registro()
        
        # Check that files are in the custom directory
        assert os.path.exists(os.path.join(custom_dir, "registro_trading.csv"))
        assert os.path.exists(os.path.join(custom_dir, "saldo.txt"))
        assert os.path.exists(os.path.join(custom_dir, "posizioni.json"))
        assert os.path.exists(os.path.join(custom_dir, "premi.json"))
