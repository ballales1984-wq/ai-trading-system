import os
file_path = r'c:\ai-trading-system\main_auto_trader.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

target = '''        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):'''

replacement = '''        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # HFT Setup (Virtual Wallet Concept)
        self.hft_queue = queue.Queue()
        self.hft_engine = None
        self.hft_virtual_balance = config.initial_balance * config.hft_budget_pct if config.enable_hft else 0.0
        
        if config.enable_hft:
            logger.info(f"Initialized HFT Engine with sub-wallet: {self.hft_virtual_balance} USDT")
            try:
                from src.hft.hft_trading_engine import create_hft_engine
                # Usiamo il primo asset come default per HFT
                first_asset = config.assets[0] if config.assets else "BTCUSDT"
                self.hft_engine = create_hft_engine(symbol=first_asset, initial_price=100) # Mock price
                self.hft_engine.set_callbacks(on_signal=self._on_hft_signal)
            except Exception as e:
                logger.error(f"Failed to start HFT Engine: {e}")

    def _on_hft_signal(self, signal):
        """Callback for HFT."""
        self.hft_queue.put(signal)
    
    def _signal_handler(self, signum, frame):'''

# Normalizziamo newlines
content = content.replace('\r\n', '\n')
target = target.replace('\r\n', '\n')
replacement = replacement.replace('\r\n', '\n')

print("Target in content:", target in content)
if target in content:
    content = content.replace(target, replacement)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Success")
else:
    print("Failed")
