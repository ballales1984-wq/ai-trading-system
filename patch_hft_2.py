import os
file_path = r'c:\ai-trading-system\main_auto_trader.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# patch config
target_config = '''    # Exchange
    exchange: str = "binance"
    testnet: bool = True
    api_key: str = ""
    api_secret: str = ""
    
    def __post_init__(self):'''
repl_config = '''    # Exchange
    exchange: str = "binance"
    testnet: bool = True
    api_key: str = ""
    api_secret: str = ""
    
    # HFT Integration
    enable_hft: bool = False
    hft_budget_pct: float = 0.10
    
    def __post_init__(self):'''

# patch logs
target_logs = '''        logger.info(f"Loop interval: {self.config.loop_interval}s")
        logger.info("=" * 60)'''
repl_logs = '''        logger.info(f"Loop interval: {self.config.loop_interval}s")
        if self.config.enable_hft:
            logger.info(f"HFT Engine: ENABLED (Budget: {self.hft_virtual_balance:.2f} USDT)")
            if self.hft_engine:
                self.hft_engine.start()
        logger.info("=" * 60)'''

# patch stop
target_stop = '''    def stop(self):
        """Ferma il trading bot."""
        logger.info("Stopping AutoTrader...")
        self.running = False
        
        # Stampa statistiche finali'''
repl_stop = '''    def stop(self):
        """Ferma il trading bot."""
        logger.info("Stopping AutoTrader...")
        self.running = False
        
        if self.hft_engine:
            self.hft_engine.stop()
            
        # Stampa statistiche finali'''

# patch args part 1
target_args1 = '''    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Assets to trade"
    )
    
    args = parser.parse_args()'''
repl_args1 = '''    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Assets to trade"
    )
    parser.add_argument(
        "--enable-hft",
        action="store_true",
        help="Enable High Frequency Trading Engine"
    )
    parser.add_argument(
        "--hft-budget-pct",
        type=float,
        default=0.10,
        help="Percentage of balance dedicated to HFT (default: 0.10)"
    )
    
    args = parser.parse_args()'''

target_args2 = '''    # Crea configurazione
    config = TradingConfig(
        loop_interval=args.interval,
        assets=args.assets,
        initial_balance=args.balance,
        dry_run=args.dry_run
    )'''
repl_args2 = '''    # Crea configurazione
    config = TradingConfig(
        loop_interval=args.interval,
        assets=args.assets,
        initial_balance=args.balance,
        dry_run=args.dry_run,
        enable_hft=args.enable_hft,
        hft_budget_pct=args.hft_budget_pct
    )'''

def do_patch(t, r):
    global content
    t = t.replace('\r\n', '\n')
    r = r.replace('\r\n', '\n')
    if t in content:
        content = content.replace(t, r)
        print("Patched one block")
    else:
        print("Target not found block")

content = content.replace('\r\n', '\n')
do_patch(target_config, repl_config)
do_patch(target_logs, repl_logs)
do_patch(target_stop, repl_stop)
do_patch(target_args1, repl_args1)
do_patch(target_args2, repl_args2)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
