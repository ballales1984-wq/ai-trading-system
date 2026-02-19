#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Commodity Trading System - Main Entry Point
Experimental system for crypto + commodity-linked trading signals + auto trading
"""

import sys
import io
import os
import argparse
import logging
from datetime import datetime

# Fix Unicode output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import config

from data_collector import DataCollector
from technical_analysis import TechnicalAnalyzer
from sentiment_news import SentimentAnalyzer
from decision_engine import DecisionEngine
from dashboard import TradingDashboard, print_dashboard_summary
from auto_trader import AutoTradingBot
from trading_simulator import TradingSimulator
from live_multi_asset import LiveMultiAssetTrader


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, config.LOGGING_CONFIG['level']),
        format=config.LOGGING_CONFIG['format'],
        handlers=[logging.StreamHandler()]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crypto Commodity Trading System"
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['signals', 'analysis', 'dashboard', 'test', 'auto', 'backtest', 'simulate', 'portfolio', 'live', 'menu'],
        default='menu',
        help='Execution mode'
    )
    
    # Portfolio control arguments
    parser.add_argument('--portfolio-action', type=str, 
                        choices=['check', 'close', 'close-all', 'reset', 'analyze', 'history'],
                        help='Portfolio action to perform')
    parser.add_argument('--symbol', '-s', type=str, default=None)
    parser.add_argument('--stop-loss', type=float, help='Set stop-loss percentage')
    parser.add_argument('--take-profit', type=float, help='Set take-profit percentage')
    parser.add_argument('--position-size', type=float, help='Set position size percentage')
    parser.add_argument('--exchange', '-e', type=str, default='binance')
    # Simulation mode - default from config, can be overridden with --simulation or --no-simulation
    parser.add_argument('--simulation', '-sim', action='store_true', default=None,
                        help='Use simulation mode (default: based on config.py)')
    parser.add_argument('--no-simulation', dest='simulation', action='store_false',
                        help='Use real API data (ignores config)')
    parser.add_argument('--dashboard', '-d', action='store_true')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', '-p', type=int, default=8050)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--balance', type=float, default=10000.0)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--duration', type=int, default=60, help='Duration for simulate mode (seconds)')
    parser.add_argument('--assets', type=str, default='BTCUSDT,ETHUSDT,SOLUSDT', help='Comma-separated list of assets for live mode')
    parser.add_argument('--allocation', type=str, default='equal_weight', choices=['equal_weight', 'volatility_parity', 'risk_parity', 'momentum'], help='Portfolio allocation strategy')
    parser.add_argument('--interval', type=str, default='1m', choices=['1m', '5m', '15m', '1h', '4h', '1d'], help='Kline interval for live mode')
    parser.add_argument('--telegram-token', type=str, default='', help='Telegram bot token for notifications')
    parser.add_argument('--telegram-chat-id', type=str, default='', help='Telegram chat ID for notifications')
    
    return parser.parse_args()


def run_signals_mode(args):
    print("\n" + "="*70)
    print("CRYPTO + COMMODITY TRADING SYSTEM")
    print("Generating Trading Signals")
    print("="*70 + "\n")
    
    collector = DataCollector(exchange=args.exchange, simulation=args.simulation)
    engine = DecisionEngine(collector)
    
    symbols = [args.symbol] if args.symbol else collector.get_supported_symbols()
    
    print(f"📊 Analyzing {len(symbols)} assets...\n")
    signals = engine.generate_signals(symbols)
    
    print(engine.generate_signal_report(signals))
    
    print("\n🏆 TOP OPPORTUNITIES:\n")
    for signal in engine.get_top_signals(signals, 5):
        if signal.action != 'HOLD':
            print(engine.format_signal_display(signal))
    
    output_file = config.DATA_DIR / f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    engine.export_signals(signals, str(output_file))
    print(f"\n💾 Signals exported to: {output_file}")


def run_analysis_mode(args):
    print("\n" + "="*70)
    print("DETAILED TECHNICAL ANALYSIS")
    print("="*70 + "\n")
    
    if not args.symbol:
        print("Error: --symbol required")
        return
    
    collector = DataCollector(exchange=args.exchange, simulation=args.simulation)
    analyzer = TechnicalAnalyzer()
    sentiment = SentimentAnalyzer()
    
    symbol = args.symbol
    
    market_data = collector.fetch_market_data(symbol)
    print(f"💰 Current Price: ${market_data.current_price:,.2f}")
    print(f"📈 24h Change: {market_data.price_change_percent_24h:+.2f}%\n")
    
    df = collector.fetch_ohlcv(symbol, config.DEFAULT_TIMEFRAME, 100)
    
    if df is not None and not df.empty:
        analysis = analyzer.analyze(df, symbol)
        
        print(f"📈 TREND: {analysis.trend.upper()}")
        print(f"   RSI: {analysis.rsi:.2f} ({analysis.rsi_signal})")
        print(f"   MACD Histogram: {analysis.macd_histogram:.4f}")
        print(f"   Technical Score: {analysis.technical_score:.1%}")
    
    asset_name = symbol.split('/')[0]
    sent = sentiment.get_combined_sentiment(asset_name)
    print(f"\n💭 Sentiment: {sent['combined_score']:+.2f}")
    print(f"   Fear/Greed: {sent['social_sentiment']['fear_greed_index']}")


def run_dashboard_mode(args):
    print("\n" + "="*70)
    print("STARTING INTERACTIVE DASHBOARD")
    print("="*70)
    print(f"\n🌐 http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")
    
    # Check if user wants API-integrated dashboard
    use_api_dashboard = os.getenv('USE_API_DASHBOARD', 'false').lower() == 'true'
    
    if use_api_dashboard:
        try:
            from dashboard_api import app
            print("Using API-integrated dashboard (connects to FastAPI)")
            app.run_server(debug=args.debug, host=args.host, port=args.port)
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("Falling back to standard dashboard...")
            use_api_dashboard = False
    
    if not use_api_dashboard:
        try:
            dashboard = TradingDashboard(debug=args.debug)
            dashboard.run(host=args.host, port=args.port, debug=args.debug)
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("Install dash: pip install dash plotly\n")
            print_dashboard_summary()


def run_test_mode(args):
    print("\n" + "="*70)
    print("RUNNING SYSTEM TESTS")
    print("="*70 + "\n")
    
    # Test Data Collector
    print("🔄 Test 1: Data Collector")
    try:
        collector = DataCollector(simulation=True)
        price = collector.fetch_current_price('BTC/USDT')
        print(f"  ✓ BTC: ${price:,.2f}")
        
        df = collector.fetch_ohlcv('ETH/USDT', '1h', 50)
        print(f"  ✓ OHLCV: {len(df)} candles")
        
        corr = collector.calculate_correlation('BTC/USDT', 'ETH/USDT', 24)
        print(f"  ✓ Correlation: {corr.correlation:.4f}")
        print("  ✅ PASSED\n")
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
    
    # Test Technical Analysis
    print("🔄 Test 2: Technical Analysis")
    try:
        analyzer = TechnicalAnalyzer()
        df = collector.fetch_ohlcv('BTC/USDT', '1h', 100)
        analysis = analyzer.analyze(df, 'BTC/USDT')
        print(f"  ✓ RSI: {analysis.rsi:.2f}")
        print(f"  ✓ Score: {analysis.technical_score:.1%}")
        print("  ✅ PASSED\n")
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
    
    # Test Sentiment
    print("🔄 Test 3: Sentiment Analysis")
    try:
        sentiment = SentimentAnalyzer()
        sent = sentiment.get_combined_sentiment('Bitcoin')
        print(f"  ✓ Score: {sent['combined_score']:.2f}")
        print(f"  ✓ Fear/Greed: {sent['social_sentiment']['fear_greed_index']}")
        print("  ✅ PASSED\n")
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
    
    # Test Decision Engine
    print("🔄 Test 4: Decision Engine")
    try:
        engine = DecisionEngine(collector, sentiment)
        signals = engine.generate_signals()
        print(f"  ✓ Signals: {len(signals)}")
        buy = len([s for s in signals if s.action == 'BUY'])
        sell = len([s for s in signals if s.action == 'SELL'])
        print(f"  ✓ Buy: {buy}, Sell: {sell}")
        print("  ✅ PASSED\n")
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
    
    print("="*70)
    print("ALL TESTS COMPLETED")
    print("="*70 + "\n")


def run_backtest_mode(args):
    print("\n" + "="*70)
    print("BACKTEST MODE")
    print("="*70 + "\n")
    
    print(f"📊 Running backtest...")
    print(f"   Initial Balance: ${args.balance:,.2f}")
    print(f"   Days: {args.days}")
    print()
    
    bot = AutoTradingBot(initial_balance=args.balance, paper_trading=True)
    results = bot.backtest(days=args.days)
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"\n💰 Initial:  ${results['initial_balance']:,.2f}")
    print(f"💵 Final:    ${results['final_balance']:,.2f}")
    print(f"📈 Return:   {results['total_return']:.2f}%")
    print(f"📊 Trades:   {results['total_trades']}")
    print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
    
    output_file = config.DATA_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    bot.export_results(str(output_file))
    print(f"\n💾 Saved to: {output_file}")


def run_auto_mode(args):
    print("\n" + "="*70)
    print("AUTO TRADING BOT")
    print("="*70 + "\n")
    
    print(f"🤖 Starting Auto Trading Bot...")
    print(f"   Mode: PAPER TRADING")
    print(f"   Balance: ${args.balance:,.2f}")
    print(f"   Stop Loss: 2%")
    print(f"   Take Profit: 5%")
    print("\n⚠️  Press Ctrl+C to stop\n")
    
    bot = AutoTradingBot(initial_balance=args.balance, paper_trading=True)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")
        bot.stop()
        
        print("\n" + "="*50)
        print("SESSION RESULTS")
        print("="*50)
        print(f"💵 Final:   ${bot.portfolio.get_total_value():,.2f}")
        print(f"📈 PnL:     ${bot.portfolio.total_pnl:,.2f}")
        print(f"📊 Trades:  {bot.portfolio.trade_count}")
        print(f"🎯 Win Rate: {bot.portfolio.get_win_rate():.1f}%")


def run_simulate_mode(args):
    """Run trading simulation"""
    from trading_simulator import TradingSimulator
    
    print("\n" + "="*70)
    print("TRADING SIMULATOR")
    print("="*70 + "\n")
    
    print(f"🎮 Starting Trading Simulation...")
    print(f"   Initial Balance: ${args.balance:,.2f}")
    print(f"   Duration: {args.duration} seconds")
    print(f"   Position Size: 20%")
    print(f"   Stop Loss: 2%")
    print(f"   Take Profit: 5%")
    print()
    
    simulator = TradingSimulator(initial_balance=args.balance)
    
    try:
        simulator.start(duration_seconds=args.duration)
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation stopped by user")
        simulator._print_final_results()


def run_portfolio_mode(args):
    """Run portfolio control commands"""
    from trading_simulator import TradingSimulator
    
    print("\n" + "="*70)
    print("PORTFOLIO CONTROL")
    print("="*70 + "\n")
    
    # Initialize simulator (loads saved state if available)
    simulator = TradingSimulator(initial_balance=args.balance)
    
    # Handle settings changes first
    if args.stop_loss is not None:
        result = simulator.set_stop_loss(args.stop_loss)
        print(f"{'✅' if result['status'] == 'SUCCESS' else '❌'} {result['message']}")
    
    if args.take_profit is not None:
        result = simulator.set_take_profit(args.take_profit)
        print(f"{'✅' if result['status'] == 'SUCCESS' else '❌'} {result['message']}")
    
    if args.position_size is not None:
        result = simulator.set_position_size(args.position_size)
        print(f"{'✅' if result['status'] == 'SUCCESS' else '❌'} {result['message']}")
    
    # Handle portfolio actions
    action = args.portfolio_action
    
    if action == 'check' or action is None:
        # Check portfolio status
        status = simulator.check_portfolio()
        print("\n📊 PORTFOLIO STATUS:")
        print("-" * 40)
        print(f"💰 Balance:        ${status['balance']:,.2f}")
        print(f"📈 Total Value:    ${status['total_value']:,.2f}")
        print(f"💵 Total P&L:     ${status['total_pnl']:+,.2f} ({status['pnl_percent']:+,.2f}%)")
        print(f"📋 Open Positions: {status['open_positions']}/{status['max_positions']}")
        print(f"🎯 Win Rate:      {status['win_rate']:.1f}%")
        print(f"📊 Total Trades:  {status['trade_count']}")
        print("\n⚙️ SETTINGS:")
        print("-" * 40)
        print(f"Position Size:  {status['settings']['position_size_percent']:.1f}%")
        print(f"Stop Loss:      {status['settings']['stop_loss_percent']:.1f}%")
        print(f"Take Profit:    {status['settings']['take_profit_percent']:.1f}%")
        
        if status['positions_detail']:
            print("\n📋 OPEN POSITIONS:")
            print("-" * 40)
            for sym, pos in status['positions_detail'].items():
                pnl = (pos['current_price'] - pos['entry_price']) * pos['quantity']
                pnl_pct = ((pos['current_price'] - pos['entry_price']) / pos['entry_price']) * 100
                print(f"  {sym}: ${pos['current_price']:,.2f} ({pnl_pct:+.2f}%)")
    
    elif action == 'close':
        if not args.symbol:
            print("❌ Error: --symbol required to close a position")
            return
        result = simulator.close_position(args.symbol)
        print(f"{'✅' if result['status'] == 'SUCCESS' else '❌'} {result['message']}")
        if result['status'] == 'SUCCESS':
            print(f"   Price: ${result['price']:,.2f}")
            print(f"   P&L: ${result['pnl']:+,.2f}")
    
    elif action == 'close-all':
        result = simulator.close_all_positions()
        print(f"✅ Closed {result['total_closed']} positions")
        for pos in result['closed_positions']:
            print(f"   {pos['message']} - P&L: ${pos['pnl']:+,.2f}")
    
    elif action == 'reset':
        result = simulator.reset_portfolio()
        print(f"✅ {result['message']}")
        print(f"   New Balance: ${result['new_balance']:,.2f}")
    
    elif action == 'analyze':
        analysis = simulator.get_portfolio_analysis()
        if analysis['status'] == 'NO_DATA':
            print("📊 No trades to analyze yet.")
        else:
            print("\n📊 PORTFOLIO ANALYSIS:")
            print("-" * 40)
            print(f"Total Trades:    {analysis['total_trades']}")
            print(f"Winning Trades:  {analysis['winning_trades']}")
            print(f"Losing Trades:   {analysis['losing_trades']}")
            print(f"Win Rate:        {analysis['win_rate']:.1f}%")
            print(f"Total P&L:       ${analysis['total_pnl']:+,.2f}")
            print(f"Average Win:     ${analysis['average_win']:+,.2f}")
            print(f"Average Loss:    ${analysis['average_loss']:,.2f}")
            print(f"Profit Factor:   {analysis['profit_factor']:.2f}")
            print(f"Best Trade:      ${analysis['best_trade']:+,.2f}")
            print(f"Worst Trade:     ${analysis['worst_trade']:+,.2f}")
    
    elif action == 'history':
        trades = simulator.get_trade_history()
        if not trades:
            print("📜 No trade history yet.")
        else:
            print("\n📜 TRADE HISTORY:")
            print("-" * 60)
            for t in trades:
                print(f"  {t['timestamp']} | {t['symbol']} | {t['action']} | "
                      f"${t['price']:,.2f} | Qty: {t['quantity']:.4f} | P&L: ${t['pnl']:+,.2f}")


def run_live_multi_asset_mode(args):
    """Run live multi-asset trading"""
    from live_multi_asset import LiveMultiAssetTrader
    import signal
    
    # Parse assets
    assets = args.assets.split(',') if hasattr(args, 'assets') else ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    assets = [a.strip().upper() for a in assets]
    
    # Get strategy
    strategy = getattr(args, 'allocation', 'equal_weight')
    
    print("\n" + "="*70)
    print("🚀 LIVE MULTI-ASSET TRADING SYSTEM")
    print("="*70)
    print(f"\n📈 Assets: {', '.join(assets)}")
    print(f"💵 Capital: ${args.balance:,.2f}")
    print(f"⏱️  Interval: {getattr(args, 'interval', '1m')}")
    print(f"⚖️  Strategy: {strategy}")
    print("\n" + "="*70)
    
    # Create trader
    trader = LiveMultiAssetTrader(
        assets=assets,
        initial_capital=args.balance,
        interval=getattr(args, 'interval', '1m'),
        allocation_strategy=strategy,
        testnet=getattr(args, 'testnet', False),
        paper_trading=not getattr(args, 'live', False),
        telegram_bot_token=getattr(args, 'telegram_token', '') or None,
        telegram_chat_id=getattr(args, 'telegram_chat_id', '') or None
    )
    
    # Handle signals
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down...")
        trader.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start trading
    trader.start()


def main():
    args = parse_args()
    setup_logging()
    
    # Determine simulation mode: use args value if provided, otherwise use config
    if args.simulation is None:
        # Use config.py default
        args.simulation = config.SIMULATION_MODE
    
    print(f"\n>>> SIMULATION MODE: {args.simulation}")
    
    # Default to menu mode if no specific mode is requested
    if args.mode == 'menu' or len(sys.argv) == 1:
        # Menu mode - inline implementation
        while True:
            print("\n" + "="*70)
            print("🚀 QUANTUM AI TRADING SYSTEM - MAIN MENU")
            print("="*70)
            print("""
╔═══════════════════════════════════════════════════════════════════╗
║                     AVAILABLE OPTIONS                            ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  [1] 📊 SIGNALS          Generate trading signals for all assets ║
║  [2] 📈 ANALYSIS         Detailed technical analysis             ║
║  [3] 📉 BACKTEST         Run backtest with historical data       ║
║  [4] 🎮 SIMULATE         Run trading simulation                 ║
║  [5] 🤖 AUTO TRADE       Start auto trading bot (paper)         ║
║  [6] 💼 PORTFOLIO        Check portfolio status                  ║
║  [7] 🖥️  DASHBOARD        Launch interactive dashboard           ║
║  [8] 🔴 LIVE TRADING     Start live multi-asset trading          ║
║  [9] ✅ TEST             Run system tests                        ║
║                                                                   ║
║  [0] ❌ EXIT             Exit the program                        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")
            
            choice = input("👉 Select an option [0-9]: ").strip()
            
            if choice == '1':
                print("\n" + "="*50)
                print("Generating Trading Signals...")
                print("="*50)
                run_signals_mode(args)
                input("\n⏎ Press Enter to continue...")
                
            elif choice == '2':
                print("\n📈 ANALYSIS MODE")
                symbol = input("   Enter symbol (e.g., BTC/USDT): ").strip().upper()
                if symbol:
                    args.symbol = symbol
                    run_analysis_mode(args)
                else:
                    print("   ⚠️ Symbol required!")
                input("\n⏎ Press Enter to continue...")
                
            elif choice == '3':
                print("\n📉 BACKTEST MODE")
                try:
                    days_input = input("   Days to backtest [30]: ").strip()
                    balance_input = input("   Initial balance [10000]: ").strip()
                    args.days = int(days_input) if days_input else 30
                    args.balance = float(balance_input) if balance_input else 10000.0
                except ValueError:
                    print("   ⚠️ Invalid input, using defaults")
                    args.days = 30
                    args.balance = 10000.0
                run_backtest_mode(args)
                input("\n⏎ Press Enter to continue...")
                
            elif choice == '4':
                print("\n🎮 SIMULATION MODE")
                try:
                    duration_input = input("   Duration in seconds [60]: ").strip()
                    balance_input = input("   Initial balance [10000]: ").strip()
                    args.duration = int(duration_input) if duration_input else 60
                    args.balance = float(balance_input) if balance_input else 10000.0
                except ValueError:
                    print("   ⚠️ Invalid input, using defaults")
                    args.duration = 60
                    args.balance = 10000.0
                run_simulate_mode(args)
                input("\n⏎ Press Enter to continue...")
                
            elif choice == '5':
                print("\n🤖 AUTO TRADING BOT")
                balance_input = input("   Initial balance [10000]: ").strip()
                try:
                    args.balance = float(balance_input) if balance_input else 10000.0
                except ValueError:
                    args.balance = 10000.0
                run_auto_mode(args)
                input("\n⏎ Press Enter to continue...")
                
            elif choice == '6':
                print("\n💼 PORTFOLIO")
                args.portfolio_action = 'check'
                run_portfolio_mode(args)
                input("\n⏎ Press Enter to continue...")
                
            elif choice == '7':
                print("\n🖥️ LAUNCHING DASHBOARD...")
                print("   Opening browser at http://localhost:8050")
                run_dashboard_mode(args)
                
            elif choice == '8':
                print("\n🔴 LIVE TRADING")
                print("   ⚠️ WARNING: This will connect to live exchange!")
                confirm = input("   Continue? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    assets_input = input("   Assets (comma-separated) [BTCUSDT,ETHUSDT]: ").strip()
                    args.assets = assets_input if assets_input else "BTCUSDT,ETHUSDT"
                    run_live_multi_asset_mode(args)
                else:
                    print("   Cancelled.")
                input("\n⏎ Press Enter to continue...")
                
            elif choice == '9':
                print("\n✅ RUNNING SYSTEM TESTS...")
                run_test_mode(args)
                input("\n⏎ Press Enter to continue...")
                
            elif choice == '0':
                print("\n" + "="*70)
                print("👋 Goodbye! Thank you for using Quantum AI Trading System")
                print("="*70 + "\n")
                break
                
            else:
                print("\n⚠️ Invalid option! Please try again.")
                input("\n⏎ Press Enter to continue...")
        return
    
    try:
        if args.dashboard or args.mode == 'dashboard':
            run_dashboard_mode(args)
        elif args.mode == 'signals':
            run_signals_mode(args)
        elif args.mode == 'analysis':
            run_analysis_mode(args)
        elif args.mode == 'test':
            run_test_mode(args)
        elif args.mode == 'backtest':
            run_backtest_mode(args)
        elif args.mode == 'auto':
            run_auto_mode(args)
        elif args.mode == 'simulate':
            run_simulate_mode(args)
        elif args.mode == 'portfolio':
            run_portfolio_mode(args)
        elif args.mode == 'live':
            run_live_multi_asset_mode(args)
        else:
            print(f"Unknown mode: {args.mode}")
            
    except KeyboardInterrupt:
        print("\n\n👋 Bye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

