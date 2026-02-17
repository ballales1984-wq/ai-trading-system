#!/usr/bin/env python3
"""
Crypto Commodity Trading System - Main Entry Point
Experimental system for crypto + commodity-linked trading signals + auto trading
"""

import sys
import argparse
import logging
from datetime import datetime

import config

from data_collector import DataCollector
from technical_analysis import TechnicalAnalyzer
from sentiment_news import SentimentAnalyzer
from decision_engine import DecisionEngine
from dashboard import TradingDashboard, print_dashboard_summary
from auto_trader import AutoTradingBot


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
        choices=['signals', 'analysis', 'dashboard', 'test', 'auto', 'backtest'],
        default='signals',
        help='Execution mode'
    )
    
    parser.add_argument('--symbol', '-s', type=str, default=None)
    parser.add_argument('--exchange', '-e', type=str, default='binance')
    parser.add_argument('--simulation', '-sim', action='store_true', default=True)
    parser.add_argument('--dashboard', '-d', action='store_true')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', '-p', type=int, default=8050)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--balance', type=float, default=10000.0)
    parser.add_argument('--days', type=int, default=30)
    
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


def main():
    args = parse_args()
    setup_logging()
    
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

