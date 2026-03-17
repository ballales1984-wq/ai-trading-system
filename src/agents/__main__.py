"""
Autonomous Quant Agent CLI Entry Point
=======================================
Command-line interface for the Level 5 Autonomous Quant Agent.

Usage:
    # Daily report
    python -m src.agents BTCUSDT
    
    # Full analysis with action proposals
    python -m src.agents BTCUSDT --proposals
    
    # Multi-symbol report
    python -m src.agents BTCUSDT,ETHUSDT,SOLUSDT
    
    # Portfolio status
    python -m src.agents --status
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import List, Optional

from src.agents.autonomous_quant_agent import (
    AutonomousQuantAgent,
    AgentConfig,
)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_report(report: dict) -> None:
    """Print the daily report in a formatted way."""
    print_header("AUTONOMOUS AGENT DAILY REPORT")
    
    # Timestamp
    print(f"\n📅 Generated: {report.get('timestamp', 'N/A')}")
    print(f"🕐 Trading Mode: {report.get('trading_mode', 'N/A')}")
    
    # Market Regime
    regime = report.get("regime", {})
    print("\n📊 MARKET REGIME")
    print(f"   Regime: {regime.get('regime', 'unknown').upper()}")
    print(f"   Confidence: {regime.get('confidence', 0):.2%}")
    print(f"   Volatility: {regime.get('volatility', 0):.2%}")
    
    # Monte Carlo
    mc = report.get("monte_carlo", {})
    print("\n🎲 MONTE CARLO SIMULATION")
    print(f"   Mean Price: ${mc.get('mean_price', 0):,.2f}")
    print(f"   Median Price: ${mc.get('median_price', 0):,.2f}")
    p5 = mc.get('percentile_5', 0)
    p95 = mc.get('percentile_95', 0)
    print(f"   5th-95th Percentile: ${p5:,.2f} - ${p95:,.2f}")
    
    # Portfolio
    portfolio = report.get("portfolio", {})
    print("\n💼 PORTFOLIO STATUS")
    print(f"   Equity: ${portfolio.get('equity', 0):,.2f}")
    print(f"   P&L: ${portfolio.get('pnl', 0):,.2f} ({portfolio.get('pnl_pct', 0):.2%})")
    print(f"   Positions: {portfolio.get('position_count', 0)}")
    
    # Risk
    risk = report.get("risk", {})
    print("\n⚠️ RISK STATUS")
    print(f"   Within Limits: {'✅ YES' if risk.get('within_limits') else '❌ NO'}")
    print(f"   VaR (95%): {risk.get('var_95', 0):.2%}")
    print(f"   CVaR (95%): {risk.get('cvar_95', 0):.2%}")
    print(f"   Drawdown: {risk.get('drawdown_pct', 0):.2%}")
    
    # Models
    models = report.get("models", {})
    print("\n🤖 ML MODELS")
    champion = models.get("champion")
    if champion:
        print(f"   Champion: {champion.get('name', 'N/A')}")
        print(f"   Version: {champion.get('version', 'N/A')}")
        print(f"   Metrics: {champion.get('metrics', {})}")
    else:
        print("   No champion model registered")


def print_proposals(proposals: List[dict]) -> None:
    """Print action proposals in a formatted way."""
    print_header("ACTION PROPOSALS")
    
    if not proposals:
        print("\n   No proposals generated.")
        return
    
    for i, proposal in enumerate(proposals, 1):
        print(f"\n{i}. {proposal.get('action', 'N/A').upper()} {proposal.get('symbol', 'N/A')}")
        print(f"   Size: {proposal.get('size', 0):.4f}")
        print(f"   Reason: {proposal.get('reason', 'N/A')}")
        print(f"   Confidence: {proposal.get('confidence', 0):.2%}")
        print(f"   Risk Score: {proposal.get('risk_score', 0):.2f}")


def print_portfolio_status(status: dict) -> None:
    """Print portfolio status."""
    print_header("PORTFOLIO STATUS")
    
    print(f"\n💰 Equity: ${status.get('equity', 0):,.2f}")
    print(f"📈 Initial: ${status.get('initial_equity', 0):,.2f}")
    print(f"📊 P&L: ${status.get('total_pnl', 0):,.2f} ({status.get('pnl_pct', 0):.2%})")
    print(f"📁 Positions: {status.get('position_count', 0)}")
    print(f"🔢 Decisions: {status.get('decision_count', 0)}")


async def run_daily_report(symbols: List[str], proposals: bool = False) -> None:
    """Run the daily report for given symbols."""
    print("\n🚀 Initializing Autonomous Quant Agent...")
    
    config = AgentConfig(
        default_symbols=symbols,
        max_position_pct=0.10,
        max_drawdown_pct=0.05,
    )
    
    agent = AutonomousQuantAgent(config)
    
    # Get report for first symbol
    symbol = symbols[0] if symbols else "BTCUSDT"
    
    print(f"\n📊 Analyzing {symbol}...")
    report = agent.daily_report(symbol)
    
    print_report(report)
    
    if proposals:
        print("\n📋 Generating action proposals...")
        actions = agent.propose_actions(symbol)
        print_proposals(actions)
    
    print("\n✅ Analysis complete.")


async def run_portfolio_status(agent: AutonomousQuantAgent) -> None:
    """Run portfolio status check."""
    status = agent.get_portfolio_status()
    print_portfolio_status(status)


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    symbols = args.symbols.split(",") if args.symbols else ["BTCUSDT"]
    
    if args.status:
        # Portfolio status only
        config = AgentConfig(default_symbols=symbols)
        agent = AutonomousQuantAgent(config)
        await run_portfolio_status(agent)
    else:
        # Daily report
        await run_daily_report(symbols, proposals=args.proposals)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Quant Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.agents BTCUSDT
  python -m src.agents BTCUSDT --proposals
  python -m src.agents BTCUSDT,ETHUSDT,SOLUSDT
  python -m src.agents --status
        """
    )
    
    parser.add_argument(
        "symbols",
        nargs="?",
        default="BTCUSDT",
        help="Comma-separated list of symbols (default: BTCUSDT)"
    )
    
    parser.add_argument(
        "--proposals",
        action="store_true",
        help="Generate action proposals"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show portfolio status"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
