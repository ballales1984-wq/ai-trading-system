#!/usr/bin/env python3
"""
Database Migration Runner Script
Handles Alembic migrations for the AI Trading System
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd: list, capture_output: bool = False) -> tuple:
    """Run a shell command and return result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=capture_output,
        text=True
    )
    return result.returncode, result.stdout if capture_output else "", result.stderr if capture_output else ""


def check_alembic_installed() -> bool:
    """Check if Alembic is installed."""
    try:
        import alembic
        return True
    except ImportError:
        print("❌ Alembic not installed. Run: pip install alembic")
        return False


def get_current_revision() -> str:
    """Get current database revision."""
    code, stdout, _ = run_command(["alembic", "current"], capture_output=True)
    if code == 0:
        return stdout.strip()
    return "Unknown"


def create_migration(message: str = "auto_migration") -> bool:
    """Create a new migration."""
    print(f"\n📝 Creating new migration: {message}")
    code, _, _ = run_command(["alembic", "revision", "--autogenerate", "-m", message])
    return code == 0


def upgrade_database(revision: str = "head") -> bool:
    """Upgrade database to specified revision."""
    print(f"\n⬆️ Upgrading database to: {revision}")
    code, _, _ = run_command(["alembic", "upgrade", revision])
    return code == 0


def downgrade_database(revision: str = "-1") -> bool:
    """Downgrade database by specified steps."""
    print(f"\n⬇️ Downgrading database: {revision}")
    code, _, _ = run_command(["alembic", "downgrade", revision])
    return code == 0


def show_history() -> bool:
    """Show migration history."""
    print("\n📜 Migration History:")
    code, _, _ = run_command(["alembic", "history"])
    return code == 0


def show_current() -> bool:
    """Show current revision."""
    print("\n📍 Current Revision:")
    code, _, _ = run_command(["alembic", "current"])
    return code == 0


def reset_database() -> bool:
    """Reset database to base (downgrade all)."""
    print("\n🔄 Resetting database...")
    code, _, _ = run_command(["alembic", "downgrade", "base"])
    return code == 0


def stamp_database(revision: str = "head") -> bool:
    """Stamp database to specific revision without running migrations."""
    print(f"\n🏷️ Stamping database to: {revision}")
    code, _, _ = run_command(["alembic", "stamp", revision])
    return code == 0


def init_database() -> bool:
    """Initialize database with all migrations."""
    print("\n🚀 Initializing database...")
    
    # Check if database exists
    db_path = PROJECT_ROOT / "data" / "ml_trading.db"
    if db_path.exists():
        print(f"Database exists at: {db_path}")
        response = input("Database already exists. Reset and reinitialize? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return False
        # Remove existing database
        os.remove(db_path)
        print("Removed existing database.")
    
    # Run migrations
    return upgrade_database("head")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Database Migration Runner for AI Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_migrations.py init          # Initialize fresh database
  python scripts/run_migrations.py upgrade       # Upgrade to latest
  python scripts/run_migrations.py downgrade     # Downgrade by 1
  python scripts/run_migrations.py create "msg"  # Create new migration
  python scripts/run_migrations.py history       # Show migration history
  python scripts/run_migrations.py status        # Show current status
  python scripts/run_migrations.py reset         # Reset to base
        """
    )
    
    parser.add_argument(
        "command",
        choices=["init", "upgrade", "downgrade", "create", "history", "status", "reset", "stamp"],
        help="Migration command to run"
    )
    
    parser.add_argument(
        "-m", "--message",
        default=None,
        help="Migration message (for create command)"
    )
    
    parser.add_argument(
        "-r", "--revision",
        default=None,
        help="Target revision (for upgrade/downgrade/stamp)"
    )
    
    args = parser.parse_args()
    
    if not check_alembic_installed():
        sys.exit(1)
    
    print("=" * 60)
    print("🗄️ AI Trading System - Database Migration Runner")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    success = False
    
    if args.command == "init":
        success = init_database()
    elif args.command == "upgrade":
        revision = args.revision or "head"
        success = upgrade_database(revision)
    elif args.command == "downgrade":
        revision = args.revision or "-1"
        success = downgrade_database(revision)
    elif args.command == "create":
        message = args.message or f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        success = create_migration(message)
    elif args.command == "history":
        success = show_history()
    elif args.command == "status":
        success = show_current()
    elif args.command == "reset":
        success = reset_database()
    elif args.command == "stamp":
        revision = args.revision or "head"
        success = stamp_database(revision)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Migration completed successfully!")
    else:
        print("❌ Migration failed!")
    print("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
