#!/usr/bin/env python3
"""
Database Migration Runner
=========================
Script per gestire le migrazioni Alembic del database.

Usage:
    python scripts/run_migrations.py --help
    python scripts/run_migrations.py status
    python scripts/run_migrations.py upgrade
    python scripts/run_migrations.py downgrade
    python scripts/run_migrations.py create "description"
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_alembic(args: list[str]) -> int:
    """Run alembic command with given arguments."""
    cmd = ["alembic"] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def get_database_url() -> str:
    """Get database URL from environment or return default."""
    return os.environ.get(
        "DATABASE_URL",
        "sqlite:///data/trading.db"
    )


def cmd_status(args):
    """Show current migration status."""
    print("=" * 60)
    print("Database Migration Status")
    print("=" * 60)
    print(f"Database URL: {get_database_url()}")
    print("-" * 60)
    
    # Show current revision
    run_alembic(["current"])
    print()
    
    # Show history
    print("Migration History:")
    run_alembic(["history", "--verbose"])


def cmd_upgrade(args):
    """Upgrade database to latest revision."""
    print("=" * 60)
    print("Upgrading Database")
    print("=" * 60)
    print(f"Database URL: {get_database_url()}")
    print("-" * 60)
    
    if args.revision:
        revision = args.revision
    else:
        revision = "head"
    
    print(f"Upgrading to: {revision}")
    result = run_alembic(["upgrade", revision])
    
    if result == 0:
        print("\n✅ Upgrade completed successfully!")
    else:
        print(f"\n❌ Upgrade failed with code {result}")
    
    return result


def cmd_downgrade(args):
    """Downgrade database to previous revision."""
    print("=" * 60)
    print("Downgrading Database")
    print("=" * 60)
    print(f"Database URL: {get_database_url()}")
    print("-" * 60)
    
    if args.revision:
        revision = args.revision
    else:
        revision = "-1"  # One step back
    
    print(f"Downgrading to: {revision}")
    result = run_alembic(["downgrade", revision])
    
    if result == 0:
        print("\n✅ Downgrade completed successfully!")
    else:
        print(f"\n❌ Downgrade failed with code {result}")
    
    return result


def cmd_create(args):
    """Create a new migration."""
    print("=" * 60)
    print("Creating New Migration")
    print("=" * 60)
    
    if not args.message:
        print("❌ Error: --message is required for creating migrations")
        return 1
    
    cmd_args = ["revision", "--autogenerate", "-m", args.message]
    
    if args.empty:
        cmd_args = ["revision", "-m", args.message]
    
    result = run_alembic(cmd_args)
    
    if result == 0:
        print("\n✅ Migration created successfully!")
        print("Review the generated file in migrations/versions/")
    else:
        print(f"\n❌ Migration creation failed with code {result}")
    
    return result


def cmd_init(args):
    """Initialize database with all tables."""
    print("=" * 60)
    print("Initializing Database")
    print("=" * 60)
    print(f"Database URL: {get_database_url()}")
    print("-" * 60)
    
    # Ensure data directory exists for SQLite
    db_path = PROJECT_ROOT / "data"
    db_path.mkdir(exist_ok=True)
    
    # Run upgrade to head
    result = run_alembic(["upgrade", "head"])
    
    if result == 0:
        print("\n✅ Database initialized successfully!")
    else:
        print(f"\n❌ Database initialization failed with code {result}")
    
    return result


def cmd_reset(args):
    """Reset database (downgrade to base, then upgrade to head)."""
    print("=" * 60)
    print("Resetting Database")
    print("=" * 60)
    print(f"Database URL: {get_database_url()}")
    print("-" * 60)
    
    if not args.force:
        response = input("⚠️  This will delete all data. Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return 1
    
    # Downgrade to base
    print("\n1. Downgrading to base...")
    result = run_alembic(["downgrade", "base"])
    if result != 0:
        print(f"❌ Downgrade failed with code {result}")
        return result
    
    # Upgrade to head
    print("\n2. Upgrading to head...")
    result = run_alembic(["upgrade", "head"])
    if result != 0:
        print(f"❌ Upgrade failed with code {result}")
        return result
    
    print("\n✅ Database reset successfully!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Database Migration Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_migrations.py status
    python scripts/run_migrations.py upgrade
    python scripts/run_migrations.py upgrade --revision a1b2c3d4e5f6
    python scripts/run_migrations.py downgrade
    python scripts/run_migrations.py create --message "add user table"
    python scripts/run_migrations.py init
    python scripts/run_migrations.py reset --force
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    subparsers.add_parser("status", help="Show migration status")
    
    # Upgrade command
    upgrade_parser = subparsers.add_parser("upgrade", help="Upgrade database")
    upgrade_parser.add_argument("--revision", help="Target revision (default: head)")
    
    # Downgrade command
    downgrade_parser = subparsers.add_parser("downgrade", help="Downgrade database")
    downgrade_parser.add_argument("--revision", help="Target revision (default: -1)")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create new migration")
    create_parser.add_argument("--message", "-m", help="Migration message")
    create_parser.add_argument("--empty", action="store_true", help="Create empty migration")
    
    # Init command
    subparsers.add_parser("init", help="Initialize database")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset database")
    reset_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        "status": cmd_status,
        "upgrade": cmd_upgrade,
        "downgrade": cmd_downgrade,
        "create": cmd_create,
        "init": cmd_init,
        "reset": cmd_reset,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
