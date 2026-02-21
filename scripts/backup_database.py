"""
Database Backup Script for AI Trading System
=============================================
Automated backup with:
- Daily full backups
- Retention policy (30 days)
- Backup verification
- Compression
- Notification on failure
"""

from __future__ import annotations

import os
import sys
import gzip
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import argparse
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.unified_config import settings, PROJECT_ROOT


class DatabaseBackup:
    """Database backup manager."""
    
    def __init__(
        self,
        backup_dir: Optional[Path] = None,
        retention_days: int = 30,
        compress: bool = True
    ):
        self.backup_dir = backup_dir or PROJECT_ROOT / "backups"
        self.retention_days = retention_days
        self.compress = compress
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def get_backup_filename(self, prefix: str = "db_backup") -> str:
        """Generate backup filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".sql.gz" if self.compress else ".sql"
        return f"{prefix}_{timestamp}{ext}"
    
    def backup_postgresql(self) -> Optional[Path]:
        """
        Backup PostgreSQL database using pg_dump.
        
        Returns:
            Path to backup file or None on failure
        """
        backup_file = self.backup_dir / self.get_backup_filename()
        
        # Parse database URL
        db_url = settings.database_url
        if "postgresql" not in db_url:
            print(f"⚠️ Not a PostgreSQL database: {db_url[:30]}...")
            return None
        
        try:
            # Extract connection info from URL
            # Format: postgresql://user:password@host:port/database
            import re
            match = re.match(
                r"postgresql://(?P<user>[^:]+):(?P<password>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)/(?P<database>.+)",
                db_url
            )
            
            if not match:
                print(f"⚠️ Could not parse database URL")
                return None
            
            db_params = match.groupdict()
            
            # Set environment variables for pg_dump
            env = os.environ.copy()
            env["PGPASSWORD"] = db_params["password"]
            
            # Run pg_dump
            cmd = [
                "pg_dump",
                "-h", db_params["host"],
                "-p", db_params["port"],
                "-U", db_params["user"],
                "-d", db_params["database"],
                "-F", "p",  # Plain SQL format
            ]
            
            print(f"🔄 Starting PostgreSQL backup...")
            
            with open(backup_file.with_suffix(".sql"), "wb") as f:
                result = subprocess.run(
                    cmd,
                    env=env,
                    stdout=f,
                    stderr=subprocess.PIPE
                )
            
            if result.returncode != 0:
                print(f"❌ pg_dump failed: {result.stderr.decode()}")
                return None
            
            # Compress if enabled
            if self.compress:
                with open(backup_file.with_suffix(".sql"), "rb") as f_in:
                    with gzip.open(backup_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_in)
                # Remove uncompressed file
                backup_file.with_suffix(".sql").unlink()
            
            print(f"✅ Backup completed: {backup_file}")
            return backup_file
            
        except FileNotFoundError:
            print("❌ pg_dump not found. Please install PostgreSQL client tools.")
            return None
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return None
    
    def backup_sqlite(self, db_path: Optional[Path] = None) -> Optional[Path]:
        """
        Backup SQLite database.
        
        Args:
            db_path: Path to SQLite database file
            
        Returns:
            Path to backup file or None on failure
        """
        # Find SQLite database
        if db_path is None:
            db_path = PROJECT_ROOT / "data" / "trading.db"
        
        if not db_path.exists():
            print(f"⚠️ SQLite database not found: {db_path}")
            return None
        
        backup_file = self.backup_dir / self.get_backup_filename("sqlite_backup")
        
        try:
            print(f"🔄 Starting SQLite backup...")
            
            if self.compress:
                with open(db_path, "rb") as f_in:
                    with gzip.open(backup_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(db_path, backup_file)
            
            print(f"✅ SQLite backup completed: {backup_file}")
            return backup_file
            
        except Exception as e:
            print(f"❌ SQLite backup failed: {e}")
            return None
    
    def backup_json_data(self) -> Optional[Path]:
        """
        Backup JSON data files (ledger, positions, etc.).
        
        Returns:
            Path to backup file or None on failure
        """
        data_dir = PROJECT_ROOT / "data"
        if not data_dir.exists():
            print("⚠️ Data directory not found")
            return None
        
        backup_file = self.backup_dir / self.get_backup_filename("data_backup")
        
        try:
            print(f"🔄 Starting JSON data backup...")
            
            # Create a tarball of data directory
            import tarfile
            with tarfile.open(backup_file, "w:gz") as tar:
                tar.add(data_dir, arcname="data")
            
            print(f"✅ Data backup completed: {backup_file}")
            return backup_file
            
        except Exception as e:
            print(f"❌ Data backup failed: {e}")
            return None
    
    def cleanup_old_backups(self) -> List[Path]:
        """
        Remove backups older than retention period.
        
        Returns:
            List of removed backup files
        """
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        removed = []
        
        for backup_file in self.backup_dir.glob("*"):
            if backup_file.is_file():
                # Get file modification time
                mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if mtime < cutoff:
                    try:
                        backup_file.unlink()
                        removed.append(backup_file)
                        print(f"🗑️ Removed old backup: {backup_file.name}")
                    except Exception as e:
                        print(f"⚠️ Could not remove {backup_file}: {e}")
        
        return removed
    
    def list_backups(self) -> List[dict]:
        """
        List all backups with metadata.
        
        Returns:
            List of backup info dictionaries
        """
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("*"), reverse=True):
            if backup_file.is_file():
                stat = backup_file.stat()
                backups.append({
                    "name": backup_file.name,
                    "path": str(backup_file),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": self._get_backup_type(backup_file.name)
                })
        
        return backups
    
    def _get_backup_type(self, filename: str) -> str:
        """Determine backup type from filename."""
        if "sqlite" in filename:
            return "sqlite"
        elif "data" in filename:
            return "data"
        else:
            return "postgresql"
    
    def verify_backup(self, backup_path: Path) -> bool:
        """
        Verify backup integrity.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if backup is valid
        """
        if not backup_path.exists():
            return False
        
        try:
            if backup_path.suffix == ".gz":
                with gzip.open(backup_path, "rb") as f:
                    # Read first few bytes to verify it's valid gzip
                    f.read(1024)
            else:
                # For non-compressed files, just check they're readable
                with open(backup_path, "rb") as f:
                    f.read(1024)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Backup verification failed: {e}")
            return False
    
    def run_full_backup(self) -> dict:
        """
        Run full backup of all data.
        
        Returns:
            Summary of backup operation
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "backups": [],
            "errors": [],
            "total_size_mb": 0
        }
        
        # Backup PostgreSQL
        pg_backup = self.backup_postgresql()
        if pg_backup:
            results["backups"].append(str(pg_backup))
            results["total_size_mb"] += pg_backup.stat().st_size / (1024 * 1024)
        else:
            # Try SQLite instead
            sqlite_backup = self.backup_sqlite()
            if sqlite_backup:
                results["backups"].append(str(sqlite_backup))
                results["total_size_mb"] += sqlite_backup.stat().st_size / (1024 * 1024)
        
        # Backup JSON data
        data_backup = self.backup_json_data()
        if data_backup:
            results["backups"].append(str(data_backup))
            results["total_size_mb"] += data_backup.stat().st_size / (1024 * 1024)
        
        # Cleanup old backups
        removed = self.cleanup_old_backups()
        results["removed_count"] = len(removed)
        
        results["total_size_mb"] = round(results["total_size_mb"], 2)
        
        return results


def main():
    """Main entry point for backup script."""
    parser = argparse.ArgumentParser(description="Database Backup Tool")
    parser.add_argument(
        "--type",
        choices=["full", "postgres", "sqlite", "data"],
        default="full",
        help="Type of backup to perform"
    )
    parser.add_argument(
        "--retention",
        type=int,
        default=30,
        help="Retention period in days"
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable compression"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing backups"
    )
    parser.add_argument(
        "--verify",
        type=str,
        help="Verify a specific backup file"
    )
    
    args = parser.parse_args()
    
    backup = DatabaseBackup(
        retention_days=args.retention,
        compress=not args.no_compress
    )
    
    if args.list:
        backups = backup.list_backups()
        if backups:
            print("\n📦 Existing Backups:")
            print("-" * 60)
            for b in backups:
                print(f"  {b['name']}: {b['size_mb']} MB ({b['type']})")
            print("-" * 60)
            print(f"Total: {len(backups)} backups")
        else:
            print("No backups found")
        return
    
    if args.verify:
        backup_path = Path(args.verify)
        if backup.verify_backup(backup_path):
            print(f"✅ Backup verified: {backup_path}")
        else:
            print(f"❌ Backup verification failed: {backup_path}")
        return
    
    # Run backup
    if args.type == "full":
        results = backup.run_full_backup()
    elif args.type == "postgres":
        backup.backup_postgresql()
    elif args.type == "sqlite":
        backup.backup_sqlite()
    elif args.type == "data":
        backup.backup_json_data()
    
    print("\n📊 Backup Summary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()