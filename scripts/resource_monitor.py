#!/usr/bin/env python3
"""
AI Trading System - Resource Monitor
=====================================
Monitors RAM and ROM usage to stay within limits:
- RAM: 4 GB total
- ROM: 3 GB total

Can be run as:
1. Standalone check: python scripts/resource_monitor.py
2. Cron job inside container
3. Integrated with Prometheus metrics
"""

import os
import sys
import shutil
import psutil
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits configuration"""
    MAX_RAM_BYTES: int = 4 * 1024 ** 3  # 4 GB
    MAX_ROM_BYTES: int = 3 * 1024 ** 3  # 3 GB
    
    # Per-volume limits (in bytes)
    VOLUME_LIMITS: Dict[str, int] = None
    
    def __post_init__(self):
        if self.VOLUME_LIMITS is None:
            self.VOLUME_LIMITS = {
                "pgdata": int(1.5 * 1024 ** 3),      # 1.5 GB
                "ml_temp": int(300 * 1024 ** 2),      # 300 MB
                "models": int(200 * 1024 ** 2),       # 200 MB
                "logs": int(500 * 1024 ** 2),         # 500 MB
                "redisdata": int(300 * 1024 ** 2),    # 300 MB
            }


class ResourceMonitor:
    """
    Monitors system resources (RAM and ROM).
    Alerts when limits are approached or exceeded.
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.process = psutil.Process(os.getpid())
        
        # Volume paths (Docker mount points)
        self.volume_paths = {
            "pgdata": "/var/lib/postgresql/data",
            "ml_temp": "/app/ml_temp",
            "models": "/app/models",
            "logs": "/app/logs",
            "redisdata": "/data",
        }
    
    def get_ram_usage(self) -> Dict[str, float]:
        """
        Get current RAM usage.
        
        Returns:
            Dict with total, used, available, percent, and process_rss
        """
        mem = psutil.virtual_memory()
        
        return {
            "total_gb": mem.total / (1024 ** 3),
            "used_gb": mem.used / (1024 ** 3),
            "available_gb": mem.available / (1024 ** 3),
            "percent": mem.percent,
            "process_rss_mb": self.process.memory_info().rss / (1024 ** 2),
        }
    
    def get_rom_usage(self) -> Dict[str, Dict[str, float]]:
        """
        Get ROM (disk) usage for all volumes.
        
        Returns:
            Dict with usage info for each volume
        """
        usage = {}
        
        for name, path in self.volume_paths.items():
            try:
                if Path(path).exists():
                    disk = shutil.disk_usage(path)
                    limit = self.limits.VOLUME_LIMITS.get(name, self.limits.MAX_ROM_BYTES)
                    
                    usage[name] = {
                        "total_gb": disk.total / (1024 ** 3),
                        "used_gb": disk.used / (1024 ** 3),
                        "free_gb": disk.free / (1024 ** 3),
                        "percent": (disk.used / disk.total) * 100,
                        "limit_gb": limit / (1024 ** 3),
                        "within_limit": disk.used <= limit,
                    }
                else:
                    usage[name] = {"error": f"Path not found: {path}"}
            except Exception as e:
                usage[name] = {"error": str(e)}
        
        return usage
    
    def check_ram_limit(self, warn_threshold: float = 0.8) -> Dict[str, any]:
        """
        Check if RAM usage is within limits.
        
        Args:
            warn_threshold: Fraction of limit to trigger warning (0.8 = 80%)
            
        Returns:
            Dict with status and message
        """
        ram = self.get_ram_usage()
        used_bytes = ram["used_gb"] * (1024 ** 3)
        limit = self.limits.MAX_RAM_BYTES
        
        result = {
            "status": "ok",
            "used_gb": ram["used_gb"],
            "limit_gb": limit / (1024 ** 3),
            "percent_of_limit": (used_bytes / limit) * 100,
        }
        
        if used_bytes > limit:
            result["status"] = "exceeded"
            result["message"] = f"⚠️ RAM LIMIT EXCEEDED! {ram['used_gb']:.2f} GB used of {limit / (1024**3):.1f} GB limit"
            logger.error(result["message"])
        elif used_bytes > limit * warn_threshold:
            result["status"] = "warning"
            result["message"] = f"⚠️ RAM usage at {result['percent_of_limit']:.1f}% of limit"
            logger.warning(result["message"])
        else:
            result["message"] = f"✅ RAM usage OK: {ram['used_gb']:.2f} GB ({result['percent_of_limit']:.1f}% of limit)"
            logger.info(result["message"])
        
        return result
    
    def check_rom_limit(self, warn_threshold: float = 0.8) -> Dict[str, any]:
        """
        Check if ROM usage is within limits for all volumes.
        
        Args:
            warn_threshold: Fraction of limit to trigger warning
            
        Returns:
            Dict with status per volume
        """
        usage = self.get_rom_usage()
        results = {}
        
        for name, info in usage.items():
            if "error" in info:
                results[name] = {"status": "error", "message": info["error"]}
                continue
            
            used_bytes = info["used_gb"] * (1024 ** 3)
            limit = self.limits.VOLUME_LIMITS.get(name, self.limits.MAX_ROM_BYTES)
            
            result = {
                "used_gb": info["used_gb"],
                "limit_gb": limit / (1024 ** 3),
                "percent_of_limit": (used_bytes / limit) * 100,
            }
            
            if used_bytes > limit:
                result["status"] = "exceeded"
                result["message"] = f"⚠️ Volume {name} EXCEEDED! {info['used_gb']:.2f} GB of {limit/(1024**3):.1f} GB limit"
                logger.error(result["message"])
            elif used_bytes > limit * warn_threshold:
                result["status"] = "warning"
                result["message"] = f"⚠️ Volume {name} at {result['percent_of_limit']:.1f}% of limit"
                logger.warning(result["message"])
            else:
                result["status"] = "ok"
                result["message"] = f"✅ Volume {name} OK: {info['used_gb']:.2f} GB"
                logger.info(result["message"])
            
            results[name] = result
        
        return results
    
    def get_process_memory_breakdown(self) -> Dict[str, float]:
        """
        Get memory breakdown for current process.
        
        Returns:
            Dict with memory metrics in MB
        """
        mem_info = self.process.memory_info()
        
        return {
            "rss_mb": mem_info.rss / (1024 ** 2),      # Resident Set Size
            "vms_mb": mem_info.vms / (1024 ** 2),      # Virtual Memory Size
            "shared_mb": getattr(mem_info, 'shared', 0) / (1024 ** 2),
            "data_mb": getattr(mem_info, 'data', 0) / (1024 ** 2),
        }
    
    def check_process_memory(self, max_mb: float = 2048) -> Dict[str, any]:
        """
        Check if current process memory is within limit.
        
        Args:
            max_mb: Maximum allowed memory in MB for this process
            
        Returns:
            Dict with status and message
        """
        breakdown = self.get_process_memory_breakdown()
        rss_mb = breakdown["rss_mb"]
        
        result = {
            "rss_mb": rss_mb,
            "max_mb": max_mb,
            "percent": (rss_mb / max_mb) * 100,
        }
        
        if rss_mb > max_mb:
            result["status"] = "exceeded"
            result["message"] = f"⚠️ Process memory EXCEEDED! {rss_mb:.1f} MB of {max_mb} MB limit"
            logger.error(result["message"])
        elif rss_mb > max_mb * 0.8:
            result["status"] = "warning"
            result["message"] = f"⚠️ Process memory at {result['percent']:.1f}% of limit"
            logger.warning(result["message"])
        else:
            result["status"] = "ok"
            result["message"] = f"✅ Process memory OK: {rss_mb:.1f} MB"
            logger.info(result["message"])
        
        return result
    
    def full_check(self) -> Dict[str, any]:
        """
        Run full resource check.
        
        Returns:
            Dict with all check results
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "ram": self.check_ram_limit(),
            "rom": self.check_rom_limit(),
            "process": self.check_process_memory(),
        }
    
    def print_report(self):
        """Print a formatted resource report."""
        print("\n" + "=" * 60)
        print("AI TRADING SYSTEM - RESOURCE MONITOR")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("-" * 60)
        
        # RAM Report
        ram = self.get_ram_usage()
        print(f"\n📊 RAM USAGE:")
        print(f"   Total:     {ram['total_gb']:.2f} GB")
        print(f"   Used:      {ram['used_gb']:.2f} GB ({ram['percent']:.1f}%)")
        print(f"   Available: {ram['available_gb']:.2f} GB")
        print(f"   Process:   {ram['process_rss_mb']:.1f} MB")
        
        ram_check = self.check_ram_limit()
        print(f"   Status:    {ram_check['status'].upper()}")
        
        # ROM Report
        print(f"\n💾 ROM USAGE (Volumes):")
        rom = self.get_rom_usage()
        for name, info in rom.items():
            if "error" in info:
                print(f"   {name}: ERROR - {info['error']}")
            else:
                status = "✅" if info.get('within_limit', True) else "⚠️"
                print(f"   {status} {name}: {info['used_gb']:.2f} GB / {info.get('limit_gb', 0):.2f} GB limit")
        
        # Process Memory
        print(f"\n🔧 PROCESS MEMORY:")
        proc = self.get_process_memory_breakdown()
        print(f"   RSS:    {proc['rss_mb']:.1f} MB")
        print(f"   VMS:    {proc['vms_mb']:.1f} MB")
        
        print("\n" + "=" * 60 + "\n")


def main():
    """Main entry point for resource monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Trading System Resource Monitor")
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Run a single check and exit"
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Continuously monitor resources"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        help="Interval in seconds for watch mode (default: 60)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--max-ram-mb",
        type=int,
        default=2048,
        help="Maximum RAM in MB for process check (default: 2048)"
    )
    
    args = parser.parse_args()
    
    monitor = ResourceMonitor()
    
    if args.json:
        import json
        result = monitor.full_check()
        print(json.dumps(result, indent=2))
        return 0 if result["ram"]["status"] != "exceeded" else 1
    
    if args.check:
        monitor.print_report()
        check = monitor.full_check()
        return 0 if check["ram"]["status"] != "exceeded" else 1
    
    if args.watch:
        import time
        print("Starting continuous monitoring (Ctrl+C to stop)...")
        try:
            while True:
                monitor.print_report()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
        return 0
    
    # Default: single check with report
    monitor.print_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
