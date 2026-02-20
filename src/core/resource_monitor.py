"""
Resource Monitor
==============
Monitors RAM and ROM usage for the trading system.
Can run as a background task or cron job.

Usage:
    # Run as standalone script
    python src/core/resource_monitor.py
    
    # Import and use in your code
    from src.core.resource_monitor import ResourceMonitor, check_resources
    
    monitor = ResourceMonitor(max_ram_gb=4, max_rom_gb=3)
    status = monitor.check_all()
    print(status)
"""

import os
import sys
import time
import logging
import threading
import shutil
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, resource monitoring will be limited")


logger = logging.getLogger(__name__)


@dataclass
class ResourceStatus:
    """Resource usage status."""
    timestamp: str
    ram_used_mb: float
    ram_percent: float
    rom_used_mb: float
    rom_percent: float
    ram_ok: bool
    rom_ok: bool
    details: Dict


class ResourceMonitor:
    """
    Monitor system resources (RAM and ROM).
    
    Ensures the trading system stays within resource limits.
    """
    
    # Volume paths to monitor (customize for your setup)
    DEFAULT_VOLUMES = {
        "postgres": "/var/lib/postgresql/data",
        "redis": "/data",
        "ml_temp": "/app/ml_temp",
        "logs": "/app/logs",
        "dashboard": "/app/data",
        "prometheus": "/prometheus",
        "nginx": "/var/log/nginx",
        "cache": "/app/cache",
        "app_data": "/app/data"
    }
    
    def __init__(
        self,
        max_ram_gb: float = 4,
        max_rom_gb: float = 3,
        volumes: Optional[Dict[str, str]] = None,
        check_interval_seconds: int = 60,
        alert_threshold_percent: float = 90
    ):
        """
        Initialize resource monitor.
        
        Args:
            max_ram_gb: Maximum RAM in GB
            max_rom_gb: Maximum ROM in GB
            volumes: Dict of volume name -> path
            check_interval_seconds: How often to check
            alert_threshold_percent: Alert when usage exceeds this %
        """
        self.max_ram_bytes = max_ram_gb * 1024 * 1024 * 1024
        self.max_rom_bytes = max_rom_gb * 1024 * 1024 * 1024
        self.volumes = volumes or self.DEFAULT_VOLUMES
        self.check_interval = check_interval_seconds
        self.alert_threshold = alert_threshold_percent
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[callable] = []
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add callback to be called when resource limit is exceeded."""
        self._callbacks.append(callback)
    
    def get_ram_usage(self) -> Dict:
        """Get current RAM usage."""
        if not PSUTIL_AVAILABLE:
            return {"used_mb": 0, "percent": 0, "available_mb": 0}
        
        mem = psutil.virtual_memory()
        
        return {
            "total_mb": mem.total / (1024 * 1024),
            "available_mb": mem.available / (1024 * 1024),
            "used_mb": mem.used / (1024 * 1024),
            "percent": mem.percent,
            "process_mb": psutil.Process().memory_info().rss / (1024 * 1024)
        }
    
    def get_rom_usage(self, base_path: str = ".") -> Dict:
        """Get ROM usage for monitored volumes."""
        total_used = 0
        volume_details = {}
        
        for name, rel_path in self.volumes.items():
            # Try both absolute and relative paths
            paths_to_try = [
                rel_path,
                os.path.join(base_path, rel_path),
                f"/app/{rel_path.lstrip('/')}",
                f"./{rel_path}"
            ]
            
            volume_path = None
            for p in paths_to_try:
                if os.path.exists(p):
                    volume_path = p
                    break
            
            if volume_path and os.path.exists(volume_path):
                try:
                    usage = shutil.disk_usage(volume_path)
                    used_mb = usage.used / (1024 * 1024)
                    total_used += used_mb
                    volume_details[name] = {
                        "used_mb": used_mb,
                        "free_mb": usage.free / (1024 * 1024),
                        "total_mb": usage.total / (1024 * 1024)
                    }
                except Exception as e:
                    volume_details[name] = {"error": str(e)}
            else:
                volume_details[name] = {"status": "not_found"}
        
        return {
            "total_mb": total_used,
            "percent": (total_used / (self.max_rom_bytes / (1024 * 1024))) * 100,
            "volumes": volume_details
        }
    
    def check_all(self, base_path: str = ".") -> ResourceStatus:
        """Check all resources and return status."""
        ram = self.get_ram_usage()
        rom = self.get_rom_usage(base_path)
        
        ram_ok = ram["percent"] < self.alert_threshold
        rom_ok = rom["percent"] < self.alert_threshold
        
        return ResourceStatus(
            timestamp=datetime.now().isoformat(),
            ram_used_mb=ram["used_mb"],
            ram_percent=ram["percent"],
            rom_used_mb=rom["total_mb"],
            rom_percent=rom["percent"],
            ram_ok=ram_ok,
            rom_ok=rom_ok,
            details={
                "ram": ram,
                "rom": rom
            }
        )
    
    def start_monitoring(self, base_path: str = ".") -> None:
        """Start background monitoring."""
        if self._running:
            return
        
        self._running = True
        self._base_path = base_path
        
        def monitor_loop():
            while self._running:
                try:
                    status = self.check_all(self._base_path)
                    
                    if not status.ram_ok or not status.rom_ok:
                        alert_msg = self._format_alert(status)
                        logger.warning(alert_msg)
                        
                        for callback in self._callbacks:
                            try:
                                callback(status)
                            except Exception as e:
                                logger.error(f"Alert callback error: {e}")
                    
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                    time.sleep(self.check_interval)
        
        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _format_alert(self, status: ResourceStatus) -> str:
        """Format alert message."""
        msg = ["⚠️ RESOURCE ALERT!"]
        
        if not status.ram_ok:
            msg.append(f"RAM: {status.ram_used_mb:.0f}MB ({status.ram_percent:.1f}%)")
        
        if not status.rom_ok:
            msg.append(f"ROM: {status.rom_used_mb:.0f}MB ({status.rom_percent:.1f}%)")
        
        return " | ".join(msg)


# Global monitor instance
_monitor: Optional[ResourceMonitor] = None


def get_monitor(
    max_ram_gb: float = 4,
    max_rom_gb: float = 3
) -> ResourceMonitor:
    """Get global resource monitor."""
    global _monitor
    if _monitor is None:
        _monitor = ResourceMonitor(max_ram_gb, max_rom_gb)
    return _monitor


def check_resources() -> ResourceStatus:
    """Quick check of resources."""
    monitor = get_monitor()
    return monitor.check_all()


# Simple CLI
def main():
    """Run resource monitor from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor system resources")
    parser.add_argument("--ram", type=float, default=4, help="Max RAM in GB")
    parser.add_argument("--rom", type=float, default=3, help="Max ROM in GB")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=60, help="Check interval seconds")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    monitor = ResourceMonitor(
        max_ram_gb=args.ram,
        max_rom_gb=args.rom,
        check_interval_seconds=args.interval
    )
    
    if args.continuous:
        print(f"Starting continuous monitoring (RAM: {args.ram}GB, ROM: {args.rom}GB)")
        monitor.start_monitoring()
        
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nStopping monitor...")
            monitor.stop_monitoring()
    else:
        print(f"Checking resources (RAM: {args.ram}GB, ROM: {args.rom}GB)")
        status = monitor.check_all()
        
        print("\n" + "=" * 50)
        print("RESOURCE STATUS")
        print("=" * 50)
        print(f"Timestamp: {status.timestamp}")
        print(f"RAM: {status.ram_used_mb:.1f} MB ({status.ram_percent:.1f}%) - {'✓ OK' if status.ram_ok else '⚠️ ALERT'}")
        print(f"ROM: {status.rom_used_mb:.1f} MB ({status.rom_percent:.1f}%) - {'✓ OK' if status.rom_ok else '⚠️ ALERT'}")
        print("=" * 50)
        
        if status.rom_percent > 0:
            print("\nVolume Details:")
            for name, vol in status.details["rom"]["volumes"].items():
                if "error" not in vol and "not_found" not in vol:
                    print(f"  {name}: {vol.get('used_mb', 0):.1f} MB")


if __name__ == "__main__":
    main()

