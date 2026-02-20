#!/usr/bin/env python3
"""
Volume Monitor for AI Trading System Docker Setup
Monitors RAM and ROM usage across all containers and volumes.
Alerts when approaching limits and can auto-cleanup.

Usage:
    python monitor_volumes.py              # Show current status
    python monitor_volumes.py --watch      # Continuous monitoring
    python monitor_volumes.py --cleanup    # Clean up old logs/temp files
"""

import subprocess
import json
import os
import shutil
import time
import argparse
from datetime import datetime
from pathlib import Path

# Configuration - matches docker-compose.stable.yml
LIMITS = {
    "containers": {
        "ai_trading_engine": {"ram_mb": 2048, "name": "Trading Engine"},
        "ai_trading_dashboard": {"ram_mb": 512, "name": "Dashboard"},
        "ai_trading_db": {"ram_mb": 1024, "name": "Database"},
        "ai_trading_redis": {"ram_mb": 512, "name": "Redis Cache"},
    },
    "volumes": {
        "pgdata": {"rom_mb": 1500, "name": "Database Storage"},
        "redisdata": {"rom_mb": 300, "name": "Redis Storage"},
        "ml_temp": {"rom_mb": 300, "name": "ML Temp Files"},
        "models": {"rom_mb": 200, "name": "ML Models"},
        "logs": {"rom_mb": 500, "name": "Log Files"},
    }
}

TOTAL_RAM_LIMIT_MB = 4096  # 4 GB
TOTAL_ROM_LIMIT_MB = 3000  # 3 GB

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def run_command(cmd: list) -> str:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception as e:
        return ""


def get_container_stats() -> dict:
    """Get RAM usage for all containers."""
    stats = {}
    
    # Get container list
    output = run_command(["docker", "ps", "--format", "{{.Names}}"])
    containers = output.split('\n') if output else []
    
    for container in containers:
        if not container:
            continue
            
        # Get memory usage
        stats_output = run_command([
            "docker", "stats", container, 
            "--no-stream", "--format", 
            "{{.MemUsage}}"
        ])
        
        if stats_output:
            # Parse "50.5MiB / 2GiB" format
            try:
                used = stats_output.split(' / ')[0]
                used_mb = parse_size_to_mb(used)
                stats[container] = {
                    "ram_used_mb": used_mb,
                    "ram_limit_mb": LIMITS["containers"].get(container, {}).get("ram_mb", 0),
                    "name": LIMITS["containers"].get(container, {}).get("name", container)
                }
            except:
                pass
    
    return stats


def parse_size_to_mb(size_str: str) -> float:
    """Convert size string to MB."""
    size_str = size_str.strip().upper()
    
    if 'GIB' in size_str or 'GB' in size_str:
        return float(''.join(c for c in size_str if c.isdigit() or c == '.')) * 1024
    elif 'MIB' in size_str or 'MB' in size_str:
        return float(''.join(c for c in size_str if c.isdigit() or c == '.'))
    elif 'KIB' in size_str or 'KB' in size_str:
        return float(''.join(c for c in size_str if c.isdigit() or c == '.')) / 1024
    return 0


def get_volume_sizes() -> dict:
    """Get ROM usage for all Docker volumes."""
    volumes = {}
    
    # Get volume paths
    output = run_command(["docker", "volume", "ls", "-q"])
    volume_names = output.split('\n') if output else []
    
    for vol_name in volume_names:
        if not vol_name:
            continue
            
        # Get volume path
        inspect = run_command(["docker", "volume", "inspect", vol_name, "--format", "{{.Mountpoint}}"])
        if not inspect:
            continue
            
        # Get size using du (inside Docker context)
        size_output = run_command([
            "docker", "run", "--rm", "-v", f"{vol_name}:/data",
            "alpine", "du", "-sm", "/data"
        ])
        
        if size_output:
            try:
                size_mb = float(size_output.split()[0])
                volumes[vol_name] = {
                    "rom_used_mb": size_mb,
                    "rom_limit_mb": LIMITS["volumes"].get(vol_name, {}).get("rom_mb", 0),
                    "name": LIMITS["volumes"].get(vol_name, {}).get("name", vol_name)
                }
            except:
                pass
    
    return volumes


def get_local_volume_sizes() -> dict:
    """Get sizes of local directories that act as volumes."""
    local_volumes = {}
    base_path = Path(".")
    
    # Check local directories
    local_dirs = ["logs", "ml_temp", "models", "data"]
    
    for dir_name in local_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            vol_key = dir_name
            if dir_name in LIMITS["volumes"]:
                local_volumes[vol_key] = {
                    "rom_used_mb": size_mb,
                    "rom_limit_mb": LIMITS["volumes"][dir_name]["rom_mb"],
                    "name": LIMITS["volumes"][dir_name]["name"]
                }
    
    return local_volumes


def print_status(container_stats: dict, volume_stats: dict, local_volumes: dict):
    """Print formatted status report."""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  AI TRADING SYSTEM - RESOURCE MONITOR{Colors.END}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    # Container RAM usage
    print(f"{Colors.BOLD}📊 CONTAINER RAM USAGE:{Colors.END}")
    print("-" * 50)
    total_ram = 0
    
    for container, stats in container_stats.items():
        used = stats['ram_used_mb']
        limit = stats['ram_limit_mb']
        name = stats['name']
        pct = (used / limit * 100) if limit > 0 else 0
        total_ram += used
        
        color = Colors.GREEN if pct < 70 else Colors.YELLOW if pct < 90 else Colors.RED
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        
        print(f"  {name:20} [{color}{bar}{Colors.END}] {used:6.1f}/{limit} MB ({pct:5.1f}%)")
    
    print(f"\n  {Colors.BOLD}Total RAM: {total_ram:.1f}/{TOTAL_RAM_LIMIT_MB} MB{Colors.END}")
    ram_pct = (total_ram / TOTAL_RAM_LIMIT_MB) * 100
    if ram_pct > 90:
        print(f"  {Colors.RED}⚠ WARNING: Approaching RAM limit!{Colors.END}")
    
    # Volume ROM usage
    print(f"\n{Colors.BOLD}💾 VOLUME ROM USAGE:{Colors.END}")
    print("-" * 50)
    total_rom = 0
    
    all_volumes = {**volume_stats, **local_volumes}
    
    for volume, stats in all_volumes.items():
        used = stats['rom_used_mb']
        limit = stats['rom_limit_mb']
        name = stats['name']
        pct = (used / limit * 100) if limit > 0 else 0
        total_rom += used
        
        color = Colors.GREEN if pct < 70 else Colors.YELLOW if pct < 90 else Colors.RED
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        
        print(f"  {name:20} [{color}{bar}{Colors.END}] {used:6.1f}/{limit} MB ({pct:5.1f}%)")
    
    print(f"\n  {Colors.BOLD}Total ROM: {total_rom:.1f}/{TOTAL_ROM_LIMIT_MB} MB{Colors.END}")
    rom_pct = (total_rom / TOTAL_ROM_LIMIT_MB) * 100
    if rom_pct > 90:
        print(f"  {Colors.RED}⚠ WARNING: Approaching ROM limit!{Colors.END}")
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    return ram_pct, rom_pct


def cleanup_old_files():
    """Clean up old logs and temp files."""
    print(f"\n{Colors.BLUE}🧹 CLEANING UP OLD FILES...{Colors.END}\n")
    
    cleaned_mb = 0
    
    # Clean logs older than 7 days
    logs_path = Path("logs")
    if logs_path.exists():
        for log_file in logs_path.glob("*.log.*"):
            try:
                # Check file age
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                age_days = (datetime.now() - mtime).days
                
                if age_days > 7:
                    size = log_file.stat().st_size / (1024 * 1024)
                    log_file.unlink()
                    cleaned_mb += size
                    print(f"  Deleted: {log_file.name} ({size:.1f} MB, {age_days} days old)")
            except Exception as e:
                print(f"  Error deleting {log_file}: {e}")
    
    # Clean temp ML files
    ml_temp_path = Path("ml_temp")
    if ml_temp_path.exists():
        for temp_file in ml_temp_path.glob("*.tmp"):
            try:
                size = temp_file.stat().st_size / (1024 * 1024)
                temp_file.unlink()
                cleaned_mb += size
                print(f"  Deleted temp: {temp_file.name} ({size:.1f} MB)")
            except:
                pass
    
    # Clean old model checkpoints (keep last 3)
    models_path = Path("models")
    if models_path.exists():
        checkpoints = sorted(models_path.glob("checkpoint_*"))
        for old_ckpt in checkpoints[:-3]:
            try:
                size = sum(f.stat().st_size for f in old_ckpt.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                shutil.rmtree(old_ckpt)
                cleaned_mb += size_mb
                print(f"  Deleted checkpoint: {old_ckpt.name} ({size_mb:.1f} MB)")
            except:
                pass
    
    print(f"\n  {Colors.GREEN}✓ Cleaned {cleaned_mb:.1f} MB{Colors.END}\n")
    return cleaned_mb


def watch_mode(interval: int = 30):
    """Continuous monitoring mode."""
    print(f"\n{Colors.BLUE}Starting watch mode (refresh every {interval}s)...{Colors.END}")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            container_stats = get_container_stats()
            volume_stats = get_volume_sizes()
            local_volumes = get_local_volume_sizes()
            
            ram_pct, rom_pct = print_status(container_stats, volume_stats, local_volumes)
            
            # Auto-cleanup if approaching limits
            if rom_pct > 85:
                print(f"\n{Colors.YELLOW}Auto-cleanup triggered (ROM > 85%){Colors.END}")
                cleanup_old_files()
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n{Colors.BLUE}Monitoring stopped.{Colors.END}")


def main():
    parser = argparse.ArgumentParser(description="Monitor Docker resources for AI Trading System")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Watch interval in seconds")
    parser.add_argument("--cleanup", "-c", action="store_true", help="Clean up old files")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_old_files()
        return
    
    if args.watch:
        watch_mode(args.interval)
        return
    
    # Single status check
    container_stats = get_container_stats()
    volume_stats = get_volume_sizes()
    local_volumes = get_local_volume_sizes()
    
    if args.json:
        output = {
            "timestamp": datetime.now().isoformat(),
            "containers": container_stats,
            "volumes": {**volume_stats, **local_volumes},
            "totals": {
                "ram_mb": sum(s['ram_used_mb'] for s in container_stats.values()),
                "rom_mb": sum(s['rom_used_mb'] for s in {**volume_stats, **local_volumes}.values())
            }
        }
        print(json.dumps(output, indent=2))
    else:
        print_status(container_stats, volume_stats, local_volumes)


if __name__ == "__main__":
    main()
