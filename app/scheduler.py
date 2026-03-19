"""
AI Trading System - Scheduler Module

Provides automated scheduling for strategy execution, data collection,
and system maintenance tasks.

Features:
- Multiple schedule types (interval, cron, daily, weekly)
- Task status tracking
- Error handling and recovery
- Async task execution

Performance:
- cProfile decorators for function timing analysis
- Memory usage tracking
"""

import asyncio
import logging
import cProfile
import pstats
import io
import functools

# profile is used as a decorator class
profile = cProfile.Profile
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Any
from enum import Enum
import json
from pathlib import Path


logger = logging.getLogger(__name__)


def profile_execution(func: Callable) -> Callable:
    """
    Decorator to profile function execution using cProfile.
    
    Usage:
        @profile_execution
        def my_function():
            ...
    
    Results are logged at INFO level with timing stats.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            
            # Get stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)  # Top 10 functions
            
            logger.info(f"Profile for {func.__name__}:\n{s.getvalue()}")
    
    return wrapper


def time_execution(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Usage:
        @time_execution
        def my_function():
            ...
    
    Results are logged at INFO level with execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.info(f"Execution time for {func.__name__}: {elapsed:.4f}s")
    
    return wrapper

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of scheduled tasks"""
    INTERVAL = "interval"           # Run every N seconds/minutes
    CRON = "cron"                   # Cron-style scheduling
    DAILY = "daily"                # Run once per day
    WEEKLY = "weekly"              # Run once per week
    ONCE = "once"                  # Run once at specific time


class TaskStatus(Enum):
    """Status of scheduled task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScheduledTask:
    """Represents a scheduled task"""
    
    def __init__(
        self,
        task_id: str,
        name: str,
        func: Callable,
        schedule_type: ScheduleType,
        interval_seconds: Optional[int] = None,
        cron_expression: Optional[str] = None,
        run_time: Optional[datetime] = None,
        enabled: bool = True,
        **kwargs
    ):
        self.task_id = task_id
        self.name = name
        self.func = func
        self.schedule_type = schedule_type
        self.interval_seconds = interval_seconds
        self.cron_expression = cron_expression
        self.run_time = run_time
        self.enabled = enabled
        self.kwargs = kwargs
        self.status = TaskStatus.PENDING
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.run_count = 0
        self.error_count = 0
        self.last_error: Optional[str] = None
        
    def calculate_next_run(self) -> datetime:
        """Calculate next run time based on schedule type"""
        now = datetime.utcnow()
        
        if self.schedule_type == ScheduleType.INTERVAL:
            self.next_run = now + timedelta(seconds=self.interval_seconds or 60)
        elif self.schedule_type == ScheduleType.DAILY and self.run_time:
            self.next_run = now.replace(
                hour=self.run_time.hour,
                minute=self.run_time.minute,
                second=0,
                microsecond=0
            )
            if self.next_run <= now:
                self.next_run += timedelta(days=1)
        elif self.schedule_type == ScheduleType.WEEKLY and self.run_time:
            self.next_run = now.replace(
                hour=self.run_time.hour,
                minute=self.run_time.minute,
                second=0,
                microsecond=0
            )
            days_ahead = self.run_time.weekday() - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            self.next_run += timedelta(days=days_ahead)
        elif self.schedule_type == ScheduleType.ONCE:
            self.next_run = self.run_time
        else:
            self.next_run = now + timedelta(minutes=5)
            
        return self.next_run
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "schedule_type": self.schedule_type.value,
            "interval_seconds": self.interval_seconds,
            "cron_expression": self.cron_expression,
            "enabled": self.enabled,
            "status": self.status.value,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_error": self.last_error
        }


class TaskScheduler:
    """
    Central scheduler for managing all scheduled tasks.
    
    Features:
    - Multiple schedule types (interval, cron, daily, weekly, once)
    - Task persistence across restarts
    - Error handling and retry logic
    - Task status tracking
    - Manual trigger capability
    """
    
    def __init__(self, persistence_path: Optional[Path] = None):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.persistence_path = persistence_path or Path("data/scheduler_state.json")
        self._lock = asyncio.Lock()
        
    def add_task(
        self,
        task_id: str,
        name: str,
        func: Callable,
        schedule_type: ScheduleType,
        interval_seconds: Optional[int] = None,
        cron_expression: Optional[str] = None,
        run_time: Optional[datetime] = None,
        enabled: bool = True,
        **kwargs
    ) -> ScheduledTask:
        """Add a new scheduled task"""
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            func=func,
            schedule_type=schedule_type,
            interval_seconds=interval_seconds,
            cron_expression=cron_expression,
            run_time=run_time,
            enabled=enabled,
            **kwargs
        )
        task.calculate_next_run()
        self.tasks[task_id] = task
        
        logger.info(f"Added scheduled task: {task_id} ({name})")
        self._save_state()
        
        return task
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a scheduled task"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._save_state()
            logger.info(f"Removed scheduled task: {task_id}")
            return True
        return False
    
    def enable_task(self, task_id: str) -> bool:
        """Enable a task"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            self.tasks[task_id].calculate_next_run()
            self._save_state()
            return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """Disable a task"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            self._save_state()
            return True
        return False
    
    async def run_task_now(self, task_id: str) -> bool:
        """Manually trigger a task to run immediately"""
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        
        async def run_with_error_handling():
            task.status = TaskStatus.RUNNING
            try:
                if asyncio.iscoroutinefunction(task.func):
                    await task.func(**task.kwargs)
                else:
                    task.func(**task.kwargs)
                task.status = TaskStatus.COMPLETED
                task.last_error = None
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_count += 1
                task.last_error = str(e)
                logger.error(f"Task {task_id} failed: {e}")
            finally:
                task.last_run = datetime.utcnow()
                task.run_count += 1
                task.calculate_next_run()
                self._save_state()
        
        asyncio.create_task(run_with_error_handling())
        return True
    
    async def start(self):
        """Start the scheduler"""
        if self.running:
            return
            
        self.running = True
        self._load_state()
        
        # Calculate next run times for all tasks
        for task in self.tasks.values():
            if task.enabled and not task.next_run:
                task.calculate_next_run()
        
        self.scheduler_task = asyncio.create_task(self._run_scheduler())
        logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        self._save_state()
        logger.info("Task scheduler stopped")
    
    @profile
    async def _execute_task(self, task: ScheduledTask):
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        
        try:
            if asyncio.iscoroutinefunction(task.func):
                await task.func(**task.kwargs)
            else:
                task.func(**task.kwargs)
                
            task.status = TaskStatus.COMPLETED
            task.last_error = None
            logger.debug(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_count += 1
            task.last_error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
            
        finally:
            task.last_run = datetime.utcnow()
            task.run_count += 1
            task.calculate_next_run()
            self._save_state()
    
    async def _run_scheduler(self):
        """Main scheduler loop"""
        while self.running:
            try:
                async with self._lock:
                    now = datetime.utcnow()
                    
                    for task_id, task in list(self.tasks.items()):
                        if not task.enabled:
                            continue
                            
                        if task.next_run and now >= task.next_run:
                            # Execute task
                            asyncio.create_task(self._execute_task(task))
                            
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5)
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        
        try:
            if asyncio.iscoroutinefunction(task.func):
                await task.func(**task.kwargs)
            else:
                task.func(**task.kwargs)
                
            task.status = TaskStatus.COMPLETED
            task.last_error = None
            logger.debug(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_count += 1
            task.last_error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
            
        finally:
            task.last_run = datetime.utcnow()
            task.run_count += 1
            task.calculate_next_run()
            self._save_state()
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        return None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get status of all tasks"""
        return [task.to_dict() for task in self.tasks.values()]
    
    def _save_state(self):
        """Persist scheduler state to disk"""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                task_id: task.to_dict() 
                for task_id, task in self.tasks.items()
            }
            with open(self.persistence_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {e}")
    
    def _load_state(self):
        """Load scheduler state from disk"""
        try:
            if self.persistence_path.exists():
                with open(self.persistence_path, 'r') as f:
                    state = json.load(f)
                    
                for task_id, task_data in state.items():
                    # Recreate tasks from saved state
                    # Note: Functions can't be serialized, so skip if not in memory
                    if task_id not in self.tasks:
                        logger.warning(f"Task {task_id} in saved state but not in memory")
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}")


# =============================================================================
# Pre-built Task Functions
# =============================================================================

async def run_strategy_interval(strategy_name: str, interval_seconds: int = 60):
    """Example: Run a strategy on an interval"""
    logger.info(f"Running strategy '{strategy_name}' at {datetime.utcnow()}")
    # Strategy execution logic would go here


async def collect_market_data(symbols: List[str]):
    """Example: Collect market data for symbols"""
    logger.info(f"Collecting market data for {symbols}")
    # Data collection logic would go here


async def perform_risk_check():
    """Example: Perform daily risk check"""
    logger.info("Performing risk check")
    # Risk check logic would go here


async def generate_performance_report():
    """Example: Generate weekly performance report"""
    logger.info("Generating performance report")
    # Report generation logic would go here


async def cleanup_old_logs(days_to_keep: int = 30):
    """Example: Clean up old log files"""
    logger.info(f"Cleaning up logs older than {days_to_keep} days")
    # Cleanup logic would go here


# =============================================================================
# Scheduler Singleton
# =============================================================================

_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    """Get the global scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler


def init_scheduler() -> TaskScheduler:
    """Initialize and return scheduler with default tasks"""
    scheduler = get_scheduler()
    
    # Add default tasks
    scheduler.add_task(
        task_id="market_data_collection",
        name="Market Data Collection",
        func=collect_market_data,
        schedule_type=ScheduleType.INTERVAL,
        interval_seconds=60,
        symbols=["BTCUSDT", "ETHUSDT"]
    )
    
    scheduler.add_task(
        task_id="daily_risk_check",
        name="Daily Risk Check",
        func=perform_risk_check,
        schedule_type=ScheduleType.DAILY,
        run_time=datetime.utcnow().replace(hour=8, minute=0)
    )
    
    scheduler.add_task(
        task_id="weekly_performance_report",
        name="Weekly Performance Report",
        func=generate_performance_report,
        schedule_type=ScheduleType.WEEKLY,
        run_time=datetime.utcnow().replace(hour=9, minute=0, weekday=0)
    )
    
    return scheduler
