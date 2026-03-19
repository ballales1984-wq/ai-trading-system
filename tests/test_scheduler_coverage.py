"""
Test Coverage for Scheduler Module
=================================
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from app.scheduler import (
    ScheduleType,
    ScheduledTask,
    TaskStatus,
    TaskScheduler,
    profile_execution,
    time_execution,
)


class TestScheduleType:
    """Test ScheduleType enum"""
    
    def test_schedule_type_values(self):
        """Test ScheduleType enum values"""
        assert ScheduleType.INTERVAL.value == "interval"
        assert ScheduleType.CRON.value == "cron"
        assert ScheduleType.DAILY.value == "daily"
        assert ScheduleType.WEEKLY.value == "weekly"
        assert ScheduleType.ONCE.value == "once"


class TestTaskStatus:
    """Test TaskStatus enum"""
    
    def test_task_status_values(self):
        """Test TaskStatus enum values"""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestScheduledTask:
    """Test ScheduledTask dataclass"""
    
    def test_scheduled_task_creation(self):
        """Test ScheduledTask creation"""
        def dummy_func():
            return "test"
        
        task = ScheduledTask(
            task_id="task-1",
            name="test_task",
            func=dummy_func,
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
            enabled=True
        )
        assert task.task_id == "task-1"
        assert task.name == "test_task"
        assert task.schedule_type == ScheduleType.INTERVAL
        assert task.interval_seconds == 60
        assert task.enabled is True
    
    def test_scheduled_task_defaults(self):
        """Test ScheduledTask default values"""
        def dummy_func():
            return "test"
        
        task = ScheduledTask(
            task_id="task-1",
            name="test_task",
            func=dummy_func,
            schedule_type=ScheduleType.INTERVAL
        )
        assert task.enabled is True
        assert task.interval_seconds is None
        assert task.status == TaskStatus.PENDING
    
    def test_calculate_next_run_interval(self):
        """Test calculate next run for interval"""
        def dummy_func():
            return "test"
        
        task = ScheduledTask(
            task_id="task-1",
            name="test_task",
            func=dummy_func,
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60
        )
        next_run = task.calculate_next_run()
        assert next_run is not None
        expected = datetime.utcnow() + timedelta(seconds=60)
        diff = abs((next_run - expected).total_seconds())
        assert diff < 1  # Within 1 second
    
    def test_calculate_next_run_daily(self):
        """Test calculate next run for daily"""
        def dummy_func():
            return "test"
        
        run_time = datetime.utcnow().replace(hour=10, minute=0, second=0, microsecond=0)
        task = ScheduledTask(
            task_id="task-1",
            name="test_task",
            func=dummy_func,
            schedule_type=ScheduleType.DAILY,
            run_time=run_time
        )
        next_run = task.calculate_next_run()
        assert next_run is not None
        assert next_run.hour == 10
        assert next_run.minute == 0
    
    def test_to_dict(self):
        """Test to_dict method"""
        def dummy_func():
            return "test"
        
        task = ScheduledTask(
            task_id="task-1",
            name="test_task",
            func=dummy_func,
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60
        )
        task_dict = task.to_dict()
        assert task_dict["task_id"] == "task-1"
        assert task_dict["name"] == "test_task"
        assert task_dict["schedule_type"] == "interval"
        assert task_dict["interval_seconds"] == 60
        assert task_dict["enabled"] is True


class TestTaskScheduler:
    """Test TaskScheduler class"""
    
    def test_scheduler_initialization(self):
        """Test TaskScheduler initialization"""
        scheduler = TaskScheduler()
        assert scheduler.tasks == {}
        assert scheduler.running is False
    
    def test_add_task(self):
        """Test adding a task"""
        scheduler = TaskScheduler()
        
        def dummy_task():
            return "test"
        
        task = scheduler.add_task(
            task_id="task-1",
            name="test_task",
            func=dummy_task,
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60
        )
        
        assert task is not None
        assert task.task_id == "task-1"
        assert "task-1" in scheduler.tasks
    
    def test_add_duplicate_task(self):
        """Test adding duplicate task overwrites"""
        scheduler = TaskScheduler()
        
        def dummy_task():
            return "test"
        
        task1 = scheduler.add_task(
            task_id="task-1",
            name="test_task",
            func=dummy_task,
            schedule_type=ScheduleType.INTERVAL
        )
        
        # Adding same task_id should overwrite
        task2 = scheduler.add_task(
            task_id="task-1",
            name="test_task_updated",
            func=dummy_task,
            schedule_type=ScheduleType.INTERVAL
        )
        
        assert task2.name == "test_task_updated"
        assert len(scheduler.tasks) == 1

    def test_remove_task(self):
        """Test removing a task"""
        scheduler = TaskScheduler()
        
        def dummy_task():
            return "test"
        
        scheduler.add_task(
            task_id="task-1",
            name="test_task",
            func=dummy_task,
            schedule_type=ScheduleType.INTERVAL
        )
        
        result = scheduler.remove_task("task-1")
        assert result is True
        assert "task-1" not in scheduler.tasks

    def test_remove_nonexistent_task(self):
        """Test removing nonexistent task returns False"""
        scheduler = TaskScheduler()
        result = scheduler.remove_task("nonexistent")
        assert result is False

    def test_get_task_status(self):
        """Test getting task status"""
        scheduler = TaskScheduler()
        
        def dummy_task():
            return "test"
        
        scheduler.add_task(
            task_id="task-1",
            name="test_task",
            func=dummy_task,
            schedule_type=ScheduleType.INTERVAL
        )
        
        status = scheduler.get_task_status("task-1")
        assert status is not None
        assert status["task_id"] == "task-1"

    def test_list_tasks(self):
        """Test listing all tasks via attribute"""
        scheduler = TaskScheduler()
        
        def dummy_task():
            return "test"
        
        scheduler.add_task(
            task_id="task-1",
            name="test_task_1",
            func=dummy_task,
            schedule_type=ScheduleType.INTERVAL
        )
        scheduler.add_task(
            task_id="task-2",
            name="test_task_2",
            func=dummy_task,
            schedule_type=ScheduleType.DAILY
        )
        
        # Access tasks via attribute
        tasks = scheduler.tasks
        assert len(tasks) == 2

    def test_enable_task(self):
        """Test enabling a task"""
        scheduler = TaskScheduler()
        
        def dummy_task():
            return "test"
        
        scheduler.add_task(
            task_id="task-1",
            name="test_task",
            func=dummy_task,
            schedule_type=ScheduleType.INTERVAL,
            enabled=False
        )
        
        result = scheduler.enable_task("task-1")
        assert result is True
        task = scheduler.tasks.get("task-1")
        assert task.enabled is True

    def test_disable_task(self):
        """Test disabling a task"""
        scheduler = TaskScheduler()
        
        def dummy_task():
            return "test"
        
        scheduler.add_task(
            task_id="task-1",
            name="test_task",
            func=dummy_task,
            schedule_type=ScheduleType.INTERVAL,
            enabled=True
        )
        
        result = scheduler.disable_task("task-1")
        assert result is True
        task = scheduler.tasks.get("task-1")
        assert task.enabled is False


class TestDecorators:
    """Test decorators"""
    
    def test_time_execution_decorator(self):
        """Test time_execution decorator"""
        @time_execution
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        result = slow_function()
        assert result == "done"

    def test_profile_execution_decorator(self):
        """Test profile_execution decorator"""
        @profile_execution
        def test_function():
            return "test result"
        
        result = test_function()
        assert result == "test result"