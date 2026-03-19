"""
Test Coverage for Performance Module
===================================
"""

import pytest
import time
from app.core.performance import (
    PerformanceMetrics,
    PerformanceProfiler,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass"""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation"""
        metrics = PerformanceMetrics(function_name="test_function")
        assert metrics.function_name == "test_function"
        assert metrics.call_count == 0
        assert metrics.total_time == 0.0
        assert metrics.min_time == float('inf')
        assert metrics.max_time == 0.0
        assert metrics.avg_time == 0.0
    
    def test_record_call(self):
        """Test record_call method"""
        metrics = PerformanceMetrics(function_name="test_function")
        metrics.record_call(0.1)  # 100ms
        assert metrics.call_count == 1
        assert metrics.total_time == 0.1
        assert metrics.min_time == 0.1
        assert metrics.max_time == 0.1
        assert metrics.avg_time == 0.1
    
    def test_record_multiple_calls(self):
        """Test recording multiple calls"""
        metrics = PerformanceMetrics(function_name="test_function")
        metrics.record_call(0.1)
        metrics.record_call(0.2)
        metrics.record_call(0.3)
        
        assert metrics.call_count == 3
        assert abs(metrics.total_time - 0.6) < 0.0001
        assert metrics.min_time == 0.1
        assert metrics.max_time == 0.3
        assert abs(metrics.avg_time - 0.2) < 0.0001
    
    def test_to_dict(self):
        """Test to_dict method"""
        metrics = PerformanceMetrics(function_name="test_function")
        metrics.record_call(0.1)
        
        result = metrics.to_dict()
        assert result["function_name"] == "test_function"
        assert result["call_count"] == 1
        assert result["total_time_ms"] == 100.0
        assert result["min_time_ms"] == 100.0
        assert result["max_time_ms"] == 100.0
        assert result["avg_time_ms"] == 100.0


class TestPerformanceProfiler:
    """Test PerformanceProfiler class"""
    
    def test_profiler_initialization(self):
        """Test PerformanceProfiler initialization"""
        profiler = PerformanceProfiler()
        assert profiler._metrics == {}
    
    def test_record_new_function(self):
        """Test recording a new function"""
        profiler = PerformanceProfiler()
        profiler.record("test_function", 0.1)
        
        assert "test_function" in profiler._metrics
        assert profiler._metrics["test_function"].call_count == 1
    
    def test_record_existing_function(self):
        """Test recording an existing function"""
        profiler = PerformanceProfiler()
        profiler.record("test_function", 0.1)
        profiler.record("test_function", 0.2)
        
        assert profiler._metrics["test_function"].call_count == 2
        assert abs(profiler._metrics["test_function"].total_time - 0.3) < 0.0001
    
    def test_get_metrics_specific_function(self):
        """Test get_metrics for specific function"""
        profiler = PerformanceProfiler()
        profiler.record("test_function", 0.1)
        
        result = profiler.get_metrics("test_function")
        assert result["function_name"] == "test_function"
        assert result["call_count"] == 1
    
    def test_get_metrics_nonexistent_function(self):
        """Test get_metrics for nonexistent function"""
        profiler = PerformanceProfiler()
        
        result = profiler.get_metrics("nonexistent")
        assert result == {}
    
    def test_get_metrics_all_functions(self):
        """Test get_metrics for all functions"""
        profiler = PerformanceProfiler()
        profiler.record("func1", 0.1)
        profiler.record("func2", 0.2)
        
        result = profiler.get_metrics()
        assert "func1" in result
        assert "func2" in result
        assert result["func1"]["call_count"] == 1
        assert result["func2"]["call_count"] == 1
    
    def test_reset_specific_function(self):
        """Test reset for specific function"""
        profiler = PerformanceProfiler()
        profiler.record("test_function", 0.1)
        profiler.reset("test_function")
        
        assert profiler._metrics["test_function"].call_count == 0
        assert profiler._metrics["test_function"].total_time == 0.0
    
    def test_reset_all_functions(self):
        """Test reset for all functions"""
        profiler = PerformanceProfiler()
        profiler.record("func1", 0.1)
        profiler.record("func2", 0.2)
        profiler.reset()
        
        assert len(profiler._metrics) == 0
    
    def test_get_slowest_functions(self):
        """Test get_slowest_functions"""
        profiler = PerformanceProfiler()
        profiler.record("slow_func", 0.5)
        profiler.record("fast_func", 0.1)
        profiler.record("medium_func", 0.3)
        
        slowest = profiler.get_slowest_functions(2)
        assert len(slowest) == 2
        assert slowest[0]["function_name"] == "slow_func"
        assert slowest[1]["function_name"] == "medium_func"
    
    def test_get_most_called_functions(self):
        """Test get_most_called_functions"""
        profiler = PerformanceProfiler()
        profiler.record("func1", 0.1)
        profiler.record("func1", 0.1)
        profiler.record("func2", 0.1)
        
        most_called = profiler.get_most_called_functions(2)
        assert len(most_called) == 2
        assert most_called[0]["function_name"] == "func1"
        assert most_called[0]["call_count"] == 2
        assert most_called[1]["function_name"] == "func2"


class TestPerformanceIntegration:
    """Integration tests for performance module"""
    
    def test_full_profiling_workflow(self):
        """Test full profiling workflow"""
        profiler = PerformanceProfiler()
        
        # Simulate function calls
        for _ in range(5):
            start = time.perf_counter()
            # Simulate work
            time.sleep(0.001)
            duration = time.perf_counter() - start
            profiler.record("my_function", duration)
        
        # Get metrics
        metrics = profiler.get_metrics("my_function")
        assert metrics["call_count"] == 5
        
        # Get slowest
        slowest = profiler.get_slowest_functions(1)
        assert len(slowest) == 1
        
        # Get most called
        most_called = profiler.get_most_called_functions(1)
        assert len(most_called) == 1
        assert most_called[0]["call_count"] == 5
