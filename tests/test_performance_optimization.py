"""
Tests for performance optimization and tuning system.
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from performance.profiler import SystemProfiler, AgentProfiler, profile_agent_operation
from performance.optimizer import PerformanceOptimizer, CacheManager, LRUCache
from performance.benchmarks import PerformanceBenchmarks, LoadTester, LoadTestConfig
from performance.metrics import PerformanceMetrics, ResourceMonitor, TimingContext

class TestSystemProfiler:
    """Test system profiler functionality."""
    
    def test_system_profiler_initialization(self):
        """Test system profiler initialization."""
        profiler = SystemProfiler()
        
        assert profiler.profiles == []
        assert not profiler.is_profiling
        assert profiler.profile_interval == 1.0
        
    def test_start_stop_profiling(self):
        """Test starting and stopping profiling."""
        profiler = SystemProfiler()
        
        # Start profiling
        profiler.start_profiling(interval=0.1)
        assert profiler.is_profiling
        
        # Let it collect some profiles
        time.sleep(0.3)
        
        # Stop profiling
        profiler.stop_profiling()
        assert not profiler.is_profiling
        
        # Should have collected some profiles
        assert len(profiler.profiles) > 0
        
    def test_profile_function_context_manager(self):
        """Test function profiling context manager."""
        profiler = SystemProfiler()
        
        def test_function():
            time.sleep(0.1)
            return "test_result"
            
        with profiler.profile_function("test_function"):
            result = test_function()
            
        assert result == "test_result"
        assert len(profiler.profiles) == 1
        assert profiler.profiles[0].function_stats is not None
        assert profiler.profiles[0].function_stats['function_name'] == "test_function"
        
    def test_performance_summary(self):
        """Test performance summary generation."""
        profiler = SystemProfiler()
        
        # Start profiling briefly
        profiler.start_profiling(interval=0.1)
        time.sleep(0.3)
        profiler.stop_profiling()
        
        summary = profiler.get_performance_summary()
        
        assert 'profile_count' in summary
        assert 'cpu_usage' in summary
        assert 'memory_usage' in summary
        assert summary['profile_count'] > 0
        
    def test_export_profiles(self, tmp_path):
        """Test exporting profiles to file."""
        profiler = SystemProfiler()
        
        # Generate some profiles
        profiler.start_profiling(interval=0.1)
        time.sleep(0.2)
        profiler.stop_profiling()
        
        export_file = tmp_path / "profiles.json"
        profiler.export_profiles(str(export_file))
        
        assert export_file.exists()
        
        # Verify file content
        import json
        with open(export_file) as f:
            data = json.load(f)
            
        assert isinstance(data, list)
        assert len(data) > 0

class TestAgentProfiler:
    """Test agent profiler functionality."""
    
    def test_agent_profiler_initialization(self):
        """Test agent profiler initialization."""
        profiler = AgentProfiler()
        
        assert profiler.agent_profiles == {}
        assert profiler._active_sessions == {}
        
    def test_agent_session_lifecycle(self):
        """Test complete agent profiling session."""
        profiler = AgentProfiler()
        agent_id = "test_agent_1"
        
        # Start session
        profiler.start_agent_session(agent_id)
        assert agent_id in profiler._active_sessions
        
        # Mark different phases
        profiler.mark_decision_start(agent_id)
        time.sleep(0.01)
        profiler.mark_decision_end(agent_id)
        
        profiler.mark_memory_retrieval_start(agent_id)
        time.sleep(0.005)
        profiler.mark_memory_retrieval_end(agent_id)
        
        profiler.mark_action_execution_start(agent_id)
        time.sleep(0.02)
        profiler.mark_action_execution_end(agent_id)
        
        # End session
        profile = profiler.end_agent_session(agent_id, memory_usage=50.0, cache_hit_rate=0.8)
        
        assert profile.agent_id == agent_id
        assert profile.decision_time > 0
        assert profile.memory_retrieval_time > 0
        assert profile.action_execution_time > 0
        assert profile.total_response_time > 0
        assert profile.memory_usage == 50.0
        assert profile.cache_hit_rate == 0.8
        
        # Session should be cleaned up
        assert agent_id not in profiler._active_sessions
        
        # Profile should be stored
        assert agent_id in profiler.agent_profiles
        assert len(profiler.agent_profiles[agent_id]) == 1
        
    def test_agent_performance_summary(self):
        """Test agent performance summary."""
        profiler = AgentProfiler()
        agent_id = "test_agent_1"
        
        # Create multiple profiles
        for i in range(5):
            profiler.start_agent_session(agent_id)
            profiler.mark_decision_start(agent_id)
            time.sleep(0.01)
            profiler.mark_decision_end(agent_id)
            profiler.end_agent_session(agent_id, cache_hit_rate=0.7 + i * 0.05)
            
        summary = profiler.get_agent_performance_summary(agent_id)
        
        assert summary['agent_id'] == agent_id
        assert summary['profile_count'] == 5
        assert summary['avg_decision_time'] > 0
        assert summary['avg_cache_hit_rate'] > 0.7
        assert 'performance_trend' in summary
        
    def test_profile_agent_operation_context_manager(self):
        """Test agent operation profiling context manager."""
        profiler = AgentProfiler()
        agent_id = "test_agent_1"
        
        profiler.start_agent_session(agent_id)
        
        with profile_agent_operation(profiler, agent_id, "decision"):
            time.sleep(0.01)
            
        profile = profiler.end_agent_session(agent_id)
        assert profile.decision_time > 0

class TestLRUCache:
    """Test LRU cache implementation."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = LRUCache(max_size=100)
        
        assert cache.max_size == 100
        assert len(cache._cache) == 0
        
    def test_cache_put_get(self):
        """Test basic cache operations."""
        cache = LRUCache(max_size=3)
        
        # Put values
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Get values
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("nonexistent") is None
        
    def test_cache_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(max_size=2)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        
    def test_cache_ttl(self):
        """Test cache TTL expiration."""
        cache = LRUCache(max_size=10, ttl=timedelta(milliseconds=50))
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for TTL expiration
        time.sleep(0.1)
        assert cache.get("key1") is None
        
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=10)
        
        # Generate some hits and misses
        cache.put("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        
        stats = cache.get_stats()
        
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['entries'] == 1
        assert stats['hit_rate'] == 0.5

class TestCacheManager:
    """Test cache manager functionality."""
    
    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        manager = CacheManager()
        
        assert manager._caches == {}
        assert manager._cache_configs == {}
        
    def test_create_and_get_cache(self):
        """Test creating and retrieving caches."""
        manager = CacheManager()
        
        # Create cache
        cache = manager.create_cache("test_cache", max_size=100)
        assert isinstance(cache, LRUCache)
        assert cache.max_size == 100
        
        # Get cache
        retrieved_cache = manager.get_cache("test_cache")
        assert retrieved_cache is cache
        
        # Get non-existent cache
        assert manager.get_cache("nonexistent") is None
        
    def test_cache_stats(self):
        """Test cache statistics collection."""
        manager = CacheManager()
        
        cache = manager.create_cache("test_cache", max_size=10)
        cache.put("key1", "value1")
        cache.get("key1")
        
        stats = manager.get_all_stats()
        
        assert "test_cache" in stats
        assert stats["test_cache"]["hits"] == 1
        assert stats["test_cache"]["max_size"] == 10

class TestPerformanceOptimizer:
    """Test performance optimizer functionality."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = PerformanceOptimizer()
        
        assert optimizer.cache_manager is not None
        assert optimizer.query_optimizer is not None
        assert optimizer.memory_optimizer is not None
        
        # Check default caches were created
        assert optimizer.cache_manager.get_cache("agent_memory") is not None
        assert optimizer.cache_manager.get_cache("llm_responses") is not None
        
    def test_agent_decision_optimization(self):
        """Test agent decision-making optimization."""
        optimizer = PerformanceOptimizer()
        
        # Mock agent profiles
        mock_profiles = [Mock() for _ in range(5)]
        
        result = optimizer.optimize_agent_decision_making(mock_profiles)
        
        assert result.optimization_type == "agent_decision_making"
        assert result.improvement_percentage >= 0
        assert len(result.recommendations) > 0
        assert result in optimizer.optimization_history
        
    def test_database_query_optimization(self):
        """Test database query optimization."""
        optimizer = PerformanceOptimizer()
        
        # Add some slow queries
        optimizer.query_optimizer.track_query("slow_query_1", 2.5)
        optimizer.query_optimizer.track_query("slow_query_2", 1.8)
        
        result = optimizer.optimize_database_queries()
        
        assert result.optimization_type == "database_queries"
        assert len(result.recommendations) > 0
        
    def test_memory_optimization(self):
        """Test memory optimization."""
        optimizer = PerformanceOptimizer()
        
        result = optimizer.optimize_memory_usage()
        
        assert result.optimization_type == "memory_usage"
        assert len(result.recommendations) > 0
        
    def test_comprehensive_optimization_report(self):
        """Test comprehensive optimization report."""
        optimizer = PerformanceOptimizer()
        
        # Run some optimizations
        optimizer.optimize_agent_decision_making([])
        optimizer.optimize_database_queries()
        
        report = optimizer.get_comprehensive_optimization_report()
        
        assert 'timestamp' in report
        assert 'cache_performance' in report
        assert 'optimization_history' in report
        assert 'recommendations' in report
        assert len(report['optimization_history']) == 2

class TestPerformanceBenchmarks:
    """Test performance benchmarking functionality."""
    
    def test_benchmarks_initialization(self):
        """Test benchmarks initialization."""
        benchmarks = PerformanceBenchmarks()
        
        assert benchmarks.benchmark_results == []
        assert benchmarks.baseline_results == {}
        
    def test_agent_decision_benchmark(self):
        """Test agent decision-making benchmark."""
        benchmarks = PerformanceBenchmarks()
        
        def mock_agent_factory():
            return Mock()
            
        result = benchmarks.run_agent_decision_benchmark(mock_agent_factory, iterations=10)
        
        assert result.benchmark_name == "agent_decision_making"
        assert result.iterations == 10
        assert result.operations_per_second > 0
        assert result.success_rate >= 0
        assert len(benchmarks.benchmark_results) == 1
        
    def test_memory_retrieval_benchmark(self):
        """Test memory retrieval benchmark."""
        benchmarks = PerformanceBenchmarks()
        
        mock_memory_system = Mock()
        
        result = benchmarks.run_memory_retrieval_benchmark(mock_memory_system, iterations=50)
        
        assert result.benchmark_name == "memory_retrieval"
        assert result.iterations == 50
        assert result.operations_per_second > 0
        
    def test_communication_benchmark(self):
        """Test communication benchmark."""
        benchmarks = PerformanceBenchmarks()
        
        mock_message_bus = Mock()
        
        result = benchmarks.run_communication_benchmark(mock_message_bus, iterations=100)
        
        assert result.benchmark_name == "agent_communication"
        assert result.iterations == 100
        assert result.operations_per_second > 0
        
    def test_baseline_comparison(self):
        """Test baseline setting and comparison."""
        benchmarks = PerformanceBenchmarks()
        
        # Run benchmark and set baseline
        mock_agent_factory = lambda: Mock()
        benchmarks.run_agent_decision_benchmark(mock_agent_factory, iterations=10)
        benchmarks.set_baseline("agent_decision_making")
        
        # Run another benchmark
        benchmarks.run_agent_decision_benchmark(mock_agent_factory, iterations=10)
        
        comparison = benchmarks.compare_to_baseline("agent_decision_making")
        
        assert comparison is not None
        assert 'benchmark_name' in comparison
        assert 'operations_per_second' in comparison
        assert 'performance_trend' in comparison
        
    def test_benchmark_summary(self):
        """Test benchmark summary."""
        benchmarks = PerformanceBenchmarks()
        
        # Run multiple benchmarks
        mock_agent_factory = lambda: Mock()
        benchmarks.run_agent_decision_benchmark(mock_agent_factory, iterations=5)
        benchmarks.run_agent_decision_benchmark(mock_agent_factory, iterations=5)
        
        summary = benchmarks.get_benchmark_summary()
        
        assert "agent_decision_making" in summary
        assert summary["agent_decision_making"]["total_runs"] == 2
        assert "latest_result" in summary["agent_decision_making"]
        
    def test_export_results(self, tmp_path):
        """Test exporting benchmark results."""
        benchmarks = PerformanceBenchmarks()
        
        # Run a benchmark
        mock_agent_factory = lambda: Mock()
        benchmarks.run_agent_decision_benchmark(mock_agent_factory, iterations=5)
        
        export_file = tmp_path / "benchmarks.json"
        benchmarks.export_results(str(export_file))
        
        assert export_file.exists()
        
        # Verify file content
        import json
        with open(export_file) as f:
            data = json.load(f)
            
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["benchmark_name"] == "agent_decision_making"

class TestLoadTester:
    """Test load testing functionality."""
    
    def test_load_tester_initialization(self):
        """Test load tester initialization."""
        load_tester = LoadTester()
        
        assert load_tester.load_test_results == []
        
    def test_concurrent_agent_load_test(self):
        """Test concurrent agent load test."""
        load_tester = LoadTester()
        
        config = LoadTestConfig(
            concurrent_users=3,
            duration=timedelta(seconds=1),
            ramp_up_time=timedelta(milliseconds=100),
            operations_per_user=5,
            think_time_range=(0.01, 0.02)
        )
        
        def mock_agent_factory():
            return Mock()
            
        result = load_tester.run_concurrent_agent_load_test(mock_agent_factory, config)
        
        assert result.config == config
        assert result.total_operations > 0
        assert result.operations_per_second > 0
        assert result.successful_operations >= 0
        assert len(load_tester.load_test_results) == 1

class TestPerformanceMetrics:
    """Test performance metrics functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PerformanceMetrics()
        
        assert len(metrics.metrics) == 0
        assert len(metrics.counters) == 0
        assert len(metrics.gauges) == 0
        
    def test_counter_metrics(self):
        """Test counter metrics."""
        metrics = PerformanceMetrics()
        
        metrics.record_counter("test_counter", 5)
        metrics.record_counter("test_counter", 3)
        
        assert metrics.get_counter_value("test_counter") == 8
        
    def test_gauge_metrics(self):
        """Test gauge metrics."""
        metrics = PerformanceMetrics()
        
        metrics.record_gauge("test_gauge", 42.5)
        metrics.record_gauge("test_gauge", 37.2)
        
        assert metrics.get_gauge_value("test_gauge") == 37.2
        
    def test_histogram_metrics(self):
        """Test histogram metrics."""
        metrics = PerformanceMetrics()
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            metrics.record_histogram("test_histogram", value)
            
        stats = metrics.get_histogram_stats("test_histogram")
        
        assert stats['count'] == 5
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['mean'] == 3.0
        assert stats['median'] == 3.0
        
    def test_timing_context(self):
        """Test timing context manager."""
        metrics = PerformanceMetrics()
        
        with TimingContext(metrics, "test_operation"):
            time.sleep(0.01)
            
        stats = metrics.get_histogram_stats("test_operation_duration")
        assert stats['count'] == 1
        assert stats['min'] > 0.005  # Should be at least 5ms
        
    def test_metric_history(self):
        """Test metric history retrieval."""
        metrics = PerformanceMetrics()
        
        metrics.record_gauge("test_metric", 10.0)
        time.sleep(0.01)
        metrics.record_gauge("test_metric", 20.0)
        
        history = metrics.get_metric_history("test_metric")
        assert len(history) == 2
        assert history[0].value == 10.0
        assert history[1].value == 20.0
        
        # Test time window filtering
        recent_history = metrics.get_metric_history("test_metric", timedelta(milliseconds=5))
        assert len(recent_history) == 1
        assert recent_history[0].value == 20.0
        
    def test_metrics_summary(self):
        """Test metrics summary."""
        metrics = PerformanceMetrics()
        
        metrics.record_counter("test_counter", 5)
        metrics.record_gauge("test_gauge", 42.0)
        metrics.record_histogram("test_histogram", 1.5)
        
        summary = metrics.get_all_metrics_summary()
        
        assert summary['counters']['test_counter'] == 5
        assert summary['gauges']['test_gauge'] == 42.0
        assert 'test_histogram' in summary['histograms']
        assert summary['metric_count'] == 3
        
    def test_export_metrics(self, tmp_path):
        """Test metrics export."""
        metrics = PerformanceMetrics()
        
        metrics.record_counter("test_counter", 1)
        metrics.record_gauge("test_gauge", 10.0)
        
        export_file = tmp_path / "metrics.json"
        metrics.export_metrics(str(export_file))
        
        assert export_file.exists()
        
        # Verify file content
        import json
        with open(export_file) as f:
            data = json.load(f)
            
        assert 'export_timestamp' in data
        assert 'metrics' in data
        assert len(data['metrics']) == 2

class TestResourceMonitor:
    """Test resource monitoring functionality."""
    
    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor(collection_interval=0.1)
        
        assert monitor.collection_interval == 0.1
        assert not monitor.is_monitoring
        assert len(monitor.resource_history) == 0
        
    def test_start_stop_monitoring(self):
        """Test starting and stopping resource monitoring."""
        monitor = ResourceMonitor(collection_interval=0.1)
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.is_monitoring
        
        # Let it collect some data
        time.sleep(0.3)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.is_monitoring
        
        # Should have collected some snapshots
        assert len(monitor.resource_history) > 0
        
    def test_current_snapshot(self):
        """Test current resource snapshot."""
        monitor = ResourceMonitor()
        
        snapshot = monitor.get_current_snapshot()
        
        assert snapshot.cpu_percent >= 0
        assert snapshot.memory_percent >= 0
        assert snapshot.memory_used_mb >= 0
        assert snapshot.process_count > 0
        
    def test_resource_summary(self):
        """Test resource usage summary."""
        monitor = ResourceMonitor(collection_interval=0.1)
        
        # Start monitoring briefly
        monitor.start_monitoring()
        time.sleep(0.3)
        monitor.stop_monitoring()
        
        summary = monitor.get_resource_summary()
        
        assert 'time_range' in summary
        assert 'cpu_usage' in summary
        assert 'memory_usage' in summary
        assert 'process_info' in summary
        assert summary['sample_count'] > 0
        
    def test_anomaly_detection(self):
        """Test resource anomaly detection."""
        monitor = ResourceMonitor()
        
        # Generate some fake snapshots with anomalies
        from performance.metrics import ResourceSnapshot
        
        base_time = datetime.now()
        for i in range(20):
            # Most snapshots have normal CPU usage
            cpu_percent = 20.0 if i != 10 else 95.0  # Anomaly at index 10
            
            snapshot = ResourceSnapshot(
                timestamp=base_time + timedelta(seconds=i),
                cpu_percent=cpu_percent,
                memory_percent=50.0,
                memory_used_mb=1000.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_io_sent_mb=0.0,
                network_io_recv_mb=0.0,
                process_count=100,
                thread_count=10
            )
            monitor.resource_history.append(snapshot)
            
        anomalies = monitor.detect_resource_anomalies()
        
        # Should detect the CPU anomaly
        assert len(anomalies) > 0
        cpu_anomalies = [a for a in anomalies if a['type'] == 'cpu_anomaly']
        assert len(cpu_anomalies) > 0
        assert cpu_anomalies[0]['value'] == 95.0

class TestPerformanceIntegration:
    """Integration tests for performance system."""
    
    def test_end_to_end_performance_monitoring(self):
        """Test end-to-end performance monitoring scenario."""
        # Initialize components
        profiler = SystemProfiler()
        agent_profiler = AgentProfiler()
        optimizer = PerformanceOptimizer()
        benchmarks = PerformanceBenchmarks()
        metrics = PerformanceMetrics()
        
        # Start system profiling
        profiler.start_profiling(interval=0.1)
        
        # Simulate agent operations
        agent_id = "test_agent"
        agent_profiler.start_agent_session(agent_id)
        
        with profile_agent_operation(agent_profiler, agent_id, "decision"):
            time.sleep(0.02)
            
        agent_profile = agent_profiler.end_agent_session(agent_id, cache_hit_rate=0.75)
        
        # Record some metrics
        metrics.record_counter("operations_completed", 1)
        metrics.record_timing("operation_duration", agent_profile.total_response_time)
        
        # Run optimization
        optimization_result = optimizer.optimize_agent_decision_making([agent_profile])
        
        # Run benchmark
        mock_agent_factory = lambda: Mock()
        benchmark_result = benchmarks.run_agent_decision_benchmark(mock_agent_factory, iterations=5)
        
        # Stop profiling
        profiler.stop_profiling()
        
        # Verify all components worked together
        assert len(profiler.profiles) > 0
        assert agent_profile.decision_time > 0
        assert optimization_result.improvement_percentage >= 0
        assert benchmark_result.operations_per_second > 0
        assert metrics.get_counter_value("operations_completed") == 1
        
        # Generate comprehensive report
        system_summary = profiler.get_performance_summary()
        optimization_report = optimizer.get_comprehensive_optimization_report()
        benchmark_summary = benchmarks.get_benchmark_summary()
        metrics_summary = metrics.get_all_metrics_summary()
        
        assert all([
            system_summary,
            optimization_report,
            benchmark_summary,
            metrics_summary
        ])

if __name__ == "__main__":
    pytest.main([__file__])