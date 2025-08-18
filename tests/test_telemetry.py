"""
Comprehensive tests for the telemetry and observability system.

Tests cover:
- OpenTelemetry integration
- Distributed tracing functionality
- Time-warping capabilities
- Performance profiling accuracy
- Telemetry completeness validation
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import threading

from monitoring.telemetry import (
    TelemetrySystem, TelemetryEvent, PerformanceMetrics, TraceContext,
    TimeWarpManager, DistributedTracer, MetricsCollector, PerformanceProfiler,
    get_telemetry, initialize_telemetry
)
from monitoring.observability import (
    ObservabilitySystem, AlertManager, LogCorrelator, TopologyDiscovery,
    DashboardManager, Alert, AlertSeverity, AlertStatus, LogEntry,
    get_observability, initialize_observability
)


class TestTelemetryEvent:
    """Test TelemetryEvent data structure."""
    
    def test_event_creation(self):
        """Test creating a telemetry event."""
        event = TelemetryEvent(
            event_id="test_001",
            timestamp=datetime.utcnow(),
            event_type="test_event",
            source="test_source",
            data={"key": "value"},
            trace_id="trace_123",
            span_id="span_456"
        )
        
        assert event.event_id == "test_001"
        assert event.event_type == "test_event"
        assert event.source == "test_source"
        assert event.data == {"key": "value"}
        assert event.trace_id == "trace_123"
        assert event.span_id == "span_456"
    
    def test_event_serialization(self):
        """Test event serialization to dictionary."""
        timestamp = datetime.utcnow()
        event = TelemetryEvent(
            event_id="test_001",
            timestamp=timestamp,
            event_type="test_event",
            source="test_source",
            data={"key": "value"}
        )
        
        event_dict = event.to_dict()
        assert event_dict["event_id"] == "test_001"
        assert event_dict["timestamp"] == timestamp.isoformat()
        assert event_dict["data"] == {"key": "value"}


class TestTimeWarpManager:
    """Test time-warping capabilities."""
    
    def test_normal_time(self):
        """Test normal time operation."""
        time_warp = TimeWarpManager()
        current_time = time_warp.get_current_time()
        
        # Should be close to actual current time
        assert abs((datetime.utcnow() - current_time).total_seconds()) < 1
    
    def test_replay_mode(self):
        """Test time-warped replay functionality."""
        time_warp = TimeWarpManager()
        start_time = datetime.utcnow() - timedelta(hours=1)
        
        time_warp.start_replay(start_time, speed=2.0)
        assert time_warp.is_replaying
        assert time_warp.playback_speed == 2.0
        
        # Get time after a small delay
        time.sleep(0.1)
        warped_time = time_warp.get_current_time()
        
        # Should be based on replay start time
        assert warped_time >= start_time
        
        time_warp.stop_replay()
        assert not time_warp.is_replaying
    
    def test_time_offset(self):
        """Test time offset functionality."""
        time_warp = TimeWarpManager()
        offset = timedelta(hours=2)
        
        time_warp.set_time_offset(offset)
        assert time_warp.time_offset == offset


class TestDistributedTracer:
    """Test distributed tracing functionality."""
    
    @patch('monitoring.telemetry.trace')
    def test_span_creation(self, mock_trace):
        """Test span creation and context management."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span_context = Mock()
        mock_span_context.span_id = 12345
        mock_span_context.trace_id = 67890
        mock_span.get_span_context.return_value = mock_span_context
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        mock_trace.get_tracer.return_value = mock_tracer
        
        tracer = DistributedTracer("test_service")
        
        with tracer.start_span("test_operation") as span:
            assert span == mock_span
        
        mock_tracer.start_as_current_span.assert_called_once()
    
    @patch('monitoring.telemetry.trace')
    def test_span_error_handling(self, mock_trace):
        """Test span error handling."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span_context = Mock()
        mock_span_context.span_id = 12345
        mock_span_context.trace_id = 67890
        mock_span.get_span_context.return_value = mock_span_context
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        mock_trace.get_tracer.return_value = mock_tracer
        
        tracer = DistributedTracer("test_service")
        
        with pytest.raises(ValueError):
            with tracer.start_span("test_operation") as span:
                raise ValueError("Test error")
        
        # Verify error was recorded on span
        mock_span.set_status.assert_called_once()
        mock_span.set_attribute.assert_called()


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    @patch('monitoring.telemetry.metrics')
    def test_counter_creation(self, mock_metrics):
        """Test counter metric creation."""
        mock_meter = Mock()
        mock_counter = Mock()
        mock_meter.create_counter.return_value = mock_counter
        mock_metrics.get_meter.return_value = mock_meter
        
        collector = MetricsCollector()
        counter = collector.get_counter("test_counter", "Test counter")
        
        assert counter == mock_counter
        mock_meter.create_counter.assert_called_with(
            name="test_counter",
            description="Test counter"
        )
    
    @patch('monitoring.telemetry.metrics')
    def test_performance_metric_recording(self, mock_metrics):
        """Test performance metric recording."""
        mock_meter = Mock()
        mock_histogram = Mock()
        mock_meter.create_histogram.return_value = mock_histogram
        mock_metrics.get_meter.return_value = mock_meter
        
        collector = MetricsCollector()
        
        metric = PerformanceMetrics(
            component="test_component",
            operation="test_operation",
            duration_ms=100.0,
            cpu_usage=50.0,
            memory_usage=1024.0,
            success=True
        )
        
        collector.record_performance_metric(metric)
        
        # Verify histogram recording
        mock_histogram.record.assert_called()
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        collector = MetricsCollector()
        
        # Add test metrics
        for i in range(5):
            metric = PerformanceMetrics(
                component="test_component",
                operation="test_op",
                duration_ms=100.0 + i * 10,
                cpu_usage=50.0 + i * 5,
                memory_usage=1024.0,
                success=i < 4  # One failure
            )
            collector.record_performance_metric(metric)
        
        summary = collector.get_performance_summary("test_component")
        
        assert summary["component"] == "test_component"
        assert summary["total_operations"] == 5
        assert summary["success_rate"] == 0.8  # 4/5
        assert "duration_stats" in summary
        assert "cpu_stats" in summary


class TestPerformanceProfiler:
    """Test performance profiling functionality."""
    
    def test_operation_profiling(self):
        """Test operation profiling context manager."""
        profiler = PerformanceProfiler()
        
        with profiler.profile_operation("test_operation", "test_component"):
            time.sleep(0.01)  # Small delay for measurable duration
        
        # Check that profiling data was recorded
        assert "test_component" in profiler.profiling_data
        assert len(profiler.profiling_data["test_component"]) == 1
        
        metric = profiler.profiling_data["test_component"][0]
        assert metric.operation == "test_operation"
        assert metric.duration_ms > 0
        assert metric.success
    
    def test_error_profiling(self):
        """Test profiling with errors."""
        profiler = PerformanceProfiler()
        
        with pytest.raises(ValueError):
            with profiler.profile_operation("error_operation", "test_component"):
                raise ValueError("Test error")
        
        # Check that error was recorded
        metric = profiler.profiling_data["test_component"][0]
        assert not metric.success
        assert metric.error_message == "Test error"
    
    def test_bottleneck_detection(self):
        """Test bottleneck detection."""
        profiler = PerformanceProfiler()
        
        # Create a slow operation
        with profiler.profile_operation("slow_operation", "test_component"):
            # Simulate slow operation by setting duration directly
            pass
        
        # Manually add a slow metric to trigger bottleneck detection
        slow_metric = PerformanceMetrics(
            component="test_component",
            operation="slow_operation",
            duration_ms=6000.0,  # 6 seconds - above threshold
            cpu_usage=100.0,
            memory_usage=1024.0,
            success=True
        )
        profiler.profiling_data["test_component"].append(slow_metric)
        profiler._analyze_bottlenecks("test_component", slow_metric)
        
        bottlenecks = profiler.get_bottlenecks()
        assert len(bottlenecks) > 0
        assert bottlenecks[0]["type"] == "slow_operation"
    
    def test_performance_report(self):
        """Test performance report generation."""
        profiler = PerformanceProfiler()
        
        # Add some test data
        metric = PerformanceMetrics(
            component="test_component",
            operation="test_operation",
            duration_ms=100.0,
            cpu_usage=50.0,
            memory_usage=1024.0,
            success=True
        )
        profiler.profiling_data["test_component"].append(metric)
        
        report = profiler.get_performance_report()
        
        assert "timestamp" in report
        assert "components" in report
        assert "bottlenecks" in report
        assert "summary" in report
        assert "test_component" in report["components"]


class TestTelemetrySystem:
    """Test main telemetry system."""
    
    @patch('monitoring.telemetry.trace')
    @patch('monitoring.telemetry.metrics')
    def test_system_initialization(self, mock_metrics, mock_trace):
        """Test telemetry system initialization."""
        system = TelemetrySystem("test_service")
        
        assert system.service_name == "test_service"
        assert system.time_warp is not None
        assert system.tracer is not None
        assert system.metrics is not None
        assert system.profiler is not None
    
    def test_event_recording(self):
        """Test event recording functionality."""
        system = TelemetrySystem("test_service")
        
        event_id = system.record_event(
            "test_event",
            "test_source",
            {"key": "value"}
        )
        
        assert event_id is not None
        assert len(system.events) == 1
        
        event = system.events[0]
        assert event.event_type == "test_event"
        assert event.source == "test_source"
        assert event.data == {"key": "value"}
    
    def test_event_filtering(self):
        """Test event filtering functionality."""
        system = TelemetrySystem("test_service")
        
        # Record multiple events
        system.record_event("type1", "source1", {"data": 1})
        system.record_event("type2", "source1", {"data": 2})
        system.record_event("type1", "source2", {"data": 3})
        
        # Test filtering by type
        type1_events = system.get_events(event_type="type1")
        assert len(type1_events) == 2
        
        # Test filtering by source
        source1_events = system.get_events(source="source1")
        assert len(source1_events) == 2
    
    @patch('monitoring.telemetry.trace')
    @patch('monitoring.telemetry.metrics')
    def test_trace_operation(self, mock_metrics, mock_trace):
        """Test operation tracing."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        mock_trace.get_tracer.return_value = mock_tracer
        
        system = TelemetrySystem("test_service")
        system.tracer.tracer = mock_tracer
        
        with system.trace_operation("test_op", "test_component") as span:
            assert span == mock_span
        
        mock_span.set_attribute.assert_called()
    
    def test_forensic_replay(self):
        """Test forensic replay functionality."""
        system = TelemetrySystem("test_service")
        
        # Record some events
        start_time = datetime.utcnow() - timedelta(minutes=10)
        end_time = datetime.utcnow() - timedelta(minutes=5)
        
        system.record_event("test_event", "test_source", {"data": "test"})
        
        replay_data = system.start_forensic_replay(start_time, end_time, speed=2.0)
        
        assert "replay_id" in replay_data
        assert replay_data["speed"] == 2.0
        assert system.time_warp.is_replaying
        
        system.stop_forensic_replay()
        assert not system.time_warp.is_replaying
    
    def test_system_health(self):
        """Test system health monitoring."""
        system = TelemetrySystem("test_service")
        
        health = system.get_system_health()
        
        assert "health_score" in health
        assert "status" in health
        assert "timestamp" in health
        assert health["health_score"] >= 0
        assert health["health_score"] <= 100
    
    def test_telemetry_export(self):
        """Test telemetry data export."""
        system = TelemetrySystem("test_service")
        
        # Add some test data
        system.record_event("test_event", "test_source", {"data": "test"})
        
        export_data = system.export_telemetry_data("json")
        
        assert isinstance(export_data, str)
        parsed_data = json.loads(export_data)
        assert "service_name" in parsed_data
        assert "events" in parsed_data
        assert "performance_report" in parsed_data


class TestAlertManager:
    """Test alert management functionality."""
    
    def test_alert_creation(self):
        """Test alert creation."""
        alert_manager = AlertManager()
        
        alert_id = alert_manager.create_alert(
            "Test Alert",
            "Test description",
            AlertSeverity.HIGH,
            "test_source"
        )
        
        assert alert_id is not None
        assert alert_id in alert_manager.alerts
        
        alert = alert_manager.alerts[alert_id]
        assert alert.title == "Test Alert"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.status == AlertStatus.ACTIVE
    
    def test_alert_acknowledgment(self):
        """Test alert acknowledgment."""
        alert_manager = AlertManager()
        
        alert_id = alert_manager.create_alert(
            "Test Alert",
            "Test description",
            AlertSeverity.MEDIUM,
            "test_source"
        )
        
        success = alert_manager.acknowledge_alert(alert_id, "test_user")
        assert success
        
        alert = alert_manager.alerts[alert_id]
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "test_user"
        assert alert.acknowledged_at is not None
    
    def test_alert_resolution(self):
        """Test alert resolution."""
        alert_manager = AlertManager()
        
        alert_id = alert_manager.create_alert(
            "Test Alert",
            "Test description",
            AlertSeverity.LOW,
            "test_source"
        )
        
        success = alert_manager.resolve_alert(alert_id)
        assert success
        
        alert = alert_manager.alerts[alert_id]
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None
    
    def test_alert_filtering(self):
        """Test alert filtering."""
        alert_manager = AlertManager()
        
        # Create alerts with different properties
        alert_manager.create_alert("Alert 1", "Desc 1", AlertSeverity.HIGH, "source1")
        alert_manager.create_alert("Alert 2", "Desc 2", AlertSeverity.LOW, "source2")
        alert_manager.create_alert("Alert 3", "Desc 3", AlertSeverity.HIGH, "source1")
        
        # Test filtering by severity
        high_alerts = alert_manager.get_alerts(severity=AlertSeverity.HIGH)
        assert len(high_alerts) == 2
        
        # Test filtering by source
        source1_alerts = alert_manager.get_alerts(source="source1")
        assert len(source1_alerts) == 2
    
    def test_alert_rules(self):
        """Test alert rule evaluation."""
        alert_manager = AlertManager()
        
        # Add a test rule
        def test_rule(data):
            if data.get("error_count", 0) > 5:
                return Alert(
                    alert_id=f"rule_alert_{int(time.time())}",
                    title="High Error Count",
                    description="Too many errors",
                    severity=AlertSeverity.HIGH,
                    status=AlertStatus.ACTIVE,
                    source="error_monitor",
                    timestamp=datetime.utcnow(),
                    tags={},
                    metadata=data
                )
            return None
        
        alert_manager.add_alert_rule(test_rule)
        
        # Evaluate with data that should trigger alert
        alert_manager.evaluate_alert_rules({"error_count": 10})
        
        # Check that alert was created
        alerts = alert_manager.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].title == "High Error Count"


class TestLogCorrelator:
    """Test log correlation functionality."""
    
    def test_log_entry_addition(self):
        """Test adding log entries."""
        correlator = LogCorrelator()
        
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            source="test_source",
            message="Test message",
            context={"key": "value"}
        )
        
        correlator.add_log_entry(log_entry)
        
        assert len(correlator.log_buffer) == 1
        assert correlator.log_buffer[0] == log_entry
    
    def test_trace_correlation(self):
        """Test finding logs by trace ID."""
        correlator = LogCorrelator()
        
        # Add logs with same trace ID
        trace_id = "trace_123"
        for i in range(3):
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level="INFO",
                source=f"source_{i}",
                message=f"Message {i}",
                context={},
                trace_id=trace_id
            )
            correlator.add_log_entry(log_entry)
        
        # Add log with different trace ID
        other_log = LogEntry(
            timestamp=datetime.utcnow(),
            level="INFO",
            source="other_source",
            message="Other message",
            context={},
            trace_id="other_trace"
        )
        correlator.add_log_entry(other_log)
        
        related_logs = correlator.find_related_logs(trace_id)
        assert len(related_logs) == 3
        assert all(log.trace_id == trace_id for log in related_logs)
    
    def test_log_search(self):
        """Test log search functionality."""
        correlator = LogCorrelator()
        
        # Add test logs
        logs = [
            LogEntry(datetime.utcnow(), "INFO", "source1", "User login successful", {}),
            LogEntry(datetime.utcnow(), "ERROR", "source2", "Database connection failed", {}),
            LogEntry(datetime.utcnow(), "INFO", "source3", "User logout", {})
        ]
        
        for log in logs:
            correlator.add_log_entry(log)
        
        # Search for logs containing "user"
        user_logs = correlator.search_logs("user")
        assert len(user_logs) == 2
        
        # Search for logs containing "database"
        db_logs = correlator.search_logs("database")
        assert len(db_logs) == 1


class TestObservabilitySystem:
    """Test complete observability system integration."""
    
    def test_system_initialization(self):
        """Test observability system initialization."""
        system = ObservabilitySystem()
        
        assert system.telemetry is not None
        assert system.alert_manager is not None
        assert system.log_correlator is not None
        assert system.topology_discovery is not None
        assert system.dashboard_manager is not None
    
    def test_observability_report(self):
        """Test comprehensive observability report."""
        system = ObservabilitySystem()
        
        report = system.get_observability_report()
        
        assert "timestamp" in report
        assert "system_health" in report
        assert "alerts" in report
        assert "performance" in report
        assert "dashboard_data" in report
    
    def test_monitoring_integration(self):
        """Test integration between monitoring components."""
        system = ObservabilitySystem()
        
        # Create an alert
        alert_id = system.alert_manager.create_alert(
            "Integration Test Alert",
            "Testing integration",
            AlertSeverity.MEDIUM,
            "integration_test"
        )
        
        # Add a log entry
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level="WARN",
            source="integration_test",
            message="Integration test warning",
            context={"alert_id": alert_id}
        )
        system.log_correlator.add_log_entry(log_entry)
        
        # Record telemetry event
        system.telemetry.record_event(
            "integration_test",
            "integration_test",
            {"alert_id": alert_id, "log_added": True}
        )
        
        # Verify data is accessible through report
        report = system.get_observability_report()
        assert report["alerts"]["active"] >= 1


class TestTelemetryCompleteness:
    """Test telemetry completeness and accuracy."""
    
    def test_trace_completeness(self):
        """Test that all operations are properly traced."""
        system = TelemetrySystem("completeness_test")
        
        operations = ["op1", "op2", "op3"]
        
        for op in operations:
            with system.trace_operation(op, "test_component"):
                time.sleep(0.001)  # Small delay
        
        # Verify all operations were recorded
        events = system.get_events()
        assert len(events) >= len(operations)
    
    def test_metric_accuracy(self):
        """Test metric measurement accuracy."""
        system = TelemetrySystem("accuracy_test")
        
        # Perform a measured operation
        start_time = time.time()
        with system.profiler.profile_operation("timed_operation", "test_component"):
            time.sleep(0.05)  # 50ms delay
        end_time = time.time()
        
        actual_duration = (end_time - start_time) * 1000  # Convert to ms
        
        # Get recorded metric
        metrics = system.profiler.profiling_data["test_component"]
        assert len(metrics) == 1
        
        recorded_duration = metrics[0].duration_ms
        
        # Allow for some measurement variance (±10ms)
        assert abs(recorded_duration - actual_duration) < 10
    
    def test_event_ordering(self):
        """Test that events maintain proper ordering."""
        system = TelemetrySystem("ordering_test")
        
        event_ids = []
        for i in range(10):
            event_id = system.record_event(
                "ordered_event",
                "test_source",
                {"sequence": i}
            )
            event_ids.append(event_id)
            time.sleep(0.001)  # Small delay to ensure different timestamps
        
        events = system.get_events(event_type="ordered_event")
        
        # Events should be in chronological order
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i-1].timestamp


# Integration tests
class TestSystemIntegration:
    """Test integration between telemetry and observability systems."""
    
    def test_telemetry_observability_integration(self):
        """Test integration between telemetry and observability."""
        # Initialize both systems
        telemetry = initialize_telemetry("integration_test")
        observability = initialize_observability()
        
        # Record some telemetry data
        with telemetry.trace_operation("integration_op", "integration_component"):
            telemetry.record_event(
                "integration_event",
                "integration_source",
                {"test": "data"}
            )
        
        # Generate observability report
        report = observability.get_observability_report()
        
        # Verify telemetry data is included
        assert "system_health" in report
        assert "performance" in report
    
    def test_end_to_end_monitoring(self):
        """Test end-to-end monitoring scenario."""
        telemetry = initialize_telemetry("e2e_test")
        observability = initialize_observability()
        
        # Simulate a complete monitoring scenario
        
        # 1. Record normal operations
        for i in range(5):
            with telemetry.trace_operation(f"normal_op_{i}", "e2e_component"):
                telemetry.record_event(
                    "normal_operation",
                    "e2e_source",
                    {"operation_id": i}
                )
        
        # 2. Simulate an error condition
        telemetry.record_event(
            "error",
            "e2e_source",
            {"error_type": "simulation", "severity": "high"}
        )
        
        # 3. Create an alert
        observability.alert_manager.create_alert(
            "E2E Test Alert",
            "End-to-end test alert",
            AlertSeverity.HIGH,
            "e2e_source"
        )
        
        # 4. Generate comprehensive report
        report = observability.get_observability_report()
        
        # Verify all components are working
        assert report["alerts"]["active"] >= 1
        assert "system_health" in report
        assert "performance" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])