"""
Comprehensive telemetry and observability system with OpenTelemetry integration.

This module provides:
- OpenTelemetry integration for traces, logs, and metrics
- Distributed tracing across agent interactions
- Time-warping capabilities for forensic analysis
- Performance profiling and bottleneck identification
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from threading import Lock
import logging
import traceback
from collections import defaultdict, deque
import statistics

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.semconv.trace import SpanAttributes


@dataclass
class TelemetryEvent:
    """Represents a telemetry event with full context."""
    event_id: str
    timestamp: datetime
    event_type: str
    source: str
    data: Dict[str, Any]
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    duration_ms: Optional[float] = None
    tags: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class PerformanceMetrics:
    """Performance metrics for system components."""
    component: str
    operation: str
    duration_ms: float
    cpu_usage: float
    memory_usage: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class TraceContext:
    """Context for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    tags: Dict[str, str]
    baggage: Dict[str, str]


class TimeWarpManager:
    """Manages time-warping capabilities for forensic analysis."""
    
    def __init__(self):
        self.time_offset = timedelta(0)
        self.playback_speed = 1.0
        self.is_replaying = False
        self.replay_start_time = None
        self.original_start_time = None
        self._lock = Lock()
    
    def start_replay(self, start_time: datetime, speed: float = 1.0):
        """Start time-warped replay from a specific point."""
        with self._lock:
            self.is_replaying = True
            self.replay_start_time = datetime.utcnow()
            self.original_start_time = start_time
            self.playback_speed = speed
    
    def stop_replay(self):
        """Stop time-warped replay."""
        with self._lock:
            self.is_replaying = False
            self.replay_start_time = None
            self.original_start_time = None
            self.playback_speed = 1.0
    
    def get_current_time(self) -> datetime:
        """Get current time considering time-warp state."""
        with self._lock:
            if not self.is_replaying:
                return datetime.utcnow()
            
            elapsed = datetime.utcnow() - self.replay_start_time
            warped_elapsed = elapsed * self.playback_speed
            return self.original_start_time + warped_elapsed
    
    def set_time_offset(self, offset: timedelta):
        """Set time offset for forensic analysis."""
        with self._lock:
            self.time_offset = offset


class DistributedTracer:
    """Manages distributed tracing across agent interactions."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = trace.get_tracer(__name__)
        self.active_spans = {}
        self._lock = Lock()
    
    @contextmanager
    def start_span(self, operation_name: str, parent_context=None, **kwargs):
        """Start a new span with proper context propagation."""
        with self.tracer.start_as_current_span(
            operation_name,
            context=parent_context,
            **kwargs
        ) as span:
            span_id = format(span.get_span_context().span_id, '016x')
            trace_id = format(span.get_span_context().trace_id, '032x')
            
            with self._lock:
                self.active_spans[span_id] = {
                    'span': span,
                    'start_time': datetime.utcnow(),
                    'operation': operation_name
                }
            
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise
            finally:
                with self._lock:
                    self.active_spans.pop(span_id, None)
    
    def add_event(self, span, event_name: str, attributes: Dict[str, Any] = None):
        """Add an event to the current span."""
        if attributes is None:
            attributes = {}
        span.add_event(event_name, attributes)
    
    def set_attributes(self, span, attributes: Dict[str, Any]):
        """Set attributes on the current span."""
        for key, value in attributes.items():
            span.set_attribute(key, value)


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self):
        self.meter = metrics.get_meter(__name__)
        self.counters = {}
        self.histograms = {}
        self.gauges = {}
        self.metrics_data = defaultdict(list)
        self._lock = Lock()
    
    def get_counter(self, name: str, description: str = "") -> metrics.Counter:
        """Get or create a counter metric."""
        if name not in self.counters:
            self.counters[name] = self.meter.create_counter(
                name=name,
                description=description
            )
        return self.counters[name]
    
    def get_histogram(self, name: str, description: str = "") -> metrics.Histogram:
        """Get or create a histogram metric."""
        if name not in self.histograms:
            self.histograms[name] = self.meter.create_histogram(
                name=name,
                description=description
            )
        return self.histograms[name]
    
    def get_gauge(self, name: str, description: str = "") -> metrics.ObservableGauge:
        """Get or create a gauge metric."""
        if name not in self.gauges:
            self.gauges[name] = self.meter.create_observable_gauge(
                name=name,
                description=description
            )
        return self.gauges[name]
    
    def record_performance_metric(self, metric: PerformanceMetrics):
        """Record a performance metric."""
        with self._lock:
            self.metrics_data[metric.component].append(metric)
            
            # Record to OpenTelemetry
            duration_histogram = self.get_histogram(
                f"{metric.component}_duration_ms",
                f"Duration of {metric.component} operations"
            )
            duration_histogram.record(
                metric.duration_ms,
                {"operation": metric.operation, "success": str(metric.success)}
            )
            
            cpu_histogram = self.get_histogram(
                f"{metric.component}_cpu_usage",
                f"CPU usage of {metric.component}"
            )
            cpu_histogram.record(
                metric.cpu_usage,
                {"operation": metric.operation}
            )
    
    def get_performance_summary(self, component: str, time_window: timedelta = None) -> Dict[str, Any]:
        """Get performance summary for a component."""
        with self._lock:
            metrics_list = self.metrics_data.get(component, [])
            
            if time_window:
                cutoff_time = datetime.utcnow() - time_window
                metrics_list = [m for m in metrics_list if m.timestamp >= cutoff_time]
            
            if not metrics_list:
                return {}
            
            durations = [m.duration_ms for m in metrics_list]
            cpu_usages = [m.cpu_usage for m in metrics_list]
            success_rate = sum(1 for m in metrics_list if m.success) / len(metrics_list)
            
            return {
                "component": component,
                "total_operations": len(metrics_list),
                "success_rate": success_rate,
                "duration_stats": {
                    "mean": statistics.mean(durations),
                    "median": statistics.median(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "p95": statistics.quantiles(durations, n=20)[18] if len(durations) > 1 else durations[0]
                },
                "cpu_stats": {
                    "mean": statistics.mean(cpu_usages),
                    "max": max(cpu_usages)
                }
            }


class PerformanceProfiler:
    """Profiles system performance and identifies bottlenecks."""
    
    def __init__(self):
        self.profiling_data = defaultdict(list)
        self.bottlenecks = []
        self._lock = Lock()
    
    @contextmanager
    def profile_operation(self, operation_name: str, component: str):
        """Profile an operation and collect performance data."""
        start_time = time.time()
        start_cpu = time.process_time()
        
        try:
            yield
            success = True
            error_msg = None
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            end_time = time.time()
            end_cpu = time.process_time()
            
            duration_ms = (end_time - start_time) * 1000
            cpu_usage = (end_cpu - start_cpu) * 1000
            
            metric = PerformanceMetrics(
                component=component,
                operation=operation_name,
                duration_ms=duration_ms,
                cpu_usage=cpu_usage,
                memory_usage=0.0,  # Would need psutil for actual memory usage
                success=success,
                error_message=error_msg
            )
            
            with self._lock:
                self.profiling_data[component].append(metric)
                self._analyze_bottlenecks(component, metric)
    
    def _analyze_bottlenecks(self, component: str, metric: PerformanceMetrics):
        """Analyze performance data to identify bottlenecks."""
        # Simple bottleneck detection based on thresholds
        if metric.duration_ms > 5000:  # 5 second threshold
            bottleneck = {
                "type": "slow_operation",
                "component": component,
                "operation": metric.operation,
                "duration_ms": metric.duration_ms,
                "timestamp": metric.timestamp,
                "severity": "high" if metric.duration_ms > 10000 else "medium"
            }
            self.bottlenecks.append(bottleneck)
        
        if metric.cpu_usage > 1000:  # High CPU usage threshold
            bottleneck = {
                "type": "high_cpu",
                "component": component,
                "operation": metric.operation,
                "cpu_usage": metric.cpu_usage,
                "timestamp": metric.timestamp,
                "severity": "high" if metric.cpu_usage > 2000 else "medium"
            }
            self.bottlenecks.append(bottleneck)
    
    def get_bottlenecks(self, severity: str = None) -> List[Dict[str, Any]]:
        """Get identified bottlenecks, optionally filtered by severity."""
        with self._lock:
            if severity:
                return [b for b in self.bottlenecks if b.get("severity") == severity]
            return self.bottlenecks.copy()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._lock:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "components": {},
                "bottlenecks": self.bottlenecks,
                "summary": {
                    "total_operations": sum(len(ops) for ops in self.profiling_data.values()),
                    "total_bottlenecks": len(self.bottlenecks),
                    "high_severity_bottlenecks": len([b for b in self.bottlenecks if b.get("severity") == "high"])
                }
            }
            
            for component, metrics in self.profiling_data.items():
                if metrics:
                    durations = [m.duration_ms for m in metrics]
                    cpu_usages = [m.cpu_usage for m in metrics]
                    success_rate = sum(1 for m in metrics if m.success) / len(metrics)
                    
                    report["components"][component] = {
                        "operation_count": len(metrics),
                        "success_rate": success_rate,
                        "avg_duration_ms": statistics.mean(durations),
                        "max_duration_ms": max(durations),
                        "avg_cpu_usage": statistics.mean(cpu_usages),
                        "max_cpu_usage": max(cpu_usages)
                    }
            
            return report


class TelemetrySystem:
    """Main telemetry and observability system."""
    
    def __init__(self, service_name: str = "archangel-agents"):
        self.service_name = service_name
        self.time_warp = TimeWarpManager()
        self.tracer = DistributedTracer(service_name)
        self.metrics = MetricsCollector()
        self.profiler = PerformanceProfiler()
        self.events = deque(maxlen=10000)  # Keep last 10k events
        self._lock = Lock()
        
        # Initialize OpenTelemetry
        self._setup_opentelemetry()
    
    def _setup_opentelemetry(self):
        """Set up OpenTelemetry providers and exporters."""
        # Set up tracing
        resource = Resource.create({"service.name": self.service_name})
        trace.set_tracer_provider(TracerProvider(resource=resource))
        
        # Add Jaeger exporter for traces
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=14268,
        )
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Set up metrics with Prometheus
        prometheus_reader = PrometheusMetricReader()
        metrics.set_meter_provider(MeterProvider(
            resource=resource,
            metric_readers=[prometheus_reader]
        ))
        
        # Instrument common libraries
        LoggingInstrumentor().instrument()
        RequestsInstrumentor().instrument()
        AsyncioInstrumentor().instrument()
    
    def record_event(self, event_type: str, source: str, data: Dict[str, Any], 
                    trace_context: Optional[TraceContext] = None) -> str:
        """Record a telemetry event."""
        event_id = f"{source}_{int(time.time() * 1000000)}"
        
        event = TelemetryEvent(
            event_id=event_id,
            timestamp=self.time_warp.get_current_time(),
            event_type=event_type,
            source=source,
            data=data,
            trace_id=trace_context.trace_id if trace_context else None,
            span_id=trace_context.span_id if trace_context else None,
            parent_span_id=trace_context.parent_span_id if trace_context else None
        )
        
        with self._lock:
            self.events.append(event)
        
        return event_id
    
    @contextmanager
    def trace_operation(self, operation_name: str, component: str, **attributes):
        """Trace an operation with performance profiling."""
        with self.tracer.start_span(operation_name) as span:
            # Set span attributes
            span.set_attribute("component", component)
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            
            with self.profiler.profile_operation(operation_name, component):
                yield span
    
    def get_events(self, event_type: str = None, source: str = None, 
                  time_range: tuple = None) -> List[TelemetryEvent]:
        """Get events with optional filtering."""
        with self._lock:
            events = list(self.events)
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if source:
            events = [e for e in events if e.source == source]
        
        if time_range:
            start_time, end_time = time_range
            events = [e for e in events if start_time <= e.timestamp <= end_time]
        
        return events
    
    def export_telemetry_data(self, format_type: str = "json") -> str:
        """Export telemetry data in specified format."""
        data = {
            "service_name": self.service_name,
            "export_timestamp": datetime.utcnow().isoformat(),
            "events": [event.to_dict() for event in self.events],
            "performance_report": self.profiler.get_performance_report(),
            "bottlenecks": self.profiler.get_bottlenecks()
        }
        
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def start_forensic_replay(self, start_time: datetime, end_time: datetime, speed: float = 1.0):
        """Start forensic replay of events in a time range."""
        self.time_warp.start_replay(start_time, speed)
        
        # Filter events in the time range
        replay_events = self.get_events(time_range=(start_time, end_time))
        
        return {
            "replay_id": f"replay_{int(time.time())}",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "speed": speed,
            "event_count": len(replay_events),
            "events": [event.to_dict() for event in replay_events]
        }
    
    def stop_forensic_replay(self):
        """Stop forensic replay."""
        self.time_warp.stop_replay()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        recent_events = self.get_events(
            time_range=(datetime.utcnow() - timedelta(minutes=5), datetime.utcnow())
        )
        
        error_events = [e for e in recent_events if e.event_type == "error"]
        bottlenecks = self.profiler.get_bottlenecks("high")
        
        health_score = 100
        if error_events:
            health_score -= min(len(error_events) * 5, 50)
        if bottlenecks:
            health_score -= min(len(bottlenecks) * 10, 30)
        
        return {
            "health_score": max(health_score, 0),
            "status": "healthy" if health_score > 80 else "degraded" if health_score > 50 else "unhealthy",
            "recent_events": len(recent_events),
            "error_events": len(error_events),
            "high_severity_bottlenecks": len(bottlenecks),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global telemetry instance
_telemetry_system = None

def get_telemetry() -> TelemetrySystem:
    """Get the global telemetry system instance."""
    global _telemetry_system
    if _telemetry_system is None:
        _telemetry_system = TelemetrySystem()
    return _telemetry_system

def initialize_telemetry(service_name: str = "archangel-agents") -> TelemetrySystem:
    """Initialize the global telemetry system."""
    global _telemetry_system
    _telemetry_system = TelemetrySystem(service_name)
    return _telemetry_system