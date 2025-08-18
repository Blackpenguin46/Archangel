"""
Performance metrics collection and monitoring.
"""

import time
import threading
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import statistics

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class ResourceSnapshot:
    """System resource snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    process_count: int
    thread_count: int

class PerformanceMetrics:
    """Performance metrics collection and analysis."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        with self._lock:
            self.counters[name] += value
            self.metrics[name].append(MetricPoint(
                timestamp=datetime.now(),
                value=value,
                tags=tags or {}
            ))
            
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        with self._lock:
            self.gauges[name] = value
            self.metrics[name].append(MetricPoint(
                timestamp=datetime.now(),
                value=value,
                tags=tags or {}
            ))
            
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only recent values for histogram
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
                
            self.metrics[name].append(MetricPoint(
                timestamp=datetime.now(),
                value=value,
                tags=tags or {}
            ))
            
    def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric."""
        self.record_histogram(f"{name}_duration", duration, tags)
        
    def get_counter_value(self, name: str) -> int:
        """Get current counter value."""
        return self.counters.get(name, 0)
        
    def get_gauge_value(self, name: str) -> float:
        """Get current gauge value."""
        return self.gauges.get(name, 0.0)
        
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        values = self.histograms.get(name, [])
        if not values:
            return {}
            
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p90': self._percentile(values, 90),
            'p95': self._percentile(values, 95),
            'p99': self._percentile(values, 99)
        }
        
    def get_metric_history(self, name: str, 
                          time_window: Optional[timedelta] = None) -> List[MetricPoint]:
        """Get metric history within time window."""
        with self._lock:
            points = list(self.metrics.get(name, []))
            
            if time_window:
                cutoff_time = datetime.now() - time_window
                points = [p for p in points if p.timestamp >= cutoff_time]
                
            return points
            
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            summary = {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {
                    name: self.get_histogram_stats(name)
                    for name in self.histograms.keys()
                },
                'metric_count': sum(len(deque_obj) for deque_obj in self.metrics.values()),
                'timestamp': datetime.now().isoformat()
            }
            
        return summary
        
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            
        logger.info("Reset all performance metrics")
        
    def export_metrics(self, filepath: str, time_window: Optional[timedelta] = None) -> None:
        """Export metrics to JSON file."""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_window': time_window.total_seconds() if time_window else None,
            'metrics': {}
        }
        
        for metric_name in self.metrics.keys():
            history = self.get_metric_history(metric_name, time_window)
            export_data['metrics'][metric_name] = [
                {
                    'timestamp': point.timestamp.isoformat(),
                    'value': point.value,
                    'tags': point.tags
                }
                for point in history
            ]
            
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported metrics to {filepath}")
        
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
            
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

class ResourceMonitor:
    """System resource monitoring."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.is_monitoring = False
        self.resource_history: deque = deque(maxlen=3600)  # 1 hour at 1s intervals
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.metrics = PerformanceMetrics()
        
        # Initialize baseline measurements
        self._initial_disk_io = psutil.disk_io_counters()
        self._initial_network_io = psutil.net_io_counters()
        
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.is_monitoring:
            logger.warning("Resource monitoring already active")
            return
            
        self.is_monitoring = True
        self._stop_event.clear()
        
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info(f"Started resource monitoring with {self.collection_interval}s interval")
        
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        self._stop_event.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
            
        logger.info("Stopped resource monitoring")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.wait(self.collection_interval):
            try:
                snapshot = self._capture_resource_snapshot()
                self.resource_history.append(snapshot)
                
                # Record metrics
                self.metrics.record_gauge('cpu_percent', snapshot.cpu_percent)
                self.metrics.record_gauge('memory_percent', snapshot.memory_percent)
                self.metrics.record_gauge('memory_used_mb', snapshot.memory_used_mb)
                self.metrics.record_gauge('process_count', snapshot.process_count)
                self.metrics.record_gauge('thread_count', snapshot.thread_count)
                
                # Record I/O rates
                self.metrics.record_gauge('disk_io_read_mb_per_sec', snapshot.disk_io_read_mb)
                self.metrics.record_gauge('disk_io_write_mb_per_sec', snapshot.disk_io_write_mb)
                self.metrics.record_gauge('network_io_sent_mb_per_sec', snapshot.network_io_sent_mb)
                self.metrics.record_gauge('network_io_recv_mb_per_sec', snapshot.network_io_recv_mb)
                
            except Exception as e:
                logger.error(f"Error during resource monitoring: {e}")
                
    def _capture_resource_snapshot(self) -> ResourceSnapshot:
        """Capture current resource usage snapshot."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Process information
        process_count = len(psutil.pids())
        
        # Current process thread count
        try:
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
        except:
            thread_count = 0
            
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io and self._initial_disk_io:
            disk_read_mb = (disk_io.read_bytes - self._initial_disk_io.read_bytes) / 1024 / 1024
            disk_write_mb = (disk_io.write_bytes - self._initial_disk_io.write_bytes) / 1024 / 1024
        else:
            disk_read_mb = disk_write_mb = 0
            
        # Network I/O
        network_io = psutil.net_io_counters()
        if network_io and self._initial_network_io:
            network_sent_mb = (network_io.bytes_sent - self._initial_network_io.bytes_sent) / 1024 / 1024
            network_recv_mb = (network_io.bytes_recv - self._initial_network_io.bytes_recv) / 1024 / 1024
        else:
            network_sent_mb = network_recv_mb = 0
            
        return ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_io_sent_mb=network_sent_mb,
            network_io_recv_mb=network_recv_mb,
            process_count=process_count,
            thread_count=thread_count
        )
        
    def get_current_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        return self._capture_resource_snapshot()
        
    def get_resource_history(self, time_window: Optional[timedelta] = None) -> List[ResourceSnapshot]:
        """Get resource history within time window."""
        snapshots = list(self.resource_history)
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
            
        return snapshots
        
    def get_resource_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get resource usage summary."""
        snapshots = self.get_resource_history(time_window)
        
        if not snapshots:
            return {}
            
        # Calculate statistics
        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_percent for s in snapshots]
        memory_used_values = [s.memory_used_mb for s in snapshots]
        
        return {
            'time_range': {
                'start': snapshots[0].timestamp.isoformat(),
                'end': snapshots[-1].timestamp.isoformat(),
                'duration_seconds': (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds()
            },
            'cpu_usage': {
                'avg': statistics.mean(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'p95': self._percentile(cpu_values, 95)
            },
            'memory_usage': {
                'avg_percent': statistics.mean(memory_values),
                'max_percent': max(memory_values),
                'avg_used_mb': statistics.mean(memory_used_values),
                'max_used_mb': max(memory_used_values)
            },
            'process_info': {
                'avg_process_count': statistics.mean([s.process_count for s in snapshots]),
                'max_process_count': max([s.process_count for s in snapshots]),
                'avg_thread_count': statistics.mean([s.thread_count for s in snapshots]),
                'max_thread_count': max([s.thread_count for s in snapshots])
            },
            'io_summary': {
                'total_disk_read_mb': sum([s.disk_io_read_mb for s in snapshots]),
                'total_disk_write_mb': sum([s.disk_io_write_mb for s in snapshots]),
                'total_network_sent_mb': sum([s.network_io_sent_mb for s in snapshots]),
                'total_network_recv_mb': sum([s.network_io_recv_mb for s in snapshots])
            },
            'sample_count': len(snapshots)
        }
        
    def detect_resource_anomalies(self, time_window: timedelta = timedelta(minutes=5)) -> List[Dict[str, Any]]:
        """Detect resource usage anomalies."""
        snapshots = self.get_resource_history(time_window)
        
        if len(snapshots) < 10:  # Need sufficient data
            return []
            
        anomalies = []
        
        # CPU anomalies
        cpu_values = [s.cpu_percent for s in snapshots]
        cpu_mean = statistics.mean(cpu_values)
        cpu_stdev = statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
        
        for snapshot in snapshots:
            if cpu_stdev > 0 and abs(snapshot.cpu_percent - cpu_mean) > 2 * cpu_stdev:
                anomalies.append({
                    'type': 'cpu_anomaly',
                    'timestamp': snapshot.timestamp.isoformat(),
                    'value': snapshot.cpu_percent,
                    'expected_range': [cpu_mean - 2*cpu_stdev, cpu_mean + 2*cpu_stdev],
                    'severity': 'high' if snapshot.cpu_percent > 90 else 'medium'
                })
                
        # Memory anomalies
        memory_values = [s.memory_percent for s in snapshots]
        memory_mean = statistics.mean(memory_values)
        memory_stdev = statistics.stdev(memory_values) if len(memory_values) > 1 else 0
        
        for snapshot in snapshots:
            if memory_stdev > 0 and abs(snapshot.memory_percent - memory_mean) > 2 * memory_stdev:
                anomalies.append({
                    'type': 'memory_anomaly',
                    'timestamp': snapshot.timestamp.isoformat(),
                    'value': snapshot.memory_percent,
                    'expected_range': [memory_mean - 2*memory_stdev, memory_mean + 2*memory_stdev],
                    'severity': 'high' if snapshot.memory_percent > 90 else 'medium'
                })
                
        return anomalies
        
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
            
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

# Context managers for automatic metric recording
class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, metrics: PerformanceMetrics, metric_name: str, tags: Optional[Dict[str, str]] = None):
        self.metrics = metrics
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.record_timing(self.metric_name, duration, self.tags)

# Decorators for automatic metric collection
def time_function(metrics: PerformanceMetrics, metric_name: Optional[str] = None):
    """Decorator for timing function execution."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}"
            with TimingContext(metrics, name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def count_calls(metrics: PerformanceMetrics, metric_name: Optional[str] = None):
    """Decorator for counting function calls."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}_calls"
            metrics.record_counter(name)
            return func(*args, **kwargs)
        return wrapper
    return decorator