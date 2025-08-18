"""
Monitoring and metrics collection module for Archangel agents.
"""

# Import existing components with error handling
try:
    from .metrics_collector import MetricsCollector, AgentMetrics
    from .health_monitor import HealthMonitor
    from .recovery_system import RecoverySystem
    _legacy_available = True
except ImportError:
    _legacy_available = False

# Import new telemetry and observability components
try:
    from .telemetry import (
        TelemetrySystem, TelemetryEvent, PerformanceMetrics, 
        TimeWarpManager, DistributedTracer, MetricsCollector as TelemetryMetricsCollector,
        PerformanceProfiler, get_telemetry, initialize_telemetry
    )
    from .observability import (
        ObservabilitySystem, AlertManager, LogCorrelator, TopologyDiscovery,
        DashboardManager, Alert, AlertSeverity, AlertStatus, LogEntry,
        get_observability, initialize_observability
    )
    _telemetry_available = True
except ImportError as e:
    _telemetry_available = False
    print(f"Warning: Telemetry components not available: {e}")

# Build __all__ list based on what's available
__all__ = []

if _legacy_available:
    __all__.extend([
        'MetricsCollector',
        'AgentMetrics', 
        'HealthMonitor',
        'RecoverySystem'
    ])

if _telemetry_available:
    __all__.extend([
        'TelemetrySystem',
        'TelemetryEvent',
        'PerformanceMetrics',
        'TimeWarpManager',
        'DistributedTracer',
        'TelemetryMetricsCollector',
        'PerformanceProfiler',
        'get_telemetry',
        'initialize_telemetry',
        'ObservabilitySystem',
        'AlertManager',
        'LogCorrelator',
        'TopologyDiscovery',
        'DashboardManager',
        'Alert',
        'AlertSeverity',
        'AlertStatus',
        'LogEntry',
        'get_observability',
        'initialize_observability'
    ])