"""
Advanced observability features for the Archangel system.

This module provides:
- Real-time monitoring dashboards
- Alert management and notification system
- Log correlation and analysis
- System topology discovery
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
import threading
from enum import Enum
import statistics

from .telemetry import get_telemetry, TelemetryEvent, PerformanceMetrics


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Represents a system alert."""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    source: str
    timestamp: datetime
    tags: Dict[str, str]
    metadata: Dict[str, Any]
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        result = asdict(self)
        result['severity'] = self.severity.value
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            result['resolved_at'] = self.resolved_at.isoformat()
        if self.acknowledged_at:
            result['acknowledged_at'] = self.acknowledged_at.isoformat()
        return result


@dataclass
class LogEntry:
    """Represents a structured log entry."""
    timestamp: datetime
    level: str
    source: str
    message: str
    context: Dict[str, Any]
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class SystemTopology:
    """Represents system topology information."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    discovered_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['discovered_at'] = self.discovered_at.isoformat()
        return result


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.alerts = {}
        self.alert_rules = []
        self.notification_handlers = []
        self._lock = threading.Lock()
        self.telemetry = get_telemetry()
    
    def add_alert_rule(self, rule: Callable[[Dict[str, Any]], Optional[Alert]]):
        """Add an alert rule function."""
        self.alert_rules.append(rule)
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler."""
        self.notification_handlers.append(handler)
    
    def create_alert(self, title: str, description: str, severity: AlertSeverity,
                    source: str, tags: Dict[str, str] = None, 
                    metadata: Dict[str, Any] = None) -> str:
        """Create a new alert."""
        alert_id = f"alert_{int(time.time() * 1000000)}"
        
        alert = Alert(
            alert_id=alert_id,
            title=title,
            description=description,
            severity=severity,
            status=AlertStatus.ACTIVE,
            source=source,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self._lock:
            self.alerts[alert_id] = alert
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Notification handler failed: {e}")
        
        # Record telemetry event
        self.telemetry.record_event(
            "alert_created",
            source,
            {
                "alert_id": alert_id,
                "title": title,
                "severity": severity.value,
                "tags": tags
            }
        )
        
        return alert_id
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by
                
                self.telemetry.record_event(
                    "alert_acknowledged",
                    alert.source,
                    {
                        "alert_id": alert_id,
                        "acknowledged_by": acknowledged_by
                    }
                )
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                
                self.telemetry.record_event(
                    "alert_resolved",
                    alert.source,
                    {"alert_id": alert_id}
                )
                return True
        return False
    
    def get_alerts(self, status: AlertStatus = None, severity: AlertSeverity = None,
                  source: str = None) -> List[Alert]:
        """Get alerts with optional filtering."""
        with self._lock:
            alerts = list(self.alerts.values())
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def evaluate_alert_rules(self, data: Dict[str, Any]):
        """Evaluate alert rules against provided data."""
        for rule in self.alert_rules:
            try:
                alert = rule(data)
                if alert:
                    with self._lock:
                        self.alerts[alert.alert_id] = alert
                    
                    # Send notifications
                    for handler in self.notification_handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            logging.error(f"Notification handler failed: {e}")
            except Exception as e:
                logging.error(f"Alert rule evaluation failed: {e}")


class LogCorrelator:
    """Correlates logs and events for analysis."""
    
    def __init__(self):
        self.log_buffer = deque(maxlen=50000)  # Keep last 50k logs
        self.correlation_rules = []
        self._lock = threading.Lock()
        self.telemetry = get_telemetry()
    
    def add_log_entry(self, entry: LogEntry):
        """Add a log entry to the buffer."""
        with self._lock:
            self.log_buffer.append(entry)
    
    def add_correlation_rule(self, rule: Callable[[List[LogEntry]], List[Dict[str, Any]]]):
        """Add a log correlation rule."""
        self.correlation_rules.append(rule)
    
    def correlate_logs(self, time_window: timedelta = timedelta(minutes=5)) -> List[Dict[str, Any]]:
        """Correlate logs within a time window."""
        cutoff_time = datetime.utcnow() - time_window
        
        with self._lock:
            recent_logs = [log for log in self.log_buffer if log.timestamp >= cutoff_time]
        
        correlations = []
        for rule in self.correlation_rules:
            try:
                rule_correlations = rule(recent_logs)
                correlations.extend(rule_correlations)
            except Exception as e:
                logging.error(f"Log correlation rule failed: {e}")
        
        return correlations
    
    def find_related_logs(self, trace_id: str) -> List[LogEntry]:
        """Find all logs related to a trace ID."""
        with self._lock:
            return [log for log in self.log_buffer if log.trace_id == trace_id]
    
    def search_logs(self, query: str, time_range: tuple = None) -> List[LogEntry]:
        """Search logs by message content."""
        with self._lock:
            logs = list(self.log_buffer)
        
        if time_range:
            start_time, end_time = time_range
            logs = [log for log in logs if start_time <= log.timestamp <= end_time]
        
        # Simple text search
        matching_logs = [log for log in logs if query.lower() in log.message.lower()]
        
        return sorted(matching_logs, key=lambda l: l.timestamp, reverse=True)


class TopologyDiscovery:
    """Discovers and maintains system topology."""
    
    def __init__(self):
        self.topology = None
        self.discovery_handlers = []
        self._lock = threading.Lock()
        self.telemetry = get_telemetry()
    
    def add_discovery_handler(self, handler: Callable[[], Dict[str, Any]]):
        """Add a topology discovery handler."""
        self.discovery_handlers.append(handler)
    
    def discover_topology(self) -> SystemTopology:
        """Discover current system topology."""
        nodes = []
        edges = []
        metadata = {}
        
        for handler in self.discovery_handlers:
            try:
                result = handler()
                nodes.extend(result.get('nodes', []))
                edges.extend(result.get('edges', []))
                metadata.update(result.get('metadata', {}))
            except Exception as e:
                logging.error(f"Topology discovery handler failed: {e}")
        
        topology = SystemTopology(
            nodes=nodes,
            edges=edges,
            metadata=metadata,
            discovered_at=datetime.utcnow()
        )
        
        with self._lock:
            self.topology = topology
        
        self.telemetry.record_event(
            "topology_discovered",
            "topology_discovery",
            {
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        )
        
        return topology
    
    def get_topology(self) -> Optional[SystemTopology]:
        """Get current topology."""
        with self._lock:
            return self.topology


class DashboardManager:
    """Manages real-time monitoring dashboards."""
    
    def __init__(self):
        self.dashboard_data = {}
        self.widgets = {}
        self._lock = threading.Lock()
        self.telemetry = get_telemetry()
        self.alert_manager = AlertManager()
        self.log_correlator = LogCorrelator()
        self.topology_discovery = TopologyDiscovery()
    
    def register_widget(self, widget_id: str, data_source: Callable[[], Dict[str, Any]]):
        """Register a dashboard widget with its data source."""
        self.widgets[widget_id] = data_source
    
    def update_dashboard_data(self):
        """Update all dashboard data."""
        updated_data = {}
        
        for widget_id, data_source in self.widgets.items():
            try:
                updated_data[widget_id] = data_source()
            except Exception as e:
                logging.error(f"Widget {widget_id} data source failed: {e}")
                updated_data[widget_id] = {"error": str(e)}
        
        with self._lock:
            self.dashboard_data = updated_data
        
        return updated_data
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        with self._lock:
            return self.dashboard_data.copy()
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system overview data."""
        health = self.telemetry.get_system_health()
        active_alerts = self.alert_manager.get_alerts(status=AlertStatus.ACTIVE)
        performance_report = self.telemetry.profiler.get_performance_report()
        
        return {
            "health": health,
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "performance": performance_report.get("summary", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent-specific metrics."""
        recent_events = self.telemetry.get_events(
            time_range=(datetime.utcnow() - timedelta(minutes=10), datetime.utcnow())
        )
        
        agent_events = defaultdict(int)
        agent_errors = defaultdict(int)
        
        for event in recent_events:
            if event.source.startswith("agent_"):
                agent_events[event.source] += 1
                if event.event_type == "error":
                    agent_errors[event.source] += 1
        
        return {
            "agent_activity": dict(agent_events),
            "agent_errors": dict(agent_errors),
            "total_agents": len(agent_events),
            "active_agents": len([agent for agent, count in agent_events.items() if count > 0])
        }
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get network topology visualization data."""
        topology = self.topology_discovery.get_topology()
        if topology:
            return topology.to_dict()
        return {"nodes": [], "edges": [], "metadata": {}}


class ObservabilitySystem:
    """Main observability system coordinating all components."""
    
    def __init__(self):
        self.telemetry = get_telemetry()
        self.alert_manager = AlertManager()
        self.log_correlator = LogCorrelator()
        self.topology_discovery = TopologyDiscovery()
        self.dashboard_manager = DashboardManager()
        self._setup_default_rules()
        self._setup_default_widgets()
    
    def _setup_default_rules(self):
        """Set up default alert and correlation rules."""
        # Default alert rules
        def high_error_rate_rule(data: Dict[str, Any]) -> Optional[Alert]:
            """Alert on high error rates."""
            if data.get("error_rate", 0) > 0.1:  # 10% error rate
                return Alert(
                    alert_id=f"high_error_rate_{int(time.time())}",
                    title="High Error Rate Detected",
                    description=f"Error rate is {data['error_rate']:.2%}",
                    severity=AlertSeverity.HIGH,
                    status=AlertStatus.ACTIVE,
                    source="error_monitor",
                    timestamp=datetime.utcnow(),
                    tags={"type": "error_rate"},
                    metadata=data
                )
            return None
        
        def performance_degradation_rule(data: Dict[str, Any]) -> Optional[Alert]:
            """Alert on performance degradation."""
            if data.get("avg_response_time", 0) > 5000:  # 5 second threshold
                return Alert(
                    alert_id=f"perf_degradation_{int(time.time())}",
                    title="Performance Degradation",
                    description=f"Average response time is {data['avg_response_time']:.0f}ms",
                    severity=AlertSeverity.MEDIUM,
                    status=AlertStatus.ACTIVE,
                    source="performance_monitor",
                    timestamp=datetime.utcnow(),
                    tags={"type": "performance"},
                    metadata=data
                )
            return None
        
        self.alert_manager.add_alert_rule(high_error_rate_rule)
        self.alert_manager.add_alert_rule(performance_degradation_rule)
        
        # Default correlation rules
        def error_burst_correlation(logs: List[LogEntry]) -> List[Dict[str, Any]]:
            """Detect error bursts."""
            error_logs = [log for log in logs if log.level == "ERROR"]
            if len(error_logs) > 10:  # More than 10 errors in time window
                return [{
                    "type": "error_burst",
                    "count": len(error_logs),
                    "sources": list(set(log.source for log in error_logs)),
                    "timestamp": datetime.utcnow().isoformat()
                }]
            return []
        
        self.log_correlator.add_correlation_rule(error_burst_correlation)
    
    def _setup_default_widgets(self):
        """Set up default dashboard widgets."""
        self.dashboard_manager.register_widget("system_overview", 
                                              self.dashboard_manager.get_system_overview)
        self.dashboard_manager.register_widget("agent_metrics", 
                                              self.dashboard_manager.get_agent_metrics)
        self.dashboard_manager.register_widget("network_topology", 
                                              self.dashboard_manager.get_network_topology)
    
    def start_monitoring(self, update_interval: int = 30):
        """Start continuous monitoring."""
        def monitoring_loop():
            while True:
                try:
                    # Update dashboard data
                    self.dashboard_manager.update_dashboard_data()
                    
                    # Evaluate alert rules
                    system_data = self.dashboard_manager.get_system_overview()
                    self.alert_manager.evaluate_alert_rules(system_data)
                    
                    # Correlate logs
                    correlations = self.log_correlator.correlate_logs()
                    if correlations:
                        self.telemetry.record_event(
                            "log_correlations_found",
                            "log_correlator",
                            {"correlations": correlations}
                        )
                    
                    # Discover topology periodically
                    if int(time.time()) % 300 == 0:  # Every 5 minutes
                        self.topology_discovery.discover_topology()
                    
                except Exception as e:
                    logging.error(f"Monitoring loop error: {e}")
                
                time.sleep(update_interval)
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
    
    def get_observability_report(self) -> Dict[str, Any]:
        """Get comprehensive observability report."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": self.telemetry.get_system_health(),
            "alerts": {
                "active": len(self.alert_manager.get_alerts(status=AlertStatus.ACTIVE)),
                "critical": len(self.alert_manager.get_alerts(severity=AlertSeverity.CRITICAL)),
                "recent": [alert.to_dict() for alert in self.alert_manager.get_alerts()[:10]]
            },
            "performance": self.telemetry.profiler.get_performance_report(),
            "topology": self.topology_discovery.get_topology().to_dict() if self.topology_discovery.get_topology() else None,
            "dashboard_data": self.dashboard_manager.get_dashboard_data()
        }


# Global observability instance
_observability_system = None

def get_observability() -> ObservabilitySystem:
    """Get the global observability system instance."""
    global _observability_system
    if _observability_system is None:
        _observability_system = ObservabilitySystem()
    return _observability_system

def initialize_observability() -> ObservabilitySystem:
    """Initialize the global observability system."""
    global _observability_system
    _observability_system = ObservabilitySystem()
    return _observability_system