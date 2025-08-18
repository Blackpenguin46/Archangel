#!/usr/bin/env python3
"""
Network Service Dependencies and Failure Simulation Module

This module simulates realistic network service dependencies and failure modes
commonly found in enterprise environments.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
import threading
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    HAS_GRAPH_LIBS = True
except ImportError:
    HAS_GRAPH_LIBS = False
    # Mock classes for when libraries aren't available
    class MockGraph:
        def __init__(self):
            self.nodes_data = {}
            self.edges_data = []
        def add_node(self, node, **kwargs): 
            self.nodes_data[node] = kwargs
        def add_edge(self, source, target): 
            self.edges_data.append((source, target))
        def nodes(self): 
            return list(self.nodes_data.keys())
        def predecessors(self, node): 
            return [s for s, t in self.edges_data if t == node]
        def successors(self, node): 
            return [t for s, t in self.edges_data if s == node]
        def number_of_edges(self): 
            return len(self.edges_data)
        def copy(self):
            new_graph = MockGraph()
            new_graph.nodes_data = self.nodes_data.copy()
            new_graph.edges_data = self.edges_data.copy()
            return new_graph
        def remove_node(self, node):
            if node in self.nodes_data:
                del self.nodes_data[node]
            self.edges_data = [(s, t) for s, t in self.edges_data if s != node and t != node]
    
    class nx:
        DiGraph = MockGraph
        @staticmethod
        def density(graph): return 0.5
        @staticmethod
        def strongly_connected_components(graph): return [set(graph.nodes())]
        @staticmethod
        def weakly_connected_components(graph): return [set(graph.nodes())]
        @staticmethod
        def dag_longest_path(graph): return []
        @staticmethod
        def is_directed_acyclic_graph(graph): return True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Types of network services"""
    DNS = "dns"
    DHCP = "dhcp"
    ACTIVE_DIRECTORY = "active_directory"
    DATABASE = "database"
    WEB_SERVER = "web_server"
    EMAIL_SERVER = "email_server"
    FILE_SERVER = "file_server"
    PRINT_SERVER = "print_server"
    BACKUP_SERVER = "backup_server"
    MONITORING = "monitoring"
    FIREWALL = "firewall"
    LOAD_BALANCER = "load_balancer"
    VPN_SERVER = "vpn_server"
    PROXY_SERVER = "proxy_server"
    NTP_SERVER = "ntp_server"
    SYSLOG_SERVER = "syslog_server"
    CERTIFICATE_AUTHORITY = "certificate_authority"
    RADIUS_SERVER = "radius_server"
    NETWORK_SWITCH = "network_switch"
    NETWORK_ROUTER = "network_router"


class FailureType(Enum):
    """Types of service failures"""
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_CRASH = "software_crash"
    NETWORK_CONNECTIVITY = "network_connectivity"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION_ERROR = "configuration_error"
    SECURITY_INCIDENT = "security_incident"
    POWER_OUTAGE = "power_outage"
    MAINTENANCE = "maintenance"
    DEPENDENCY_FAILURE = "dependency_failure"
    OVERLOAD = "overload"


class ServiceStatus(Enum):
    """Service status states"""
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class ServiceFailure:
    """Represents a service failure event"""
    failure_id: str
    service_id: str
    failure_type: FailureType
    start_time: datetime
    end_time: Optional[datetime] = None
    description: str = ""
    impact_level: str = "medium"  # low, medium, high, critical
    root_cause: str = ""
    resolution_steps: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get failure duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if failure is still active"""
        return self.end_time is None


@dataclass
class NetworkService:
    """Represents a network service with dependencies"""
    service_id: str
    service_type: ServiceType
    name: str
    hostname: str
    ip_address: str
    port: int
    status: ServiceStatus = ServiceStatus.RUNNING
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    health_check_interval: int = 30  # seconds
    last_health_check: datetime = field(default_factory=datetime.now)
    uptime_percentage: float = 99.9
    response_time_ms: float = 50.0
    cpu_usage: float = 10.0
    memory_usage: float = 20.0
    disk_usage: float = 30.0
    network_usage: float = 5.0
    failure_probability: float = 0.001  # 0.1% per check
    recovery_time_minutes: int = 5
    criticality_level: str = "medium"  # low, medium, high, critical
    business_impact: str = "medium"
    sla_target: float = 99.5  # SLA uptime target percentage
    maintenance_window: Optional[str] = None
    backup_services: List[str] = field(default_factory=list)
    monitoring_enabled: bool = True
    alerting_enabled: bool = True
    
    def add_dependency(self, service_id: str):
        """Add a service dependency"""
        self.dependencies.add(service_id)
    
    def add_dependent(self, service_id: str):
        """Add a dependent service"""
        self.dependents.add(service_id)
    
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return self.status == ServiceStatus.RUNNING
    
    def calculate_availability(self, time_period: timedelta) -> float:
        """Calculate service availability over time period"""
        # This would integrate with actual monitoring data
        # For simulation, return uptime percentage with some variance
        variance = random.uniform(-0.5, 0.5)
        return max(0, min(100, self.uptime_percentage + variance))


class ServiceDependencyGraph:
    """Manages service dependencies using a directed graph"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.services: Dict[str, NetworkService] = {}
    
    def add_service(self, service: NetworkService):
        """Add a service to the dependency graph"""
        self.services[service.service_id] = service
        self.graph.add_node(service.service_id, service=service)
        logger.info(f"Added service: {service.name} ({service.service_id})")
    
    def add_dependency(self, service_id: str, depends_on: str):
        """Add a dependency relationship"""
        if service_id in self.services and depends_on in self.services:
            self.services[service_id].add_dependency(depends_on)
            self.services[depends_on].add_dependent(service_id)
            self.graph.add_edge(depends_on, service_id)
            logger.debug(f"Added dependency: {service_id} depends on {depends_on}")
    
    def get_dependencies(self, service_id: str) -> List[str]:
        """Get direct dependencies of a service"""
        if service_id in self.graph:
            return list(self.graph.predecessors(service_id))
        return []
    
    def get_dependents(self, service_id: str) -> List[str]:
        """Get direct dependents of a service"""
        if service_id in self.graph:
            return list(self.graph.successors(service_id))
        return []
    
    def get_all_dependencies(self, service_id: str) -> Set[str]:
        """Get all dependencies (recursive) of a service"""
        if service_id not in self.graph:
            return set()
        
        all_deps = set()
        
        def _collect_deps(sid):
            for dep in self.graph.predecessors(sid):
                if dep not in all_deps:
                    all_deps.add(dep)
                    _collect_deps(dep)
        
        _collect_deps(service_id)
        return all_deps
    
    def get_all_dependents(self, service_id: str) -> Set[str]:
        """Get all dependents (recursive) of a service"""
        if service_id not in self.graph:
            return set()
        
        all_deps = set()
        
        def _collect_deps(sid):
            for dep in self.graph.successors(sid):
                if dep not in all_deps:
                    all_deps.add(dep)
                    _collect_deps(dep)
        
        _collect_deps(service_id)
        return all_deps
    
    def find_critical_path(self, start_service: str, end_service: str) -> List[str]:
        """Find critical path between two services"""
        try:
            return nx.shortest_path(self.graph, start_service, end_service)
        except nx.NetworkXNoPath:
            return []
    
    def find_single_points_of_failure(self) -> List[str]:
        """Find services that are single points of failure"""
        spofs = []
        
        for service_id in self.graph.nodes():
            # A service is a SPOF if removing it disconnects the graph significantly
            temp_graph = self.graph.copy()
            temp_graph.remove_node(service_id)
            
            # Check if removing this node creates isolated components
            components = list(nx.weakly_connected_components(temp_graph))
            if len(components) > 1:
                # Check if any component has critical services
                for component in components:
                    critical_services = [s for s in component 
                                       if self.services[s].criticality_level in ["high", "critical"]]
                    if critical_services:
                        spofs.append(service_id)
                        break
        
        return spofs
    
    def simulate_cascade_failure(self, failed_service: str) -> List[str]:
        """Simulate cascade failure from a failed service"""
        affected_services = [failed_service]
        
        # Get all services that depend on the failed service
        all_dependents = self.get_all_dependents(failed_service)
        
        for dependent in all_dependents:
            service = self.services[dependent]
            
            # Check if service can continue without this dependency
            if self._can_service_continue_without_dependency(dependent, failed_service):
                continue
            
            # Service fails due to dependency failure
            affected_services.append(dependent)
            logger.warning(f"Cascade failure: {dependent} failed due to {failed_service}")
        
        return affected_services
    
    def _can_service_continue_without_dependency(self, service_id: str, failed_dependency: str) -> bool:
        """Check if a service can continue operating without a specific dependency"""
        service = self.services[service_id]
        
        # Check if there are backup services available
        if service.backup_services:
            return True
        
        # Check criticality of the dependency
        failed_service = self.services[failed_dependency]
        if failed_service.criticality_level in ["high", "critical"]:
            return False
        
        # Some services can operate in degraded mode
        if service.service_type in [ServiceType.WEB_SERVER, ServiceType.EMAIL_SERVER]:
            return random.choice([True, False])  # 50% chance of degraded operation
        
        return False
    
    def get_dependency_metrics(self) -> Dict[str, Any]:
        """Get dependency graph metrics"""
        return {
            "total_services": len(self.services),
            "total_dependencies": self.graph.number_of_edges(),
            "average_dependencies_per_service": self.graph.number_of_edges() / len(self.services) if self.services else 0,
            "single_points_of_failure": len(self.find_single_points_of_failure()),
            "strongly_connected_components": len(list(nx.strongly_connected_components(self.graph))),
            "weakly_connected_components": len(list(nx.weakly_connected_components(self.graph))),
            "graph_density": nx.density(self.graph),
            "longest_dependency_chain": len(nx.dag_longest_path(self.graph)) if nx.is_directed_acyclic_graph(self.graph) else 0
        }
    
    def visualize_dependencies(self, filename: str = "service_dependencies.png"):
        """Visualize the service dependency graph"""
        if not HAS_GRAPH_LIBS:
            logger.warning("Graph visualization libraries not available. Skipping visualization.")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Color nodes by service type
        node_colors = []
        for service_id in self.graph.nodes():
            service = self.services[service_id]
            if service.criticality_level == "critical":
                node_colors.append('red')
            elif service.criticality_level == "high":
                node_colors.append('orange')
            elif service.criticality_level == "medium":
                node_colors.append('yellow')
            else:
                node_colors.append('lightblue')
        
        # Draw the graph
        nx.draw(self.graph, pos, 
                node_color=node_colors,
                node_size=1000,
                with_labels=True,
                labels={sid: self.services[sid].name[:10] for sid in self.graph.nodes()},
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                alpha=0.7)
        
        plt.title("Service Dependency Graph")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dependency graph visualization saved to {filename}")


class FailureSimulator:
    """Simulates various types of service failures"""
    
    def __init__(self, dependency_graph: ServiceDependencyGraph):
        self.dependency_graph = dependency_graph
        self.active_failures: Dict[str, ServiceFailure] = {}
        self.failure_history: List[ServiceFailure] = []
    
    def simulate_random_failure(self) -> Optional[ServiceFailure]:
        """Simulate a random service failure"""
        services = list(self.dependency_graph.services.values())
        if not services:
            return None
        
        # Select a service to fail
        service = random.choice(services)
        
        # Check if service should fail based on its failure probability
        if random.random() > service.failure_probability:
            return None
        
        # Don't fail a service that's already failed
        if service.service_id in self.active_failures:
            return None
        
        # Create failure
        failure_type = random.choice(list(FailureType))
        failure = self._create_failure(service, failure_type)
        
        # Apply failure
        self._apply_failure(failure)
        
        return failure
    
    def simulate_targeted_failure(self, service_id: str, failure_type: FailureType) -> Optional[ServiceFailure]:
        """Simulate a targeted service failure"""
        if service_id not in self.dependency_graph.services:
            return None
        
        if service_id in self.active_failures:
            return None  # Service already failed
        
        service = self.dependency_graph.services[service_id]
        failure = self._create_failure(service, failure_type)
        
        # Apply failure
        self._apply_failure(failure)
        
        return failure
    
    def _create_failure(self, service: NetworkService, failure_type: FailureType) -> ServiceFailure:
        """Create a failure object"""
        failure_descriptions = {
            FailureType.HARDWARE_FAILURE: f"Hardware failure on {service.hostname}",
            FailureType.SOFTWARE_CRASH: f"Software crash in {service.name}",
            FailureType.NETWORK_CONNECTIVITY: f"Network connectivity lost to {service.hostname}",
            FailureType.RESOURCE_EXHAUSTION: f"Resource exhaustion on {service.name}",
            FailureType.CONFIGURATION_ERROR: f"Configuration error in {service.name}",
            FailureType.SECURITY_INCIDENT: f"Security incident affecting {service.name}",
            FailureType.POWER_OUTAGE: f"Power outage affecting {service.hostname}",
            FailureType.MAINTENANCE: f"Scheduled maintenance on {service.name}",
            FailureType.OVERLOAD: f"Service overload on {service.name}"
        }
        
        impact_levels = {
            "critical": ["critical", "high"],
            "high": ["high", "medium"],
            "medium": ["medium", "low"],
            "low": ["low"]
        }
        
        impact_level = random.choice(impact_levels.get(service.criticality_level, ["medium"]))
        
        failure = ServiceFailure(
            failure_id=f"fail_{int(time.time())}_{random.randint(1000, 9999)}",
            service_id=service.service_id,
            failure_type=failure_type,
            start_time=datetime.now(),
            description=failure_descriptions.get(failure_type, f"Unknown failure in {service.name}"),
            impact_level=impact_level,
            root_cause=self._generate_root_cause(failure_type),
            resolution_steps=self._generate_resolution_steps(failure_type)
        )
        
        return failure
    
    def _generate_root_cause(self, failure_type: FailureType) -> str:
        """Generate a realistic root cause for the failure"""
        root_causes = {
            FailureType.HARDWARE_FAILURE: [
                "Disk drive failure", "Memory module failure", "CPU overheating",
                "Power supply failure", "Network card failure"
            ],
            FailureType.SOFTWARE_CRASH: [
                "Memory leak", "Null pointer exception", "Stack overflow",
                "Deadlock condition", "Unhandled exception"
            ],
            FailureType.NETWORK_CONNECTIVITY: [
                "Switch port failure", "Cable disconnection", "Router misconfiguration",
                "VLAN configuration error", "DNS resolution failure"
            ],
            FailureType.RESOURCE_EXHAUSTION: [
                "Out of memory", "Disk space full", "CPU at 100%",
                "Network bandwidth saturated", "File handle exhaustion"
            ],
            FailureType.CONFIGURATION_ERROR: [
                "Incorrect service configuration", "Wrong firewall rules",
                "Database connection string error", "SSL certificate expired"
            ]
        }
        
        causes = root_causes.get(failure_type, ["Unknown root cause"])
        return random.choice(causes)
    
    def _generate_resolution_steps(self, failure_type: FailureType) -> List[str]:
        """Generate resolution steps for the failure"""
        resolution_steps = {
            FailureType.HARDWARE_FAILURE: [
                "Replace failed hardware component",
                "Verify system functionality",
                "Update hardware inventory"
            ],
            FailureType.SOFTWARE_CRASH: [
                "Restart the service",
                "Check application logs",
                "Apply software patches if available",
                "Monitor for recurring issues"
            ],
            FailureType.NETWORK_CONNECTIVITY: [
                "Check physical network connections",
                "Verify network configuration",
                "Test connectivity to dependent services",
                "Update network documentation"
            ],
            FailureType.RESOURCE_EXHAUSTION: [
                "Free up system resources",
                "Restart affected services",
                "Implement resource monitoring",
                "Plan capacity upgrades"
            ],
            FailureType.CONFIGURATION_ERROR: [
                "Review and correct configuration",
                "Restart affected services",
                "Test service functionality",
                "Document configuration changes"
            ]
        }
        
        return resolution_steps.get(failure_type, ["Investigate and resolve issue"])
    
    def _apply_failure(self, failure: ServiceFailure):
        """Apply the failure to the service and handle cascades"""
        service = self.dependency_graph.services[failure.service_id]
        
        # Change service status
        if failure.failure_type == FailureType.MAINTENANCE:
            service.status = ServiceStatus.MAINTENANCE
        else:
            service.status = ServiceStatus.FAILED
        
        # Add to active failures
        self.active_failures[failure.service_id] = failure
        
        # Simulate cascade failures
        affected_services = self.dependency_graph.simulate_cascade_failure(failure.service_id)
        failure.affected_services = affected_services
        
        # Apply cascade failures
        for affected_id in affected_services[1:]:  # Skip the original failed service
            if affected_id not in self.active_failures:
                affected_service = self.dependency_graph.services[affected_id]
                affected_service.status = ServiceStatus.DEGRADED
                
                # Create cascade failure record
                cascade_failure = ServiceFailure(
                    failure_id=f"cascade_{failure.failure_id}_{affected_id}",
                    service_id=affected_id,
                    failure_type=FailureType.DEPENDENCY_FAILURE,
                    start_time=datetime.now(),
                    description=f"Service degraded due to dependency failure: {failure.service_id}",
                    impact_level="medium"
                )
                self.active_failures[affected_id] = cascade_failure
        
        logger.warning(f"Service failure: {failure.description} (affects {len(affected_services)} services)")
    
    def resolve_failure(self, failure_id: str) -> bool:
        """Resolve a service failure"""
        failure = None
        
        # Find the failure
        for f in self.active_failures.values():
            if f.failure_id == failure_id:
                failure = f
                break
        
        if not failure:
            return False
        
        # Resolve the failure
        failure.end_time = datetime.now()
        service = self.dependency_graph.services[failure.service_id]
        service.status = ServiceStatus.RUNNING
        
        # Remove from active failures
        del self.active_failures[failure.service_id]
        
        # Add to history
        self.failure_history.append(failure)
        
        # Resolve cascade failures
        cascade_failures = [f for f in self.active_failures.values() 
                          if f.failure_type == FailureType.DEPENDENCY_FAILURE and 
                          failure.service_id in f.description]
        
        for cascade_failure in cascade_failures:
            cascade_service = self.dependency_graph.services[cascade_failure.service_id]
            cascade_service.status = ServiceStatus.RUNNING
            cascade_failure.end_time = datetime.now()
            del self.active_failures[cascade_failure.service_id]
            self.failure_history.append(cascade_failure)
        
        logger.info(f"Resolved failure: {failure.description}")
        return True
    
    def auto_resolve_failures(self):
        """Automatically resolve failures based on recovery time"""
        current_time = datetime.now()
        
        for failure in list(self.active_failures.values()):
            service = self.dependency_graph.services[failure.service_id]
            recovery_time = timedelta(minutes=service.recovery_time_minutes)
            
            if current_time - failure.start_time >= recovery_time:
                # Add some randomness to recovery
                if random.random() < 0.8:  # 80% chance of auto-recovery
                    self.resolve_failure(failure.failure_id)
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure statistics"""
        total_failures = len(self.failure_history) + len(self.active_failures)
        active_failures = len(self.active_failures)
        
        # Count by failure type
        failure_types = {}
        for failure in self.failure_history + list(self.active_failures.values()):
            ft = failure.failure_type.value
            failure_types[ft] = failure_types.get(ft, 0) + 1
        
        # Count by impact level
        impact_levels = {}
        for failure in self.failure_history + list(self.active_failures.values()):
            il = failure.impact_level
            impact_levels[il] = impact_levels.get(il, 0) + 1
        
        # Calculate MTTR (Mean Time To Recovery)
        resolved_failures = [f for f in self.failure_history if f.duration]
        if resolved_failures:
            total_duration = sum([f.duration.total_seconds() for f in resolved_failures])
            mttr_seconds = total_duration / len(resolved_failures)
            mttr_minutes = mttr_seconds / 60
        else:
            mttr_minutes = 0
        
        return {
            "total_failures": total_failures,
            "active_failures": active_failures,
            "resolved_failures": len(self.failure_history),
            "failure_types": failure_types,
            "impact_levels": impact_levels,
            "mttr_minutes": round(mttr_minutes, 2),
            "cascade_failures": len([f for f in self.failure_history + list(self.active_failures.values()) 
                                   if f.failure_type == FailureType.DEPENDENCY_FAILURE])
        }


class NetworkDependencySimulator:
    """Main simulator for network dependencies and failures"""
    
    def __init__(self):
        self.dependency_graph = ServiceDependencyGraph()
        self.failure_simulator = FailureSimulator(self.dependency_graph)
        self.simulation_running = False
        self.simulation_thread = None
        
    def setup_enterprise_services(self):
        """Setup a realistic enterprise service topology"""
        # Core infrastructure services
        dns_primary = NetworkService(
            service_id="dns_primary",
            service_type=ServiceType.DNS,
            name="Primary DNS Server",
            hostname="dns1.corp.local",
            ip_address="192.168.1.10",
            port=53,
            criticality_level="critical",
            failure_probability=0.0005
        )
        
        dns_secondary = NetworkService(
            service_id="dns_secondary",
            service_type=ServiceType.DNS,
            name="Secondary DNS Server",
            hostname="dns2.corp.local",
            ip_address="192.168.1.11",
            port=53,
            criticality_level="high",
            failure_probability=0.0005
        )
        
        dhcp_server = NetworkService(
            service_id="dhcp_server",
            service_type=ServiceType.DHCP,
            name="DHCP Server",
            hostname="dhcp1.corp.local",
            ip_address="192.168.1.20",
            port=67,
            criticality_level="high",
            failure_probability=0.001
        )
        
        ad_controller = NetworkService(
            service_id="ad_controller",
            service_type=ServiceType.ACTIVE_DIRECTORY,
            name="Active Directory Controller",
            hostname="dc1.corp.local",
            ip_address="192.168.1.30",
            port=389,
            criticality_level="critical",
            failure_probability=0.0008
        )
        
        # Database services
        db_primary = NetworkService(
            service_id="db_primary",
            service_type=ServiceType.DATABASE,
            name="Primary Database Server",
            hostname="db1.corp.local",
            ip_address="192.168.2.10",
            port=1433,
            criticality_level="critical",
            failure_probability=0.001
        )
        
        db_secondary = NetworkService(
            service_id="db_secondary",
            service_type=ServiceType.DATABASE,
            name="Secondary Database Server",
            hostname="db2.corp.local",
            ip_address="192.168.2.11",
            port=1433,
            criticality_level="high",
            backup_services=["db_primary"],
            failure_probability=0.001
        )
        
        # Application services
        web_server = NetworkService(
            service_id="web_server",
            service_type=ServiceType.WEB_SERVER,
            name="Web Server",
            hostname="web1.corp.local",
            ip_address="192.168.3.10",
            port=80,
            criticality_level="high",
            failure_probability=0.002
        )
        
        email_server = NetworkService(
            service_id="email_server",
            service_type=ServiceType.EMAIL_SERVER,
            name="Email Server",
            hostname="mail1.corp.local",
            ip_address="192.168.3.20",
            port=25,
            criticality_level="high",
            failure_probability=0.0015
        )
        
        file_server = NetworkService(
            service_id="file_server",
            service_type=ServiceType.FILE_SERVER,
            name="File Server",
            hostname="files1.corp.local",
            ip_address="192.168.3.30",
            port=445,
            criticality_level="medium",
            failure_probability=0.0012
        )
        
        # Network infrastructure
        core_switch = NetworkService(
            service_id="core_switch",
            service_type=ServiceType.NETWORK_SWITCH,
            name="Core Network Switch",
            hostname="switch1.corp.local",
            ip_address="192.168.1.1",
            port=22,
            criticality_level="critical",
            failure_probability=0.0003
        )
        
        firewall = NetworkService(
            service_id="firewall",
            service_type=ServiceType.FIREWALL,
            name="Enterprise Firewall",
            hostname="fw1.corp.local",
            ip_address="192.168.1.2",
            port=22,
            criticality_level="critical",
            failure_probability=0.0005
        )
        
        load_balancer = NetworkService(
            service_id="load_balancer",
            service_type=ServiceType.LOAD_BALANCER,
            name="Load Balancer",
            hostname="lb1.corp.local",
            ip_address="192.168.3.1",
            port=80,
            criticality_level="high",
            failure_probability=0.001
        )
        
        # Monitoring and support services
        monitoring = NetworkService(
            service_id="monitoring",
            service_type=ServiceType.MONITORING,
            name="Monitoring Server",
            hostname="monitor1.corp.local",
            ip_address="192.168.4.10",
            port=80,
            criticality_level="medium",
            failure_probability=0.002
        )
        
        backup_server = NetworkService(
            service_id="backup_server",
            service_type=ServiceType.BACKUP_SERVER,
            name="Backup Server",
            hostname="backup1.corp.local",
            ip_address="192.168.4.20",
            port=22,
            criticality_level="medium",
            failure_probability=0.0015
        )
        
        # Add all services to the graph
        services = [
            dns_primary, dns_secondary, dhcp_server, ad_controller,
            db_primary, db_secondary, web_server, email_server, file_server,
            core_switch, firewall, load_balancer, monitoring, backup_server
        ]
        
        for service in services:
            self.dependency_graph.add_service(service)
        
        # Setup dependencies
        dependencies = [
            # Core infrastructure dependencies
            ("dhcp_server", "dns_primary"),
            ("ad_controller", "dns_primary"),
            ("ad_controller", "core_switch"),
            
            # Database dependencies
            ("db_primary", "ad_controller"),
            ("db_primary", "core_switch"),
            ("db_secondary", "ad_controller"),
            ("db_secondary", "core_switch"),
            
            # Application dependencies
            ("web_server", "db_primary"),
            ("web_server", "ad_controller"),
            ("web_server", "dns_primary"),
            ("web_server", "load_balancer"),
            
            ("email_server", "ad_controller"),
            ("email_server", "dns_primary"),
            ("email_server", "core_switch"),
            
            ("file_server", "ad_controller"),
            ("file_server", "core_switch"),
            
            # Network dependencies
            ("load_balancer", "core_switch"),
            ("firewall", "core_switch"),
            
            # Monitoring dependencies
            ("monitoring", "core_switch"),
            ("monitoring", "dns_primary"),
            
            ("backup_server", "core_switch"),
            ("backup_server", "ad_controller")
        ]
        
        for service_id, depends_on in dependencies:
            self.dependency_graph.add_dependency(service_id, depends_on)
        
        logger.info("Enterprise service topology setup complete")
    
    def start_simulation(self):
        """Start the network dependency simulation"""
        if self.simulation_running:
            logger.warning("Simulation already running")
            return
        
        logger.info("Starting network dependency simulation")
        self.simulation_running = True
        
        # Start background simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info("Network dependency simulation started successfully")
    
    def stop_simulation(self):
        """Stop the simulation"""
        logger.info("Stopping network dependency simulation")
        self.simulation_running = False
        
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5)
    
    def _simulation_loop(self):
        """Main simulation loop"""
        while self.simulation_running:
            try:
                # Simulate random failures
                failure = self.failure_simulator.simulate_random_failure()
                if failure:
                    logger.info(f"Random failure occurred: {failure.description}")
                
                # Auto-resolve failures
                self.failure_simulator.auto_resolve_failures()
                
                # Update service metrics
                self._update_service_metrics()
                
                # Sleep for simulation interval
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
    
    def _update_service_metrics(self):
        """Update service performance metrics"""
        for service in self.dependency_graph.services.values():
            if service.status == ServiceStatus.RUNNING:
                # Simulate metric variations
                service.response_time_ms += random.uniform(-5, 5)
                service.response_time_ms = max(1, service.response_time_ms)
                
                service.cpu_usage += random.uniform(-2, 2)
                service.cpu_usage = max(0, min(100, service.cpu_usage))
                
                service.memory_usage += random.uniform(-1, 1)
                service.memory_usage = max(0, min(100, service.memory_usage))
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        total_services = len(self.dependency_graph.services)
        running_services = sum(1 for s in self.dependency_graph.services.values() 
                             if s.status == ServiceStatus.RUNNING)
        failed_services = sum(1 for s in self.dependency_graph.services.values() 
                            if s.status == ServiceStatus.FAILED)
        degraded_services = sum(1 for s in self.dependency_graph.services.values() 
                              if s.status == ServiceStatus.DEGRADED)
        
        # Calculate overall health score
        health_score = (running_services / total_services * 100) if total_services > 0 else 0
        
        # Determine health status
        if health_score >= 95:
            health_status = "healthy"
        elif health_score >= 80:
            health_status = "degraded"
        elif health_score >= 60:
            health_status = "impaired"
        else:
            health_status = "critical"
        
        return {
            "health_status": health_status,
            "health_score": round(health_score, 2),
            "total_services": total_services,
            "running_services": running_services,
            "failed_services": failed_services,
            "degraded_services": degraded_services,
            "active_failures": len(self.failure_simulator.active_failures),
            "single_points_of_failure": len(self.dependency_graph.find_single_points_of_failure()),
            "dependency_metrics": self.dependency_graph.get_dependency_metrics(),
            "failure_statistics": self.failure_simulator.get_failure_statistics()
        }
    
    def export_dependency_report(self, filename: str):
        """Export comprehensive dependency report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_health": self.get_system_health(),
            "services": {
                service_id: {
                    "name": service.name,
                    "type": service.service_type.value,
                    "hostname": service.hostname,
                    "ip_address": service.ip_address,
                    "status": service.status.value,
                    "criticality": service.criticality_level,
                    "dependencies": list(service.dependencies),
                    "dependents": list(service.dependents),
                    "uptime_percentage": service.uptime_percentage,
                    "response_time_ms": service.response_time_ms,
                    "failure_probability": service.failure_probability
                }
                for service_id, service in self.dependency_graph.services.items()
            },
            "active_failures": {
                failure.failure_id: {
                    "service_id": failure.service_id,
                    "failure_type": failure.failure_type.value,
                    "start_time": failure.start_time.isoformat(),
                    "description": failure.description,
                    "impact_level": failure.impact_level,
                    "affected_services": failure.affected_services
                }
                for failure in self.failure_simulator.active_failures.values()
            },
            "single_points_of_failure": self.dependency_graph.find_single_points_of_failure(),
            "dependency_metrics": self.dependency_graph.get_dependency_metrics()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Dependency report exported to {filename}")


def main():
    """Main function for testing the network dependency simulation"""
    simulator = NetworkDependencySimulator()
    
    try:
        # Setup enterprise services
        simulator.setup_enterprise_services()
        
        # Visualize dependencies
        simulator.dependency_graph.visualize_dependencies()
        
        # Start simulation
        simulator.start_simulation()
        
        # Let it run for a bit
        time.sleep(60)
        
        # Get system health
        health = simulator.get_system_health()
        print(f"System Health: {health}")
        
        # Export report
        simulator.export_dependency_report("network_dependency_report.json")
        
    finally:
        simulator.stop_simulation()


if __name__ == "__main__":
    main()