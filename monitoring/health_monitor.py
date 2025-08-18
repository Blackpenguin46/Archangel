"""
Health monitoring system for Archangel agents and infrastructure.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import psutil

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_function: Callable[[], bool]
    interval: float  # seconds
    timeout: float   # seconds
    last_check: float = field(default_factory=time.time)
    last_status: HealthStatus = HealthStatus.UNKNOWN
    failure_count: int = 0
    max_failures: int = 3
    description: str = ""

@dataclass
class ComponentHealth:
    """Health status for a system component."""
    component_id: str
    component_type: str
    status: HealthStatus
    last_healthy: float
    checks: Dict[str, HealthCheck] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class HealthMonitor:
    """
    Comprehensive health monitoring system for Archangel components.
    
    Monitors:
    - Agent health and responsiveness
    - System resources (CPU, memory, disk)
    - Service availability (databases, message queues)
    - Communication channels
    - Performance metrics
    """
    
    def __init__(self, check_interval: float = 30.0):
        """Initialize the health monitor.
        
        Args:
            check_interval: Default interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.components: Dict[str, ComponentHealth] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self._lock = threading.RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._status_callbacks: List[Callable[[str, HealthStatus], None]] = []
        
        # Initialize default health checks
        self._init_default_checks()
        
    def _init_default_checks(self):
        """Initialize default system health checks."""
        
        # System resource checks
        self.add_health_check(
            "system_cpu",
            self._check_system_cpu,
            interval=30.0,
            timeout=5.0,
            description="System CPU usage check"
        )
        
        self.add_health_check(
            "system_memory", 
            self._check_system_memory,
            interval=30.0,
            timeout=5.0,
            description="System memory usage check"
        )
        
        self.add_health_check(
            "system_disk",
            self._check_system_disk,
            interval=60.0,
            timeout=5.0,
            description="System disk space check"
        )
        
        # Service availability checks
        self.add_health_check(
            "docker_daemon",
            self._check_docker_daemon,
            interval=60.0,
            timeout=10.0,
            description="Docker daemon availability"
        )
        
    def start_monitoring(self):
        """Start the health monitoring loop."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Health monitoring already running")
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Health monitoring started")
        
    def stop_monitoring(self):
        """Stop the health monitoring loop."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10.0)
        logger.info("Health monitoring stopped")
        
    def add_health_check(self, name: str, check_function: Callable[[], bool],
                        interval: float = 30.0, timeout: float = 5.0,
                        max_failures: int = 3, description: str = ""):
        """Add a new health check.
        
        Args:
            name: Unique name for the health check
            check_function: Function that returns True if healthy, False otherwise
            interval: Interval between checks in seconds
            timeout: Timeout for the check function
            max_failures: Maximum consecutive failures before marking as critical
            description: Human-readable description of the check
        """
        with self._lock:
            check = HealthCheck(
                name=name,
                check_function=check_function,
                interval=interval,
                timeout=timeout,
                max_failures=max_failures,
                description=description
            )
            self.health_checks[name] = check
            logger.info(f"Added health check: {name}")
            
    def remove_health_check(self, name: str):
        """Remove a health check.
        
        Args:
            name: Name of the health check to remove
        """
        with self._lock:
            if name in self.health_checks:
                del self.health_checks[name]
                logger.info(f"Removed health check: {name}")
                
    def register_component(self, component_id: str, component_type: str):
        """Register a component for health monitoring.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Type of component (e.g., 'agent', 'service', 'database')
        """
        with self._lock:
            self.components[component_id] = ComponentHealth(
                component_id=component_id,
                component_type=component_type,
                status=HealthStatus.UNKNOWN,
                last_healthy=time.time()
            )
            logger.info(f"Registered component: {component_id} ({component_type})")
            
    def unregister_component(self, component_id: str):
        """Unregister a component from health monitoring.
        
        Args:
            component_id: ID of component to unregister
        """
        with self._lock:
            if component_id in self.components:
                del self.components[component_id]
                logger.info(f"Unregistered component: {component_id}")
                
    def update_component_health(self, component_id: str, status: HealthStatus,
                               metadata: Optional[Dict[str, Any]] = None):
        """Update the health status of a component.
        
        Args:
            component_id: ID of the component
            status: New health status
            metadata: Optional metadata about the health status
        """
        with self._lock:
            if component_id not in self.components:
                logger.warning(f"Health update for unregistered component: {component_id}")
                return
                
            component = self.components[component_id]
            old_status = component.status
            component.status = status
            
            if metadata:
                component.metadata.update(metadata)
                
            if status == HealthStatus.HEALTHY:
                component.last_healthy = time.time()
                
            # Notify callbacks if status changed
            if old_status != status:
                for callback in self._status_callbacks:
                    try:
                        callback(component_id, status)
                    except Exception as e:
                        logger.error(f"Error in status callback: {e}")
                        
    def add_status_callback(self, callback: Callable[[str, HealthStatus], None]):
        """Add a callback to be notified of status changes.
        
        Args:
            callback: Function to call when component status changes
        """
        self._status_callbacks.append(callback)
        
    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            ComponentHealth object or None if not found
        """
        with self._lock:
            return self.components.get(component_id)
            
    def get_all_components(self) -> Dict[str, ComponentHealth]:
        """Get health status for all components.
        
        Returns:
            Dictionary mapping component IDs to ComponentHealth objects
        """
        with self._lock:
            return self.components.copy()
            
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary.
        
        Returns:
            Dictionary with system health statistics
        """
        with self._lock:
            total_components = len(self.components)
            if total_components == 0:
                return {
                    "overall_status": HealthStatus.UNKNOWN.value,
                    "total_components": 0,
                    "healthy": 0,
                    "warning": 0,
                    "critical": 0,
                    "unknown": 0
                }
                
            status_counts = {
                HealthStatus.HEALTHY.value: 0,
                HealthStatus.WARNING.value: 0,
                HealthStatus.CRITICAL.value: 0,
                HealthStatus.UNKNOWN.value: 0
            }
            
            for component in self.components.values():
                status_counts[component.status.value] += 1
                
            # Determine overall status
            if status_counts[HealthStatus.CRITICAL.value] > 0:
                overall_status = HealthStatus.CRITICAL
            elif status_counts[HealthStatus.WARNING.value] > 0:
                overall_status = HealthStatus.WARNING
            elif status_counts[HealthStatus.HEALTHY.value] == total_components:
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.UNKNOWN
                
            return {
                "overall_status": overall_status.value,
                "total_components": total_components,
                "healthy": status_counts[HealthStatus.HEALTHY.value],
                "warning": status_counts[HealthStatus.WARNING.value], 
                "critical": status_counts[HealthStatus.CRITICAL.value],
                "unknown": status_counts[HealthStatus.UNKNOWN.value],
                "last_updated": time.time()
            }
            
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._run_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(5.0)  # Back off on errors
                
    def _run_health_checks(self):
        """Run all scheduled health checks."""
        current_time = time.time()
        
        with self._lock:
            for name, check in self.health_checks.items():
                # Check if it's time to run this check
                if current_time - check.last_check >= check.interval:
                    self._run_single_check(check)
                    
    def _run_single_check(self, check: HealthCheck):
        """Run a single health check.
        
        Args:
            check: HealthCheck object to execute
        """
        try:
            start_time = time.time()
            
            # Run the check with timeout (simplified timeout implementation)
            result = check.check_function()
            
            execution_time = time.time() - start_time
            check.last_check = time.time()
            
            if result:
                check.last_status = HealthStatus.HEALTHY
                check.failure_count = 0
            else:
                check.failure_count += 1
                if check.failure_count >= check.max_failures:
                    check.last_status = HealthStatus.CRITICAL
                else:
                    check.last_status = HealthStatus.WARNING
                    
            # Update system component health based on check results
            if check.name.startswith("system_"):
                self.update_component_health(
                    "system",
                    check.last_status,
                    {"check": check.name, "execution_time": execution_time}
                )
                
        except Exception as e:
            logger.error(f"Error running health check {check.name}: {e}")
            check.last_status = HealthStatus.CRITICAL
            check.failure_count += 1
            
    def _check_system_cpu(self) -> bool:
        """Check system CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 90.0  # Healthy if CPU < 90%
        except Exception as e:
            logger.error(f"CPU check failed: {e}")
            return False
            
    def _check_system_memory(self) -> bool:
        """Check system memory usage.""" 
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            return memory_percent < 85.0  # Healthy if memory < 85%
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False
            
    def _check_system_disk(self) -> bool:
        """Check system disk space."""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            return disk_percent < 90.0  # Healthy if disk < 90%
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return False
            
    def _check_docker_daemon(self) -> bool:
        """Check if Docker daemon is running."""
        try:
            import subprocess
            result = subprocess.run(
                ['docker', 'info'],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Docker check failed: {e}")
            return False

# Global health monitor instance
health_monitor = HealthMonitor()