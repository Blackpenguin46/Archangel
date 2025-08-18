"""
Automated recovery system for Archangel components.
"""

import time
import threading
import logging
import subprocess
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from .health_monitor import HealthMonitor, HealthStatus

logger = logging.getLogger(__name__)

class RecoveryAction(Enum):
    """Types of recovery actions."""
    RESTART_AGENT = "restart_agent"
    RESTART_SERVICE = "restart_service" 
    RESTART_CONTAINER = "restart_container"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CLEAR_CACHE = "clear_cache"
    RESET_CONNECTION = "reset_connection"
    ALERT_ONLY = "alert_only"
    CUSTOM = "custom"

@dataclass
class RecoveryRule:
    """Recovery rule definition."""
    name: str
    component_pattern: str  # Regex pattern for component matching
    trigger_status: HealthStatus
    action: RecoveryAction
    action_params: Dict[str, Any] = field(default_factory=dict)
    cooldown_seconds: float = 300.0  # 5 minutes
    max_attempts: int = 3
    custom_function: Optional[Callable[[str, Dict[str, Any]], bool]] = None
    description: str = ""
    last_execution: float = 0.0
    execution_count: int = 0

@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    timestamp: float
    component_id: str
    rule_name: str
    action: RecoveryAction
    success: bool
    error_message: str = ""
    duration: float = 0.0

class RecoverySystem:
    """
    Automated recovery system that responds to health status changes.
    
    Features:
    - Automatic restart of failed agents/services
    - Container orchestration integration
    - Resource scaling based on load
    - Custom recovery actions
    - Recovery attempt tracking and limits
    - Cooldown periods to prevent recovery storms
    """
    
    def __init__(self, health_monitor: HealthMonitor):
        """Initialize the recovery system.
        
        Args:
            health_monitor: HealthMonitor instance to monitor
        """
        self.health_monitor = health_monitor
        self.rules: Dict[str, RecoveryRule] = {}
        self.recovery_history: List[RecoveryAttempt] = []
        self._lock = threading.RLock()
        self._running = False
        self._max_history = 1000  # Keep last 1000 recovery attempts
        
        # Register for health status changes
        self.health_monitor.add_status_callback(self._handle_status_change)
        
        # Initialize default recovery rules
        self._init_default_rules()
        
    def _init_default_rules(self):
        """Initialize default recovery rules."""
        
        # Agent restart rule
        self.add_recovery_rule(
            name="restart_failed_agents",
            component_pattern=r"agent_.*",
            trigger_status=HealthStatus.CRITICAL,
            action=RecoveryAction.RESTART_AGENT,
            cooldown_seconds=60.0,
            max_attempts=3,
            description="Restart agents that become critical"
        )
        
        # Service restart rule
        self.add_recovery_rule(
            name="restart_failed_services",
            component_pattern=r"service_.*",
            trigger_status=HealthStatus.CRITICAL,
            action=RecoveryAction.RESTART_SERVICE,
            cooldown_seconds=120.0,
            max_attempts=2,
            description="Restart services that become critical"
        )
        
        # Container restart rule
        self.add_recovery_rule(
            name="restart_failed_containers",
            component_pattern=r"container_.*",
            trigger_status=HealthStatus.CRITICAL,
            action=RecoveryAction.RESTART_CONTAINER,
            cooldown_seconds=180.0,
            max_attempts=3,
            description="Restart containers that become critical"
        )
        
        # System resource warning rule (alert only)
        self.add_recovery_rule(
            name="system_resource_warning",
            component_pattern=r"system",
            trigger_status=HealthStatus.WARNING,
            action=RecoveryAction.ALERT_ONLY,
            cooldown_seconds=300.0,
            max_attempts=1,
            description="Alert on system resource warnings"
        )
        
    def add_recovery_rule(self, name: str, component_pattern: str, 
                         trigger_status: HealthStatus, action: RecoveryAction,
                         action_params: Optional[Dict[str, Any]] = None,
                         cooldown_seconds: float = 300.0, max_attempts: int = 3,
                         custom_function: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
                         description: str = ""):
        """Add a new recovery rule.
        
        Args:
            name: Unique name for the rule
            component_pattern: Regex pattern to match component IDs
            trigger_status: Health status that triggers the rule
            action: Recovery action to take
            action_params: Parameters for the recovery action
            cooldown_seconds: Minimum time between executions
            max_attempts: Maximum recovery attempts
            custom_function: Custom recovery function for CUSTOM action
            description: Human-readable description
        """
        with self._lock:
            rule = RecoveryRule(
                name=name,
                component_pattern=component_pattern,
                trigger_status=trigger_status,
                action=action,
                action_params=action_params or {},
                cooldown_seconds=cooldown_seconds,
                max_attempts=max_attempts,
                custom_function=custom_function,
                description=description
            )
            self.rules[name] = rule
            logger.info(f"Added recovery rule: {name}")
            
    def remove_recovery_rule(self, name: str):
        """Remove a recovery rule.
        
        Args:
            name: Name of the rule to remove
        """
        with self._lock:
            if name in self.rules:
                del self.rules[name]
                logger.info(f"Removed recovery rule: {name}")
                
    def get_recovery_rules(self) -> Dict[str, RecoveryRule]:
        """Get all recovery rules.
        
        Returns:
            Dictionary of recovery rules
        """
        with self._lock:
            return self.rules.copy()
            
    def get_recovery_history(self, limit: int = 100) -> List[RecoveryAttempt]:
        """Get recent recovery attempts.
        
        Args:
            limit: Maximum number of attempts to return
            
        Returns:
            List of recent RecoveryAttempt objects
        """
        with self._lock:
            return self.recovery_history[-limit:]
            
    def _handle_status_change(self, component_id: str, status: HealthStatus):
        """Handle health status changes from the monitor.
        
        Args:
            component_id: ID of the component that changed status
            status: New health status
        """
        if status == HealthStatus.HEALTHY:
            return  # No recovery needed for healthy components
            
        # Find matching recovery rules
        matching_rules = self._find_matching_rules(component_id, status)
        
        for rule in matching_rules:
            if self._should_execute_rule(rule):
                self._execute_recovery_action(component_id, rule)
                
    def _find_matching_rules(self, component_id: str, status: HealthStatus) -> List[RecoveryRule]:
        """Find recovery rules that match the component and status.
        
        Args:
            component_id: ID of the component
            status: Current health status
            
        Returns:
            List of matching recovery rules
        """
        import re
        
        matching_rules = []
        
        with self._lock:
            for rule in self.rules.values():
                # Check if component pattern matches
                if re.match(rule.component_pattern, component_id):
                    # Check if status matches
                    if rule.trigger_status == status:
                        matching_rules.append(rule)
                        
        return matching_rules
        
    def _should_execute_rule(self, rule: RecoveryRule) -> bool:
        """Check if a recovery rule should be executed.
        
        Args:
            rule: Recovery rule to check
            
        Returns:
            True if the rule should be executed
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - rule.last_execution < rule.cooldown_seconds:
            logger.debug(f"Recovery rule {rule.name} in cooldown period")
            return False
            
        # Check max attempts
        if rule.execution_count >= rule.max_attempts:
            logger.warning(f"Recovery rule {rule.name} exceeded max attempts ({rule.max_attempts})")
            return False
            
        return True
        
    def _execute_recovery_action(self, component_id: str, rule: RecoveryRule):
        """Execute a recovery action for a component.
        
        Args:
            component_id: ID of the component to recover
            rule: Recovery rule to execute
        """
        start_time = time.time()
        success = False
        error_message = ""
        
        try:
            logger.info(f"Executing recovery action {rule.action.value} for {component_id} (rule: {rule.name})")
            
            if rule.action == RecoveryAction.RESTART_AGENT:
                success = self._restart_agent(component_id, rule.action_params)
            elif rule.action == RecoveryAction.RESTART_SERVICE:
                success = self._restart_service(component_id, rule.action_params)
            elif rule.action == RecoveryAction.RESTART_CONTAINER:
                success = self._restart_container(component_id, rule.action_params)
            elif rule.action == RecoveryAction.SCALE_UP:
                success = self._scale_up(component_id, rule.action_params)
            elif rule.action == RecoveryAction.SCALE_DOWN:
                success = self._scale_down(component_id, rule.action_params)
            elif rule.action == RecoveryAction.CLEAR_CACHE:
                success = self._clear_cache(component_id, rule.action_params)
            elif rule.action == RecoveryAction.RESET_CONNECTION:
                success = self._reset_connection(component_id, rule.action_params)
            elif rule.action == RecoveryAction.ALERT_ONLY:
                success = self._send_alert(component_id, rule.action_params)
            elif rule.action == RecoveryAction.CUSTOM:
                if rule.custom_function:
                    success = rule.custom_function(component_id, rule.action_params)
                else:
                    error_message = "Custom recovery function not provided"
            else:
                error_message = f"Unknown recovery action: {rule.action}"
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Recovery action failed: {e}")
            
        # Update rule execution tracking
        with self._lock:
            rule.last_execution = time.time()
            rule.execution_count += 1
            
            # Record the attempt
            attempt = RecoveryAttempt(
                timestamp=start_time,
                component_id=component_id,
                rule_name=rule.name,
                action=rule.action,
                success=success,
                error_message=error_message,
                duration=time.time() - start_time
            )
            
            self.recovery_history.append(attempt)
            
            # Trim history if too large
            if len(self.recovery_history) > self._max_history:
                self.recovery_history = self.recovery_history[-self._max_history:]
                
        if success:
            logger.info(f"Recovery action {rule.action.value} succeeded for {component_id}")
        else:
            logger.error(f"Recovery action {rule.action.value} failed for {component_id}: {error_message}")
            
    def _restart_agent(self, component_id: str, params: Dict[str, Any]) -> bool:
        """Restart an agent.
        
        Args:
            component_id: ID of the agent to restart
            params: Action parameters
            
        Returns:
            True if successful
        """
        try:
            # In a real implementation, this would interact with the agent manager
            # For now, simulate a restart
            logger.info(f"Simulating restart of agent {component_id}")
            time.sleep(2)  # Simulate restart time
            return True
        except Exception as e:
            logger.error(f"Failed to restart agent {component_id}: {e}")
            return False
            
    def _restart_service(self, component_id: str, params: Dict[str, Any]) -> bool:
        """Restart a system service.
        
        Args:
            component_id: ID of the service to restart
            params: Action parameters
            
        Returns:
            True if successful
        """
        try:
            service_name = params.get('service_name', component_id.replace('service_', ''))
            
            # Try systemctl restart
            result = subprocess.run(
                ['sudo', 'systemctl', 'restart', service_name],
                capture_output=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to restart service {component_id}: {e}")
            return False
            
    def _restart_container(self, component_id: str, params: Dict[str, Any]) -> bool:
        """Restart a Docker container.
        
        Args:
            component_id: ID of the container to restart
            params: Action parameters
            
        Returns:
            True if successful
        """
        try:
            container_name = params.get('container_name', component_id.replace('container_', ''))
            
            # Restart the container
            result = subprocess.run(
                ['docker', 'restart', container_name],
                capture_output=True,
                timeout=60
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to restart container {component_id}: {e}")
            return False
            
    def _scale_up(self, component_id: str, params: Dict[str, Any]) -> bool:
        """Scale up a service or container.
        
        Args:
            component_id: ID of the service to scale up
            params: Action parameters
            
        Returns:
            True if successful
        """
        try:
            service_name = params.get('service_name', component_id)
            scale_count = params.get('scale_count', 1)
            
            # Use docker-compose to scale up
            result = subprocess.run(
                ['docker-compose', 'scale', f"{service_name}={scale_count}"],
                capture_output=True,
                timeout=120
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to scale up {component_id}: {e}")
            return False
            
    def _scale_down(self, component_id: str, params: Dict[str, Any]) -> bool:
        """Scale down a service or container.
        
        Args:
            component_id: ID of the service to scale down
            params: Action parameters
            
        Returns:
            True if successful
        """
        try:
            service_name = params.get('service_name', component_id)
            scale_count = params.get('scale_count', 1)
            
            # Use docker-compose to scale down
            result = subprocess.run(
                ['docker-compose', 'scale', f"{service_name}={scale_count}"],
                capture_output=True,
                timeout=120
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to scale down {component_id}: {e}")
            return False
            
    def _clear_cache(self, component_id: str, params: Dict[str, Any]) -> bool:
        """Clear cache for a component.
        
        Args:
            component_id: ID of the component
            params: Action parameters
            
        Returns:
            True if successful
        """
        try:
            # Implementation would depend on the specific caching system
            logger.info(f"Simulating cache clear for {component_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache for {component_id}: {e}")
            return False
            
    def _reset_connection(self, component_id: str, params: Dict[str, Any]) -> bool:
        """Reset network connections for a component.
        
        Args:
            component_id: ID of the component
            params: Action parameters
            
        Returns:
            True if successful
        """
        try:
            # Implementation would reset network connections
            logger.info(f"Simulating connection reset for {component_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset connections for {component_id}: {e}")
            return False
            
    def _send_alert(self, component_id: str, params: Dict[str, Any]) -> bool:
        """Send an alert about the component status.
        
        Args:
            component_id: ID of the component
            params: Action parameters
            
        Returns:
            True if successful
        """
        try:
            # Send alert (could be webhook, email, Slack, etc.)
            alert_message = params.get('message', f"Component {component_id} requires attention")
            logger.warning(f"ALERT: {alert_message}")
            
            # In a real implementation, this would send notifications
            return True
        except Exception as e:
            logger.error(f"Failed to send alert for {component_id}: {e}")
            return False
            
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery system statistics.
        
        Returns:
            Dictionary with recovery statistics
        """
        with self._lock:
            total_attempts = len(self.recovery_history)
            if total_attempts == 0:
                return {
                    "total_attempts": 0,
                    "success_rate": 0.0,
                    "most_common_action": None,
                    "most_recovered_component": None
                }
                
            successful_attempts = sum(1 for attempt in self.recovery_history if attempt.success)
            success_rate = successful_attempts / total_attempts
            
            # Count actions
            action_counts = {}
            component_counts = {}
            for attempt in self.recovery_history:
                action = attempt.action.value
                action_counts[action] = action_counts.get(action, 0) + 1
                
                component = attempt.component_id
                component_counts[component] = component_counts.get(component, 0) + 1
                
            most_common_action = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else None
            most_recovered_component = max(component_counts.items(), key=lambda x: x[1])[0] if component_counts else None
            
            return {
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "success_rate": success_rate,
                "most_common_action": most_common_action,
                "most_recovered_component": most_recovered_component,
                "action_counts": action_counts,
                "component_counts": component_counts
            }

# Global recovery system (will be initialized with health monitor)
recovery_system: Optional[RecoverySystem] = None