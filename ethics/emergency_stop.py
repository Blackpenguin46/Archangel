"""
Emergency stop mechanisms and constraint enforcement for autonomous agents.
"""

import time
import threading
import logging
import signal
import os
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

logger = logging.getLogger(__name__)

class EmergencyStopReason(Enum):
    """Reasons for emergency stop activation."""
    ETHICAL_VIOLATION = "ethical_violation"
    BOUNDARY_BREACH = "boundary_breach"
    SAFETY_THRESHOLD = "safety_threshold"
    HUMAN_INTERVENTION = "human_intervention"
    SYSTEM_MALFUNCTION = "system_malfunction"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_THREAT = "security_threat"
    MANUAL_OVERRIDE = "manual_override"

class StopScope(Enum):
    """Scope of emergency stop."""
    SINGLE_AGENT = "single_agent"        # Stop specific agent
    AGENT_TEAM = "agent_team"           # Stop entire team (red/blue)
    AGENT_TYPE = "agent_type"           # Stop all agents of specific type
    ALL_AGENTS = "all_agents"           # Stop all agents
    SYSTEM_WIDE = "system_wide"         # Stop entire system

class ConstraintType(Enum):
    """Types of constraints that can be enforced."""
    EXECUTION_TIME = "execution_time"    # Time limits on operations
    RESOURCE_USAGE = "resource_usage"    # CPU/memory limits
    ACTION_COUNT = "action_count"        # Limit number of actions
    INTERACTION_RATE = "interaction_rate" # Rate limiting
    CAPABILITY_RESTRICTION = "capability_restriction"  # Feature restrictions
    GEOGRAPHIC_BOUNDARY = "geographic_boundary"  # Virtual geography limits

@dataclass
class StopCommand:
    """Emergency stop command."""
    command_id: str
    scope: StopScope
    target: str  # Agent ID, team name, or "*" for all
    reason: EmergencyStopReason
    description: str
    issued_by: str
    timestamp: float = field(default_factory=time.time)
    authorization_level: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    auto_resolve: bool = False
    resolve_after: Optional[float] = None  # Seconds until auto-resolve
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Constraint:
    """System constraint definition."""
    constraint_id: str
    constraint_type: ConstraintType
    description: str
    target_scope: StopScope
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enforcement_level: str = "strict"  # "strict", "moderate", "advisory"
    enabled: bool = True
    violation_count: int = 0
    last_violation: Optional[float] = None

@dataclass
class ConstraintViolation:
    """Constraint violation record."""
    violation_id: str
    constraint_id: str
    agent_id: str
    violation_type: ConstraintType
    severity: str
    description: str
    measured_value: Any
    threshold_value: Any
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    automatic_response: str = ""

class EmergencyStopSystem:
    """
    Comprehensive emergency stop and constraint enforcement system.
    
    Features:
    - Multi-level emergency stops (agent, team, system-wide)
    - Constraint enforcement with violation tracking
    - Authorization levels for different stop commands
    - Automatic and manual override capabilities
    - Recovery and restart mechanisms
    - Audit trail of all emergency actions
    """
    
    def __init__(self):
        """Initialize emergency stop system."""
        self.active_stops: Dict[str, StopCommand] = {}
        self.stop_history: List[StopCommand] = []
        self.constraints: Dict[str, Constraint] = {}
        self.violations: List[ConstraintViolation] = []
        
        self._lock = threading.RLock()
        self.max_history = 1000
        
        # System state
        self.system_stopped = False
        self.stopped_agents: Set[str] = set()
        self.stopped_teams: Set[str] = set()
        
        # Callbacks
        self._stop_callbacks: List[Callable[[StopCommand], None]] = []
        self._constraint_callbacks: List[Callable[[ConstraintViolation], None]] = []
        self._recovery_callbacks: List[Callable[[str, str], None]] = []  # agent_id, reason
        
        # Authorization codes for different levels
        self._authorization_codes = {
            1: "basic_stop",
            2: "elevated_stop", 
            3: "emergency_halt",
            4: "system_override"
        }
        
        # Initialize default constraints
        self._initialize_default_constraints()
        
    def emergency_stop(self, scope: StopScope, target: str, reason: EmergencyStopReason,
                      description: str, issued_by: str, authorization_level: int = 1,
                      auto_resolve: bool = False, resolve_after: Optional[float] = None) -> str:
        """Issue emergency stop command.
        
        Args:
            scope: Scope of the stop (agent, team, system)
            target: Target identifier (agent ID, team name, etc.)
            reason: Reason for emergency stop
            description: Detailed description
            issued_by: Who issued the stop command
            authorization_level: Authorization level required (1-4)
            auto_resolve: Whether to auto-resolve after time
            resolve_after: Seconds until auto-resolve
            
        Returns:
            Command ID for the stop command
        """
        command_id = self._generate_command_id(scope, target, reason)
        
        command = StopCommand(
            command_id=command_id,
            scope=scope,
            target=target,
            reason=reason,
            description=description,
            issued_by=issued_by,
            authorization_level=authorization_level,
            auto_resolve=auto_resolve,
            resolve_after=resolve_after
        )
        
        with self._lock:
            # Execute the stop command
            self._execute_stop_command(command)
            
            # Store active command
            self.active_stops[command_id] = command
            
            # Add to history
            self.stop_history.append(command)
            self._trim_history()
            
        # Notify callbacks
        for callback in self._stop_callbacks:
            try:
                callback(command)
            except Exception as e:
                logger.error(f"Error in stop callback: {e}")
                
        logger.critical(f"Emergency stop issued: {command_id} - {description}")
        return command_id
        
    def resolve_stop(self, command_id: str, resolved_by: str, 
                    authorization_code: Optional[str] = None) -> bool:
        """Resolve an emergency stop command.
        
        Args:
            command_id: ID of the stop command to resolve
            resolved_by: Who is resolving the stop
            authorization_code: Authorization code for high-level stops
            
        Returns:
            True if stop was resolved successfully
        """
        with self._lock:
            if command_id not in self.active_stops:
                logger.warning(f"Attempted to resolve non-active stop: {command_id}")
                return False
                
            command = self.active_stops[command_id]
            
            # Check authorization for high-level stops
            if command.authorization_level >= 3 and not authorization_code:
                logger.error(f"Authorization code required to resolve stop {command_id}")
                return False
                
            if authorization_code and not self._verify_authorization_code(
                command.authorization_level, authorization_code):
                logger.error(f"Invalid authorization code for stop {command_id}")
                return False
                
            # Resolve the stop
            self._resolve_stop_command(command)
            
            # Remove from active stops
            del self.active_stops[command_id]
            
        logger.info(f"Emergency stop resolved: {command_id} by {resolved_by}")
        
        # Notify recovery callbacks
        for callback in self._recovery_callbacks:
            try:
                callback(command.target, f"Stop resolved: {command_id}")
            except Exception as e:
                logger.error(f"Error in recovery callback: {e}")
                
        return True
        
    def add_constraint(self, constraint: Constraint):
        """Add a system constraint.
        
        Args:
            constraint: Constraint to add
        """
        with self._lock:
            self.constraints[constraint.constraint_id] = constraint
            logger.info(f"Added constraint: {constraint.constraint_id}")
            
    def remove_constraint(self, constraint_id: str):
        """Remove a system constraint.
        
        Args:
            constraint_id: ID of constraint to remove
        """
        with self._lock:
            if constraint_id in self.constraints:
                del self.constraints[constraint_id]
                logger.info(f"Removed constraint: {constraint_id}")
                
    def check_constraints(self, agent_id: str, action_type: str, 
                         action_data: Dict[str, Any]) -> List[ConstraintViolation]:
        """Check if action violates any constraints.
        
        Args:
            agent_id: ID of the agent
            action_type: Type of action being performed
            action_data: Data about the action
            
        Returns:
            List of constraint violations
        """
        violations = []
        
        with self._lock:
            for constraint in self.constraints.values():
                if not constraint.enabled:
                    continue
                    
                # Check if constraint applies to this agent/action
                if not self._constraint_applies(constraint, agent_id, action_type):
                    continue
                    
                # Check specific constraint types
                violation = self._check_specific_constraint(
                    constraint, agent_id, action_type, action_data)
                
                if violation:
                    violations.append(violation)
                    
                    # Update constraint violation count
                    constraint.violation_count += 1
                    constraint.last_violation = time.time()
                    
                    # Enforce constraint based on level
                    self._enforce_constraint(constraint, violation)
                    
        return violations
        
    def is_agent_stopped(self, agent_id: str) -> bool:
        """Check if agent is currently stopped.
        
        Args:
            agent_id: ID of the agent to check
            
        Returns:
            True if agent is stopped
        """
        with self._lock:
            if self.system_stopped:
                return True
                
            if agent_id in self.stopped_agents:
                return True
                
            # Check team stops
            for command in self.active_stops.values():
                if command.scope == StopScope.AGENT_TEAM:
                    # This would need agent team information
                    pass
                elif command.scope == StopScope.ALL_AGENTS:
                    return True
                    
            return False
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get emergency stop system status.
        
        Returns:
            Dictionary with system status
        """
        with self._lock:
            return {
                "system_stopped": self.system_stopped,
                "active_stops": len(self.active_stops),
                "stopped_agents": len(self.stopped_agents),
                "stopped_teams": len(self.stopped_teams),
                "total_constraints": len(self.constraints),
                "enabled_constraints": sum(1 for c in self.constraints.values() if c.enabled),
                "total_violations": len(self.violations),
                "recent_violations": len([v for v in self.violations 
                                        if time.time() - v.timestamp < 3600])
            }
            
    def get_stop_history(self, limit: int = 100) -> List[StopCommand]:
        """Get emergency stop history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of stop commands
        """
        with self._lock:
            return self.stop_history[-limit:]
            
    def _execute_stop_command(self, command: StopCommand):
        """Execute emergency stop command.
        
        Args:
            command: Stop command to execute
        """
        if command.scope == StopScope.SINGLE_AGENT:
            self.stopped_agents.add(command.target)
            logger.warning(f"Stopped agent: {command.target}")
            
        elif command.scope == StopScope.AGENT_TEAM:
            self.stopped_teams.add(command.target)
            logger.warning(f"Stopped team: {command.target}")
            
        elif command.scope == StopScope.ALL_AGENTS:
            self.system_stopped = True
            logger.critical("Stopped all agents")
            
        elif command.scope == StopScope.SYSTEM_WIDE:
            self.system_stopped = True
            logger.critical("System-wide emergency stop activated")
            
            # In a real implementation, this might:
            # - Send SIGTERM to all agent processes
            # - Close network connections
            # - Save current state
            # - Notify external systems
            
    def _resolve_stop_command(self, command: StopCommand):
        """Resolve emergency stop command.
        
        Args:
            command: Stop command to resolve
        """
        if command.scope == StopScope.SINGLE_AGENT:
            self.stopped_agents.discard(command.target)
            logger.info(f"Resumed agent: {command.target}")
            
        elif command.scope == StopScope.AGENT_TEAM:
            self.stopped_teams.discard(command.target)
            logger.info(f"Resumed team: {command.target}")
            
        elif command.scope in [StopScope.ALL_AGENTS, StopScope.SYSTEM_WIDE]:
            # Only resume system if no other system-wide stops are active
            system_wide_stops = [cmd for cmd in self.active_stops.values()
                               if cmd.scope in [StopScope.ALL_AGENTS, StopScope.SYSTEM_WIDE]
                               and cmd.command_id != command.command_id]
            
            if not system_wide_stops:
                self.system_stopped = False
                logger.info("System-wide stop resolved")
                
    def _initialize_default_constraints(self):
        """Initialize default system constraints."""
        
        # Execution time constraint
        execution_time_constraint = Constraint(
            constraint_id="max_execution_time",
            constraint_type=ConstraintType.EXECUTION_TIME,
            description="Maximum execution time per action",
            target_scope=StopScope.ALL_AGENTS,
            target="*",
            parameters={
                "max_seconds": 300,  # 5 minutes
                "warning_threshold": 240  # 4 minutes
            }
        )
        
        # Action count constraint
        action_count_constraint = Constraint(
            constraint_id="max_actions_per_minute",
            constraint_type=ConstraintType.ACTION_COUNT,
            description="Maximum actions per minute per agent",
            target_scope=StopScope.ALL_AGENTS,
            target="*",
            parameters={
                "max_actions": 60,
                "time_window": 60
            }
        )
        
        # Resource usage constraint
        resource_constraint = Constraint(
            constraint_id="max_resource_usage",
            constraint_type=ConstraintType.RESOURCE_USAGE,
            description="Maximum resource usage per agent",
            target_scope=StopScope.ALL_AGENTS,
            target="*",
            parameters={
                "max_cpu_percent": 80,
                "max_memory_mb": 512
            }
        )
        
        self.add_constraint(execution_time_constraint)
        self.add_constraint(action_count_constraint)
        self.add_constraint(resource_constraint)
        
    def _constraint_applies(self, constraint: Constraint, agent_id: str, action_type: str) -> bool:
        """Check if constraint applies to agent/action.
        
        Args:
            constraint: Constraint to check
            agent_id: Agent ID
            action_type: Action type
            
        Returns:
            True if constraint applies
        """
        if constraint.target_scope == StopScope.ALL_AGENTS:
            return True
        elif constraint.target_scope == StopScope.SINGLE_AGENT:
            return constraint.target == agent_id
        # Add other scope checks as needed
        
        return False
        
    def _check_specific_constraint(self, constraint: Constraint, agent_id: str,
                                 action_type: str, action_data: Dict[str, Any]) -> Optional[ConstraintViolation]:
        """Check specific constraint type.
        
        Args:
            constraint: Constraint to check
            agent_id: Agent ID
            action_type: Action type  
            action_data: Action data
            
        Returns:
            ConstraintViolation if violated, None otherwise
        """
        if constraint.constraint_type == ConstraintType.EXECUTION_TIME:
            execution_time = action_data.get("execution_time", 0)
            max_time = constraint.parameters.get("max_seconds", 300)
            
            if execution_time > max_time:
                return self._create_violation(
                    constraint, agent_id, "high",
                    f"Execution time exceeded: {execution_time}s > {max_time}s",
                    execution_time, max_time, action_data
                )
                
        elif constraint.constraint_type == ConstraintType.ACTION_COUNT:
            # This would require tracking action counts per agent
            # For now, simulate based on action frequency
            recent_actions = action_data.get("recent_action_count", 0)
            max_actions = constraint.parameters.get("max_actions", 60)
            
            if recent_actions > max_actions:
                return self._create_violation(
                    constraint, agent_id, "medium", 
                    f"Action count exceeded: {recent_actions} > {max_actions}",
                    recent_actions, max_actions, action_data
                )
                
        elif constraint.constraint_type == ConstraintType.RESOURCE_USAGE:
            cpu_usage = action_data.get("cpu_percent", 0)
            memory_usage = action_data.get("memory_mb", 0)
            max_cpu = constraint.parameters.get("max_cpu_percent", 80)
            max_memory = constraint.parameters.get("max_memory_mb", 512)
            
            if cpu_usage > max_cpu:
                return self._create_violation(
                    constraint, agent_id, "high",
                    f"CPU usage exceeded: {cpu_usage}% > {max_cpu}%",
                    cpu_usage, max_cpu, action_data
                )
            elif memory_usage > max_memory:
                return self._create_violation(
                    constraint, agent_id, "high",
                    f"Memory usage exceeded: {memory_usage}MB > {max_memory}MB", 
                    memory_usage, max_memory, action_data
                )
                
        return None
        
    def _create_violation(self, constraint: Constraint, agent_id: str, severity: str,
                         description: str, measured_value: Any, threshold_value: Any,
                         evidence: Dict[str, Any]) -> ConstraintViolation:
        """Create constraint violation record.
        
        Args:
            constraint: Violated constraint
            agent_id: Agent ID
            severity: Violation severity
            description: Description of violation
            measured_value: Measured value
            threshold_value: Threshold value
            evidence: Evidence data
            
        Returns:
            ConstraintViolation object
        """
        violation_id = hashlib.md5(f"{constraint.constraint_id}_{agent_id}_{time.time()}".encode()).hexdigest()[:12]
        
        violation = ConstraintViolation(
            violation_id=violation_id,
            constraint_id=constraint.constraint_id,
            agent_id=agent_id,
            violation_type=constraint.constraint_type,
            severity=severity,
            description=description,
            measured_value=measured_value,
            threshold_value=threshold_value,
            evidence=evidence,
            automatic_response=self._determine_response(constraint, severity)
        )
        
        self.violations.append(violation)
        return violation
        
    def _enforce_constraint(self, constraint: Constraint, violation: ConstraintViolation):
        """Enforce constraint after violation.
        
        Args:
            constraint: Violated constraint
            violation: Violation details
        """
        if constraint.enforcement_level == "strict":
            if violation.severity in ["high", "critical"]:
                # Issue emergency stop for severe violations
                self.emergency_stop(
                    StopScope.SINGLE_AGENT,
                    violation.agent_id,
                    EmergencyStopReason.SAFETY_THRESHOLD,
                    f"Strict constraint violation: {violation.description}",
                    "constraint_enforcer",
                    authorization_level=2
                )
            else:
                logger.warning(f"Constraint violation (strict): {violation.description}")
                
        elif constraint.enforcement_level == "moderate":
            if violation.severity == "critical":
                logger.error(f"Critical constraint violation: {violation.description}")
            else:
                logger.warning(f"Constraint violation (moderate): {violation.description}")
                
        # Notify callbacks
        for callback in self._constraint_callbacks:
            try:
                callback(violation)
            except Exception as e:
                logger.error(f"Error in constraint callback: {e}")
                
    def _determine_response(self, constraint: Constraint, severity: str) -> str:
        """Determine automatic response to constraint violation.
        
        Args:
            constraint: Violated constraint
            severity: Violation severity
            
        Returns:
            Description of automatic response
        """
        if constraint.enforcement_level == "strict":
            if severity in ["high", "critical"]:
                return "Agent stopped due to strict constraint violation"
            else:
                return "Warning logged, constraint enforced"
        elif constraint.enforcement_level == "moderate":
            return "Warning logged, action throttled"
        else:
            return "Advisory notice logged"
            
    def _generate_command_id(self, scope: StopScope, target: str, reason: EmergencyStopReason) -> str:
        """Generate unique command ID.
        
        Args:
            scope: Stop scope
            target: Target identifier
            reason: Stop reason
            
        Returns:
            Unique command ID
        """
        timestamp = str(time.time())
        content = f"{scope.value}_{target}_{reason.value}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
        
    def _verify_authorization_code(self, level: int, code: str) -> bool:
        """Verify authorization code for emergency stop resolution.
        
        Args:
            level: Authorization level
            code: Provided authorization code
            
        Returns:
            True if code is valid
        """
        # In production, this would use proper cryptographic verification
        expected_codes = {
            1: "basic_override_123",
            2: "elevated_override_456",
            3: "emergency_override_789",
            4: "system_override_000"
        }
        
        return expected_codes.get(level) == code
        
    def _trim_history(self):
        """Trim stop command history to maximum size."""
        if len(self.stop_history) > self.max_history:
            self.stop_history = self.stop_history[-self.max_history:]

# Global emergency stop system
emergency_stop_system = EmergencyStopSystem()