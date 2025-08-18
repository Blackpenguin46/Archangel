"""
Agent heartbeat monitoring and failure detection system.
"""

import time
import threading
import logging
from typing import Dict, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent state enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"

@dataclass
class HeartbeatConfig:
    """Configuration for heartbeat monitoring."""
    interval: float = 5.0  # Heartbeat interval in seconds
    timeout: float = 15.0  # Heartbeat timeout in seconds
    missed_threshold: int = 3  # Number of missed heartbeats before marking as failed
    recovery_threshold: int = 2  # Number of successful heartbeats to mark as recovered
    degraded_threshold: float = 10.0  # Response time threshold for degraded state
    enable_adaptive_timeout: bool = True  # Dynamically adjust timeout based on performance

@dataclass
class AgentHeartbeat:
    """Agent heartbeat tracking data."""
    agent_id: str
    last_heartbeat: float = field(default_factory=time.time)
    last_response_time: float = 0.0
    missed_count: int = 0
    recovery_count: int = 0
    state: AgentState = AgentState.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    adaptive_timeout: float = 15.0
    average_response_time: float = 0.0
    response_time_history: list = field(default_factory=lambda: [])

class HeartbeatMonitor:
    """
    Comprehensive heartbeat monitoring system for Archangel agents.
    
    Features:
    - Real-time heartbeat tracking with configurable intervals
    - Adaptive timeout adjustment based on agent performance
    - Multi-tier failure detection (healthy → degraded → critical → failed)
    - Recovery tracking and validation
    - Performance-based state transitions
    - Callback notifications for state changes
    - Comprehensive failure analytics
    """
    
    def __init__(self, config: Optional[HeartbeatConfig] = None):
        """Initialize the heartbeat monitor.
        
        Args:
            config: HeartbeatConfig object with monitoring parameters
        """
        self.config = config or HeartbeatConfig()
        self.agents: Dict[str, AgentHeartbeat] = {}
        self._lock = threading.RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._state_change_callbacks: list[Callable[[str, AgentState, AgentState], None]] = []
        self._failure_callbacks: list[Callable[[str, Dict[str, Any]], None]] = []
        self._recovery_callbacks: list[Callable[[str, Dict[str, Any]], None]] = []
        
        # Performance tracking
        self._performance_history: Dict[str, list] = {}
        self._max_history_length = 100
        
    def start_monitoring(self):
        """Start the heartbeat monitoring loop."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Heartbeat monitoring already running")
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Heartbeat monitoring started")
        
    def stop_monitoring(self):
        """Stop the heartbeat monitoring loop."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Heartbeat monitoring stopped")
        
    def register_agent(self, agent_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Register an agent for heartbeat monitoring.
        
        Args:
            agent_id: Unique identifier for the agent
            metadata: Optional metadata about the agent
        """
        with self._lock:
            heartbeat = AgentHeartbeat(
                agent_id=agent_id,
                metadata=metadata or {}
            )
            self.agents[agent_id] = heartbeat
            self._performance_history[agent_id] = []
            logger.info(f"Registered agent {agent_id} for heartbeat monitoring")
            
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from heartbeat monitoring.
        
        Args:
            agent_id: Agent ID to unregister
        """
        with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                if agent_id in self._performance_history:
                    del self._performance_history[agent_id]
                logger.info(f"Unregistered agent {agent_id} from heartbeat monitoring")
                
    def record_heartbeat(self, agent_id: str, response_time: Optional[float] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """Record a heartbeat from an agent.
        
        Args:
            agent_id: ID of the agent sending heartbeat
            response_time: Time taken for the heartbeat response
            metadata: Additional metadata about the agent state
        """
        current_time = time.time()
        
        with self._lock:
            if agent_id not in self.agents:
                logger.warning(f"Heartbeat received from unregistered agent {agent_id}")
                return
                
            heartbeat = self.agents[agent_id]
            old_state = heartbeat.state
            
            # Update heartbeat data
            heartbeat.last_heartbeat = current_time
            if response_time is not None:
                heartbeat.last_response_time = response_time
                self._update_response_time_tracking(agent_id, response_time)
                
            if metadata:
                heartbeat.metadata.update(metadata)
                
            # Reset missed count on successful heartbeat
            heartbeat.missed_count = 0
            
            # Update agent state based on performance
            new_state = self._determine_agent_state(heartbeat)
            
            if new_state != old_state:
                heartbeat.state = new_state
                self._notify_state_change(agent_id, old_state, new_state)
                
                # Handle recovery tracking
                if new_state == AgentState.HEALTHY and old_state in [AgentState.DEGRADED, AgentState.CRITICAL, AgentState.FAILED]:
                    heartbeat.recovery_count += 1
                    self._notify_recovery(agent_id, {
                        "previous_state": old_state.value,
                        "recovery_time": current_time,
                        "recovery_count": heartbeat.recovery_count
                    })
                    
    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get the current state of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Current AgentState or None if not found
        """
        with self._lock:
            heartbeat = self.agents.get(agent_id)
            return heartbeat.state if heartbeat else None
            
    def get_agent_heartbeat(self, agent_id: str) -> Optional[AgentHeartbeat]:
        """Get complete heartbeat data for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            AgentHeartbeat object or None if not found
        """
        with self._lock:
            return self.agents.get(agent_id)
            
    def get_all_agents(self) -> Dict[str, AgentHeartbeat]:
        """Get heartbeat data for all agents.
        
        Returns:
            Dictionary mapping agent IDs to AgentHeartbeat objects
        """
        with self._lock:
            return self.agents.copy()
            
    def get_failed_agents(self) -> Dict[str, AgentHeartbeat]:
        """Get all agents currently in failed state.
        
        Returns:
            Dictionary of failed agents
        """
        with self._lock:
            return {
                agent_id: heartbeat
                for agent_id, heartbeat in self.agents.items()
                if heartbeat.state == AgentState.FAILED
            }
            
    def get_degraded_agents(self) -> Dict[str, AgentHeartbeat]:
        """Get all agents currently in degraded state.
        
        Returns:
            Dictionary of degraded agents
        """
        with self._lock:
            return {
                agent_id: heartbeat
                for agent_id, heartbeat in self.agents.items()
                if heartbeat.state == AgentState.DEGRADED
            }
            
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary statistics.
        
        Returns:
            Dictionary with monitoring statistics
        """
        with self._lock:
            total_agents = len(self.agents)
            if total_agents == 0:
                return {
                    "total_agents": 0,
                    "healthy": 0,
                    "degraded": 0,
                    "critical": 0,
                    "failed": 0,
                    "recovering": 0,
                    "unknown": 0,
                    "average_response_time": 0.0
                }
                
            state_counts = {state.value: 0 for state in AgentState}
            total_response_time = 0.0
            response_count = 0
            
            for heartbeat in self.agents.values():
                state_counts[heartbeat.state.value] += 1
                if heartbeat.average_response_time > 0:
                    total_response_time += heartbeat.average_response_time
                    response_count += 1
                    
            avg_response_time = total_response_time / response_count if response_count > 0 else 0.0
            
            return {
                "total_agents": total_agents,
                "healthy": state_counts[AgentState.HEALTHY.value],
                "degraded": state_counts[AgentState.DEGRADED.value],
                "critical": state_counts[AgentState.CRITICAL.value],
                "failed": state_counts[AgentState.FAILED.value],
                "recovering": state_counts[AgentState.RECOVERING.value],
                "unknown": state_counts[AgentState.UNKNOWN.value],
                "average_response_time": avg_response_time,
                "last_updated": time.time()
            }
            
    def add_state_change_callback(self, callback: Callable[[str, AgentState, AgentState], None]):
        """Add callback for agent state changes.
        
        Args:
            callback: Function to call when agent state changes
        """
        self._state_change_callbacks.append(callback)
        
    def add_failure_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for agent failures.
        
        Args:
            callback: Function to call when agent fails
        """
        self._failure_callbacks.append(callback)
        
    def add_recovery_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for agent recovery.
        
        Args:
            callback: Function to call when agent recovers
        """
        self._recovery_callbacks.append(callback)
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._check_agent_heartbeats()
                time.sleep(self.config.interval)
            except Exception as e:
                logger.error(f"Error in heartbeat monitoring loop: {e}")
                time.sleep(1.0)  # Back off on errors
                
    def _check_agent_heartbeats(self):
        """Check all agent heartbeats for timeouts."""
        current_time = time.time()
        
        with self._lock:
            for agent_id, heartbeat in self.agents.items():
                time_since_heartbeat = current_time - heartbeat.last_heartbeat
                timeout_threshold = heartbeat.adaptive_timeout if self.config.enable_adaptive_timeout else self.config.timeout
                
                if time_since_heartbeat > timeout_threshold:
                    old_state = heartbeat.state
                    heartbeat.missed_count += 1
                    
                    # Determine new state based on missed heartbeats
                    if heartbeat.missed_count >= self.config.missed_threshold:
                        new_state = AgentState.FAILED
                    elif heartbeat.missed_count >= 2:
                        new_state = AgentState.CRITICAL
                    elif heartbeat.missed_count >= 1:
                        new_state = AgentState.DEGRADED
                    else:
                        new_state = old_state
                        
                    if new_state != old_state:
                        heartbeat.state = new_state
                        self._notify_state_change(agent_id, old_state, new_state)
                        
                        if new_state == AgentState.FAILED:
                            self._notify_failure(agent_id, {
                                "failure_reason": "heartbeat_timeout",
                                "missed_count": heartbeat.missed_count,
                                "time_since_heartbeat": time_since_heartbeat,
                                "failure_time": current_time
                            })
                            
    def _determine_agent_state(self, heartbeat: AgentHeartbeat) -> AgentState:
        """Determine agent state based on current metrics.
        
        Args:
            heartbeat: AgentHeartbeat object to analyze
            
        Returns:
            Appropriate AgentState
        """
        # Check response time for degradation
        if heartbeat.average_response_time > self.config.degraded_threshold:
            return AgentState.DEGRADED
            
        # Check if recovering
        if heartbeat.state in [AgentState.FAILED, AgentState.CRITICAL] and heartbeat.recovery_count < self.config.recovery_threshold:
            return AgentState.RECOVERING
            
        # Default to healthy if no issues
        return AgentState.HEALTHY
        
    def _update_response_time_tracking(self, agent_id: str, response_time: float):
        """Update response time tracking for an agent.
        
        Args:
            agent_id: ID of the agent
            response_time: Latest response time
        """
        heartbeat = self.agents[agent_id]
        
        # Update response time history
        heartbeat.response_time_history.append(response_time)
        if len(heartbeat.response_time_history) > 10:  # Keep last 10 measurements
            heartbeat.response_time_history.pop(0)
            
        # Calculate rolling average
        heartbeat.average_response_time = sum(heartbeat.response_time_history) / len(heartbeat.response_time_history)
        
        # Update adaptive timeout if enabled
        if self.config.enable_adaptive_timeout:
            # Set timeout to 3x average response time, with min/max bounds
            adaptive_timeout = max(
                self.config.timeout,
                min(heartbeat.average_response_time * 3.0, self.config.timeout * 3.0)
            )
            heartbeat.adaptive_timeout = adaptive_timeout
            
        # Track performance history
        if agent_id in self._performance_history:
            self._performance_history[agent_id].append({
                "timestamp": time.time(),
                "response_time": response_time,
                "average_response_time": heartbeat.average_response_time
            })
            
            # Trim history
            if len(self._performance_history[agent_id]) > self._max_history_length:
                self._performance_history[agent_id].pop(0)
                
    def _notify_state_change(self, agent_id: str, old_state: AgentState, new_state: AgentState):
        """Notify callbacks of agent state changes.
        
        Args:
            agent_id: ID of the agent
            old_state: Previous agent state
            new_state: New agent state
        """
        logger.info(f"Agent {agent_id} state changed: {old_state.value} → {new_state.value}")
        
        for callback in self._state_change_callbacks:
            try:
                callback(agent_id, old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
                
    def _notify_failure(self, agent_id: str, failure_info: Dict[str, Any]):
        """Notify callbacks of agent failure.
        
        Args:
            agent_id: ID of the failed agent
            failure_info: Information about the failure
        """
        logger.error(f"Agent {agent_id} failed: {failure_info}")
        
        for callback in self._failure_callbacks:
            try:
                callback(agent_id, failure_info)
            except Exception as e:
                logger.error(f"Error in failure callback: {e}")
                
    def _notify_recovery(self, agent_id: str, recovery_info: Dict[str, Any]):
        """Notify callbacks of agent recovery.
        
        Args:
            agent_id: ID of the recovered agent
            recovery_info: Information about the recovery
        """
        logger.info(f"Agent {agent_id} recovered: {recovery_info}")
        
        for callback in self._recovery_callbacks:
            try:
                callback(agent_id, recovery_info)
            except Exception as e:
                logger.error(f"Error in recovery callback: {e}")

# Global heartbeat monitor instance
heartbeat_monitor = HeartbeatMonitor()