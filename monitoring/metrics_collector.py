"""
Prometheus metrics collection system for Archangel agents.
"""

import time
import threading
import psutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, start_http_server
from prometheus_client.core import CollectorRegistry
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentMetrics:
    """Container for agent-specific metrics."""
    agent_id: str
    agent_type: str
    team: str
    status: str = "active"
    decisions_total: int = 0
    decisions_success: int = 0
    decisions_failed: int = 0
    response_time: float = 0.0
    memory_usage: int = 0
    communication_failures: int = 0
    last_activity: float = field(default_factory=time.time)
    coordination_score: float = 1.0

class MetricsCollector:
    """
    Prometheus metrics collector for Archangel system components.
    
    Collects and exposes metrics for:
    - Agent performance and health
    - System resource usage
    - Communication patterns
    - Decision-making effectiveness
    - Team coordination
    """
    
    def __init__(self, port: int = 8888, registry: Optional[CollectorRegistry] = None):
        """Initialize the metrics collector.
        
        Args:
            port: Port to serve Prometheus metrics on
            registry: Optional custom registry (defaults to default registry)
        """
        self.port = port
        self.registry = registry or CollectorRegistry()
        self.agents: Dict[str, AgentMetrics] = {}
        self._lock = threading.RLock()
        self._server_thread = None
        self._collection_interval = 15.0  # seconds
        self._running = False
        
        # Initialize Prometheus metrics
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize all Prometheus metrics."""
        
        # Agent status metrics
        self.agent_status = Gauge(
            'archangel_agent_status',
            'Agent status (1=active, 0=inactive)',
            ['agent_id', 'agent_type', 'team'],
            registry=self.registry
        )
        
        # Decision metrics
        self.decisions_total = Counter(
            'archangel_agent_decisions_total',
            'Total number of agent decisions',
            ['agent_id', 'agent_type', 'team', 'decision_type'],
            registry=self.registry
        )
        
        self.decisions_success_total = Counter(
            'archangel_agent_decisions_success_total',
            'Total successful agent decisions',
            ['agent_id', 'agent_type', 'team'],
            registry=self.registry
        )
        
        self.decisions_failed_total = Counter(
            'archangel_agent_decisions_failed_total', 
            'Total failed agent decisions',
            ['agent_id', 'agent_type', 'team'],
            registry=self.registry
        )
        
        # Performance metrics
        self.response_time_seconds = Histogram(
            'archangel_agent_response_time_seconds',
            'Agent response time in seconds',
            ['agent_id', 'agent_type', 'team'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'archangel_agent_memory_usage_bytes',
            'Agent memory usage in bytes',
            ['agent_id', 'agent_type', 'team'],
            registry=self.registry
        )
        
        # Communication metrics
        self.communication_failures_total = Counter(
            'archangel_agent_communication_failures_total',
            'Total agent communication failures',
            ['agent_id', 'agent_type', 'team', 'failure_type'],
            registry=self.registry
        )
        
        # Team coordination metrics
        self.team_coordination_score = Gauge(
            'archangel_team_coordination_score',
            'Team coordination effectiveness score (0-1)',
            ['team'],
            registry=self.registry
        )
        
        # Game loop metrics
        self.game_loop_duration_seconds = Histogram(
            'archangel_game_loop_duration_seconds',
            'Game loop execution time in seconds',
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
            registry=self.registry
        )
        
        # Scoring engine metrics
        self.scoring_calculation_duration_seconds = Histogram(
            'archangel_scoring_calculation_duration_seconds',
            'Scoring calculation time in seconds',
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        # Vector store metrics
        self.vector_store_query_duration_seconds = Histogram(
            'archangel_vector_store_query_duration_seconds',
            'Vector store query time in seconds',
            ['operation'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        # Red Team specific metrics
        self.red_team_actions_total = Counter(
            'archangel_red_team_actions_total',
            'Total Red Team actions',
            ['agent_id', 'action_type', 'target'],
            registry=self.registry
        )
        
        self.red_team_actions_success_total = Counter(
            'archangel_red_team_actions_success_total',
            'Successful Red Team actions',
            ['agent_id', 'action_type', 'target'],
            registry=self.registry
        )
        
        # Blue Team specific metrics
        self.blue_team_detections_total = Counter(
            'archangel_blue_team_detections_total',
            'Total Blue Team threat detections',
            ['agent_id', 'detection_type', 'severity'],
            registry=self.registry
        )
        
        self.blue_team_response_time_seconds = Histogram(
            'archangel_blue_team_response_time_seconds',
            'Blue Team response time to threats',
            ['agent_id', 'threat_type'],
            buckets=[1.0, 5.0, 15.0, 30.0, 60.0, 300.0],
            registry=self.registry
        )
        
        # System info
        self.system_info = Info(
            'archangel_system_info',
            'System information',
            registry=self.registry
        )
        
    def start_server(self):
        """Start the Prometheus metrics HTTP server."""
        if self._server_thread and self._server_thread.is_alive():
            logger.warning(f"Metrics server already running on port {self.port}")
            return
            
        try:
            start_http_server(self.port, registry=self.registry)
            self._running = True
            
            # Start collection thread
            self._server_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self._server_thread.start()
            
            logger.info(f"Metrics server started on port {self.port}")
            
            # Update system info
            self.system_info.info({
                'version': '1.0.0',
                'python_version': f"{psutil.PYTHON_VERSION}",
                'platform': psutil.os.name,
                'architecture': psutil.platform.platform()
            })
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
            
    def stop_server(self):
        """Stop the metrics collection."""
        self._running = False
        if self._server_thread:
            self._server_thread.join(timeout=5.0)
        logger.info("Metrics collection stopped")
        
    def register_agent(self, agent_id: str, agent_type: str, team: str):
        """Register a new agent for metrics collection.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (e.g., 'recon', 'exploit', 'soc_analyst')
            team: Team the agent belongs to ('red' or 'blue')
        """
        with self._lock:
            metrics = AgentMetrics(
                agent_id=agent_id,
                agent_type=agent_type,
                team=team
            )
            self.agents[agent_id] = metrics
            
            # Initialize metrics for this agent
            self.agent_status.labels(
                agent_id=agent_id, 
                agent_type=agent_type, 
                team=team
            ).set(1)
            
        logger.info(f"Registered agent {agent_id} ({agent_type}, {team} team)")
        
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from metrics collection.
        
        Args:
            agent_id: ID of agent to unregister
        """
        with self._lock:
            if agent_id in self.agents:
                metrics = self.agents[agent_id]
                
                # Set status to inactive
                self.agent_status.labels(
                    agent_id=agent_id,
                    agent_type=metrics.agent_type,
                    team=metrics.team
                ).set(0)
                
                del self.agents[agent_id]
                logger.info(f"Unregistered agent {agent_id}")
                
    def record_decision(self, agent_id: str, decision_type: str, success: bool, 
                       response_time: float):
        """Record an agent decision for metrics.
        
        Args:
            agent_id: ID of the agent making the decision
            decision_type: Type of decision made
            success: Whether the decision was successful
            response_time: Time taken to make the decision (seconds)
        """
        with self._lock:
            if agent_id not in self.agents:
                logger.warning(f"Decision recorded for unregistered agent {agent_id}")
                return
                
            metrics = self.agents[agent_id]
            
            # Update counters
            self.decisions_total.labels(
                agent_id=agent_id,
                agent_type=metrics.agent_type,
                team=metrics.team,
                decision_type=decision_type
            ).inc()
            
            if success:
                self.decisions_success_total.labels(
                    agent_id=agent_id,
                    agent_type=metrics.agent_type,
                    team=metrics.team
                ).inc()
                metrics.decisions_success += 1
            else:
                self.decisions_failed_total.labels(
                    agent_id=agent_id,
                    agent_type=metrics.agent_type,
                    team=metrics.team
                ).inc()
                metrics.decisions_failed += 1
                
            # Update response time
            self.response_time_seconds.labels(
                agent_id=agent_id,
                agent_type=metrics.agent_type,
                team=metrics.team
            ).observe(response_time)
            
            metrics.decisions_total += 1
            metrics.response_time = response_time
            metrics.last_activity = time.time()
            
    def record_communication_failure(self, agent_id: str, failure_type: str):
        """Record a communication failure for an agent.
        
        Args:
            agent_id: ID of the agent with communication failure
            failure_type: Type of communication failure
        """
        with self._lock:
            if agent_id not in self.agents:
                logger.warning(f"Communication failure recorded for unregistered agent {agent_id}")
                return
                
            metrics = self.agents[agent_id]
            
            self.communication_failures_total.labels(
                agent_id=agent_id,
                agent_type=metrics.agent_type,
                team=metrics.team,
                failure_type=failure_type
            ).inc()
            
            metrics.communication_failures += 1
            
    def update_team_coordination(self, team: str, score: float):
        """Update team coordination score.
        
        Args:
            team: Team name ('red' or 'blue')
            score: Coordination score (0.0-1.0)
        """
        self.team_coordination_score.labels(team=team).set(score)
        
        # Update agent metrics
        with self._lock:
            for agent_id, metrics in self.agents.items():
                if metrics.team == team:
                    metrics.coordination_score = score
                    
    def record_red_team_action(self, agent_id: str, action_type: str, 
                              target: str, success: bool):
        """Record a Red Team action.
        
        Args:
            agent_id: ID of the Red Team agent
            action_type: Type of action (e.g., 'exploit', 'recon', 'persist')
            target: Target of the action
            success: Whether the action was successful
        """
        self.red_team_actions_total.labels(
            agent_id=agent_id,
            action_type=action_type,
            target=target
        ).inc()
        
        if success:
            self.red_team_actions_success_total.labels(
                agent_id=agent_id,
                action_type=action_type,
                target=target
            ).inc()
            
    def record_blue_team_detection(self, agent_id: str, detection_type: str, 
                                  severity: str, response_time: float):
        """Record a Blue Team threat detection.
        
        Args:
            agent_id: ID of the Blue Team agent
            detection_type: Type of threat detected
            severity: Severity level of the threat
            response_time: Time taken to respond to the threat
        """
        self.blue_team_detections_total.labels(
            agent_id=agent_id,
            detection_type=detection_type,
            severity=severity
        ).inc()
        
        self.blue_team_response_time_seconds.labels(
            agent_id=agent_id,
            threat_type=detection_type
        ).observe(response_time)
        
    def record_game_loop_duration(self, duration: float):
        """Record game loop execution time.
        
        Args:
            duration: Execution time in seconds
        """
        self.game_loop_duration_seconds.observe(duration)
        
    def record_scoring_duration(self, duration: float):
        """Record scoring calculation time.
        
        Args:
            duration: Calculation time in seconds
        """
        self.scoring_calculation_duration_seconds.observe(duration)
        
    def record_vector_store_query(self, operation: str, duration: float):
        """Record vector store query time.
        
        Args:
            operation: Type of operation (e.g., 'search', 'insert', 'update')
            duration: Query time in seconds
        """
        self.vector_store_query_duration_seconds.labels(
            operation=operation
        ).observe(duration)
        
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            AgentMetrics object or None if not found
        """
        with self._lock:
            return self.agents.get(agent_id)
            
    def get_all_agents(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all registered agents.
        
        Returns:
            Dictionary mapping agent IDs to AgentMetrics
        """
        with self._lock:
            return self.agents.copy()
            
    def _collection_loop(self):
        """Background thread for periodic metrics collection."""
        while self._running:
            try:
                self._collect_system_metrics()
                time.sleep(self._collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(5.0)  # Back off on errors
                
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        with self._lock:
            for agent_id, metrics in self.agents.items():
                # Update memory usage (simulated for now)
                try:
                    # In a real implementation, this would get actual process memory
                    memory_usage = psutil.virtual_memory().used // len(self.agents) if self.agents else 0
                    metrics.memory_usage = memory_usage
                    
                    self.memory_usage_bytes.labels(
                        agent_id=agent_id,
                        agent_type=metrics.agent_type,
                        team=metrics.team
                    ).set(memory_usage)
                    
                    # Check if agent is still active (hasn't reported in 60 seconds)
                    if time.time() - metrics.last_activity > 60:
                        metrics.status = "inactive"
                        self.agent_status.labels(
                            agent_id=agent_id,
                            agent_type=metrics.agent_type,
                            team=metrics.team
                        ).set(0)
                    else:
                        self.agent_status.labels(
                            agent_id=agent_id,
                            agent_type=metrics.agent_type,
                            team=metrics.team
                        ).set(1)
                        
                except Exception as e:
                    logger.error(f"Error collecting metrics for agent {agent_id}: {e}")

# Global metrics collector instance
metrics_collector = MetricsCollector()