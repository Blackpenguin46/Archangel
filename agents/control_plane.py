"""
Control Plane Implementation for Agent Decision-Making and Coordination

This module implements the control plane that handles agent decision-making,
coordination, and management separate from the data plane that manages
environment state and simulation execution.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from uuid import uuid4
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ControlPlaneStatus(Enum):
    """Status of control plane components"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


class DecisionType(Enum):
    """Types of decisions made by agents"""
    TACTICAL = "tactical"
    STRATEGIC = "strategic"
    COORDINATION = "coordination"
    EMERGENCY = "emergency"


@dataclass
class AgentDecision:
    """Represents a decision made by an agent"""
    decision_id: str
    agent_id: str
    decision_type: DecisionType
    action: str
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float
    timestamp: datetime
    dependencies: List[str]
    expected_outcome: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'decision_type': self.decision_type.value
        }


@dataclass
class CoordinationRequest:
    """Request for multi-agent coordination"""
    request_id: str
    initiator_agent_id: str
    target_agents: List[str]
    coordination_type: str
    payload: Dict[str, Any]
    priority: int
    timeout: timedelta
    timestamp: datetime


@dataclass
class ControlPlaneMetrics:
    """Metrics for control plane performance"""
    decisions_per_second: float
    average_decision_latency: float
    coordination_success_rate: float
    active_agents: int
    failed_decisions: int
    total_decisions: int
    uptime: timedelta


class ControlPlaneInterface(ABC):
    """Abstract interface for control plane components"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the control plane component"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the control plane component"""
        pass
    
    @abstractmethod
    def get_status(self) -> ControlPlaneStatus:
        """Get current status of the component"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        pass


class DecisionEngine(ControlPlaneInterface):
    """
    Core decision engine for agent reasoning and action selection
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.status = ControlPlaneStatus.INITIALIZING
        self.decision_history: List[AgentDecision] = []
        self.decision_callbacks: Dict[str, Callable] = {}
        self.metrics = {
            'decisions_made': 0,
            'average_latency': 0.0,
            'success_rate': 0.0
        }
        self._lock = threading.Lock()
        self._start_time = datetime.now()
    
    async def initialize(self) -> bool:
        """Initialize the decision engine"""
        try:
            logger.info(f"Initializing decision engine for agent {self.agent_id}")
            self.status = ControlPlaneStatus.ACTIVE
            return True
        except Exception as e:
            logger.error(f"Failed to initialize decision engine: {e}")
            self.status = ControlPlaneStatus.FAILED
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the decision engine"""
        try:
            logger.info(f"Shutting down decision engine for agent {self.agent_id}")
            self.status = ControlPlaneStatus.SHUTDOWN
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown decision engine: {e}")
            return False
    
    def get_status(self) -> ControlPlaneStatus:
        """Get current status"""
        return self.status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get decision engine metrics"""
        with self._lock:
            uptime = datetime.now() - self._start_time
            return {
                **self.metrics,
                'uptime_seconds': uptime.total_seconds(),
                'decisions_in_history': len(self.decision_history)
            }
    
    async def make_decision(self, 
                          decision_type: DecisionType,
                          context: Dict[str, Any],
                          constraints: List[str] = None) -> AgentDecision:
        """
        Make a decision based on context and constraints
        """
        start_time = time.time()
        
        try:
            decision_id = str(uuid4())
            
            # Simulate decision-making process
            action, parameters, reasoning, confidence = await self._process_decision(
                decision_type, context, constraints or []
            )
            
            decision = AgentDecision(
                decision_id=decision_id,
                agent_id=self.agent_id,
                decision_type=decision_type,
                action=action,
                parameters=parameters,
                reasoning=reasoning,
                confidence=confidence,
                timestamp=datetime.now(),
                dependencies=context.get('dependencies', []),
                expected_outcome=context.get('expected_outcome', 'Unknown')
            )
            
            # Store decision
            with self._lock:
                self.decision_history.append(decision)
                self.metrics['decisions_made'] += 1
                
                # Update average latency
                latency = time.time() - start_time
                current_avg = self.metrics['average_latency']
                count = self.metrics['decisions_made']
                self.metrics['average_latency'] = (current_avg * (count - 1) + latency) / count
            
            logger.info(f"Decision made: {decision.action} (confidence: {confidence})")
            return decision
            
        except Exception as e:
            logger.error(f"Failed to make decision: {e}")
            raise
    
    async def _process_decision(self, 
                              decision_type: DecisionType,
                              context: Dict[str, Any],
                              constraints: List[str]) -> tuple:
        """
        Process the decision-making logic
        """
        # Simulate decision processing time
        await asyncio.sleep(0.1)
        
        # Simple decision logic based on type
        if decision_type == DecisionType.TACTICAL:
            action = "execute_tactic"
            parameters = {"tactic": context.get("suggested_tactic", "reconnaissance")}
            reasoning = "Selected tactic based on current phase and objectives"
            confidence = 0.8
        elif decision_type == DecisionType.STRATEGIC:
            action = "update_strategy"
            parameters = {"strategy": context.get("strategy", "adaptive")}
            reasoning = "Strategic adjustment based on environment feedback"
            confidence = 0.7
        elif decision_type == DecisionType.COORDINATION:
            action = "coordinate_with_team"
            parameters = {"coordination_type": context.get("coordination_type", "intelligence_sharing")}
            reasoning = "Team coordination required for objective completion"
            confidence = 0.9
        else:  # EMERGENCY
            action = "emergency_response"
            parameters = {"response_type": context.get("emergency_type", "containment")}
            reasoning = "Emergency response triggered by threat detection"
            confidence = 1.0
        
        return action, parameters, reasoning, confidence
    
    def register_decision_callback(self, decision_type: str, callback: Callable):
        """Register callback for specific decision types"""
        self.decision_callbacks[decision_type] = callback
    
    def get_decision_history(self, limit: int = 100) -> List[AgentDecision]:
        """Get recent decision history"""
        with self._lock:
            return self.decision_history[-limit:]


class CoordinationManager(ControlPlaneInterface):
    """
    Manages multi-agent coordination and communication
    """
    
    def __init__(self):
        self.status = ControlPlaneStatus.INITIALIZING
        self.active_agents: Set[str] = set()
        self.coordination_requests: Dict[str, CoordinationRequest] = {}
        self.coordination_history: List[CoordinationRequest] = []
        self.metrics = {
            'coordination_requests': 0,
            'successful_coordinations': 0,
            'failed_coordinations': 0,
            'average_response_time': 0.0
        }
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=10)
    
    async def initialize(self) -> bool:
        """Initialize coordination manager"""
        try:
            logger.info("Initializing coordination manager")
            self.status = ControlPlaneStatus.ACTIVE
            return True
        except Exception as e:
            logger.error(f"Failed to initialize coordination manager: {e}")
            self.status = ControlPlaneStatus.FAILED
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown coordination manager"""
        try:
            logger.info("Shutting down coordination manager")
            self._executor.shutdown(wait=True)
            self.status = ControlPlaneStatus.SHUTDOWN
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown coordination manager: {e}")
            return False
    
    def get_status(self) -> ControlPlaneStatus:
        """Get current status"""
        return self.status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get coordination metrics"""
        with self._lock:
            return {
                **self.metrics,
                'active_agents': len(self.active_agents),
                'pending_requests': len(self.coordination_requests)
            }
    
    def register_agent(self, agent_id: str) -> bool:
        """Register an agent with the coordination manager"""
        try:
            with self._lock:
                self.active_agents.add(agent_id)
            logger.info(f"Agent {agent_id} registered for coordination")
            return True
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from coordination"""
        try:
            with self._lock:
                self.active_agents.discard(agent_id)
            logger.info(f"Agent {agent_id} unregistered from coordination")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def request_coordination(self, 
                                 initiator_agent_id: str,
                                 target_agents: List[str],
                                 coordination_type: str,
                                 payload: Dict[str, Any],
                                 priority: int = 5,
                                 timeout: timedelta = timedelta(seconds=30)) -> str:
        """
        Request coordination between agents
        """
        request_id = str(uuid4())
        
        request = CoordinationRequest(
            request_id=request_id,
            initiator_agent_id=initiator_agent_id,
            target_agents=target_agents,
            coordination_type=coordination_type,
            payload=payload,
            priority=priority,
            timeout=timeout,
            timestamp=datetime.now()
        )
        
        with self._lock:
            self.coordination_requests[request_id] = request
            self.metrics['coordination_requests'] += 1
        
        # Process coordination asynchronously
        asyncio.create_task(self._process_coordination_request(request))
        
        logger.info(f"Coordination request {request_id} created for {coordination_type}")
        return request_id
    
    async def _process_coordination_request(self, request: CoordinationRequest):
        """Process a coordination request"""
        start_time = time.time()
        
        try:
            # Simulate coordination processing
            await asyncio.sleep(0.2)
            
            # Check if target agents are available
            available_agents = []
            with self._lock:
                for agent_id in request.target_agents:
                    if agent_id in self.active_agents:
                        available_agents.append(agent_id)
            
            if len(available_agents) == len(request.target_agents):
                # All agents available - coordination successful
                with self._lock:
                    self.metrics['successful_coordinations'] += 1
                    if request.request_id in self.coordination_requests:
                        del self.coordination_requests[request.request_id]
                    self.coordination_history.append(request)
                
                logger.info(f"Coordination {request.request_id} completed successfully")
            else:
                # Some agents unavailable - coordination failed
                with self._lock:
                    self.metrics['failed_coordinations'] += 1
                    if request.request_id in self.coordination_requests:
                        del self.coordination_requests[request.request_id]
                
                logger.warning(f"Coordination {request.request_id} failed - agents unavailable")
            
            # Update average response time
            response_time = time.time() - start_time
            with self._lock:
                current_avg = self.metrics['average_response_time']
                total_requests = self.metrics['coordination_requests']
                self.metrics['average_response_time'] = (
                    (current_avg * (total_requests - 1) + response_time) / total_requests
                )
                
        except Exception as e:
            logger.error(f"Failed to process coordination request {request.request_id}: {e}")
            with self._lock:
                self.metrics['failed_coordinations'] += 1
                if request.request_id in self.coordination_requests:
                    del self.coordination_requests[request.request_id]
    
    def get_coordination_status(self, request_id: str) -> Optional[str]:
        """Get status of a coordination request"""
        with self._lock:
            if request_id in self.coordination_requests:
                return "pending"
            
            # Check if in history (completed)
            for request in self.coordination_history:
                if request.request_id == request_id:
                    return "completed"
            
            return "not_found"


class ControlPlaneOrchestrator:
    """
    Main orchestrator for the control plane
    """
    
    def __init__(self):
        self.status = ControlPlaneStatus.INITIALIZING
        self.decision_engines: Dict[str, DecisionEngine] = {}
        self.coordination_manager = CoordinationManager()
        self.metrics_collector = ControlPlaneMetricsCollector()
        self._lock = threading.Lock()
        self._start_time = datetime.now()
    
    async def initialize(self) -> bool:
        """Initialize the control plane orchestrator"""
        try:
            logger.info("Initializing control plane orchestrator")
            
            # Initialize coordination manager
            if not await self.coordination_manager.initialize():
                raise Exception("Failed to initialize coordination manager")
            
            # Initialize metrics collector
            if not await self.metrics_collector.initialize():
                raise Exception("Failed to initialize metrics collector")
            
            self.status = ControlPlaneStatus.ACTIVE
            logger.info("Control plane orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize control plane orchestrator: {e}")
            self.status = ControlPlaneStatus.FAILED
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the control plane orchestrator"""
        try:
            logger.info("Shutting down control plane orchestrator")
            
            # Shutdown all decision engines
            for engine in self.decision_engines.values():
                await engine.shutdown()
            
            # Shutdown coordination manager
            await self.coordination_manager.shutdown()
            
            # Shutdown metrics collector
            await self.metrics_collector.shutdown()
            
            self.status = ControlPlaneStatus.SHUTDOWN
            logger.info("Control plane orchestrator shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown control plane orchestrator: {e}")
            return False
    
    def register_agent(self, agent_id: str) -> DecisionEngine:
        """Register a new agent and create its decision engine"""
        with self._lock:
            if agent_id not in self.decision_engines:
                engine = DecisionEngine(agent_id)
                self.decision_engines[agent_id] = engine
                
                # Register with coordination manager
                self.coordination_manager.register_agent(agent_id)
                
                logger.info(f"Agent {agent_id} registered with control plane")
                return engine
            else:
                return self.decision_engines[agent_id]
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the control plane"""
        try:
            with self._lock:
                if agent_id in self.decision_engines:
                    engine = self.decision_engines[agent_id]
                    asyncio.create_task(engine.shutdown())
                    del self.decision_engines[agent_id]
                    
                    # Unregister from coordination manager
                    self.coordination_manager.unregister_agent(agent_id)
                    
                    logger.info(f"Agent {agent_id} unregistered from control plane")
                    return True
                else:
                    logger.warning(f"Agent {agent_id} not found for unregistration")
                    return False
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    def get_agent_decision_engine(self, agent_id: str) -> Optional[DecisionEngine]:
        """Get decision engine for a specific agent"""
        with self._lock:
            return self.decision_engines.get(agent_id)
    
    def get_coordination_manager(self) -> CoordinationManager:
        """Get the coordination manager"""
        return self.coordination_manager
    
    def get_overall_metrics(self) -> ControlPlaneMetrics:
        """Get overall control plane metrics"""
        with self._lock:
            total_decisions = sum(
                engine.metrics['decisions_made'] 
                for engine in self.decision_engines.values()
            )
            
            avg_latency = 0.0
            if self.decision_engines:
                avg_latency = sum(
                    engine.metrics['average_latency'] 
                    for engine in self.decision_engines.values()
                ) / len(self.decision_engines)
            
            coord_metrics = self.coordination_manager.get_metrics()
            coord_success_rate = 0.0
            if coord_metrics['coordination_requests'] > 0:
                coord_success_rate = (
                    coord_metrics['successful_coordinations'] / 
                    coord_metrics['coordination_requests']
                )
            
            uptime = datetime.now() - self._start_time
            
            return ControlPlaneMetrics(
                decisions_per_second=total_decisions / max(uptime.total_seconds(), 1),
                average_decision_latency=avg_latency,
                coordination_success_rate=coord_success_rate,
                active_agents=len(self.decision_engines),
                failed_decisions=0,  # TODO: Track failed decisions
                total_decisions=total_decisions,
                uptime=uptime
            )


class ControlPlaneMetricsCollector(ControlPlaneInterface):
    """
    Collects and aggregates metrics from control plane components
    """
    
    def __init__(self):
        self.status = ControlPlaneStatus.INITIALIZING
        self.metrics_history: List[Dict[str, Any]] = []
        self.collection_interval = 10  # seconds
        self._collection_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
    
    async def initialize(self) -> bool:
        """Initialize metrics collector"""
        try:
            logger.info("Initializing control plane metrics collector")
            self.status = ControlPlaneStatus.ACTIVE
            
            # Start metrics collection task
            self._collection_task = asyncio.create_task(self._collect_metrics_loop())
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize metrics collector: {e}")
            self.status = ControlPlaneStatus.FAILED
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown metrics collector"""
        try:
            logger.info("Shutting down metrics collector")
            
            if self._collection_task:
                self._collection_task.cancel()
                try:
                    await self._collection_task
                except asyncio.CancelledError:
                    pass
            
            self.status = ControlPlaneStatus.SHUTDOWN
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown metrics collector: {e}")
            return False
    
    def get_status(self) -> ControlPlaneStatus:
        """Get current status"""
        return self.status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        with self._lock:
            return {
                'metrics_collected': len(self.metrics_history),
                'collection_interval': self.collection_interval,
                'last_collection': self.metrics_history[-1]['timestamp'] if self.metrics_history else None
            }
    
    async def _collect_metrics_loop(self):
        """Continuous metrics collection loop"""
        while self.status == ControlPlaneStatus.ACTIVE:
            try:
                await asyncio.sleep(self.collection_interval)
                
                # Collect metrics (placeholder - would integrate with actual components)
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'control_plane_status': self.status.value,
                    'collection_interval': self.collection_interval
                }
                
                with self._lock:
                    self.metrics_history.append(metrics)
                    
                    # Keep only last 1000 entries
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(1)  # Brief pause before retry


# Global control plane instance
_control_plane_orchestrator: Optional[ControlPlaneOrchestrator] = None


async def get_control_plane() -> ControlPlaneOrchestrator:
    """Get or create the global control plane orchestrator"""
    global _control_plane_orchestrator
    
    if _control_plane_orchestrator is None:
        _control_plane_orchestrator = ControlPlaneOrchestrator()
        await _control_plane_orchestrator.initialize()
    
    return _control_plane_orchestrator


async def shutdown_control_plane():
    """Shutdown the global control plane orchestrator"""
    global _control_plane_orchestrator
    
    if _control_plane_orchestrator is not None:
        await _control_plane_orchestrator.shutdown()
        _control_plane_orchestrator = None