#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Base Agent
Foundation class for all autonomous agents in the system
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

class Team(Enum):
    """Agent team affiliation"""
    RED = "red"
    BLUE = "blue"
    NEUTRAL = "neutral"

class AgentStatus(Enum):
    """Agent operational status"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class Role(Enum):
    """Agent role specialization"""
    # Red Team Roles
    RECON = "recon"
    EXPLOIT = "exploit"
    PERSISTENCE = "persistence"
    EXFILTRATION = "exfiltration"
    
    # Blue Team Roles
    SOC_ANALYST = "soc_analyst"
    FIREWALL_CONFIG = "firewall_config"
    SIEM_INTEGRATOR = "siem_integrator"
    COMPLIANCE_AUDITOR = "compliance_auditor"
    INCIDENT_RESPONSE = "incident_response"
    THREAT_HUNTER = "threat_hunter"

@dataclass
class AgentConfig:
    """Configuration for agent initialization"""
    agent_id: str
    team: Team
    role: Role
    name: str
    description: str
    llm_model: str = "gpt-4-turbo"
    max_memory_size: int = 1000
    decision_timeout: float = 30.0
    tools: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)

@dataclass
class EnvironmentState:
    """Current state of the environment as perceived by agent"""
    timestamp: datetime
    network_topology: Dict[str, Any]
    active_services: List[Dict[str, Any]]
    security_alerts: List[Dict[str, Any]]
    system_logs: List[Dict[str, Any]]
    agent_positions: Dict[str, Dict[str, Any]]
    threat_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningResult:
    """Result of agent reasoning process"""
    situation_assessment: str
    threat_analysis: str
    opportunity_identification: List[str]
    risk_assessment: Dict[str, float]
    recommended_actions: List[str]
    confidence_score: float
    reasoning_chain: List[str]
    alternatives_considered: List[str]

@dataclass
class Action:
    """Individual action that can be taken by an agent"""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    timestamp: datetime
    target: Optional[str] = None
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActionPlan:
    """Planned actions for agent execution"""
    primary_action: str
    action_type: str
    target: Optional[str]
    parameters: Dict[str, Any]
    expected_outcome: str
    success_criteria: List[str]
    fallback_actions: List[str]
    estimated_duration: float
    risk_level: str

@dataclass
class ActionResult:
    """Result of executed action"""
    action_id: str
    action_type: str
    success: bool
    outcome: str
    data: Dict[str, Any]
    duration: float
    errors: List[str]
    side_effects: List[str]
    confidence: float
    timestamp: datetime

@dataclass
class Experience:
    """Agent experience record for learning"""
    experience_id: str
    agent_id: str
    timestamp: datetime
    action_taken: Optional[Action] = None
    success: bool = False
    reasoning: str = ""
    outcome: str = ""
    context: Optional[EnvironmentState] = None
    lessons_learned: List[str] = field(default_factory=list)
    mitre_attack_mapping: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

class BaseAgent(ABC):
    """
    Base class for all autonomous agents in the Archangel system.
    
    Provides core functionality for:
    - LLM-powered reasoning
    - Memory management
    - Communication with other agents
    - Action planning and execution
    - Learning from experiences
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.team = config.team
        self.role = config.role
        self.name = config.name
        
        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.current_objective = None
        self.active_plan = None
        self.last_action_time = None
        
        # Memory and learning
        self.short_term_memory = []
        self.experiences = []
        self.performance_metrics = {
            'actions_taken': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'learning_iterations': 0
        }
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.team_members = []
        
        # Tools and capabilities
        self.available_tools = config.tools
        self.constraints = config.constraints
        self.objectives = config.objectives
        
        # Logging
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        
        # Reasoning system components (initialized in _initialize_llm)
        self.reasoning_system = None
        self.world_state = None
        self.last_decision = None
        
    async def initialize(self) -> None:
        """Initialize the agent and prepare for operation"""
        try:
            self.logger.info(f"Initializing agent {self.name} ({self.agent_id})")
            
            # Initialize LLM interface
            await self._initialize_llm()
            
            # Initialize memory systems
            await self._initialize_memory()
            
            # Initialize tools
            await self._initialize_tools()
            
            # Set status to active
            self.status = AgentStatus.ACTIVE
            self.logger.info(f"Agent {self.name} initialized successfully")
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Failed to initialize agent {self.name}: {e}")
            raise
    
    async def _initialize_llm(self) -> None:
        """Initialize LLM interface for reasoning"""
        from .llm_interface import LLMConfig, LLMProvider
        from .reasoning_integration import IntegratedReasoningSystem, ReasoningContext
        from .planning import WorldState
        
        # Initialize LLM configuration
        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model=self.config.llm_model,
            api_key=None,  # Will be set from environment
            max_tokens=2048,
            temperature=0.7
        )
        
        # Initialize integrated reasoning system
        self.reasoning_system = IntegratedReasoningSystem(llm_config)
        
        # Initialize world state for planning
        self.world_state = self.reasoning_system.planning_engine.get_world_state_template(
            self.team.value
        )
        
        self.logger.debug("LLM interface and reasoning system initialized")
    
    async def _initialize_memory(self) -> None:
        """Initialize memory systems"""
        # This will be implemented with vector database integration
        self.logger.debug("Memory systems initialized")
    
    async def _initialize_tools(self) -> None:
        """Initialize available tools"""
        # This will be implemented with actual tool integration
        self.logger.debug(f"Tools initialized: {self.available_tools}")
    
    async def perceive_environment(self) -> EnvironmentState:
        """
        Perceive and analyze the current environment state
        
        Returns:
            EnvironmentState: Current state of the environment
        """
        try:
            # This will be implemented with actual environment sensing
            current_state = EnvironmentState(
                timestamp=datetime.now(),
                network_topology={},
                active_services=[],
                security_alerts=[],
                system_logs=[],
                agent_positions={},
                threat_level="low"
            )
            
            self.logger.debug("Environment perception complete")
            return current_state
            
        except Exception as e:
            self.logger.error(f"Failed to perceive environment: {e}")
            raise
    
    async def reason_about_situation(self, state: EnvironmentState) -> ReasoningResult:
        """
        Use LLM reasoning to analyze the situation and determine appropriate actions
        
        Args:
            state: Current environment state
            
        Returns:
            ReasoningResult: Analysis and reasoning about the situation
        """
        from .reasoning_integration import ReasoningContext
        
        # Create reasoning context
        context = ReasoningContext(
            agent_id=self.agent_id,
            team=self.team.value,
            role=self.role.value,
            environment_state={
                "network_topology": state.network_topology,
                "active_services": state.active_services,
                "security_alerts": state.security_alerts,
                "system_logs": state.system_logs,
                "agent_positions": state.agent_positions,
                "threat_level": state.threat_level
            },
            agent_memory={"experiences": self.experiences[-10:]},  # Last 10 experiences
            available_tools=self.available_tools,
            constraints=self.constraints,
            objectives=self.objectives,
            world_state=self.world_state
        )
        
        # Make integrated decision
        decision = await self.reasoning_system.make_decision(context)
        
        # Convert to ReasoningResult format
        reasoning_result = ReasoningResult(
            situation_assessment=decision.reasoning_result.get("situation_assessment", ""),
            threat_analysis=decision.reasoning_result.get("threat_analysis", ""),
            opportunity_identification=decision.reasoning_result.get("opportunity_identification", []),
            risk_assessment=decision.reasoning_result.get("risk_assessment", {}),
            recommended_actions=decision.reasoning_result.get("recommended_actions", []),
            confidence_score=decision.confidence_score,
            reasoning_chain=decision.reasoning_result.get("reasoning_chain", []),
            alternatives_considered=decision.reasoning_result.get("alternatives_considered", [])
        )
        
        # Store decision for later use
        self.last_decision = decision
        
        return reasoning_result
    
    async def plan_actions(self, reasoning: ReasoningResult) -> ActionPlan:
        """
        Create an action plan based on reasoning results
        
        Args:
            reasoning: Results from situation analysis
            
        Returns:
            ActionPlan: Planned actions to execute
        """
        # Use the selected action from the last decision
        if hasattr(self, 'last_decision') and self.last_decision.selected_action:
            selected_action = self.last_decision.selected_action
            
            action_plan = ActionPlan(
                primary_action=selected_action.name,
                action_type=selected_action.action_type.value,
                target=selected_action.parameters.get("target"),
                parameters=selected_action.parameters,
                expected_outcome=f"Execute {selected_action.name} successfully",
                success_criteria=[
                    "Action completes without errors",
                    "Expected effects are achieved",
                    "No safety violations occur"
                ],
                fallback_actions=reasoning.alternatives_considered[:3],
                estimated_duration=selected_action.duration,
                risk_level=self._assess_risk_level(reasoning.risk_assessment)
            )
            
            return action_plan
        else:
            # Fallback: create plan from reasoning recommendations
            if reasoning.recommended_actions:
                primary_action = reasoning.recommended_actions[0]
                
                action_plan = ActionPlan(
                    primary_action=primary_action,
                    action_type="general",
                    target=None,
                    parameters={},
                    expected_outcome=f"Execute {primary_action}",
                    success_criteria=["Action completes successfully"],
                    fallback_actions=reasoning.recommended_actions[1:3],
                    estimated_duration=30.0,
                    risk_level=self._assess_risk_level(reasoning.risk_assessment)
                )
                
                return action_plan
            else:
                # No actions recommended - create idle plan
                return ActionPlan(
                    primary_action="idle",
                    action_type="maintenance",
                    target=None,
                    parameters={},
                    expected_outcome="Maintain current state",
                    success_criteria=["No errors occur"],
                    fallback_actions=[],
                    estimated_duration=5.0,
                    risk_level="low"
                )
    
    @abstractmethod
    async def execute_action(self, action: ActionPlan) -> ActionResult:
        """
        Execute the planned action
        
        Args:
            action: Action plan to execute
            
        Returns:
            ActionResult: Results of action execution
        """
        pass
    
    async def learn_from_outcome(self, action: ActionPlan, result: ActionResult) -> None:
        """
        Learn from action outcomes and update memory
        
        Args:
            action: Action that was executed
            result: Result of the action
        """
        try:
            # Create experience record
            experience = Experience(
                experience_id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                timestamp=datetime.now(),
                context=await self.perceive_environment(),
                action_taken=action,
                reasoning=ReasoningResult(
                    situation_assessment="",
                    threat_analysis="",
                    opportunity_identification=[],
                    risk_assessment={},
                    recommended_actions=[],
                    confidence_score=0.0,
                    reasoning_chain=[],
                    alternatives_considered=[]
                ),
                outcome=result,
                success=result.success,
                lessons_learned=[],
                mitre_attack_mapping=[],
                confidence_score=result.confidence
            )
            
            # Store experience
            self.experiences.append(experience)
            
            # Update performance metrics
            self.performance_metrics['actions_taken'] += 1
            if result.success:
                self.performance_metrics['successful_actions'] += 1
            else:
                self.performance_metrics['failed_actions'] += 1
            
            self.logger.debug(f"Learning from action outcome: {result.success}")
            
        except Exception as e:
            self.logger.error(f"Failed to learn from outcome: {e}")
    
    async def communicate_with_team(self, message: Dict[str, Any]) -> None:
        """
        Send message to team members
        
        Args:
            message: Message to send to team
        """
        try:
            # This will be implemented with actual message bus
            self.logger.debug(f"Sending team message: {message.get('type', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Failed to communicate with team: {e}")
    
    async def update_memory(self, experience: Experience) -> None:
        """
        Update agent memory with new experience
        
        Args:
            experience: Experience to store in memory
        """
        try:
            # Add to short-term memory
            self.short_term_memory.append(experience)
            
            # Manage memory size
            if len(self.short_term_memory) > self.config.max_memory_size:
                # Move oldest to long-term memory (vector database)
                oldest = self.short_term_memory.pop(0)
                await self._store_long_term_memory(oldest)
            
            self.logger.debug("Memory updated with new experience")
            
        except Exception as e:
            self.logger.error(f"Failed to update memory: {e}")
    
    async def _store_long_term_memory(self, experience: Experience) -> None:
        """Store experience in long-term memory (vector database)"""
        # This will be implemented with vector database integration
        pass
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'team': self.team.value,
            'role': self.role.value,
            'status': self.status.value,
            'current_objective': self.current_objective,
            'performance_metrics': self.performance_metrics,
            'memory_size': len(self.short_term_memory),
            'last_action_time': self.last_action_time
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent"""
        try:
            self.logger.info(f"Shutting down agent {self.name}")
            self.status = AgentStatus.SHUTDOWN
            
            # Save any pending experiences
            for experience in self.short_term_memory:
                await self._store_long_term_memory(experience)
            
            self.logger.info(f"Agent {self.name} shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during agent shutdown: {e}")
    
    def _assess_risk_level(self, risk_assessment: Dict[str, float]) -> str:
        """Assess overall risk level from risk assessment scores"""
        if not risk_assessment:
            return "low"
        
        # Calculate average risk score
        avg_risk = sum(risk_assessment.values()) / len(risk_assessment.values())
        
        if avg_risk >= 0.7:
            return "high"
        elif avg_risk >= 0.4:
            return "medium"
        else:
            return "low"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.agent_id}, team={self.team.value}, role={self.role.value})>"