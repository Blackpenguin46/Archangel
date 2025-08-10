#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - LangGraph Coordinator
Multi-agent coordination and workflow management using LangGraph
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable
import json

from .base_agent import BaseAgent, AgentConfig, Team, Role, AgentStatus
from .communication import MessageBus, AgentMessage, TeamMessage, MessageType, Priority

logger = logging.getLogger(__name__)

class WorkflowState(Enum):
    """States of the coordination workflow"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    COORDINATING = "coordinating"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"

class Phase(Enum):
    """Game phases for structured progression"""
    RECONNAISSANCE = "reconnaissance"
    EXPLOITATION = "exploitation"
    PERSISTENCE = "persistence"
    DEFENSE = "defense"
    RECOVERY = "recovery"
    ANALYSIS = "analysis"

@dataclass
class Scenario:
    """Scenario configuration for agent coordination"""
    scenario_id: str
    name: str
    description: str
    duration: timedelta
    phases: List[Phase]
    objectives: Dict[Team, List[str]]
    constraints: List[str]
    success_criteria: Dict[Team, List[str]]
    environment_config: Dict[str, Any]

@dataclass
class WorkflowResult:
    """Result of workflow execution"""
    workflow_id: str
    scenario_id: str
    start_time: datetime
    end_time: Optional[datetime]
    state: WorkflowState
    participating_agents: List[str]
    phase_results: Dict[Phase, Dict[str, Any]]
    team_scores: Dict[Team, float]
    success: bool
    errors: List[str]
    metrics: Dict[str, Any]

@dataclass
class AgentRegistration:
    """Agent registration information"""
    agent: BaseAgent
    capabilities: List[str]
    current_task: Optional[str]
    last_heartbeat: datetime
    performance_score: float

class LangGraphCoordinator:
    """
    Central coordinator for multi-agent workflows using LangGraph patterns.
    
    Responsibilities:
    - Agent registration and lifecycle management
    - Workflow orchestration and state management
    - Phase transitions and constraint enforcement
    - Inter-agent communication coordination
    - Performance monitoring and evaluation
    """
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.coordinator_id = str(uuid.uuid4())
        
        # Agent management
        self.registered_agents: Dict[str, AgentRegistration] = {}
        self.team_assignments: Dict[Team, Set[str]] = {
            Team.RED: set(),
            Team.BLUE: set(),
            Team.NEUTRAL: set()
        }
        
        # Workflow management
        self.active_workflows: Dict[str, WorkflowResult] = {}
        self.current_scenario: Optional[Scenario] = None
        self.current_phase: Optional[Phase] = None
        self.phase_start_time: Optional[datetime] = None
        
        # Coordination state
        self.workflow_state = WorkflowState.INITIALIZING
        self.coordination_rules: Dict[str, Callable] = {}
        self.constraint_validators: List[Callable] = []
        
        # Performance tracking
        self.coordination_metrics = {
            'workflows_completed': 0,
            'workflows_failed': 0,
            'average_workflow_duration': 0.0,
            'agent_coordination_events': 0,
            'phase_transitions': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    async def initialize(self) -> None:
        """Initialize the coordinator"""
        try:
            self.logger.info("Initializing LangGraph coordinator")
            
            # Initialize message bus if not already done
            if not self.message_bus.running:
                await self.message_bus.initialize()
                await self.message_bus.start_message_processing()
            
            # Subscribe to coordination messages
            await self.message_bus.subscribe_to_topic(
                "coordination", 
                self._handle_coordination_message
            )
            
            # Subscribe to agent status updates
            await self.message_bus.subscribe_to_topic(
                "agent_status",
                self._handle_agent_status
            )
            
            # Start coordination loop
            asyncio.create_task(self._coordination_loop())
            
            # Start heartbeat monitoring
            asyncio.create_task(self._monitor_agent_heartbeats())
            
            self.running = True
            self.workflow_state = WorkflowState.PLANNING
            
            self.logger.info("LangGraph coordinator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize coordinator: {e}")
            raise
    
    async def register_agent(self, agent: BaseAgent, capabilities: List[str]) -> bool:
        """
        Register an agent with the coordinator
        
        Args:
            agent: Agent to register
            capabilities: List of agent capabilities
            
        Returns:
            bool: True if registration successful
        """
        try:
            agent_id = agent.agent_id
            
            # Create registration record
            registration = AgentRegistration(
                agent=agent,
                capabilities=capabilities,
                current_task=None,
                last_heartbeat=datetime.now(),
                performance_score=1.0
            )
            
            # Store registration
            self.registered_agents[agent_id] = registration
            
            # Add to team assignment
            self.team_assignments[agent.team].add(agent_id)
            
            self.logger.info(f"Registered agent {agent.name} ({agent_id}) for team {agent.team.value}")
            
            # Send welcome message to agent
            welcome_message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.coordinator_id,
                recipient_id=agent_id,
                message_type=MessageType.STATUS,
                content={
                    'type': 'registration_confirmed',
                    'coordinator_id': self.coordinator_id,
                    'team_members': list(self.team_assignments[agent.team])
                },
                timestamp=datetime.now(),
                priority=Priority.HIGH
            )
            
            await self.message_bus.send_direct_message(agent_id, welcome_message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the coordinator
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            bool: True if unregistration successful
        """
        try:
            if agent_id not in self.registered_agents:
                self.logger.warning(f"Attempted to unregister unknown agent {agent_id}")
                return False
            
            registration = self.registered_agents[agent_id]
            team = registration.agent.team
            
            # Remove from registrations
            del self.registered_agents[agent_id]
            
            # Remove from team assignment
            self.team_assignments[team].discard(agent_id)
            
            self.logger.info(f"Unregistered agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def start_scenario(self, scenario: Scenario) -> str:
        """
        Start a new scenario workflow
        
        Args:
            scenario: Scenario configuration
            
        Returns:
            str: Workflow ID
        """
        try:
            workflow_id = str(uuid.uuid4())
            
            # Create workflow result tracking
            workflow_result = WorkflowResult(
                workflow_id=workflow_id,
                scenario_id=scenario.scenario_id,
                start_time=datetime.now(),
                end_time=None,
                state=WorkflowState.PLANNING,
                participating_agents=list(self.registered_agents.keys()),
                phase_results={},
                team_scores={Team.RED: 0.0, Team.BLUE: 0.0},
                success=False,
                errors=[],
                metrics={}
            )
            
            # Store workflow
            self.active_workflows[workflow_id] = workflow_result
            self.current_scenario = scenario
            
            # Initialize first phase
            if scenario.phases:
                await self._transition_to_phase(scenario.phases[0])
            
            # Notify all agents of scenario start
            await self._broadcast_scenario_start(scenario, workflow_id)
            
            self.logger.info(f"Started scenario {scenario.name} with workflow {workflow_id}")
            return workflow_id
            
        except Exception as e:
            self.logger.error(f"Failed to start scenario: {e}")
            raise
    
    async def _transition_to_phase(self, new_phase: Phase) -> None:
        """
        Transition to a new phase
        
        Args:
            new_phase: Phase to transition to
        """
        try:
            old_phase = self.current_phase
            self.current_phase = new_phase
            self.phase_start_time = datetime.now()
            
            # Update metrics
            self.coordination_metrics['phase_transitions'] += 1
            
            # Notify agents of phase transition
            phase_message = {
                'type': 'phase_transition',
                'old_phase': old_phase.value if old_phase else None,
                'new_phase': new_phase.value,
                'transition_time': self.phase_start_time.isoformat()
            }
            
            await self._broadcast_to_all_agents(phase_message, MessageType.COMMAND, Priority.HIGH)
            
            self.logger.info(f"Transitioned from {old_phase} to {new_phase}")
            
        except Exception as e:
            self.logger.error(f"Failed to transition to phase {new_phase}: {e}")
            raise
    
    async def _broadcast_scenario_start(self, scenario: Scenario, workflow_id: str) -> None:
        """Broadcast scenario start to all agents"""
        message_content = {
            'type': 'scenario_start',
            'workflow_id': workflow_id,
            'scenario': {
                'id': scenario.scenario_id,
                'name': scenario.name,
                'description': scenario.description,
                'duration': scenario.duration.total_seconds(),
                'phases': [phase.value for phase in scenario.phases]
            }
        }
        
        await self._broadcast_to_all_agents(message_content, MessageType.COMMAND, Priority.CRITICAL)
    
    async def _broadcast_to_all_agents(self, content: Dict[str, Any], 
                                     message_type: MessageType, 
                                     priority: Priority) -> None:
        """Broadcast message to all registered agents"""
        for agent_id in self.registered_agents.keys():
            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.coordinator_id,
                recipient_id=agent_id,
                message_type=message_type,
                content=content,
                timestamp=datetime.now(),
                priority=priority
            )
            
            await self.message_bus.send_direct_message(agent_id, message)
    
    async def coordinate_multi_agent_workflow(self, scenario: Scenario) -> WorkflowResult:
        """
        Coordinate a complete multi-agent workflow
        
        Args:
            scenario: Scenario to execute
            
        Returns:
            WorkflowResult: Results of the workflow execution
        """
        try:
            workflow_id = await self.start_scenario(scenario)
            workflow_result = self.active_workflows[workflow_id]
            
            # Execute each phase
            for phase in scenario.phases:
                await self._execute_phase(phase, scenario, workflow_result)
                
                # Check if workflow should continue
                if workflow_result.state == WorkflowState.FAILED:
                    break
            
            # Complete workflow
            workflow_result.end_time = datetime.now()
            workflow_result.state = WorkflowState.COMPLETED
            workflow_result.success = True
            
            # Update metrics
            self.coordination_metrics['workflows_completed'] += 1
            duration = (workflow_result.end_time - workflow_result.start_time).total_seconds()
            self._update_average_duration(duration)
            
            self.logger.info(f"Completed workflow {workflow_id}")
            return workflow_result
            
        except Exception as e:
            self.logger.error(f"Workflow coordination failed: {e}")
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id].state = WorkflowState.FAILED
                self.active_workflows[workflow_id].errors.append(str(e))
                self.coordination_metrics['workflows_failed'] += 1
            raise
    
    async def _execute_phase(self, phase: Phase, scenario: Scenario, workflow_result: WorkflowResult) -> None:
        """Execute a specific phase of the workflow"""
        try:
            await self._transition_to_phase(phase)
            
            # Get phase duration (simplified - could be more sophisticated)
            phase_duration = scenario.duration / len(scenario.phases)
            
            # Monitor phase execution
            phase_start = datetime.now()
            phase_end = phase_start + phase_duration
            
            phase_results = {
                'start_time': phase_start.isoformat(),
                'planned_end_time': phase_end.isoformat(),
                'agent_activities': {},
                'objectives_completed': [],
                'constraints_violated': []
            }
            
            # Monitor agents during phase
            while datetime.now() < phase_end:
                await self._monitor_phase_execution(phase, phase_results)
                await asyncio.sleep(1.0)  # Check every second
            
            # Complete phase
            phase_results['actual_end_time'] = datetime.now().isoformat()
            workflow_result.phase_results[phase] = phase_results
            
            self.logger.info(f"Completed phase {phase.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute phase {phase}: {e}")
            workflow_result.errors.append(f"Phase {phase.value} failed: {str(e)}")
            raise
    
    async def _monitor_phase_execution(self, phase: Phase, phase_results: Dict[str, Any]) -> None:
        """Monitor agent activities during phase execution"""
        try:
            # Check agent status and activities
            for agent_id, registration in self.registered_agents.items():
                if registration.agent.status == AgentStatus.ACTIVE:
                    # Record agent activity (simplified)
                    if agent_id not in phase_results['agent_activities']:
                        phase_results['agent_activities'][agent_id] = []
                    
                    # This would be expanded to track actual agent actions
                    activity = {
                        'timestamp': datetime.now().isoformat(),
                        'status': registration.agent.status.value,
                        'current_task': registration.current_task
                    }
                    phase_results['agent_activities'][agent_id].append(activity)
            
        except Exception as e:
            self.logger.error(f"Error monitoring phase execution: {e}")
    
    async def _handle_coordination_message(self, message: AgentMessage) -> None:
        """Handle coordination messages from agents"""
        try:
            content = message.content
            message_subtype = content.get('type', 'unknown')
            
            if message_subtype == 'request_coordination':
                await self._handle_coordination_request(message)
            elif message_subtype == 'report_progress':
                await self._handle_progress_report(message)
            elif message_subtype == 'request_phase_transition':
                await self._handle_phase_transition_request(message)
            else:
                self.logger.warning(f"Unknown coordination message type: {message_subtype}")
            
            self.coordination_metrics['agent_coordination_events'] += 1
            
        except Exception as e:
            self.logger.error(f"Error handling coordination message: {e}")
    
    async def _handle_agent_status(self, message: AgentMessage) -> None:
        """Handle agent status updates"""
        try:
            agent_id = message.sender_id
            if agent_id in self.registered_agents:
                registration = self.registered_agents[agent_id]
                registration.last_heartbeat = datetime.now()
                
                # Update agent status based on message content
                status_info = message.content
                if 'current_task' in status_info:
                    registration.current_task = status_info['current_task']
                
                if 'performance_score' in status_info:
                    registration.performance_score = status_info['performance_score']
            
        except Exception as e:
            self.logger.error(f"Error handling agent status: {e}")
    
    async def _handle_coordination_request(self, message: AgentMessage) -> None:
        """Handle coordination requests from agents"""
        # Implementation for handling coordination requests
        pass
    
    async def _handle_progress_report(self, message: AgentMessage) -> None:
        """Handle progress reports from agents"""
        # Implementation for handling progress reports
        pass
    
    async def _handle_phase_transition_request(self, message: AgentMessage) -> None:
        """Handle phase transition requests from agents"""
        # Implementation for handling phase transition requests
        pass
    
    async def _coordination_loop(self) -> None:
        """Main coordination loop"""
        while self.running:
            try:
                # Process any pending coordination tasks
                await self._process_coordination_tasks()
                
                # Check for workflow timeouts
                await self._check_workflow_timeouts()
                
                # Update coordination metrics
                await self._update_coordination_metrics()
                
                await asyncio.sleep(1.0)  # Coordination loop interval
                
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(5.0)  # Back off on error
    
    async def _monitor_agent_heartbeats(self) -> None:
        """Monitor agent heartbeats and handle failures"""
        while self.running:
            try:
                current_time = datetime.now()
                heartbeat_timeout = timedelta(seconds=30)
                
                for agent_id, registration in list(self.registered_agents.items()):
                    time_since_heartbeat = current_time - registration.last_heartbeat
                    
                    if time_since_heartbeat > heartbeat_timeout:
                        self.logger.warning(f"Agent {agent_id} heartbeat timeout")
                        # Handle agent failure
                        await self._handle_agent_failure(agent_id)
                
                await asyncio.sleep(10.0)  # Check heartbeats every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring heartbeats: {e}")
                await asyncio.sleep(5.0)
    
    async def _handle_agent_failure(self, agent_id: str) -> None:
        """Handle agent failure"""
        try:
            if agent_id in self.registered_agents:
                registration = self.registered_agents[agent_id]
                self.logger.error(f"Handling failure of agent {agent_id}")
                
                # Mark agent as failed
                registration.agent.status = AgentStatus.ERROR
                
                # Notify other team members
                team = registration.agent.team
                failure_message = {
                    'type': 'agent_failure',
                    'failed_agent_id': agent_id,
                    'failure_time': datetime.now().isoformat()
                }
                
                team_message = TeamMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.coordinator_id,
                    recipient_id=f"team.{team.value}",
                    message_type=MessageType.ALERT,
                    content=failure_message,
                    timestamp=datetime.now(),
                    priority=Priority.HIGH,
                    team=team.value
                )
                
                await self.message_bus.broadcast_to_team(team.value, team_message)
                
        except Exception as e:
            self.logger.error(f"Error handling agent failure: {e}")
    
    async def _process_coordination_tasks(self) -> None:
        """Process pending coordination tasks"""
        # Implementation for processing coordination tasks
        pass
    
    async def _check_workflow_timeouts(self) -> None:
        """Check for workflow timeouts"""
        # Implementation for checking workflow timeouts
        pass
    
    async def _update_coordination_metrics(self) -> None:
        """Update coordination metrics"""
        # Implementation for updating metrics
        pass
    
    def _update_average_duration(self, duration: float) -> None:
        """Update average workflow duration"""
        completed = self.coordination_metrics['workflows_completed']
        if completed > 1:
            current_avg = self.coordination_metrics['average_workflow_duration']
            new_avg = ((current_avg * (completed - 1)) + duration) / completed
            self.coordination_metrics['average_workflow_duration'] = new_avg
        else:
            self.coordination_metrics['average_workflow_duration'] = duration
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            'coordinator_id': self.coordinator_id,
            'running': self.running,
            'workflow_state': self.workflow_state.value,
            'current_phase': self.current_phase.value if self.current_phase else None,
            'registered_agents': len(self.registered_agents),
            'active_workflows': len(self.active_workflows),
            'team_assignments': {
                team.value: len(agents) 
                for team, agents in self.team_assignments.items()
            },
            'metrics': self.coordination_metrics
        }
    
    async def shutdown(self) -> None:
        """Shutdown the coordinator"""
        try:
            self.logger.info("Shutting down LangGraph coordinator")
            self.running = False
            
            # Notify all agents of shutdown
            shutdown_message = {
                'type': 'coordinator_shutdown',
                'shutdown_time': datetime.now().isoformat()
            }
            
            await self._broadcast_to_all_agents(
                shutdown_message, 
                MessageType.COMMAND, 
                Priority.CRITICAL
            )
            
            # Unregister all agents
            for agent_id in list(self.registered_agents.keys()):
                await self.unregister_agent(agent_id)
            
            self.logger.info("LangGraph coordinator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during coordinator shutdown: {e}")