#!/usr/bin/env python3
"""
Tests for LangGraph coordinator functionality
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from agents.coordinator import (
    LangGraphCoordinator, Scenario, WorkflowResult, WorkflowState, Phase,
    AgentRegistration
)
from agents.base_agent import BaseAgent, AgentConfig, Team, Role, AgentStatus
from agents.communication import MessageBus, MessageType, Priority

class MockAgent(BaseAgent):
    """Mock agent for testing"""
    
    async def reason_about_situation(self, state):
        return Mock()
    
    async def plan_actions(self, reasoning):
        return Mock()
    
    async def execute_action(self, action):
        return Mock()

@pytest.fixture
def mock_message_bus():
    """Create mock message bus"""
    bus = Mock(spec=MessageBus)
    bus.running = True
    bus.initialize = AsyncMock()
    bus.start_message_processing = AsyncMock()
    bus.subscribe_to_topic = AsyncMock()
    bus.send_direct_message = AsyncMock()
    bus.broadcast_to_team = AsyncMock()
    return bus

@pytest.fixture
def coordinator(mock_message_bus):
    """Create test coordinator"""
    return LangGraphCoordinator(mock_message_bus)

@pytest.fixture
def mock_agent():
    """Create mock agent"""
    config = AgentConfig(
        agent_id="test_agent_001",
        team=Team.RED,
        role=Role.RECON,
        name="Test Agent",
        description="Test agent for coordination"
    )
    agent = MockAgent(config)
    agent.status = AgentStatus.ACTIVE
    return agent

@pytest.fixture
def test_scenario():
    """Create test scenario"""
    return Scenario(
        scenario_id="test_scenario_001",
        name="Test Scenario",
        description="Test scenario for coordination",
        duration=timedelta(minutes=30),
        phases=[Phase.RECONNAISSANCE, Phase.EXPLOITATION, Phase.DEFENSE],
        objectives={
            Team.RED: ["compromise_target", "maintain_persistence"],
            Team.BLUE: ["detect_intrusion", "contain_threat"]
        },
        constraints=["no_destructive_actions"],
        success_criteria={
            Team.RED: ["successful_exploitation"],
            Team.BLUE: ["threat_contained"]
        },
        environment_config={"network": "192.168.1.0/24"}
    )

@pytest.mark.asyncio
async def test_coordinator_initialization(coordinator, mock_message_bus):
    """Test coordinator initialization"""
    assert not coordinator.running
    assert coordinator.workflow_state == WorkflowState.INITIALIZING
    
    await coordinator.initialize()
    
    assert coordinator.running
    assert coordinator.workflow_state == WorkflowState.PLANNING
    mock_message_bus.initialize.assert_called_once()
    mock_message_bus.start_message_processing.assert_called_once()

@pytest.mark.asyncio
async def test_agent_registration(coordinator, mock_agent):
    """Test agent registration"""
    await coordinator.initialize()
    
    capabilities = ["network_scanning", "vulnerability_assessment"]
    success = await coordinator.register_agent(mock_agent, capabilities)
    
    assert success
    assert mock_agent.agent_id in coordinator.registered_agents
    assert mock_agent.agent_id in coordinator.team_assignments[Team.RED]
    
    registration = coordinator.registered_agents[mock_agent.agent_id]
    assert registration.agent == mock_agent
    assert registration.capabilities == capabilities

@pytest.mark.asyncio
async def test_agent_unregistration(coordinator, mock_agent):
    """Test agent unregistration"""
    await coordinator.initialize()
    
    # Register agent first
    await coordinator.register_agent(mock_agent, ["test_capability"])
    assert mock_agent.agent_id in coordinator.registered_agents
    
    # Unregister agent
    success = await coordinator.unregister_agent(mock_agent.agent_id)
    
    assert success
    assert mock_agent.agent_id not in coordinator.registered_agents
    assert mock_agent.agent_id not in coordinator.team_assignments[Team.RED]

@pytest.mark.asyncio
async def test_scenario_start(coordinator, test_scenario):
    """Test scenario start"""
    await coordinator.initialize()
    
    workflow_id = await coordinator.start_scenario(test_scenario)
    
    assert workflow_id is not None
    assert workflow_id in coordinator.active_workflows
    assert coordinator.current_scenario == test_scenario
    assert coordinator.current_phase == Phase.RECONNAISSANCE
    
    workflow_result = coordinator.active_workflows[workflow_id]
    assert workflow_result.scenario_id == test_scenario.scenario_id
    assert workflow_result.state == WorkflowState.PLANNING

@pytest.mark.asyncio
async def test_phase_transition(coordinator, test_scenario):
    """Test phase transition"""
    await coordinator.initialize()
    
    # Start scenario
    await coordinator.start_scenario(test_scenario)
    assert coordinator.current_phase == Phase.RECONNAISSANCE
    
    # Transition to next phase
    await coordinator._transition_to_phase(Phase.EXPLOITATION)
    
    assert coordinator.current_phase == Phase.EXPLOITATION
    assert coordinator.phase_start_time is not None
    assert coordinator.coordination_metrics['phase_transitions'] == 2  # Initial + transition

@pytest.mark.asyncio
async def test_workflow_coordination(coordinator, test_scenario, mock_agent):
    """Test complete workflow coordination"""
    await coordinator.initialize()
    
    # Register an agent
    await coordinator.register_agent(mock_agent, ["test_capability"])
    
    # Mock the phase execution to complete quickly
    original_execute_phase = coordinator._execute_phase
    async def mock_execute_phase(phase, scenario, workflow_result):
        # Simplified phase execution for testing
        workflow_result.phase_results[phase] = {
            'start_time': datetime.now().isoformat(),
            'actual_end_time': datetime.now().isoformat(),
            'agent_activities': {},
            'objectives_completed': [],
            'constraints_violated': []
        }
    
    coordinator._execute_phase = mock_execute_phase
    
    # Execute workflow
    workflow_result = await coordinator.coordinate_multi_agent_workflow(test_scenario)
    
    assert workflow_result.success
    assert workflow_result.state == WorkflowState.COMPLETED
    assert len(workflow_result.phase_results) == len(test_scenario.phases)
    assert coordinator.coordination_metrics['workflows_completed'] == 1

@pytest.mark.asyncio
async def test_agent_heartbeat_monitoring(coordinator, mock_agent):
    """Test agent heartbeat monitoring"""
    await coordinator.initialize()
    
    # Register agent
    await coordinator.register_agent(mock_agent, ["test_capability"])
    
    # Simulate old heartbeat
    registration = coordinator.registered_agents[mock_agent.agent_id]
    registration.last_heartbeat = datetime.now() - timedelta(minutes=5)
    
    # Trigger heartbeat check
    await coordinator._handle_agent_failure(mock_agent.agent_id)
    
    # Agent should be marked as failed
    assert registration.agent.status == AgentStatus.ERROR

def test_coordination_status(coordinator):
    """Test coordination status reporting"""
    status = coordinator.get_coordination_status()
    
    assert 'coordinator_id' in status
    assert 'running' in status
    assert 'workflow_state' in status
    assert 'registered_agents' in status
    assert 'team_assignments' in status
    assert 'metrics' in status

@pytest.mark.asyncio
async def test_coordinator_shutdown(coordinator, mock_agent):
    """Test coordinator shutdown"""
    await coordinator.initialize()
    
    # Register an agent
    await coordinator.register_agent(mock_agent, ["test_capability"])
    
    await coordinator.shutdown()
    
    assert not coordinator.running
    assert len(coordinator.registered_agents) == 0

def test_scenario_creation(test_scenario):
    """Test scenario creation"""
    assert test_scenario.scenario_id == "test_scenario_001"
    assert test_scenario.name == "Test Scenario"
    assert len(test_scenario.phases) == 3
    assert Team.RED in test_scenario.objectives
    assert Team.BLUE in test_scenario.objectives

def test_workflow_result_creation():
    """Test workflow result creation"""
    workflow_result = WorkflowResult(
        workflow_id="test_workflow",
        scenario_id="test_scenario",
        start_time=datetime.now(),
        end_time=None,
        state=WorkflowState.PLANNING,
        participating_agents=["agent1", "agent2"],
        phase_results={},
        team_scores={Team.RED: 0.0, Team.BLUE: 0.0},
        success=False,
        errors=[],
        metrics={}
    )
    
    assert workflow_result.workflow_id == "test_workflow"
    assert workflow_result.state == WorkflowState.PLANNING
    assert len(workflow_result.participating_agents) == 2

def test_agent_registration_data():
    """Test agent registration data structure"""
    mock_agent = Mock()
    mock_agent.agent_id = "test_agent"
    mock_agent.team = Team.BLUE
    
    registration = AgentRegistration(
        agent=mock_agent,
        capabilities=["monitoring", "analysis"],
        current_task=None,
        last_heartbeat=datetime.now(),
        performance_score=1.0
    )
    
    assert registration.agent == mock_agent
    assert "monitoring" in registration.capabilities
    assert registration.performance_score == 1.0

def test_phase_enum():
    """Test Phase enum values"""
    assert Phase.RECONNAISSANCE.value == "reconnaissance"
    assert Phase.EXPLOITATION.value == "exploitation"
    assert Phase.PERSISTENCE.value == "persistence"
    assert Phase.DEFENSE.value == "defense"
    assert Phase.RECOVERY.value == "recovery"
    assert Phase.ANALYSIS.value == "analysis"

def test_workflow_state_enum():
    """Test WorkflowState enum values"""
    assert WorkflowState.INITIALIZING.value == "initializing"
    assert WorkflowState.PLANNING.value == "planning"
    assert WorkflowState.EXECUTING.value == "executing"
    assert WorkflowState.COORDINATING.value == "coordinating"
    assert WorkflowState.EVALUATING.value == "evaluating"
    assert WorkflowState.COMPLETED.value == "completed"
    assert WorkflowState.FAILED.value == "failed"

@pytest.mark.asyncio
async def test_coordination_metrics_update(coordinator):
    """Test coordination metrics updates"""
    await coordinator.initialize()
    
    initial_metrics = coordinator.coordination_metrics.copy()
    
    # Simulate some coordination activity
    coordinator.coordination_metrics['agent_coordination_events'] += 1
    coordinator.coordination_metrics['phase_transitions'] += 1
    
    assert coordinator.coordination_metrics['agent_coordination_events'] > initial_metrics['agent_coordination_events']
    assert coordinator.coordination_metrics['phase_transitions'] > initial_metrics['phase_transitions']

@pytest.mark.asyncio
async def test_average_duration_calculation(coordinator):
    """Test average workflow duration calculation"""
    # Test first workflow
    coordinator._update_average_duration(10.0)
    assert coordinator.coordination_metrics['average_workflow_duration'] == 10.0
    
    # Test second workflow
    coordinator.coordination_metrics['workflows_completed'] = 2
    coordinator._update_average_duration(20.0)
    assert coordinator.coordination_metrics['average_workflow_duration'] == 15.0