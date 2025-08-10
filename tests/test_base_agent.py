#!/usr/bin/env python3
"""
Tests for BaseAgent functionality
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from agents.base_agent import (
    BaseAgent, AgentConfig, Team, Role, AgentStatus,
    EnvironmentState, ReasoningResult, ActionPlan, ActionResult
)

class TestAgent(BaseAgent):
    """Test implementation of BaseAgent"""
    
    async def reason_about_situation(self, state: EnvironmentState) -> ReasoningResult:
        return ReasoningResult(
            situation_assessment="Test situation",
            threat_analysis="Low threat",
            opportunity_identification=["test_opportunity"],
            risk_assessment={"low": 0.1},
            recommended_actions=["test_action"],
            confidence_score=0.8,
            reasoning_chain=["step1", "step2"],
            alternatives_considered=["alt1", "alt2"]
        )
    
    async def plan_actions(self, reasoning: ReasoningResult) -> ActionPlan:
        return ActionPlan(
            primary_action="test_action",
            action_type="test",
            target="test_target",
            parameters={"param1": "value1"},
            expected_outcome="success",
            success_criteria=["criteria1"],
            fallback_actions=["fallback1"],
            estimated_duration=5.0,
            risk_level="low"
        )
    
    async def execute_action(self, action: ActionPlan) -> ActionResult:
        return ActionResult(
            action_id="test_action_id",
            action_type=action.action_type,
            success=True,
            outcome="Action completed successfully",
            data={"result": "success"},
            duration=3.0,
            errors=[],
            side_effects=[],
            confidence=0.9,
            timestamp=datetime.now()
        )

@pytest.fixture
def agent_config():
    """Create test agent configuration"""
    return AgentConfig(
        agent_id="test_agent_001",
        team=Team.RED,
        role=Role.RECON,
        name="Test Recon Agent",
        description="Test agent for reconnaissance",
        llm_model="gpt-4-turbo",
        max_memory_size=100,
        decision_timeout=10.0,
        tools=["nmap", "dirb"],
        constraints=["no_destructive_actions"],
        objectives=["gather_intelligence"]
    )

@pytest.fixture
def test_agent(agent_config):
    """Create test agent instance"""
    return TestAgent(agent_config)

@pytest.mark.asyncio
async def test_agent_initialization(test_agent):
    """Test agent initialization"""
    assert test_agent.status == AgentStatus.INITIALIZING
    assert test_agent.team == Team.RED
    assert test_agent.role == Role.RECON
    assert test_agent.name == "Test Recon Agent"
    
    # Initialize agent
    await test_agent.initialize()
    
    assert test_agent.status == AgentStatus.ACTIVE

@pytest.mark.asyncio
async def test_agent_perception(test_agent):
    """Test environment perception"""
    await test_agent.initialize()
    
    state = await test_agent.perceive_environment()
    
    assert isinstance(state, EnvironmentState)
    assert state.timestamp is not None
    assert isinstance(state.network_topology, dict)
    assert isinstance(state.active_services, list)

@pytest.mark.asyncio
async def test_agent_reasoning(test_agent):
    """Test agent reasoning process"""
    await test_agent.initialize()
    
    state = await test_agent.perceive_environment()
    reasoning = await test_agent.reason_about_situation(state)
    
    assert isinstance(reasoning, ReasoningResult)
    assert reasoning.situation_assessment == "Test situation"
    assert reasoning.confidence_score == 0.8
    assert len(reasoning.recommended_actions) > 0

@pytest.mark.asyncio
async def test_agent_planning(test_agent):
    """Test action planning"""
    await test_agent.initialize()
    
    state = await test_agent.perceive_environment()
    reasoning = await test_agent.reason_about_situation(state)
    plan = await test_agent.plan_actions(reasoning)
    
    assert isinstance(plan, ActionPlan)
    assert plan.primary_action == "test_action"
    assert plan.action_type == "test"
    assert plan.risk_level == "low"

@pytest.mark.asyncio
async def test_agent_execution(test_agent):
    """Test action execution"""
    await test_agent.initialize()
    
    state = await test_agent.perceive_environment()
    reasoning = await test_agent.reason_about_situation(state)
    plan = await test_agent.plan_actions(reasoning)
    result = await test_agent.execute_action(plan)
    
    assert isinstance(result, ActionResult)
    assert result.success is True
    assert result.confidence == 0.9

@pytest.mark.asyncio
async def test_agent_learning(test_agent):
    """Test learning from outcomes"""
    await test_agent.initialize()
    
    # Execute full cycle
    state = await test_agent.perceive_environment()
    reasoning = await test_agent.reason_about_situation(state)
    plan = await test_agent.plan_actions(reasoning)
    result = await test_agent.execute_action(plan)
    
    # Learn from outcome
    initial_actions = test_agent.performance_metrics['actions_taken']
    await test_agent.learn_from_outcome(plan, result)
    
    assert test_agent.performance_metrics['actions_taken'] == initial_actions + 1
    assert test_agent.performance_metrics['successful_actions'] == 1
    assert len(test_agent.experiences) == 1

@pytest.mark.asyncio
async def test_agent_memory_management(test_agent):
    """Test memory management"""
    await test_agent.initialize()
    
    # Fill memory beyond capacity
    for i in range(test_agent.config.max_memory_size + 10):
        state = await test_agent.perceive_environment()
        reasoning = await test_agent.reason_about_situation(state)
        plan = await test_agent.plan_actions(reasoning)
        result = await test_agent.execute_action(plan)
        await test_agent.learn_from_outcome(plan, result)
    
    # Memory should not exceed max size
    assert len(test_agent.short_term_memory) <= test_agent.config.max_memory_size

@pytest.mark.asyncio
async def test_agent_status_reporting(test_agent):
    """Test status reporting"""
    await test_agent.initialize()
    
    status = await test_agent.get_status()
    
    assert status['agent_id'] == test_agent.agent_id
    assert status['name'] == test_agent.name
    assert status['team'] == test_agent.team.value
    assert status['role'] == test_agent.role.value
    assert status['status'] == AgentStatus.ACTIVE.value
    assert 'performance_metrics' in status

@pytest.mark.asyncio
async def test_agent_shutdown(test_agent):
    """Test agent shutdown"""
    await test_agent.initialize()
    assert test_agent.status == AgentStatus.ACTIVE
    
    await test_agent.shutdown()
    assert test_agent.status == AgentStatus.SHUTDOWN

def test_agent_config_creation():
    """Test agent configuration creation"""
    config = AgentConfig(
        agent_id="test_001",
        team=Team.BLUE,
        role=Role.SOC_ANALYST,
        name="Test SOC Agent",
        description="Test SOC analyst agent"
    )
    
    assert config.agent_id == "test_001"
    assert config.team == Team.BLUE
    assert config.role == Role.SOC_ANALYST
    assert config.llm_model == "gpt-4-turbo"  # Default value
    assert config.max_memory_size == 1000  # Default value

def test_team_enum():
    """Test Team enum values"""
    assert Team.RED.value == "red"
    assert Team.BLUE.value == "blue"
    assert Team.NEUTRAL.value == "neutral"

def test_role_enum():
    """Test Role enum values"""
    # Red team roles
    assert Role.RECON.value == "recon"
    assert Role.EXPLOIT.value == "exploit"
    assert Role.PERSISTENCE.value == "persistence"
    assert Role.EXFILTRATION.value == "exfiltration"
    
    # Blue team roles
    assert Role.SOC_ANALYST.value == "soc_analyst"
    assert Role.FIREWALL_CONFIG.value == "firewall_config"
    assert Role.SIEM_INTEGRATOR.value == "siem_integrator"

def test_agent_status_enum():
    """Test AgentStatus enum values"""
    assert AgentStatus.INITIALIZING.value == "initializing"
    assert AgentStatus.ACTIVE.value == "active"
    assert AgentStatus.IDLE.value == "idle"
    assert AgentStatus.BUSY.value == "busy"
    assert AgentStatus.ERROR.value == "error"
    assert AgentStatus.SHUTDOWN.value == "shutdown"