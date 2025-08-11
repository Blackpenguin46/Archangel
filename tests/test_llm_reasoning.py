#!/usr/bin/env python3
"""
Tests for LLM reasoning and behavior tree integration
"""

import asyncio
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
import time

from agents.llm_interface import (
    LLMConfig, LLMProvider, PromptType, PromptTemplate, PromptTemplateManager,
    LLMRequest, LLMResponse, OpenAIInterface, LLMReasoningEngine, SafetyValidator
)
from agents.behavior_tree import (
    BehaviorTree, BehaviorTreeBuilder, ExecutionContext, NodeResult, NodeStatus,
    SequenceNode, SelectorNode, ActionNode, ConditionNode, LLMReasoningNode
)
from agents.planning import (
    WorldState, Goal, Action, ActionType, Plan, PlanStatus,
    GOAPPlanner, PlanExecutor, ActionLibrary, PlanningEngine
)
from agents.reasoning_integration import (
    ReasoningContext, IntegratedDecision, LLMBehaviorTreeIntegration,
    IntegratedReasoningSystem
)

class TestPromptTemplateManager:
    """Test prompt template management"""
    
    def test_template_manager_initialization(self):
        """Test that template manager initializes with default templates"""
        manager = PromptTemplateManager()
        
        assert len(manager.templates) > 0
        assert "red_team_reasoning" in manager.templates
        assert "blue_team_reasoning" in manager.templates
        assert "action_planning" in manager.templates
        assert "safety_validation" in manager.templates
    
    def test_get_template(self):
        """Test getting templates by name"""
        manager = PromptTemplateManager()
        
        template = manager.get_template("red_team_reasoning")
        assert template is not None
        assert template.prompt_type == PromptType.REASONING
        assert len(template.required_variables) > 0
    
    def test_validate_variables(self):
        """Test variable validation"""
        manager = PromptTemplateManager()
        
        # Valid variables
        variables = {
            "network_topology": {},
            "active_services": [],
            "security_alerts": [],
            "threat_level": "low",
            "agent_role": "recon",
            "agent_team": "red",
            "current_objectives": [],
            "available_tools": [],
            "previous_actions": []
        }
        
        errors = manager.validate_variables("red_team_reasoning", variables)
        assert len(errors) == 0
        
        # Missing variables
        incomplete_variables = {"network_topology": {}}
        errors = manager.validate_variables("red_team_reasoning", incomplete_variables)
        assert len(errors) > 0
    
    def test_add_custom_template(self):
        """Test adding custom templates"""
        manager = PromptTemplateManager()
        
        custom_template = PromptTemplate(
            name="custom_test",
            prompt_type=PromptType.REASONING,
            system_prompt="Test system prompt",
            user_prompt_template="Test user prompt: {variable}",
            required_variables=["variable"]
        )
        
        manager.add_template(custom_template)
        assert "custom_test" in manager.templates
        
        retrieved = manager.get_template("custom_test")
        assert retrieved.name == "custom_test"

class TestLLMInterface:
    """Test LLM interface functionality"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "situation_assessment": "Test assessment",
            "threat_analysis": "Test threat analysis",
            "opportunity_identification": ["Test opportunity"],
            "risk_assessment": {"detection": 0.3},
            "recommended_actions": ["Test action"],
            "confidence_score": 0.8
        })
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 100
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    @pytest.mark.asyncio
    async def test_openai_interface_generation(self, mock_openai_client):
        """Test OpenAI interface response generation"""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo",
            api_key="test-key"
        )
        
        interface = OpenAIInterface(config)
        interface.client = mock_openai_client
        
        request = LLMRequest(
            prompt_type=PromptType.REASONING,
            template_name="red_team_reasoning",
            variables={
                "network_topology": {},
                "active_services": [],
                "security_alerts": [],
                "threat_level": "low",
                "agent_role": "recon",
                "agent_team": "red",
                "current_objectives": [],
                "available_tools": [],
                "previous_actions": []
            }
        )
        
        response = await interface.generate_response(request)
        
        assert response.content is not None
        assert response.prompt_type == PromptType.REASONING
        assert response.tokens_used == 100
        assert response.validation_passed
        assert response.safety_passed
    
    @pytest.mark.asyncio
    async def test_response_validation(self):
        """Test LLM response validation"""
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4-turbo")
        interface = OpenAIInterface(config)
        
        # Valid JSON response
        valid_response = LLMResponse(
            content='{"situation_assessment": "test", "confidence_score": 0.8}',
            prompt_type=PromptType.REASONING,
            template_name="test",
            model="gpt-4-turbo",
            tokens_used=50,
            response_time=1.0,
            confidence_score=0.8,
            validation_passed=False,
            safety_passed=False
        )
        
        is_valid = await interface.validate_response(valid_response)
        assert is_valid
        
        # Invalid JSON response
        invalid_response = LLMResponse(
            content='invalid json',
            prompt_type=PromptType.REASONING,
            template_name="test",
            model="gpt-4-turbo",
            tokens_used=50,
            response_time=1.0,
            confidence_score=0.8,
            validation_passed=False,
            safety_passed=False
        )
        
        is_valid = await interface.validate_response(invalid_response)
        assert not is_valid

class TestBehaviorTree:
    """Test behavior tree functionality"""
    
    @pytest.mark.asyncio
    async def test_sequence_node_success(self):
        """Test sequence node with all children succeeding"""
        
        async def success_action(context):
            return {"success": True}
        
        sequence = SequenceNode("test_sequence")
        sequence.add_child(ActionNode("action1", success_action))
        sequence.add_child(ActionNode("action2", success_action))
        
        context = ExecutionContext(
            agent_id="test_agent",
            environment_state={},
            agent_memory={},
            available_tools=[],
            constraints=[],
            objectives=[]
        )
        
        result = await sequence.execute(context)
        assert result.status == NodeStatus.SUCCESS
        assert len(result.children_results) == 2
    
    @pytest.mark.asyncio
    async def test_sequence_node_failure(self):
        """Test sequence node with one child failing"""
        
        async def success_action(context):
            return {"success": True}
        
        async def failure_action(context):
            return {"success": False}
        
        sequence = SequenceNode("test_sequence")
        sequence.add_child(ActionNode("action1", success_action))
        sequence.add_child(ActionNode("action2", failure_action))
        sequence.add_child(ActionNode("action3", success_action))
        
        context = ExecutionContext(
            agent_id="test_agent",
            environment_state={},
            agent_memory={},
            available_tools=[],
            constraints=[],
            objectives=[]
        )
        
        result = await sequence.execute(context)
        assert result.status == NodeStatus.FAILURE
        assert len(result.children_results) == 2  # Should stop at failure
    
    @pytest.mark.asyncio
    async def test_selector_node_success(self):
        """Test selector node with first child succeeding"""
        
        async def success_action(context):
            return {"success": True}
        
        async def failure_action(context):
            return {"success": False}
        
        selector = SelectorNode("test_selector")
        selector.add_child(ActionNode("action1", failure_action))
        selector.add_child(ActionNode("action2", success_action))
        selector.add_child(ActionNode("action3", success_action))
        
        context = ExecutionContext(
            agent_id="test_agent",
            environment_state={},
            agent_memory={},
            available_tools=[],
            constraints=[],
            objectives=[]
        )
        
        result = await selector.execute(context)
        assert result.status == NodeStatus.SUCCESS
        assert len(result.children_results) == 2  # Should stop at success
    
    @pytest.mark.asyncio
    async def test_behavior_tree_builder(self):
        """Test behavior tree builder"""
        
        async def test_action(context):
            return {"success": True}
        
        async def test_condition(context):
            return True
        
        tree = (BehaviorTreeBuilder("test_tree")
                .sequence("main")
                    .condition("check", test_condition)
                    .action("act", test_action)
                .end()
                .build())
        
        assert tree.name == "test_tree"
        assert tree.root_node.name == "main"
        assert len(tree.root_node.children) == 2
        
        context = ExecutionContext(
            agent_id="test_agent",
            environment_state={},
            agent_memory={},
            available_tools=[],
            constraints=[],
            objectives=[]
        )
        
        result = await tree.execute(context)
        assert result.status == NodeStatus.SUCCESS

class TestPlanning:
    """Test planning system functionality"""
    
    def test_world_state_operations(self):
        """Test world state operations"""
        state = WorldState()
        
        # Test setting and getting facts
        state.set_fact("network_mapped", True)
        assert state.has_fact("network_mapped", True)
        assert state.get_fact("network_mapped") == True
        
        # Test satisfies conditions
        conditions = {"network_mapped": True, "services_discovered": False}
        state.set_fact("services_discovered", False)
        assert state.satisfies(conditions)
        
        # Test copy
        state_copy = state.copy()
        assert state_copy.facts == state.facts
        assert state_copy is not state
    
    def test_goal_satisfaction(self):
        """Test goal satisfaction checking"""
        goal = Goal(
            goal_id="test_goal",
            name="Test Goal",
            conditions={"network_mapped": True, "vulnerabilities_identified": True}
        )
        
        # Unsatisfied state
        state = WorldState(facts={"network_mapped": True, "vulnerabilities_identified": False})
        assert not goal.is_satisfied(state)
        
        # Satisfied state
        state.set_fact("vulnerabilities_identified", True)
        assert goal.is_satisfied(state)
    
    def test_action_execution(self):
        """Test action preconditions and effects"""
        action = Action(
            action_id="test_action",
            name="Test Action",
            action_type=ActionType.RECONNAISSANCE,
            preconditions={"network_access": True},
            effects={"network_mapped": True}
        )
        
        # Test preconditions
        state = WorldState(facts={"network_access": False})
        assert not action.can_execute(state)
        
        state.set_fact("network_access", True)
        assert action.can_execute(state)
        
        # Test effects
        new_state = action.apply_effects(state)
        assert new_state.has_fact("network_mapped", True)
        assert new_state.has_fact("network_access", True)  # Original fact preserved
    
    @pytest.mark.asyncio
    async def test_goap_planner(self):
        """Test GOAP planner"""
        planner = GOAPPlanner(max_depth=5, max_nodes=100)
        
        # Create simple planning scenario
        initial_state = WorldState(facts={"network_access": True})
        
        goal = Goal(
            goal_id="test_goal",
            name="Map Network",
            conditions={"network_mapped": True}
        )
        
        actions = [
            Action(
                action_id="scan_network",
                name="Scan Network",
                action_type=ActionType.RECONNAISSANCE,
                preconditions={"network_access": True},
                effects={"network_mapped": True},
                cost=1.0
            )
        ]
        
        plan = await planner.create_plan(initial_state, goal, actions)
        
        assert plan is not None
        assert len(plan.actions) == 1
        assert plan.actions[0].name == "Scan Network"
        assert plan.is_valid(initial_state)
    
    @pytest.mark.asyncio
    async def test_plan_executor(self):
        """Test plan execution"""
        executor = PlanExecutor()
        
        # Create simple plan
        goal = Goal(
            goal_id="test_goal",
            name="Test Goal",
            conditions={"task_completed": True}
        )
        
        async def test_executor(action, state):
            return {"success": True, "message": "Action completed"}
        
        action = Action(
            action_id="test_action",
            name="Test Action",
            action_type=ActionType.RECONNAISSANCE,
            preconditions={},
            effects={"task_completed": True},
            executor=test_executor
        )
        
        plan = Plan(plan_id="test_plan", goal=goal)
        plan.add_action(action)
        
        initial_state = WorldState()
        result = await executor.execute_plan(plan, initial_state)
        
        assert result["success"]
        assert result["goal_satisfied"]
        assert len(result["executed_actions"]) == 1

class TestIntegratedReasoning:
    """Test integrated reasoning system"""
    
    @pytest.fixture
    def mock_llm_engine(self):
        """Mock LLM reasoning engine"""
        engine = Mock()
        engine.reason_about_situation = AsyncMock(return_value={
            "situation_assessment": "Test assessment",
            "threat_analysis": "Test threat analysis",
            "opportunity_identification": ["Test opportunity"],
            "risk_assessment": {"detection": 0.3},
            "recommended_actions": ["reconnaissance", "exploitation"],
            "confidence_score": 0.8
        })
        engine.validate_action_safety = AsyncMock(return_value={
            "safety_assessment": "safe",
            "constraint_violations": [],
            "risk_level": "low",
            "approval_status": "approved"
        })
        return engine
    
    @pytest.mark.asyncio
    async def test_behavior_tree_integration(self, mock_llm_engine):
        """Test LLM behavior tree integration"""
        integration = LLMBehaviorTreeIntegration(mock_llm_engine)
        
        # Test Red Team tree creation
        red_tree = integration.create_reasoning_tree("red", "recon")
        assert red_tree.name == "red_team_reasoning"
        assert red_tree.root_node is not None
        
        # Test Blue Team tree creation
        blue_tree = integration.create_reasoning_tree("blue", "soc_analyst")
        assert blue_tree.name == "blue_team_reasoning"
        assert blue_tree.root_node is not None
    
    @pytest.mark.asyncio
    async def test_integrated_decision_making(self, mock_llm_engine):
        """Test integrated decision making process"""
        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo",
            api_key="test-key"
        )
        
        # Mock the LLM engine in the integrated system
        system = IntegratedReasoningSystem(llm_config)
        system.llm_engine = mock_llm_engine
        
        # Create reasoning context
        context = ReasoningContext(
            agent_id="test_agent",
            team="red",
            role="recon",
            environment_state={
                "network_topology": {},
                "active_services": [],
                "security_alerts": [],
                "threat_level": "low"
            },
            agent_memory={},
            available_tools=["nmap", "nessus"],
            constraints=["simulation_only"],
            objectives=["map_network"],
            world_state=WorldState(facts={"network_access": True})
        )
        
        decision = await system.make_decision(context)
        
        assert decision.decision_id is not None
        assert decision.reasoning_result is not None
        assert decision.confidence_score > 0
        assert decision.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_safety_validation(self, mock_llm_engine):
        """Test decision safety validation"""
        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo",
            api_key="test-key"
        )
        
        system = IntegratedReasoningSystem(llm_config)
        system.llm_engine = mock_llm_engine
        
        # Create test decision
        decision = IntegratedDecision(
            decision_id="test_decision",
            reasoning_result={"situation_assessment": "test"},
            selected_action=Action(
                action_id="test_action",
                name="Network Scan",
                action_type=ActionType.RECONNAISSANCE
            ),
            behavior_tree_result=None,
            plan_fragment=None,
            confidence_score=0.8,
            execution_time=1.0
        )
        
        context = ReasoningContext(
            agent_id="test_agent",
            team="red",
            role="recon",
            environment_state={},
            agent_memory={},
            available_tools=[],
            constraints=["simulation_only"],
            objectives=[],
            world_state=WorldState()
        )
        
        safety_result = await system.validate_decision_safety(decision, context)
        
        assert safety_result["approval_status"] == "approved"
        assert len(safety_result["constraint_violations"]) == 0

class TestSafetyValidator:
    """Test safety validation functionality"""
    
    @pytest.mark.asyncio
    async def test_action_validation(self):
        """Test action safety validation"""
        validator = SafetyValidator()
        
        # Safe action
        safe_action = {
            "name": "Network Scan",
            "type": "reconnaissance",
            "parameters": {"target": "simulation_network"}
        }
        
        is_safe = await validator.validate_action(safe_action)
        assert is_safe
        
        # Dangerous action
        dangerous_action = {
            "name": "Format Drive",
            "type": "destruction",
            "parameters": {"command": "rm -rf /"}
        }
        
        is_safe = await validator.validate_action(dangerous_action)
        assert not is_safe

@pytest.mark.asyncio
async def test_end_to_end_reasoning():
    """Test complete end-to-end reasoning flow"""
    
    # Mock LLM responses
    mock_llm_engine = Mock()
    mock_llm_engine.reason_about_situation = AsyncMock(return_value={
        "situation_assessment": "Network accessible, no immediate threats detected",
        "threat_analysis": "Low threat environment suitable for reconnaissance",
        "opportunity_identification": ["Network scanning", "Service enumeration"],
        "risk_assessment": {"detection": 0.2, "success": 0.8},
        "recommended_actions": ["network_scan", "vulnerability_scan"],
        "confidence_score": 0.85
    })
    
    mock_llm_engine.validate_action_safety = AsyncMock(return_value={
        "safety_assessment": "safe",
        "constraint_violations": [],
        "risk_level": "low",
        "approval_status": "approved"
    })
    
    # Create integrated reasoning system
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4-turbo",
        api_key="test-key"
    )
    
    system = IntegratedReasoningSystem(llm_config)
    system.llm_engine = mock_llm_engine
    
    # Create comprehensive reasoning context
    context = ReasoningContext(
        agent_id="red_agent_001",
        team="red",
        role="recon",
        environment_state={
            "network_topology": {"subnets": ["192.168.1.0/24", "10.0.0.0/8"]},
            "active_services": [
                {"ip": "192.168.1.10", "port": 80, "service": "http"},
                {"ip": "192.168.1.20", "port": 22, "service": "ssh"}
            ],
            "security_alerts": [],
            "threat_level": "low"
        },
        agent_memory={"previous_scans": [], "discovered_vulnerabilities": []},
        available_tools=["nmap", "nessus", "metasploit"],
        constraints=["simulation_only", "no_destructive_actions"],
        objectives=["map_network", "identify_vulnerabilities", "establish_foothold"],
        world_state=WorldState(facts={
            "network_access": True,
            "network_mapped": False,
            "services_discovered": False,
            "vulnerabilities_identified": False,
            "system_compromised": False
        })
    )
    
    # Make integrated decision
    decision = await system.make_decision(context)
    
    # Validate decision
    assert decision.decision_id is not None
    assert decision.reasoning_result["confidence_score"] == 0.85
    assert decision.selected_action is not None
    assert decision.confidence_score > 0.7
    assert decision.execution_time > 0
    
    # Validate safety
    safety_result = await system.validate_decision_safety(decision, context)
    assert safety_result["approval_status"] == "approved"
    
    # Check that reasoning history is updated
    assert len(context.reasoning_history) == 1
    assert context.reasoning_history[0]["decision_id"] == decision.decision_id

if __name__ == "__main__":
    pytest.main([__file__, "-v"])