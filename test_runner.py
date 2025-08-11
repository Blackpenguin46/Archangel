#!/usr/bin/env python3
"""
Simple test runner for LLM reasoning integration
"""

import asyncio
import sys
import os
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_basic_imports():
    """Test that all modules can be imported"""
    print("Testing basic imports...")
    
    try:
        from agents.llm_interface import (
            LLMConfig, LLMProvider, PromptTemplateManager, LLMReasoningEngine
        )
        print("✓ LLM interface imports successful")
        
        from agents.behavior_tree import (
            BehaviorTree, BehaviorTreeBuilder, ExecutionContext, NodeStatus
        )
        print("✓ Behavior tree imports successful")
        
        from agents.planning import (
            WorldState, Goal, Action, ActionType, GOAPPlanner, PlanningEngine
        )
        print("✓ Planning system imports successful")
        
        from agents.reasoning_integration import (
            IntegratedReasoningSystem, ReasoningContext
        )
        print("✓ Reasoning integration imports successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

async def test_prompt_template_manager():
    """Test prompt template manager functionality"""
    print("\nTesting PromptTemplateManager...")
    
    try:
        from agents.llm_interface import PromptTemplateManager
        
        manager = PromptTemplateManager()
        
        # Test template retrieval
        red_template = manager.get_template("red_team_reasoning")
        assert red_template is not None, "Red team template should exist"
        
        blue_template = manager.get_template("blue_team_reasoning")
        assert blue_template is not None, "Blue team template should exist"
        
        # Test variable validation
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
        assert len(errors) == 0, f"Should have no validation errors, got: {errors}"
        
        print("✓ PromptTemplateManager tests passed")
        return True
        
    except Exception as e:
        print(f"✗ PromptTemplateManager test failed: {e}")
        traceback.print_exc()
        return False

async def test_behavior_tree_builder():
    """Test behavior tree builder"""
    print("\nTesting BehaviorTreeBuilder...")
    
    try:
        from agents.behavior_tree import BehaviorTreeBuilder, ExecutionContext, NodeStatus
        
        async def test_action(context):
            return {"success": True, "message": "Test action executed"}
        
        async def test_condition(context):
            return True
        
        # Build a simple tree
        tree = (BehaviorTreeBuilder("test_tree")
                .sequence("main_sequence")
                    .condition("check_condition", test_condition)
                    .action("execute_action", test_action)
                .end()
                .build())
        
        assert tree.name == "test_tree"
        assert tree.root_node.name == "main_sequence"
        assert len(tree.root_node.children) == 2
        
        # Test execution
        context = ExecutionContext(
            agent_id="test_agent",
            environment_state={},
            agent_memory={},
            available_tools=[],
            constraints=[],
            objectives=[]
        )
        
        result = await tree.execute(context)
        assert result.status == NodeStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
        
        print("✓ BehaviorTreeBuilder tests passed")
        return True
        
    except Exception as e:
        print(f"✗ BehaviorTreeBuilder test failed: {e}")
        traceback.print_exc()
        return False

async def test_planning_system():
    """Test planning system"""
    print("\nTesting Planning System...")
    
    try:
        from agents.planning import WorldState, Goal, Action, ActionType, GOAPPlanner
        
        # Test world state
        state = WorldState()
        state.set_fact("network_access", True)
        assert state.has_fact("network_access", True)
        
        # Test goal
        goal = Goal(
            goal_id="test_goal",
            name="Test Goal",
            conditions={"network_mapped": True}
        )
        
        # Test action
        action = Action(
            action_id="scan_network",
            name="Scan Network",
            action_type=ActionType.RECONNAISSANCE,
            preconditions={"network_access": True},
            effects={"network_mapped": True}
        )
        
        assert action.can_execute(state)
        new_state = action.apply_effects(state)
        assert new_state.has_fact("network_mapped", True)
        
        # Test planner
        planner = GOAPPlanner(max_depth=3, max_nodes=50)
        plan = await planner.create_plan(state, goal, [action])
        
        assert plan is not None, "Planner should create a valid plan"
        assert len(plan.actions) == 1, "Plan should have one action"
        assert plan.actions[0].name == "Scan Network"
        
        print("✓ Planning system tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Planning system test failed: {e}")
        traceback.print_exc()
        return False

async def test_integrated_system():
    """Test integrated reasoning system (without actual LLM calls)"""
    print("\nTesting Integrated Reasoning System...")
    
    try:
        from agents.llm_interface import LLMConfig, LLMProvider
        from agents.reasoning_integration import IntegratedReasoningSystem, ReasoningContext
        from agents.planning import WorldState
        from unittest.mock import Mock, AsyncMock
        
        # Create mock LLM engine
        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo",
            api_key="test-key"
        )
        
        system = IntegratedReasoningSystem(llm_config)
        
        # Mock the LLM engine to avoid actual API calls
        system.llm_engine.reason_about_situation = AsyncMock(return_value={
            "situation_assessment": "Test assessment",
            "threat_analysis": "Test threat analysis",
            "opportunity_identification": ["Test opportunity"],
            "risk_assessment": {"detection": 0.3},
            "recommended_actions": ["reconnaissance"],
            "confidence_score": 0.8
        })
        
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
            available_tools=["nmap"],
            constraints=["simulation_only"],
            objectives=["map_network"],
            world_state=WorldState(facts={"network_access": True})
        )
        
        # Make decision
        decision = await system.make_decision(context)
        
        assert decision.decision_id is not None
        assert decision.reasoning_result is not None
        assert decision.confidence_score > 0
        
        print("✓ Integrated reasoning system tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Integrated reasoning system test failed: {e}")
        traceback.print_exc()
        return False

async def test_base_agent_integration():
    """Test base agent integration with reasoning system"""
    print("\nTesting BaseAgent Integration...")
    
    try:
        from agents.base_agent import BaseAgent, AgentConfig, Team, Role, EnvironmentState
        from datetime import datetime
        from unittest.mock import Mock, AsyncMock
        
        # Create test agent config
        config = AgentConfig(
            agent_id="test_agent",
            team=Team.RED,
            role=Role.RECON,
            name="Test Agent",
            description="Test agent for integration testing"
        )
        
        # Create concrete implementation for testing
        class TestAgent(BaseAgent):
            async def execute_action(self, action):
                return {
                    "action_id": "test_action",
                    "action_type": action.action_type,
                    "success": True,
                    "outcome": "Test action executed successfully",
                    "data": {},
                    "duration": 1.0,
                    "errors": [],
                    "side_effects": [],
                    "confidence": 0.9,
                    "timestamp": datetime.now()
                }
        
        agent = TestAgent(config)
        
        # Mock the reasoning system to avoid LLM calls
        mock_reasoning_system = Mock()
        mock_reasoning_system.make_decision = AsyncMock()
        mock_reasoning_system.make_decision.return_value = Mock(
            reasoning_result={
                "situation_assessment": "Test assessment",
                "threat_analysis": "Test analysis",
                "opportunity_identification": ["Test opportunity"],
                "risk_assessment": {"detection": 0.2},
                "recommended_actions": ["network_scan"],
                "confidence_score": 0.8
            },
            selected_action=Mock(
                name="Network Scan",
                action_type=Mock(value="reconnaissance"),
                parameters={"target": "test_network"},
                duration=5.0
            ),
            confidence_score=0.8
        )
        
        agent.reasoning_system = mock_reasoning_system
        agent.world_state = Mock()
        
        # Test reasoning
        env_state = EnvironmentState(
            timestamp=datetime.now(),
            network_topology={},
            active_services=[],
            security_alerts=[],
            system_logs=[],
            agent_positions={},
            threat_level="low"
        )
        
        reasoning_result = await agent.reason_about_situation(env_state)
        
        assert reasoning_result.situation_assessment == "Test assessment"
        assert reasoning_result.confidence_score == 0.8
        assert len(reasoning_result.recommended_actions) > 0
        
        # Test action planning
        action_plan = await agent.plan_actions(reasoning_result)
        
        assert action_plan.primary_action == "Network Scan"
        assert action_plan.action_type == "reconnaissance"
        
        print("✓ BaseAgent integration tests passed")
        return True
        
    except Exception as e:
        print(f"✗ BaseAgent integration test failed: {e}")
        traceback.print_exc()
        return False

async def run_all_tests():
    """Run all tests"""
    print("Running LLM Reasoning Integration Tests")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_prompt_template_manager,
        test_behavior_tree_builder,
        test_planning_system,
        test_integrated_system,
        test_base_agent_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)