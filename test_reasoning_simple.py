#!/usr/bin/env python3
"""
Simple test for LLM reasoning integration without external dependencies
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_llm_interface():
    """Test LLM interface components"""
    print("Testing LLM Interface...")
    
    try:
        from agents.llm_interface import (
            LLMConfig, LLMProvider, PromptType, PromptTemplateManager
        )
        
        # Test configuration
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo",
            api_key="test-key"
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4-turbo"
        
        # Test template manager
        manager = PromptTemplateManager()
        assert len(manager.templates) > 0
        
        red_template = manager.get_template("red_team_reasoning")
        assert red_template is not None
        assert red_template.prompt_type == PromptType.REASONING
        
        print("✓ LLM Interface tests passed")
        return True
        
    except Exception as e:
        print(f"✗ LLM Interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_behavior_tree():
    """Test behavior tree components"""
    print("Testing Behavior Tree...")
    
    try:
        from agents.behavior_tree import (
            BehaviorTreeBuilder, ExecutionContext, NodeStatus, ActionNode
        )
        
        async def test_action(context):
            return {"success": True, "message": "Test completed"}
        
        async def test_condition(context):
            return True
        
        # Build tree
        tree = (BehaviorTreeBuilder("test_tree")
                .sequence("main")
                    .condition("check", test_condition)
                    .action("act", test_action)
                .end()
                .build())
        
        assert tree.name == "test_tree"
        assert tree.root_node.name == "main"
        
        # Test execution
        context = ExecutionContext(
            agent_id="test",
            environment_state={},
            agent_memory={},
            available_tools=[],
            constraints=[],
            objectives=[]
        )
        
        result = await tree.execute(context)
        assert result.status == NodeStatus.SUCCESS
        
        print("✓ Behavior Tree tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Behavior Tree test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_planning():
    """Test planning components"""
    print("Testing Planning System...")
    
    try:
        from agents.planning import (
            WorldState, Goal, Action, ActionType, GOAPPlanner
        )
        
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
            action_id="scan",
            name="Network Scan",
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
        
        assert plan is not None
        assert len(plan.actions) == 1
        assert plan.actions[0].name == "Network Scan"
        
        print("✓ Planning System tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Planning System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integration():
    """Test reasoning integration"""
    print("Testing Reasoning Integration...")
    
    try:
        from agents.llm_interface import LLMConfig, LLMProvider
        from agents.reasoning_integration import (
            IntegratedReasoningSystem, ReasoningContext, LLMBehaviorTreeIntegration
        )
        from agents.planning import WorldState
        from unittest.mock import Mock, AsyncMock
        
        # Test LLM-BT integration
        mock_llm_engine = Mock()
        integration = LLMBehaviorTreeIntegration(mock_llm_engine)
        
        red_tree = integration.create_reasoning_tree("red", "recon")
        assert red_tree.name == "red_team_reasoning"
        
        blue_tree = integration.create_reasoning_tree("blue", "soc_analyst")
        assert blue_tree.name == "blue_team_reasoning"
        
        # Test integrated system (with mocked LLM)
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo",
            api_key="test-key"
        )
        
        system = IntegratedReasoningSystem(config)
        
        # Mock LLM calls
        system.llm_engine.reason_about_situation = AsyncMock(return_value={
            "situation_assessment": "Test assessment",
            "threat_analysis": "Test analysis",
            "opportunity_identification": ["Test opportunity"],
            "risk_assessment": {"detection": 0.3},
            "recommended_actions": ["reconnaissance"],
            "confidence_score": 0.8
        })
        
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
        
        decision = await system.make_decision(context)
        
        assert decision.decision_id is not None
        assert decision.reasoning_result is not None
        assert decision.confidence_score > 0
        
        print("✓ Reasoning Integration tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Reasoning Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_tests():
    """Run all tests"""
    print("LLM Reasoning Integration Tests")
    print("=" * 40)
    
    tests = [
        test_llm_interface,
        test_behavior_tree,
        test_planning,
        test_integration
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
        print()
    
    print("=" * 40)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(run_tests())
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
    sys.exit(0 if success else 1)