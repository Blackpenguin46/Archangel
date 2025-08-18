#!/usr/bin/env python3
"""
Simple test script for LLM reasoning and behavior tree integration
"""

import asyncio
import sys
from agents.llm_reasoning import LLMReasoningEngine, ReasoningContext, ReasoningType, LocalLLMInterface
from agents.behavior_tree import BehaviorTreeBuilder, ExecutionContext, BehaviorTreeStatus
from agents.planning import WorldState, GOAPPlanner, ActionLibrary


async def test_llm_reasoning():
    """Test basic LLM reasoning functionality"""
    print("Testing LLM Reasoning...")
    
    # Create local LLM interface
    llm_interface = LocalLLMInterface("test-model")
    reasoning_engine = LLMReasoningEngine(llm_interface)
    
    # Create test context
    context = ReasoningContext(
        agent_id="test_agent",
        team="red",
        role="reconnaissance",
        current_phase="recon",
        environment_state={"network_discovered": False},
        objectives=["Test objective"],
        available_tools=["nmap"],
        memory_context={}
    )
    
    # Test reasoning
    result = await reasoning_engine.reason(context, ReasoningType.TACTICAL)
    
    assert result is not None
    assert result.confidence >= 0.0 and result.confidence <= 1.0
    assert len(result.recommended_actions) > 0
    
    print("✓ LLM reasoning test passed")


async def test_behavior_trees():
    """Test behavior tree execution"""
    print("Testing Behavior Trees...")
    
    # Create reasoning engine
    llm_interface = LocalLLMInterface("test-model")
    reasoning_engine = LLMReasoningEngine(llm_interface)
    
    # Create behavior tree builder
    builder = BehaviorTreeBuilder(reasoning_engine)
    
    # Test Red Team tree
    red_tree = builder.build_red_team_tree()
    
    world_state = WorldState({
        "has_target": True,
        "tools_available": True,
        "network_scanned": False
    })
    
    context = ExecutionContext(
        agent_id="test_red",
        team="red",
        role="tester",
        current_phase="recon",
        world_state=world_state,
        objectives=["Test"],
        available_tools=["nmap"],
        memory_context={},
        execution_data={}
    )
    
    result = await red_tree.execute(context)
    
    assert result in [BehaviorTreeStatus.SUCCESS, BehaviorTreeStatus.RUNNING, BehaviorTreeStatus.FAILURE]
    
    print("✓ Behavior tree test passed")


def test_goap_planning():
    """Test GOAP planning functionality"""
    print("Testing GOAP Planning...")
    
    # Create planner
    planner = GOAPPlanner()
    
    # Add actions
    for action in ActionLibrary.get_red_team_actions():
        planner.add_action(action)
    
    # Test planning
    initial_state = WorldState({
        "has_target": True,
        "tools_available": True,
        "network_scanned": False,
        "system_compromised": False
    })
    
    from agents.planning import Goal
    goal = Goal("test_goal", {"system_compromised": True, "access_gained": True})
    
    plan = planner.plan(initial_state, goal)
    
    assert plan is not None
    assert len(plan) > 0
    
    print("✓ GOAP planning test passed")


async def main():
    """Run all tests"""
    print("Running LLM Integration Tests...")
    print("=" * 40)
    
    try:
        await test_llm_reasoning()
        await test_behavior_trees()
        test_goap_planning()
        
        print("=" * 40)
        print("All tests passed! ✓")
        return 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)