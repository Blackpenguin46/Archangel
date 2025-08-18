"""
Comprehensive tests for LLM reasoning and behavior tree integration
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from agents.llm_reasoning import (
    LLMReasoningEngine, ReasoningContext, ReasoningType, ReasoningResult,
    PromptTemplateManager, LocalLLMInterface, OpenAIInterface
)
from agents.behavior_tree import (
    BehaviorTreeBuilder, ExecutionContext, BehaviorTreeStatus,
    SequenceNode, SelectorNode, ConditionNode, ActionNode, LLMReasoningNode
)
from agents.planning import WorldState, Goal, GOAPPlanner, ActionLibrary


class TestLLMReasoningEngine:
    """Test cases for LLM reasoning engine"""
    
    @pytest.fixture
    def mock_llm_interface(self):
        """Create mock LLM interface"""
        mock_llm = Mock()
        mock_llm.generate_response = AsyncMock(return_value="Test LLM response with analysis and recommendations")
        mock_llm.validate_response = Mock(return_value=True)
        return mock_llm
    
    @pytest.fixture
    def reasoning_engine(self, mock_llm_interface):
        """Create reasoning engine with mock LLM"""
        return LLMReasoningEngine(mock_llm_interface)
    
    @pytest.fixture
    def sample_context(self):
        """Create sample reasoning context"""
        return ReasoningContext(
            agent_id="test_agent",
            team="red",
            role="reconnaissance",
            current_phase="recon",
            environment_state={"network_discovered": False},
            objectives=["Discover network topology"],
            available_tools=["nmap", "masscan"],
            memory_context={"previous_scans": []}
        )
    
    @pytest.mark.asyncio
    async def test_reasoning_engine_initialization(self, reasoning_engine):
        """Test reasoning engine initialization"""
        assert reasoning_engine.primary_llm is not None
        assert reasoning_engine.template_manager is not None
        assert isinstance(reasoning_engine.reasoning_history, list)
    
    @pytest.mark.asyncio
    async def test_basic_reasoning(self, reasoning_engine, sample_context):
        """Test basic reasoning functionality"""
        result = await reasoning_engine.reason(sample_context, ReasoningType.TACTICAL)
        
        assert isinstance(result, ReasoningResult)
        assert result.reasoning_type == ReasoningType.TACTICAL
        assert result.confidence >= 0.0 and result.confidence <= 1.0
        assert isinstance(result.recommended_actions, list)
        assert len(result.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_reasoning_with_fallback(self):
        """Test reasoning with fallback LLM"""
        primary_llm = Mock()
        primary_llm.generate_response = AsyncMock(side_effect=Exception("Primary LLM failed"))
        
        fallback_llm = Mock()
        fallback_llm.generate_response = AsyncMock(return_value="Fallback response")
        fallback_llm.validate_response = Mock(return_value=True)
        
        engine = LLMReasoningEngine(primary_llm, fallback_llm)
        context = ReasoningContext(
            agent_id="test", team="red", role="test", current_phase="test",
            environment_state={}, objectives=[], available_tools=[], memory_context={}
        )
        
        result = await engine.reason(context, ReasoningType.TACTICAL)
        
        # Should succeed using fallback
        assert isinstance(result, ReasoningResult)
        fallback_llm.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_template_selection(self, reasoning_engine):
        """Test template selection based on team and role"""
        # Test Red Team template selection
        red_context = ReasoningContext(
            agent_id="red_agent", team="red", role="reconnaissance", current_phase="recon",
            environment_state={}, objectives=[], available_tools=[], memory_context={}
        )
        
        template_name = reasoning_engine._select_template("red", "reconnaissance")
        assert template_name == "red_recon"
        
        # Test Blue Team template selection
        blue_context = ReasoningContext(
            agent_id="blue_agent", team="blue", role="soc_analyst", current_phase="monitor",
            environment_state={}, objectives=[], available_tools=[], memory_context={}
        )
        
        template_name = reasoning_engine._select_template("blue", "soc_analyst")
        assert template_name == "blue_soc"
    
    @pytest.mark.asyncio
    async def test_response_validation(self, reasoning_engine):
        """Test response validation and safety checking"""
        # Test with harmful content
        reasoning_engine.primary_llm.validate_response = Mock(return_value=False)
        reasoning_engine.fallback_llm = Mock()
        reasoning_engine.fallback_llm.generate_response = AsyncMock(return_value="Safe response")
        reasoning_engine.fallback_llm.validate_response = Mock(return_value=True)
        
        context = ReasoningContext(
            agent_id="test", team="red", role="test", current_phase="test",
            environment_state={}, objectives=[], available_tools=[], memory_context={}
        )
        
        result = await reasoning_engine.reason(context, ReasoningType.TACTICAL)
        
        # Should use fallback due to validation failure
        assert isinstance(result, ReasoningResult)
        reasoning_engine.fallback_llm.generate_response.assert_called_once()


class TestPromptTemplateManager:
    """Test cases for prompt template management"""
    
    @pytest.fixture
    def template_manager(self):
        """Create template manager"""
        return PromptTemplateManager()
    
    def test_template_manager_initialization(self, template_manager):
        """Test template manager initialization with default templates"""
        assert "red_recon" in template_manager.templates
        assert "blue_soc" in template_manager.templates
    
    def test_get_template(self, template_manager):
        """Test getting templates by name"""
        template = template_manager.get_template("red_recon")
        assert template.template_name == "red_recon"
        assert "reconnaissance" in template.template.lower()
    
    def test_get_nonexistent_template(self, template_manager):
        """Test getting non-existent template raises error"""
        with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
            template_manager.get_template("nonexistent")
    
    def test_template_formatting(self, template_manager):
        """Test template formatting with variables"""
        template = template_manager.get_template("red_recon")
        
        formatted = template.format(
            role="reconnaissance",
            current_phase="recon",
            objectives="[]",
            environment_state="{}",
            available_tools="[]",
            memory_context="{}"
        )
        
        assert "reconnaissance" in formatted
        assert "recon" in formatted


class TestBehaviorTreeIntegration:
    """Test cases for behavior tree integration with LLM reasoning"""
    
    @pytest.fixture
    def mock_reasoning_engine(self):
        """Create mock reasoning engine"""
        engine = Mock()
        engine.reason = AsyncMock(return_value=ReasoningResult(
            reasoning_type=ReasoningType.TACTICAL,
            decision="Test decision",
            confidence=0.8,
            reasoning_chain=["Analysis step 1", "Analysis step 2"],
            recommended_actions=[{"action": "scan_network", "priority": "high"}],
            risk_assessment={"overall": 0.3},
            metadata={}
        ))
        return engine
    
    @pytest.fixture
    def behavior_tree_builder(self, mock_reasoning_engine):
        """Create behavior tree builder with mock reasoning engine"""
        return BehaviorTreeBuilder(mock_reasoning_engine)
    
    @pytest.fixture
    def sample_execution_context(self):
        """Create sample execution context"""
        world_state = WorldState({
            "network_scanned": False,
            "vulnerabilities_found": False,
            "system_compromised": False
        })
        
        return ExecutionContext(
            agent_id="test_agent",
            team="red",
            role="penetration_tester",
            current_phase="recon",
            world_state=world_state,
            objectives=["Compromise target"],
            available_tools=["nmap", "metasploit"],
            memory_context={},
            execution_data={}
        )
    
    @pytest.mark.asyncio
    async def test_llm_reasoning_node(self, mock_reasoning_engine, sample_execution_context):
        """Test LLM reasoning node execution"""
        node = LLMReasoningNode("TestReasoning", mock_reasoning_engine, ReasoningType.TACTICAL)
        
        result = await node.execute(sample_execution_context)
        
        assert result == BehaviorTreeStatus.SUCCESS  # High confidence (0.8)
        assert node.last_reasoning_result is not None
        assert "TestReasoning_reasoning" in sample_execution_context.execution_data
        mock_reasoning_engine.reason.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sequence_node_execution(self, sample_execution_context):
        """Test sequence node with multiple children"""
        sequence = SequenceNode("TestSequence")
        
        # Add condition that should succeed
        condition1 = ConditionNode("Condition1", lambda ctx: True)
        
        # Add action that should succeed
        action1 = ActionNode("Action1", lambda ctx: True)
        
        sequence.add_child(condition1)
        sequence.add_child(action1)
        
        result = await sequence.execute(sample_execution_context)
        
        assert result == BehaviorTreeStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_sequence_node_failure(self, sample_execution_context):
        """Test sequence node failure when child fails"""
        sequence = SequenceNode("TestSequence")
        
        # Add condition that should fail
        condition1 = ConditionNode("Condition1", lambda ctx: False)
        
        # Add action that should not be reached
        action1 = ActionNode("Action1", lambda ctx: True)
        
        sequence.add_child(condition1)
        sequence.add_child(action1)
        
        result = await sequence.execute(sample_execution_context)
        
        assert result == BehaviorTreeStatus.FAILURE
    
    @pytest.mark.asyncio
    async def test_selector_node_execution(self, sample_execution_context):
        """Test selector node execution"""
        selector = SelectorNode("TestSelector")
        
        # Add condition that should fail
        condition1 = ConditionNode("Condition1", lambda ctx: False)
        
        # Add condition that should succeed
        condition2 = ConditionNode("Condition2", lambda ctx: True)
        
        selector.add_child(condition1)
        selector.add_child(condition2)
        
        result = await selector.execute(sample_execution_context)
        
        assert result == BehaviorTreeStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_red_team_behavior_tree(self, behavior_tree_builder, sample_execution_context):
        """Test Red Team behavior tree execution"""
        red_tree = behavior_tree_builder.build_red_team_tree()
        
        result = await red_tree.execute(sample_execution_context)
        
        # Should execute successfully (at least one branch should succeed)
        assert result in [BehaviorTreeStatus.SUCCESS, BehaviorTreeStatus.RUNNING]
    
    @pytest.mark.asyncio
    async def test_blue_team_behavior_tree(self, behavior_tree_builder):
        """Test Blue Team behavior tree execution"""
        blue_tree = behavior_tree_builder.build_blue_team_tree()
        
        # Create Blue Team context
        world_state = WorldState({
            "siem_active": True,
            "monitoring_enabled": True,
            "threat_detected": False
        })
        
        blue_context = ExecutionContext(
            agent_id="blue_agent",
            team="blue",
            role="soc_analyst",
            current_phase="monitor",
            world_state=world_state,
            objectives=["Monitor threats"],
            available_tools=["siem", "firewall"],
            memory_context={},
            execution_data={}
        )
        
        result = await blue_tree.execute(blue_context)
        
        # Parallel node should succeed if at least one child succeeds
        assert result in [BehaviorTreeStatus.SUCCESS, BehaviorTreeStatus.RUNNING]


class TestGOAPIntegration:
    """Test cases for GOAP integration with behavior trees"""
    
    @pytest.fixture
    def goap_planner(self):
        """Create GOAP planner with actions"""
        planner = GOAPPlanner()
        for action in ActionLibrary.get_red_team_actions():
            planner.add_action(action)
        return planner
    
    def test_goap_planning_basic(self, goap_planner):
        """Test basic GOAP planning"""
        initial_state = WorldState({
            "has_target": True,
            "tools_available": True,
            "network_scanned": False,
            "services_discovered": False,
            "vulnerabilities_identified": False,
            "system_compromised": False
        })
        
        goal = Goal("compromise_system", {
            "system_compromised": True,
            "access_gained": True
        })
        
        plan = goap_planner.plan(initial_state, goal)
        
        assert plan is not None
        assert len(plan) > 0
        assert all(hasattr(action, 'name') for action in plan)
    
    def test_goap_planning_already_satisfied(self, goap_planner):
        """Test GOAP planning when goal is already satisfied"""
        initial_state = WorldState({
            "system_compromised": True,
            "access_gained": True
        })
        
        goal = Goal("compromise_system", {
            "system_compromised": True,
            "access_gained": True
        })
        
        plan = goap_planner.plan(initial_state, goal)
        
        assert plan == []  # Empty plan since goal is already satisfied
    
    def test_goap_planning_impossible_goal(self, goap_planner):
        """Test GOAP planning with impossible goal"""
        initial_state = WorldState({
            "has_target": False,  # No target available
            "tools_available": False  # No tools available
        })
        
        goal = Goal("compromise_system", {
            "system_compromised": True,
            "access_gained": True
        })
        
        plan = goap_planner.plan(initial_state, goal)
        
        assert plan is None  # No plan possible


class TestIntegrationScenarios:
    """Integration test scenarios combining all components"""
    
    @pytest.mark.asyncio
    async def test_full_red_team_scenario(self):
        """Test complete Red Team scenario with LLM reasoning and behavior trees"""
        # Create mock LLM interface
        mock_llm = Mock()
        mock_llm.generate_response = AsyncMock(return_value="""
        Situation Analysis: Target network appears to be a typical enterprise environment.
        Risk Assessment: Low risk of detection during initial reconnaissance.
        Recommended Actions:
        1. Perform network scan using nmap
        2. Enumerate services on discovered hosts
        3. Identify potential vulnerabilities
        """)
        mock_llm.validate_response = Mock(return_value=True)
        
        # Create reasoning engine
        reasoning_engine = LLMReasoningEngine(mock_llm)
        
        # Create behavior tree
        builder = BehaviorTreeBuilder(reasoning_engine)
        red_tree = builder.build_red_team_tree()
        
        # Create execution context
        world_state = WorldState({
            "has_target": True,
            "tools_available": True,
            "network_scanned": False,
            "services_discovered": False
        })
        
        context = ExecutionContext(
            agent_id="red_agent_001",
            team="red",
            role="penetration_tester",
            current_phase="recon",
            world_state=world_state,
            objectives=["Compromise target system"],
            available_tools=["nmap", "metasploit"],
            memory_context={},
            execution_data={}
        )
        
        # Execute behavior tree
        result = await red_tree.execute(context)
        
        # Verify execution
        assert result in [BehaviorTreeStatus.SUCCESS, BehaviorTreeStatus.RUNNING]
        assert mock_llm.generate_response.called
        
        # Check that reasoning results were stored
        reasoning_keys = [k for k in context.execution_data.keys() if 'reasoning' in k]
        assert len(reasoning_keys) > 0
    
    @pytest.mark.asyncio
    async def test_full_blue_team_scenario(self):
        """Test complete Blue Team scenario with LLM reasoning and behavior trees"""
        # Create mock LLM interface
        mock_llm = Mock()
        mock_llm.generate_response = AsyncMock(return_value="""
        Threat Analysis: Multiple suspicious activities detected in network logs.
        Alert Prioritization: High priority - potential port scanning activity.
        Response Strategy:
        1. Block suspicious source IP addresses
        2. Increase monitoring on affected systems
        3. Notify security team
        """)
        mock_llm.validate_response = Mock(return_value=True)
        
        # Create reasoning engine
        reasoning_engine = LLMReasoningEngine(mock_llm)
        
        # Create behavior tree
        builder = BehaviorTreeBuilder(reasoning_engine)
        blue_tree = builder.build_blue_team_tree()
        
        # Create execution context
        world_state = WorldState({
            "siem_active": True,
            "monitoring_enabled": True,
            "alerts_pending": True,
            "threat_detected": False
        })
        
        context = ExecutionContext(
            agent_id="blue_agent_001",
            team="blue",
            role="soc_analyst",
            current_phase="monitor",
            world_state=world_state,
            objectives=["Detect and respond to threats"],
            available_tools=["siem", "firewall", "ids"],
            memory_context={},
            execution_data={}
        )
        
        # Execute behavior tree
        result = await blue_tree.execute(context)
        
        # Verify execution
        assert result in [BehaviorTreeStatus.SUCCESS, BehaviorTreeStatus.RUNNING]
        assert mock_llm.generate_response.called
        
        # Check that monitoring was performed
        assert context.execution_data.get("alerts_monitored", False)


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.asyncio
    async def test_reasoning_performance(self):
        """Test reasoning engine performance under load"""
        mock_llm = Mock()
        mock_llm.generate_response = AsyncMock(return_value="Quick response")
        mock_llm.validate_response = Mock(return_value=True)
        
        engine = LLMReasoningEngine(mock_llm)
        
        context = ReasoningContext(
            agent_id="perf_test", team="red", role="test", current_phase="test",
            environment_state={}, objectives=[], available_tools=[], memory_context={}
        )
        
        # Measure time for multiple reasoning operations
        start_time = datetime.now()
        
        tasks = []
        for i in range(10):
            task = engine.reason(context, ReasoningType.TACTICAL)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete 10 reasoning operations in reasonable time
        assert duration < 5.0  # 5 seconds max
        assert len(results) == 10
        assert all(isinstance(r, ReasoningResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_behavior_tree_performance(self):
        """Test behavior tree performance with complex trees"""
        mock_llm = Mock()
        mock_llm.generate_response = AsyncMock(return_value="Performance test response")
        mock_llm.validate_response = Mock(return_value=True)
        
        reasoning_engine = LLMReasoningEngine(mock_llm)
        builder = BehaviorTreeBuilder(reasoning_engine)
        
        # Create complex behavior tree
        tree = builder.build_red_team_tree()
        
        world_state = WorldState({"has_target": True, "tools_available": True})
        context = ExecutionContext(
            agent_id="perf_agent", team="red", role="test", current_phase="recon",
            world_state=world_state, objectives=[], available_tools=[], 
            memory_context={}, execution_data={}
        )
        
        # Measure execution time
        start_time = datetime.now()
        
        result = await tree.execute(context)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time
        assert duration < 10.0  # 10 seconds max
        assert result in [BehaviorTreeStatus.SUCCESS, BehaviorTreeStatus.RUNNING, BehaviorTreeStatus.FAILURE]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])