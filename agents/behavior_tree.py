"""
Behavior Tree Framework with LLM Integration

This module provides a comprehensive behavior tree implementation that integrates
with LLM reasoning for autonomous agent decision-making.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
import json

from .llm_reasoning import LLMReasoningEngine, ReasoningContext, ReasoningType, NodeResult
from .planning import GOAPPlanner, WorldState, Goal, ActionLibrary, PlanExecutor

logger = logging.getLogger(__name__)


class BehaviorTreeStatus(Enum):
    """Status of behavior tree execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"
    INVALID = "invalid"


@dataclass
class ExecutionContext:
    """Context for behavior tree execution"""
    agent_id: str
    team: str
    role: str
    current_phase: str
    world_state: WorldState
    objectives: List[str]
    available_tools: List[str]
    memory_context: Dict[str, Any]
    execution_data: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM reasoning"""
        return {
            'agent_id': self.agent_id,
            'team': self.team,
            'role': self.role,
            'current_phase': self.current_phase,
            'environment_state': self.world_state.facts,
            'objectives': self.objectives,
            'available_tools': self.available_tools,
            'memory_context': self.memory_context,
            'execution_data': self.execution_data
        }


class BehaviorNode(ABC):
    """Abstract base class for all behavior tree nodes"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.parent: Optional['BehaviorNode'] = None
        self.children: List['BehaviorNode'] = []
        self.status = BehaviorTreeStatus.INVALID
        self.last_execution_time: Optional[datetime] = None
        self.execution_count = 0
    
    def add_child(self, child: 'BehaviorNode') -> 'BehaviorNode':
        """Add a child node"""
        child.parent = self
        self.children.append(child)
        return self
    
    def remove_child(self, child: 'BehaviorNode') -> bool:
        """Remove a child node"""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            return True
        return False
    
    @abstractmethod
    async def execute(self, context: ExecutionContext) -> BehaviorTreeStatus:
        """Execute this node"""
        pass
    
    def reset(self):
        """Reset this node and all children"""
        self.status = BehaviorTreeStatus.INVALID
        for child in self.children:
            child.reset()
    
    def get_status(self) -> BehaviorTreeStatus:
        """Get current status"""
        return self.status
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class CompositeNode(BehaviorNode):
    """Base class for composite nodes (nodes with children)"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.current_child_index = 0
    
    def reset(self):
        """Reset composite node"""
        super().reset()
        self.current_child_index = 0


class SequenceNode(CompositeNode):
    """Sequence node - executes children in order until one fails"""
    
    async def execute(self, context: ExecutionContext) -> BehaviorTreeStatus:
        """Execute children in sequence"""
        self.execution_count += 1
        self.last_execution_time = datetime.now()
        
        logger.debug(f"Executing sequence node: {self.name}")
        
        # Execute children in order
        while self.current_child_index < len(self.children):
            child = self.children[self.current_child_index]
            child_status = await child.execute(context)
            
            if child_status == BehaviorTreeStatus.FAILURE:
                logger.debug(f"Sequence {self.name} failed at child {child.name}")
                self.status = BehaviorTreeStatus.FAILURE
                self.reset()
                return self.status
            elif child_status == BehaviorTreeStatus.RUNNING:
                logger.debug(f"Sequence {self.name} running at child {child.name}")
                self.status = BehaviorTreeStatus.RUNNING
                return self.status
            else:  # SUCCESS
                self.current_child_index += 1
        
        # All children succeeded
        logger.debug(f"Sequence {self.name} completed successfully")
        self.status = BehaviorTreeStatus.SUCCESS
        self.reset()
        return self.status


class SelectorNode(CompositeNode):
    """Selector node - executes children until one succeeds"""
    
    async def execute(self, context: ExecutionContext) -> BehaviorTreeStatus:
        """Execute children until one succeeds"""
        self.execution_count += 1
        self.last_execution_time = datetime.now()
        
        logger.debug(f"Executing selector node: {self.name}")
        
        # Try children in order
        while self.current_child_index < len(self.children):
            child = self.children[self.current_child_index]
            child_status = await child.execute(context)
            
            if child_status == BehaviorTreeStatus.SUCCESS:
                logger.debug(f"Selector {self.name} succeeded at child {child.name}")
                self.status = BehaviorTreeStatus.SUCCESS
                self.reset()
                return self.status
            elif child_status == BehaviorTreeStatus.RUNNING:
                logger.debug(f"Selector {self.name} running at child {child.name}")
                self.status = BehaviorTreeStatus.RUNNING
                return self.status
            else:  # FAILURE
                self.current_child_index += 1
        
        # All children failed
        logger.debug(f"Selector {self.name} failed - all children failed")
        self.status = BehaviorTreeStatus.FAILURE
        self.reset()
        return self.status


class ParallelNode(CompositeNode):
    """Parallel node - executes all children simultaneously"""
    
    def __init__(self, name: str, success_threshold: int = 1, description: str = ""):
        super().__init__(name, description)
        self.success_threshold = success_threshold
        self.child_statuses: Dict[int, BehaviorTreeStatus] = {}
    
    async def execute(self, context: ExecutionContext) -> BehaviorTreeStatus:
        """Execute all children in parallel"""
        self.execution_count += 1
        self.last_execution_time = datetime.now()
        
        logger.debug(f"Executing parallel node: {self.name}")
        
        # Execute all children concurrently
        tasks = []
        for i, child in enumerate(self.children):
            task = asyncio.create_task(child.execute(context))
            tasks.append((i, task))
        
        # Wait for all tasks to complete
        success_count = 0
        failure_count = 0
        running_count = 0
        
        for i, task in tasks:
            try:
                child_status = await task
                self.child_statuses[i] = child_status
                
                if child_status == BehaviorTreeStatus.SUCCESS:
                    success_count += 1
                elif child_status == BehaviorTreeStatus.FAILURE:
                    failure_count += 1
                else:  # RUNNING
                    running_count += 1
            except Exception as e:
                logger.error(f"Error in parallel child {i}: {e}")
                self.child_statuses[i] = BehaviorTreeStatus.FAILURE
                failure_count += 1
        
        # Determine overall status
        if success_count >= self.success_threshold:
            self.status = BehaviorTreeStatus.SUCCESS
        elif running_count > 0:
            self.status = BehaviorTreeStatus.RUNNING
        else:
            self.status = BehaviorTreeStatus.FAILURE
        
        logger.debug(f"Parallel {self.name} result: {self.status} "
                    f"(success: {success_count}, failure: {failure_count}, running: {running_count})")
        
        if self.status != BehaviorTreeStatus.RUNNING:
            self.reset()
        
        return self.status
    
    def reset(self):
        """Reset parallel node"""
        super().reset()
        self.child_statuses.clear()


class DecoratorNode(BehaviorNode):
    """Base class for decorator nodes (nodes that modify child behavior)"""
    
    def __init__(self, name: str, child: Optional[BehaviorNode] = None, description: str = ""):
        super().__init__(name, description)
        if child:
            self.add_child(child)
    
    @property
    def child(self) -> Optional[BehaviorNode]:
        """Get the single child node"""
        return self.children[0] if self.children else None


class InverterNode(DecoratorNode):
    """Inverter node - inverts the result of its child"""
    
    async def execute(self, context: ExecutionContext) -> BehaviorTreeStatus:
        """Execute child and invert result"""
        self.execution_count += 1
        self.last_execution_time = datetime.now()
        
        if not self.child:
            self.status = BehaviorTreeStatus.FAILURE
            return self.status
        
        child_status = await self.child.execute(context)
        
        if child_status == BehaviorTreeStatus.SUCCESS:
            self.status = BehaviorTreeStatus.FAILURE
        elif child_status == BehaviorTreeStatus.FAILURE:
            self.status = BehaviorTreeStatus.SUCCESS
        else:  # RUNNING
            self.status = BehaviorTreeStatus.RUNNING
        
        return self.status


class RetryNode(DecoratorNode):
    """Retry node - retries child execution up to max_attempts"""
    
    def __init__(self, name: str, max_attempts: int = 3, child: Optional[BehaviorNode] = None, description: str = ""):
        super().__init__(name, child, description)
        self.max_attempts = max_attempts
        self.current_attempts = 0
    
    async def execute(self, context: ExecutionContext) -> BehaviorTreeStatus:
        """Execute child with retry logic"""
        self.execution_count += 1
        self.last_execution_time = datetime.now()
        
        if not self.child:
            self.status = BehaviorTreeStatus.FAILURE
            return self.status
        
        while self.current_attempts < self.max_attempts:
            child_status = await self.child.execute(context)
            
            if child_status == BehaviorTreeStatus.SUCCESS:
                self.status = BehaviorTreeStatus.SUCCESS
                self.current_attempts = 0
                return self.status
            elif child_status == BehaviorTreeStatus.RUNNING:
                self.status = BehaviorTreeStatus.RUNNING
                return self.status
            else:  # FAILURE
                self.current_attempts += 1
                logger.debug(f"Retry {self.name} attempt {self.current_attempts}/{self.max_attempts}")
        
        # Max attempts reached
        self.status = BehaviorTreeStatus.FAILURE
        self.current_attempts = 0
        return self.status
    
    def reset(self):
        """Reset retry node"""
        super().reset()
        self.current_attempts = 0


class ConditionNode(BehaviorNode):
    """Condition node - evaluates a condition function"""
    
    def __init__(self, name: str, condition_func: Callable[[ExecutionContext], bool], description: str = ""):
        super().__init__(name, description)
        self.condition_func = condition_func
    
    async def execute(self, context: ExecutionContext) -> BehaviorTreeStatus:
        """Evaluate condition"""
        self.execution_count += 1
        self.last_execution_time = datetime.now()
        
        try:
            if self.condition_func(context):
                self.status = BehaviorTreeStatus.SUCCESS
            else:
                self.status = BehaviorTreeStatus.FAILURE
        except Exception as e:
            logger.error(f"Condition evaluation error in {self.name}: {e}")
            self.status = BehaviorTreeStatus.FAILURE
        
        return self.status


class ActionNode(BehaviorNode):
    """Action node - executes an action function"""
    
    def __init__(self, name: str, action_func: Callable[[ExecutionContext], Any], description: str = ""):
        super().__init__(name, description)
        self.action_func = action_func
        self.is_running = False
    
    async def execute(self, context: ExecutionContext) -> BehaviorTreeStatus:
        """Execute action"""
        self.execution_count += 1
        self.last_execution_time = datetime.now()
        
        try:
            if asyncio.iscoroutinefunction(self.action_func):
                result = await self.action_func(context)
            else:
                result = self.action_func(context)
            
            if result is True:
                self.status = BehaviorTreeStatus.SUCCESS
            elif result is False:
                self.status = BehaviorTreeStatus.FAILURE
            else:  # None or other - still running
                self.status = BehaviorTreeStatus.RUNNING
            
        except Exception as e:
            logger.error(f"Action execution error in {self.name}: {e}")
            self.status = BehaviorTreeStatus.FAILURE
        
        return self.status


class LLMReasoningNode(BehaviorNode):
    """Node that uses LLM reasoning for decision making"""
    
    def __init__(self, name: str, reasoning_engine: LLMReasoningEngine, 
                 reasoning_type: ReasoningType, description: str = ""):
        super().__init__(name, description)
        self.reasoning_engine = reasoning_engine
        self.reasoning_type = reasoning_type
        self.last_reasoning_result = None
    
    async def execute(self, context: ExecutionContext) -> BehaviorTreeStatus:
        """Execute LLM reasoning"""
        self.execution_count += 1
        self.last_execution_time = datetime.now()
        
        try:
            # Create reasoning context
            reasoning_context = ReasoningContext(
                agent_id=context.agent_id,
                team=context.team,
                role=context.role,
                current_phase=context.current_phase,
                environment_state=context.world_state.facts,
                objectives=context.objectives,
                available_tools=context.available_tools,
                memory_context=context.memory_context
            )
            
            # Perform reasoning
            reasoning_result = await self.reasoning_engine.reason(reasoning_context, self.reasoning_type)
            self.last_reasoning_result = reasoning_result
            
            # Store reasoning result in context
            context.execution_data[f'{self.name}_reasoning'] = reasoning_result
            
            # Determine success based on confidence threshold
            if reasoning_result.confidence > 0.7:
                self.status = BehaviorTreeStatus.SUCCESS
            elif reasoning_result.confidence > 0.3:
                self.status = BehaviorTreeStatus.RUNNING
            else:
                self.status = BehaviorTreeStatus.FAILURE
            
            logger.debug(f"LLM reasoning {self.name}: confidence={reasoning_result.confidence}, status={self.status}")
            
        except Exception as e:
            logger.error(f"LLM reasoning error in {self.name}: {e}")
            self.status = BehaviorTreeStatus.FAILURE
        
        return self.status


class GOAPPlanningNode(BehaviorNode):
    """Node that uses GOAP planning for action selection"""
    
    def __init__(self, name: str, planner: GOAPPlanner, goal: Goal, description: str = ""):
        super().__init__(name, description)
        self.planner = planner
        self.goal = goal
        self.executor = PlanExecutor()
        self.current_plan = None
    
    async def execute(self, context: ExecutionContext) -> BehaviorTreeStatus:
        """Execute GOAP planning and execution"""
        self.execution_count += 1
        self.last_execution_time = datetime.now()
        
        try:
            # Check if we need a new plan
            if self.current_plan is None or self.executor.is_plan_complete():
                # Create new plan
                plan = self.planner.plan(context.world_state, self.goal)
                
                if plan:
                    self.current_plan = plan
                    self.executor.set_plan(plan, context.to_dict())
                    logger.info(f"GOAP {self.name}: Created plan with {len(plan)} actions")
                else:
                    logger.warning(f"GOAP {self.name}: No plan found for goal {self.goal.name}")
                    self.status = BehaviorTreeStatus.FAILURE
                    return self.status
            
            # Execute next action in plan
            if not self.executor.is_plan_complete():
                result = await self.executor.execute_next_action()
                
                if result is None:  # Plan complete
                    self.status = BehaviorTreeStatus.SUCCESS
                    self.current_plan = None
                elif result.name == "SUCCESS":
                    # Continue with next action
                    self.status = BehaviorTreeStatus.RUNNING
                elif result.name == "IN_PROGRESS":
                    self.status = BehaviorTreeStatus.RUNNING
                else:  # FAILURE
                    self.status = BehaviorTreeStatus.FAILURE
                    self.current_plan = None
            else:
                self.status = BehaviorTreeStatus.SUCCESS
                self.current_plan = None
            
        except Exception as e:
            logger.error(f"GOAP planning error in {self.name}: {e}")
            self.status = BehaviorTreeStatus.FAILURE
            self.current_plan = None
        
        return self.status


class BehaviorTreeBuilder:
    """Builder for creating behavior trees with LLM integration"""
    
    def __init__(self, reasoning_engine: LLMReasoningEngine):
        self.reasoning_engine = reasoning_engine
        self.planner = GOAPPlanner()
    
    def build_red_team_tree(self) -> BehaviorNode:
        """Build comprehensive Red Team behavior tree"""
        root = SelectorNode("RedTeamRoot", "Main Red Team decision tree")
        
        # Phase-based sequences
        recon_phase = self._build_recon_phase()
        exploit_phase = self._build_exploit_phase()
        persist_phase = self._build_persistence_phase()
        
        root.add_child(recon_phase)
        root.add_child(exploit_phase)
        root.add_child(persist_phase)
        
        return root
    
    def build_blue_team_tree(self) -> BehaviorNode:
        """Build comprehensive Blue Team behavior tree"""
        root = ParallelNode("BlueTeamRoot", success_threshold=1, 
                           description="Main Blue Team parallel operations")
        
        # Continuous monitoring
        monitoring_tree = self._build_monitoring_tree()
        
        # Incident response
        response_tree = self._build_response_tree()
        
        # Threat hunting
        hunting_tree = self._build_hunting_tree()
        
        root.add_child(monitoring_tree)
        root.add_child(response_tree)
        root.add_child(hunting_tree)
        
        return root
    
    def _build_recon_phase(self) -> BehaviorNode:
        """Build reconnaissance phase behavior tree"""
        recon_seq = SequenceNode("ReconPhase", "Reconnaissance phase operations")
        
        # Check if in recon phase
        phase_check = ConditionNode("CheckReconPhase", 
                                   lambda ctx: ctx.current_phase == "recon")
        
        # LLM reasoning for recon strategy
        recon_reasoning = LLMReasoningNode("ReconReasoning", 
                                         self.reasoning_engine, 
                                         ReasoningType.TACTICAL)
        
        # GOAP planning for recon actions
        recon_goal = Goal("complete_recon", {"network_scanned": True, "services_discovered": True})
        
        # Add Red Team actions to planner
        for action in ActionLibrary.get_red_team_actions():
            self.planner.add_action(action)
        
        recon_planning = GOAPPlanningNode("ReconPlanning", self.planner, recon_goal)
        
        recon_seq.add_child(phase_check)
        recon_seq.add_child(recon_reasoning)
        recon_seq.add_child(recon_planning)
        
        return recon_seq
    
    def _build_exploit_phase(self) -> BehaviorNode:
        """Build exploitation phase behavior tree"""
        exploit_seq = SequenceNode("ExploitPhase", "Exploitation phase operations")
        
        # Check if in exploit phase
        phase_check = ConditionNode("CheckExploitPhase",
                                   lambda ctx: ctx.current_phase == "exploit")
        
        # Check if vulnerabilities are available
        vuln_check = ConditionNode("CheckVulnerabilities",
                                  lambda ctx: ctx.world_state.get("vulnerabilities_identified", False))
        
        # LLM reasoning for exploit strategy
        exploit_reasoning = LLMReasoningNode("ExploitReasoning",
                                           self.reasoning_engine,
                                           ReasoningType.STRATEGIC)
        
        # Retry wrapper for exploitation attempts
        exploit_retry = RetryNode("ExploitRetry", max_attempts=3)
        
        # Actual exploitation action
        exploit_action = ActionNode("ExecuteExploit", self._exploit_action)
        exploit_retry.add_child(exploit_action)
        
        exploit_seq.add_child(phase_check)
        exploit_seq.add_child(vuln_check)
        exploit_seq.add_child(exploit_reasoning)
        exploit_seq.add_child(exploit_retry)
        
        return exploit_seq
    
    def _build_persistence_phase(self) -> BehaviorNode:
        """Build persistence phase behavior tree"""
        persist_seq = SequenceNode("PersistencePhase", "Persistence phase operations")
        
        # Check if system is compromised
        access_check = ConditionNode("CheckAccess",
                                    lambda ctx: ctx.world_state.get("system_compromised", False))
        
        # LLM reasoning for persistence strategy
        persist_reasoning = LLMReasoningNode("PersistenceReasoning",
                                           self.reasoning_engine,
                                           ReasoningType.STRATEGIC)
        
        # Persistence actions
        persist_action = ActionNode("EstablishPersistence", self._persistence_action)
        
        persist_seq.add_child(access_check)
        persist_seq.add_child(persist_reasoning)
        persist_seq.add_child(persist_action)
        
        return persist_seq
    
    def _build_monitoring_tree(self) -> BehaviorNode:
        """Build monitoring behavior tree for Blue Team"""
        monitor_seq = SequenceNode("MonitoringTree", "Continuous monitoring operations")
        
        # LLM reasoning for threat analysis
        threat_analysis = LLMReasoningNode("ThreatAnalysis",
                                         self.reasoning_engine,
                                         ReasoningType.ANALYTICAL)
        
        # Monitor alerts action
        monitor_action = ActionNode("MonitorAlerts", self._monitor_alerts_action)
        
        monitor_seq.add_child(threat_analysis)
        monitor_seq.add_child(monitor_action)
        
        return monitor_seq
    
    def _build_response_tree(self) -> BehaviorNode:
        """Build incident response tree for Blue Team"""
        response_selector = SelectorNode("ResponseTree", "Incident response operations")
        
        # High priority response
        high_priority = SequenceNode("HighPriorityResponse")
        high_priority_check = ConditionNode("CheckHighPriority",
                                          lambda ctx: ctx.execution_data.get("threat_level") == "high")
        high_priority_action = ActionNode("ImmediateResponse", self._immediate_response_action)
        high_priority.add_child(high_priority_check)
        high_priority.add_child(high_priority_action)
        
        # Standard response
        standard_response = SequenceNode("StandardResponse")
        standard_reasoning = LLMReasoningNode("ResponseReasoning",
                                            self.reasoning_engine,
                                            ReasoningType.REACTIVE)
        standard_action = ActionNode("StandardResponse", self._standard_response_action)
        standard_response.add_child(standard_reasoning)
        standard_response.add_child(standard_action)
        
        response_selector.add_child(high_priority)
        response_selector.add_child(standard_response)
        
        return response_selector
    
    def _build_hunting_tree(self) -> BehaviorNode:
        """Build threat hunting tree for Blue Team"""
        hunting_seq = SequenceNode("ThreatHunting", "Proactive threat hunting")
        
        # Hunting reasoning
        hunting_reasoning = LLMReasoningNode("HuntingReasoning",
                                           self.reasoning_engine,
                                           ReasoningType.ANALYTICAL)
        
        # Hunting action
        hunting_action = ActionNode("ExecuteHunt", self._threat_hunting_action)
        
        hunting_seq.add_child(hunting_reasoning)
        hunting_seq.add_child(hunting_action)
        
        return hunting_seq
    
    # Action implementations
    async def _exploit_action(self, context: ExecutionContext) -> bool:
        """Execute exploitation action"""
        logger.info("Executing exploitation action")
        # Simulate exploitation
        vulnerabilities = context.execution_data.get("vulnerabilities", [])
        if vulnerabilities:
            context.world_state.set("system_compromised", True)
            context.world_state.set("access_gained", True)
            return True
        return False
    
    async def _persistence_action(self, context: ExecutionContext) -> bool:
        """Execute persistence action"""
        logger.info("Establishing persistence")
        # Simulate persistence establishment
        if context.world_state.get("system_compromised", False):
            context.world_state.set("persistence_established", True)
            return True
        return False
    
    async def _monitor_alerts_action(self, context: ExecutionContext) -> bool:
        """Monitor security alerts"""
        logger.info("Monitoring security alerts")
        # Simulate alert monitoring
        context.execution_data["alerts_monitored"] = True
        context.execution_data["threat_level"] = "medium"  # Simulate threat detection
        return True
    
    async def _immediate_response_action(self, context: ExecutionContext) -> bool:
        """Execute immediate response to high priority threats"""
        logger.info("Executing immediate response")
        # Simulate immediate response
        context.world_state.set("threat_contained", True)
        return True
    
    async def _standard_response_action(self, context: ExecutionContext) -> bool:
        """Execute standard incident response"""
        logger.info("Executing standard response")
        # Simulate standard response
        context.world_state.set("incident_logged", True)
        return True
    
    async def _threat_hunting_action(self, context: ExecutionContext) -> bool:
        """Execute threat hunting activities"""
        logger.info("Executing threat hunting")
        # Simulate threat hunting
        context.execution_data["hunting_completed"] = True
        return True


# Example usage and testing
async def test_behavior_tree_system():
    """Test the behavior tree system with LLM integration"""
    
    from .llm_reasoning import LocalLLMInterface, LLMReasoningEngine
    
    # Initialize LLM reasoning engine
    llm_interface = LocalLLMInterface("test-model")
    reasoning_engine = LLMReasoningEngine(llm_interface)
    
    # Create behavior tree builder
    builder = BehaviorTreeBuilder(reasoning_engine)
    
    # Test Red Team behavior tree
    print("=== Red Team Behavior Tree Test ===")
    
    red_tree = builder.build_red_team_tree()
    
    # Create execution context for Red Team
    red_world_state = WorldState({
        "has_target": True,
        "tools_available": True,
        "network_scanned": False,
        "services_discovered": False,
        "vulnerabilities_identified": False,
        "system_compromised": False
    })
    
    red_context = ExecutionContext(
        agent_id="red_agent_001",
        team="red",
        role="penetration_tester",
        current_phase="recon",
        world_state=red_world_state,
        objectives=["Compromise target system", "Establish persistence"],
        available_tools=["nmap", "metasploit", "sqlmap"],
        memory_context={"previous_targets": [], "known_vulnerabilities": []},
        execution_data={}
    )
    
    # Execute Red Team behavior tree
    print("Executing Red Team behavior tree...")
    red_result = await red_tree.execute(red_context)
    print(f"Red Team result: {red_result}")
    print(f"Final world state: {red_context.world_state.facts}")
    
    # Test Blue Team behavior tree
    print("\n=== Blue Team Behavior Tree Test ===")
    
    blue_tree = builder.build_blue_team_tree()
    
    # Create execution context for Blue Team
    blue_world_state = WorldState({
        "siem_active": True,
        "monitoring_enabled": True,
        "alerts_pending": True,
        "threat_detected": False,
        "incident_logged": False
    })
    
    blue_context = ExecutionContext(
        agent_id="blue_agent_001",
        team="blue",
        role="soc_analyst",
        current_phase="monitor",
        world_state=blue_world_state,
        objectives=["Monitor for threats", "Respond to incidents"],
        available_tools=["siem", "firewall", "ids"],
        memory_context={"known_threats": [], "response_procedures": []},
        execution_data={}
    )
    
    # Execute Blue Team behavior tree
    print("Executing Blue Team behavior tree...")
    blue_result = await blue_tree.execute(blue_context)
    print(f"Blue Team result: {blue_result}")
    print(f"Final world state: {blue_context.world_state.facts}")
    print(f"Execution data: {blue_context.execution_data}")


if __name__ == "__main__":
    asyncio.run(test_behavior_tree_system())