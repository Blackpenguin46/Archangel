#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Behavior Tree Framework
Structured decision-making system for autonomous agents
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import uuid

logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    """Status of behavior tree node execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"
    INVALID = "invalid"

class NodeType(Enum):
    """Types of behavior tree nodes"""
    COMPOSITE = "composite"
    DECORATOR = "decorator"
    LEAF = "leaf"

@dataclass
class ExecutionContext:
    """Context for behavior tree execution"""
    agent_id: str
    environment_state: Dict[str, Any]
    agent_memory: Dict[str, Any]
    available_tools: List[str]
    constraints: List[str]
    objectives: List[str]
    blackboard: Dict[str, Any] = field(default_factory=dict)
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    max_execution_time: float = 300.0  # 5 minutes default

@dataclass
class NodeResult:
    """Result of node execution"""
    status: NodeStatus
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    execution_time: float = 0.0
    children_results: List['NodeResult'] = field(default_factory=list)

class BehaviorTreeNode(ABC):
    """Abstract base class for all behavior tree nodes"""
    
    def __init__(self, name: str, node_type: NodeType):
        self.name = name
        self.node_type = node_type
        self.node_id = str(uuid.uuid4())
        self.parent: Optional['BehaviorTreeNode'] = None
        self.children: List['BehaviorTreeNode'] = []
        self.metadata: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"bt_node.{name}")
    
    @abstractmethod
    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute the node with given context"""
        pass
    
    def add_child(self, child: 'BehaviorTreeNode') -> None:
        """Add a child node"""
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child: 'BehaviorTreeNode') -> None:
        """Remove a child node"""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
    
    def get_path(self) -> str:
        """Get the path from root to this node"""
        if self.parent:
            return f"{self.parent.get_path()}/{self.name}"
        return self.name
    
    async def reset(self) -> None:
        """Reset node state"""
        for child in self.children:
            await child.reset()
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, children={len(self.children)})>"

class CompositeNode(BehaviorTreeNode):
    """Base class for composite nodes (nodes with children)"""
    
    def __init__(self, name: str):
        super().__init__(name, NodeType.COMPOSITE)

class SequenceNode(CompositeNode):
    """Execute children in sequence, fail on first failure"""
    
    async def execute(self, context: ExecutionContext) -> NodeResult:
        start_time = time.time()
        results = []
        
        self.logger.debug(f"Executing sequence node: {self.name}")
        
        for child in self.children:
            result = await child.execute(context)
            results.append(result)
            
            if result.status == NodeStatus.FAILURE:
                execution_time = time.time() - start_time
                return NodeResult(
                    status=NodeStatus.FAILURE,
                    message=f"Sequence failed at child: {child.name}",
                    execution_time=execution_time,
                    children_results=results
                )
            elif result.status == NodeStatus.RUNNING:
                execution_time = time.time() - start_time
                return NodeResult(
                    status=NodeStatus.RUNNING,
                    message=f"Sequence running at child: {child.name}",
                    execution_time=execution_time,
                    children_results=results
                )
        
        execution_time = time.time() - start_time
        return NodeResult(
            status=NodeStatus.SUCCESS,
            message=f"Sequence completed successfully",
            execution_time=execution_time,
            children_results=results
        )

class SelectorNode(CompositeNode):
    """Execute children until one succeeds, succeed on first success"""
    
    async def execute(self, context: ExecutionContext) -> NodeResult:
        start_time = time.time()
        results = []
        
        self.logger.debug(f"Executing selector node: {self.name}")
        
        for child in self.children:
            result = await child.execute(context)
            results.append(result)
            
            if result.status == NodeStatus.SUCCESS:
                execution_time = time.time() - start_time
                return NodeResult(
                    status=NodeStatus.SUCCESS,
                    message=f"Selector succeeded at child: {child.name}",
                    execution_time=execution_time,
                    children_results=results
                )
            elif result.status == NodeStatus.RUNNING:
                execution_time = time.time() - start_time
                return NodeResult(
                    status=NodeStatus.RUNNING,
                    message=f"Selector running at child: {child.name}",
                    execution_time=execution_time,
                    children_results=results
                )
        
        execution_time = time.time() - start_time
        return NodeResult(
            status=NodeStatus.FAILURE,
            message="All selector children failed",
            execution_time=execution_time,
            children_results=results
        )

class ParallelNode(CompositeNode):
    """Execute all children in parallel"""
    
    def __init__(self, name: str, success_threshold: int = 1, failure_threshold: int = 1):
        super().__init__(name)
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
    
    async def execute(self, context: ExecutionContext) -> NodeResult:
        start_time = time.time()
        
        self.logger.debug(f"Executing parallel node: {self.name}")
        
        # Execute all children concurrently
        tasks = [child.execute(context) for child in self.children]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        node_results = []
        successes = 0
        failures = 0
        running = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                node_result = NodeResult(
                    status=NodeStatus.FAILURE,
                    message=f"Child {self.children[i].name} raised exception: {result}"
                )
                failures += 1
            else:
                node_result = result
                if result.status == NodeStatus.SUCCESS:
                    successes += 1
                elif result.status == NodeStatus.FAILURE:
                    failures += 1
                elif result.status == NodeStatus.RUNNING:
                    running += 1
            
            node_results.append(node_result)
        
        execution_time = time.time() - start_time
        
        # Determine overall status
        if successes >= self.success_threshold:
            status = NodeStatus.SUCCESS
            message = f"Parallel node succeeded ({successes}/{len(self.children)} children)"
        elif failures >= self.failure_threshold:
            status = NodeStatus.FAILURE
            message = f"Parallel node failed ({failures}/{len(self.children)} children)"
        else:
            status = NodeStatus.RUNNING
            message = f"Parallel node running ({running}/{len(self.children)} children)"
        
        return NodeResult(
            status=status,
            message=message,
            execution_time=execution_time,
            children_results=node_results
        )

class DecoratorNode(BehaviorTreeNode):
    """Base class for decorator nodes (modify child behavior)"""
    
    def __init__(self, name: str, child: Optional[BehaviorTreeNode] = None):
        super().__init__(name, NodeType.DECORATOR)
        if child:
            self.add_child(child)
    
    @property
    def child(self) -> Optional[BehaviorTreeNode]:
        """Get the single child node"""
        return self.children[0] if self.children else None

class InverterNode(DecoratorNode):
    """Invert the result of child node"""
    
    async def execute(self, context: ExecutionContext) -> NodeResult:
        if not self.child:
            return NodeResult(
                status=NodeStatus.FAILURE,
                message="Inverter has no child node"
            )
        
        start_time = time.time()
        result = await self.child.execute(context)
        execution_time = time.time() - start_time
        
        # Invert success/failure
        if result.status == NodeStatus.SUCCESS:
            inverted_status = NodeStatus.FAILURE
        elif result.status == NodeStatus.FAILURE:
            inverted_status = NodeStatus.SUCCESS
        else:
            inverted_status = result.status  # Keep RUNNING as is
        
        return NodeResult(
            status=inverted_status,
            message=f"Inverted result: {result.message}",
            execution_time=execution_time,
            children_results=[result]
        )

class RetryNode(DecoratorNode):
    """Retry child node on failure"""
    
    def __init__(self, name: str, max_attempts: int = 3, child: Optional[BehaviorTreeNode] = None):
        super().__init__(name, child)
        self.max_attempts = max_attempts
    
    async def execute(self, context: ExecutionContext) -> NodeResult:
        if not self.child:
            return NodeResult(
                status=NodeStatus.FAILURE,
                message="Retry has no child node"
            )
        
        start_time = time.time()
        results = []
        
        for attempt in range(self.max_attempts):
            self.logger.debug(f"Retry attempt {attempt + 1}/{self.max_attempts}")
            result = await self.child.execute(context)
            results.append(result)
            
            if result.status == NodeStatus.SUCCESS:
                execution_time = time.time() - start_time
                return NodeResult(
                    status=NodeStatus.SUCCESS,
                    message=f"Retry succeeded on attempt {attempt + 1}",
                    execution_time=execution_time,
                    children_results=results
                )
            elif result.status == NodeStatus.RUNNING:
                execution_time = time.time() - start_time
                return NodeResult(
                    status=NodeStatus.RUNNING,
                    message=f"Retry running on attempt {attempt + 1}",
                    execution_time=execution_time,
                    children_results=results
                )
            
            # Brief delay between retries
            if attempt < self.max_attempts - 1:
                await asyncio.sleep(0.1)
        
        execution_time = time.time() - start_time
        return NodeResult(
            status=NodeStatus.FAILURE,
            message=f"Retry failed after {self.max_attempts} attempts",
            execution_time=execution_time,
            children_results=results
        )

class TimeoutNode(DecoratorNode):
    """Execute child with timeout"""
    
    def __init__(self, name: str, timeout_seconds: float, child: Optional[BehaviorTreeNode] = None):
        super().__init__(name, child)
        self.timeout_seconds = timeout_seconds
    
    async def execute(self, context: ExecutionContext) -> NodeResult:
        if not self.child:
            return NodeResult(
                status=NodeStatus.FAILURE,
                message="Timeout has no child node"
            )
        
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.child.execute(context),
                timeout=self.timeout_seconds
            )
            execution_time = time.time() - start_time
            return NodeResult(
                status=result.status,
                message=result.message,
                execution_time=execution_time,
                children_results=[result]
            )
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return NodeResult(
                status=NodeStatus.FAILURE,
                message=f"Child execution timed out after {self.timeout_seconds}s",
                execution_time=execution_time
            )

class LeafNode(BehaviorTreeNode):
    """Base class for leaf nodes (action/condition nodes)"""
    
    def __init__(self, name: str):
        super().__init__(name, NodeType.LEAF)

class ActionNode(LeafNode):
    """Execute an action"""
    
    def __init__(self, name: str, action_func: Callable[[ExecutionContext], Any]):
        super().__init__(name)
        self.action_func = action_func
    
    async def execute(self, context: ExecutionContext) -> NodeResult:
        start_time = time.time()
        
        try:
            self.logger.debug(f"Executing action: {self.name}")
            
            # Execute the action function
            if asyncio.iscoroutinefunction(self.action_func):
                result = await self.action_func(context)
            else:
                result = self.action_func(context)
            
            execution_time = time.time() - start_time
            
            # Interpret result
            if isinstance(result, bool):
                status = NodeStatus.SUCCESS if result else NodeStatus.FAILURE
                data = {"result": result}
            elif isinstance(result, dict):
                status = NodeStatus.SUCCESS if result.get("success", True) else NodeStatus.FAILURE
                data = result
            else:
                status = NodeStatus.SUCCESS
                data = {"result": result}
            
            return NodeResult(
                status=status,
                data=data,
                message=f"Action {self.name} completed",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Action {self.name} failed: {e}")
            return NodeResult(
                status=NodeStatus.FAILURE,
                message=f"Action {self.name} failed: {str(e)}",
                execution_time=execution_time
            )

class ConditionNode(LeafNode):
    """Check a condition"""
    
    def __init__(self, name: str, condition_func: Callable[[ExecutionContext], bool]):
        super().__init__(name)
        self.condition_func = condition_func
    
    async def execute(self, context: ExecutionContext) -> NodeResult:
        start_time = time.time()
        
        try:
            self.logger.debug(f"Checking condition: {self.name}")
            
            # Check the condition
            if asyncio.iscoroutinefunction(self.condition_func):
                result = await self.condition_func(context)
            else:
                result = self.condition_func(context)
            
            execution_time = time.time() - start_time
            status = NodeStatus.SUCCESS if result else NodeStatus.FAILURE
            
            return NodeResult(
                status=status,
                data={"condition_result": result},
                message=f"Condition {self.name}: {result}",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Condition {self.name} failed: {e}")
            return NodeResult(
                status=NodeStatus.FAILURE,
                message=f"Condition {self.name} failed: {str(e)}",
                execution_time=execution_time
            )

class LLMReasoningNode(LeafNode):
    """Node that uses LLM for reasoning"""
    
    def __init__(self, name: str, llm_engine, reasoning_type: str = "general"):
        super().__init__(name)
        self.llm_engine = llm_engine
        self.reasoning_type = reasoning_type
    
    async def execute(self, context: ExecutionContext) -> NodeResult:
        start_time = time.time()
        
        try:
            self.logger.debug(f"Executing LLM reasoning: {self.name}")
            
            # Prepare agent context
            agent_context = {
                "agent_id": context.agent_id,
                "team": context.blackboard.get("team", "unknown"),
                "role": context.blackboard.get("role", "unknown"),
                "available_tools": context.available_tools,
                "constraints": context.constraints,
                "objectives": context.objectives,
                "previous_actions": context.blackboard.get("previous_actions", [])
            }
            
            # Get reasoning from LLM
            reasoning_result = await self.llm_engine.reason_about_situation(
                agent_context, context.environment_state
            )
            
            # Store result in blackboard for other nodes
            context.blackboard["reasoning_result"] = reasoning_result
            context.blackboard["last_reasoning_time"] = time.time()
            
            execution_time = time.time() - start_time
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                data=reasoning_result,
                message=f"LLM reasoning completed: {reasoning_result.get('situation_assessment', 'N/A')[:100]}",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"LLM reasoning {self.name} failed: {e}")
            return NodeResult(
                status=NodeStatus.FAILURE,
                message=f"LLM reasoning {self.name} failed: {str(e)}",
                execution_time=execution_time
            )

class BehaviorTree:
    """Main behavior tree class"""
    
    def __init__(self, name: str, root_node: BehaviorTreeNode):
        self.name = name
        self.root_node = root_node
        self.tree_id = str(uuid.uuid4())
        self.execution_history: List[NodeResult] = []
        self.logger = logging.getLogger(f"behavior_tree.{name}")
    
    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute the behavior tree"""
        self.logger.info(f"Executing behavior tree: {self.name}")
        
        try:
            # Check execution timeout
            if time.time() - context.start_time > context.max_execution_time:
                return NodeResult(
                    status=NodeStatus.FAILURE,
                    message="Behavior tree execution timed out"
                )
            
            # Execute root node
            result = await self.root_node.execute(context)
            
            # Store in history
            self.execution_history.append(result)
            
            # Limit history size
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-50:]
            
            self.logger.info(f"Behavior tree execution completed: {result.status.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Behavior tree execution failed: {e}")
            error_result = NodeResult(
                status=NodeStatus.FAILURE,
                message=f"Behavior tree execution failed: {str(e)}"
            )
            self.execution_history.append(error_result)
            return error_result
    
    async def reset(self) -> None:
        """Reset the behavior tree"""
        await self.root_node.reset()
        self.logger.debug("Behavior tree reset")
    
    def get_node_by_name(self, name: str) -> Optional[BehaviorTreeNode]:
        """Find a node by name"""
        return self._find_node_recursive(self.root_node, name)
    
    def _find_node_recursive(self, node: BehaviorTreeNode, name: str) -> Optional[BehaviorTreeNode]:
        """Recursively find node by name"""
        if node.name == name:
            return node
        
        for child in node.children:
            found = self._find_node_recursive(child, name)
            if found:
                return found
        
        return None
    
    def get_tree_structure(self) -> Dict[str, Any]:
        """Get the structure of the behavior tree"""
        return self._get_node_structure(self.root_node)
    
    def _get_node_structure(self, node: BehaviorTreeNode) -> Dict[str, Any]:
        """Recursively get node structure"""
        return {
            "name": node.name,
            "type": node.__class__.__name__,
            "node_type": node.node_type.value,
            "children": [self._get_node_structure(child) for child in node.children]
        }

class BehaviorTreeBuilder:
    """Builder for creating behavior trees"""
    
    def __init__(self, name: str):
        self.name = name
        self.root_node: Optional[BehaviorTreeNode] = None
        self.current_node: Optional[BehaviorTreeNode] = None
    
    def sequence(self, name: str) -> 'BehaviorTreeBuilder':
        """Add a sequence node"""
        node = SequenceNode(name)
        return self._add_node(node)
    
    def selector(self, name: str) -> 'BehaviorTreeBuilder':
        """Add a selector node"""
        node = SelectorNode(name)
        return self._add_node(node)
    
    def parallel(self, name: str, success_threshold: int = 1, failure_threshold: int = 1) -> 'BehaviorTreeBuilder':
        """Add a parallel node"""
        node = ParallelNode(name, success_threshold, failure_threshold)
        return self._add_node(node)
    
    def action(self, name: str, action_func: Callable) -> 'BehaviorTreeBuilder':
        """Add an action node"""
        node = ActionNode(name, action_func)
        return self._add_node(node)
    
    def condition(self, name: str, condition_func: Callable) -> 'BehaviorTreeBuilder':
        """Add a condition node"""
        node = ConditionNode(name, condition_func)
        return self._add_node(node)
    
    def llm_reasoning(self, name: str, llm_engine, reasoning_type: str = "general") -> 'BehaviorTreeBuilder':
        """Add an LLM reasoning node"""
        node = LLMReasoningNode(name, llm_engine, reasoning_type)
        return self._add_node(node)
    
    def inverter(self, name: str) -> 'BehaviorTreeBuilder':
        """Add an inverter decorator"""
        node = InverterNode(name)
        return self._add_node(node)
    
    def retry(self, name: str, max_attempts: int = 3) -> 'BehaviorTreeBuilder':
        """Add a retry decorator"""
        node = RetryNode(name, max_attempts)
        return self._add_node(node)
    
    def timeout(self, name: str, timeout_seconds: float) -> 'BehaviorTreeBuilder':
        """Add a timeout decorator"""
        node = TimeoutNode(name, timeout_seconds)
        return self._add_node(node)
    
    def _add_node(self, node: BehaviorTreeNode) -> 'BehaviorTreeBuilder':
        """Add a node to the tree"""
        if self.root_node is None:
            self.root_node = node
        elif self.current_node is not None:
            self.current_node.add_child(node)
        
        self.current_node = node
        return self
    
    def end(self) -> 'BehaviorTreeBuilder':
        """Move up one level in the tree"""
        if self.current_node and self.current_node.parent:
            self.current_node = self.current_node.parent
        return self
    
    def build(self) -> BehaviorTree:
        """Build the behavior tree"""
        if self.root_node is None:
            raise ValueError("Cannot build empty behavior tree")
        
        return BehaviorTree(self.name, self.root_node)