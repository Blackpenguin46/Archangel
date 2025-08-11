#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Reasoning Integration
Integration layer combining LLM reasoning, behavior trees, and planning systems
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
import uuid

from .llm_interface import LLMReasoningEngine, LLMConfig, LLMProvider
from .behavior_tree import (
    BehaviorTree, BehaviorTreeBuilder, ExecutionContext, NodeResult, NodeStatus,
    ActionNode, ConditionNode, LLMReasoningNode
)
from .planning import (
    PlanningEngine, WorldState, Goal, Action, ActionType, Plan, PlanStatus
)

logger = logging.getLogger(__name__)

@dataclass
class ReasoningContext:
    """Context for integrated reasoning system"""
    agent_id: str
    team: str
    role: str
    environment_state: Dict[str, Any]
    agent_memory: Dict[str, Any]
    available_tools: List[str]
    constraints: List[str]
    objectives: List[str]
    world_state: WorldState
    current_goal: Optional[Goal] = None
    active_plan: Optional[Plan] = None
    reasoning_history: List[Dict[str, Any]] = field(default_factory=list)
    execution_context: Optional[ExecutionContext] = None

@dataclass
class IntegratedDecision:
    """Result of integrated reasoning process"""
    decision_id: str
    reasoning_result: Dict[str, Any]
    selected_action: Optional[Action]
    behavior_tree_result: Optional[NodeResult]
    plan_fragment: Optional[Plan]
    confidence_score: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class LLMBehaviorTreeIntegration:
    """Integration between LLM reasoning and behavior trees"""
    
    def __init__(self, llm_engine: LLMReasoningEngine):
        self.llm_engine = llm_engine
        self.logger = logging.getLogger("llm_bt_integration")
    
    def create_reasoning_tree(self, agent_team: str, agent_role: str) -> BehaviorTree:
        """Create a behavior tree with LLM reasoning nodes"""
        
        if agent_team.lower() == "red":
            return self._create_red_team_tree(agent_role)
        elif agent_team.lower() == "blue":
            return self._create_blue_team_tree(agent_role)
        else:
            return self._create_generic_tree(agent_role)
    
    def _create_red_team_tree(self, role: str) -> BehaviorTree:
        """Create behavior tree for Red Team agents"""
        
        builder = BehaviorTreeBuilder("red_team_reasoning")
        
        # Main sequence: Assess -> Plan -> Execute
        tree = (builder
            .sequence("main_sequence")
                .llm_reasoning("assess_situation", self.llm_engine, "red_team_assessment")
                .selector("choose_strategy")
                    .sequence("reconnaissance_strategy")
                        .condition("need_reconnaissance", self._need_reconnaissance)
                        .llm_reasoning("plan_reconnaissance", self.llm_engine, "reconnaissance_planning")
                        .action("execute_reconnaissance", self._execute_reconnaissance)
                    .end()
                    .sequence("exploitation_strategy")
                        .condition("can_exploit", self._can_exploit)
                        .llm_reasoning("plan_exploitation", self.llm_engine, "exploitation_planning")
                        .action("execute_exploitation", self._execute_exploitation)
                    .end()
                    .sequence("persistence_strategy")
                        .condition("need_persistence", self._need_persistence)
                        .llm_reasoning("plan_persistence", self.llm_engine, "persistence_planning")
                        .action("execute_persistence", self._execute_persistence)
                    .end()
                .end()
                .action("update_memory", self._update_agent_memory)
            .end()
            .build()
        )
        
        return tree
    
    def _create_blue_team_tree(self, role: str) -> BehaviorTree:
        """Create behavior tree for Blue Team agents"""
        
        builder = BehaviorTreeBuilder("blue_team_reasoning")
        
        tree = (builder
            .sequence("main_sequence")
                .llm_reasoning("assess_threats", self.llm_engine, "blue_team_assessment")
                .selector("choose_response")
                    .sequence("monitoring_strategy")
                        .condition("need_monitoring", self._need_monitoring)
                        .llm_reasoning("plan_monitoring", self.llm_engine, "monitoring_planning")
                        .action("execute_monitoring", self._execute_monitoring)
                    .end()
                    .sequence("incident_response")
                        .condition("incident_detected", self._incident_detected)
                        .llm_reasoning("plan_response", self.llm_engine, "incident_planning")
                        .action("execute_response", self._execute_incident_response)
                    .end()
                    .sequence("defensive_measures")
                        .condition("need_defense", self._need_defense)
                        .llm_reasoning("plan_defense", self.llm_engine, "defense_planning")
                        .action("execute_defense", self._execute_defense)
                    .end()
                .end()
                .action("update_memory", self._update_agent_memory)
            .end()
            .build()
        )
        
        return tree
    
    def _create_generic_tree(self, role: str) -> BehaviorTree:
        """Create generic behavior tree"""
        
        builder = BehaviorTreeBuilder("generic_reasoning")
        
        tree = (builder
            .sequence("main_sequence")
                .llm_reasoning("general_assessment", self.llm_engine, "general")
                .action("execute_action", self._execute_generic_action)
                .action("update_memory", self._update_agent_memory)
            .end()
            .build()
        )
        
        return tree
    
    # Condition functions for behavior tree
    async def _need_reconnaissance(self, context: ExecutionContext) -> bool:
        """Check if reconnaissance is needed"""
        world_state = context.blackboard.get("world_state", {})
        return not world_state.get("network_mapped", False)
    
    async def _can_exploit(self, context: ExecutionContext) -> bool:
        """Check if exploitation is possible"""
        world_state = context.blackboard.get("world_state", {})
        return world_state.get("vulnerabilities_identified", False)
    
    async def _need_persistence(self, context: ExecutionContext) -> bool:
        """Check if persistence is needed"""
        world_state = context.blackboard.get("world_state", {})
        return (world_state.get("system_compromised", False) and 
                not world_state.get("persistence_established", False))
    
    async def _need_monitoring(self, context: ExecutionContext) -> bool:
        """Check if monitoring is needed"""
        world_state = context.blackboard.get("world_state", {})
        return not world_state.get("network_monitored", False)
    
    async def _incident_detected(self, context: ExecutionContext) -> bool:
        """Check if incident is detected"""
        world_state = context.blackboard.get("world_state", {})
        return world_state.get("alerts_generated", False)
    
    async def _need_defense(self, context: ExecutionContext) -> bool:
        """Check if defensive measures are needed"""
        world_state = context.blackboard.get("world_state", {})
        return world_state.get("threats_identified", False)
    
    # Action functions for behavior tree
    async def _execute_reconnaissance(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute reconnaissance action"""
        self.logger.debug("Executing reconnaissance action")
        
        # Simulate reconnaissance
        await asyncio.sleep(0.5)
        
        # Update world state
        world_state = context.blackboard.get("world_state", {})
        world_state["network_mapped"] = True
        world_state["services_discovered"] = True
        context.blackboard["world_state"] = world_state
        
        return {"success": True, "action": "reconnaissance", "result": "Network mapped successfully"}
    
    async def _execute_exploitation(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute exploitation action"""
        self.logger.debug("Executing exploitation action")
        
        # Simulate exploitation
        await asyncio.sleep(1.0)
        
        # Update world state
        world_state = context.blackboard.get("world_state", {})
        world_state["system_compromised"] = True
        world_state["initial_access"] = True
        context.blackboard["world_state"] = world_state
        
        return {"success": True, "action": "exploitation", "result": "System compromised"}
    
    async def _execute_persistence(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute persistence action"""
        self.logger.debug("Executing persistence action")
        
        # Simulate persistence establishment
        await asyncio.sleep(0.8)
        
        # Update world state
        world_state = context.blackboard.get("world_state", {})
        world_state["persistence_established"] = True
        world_state["backdoor_installed"] = True
        context.blackboard["world_state"] = world_state
        
        return {"success": True, "action": "persistence", "result": "Persistence established"}
    
    async def _execute_monitoring(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute monitoring action"""
        self.logger.debug("Executing monitoring action")
        
        # Simulate monitoring
        await asyncio.sleep(0.3)
        
        # Update world state
        world_state = context.blackboard.get("world_state", {})
        world_state["network_monitored"] = True
        world_state["alerts_generated"] = True
        context.blackboard["world_state"] = world_state
        
        return {"success": True, "action": "monitoring", "result": "Network monitoring active"}
    
    async def _execute_incident_response(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute incident response action"""
        self.logger.debug("Executing incident response action")
        
        # Simulate incident response
        await asyncio.sleep(1.2)
        
        # Update world state
        world_state = context.blackboard.get("world_state", {})
        world_state["incident_contained"] = True
        world_state["forensics_collected"] = True
        context.blackboard["world_state"] = world_state
        
        return {"success": True, "action": "incident_response", "result": "Incident contained"}
    
    async def _execute_defense(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute defensive measures"""
        self.logger.debug("Executing defensive measures")
        
        # Simulate defense
        await asyncio.sleep(0.7)
        
        # Update world state
        world_state = context.blackboard.get("world_state", {})
        world_state["attack_blocked"] = True
        world_state["network_secured"] = True
        context.blackboard["world_state"] = world_state
        
        return {"success": True, "action": "defense", "result": "Defensive measures deployed"}
    
    async def _execute_generic_action(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute generic action"""
        self.logger.debug("Executing generic action")
        
        # Get reasoning result from blackboard
        reasoning_result = context.blackboard.get("reasoning_result", {})
        recommended_actions = reasoning_result.get("recommended_actions", [])
        
        if recommended_actions:
            action = recommended_actions[0]
            await asyncio.sleep(0.5)
            return {"success": True, "action": action, "result": f"Executed: {action}"}
        else:
            return {"success": False, "error": "No recommended actions"}
    
    async def _update_agent_memory(self, context: ExecutionContext) -> Dict[str, Any]:
        """Update agent memory with experience"""
        self.logger.debug("Updating agent memory")
        
        # This would integrate with the memory system
        memory_update = {
            "timestamp": time.time(),
            "context": context.environment_state,
            "reasoning": context.blackboard.get("reasoning_result", {}),
            "world_state": context.blackboard.get("world_state", {})
        }
        
        context.agent_memory.setdefault("experiences", []).append(memory_update)
        
        return {"success": True, "action": "memory_update", "result": "Memory updated"}

class IntegratedReasoningSystem:
    """Main system that integrates LLM reasoning, behavior trees, and planning"""
    
    def __init__(self, llm_config: LLMConfig):
        self.llm_engine = LLMReasoningEngine(llm_config)
        self.planning_engine = PlanningEngine()
        self.bt_integration = LLMBehaviorTreeIntegration(self.llm_engine)
        self.logger = logging.getLogger("integrated_reasoning")
        
        # Cache for behavior trees
        self.behavior_trees: Dict[str, BehaviorTree] = {}
    
    async def make_decision(self, context: ReasoningContext) -> IntegratedDecision:
        """Make an integrated decision using LLM, behavior trees, and planning"""
        start_time = time.time()
        decision_id = str(uuid.uuid4())
        
        self.logger.info(f"Making integrated decision for agent {context.agent_id}")
        
        try:
            # Step 1: LLM Reasoning
            reasoning_result = await self._perform_llm_reasoning(context)
            
            # Step 2: Behavior Tree Execution
            bt_result = await self._execute_behavior_tree(context, reasoning_result)
            
            # Step 3: Planning (if needed)
            plan_fragment = await self._create_plan_fragment(context, reasoning_result)
            
            # Step 4: Action Selection
            selected_action = await self._select_action(context, reasoning_result, bt_result, plan_fragment)
            
            # Step 5: Calculate confidence
            confidence_score = self._calculate_confidence(reasoning_result, bt_result, plan_fragment)
            
            execution_time = time.time() - start_time
            
            decision = IntegratedDecision(
                decision_id=decision_id,
                reasoning_result=reasoning_result,
                selected_action=selected_action,
                behavior_tree_result=bt_result,
                plan_fragment=plan_fragment,
                confidence_score=confidence_score,
                execution_time=execution_time,
                metadata={
                    "agent_id": context.agent_id,
                    "team": context.team,
                    "role": context.role,
                    "timestamp": time.time()
                }
            )
            
            # Update reasoning history
            context.reasoning_history.append({
                "decision_id": decision_id,
                "timestamp": time.time(),
                "reasoning_summary": reasoning_result.get("situation_assessment", ""),
                "selected_action": selected_action.name if selected_action else None,
                "confidence": confidence_score
            })
            
            self.logger.info(f"Decision made: {decision_id} (confidence: {confidence_score:.2f})")
            return decision
            
        except Exception as e:
            self.logger.error(f"Decision making failed: {e}")
            
            # Return fallback decision
            return IntegratedDecision(
                decision_id=decision_id,
                reasoning_result={"error": str(e)},
                selected_action=None,
                behavior_tree_result=None,
                plan_fragment=None,
                confidence_score=0.0,
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def _perform_llm_reasoning(self, context: ReasoningContext) -> Dict[str, Any]:
        """Perform LLM reasoning about the situation"""
        
        agent_context = {
            "agent_id": context.agent_id,
            "team": context.team,
            "role": context.role,
            "available_tools": context.available_tools,
            "constraints": context.constraints,
            "objectives": context.objectives,
            "previous_actions": [entry.get("selected_action") for entry in context.reasoning_history[-5:]]
        }
        
        return await self.llm_engine.reason_about_situation(agent_context, context.environment_state)
    
    async def _execute_behavior_tree(self, context: ReasoningContext, 
                                   reasoning_result: Dict[str, Any]) -> Optional[NodeResult]:
        """Execute behavior tree for structured decision making"""
        
        # Get or create behavior tree for this agent
        tree_key = f"{context.team}_{context.role}"
        if tree_key not in self.behavior_trees:
            self.behavior_trees[tree_key] = self.bt_integration.create_reasoning_tree(
                context.team, context.role
            )
        
        behavior_tree = self.behavior_trees[tree_key]
        
        # Create execution context
        exec_context = ExecutionContext(
            agent_id=context.agent_id,
            environment_state=context.environment_state,
            agent_memory=context.agent_memory,
            available_tools=context.available_tools,
            constraints=context.constraints,
            objectives=context.objectives
        )
        
        # Add reasoning result and world state to blackboard
        exec_context.blackboard["reasoning_result"] = reasoning_result
        exec_context.blackboard["world_state"] = context.world_state.facts
        exec_context.blackboard["team"] = context.team
        exec_context.blackboard["role"] = context.role
        
        # Store execution context in reasoning context
        context.execution_context = exec_context
        
        # Execute behavior tree
        return await behavior_tree.execute(exec_context)
    
    async def _create_plan_fragment(self, context: ReasoningContext, 
                                  reasoning_result: Dict[str, Any]) -> Optional[Plan]:
        """Create a plan fragment for strategic actions"""
        
        # Only create plans for complex objectives
        recommended_actions = reasoning_result.get("recommended_actions", [])
        if len(recommended_actions) <= 1:
            return None
        
        try:
            # Create goal based on reasoning
            goal_conditions = {}
            if context.team.lower() == "red":
                if "reconnaissance" in str(recommended_actions).lower():
                    goal_conditions = {"network_mapped": True, "vulnerabilities_identified": True}
                elif "exploit" in str(recommended_actions).lower():
                    goal_conditions = {"system_compromised": True}
                elif "persist" in str(recommended_actions).lower():
                    goal_conditions = {"persistence_established": True}
            elif context.team.lower() == "blue":
                if "monitor" in str(recommended_actions).lower():
                    goal_conditions = {"threats_identified": True}
                elif "respond" in str(recommended_actions).lower():
                    goal_conditions = {"incident_contained": True}
                elif "defend" in str(recommended_actions).lower():
                    goal_conditions = {"attack_blocked": True}
            
            if not goal_conditions:
                return None
            
            goal = Goal(
                goal_id=str(uuid.uuid4()),
                name=f"Strategic Goal for {context.agent_id}",
                conditions=goal_conditions,
                priority=1.0
            )
            
            # Create plan
            plan = await self.planning_engine.create_plan(context.world_state, goal)
            return plan
            
        except Exception as e:
            self.logger.warning(f"Plan creation failed: {e}")
            return None
    
    async def _select_action(self, context: ReasoningContext, reasoning_result: Dict[str, Any],
                           bt_result: Optional[NodeResult], plan_fragment: Optional[Plan]) -> Optional[Action]:
        """Select the best action based on all reasoning components"""
        
        # Priority 1: Action from successful behavior tree execution
        if bt_result and bt_result.status == NodeStatus.SUCCESS:
            action_data = bt_result.data
            if "action" in action_data:
                action_name = action_data["action"]
                # Create action object
                return Action(
                    action_id=str(uuid.uuid4()),
                    name=action_name,
                    action_type=self._infer_action_type(action_name),
                    cost=1.0,
                    duration=1.0,
                    parameters=action_data
                )
        
        # Priority 2: First action from plan
        if plan_fragment and plan_fragment.actions:
            return plan_fragment.actions[0]
        
        # Priority 3: First recommended action from LLM
        recommended_actions = reasoning_result.get("recommended_actions", [])
        if recommended_actions:
            action_name = recommended_actions[0]
            return Action(
                action_id=str(uuid.uuid4()),
                name=action_name,
                action_type=self._infer_action_type(action_name),
                cost=1.0,
                duration=1.0,
                parameters={"llm_recommended": True}
            )
        
        return None
    
    def _infer_action_type(self, action_name: str) -> ActionType:
        """Infer action type from action name"""
        action_lower = action_name.lower()
        
        if any(keyword in action_lower for keyword in ["scan", "recon", "discover", "enumerate"]):
            return ActionType.RECONNAISSANCE
        elif any(keyword in action_lower for keyword in ["exploit", "attack", "compromise"]):
            return ActionType.EXPLOITATION
        elif any(keyword in action_lower for keyword in ["persist", "backdoor", "maintain"]):
            return ActionType.PERSISTENCE
        elif any(keyword in action_lower for keyword in ["lateral", "move", "pivot"]):
            return ActionType.LATERAL_MOVEMENT
        elif any(keyword in action_lower for keyword in ["exfiltrate", "steal", "extract"]):
            return ActionType.EXFILTRATION
        elif any(keyword in action_lower for keyword in ["defend", "block", "protect"]):
            return ActionType.DEFENSE
        elif any(keyword in action_lower for keyword in ["monitor", "watch", "observe"]):
            return ActionType.MONITORING
        elif any(keyword in action_lower for keyword in ["respond", "contain", "mitigate"]):
            return ActionType.RESPONSE
        elif any(keyword in action_lower for keyword in ["analyze", "investigate", "examine"]):
            return ActionType.ANALYSIS
        else:
            return ActionType.RECONNAISSANCE  # Default
    
    def _calculate_confidence(self, reasoning_result: Dict[str, Any], 
                            bt_result: Optional[NodeResult], 
                            plan_fragment: Optional[Plan]) -> float:
        """Calculate overall confidence score"""
        
        confidence_factors = []
        
        # LLM confidence
        llm_confidence = reasoning_result.get("confidence_score", 0.5)
        confidence_factors.append(llm_confidence * 0.4)  # 40% weight
        
        # Behavior tree success
        if bt_result:
            bt_confidence = 1.0 if bt_result.status == NodeStatus.SUCCESS else 0.3
            confidence_factors.append(bt_confidence * 0.3)  # 30% weight
        else:
            confidence_factors.append(0.1)
        
        # Plan validity
        if plan_fragment:
            plan_confidence = 0.8 if len(plan_fragment.actions) > 0 else 0.2
            confidence_factors.append(plan_confidence * 0.3)  # 30% weight
        else:
            confidence_factors.append(0.2)
        
        return min(1.0, sum(confidence_factors))
    
    async def validate_decision_safety(self, decision: IntegratedDecision, 
                                     context: ReasoningContext) -> Dict[str, Any]:
        """Validate decision safety using LLM"""
        
        if not decision.selected_action:
            return {"approved": True, "reason": "No action selected"}
        
        # Prepare action for safety validation
        proposed_action = {
            "name": decision.selected_action.name,
            "type": decision.selected_action.action_type.value,
            "parameters": decision.selected_action.parameters,
            "reasoning": decision.reasoning_result.get("situation_assessment", "")
        }
        
        agent_context = {
            "team": context.team,
            "role": context.role,
            "constraints": context.constraints
        }
        
        return await self.llm_engine.validate_action_safety(proposed_action, agent_context)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the integrated reasoning system"""
        return {
            "llm_engine": "active",
            "planning_engine": "active",
            "behavior_trees_cached": len(self.behavior_trees),
            "available_actions": len(self.planning_engine.action_library.actions)
        }