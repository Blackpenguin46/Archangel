#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Planning Systems
GOAP (Goal-Oriented Action Planning) and PDDL integration for strategic action selection
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
import heapq
import uuid
import json

logger = logging.getLogger(__name__)

class PlanStatus(Enum):
    """Status of plan execution"""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"

class ActionType(Enum):
    """Types of actions in planning"""
    RECONNAISSANCE = "reconnaissance"
    EXPLOITATION = "exploitation"
    PERSISTENCE = "persistence"
    LATERAL_MOVEMENT = "lateral_movement"
    EXFILTRATION = "exfiltration"
    DEFENSE = "defense"
    MONITORING = "monitoring"
    RESPONSE = "response"
    ANALYSIS = "analysis"

@dataclass
class WorldState:
    """Represents the current state of the world"""
    facts: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def has_fact(self, key: str, value: Any = True) -> bool:
        """Check if a fact exists with given value"""
        return self.facts.get(key) == value
    
    def set_fact(self, key: str, value: Any) -> None:
        """Set a fact in the world state"""
        self.facts[key] = value
    
    def get_fact(self, key: str, default: Any = None) -> Any:
        """Get a fact from the world state"""
        return self.facts.get(key, default)
    
    def copy(self) -> 'WorldState':
        """Create a copy of the world state"""
        return WorldState(facts=self.facts.copy(), timestamp=self.timestamp)
    
    def satisfies(self, conditions: Dict[str, Any]) -> bool:
        """Check if world state satisfies all conditions"""
        for key, value in conditions.items():
            if not self.has_fact(key, value):
                return False
        return True

@dataclass
class Goal:
    """Represents a goal to achieve"""
    goal_id: str
    name: str
    conditions: Dict[str, Any]
    priority: float = 1.0
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_satisfied(self, world_state: WorldState) -> bool:
        """Check if goal is satisfied by world state"""
        return world_state.satisfies(self.conditions)

@dataclass
class Action:
    """Represents an action that can be performed"""
    action_id: str
    name: str
    action_type: ActionType
    preconditions: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0
    duration: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    executor: Optional[Callable] = None
    
    def can_execute(self, world_state: WorldState) -> bool:
        """Check if action can be executed in current world state"""
        return world_state.satisfies(self.preconditions)
    
    def apply_effects(self, world_state: WorldState) -> WorldState:
        """Apply action effects to world state"""
        new_state = world_state.copy()
        for key, value in self.effects.items():
            new_state.set_fact(key, value)
        return new_state

@dataclass
class Plan:
    """Represents a sequence of actions to achieve a goal"""
    plan_id: str
    goal: Goal
    actions: List[Action] = field(default_factory=list)
    total_cost: float = 0.0
    estimated_duration: float = 0.0
    status: PlanStatus = PlanStatus.PENDING
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_action(self, action: Action) -> None:
        """Add an action to the plan"""
        self.actions.append(action)
        self.total_cost += action.cost
        self.estimated_duration += action.duration
    
    def is_valid(self, initial_state: WorldState) -> bool:
        """Check if plan is valid from initial state"""
        current_state = initial_state.copy()
        
        for action in self.actions:
            if not action.can_execute(current_state):
                return False
            current_state = action.apply_effects(current_state)
        
        return self.goal.is_satisfied(current_state)

class Planner(ABC):
    """Abstract base class for planners"""
    
    @abstractmethod
    async def create_plan(self, initial_state: WorldState, goal: Goal, 
                         available_actions: List[Action]) -> Optional[Plan]:
        """Create a plan to achieve the goal"""
        pass
    
    @abstractmethod
    async def validate_plan(self, plan: Plan, initial_state: WorldState) -> bool:
        """Validate that a plan is executable"""
        pass

class GOAPPlanner(Planner):
    """Goal-Oriented Action Planning implementation"""
    
    def __init__(self, max_depth: int = 10, max_nodes: int = 1000):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.logger = logging.getLogger("goap_planner")
    
    async def create_plan(self, initial_state: WorldState, goal: Goal, 
                         available_actions: List[Action]) -> Optional[Plan]:
        """Create a plan using A* search"""
        self.logger.debug(f"Creating plan for goal: {goal.name}")
        
        # Priority queue for A* search: (f_score, g_score, state, actions_taken)
        open_set = [(0, 0, initial_state, [])]
        closed_set: Set[str] = set()
        nodes_explored = 0
        
        while open_set and nodes_explored < self.max_nodes:
            f_score, g_score, current_state, actions_taken = heapq.heappop(open_set)
            nodes_explored += 1
            
            # Create state hash for closed set
            state_hash = self._hash_state(current_state)
            if state_hash in closed_set:
                continue
            closed_set.add(state_hash)
            
            # Check if goal is satisfied
            if goal.is_satisfied(current_state):
                plan = Plan(
                    plan_id=str(uuid.uuid4()),
                    goal=goal,
                    actions=actions_taken.copy(),
                    total_cost=g_score,
                    estimated_duration=sum(action.duration for action in actions_taken)
                )
                self.logger.debug(f"Plan created with {len(actions_taken)} actions, cost: {g_score}")
                return plan
            
            # Don't expand beyond max depth
            if len(actions_taken) >= self.max_depth:
                continue
            
            # Expand neighbors
            for action in available_actions:
                if action.can_execute(current_state):
                    new_state = action.apply_effects(current_state)
                    new_actions = actions_taken + [action]
                    new_g_score = g_score + action.cost
                    
                    # Heuristic: number of unsatisfied goal conditions
                    h_score = self._heuristic(new_state, goal)
                    new_f_score = new_g_score + h_score
                    
                    heapq.heappush(open_set, (new_f_score, new_g_score, new_state, new_actions))
        
        self.logger.warning(f"No plan found for goal: {goal.name} (explored {nodes_explored} nodes)")
        return None
    
    def _hash_state(self, state: WorldState) -> str:
        """Create a hash of the world state for closed set"""
        return json.dumps(state.facts, sort_keys=True)
    
    def _heuristic(self, state: WorldState, goal: Goal) -> float:
        """Heuristic function for A* search"""
        unsatisfied = 0
        for key, value in goal.conditions.items():
            if not state.has_fact(key, value):
                unsatisfied += 1
        return float(unsatisfied)
    
    async def validate_plan(self, plan: Plan, initial_state: WorldState) -> bool:
        """Validate that a plan is executable"""
        return plan.is_valid(initial_state)

class PDDLPlanner(Planner):
    """PDDL (Planning Domain Definition Language) planner"""
    
    def __init__(self):
        self.logger = logging.getLogger("pddl_planner")
        self.domain_definition = ""
        self.problem_definition = ""
    
    async def create_plan(self, initial_state: WorldState, goal: Goal, 
                         available_actions: List[Action]) -> Optional[Plan]:
        """Create a plan using PDDL"""
        self.logger.debug(f"Creating PDDL plan for goal: {goal.name}")
        
        try:
            # Generate PDDL domain and problem
            domain = self._generate_domain(available_actions)
            problem = self._generate_problem(initial_state, goal)
            
            # For now, fall back to GOAP since external PDDL solver integration
            # would require additional dependencies
            goap_planner = GOAPPlanner()
            return await goap_planner.create_plan(initial_state, goal, available_actions)
            
        except Exception as e:
            self.logger.error(f"PDDL planning failed: {e}")
            return None
    
    def _generate_domain(self, actions: List[Action]) -> str:
        """Generate PDDL domain definition"""
        domain_lines = [
            "(define (domain cybersecurity)",
            "  (:requirements :strips :typing)",
            "  (:types",
            "    system network service - object",
            "    agent - object",
            "  )",
            "  (:predicates"
        ]
        
        # Add predicates based on actions
        predicates = set()
        for action in actions:
            for key in action.preconditions.keys():
                predicates.add(f"    ({key} ?obj)")
            for key in action.effects.keys():
                predicates.add(f"    ({key} ?obj)")
        
        domain_lines.extend(sorted(predicates))
        domain_lines.append("  )")
        
        # Add actions
        for action in actions:
            domain_lines.extend(self._action_to_pddl(action))
        
        domain_lines.append(")")
        return "\n".join(domain_lines)
    
    def _action_to_pddl(self, action: Action) -> List[str]:
        """Convert action to PDDL format"""
        lines = [
            f"  (:action {action.name.lower().replace(' ', '_')}",
            "    :parameters (?agent - agent ?target - object)",
            "    :precondition (and"
        ]
        
        for key, value in action.preconditions.items():
            if value:
                lines.append(f"      ({key} ?target)")
            else:
                lines.append(f"      (not ({key} ?target))")
        
        lines.extend([
            "    )",
            "    :effect (and"
        ])
        
        for key, value in action.effects.items():
            if value:
                lines.append(f"      ({key} ?target)")
            else:
                lines.append(f"      (not ({key} ?target))")
        
        lines.extend([
            "    )",
            "  )"
        ])
        
        return lines
    
    def _generate_problem(self, initial_state: WorldState, goal: Goal) -> str:
        """Generate PDDL problem definition"""
        problem_lines = [
            "(define (problem cybersecurity-scenario)",
            "  (:domain cybersecurity)",
            "  (:objects",
            "    agent1 - agent",
            "    target1 - system",
            "  )",
            "  (:init"
        ]
        
        # Add initial state facts
        for key, value in initial_state.facts.items():
            if value:
                problem_lines.append(f"    ({key} target1)")
        
        problem_lines.extend([
            "  )",
            "  (:goal (and"
        ])
        
        # Add goal conditions
        for key, value in goal.conditions.items():
            if value:
                problem_lines.append(f"    ({key} target1)")
            else:
                problem_lines.append(f"    (not ({key} target1))")
        
        problem_lines.extend([
            "  ))",
            ")"
        ])
        
        return "\n".join(problem_lines)
    
    async def validate_plan(self, plan: Plan, initial_state: WorldState) -> bool:
        """Validate PDDL plan"""
        return plan.is_valid(initial_state)

class PlanExecutor:
    """Executes plans by running actions in sequence"""
    
    def __init__(self):
        self.logger = logging.getLogger("plan_executor")
        self.current_plan: Optional[Plan] = None
        self.execution_history: List[Dict[str, Any]] = []
    
    async def execute_plan(self, plan: Plan, initial_state: WorldState) -> Dict[str, Any]:
        """Execute a plan and return results"""
        self.logger.info(f"Executing plan: {plan.plan_id} with {len(plan.actions)} actions")
        
        self.current_plan = plan
        plan.status = PlanStatus.EXECUTING
        
        current_state = initial_state.copy()
        executed_actions = []
        execution_start = time.time()
        
        try:
            for i, action in enumerate(plan.actions):
                self.logger.debug(f"Executing action {i+1}/{len(plan.actions)}: {action.name}")
                
                # Check preconditions
                if not action.can_execute(current_state):
                    error_msg = f"Action {action.name} preconditions not met"
                    self.logger.error(error_msg)
                    plan.status = PlanStatus.FAILURE
                    return {
                        "success": False,
                        "error": error_msg,
                        "executed_actions": executed_actions,
                        "final_state": current_state,
                        "execution_time": time.time() - execution_start
                    }
                
                # Execute action
                action_result = await self._execute_action(action, current_state)
                executed_actions.append({
                    "action": action.name,
                    "result": action_result,
                    "timestamp": time.time()
                })
                
                if not action_result.get("success", True):
                    error_msg = f"Action {action.name} execution failed: {action_result.get('error', 'Unknown error')}"
                    self.logger.error(error_msg)
                    plan.status = PlanStatus.FAILURE
                    return {
                        "success": False,
                        "error": error_msg,
                        "executed_actions": executed_actions,
                        "final_state": current_state,
                        "execution_time": time.time() - execution_start
                    }
                
                # Apply effects
                current_state = action.apply_effects(current_state)
                
                # Brief delay between actions
                await asyncio.sleep(0.1)
            
            # Check if goal is satisfied
            goal_satisfied = plan.goal.is_satisfied(current_state)
            if goal_satisfied:
                plan.status = PlanStatus.SUCCESS
                self.logger.info(f"Plan executed successfully: {plan.plan_id}")
            else:
                plan.status = PlanStatus.FAILURE
                self.logger.warning(f"Plan completed but goal not satisfied: {plan.plan_id}")
            
            execution_time = time.time() - execution_start
            
            result = {
                "success": goal_satisfied,
                "goal_satisfied": goal_satisfied,
                "executed_actions": executed_actions,
                "final_state": current_state,
                "execution_time": execution_time,
                "plan_id": plan.plan_id
            }
            
            # Store in history
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            plan.status = PlanStatus.FAILURE
            error_msg = f"Plan execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "executed_actions": executed_actions,
                "final_state": current_state,
                "execution_time": time.time() - execution_start
            }
    
    async def _execute_action(self, action: Action, current_state: WorldState) -> Dict[str, Any]:
        """Execute a single action"""
        try:
            if action.executor:
                # Execute custom action function
                if asyncio.iscoroutinefunction(action.executor):
                    result = await action.executor(action, current_state)
                else:
                    result = action.executor(action, current_state)
                
                if isinstance(result, dict):
                    return result
                else:
                    return {"success": bool(result), "result": result}
            else:
                # Default execution (just apply effects)
                await asyncio.sleep(action.duration * 0.1)  # Simulate execution time
                return {"success": True, "message": f"Action {action.name} completed"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def cancel_plan(self) -> bool:
        """Cancel current plan execution"""
        if self.current_plan and self.current_plan.status == PlanStatus.EXECUTING:
            self.current_plan.status = PlanStatus.CANCELLED
            self.logger.info(f"Plan cancelled: {self.current_plan.plan_id}")
            return True
        return False

class ActionLibrary:
    """Library of available actions for planning"""
    
    def __init__(self):
        self.actions: Dict[str, Action] = {}
        self.logger = logging.getLogger("action_library")
        self._initialize_default_actions()
    
    def _initialize_default_actions(self):
        """Initialize default actions for cybersecurity scenarios"""
        
        # Red Team Actions
        self.add_action(Action(
            action_id="network_scan",
            name="Network Scan",
            action_type=ActionType.RECONNAISSANCE,
            preconditions={"network_access": True},
            effects={"network_mapped": True, "services_discovered": True},
            cost=2.0,
            duration=5.0
        ))
        
        self.add_action(Action(
            action_id="vulnerability_scan",
            name="Vulnerability Scan",
            action_type=ActionType.RECONNAISSANCE,
            preconditions={"services_discovered": True},
            effects={"vulnerabilities_identified": True},
            cost=3.0,
            duration=10.0
        ))
        
        self.add_action(Action(
            action_id="exploit_vulnerability",
            name="Exploit Vulnerability",
            action_type=ActionType.EXPLOITATION,
            preconditions={"vulnerabilities_identified": True},
            effects={"initial_access": True, "system_compromised": True},
            cost=5.0,
            duration=15.0
        ))
        
        self.add_action(Action(
            action_id="establish_persistence",
            name="Establish Persistence",
            action_type=ActionType.PERSISTENCE,
            preconditions={"system_compromised": True},
            effects={"persistence_established": True, "backdoor_installed": True},
            cost=4.0,
            duration=8.0
        ))
        
        self.add_action(Action(
            action_id="lateral_movement",
            name="Lateral Movement",
            action_type=ActionType.LATERAL_MOVEMENT,
            preconditions={"system_compromised": True},
            effects={"additional_systems_compromised": True},
            cost=6.0,
            duration=20.0
        ))
        
        self.add_action(Action(
            action_id="data_exfiltration",
            name="Data Exfiltration",
            action_type=ActionType.EXFILTRATION,
            preconditions={"system_compromised": True},
            effects={"data_exfiltrated": True, "mission_complete": True},
            cost=7.0,
            duration=12.0
        ))
        
        # Blue Team Actions
        self.add_action(Action(
            action_id="monitor_network",
            name="Monitor Network",
            action_type=ActionType.MONITORING,
            preconditions={"monitoring_enabled": True},
            effects={"network_monitored": True, "alerts_generated": True},
            cost=1.0,
            duration=1.0
        ))
        
        self.add_action(Action(
            action_id="analyze_alerts",
            name="Analyze Alerts",
            action_type=ActionType.ANALYSIS,
            preconditions={"alerts_generated": True},
            effects={"threats_identified": True},
            cost=2.0,
            duration=5.0
        ))
        
        self.add_action(Action(
            action_id="block_traffic",
            name="Block Malicious Traffic",
            action_type=ActionType.DEFENSE,
            preconditions={"threats_identified": True},
            effects={"attack_blocked": True, "network_secured": True},
            cost=3.0,
            duration=2.0
        ))
        
        self.add_action(Action(
            action_id="incident_response",
            name="Incident Response",
            action_type=ActionType.RESPONSE,
            preconditions={"threats_identified": True},
            effects={"incident_contained": True, "forensics_collected": True},
            cost=5.0,
            duration=15.0
        ))
        
        self.add_action(Action(
            action_id="patch_vulnerabilities",
            name="Patch Vulnerabilities",
            action_type=ActionType.DEFENSE,
            preconditions={"vulnerabilities_identified": True},
            effects={"vulnerabilities_patched": True, "system_hardened": True},
            cost=4.0,
            duration=10.0
        ))
    
    def add_action(self, action: Action) -> None:
        """Add an action to the library"""
        self.actions[action.action_id] = action
        self.logger.debug(f"Added action: {action.name}")
    
    def get_action(self, action_id: str) -> Optional[Action]:
        """Get an action by ID"""
        return self.actions.get(action_id)
    
    def get_actions_by_type(self, action_type: ActionType) -> List[Action]:
        """Get all actions of a specific type"""
        return [action for action in self.actions.values() if action.action_type == action_type]
    
    def get_applicable_actions(self, world_state: WorldState) -> List[Action]:
        """Get all actions that can be executed in the current world state"""
        return [action for action in self.actions.values() if action.can_execute(world_state)]
    
    def list_actions(self) -> List[str]:
        """List all available action IDs"""
        return list(self.actions.keys())

class PlanningEngine:
    """Main planning engine that coordinates different planners"""
    
    def __init__(self, use_goap: bool = True, use_pddl: bool = False):
        self.goap_planner = GOAPPlanner() if use_goap else None
        self.pddl_planner = PDDLPlanner() if use_pddl else None
        self.action_library = ActionLibrary()
        self.plan_executor = PlanExecutor()
        self.logger = logging.getLogger("planning_engine")
    
    async def create_plan(self, initial_state: WorldState, goal: Goal, 
                         planner_type: str = "goap") -> Optional[Plan]:
        """Create a plan using specified planner"""
        available_actions = self.action_library.get_applicable_actions(initial_state)
        
        if planner_type == "goap" and self.goap_planner:
            return await self.goap_planner.create_plan(initial_state, goal, available_actions)
        elif planner_type == "pddl" and self.pddl_planner:
            return await self.pddl_planner.create_plan(initial_state, goal, available_actions)
        else:
            self.logger.error(f"Planner type '{planner_type}' not available")
            return None
    
    async def execute_plan(self, plan: Plan, initial_state: WorldState) -> Dict[str, Any]:
        """Execute a plan"""
        return await self.plan_executor.execute_plan(plan, initial_state)
    
    async def plan_and_execute(self, initial_state: WorldState, goal: Goal, 
                              planner_type: str = "goap") -> Dict[str, Any]:
        """Create and execute a plan in one step"""
        plan = await self.create_plan(initial_state, goal, planner_type)
        
        if plan is None:
            return {
                "success": False,
                "error": "Failed to create plan",
                "plan": None,
                "execution_result": None
            }
        
        execution_result = await self.execute_plan(plan, initial_state)
        
        return {
            "success": execution_result["success"],
            "plan": plan,
            "execution_result": execution_result
        }
    
    def add_custom_action(self, action: Action) -> None:
        """Add a custom action to the action library"""
        self.action_library.add_action(action)
    
    def get_world_state_template(self, agent_team: str) -> WorldState:
        """Get a template world state for an agent team"""
        if agent_team.lower() == "red":
            return WorldState(facts={
                "network_access": True,
                "monitoring_enabled": True,
                "network_mapped": False,
                "services_discovered": False,
                "vulnerabilities_identified": False,
                "system_compromised": False,
                "persistence_established": False,
                "data_exfiltrated": False,
                "mission_complete": False
            })
        elif agent_team.lower() == "blue":
            return WorldState(facts={
                "monitoring_enabled": True,
                "network_monitored": False,
                "alerts_generated": False,
                "threats_identified": False,
                "attack_blocked": False,
                "incident_contained": False,
                "network_secured": False,
                "vulnerabilities_patched": False,
                "system_hardened": False
            })
        else:
            return WorldState()
    
    def create_goal_template(self, agent_team: str, goal_type: str) -> Goal:
        """Create a goal template for an agent team"""
        if agent_team.lower() == "red":
            if goal_type == "reconnaissance":
                return Goal(
                    goal_id=str(uuid.uuid4()),
                    name="Complete Reconnaissance",
                    conditions={"network_mapped": True, "vulnerabilities_identified": True},
                    priority=1.0
                )
            elif goal_type == "compromise":
                return Goal(
                    goal_id=str(uuid.uuid4()),
                    name="Compromise System",
                    conditions={"system_compromised": True, "persistence_established": True},
                    priority=2.0
                )
            elif goal_type == "exfiltration":
                return Goal(
                    goal_id=str(uuid.uuid4()),
                    name="Complete Mission",
                    conditions={"data_exfiltrated": True, "mission_complete": True},
                    priority=3.0
                )
        elif agent_team.lower() == "blue":
            if goal_type == "detection":
                return Goal(
                    goal_id=str(uuid.uuid4()),
                    name="Detect Threats",
                    conditions={"threats_identified": True},
                    priority=1.0
                )
            elif goal_type == "response":
                return Goal(
                    goal_id=str(uuid.uuid4()),
                    name="Respond to Incident",
                    conditions={"incident_contained": True, "attack_blocked": True},
                    priority=2.0
                )
            elif goal_type == "hardening":
                return Goal(
                    goal_id=str(uuid.uuid4()),
                    name="Harden Systems",
                    conditions={"vulnerabilities_patched": True, "system_hardened": True},
                    priority=3.0
                )
        
        # Default goal
        return Goal(
            goal_id=str(uuid.uuid4()),
            name="Default Goal",
            conditions={},
            priority=1.0
        )