"""
Goal-Oriented Action Planning (GOAP) System

This module implements GOAP planning for strategic action selection in autonomous agents.
GOAP allows agents to dynamically plan sequences of actions to achieve their goals.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
import heapq
import copy

logger = logging.getLogger(__name__)


class ActionResult(Enum):
    """Results of action execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"


@dataclass
class WorldState:
    """Represents the current state of the world"""
    facts: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default=None):
        """Get a fact from the world state"""
        return self.facts.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a fact in the world state"""
        self.facts[key] = value
    
    def has(self, key: str) -> bool:
        """Check if a fact exists in the world state"""
        return key in self.facts
    
    def satisfies(self, conditions: Dict[str, Any]) -> bool:
        """Check if this world state satisfies the given conditions"""
        for key, value in conditions.items():
            if key not in self.facts or self.facts[key] != value:
                return False
        return True
    
    def copy(self) -> 'WorldState':
        """Create a copy of this world state"""
        return WorldState(copy.deepcopy(self.facts))
    
    def __str__(self) -> str:
        return f"WorldState({self.facts})"


@dataclass
class Goal:
    """Represents a goal to be achieved"""
    name: str
    conditions: Dict[str, Any]
    priority: float = 1.0
    
    def is_satisfied(self, world_state: WorldState) -> bool:
        """Check if this goal is satisfied by the given world state"""
        return world_state.satisfies(self.conditions)
    
    def __str__(self) -> str:
        return f"Goal({self.name}: {self.conditions})"


class Action(ABC):
    """Abstract base class for GOAP actions"""
    
    def __init__(self, name: str, cost: float = 1.0):
        self.name = name
        self.cost = cost
        self.preconditions: Dict[str, Any] = {}
        self.effects: Dict[str, Any] = {}
    
    def set_preconditions(self, preconditions: Dict[str, Any]):
        """Set the preconditions for this action"""
        self.preconditions = preconditions
        return self
    
    def set_effects(self, effects: Dict[str, Any]):
        """Set the effects of this action"""
        self.effects = effects
        return self
    
    def can_execute(self, world_state: WorldState) -> bool:
        """Check if this action can be executed in the given world state"""
        return world_state.satisfies(self.preconditions)
    
    def apply_effects(self, world_state: WorldState) -> WorldState:
        """Apply the effects of this action to the world state"""
        new_state = world_state.copy()
        for key, value in self.effects.items():
            new_state.set(key, value)
        return new_state
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> ActionResult:
        """Execute this action in the given context"""
        pass
    
    def __str__(self) -> str:
        return f"Action({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()


class ReconScanAction(Action):
    """Reconnaissance scanning action for Red Team agents"""
    
    def __init__(self):
        super().__init__("recon_scan", cost=2.0)
        self.set_preconditions({
            "has_target": True,
            "tools_available": True
        })
        self.set_effects({
            "network_scanned": True,
            "services_discovered": True
        })
    
    async def execute(self, context: Dict[str, Any]) -> ActionResult:
        """Execute network reconnaissance scan"""
        logger.info("Executing reconnaissance scan")
        # Simulate scanning
        target = context.get('target', 'unknown')
        tools = context.get('available_tools', [])
        
        if 'nmap' in tools:
            logger.info(f"Scanning target {target} with nmap")
            # Simulate scan results
            context['scan_results'] = {
                'open_ports': [22, 80, 443, 3306],
                'services': ['ssh', 'http', 'https', 'mysql'],
                'os_detection': 'Linux'
            }
            return ActionResult.SUCCESS
        else:
            logger.warning("No scanning tools available")
            return ActionResult.FAILURE


class VulnerabilityAssessmentAction(Action):
    """Vulnerability assessment action"""
    
    def __init__(self):
        super().__init__("vuln_assessment", cost=3.0)
        self.set_preconditions({
            "network_scanned": True,
            "services_discovered": True
        })
        self.set_effects({
            "vulnerabilities_identified": True,
            "exploit_targets_found": True
        })
    
    async def execute(self, context: Dict[str, Any]) -> ActionResult:
        """Execute vulnerability assessment"""
        logger.info("Executing vulnerability assessment")
        scan_results = context.get('scan_results', {})
        
        if scan_results:
            # Simulate vulnerability discovery
            context['vulnerabilities'] = [
                {'cve': 'CVE-2021-44228', 'service': 'http', 'severity': 'critical'},
                {'cve': 'CVE-2019-6340', 'service': 'ssh', 'severity': 'high'}
            ]
            return ActionResult.SUCCESS
        else:
            return ActionResult.FAILURE


class ExploitAction(Action):
    """Exploitation action for Red Team agents"""
    
    def __init__(self):
        super().__init__("exploit", cost=4.0)
        self.set_preconditions({
            "vulnerabilities_identified": True,
            "exploit_targets_found": True
        })
        self.set_effects({
            "system_compromised": True,
            "access_gained": True
        })
    
    async def execute(self, context: Dict[str, Any]) -> ActionResult:
        """Execute exploitation attempt"""
        logger.info("Executing exploitation")
        vulnerabilities = context.get('vulnerabilities', [])
        
        if vulnerabilities:
            # Simulate exploitation
            target_vuln = vulnerabilities[0]  # Target highest severity
            logger.info(f"Exploiting {target_vuln['cve']}")
            context['compromised_system'] = {
                'access_level': 'user',
                'shell_type': 'reverse',
                'persistence': False
            }
            return ActionResult.SUCCESS
        else:
            return ActionResult.FAILURE


class MonitorAlertsAction(Action):
    """Monitor security alerts action for Blue Team agents"""
    
    def __init__(self):
        super().__init__("monitor_alerts", cost=1.0)
        self.set_preconditions({
            "siem_active": True,
            "monitoring_enabled": True
        })
        self.set_effects({
            "alerts_checked": True,
            "threat_status_updated": True
        })
    
    async def execute(self, context: Dict[str, Any]) -> ActionResult:
        """Monitor security alerts"""
        logger.info("Monitoring security alerts")
        # Simulate alert monitoring
        context['current_alerts'] = [
            {'type': 'suspicious_login', 'severity': 'medium', 'source': '192.168.1.100'},
            {'type': 'port_scan', 'severity': 'high', 'source': '10.0.0.50'}
        ]
        return ActionResult.SUCCESS


class AnalyzeThreatAction(Action):
    """Analyze threat action for Blue Team agents"""
    
    def __init__(self):
        super().__init__("analyze_threat", cost=2.0)
        self.set_preconditions({
            "alerts_checked": True,
            "threat_status_updated": True
        })
        self.set_effects({
            "threat_analyzed": True,
            "response_plan_ready": True
        })
    
    async def execute(self, context: Dict[str, Any]) -> ActionResult:
        """Analyze detected threats"""
        logger.info("Analyzing threats")
        alerts = context.get('current_alerts', [])
        
        if alerts:
            # Simulate threat analysis
            high_severity_alerts = [a for a in alerts if a['severity'] == 'high']
            context['threat_analysis'] = {
                'threat_level': 'high' if high_severity_alerts else 'medium',
                'attack_type': 'reconnaissance' if any('scan' in a['type'] for a in alerts) else 'unknown',
                'recommended_response': 'block_source' if high_severity_alerts else 'monitor'
            }
            return ActionResult.SUCCESS
        else:
            context['threat_analysis'] = {'threat_level': 'low'}
            return ActionResult.SUCCESS


class BlockThreatAction(Action):
    """Block threat action for Blue Team agents"""
    
    def __init__(self):
        super().__init__("block_threat", cost=1.5)
        self.set_preconditions({
            "threat_analyzed": True,
            "response_plan_ready": True
        })
        self.set_effects({
            "threat_blocked": True,
            "security_updated": True
        })
    
    async def execute(self, context: Dict[str, Any]) -> ActionResult:
        """Block identified threats"""
        logger.info("Blocking threats")
        threat_analysis = context.get('threat_analysis', {})
        
        if threat_analysis.get('recommended_response') == 'block_source':
            # Simulate blocking action
            context['blocked_sources'] = ['192.168.1.100', '10.0.0.50']
            logger.info(f"Blocked sources: {context['blocked_sources']}")
            return ActionResult.SUCCESS
        else:
            logger.info("No blocking action required")
            return ActionResult.SUCCESS


@dataclass
class PlanNode:
    """Node in the planning graph"""
    world_state: WorldState
    action: Optional[Action]
    parent: Optional['PlanNode']
    cost: float
    heuristic: float
    
    @property
    def total_cost(self) -> float:
        return self.cost + self.heuristic
    
    def __lt__(self, other: 'PlanNode') -> bool:
        return self.total_cost < other.total_cost


class GOAPPlanner:
    """Goal-Oriented Action Planning system"""
    
    def __init__(self):
        self.actions: List[Action] = []
        self.max_depth = 10
        self.max_nodes = 1000
    
    def add_action(self, action: Action):
        """Add an action to the planner"""
        self.actions.append(action)
    
    def plan(self, current_state: WorldState, goal: Goal) -> Optional[List[Action]]:
        """Create a plan to achieve the goal from the current state"""
        if goal.is_satisfied(current_state):
            return []  # Goal already satisfied
        
        # A* search for optimal plan
        open_set = []
        closed_set = set()
        node_count = 0
        
        # Start node
        start_node = PlanNode(
            world_state=current_state,
            action=None,
            parent=None,
            cost=0.0,
            heuristic=self._calculate_heuristic(current_state, goal)
        )
        
        heapq.heappush(open_set, start_node)
        
        while open_set and node_count < self.max_nodes:
            current_node = heapq.heappop(open_set)
            node_count += 1
            
            # Convert world state to hashable representation for closed set
            state_key = tuple(sorted(current_node.world_state.facts.items()))
            if state_key in closed_set:
                continue
            
            closed_set.add(state_key)
            
            # Check if goal is satisfied
            if goal.is_satisfied(current_node.world_state):
                return self._reconstruct_plan(current_node)
            
            # Expand neighbors
            if current_node.parent is None or len(self._get_plan_from_node(current_node)) < self.max_depth:
                for action in self.actions:
                    if action.can_execute(current_node.world_state):
                        new_state = action.apply_effects(current_node.world_state)
                        new_cost = current_node.cost + action.cost
                        new_heuristic = self._calculate_heuristic(new_state, goal)
                        
                        new_node = PlanNode(
                            world_state=new_state,
                            action=action,
                            parent=current_node,
                            cost=new_cost,
                            heuristic=new_heuristic
                        )
                        
                        heapq.heappush(open_set, new_node)
        
        logger.warning(f"No plan found after exploring {node_count} nodes")
        return None
    
    def _calculate_heuristic(self, state: WorldState, goal: Goal) -> float:
        """Calculate heuristic cost to reach goal from state"""
        unsatisfied_conditions = 0
        for key, value in goal.conditions.items():
            if not state.has(key) or state.get(key) != value:
                unsatisfied_conditions += 1
        
        return float(unsatisfied_conditions)
    
    def _reconstruct_plan(self, node: PlanNode) -> List[Action]:
        """Reconstruct the plan from the goal node"""
        plan = []
        current = node
        
        while current.parent is not None:
            if current.action:
                plan.append(current.action)
            current = current.parent
        
        plan.reverse()
        return plan
    
    def _get_plan_from_node(self, node: PlanNode) -> List[Action]:
        """Get the current plan from a node"""
        plan = []
        current = node
        
        while current.parent is not None:
            if current.action:
                plan.append(current.action)
            current = current.parent
        
        return plan


class ActionLibrary:
    """Library of available actions for different agent types"""
    
    @staticmethod
    def get_red_team_actions() -> List[Action]:
        """Get actions available to Red Team agents"""
        return [
            ReconScanAction(),
            VulnerabilityAssessmentAction(),
            ExploitAction()
        ]
    
    @staticmethod
    def get_blue_team_actions() -> List[Action]:
        """Get actions available to Blue Team agents"""
        return [
            MonitorAlertsAction(),
            AnalyzeThreatAction(),
            BlockThreatAction()
        ]


class PlanExecutor:
    """Executes GOAP plans"""
    
    def __init__(self):
        self.current_plan: Optional[List[Action]] = None
        self.current_action_index = 0
        self.execution_context: Dict[str, Any] = {}
    
    def set_plan(self, plan: List[Action], context: Dict[str, Any]):
        """Set a new plan for execution"""
        self.current_plan = plan
        self.current_action_index = 0
        self.execution_context = context.copy()
        logger.info(f"New plan set with {len(plan)} actions")
    
    async def execute_next_action(self) -> Optional[ActionResult]:
        """Execute the next action in the plan"""
        if not self.current_plan or self.current_action_index >= len(self.current_plan):
            return None
        
        action = self.current_plan[self.current_action_index]
        logger.info(f"Executing action {self.current_action_index + 1}/{len(self.current_plan)}: {action.name}")
        
        try:
            result = await action.execute(self.execution_context)
            
            if result == ActionResult.SUCCESS:
                self.current_action_index += 1
                logger.info(f"Action {action.name} completed successfully")
            elif result == ActionResult.FAILURE:
                logger.error(f"Action {action.name} failed")
                # Plan failed, need to replan
                self.current_plan = None
            # If IN_PROGRESS, keep trying the same action
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing action {action.name}: {e}")
            self.current_plan = None
            return ActionResult.FAILURE
    
    def is_plan_complete(self) -> bool:
        """Check if the current plan is complete"""
        return (self.current_plan is None or 
                self.current_action_index >= len(self.current_plan))
    
    def get_progress(self) -> Tuple[int, int]:
        """Get current progress (completed, total)"""
        if not self.current_plan:
            return (0, 0)
        return (self.current_action_index, len(self.current_plan))


# Example usage and testing
async def test_goap_system():
    """Test the GOAP planning system"""
    
    # Create planner
    planner = GOAPPlanner()
    
    # Test Red Team scenario
    print("=== Red Team GOAP Test ===")
    
    # Add Red Team actions
    for action in ActionLibrary.get_red_team_actions():
        planner.add_action(action)
    
    # Initial world state for Red Team
    red_initial_state = WorldState({
        "has_target": True,
        "tools_available": True,
        "network_scanned": False,
        "services_discovered": False,
        "vulnerabilities_identified": False,
        "exploit_targets_found": False,
        "system_compromised": False,
        "access_gained": False
    })
    
    # Goal: Compromise the system
    red_goal = Goal("compromise_system", {
        "system_compromised": True,
        "access_gained": True
    })
    
    # Create plan
    red_plan = planner.plan(red_initial_state, red_goal)
    
    if red_plan:
        print(f"Red Team plan created with {len(red_plan)} actions:")
        for i, action in enumerate(red_plan, 1):
            print(f"  {i}. {action.name}")
        
        # Execute plan
        executor = PlanExecutor()
        context = {
            'target': '192.168.1.100',
            'available_tools': ['nmap', 'metasploit', 'sqlmap']
        }
        executor.set_plan(red_plan, context)
        
        while not executor.is_plan_complete():
            result = await executor.execute_next_action()
            if result == ActionResult.FAILURE:
                print("Plan execution failed!")
                break
            elif result is None:
                print("Plan execution completed!")
                break
        
        print(f"Final context: {executor.execution_context}")
    else:
        print("No plan found for Red Team goal")
    
    print("\n=== Blue Team GOAP Test ===")
    
    # Reset planner for Blue Team
    planner = GOAPPlanner()
    for action in ActionLibrary.get_blue_team_actions():
        planner.add_action(action)
    
    # Initial world state for Blue Team
    blue_initial_state = WorldState({
        "siem_active": True,
        "monitoring_enabled": True,
        "alerts_checked": False,
        "threat_status_updated": False,
        "threat_analyzed": False,
        "response_plan_ready": False,
        "threat_blocked": False,
        "security_updated": False
    })
    
    # Goal: Secure the environment
    blue_goal = Goal("secure_environment", {
        "threat_blocked": True,
        "security_updated": True
    })
    
    # Create plan
    blue_plan = planner.plan(blue_initial_state, blue_goal)
    
    if blue_plan:
        print(f"Blue Team plan created with {len(blue_plan)} actions:")
        for i, action in enumerate(blue_plan, 1):
            print(f"  {i}. {action.name}")
        
        # Execute plan
        executor = PlanExecutor()
        context = {
            'siem_system': 'wazuh',
            'firewall_type': 'iptables'
        }
        executor.set_plan(blue_plan, context)
        
        while not executor.is_plan_complete():
            result = await executor.execute_next_action()
            if result == ActionResult.FAILURE:
                print("Plan execution failed!")
                break
            elif result is None:
                print("Plan execution completed!")
                break
        
        print(f"Final context: {executor.execution_context}")
    else:
        print("No plan found for Blue Team goal")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_goap_system())