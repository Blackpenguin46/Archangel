"""
Multi-Agent Reinforcement Learning Coordinator
Advanced AI vs AI cybersecurity system for BlackHat demonstration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import json
import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import random
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    RED_TEAM = "red_team"
    BLUE_TEAM = "blue_team"

class ActionType(Enum):
    # Red Team Actions
    RECONNAISSANCE = "reconnaissance"
    VULNERABILITY_SCAN = "vulnerability_scan"
    EXPLOIT_ATTEMPT = "exploit_attempt"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    PERSISTENCE = "persistence"
    
    # Blue Team Actions
    MONITOR_NETWORK = "monitor_network"
    PATCH_VULNERABILITY = "patch_vulnerability"
    ISOLATE_HOST = "isolate_host"
    ANALYZE_LOGS = "analyze_logs"
    DEPLOY_HONEYPOT = "deploy_honeypot"
    UPDATE_FIREWALL = "update_firewall"
    INCIDENT_RESPONSE = "incident_response"

@dataclass
class GameState:
    """Represents the current state of the cyber warfare game"""
    network_topology: Dict[str, Any]
    compromised_hosts: List[str]
    detected_attacks: List[str]
    active_vulnerabilities: List[str]
    defensive_measures: List[str]
    asset_values: Dict[str, float]
    time_step: int
    game_score: Dict[str, float]

@dataclass
class Action:
    """Represents an action taken by an agent"""
    agent_id: str
    action_type: ActionType
    target: str
    parameters: Dict[str, Any]
    timestamp: datetime
    success_probability: float

class MultiAgentQLearning:
    """Advanced Q-Learning for multi-agent cybersecurity scenarios"""
    
    def __init__(self, state_size: int, action_size: int, agent_id: str, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.lr = lr
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Neural network for Q-learning
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Multi-agent specific
        self.coordination_history = deque(maxlen=1000)
        self.team_reward_weight = 0.3
        
    def _build_network(self) -> nn.Module:
        """Build the neural network for Q-learning"""
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool, team_reward: float = 0.0):
        """Store experience in replay memory"""
        total_reward = reward + (self.team_reward_weight * team_reward)
        self.memory.append((state, action, total_reward, next_state, done))
    
    def act(self, state: np.ndarray, legal_actions: List[int] = None) -> int:
        """Choose action using epsilon-greedy policy with coordination"""
        if legal_actions is None:
            legal_actions = list(range(self.action_size))
            
        # Exploration
        if np.random.random() <= self.epsilon:
            return random.choice(legal_actions)
        
        # Exploitation with coordination consideration
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        
        # Mask illegal actions
        masked_q_values = q_values.clone()
        for i in range(self.action_size):
            if i not in legal_actions:
                masked_q_values[0][i] = float('-inf')
        
        return masked_q_values.argmax().item()
    
    def replay(self):
        """Train the neural network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

class CyberWarfareAgent:
    """Individual agent in the cyber warfare simulation"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, specialization: str = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.specialization = specialization
        self.skill_level = random.uniform(0.7, 1.0)
        
        # RL components
        self.state_size = 50  # Comprehensive state representation
        self.action_size = len(ActionType)
        self.rl_agent = MultiAgentQLearning(self.state_size, self.action_size, agent_id)
        
        # Agent memory and learning
        self.action_history = deque(maxlen=1000)
        self.success_rate = defaultdict(float)
        self.coordination_memory = deque(maxlen=500)
        
        # Specialization bonuses
        self.specialization_bonus = self._get_specialization_bonus()
        
    def _get_specialization_bonus(self) -> Dict[ActionType, float]:
        """Get bonus multipliers based on agent specialization"""
        if self.specialization == "network_scanner":
            return {ActionType.RECONNAISSANCE: 1.5, ActionType.VULNERABILITY_SCAN: 1.3}
        elif self.specialization == "exploiter":
            return {ActionType.EXPLOIT_ATTEMPT: 1.4, ActionType.PRIVILEGE_ESCALATION: 1.2}
        elif self.specialization == "lateral_movement":
            return {ActionType.LATERAL_MOVEMENT: 1.5, ActionType.PERSISTENCE: 1.3}
        elif self.specialization == "analyst":
            return {ActionType.ANALYZE_LOGS: 1.4, ActionType.MONITOR_NETWORK: 1.2}
        elif self.specialization == "responder":
            return {ActionType.INCIDENT_RESPONSE: 1.3, ActionType.ISOLATE_HOST: 1.2}
        else:
            return {}
    
    def encode_state(self, game_state: GameState) -> np.ndarray:
        """Encode game state into neural network input"""
        state = np.zeros(self.state_size)
        
        # Network topology features (0-9)
        state[0] = len(game_state.network_topology.get('hosts', []))
        state[1] = len(game_state.network_topology.get('services', []))
        state[2] = len(game_state.compromised_hosts)
        state[3] = len(game_state.active_vulnerabilities)
        state[4] = len(game_state.defensive_measures)
        
        # Asset value features (5-14)
        total_assets = sum(game_state.asset_values.values())
        compromised_value = sum(game_state.asset_values.get(host, 0) 
                              for host in game_state.compromised_hosts)
        state[5] = total_assets / 1000000  # Normalize to millions
        state[6] = compromised_value / 1000000 if total_assets > 0 else 0
        state[7] = compromised_value / total_assets if total_assets > 0 else 0
        
        # Game progress features (8-19)
        state[8] = game_state.time_step / 1000  # Normalize time
        state[9] = len(game_state.detected_attacks)
        
        # Team performance (10-19)
        my_team = "red_team" if self.agent_type == AgentType.RED_TEAM else "blue_team"
        enemy_team = "blue_team" if self.agent_type == AgentType.RED_TEAM else "red_team"
        
        state[10] = game_state.game_score.get(my_team, 0) / 1000
        state[11] = game_state.game_score.get(enemy_team, 0) / 1000
        
        # Recent action success rates (12-29)
        for i, action_type in enumerate(ActionType):
            if i < 18:  # Prevent index overflow
                state[12 + i] = self.success_rate.get(action_type, 0.5)
        
        # Coordination features (30-39)
        recent_coords = list(self.coordination_memory)[-10:]
        for i, coord in enumerate(recent_coords):
            if i < 10:
                state[30 + i] = coord.get('success', 0)
        
        # Agent-specific features (40-49)
        state[40] = self.skill_level
        state[41] = len(self.action_history) / 1000
        
        # Fill remaining with random noise to prevent overfitting
        state[42:] = np.random.normal(0, 0.1, size=self.state_size - 42)
        
        return state
    
    def select_action(self, game_state: GameState, 
                     coordination_signal: Dict[str, Any] = None) -> Action:
        """Select next action using RL and coordination"""
        state = self.encode_state(game_state)
        
        # Get legal actions based on current state
        legal_actions = self._get_legal_actions(game_state)
        
        # Get action from RL agent
        action_idx = self.rl_agent.act(state, legal_actions)
        
        # Ensure action_idx is an integer
        if isinstance(action_idx, str):
            try:
                action_idx = int(action_idx)
            except ValueError:
                logger.warning(f"Invalid action index '{action_idx}', using default")
                action_idx = 0
        
        # Ensure action_idx is within bounds
        action_list = list(ActionType)
        if not isinstance(action_idx, int) or action_idx >= len(action_list):
            logger.warning(f"Action index {action_idx} out of bounds, using 0")
            action_idx = 0
            
        action_type = action_list[action_idx]
        
        # Select target based on action type and game state
        target = self._select_target(action_type, game_state, coordination_signal)
        
        # Calculate success probability
        success_prob = self._calculate_success_probability(action_type, target, game_state)
        
        action = Action(
            agent_id=self.agent_id,
            action_type=action_type,
            target=target,
            parameters=self._get_action_parameters(action_type, target),
            timestamp=datetime.now(),
            success_probability=success_prob
        )
        
        self.action_history.append(action)
        return action
    
    def _get_legal_actions(self, game_state: GameState) -> List[int]:
        """Get list of legal action indices based on current state"""
        legal_action_types = []
        
        if self.agent_type == AgentType.RED_TEAM:
            # Red team actions
            legal_action_types.extend([
                ActionType.RECONNAISSANCE,
                ActionType.VULNERABILITY_SCAN
            ])
            
            if game_state.active_vulnerabilities:
                legal_action_types.append(ActionType.EXPLOIT_ATTEMPT)
            
            if game_state.compromised_hosts:
                legal_action_types.extend([
                    ActionType.LATERAL_MOVEMENT,
                    ActionType.PRIVILEGE_ESCALATION,
                    ActionType.DATA_EXFILTRATION,
                    ActionType.PERSISTENCE
                ])
                
        else:  # Blue team
            legal_action_types.extend([
                ActionType.MONITOR_NETWORK,
                ActionType.ANALYZE_LOGS
            ])
            
            if game_state.active_vulnerabilities:
                legal_action_types.append(ActionType.PATCH_VULNERABILITY)
            
            if game_state.detected_attacks:
                legal_action_types.extend([
                    ActionType.INCIDENT_RESPONSE,
                    ActionType.ISOLATE_HOST,
                    ActionType.UPDATE_FIREWALL
                ])
            
            legal_action_types.append(ActionType.DEPLOY_HONEYPOT)
        
        # Convert action types to indices
        action_list = list(ActionType)
        legal_indices = []
        for action_type in legal_action_types:
            try:
                index = action_list.index(action_type)
                legal_indices.append(index)
            except ValueError:
                logger.warning(f"Action type {action_type} not found in ActionType enum")
        
        return legal_indices if legal_indices else [0]  # Return at least one legal action
    
    def _select_target(self, action_type: ActionType, game_state: GameState, 
                      coordination_signal: Dict[str, Any] = None) -> str:
        """Select target for the action"""
        if coordination_signal and 'suggested_target' in coordination_signal:
            return coordination_signal['suggested_target']
        
        # Target selection based on action type and asset values
        hosts = list(game_state.network_topology.get('hosts', []))
        
        if not hosts:
            return "unknown"
        
        if action_type in [ActionType.DATA_EXFILTRATION, ActionType.EXPLOIT_ATTEMPT]:
            # Target high-value assets
            targets_by_value = sorted(hosts, 
                                    key=lambda h: game_state.asset_values.get(h, 0), 
                                    reverse=True)
            return targets_by_value[0] if targets_by_value else random.choice(hosts)
        
        elif action_type == ActionType.LATERAL_MOVEMENT:
            # Target uncompromised hosts adjacent to compromised ones
            uncompromised = [h for h in hosts if h not in game_state.compromised_hosts]
            return random.choice(uncompromised) if uncompromised else random.choice(hosts)
        
        else:
            return random.choice(hosts)
    
    def _calculate_success_probability(self, action_type: ActionType, target: str, 
                                     game_state: GameState) -> float:
        """Calculate probability of action success"""
        base_prob = 0.5
        
        # Apply skill level
        prob = base_prob * self.skill_level
        
        # Apply specialization bonus
        if action_type in self.specialization_bonus:
            prob *= self.specialization_bonus[action_type]
        
        # Apply historical success rate
        historical_rate = self.success_rate.get(action_type, 0.5)
        prob = 0.7 * prob + 0.3 * historical_rate
        
        # Environmental factors
        if self.agent_type == AgentType.RED_TEAM:
            # Harder to attack well-defended systems
            defense_factor = len(game_state.defensive_measures) * 0.05
            prob = max(0.1, prob - defense_factor)
        else:
            # Easier to defend with more defensive measures
            defense_factor = len(game_state.defensive_measures) * 0.02
            prob = min(0.95, prob + defense_factor)
        
        return np.clip(prob, 0.1, 0.95)
    
    def _get_action_parameters(self, action_type: ActionType, target: str) -> Dict[str, Any]:
        """Get parameters for the action"""
        params = {
            'target': target,
            'agent_specialization': self.specialization,
            'timestamp': datetime.now().isoformat()
        }
        
        if action_type == ActionType.VULNERABILITY_SCAN:
            params.update({
                'scan_type': random.choice(['tcp', 'udp', 'comprehensive']),
                'intensity': random.choice(['stealth', 'normal', 'aggressive'])
            })
        elif action_type == ActionType.EXPLOIT_ATTEMPT:
            params.update({
                'exploit_type': random.choice(['buffer_overflow', 'sql_injection', 'rce']),
                'payload': f"payload_{random.randint(1000, 9999)}"
            })
        
        return params
    
    def update_from_result(self, action: Action, success: bool, reward: float, 
                          new_game_state: GameState, team_reward: float = 0.0):
        """Update agent learning from action result"""
        # Update success rate
        action_type = action.action_type
        current_rate = self.success_rate[action_type]
        self.success_rate[action_type] = 0.9 * current_rate + 0.1 * (1.0 if success else 0.0)
        
        # Update RL agent
        if len(self.action_history) >= 2:
            prev_state = self.encode_state(self._get_previous_game_state())
            current_state = self.encode_state(new_game_state)
            action_idx = list(ActionType).index(action_type)
            
            self.rl_agent.remember(
                prev_state, action_idx, reward, current_state, 
                False, team_reward
            )
            
            self.rl_agent.replay()
    
    def _get_previous_game_state(self) -> GameState:
        """Get previous game state (simplified for demo)"""
        # In a real implementation, this would be stored
        return GameState(
            network_topology={},
            compromised_hosts=[],
            detected_attacks=[],
            active_vulnerabilities=[],
            defensive_measures=[],
            asset_values={},
            time_step=0,
            game_score={}
        )

class MARLCoordinator:
    """Multi-Agent Reinforcement Learning Coordinator for cyber warfare"""
    
    def __init__(self):
        self.red_team_agents: List[CyberWarfareAgent] = []
        self.blue_team_agents: List[CyberWarfareAgent] = []
        self.game_state = None
        self.coordination_network = CoordinationNetwork()
        self.game_history = deque(maxlen=10000)
        
        logger.info("MARL Coordinator initialized")
    
    def initialize_agents(self, red_team_config: Dict, blue_team_config: Dict):
        """Initialize red and blue team agents"""
        # Red team specializations
        red_specializations = [
            "network_scanner", "exploiter", "lateral_movement", 
            "data_exfiltrator", "persistence_specialist"
        ]
        
        # Blue team specializations  
        blue_specializations = [
            "network_monitor", "analyst", "responder", 
            "forensics_expert", "threat_hunter"
        ]
        
        # Create red team agents
        for i in range(red_team_config.get('num_agents', 3)):
            agent = CyberWarfareAgent(
                agent_id=f"red_agent_{i}",
                agent_type=AgentType.RED_TEAM,
                specialization=red_specializations[i % len(red_specializations)]
            )
            self.red_team_agents.append(agent)
            logger.info(f"Created red team agent: {agent.agent_id} ({agent.specialization})")
        
        # Create blue team agents
        for i in range(blue_team_config.get('num_agents', 3)):
            agent = CyberWarfareAgent(
                agent_id=f"blue_agent_{i}",
                agent_type=AgentType.BLUE_TEAM,
                specialization=blue_specializations[i % len(blue_specializations)]
            )
            self.blue_team_agents.append(agent)
            logger.info(f"Created blue team agent: {agent.agent_id} ({agent.specialization})")
    
    async def run_simulation_step(self, game_state: GameState) -> Dict[str, Any]:
        """Run one step of the multi-agent simulation"""
        self.game_state = game_state
        step_results = {
            'red_actions': [],
            'blue_actions': [],
            'coordination_signals': {},
            'rewards': {},
            'emergent_behaviors': []
        }
        
        # Generate coordination signals
        red_coordination = await self.coordination_network.generate_coordination_signal(
            self.red_team_agents, game_state, AgentType.RED_TEAM
        )
        blue_coordination = await self.coordination_network.generate_coordination_signal(
            self.blue_team_agents, game_state, AgentType.BLUE_TEAM
        )
        
        # Red team actions (simultaneous)
        red_actions = []
        for agent in self.red_team_agents:
            coordination_signal = red_coordination.get(agent.agent_id, {})
            action = agent.select_action(game_state, coordination_signal)
            red_actions.append(action)
        
        # Blue team actions (simultaneous)
        blue_actions = []
        for agent in self.blue_team_agents:
            coordination_signal = blue_coordination.get(agent.agent_id, {})
            action = agent.select_action(game_state, coordination_signal)
            blue_actions.append(action)
        
        step_results['red_actions'] = red_actions
        step_results['blue_actions'] = blue_actions
        step_results['coordination_signals'] = {
            'red_team': red_coordination,
            'blue_team': blue_coordination
        }
        
        # Detect emergent behaviors
        emergent_behaviors = self._detect_emergent_behaviors(red_actions, blue_actions)
        step_results['emergent_behaviors'] = emergent_behaviors
        
        self.game_history.append(step_results)
        
        logger.info(f"Simulation step completed: {len(red_actions)} red actions, "
                   f"{len(blue_actions)} blue actions, "
                   f"{len(emergent_behaviors)} emergent behaviors detected")
        
        return step_results
    
    def _detect_emergent_behaviors(self, red_actions: List[Action], 
                                  blue_actions: List[Action]) -> List[Dict[str, Any]]:
        """Detect emergent behaviors in agent actions"""
        behaviors = []
        
        # Coordinated attack patterns
        red_targets = [action.target for action in red_actions]
        if len(set(red_targets)) == 1 and len(red_actions) > 1:
            behaviors.append({
                'type': 'coordinated_focus_attack',
                'description': f'Multiple red agents targeting {red_targets[0]}',
                'agents': [action.agent_id for action in red_actions],
                'target': red_targets[0]
            })
        
        # Defensive coordination
        if any(action.action_type == ActionType.ISOLATE_HOST for action in blue_actions):
            isolation_actions = [a for a in blue_actions if a.action_type == ActionType.ISOLATE_HOST]
            if len(isolation_actions) > 1:
                behaviors.append({
                    'type': 'coordinated_isolation',
                    'description': 'Multiple blue agents coordinating isolation',
                    'agents': [action.agent_id for action in isolation_actions]
                })
        
        # Advanced persistent threat simulation
        red_action_types = [action.action_type for action in red_actions]
        apt_sequence = [
            ActionType.RECONNAISSANCE, 
            ActionType.EXPLOIT_ATTEMPT, 
            ActionType.LATERAL_MOVEMENT
        ]
        if all(action_type in red_action_types for action_type in apt_sequence):
            behaviors.append({
                'type': 'apt_simulation',
                'description': 'Red team exhibiting APT-like behavior sequence',
                'sequence': apt_sequence
            })
        
        return behaviors
    
    def get_team_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for both teams"""
        if not self.game_history:
            return {}
        
        recent_steps = list(self.game_history)[-100:]  # Last 100 steps
        
        red_success_rate = 0
        blue_success_rate = 0
        coordination_effectiveness = 0
        
        # Calculate metrics from recent history
        for step in recent_steps:
            # This would be populated with actual success results
            pass
        
        return {
            'red_team': {
                'success_rate': red_success_rate,
                'coordination_score': coordination_effectiveness,
                'agent_count': len(self.red_team_agents),
                'emergent_behaviors': len([b for step in recent_steps 
                                         for b in step['emergent_behaviors']])
            },
            'blue_team': {
                'success_rate': blue_success_rate,
                'coordination_score': coordination_effectiveness,
                'agent_count': len(self.blue_team_agents)
            }
        }

class CoordinationNetwork:
    """Neural network for agent coordination"""
    
    def __init__(self):
        self.coordination_history = deque(maxlen=1000)
        
    async def generate_coordination_signal(self, agents: List[CyberWarfareAgent], 
                                         game_state: GameState, 
                                         team_type: AgentType) -> Dict[str, Dict[str, Any]]:
        """Generate coordination signals for agents"""
        signals = {}
        
        # Simple coordination logic (can be enhanced with neural networks)
        if team_type == AgentType.RED_TEAM:
            # Focus on high-value targets
            high_value_targets = sorted(
                game_state.asset_values.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            for i, agent in enumerate(agents):
                if high_value_targets:
                    target_idx = i % len(high_value_targets)
                    signals[agent.agent_id] = {
                        'suggested_target': high_value_targets[target_idx][0],
                        'coordination_type': 'focus_fire',
                        'priority': 'high'
                    }
        
        else:  # Blue team
            # Coordinate defense of critical assets
            critical_assets = [host for host, value in game_state.asset_values.items() 
                             if value > 1000000]
            
            for i, agent in enumerate(agents):
                if critical_assets:
                    asset_idx = i % len(critical_assets)
                    signals[agent.agent_id] = {
                        'suggested_target': critical_assets[asset_idx],
                        'coordination_type': 'defensive_priority',
                        'priority': 'critical'
                    }
        
        return signals

# Example usage and testing
if __name__ == "__main__":
    async def test_marl_system():
        """Test the MARL coordination system"""
        coordinator = MARLCoordinator()
        
        # Initialize agents
        red_config = {'num_agents': 3}
        blue_config = {'num_agents': 3}
        coordinator.initialize_agents(red_config, blue_config)
        
        # Create test game state
        test_game_state = GameState(
            network_topology={
                'hosts': ['web-server', 'db-server', 'file-server', 'domain-controller'],
                'services': ['http', 'mysql', 'ftp', 'ldap']
            },
            compromised_hosts=[],
            detected_attacks=[],
            active_vulnerabilities=['CVE-2024-001', 'CVE-2024-002'],
            defensive_measures=['firewall', 'ids'],
            asset_values={
                'web-server': 500000,
                'db-server': 2000000,
                'file-server': 1000000,
                'domain-controller': 3000000
            },
            time_step=1,
            game_score={'red_team': 0, 'blue_team': 0}
        )
        
        # Run simulation step
        results = await coordinator.run_simulation_step(test_game_state)
        
        print("MARL Simulation Results:")
        print(f"Red team actions: {len(results['red_actions'])}")
        print(f"Blue team actions: {len(results['blue_actions'])}")
        print(f"Emergent behaviors: {len(results['emergent_behaviors'])}")
        
        for behavior in results['emergent_behaviors']:
            print(f"  - {behavior['type']}: {behavior['description']}")
    
    # Run test
    asyncio.run(test_marl_system())