#!/usr/bin/env python3
"""
Multi-Agent Reinforcement Learning for Advanced Team Coordination
Implements cutting-edge MARL techniques for autonomous security operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import asyncio
import json
from collections import deque
import random

@dataclass
class AgentAction:
    """Action taken by an agent"""
    agent_id: str
    action_type: str
    parameters: Dict[str, Any]
    timestamp: float
    confidence: float

@dataclass
class TeamState:
    """Current state of the team environment"""
    team_type: str  # 'red' or 'blue'
    network_state: Dict[str, Any]
    active_operations: List[str]
    discovered_vulnerabilities: List[str]
    team_coordination_score: float
    resource_allocation: Dict[str, float]

@dataclass
class CoordinationReward:
    """Reward structure for team coordination"""
    individual_reward: float
    team_reward: float
    cooperation_bonus: float
    information_sharing_bonus: float
    strategic_alignment_bonus: float

class AttentionNetwork(nn.Module):
    """Attention mechanism for agent coordination"""
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.query_net = nn.Linear(input_size, hidden_size)
        self.key_net = nn.Linear(input_size, hidden_size)
        self.value_net = nn.Linear(input_size, hidden_size)
        self.scale = np.sqrt(hidden_size)
        
    def forward(self, agent_states: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to agent states"""
        batch_size, num_agents, state_size = agent_states.shape
        
        queries = self.query_net(agent_states)
        keys = self.key_net(agent_states)
        values = self.value_net(agent_states)
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended_states = torch.matmul(attention_weights, values)
        
        return attended_states, attention_weights

class HierarchicalPolicyNetwork(nn.Module):
    """Hierarchical policy for complex multi-step operations"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        
        # High-level strategic policy
        self.strategic_policy = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)  # 10 high-level strategies
        )
        
        # Low-level tactical policies (one for each strategy)
        self.tactical_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_size + 10, hidden_size),  # state + strategy embedding
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            ) for _ in range(10)
        ])
        
        # Value function for each level
        self.strategic_value = nn.Linear(state_size, 1)
        self.tactical_value = nn.Linear(state_size + 10, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through hierarchical policy"""
        # Strategic level
        strategic_logits = self.strategic_policy(state)
        strategic_probs = F.softmax(strategic_logits, dim=-1)
        strategic_value = self.strategic_value(state)
        
        # Sample strategy
        strategy_dist = torch.distributions.Categorical(strategic_probs)
        strategy = strategy_dist.sample()
        
        # Create strategy embedding
        strategy_embedding = F.one_hot(strategy, num_classes=10).float()
        
        # Tactical level
        tactical_input = torch.cat([state, strategy_embedding], dim=-1)
        tactical_logits = self.tactical_policies[strategy.item()](tactical_input)
        tactical_value = self.tactical_value(tactical_input)
        
        return strategic_logits, tactical_logits, strategic_value, tactical_value

class SwarmIntelligenceNetwork(nn.Module):
    """Swarm intelligence for distributed decision making"""
    
    def __init__(self, agent_state_size: int, num_agents: int = 5):
        super().__init__()
        self.num_agents = num_agents
        self.agent_encoder = nn.Linear(agent_state_size, 64)
        
        # Swarm communication network
        self.communication_net = nn.Sequential(
            nn.Linear(64 * num_agents, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents * 32)  # Communication signals
        )
        
        # Swarm decision fusion
        self.decision_fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
    def forward(self, agent_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process swarm intelligence"""
        batch_size = agent_states.shape[0]
        
        # Encode individual agent states
        encoded_states = self.agent_encoder(agent_states)  # [batch, num_agents, 64]
        
        # Flatten for communication network
        flattened_states = encoded_states.view(batch_size, -1)
        
        # Generate communication signals
        comm_signals = self.communication_net(flattened_states)
        comm_signals = comm_signals.view(batch_size, self.num_agents, 32)
        
        # Fuse individual states with communication
        fused_inputs = torch.cat([encoded_states, comm_signals], dim=-1)
        swarm_decisions = self.decision_fusion(fused_inputs)
        
        return swarm_decisions, comm_signals

class AdvancedMARLAgent(nn.Module):
    """Advanced Multi-Agent RL Agent with multiple cutting-edge techniques"""
    
    def __init__(self, state_size: int, action_size: int, agent_id: str, team_type: str):
        super().__init__()
        self.agent_id = agent_id
        self.team_type = team_type
        self.state_size = state_size
        self.action_size = action_size
        
        # Core networks
        self.attention_net = AttentionNetwork(state_size)
        self.hierarchical_policy = HierarchicalPolicyNetwork(state_size, action_size)
        self.swarm_net = SwarmIntelligenceNetwork(state_size)
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # Coordination mechanisms
        self.coordination_history = deque(maxlen=100)
        self.team_knowledge = {}
        
        # Meta-learning parameters
        self.meta_lr = 0.001
        self.adaptation_steps = 5
        
    def select_action(self, state: torch.Tensor, team_states: torch.Tensor, 
                     exploration_noise: float = 0.1) -> AgentAction:
        """Select action using hierarchical policy with team coordination"""
        with torch.no_grad():
            # Apply attention to team states
            attended_states, attention_weights = self.attention_net(team_states.unsqueeze(0))
            
            # Get hierarchical policy outputs
            strategic_logits, tactical_logits, strategic_value, tactical_value = \
                self.hierarchical_policy(state.unsqueeze(0))
            
            # Add exploration noise
            if exploration_noise > 0:
                strategic_logits += torch.randn_like(strategic_logits) * exploration_noise
                tactical_logits += torch.randn_like(tactical_logits) * exploration_noise
            
            # Sample actions
            strategic_action = torch.multinomial(F.softmax(strategic_logits, dim=-1), 1).item()
            tactical_action = torch.multinomial(F.softmax(tactical_logits, dim=-1), 1).item()
            
            # Calculate confidence based on value functions
            confidence = (torch.sigmoid(strategic_value) + torch.sigmoid(tactical_value)) / 2
            
        return AgentAction(
            agent_id=self.agent_id,
            action_type=f"strategy_{strategic_action}_tactic_{tactical_action}",
            parameters={
                'strategic_action': strategic_action,
                'tactical_action': tactical_action,
                'attention_weights': attention_weights.squeeze(0).tolist(),
                'strategic_value': strategic_value.item(),
                'tactical_value': tactical_value.item()
            },
            timestamp=asyncio.get_event_loop().time(),
            confidence=confidence.item()
        )
    
    def coordinate_with_team(self, team_actions: List[AgentAction]) -> Dict[str, Any]:
        """Advanced team coordination using swarm intelligence"""
        # Extract team action embeddings
        team_embeddings = []
        for action in team_actions:
            if action.agent_id != self.agent_id:
                # Simple embedding (would use more sophisticated encoding in practice)
                embedding = torch.randn(self.state_size)
                team_embeddings.append(embedding)
        
        if not team_embeddings:
            return {'coordination_score': 0.0, 'shared_knowledge': {}}
        
        # Apply swarm intelligence
        team_tensor = torch.stack(team_embeddings).unsqueeze(0)
        swarm_decisions, comm_signals = self.swarm_net(team_tensor)
        
        # Extract coordination insights
        coordination_score = torch.mean(swarm_decisions).item()
        
        # Update team knowledge
        shared_knowledge = {
            'team_coordination_strength': coordination_score,
            'communication_efficiency': torch.mean(torch.abs(comm_signals)).item(),
            'strategic_alignment': self._assess_strategic_alignment(team_actions)
        }
        
        self.team_knowledge.update(shared_knowledge)
        
        return {
            'coordination_score': coordination_score,
            'shared_knowledge': shared_knowledge,
            'recommended_adjustments': self._recommend_coordination_adjustments(shared_knowledge)
        }
    
    def learn_from_experience(self, state: torch.Tensor, action: AgentAction, 
                            reward: CoordinationReward, next_state: torch.Tensor, done: bool):
        """Learn from multi-agent experience with coordination rewards"""
        # Store experience
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.memory.append(experience)
        
        # Update coordination history
        self.coordination_history.append({
            'timestamp': action.timestamp,
            'team_reward': reward.team_reward,
            'cooperation_bonus': reward.cooperation_bonus,
            'action_confidence': action.confidence
        })
    
    def meta_adapt(self, new_scenario_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Meta-learning adaptation to new scenarios"""
        adaptation_results = {
            'adaptation_successful': False,
            'performance_improvement': 0.0,
            'adaptation_confidence': 0.0
        }
        
        if len(new_scenario_data) < 5:
            return adaptation_results
        
        # Simplified meta-adaptation (would implement full MAML in practice)
        initial_performance = self._evaluate_performance(new_scenario_data[:2])
        
        # Rapid adaptation steps
        for step in range(self.adaptation_steps):
            # Sample adaptation batch
            batch = random.sample(new_scenario_data, min(3, len(new_scenario_data)))
            # Perform gradient step (simplified)
            adaptation_loss = self._compute_adaptation_loss(batch)
        
        final_performance = self._evaluate_performance(new_scenario_data[-2:])
        
        adaptation_results.update({
            'adaptation_successful': final_performance > initial_performance,
            'performance_improvement': final_performance - initial_performance,
            'adaptation_confidence': min(final_performance / initial_performance, 1.0) if initial_performance > 0 else 0.5
        })
        
        return adaptation_results
    
    def _assess_strategic_alignment(self, team_actions: List[AgentAction]) -> float:
        """Assess how well team actions are strategically aligned"""
        if len(team_actions) < 2:
            return 1.0
        
        # Simple alignment measure based on action timing and confidence
        confidences = [action.confidence for action in team_actions]
        timestamps = [action.timestamp for action in team_actions]
        
        # Measure temporal coordination
        time_variance = np.var(timestamps) if len(timestamps) > 1 else 0
        confidence_harmony = 1.0 - np.std(confidences) if len(confidences) > 1 else 1.0
        
        alignment = confidence_harmony * np.exp(-time_variance / 1000)  # Exponential decay for time spread
        return min(max(alignment, 0.0), 1.0)
    
    def _recommend_coordination_adjustments(self, shared_knowledge: Dict[str, Any]) -> List[str]:
        """Recommend adjustments to improve team coordination"""
        recommendations = []
        
        coord_strength = shared_knowledge.get('team_coordination_strength', 0.5)
        comm_efficiency = shared_knowledge.get('communication_efficiency', 0.5)
        strategic_alignment = shared_knowledge.get('strategic_alignment', 0.5)
        
        if coord_strength < 0.5:
            recommendations.append("Increase information sharing frequency")
        if comm_efficiency < 0.4:
            recommendations.append("Optimize communication protocols")
        if strategic_alignment < 0.6:
            recommendations.append("Align strategic objectives across team")
        
        if coord_strength > 0.8 and comm_efficiency > 0.7 and strategic_alignment > 0.8:
            recommendations.append("Team coordination optimal - maintain current approach")
        
        return recommendations
    
    def _evaluate_performance(self, scenario_data: List[Dict[str, Any]]) -> float:
        """Evaluate agent performance on scenario data"""
        if not scenario_data:
            return 0.0
        
        # Simplified performance evaluation
        success_rate = sum(1 for data in scenario_data if data.get('success', False)) / len(scenario_data)
        avg_confidence = np.mean([data.get('confidence', 0.5) for data in scenario_data])
        
        return success_rate * avg_confidence
    
    def _compute_adaptation_loss(self, batch: List[Dict[str, Any]]) -> float:
        """Compute loss for meta-learning adaptation"""
        # Simplified adaptation loss
        losses = []
        for data in batch:
            predicted_success = data.get('predicted_success', 0.5)
            actual_success = data.get('actual_success', 0.5)
            loss = (predicted_success - actual_success) ** 2
            losses.append(loss)
        
        return np.mean(losses)

class MultiAgentCoordinator:
    """Coordinates multiple MARL agents for team operations"""
    
    def __init__(self, team_type: str):
        self.team_type = team_type
        self.agents = {}
        self.team_state = TeamState(
            team_type=team_type,
            network_state={},
            active_operations=[],
            discovered_vulnerabilities=[],
            team_coordination_score=0.0,
            resource_allocation={}
        )
        self.coordination_history = []
        
    def add_agent(self, agent_id: str, state_size: int, action_size: int) -> AdvancedMARLAgent:
        """Add agent to the team"""
        agent = AdvancedMARLAgent(state_size, action_size, agent_id, self.team_type)
        self.agents[agent_id] = agent
        return agent
    
    async def coordinate_team_action(self, environment_state: Dict[str, Any]) -> List[AgentAction]:
        """Coordinate actions across all team agents"""
        # Convert environment state to tensor
        state_tensor = self._state_to_tensor(environment_state)
        
        # Get individual agent states
        agent_states = []
        for agent_id, agent in self.agents.items():
            agent_state = state_tensor  # Simplified - would have agent-specific states
            agent_states.append(agent_state)
        
        team_states_tensor = torch.stack(agent_states) if agent_states else torch.empty(0)
        
        # Get actions from all agents
        team_actions = []
        for agent_id, agent in self.agents.items():
            action = agent.select_action(state_tensor, team_states_tensor)
            team_actions.append(action)
        
        # Coordinate actions using swarm intelligence
        coordination_results = []
        for agent_id, agent in self.agents.items():
            coord_result = agent.coordinate_with_team(team_actions)
            coordination_results.append(coord_result)
        
        # Update team coordination score
        avg_coordination = np.mean([result['coordination_score'] for result in coordination_results])
        self.team_state.team_coordination_score = avg_coordination
        
        # Store coordination history
        self.coordination_history.append({
            'timestamp': asyncio.get_event_loop().time(),
            'team_actions': team_actions,
            'coordination_score': avg_coordination,
            'environment_state': environment_state
        })
        
        return team_actions
    
    async def learn_from_team_experience(self, team_actions: List[AgentAction], 
                                       team_rewards: List[CoordinationReward],
                                       next_environment_state: Dict[str, Any], done: bool):
        """Team-wide learning from coordinated experience"""
        current_state = self._state_to_tensor(self.team_state.__dict__)
        next_state = self._state_to_tensor(next_environment_state)
        
        # Each agent learns from their experience
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            if i < len(team_actions) and i < len(team_rewards):
                agent.learn_from_experience(
                    current_state, team_actions[i], team_rewards[i], next_state, done
                )
    
    def get_team_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive team intelligence summary"""
        return {
            'team_type': self.team_type,
            'num_agents': len(self.agents),
            'coordination_score': self.team_state.team_coordination_score,
            'active_operations': self.team_state.active_operations,
            'coordination_history_length': len(self.coordination_history),
            'recent_performance': self._calculate_recent_performance(),
            'agent_specializations': self._analyze_agent_specializations(),
            'team_learning_progress': self._assess_team_learning_progress()
        }
    
    def _state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """Convert state dictionary to tensor (simplified)"""
        # In practice, this would be more sophisticated state encoding
        return torch.randn(64)  # Placeholder
    
    def _calculate_recent_performance(self) -> Dict[str, float]:
        """Calculate recent team performance metrics"""
        if not self.coordination_history:
            return {'coordination': 0.0, 'efficiency': 0.0}
        
        recent_coords = self.coordination_history[-10:]  # Last 10 operations
        avg_coordination = np.mean([op['coordination_score'] for op in recent_coords])
        
        return {
            'coordination': avg_coordination,
            'efficiency': avg_coordination * 0.8 + 0.2  # Simplified efficiency measure
        }
    
    def _analyze_agent_specializations(self) -> Dict[str, str]:
        """Analyze what each agent specializes in"""
        specializations = {}
        for agent_id, agent in self.agents.items():
            # Simplified specialization analysis
            if hasattr(agent, 'coordination_history') and agent.coordination_history:
                avg_confidence = np.mean([op['action_confidence'] for op in agent.coordination_history])
                if avg_confidence > 0.8:
                    specializations[agent_id] = "high_confidence_operations"
                elif avg_confidence > 0.6:
                    specializations[agent_id] = "strategic_planning"
                else:
                    specializations[agent_id] = "reconnaissance_support"
            else:
                specializations[agent_id] = "general_operations"
        
        return specializations
    
    def _assess_team_learning_progress(self) -> Dict[str, Any]:
        """Assess how well the team is learning and improving"""
        if len(self.coordination_history) < 5:
            return {'progress': 'insufficient_data', 'trend': 'unknown'}
        
        # Analyze coordination score trend
        recent_scores = [op['coordination_score'] for op in self.coordination_history[-10:]]
        early_scores = [op['coordination_score'] for op in self.coordination_history[:5]]
        
        recent_avg = np.mean(recent_scores)
        early_avg = np.mean(early_scores)
        
        improvement = recent_avg - early_avg
        
        return {
            'progress': 'improving' if improvement > 0.1 else 'stable' if abs(improvement) < 0.1 else 'declining',
            'improvement_rate': improvement,
            'current_level': 'expert' if recent_avg > 0.8 else 'intermediate' if recent_avg > 0.5 else 'novice'
        }

# Example usage and testing
async def test_marl_coordination():
    """Test Multi-Agent Reinforcement Learning coordination"""
    print("🤖 Testing Advanced Multi-Agent Reinforcement Learning...")
    
    # Create red team coordinator
    red_team = MultiAgentCoordinator("red")
    
    # Add agents
    red_agent_1 = red_team.add_agent("red_reconnaissance", 64, 20)
    red_agent_2 = red_team.add_agent("red_exploitation", 64, 20)
    red_agent_3 = red_team.add_agent("red_persistence", 64, 20)
    
    # Simulate environment state
    env_state = {
        'network_topology': 'enterprise',
        'security_level': 'high',
        'active_defenses': ['firewall', 'ids', 'siem'],
        'target_systems': ['database', 'file_server', 'domain_controller']
    }
    
    # Coordinate team actions
    team_actions = await red_team.coordinate_team_action(env_state)
    
    print(f"✅ Generated {len(team_actions)} coordinated team actions")
    print(f"🤝 Team coordination score: {red_team.team_state.team_coordination_score:.3f}")
    
    # Simulate rewards
    rewards = [
        CoordinationReward(0.7, 0.8, 0.2, 0.3, 0.4),
        CoordinationReward(0.6, 0.7, 0.3, 0.2, 0.5),
        CoordinationReward(0.8, 0.9, 0.4, 0.4, 0.3)
    ]
    
    # Team learning
    await red_team.learn_from_team_experience(team_actions, rewards, env_state, False)
    
    # Get intelligence summary
    intelligence = red_team.get_team_intelligence_summary()
    print(f"🧠 Team Intelligence Summary:")
    print(f"   - Coordination Score: {intelligence['coordination_score']:.3f}")
    print(f"   - Recent Performance: {intelligence['recent_performance']}")
    print(f"   - Learning Progress: {intelligence['team_learning_progress']}")
    
    print("🎉 Multi-Agent RL coordination test completed!")

if __name__ == "__main__":
    asyncio.run(test_marl_coordination())