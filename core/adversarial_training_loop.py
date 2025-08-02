"""
Adversarial Training Loop for AI vs AI Cybersecurity
Self-evolving continuous training system for competitive AI agents
"""

import asyncio
import logging
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from datetime import datetime, timedelta
import random
import pickle
import os
from enum import Enum
import threading
import time

from marl_coordinator import MARLCoordinator, GameState, Action, AgentType
from llm_reasoning_engine import AdversarialLLMFramework, ReasoningContext, ReasoningType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvolutionPhase(Enum):
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    ADAPTATION = "adaptation"
    CONVERGENCE = "convergence"

class CompetitiveOutcome(Enum):
    RED_VICTORY = "red_victory"
    BLUE_VICTORY = "blue_victory"
    STALEMATE = "stalemate"
    ONGOING = "ongoing"

@dataclass
class TrainingEpisode:
    """Single training episode data"""
    episode_id: str
    red_team_performance: Dict[str, float]
    blue_team_performance: Dict[str, float]
    outcome: CompetitiveOutcome
    duration: float
    novel_strategies: List[Dict[str, Any]]
    emergent_behaviors: List[Dict[str, Any]]
    learning_metrics: Dict[str, float]
    timestamp: datetime

@dataclass
class EvolutionaryPressure:
    """Represents evolutionary pressure in the system"""
    pressure_type: str
    intensity: float  # 0-1 scale
    target_team: str  # 'red', 'blue', or 'both'
    adaptation_driver: str
    duration: int  # episodes

class AdversarialTrainingLoop:
    """Main adversarial training system orchestrating AI vs AI evolution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_active = False
        self.current_episode = 0
        self.max_episodes = config.get('max_episodes', 10000)
        
        # Core components
        self.marl_coordinator = MARLCoordinator()
        self.llm_framework = AdversarialLLMFramework()
        self.evolution_engine = EvolutionEngine()
        self.meta_learner = MetaLearningSystem()
        
        # Training state
        self.episode_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        self.evolutionary_pressures = []
        self.current_phase = EvolutionPhase.EXPLORATION
        
        # Adaptive parameters
        self.learning_rates = {
            'red_team': 0.001,
            'blue_team': 0.001,
            'meta_learning': 0.0001
        }
        
        # Self-modification tracking
        self.strategy_mutations = deque(maxlen=500)
        self.successful_adaptations = defaultdict(list)
        
        logger.info("Adversarial Training Loop initialized")
    
    async def start_continuous_training(self):
        """Start the continuous adversarial training process"""
        self.training_active = True
        logger.info("Starting continuous adversarial training")
        
        # Initialize agents
        await self._initialize_training_environment()
        
        # Main training loop
        while self.training_active and self.current_episode < self.max_episodes:
            try:
                # Run training episode
                episode_result = await self._run_training_episode()
                
                # Process results and adapt
                await self._process_episode_results(episode_result)
                
                # Check for phase transitions
                await self._evaluate_evolution_phase()
                
                # Apply evolutionary pressures
                await self._apply_evolutionary_pressures()
                
                # Meta-learning updates
                await self._meta_learning_update()
                
                self.current_episode += 1
                
                # Periodic checkpointing
                if self.current_episode % 100 == 0:
                    await self._checkpoint_training_state()
                
                # Brief pause between episodes
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Training episode {self.current_episode} failed: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Training completed after {self.current_episode} episodes")
    
    async def _initialize_training_environment(self):
        """Initialize the training environment and agents"""
        # Initialize MARL agents
        red_config = {
            'num_agents': self.config.get('red_agents', 3),
            'specializations': ['scanner', 'exploiter', 'lateral_mover']
        }
        blue_config = {
            'num_agents': self.config.get('blue_agents', 3),
            'specializations': ['monitor', 'analyst', 'responder']
        }
        
        self.marl_coordinator.initialize_agents(red_config, blue_config)
        
        # Initialize evolutionary pressures
        self._initialize_evolutionary_pressures()
        
        logger.info("Training environment initialized")
    
    def _initialize_evolutionary_pressures(self):
        """Initialize basic evolutionary pressures"""
        pressures = [
            EvolutionaryPressure(
                pressure_type="performance_pressure",
                intensity=0.5,
                target_team="both",
                adaptation_driver="win_rate_optimization",
                duration=50
            ),
            EvolutionaryPressure(
                pressure_type="novelty_pressure",
                intensity=0.3,
                target_team="both", 
                adaptation_driver="strategy_diversification",
                duration=100
            ),
            EvolutionaryPressure(
                pressure_type="efficiency_pressure",
                intensity=0.4,
                target_team="both",
                adaptation_driver="resource_optimization",
                duration=75
            )
        ]
        
        self.evolutionary_pressures.extend(pressures)
    
    async def _run_training_episode(self) -> TrainingEpisode:
        """Run a single training episode"""
        episode_start = time.time()
        episode_id = f"episode_{self.current_episode:06d}"
        
        # Create dynamic game state
        game_state = await self._generate_dynamic_game_state()
        
        # Initialize episode tracking
        red_actions_log = []
        blue_actions_log = []
        novel_strategies = []
        emergent_behaviors = []
        
        # Episode loop (multiple simulation steps)
        max_steps = self.config.get('max_steps_per_episode', 50)
        for step in range(max_steps):
            
            # Run MARL simulation step
            step_results = await self.marl_coordinator.run_simulation_step(game_state)
            
            # Run LLM adversarial reasoning
            llm_results = await self._run_adversarial_reasoning_step(game_state, step_results)
            
            # Update game state based on actions
            game_state = await self._update_game_state(game_state, step_results, llm_results)
            
            # Collect data
            red_actions_log.extend(step_results['red_actions'])
            blue_actions_log.extend(step_results['blue_actions'])
            emergent_behaviors.extend(step_results.get('emergent_behaviors', []))
            
            # Detect novel strategies
            novel_strategies.extend(await self._detect_novel_strategies(step_results, llm_results))
            
            # Check for episode termination conditions
            outcome = self._evaluate_episode_outcome(game_state)
            if outcome != CompetitiveOutcome.ONGOING:
                break
        
        # Calculate performance metrics
        red_performance = await self._calculate_team_performance(red_actions_log, 'red')
        blue_performance = await self._calculate_team_performance(blue_actions_log, 'blue')
        
        episode_duration = time.time() - episode_start
        
        # Create episode record
        episode = TrainingEpisode(
            episode_id=episode_id,
            red_team_performance=red_performance,
            blue_team_performance=blue_performance,
            outcome=outcome,
            duration=episode_duration,
            novel_strategies=novel_strategies,
            emergent_behaviors=emergent_behaviors,
            learning_metrics=await self._calculate_learning_metrics(),
            timestamp=datetime.now()
        )
        
        self.episode_history.append(episode)
        
        logger.info(f"Episode {episode_id} completed: {outcome.value} in {episode_duration:.2f}s")
        return episode
    
    async def _generate_dynamic_game_state(self) -> GameState:
        """Generate a dynamic game state for training"""
        # Base network topology
        base_hosts = ['web-server', 'db-server', 'file-server', 'domain-controller', 'workstation-1', 'workstation-2']
        
        # Randomly modify topology for variety
        num_hosts = random.randint(4, len(base_hosts))
        selected_hosts = random.sample(base_hosts, num_hosts)
        
        # Generate random vulnerabilities
        possible_vulns = [
            'CVE-2024-001', 'CVE-2024-002', 'CVE-2024-003',
            'SQL-INJECTION-001', 'XSS-002', 'RCE-003'
        ]
        num_vulns = random.randint(1, 4)
        active_vulns = random.sample(possible_vulns, num_vulns)
        
        # Generate asset values with some randomization
        base_values = {
            'web-server': 500000,
            'db-server': 2000000,
            'file-server': 1000000,
            'domain-controller': 3000000,
            'workstation-1': 50000,
            'workstation-2': 50000
        }
        
        # Add random variation (±20%)
        asset_values = {}
        for host in selected_hosts:
            base_value = base_values.get(host, 100000)
            variation = random.uniform(0.8, 1.2)
            asset_values[host] = int(base_value * variation)
        
        return GameState(
            network_topology={
                'hosts': selected_hosts,
                'services': ['http', 'mysql', 'ftp', 'ldap', 'ssh']
            },
            compromised_hosts=[],
            detected_attacks=[],
            active_vulnerabilities=active_vulns,
            defensive_measures=['firewall', 'ids'],
            asset_values=asset_values,
            time_step=0,
            game_score={'red_team': 0, 'blue_team': 0}
        )
    
    async def _run_adversarial_reasoning_step(self, game_state: GameState, 
                                           step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run adversarial LLM reasoning for the current step"""
        
        # Create reasoning contexts
        red_context = ReasoningContext(
            agent_id="red_team_coordinator",
            team="red",
            specialization="strategic_coordinator",
            current_objective="Maximize asset compromise",
            available_tools=['nmap', 'metasploit', 'sqlmap', 'hydra'],
            network_state=asdict(game_state),
            threat_landscape={'active_threats': len(game_state.active_vulnerabilities)},
            historical_actions=step_results.get('red_actions', []),
            adversary_patterns={},
            time_pressure=0.7,
            risk_tolerance=0.6
        )
        
        blue_context = ReasoningContext(
            agent_id="blue_team_coordinator",
            team="blue",
            specialization="defense_coordinator",
            current_objective="Protect critical assets",
            available_tools=['splunk', 'wireshark', 'osquery', 'yara'],
            network_state=asdict(game_state),
            threat_landscape={'detected_attacks': len(game_state.detected_attacks)},
            historical_actions=step_results.get('blue_actions', []),
            adversary_patterns={},
            time_pressure=0.5,
            risk_tolerance=0.3
        )
        
        # Run adversarial reasoning cycle
        reasoning_results = await self.llm_framework.adversarial_reasoning_cycle(
            red_context, blue_context
        )
        
        return reasoning_results
    
    async def _update_game_state(self, game_state: GameState, 
                               step_results: Dict[str, Any],
                               llm_results: Dict[str, Any]) -> GameState:
        """Update game state based on actions and reasoning"""
        
        # Process red team actions
        for action in step_results.get('red_actions', []):
            success = self._simulate_action_success(action, game_state)
            
            if success:
                if action.action_type.value == 'exploit_attempt':
                    if action.target not in game_state.compromised_hosts:
                        game_state.compromised_hosts.append(action.target)
                        game_state.game_score['red_team'] += game_state.asset_values.get(action.target, 0)
                
                elif action.action_type.value == 'vulnerability_scan':
                    # Might detect new vulnerabilities
                    if random.random() < 0.3:
                        new_vuln = f"DISCOVERED-{random.randint(1000, 9999)}"
                        if new_vuln not in game_state.active_vulnerabilities:
                            game_state.active_vulnerabilities.append(new_vuln)
        
        # Process blue team actions
        for action in step_results.get('blue_actions', []):
            success = self._simulate_action_success(action, game_state)
            
            if success:
                if action.action_type.value == 'patch_vulnerability':
                    if game_state.active_vulnerabilities:
                        patched_vuln = random.choice(game_state.active_vulnerabilities)
                        game_state.active_vulnerabilities.remove(patched_vuln)
                        game_state.game_score['blue_team'] += 100000
                
                elif action.action_type.value == 'isolate_host':
                    if action.target in game_state.compromised_hosts:
                        game_state.compromised_hosts.remove(action.target)
                        game_state.game_score['blue_team'] += 50000
                
                elif action.action_type.value == 'monitor_network':
                    # Might detect attacks
                    if random.random() < 0.4:
                        detected_attack = f"DETECTED-{random.randint(1000, 9999)}"
                        game_state.detected_attacks.append(detected_attack)
        
        game_state.time_step += 1
        return game_state
    
    def _simulate_action_success(self, action: Action, game_state: GameState) -> bool:
        """Simulate whether an action succeeds"""
        # Use the action's success probability with some randomness
        base_prob = action.success_probability
        
        # Environmental factors
        if hasattr(action.action_type, 'value'):
            action_type = action.action_type.value
        else:
            action_type = str(action.action_type)
        
        # Red team actions are harder if more defensive measures are active
        if action_type in ['exploit_attempt', 'lateral_movement']:
            defense_penalty = len(game_state.defensive_measures) * 0.1
            base_prob = max(0.1, base_prob - defense_penalty)
        
        # Blue team actions are easier with more defensive measures
        elif action_type in ['monitor_network', 'analyze_logs']:
            defense_bonus = len(game_state.defensive_measures) * 0.05
            base_prob = min(0.95, base_prob + defense_bonus)
        
        return random.random() < base_prob
    
    def _evaluate_episode_outcome(self, game_state: GameState) -> CompetitiveOutcome:
        """Evaluate the current episode outcome"""
        red_score = game_state.game_score.get('red_team', 0)
        blue_score = game_state.game_score.get('blue_team', 0)
        
        total_assets = sum(game_state.asset_values.values())
        
        # Red team wins if they compromise >50% of assets
        if red_score > total_assets * 0.5:
            return CompetitiveOutcome.RED_VICTORY
        
        # Blue team wins if they maintain >80% asset integrity
        elif red_score < total_assets * 0.2:
            return CompetitiveOutcome.BLUE_VICTORY
        
        # Check for time-based outcomes
        elif game_state.time_step > 45:  # Near max steps
            if red_score > blue_score:
                return CompetitiveOutcome.RED_VICTORY
            elif blue_score > red_score:
                return CompetitiveOutcome.BLUE_VICTORY
            else:
                return CompetitiveOutcome.STALEMATE
        
        return CompetitiveOutcome.ONGOING
    
    async def _detect_novel_strategies(self, step_results: Dict[str, Any], 
                                     llm_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect novel strategies in agent behavior"""
        novel_strategies = []
        
        # Analyze action patterns for novelty
        red_actions = step_results.get('red_actions', [])
        blue_actions = step_results.get('blue_actions', [])
        
        # Check for new action sequences
        red_sequence = [action.action_type.value if hasattr(action.action_type, 'value') 
                       else str(action.action_type) for action in red_actions]
        
        if len(red_sequence) >= 2:
            sequence_key = '->'.join(red_sequence)
            if not self._has_seen_sequence(sequence_key, 'red'):
                novel_strategies.append({
                    'type': 'novel_action_sequence',
                    'team': 'red',
                    'sequence': sequence_key,
                    'description': f'New red team action sequence: {sequence_key}'
                })
                self._record_sequence(sequence_key, 'red')
        
        # Check for novel coordination patterns
        emergent_behaviors = step_results.get('emergent_behaviors', [])
        for behavior in emergent_behaviors:
            if behavior['type'] not in self._get_known_behaviors():
                novel_strategies.append({
                    'type': 'novel_coordination',
                    'team': 'both',
                    'behavior': behavior,
                    'description': f"New emergent behavior: {behavior['type']}"
                })
        
        # Check LLM reasoning novelty
        if llm_results.get('game_analysis', {}).get('predicted_winner') == 'stalemate':
            # Stalemate might indicate novel balance
            novel_strategies.append({
                'type': 'strategic_balance',
                'team': 'both',
                'description': 'Novel strategic balance achieved between teams'
            })
        
        return novel_strategies
    
    def _has_seen_sequence(self, sequence: str, team: str) -> bool:
        """Check if we've seen this action sequence before"""
        # Simplified implementation - in practice would use more sophisticated tracking
        sequence_history = getattr(self, f'_{team}_sequences', set())
        return sequence in sequence_history
    
    def _record_sequence(self, sequence: str, team: str):
        """Record a new action sequence"""
        if not hasattr(self, f'_{team}_sequences'):
            setattr(self, f'_{team}_sequences', set())
        
        sequence_history = getattr(self, f'_{team}_sequences')
        sequence_history.add(sequence)
    
    def _get_known_behaviors(self) -> List[str]:
        """Get list of known emergent behaviors"""
        return [
            'coordinated_focus_attack',
            'coordinated_isolation',
            'apt_simulation',
            'defensive_clustering'
        ]
    
    async def _calculate_team_performance(self, actions: List[Action], team: str) -> Dict[str, float]:
        """Calculate performance metrics for a team"""
        if not actions:
            return {'success_rate': 0, 'efficiency': 0, 'coordination': 0}
        
        # Calculate success rate (simplified)
        total_success_prob = sum(action.success_probability for action in actions)
        avg_success_rate = total_success_prob / len(actions)
        
        # Calculate efficiency (actions per time unit)
        time_span = (actions[-1].timestamp - actions[0].timestamp).total_seconds()
        efficiency = len(actions) / max(time_span, 1)
        
        # Calculate coordination score (based on target overlap)
        targets = [action.target for action in actions]
        unique_targets = len(set(targets))
        coordination = 1.0 - (unique_targets / len(targets)) if targets else 0
        
        return {
            'success_rate': avg_success_rate,
            'efficiency': min(efficiency / 10, 1.0),  # Normalize
            'coordination': coordination,
            'action_count': len(actions)
        }
    
    async def _calculate_learning_metrics(self) -> Dict[str, float]:
        """Calculate learning progress metrics"""
        if len(self.episode_history) < 2:
            return {'learning_rate': 0, 'adaptation_speed': 0, 'novelty_rate': 0}
        
        recent_episodes = list(self.episode_history)[-10:]
        
        # Calculate learning rate (improvement over time)
        red_scores = [ep.red_team_performance['success_rate'] for ep in recent_episodes]
        blue_scores = [ep.blue_team_performance['success_rate'] for ep in recent_episodes]
        
        red_trend = np.polyfit(range(len(red_scores)), red_scores, 1)[0] if len(red_scores) > 1 else 0
        blue_trend = np.polyfit(range(len(blue_scores)), blue_scores, 1)[0] if len(blue_scores) > 1 else 0
        
        # Calculate adaptation speed (how quickly strategies change)
        novel_strategies_count = sum(len(ep.novel_strategies) for ep in recent_episodes)
        adaptation_speed = novel_strategies_count / len(recent_episodes)
        
        # Calculate novelty rate
        novelty_rate = adaptation_speed / 5.0  # Normalize
        
        return {
            'red_learning_rate': red_trend,
            'blue_learning_rate': blue_trend,
            'adaptation_speed': adaptation_speed,
            'novelty_rate': min(novelty_rate, 1.0)
        }
    
    async def _process_episode_results(self, episode: TrainingEpisode):
        """Process episode results and update systems"""
        
        # Update performance tracking
        self.performance_metrics['red_success_rate'].append(
            episode.red_team_performance['success_rate']
        )
        self.performance_metrics['blue_success_rate'].append(
            episode.blue_team_performance['success_rate']
        )
        
        # Record strategy mutations
        for strategy in episode.novel_strategies:
            self.strategy_mutations.append({
                'episode': episode.episode_id,
                'strategy': strategy,
                'outcome': episode.outcome.value
            })
        
        # Update successful adaptations
        if episode.outcome != CompetitiveOutcome.STALEMATE:
            winning_team = 'red' if episode.outcome == CompetitiveOutcome.RED_VICTORY else 'blue'
            
            for strategy in episode.novel_strategies:
                if strategy.get('team') == winning_team or strategy.get('team') == 'both':
                    self.successful_adaptations[winning_team].append(strategy)
        
        # Adjust learning rates based on performance
        await self._adjust_learning_rates(episode)
        
        logger.info(f"Processed episode {episode.episode_id}: "
                   f"{len(episode.novel_strategies)} novel strategies, "
                   f"{len(episode.emergent_behaviors)} emergent behaviors")
    
    async def _adjust_learning_rates(self, episode: TrainingEpisode):
        """Dynamically adjust learning rates based on performance"""
        
        # If one team is consistently winning, increase the other team's learning rate
        recent_outcomes = [ep.outcome for ep in list(self.episode_history)[-20:]]
        
        red_wins = sum(1 for outcome in recent_outcomes if outcome == CompetitiveOutcome.RED_VICTORY)
        blue_wins = sum(1 for outcome in recent_outcomes if outcome == CompetitiveOutcome.BLUE_VICTORY)
        
        total_games = len(recent_outcomes)
        
        if total_games > 10:
            red_win_rate = red_wins / total_games
            blue_win_rate = blue_wins / total_games
            
            # Adjust learning rates to balance competition
            if red_win_rate > 0.7:  # Red team dominating
                self.learning_rates['blue_team'] *= 1.1
                self.learning_rates['red_team'] *= 0.95
            elif blue_win_rate > 0.7:  # Blue team dominating
                self.learning_rates['red_team'] *= 1.1
                self.learning_rates['blue_team'] *= 0.95
            
            # Keep learning rates in reasonable bounds
            for team in ['red_team', 'blue_team']:
                self.learning_rates[team] = np.clip(self.learning_rates[team], 0.0001, 0.01)
    
    async def _evaluate_evolution_phase(self):
        """Evaluate current evolution phase and transition if needed"""
        if len(self.episode_history) < 50:
            self.current_phase = EvolutionPhase.EXPLORATION
            return
        
        recent_episodes = list(self.episode_history)[-50:]
        
        # Calculate novelty rate
        total_novel_strategies = sum(len(ep.novel_strategies) for ep in recent_episodes)
        novelty_rate = total_novel_strategies / len(recent_episodes)
        
        # Calculate performance stability
        red_scores = [ep.red_team_performance['success_rate'] for ep in recent_episodes]
        blue_scores = [ep.blue_team_performance['success_rate'] for ep in recent_episodes]
        
        red_stability = 1.0 - np.std(red_scores)
        blue_stability = 1.0 - np.std(blue_scores)
        avg_stability = (red_stability + blue_stability) / 2
        
        # Phase transition logic
        if novelty_rate > 2.0:
            self.current_phase = EvolutionPhase.EXPLORATION
        elif novelty_rate > 1.0 and avg_stability < 0.8:
            self.current_phase = EvolutionPhase.EXPLOITATION
        elif avg_stability > 0.8:
            self.current_phase = EvolutionPhase.ADAPTATION
        else:
            self.current_phase = EvolutionPhase.CONVERGENCE
        
        logger.info(f"Evolution phase: {self.current_phase.value} "
                   f"(novelty: {novelty_rate:.2f}, stability: {avg_stability:.2f})")
    
    async def _apply_evolutionary_pressures(self):
        """Apply evolutionary pressures to drive adaptation"""
        
        for pressure in self.evolutionary_pressures:
            if pressure.duration > 0:
                await self._apply_single_pressure(pressure)
                pressure.duration -= 1
            else:
                # Remove expired pressures
                self.evolutionary_pressures.remove(pressure)
        
        # Add new pressures based on current phase
        if self.current_phase == EvolutionPhase.EXPLORATION:
            await self._add_exploration_pressure()
        elif self.current_phase == EvolutionPhase.EXPLOITATION:
            await self._add_exploitation_pressure()
    
    async def _apply_single_pressure(self, pressure: EvolutionaryPressure):
        """Apply a single evolutionary pressure"""
        
        if pressure.pressure_type == "performance_pressure":
            # Increase reward for winning
            self._modify_reward_structure(pressure.intensity)
            
        elif pressure.pressure_type == "novelty_pressure":
            # Reward novel strategies more
            self._modify_novelty_rewards(pressure.intensity)
            
        elif pressure.pressure_type == "efficiency_pressure":
            # Penalize slow or wasteful actions
            self._modify_efficiency_penalties(pressure.intensity)
    
    def _modify_reward_structure(self, intensity: float):
        """Modify reward structure based on pressure intensity"""
        # This would integrate with the MARL system's reward calculation
        # For now, we adjust the learning rates
        multiplier = 1.0 + (intensity * 0.1)
        
        for team in ['red_team', 'blue_team']:
            self.learning_rates[team] *= multiplier
    
    def _modify_novelty_rewards(self, intensity: float):
        """Modify rewards for novel strategies"""
        # In practice, this would modify the reward function in MARL agents
        # to give bonus rewards for novel action sequences
        logger.info(f"Applying novelty pressure with intensity {intensity}")
    
    def _modify_efficiency_penalties(self, intensity: float):
        """Modify penalties for inefficient actions"""
        # This would penalize actions that take too long or use too many resources
        logger.info(f"Applying efficiency pressure with intensity {intensity}")
    
    async def _add_exploration_pressure(self):
        """Add pressures that encourage exploration"""
        if random.random() < 0.1:  # 10% chance per episode
            new_pressure = EvolutionaryPressure(
                pressure_type="curiosity_pressure",
                intensity=random.uniform(0.2, 0.6),
                target_team="both",
                adaptation_driver="exploration_bonus",
                duration=random.randint(20, 50)
            )
            self.evolutionary_pressures.append(new_pressure)
    
    async def _add_exploitation_pressure(self):
        """Add pressures that encourage exploitation of successful strategies"""
        if random.random() < 0.15:  # 15% chance per episode
            new_pressure = EvolutionaryPressure(
                pressure_type="exploitation_pressure",
                intensity=random.uniform(0.3, 0.8),
                target_team="both",
                adaptation_driver="success_amplification",
                duration=random.randint(30, 80)
            )
            self.evolutionary_pressures.append(new_pressure)
    
    async def _meta_learning_update(self):
        """Update meta-learning system"""
        await self.meta_learner.update(self.episode_history, self.performance_metrics)
    
    async def _checkpoint_training_state(self):
        """Save training state checkpoint"""
        checkpoint_data = {
            'episode': self.current_episode,
            'performance_metrics': dict(self.performance_metrics),
            'learning_rates': self.learning_rates,
            'current_phase': self.current_phase.value,
            'successful_adaptations': dict(self.successful_adaptations)
        }
        
        checkpoint_path = f"checkpoints/training_checkpoint_{self.current_episode}.json"
        os.makedirs("checkpoints", exist_ok=True)
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"Training checkpoint saved: {checkpoint_path}")
    
    def stop_training(self):
        """Stop the training loop"""
        self.training_active = False
        logger.info("Training loop stopped")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        if not self.episode_history:
            return {}
        
        episodes = list(self.episode_history)
        
        # Calculate outcome distribution
        outcomes = [ep.outcome for ep in episodes]
        outcome_counts = {
            CompetitiveOutcome.RED_VICTORY: outcomes.count(CompetitiveOutcome.RED_VICTORY),
            CompetitiveOutcome.BLUE_VICTORY: outcomes.count(CompetitiveOutcome.BLUE_VICTORY),
            CompetitiveOutcome.STALEMATE: outcomes.count(CompetitiveOutcome.STALEMATE)
        }
        
        # Calculate average performance
        avg_red_performance = np.mean([ep.red_team_performance['success_rate'] for ep in episodes])
        avg_blue_performance = np.mean([ep.blue_team_performance['success_rate'] for ep in episodes])
        
        # Count total innovations
        total_novel_strategies = sum(len(ep.novel_strategies) for ep in episodes)
        total_emergent_behaviors = sum(len(ep.emergent_behaviors) for ep in episodes)
        
        return {
            'total_episodes': len(episodes),
            'current_phase': self.current_phase.value,
            'outcome_distribution': {k.value: v for k, v in outcome_counts.items()},
            'average_performance': {
                'red_team': avg_red_performance,
                'blue_team': avg_blue_performance
            },
            'innovation_metrics': {
                'novel_strategies': total_novel_strategies,
                'emergent_behaviors': total_emergent_behaviors,
                'successful_adaptations': {k: len(v) for k, v in self.successful_adaptations.items()}
            },
            'learning_rates': self.learning_rates.copy(),
            'active_pressures': len(self.evolutionary_pressures)
        }

class EvolutionEngine:
    """Engine for managing evolutionary dynamics in the training system"""
    
    def __init__(self):
        self.mutation_rate = 0.1
        self.crossover_rate = 0.3
        self.selection_pressure = 0.7
        
    async def evolve_strategies(self, successful_strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evolve successful strategies into new variants"""
        evolved_strategies = []
        
        for strategy in successful_strategies:
            # Mutation
            if random.random() < self.mutation_rate:
                mutated = await self._mutate_strategy(strategy)
                evolved_strategies.append(mutated)
            
            # Crossover with other strategies
            if random.random() < self.crossover_rate and len(successful_strategies) > 1:
                partner = random.choice([s for s in successful_strategies if s != strategy])
                crossover = await self._crossover_strategies(strategy, partner)
                evolved_strategies.append(crossover)
        
        return evolved_strategies
    
    async def _mutate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a strategy to create a variant"""
        mutated = strategy.copy()
        
        # Simple mutation: modify description
        if 'description' in mutated:
            mutated['description'] = f"Evolved: {mutated['description']}"
        
        # Add mutation marker
        mutated['mutation_generation'] = mutated.get('mutation_generation', 0) + 1
        mutated['parent_strategy'] = strategy.get('type', 'unknown')
        
        return mutated
    
    async def _crossover_strategies(self, strategy1: Dict[str, Any], 
                                  strategy2: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new strategy by combining two successful strategies"""
        crossover = {
            'type': f"hybrid_{strategy1.get('type', 'A')}_{strategy2.get('type', 'B')}",
            'description': f"Hybrid of {strategy1.get('type', 'A')} and {strategy2.get('type', 'B')}",
            'parent_strategies': [strategy1.get('type'), strategy2.get('type')],
            'crossover_generation': 1
        }
        
        return crossover

class MetaLearningSystem:
    """Meta-learning system for learning how to learn better"""
    
    def __init__(self):
        self.learning_history = deque(maxlen=1000)
        self.adaptation_patterns = defaultdict(list)
        
    async def update(self, episode_history: deque, performance_metrics: Dict[str, List[float]]):
        """Update meta-learning from episode history"""
        
        if len(episode_history) < 10:
            return
        
        recent_episodes = list(episode_history)[-10:]
        
        # Analyze learning patterns
        learning_velocity = self._calculate_learning_velocity(performance_metrics)
        adaptation_effectiveness = self._analyze_adaptation_effectiveness(recent_episodes)
        
        # Store learning insights
        insight = {
            'timestamp': datetime.now(),
            'learning_velocity': learning_velocity,
            'adaptation_effectiveness': adaptation_effectiveness,
            'phase_transitions': self._analyze_phase_transitions(recent_episodes)
        }
        
        self.learning_history.append(insight)
    
    def _calculate_learning_velocity(self, metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate how fast each team is learning"""
        velocity = {}
        
        for team, scores in metrics.items():
            if len(scores) >= 5:
                recent_scores = scores[-5:]
                older_scores = scores[-10:-5] if len(scores) >= 10 else scores[:-5]
                
                recent_avg = np.mean(recent_scores)
                older_avg = np.mean(older_scores) if older_scores else recent_avg
                
                velocity[team] = recent_avg - older_avg
            else:
                velocity[team] = 0
        
        return velocity
    
    def _analyze_adaptation_effectiveness(self, episodes: List[TrainingEpisode]) -> float:
        """Analyze how effective adaptations are"""
        total_adaptations = sum(len(ep.novel_strategies) for ep in episodes)
        
        if total_adaptations == 0:
            return 0
        
        # Simplified effectiveness measure
        successful_episodes = sum(1 for ep in episodes 
                                if ep.outcome != CompetitiveOutcome.STALEMATE)
        
        return successful_episodes / len(episodes)
    
    def _analyze_phase_transitions(self, episodes: List[TrainingEpisode]) -> List[str]:
        """Analyze patterns in phase transitions"""
        # Simplified analysis
        return ["exploration_to_exploitation", "adaptation_cycles"]

# Example usage
if __name__ == "__main__":
    async def test_adversarial_training():
        """Test the adversarial training system"""
        config = {
            'max_episodes': 100,
            'max_steps_per_episode': 20,
            'red_agents': 2,
            'blue_agents': 2
        }
        
        training_loop = AdversarialTrainingLoop(config)
        
        # Run a few episodes
        for i in range(5):
            episode = await training_loop._run_training_episode()
            await training_loop._process_episode_results(episode)
            print(f"Episode {i}: {episode.outcome.value}")
        
        # Get summary
        summary = training_loop.get_training_summary()
        print("\nTraining Summary:")
        print(json.dumps(summary, indent=2, default=str))
    
    # Run test
    asyncio.run(test_adversarial_training())