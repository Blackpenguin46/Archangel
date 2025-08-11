#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Learning System
Adversarial self-play and reinforcement learning for agent strategy evolution
"""

import asyncio
import logging
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid

from .base_agent import Experience, Team, Role, ActionResult, ActionPlan
from memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class LearningMode(Enum):
    """Learning mode types"""
    SELF_PLAY = "self_play"
    ADVERSARIAL = "adversarial"
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"

class StrategyType(Enum):
    """Strategy types for learning"""
    OFFENSIVE = "offensive"
    DEFENSIVE = "defensive"
    RECONNAISSANCE = "reconnaissance"
    PERSISTENCE = "persistence"
    EVASION = "evasion"

@dataclass
class LearningConfig:
    """Configuration for learning system"""
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01
    replay_buffer_size: int = 10000
    batch_size: int = 32
    target_update_frequency: int = 100
    performance_window: int = 100
    improvement_threshold: float = 0.05
    self_play_episodes: int = 1000
    strategy_evolution_cycles: int = 10

@dataclass
class PerformanceMetrics:
    """Agent performance metrics for learning"""
    agent_id: str
    timestamp: datetime
    success_rate: float
    average_reward: float
    exploration_rate: float
    learning_iterations: int
    strategy_effectiveness: Dict[str, float]
    improvement_rate: float
    confidence_score: float
    tactical_diversity: float
    adaptation_speed: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyUpdate:
    """Strategy update from learning"""
    strategy_id: str
    strategy_type: StrategyType
    agent_id: str
    old_parameters: Dict[str, Any]
    new_parameters: Dict[str, Any]
    improvement_score: float
    confidence: float
    validation_results: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SelfPlaySession:
    """Self-play session configuration and results"""
    session_id: str
    participants: List[str]  # Agent IDs
    scenario_config: Dict[str, Any]
    duration: timedelta
    episodes_completed: int
    learning_outcomes: List[Dict[str, Any]]
    performance_improvements: Dict[str, PerformanceMetrics]
    strategy_updates: List[StrategyUpdate]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class ReplayBuffer:
    """Experience replay buffer for reinforcement learning"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
        
    def add(self, experience: Experience) -> None:
        """Add experience to replay buffer"""
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences"""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear the replay buffer"""
        self.buffer.clear()
        self.position = 0

class StrategyEvolution:
    """Strategy evolution through genetic algorithm-like approach"""
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.strategies = {}  # strategy_id -> strategy_parameters
        self.fitness_scores = {}  # strategy_id -> fitness_score
        self.generation = 0
        
    def add_strategy(self, strategy_id: str, parameters: Dict[str, Any], fitness: float = 0.0) -> None:
        """Add a strategy to the evolution pool"""
        self.strategies[strategy_id] = parameters
        self.fitness_scores[strategy_id] = fitness
    
    def update_fitness(self, strategy_id: str, fitness: float) -> None:
        """Update fitness score for a strategy"""
        if strategy_id in self.fitness_scores:
            # Use exponential moving average
            alpha = 0.1
            self.fitness_scores[strategy_id] = (
                alpha * fitness + (1 - alpha) * self.fitness_scores[strategy_id]
            )
    
    def evolve_strategies(self) -> List[StrategyUpdate]:
        """Evolve strategies based on fitness scores"""
        if len(self.strategies) < 2:
            return []
        
        updates = []
        
        # Select top performers
        sorted_strategies = sorted(
            self.fitness_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_strategies = sorted_strategies[:max(2, len(sorted_strategies) // 2)]
        
        # Create new strategies through crossover and mutation
        for i in range(len(sorted_strategies) // 2):
            parent1_id, parent1_fitness = top_strategies[i % len(top_strategies)]
            parent2_id, parent2_fitness = top_strategies[(i + 1) % len(top_strategies)]
            
            # Crossover
            child_parameters = self._crossover(
                self.strategies[parent1_id],
                self.strategies[parent2_id]
            )
            
            # Mutation
            child_parameters = self._mutate(child_parameters)
            
            # Create strategy update
            child_id = f"evolved_{self.generation}_{i}"
            update = StrategyUpdate(
                strategy_id=child_id,
                strategy_type=StrategyType.OFFENSIVE,  # Would be determined from context
                agent_id="",  # Would be set by calling agent
                old_parameters=self.strategies.get(parent1_id, {}),
                new_parameters=child_parameters,
                improvement_score=(parent1_fitness + parent2_fitness) / 2,
                confidence=0.7,
                validation_results={},
                timestamp=datetime.now()
            )
            
            updates.append(update)
            self.add_strategy(child_id, child_parameters)
        
        self.generation += 1
        return updates
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two strategy parameter sets"""
        child = {}
        
        for key in set(parent1.keys()) | set(parent2.keys()):
            if key in parent1 and key in parent2:
                # Random selection from parents
                child[key] = parent1[key] if np.random.random() < 0.5 else parent2[key]
            elif key in parent1:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        return child
    
    def _mutate(self, parameters: Dict[str, Any], mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Mutate strategy parameters"""
        mutated = parameters.copy()
        
        for key, value in mutated.items():
            if np.random.random() < mutation_rate:
                if isinstance(value, (int, float)):
                    # Add gaussian noise
                    noise = np.random.normal(0, abs(value) * 0.1)
                    mutated[key] = value + noise
                elif isinstance(value, bool):
                    mutated[key] = not value
                elif isinstance(value, str):
                    # For strings, we might modify based on context
                    pass
        
        return mutated

class LearningSystem:
    """
    Main learning system for adversarial self-play and strategy evolution
    """
    
    def __init__(self, config: LearningConfig = None, memory_manager: MemoryManager = None):
        self.config = config or LearningConfig()
        self.memory_manager = memory_manager
        
        # Learning components
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        self.strategy_evolution = StrategyEvolution(self.config)
        
        # Performance tracking
        self.performance_history = {}  # agent_id -> List[PerformanceMetrics]
        self.learning_sessions = {}  # session_id -> SelfPlaySession
        
        # Learning state
        self.current_exploration_rate = self.config.exploration_rate
        self.learning_iteration = 0
        self.active_sessions = {}
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the learning system"""
        try:
            self.logger.info("Initializing learning system")
            
            # Initialize memory manager if not provided
            if self.memory_manager is None:
                from memory.memory_manager import create_memory_manager
                self.memory_manager = create_memory_manager()
                await self.memory_manager.initialize()
            
            self.initialized = True
            self.logger.info("Learning system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize learning system: {e}")
            raise
    
    async def start_self_play_session(self, 
                                    agent_ids: List[str],
                                    scenario_config: Dict[str, Any],
                                    duration: timedelta = None) -> str:
        """
        Start a self-play learning session
        
        Args:
            agent_ids: List of agent IDs to participate
            scenario_config: Scenario configuration
            duration: Session duration (default: 1 hour)
            
        Returns:
            str: Session ID
        """
        try:
            session_id = str(uuid.uuid4())
            duration = duration or timedelta(hours=1)
            
            session = SelfPlaySession(
                session_id=session_id,
                participants=agent_ids,
                scenario_config=scenario_config,
                duration=duration,
                episodes_completed=0,
                learning_outcomes=[],
                performance_improvements={},
                strategy_updates=[],
                timestamp=datetime.now()
            )
            
            self.learning_sessions[session_id] = session
            self.active_sessions[session_id] = session
            
            # Start session task
            asyncio.create_task(self._run_self_play_session(session))
            
            self.logger.info(f"Started self-play session {session_id} with {len(agent_ids)} agents")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start self-play session: {e}")
            raise
    
    async def _run_self_play_session(self, session: SelfPlaySession) -> None:
        """Run a self-play session"""
        try:
            start_time = datetime.now()
            episode = 0
            
            while (datetime.now() - start_time) < session.duration and episode < self.config.self_play_episodes:
                # Run episode
                episode_results = await self._run_self_play_episode(session, episode)
                
                # Process results
                session.learning_outcomes.append(episode_results)
                session.episodes_completed += 1
                
                # Update performance metrics
                await self._update_performance_metrics(session, episode_results)
                
                # Evolve strategies periodically
                if episode % self.config.strategy_evolution_cycles == 0:
                    strategy_updates = await self._evolve_strategies(session)
                    session.strategy_updates.extend(strategy_updates)
                
                episode += 1
                
                # Brief pause between episodes
                await asyncio.sleep(1)
            
            # Session complete
            await self._finalize_self_play_session(session)
            
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
            
            self.logger.info(f"Self-play session {session.session_id} completed: {episode} episodes")
            
        except Exception as e:
            self.logger.error(f"Error in self-play session {session.session_id}: {e}")
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
    
    async def _run_self_play_episode(self, session: SelfPlaySession, episode: int) -> Dict[str, Any]:
        """Run a single self-play episode"""
        try:
            # Simulate agent interactions
            # In a real implementation, this would involve actual agent execution
            
            episode_results = {
                "episode": episode,
                "timestamp": datetime.now(),
                "participants": session.participants,
                "actions_taken": [],
                "outcomes": {},
                "learning_signals": {},
                "performance_deltas": {}
            }
            
            # Simulate actions and outcomes for each agent
            for agent_id in session.participants:
                # Simulate action selection and execution
                action_result = await self._simulate_agent_action(agent_id, session.scenario_config)
                
                episode_results["actions_taken"].append({
                    "agent_id": agent_id,
                    "action": action_result["action"],
                    "outcome": action_result["outcome"],
                    "reward": action_result["reward"]
                })
                
                episode_results["outcomes"][agent_id] = action_result["outcome"]
                episode_results["learning_signals"][agent_id] = action_result["reward"]
            
            # Calculate performance changes
            for agent_id in session.participants:
                current_performance = await self._calculate_agent_performance(agent_id)
                episode_results["performance_deltas"][agent_id] = current_performance
            
            return episode_results
            
        except Exception as e:
            self.logger.error(f"Error in self-play episode {episode}: {e}")
            return {"error": str(e), "episode": episode}
    
    async def _simulate_agent_action(self, agent_id: str, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate an agent action for self-play"""
        # This is a simplified simulation
        # In practice, this would involve actual agent reasoning and action execution
        
        actions = ["scan_network", "exploit_vulnerability", "establish_persistence", "detect_intrusion", "block_attack"]
        action = np.random.choice(actions)
        
        # Simulate outcome based on action and scenario
        success_probability = 0.6 + np.random.normal(0, 0.1)
        success = np.random.random() < success_probability
        
        # Calculate reward
        base_reward = 1.0 if success else -0.5
        reward = base_reward + np.random.normal(0, 0.1)
        
        return {
            "action": action,
            "outcome": "success" if success else "failure",
            "reward": reward,
            "confidence": success_probability
        }
    
    async def _calculate_agent_performance(self, agent_id: str) -> PerformanceMetrics:
        """Calculate current performance metrics for an agent"""
        try:
            # Get recent experiences from memory
            if self.memory_manager:
                recent_experiences = await self.memory_manager.retrieve_similar_experiences(
                    query_text="",
                    agent_id=agent_id,
                    max_results=self.config.performance_window,
                    similarity_threshold=0.0
                )
            else:
                recent_experiences = []
            
            # Calculate metrics
            if recent_experiences:
                success_count = sum(1 for exp in recent_experiences if exp.metadata.get("success", False))
                success_rate = success_count / len(recent_experiences)
                
                avg_confidence = sum(exp.metadata.get("confidence", 0.0) for exp in recent_experiences) / len(recent_experiences)
                avg_reward = sum(exp.metadata.get("reward", 0.0) for exp in recent_experiences) / len(recent_experiences)
            else:
                success_rate = 0.0
                avg_confidence = 0.0
                avg_reward = 0.0
            
            # Get historical performance for improvement calculation
            historical_performance = self.performance_history.get(agent_id, [])
            if historical_performance:
                last_performance = historical_performance[-1]
                improvement_rate = success_rate - last_performance.success_rate
            else:
                improvement_rate = 0.0
            
            metrics = PerformanceMetrics(
                agent_id=agent_id,
                timestamp=datetime.now(),
                success_rate=success_rate,
                average_reward=avg_reward,
                exploration_rate=self.current_exploration_rate,
                learning_iterations=self.learning_iteration,
                strategy_effectiveness={
                    "offensive": success_rate * 0.8 + np.random.normal(0, 0.1),
                    "defensive": success_rate * 0.9 + np.random.normal(0, 0.1),
                    "reconnaissance": success_rate * 0.7 + np.random.normal(0, 0.1)
                },
                improvement_rate=improvement_rate,
                confidence_score=avg_confidence,
                tactical_diversity=self._calculate_tactical_diversity(recent_experiences),
                adaptation_speed=self._calculate_adaptation_speed(historical_performance)
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate agent performance: {e}")
            return PerformanceMetrics(
                agent_id=agent_id,
                timestamp=datetime.now(),
                success_rate=0.0,
                average_reward=0.0,
                exploration_rate=self.current_exploration_rate,
                learning_iterations=self.learning_iteration,
                strategy_effectiveness={},
                improvement_rate=0.0,
                confidence_score=0.0,
                tactical_diversity=0.0,
                adaptation_speed=0.0
            )
    
    def _calculate_tactical_diversity(self, experiences: List[Any]) -> float:
        """Calculate tactical diversity from experiences"""
        if not experiences:
            return 0.0
        
        # Count unique action types
        action_types = set()
        for exp in experiences:
            if hasattr(exp, 'metadata') and 'action_type' in exp.metadata:
                action_types.add(exp.metadata['action_type'])
        
        # Normalize by maximum possible diversity
        max_diversity = 10  # Assume 10 different action types
        return len(action_types) / max_diversity
    
    def _calculate_adaptation_speed(self, historical_performance: List[PerformanceMetrics]) -> float:
        """Calculate how quickly agent adapts to new situations"""
        if len(historical_performance) < 2:
            return 0.0
        
        # Calculate rate of improvement over time
        recent_metrics = historical_performance[-5:]  # Last 5 measurements
        if len(recent_metrics) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(recent_metrics)):
            improvement = recent_metrics[i].success_rate - recent_metrics[i-1].success_rate
            improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
    async def _update_performance_metrics(self, session: SelfPlaySession, episode_results: Dict[str, Any]) -> None:
        """Update performance metrics based on episode results"""
        try:
            for agent_id in session.participants:
                if agent_id in episode_results["performance_deltas"]:
                    metrics = episode_results["performance_deltas"][agent_id]
                    
                    # Add to performance history
                    if agent_id not in self.performance_history:
                        self.performance_history[agent_id] = []
                    
                    self.performance_history[agent_id].append(metrics)
                    
                    # Keep only recent history
                    if len(self.performance_history[agent_id]) > self.config.performance_window:
                        self.performance_history[agent_id] = self.performance_history[agent_id][-self.config.performance_window:]
                    
                    # Update session performance improvements
                    session.performance_improvements[agent_id] = metrics
            
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")
    
    async def _evolve_strategies(self, session: SelfPlaySession) -> List[StrategyUpdate]:
        """Evolve strategies based on session performance"""
        try:
            updates = []
            
            for agent_id in session.participants:
                if agent_id in session.performance_improvements:
                    metrics = session.performance_improvements[agent_id]
                    
                    # Update strategy fitness
                    strategy_id = f"strategy_{agent_id}"
                    fitness = metrics.success_rate * 0.7 + metrics.average_reward * 0.3
                    
                    # Add strategy if not exists
                    if strategy_id not in self.strategy_evolution.strategies:
                        self.strategy_evolution.add_strategy(
                            strategy_id,
                            {
                                "exploration_rate": self.current_exploration_rate,
                                "risk_tolerance": 0.5,
                                "cooperation_level": 0.7,
                                "aggression_level": 0.6
                            },
                            fitness
                        )
                    else:
                        self.strategy_evolution.update_fitness(strategy_id, fitness)
            
            # Evolve strategies
            strategy_updates = self.strategy_evolution.evolve_strategies()
            
            # Set agent IDs for updates
            for i, update in enumerate(strategy_updates):
                if i < len(session.participants):
                    update.agent_id = session.participants[i]
            
            updates.extend(strategy_updates)
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Failed to evolve strategies: {e}")
            return []
    
    async def _finalize_self_play_session(self, session: SelfPlaySession) -> None:
        """Finalize self-play session and extract learning outcomes"""
        try:
            # Calculate overall session performance
            total_episodes = session.episodes_completed
            total_improvements = len(session.strategy_updates)
            
            # Extract key learning outcomes
            learning_summary = {
                "session_id": session.session_id,
                "total_episodes": total_episodes,
                "strategy_updates": total_improvements,
                "performance_improvements": {},
                "key_insights": []
            }
            
            # Analyze performance improvements
            for agent_id, metrics in session.performance_improvements.items():
                learning_summary["performance_improvements"][agent_id] = {
                    "success_rate": metrics.success_rate,
                    "improvement_rate": metrics.improvement_rate,
                    "tactical_diversity": metrics.tactical_diversity,
                    "adaptation_speed": metrics.adaptation_speed
                }
            
            # Generate insights
            insights = await self._generate_learning_insights(session)
            learning_summary["key_insights"] = insights
            
            # Store learning outcomes in memory
            if self.memory_manager:
                await self._store_learning_outcomes(session, learning_summary)
            
            self.logger.info(f"Finalized self-play session {session.session_id}: {learning_summary}")
            
        except Exception as e:
            self.logger.error(f"Failed to finalize self-play session: {e}")
    
    async def _generate_learning_insights(self, session: SelfPlaySession) -> List[str]:
        """Generate learning insights from session data"""
        insights = []
        
        try:
            # Analyze performance trends
            for agent_id in session.participants:
                if agent_id in session.performance_improvements:
                    metrics = session.performance_improvements[agent_id]
                    
                    if metrics.improvement_rate > self.config.improvement_threshold:
                        insights.append(f"Agent {agent_id} showed significant improvement ({metrics.improvement_rate:.2f})")
                    
                    if metrics.tactical_diversity > 0.7:
                        insights.append(f"Agent {agent_id} demonstrated high tactical diversity ({metrics.tactical_diversity:.2f})")
                    
                    if metrics.adaptation_speed > 0.1:
                        insights.append(f"Agent {agent_id} showed fast adaptation ({metrics.adaptation_speed:.2f})")
            
            # Analyze strategy evolution
            if session.strategy_updates:
                avg_improvement = np.mean([update.improvement_score for update in session.strategy_updates])
                insights.append(f"Strategy evolution yielded average improvement of {avg_improvement:.2f}")
            
            # Analyze episode outcomes
            if session.learning_outcomes:
                success_rates = []
                for outcome in session.learning_outcomes:
                    if "learning_signals" in outcome:
                        rewards = list(outcome["learning_signals"].values())
                        if rewards:
                            success_rates.append(np.mean([r > 0 for r in rewards]))
                
                if success_rates:
                    overall_success = np.mean(success_rates)
                    insights.append(f"Overall session success rate: {overall_success:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate learning insights: {e}")
        
        return insights
    
    async def _store_learning_outcomes(self, session: SelfPlaySession, learning_summary: Dict[str, Any]) -> None:
        """Store learning outcomes in memory system"""
        try:
            # Create experience record for the learning session
            from .base_agent import Experience
            
            experience = Experience(
                experience_id=str(uuid.uuid4()),
                agent_id="learning_system",
                timestamp=datetime.now(),
                context=None,  # Would need to create appropriate context
                action_taken=None,  # Would need to create appropriate action
                reasoning=None,  # Would need to create appropriate reasoning
                outcome=None,  # Would need to create appropriate outcome
                success=len(session.strategy_updates) > 0,
                lessons_learned=learning_summary["key_insights"],
                mitre_attack_mapping=[],
                confidence_score=0.8
            )
            
            # Store in memory manager
            await self.memory_manager.store_agent_experience("learning_system", experience)
            
            self.logger.debug(f"Stored learning outcomes for session {session.session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store learning outcomes: {e}")
    
    async def add_experience_to_replay_buffer(self, experience: Experience) -> None:
        """Add experience to replay buffer for learning"""
        self.replay_buffer.add(experience)
        
        # Update exploration rate
        self.current_exploration_rate = max(
            self.config.min_exploration_rate,
            self.current_exploration_rate * self.config.exploration_decay
        )
        
        self.learning_iteration += 1
    
    async def get_learning_batch(self, batch_size: int = None) -> List[Experience]:
        """Get batch of experiences for learning"""
        batch_size = batch_size or self.config.batch_size
        return self.replay_buffer.sample(batch_size)
    
    async def get_agent_performance_history(self, agent_id: str) -> List[PerformanceMetrics]:
        """Get performance history for an agent"""
        return self.performance_history.get(agent_id, [])
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning system statistics"""
        try:
            return {
                "learning_system": {
                    "initialized": self.initialized,
                    "learning_iteration": self.learning_iteration,
                    "current_exploration_rate": self.current_exploration_rate,
                    "replay_buffer_size": self.replay_buffer.size(),
                    "active_sessions": len(self.active_sessions),
                    "total_sessions": len(self.learning_sessions),
                    "config": {
                        "learning_rate": self.config.learning_rate,
                        "discount_factor": self.config.discount_factor,
                        "batch_size": self.config.batch_size,
                        "self_play_episodes": self.config.self_play_episodes
                    }
                },
                "performance_tracking": {
                    "agents_tracked": len(self.performance_history),
                    "total_performance_records": sum(len(history) for history in self.performance_history.values())
                },
                "strategy_evolution": {
                    "generation": self.strategy_evolution.generation,
                    "strategies_tracked": len(self.strategy_evolution.strategies),
                    "fitness_scores": dict(self.strategy_evolution.fitness_scores)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get learning statistics: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown the learning system"""
        try:
            self.logger.info("Shutting down learning system")
            
            # Cancel active sessions
            for session_id in list(self.active_sessions.keys()):
                self.logger.info(f"Cancelling active session {session_id}")
                # Sessions will be cancelled by their tasks
            
            # Clear buffers
            self.replay_buffer.clear()
            
            self.initialized = False
            self.logger.info("Learning system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during learning system shutdown: {e}")

# Factory function
def create_learning_system(config: LearningConfig = None, memory_manager: MemoryManager = None) -> LearningSystem:
    """Create a learning system instance"""
    return LearningSystem(config or LearningConfig(), memory_manager)