#!/usr/bin/env python3
"""
Tests for the Learning System
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from agents.learning_system import (
    LearningSystem, LearningConfig, PerformanceMetrics, StrategyUpdate,
    SelfPlaySession, LearningMode, StrategyType, ReplayBuffer, StrategyEvolution
)
from agents.base_agent import Experience, Team, Role, ActionResult, ActionPlan
from memory.memory_manager import MemoryManager

class TestReplayBuffer:
    """Test the replay buffer component"""
    
    def test_replay_buffer_initialization(self):
        """Test replay buffer initialization"""
        buffer = ReplayBuffer(max_size=100)
        
        assert buffer.max_size == 100
        assert buffer.size() == 0
        assert len(buffer.buffer) == 0
    
    def test_replay_buffer_add_experience(self):
        """Test adding experiences to replay buffer"""
        buffer = ReplayBuffer(max_size=3)
        
        # Create mock experiences
        exp1 = Mock(spec=Experience)
        exp1.experience_id = "exp1"
        
        exp2 = Mock(spec=Experience)
        exp2.experience_id = "exp2"
        
        # Add experiences
        buffer.add(exp1)
        buffer.add(exp2)
        
        assert buffer.size() == 2
        assert buffer.buffer[0] == exp1
        assert buffer.buffer[1] == exp2
    
    def test_replay_buffer_overflow(self):
        """Test replay buffer overflow behavior"""
        buffer = ReplayBuffer(max_size=2)
        
        # Create mock experiences
        experiences = []
        for i in range(4):
            exp = Mock(spec=Experience)
            exp.experience_id = f"exp{i}"
            experiences.append(exp)
            buffer.add(exp)
        
        # Buffer should only contain last 2 experiences
        assert buffer.size() == 2
        assert buffer.buffer[0] == experiences[2]  # exp2 overwrote exp0
        assert buffer.buffer[1] == experiences[3]  # exp3 overwrote exp1
    
    def test_replay_buffer_sample(self):
        """Test sampling from replay buffer"""
        buffer = ReplayBuffer(max_size=10)
        
        # Add experiences
        experiences = []
        for i in range(5):
            exp = Mock(spec=Experience)
            exp.experience_id = f"exp{i}"
            experiences.append(exp)
            buffer.add(exp)
        
        # Sample batch
        batch = buffer.sample(3)
        
        assert len(batch) == 3
        assert all(exp in experiences for exp in batch)
    
    def test_replay_buffer_sample_larger_than_buffer(self):
        """Test sampling more experiences than available"""
        buffer = ReplayBuffer(max_size=10)
        
        # Add 2 experiences
        for i in range(2):
            exp = Mock(spec=Experience)
            exp.experience_id = f"exp{i}"
            buffer.add(exp)
        
        # Sample 5 (more than available)
        batch = buffer.sample(5)
        
        assert len(batch) == 2  # Should return all available
    
    def test_replay_buffer_clear(self):
        """Test clearing replay buffer"""
        buffer = ReplayBuffer(max_size=10)
        
        # Add experiences
        for i in range(3):
            exp = Mock(spec=Experience)
            buffer.add(exp)
        
        assert buffer.size() == 3
        
        # Clear buffer
        buffer.clear()
        
        assert buffer.size() == 0
        assert len(buffer.buffer) == 0

class TestStrategyEvolution:
    """Test the strategy evolution component"""
    
    def test_strategy_evolution_initialization(self):
        """Test strategy evolution initialization"""
        config = LearningConfig()
        evolution = StrategyEvolution(config)
        
        assert evolution.config == config
        assert len(evolution.strategies) == 0
        assert len(evolution.fitness_scores) == 0
        assert evolution.generation == 0
    
    def test_add_strategy(self):
        """Test adding strategies to evolution pool"""
        config = LearningConfig()
        evolution = StrategyEvolution(config)
        
        strategy_params = {"param1": 0.5, "param2": 0.8}
        evolution.add_strategy("strategy1", strategy_params, 0.7)
        
        assert "strategy1" in evolution.strategies
        assert evolution.strategies["strategy1"] == strategy_params
        assert evolution.fitness_scores["strategy1"] == 0.7
    
    def test_update_fitness(self):
        """Test updating fitness scores"""
        config = LearningConfig()
        evolution = StrategyEvolution(config)
        
        # Add strategy
        evolution.add_strategy("strategy1", {"param": 0.5}, 0.5)
        
        # Update fitness
        evolution.update_fitness("strategy1", 0.8)
        
        # Should use exponential moving average
        expected_fitness = 0.1 * 0.8 + 0.9 * 0.5  # alpha=0.1
        assert abs(evolution.fitness_scores["strategy1"] - expected_fitness) < 0.01
    
    def test_evolve_strategies(self):
        """Test strategy evolution"""
        config = LearningConfig()
        evolution = StrategyEvolution(config)
        
        # Add multiple strategies
        evolution.add_strategy("strategy1", {"param1": 0.5, "param2": 0.3}, 0.8)
        evolution.add_strategy("strategy2", {"param1": 0.7, "param2": 0.6}, 0.6)
        evolution.add_strategy("strategy3", {"param1": 0.2, "param2": 0.9}, 0.4)
        
        # Evolve strategies
        updates = evolution.evolve_strategies()
        
        assert len(updates) > 0
        assert evolution.generation == 1
        
        # Check that new strategies were created
        initial_count = 3
        assert len(evolution.strategies) > initial_count

class TestLearningSystem:
    """Test the main learning system"""
    
    @pytest.fixture
    async def learning_system(self):
        """Create a learning system for testing"""
        config = LearningConfig(
            learning_rate=0.01,
            self_play_episodes=10,  # Reduced for testing
            replay_buffer_size=100
        )
        
        # Mock memory manager
        memory_manager = Mock(spec=MemoryManager)
        memory_manager.initialize = AsyncMock()
        memory_manager.retrieve_similar_experiences = AsyncMock(return_value=[])
        
        system = LearningSystem(config, memory_manager)
        await system.initialize()
        
        yield system
        
        await system.shutdown()
    
    @pytest.mark.asyncio
    async def test_learning_system_initialization(self, learning_system):
        """Test learning system initialization"""
        assert learning_system.initialized
        assert learning_system.config is not None
        assert learning_system.replay_buffer is not None
        assert learning_system.strategy_evolution is not None
    
    @pytest.mark.asyncio
    async def test_start_self_play_session(self, learning_system):
        """Test starting a self-play session"""
        agent_ids = ["agent1", "agent2"]
        scenario_config = {"type": "test_scenario"}
        
        session_id = await learning_system.start_self_play_session(
            agent_ids, scenario_config, timedelta(seconds=1)  # Very short for testing
        )
        
        assert session_id is not None
        assert session_id in learning_system.learning_sessions
        assert session_id in learning_system.active_sessions
        
        # Wait for session to complete
        await asyncio.sleep(2)
        
        # Session should be completed
        assert session_id not in learning_system.active_sessions
    
    @pytest.mark.asyncio
    async def test_calculate_agent_performance(self, learning_system):
        """Test agent performance calculation"""
        agent_id = "test_agent"
        
        # Mock memory manager to return some experiences
        mock_experiences = []
        for i in range(5):
            exp = Mock()
            exp.metadata = {
                "success": i % 2 == 0,  # Alternate success/failure
                "confidence": 0.7 + i * 0.05,
                "reward": 1.0 if i % 2 == 0 else -0.5
            }
            mock_experiences.append(exp)
        
        learning_system.memory_manager.retrieve_similar_experiences.return_value = mock_experiences
        
        # Calculate performance
        performance = await learning_system._calculate_agent_performance(agent_id)
        
        assert isinstance(performance, PerformanceMetrics)
        assert performance.agent_id == agent_id
        assert 0.0 <= performance.success_rate <= 1.0
        assert performance.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_add_experience_to_replay_buffer(self, learning_system):
        """Test adding experience to replay buffer"""
        # Create mock experience
        experience = Mock(spec=Experience)
        experience.experience_id = "test_exp"
        experience.timestamp = datetime.now()
        experience.success = True
        experience.confidence_score = 0.8
        
        initial_size = learning_system.replay_buffer.size()
        
        # Add experience
        await learning_system.add_experience_to_replay_buffer(experience)
        
        assert learning_system.replay_buffer.size() == initial_size + 1
        assert learning_system.learning_iteration > 0
    
    @pytest.mark.asyncio
    async def test_get_learning_batch(self, learning_system):
        """Test getting learning batch"""
        # Add some experiences to buffer
        for i in range(10):
            experience = Mock(spec=Experience)
            experience.experience_id = f"exp_{i}"
            experience.timestamp = datetime.now()
            experience.success = i % 2 == 0
            experience.confidence_score = 0.5 + i * 0.05
            
            await learning_system.add_experience_to_replay_buffer(experience)
        
        # Get learning batch
        batch = await learning_system.get_learning_batch(5)
        
        assert len(batch) == 5
        assert all(isinstance(exp, Mock) for exp in batch)
    
    @pytest.mark.asyncio
    async def test_get_agent_performance_history(self, learning_system):
        """Test getting agent performance history"""
        agent_id = "test_agent"
        
        # Add some performance metrics
        for i in range(3):
            metrics = PerformanceMetrics(
                agent_id=agent_id,
                timestamp=datetime.now(),
                success_rate=0.5 + i * 0.1,
                average_reward=i * 0.2,
                exploration_rate=0.1,
                learning_iterations=i,
                strategy_effectiveness={},
                improvement_rate=0.05,
                confidence_score=0.7,
                tactical_diversity=0.5,
                adaptation_speed=0.3
            )
            
            if agent_id not in learning_system.performance_history:
                learning_system.performance_history[agent_id] = []
            learning_system.performance_history[agent_id].append(metrics)
        
        # Get history
        history = await learning_system.get_agent_performance_history(agent_id)
        
        assert len(history) == 3
        assert all(isinstance(metric, PerformanceMetrics) for metric in history)
        assert all(metric.agent_id == agent_id for metric in history)
    
    @pytest.mark.asyncio
    async def test_get_learning_statistics(self, learning_system):
        """Test getting learning statistics"""
        stats = await learning_system.get_learning_statistics()
        
        assert "learning_system" in stats
        assert "performance_tracking" in stats
        assert "strategy_evolution" in stats
        
        assert "initialized" in stats["learning_system"]
        assert "learning_iteration" in stats["learning_system"]
        assert "current_exploration_rate" in stats["learning_system"]
        assert "replay_buffer_size" in stats["learning_system"]
    
    @pytest.mark.asyncio
    async def test_simulate_agent_action(self, learning_system):
        """Test agent action simulation"""
        agent_id = "test_agent"
        scenario_config = {"type": "test"}
        
        result = await learning_system._simulate_agent_action(agent_id, scenario_config)
        
        assert "action" in result
        assert "outcome" in result
        assert "reward" in result
        assert "confidence" in result
        
        assert result["outcome"] in ["success", "failure"]
        assert isinstance(result["reward"], (int, float))
        assert 0.0 <= result["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_exploration_rate_decay(self, learning_system):
        """Test exploration rate decay"""
        initial_rate = learning_system.current_exploration_rate
        
        # Add multiple experiences to trigger decay
        for i in range(10):
            experience = Mock(spec=Experience)
            experience.experience_id = f"exp_{i}"
            experience.timestamp = datetime.now()
            experience.success = True
            experience.confidence_score = 0.8
            
            await learning_system.add_experience_to_replay_buffer(experience)
        
        # Exploration rate should have decayed
        assert learning_system.current_exploration_rate < initial_rate
        assert learning_system.current_exploration_rate >= learning_system.config.min_exploration_rate

class TestPerformanceMetrics:
    """Test performance metrics functionality"""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics"""
        metrics = PerformanceMetrics(
            agent_id="test_agent",
            timestamp=datetime.now(),
            success_rate=0.75,
            average_reward=0.5,
            exploration_rate=0.1,
            learning_iterations=100,
            strategy_effectiveness={"offensive": 0.8, "defensive": 0.6},
            improvement_rate=0.05,
            confidence_score=0.85,
            tactical_diversity=0.7,
            adaptation_speed=0.4
        )
        
        assert metrics.agent_id == "test_agent"
        assert metrics.success_rate == 0.75
        assert metrics.average_reward == 0.5
        assert "offensive" in metrics.strategy_effectiveness
        assert "defensive" in metrics.strategy_effectiveness
    
    def test_performance_metrics_validation(self):
        """Test performance metrics validation"""
        # Test with valid values
        metrics = PerformanceMetrics(
            agent_id="test_agent",
            timestamp=datetime.now(),
            success_rate=0.5,
            average_reward=0.0,
            exploration_rate=0.1,
            learning_iterations=0,
            strategy_effectiveness={},
            improvement_rate=0.0,
            confidence_score=0.5,
            tactical_diversity=0.0,
            adaptation_speed=0.0
        )
        
        # All values should be within expected ranges
        assert 0.0 <= metrics.success_rate <= 1.0
        assert 0.0 <= metrics.exploration_rate <= 1.0
        assert 0.0 <= metrics.confidence_score <= 1.0
        assert 0.0 <= metrics.tactical_diversity <= 1.0
        assert 0.0 <= metrics.adaptation_speed <= 1.0

class TestStrategyUpdate:
    """Test strategy update functionality"""
    
    def test_strategy_update_creation(self):
        """Test creating strategy updates"""
        update = StrategyUpdate(
            strategy_id="strategy_1",
            strategy_type=StrategyType.OFFENSIVE,
            agent_id="agent_1",
            old_parameters={"param1": 0.5},
            new_parameters={"param1": 0.7},
            improvement_score=0.2,
            confidence=0.8,
            validation_results={"test": "passed"},
            timestamp=datetime.now()
        )
        
        assert update.strategy_id == "strategy_1"
        assert update.strategy_type == StrategyType.OFFENSIVE
        assert update.agent_id == "agent_1"
        assert update.improvement_score == 0.2
        assert update.confidence == 0.8

class TestIntegration:
    """Integration tests for learning system components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_learning_cycle(self):
        """Test complete learning cycle"""
        # Create learning system
        config = LearningConfig(
            self_play_episodes=5,  # Small number for testing
            replay_buffer_size=50
        )
        
        memory_manager = Mock(spec=MemoryManager)
        memory_manager.initialize = AsyncMock()
        memory_manager.retrieve_similar_experiences = AsyncMock(return_value=[])
        
        learning_system = LearningSystem(config, memory_manager)
        await learning_system.initialize()
        
        try:
            # Add some experiences
            for i in range(10):
                experience = Mock(spec=Experience)
                experience.experience_id = f"exp_{i}"
                experience.timestamp = datetime.now()
                experience.success = i % 3 == 0  # 1/3 success rate
                experience.confidence_score = 0.5 + (i % 5) * 0.1
                experience.lessons_learned = [f"lesson_{i}"]
                experience.mitre_attack_mapping = [f"T{1000 + i}"]
                
                await learning_system.add_experience_to_replay_buffer(experience)
            
            # Start a short self-play session
            session_id = await learning_system.start_self_play_session(
                agent_ids=["agent1", "agent2"],
                scenario_config={"type": "integration_test"},
                duration=timedelta(seconds=2)
            )
            
            # Wait for session to complete
            await asyncio.sleep(3)
            
            # Verify session completed
            assert session_id not in learning_system.active_sessions
            assert session_id in learning_system.learning_sessions
            
            # Get learning statistics
            stats = await learning_system.get_learning_statistics()
            
            assert stats["learning_system"]["replay_buffer_size"] == 10
            assert stats["learning_system"]["learning_iteration"] == 10
            
        finally:
            await learning_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_tracking_integration(self):
        """Test integration of performance tracking"""
        config = LearningConfig()
        memory_manager = Mock(spec=MemoryManager)
        memory_manager.initialize = AsyncMock()
        
        # Mock experiences with varying success rates
        mock_experiences = []
        for i in range(20):
            exp = Mock()
            exp.metadata = {
                "success": i < 15,  # 75% success rate
                "confidence": 0.6 + (i % 5) * 0.08,
                "reward": 1.0 if i < 15 else -0.5,
                "action_type": f"action_{i % 3}"
            }
            mock_experiences.append(exp)
        
        memory_manager.retrieve_similar_experiences = AsyncMock(return_value=mock_experiences)
        
        learning_system = LearningSystem(config, memory_manager)
        await learning_system.initialize()
        
        try:
            # Calculate performance for multiple agents
            agent_ids = ["agent1", "agent2", "agent3"]
            
            for agent_id in agent_ids:
                performance = await learning_system._calculate_agent_performance(agent_id)
                
                assert isinstance(performance, PerformanceMetrics)
                assert performance.agent_id == agent_id
                assert performance.success_rate == 0.75  # Based on mock data
                
                # Add to history
                if agent_id not in learning_system.performance_history:
                    learning_system.performance_history[agent_id] = []
                learning_system.performance_history[agent_id].append(performance)
            
            # Verify all agents have performance history
            for agent_id in agent_ids:
                history = await learning_system.get_agent_performance_history(agent_id)
                assert len(history) == 1
                assert history[0].agent_id == agent_id
            
        finally:
            await learning_system.shutdown()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])