#!/usr/bin/env python3
"""
Tests for the Self-Play Coordinator
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from agents.self_play_coordinator import (
    SelfPlayCoordinator, SelfPlayConfig, AgentMatchup, SessionResult,
    SelfPlayMode, MatchupType
)
from agents.base_agent import BaseAgent, Team, Role, AgentConfig
from agents.learning_system import LearningSystem, PerformanceMetrics

class TestSelfPlayCoordinator:
    """Test the self-play coordinator"""
    
    @pytest.fixture
    async def coordinator(self):
        """Create a coordinator for testing"""
        config = SelfPlayConfig(
            default_session_duration=timedelta(seconds=5),  # Short for testing
            max_concurrent_sessions=2,
            auto_matchmaking=False  # Disable for controlled testing
        )
        
        # Mock learning system
        learning_system = Mock(spec=LearningSystem)
        learning_system.initialize = AsyncMock()
        learning_system.start_self_play_session = AsyncMock(return_value="session_123")
        learning_system.active_sessions = {}
        learning_system.learning_sessions = {}
        learning_system._calculate_agent_performance = AsyncMock()
        
        coordinator = SelfPlayCoordinator(config, learning_system)
        await coordinator.initialize()
        
        yield coordinator
        
        await coordinator.shutdown()
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing"""
        agents = []
        
        # Red team agents
        for i in range(2):
            agent = Mock(spec=BaseAgent)
            agent.agent_id = f"red_agent_{i}"
            agent.team = Team.RED
            agent.role = Role.RECON if i == 0 else Role.EXPLOIT
            agent.name = f"Red Agent {i}"
            agents.append(agent)
        
        # Blue team agents
        for i in range(2):
            agent = Mock(spec=BaseAgent)
            agent.agent_id = f"blue_agent_{i}"
            agent.team = Team.BLUE
            agent.role = Role.SOC_ANALYST if i == 0 else Role.FIREWALL_CONFIG
            agent.name = f"Blue Agent {i}"
            agents.append(agent)
        
        return agents
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization"""
        assert coordinator.initialized
        assert coordinator.running
        assert coordinator.config is not None
        assert coordinator.learning_system is not None
    
    @pytest.mark.asyncio
    async def test_register_agent(self, coordinator, mock_agents):
        """Test agent registration"""
        agent = mock_agents[0]
        
        await coordinator.register_agent(agent)
        
        assert agent.agent_id in coordinator.registered_agents
        assert coordinator.registered_agents[agent.agent_id] == agent
        assert coordinator.agent_availability[agent.agent_id] is True
        assert agent.agent_id in coordinator.agent_performance
    
    @pytest.mark.asyncio
    async def test_unregister_agent(self, coordinator, mock_agents):
        """Test agent unregistration"""
        agent = mock_agents[0]
        
        # Register first
        await coordinator.register_agent(agent)
        assert agent.agent_id in coordinator.registered_agents
        
        # Unregister
        await coordinator.unregister_agent(agent.agent_id)
        
        assert agent.agent_id not in coordinator.registered_agents
        assert agent.agent_id not in coordinator.agent_availability
    
    @pytest.mark.asyncio
    async def test_create_manual_matchup(self, coordinator, mock_agents):
        """Test creating manual matchup"""
        # Register agents
        for agent in mock_agents:
            await coordinator.register_agent(agent)
        
        red_agents = [agent.agent_id for agent in mock_agents if agent.team == Team.RED]
        blue_agents = [agent.agent_id for agent in mock_agents if agent.team == Team.BLUE]
        
        session_id = await coordinator.create_manual_matchup(
            red_agents, blue_agents, duration=timedelta(seconds=1)
        )
        
        assert session_id is not None
        assert session_id in coordinator.active_sessions
        
        # Agents should be marked as unavailable
        for agent_id in red_agents + blue_agents:
            assert coordinator.agent_availability[agent_id] is False
    
    @pytest.mark.asyncio
    async def test_create_manual_matchup_invalid_agents(self, coordinator):
        """Test creating matchup with invalid agents"""
        with pytest.raises(ValueError, match="not registered"):
            await coordinator.create_manual_matchup(
                ["nonexistent_red"], ["nonexistent_blue"]
            )
    
    @pytest.mark.asyncio
    async def test_start_tournament(self, coordinator, mock_agents):
        """Test starting tournament"""
        # Register agents
        for agent in mock_agents:
            await coordinator.register_agent(agent)
        
        agent_ids = [agent.agent_id for agent in mock_agents]
        
        session_id = await coordinator.start_tournament(
            agent_ids,
            {"rounds": 2, "round_duration": timedelta(seconds=1)}
        )
        
        assert session_id is not None
        coordinator.learning_system.start_self_play_session.assert_called_once()
    
    def test_calculate_skill_balance(self, coordinator, mock_agents):
        """Test skill balance calculation"""
        # Register agents and set performance
        for agent in mock_agents:
            coordinator.registered_agents[agent.agent_id] = agent
            coordinator.agent_performance[agent.agent_id] = PerformanceMetrics(
                agent_id=agent.agent_id,
                timestamp=datetime.now(),
                success_rate=0.6 if agent.team == Team.RED else 0.7,
                average_reward=0.0,
                exploration_rate=0.1,
                learning_iterations=0,
                strategy_effectiveness={},
                improvement_rate=0.0,
                confidence_score=0.5,
                tactical_diversity=0.0,
                adaptation_speed=0.0
            )
        
        red_agents = [agent.agent_id for agent in mock_agents if agent.team == Team.RED]
        blue_agents = [agent.agent_id for agent in mock_agents if agent.team == Team.BLUE]
        
        balance = coordinator._calculate_skill_balance(red_agents, blue_agents)
        
        assert 0.0 <= balance <= 1.0
        # With red=0.6 and blue=0.7, balance should be 1.0 - |0.6 - 0.7| = 0.9
        assert abs(balance - 0.9) < 0.01
    
    def test_calculate_expected_competitiveness(self, coordinator, mock_agents):
        """Test expected competitiveness calculation"""
        # Register agents
        for agent in mock_agents:
            coordinator.registered_agents[agent.agent_id] = agent
            coordinator.agent_performance[agent.agent_id] = PerformanceMetrics(
                agent_id=agent.agent_id,
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
        
        red_agents = [agent.agent_id for agent in mock_agents if agent.team == Team.RED]
        blue_agents = [agent.agent_id for agent in mock_agents if agent.team == Team.BLUE]
        
        competitiveness = coordinator._calculate_expected_competitiveness(red_agents, blue_agents)
        
        assert 0.0 <= competitiveness <= 1.0
    
    def test_get_available_agents(self, coordinator, mock_agents):
        """Test getting available agents"""
        # Register some agents
        for i, agent in enumerate(mock_agents[:2]):
            coordinator.registered_agents[agent.agent_id] = agent
            coordinator.agent_availability[agent.agent_id] = i == 0  # Only first is available
        
        available = coordinator._get_available_agents()
        
        assert len(available) == 1
        assert available[0] == mock_agents[0].agent_id
    
    def test_balance_teams(self, coordinator, mock_agents):
        """Test team balancing"""
        # Set up agents with different performance levels
        red_agents = []
        blue_agents = []
        
        for agent in mock_agents:
            coordinator.registered_agents[agent.agent_id] = agent
            
            if agent.team == Team.RED:
                red_agents.append(agent.agent_id)
                # Red agents have varying performance
                success_rate = 0.8 if "0" in agent.agent_id else 0.4
            else:
                blue_agents.append(agent.agent_id)
                # Blue agents have varying performance
                success_rate = 0.9 if "0" in agent.agent_id else 0.3
            
            coordinator.agent_performance[agent.agent_id] = PerformanceMetrics(
                agent_id=agent.agent_id,
                timestamp=datetime.now(),
                success_rate=success_rate,
                average_reward=0.0,
                exploration_rate=0.1,
                learning_iterations=0,
                strategy_effectiveness={},
                improvement_rate=0.0,
                confidence_score=0.5,
                tactical_diversity=0.0,
                adaptation_speed=0.0
            )
        
        selected_red, selected_blue = coordinator._balance_teams(red_agents, blue_agents)
        
        assert len(selected_red) > 0
        assert len(selected_blue) > 0
        assert all(agent_id in red_agents for agent_id in selected_red)
        assert all(agent_id in blue_agents for agent_id in selected_blue)
    
    def test_get_default_scenario_config(self, coordinator):
        """Test default scenario configuration"""
        config = coordinator._get_default_scenario_config()
        
        assert "type" in config
        assert "environment" in config
        assert "objectives" in config
        assert "red_team" in config["objectives"]
        assert "blue_team" in config["objectives"]
        assert "constraints" in config
    
    def test_get_adaptive_scenario_config(self, coordinator, mock_agents):
        """Test adaptive scenario configuration"""
        # Register agents with different performance levels
        agent_ids = []
        for agent in mock_agents:
            coordinator.registered_agents[agent.agent_id] = agent
            agent_ids.append(agent.agent_id)
            
            # High performance agents
            coordinator.agent_performance[agent.agent_id] = PerformanceMetrics(
                agent_id=agent.agent_id,
                timestamp=datetime.now(),
                success_rate=0.8,  # High performance
                average_reward=0.0,
                exploration_rate=0.1,
                learning_iterations=0,
                strategy_effectiveness={},
                improvement_rate=0.0,
                confidence_score=0.5,
                tactical_diversity=0.0,
                adaptation_speed=0.0
            )
        
        config = coordinator._get_adaptive_scenario_config(agent_ids)
        
        assert "difficulty" in config
        assert config["difficulty"] == "hard"  # Should be hard for high-performance agents
        assert "type" in config
        assert "objectives" in config
    
    def test_create_tournament_matchups(self, coordinator, mock_agents):
        """Test tournament matchup creation"""
        # Register agents
        for agent in mock_agents:
            coordinator.registered_agents[agent.agent_id] = agent
        
        agent_ids = [agent.agent_id for agent in mock_agents]
        config = {"rounds": 2}
        
        matchups = coordinator._create_tournament_matchups(agent_ids, config)
        
        assert len(matchups) > 0
        
        for matchup in matchups:
            assert isinstance(matchup, AgentMatchup)
            assert len(matchup.red_agents) > 0
            assert len(matchup.blue_agents) > 0
            assert matchup.matchup_type == MatchupType.EVOLUTIONARY
    
    @pytest.mark.asyncio
    async def test_create_automatic_matchup(self, coordinator, mock_agents):
        """Test automatic matchup creation"""
        # Register agents
        for agent in mock_agents:
            await coordinator.register_agent(agent)
        
        available_agents = coordinator._get_available_agents()
        matchup = await coordinator._create_automatic_matchup(available_agents)
        
        assert matchup is not None
        assert isinstance(matchup, AgentMatchup)
        assert len(matchup.red_agents) > 0
        assert len(matchup.blue_agents) > 0
        assert 0.0 <= matchup.skill_balance <= 1.0
        assert 0.0 <= matchup.expected_competitiveness <= 1.0
    
    @pytest.mark.asyncio
    async def test_create_automatic_matchup_insufficient_agents(self, coordinator, mock_agents):
        """Test automatic matchup with insufficient agents"""
        # Register only red agents
        red_agents = [agent for agent in mock_agents if agent.team == Team.RED]
        for agent in red_agents:
            await coordinator.register_agent(agent)
        
        available_agents = coordinator._get_available_agents()
        matchup = await coordinator._create_automatic_matchup(available_agents)
        
        # Should return None because no blue agents available
        assert matchup is None
    
    def test_determine_winner(self, coordinator, mock_agents):
        """Test winner determination"""
        # Create mock session
        session = Mock()
        session.performance_improvements = {}
        
        # Register agents
        for agent in mock_agents:
            coordinator.registered_agents[agent.agent_id] = agent
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                agent_id=agent.agent_id,
                timestamp=datetime.now(),
                success_rate=0.8 if agent.team == Team.RED else 0.6,  # Red team performs better
                average_reward=0.0,
                exploration_rate=0.1,
                learning_iterations=0,
                strategy_effectiveness={},
                improvement_rate=0.0,
                confidence_score=0.5,
                tactical_diversity=0.0,
                adaptation_speed=0.0
            )
            session.performance_improvements[agent.agent_id] = metrics
        
        winner = coordinator._determine_winner(session)
        
        assert winner == Team.RED  # Red team should win with higher success rate
    
    def test_calculate_final_scores(self, coordinator, mock_agents):
        """Test final score calculation"""
        # Create mock session
        session = Mock()
        session.performance_improvements = {}
        
        # Add performance metrics
        for agent in mock_agents:
            metrics = PerformanceMetrics(
                agent_id=agent.agent_id,
                timestamp=datetime.now(),
                success_rate=0.7,
                average_reward=0.5,
                exploration_rate=0.1,
                learning_iterations=0,
                strategy_effectiveness={},
                improvement_rate=0.0,
                confidence_score=0.5,
                tactical_diversity=0.6,
                adaptation_speed=0.3
            )
            session.performance_improvements[agent.agent_id] = metrics
        
        scores = coordinator._calculate_final_scores(session)
        
        assert len(scores) == len(mock_agents)
        for agent in mock_agents:
            assert agent.agent_id in scores
            assert 0.0 <= scores[agent.agent_id] <= 1.0
    
    def test_extract_learning_outcomes(self, coordinator):
        """Test learning outcome extraction"""
        # Create mock session
        session = Mock()
        session.learning_outcomes = [
            {"insights": ["insight1", "insight2"]},
            {"insights": ["insight3"]}
        ]
        session.strategy_updates = [
            Mock(strategy_type=Mock(value="offensive"), improvement_score=0.2),
            Mock(strategy_type=Mock(value="defensive"), improvement_score=0.3)
        ]
        session.performance_improvements = {
            "agent1": PerformanceMetrics(
                agent_id="agent1",
                timestamp=datetime.now(),
                success_rate=0.7,
                average_reward=0.0,
                exploration_rate=0.1,
                learning_iterations=0,
                strategy_effectiveness={},
                improvement_rate=0.15,  # Significant improvement
                confidence_score=0.5,
                tactical_diversity=0.0,
                adaptation_speed=0.0
            )
        }
        
        outcomes = coordinator._extract_learning_outcomes(session)
        
        assert len(outcomes) > 0
        assert "insight1" in outcomes
        assert "insight2" in outcomes
        assert "insight3" in outcomes
        assert any("Strategy evolution" in outcome for outcome in outcomes)
        assert any("significant improvement" in outcome for outcome in outcomes)
    
    @pytest.mark.asyncio
    async def test_get_coordinator_status(self, coordinator, mock_agents):
        """Test getting coordinator status"""
        # Register some agents
        for agent in mock_agents[:2]:
            await coordinator.register_agent(agent)
        
        status = await coordinator.get_coordinator_status()
        
        assert "coordinator" in status
        assert "statistics" in status
        assert "recent_sessions" in status
        
        coordinator_info = status["coordinator"]
        assert coordinator_info["initialized"] is True
        assert coordinator_info["running"] is True
        assert coordinator_info["registered_agents"] == 2
        assert coordinator_info["available_agents"] == 2
    
    @pytest.mark.asyncio
    async def test_session_completion_handling(self, coordinator, mock_agents):
        """Test session completion handling"""
        # Register agents
        for agent in mock_agents:
            await coordinator.register_agent(agent)
        
        # Create mock matchup
        matchup = AgentMatchup(
            matchup_id="test_matchup",
            red_agents=[mock_agents[0].agent_id],
            blue_agents=[mock_agents[2].agent_id],
            matchup_type=MatchupType.ROLE_BASED,
            skill_balance=0.8,
            expected_competitiveness=0.7,
            scenario_config={},
            timestamp=datetime.now()
        )
        
        # Mock session in learning system
        mock_session = Mock()
        mock_session.timestamp = datetime.now()
        mock_session.episodes_completed = 5
        mock_session.performance_improvements = {}
        mock_session.strategy_updates = []
        mock_session.learning_outcomes = []
        
        coordinator.learning_system.learning_sessions["session_123"] = mock_session
        coordinator.learning_system.active_sessions = {}  # Session completed
        
        # Handle completion
        await coordinator._handle_session_completion("session_123", matchup)
        
        # Check that agents are available again
        assert coordinator.agent_availability[mock_agents[0].agent_id] is True
        assert coordinator.agent_availability[mock_agents[2].agent_id] is True
        
        # Check that session result was stored
        assert len(coordinator.session_history) > 0

class TestSelfPlayConfig:
    """Test self-play configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SelfPlayConfig()
        
        assert config.default_session_duration == timedelta(hours=1)
        assert config.max_concurrent_sessions == 5
        assert config.min_agents_per_session == 2
        assert config.max_agents_per_session == 8
        assert config.auto_matchmaking is True
        assert config.skill_balancing is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = SelfPlayConfig(
            default_session_duration=timedelta(minutes=30),
            max_concurrent_sessions=3,
            auto_matchmaking=False
        )
        
        assert config.default_session_duration == timedelta(minutes=30)
        assert config.max_concurrent_sessions == 3
        assert config.auto_matchmaking is False

class TestAgentMatchup:
    """Test agent matchup functionality"""
    
    def test_matchup_creation(self):
        """Test creating agent matchup"""
        matchup = AgentMatchup(
            matchup_id="test_matchup",
            red_agents=["red1", "red2"],
            blue_agents=["blue1", "blue2"],
            matchup_type=MatchupType.SKILL_BASED,
            skill_balance=0.85,
            expected_competitiveness=0.75,
            scenario_config={"type": "test"},
            timestamp=datetime.now()
        )
        
        assert matchup.matchup_id == "test_matchup"
        assert len(matchup.red_agents) == 2
        assert len(matchup.blue_agents) == 2
        assert matchup.matchup_type == MatchupType.SKILL_BASED
        assert matchup.skill_balance == 0.85
        assert matchup.expected_competitiveness == 0.75

class TestSessionResult:
    """Test session result functionality"""
    
    def test_session_result_creation(self):
        """Test creating session result"""
        matchup = AgentMatchup(
            matchup_id="test_matchup",
            red_agents=["red1"],
            blue_agents=["blue1"],
            matchup_type=MatchupType.RANDOM,
            skill_balance=0.5,
            expected_competitiveness=0.5,
            scenario_config={},
            timestamp=datetime.now()
        )
        
        result = SessionResult(
            session_id="session_123",
            matchup=matchup,
            duration=timedelta(minutes=30),
            episodes_completed=10,
            winner=Team.RED,
            final_scores={"red1": 0.8, "blue1": 0.6},
            performance_improvements={},
            learning_outcomes=["outcome1", "outcome2"],
            strategy_updates=2,
            timestamp=datetime.now()
        )
        
        assert result.session_id == "session_123"
        assert result.matchup == matchup
        assert result.duration == timedelta(minutes=30)
        assert result.episodes_completed == 10
        assert result.winner == Team.RED
        assert len(result.final_scores) == 2
        assert len(result.learning_outcomes) == 2
        assert result.strategy_updates == 2

class TestIntegration:
    """Integration tests for self-play coordinator"""
    
    @pytest.mark.asyncio
    async def test_full_matchup_cycle(self):
        """Test complete matchup cycle"""
        # Create coordinator with short durations for testing
        config = SelfPlayConfig(
            default_session_duration=timedelta(seconds=1),
            auto_matchmaking=False
        )
        
        learning_system = Mock(spec=LearningSystem)
        learning_system.initialize = AsyncMock()
        learning_system.start_self_play_session = AsyncMock(return_value="session_123")
        learning_system.active_sessions = {}
        learning_system.learning_sessions = {}
        learning_system._calculate_agent_performance = AsyncMock()
        
        coordinator = SelfPlayCoordinator(config, learning_system)
        await coordinator.initialize()
        
        try:
            # Create mock agents
            red_agent = Mock(spec=BaseAgent)
            red_agent.agent_id = "red_agent"
            red_agent.team = Team.RED
            red_agent.role = Role.RECON
            
            blue_agent = Mock(spec=BaseAgent)
            blue_agent.agent_id = "blue_agent"
            blue_agent.team = Team.BLUE
            blue_agent.role = Role.SOC_ANALYST
            
            # Register agents
            await coordinator.register_agent(red_agent)
            await coordinator.register_agent(blue_agent)
            
            # Create manual matchup
            session_id = await coordinator.create_manual_matchup(
                ["red_agent"], ["blue_agent"], duration=timedelta(seconds=1)
            )
            
            # Verify session was created
            assert session_id is not None
            learning_system.start_self_play_session.assert_called_once()
            
            # Verify agents are unavailable
            assert coordinator.agent_availability["red_agent"] is False
            assert coordinator.agent_availability["blue_agent"] is False
            
        finally:
            await coordinator.shutdown()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])