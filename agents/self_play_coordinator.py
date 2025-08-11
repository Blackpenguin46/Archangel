#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Self-Play Coordinator
Coordinates self-play sessions between Red and Blue team agents
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .base_agent import BaseAgent, Team, Role, Experience
from .learning_system import LearningSystem, SelfPlaySession, PerformanceMetrics

logger = logging.getLogger(__name__)

class SelfPlayMode(Enum):
    """Self-play mode types"""
    RED_VS_BLUE = "red_vs_blue"
    RED_VS_RED = "red_vs_red"
    BLUE_VS_BLUE = "blue_vs_blue"
    COOPERATIVE = "cooperative"
    TOURNAMENT = "tournament"

class MatchupType(Enum):
    """Agent matchup types"""
    ROLE_BASED = "role_based"
    SKILL_BASED = "skill_based"
    RANDOM = "random"
    EVOLUTIONARY = "evolutionary"

@dataclass
class SelfPlayConfig:
    """Configuration for self-play coordinator"""
    default_session_duration: timedelta = timedelta(hours=1)
    max_concurrent_sessions: int = 5
    min_agents_per_session: int = 2
    max_agents_per_session: int = 8
    performance_evaluation_interval: timedelta = timedelta(minutes=10)
    strategy_update_interval: timedelta = timedelta(minutes=30)
    session_cooldown: timedelta = timedelta(minutes=5)
    auto_matchmaking: bool = True
    skill_balancing: bool = True
    learning_rate_adaptation: bool = True

@dataclass
class AgentMatchup:
    """Agent matchup for self-play"""
    matchup_id: str
    red_agents: List[str]
    blue_agents: List[str]
    matchup_type: MatchupType
    skill_balance: float
    expected_competitiveness: float
    scenario_config: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionResult:
    """Results from a self-play session"""
    session_id: str
    matchup: AgentMatchup
    duration: timedelta
    episodes_completed: int
    winner: Optional[Team]
    final_scores: Dict[str, float]
    performance_improvements: Dict[str, PerformanceMetrics]
    learning_outcomes: List[str]
    strategy_updates: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class SelfPlayCoordinator:
    """
    Coordinates self-play sessions between agents for learning and strategy evolution
    """
    
    def __init__(self, config: SelfPlayConfig = None, learning_system: LearningSystem = None):
        self.config = config or SelfPlayConfig()
        self.learning_system = learning_system
        
        # Agent management
        self.registered_agents = {}  # agent_id -> BaseAgent
        self.agent_performance = {}  # agent_id -> PerformanceMetrics
        self.agent_availability = {}  # agent_id -> bool
        
        # Session management
        self.active_sessions = {}  # session_id -> SelfPlaySession
        self.session_history = []  # List[SessionResult]
        self.pending_matchups = []  # List[AgentMatchup]
        
        # Matchmaking
        self.matchmaking_task = None
        self.performance_evaluation_task = None
        
        # Statistics
        self.stats = {
            'total_sessions': 0,
            'successful_sessions': 0,
            'total_episodes': 0,
            'total_learning_outcomes': 0,
            'average_session_duration': 0.0,
            'agent_improvement_rate': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.running = False
    
    async def initialize(self) -> None:
        """Initialize the self-play coordinator"""
        try:
            self.logger.info("Initializing self-play coordinator")
            
            # Initialize learning system if not provided
            if self.learning_system is None:
                from .learning_system import create_learning_system
                self.learning_system = create_learning_system()
                await self.learning_system.initialize()
            
            # Start background tasks
            if self.config.auto_matchmaking:
                self.matchmaking_task = asyncio.create_task(self._matchmaking_loop())
            
            self.performance_evaluation_task = asyncio.create_task(self._performance_evaluation_loop())
            
            self.initialized = True
            self.running = True
            
            self.logger.info("Self-play coordinator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize self-play coordinator: {e}")
            raise
    
    async def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for self-play sessions"""
        try:
            self.registered_agents[agent.agent_id] = agent
            self.agent_availability[agent.agent_id] = True
            
            # Initialize performance tracking
            if agent.agent_id not in self.agent_performance:
                initial_performance = PerformanceMetrics(
                    agent_id=agent.agent_id,
                    timestamp=datetime.now(),
                    success_rate=0.5,  # Start with neutral performance
                    average_reward=0.0,
                    exploration_rate=0.1,
                    learning_iterations=0,
                    strategy_effectiveness={},
                    improvement_rate=0.0,
                    confidence_score=0.5,
                    tactical_diversity=0.0,
                    adaptation_speed=0.0
                )
                self.agent_performance[agent.agent_id] = initial_performance
            
            self.logger.info(f"Registered agent {agent.agent_id} ({agent.team.value}, {agent.role.value})")
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {e}")
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from self-play sessions"""
        try:
            if agent_id in self.registered_agents:
                del self.registered_agents[agent_id]
            
            if agent_id in self.agent_availability:
                del self.agent_availability[agent_id]
            
            # Cancel any active sessions involving this agent
            sessions_to_cancel = []
            for session_id, session in self.active_sessions.items():
                if agent_id in session.participants:
                    sessions_to_cancel.append(session_id)
            
            for session_id in sessions_to_cancel:
                await self._cancel_session(session_id, f"Agent {agent_id} unregistered")
            
            self.logger.info(f"Unregistered agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
    
    async def create_manual_matchup(self, 
                                  red_agent_ids: List[str],
                                  blue_agent_ids: List[str],
                                  scenario_config: Dict[str, Any] = None,
                                  session_duration: timedelta = None) -> str:
        """
        Create a manual matchup between specific agents
        
        Args:
            red_agent_ids: List of Red team agent IDs
            blue_agent_ids: List of Blue team agent IDs
            scenario_config: Optional scenario configuration
            session_duration: Optional session duration
            
        Returns:
            str: Session ID
        """
        try:
            # Validate agents
            all_agent_ids = red_agent_ids + blue_agent_ids
            for agent_id in all_agent_ids:
                if agent_id not in self.registered_agents:
                    raise ValueError(f"Agent {agent_id} not registered")
                if not self.agent_availability.get(agent_id, False):
                    raise ValueError(f"Agent {agent_id} not available")
            
            # Create matchup
            matchup = AgentMatchup(
                matchup_id=str(uuid.uuid4()),
                red_agents=red_agent_ids,
                blue_agents=blue_agent_ids,
                matchup_type=MatchupType.ROLE_BASED,
                skill_balance=self._calculate_skill_balance(red_agent_ids, blue_agent_ids),
                expected_competitiveness=0.5,  # Will be calculated
                scenario_config=scenario_config or self._get_default_scenario_config(),
                timestamp=datetime.now()
            )
            
            # Start session
            session_id = await self._start_session_from_matchup(matchup, session_duration)
            
            self.logger.info(f"Created manual matchup {matchup.matchup_id} -> session {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to create manual matchup: {e}")
            raise
    
    async def start_tournament(self, 
                             agent_ids: List[str],
                             tournament_config: Dict[str, Any] = None) -> str:
        """
        Start a tournament-style self-play session
        
        Args:
            agent_ids: List of agent IDs to participate
            tournament_config: Tournament configuration
            
        Returns:
            str: Tournament session ID
        """
        try:
            config = tournament_config or {
                "rounds": 3,
                "elimination": False,
                "round_duration": timedelta(minutes=30)
            }
            
            # Create tournament matchups
            matchups = self._create_tournament_matchups(agent_ids, config)
            
            # Start tournament session
            session_id = await self.learning_system.start_self_play_session(
                agent_ids=agent_ids,
                scenario_config={
                    "type": "tournament",
                    "matchups": [matchup.__dict__ for matchup in matchups],
                    "config": config
                },
                duration=config.get("total_duration", timedelta(hours=2))
            )
            
            self.logger.info(f"Started tournament {session_id} with {len(agent_ids)} agents")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start tournament: {e}")
            raise
    
    async def _matchmaking_loop(self) -> None:
        """Background matchmaking loop"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not self.running:
                    break
                
                # Check if we can create new sessions
                if len(self.active_sessions) >= self.config.max_concurrent_sessions:
                    continue
                
                # Find available agents
                available_agents = self._get_available_agents()
                
                if len(available_agents) < self.config.min_agents_per_session:
                    continue
                
                # Create matchup
                matchup = await self._create_automatic_matchup(available_agents)
                
                if matchup:
                    session_id = await self._start_session_from_matchup(matchup)
                    self.logger.info(f"Auto-created session {session_id} from matchmaking")
                
            except Exception as e:
                self.logger.error(f"Error in matchmaking loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _performance_evaluation_loop(self) -> None:
        """Background performance evaluation loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config.performance_evaluation_interval.total_seconds())
                
                if not self.running:
                    break
                
                # Update agent performance metrics
                for agent_id in self.registered_agents:
                    try:
                        performance = await self.learning_system._calculate_agent_performance(agent_id)
                        self.agent_performance[agent_id] = performance
                    except Exception as e:
                        self.logger.error(f"Failed to update performance for agent {agent_id}: {e}")
                
                # Update statistics
                await self._update_coordinator_statistics()
                
            except Exception as e:
                self.logger.error(f"Error in performance evaluation loop: {e}")
                await asyncio.sleep(60)
    
    def _get_available_agents(self) -> List[str]:
        """Get list of available agent IDs"""
        return [
            agent_id for agent_id, available in self.agent_availability.items()
            if available and agent_id in self.registered_agents
        ]
    
    async def _create_automatic_matchup(self, available_agents: List[str]) -> Optional[AgentMatchup]:
        """Create an automatic matchup from available agents"""
        try:
            # Separate by team
            red_agents = []
            blue_agents = []
            
            for agent_id in available_agents:
                agent = self.registered_agents[agent_id]
                if agent.team == Team.RED:
                    red_agents.append(agent_id)
                elif agent.team == Team.BLUE:
                    blue_agents.append(agent_id)
            
            # Need at least one agent from each team
            if not red_agents or not blue_agents:
                return None
            
            # Select agents based on skill balancing
            if self.config.skill_balancing:
                selected_red, selected_blue = self._balance_teams(red_agents, blue_agents)
            else:
                # Random selection
                import random
                selected_red = random.sample(red_agents, min(2, len(red_agents)))
                selected_blue = random.sample(blue_agents, min(2, len(blue_agents)))
            
            # Create matchup
            matchup = AgentMatchup(
                matchup_id=str(uuid.uuid4()),
                red_agents=selected_red,
                blue_agents=selected_blue,
                matchup_type=MatchupType.SKILL_BASED if self.config.skill_balancing else MatchupType.RANDOM,
                skill_balance=self._calculate_skill_balance(selected_red, selected_blue),
                expected_competitiveness=self._calculate_expected_competitiveness(selected_red, selected_blue),
                scenario_config=self._get_adaptive_scenario_config(selected_red + selected_blue),
                timestamp=datetime.now()
            )
            
            return matchup
            
        except Exception as e:
            self.logger.error(f"Failed to create automatic matchup: {e}")
            return None
    
    def _balance_teams(self, red_agents: List[str], blue_agents: List[str]) -> Tuple[List[str], List[str]]:
        """Balance teams based on agent performance"""
        try:
            # Get performance scores
            red_scores = [(agent_id, self.agent_performance.get(agent_id, PerformanceMetrics(
                agent_id=agent_id, timestamp=datetime.now(), success_rate=0.5,
                average_reward=0.0, exploration_rate=0.1, learning_iterations=0,
                strategy_effectiveness={}, improvement_rate=0.0, confidence_score=0.5,
                tactical_diversity=0.0, adaptation_speed=0.0
            )).success_rate) for agent_id in red_agents]
            
            blue_scores = [(agent_id, self.agent_performance.get(agent_id, PerformanceMetrics(
                agent_id=agent_id, timestamp=datetime.now(), success_rate=0.5,
                average_reward=0.0, exploration_rate=0.1, learning_iterations=0,
                strategy_effectiveness={}, improvement_rate=0.0, confidence_score=0.5,
                tactical_diversity=0.0, adaptation_speed=0.0
            )).success_rate) for agent_id in blue_agents]
            
            # Sort by performance
            red_scores.sort(key=lambda x: x[1], reverse=True)
            blue_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select balanced teams
            selected_red = [red_scores[0][0]]  # Best red agent
            selected_blue = [blue_scores[0][0]]  # Best blue agent
            
            # Add more agents if available, trying to balance
            if len(red_scores) > 1 and len(blue_scores) > 1:
                selected_red.append(red_scores[-1][0])  # Worst red agent
                selected_blue.append(blue_scores[-1][0])  # Worst blue agent
            
            return selected_red, selected_blue
            
        except Exception as e:
            self.logger.error(f"Failed to balance teams: {e}")
            # Fallback to random selection
            import random
            return (
                random.sample(red_agents, min(2, len(red_agents))),
                random.sample(blue_agents, min(2, len(blue_agents)))
            )
    
    def _calculate_skill_balance(self, red_agents: List[str], blue_agents: List[str]) -> float:
        """Calculate skill balance between teams"""
        try:
            red_avg = sum(self.agent_performance.get(agent_id, PerformanceMetrics(
                agent_id=agent_id, timestamp=datetime.now(), success_rate=0.5,
                average_reward=0.0, exploration_rate=0.1, learning_iterations=0,
                strategy_effectiveness={}, improvement_rate=0.0, confidence_score=0.5,
                tactical_diversity=0.0, adaptation_speed=0.0
            )).success_rate for agent_id in red_agents) / len(red_agents)
            
            blue_avg = sum(self.agent_performance.get(agent_id, PerformanceMetrics(
                agent_id=agent_id, timestamp=datetime.now(), success_rate=0.5,
                average_reward=0.0, exploration_rate=0.1, learning_iterations=0,
                strategy_effectiveness={}, improvement_rate=0.0, confidence_score=0.5,
                tactical_diversity=0.0, adaptation_speed=0.0
            )).success_rate for agent_id in blue_agents) / len(blue_agents)
            
            # Return balance score (1.0 = perfectly balanced, 0.0 = completely unbalanced)
            return 1.0 - abs(red_avg - blue_avg)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate skill balance: {e}")
            return 0.5  # Neutral balance
    
    def _calculate_expected_competitiveness(self, red_agents: List[str], blue_agents: List[str]) -> float:
        """Calculate expected competitiveness of the matchup"""
        try:
            skill_balance = self._calculate_skill_balance(red_agents, blue_agents)
            
            # Factor in team diversity
            red_roles = set(self.registered_agents[agent_id].role for agent_id in red_agents)
            blue_roles = set(self.registered_agents[agent_id].role for agent_id in blue_agents)
            
            role_diversity = (len(red_roles) + len(blue_roles)) / (len(red_agents) + len(blue_agents))
            
            # Combine factors
            competitiveness = (skill_balance * 0.7) + (role_diversity * 0.3)
            
            return min(1.0, max(0.0, competitiveness))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate expected competitiveness: {e}")
            return 0.5
    
    def _get_default_scenario_config(self) -> Dict[str, Any]:
        """Get default scenario configuration"""
        return {
            "type": "standard_engagement",
            "environment": "mock_enterprise",
            "duration": "30_minutes",
            "objectives": {
                "red_team": ["gain_initial_access", "establish_persistence", "exfiltrate_data"],
                "blue_team": ["detect_intrusion", "contain_threat", "maintain_uptime"]
            },
            "constraints": {
                "no_real_world_impact": True,
                "ethical_boundaries": True,
                "logging_required": True
            }
        }
    
    def _get_adaptive_scenario_config(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Get adaptive scenario configuration based on agent capabilities"""
        base_config = self._get_default_scenario_config()
        
        try:
            # Analyze agent capabilities
            roles = set()
            avg_performance = 0.0
            
            for agent_id in agent_ids:
                agent = self.registered_agents[agent_id]
                roles.add(agent.role)
                
                performance = self.agent_performance.get(agent_id)
                if performance:
                    avg_performance += performance.success_rate
            
            avg_performance /= len(agent_ids)
            
            # Adapt scenario based on performance
            if avg_performance > 0.7:
                base_config["difficulty"] = "hard"
                base_config["objectives"]["red_team"].append("advanced_persistence")
                base_config["objectives"]["blue_team"].append("threat_hunting")
            elif avg_performance < 0.3:
                base_config["difficulty"] = "easy"
                base_config["constraints"]["guided_learning"] = True
            else:
                base_config["difficulty"] = "medium"
            
            # Adapt based on role diversity
            if len(roles) > 4:
                base_config["complexity"] = "high"
                base_config["multi_phase"] = True
            
            return base_config
            
        except Exception as e:
            self.logger.error(f"Failed to create adaptive scenario config: {e}")
            return base_config
    
    def _create_tournament_matchups(self, agent_ids: List[str], config: Dict[str, Any]) -> List[AgentMatchup]:
        """Create tournament matchups"""
        matchups = []
        
        try:
            # Simple round-robin tournament
            red_agents = [agent_id for agent_id in agent_ids if self.registered_agents[agent_id].team == Team.RED]
            blue_agents = [agent_id for agent_id in agent_ids if self.registered_agents[agent_id].team == Team.BLUE]
            
            # Create all possible matchups
            for red_agent in red_agents:
                for blue_agent in blue_agents:
                    matchup = AgentMatchup(
                        matchup_id=str(uuid.uuid4()),
                        red_agents=[red_agent],
                        blue_agents=[blue_agent],
                        matchup_type=MatchupType.EVOLUTIONARY,
                        skill_balance=self._calculate_skill_balance([red_agent], [blue_agent]),
                        expected_competitiveness=0.5,
                        scenario_config=self._get_default_scenario_config(),
                        timestamp=datetime.now(),
                        metadata={"tournament": True}
                    )
                    matchups.append(matchup)
            
            return matchups
            
        except Exception as e:
            self.logger.error(f"Failed to create tournament matchups: {e}")
            return []
    
    async def _start_session_from_matchup(self, matchup: AgentMatchup, duration: timedelta = None) -> str:
        """Start a self-play session from a matchup"""
        try:
            # Mark agents as unavailable
            all_agents = matchup.red_agents + matchup.blue_agents
            for agent_id in all_agents:
                self.agent_availability[agent_id] = False
            
            # Start learning session
            session_id = await self.learning_system.start_self_play_session(
                agent_ids=all_agents,
                scenario_config=matchup.scenario_config,
                duration=duration or self.config.default_session_duration
            )
            
            # Track session
            self.active_sessions[session_id] = matchup
            
            # Schedule session completion handling
            asyncio.create_task(self._handle_session_completion(session_id, matchup))
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start session from matchup: {e}")
            # Mark agents as available again
            all_agents = matchup.red_agents + matchup.blue_agents
            for agent_id in all_agents:
                self.agent_availability[agent_id] = True
            raise
    
    async def _handle_session_completion(self, session_id: str, matchup: AgentMatchup) -> None:
        """Handle completion of a self-play session"""
        try:
            # Wait for session to complete
            while session_id in self.learning_system.active_sessions:
                await asyncio.sleep(10)
            
            # Get session results
            if session_id in self.learning_system.learning_sessions:
                session = self.learning_system.learning_sessions[session_id]
                
                # Create session result
                result = SessionResult(
                    session_id=session_id,
                    matchup=matchup,
                    duration=datetime.now() - session.timestamp,
                    episodes_completed=session.episodes_completed,
                    winner=self._determine_winner(session),
                    final_scores=self._calculate_final_scores(session),
                    performance_improvements=session.performance_improvements,
                    learning_outcomes=self._extract_learning_outcomes(session),
                    strategy_updates=len(session.strategy_updates),
                    timestamp=datetime.now()
                )
                
                # Store result
                self.session_history.append(result)
                
                # Update statistics
                self.stats['total_sessions'] += 1
                self.stats['successful_sessions'] += 1 if session.episodes_completed > 0 else 0
                self.stats['total_episodes'] += session.episodes_completed
                self.stats['total_learning_outcomes'] += len(result.learning_outcomes)
                
                self.logger.info(f"Session {session_id} completed: {result.episodes_completed} episodes, {result.strategy_updates} strategy updates")
            
            # Mark agents as available again
            all_agents = matchup.red_agents + matchup.blue_agents
            for agent_id in all_agents:
                self.agent_availability[agent_id] = True
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Cooldown period
            await asyncio.sleep(self.config.session_cooldown.total_seconds())
            
        except Exception as e:
            self.logger.error(f"Error handling session completion {session_id}: {e}")
    
    def _determine_winner(self, session: SelfPlaySession) -> Optional[Team]:
        """Determine the winner of a session"""
        try:
            red_score = 0.0
            blue_score = 0.0
            
            for agent_id, metrics in session.performance_improvements.items():
                agent = self.registered_agents.get(agent_id)
                if agent:
                    if agent.team == Team.RED:
                        red_score += metrics.success_rate
                    elif agent.team == Team.BLUE:
                        blue_score += metrics.success_rate
            
            if red_score > blue_score:
                return Team.RED
            elif blue_score > red_score:
                return Team.BLUE
            else:
                return None  # Tie
                
        except Exception as e:
            self.logger.error(f"Failed to determine winner: {e}")
            return None
    
    def _calculate_final_scores(self, session: SelfPlaySession) -> Dict[str, float]:
        """Calculate final scores for session participants"""
        scores = {}
        
        try:
            for agent_id, metrics in session.performance_improvements.items():
                # Combine multiple performance factors
                score = (
                    metrics.success_rate * 0.4 +
                    metrics.average_reward * 0.3 +
                    metrics.tactical_diversity * 0.2 +
                    metrics.adaptation_speed * 0.1
                )
                scores[agent_id] = max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate final scores: {e}")
        
        return scores
    
    def _extract_learning_outcomes(self, session: SelfPlaySession) -> List[str]:
        """Extract learning outcomes from session"""
        outcomes = []
        
        try:
            # Extract from session learning outcomes
            for outcome in session.learning_outcomes:
                if isinstance(outcome, dict) and "insights" in outcome:
                    outcomes.extend(outcome["insights"])
            
            # Extract from strategy updates
            for update in session.strategy_updates:
                outcomes.append(f"Strategy evolution: {update.strategy_type.value} improved by {update.improvement_score:.2f}")
            
            # Extract from performance improvements
            for agent_id, metrics in session.performance_improvements.items():
                if metrics.improvement_rate > 0.1:
                    outcomes.append(f"Agent {agent_id} showed significant improvement: {metrics.improvement_rate:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to extract learning outcomes: {e}")
        
        return outcomes
    
    async def _cancel_session(self, session_id: str, reason: str) -> None:
        """Cancel an active session"""
        try:
            self.logger.info(f"Cancelling session {session_id}: {reason}")
            
            # Mark agents as available
            if session_id in self.active_sessions:
                matchup = self.active_sessions[session_id]
                all_agents = matchup.red_agents + matchup.blue_agents
                for agent_id in all_agents:
                    self.agent_availability[agent_id] = True
                
                del self.active_sessions[session_id]
            
        except Exception as e:
            self.logger.error(f"Failed to cancel session {session_id}: {e}")
    
    async def _update_coordinator_statistics(self) -> None:
        """Update coordinator statistics"""
        try:
            if self.session_history:
                total_duration = sum(result.duration.total_seconds() for result in self.session_history)
                self.stats['average_session_duration'] = total_duration / len(self.session_history)
            
            # Calculate agent improvement rate
            if self.agent_performance:
                improvement_rates = [metrics.improvement_rate for metrics in self.agent_performance.values()]
                self.stats['agent_improvement_rate'] = sum(improvement_rates) / len(improvement_rates)
            
        except Exception as e:
            self.logger.error(f"Failed to update coordinator statistics: {e}")
    
    async def get_coordinator_status(self) -> Dict[str, Any]:
        """Get comprehensive coordinator status"""
        try:
            return {
                "coordinator": {
                    "initialized": self.initialized,
                    "running": self.running,
                    "registered_agents": len(self.registered_agents),
                    "available_agents": sum(self.agent_availability.values()),
                    "active_sessions": len(self.active_sessions),
                    "pending_matchups": len(self.pending_matchups),
                    "config": {
                        "max_concurrent_sessions": self.config.max_concurrent_sessions,
                        "auto_matchmaking": self.config.auto_matchmaking,
                        "skill_balancing": self.config.skill_balancing
                    }
                },
                "statistics": self.stats,
                "recent_sessions": [
                    {
                        "session_id": result.session_id,
                        "duration": result.duration.total_seconds(),
                        "episodes": result.episodes_completed,
                        "winner": result.winner.value if result.winner else None,
                        "learning_outcomes": len(result.learning_outcomes)
                    }
                    for result in self.session_history[-5:]  # Last 5 sessions
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get coordinator status: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown the self-play coordinator"""
        try:
            self.logger.info("Shutting down self-play coordinator")
            self.running = False
            
            # Cancel background tasks
            if self.matchmaking_task:
                self.matchmaking_task.cancel()
                try:
                    await self.matchmaking_task
                except asyncio.CancelledError:
                    pass
            
            if self.performance_evaluation_task:
                self.performance_evaluation_task.cancel()
                try:
                    await self.performance_evaluation_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel active sessions
            for session_id in list(self.active_sessions.keys()):
                await self._cancel_session(session_id, "Coordinator shutdown")
            
            self.initialized = False
            self.logger.info("Self-play coordinator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during coordinator shutdown: {e}")

# Factory function
def create_self_play_coordinator(config: SelfPlayConfig = None, learning_system: LearningSystem = None) -> SelfPlayCoordinator:
    """Create a self-play coordinator instance"""
    return SelfPlayCoordinator(config or SelfPlayConfig(), learning_system)