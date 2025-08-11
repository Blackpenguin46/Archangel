#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Dynamic Scoring and Evaluation Engine
Real-time scoring system for Red vs Blue team performance evaluation
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import json
import statistics
from collections import defaultdict

from .base_agent import Team, Role, ActionResult, Experience

logger = logging.getLogger(__name__)

class ScoreCategory(Enum):
    """Categories for scoring metrics"""
    ATTACK_SUCCESS = "attack_success"
    DEFENSE_SUCCESS = "defense_success"
    DETECTION_SPEED = "detection_speed"
    CONTAINMENT_TIME = "containment_time"
    STEALTH_MAINTENANCE = "stealth_maintenance"
    COLLABORATION = "collaboration"
    LEARNING_ADAPTATION = "learning_adaptation"

class MetricType(Enum):
    """Types of metrics for evaluation"""
    COUNTER = "counter"
    TIMER = "timer"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    SCORE = "score"

@dataclass
class ScoringWeight:
    """Weight configuration for scoring metrics"""
    category: ScoreCategory
    weight: float
    max_score: float
    min_score: float = 0.0
    description: str = ""

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    metric_id: str
    agent_id: str
    team: Team
    category: ScoreCategory
    metric_type: MetricType
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

@dataclass
class TeamScore:
    """Team scoring information"""
    team: Team
    total_score: float
    category_scores: Dict[ScoreCategory, float]
    metrics_count: int
    last_updated: datetime
    performance_trend: List[float] = field(default_factory=list)

@dataclass
class ScoringConfig:
    """Configuration for the scoring engine"""
    weights: Dict[ScoreCategory, ScoringWeight]
    evaluation_window: timedelta = timedelta(minutes=5)
    trend_history_size: int = 100
    real_time_updates: bool = True
    fairness_adjustments: bool = True

class DynamicScoringEngine:
    """
    Dynamic scoring and evaluation engine for Red vs Blue team competition.
    
    Features:
    - Real-time score calculation with weighted metrics
    - Objective-based evaluation with customizable criteria
    - Performance tracking for detection speed, containment time, success rates
    - Comparative analysis and team effectiveness reporting
    - Fairness adjustments to ensure balanced competition
    """
    
    def __init__(self, config: ScoringConfig):
        self.config = config
        self.engine_id = str(uuid.uuid4())
        
        # Scoring state
        self.team_scores: Dict[Team, TeamScore] = {
            Team.RED: TeamScore(
                team=Team.RED,
                total_score=0.0,
                category_scores={cat: 0.0 for cat in ScoreCategory},
                metrics_count=0,
                last_updated=datetime.now()
            ),
            Team.BLUE: TeamScore(
                team=Team.BLUE,
                total_score=0.0,
                category_scores={cat: 0.0 for cat in ScoreCategory},
                metrics_count=0,
                last_updated=datetime.now()
            )
        }
        
        # Metrics storage
        self.metrics_history: List[PerformanceMetric] = []
        self.agent_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        
        # Performance tracking
        self.detection_times: List[float] = []
        self.containment_times: List[float] = []
        self.success_rates: Dict[Team, List[bool]] = {
            Team.RED: [],
            Team.BLUE: []
        }
        
        # Evaluation state
        self.evaluation_tasks: List[asyncio.Task] = []
        self.running = False
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the scoring engine"""
        try:
            self.logger.info("Initializing dynamic scoring engine")
            
            # Validate configuration
            await self._validate_config()
            
            # Start real-time evaluation if enabled
            if self.config.real_time_updates:
                self.evaluation_tasks.append(
                    asyncio.create_task(self._real_time_evaluation_loop())
                )
            
            # Start performance tracking
            self.evaluation_tasks.append(
                asyncio.create_task(self._performance_tracking_loop())
            )
            
            self.running = True
            self.logger.info("Dynamic scoring engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize scoring engine: {e}")
            raise
    
    async def _validate_config(self) -> None:
        """Validate scoring configuration"""
        if not self.config.weights:
            raise ValueError("Scoring weights configuration is required")
        
        # Validate weight values
        for category, weight in self.config.weights.items():
            if weight.weight < 0 or weight.weight > 1:
                raise ValueError(f"Weight for {category} must be between 0 and 1")
            
            if weight.max_score <= weight.min_score:
                raise ValueError(f"Max score must be greater than min score for {category}")
    
    async def record_metric(self, agent_id: str, team: Team, category: ScoreCategory,
                          metric_type: MetricType, value: float, 
                          context: Optional[Dict[str, Any]] = None,
                          confidence: float = 1.0) -> str:
        """
        Record a performance metric for scoring
        
        Args:
            agent_id: ID of the agent generating the metric
            team: Team the agent belongs to
            category: Category of the metric
            metric_type: Type of metric
            value: Metric value
            context: Additional context information
            confidence: Confidence in the metric (0.0 to 1.0)
            
        Returns:
            str: Metric ID
        """
        try:
            metric = PerformanceMetric(
                metric_id=str(uuid.uuid4()),
                agent_id=agent_id,
                team=team,
                category=category,
                metric_type=metric_type,
                value=value,
                timestamp=datetime.now(),
                context=context or {},
                confidence=confidence
            )
            
            # Store metric
            self.metrics_history.append(metric)
            self.agent_metrics[agent_id].append(metric)
            
            # Update scores if real-time updates are enabled
            if self.config.real_time_updates:
                await self._update_team_scores()
            
            self.logger.debug(f"Recorded metric {category.value} for agent {agent_id}: {value}")
            return metric.metric_id
            
        except Exception as e:
            self.logger.error(f"Failed to record metric: {e}")
            raise
    
    async def record_attack_success(self, agent_id: str, target: str, 
                                  attack_type: str, success: bool,
                                  duration: float, stealth_score: float = 0.5) -> str:
        """Record an attack attempt and its outcome"""
        context = {
            "target": target,
            "attack_type": attack_type,
            "duration": duration,
            "stealth_score": stealth_score
        }
        
        # Record success/failure
        success_value = 1.0 if success else 0.0
        await self.record_metric(
            agent_id, Team.RED, ScoreCategory.ATTACK_SUCCESS,
            MetricType.SCORE, success_value, context
        )
        
        # Record stealth maintenance if successful
        if success:
            await self.record_metric(
                agent_id, Team.RED, ScoreCategory.STEALTH_MAINTENANCE,
                MetricType.SCORE, stealth_score, context
            )
        
        # Update success rates
        self.success_rates[Team.RED].append(success)
        
        return f"attack_{agent_id}_{datetime.now().timestamp()}"
    
    async def record_detection_event(self, agent_id: str, detected_agent: str,
                                   detection_time: float, accuracy: float = 1.0) -> str:
        """Record a detection event by Blue team"""
        context = {
            "detected_agent": detected_agent,
            "detection_time": detection_time,
            "accuracy": accuracy
        }
        
        # Record detection success
        await self.record_metric(
            agent_id, Team.BLUE, ScoreCategory.DEFENSE_SUCCESS,
            MetricType.SCORE, accuracy, context
        )
        
        # Record detection speed (inverse of time - faster is better)
        speed_score = max(0.0, 1.0 - (detection_time / 300.0))  # 5 minutes max
        await self.record_metric(
            agent_id, Team.BLUE, ScoreCategory.DETECTION_SPEED,
            MetricType.TIMER, speed_score, context
        )
        
        # Store detection time for analysis
        self.detection_times.append(detection_time)
        self.success_rates[Team.BLUE].append(True)
        
        return f"detection_{agent_id}_{datetime.now().timestamp()}"
    
    async def record_containment_action(self, agent_id: str, threat_id: str,
                                      containment_time: float, effectiveness: float) -> str:
        """Record a containment action by Blue team"""
        context = {
            "threat_id": threat_id,
            "containment_time": containment_time,
            "effectiveness": effectiveness
        }
        
        # Record containment success
        await self.record_metric(
            agent_id, Team.BLUE, ScoreCategory.DEFENSE_SUCCESS,
            MetricType.SCORE, effectiveness, context
        )
        
        # Record containment speed
        speed_score = max(0.0, 1.0 - (containment_time / 600.0))  # 10 minutes max
        await self.record_metric(
            agent_id, Team.BLUE, ScoreCategory.CONTAINMENT_TIME,
            MetricType.TIMER, speed_score, context
        )
        
        # Store containment time for analysis
        self.containment_times.append(containment_time)
        
        return f"containment_{agent_id}_{datetime.now().timestamp()}"
    
    async def record_collaboration_event(self, agent_id: str, team: Team,
                                       collaboration_type: str, effectiveness: float) -> str:
        """Record a team collaboration event"""
        context = {
            "collaboration_type": collaboration_type,
            "effectiveness": effectiveness
        }
        
        await self.record_metric(
            agent_id, team, ScoreCategory.COLLABORATION,
            MetricType.SCORE, effectiveness, context
        )
        
        return f"collaboration_{agent_id}_{datetime.now().timestamp()}"
    
    async def record_learning_adaptation(self, agent_id: str, team: Team,
                                       adaptation_score: float, context: Dict[str, Any]) -> str:
        """Record learning and adaptation performance"""
        await self.record_metric(
            agent_id, team, ScoreCategory.LEARNING_ADAPTATION,
            MetricType.SCORE, adaptation_score, context
        )
        
        return f"learning_{agent_id}_{datetime.now().timestamp()}"
    
    async def _update_team_scores(self) -> None:
        """Update team scores based on recent metrics"""
        try:
            current_time = datetime.now()
            evaluation_window_start = current_time - self.config.evaluation_window
            
            # Get recent metrics within evaluation window
            recent_metrics = [
                metric for metric in self.metrics_history
                if metric.timestamp >= evaluation_window_start
            ]
            
            # Calculate scores for each team
            for team in [Team.RED, Team.BLUE]:
                team_metrics = [m for m in recent_metrics if m.team == team]
                await self._calculate_team_score(team, team_metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to update team scores: {e}")
    
    async def _calculate_team_score(self, team: Team, metrics: List[PerformanceMetric]) -> None:
        """Calculate score for a specific team"""
        try:
            team_score = self.team_scores[team]
            category_scores = {}
            
            # Calculate score for each category
            for category in ScoreCategory:
                category_metrics = [m for m in metrics if m.category == category]
                
                if category_metrics:
                    # Calculate weighted average with confidence
                    weighted_sum = sum(m.value * m.confidence for m in category_metrics)
                    confidence_sum = sum(m.confidence for m in category_metrics)
                    
                    if confidence_sum > 0:
                        avg_score = weighted_sum / confidence_sum
                    else:
                        avg_score = 0.0
                    
                    # Apply category weight and bounds
                    if category in self.config.weights:
                        weight_config = self.config.weights[category]
                        bounded_score = max(weight_config.min_score, 
                                          min(weight_config.max_score, avg_score))
                        category_scores[category] = bounded_score * weight_config.weight
                    else:
                        category_scores[category] = avg_score
                else:
                    category_scores[category] = 0.0
            
            # Apply fairness adjustments if enabled
            if self.config.fairness_adjustments:
                category_scores = await self._apply_fairness_adjustments(team, category_scores)
            
            # Calculate total score
            total_score = sum(category_scores.values())
            
            # Update team score
            team_score.category_scores = category_scores
            team_score.total_score = total_score
            team_score.metrics_count = len(metrics)
            team_score.last_updated = datetime.now()
            
            # Update performance trend
            team_score.performance_trend.append(total_score)
            if len(team_score.performance_trend) > self.config.trend_history_size:
                team_score.performance_trend.pop(0)
            
            self.logger.debug(f"Updated {team.value} team score: {total_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate team score for {team.value}: {e}")
    
    async def _apply_fairness_adjustments(self, team: Team, 
                                        category_scores: Dict[ScoreCategory, float]) -> Dict[ScoreCategory, float]:
        """Apply fairness adjustments to ensure balanced competition"""
        try:
            adjusted_scores = category_scores.copy()
            
            # Get opposing team scores for comparison
            opposing_team = Team.BLUE if team == Team.RED else Team.RED
            opposing_scores = self.team_scores[opposing_team].category_scores
            
            # Apply adjustments based on score disparities
            for category, score in category_scores.items():
                opposing_score = opposing_scores.get(category, 0.0)
                
                # If one team is significantly ahead, apply diminishing returns
                if score > 0 and opposing_score > 0:
                    ratio = score / opposing_score
                    if ratio > 2.0:  # Team is more than 2x ahead
                        adjustment_factor = 1.0 - ((ratio - 2.0) * 0.1)  # Reduce by 10% per ratio point
                        adjusted_scores[category] = score * max(0.5, adjustment_factor)
            
            return adjusted_scores
            
        except Exception as e:
            self.logger.error(f"Failed to apply fairness adjustments: {e}")
            return category_scores
    
    async def get_current_scores(self) -> Dict[Team, TeamScore]:
        """Get current team scores"""
        # Update scores before returning
        await self._update_team_scores()
        return self.team_scores.copy()
    
    async def get_performance_analysis(self) -> Dict[str, Any]:
        """Get comprehensive performance analysis"""
        try:
            current_scores = await self.get_current_scores()
            
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "team_scores": {
                    team.value: {
                        "total_score": score.total_score,
                        "category_scores": {cat.value: val for cat, val in score.category_scores.items()},
                        "metrics_count": score.metrics_count,
                        "performance_trend": score.performance_trend[-10:],  # Last 10 data points
                        "trend_direction": self._calculate_trend_direction(score.performance_trend)
                    }
                    for team, score in current_scores.items()
                },
                "comparative_analysis": await self._generate_comparative_analysis(),
                "performance_metrics": {
                    "average_detection_time": statistics.mean(self.detection_times) if self.detection_times else 0.0,
                    "average_containment_time": statistics.mean(self.containment_times) if self.containment_times else 0.0,
                    "red_team_success_rate": self._calculate_success_rate(Team.RED),
                    "blue_team_success_rate": self._calculate_success_rate(Team.BLUE),
                    "total_metrics_recorded": len(self.metrics_history)
                },
                "recommendations": await self._generate_recommendations()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance analysis: {e}")
            raise
    
    def _calculate_trend_direction(self, trend_data: List[float]) -> str:
        """Calculate trend direction from performance data"""
        if len(trend_data) < 2:
            return "insufficient_data"
        
        recent_avg = statistics.mean(trend_data[-5:]) if len(trend_data) >= 5 else trend_data[-1]
        older_avg = statistics.mean(trend_data[-10:-5]) if len(trend_data) >= 10 else trend_data[0]
        
        if recent_avg > older_avg * 1.05:  # 5% improvement threshold
            return "improving"
        elif recent_avg < older_avg * 0.95:  # 5% decline threshold
            return "declining"
        else:
            return "stable"
    
    def _calculate_success_rate(self, team: Team) -> float:
        """Calculate success rate for a team"""
        successes = self.success_rates[team]
        if not successes:
            return 0.0
        
        return sum(successes) / len(successes)
    
    async def _generate_comparative_analysis(self) -> Dict[str, Any]:
        """Generate comparative analysis between teams"""
        red_score = self.team_scores[Team.RED]
        blue_score = self.team_scores[Team.BLUE]
        
        analysis = {
            "score_difference": red_score.total_score - blue_score.total_score,
            "leading_team": Team.RED.value if red_score.total_score > blue_score.total_score else Team.BLUE.value,
            "category_comparison": {},
            "performance_balance": abs(red_score.total_score - blue_score.total_score) / max(red_score.total_score, blue_score.total_score, 1.0)
        }
        
        # Compare categories
        for category in ScoreCategory:
            red_cat_score = red_score.category_scores.get(category, 0.0)
            blue_cat_score = blue_score.category_scores.get(category, 0.0)
            
            analysis["category_comparison"][category.value] = {
                "red_score": red_cat_score,
                "blue_score": blue_cat_score,
                "difference": red_cat_score - blue_cat_score,
                "leader": "red" if red_cat_score > blue_cat_score else "blue"
            }
        
        return analysis
    
    async def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations for improving performance"""
        recommendations = []
        
        # Analyze team performance and suggest improvements
        for team in [Team.RED, Team.BLUE]:
            team_score = self.team_scores[team]
            
            # Find weakest categories
            sorted_categories = sorted(
                team_score.category_scores.items(),
                key=lambda x: x[1]
            )
            
            if sorted_categories:
                weakest_category = sorted_categories[0][0]
                recommendations.append({
                    "team": team.value,
                    "type": "improvement",
                    "category": weakest_category.value,
                    "recommendation": f"Focus on improving {weakest_category.value} performance"
                })
        
        # Suggest balance improvements
        comparative = await self._generate_comparative_analysis()
        if comparative["performance_balance"] > 0.5:
            leading_team = comparative["leading_team"]
            recommendations.append({
                "team": "both",
                "type": "balance",
                "recommendation": f"Consider adjusting difficulty to balance competition - {leading_team} team is significantly ahead"
            })
        
        return recommendations
    
    async def _real_time_evaluation_loop(self) -> None:
        """Real-time evaluation loop for continuous scoring updates"""
        while self.running:
            try:
                await self._update_team_scores()
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in real-time evaluation loop: {e}")
                await asyncio.sleep(10.0)  # Back off on error
    
    async def _performance_tracking_loop(self) -> None:
        """Performance tracking loop for metrics analysis"""
        while self.running:
            try:
                # Clean up old metrics to prevent memory bloat
                cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours of data
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp >= cutoff_time
                ]
                
                # Clean up agent metrics
                for agent_id in self.agent_metrics:
                    self.agent_metrics[agent_id] = [
                        m for m in self.agent_metrics[agent_id]
                        if m.timestamp >= cutoff_time
                    ]
                
                await asyncio.sleep(300.0)  # Clean up every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(60.0)
    
    async def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics data in specified format"""
        try:
            if format_type.lower() == "json":
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "engine_id": self.engine_id,
                    "team_scores": {
                        team.value: {
                            "total_score": score.total_score,
                            "category_scores": {cat.value: val for cat, val in score.category_scores.items()},
                            "metrics_count": score.metrics_count,
                            "last_updated": score.last_updated.isoformat(),
                            "performance_trend": score.performance_trend
                        }
                        for team, score in self.team_scores.items()
                    },
                    "metrics_history": [
                        {
                            "metric_id": m.metric_id,
                            "agent_id": m.agent_id,
                            "team": m.team.value,
                            "category": m.category.value,
                            "metric_type": m.metric_type.value,
                            "value": m.value,
                            "timestamp": m.timestamp.isoformat(),
                            "context": m.context,
                            "confidence": m.confidence
                        }
                        for m in self.metrics_history
                    ]
                }
                
                return json.dumps(export_data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the scoring engine"""
        try:
            self.logger.info("Shutting down dynamic scoring engine")
            self.running = False
            
            # Cancel evaluation tasks
            for task in self.evaluation_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Dynamic scoring engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during scoring engine shutdown: {e}")

# Default scoring configuration
DEFAULT_SCORING_CONFIG = ScoringConfig(
    weights={
        ScoreCategory.ATTACK_SUCCESS: ScoringWeight(
            category=ScoreCategory.ATTACK_SUCCESS,
            weight=0.25,
            max_score=100.0,
            description="Red team attack success rate and effectiveness"
        ),
        ScoreCategory.DEFENSE_SUCCESS: ScoringWeight(
            category=ScoreCategory.DEFENSE_SUCCESS,
            weight=0.25,
            max_score=100.0,
            description="Blue team defense success rate and effectiveness"
        ),
        ScoreCategory.DETECTION_SPEED: ScoringWeight(
            category=ScoreCategory.DETECTION_SPEED,
            weight=0.20,
            max_score=100.0,
            description="Speed of threat detection by Blue team"
        ),
        ScoreCategory.CONTAINMENT_TIME: ScoringWeight(
            category=ScoreCategory.CONTAINMENT_TIME,
            weight=0.15,
            max_score=100.0,
            description="Time to contain threats after detection"
        ),
        ScoreCategory.STEALTH_MAINTENANCE: ScoringWeight(
            category=ScoreCategory.STEALTH_MAINTENANCE,
            weight=0.10,
            max_score=100.0,
            description="Red team ability to maintain stealth and avoid detection"
        ),
        ScoreCategory.COLLABORATION: ScoringWeight(
            category=ScoreCategory.COLLABORATION,
            weight=0.05,
            max_score=100.0,
            description="Team collaboration and coordination effectiveness"
        )
    },
    evaluation_window=timedelta(minutes=5),
    trend_history_size=100,
    real_time_updates=True,
    fairness_adjustments=True
)