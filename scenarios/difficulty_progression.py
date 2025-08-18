#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Scenario Difficulty Progression System
Dynamic difficulty scaling and complexity progression for adaptive learning
"""

import asyncio
import logging
import uuid
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict
import json

from .scenario_templates import ScenarioTemplate, ComplexityLevel, ScenarioCategory
from .dynamic_generation import AgentLearningProfile, LearningOutcome

logger = logging.getLogger(__name__)

class ProgressionStrategy(Enum):
    """Difficulty progression strategies"""
    LINEAR = "linear"                    # Steady linear progression
    EXPONENTIAL = "exponential"          # Exponential difficulty increase
    ADAPTIVE = "adaptive"                # Based on performance
    MASTERY_BASED = "mastery_based"      # Progress after mastery
    SPIRAL = "spiral"                    # Revisit concepts at higher levels
    BRANCHING = "branching"              # Multiple progression paths

class DifficultyMetric(Enum):
    """Metrics for measuring scenario difficulty"""
    COMPLEXITY_SCORE = "complexity_score"
    TIME_PRESSURE = "time_pressure"
    RESOURCE_CONSTRAINTS = "resource_constraints"
    INFORMATION_AVAILABILITY = "information_availability"
    MULTI_STEP_COMPLEXITY = "multi_step_complexity"
    COLLABORATION_REQUIREMENT = "collaboration_requirement"
    NOVELTY_FACTOR = "novelty_factor"
    FAILURE_TOLERANCE = "failure_tolerance"

@dataclass
class DifficultyProfile:
    """Comprehensive difficulty profile for scenarios"""
    # Core difficulty metrics (0.0 to 1.0)
    complexity_score: float = 0.5
    time_pressure: float = 0.5
    resource_constraints: float = 0.5
    information_availability: float = 0.5  # Lower = less info available
    multi_step_complexity: float = 0.5
    collaboration_requirement: float = 0.5
    novelty_factor: float = 0.5
    failure_tolerance: float = 0.5  # Lower = less tolerance for mistakes
    
    # Derived metrics
    overall_difficulty: float = field(init=False)
    cognitive_load: float = field(init=False)
    stress_level: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        # Overall difficulty is weighted average
        weights = {
            'complexity_score': 0.25,
            'time_pressure': 0.15,
            'resource_constraints': 0.15,
            'information_availability': 0.15,
            'multi_step_complexity': 0.15,
            'collaboration_requirement': 0.10,
            'novelty_factor': 0.05
        }
        
        self.overall_difficulty = (
            self.complexity_score * weights['complexity_score'] +
            self.time_pressure * weights['time_pressure'] +
            self.resource_constraints * weights['resource_constraints'] +
            (1.0 - self.information_availability) * weights['information_availability'] +
            self.multi_step_complexity * weights['multi_step_complexity'] +
            self.collaboration_requirement * weights['collaboration_requirement'] +
            self.novelty_factor * weights['novelty_factor']
        )
        
        # Cognitive load considers information processing requirements
        self.cognitive_load = (
            self.complexity_score * 0.4 +
            self.multi_step_complexity * 0.3 +
            (1.0 - self.information_availability) * 0.2 +
            self.novelty_factor * 0.1
        )
        
        # Stress level considers time and failure pressure
        self.stress_level = (
            self.time_pressure * 0.4 +
            (1.0 - self.failure_tolerance) * 0.3 +
            self.resource_constraints * 0.2 +
            self.collaboration_requirement * 0.1
        )

@dataclass
class ProgressionPath:
    """Defines a learning progression path"""
    path_id: str
    name: str
    description: str
    
    # Path configuration
    strategy: ProgressionStrategy
    target_category: ScenarioCategory
    
    # Progression stages
    stages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Prerequisites and requirements
    prerequisites: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    
    # Progression parameters
    mastery_threshold: float = 0.8  # Success rate required to advance
    min_attempts: int = 3  # Minimum attempts before advancement
    max_attempts: int = 10  # Maximum attempts before forced advancement
    
    # Adaptation parameters
    difficulty_increment: float = 0.1  # How much to increase difficulty
    adaptation_sensitivity: float = 0.2  # How quickly to adapt
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    success_rate: float = 0.0
    completion_rate: float = 0.0

@dataclass
class ProgressionState:
    """Tracks an agent's progression state"""
    agent_id: str
    path_id: str
    
    # Current state
    current_stage: int = 0
    current_difficulty: DifficultyProfile = field(default_factory=DifficultyProfile)
    
    # Performance tracking
    stage_attempts: int = 0
    stage_successes: int = 0
    stage_success_rate: float = 0.0
    
    # Historical data
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    difficulty_history: List[DifficultyProfile] = field(default_factory=list)
    
    # Adaptation state
    last_adaptation: Optional[datetime] = None
    adaptation_direction: int = 0  # -1: easier, 0: same, 1: harder
    
    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

class DifficultyProgressionEngine:
    """
    Advanced difficulty progression and complexity scaling system.
    
    Features:
    - Multiple progression strategies
    - Adaptive difficulty adjustment
    - Performance-based scaling
    - Multi-dimensional difficulty metrics
    - Personalized progression paths
    """
    
    def __init__(self):
        # Progression paths and states
        self.progression_paths: Dict[str, ProgressionPath] = {}
        self.agent_progressions: Dict[str, Dict[str, ProgressionState]] = defaultdict(dict)
        
        # Difficulty calculation models
        self.difficulty_models: Dict[ScenarioCategory, Dict[str, Any]] = {}
        
        # Performance analytics
        self.progression_analytics = {
            "total_progressions": 0,
            "completion_rates": {},
            "average_progression_time": {},
            "difficulty_effectiveness": {}
        }
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the difficulty progression engine"""
        try:
            self.logger.info("Initializing difficulty progression engine")
            
            # Initialize difficulty models
            await self._initialize_difficulty_models()
            
            # Create default progression paths
            await self._create_default_progression_paths()
            
            # Load existing progression states
            await self._load_progression_states()
            
            self.initialized = True
            self.logger.info("Difficulty progression engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize difficulty progression engine: {e}")
            raise
    
    async def calculate_scenario_difficulty(self, 
                                          scenario: ScenarioTemplate,
                                          agent_profile: AgentLearningProfile) -> DifficultyProfile:
        """Calculate comprehensive difficulty profile for a scenario"""
        try:
            # Base difficulty from scenario complexity
            base_complexity = scenario.complexity.value / 5.0
            
            # Adjust based on agent skill level
            agent_skill = sum(agent_profile.skill_levels.values()) / max(1, len(agent_profile.skill_levels))
            skill_adjustment = base_complexity - agent_skill
            
            # Calculate individual difficulty metrics
            complexity_score = max(0.0, min(1.0, base_complexity + skill_adjustment * 0.3))
            
            # Time pressure based on estimated duration and agent experience
            time_pressure = self._calculate_time_pressure(scenario, agent_profile)
            
            # Resource constraints based on scenario requirements
            resource_constraints = self._calculate_resource_constraints(scenario)
            
            # Information availability based on scenario documentation
            info_availability = self._calculate_information_availability(scenario)
            
            # Multi-step complexity based on objectives
            multi_step = self._calculate_multi_step_complexity(scenario)
            
            # Collaboration requirement
            collaboration = self._calculate_collaboration_requirement(scenario)
            
            # Novelty factor based on agent experience
            novelty = self._calculate_novelty_factor(scenario, agent_profile)
            
            # Failure tolerance based on scenario type
            failure_tolerance = self._calculate_failure_tolerance(scenario)
            
            return DifficultyProfile(
                complexity_score=complexity_score,
                time_pressure=time_pressure,
                resource_constraints=resource_constraints,
                information_availability=info_availability,
                multi_step_complexity=multi_step,
                collaboration_requirement=collaboration,
                novelty_factor=novelty,
                failure_tolerance=failure_tolerance
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate scenario difficulty: {e}")
            return DifficultyProfile()  # Return default difficulty
    
    async def create_progression_path(self, path: ProgressionPath) -> str:
        """Create a new progression path"""
        try:
            # Validate progression path
            validation_result = await self._validate_progression_path(path)
            if not validation_result["valid"]:
                raise ValueError(f"Progression path validation failed: {validation_result['errors']}")
            
            # Store progression path
            self.progression_paths[path.path_id] = path
            
            self.logger.info(f"Created progression path: {path.name} ({path.path_id})")
            return path.path_id
            
        except Exception as e:
            self.logger.error(f"Failed to create progression path: {e}")
            raise
    
    async def start_agent_progression(self, 
                                    agent_id: str,
                                    path_id: str,
                                    initial_difficulty: Optional[DifficultyProfile] = None) -> bool:
        """Start an agent on a progression path"""
        try:
            if path_id not in self.progression_paths:
                raise ValueError(f"Progression path {path_id} not found")
            
            path = self.progression_paths[path_id]
            
            # Create initial progression state
            progression_state = ProgressionState(
                agent_id=agent_id,
                path_id=path_id,
                current_difficulty=initial_difficulty or DifficultyProfile()
            )
            
            # Store progression state
            self.agent_progressions[agent_id][path_id] = progression_state
            
            self.logger.info(f"Started agent {agent_id} on progression path {path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start agent progression: {e}")
            return False
    
    async def update_progression(self, 
                               agent_id: str,
                               path_id: str,
                               scenario_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update agent progression based on scenario results"""
        try:
            if agent_id not in self.agent_progressions or path_id not in self.agent_progressions[agent_id]:
                raise ValueError(f"No progression found for agent {agent_id} on path {path_id}")
            
            progression = self.agent_progressions[agent_id][path_id]
            path = self.progression_paths[path_id]
            
            # Update attempt tracking
            progression.stage_attempts += 1
            if scenario_result.get("success", False):
                progression.stage_successes += 1
            
            # Calculate current success rate
            progression.stage_success_rate = progression.stage_successes / progression.stage_attempts
            
            # Add to performance history
            progression.performance_history.append({
                "timestamp": datetime.now().isoformat(),
                "success": scenario_result.get("success", False),
                "completion_time": scenario_result.get("completion_time", 0),
                "score": scenario_result.get("score", 0),
                "difficulty": progression.current_difficulty
            })
            
            # Determine if adaptation is needed
            adaptation_result = await self._evaluate_difficulty_adaptation(progression, path)
            
            # Apply adaptation if needed
            if adaptation_result["adapt"]:
                await self._apply_difficulty_adaptation(progression, adaptation_result)
            
            # Check for stage advancement
            advancement_result = await self._evaluate_stage_advancement(progression, path)
            
            # Apply stage advancement if needed
            if advancement_result["advance"]:
                await self._advance_progression_stage(progression, path)
            
            # Update metadata
            progression.last_updated = datetime.now()
            
            return {
                "progression_updated": True,
                "current_stage": progression.current_stage,
                "success_rate": progression.stage_success_rate,
                "difficulty_adapted": adaptation_result["adapt"],
                "stage_advanced": advancement_result["advance"],
                "recommendations": advancement_result.get("recommendations", [])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update progression: {e}")
            return {"progression_updated": False, "error": str(e)}
    
    async def get_next_difficulty(self, 
                                agent_id: str,
                                path_id: str) -> Optional[DifficultyProfile]:
        """Get the next difficulty level for an agent"""
        try:
            if agent_id not in self.agent_progressions or path_id not in self.agent_progressions[agent_id]:
                return None
            
            progression = self.agent_progressions[agent_id][path_id]
            return progression.current_difficulty
            
        except Exception as e:
            self.logger.error(f"Failed to get next difficulty: {e}")
            return None
    
    async def get_progression_recommendations(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get progression recommendations for an agent"""
        try:
            recommendations = []
            
            if agent_id not in self.agent_progressions:
                # Recommend starting progression paths
                for path in self.progression_paths.values():
                    recommendations.append({
                        "type": "start_progression",
                        "path_id": path.path_id,
                        "path_name": path.name,
                        "description": path.description,
                        "category": path.target_category.value,
                        "priority": "medium"
                    })
                return recommendations[:3]  # Top 3 recommendations
            
            # Analyze current progressions
            for path_id, progression in self.agent_progressions[agent_id].items():
                path = self.progression_paths[path_id]
                
                # Check if stuck on current stage
                if progression.stage_attempts > path.max_attempts // 2:
                    recommendations.append({
                        "type": "difficulty_adjustment",
                        "path_id": path_id,
                        "path_name": path.name,
                        "recommendation": "Consider reducing difficulty or providing additional support",
                        "priority": "high"
                    })
                
                # Check if ready for advancement
                if progression.stage_success_rate >= path.mastery_threshold:
                    recommendations.append({
                        "type": "stage_advancement",
                        "path_id": path_id,
                        "path_name": path.name,
                        "recommendation": "Ready to advance to next stage",
                        "priority": "high"
                    })
                
                # Check for new progression paths
                if progression.current_stage >= len(path.stages) - 1:
                    # Look for related progression paths
                    for other_path in self.progression_paths.values():
                        if (other_path.path_id != path_id and 
                            other_path.target_category == path.target_category and
                            other_path.path_id not in self.agent_progressions[agent_id]):
                            recommendations.append({
                                "type": "new_progression",
                                "path_id": other_path.path_id,
                                "path_name": other_path.name,
                                "recommendation": f"Continue learning in {other_path.target_category.value}",
                                "priority": "medium"
                            })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get progression recommendations: {e}")
            return []
    
    def _calculate_time_pressure(self, scenario: ScenarioTemplate, agent_profile: AgentLearningProfile) -> float:
        """Calculate time pressure metric"""
        # Base time pressure from scenario duration
        duration_hours = scenario.estimated_duration.total_seconds() / 3600
        
        # Shorter scenarios create more time pressure
        if duration_hours < 1:
            base_pressure = 0.8
        elif duration_hours < 2:
            base_pressure = 0.6
        elif duration_hours < 4:
            base_pressure = 0.4
        else:
            base_pressure = 0.2
        
        # Adjust based on agent experience
        agent_experience = len(agent_profile.completed_scenarios)
        experience_factor = max(0.5, 1.0 - (agent_experience / 100))
        
        return min(1.0, base_pressure * experience_factor)
    
    def _calculate_resource_constraints(self, scenario: ScenarioTemplate) -> float:
        """Calculate resource constraints metric"""
        # Base on number of required assets and complexity
        asset_count = len(scenario.required_assets)
        complexity_factor = scenario.complexity.value / 5.0
        
        # More assets and higher complexity = more resource constraints
        constraint_score = min(1.0, (asset_count / 10) * 0.5 + complexity_factor * 0.5)
        
        return constraint_score
    
    def _calculate_information_availability(self, scenario: ScenarioTemplate) -> float:
        """Calculate information availability metric"""
        # Base on documentation length and objective clarity
        doc_length = len(scenario.documentation) if scenario.documentation else 0
        objective_count = len(scenario.objectives)
        
        # More documentation and objectives = higher information availability
        info_score = min(1.0, (doc_length / 1000) * 0.5 + (objective_count / 10) * 0.5)
        
        return max(0.1, info_score)  # Minimum 10% information availability
    
    def _calculate_multi_step_complexity(self, scenario: ScenarioTemplate) -> float:
        """Calculate multi-step complexity metric"""
        # Base on number of objectives and their dependencies
        objective_count = len(scenario.objectives)
        
        # Count objectives with dependencies
        dependent_objectives = sum(1 for obj in scenario.objectives if obj.dependencies)
        
        complexity_score = min(1.0, (objective_count / 10) * 0.6 + (dependent_objectives / objective_count) * 0.4)
        
        return complexity_score
    
    def _calculate_collaboration_requirement(self, scenario: ScenarioTemplate) -> float:
        """Calculate collaboration requirement metric"""
        # Base on minimum participants and team requirements
        if scenario.min_participants > 1:
            return min(1.0, scenario.min_participants / 5)
        
        if scenario.recommended_teams:
            return 0.5
        
        return 0.1  # Minimal collaboration for single-agent scenarios
    
    def _calculate_novelty_factor(self, scenario: ScenarioTemplate, agent_profile: AgentLearningProfile) -> float:
        """Calculate novelty factor based on agent experience"""
        # Check if agent has completed similar scenarios
        similar_scenarios = sum(1 for completed in agent_profile.completed_scenarios 
                              if scenario.category.value in completed)
        
        # More similar scenarios = less novelty
        novelty_score = max(0.1, 1.0 - (similar_scenarios / 20))
        
        return novelty_score
    
    def _calculate_failure_tolerance(self, scenario: ScenarioTemplate) -> float:
        """Calculate failure tolerance metric"""
        # Base on scenario type and complexity
        if scenario.scenario_type.value == "assessment":
            return 0.3  # Low tolerance for assessment scenarios
        elif scenario.scenario_type.value == "competition":
            return 0.2  # Very low tolerance for competitions
        else:
            return 0.7  # Higher tolerance for training scenarios
    
    async def _evaluate_difficulty_adaptation(self, 
                                            progression: ProgressionState,
                                            path: ProgressionPath) -> Dict[str, Any]:
        """Evaluate if difficulty adaptation is needed"""
        try:
            # Don't adapt too frequently
            if (progression.last_adaptation and 
                datetime.now() - progression.last_adaptation < timedelta(hours=1)):
                return {"adapt": False, "reason": "too_recent"}
            
            # Need minimum attempts for reliable assessment
            if progression.stage_attempts < 3:
                return {"adapt": False, "reason": "insufficient_data"}
            
            success_rate = progression.stage_success_rate
            
            # Adapt if performance is consistently too high or too low
            if success_rate > 0.9 and progression.stage_attempts >= 3:
                return {
                    "adapt": True,
                    "direction": "increase",
                    "reason": "too_easy",
                    "adjustment": path.difficulty_increment
                }
            elif success_rate < 0.3 and progression.stage_attempts >= 5:
                return {
                    "adapt": True,
                    "direction": "decrease",
                    "reason": "too_difficult",
                    "adjustment": -path.difficulty_increment
                }
            
            return {"adapt": False, "reason": "performance_acceptable"}
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate difficulty adaptation: {e}")
            return {"adapt": False, "reason": "error"}
    
    async def _apply_difficulty_adaptation(self, 
                                         progression: ProgressionState,
                                         adaptation: Dict[str, Any]) -> None:
        """Apply difficulty adaptation"""
        try:
            adjustment = adaptation["adjustment"]
            
            # Apply adjustment to all difficulty metrics
            current = progression.current_difficulty
            
            # Adjust primary metrics
            current.complexity_score = max(0.0, min(1.0, current.complexity_score + adjustment))
            current.time_pressure = max(0.0, min(1.0, current.time_pressure + adjustment * 0.5))
            current.resource_constraints = max(0.0, min(1.0, current.resource_constraints + adjustment * 0.3))
            
            # Recalculate derived metrics
            current.__post_init__()
            
            # Record adaptation
            progression.last_adaptation = datetime.now()
            progression.adaptation_direction = 1 if adjustment > 0 else -1
            progression.difficulty_history.append(current)
            
            self.logger.info(f"Applied difficulty adaptation: {adaptation['direction']} by {abs(adjustment)}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply difficulty adaptation: {e}")
    
    async def _evaluate_stage_advancement(self, 
                                        progression: ProgressionState,
                                        path: ProgressionPath) -> Dict[str, Any]:
        """Evaluate if stage advancement is warranted"""
        try:
            # Check if at final stage
            if progression.current_stage >= len(path.stages) - 1:
                return {
                    "advance": False,
                    "reason": "final_stage",
                    "recommendations": ["Consider new progression path"]
                }
            
            # Check mastery threshold
            if progression.stage_success_rate >= path.mastery_threshold:
                if progression.stage_attempts >= path.min_attempts:
                    return {
                        "advance": True,
                        "reason": "mastery_achieved",
                        "recommendations": ["Ready for next challenge level"]
                    }
                else:
                    return {
                        "advance": False,
                        "reason": "insufficient_attempts",
                        "recommendations": ["Complete more scenarios at current level"]
                    }
            
            # Force advancement if stuck too long
            if progression.stage_attempts >= path.max_attempts:
                return {
                    "advance": True,
                    "reason": "max_attempts_reached",
                    "recommendations": ["Moving to next stage despite low success rate"]
                }
            
            return {
                "advance": False,
                "reason": "mastery_not_achieved",
                "recommendations": ["Continue practicing at current level"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate stage advancement: {e}")
            return {"advance": False, "reason": "error"}
    
    async def _advance_progression_stage(self, 
                                       progression: ProgressionState,
                                       path: ProgressionPath) -> None:
        """Advance agent to next progression stage"""
        try:
            # Move to next stage
            progression.current_stage += 1
            
            # Reset stage tracking
            progression.stage_attempts = 0
            progression.stage_successes = 0
            progression.stage_success_rate = 0.0
            
            # Increase difficulty for new stage
            if progression.current_stage < len(path.stages):
                stage_config = path.stages[progression.current_stage]
                difficulty_increase = stage_config.get("difficulty_increase", 0.1)
                
                # Apply difficulty increase
                current = progression.current_difficulty
                current.complexity_score = min(1.0, current.complexity_score + difficulty_increase)
                current.multi_step_complexity = min(1.0, current.multi_step_complexity + difficulty_increase * 0.5)
                current.__post_init__()
            
            self.logger.info(f"Advanced agent {progression.agent_id} to stage {progression.current_stage}")
            
        except Exception as e:
            self.logger.error(f"Failed to advance progression stage: {e}")
    
    async def _initialize_difficulty_models(self) -> None:
        """Initialize difficulty calculation models"""
        try:
            # Create category-specific difficulty models
            for category in ScenarioCategory:
                self.difficulty_models[category] = {
                    "base_complexity": 0.5,
                    "time_pressure_factor": 1.0,
                    "resource_factor": 1.0,
                    "collaboration_factor": 1.0
                }
            
            # Customize models for specific categories
            self.difficulty_models[ScenarioCategory.RECONNAISSANCE]["time_pressure_factor"] = 0.8
            self.difficulty_models[ScenarioCategory.INCIDENT_RESPONSE]["time_pressure_factor"] = 1.5
            self.difficulty_models[ScenarioCategory.THREAT_HUNTING]["collaboration_factor"] = 1.3
            
        except Exception as e:
            self.logger.error(f"Failed to initialize difficulty models: {e}")
    
    async def _create_default_progression_paths(self) -> None:
        """Create default progression paths"""
        try:
            # Red Team progression path
            red_team_path = ProgressionPath(
                path_id="red_team_progression",
                name="Red Team Skill Progression",
                description="Progressive red team skill development from reconnaissance to advanced persistence",
                strategy=ProgressionStrategy.MASTERY_BASED,
                target_category=ScenarioCategory.RECONNAISSANCE,
                stages=[
                    {"name": "Basic Reconnaissance", "difficulty_increase": 0.0},
                    {"name": "Initial Access", "difficulty_increase": 0.15},
                    {"name": "Privilege Escalation", "difficulty_increase": 0.15},
                    {"name": "Persistence", "difficulty_increase": 0.15},
                    {"name": "Advanced Techniques", "difficulty_increase": 0.20}
                ]
            )
            
            # Blue Team progression path
            blue_team_path = ProgressionPath(
                path_id="blue_team_progression",
                name="Blue Team Defense Progression",
                description="Progressive defensive skill development from monitoring to threat hunting",
                strategy=ProgressionStrategy.MASTERY_BASED,
                target_category=ScenarioCategory.INCIDENT_RESPONSE,
                stages=[
                    {"name": "Basic Monitoring", "difficulty_increase": 0.0},
                    {"name": "Incident Response", "difficulty_increase": 0.15},
                    {"name": "Threat Hunting", "difficulty_increase": 0.15},
                    {"name": "Advanced Analysis", "difficulty_increase": 0.20},
                    {"name": "Proactive Defense", "difficulty_increase": 0.20}
                ]
            )
            
            self.progression_paths[red_team_path.path_id] = red_team_path
            self.progression_paths[blue_team_path.path_id] = blue_team_path
            
        except Exception as e:
            self.logger.error(f"Failed to create default progression paths: {e}")
    
    async def _load_progression_states(self) -> None:
        """Load existing progression states"""
        try:
            # In a real implementation, this would load from persistent storage
            # For now, we'll start with empty states
            pass
            
        except Exception as e:
            self.logger.error(f"Failed to load progression states: {e}")
    
    async def _validate_progression_path(self, path: ProgressionPath) -> Dict[str, Any]:
        """Validate a progression path"""
        errors = []
        
        if not path.name or not path.description:
            errors.append("Path must have name and description")
        
        if not path.stages:
            errors.append("Path must have at least one stage")
        
        if path.mastery_threshold < 0.0 or path.mastery_threshold > 1.0:
            errors.append("Mastery threshold must be between 0.0 and 1.0")
        
        if path.min_attempts > path.max_attempts:
            errors.append("Minimum attempts cannot exceed maximum attempts")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }