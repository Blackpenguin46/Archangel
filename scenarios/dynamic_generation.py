#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Dynamic Scenario Generation
AI-driven scenario generation based on learning outcomes and agent performance
"""

import asyncio
import logging
import uuid
import random
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from collections import defaultdict
import json

from .scenario_templates import (
    ScenarioTemplate, ScenarioInstance, ScenarioObjective, ScenarioParameter, ScenarioAsset,
    ScenarioType, ScenarioCategory, ComplexityLevel, NetworkTopology
)
from agents.base_agent import Experience, Team, Role

logger = logging.getLogger(__name__)

class GenerationType(Enum):
    """Types of scenario generation"""
    ADAPTIVE_LEARNING = "adaptive_learning"     # Based on learning progress
    SKILL_ASSESSMENT = "skill_assessment"       # For skills evaluation
    WEAKNESS_TARGETING = "weakness_targeting"   # Target identified weaknesses
    PROGRESSION_BASED = "progression_based"     # Sequential skill building
    CHALLENGE_SCALING = "challenge_scaling"     # Dynamic difficulty adjustment
    COLLABORATIVE = "collaborative"             # Multi-agent scenarios
    RESEARCH_DRIVEN = "research_driven"         # For research objectives

class LearningOutcome(Enum):
    """Learning outcome categories"""
    SKILL_MASTERY = "skill_mastery"
    KNOWLEDGE_GAP = "knowledge_gap"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    COLLABORATION_SKILL = "collaboration_skill"
    CREATIVE_THINKING = "creative_thinking"
    PROBLEM_SOLVING = "problem_solving"
    ADAPTATION_ABILITY = "adaptation_ability"
    DECISION_MAKING = "decision_making"

@dataclass
class AgentLearningProfile:
    """Comprehensive agent learning profile"""
    agent_id: str
    
    # Skill assessments
    skill_levels: Dict[str, float] = field(default_factory=dict)  # skill -> proficiency (0-1)
    learning_rate: float = 0.5  # How quickly agent learns
    retention_rate: float = 0.8  # How well agent retains knowledge
    
    # Performance patterns
    success_rates: Dict[ScenarioCategory, float] = field(default_factory=dict)
    average_completion_time: Dict[ScenarioCategory, float] = field(default_factory=dict)
    confidence_levels: Dict[ScenarioCategory, float] = field(default_factory=dict)
    
    # Learning preferences
    preferred_complexity: ComplexityLevel = ComplexityLevel.INTERMEDIATE
    preferred_categories: List[ScenarioCategory] = field(default_factory=list)
    learning_style: str = "balanced"  # visual, hands_on, analytical, collaborative
    
    # Weakness and strength analysis
    identified_weaknesses: List[str] = field(default_factory=list)
    core_strengths: List[str] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)
    
    # Learning history
    completed_scenarios: List[str] = field(default_factory=list)
    recent_outcomes: List[LearningOutcome] = field(default_factory=list)
    learning_trajectory: List[Tuple[datetime, float]] = field(default_factory=list)  # (time, skill_level)
    
    # Collaboration metrics
    team_performance: Dict[str, float] = field(default_factory=dict)  # team_id -> performance
    collaboration_effectiveness: float = 0.5
    leadership_potential: float = 0.5
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    profile_confidence: float = 0.5  # Confidence in profile accuracy

@dataclass
class GenerationContext:
    """Context for dynamic scenario generation"""
    # Target agents
    target_agents: List[str]
    agent_profiles: Dict[str, AgentLearningProfile] = field(default_factory=dict)
    
    # Learning objectives
    primary_objectives: List[LearningOutcome] = field(default_factory=list)
    skill_targets: Dict[str, float] = field(default_factory=dict)  # skill -> target_level
    
    # Generation constraints
    time_constraint: Optional[timedelta] = None
    complexity_range: Tuple[ComplexityLevel, ComplexityLevel] = (ComplexityLevel.BEGINNER, ComplexityLevel.EXPERT)
    allowed_categories: List[ScenarioCategory] = field(default_factory=list)
    
    # Context factors
    previous_scenarios: List[str] = field(default_factory=list)
    available_resources: Dict[str, Any] = field(default_factory=dict)
    environmental_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Generation preferences
    generation_type: GenerationType = GenerationType.ADAPTIVE_LEARNING
    novelty_factor: float = 0.3  # 0 = familiar, 1 = novel
    challenge_factor: float = 0.5  # 0 = easy, 1 = challenging
    collaboration_factor: float = 0.5  # 0 = individual, 1 = team-focused

@dataclass
class GeneratedScenario:
    """Result of dynamic scenario generation"""
    scenario: ScenarioTemplate
    generation_rationale: str
    confidence_score: float
    
    # Generation metadata
    generation_type: GenerationType
    target_outcomes: List[LearningOutcome]
    expected_difficulty: float
    novelty_score: float
    
    # Adaptation recommendations
    parameter_suggestions: Dict[str, Any] = field(default_factory=dict)
    monitoring_points: List[str] = field(default_factory=list)
    adaptation_triggers: Dict[str, Any] = field(default_factory=dict)
    
    # Predicted outcomes
    success_probability: float = 0.5
    learning_potential: float = 0.5
    engagement_score: float = 0.5

class DynamicScenarioGenerator:
    """
    AI-driven dynamic scenario generation system.
    
    Features:
    - Learning outcome-based generation
    - Agent profile analysis and adaptation
    - Dynamic difficulty adjustment
    - Collaborative scenario creation
    - Weakness targeting and skill progression
    - Novel scenario synthesis
    """
    
    def __init__(self, template_manager=None):
        self.template_manager = template_manager
        
        # Learning analytics
        self.agent_profiles: Dict[str, AgentLearningProfile] = {}
        self.learning_outcomes_history: List[Dict[str, Any]] = []
        
        # Generation models and algorithms
        self.skill_progression_models: Dict[str, Any] = {}
        self.difficulty_prediction_models: Dict[str, Any] = {}
        self.engagement_models: Dict[str, Any] = {}
        
        # Generated scenario cache
        self.generated_scenarios: Dict[str, GeneratedScenario] = {}
        self.generation_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.generation_analytics = {
            "total_generated": 0,
            "generation_success_rate": 0.0,
            "avg_learning_effectiveness": 0.0,
            "generation_time_avg": 0.0,
            "model_accuracy": {}
        }
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self, template_manager=None) -> None:
        """Initialize the dynamic scenario generator"""
        try:
            self.logger.info("Initializing dynamic scenario generator")
            
            if template_manager:
                self.template_manager = template_manager
            
            # Initialize learning models
            await self._initialize_learning_models()
            
            # Load existing agent profiles
            await self._load_agent_profiles()
            
            # Initialize generation algorithms
            await self._initialize_generation_algorithms()
            
            self.initialized = True
            self.logger.info("Dynamic scenario generator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dynamic scenario generator: {e}")
            raise
    
    async def generate_scenario(self, context: GenerationContext) -> GeneratedScenario:
        """
        Generate a dynamic scenario based on learning context
        
        Args:
            context: Generation context with agents, objectives, and constraints
            
        Returns:
            GeneratedScenario: Generated scenario with metadata
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"Generating scenario for {len(context.target_agents)} agents")
            
            # Analyze agent profiles
            await self._analyze_agent_profiles(context)
            
            # Determine generation strategy
            generation_strategy = await self._select_generation_strategy(context)
            
            # Generate scenario based on strategy
            if generation_strategy == GenerationType.ADAPTIVE_LEARNING:
                generated = await self._generate_adaptive_learning_scenario(context)
            elif generation_strategy == GenerationType.SKILL_ASSESSMENT:
                generated = await self._generate_skill_assessment_scenario(context)
            elif generation_strategy == GenerationType.WEAKNESS_TARGETING:
                generated = await self._generate_weakness_targeting_scenario(context)
            elif generation_strategy == GenerationType.PROGRESSION_BASED:
                generated = await self._generate_progression_based_scenario(context)
            elif generation_strategy == GenerationType.CHALLENGE_SCALING:
                generated = await self._generate_challenge_scaling_scenario(context)
            elif generation_strategy == GenerationType.COLLABORATIVE:
                generated = await self._generate_collaborative_scenario(context)
            else:
                generated = await self._generate_research_driven_scenario(context)
            
            # Post-process and validate
            validated_scenario = await self._validate_generated_scenario(generated, context)
            
            # Record generation
            generation_time = (datetime.now() - start_time).total_seconds()
            await self._record_generation(validated_scenario, context, generation_time)
            
            self.logger.info(f"Generated scenario '{validated_scenario.scenario.name}' in {generation_time:.2f}s")
            return validated_scenario
            
        except Exception as e:
            self.logger.error(f"Failed to generate scenario: {e}")
            raise
    
    async def update_agent_profile(self, agent_id: str, scenario_result: Dict[str, Any]) -> None:
        """Update agent learning profile based on scenario results"""
        try:
            if agent_id not in self.agent_profiles:
                self.agent_profiles[agent_id] = AgentLearningProfile(agent_id=agent_id)
            
            profile = self.agent_profiles[agent_id]
            
            # Extract results
            success = scenario_result.get("success", False)
            completion_time = scenario_result.get("completion_time", 0)
            confidence = scenario_result.get("confidence", 0.5)
            scenario_category = scenario_result.get("category", ScenarioCategory.RECONNAISSANCE)
            skills_demonstrated = scenario_result.get("skills_demonstrated", [])
            lessons_learned = scenario_result.get("lessons_learned", [])
            
            # Update success rates
            current_success = profile.success_rates.get(scenario_category, 0.5)
            profile.success_rates[scenario_category] = self._update_exponential_average(
                current_success, 1.0 if success else 0.0, 0.2
            )
            
            # Update completion time
            if completion_time > 0:
                current_time = profile.average_completion_time.get(scenario_category, completion_time)
                profile.average_completion_time[scenario_category] = self._update_exponential_average(
                    current_time, completion_time, 0.3
                )
            
            # Update confidence levels
            current_confidence = profile.confidence_levels.get(scenario_category, 0.5)
            profile.confidence_levels[scenario_category] = self._update_exponential_average(
                current_confidence, confidence, 0.2
            )
            
            # Update skill levels
            for skill in skills_demonstrated:
                current_level = profile.skill_levels.get(skill, 0.5)
                improvement = 0.1 if success else -0.05  # Success improves, failure decreases slightly
                profile.skill_levels[skill] = max(0.0, min(1.0, current_level + improvement))
            
            # Analyze learning outcomes
            outcomes = await self._analyze_learning_outcomes(scenario_result, profile)
            profile.recent_outcomes.extend(outcomes)
            
            # Keep only recent outcomes (last 10)
            profile.recent_outcomes = profile.recent_outcomes[-10:]
            
            # Update learning trajectory
            avg_skill = sum(profile.skill_levels.values()) / max(1, len(profile.skill_levels))
            profile.learning_trajectory.append((datetime.now(), avg_skill))
            
            # Keep only recent trajectory points (last 50)
            profile.learning_trajectory = profile.learning_trajectory[-50:]
            
            # Update weaknesses and strengths
            await self._update_weaknesses_and_strengths(profile)
            
            # Update metadata
            profile.last_updated = datetime.now()
            profile.profile_confidence = min(1.0, profile.profile_confidence + 0.05)
            
            self.logger.debug(f"Updated profile for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update agent profile {agent_id}: {e}")
    
    async def get_learning_recommendations(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get learning recommendations for an agent"""
        try:
            if agent_id not in self.agent_profiles:
                return []
            
            profile = self.agent_profiles[agent_id]
            recommendations = []
            
            # Weakness-based recommendations
            for weakness in profile.identified_weaknesses[:3]:  # Top 3 weaknesses
                context = GenerationContext(
                    target_agents=[agent_id],
                    generation_type=GenerationType.WEAKNESS_TARGETING,
                    primary_objectives=[LearningOutcome.SKILL_MASTERY],
                    skill_targets={weakness: min(1.0, profile.skill_levels.get(weakness, 0.5) + 0.2)}
                )
                
                recommendations.append({
                    "type": "weakness_targeting",
                    "target_skill": weakness,
                    "current_level": profile.skill_levels.get(weakness, 0.5),
                    "target_level": context.skill_targets[weakness],
                    "recommendation": f"Focus on improving {weakness} skills",
                    "priority": "high"
                })
            
            # Progression-based recommendations
            ready_for_advancement = []
            for skill, level in profile.skill_levels.items():
                if level > 0.7 and skill not in profile.core_strengths:
                    ready_for_advancement.append((skill, level))
            
            for skill, level in sorted(ready_for_advancement, key=lambda x: x[1], reverse=True)[:2]:
                recommendations.append({
                    "type": "skill_advancement",
                    "target_skill": skill,
                    "current_level": level,
                    "recommendation": f"Advanced challenges in {skill}",
                    "priority": "medium"
                })
            
            # Collaborative recommendations
            if profile.collaboration_effectiveness < 0.6:
                recommendations.append({
                    "type": "collaboration_improvement",
                    "current_level": profile.collaboration_effectiveness,
                    "recommendation": "Practice team-based scenarios",
                    "priority": "medium"
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get learning recommendations for {agent_id}: {e}")
            return []
    
    async def _analyze_agent_profiles(self, context: GenerationContext) -> None:
        """Analyze and update agent profiles in context"""
        try:
            for agent_id in context.target_agents:
                if agent_id not in self.agent_profiles:
                    # Create new profile
                    profile = AgentLearningProfile(agent_id=agent_id)
                    self.agent_profiles[agent_id] = profile
                
                context.agent_profiles[agent_id] = self.agent_profiles[agent_id]
            
        except Exception as e:
            self.logger.error(f"Failed to analyze agent profiles: {e}")
    
    async def _select_generation_strategy(self, context: GenerationContext) -> GenerationType:
        """Select the most appropriate generation strategy"""
        try:
            # Use specified strategy if available
            if context.generation_type != GenerationType.ADAPTIVE_LEARNING:
                return context.generation_type
            
            # Analyze agent profiles to select strategy
            profiles = list(context.agent_profiles.values())
            
            if not profiles:
                return GenerationType.ADAPTIVE_LEARNING
            
            # Multiple agents suggest collaborative scenarios
            if len(profiles) > 1 and context.collaboration_factor > 0.6:
                return GenerationType.COLLABORATIVE
            
            # Check for significant weaknesses
            has_major_weaknesses = any(
                len(profile.identified_weaknesses) > 2 for profile in profiles
            )
            if has_major_weaknesses:
                return GenerationType.WEAKNESS_TARGETING
            
            # Check skill progression readiness
            ready_for_progression = any(
                max(profile.skill_levels.values(), default=0.0) > 0.7 for profile in profiles
            )
            if ready_for_progression:
                return GenerationType.PROGRESSION_BASED
            
            # Default to adaptive learning
            return GenerationType.ADAPTIVE_LEARNING
            
        except Exception as e:
            self.logger.error(f"Failed to select generation strategy: {e}")
            return GenerationType.ADAPTIVE_LEARNING
    
    async def _generate_adaptive_learning_scenario(self, context: GenerationContext) -> GeneratedScenario:
        """Generate adaptive learning scenario"""
        try:
            # Analyze current skill levels
            avg_skill_levels = {}
            for profile in context.agent_profiles.values():
                for skill, level in profile.skill_levels.items():
                    if skill not in avg_skill_levels:
                        avg_skill_levels[skill] = []
                    avg_skill_levels[skill].append(level)
            
            # Calculate average skill levels
            for skill in avg_skill_levels:
                avg_skill_levels[skill] = sum(avg_skill_levels[skill]) / len(avg_skill_levels[skill])
            
            # Find skill gaps
            skill_gaps = {}
            for skill, level in avg_skill_levels.items():
                target_level = context.skill_targets.get(skill, 0.7)
                if level < target_level:
                    skill_gaps[skill] = target_level - level
            
            # Select primary skill to target
            if skill_gaps:
                target_skill = max(skill_gaps.keys(), key=lambda s: skill_gaps[s])
            else:
                target_skill = random.choice(list(avg_skill_levels.keys())) if avg_skill_levels else "reconnaissance"
            
            # Map skill to scenario category
            skill_category_map = {
                "reconnaissance": ScenarioCategory.RECONNAISSANCE,
                "exploitation": ScenarioCategory.INITIAL_ACCESS,
                "persistence": ScenarioCategory.PERSISTENCE,
                "lateral_movement": ScenarioCategory.LATERAL_MOVEMENT,
                "incident_response": ScenarioCategory.INCIDENT_RESPONSE,
                "threat_hunting": ScenarioCategory.THREAT_HUNTING
            }
            
            target_category = skill_category_map.get(target_skill, ScenarioCategory.RECONNAISSANCE)
            
            # Determine complexity based on skill level
            avg_skill = avg_skill_levels.get(target_skill, 0.5)
            if avg_skill < 0.3:
                complexity = ComplexityLevel.BEGINNER
            elif avg_skill < 0.6:
                complexity = ComplexityLevel.INTERMEDIATE
            elif avg_skill < 0.8:
                complexity = ComplexityLevel.ADVANCED
            else:
                complexity = ComplexityLevel.EXPERT
            
            # Create scenario template
            scenario_template = await self._create_adaptive_scenario_template(
                target_category, complexity, target_skill, context
            )
            
            # Calculate confidence and metrics
            confidence_score = 0.7  # Moderate confidence for adaptive scenarios
            novelty_score = context.novelty_factor
            
            return GeneratedScenario(
                scenario=scenario_template,
                generation_rationale=f"Adaptive learning scenario targeting {target_skill} skill gap",
                confidence_score=confidence_score,
                generation_type=GenerationType.ADAPTIVE_LEARNING,
                target_outcomes=[LearningOutcome.SKILL_MASTERY, LearningOutcome.PERFORMANCE_IMPROVEMENT],
                expected_difficulty=complexity.value / 5.0,
                novelty_score=novelty_score,
                success_probability=0.6,
                learning_potential=0.8,
                engagement_score=0.7
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate adaptive learning scenario: {e}")
            raise
    
    async def _generate_weakness_targeting_scenario(self, context: GenerationContext) -> GeneratedScenario:
        """Generate scenario targeting specific weaknesses"""
        try:
            # Identify common weaknesses
            weakness_counts = defaultdict(int)
            for profile in context.agent_profiles.values():
                for weakness in profile.identified_weaknesses:
                    weakness_counts[weakness] += 1
            
            # Select most common weakness
            if weakness_counts:
                target_weakness = max(weakness_counts.keys(), key=lambda w: weakness_counts[w])
            else:
                # Default weakness if none identified
                target_weakness = "reconnaissance"
            
            # Create focused scenario
            scenario_template = await self._create_weakness_focused_template(target_weakness, context)
            
            return GeneratedScenario(
                scenario=scenario_template,
                generation_rationale=f"Weakness targeting scenario focusing on {target_weakness}",
                confidence_score=0.8,
                generation_type=GenerationType.WEAKNESS_TARGETING,
                target_outcomes=[LearningOutcome.SKILL_MASTERY, LearningOutcome.KNOWLEDGE_GAP],
                expected_difficulty=0.7,  # Slightly challenging to address weakness
                novelty_score=0.4,  # Less novel, more focused
                success_probability=0.5,  # Lower success expected initially
                learning_potential=0.9,  # High learning potential
                engagement_score=0.6
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate weakness targeting scenario: {e}")
            raise
    
    async def _generate_collaborative_scenario(self, context: GenerationContext) -> GeneratedScenario:
        """Generate collaborative multi-agent scenario"""
        try:
            # Analyze team composition
            team_skills = defaultdict(list)
            for profile in context.agent_profiles.values():
                for skill, level in profile.skill_levels.items():
                    team_skills[skill].append(level)
            
            # Find complementary skills
            strong_skills = []
            weak_skills = []
            
            for skill, levels in team_skills.items():
                avg_level = sum(levels) / len(levels)
                if avg_level > 0.7:
                    strong_skills.append(skill)
                elif avg_level < 0.4:
                    weak_skills.append(skill)
            
            # Create collaborative scenario that leverages strengths and improves weaknesses
            scenario_template = await self._create_collaborative_template(
                strong_skills, weak_skills, context
            )
            
            return GeneratedScenario(
                scenario=scenario_template,
                generation_rationale="Collaborative scenario leveraging team strengths and addressing weaknesses",
                confidence_score=0.7,
                generation_type=GenerationType.COLLABORATIVE,
                target_outcomes=[LearningOutcome.COLLABORATION_SKILL, LearningOutcome.PROBLEM_SOLVING],
                expected_difficulty=0.6,
                novelty_score=0.6,
                success_probability=0.7,  # Higher success with collaboration
                learning_potential=0.8,
                engagement_score=0.9  # High engagement for collaborative scenarios
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate collaborative scenario: {e}")
            raise
    
    async def _generate_skill_assessment_scenario(self, context: GenerationContext) -> GeneratedScenario:
        """Generate skill assessment scenario"""
        try:
            # Create assessment-focused scenario
            scenario_template = await self._create_assessment_template(context)
            
            return GeneratedScenario(
                scenario=scenario_template,
                generation_rationale="Skill assessment scenario for comprehensive evaluation",
                confidence_score=0.9,
                generation_type=GenerationType.SKILL_ASSESSMENT,
                target_outcomes=[LearningOutcome.SKILL_MASTERY],
                expected_difficulty=0.5,
                novelty_score=0.3,
                success_probability=0.6,
                learning_potential=0.6,
                engagement_score=0.7
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate skill assessment scenario: {e}")
            raise
    
    async def _generate_progression_based_scenario(self, context: GenerationContext) -> GeneratedScenario:
        """Generate progression-based scenario"""
        try:
            # Create progression scenario
            scenario_template = await self._create_progression_template(context)
            
            return GeneratedScenario(
                scenario=scenario_template,
                generation_rationale="Progression-based scenario for skill advancement",
                confidence_score=0.8,
                generation_type=GenerationType.PROGRESSION_BASED,
                target_outcomes=[LearningOutcome.SKILL_MASTERY, LearningOutcome.PERFORMANCE_IMPROVEMENT],
                expected_difficulty=0.8,
                novelty_score=0.5,
                success_probability=0.6,
                learning_potential=0.8,
                engagement_score=0.8
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate progression-based scenario: {e}")
            raise
    
    async def _generate_challenge_scaling_scenario(self, context: GenerationContext) -> GeneratedScenario:
        """Generate challenge scaling scenario"""
        try:
            # Create challenge scaling scenario
            scenario_template = await self._create_challenge_scaling_template(context)
            
            return GeneratedScenario(
                scenario=scenario_template,
                generation_rationale="Challenge scaling scenario with dynamic difficulty",
                confidence_score=0.7,
                generation_type=GenerationType.CHALLENGE_SCALING,
                target_outcomes=[LearningOutcome.ADAPTATION_ABILITY, LearningOutcome.PROBLEM_SOLVING],
                expected_difficulty=0.7,
                novelty_score=0.7,
                success_probability=0.6,
                learning_potential=0.8,
                engagement_score=0.8
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate challenge scaling scenario: {e}")
            raise
    
    async def _generate_research_driven_scenario(self, context: GenerationContext) -> GeneratedScenario:
        """Generate research-driven scenario"""
        try:
            # Create research scenario
            scenario_template = await self._create_research_template(context)
            
            return GeneratedScenario(
                scenario=scenario_template,
                generation_rationale="Research-driven scenario for experimental learning",
                confidence_score=0.6,
                generation_type=GenerationType.RESEARCH_DRIVEN,
                target_outcomes=[LearningOutcome.CREATIVE_THINKING, LearningOutcome.PROBLEM_SOLVING],
                expected_difficulty=0.8,
                novelty_score=0.9,
                success_probability=0.5,
                learning_potential=0.9,
                engagement_score=0.8
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate research-driven scenario: {e}")
            raise
    
    async def _create_adaptive_scenario_template(self, 
                                               category: ScenarioCategory, 
                                               complexity: ComplexityLevel,
                                               target_skill: str,
                                               context: GenerationContext) -> ScenarioTemplate:
        """Create adaptive scenario template"""
        try:
            template_id = f"dynamic_{category.value}_{complexity.value}_{uuid.uuid4().hex[:8]}"
            
            # Base scenario configuration
            scenario_template = ScenarioTemplate(
                template_id=template_id,
                name=f"Adaptive {category.value.title()} Challenge",
                description=f"Dynamically generated {category.value} scenario targeting {target_skill} skills",
                scenario_type=ScenarioType.TRAINING,
                category=category,
                complexity=complexity,
                estimated_duration=timedelta(hours=2),
                min_participants=len(context.target_agents),
                max_participants=len(context.target_agents),
                tags={category.value, target_skill, "adaptive", "dynamic"}
            )
            
            # Add objectives based on category and skill
            scenario_template.objectives = await self._generate_objectives_for_skill(target_skill, complexity)
            
            # Add parameters
            scenario_template.parameters = await self._generate_adaptive_parameters(context)
            
            return scenario_template
            
        except Exception as e:
            self.logger.error(f"Failed to create adaptive scenario template: {e}")
            raise
    
    async def _create_weakness_focused_template(self, weakness: str, context: GenerationContext) -> ScenarioTemplate:
        """Create template focused on specific weakness"""
        template_id = f"weakness_{weakness}_{uuid.uuid4().hex[:8]}"
        
        return ScenarioTemplate(
            template_id=template_id,
            name=f"Weakness Training: {weakness.title()}",
            description=f"Focused training scenario to address {weakness} weaknesses",
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.RECONNAISSANCE,  # Default, would map weakness to category
            complexity=ComplexityLevel.INTERMEDIATE,
            estimated_duration=timedelta(hours=1, minutes=30),
            min_participants=len(context.target_agents),
            max_participants=len(context.target_agents),
            tags={weakness, "weakness_targeting", "focused_training"}
        )
    
    async def _create_collaborative_template(self, strong_skills: List[str], weak_skills: List[str], 
                                           context: GenerationContext) -> ScenarioTemplate:
        """Create collaborative scenario template"""
        template_id = f"collaborative_{uuid.uuid4().hex[:8]}"
        
        return ScenarioTemplate(
            template_id=template_id,
            name="Team Collaboration Exercise",
            description="Multi-agent collaborative scenario leveraging team strengths",
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.FULL_CAMPAIGN,
            complexity=ComplexityLevel.ADVANCED,
            estimated_duration=timedelta(hours=3),
            min_participants=len(context.target_agents),
            max_participants=len(context.target_agents),
            recommended_teams=[Team.RED_TEAM, Team.BLUE_TEAM],
            tags={"collaborative", "team_exercise", "multi_agent"}
        )
    
    async def _create_assessment_template(self, context: GenerationContext) -> ScenarioTemplate:
        """Create assessment scenario template"""
        template_id = f"assessment_{uuid.uuid4().hex[:8]}"
        
        return ScenarioTemplate(
            template_id=template_id,
            name="Comprehensive Skill Assessment",
            description="Multi-faceted assessment scenario evaluating various skills",
            scenario_type=ScenarioType.ASSESSMENT,
            category=ScenarioCategory.FULL_CAMPAIGN,
            complexity=ComplexityLevel.INTERMEDIATE,
            estimated_duration=timedelta(hours=2, minutes=30),
            min_participants=len(context.target_agents),
            max_participants=len(context.target_agents),
            tags={"assessment", "evaluation", "comprehensive"}
        )
    
    async def _create_progression_template(self, context: GenerationContext) -> ScenarioTemplate:
        """Create progression-based scenario template"""
        template_id = f"progression_{uuid.uuid4().hex[:8]}"
        
        return ScenarioTemplate(
            template_id=template_id,
            name="Skill Progression Challenge",
            description="Advanced scenario for skill progression and mastery",
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.DISCOVERY,
            complexity=ComplexityLevel.ADVANCED,
            estimated_duration=timedelta(hours=4),
            min_participants=len(context.target_agents),
            max_participants=len(context.target_agents),
            tags={"progression", "advanced", "mastery"}
        )
    
    async def _create_challenge_scaling_template(self, context: GenerationContext) -> ScenarioTemplate:
        """Create challenge scaling scenario template"""
        template_id = f"scaling_{uuid.uuid4().hex[:8]}"
        
        return ScenarioTemplate(
            template_id=template_id,
            name="Adaptive Challenge Scaling",
            description="Dynamic scenario with adaptive difficulty scaling",
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.LATERAL_MOVEMENT,
            complexity=ComplexityLevel.EXPERT,
            estimated_duration=timedelta(hours=3),
            min_participants=len(context.target_agents),
            max_participants=len(context.target_agents),
            tags={"scaling", "adaptive", "dynamic_difficulty"}
        )
    
    async def _create_research_template(self, context: GenerationContext) -> ScenarioTemplate:
        """Create research-driven scenario template"""
        template_id = f"research_{uuid.uuid4().hex[:8]}"
        
        return ScenarioTemplate(
            template_id=template_id,
            name="Research and Innovation Challenge",
            description="Open-ended research scenario for creative problem solving",
            scenario_type=ScenarioType.RESEARCH,
            category=ScenarioCategory.THREAT_HUNTING,
            complexity=ComplexityLevel.MASTER,
            estimated_duration=timedelta(hours=6),
            min_participants=len(context.target_agents),
            max_participants=len(context.target_agents),
            tags={"research", "innovation", "creative", "open_ended"}
        )
    
    async def _generate_objectives_for_skill(self, skill: str, complexity: ComplexityLevel) -> List[ScenarioObjective]:
        """Generate objectives based on skill and complexity"""
        objectives = []
        
        base_points = 50 * complexity.value
        
        if skill == "reconnaissance":
            objectives.append(ScenarioObjective(
                objective_id="obj_recon_1",
                name="Network Discovery",
                description="Discover and map target network infrastructure",
                success_criteria=["Identify live hosts", "Map network topology"],
                points=base_points
            ))
        elif skill == "exploitation":
            objectives.append(ScenarioObjective(
                objective_id="obj_exploit_1",
                name="Initial Access",
                description="Gain initial access to target systems",
                success_criteria=["Exploit vulnerability", "Establish foothold"],
                points=base_points
            ))
        else:
            # Generic objective
            objectives.append(ScenarioObjective(
                objective_id="obj_generic_1",
                name=f"{skill.title()} Challenge",
                description=f"Demonstrate proficiency in {skill}",
                success_criteria=["Complete primary task", "Document approach"],
                points=base_points
            ))
        
        return objectives
    
    async def _generate_adaptive_parameters(self, context: GenerationContext) -> List[ScenarioParameter]:
        """Generate adaptive parameters based on context"""
        parameters = []
        
        # Difficulty parameter
        parameters.append(ScenarioParameter(
            name="difficulty_level",
            parameter_type="enum",
            default_value="adaptive",
            allowed_values=["easy", "normal", "hard", "adaptive"],
            description="Scenario difficulty level"
        ))
        
        # Time constraint parameter
        parameters.append(ScenarioParameter(
            name="time_limit",
            parameter_type="int",
            default_value=120,  # 2 hours
            min_value=30,
            max_value=480,
            description="Time limit in minutes"
        ))
        
        return parameters
    
    async def _analyze_learning_outcomes(self, scenario_result: Dict[str, Any], 
                                       profile: AgentLearningProfile) -> List[LearningOutcome]:
        """Analyze learning outcomes from scenario results"""
        outcomes = []
        
        success = scenario_result.get("success", False)
        improvement_shown = scenario_result.get("improvement_shown", False)
        collaboration_effective = scenario_result.get("collaboration_effective", False)
        creative_approach = scenario_result.get("creative_approach", False)
        
        if success and improvement_shown:
            outcomes.append(LearningOutcome.SKILL_MASTERY)
        
        if improvement_shown:
            outcomes.append(LearningOutcome.PERFORMANCE_IMPROVEMENT)
        
        if collaboration_effective:
            outcomes.append(LearningOutcome.COLLABORATION_SKILL)
        
        if creative_approach:
            outcomes.append(LearningOutcome.CREATIVE_THINKING)
        
        if not success:
            outcomes.append(LearningOutcome.KNOWLEDGE_GAP)
        
        return outcomes
    
    async def _update_weaknesses_and_strengths(self, profile: AgentLearningProfile) -> None:
        """Update identified weaknesses and strengths"""
        # Identify weaknesses (skills below 0.4)
        weaknesses = [skill for skill, level in profile.skill_levels.items() if level < 0.4]
        profile.identified_weaknesses = weaknesses[:5]  # Top 5 weaknesses
        
        # Identify strengths (skills above 0.7)
        strengths = [skill for skill, level in profile.skill_levels.items() if level > 0.7]
        profile.core_strengths = strengths[:5]  # Top 5 strengths
        
        # Identify improvement areas (skills between 0.4 and 0.7)
        improvements = [skill for skill, level in profile.skill_levels.items() if 0.4 <= level <= 0.7]
        profile.improvement_areas = improvements[:5]  # Top 5 improvement areas
    
    def _update_exponential_average(self, current: float, new_value: float, alpha: float) -> float:
        """Update exponential moving average"""
        return alpha * new_value + (1 - alpha) * current
    
    async def _validate_generated_scenario(self, generated: GeneratedScenario, 
                                         context: GenerationContext) -> GeneratedScenario:
        """Validate generated scenario"""
        # Basic validation
        if not generated.scenario.name or not generated.scenario.description:
            generated.confidence_score *= 0.5
        
        # Ensure objectives exist
        if not generated.scenario.objectives:
            generated.scenario.objectives = [
                ScenarioObjective(
                    objective_id="default_obj",
                    name="Complete Scenario",
                    description="Successfully complete the scenario",
                    success_criteria=["Finish all tasks"],
                    points=100
                )
            ]
        
        return generated
    
    async def _record_generation(self, generated: GeneratedScenario, 
                                context: GenerationContext, generation_time: float) -> None:
        """Record generation for analytics"""
        self.generation_analytics["total_generated"] += 1
        
        # Update average generation time
        total_gen = self.generation_analytics["total_generated"]
        current_avg = self.generation_analytics["generation_time_avg"]
        self.generation_analytics["generation_time_avg"] = (
            (current_avg * (total_gen - 1) + generation_time) / total_gen
        )
        
        # Store generated scenario
        self.generated_scenarios[generated.scenario.template_id] = generated
        
        # Record in history
        self.generation_history.append({
            "timestamp": datetime.now().isoformat(),
            "scenario_id": generated.scenario.template_id,
            "generation_type": generated.generation_type.value,
            "confidence_score": generated.confidence_score,
            "generation_time": generation_time,
            "target_agents": context.target_agents
        })
    
    async def _initialize_learning_models(self) -> None:
        """Initialize learning and prediction models"""
        try:
            # Placeholder for ML model initialization
            self.logger.debug("Initializing learning models (placeholder)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize learning models: {e}")
    
    async def _load_agent_profiles(self) -> None:
        """Load existing agent profiles"""
        try:
            # In production, would load from persistent storage
            self.logger.debug("Loading agent profiles (placeholder)")
            
        except Exception as e:
            self.logger.error(f"Failed to load agent profiles: {e}")
    
    async def _initialize_generation_algorithms(self) -> None:
        """Initialize generation algorithms"""
        try:
            # Placeholder for algorithm initialization
            self.logger.debug("Initializing generation algorithms")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize generation algorithms: {e}")
    
    def get_generation_analytics(self) -> Dict[str, Any]:
        """Get comprehensive generation analytics"""
        try:
            analytics = dict(self.generation_analytics)
            
            # Add profile statistics
            analytics["agent_profiles"] = {
                "total_profiles": len(self.agent_profiles),
                "avg_skill_level": 0.0,
                "common_weaknesses": [],
                "common_strengths": []
            }
            
            if self.agent_profiles:
                all_skills = defaultdict(list)
                all_weaknesses = []
                all_strengths = []
                
                for profile in self.agent_profiles.values():
                    for skill, level in profile.skill_levels.items():
                        all_skills[skill].append(level)
                    all_weaknesses.extend(profile.identified_weaknesses)
                    all_strengths.extend(profile.core_strengths)
                
                # Calculate average skill levels
                if all_skills:
                    avg_skill = sum(
                        sum(levels) / len(levels) for levels in all_skills.values()
                    ) / len(all_skills)
                    analytics["agent_profiles"]["avg_skill_level"] = avg_skill
                
                # Find common weaknesses and strengths
                from collections import Counter
                weakness_counts = Counter(all_weaknesses)
                strength_counts = Counter(all_strengths)
                
                analytics["agent_profiles"]["common_weaknesses"] = [
                    weakness for weakness, count in weakness_counts.most_common(5)
                ]
                analytics["agent_profiles"]["common_strengths"] = [
                    strength for strength, count in strength_counts.most_common(5)
                ]
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to get generation analytics: {e}")
            return {}

# Factory function
def create_dynamic_generator(template_manager=None) -> DynamicScenarioGenerator:
    """Create a dynamic scenario generator instance"""
    return DynamicScenarioGenerator(template_manager)