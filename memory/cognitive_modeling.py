#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Cognitive Modeling System
Memory decay models and agent personality vectors for realistic cognitive limitations
"""

import asyncio
import logging
import math
import random
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

from agents.base_agent import Experience, Team, Role, AgentConfig

logger = logging.getLogger(__name__)

class PersonalityTrait(Enum):
    """Agent personality traits that affect behavior"""
    CAUTIOUSNESS = "cautiousness"
    AGGRESSIVENESS = "aggressiveness"
    STEALTH = "stealth"
    CURIOSITY = "curiosity"
    PERSISTENCE = "persistence"
    ADAPTABILITY = "adaptability"
    RISK_TOLERANCE = "risk_tolerance"
    COLLABORATION = "collaboration"

class OperationalStyle(Enum):
    """High-level operational styles for agents"""
    CAUTIOUS = "cautious"
    AGGRESSIVE = "aggressive"
    STEALTHY = "stealthy"
    BALANCED = "balanced"
    OPPORTUNISTIC = "opportunistic"
    METHODICAL = "methodical"

class MemoryDecayType(Enum):
    """Types of memory decay models"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    STEPPED = "stepped"
    FORGETTING_CURVE = "forgetting_curve"  # Ebbinghaus forgetting curve

@dataclass
class PersonalityVector:
    """Agent personality represented as trait scores"""
    traits: Dict[PersonalityTrait, float] = field(default_factory=dict)
    operational_style: OperationalStyle = OperationalStyle.BALANCED
    consistency_factor: float = 0.8  # How consistent the agent is (0.0-1.0)
    adaptation_rate: float = 0.1  # How quickly personality adapts (0.0-1.0)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize default trait values if not provided"""
        if not self.traits:
            # Initialize with random values for each trait
            for trait in PersonalityTrait:
                self.traits[trait] = random.uniform(0.2, 0.8)
        
        # Ensure all traits are present
        for trait in PersonalityTrait:
            if trait not in self.traits:
                self.traits[trait] = 0.5  # Default neutral value

@dataclass
class MemoryDecayConfig:
    """Configuration for memory decay models"""
    decay_type: MemoryDecayType = MemoryDecayType.EXPONENTIAL
    half_life: timedelta = timedelta(days=7)  # Time for memory to decay to 50%
    minimum_strength: float = 0.1  # Minimum memory strength (never fully forgotten)
    importance_factor: float = 2.0  # How much importance affects decay resistance
    recency_boost: float = 1.5  # Boost for recently accessed memories
    emotional_weight: float = 1.2  # Weight for emotionally significant memories
    success_bias: float = 1.3  # Bias towards remembering successful experiences

@dataclass
class CognitiveState:
    """Current cognitive state of an agent"""
    attention_span: float = 1.0  # Current attention capacity (0.0-1.0)
    stress_level: float = 0.0  # Current stress level (0.0-1.0)
    confidence: float = 0.5  # Current confidence level (0.0-1.0)
    fatigue: float = 0.0  # Mental fatigue level (0.0-1.0)
    focus_target: Optional[str] = None  # Current focus of attention
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class BehaviorFuzzingConfig:
    """Configuration for behavior fuzzing to introduce unpredictability"""
    enabled: bool = True
    base_randomness: float = 0.1  # Base level of randomness (0.0-1.0)
    stress_amplifier: float = 2.0  # How stress amplifies randomness
    fatigue_amplifier: float = 1.5  # How fatigue amplifies randomness
    personality_influence: float = 0.3  # How personality affects randomness
    decision_noise: float = 0.05  # Random noise in decision making
    action_variance: float = 0.1  # Variance in action execution

class MemoryDecayModel:
    """
    Models realistic memory decay for agent experiences.
    
    Implements various decay functions to simulate how memories fade over time,
    with factors for importance, recency, and emotional significance.
    """
    
    def __init__(self, config: MemoryDecayConfig = None):
        self.config = config or MemoryDecayConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate_memory_strength(self, 
                                experience: Experience,
                                current_time: datetime = None,
                                access_count: int = 0,
                                last_access: datetime = None) -> float:
        """
        Calculate current memory strength based on decay model
        
        Args:
            experience: The experience to calculate strength for
            current_time: Current time (defaults to now)
            access_count: Number of times memory has been accessed
            last_access: Last time memory was accessed
            
        Returns:
            float: Memory strength (0.0-1.0)
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Calculate base age
        age = current_time - experience.timestamp
        
        # Calculate base decay
        base_strength = self._calculate_base_decay(age)
        
        # Apply importance factor
        importance = self._calculate_importance(experience)
        importance_boost = 1.0 + (importance - 0.5) * self.config.importance_factor
        
        # Apply recency boost if recently accessed
        recency_boost = 1.0
        if last_access and access_count > 0:
            recency_age = current_time - last_access
            if recency_age < timedelta(hours=24):
                recency_boost = self.config.recency_boost
        
        # Apply emotional weight
        emotional_boost = 1.0
        if self._is_emotionally_significant(experience):
            emotional_boost = self.config.emotional_weight
        
        # Apply success bias
        success_boost = 1.0
        if experience.success:
            success_boost = self.config.success_bias
        
        # Calculate final strength
        final_strength = (base_strength * 
                         importance_boost * 
                         recency_boost * 
                         emotional_boost * 
                         success_boost)
        
        # Ensure minimum strength and cap at 1.0
        return max(self.config.minimum_strength, min(1.0, final_strength))
    
    def _calculate_base_decay(self, age: timedelta) -> float:
        """Calculate base decay based on age and decay type"""
        age_hours = age.total_seconds() / 3600
        half_life_hours = self.config.half_life.total_seconds() / 3600
        
        if self.config.decay_type == MemoryDecayType.EXPONENTIAL:
            # Exponential decay: strength = e^(-λt) where λ = ln(2)/half_life
            decay_constant = math.log(2) / half_life_hours
            return math.exp(-decay_constant * age_hours)
        
        elif self.config.decay_type == MemoryDecayType.LINEAR:
            # Linear decay: at half_life, strength = 0.5
            return max(0.0, 1.0 - (age_hours / (half_life_hours * 2)))
        
        elif self.config.decay_type == MemoryDecayType.LOGARITHMIC:
            # Logarithmic decay
            if age_hours <= 0:
                return 1.0
            # Adjust to ensure 0.5 at half-life
            scale_factor = half_life_hours * 2
            return max(0.0, 1.0 - math.log(age_hours + 1) / math.log(scale_factor + 1))
        
        elif self.config.decay_type == MemoryDecayType.STEPPED:
            # Stepped decay (discrete levels)
            steps = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
            step_duration = half_life_hours / 3  # Adjusted for more realistic decay
            step_index = min(len(steps) - 1, int(age_hours / step_duration))
            return steps[step_index]
        
        elif self.config.decay_type == MemoryDecayType.FORGETTING_CURVE:
            # Ebbinghaus forgetting curve: R = e^(-t/S)
            # Adjusted to match half-life behavior
            decay_constant = math.log(2) / half_life_hours
            return math.exp(-decay_constant * age_hours)
        
        else:
            # Default to exponential
            decay_constant = math.log(2) / half_life_hours
            return math.exp(-decay_constant * age_hours)
    
    def _calculate_importance(self, experience: Experience) -> float:
        """Calculate importance score for an experience"""
        importance = 0.5  # Base importance
        
        # Factor in confidence score
        importance += (experience.confidence_score - 0.5) * 0.3
        
        # Factor in success/failure
        if experience.success:
            importance += 0.2
        else:
            importance += 0.1  # Failures can be important too
        
        # Factor in lessons learned
        if experience.lessons_learned:
            importance += len(experience.lessons_learned) * 0.1
        
        # Factor in MITRE ATT&CK mapping (indicates tactical significance)
        if experience.mitre_attack_mapping:
            importance += len(experience.mitre_attack_mapping) * 0.05
        
        return max(0.0, min(1.0, importance))
    
    def _is_emotionally_significant(self, experience: Experience) -> bool:
        """Determine if an experience is emotionally significant"""
        # High confidence experiences
        if experience.confidence_score > 0.8:
            return True
        
        # Major failures (low confidence, unsuccessful)
        if not experience.success and experience.confidence_score < 0.3:
            return True
        
        # Experiences with many lessons learned
        if len(experience.lessons_learned) >= 3:
            return True
        
        return False

class PersonalitySystem:
    """
    Manages agent personality vectors and behavioral consistency.
    
    Provides personality-driven behavior modification and adaptation
    based on experiences and environmental factors.
    """
    
    def __init__(self):
        self.personalities: Dict[str, PersonalityVector] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_personality(self, 
                         agent_id: str,
                         operational_style: OperationalStyle = None,
                         trait_overrides: Dict[PersonalityTrait, float] = None) -> PersonalityVector:
        """
        Create a new personality vector for an agent
        
        Args:
            agent_id: Agent identifier
            operational_style: Desired operational style
            trait_overrides: Specific trait values to override
            
        Returns:
            PersonalityVector: Created personality
        """
        # Generate base personality
        personality = PersonalityVector()
        
        # Apply operational style if specified
        if operational_style:
            personality = self._apply_operational_style(personality, operational_style)
        
        # Apply trait overrides
        if trait_overrides:
            for trait, value in trait_overrides.items():
                personality.traits[trait] = max(0.0, min(1.0, value))
        
        # Ensure personality consistency
        personality = self._ensure_consistency(personality)
        
        # Store personality
        self.personalities[agent_id] = personality
        
        self.logger.info(f"Created personality for agent {agent_id}: {operational_style}")
        return personality
    
    def _apply_operational_style(self, 
                               personality: PersonalityVector,
                               style: OperationalStyle) -> PersonalityVector:
        """Apply operational style to personality traits"""
        if style == OperationalStyle.CAUTIOUS:
            personality.traits[PersonalityTrait.CAUTIOUSNESS] = random.uniform(0.7, 0.9)
            personality.traits[PersonalityTrait.RISK_TOLERANCE] = random.uniform(0.1, 0.3)
            personality.traits[PersonalityTrait.AGGRESSIVENESS] = random.uniform(0.1, 0.4)
            personality.traits[PersonalityTrait.STEALTH] = random.uniform(0.6, 0.8)
        
        elif style == OperationalStyle.AGGRESSIVE:
            personality.traits[PersonalityTrait.AGGRESSIVENESS] = random.uniform(0.7, 0.9)
            personality.traits[PersonalityTrait.RISK_TOLERANCE] = random.uniform(0.6, 0.9)
            personality.traits[PersonalityTrait.CAUTIOUSNESS] = random.uniform(0.1, 0.4)
            personality.traits[PersonalityTrait.PERSISTENCE] = random.uniform(0.6, 0.8)
        
        elif style == OperationalStyle.STEALTHY:
            personality.traits[PersonalityTrait.STEALTH] = random.uniform(0.8, 0.95)
            personality.traits[PersonalityTrait.CAUTIOUSNESS] = random.uniform(0.6, 0.8)
            personality.traits[PersonalityTrait.AGGRESSIVENESS] = random.uniform(0.1, 0.3)
            personality.traits[PersonalityTrait.ADAPTABILITY] = random.uniform(0.7, 0.9)
        
        elif style == OperationalStyle.OPPORTUNISTIC:
            personality.traits[PersonalityTrait.ADAPTABILITY] = random.uniform(0.7, 0.9)
            personality.traits[PersonalityTrait.CURIOSITY] = random.uniform(0.6, 0.8)
            personality.traits[PersonalityTrait.RISK_TOLERANCE] = random.uniform(0.5, 0.7)
            personality.traits[PersonalityTrait.AGGRESSIVENESS] = random.uniform(0.4, 0.7)
        
        elif style == OperationalStyle.METHODICAL:
            personality.traits[PersonalityTrait.PERSISTENCE] = random.uniform(0.7, 0.9)
            personality.traits[PersonalityTrait.CAUTIOUSNESS] = random.uniform(0.6, 0.8)
            personality.traits[PersonalityTrait.ADAPTABILITY] = random.uniform(0.3, 0.5)
            personality.traits[PersonalityTrait.COLLABORATION] = random.uniform(0.6, 0.8)
        
        # BALANCED style keeps random values
        
        personality.operational_style = style
        return personality
    
    def _ensure_consistency(self, personality: PersonalityVector) -> PersonalityVector:
        """Ensure personality traits are internally consistent"""
        # Cautiousness and aggressiveness should be somewhat inverse
        if (personality.traits[PersonalityTrait.CAUTIOUSNESS] > 0.7 and 
            personality.traits[PersonalityTrait.AGGRESSIVENESS] > 0.7):
            # Reduce the lower one
            if personality.traits[PersonalityTrait.CAUTIOUSNESS] > personality.traits[PersonalityTrait.AGGRESSIVENESS]:
                personality.traits[PersonalityTrait.AGGRESSIVENESS] *= 0.6
            else:
                personality.traits[PersonalityTrait.CAUTIOUSNESS] *= 0.6
        
        # Risk tolerance and cautiousness should be somewhat inverse
        if (personality.traits[PersonalityTrait.RISK_TOLERANCE] > 0.7 and 
            personality.traits[PersonalityTrait.CAUTIOUSNESS] > 0.7):
            personality.traits[PersonalityTrait.RISK_TOLERANCE] *= 0.7
        
        # Stealth and aggressiveness should be somewhat inverse
        if (personality.traits[PersonalityTrait.STEALTH] > 0.7 and 
            personality.traits[PersonalityTrait.AGGRESSIVENESS] > 0.7):
            personality.traits[PersonalityTrait.AGGRESSIVENESS] *= 0.7
        
        return personality
    
    def get_personality(self, agent_id: str) -> Optional[PersonalityVector]:
        """Get personality vector for an agent"""
        return self.personalities.get(agent_id)
    
    def update_personality(self, 
                         agent_id: str,
                         experience: Experience,
                         adaptation_factor: float = 1.0) -> None:
        """
        Update agent personality based on experience
        
        Args:
            agent_id: Agent identifier
            experience: Recent experience
            adaptation_factor: How much to adapt (0.0-1.0)
        """
        personality = self.personalities.get(agent_id)
        if not personality:
            return
        
        # Calculate adaptation amount
        base_adaptation = personality.adaptation_rate * adaptation_factor
        
        # Adapt based on experience outcome
        if experience.success:
            # Successful experiences reinforce current traits slightly
            if hasattr(experience.action_taken, 'action_type') and experience.action_taken.action_type:
                action_type = str(experience.action_taken.action_type)
                
                # Increase traits that led to success
                if 'stealth' in action_type.lower():
                    personality.traits[PersonalityTrait.STEALTH] += base_adaptation * 0.1
                elif 'aggressive' in action_type.lower():
                    personality.traits[PersonalityTrait.AGGRESSIVENESS] += base_adaptation * 0.1
                
                # Increase confidence-related traits
                personality.traits[PersonalityTrait.PERSISTENCE] += base_adaptation * 0.05
        else:
            # Failed experiences may cause trait adjustment
            if experience.confidence_score < 0.3:
                # Major failure - increase cautiousness
                personality.traits[PersonalityTrait.CAUTIOUSNESS] += base_adaptation * 0.15
                personality.traits[PersonalityTrait.RISK_TOLERANCE] -= base_adaptation * 0.1
            
            # Increase adaptability after failures
            personality.traits[PersonalityTrait.ADAPTABILITY] += base_adaptation * 0.08
        
        # Ensure traits stay in valid range
        for trait in personality.traits:
            personality.traits[trait] = max(0.0, min(1.0, personality.traits[trait]))
        
        # Update timestamp
        personality.last_updated = datetime.now()
        
        self.logger.debug(f"Updated personality for agent {agent_id} based on experience")
    
    def get_behavior_modifiers(self, agent_id: str) -> Dict[str, float]:
        """
        Get behavior modifiers based on personality
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dict of behavior modifiers
        """
        personality = self.personalities.get(agent_id)
        if not personality:
            return {}
        
        return {
            'decision_speed': 1.0 - personality.traits[PersonalityTrait.CAUTIOUSNESS] * 0.5,
            'risk_acceptance': personality.traits[PersonalityTrait.RISK_TOLERANCE],
            'stealth_preference': personality.traits[PersonalityTrait.STEALTH],
            'aggression_level': personality.traits[PersonalityTrait.AGGRESSIVENESS],
            'exploration_tendency': personality.traits[PersonalityTrait.CURIOSITY],
            'persistence_factor': personality.traits[PersonalityTrait.PERSISTENCE],
            'adaptation_speed': personality.traits[PersonalityTrait.ADAPTABILITY],
            'collaboration_preference': personality.traits[PersonalityTrait.COLLABORATION]
        }

class BehaviorFuzzingSystem:
    """
    Introduces controlled unpredictability to agent behavior.
    
    Simulates realistic cognitive variations and prevents overly
    deterministic behavior patterns.
    """
    
    def __init__(self, config: BehaviorFuzzingConfig = None):
        self.config = config or BehaviorFuzzingConfig()
        self.logger = logging.getLogger(__name__)
    
    def apply_decision_fuzzing(self, 
                             agent_id: str,
                             decision_scores: Dict[str, float],
                             cognitive_state: CognitiveState,
                             personality: PersonalityVector = None) -> Dict[str, float]:
        """
        Apply fuzzing to decision scores to introduce unpredictability
        
        Args:
            agent_id: Agent identifier
            decision_scores: Original decision scores
            cognitive_state: Current cognitive state
            personality: Agent personality (optional)
            
        Returns:
            Dict of fuzzed decision scores
        """
        if not self.config.enabled:
            return decision_scores
        
        # Calculate fuzzing intensity
        fuzzing_intensity = self._calculate_fuzzing_intensity(cognitive_state, personality)
        
        # Apply fuzzing to each decision
        fuzzed_scores = {}
        for decision, score in decision_scores.items():
            # Add random noise
            noise = random.gauss(0, self.config.decision_noise * fuzzing_intensity)
            fuzzed_score = score + noise
            
            # Apply personality-based variance
            if personality:
                personality_variance = self._get_personality_variance(personality, decision)
                fuzzed_score += personality_variance * fuzzing_intensity
            
            # Ensure score stays in valid range
            fuzzed_scores[decision] = max(0.0, min(1.0, fuzzed_score))
        
        self.logger.debug(f"Applied decision fuzzing to agent {agent_id} with intensity {fuzzing_intensity:.3f}")
        return fuzzed_scores
    
    def apply_action_fuzzing(self, 
                           agent_id: str,
                           action_parameters: Dict[str, Any],
                           cognitive_state: CognitiveState,
                           personality: PersonalityVector = None) -> Dict[str, Any]:
        """
        Apply fuzzing to action parameters
        
        Args:
            agent_id: Agent identifier
            action_parameters: Original action parameters
            cognitive_state: Current cognitive state
            personality: Agent personality (optional)
            
        Returns:
            Dict of fuzzed action parameters
        """
        if not self.config.enabled:
            return action_parameters
        
        # Calculate fuzzing intensity
        fuzzing_intensity = self._calculate_fuzzing_intensity(cognitive_state, personality)
        
        # Apply fuzzing to numeric parameters
        fuzzed_parameters = action_parameters.copy()
        
        for key, value in action_parameters.items():
            if isinstance(value, (int, float)):
                # Apply variance to numeric values
                variance = abs(value) * self.config.action_variance * fuzzing_intensity
                noise = random.gauss(0, variance)
                fuzzed_parameters[key] = value + noise
            
            elif isinstance(value, str) and key in ['timeout', 'delay', 'duration']:
                # Apply fuzzing to time-related string parameters
                try:
                    numeric_value = float(value)
                    variance = numeric_value * self.config.action_variance * fuzzing_intensity
                    noise = random.gauss(0, variance)
                    fuzzed_parameters[key] = str(max(0.1, numeric_value + noise))
                except ValueError:
                    pass  # Keep original if not numeric
        
        return fuzzed_parameters
    
    def _calculate_fuzzing_intensity(self, 
                                   cognitive_state: CognitiveState,
                                   personality: PersonalityVector = None) -> float:
        """Calculate overall fuzzing intensity based on state and personality"""
        # Base randomness
        intensity = self.config.base_randomness
        
        # Stress amplification
        intensity += cognitive_state.stress_level * self.config.stress_amplifier * 0.1
        
        # Fatigue amplification
        intensity += cognitive_state.fatigue * self.config.fatigue_amplifier * 0.1
        
        # Attention span reduction increases randomness
        intensity += (1.0 - cognitive_state.attention_span) * 0.2
        
        # Personality influence
        if personality:
            # Less consistent personalities have more randomness
            consistency_factor = 1.0 - personality.consistency_factor
            intensity += consistency_factor * self.config.personality_influence
            
            # High adaptability increases randomness
            adaptability_factor = personality.traits.get(PersonalityTrait.ADAPTABILITY, 0.5)
            intensity += adaptability_factor * 0.1
        
        # Ensure intensity stays in reasonable range
        return max(0.0, min(1.0, intensity))
    
    def _get_personality_variance(self, 
                                personality: PersonalityVector,
                                decision: str) -> float:
        """Get personality-based variance for a specific decision"""
        # Map decision types to personality traits
        trait_mapping = {
            'attack': PersonalityTrait.AGGRESSIVENESS,
            'defend': PersonalityTrait.CAUTIOUSNESS,
            'stealth': PersonalityTrait.STEALTH,
            'explore': PersonalityTrait.CURIOSITY,
            'persist': PersonalityTrait.PERSISTENCE,
            'adapt': PersonalityTrait.ADAPTABILITY,
            'collaborate': PersonalityTrait.COLLABORATION
        }
        
        # Find relevant trait
        relevant_trait = None
        for decision_type, trait in trait_mapping.items():
            if decision_type in decision.lower():
                relevant_trait = trait
                break
        
        if relevant_trait:
            trait_value = personality.traits.get(relevant_trait, 0.5)
            # Higher trait values reduce variance (more consistent behavior)
            return (1.0 - trait_value) * 0.1
        
        return 0.0

class CognitiveModelingSystem:
    """
    Main system that integrates memory decay, personality, and behavior fuzzing.
    
    Provides a comprehensive cognitive model for agents with realistic
    limitations and behavioral diversity.
    """
    
    def __init__(self, 
                 memory_decay_config: MemoryDecayConfig = None,
                 fuzzing_config: BehaviorFuzzingConfig = None):
        
        self.memory_decay = MemoryDecayModel(memory_decay_config)
        self.personality_system = PersonalitySystem()
        self.behavior_fuzzing = BehaviorFuzzingSystem(fuzzing_config)
        
        # Cognitive states for agents
        self.cognitive_states: Dict[str, CognitiveState] = {}
        
        # Memory access tracking
        self.memory_access_counts: Dict[str, int] = {}
        self.memory_last_access: Dict[str, datetime] = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize_agent_cognition(self, 
                                       agent_id: str,
                                       config: AgentConfig,
                                       operational_style: OperationalStyle = None) -> None:
        """
        Initialize cognitive modeling for an agent
        
        Args:
            agent_id: Agent identifier
            config: Agent configuration
            operational_style: Desired operational style
        """
        # Create personality
        style = operational_style or self._infer_style_from_role(config.role)
        personality = self.personality_system.create_personality(agent_id, style)
        
        # Initialize cognitive state
        cognitive_state = CognitiveState(
            attention_span=random.uniform(0.7, 1.0),
            stress_level=random.uniform(0.0, 0.2),
            confidence=random.uniform(0.4, 0.6),
            fatigue=random.uniform(0.0, 0.1)
        )
        self.cognitive_states[agent_id] = cognitive_state
        
        self.logger.info(f"Initialized cognitive modeling for agent {agent_id}")
    
    def _infer_style_from_role(self, role: Role) -> OperationalStyle:
        """Infer operational style from agent role"""
        role_style_mapping = {
            Role.RECON: OperationalStyle.STEALTHY,
            Role.EXPLOIT: OperationalStyle.AGGRESSIVE,
            Role.PERSISTENCE: OperationalStyle.METHODICAL,
            Role.EXFILTRATION: OperationalStyle.CAUTIOUS,
            Role.SOC_ANALYST: OperationalStyle.METHODICAL,
            Role.FIREWALL_CONFIG: OperationalStyle.CAUTIOUS,
            Role.SIEM_INTEGRATOR: OperationalStyle.BALANCED,
            Role.COMPLIANCE_AUDITOR: OperationalStyle.METHODICAL,
            Role.INCIDENT_RESPONSE: OperationalStyle.AGGRESSIVE,
            Role.THREAT_HUNTER: OperationalStyle.OPPORTUNISTIC
        }
        
        return role_style_mapping.get(role, OperationalStyle.BALANCED)
    
    def get_memory_strength(self, 
                          agent_id: str,
                          experience: Experience) -> float:
        """
        Get current memory strength for an experience
        
        Args:
            agent_id: Agent identifier
            experience: Experience to check
            
        Returns:
            float: Current memory strength (0.0-1.0)
        """
        access_count = self.memory_access_counts.get(experience.experience_id, 0)
        last_access = self.memory_last_access.get(experience.experience_id)
        
        return self.memory_decay.calculate_memory_strength(
            experience, 
            access_count=access_count,
            last_access=last_access
        )
    
    def access_memory(self, 
                     agent_id: str,
                     experience_id: str) -> None:
        """
        Record memory access for decay calculation
        
        Args:
            agent_id: Agent identifier
            experience_id: Experience identifier
        """
        self.memory_access_counts[experience_id] = self.memory_access_counts.get(experience_id, 0) + 1
        self.memory_last_access[experience_id] = datetime.now()
    
    def update_cognitive_state(self, 
                             agent_id: str,
                             stress_delta: float = 0.0,
                             fatigue_delta: float = 0.0,
                             confidence_delta: float = 0.0) -> None:
        """
        Update agent cognitive state
        
        Args:
            agent_id: Agent identifier
            stress_delta: Change in stress level
            fatigue_delta: Change in fatigue level
            confidence_delta: Change in confidence level
        """
        state = self.cognitive_states.get(agent_id)
        if not state:
            return
        
        # Apply deltas
        state.stress_level = max(0.0, min(1.0, state.stress_level + stress_delta))
        state.fatigue = max(0.0, min(1.0, state.fatigue + fatigue_delta))
        state.confidence = max(0.0, min(1.0, state.confidence + confidence_delta))
        
        # Attention span is affected by stress and fatigue
        base_attention = 1.0 - (state.stress_level * 0.3) - (state.fatigue * 0.4)
        state.attention_span = max(0.1, min(1.0, base_attention))
        
        state.last_updated = datetime.now()
    
    def apply_cognitive_effects(self, 
                              agent_id: str,
                              decision_scores: Dict[str, float],
                              action_parameters: Dict[str, Any] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Apply all cognitive effects to agent decisions and actions
        
        Args:
            agent_id: Agent identifier
            decision_scores: Original decision scores
            action_parameters: Original action parameters
            
        Returns:
            Tuple of (fuzzed_decisions, fuzzed_parameters)
        """
        cognitive_state = self.cognitive_states.get(agent_id)
        personality = self.personality_system.get_personality(agent_id)
        
        if not cognitive_state:
            return decision_scores, action_parameters or {}
        
        # Apply decision fuzzing
        fuzzed_decisions = self.behavior_fuzzing.apply_decision_fuzzing(
            agent_id, decision_scores, cognitive_state, personality
        )
        
        # Apply action fuzzing
        fuzzed_parameters = action_parameters or {}
        if action_parameters:
            fuzzed_parameters = self.behavior_fuzzing.apply_action_fuzzing(
                agent_id, action_parameters, cognitive_state, personality
            )
        
        return fuzzed_decisions, fuzzed_parameters
    
    def learn_from_experience(self, 
                            agent_id: str,
                            experience: Experience) -> None:
        """
        Update cognitive model based on experience
        
        Args:
            agent_id: Agent identifier
            experience: Recent experience
        """
        # Update personality
        self.personality_system.update_personality(agent_id, experience)
        
        # Update cognitive state based on experience
        if experience.success:
            self.update_cognitive_state(agent_id, confidence_delta=0.05, stress_delta=-0.02)
        else:
            self.update_cognitive_state(agent_id, confidence_delta=-0.03, stress_delta=0.05)
        
        # Add some fatigue from any experience
        self.update_cognitive_state(agent_id, fatigue_delta=0.01)
    
    def get_agent_profile(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive cognitive profile for an agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dict containing cognitive profile
        """
        personality = self.personality_system.get_personality(agent_id)
        cognitive_state = self.cognitive_states.get(agent_id)
        behavior_modifiers = self.personality_system.get_behavior_modifiers(agent_id)
        
        profile = {
            "agent_id": agent_id,
            "personality": {
                "operational_style": personality.operational_style.value if personality else "unknown",
                "traits": {trait.value: score for trait, score in personality.traits.items()} if personality else {},
                "consistency_factor": personality.consistency_factor if personality else 0.5,
                "adaptation_rate": personality.adaptation_rate if personality else 0.1
            },
            "cognitive_state": {
                "attention_span": cognitive_state.attention_span if cognitive_state else 1.0,
                "stress_level": cognitive_state.stress_level if cognitive_state else 0.0,
                "confidence": cognitive_state.confidence if cognitive_state else 0.5,
                "fatigue": cognitive_state.fatigue if cognitive_state else 0.0,
                "focus_target": cognitive_state.focus_target if cognitive_state else None
            },
            "behavior_modifiers": behavior_modifiers,
            "memory_decay_config": {
                "decay_type": self.memory_decay.config.decay_type.value,
                "half_life_days": self.memory_decay.config.half_life.days,
                "minimum_strength": self.memory_decay.config.minimum_strength
            }
        }
        
        return profile
    
    def reset_cognitive_state(self, agent_id: str) -> None:
        """Reset cognitive state for an agent (e.g., after rest period)"""
        state = self.cognitive_states.get(agent_id)
        if state:
            state.stress_level = max(0.0, state.stress_level * 0.5)
            state.fatigue = max(0.0, state.fatigue * 0.3)
            state.attention_span = min(1.0, state.attention_span + 0.2)
            state.last_updated = datetime.now()
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide cognitive modeling statistics"""
        total_agents = len(self.cognitive_states)
        
        if total_agents == 0:
            return {"total_agents": 0}
        
        # Calculate averages
        avg_stress = sum(state.stress_level for state in self.cognitive_states.values()) / total_agents
        avg_fatigue = sum(state.fatigue for state in self.cognitive_states.values()) / total_agents
        avg_confidence = sum(state.confidence for state in self.cognitive_states.values()) / total_agents
        avg_attention = sum(state.attention_span for state in self.cognitive_states.values()) / total_agents
        
        # Personality distribution
        style_counts = {}
        for personality in self.personality_system.personalities.values():
            style = personality.operational_style.value
            style_counts[style] = style_counts.get(style, 0) + 1
        
        return {
            "total_agents": total_agents,
            "average_metrics": {
                "stress_level": avg_stress,
                "fatigue": avg_fatigue,
                "confidence": avg_confidence,
                "attention_span": avg_attention
            },
            "personality_distribution": style_counts,
            "memory_accesses": len(self.memory_access_counts),
            "fuzzing_enabled": self.behavior_fuzzing.config.enabled
        }

# Factory functions
def create_cognitive_modeling_system(
    memory_decay_config: MemoryDecayConfig = None,
    fuzzing_config: BehaviorFuzzingConfig = None
) -> CognitiveModelingSystem:
    """Create a cognitive modeling system instance"""
    return CognitiveModelingSystem(memory_decay_config, fuzzing_config)

def create_personality_for_role(role: Role) -> Tuple[OperationalStyle, Dict[PersonalityTrait, float]]:
    """Create appropriate personality configuration for a role"""
    if role in [Role.RECON, Role.EXFILTRATION]:
        return OperationalStyle.STEALTHY, {
            PersonalityTrait.STEALTH: random.uniform(0.7, 0.9),
            PersonalityTrait.CAUTIOUSNESS: random.uniform(0.6, 0.8),
            PersonalityTrait.CURIOSITY: random.uniform(0.7, 0.9)
        }
    elif role in [Role.EXPLOIT, Role.INCIDENT_RESPONSE]:
        return OperationalStyle.AGGRESSIVE, {
            PersonalityTrait.AGGRESSIVENESS: random.uniform(0.7, 0.9),
            PersonalityTrait.RISK_TOLERANCE: random.uniform(0.6, 0.8),
            PersonalityTrait.PERSISTENCE: random.uniform(0.6, 0.8)
        }
    elif role in [Role.PERSISTENCE, Role.COMPLIANCE_AUDITOR]:
        return OperationalStyle.METHODICAL, {
            PersonalityTrait.PERSISTENCE: random.uniform(0.8, 0.95),
            PersonalityTrait.CAUTIOUSNESS: random.uniform(0.6, 0.8),
            PersonalityTrait.COLLABORATION: random.uniform(0.6, 0.8)
        }
    else:
        return OperationalStyle.BALANCED, {}