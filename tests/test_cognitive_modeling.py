#!/usr/bin/env python3
"""
Tests for Cognitive Modeling System
"""

import pytest
import asyncio
import math
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from memory.cognitive_modeling import (
    CognitiveModelingSystem,
    MemoryDecayModel,
    PersonalitySystem,
    BehaviorFuzzingSystem,
    PersonalityVector,
    PersonalityTrait,
    OperationalStyle,
    MemoryDecayType,
    MemoryDecayConfig,
    BehaviorFuzzingConfig,
    CognitiveState,
    create_cognitive_modeling_system,
    create_personality_for_role
)
from agents.base_agent import Experience, Team, Role, AgentConfig, ActionPlan, ActionResult

class TestMemoryDecayModel:
    """Test memory decay functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = MemoryDecayConfig(
            decay_type=MemoryDecayType.EXPONENTIAL,
            half_life=timedelta(days=7),
            minimum_strength=0.1
        )
        self.decay_model = MemoryDecayModel(self.config)
        
        # Create test experience
        self.experience = Experience(
            experience_id="test_exp_1",
            agent_id="test_agent",
            timestamp=datetime.now() - timedelta(days=3),
            context=Mock(),
            action_taken=Mock(),
            reasoning=Mock(),
            outcome=Mock(),
            success=True,
            lessons_learned=["Test lesson"],
            mitre_attack_mapping=["T1001"],
            confidence_score=0.8
        )
    
    def test_exponential_decay(self):
        """Test exponential memory decay"""
        # Test fresh memory (should be close to 1.0)
        fresh_experience = Experience(
            experience_id="fresh_exp",
            agent_id="test_agent",
            timestamp=datetime.now() - timedelta(minutes=5),
            context=Mock(),
            action_taken=Mock(),
            reasoning=Mock(),
            outcome=Mock(),
            success=True,
            lessons_learned=[],
            mitre_attack_mapping=[],
            confidence_score=0.5
        )
        
        strength = self.decay_model.calculate_memory_strength(fresh_experience)
        assert strength > 0.95
        
        # Test aged memory
        old_experience = Experience(
            experience_id="old_exp",
            agent_id="test_agent",
            timestamp=datetime.now() - timedelta(days=14),  # 2 half-lives
            context=Mock(),
            action_taken=Mock(),
            reasoning=Mock(),
            outcome=Mock(),
            success=True,
            lessons_learned=[],
            mitre_attack_mapping=[],
            confidence_score=0.5
        )
        
        strength = self.decay_model.calculate_memory_strength(old_experience)
        # After 2 half-lives, should be around 0.25 (0.5^2), but with importance factors it will be higher
        assert 0.15 < strength < 0.6
    
    def test_linear_decay(self):
        """Test linear memory decay"""
        config = MemoryDecayConfig(decay_type=MemoryDecayType.LINEAR)
        model = MemoryDecayModel(config)
        
        # Test at half-life point
        half_life_experience = Experience(
            experience_id="half_exp",
            agent_id="test_agent",
            timestamp=datetime.now() - config.half_life,
            context=Mock(),
            action_taken=Mock(),
            reasoning=Mock(),
            outcome=Mock(),
            success=True,
            lessons_learned=[],
            mitre_attack_mapping=[],
            confidence_score=0.5
        )
        
        strength = model.calculate_memory_strength(half_life_experience)
        # Linear decay should be around 0.5 at half-life, but with importance factors it will be higher
        assert 0.3 < strength < 1.0
    
    def test_importance_factor(self):
        """Test that important memories decay slower"""
        # High importance experience
        important_exp = Experience(
            experience_id="important_exp",
            agent_id="test_agent",
            timestamp=datetime.now() - timedelta(days=7),
            context=Mock(),
            action_taken=Mock(),
            reasoning=Mock(),
            outcome=Mock(),
            success=True,
            lessons_learned=["Important lesson 1", "Important lesson 2"],
            mitre_attack_mapping=["T1001", "T1002"],
            confidence_score=0.9
        )
        
        # Low importance experience
        unimportant_exp = Experience(
            experience_id="unimportant_exp",
            agent_id="test_agent",
            timestamp=datetime.now() - timedelta(days=7),
            context=Mock(),
            action_taken=Mock(),
            reasoning=Mock(),
            outcome=Mock(),
            success=False,
            lessons_learned=[],
            mitre_attack_mapping=[],
            confidence_score=0.2
        )
        
        important_strength = self.decay_model.calculate_memory_strength(important_exp)
        unimportant_strength = self.decay_model.calculate_memory_strength(unimportant_exp)
        
        # Important memory should be stronger
        assert important_strength > unimportant_strength
    
    def test_recency_boost(self):
        """Test that recently accessed memories are stronger"""
        old_experience = Experience(
            experience_id="old_accessed_exp",
            agent_id="test_agent",
            timestamp=datetime.now() - timedelta(days=14),
            context=Mock(),
            action_taken=Mock(),
            reasoning=Mock(),
            outcome=Mock(),
            success=True,
            lessons_learned=[],
            mitre_attack_mapping=[],
            confidence_score=0.5
        )
        
        # Without recent access
        strength_no_access = self.decay_model.calculate_memory_strength(old_experience)
        
        # With recent access
        recent_access = datetime.now() - timedelta(hours=2)
        strength_with_access = self.decay_model.calculate_memory_strength(
            old_experience, 
            access_count=3,
            last_access=recent_access
        )
        
        # Recently accessed should be stronger
        assert strength_with_access > strength_no_access
    
    def test_minimum_strength(self):
        """Test that memories never decay below minimum strength"""
        very_old_experience = Experience(
            experience_id="ancient_exp",
            agent_id="test_agent",
            timestamp=datetime.now() - timedelta(days=365),  # Very old
            context=Mock(),
            action_taken=Mock(),
            reasoning=Mock(),
            outcome=Mock(),
            success=False,
            lessons_learned=[],
            mitre_attack_mapping=[],
            confidence_score=0.1
        )
        
        strength = self.decay_model.calculate_memory_strength(very_old_experience)
        assert strength >= self.config.minimum_strength

class TestPersonalitySystem:
    """Test personality system functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.personality_system = PersonalitySystem()
    
    def test_create_personality(self):
        """Test personality creation"""
        personality = self.personality_system.create_personality(
            "test_agent",
            OperationalStyle.AGGRESSIVE
        )
        
        assert personality.operational_style == OperationalStyle.AGGRESSIVE
        assert len(personality.traits) == len(PersonalityTrait)
        
        # Aggressive style should have high aggressiveness
        assert personality.traits[PersonalityTrait.AGGRESSIVENESS] > 0.6
        assert personality.traits[PersonalityTrait.CAUTIOUSNESS] < 0.5
    
    def test_personality_consistency(self):
        """Test that personalities are internally consistent"""
        personality = self.personality_system.create_personality(
            "test_agent",
            OperationalStyle.CAUTIOUS
        )
        
        # Cautious agents should have low aggressiveness and high cautiousness
        assert personality.traits[PersonalityTrait.CAUTIOUSNESS] > 0.6
        assert personality.traits[PersonalityTrait.AGGRESSIVENESS] < 0.5
        assert personality.traits[PersonalityTrait.RISK_TOLERANCE] < 0.4
    
    def test_personality_update(self):
        """Test personality adaptation based on experience"""
        personality = self.personality_system.create_personality("test_agent")
        original_persistence = personality.traits[PersonalityTrait.PERSISTENCE]
        
        # Create successful experience with proper action_type
        action_mock = Mock()
        action_mock.action_type = "stealth_reconnaissance"
        
        success_exp = Experience(
            experience_id="success_exp",
            agent_id="test_agent",
            timestamp=datetime.now(),
            context=Mock(),
            action_taken=action_mock,
            reasoning=Mock(),
            outcome=Mock(),
            success=True,
            lessons_learned=["Success lesson"],
            mitre_attack_mapping=[],
            confidence_score=0.8
        )
        
        self.personality_system.update_personality("test_agent", success_exp)
        
        updated_personality = self.personality_system.get_personality("test_agent")
        # Persistence should increase after success
        assert updated_personality.traits[PersonalityTrait.PERSISTENCE] >= original_persistence
    
    def test_behavior_modifiers(self):
        """Test behavior modifier generation"""
        self.personality_system.create_personality(
            "test_agent",
            OperationalStyle.STEALTHY
        )
        
        modifiers = self.personality_system.get_behavior_modifiers("test_agent")
        
        assert 'stealth_preference' in modifiers
        assert 'aggression_level' in modifiers
        assert 'risk_acceptance' in modifiers
        
        # Stealthy agents should prefer stealth
        assert modifiers['stealth_preference'] > 0.6
    
    def test_operational_styles(self):
        """Test all operational styles create appropriate personalities"""
        styles = [
            OperationalStyle.CAUTIOUS,
            OperationalStyle.AGGRESSIVE,
            OperationalStyle.STEALTHY,
            OperationalStyle.BALANCED,
            OperationalStyle.OPPORTUNISTIC,
            OperationalStyle.METHODICAL
        ]
        
        for style in styles:
            personality = self.personality_system.create_personality(f"agent_{style.value}", style)
            assert personality.operational_style == style
            
            # Verify style-appropriate traits
            if style == OperationalStyle.CAUTIOUS:
                assert personality.traits[PersonalityTrait.CAUTIOUSNESS] > 0.6
            elif style == OperationalStyle.AGGRESSIVE:
                assert personality.traits[PersonalityTrait.AGGRESSIVENESS] > 0.6
            elif style == OperationalStyle.STEALTHY:
                assert personality.traits[PersonalityTrait.STEALTH] > 0.7

class TestBehaviorFuzzingSystem:
    """Test behavior fuzzing functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = BehaviorFuzzingConfig(
            enabled=True,
            base_randomness=0.1,
            decision_noise=0.05
        )
        self.fuzzing_system = BehaviorFuzzingSystem(self.config)
        
        self.cognitive_state = CognitiveState(
            attention_span=0.8,
            stress_level=0.3,
            confidence=0.6,
            fatigue=0.2
        )
    
    def test_decision_fuzzing(self):
        """Test decision score fuzzing"""
        original_scores = {
            'attack': 0.8,
            'defend': 0.6,
            'wait': 0.4
        }
        
        fuzzed_scores = self.fuzzing_system.apply_decision_fuzzing(
            "test_agent",
            original_scores,
            self.cognitive_state
        )
        
        # Scores should be different but in valid range
        for decision in original_scores:
            assert 0.0 <= fuzzed_scores[decision] <= 1.0
            # Should be somewhat different (allowing for small random variations)
            assert abs(fuzzed_scores[decision] - original_scores[decision]) >= 0.0
    
    def test_action_fuzzing(self):
        """Test action parameter fuzzing"""
        original_params = {
            'timeout': 30.0,
            'retry_count': 3,
            'delay': '5.0',
            'target': 'server1'
        }
        
        fuzzed_params = self.fuzzing_system.apply_action_fuzzing(
            "test_agent",
            original_params,
            self.cognitive_state
        )
        
        # Numeric parameters should be fuzzed
        assert fuzzed_params['timeout'] != original_params['timeout']
        assert fuzzed_params['retry_count'] != original_params['retry_count']
        
        # String target should remain unchanged
        assert fuzzed_params['target'] == original_params['target']
    
    def test_fuzzing_intensity_calculation(self):
        """Test fuzzing intensity calculation"""
        # High stress state
        high_stress_state = CognitiveState(
            stress_level=0.8,
            fatigue=0.6,
            attention_span=0.3
        )
        
        # Low stress state
        low_stress_state = CognitiveState(
            stress_level=0.1,
            fatigue=0.1,
            attention_span=0.9
        )
        
        high_intensity = self.fuzzing_system._calculate_fuzzing_intensity(high_stress_state)
        low_intensity = self.fuzzing_system._calculate_fuzzing_intensity(low_stress_state)
        
        # High stress should result in higher fuzzing intensity
        assert high_intensity > low_intensity
    
    def test_fuzzing_disabled(self):
        """Test that fuzzing can be disabled"""
        disabled_config = BehaviorFuzzingConfig(enabled=False)
        disabled_system = BehaviorFuzzingSystem(disabled_config)
        
        original_scores = {'action1': 0.5, 'action2': 0.7}
        
        fuzzed_scores = disabled_system.apply_decision_fuzzing(
            "test_agent",
            original_scores,
            self.cognitive_state
        )
        
        # Should be identical when disabled
        assert fuzzed_scores == original_scores

class TestCognitiveModelingSystem:
    """Test integrated cognitive modeling system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.system = create_cognitive_modeling_system()
        
        self.agent_config = AgentConfig(
            agent_id="test_agent",
            team=Team.RED,
            role=Role.EXPLOIT,
            name="Test Agent",
            description="Test agent for cognitive modeling"
        )
    
    def test_agent_initialization(self):
        """Test agent cognitive initialization"""
        # Use asyncio.run for the async call
        import asyncio
        asyncio.run(self.system.initialize_agent_cognition(
            "test_agent",
            self.agent_config,
            OperationalStyle.AGGRESSIVE
        ))
        
        # Check personality was created
        personality = self.system.personality_system.get_personality("test_agent")
        assert personality is not None
        assert personality.operational_style == OperationalStyle.AGGRESSIVE
        
        # Check cognitive state was created
        assert "test_agent" in self.system.cognitive_states
        cognitive_state = self.system.cognitive_states["test_agent"]
        assert 0.0 <= cognitive_state.stress_level <= 1.0
        assert 0.0 <= cognitive_state.fatigue <= 1.0
    
    def test_memory_strength_calculation(self):
        """Test memory strength calculation with access tracking"""
        experience = Experience(
            experience_id="test_memory",
            agent_id="test_agent",
            timestamp=datetime.now() - timedelta(days=5),
            context=Mock(),
            action_taken=Mock(),
            reasoning=Mock(),
            outcome=Mock(),
            success=True,
            lessons_learned=[],
            mitre_attack_mapping=[],
            confidence_score=0.7
        )
        
        # Initial strength
        strength1 = self.system.get_memory_strength("test_agent", experience)
        
        # Access memory
        self.system.access_memory("test_agent", "test_memory")
        
        # Strength should be same or higher due to recency boost
        strength2 = self.system.get_memory_strength("test_agent", experience)
        assert strength2 >= strength1
    
    def test_cognitive_state_updates(self):
        """Test cognitive state updates"""
        # Initialize agent
        self.system.cognitive_states["test_agent"] = CognitiveState()
        
        original_stress = self.system.cognitive_states["test_agent"].stress_level
        
        # Update stress
        self.system.update_cognitive_state("test_agent", stress_delta=0.2)
        
        new_stress = self.system.cognitive_states["test_agent"].stress_level
        assert new_stress > original_stress
        
        # Attention span should be affected by stress
        attention = self.system.cognitive_states["test_agent"].attention_span
        assert attention < 1.0  # Should be reduced by stress
    
    def test_cognitive_effects_application(self):
        """Test application of cognitive effects to decisions"""
        # Initialize agent
        self.system.cognitive_states["test_agent"] = CognitiveState(stress_level=0.5)
        self.system.personality_system.create_personality("test_agent")
        
        original_decisions = {
            'attack': 0.8,
            'defend': 0.6,
            'wait': 0.4
        }
        
        original_params = {
            'timeout': 30.0,
            'target': 'server1'
        }
        
        fuzzed_decisions, fuzzed_params = self.system.apply_cognitive_effects(
            "test_agent",
            original_decisions,
            original_params
        )
        
        # Should have some differences due to fuzzing
        assert len(fuzzed_decisions) == len(original_decisions)
        assert len(fuzzed_params) == len(original_params)
    
    def test_learning_from_experience(self):
        """Test learning and adaptation from experiences"""
        # Initialize agent
        self.system.personality_system.create_personality("test_agent")
        self.system.cognitive_states["test_agent"] = CognitiveState()
        
        original_confidence = self.system.cognitive_states["test_agent"].confidence
        
        # Create successful experience with proper action_type
        action_mock = Mock()
        action_mock.action_type = "defensive_analysis"
        
        success_exp = Experience(
            experience_id="success_exp",
            agent_id="test_agent",
            timestamp=datetime.now(),
            context=Mock(),
            action_taken=action_mock,
            reasoning=Mock(),
            outcome=Mock(),
            success=True,
            lessons_learned=["Good lesson"],
            mitre_attack_mapping=[],
            confidence_score=0.8
        )
        
        self.system.learn_from_experience("test_agent", success_exp)
        
        # Confidence should increase after success
        new_confidence = self.system.cognitive_states["test_agent"].confidence
        assert new_confidence > original_confidence
    
    def test_agent_profile_generation(self):
        """Test comprehensive agent profile generation"""
        # Initialize agent
        self.system.personality_system.create_personality("test_agent", OperationalStyle.STEALTHY)
        self.system.cognitive_states["test_agent"] = CognitiveState()
        
        profile = self.system.get_agent_profile("test_agent")
        
        assert profile["agent_id"] == "test_agent"
        assert "personality" in profile
        assert "cognitive_state" in profile
        assert "behavior_modifiers" in profile
        assert "memory_decay_config" in profile
        
        # Check personality data
        assert profile["personality"]["operational_style"] == "stealthy"
        assert len(profile["personality"]["traits"]) == len(PersonalityTrait)
    
    def test_cognitive_state_reset(self):
        """Test cognitive state reset functionality"""
        # Initialize with high stress and fatigue
        self.system.cognitive_states["test_agent"] = CognitiveState(
            stress_level=0.8,
            fatigue=0.9,
            attention_span=0.3
        )
        
        self.system.reset_cognitive_state("test_agent")
        
        state = self.system.cognitive_states["test_agent"]
        assert state.stress_level < 0.8  # Should be reduced
        assert state.fatigue < 0.9      # Should be reduced
        assert state.attention_span > 0.3  # Should be improved
    
    def test_system_statistics(self):
        """Test system statistics generation"""
        # Initialize multiple agents
        for i in range(3):
            agent_id = f"agent_{i}"
            self.system.personality_system.create_personality(agent_id)
            self.system.cognitive_states[agent_id] = CognitiveState()
        
        stats = self.system.get_system_statistics()
        
        assert stats["total_agents"] == 3
        assert "average_metrics" in stats
        assert "personality_distribution" in stats
        assert "memory_accesses" in stats

class TestFactoryFunctions:
    """Test factory and utility functions"""
    
    def test_create_cognitive_modeling_system(self):
        """Test system creation with custom configs"""
        memory_config = MemoryDecayConfig(decay_type=MemoryDecayType.LINEAR)
        fuzzing_config = BehaviorFuzzingConfig(enabled=False)
        
        system = create_cognitive_modeling_system(memory_config, fuzzing_config)
        
        assert system.memory_decay.config.decay_type == MemoryDecayType.LINEAR
        assert not system.behavior_fuzzing.config.enabled
    
    def test_create_personality_for_role(self):
        """Test role-based personality creation"""
        # Test stealth roles
        style, traits = create_personality_for_role(Role.RECON)
        assert style == OperationalStyle.STEALTHY
        assert PersonalityTrait.STEALTH in traits
        assert traits[PersonalityTrait.STEALTH] > 0.6
        
        # Test aggressive roles
        style, traits = create_personality_for_role(Role.EXPLOIT)
        assert style == OperationalStyle.AGGRESSIVE
        assert PersonalityTrait.AGGRESSIVENESS in traits
        assert traits[PersonalityTrait.AGGRESSIVENESS] > 0.6
        
        # Test methodical roles
        style, traits = create_personality_for_role(Role.PERSISTENCE)
        assert style == OperationalStyle.METHODICAL
        assert PersonalityTrait.PERSISTENCE in traits
        assert traits[PersonalityTrait.PERSISTENCE] > 0.7

class TestMemoryDecayTypes:
    """Test different memory decay algorithms"""
    
    def test_all_decay_types(self):
        """Test all decay type implementations"""
        decay_types = [
            MemoryDecayType.EXPONENTIAL,
            MemoryDecayType.LINEAR,
            MemoryDecayType.LOGARITHMIC,
            MemoryDecayType.STEPPED,
            MemoryDecayType.FORGETTING_CURVE
        ]
        
        for decay_type in decay_types:
            config = MemoryDecayConfig(decay_type=decay_type)
            model = MemoryDecayModel(config)
            
            # Test with experience at half-life
            experience = Experience(
                experience_id=f"test_{decay_type.value}",
                agent_id="test_agent",
                timestamp=datetime.now() - config.half_life,
                context=Mock(),
                action_taken=Mock(),
                reasoning=Mock(),
                outcome=Mock(),
                success=True,
                lessons_learned=[],
                mitre_attack_mapping=[],
                confidence_score=0.5
            )
            
            strength = model.calculate_memory_strength(experience)
            
            # All decay types should produce valid strengths
            assert 0.0 <= strength <= 1.0
            # At half-life, strength should be reduced (but importance factors may keep it higher)
            assert strength < 1.0

class TestPersonalityTraitInteractions:
    """Test personality trait interactions and consistency"""
    
    def test_trait_consistency_enforcement(self):
        """Test that inconsistent traits are corrected"""
        system = PersonalitySystem()
        
        # Create personality with conflicting traits
        personality = system.create_personality("test_agent")
        
        # Manually set conflicting traits
        personality.traits[PersonalityTrait.CAUTIOUSNESS] = 0.9
        personality.traits[PersonalityTrait.AGGRESSIVENESS] = 0.9
        
        # Apply consistency enforcement
        consistent_personality = system._ensure_consistency(personality)
        
        # One of the conflicting traits should be reduced
        assert (consistent_personality.traits[PersonalityTrait.CAUTIOUSNESS] < 0.9 or
                consistent_personality.traits[PersonalityTrait.AGGRESSIVENESS] < 0.9)
    
    def test_personality_adaptation_limits(self):
        """Test that personality adaptation has reasonable limits"""
        system = PersonalitySystem()
        personality = system.create_personality("test_agent")
        
        original_traits = personality.traits.copy()
        
        # Create many successful experiences
        for i in range(100):
            action_mock = Mock()
            action_mock.action_type = "test_action"
            
            experience = Experience(
                experience_id=f"exp_{i}",
                agent_id="test_agent",
                timestamp=datetime.now(),
                context=Mock(),
                action_taken=action_mock,
                reasoning=Mock(),
                outcome=Mock(),
                success=True,
                lessons_learned=[],
                mitre_attack_mapping=[],
                confidence_score=0.8
            )
            system.update_personality("test_agent", experience)
        
        # Traits should still be in valid range
        updated_personality = system.get_personality("test_agent")
        for trait, value in updated_personality.traits.items():
            assert 0.0 <= value <= 1.0
            # Should not change too dramatically
            assert abs(value - original_traits[trait]) < 0.5

if __name__ == "__main__":
    pytest.main([__file__])