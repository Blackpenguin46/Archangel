#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Cognitive Modeling Demo
Demonstrates memory decay, personality vectors, and behavior fuzzing
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List

from memory.cognitive_modeling import (
    CognitiveModelingSystem,
    MemoryDecayConfig,
    BehaviorFuzzingConfig,
    MemoryDecayType,
    OperationalStyle,
    PersonalityTrait,
    create_cognitive_modeling_system,
    create_personality_for_role
)
from agents.base_agent import Experience, Team, Role, AgentConfig, ActionPlan, ActionResult

class CognitiveModelingDemo:
    """
    Comprehensive demonstration of the cognitive modeling system.
    
    Shows memory decay, personality-driven behavior, and fuzzing effects
    in realistic agent scenarios.
    """
    
    def __init__(self):
        # Create cognitive modeling system with custom configs
        memory_config = MemoryDecayConfig(
            decay_type=MemoryDecayType.EXPONENTIAL,
            half_life=timedelta(days=5),
            minimum_strength=0.15,
            importance_factor=1.8,
            success_bias=1.4
        )
        
        fuzzing_config = BehaviorFuzzingConfig(
            enabled=True,
            base_randomness=0.12,
            stress_amplifier=2.2,
            decision_noise=0.08
        )
        
        self.cognitive_system = create_cognitive_modeling_system(memory_config, fuzzing_config)
        
        # Demo agents with different roles and personalities
        self.demo_agents = [
            {
                'id': 'red_recon_001',
                'config': AgentConfig(
                    agent_id='red_recon_001',
                    team=Team.RED,
                    role=Role.RECON,
                    name='Stealthy Scanner',
                    description='Red team reconnaissance specialist'
                ),
                'style': OperationalStyle.STEALTHY
            },
            {
                'id': 'red_exploit_001',
                'config': AgentConfig(
                    agent_id='red_exploit_001',
                    team=Team.RED,
                    role=Role.EXPLOIT,
                    name='Aggressive Exploiter',
                    description='Red team exploitation specialist'
                ),
                'style': OperationalStyle.AGGRESSIVE
            },
            {
                'id': 'blue_soc_001',
                'config': AgentConfig(
                    agent_id='blue_soc_001',
                    team=Team.BLUE,
                    role=Role.SOC_ANALYST,
                    name='Methodical Analyst',
                    description='Blue team SOC analyst'
                ),
                'style': OperationalStyle.METHODICAL
            },
            {
                'id': 'blue_hunter_001',
                'config': AgentConfig(
                    agent_id='blue_hunter_001',
                    team=Team.BLUE,
                    role=Role.THREAT_HUNTER,
                    name='Opportunistic Hunter',
                    description='Blue team threat hunter'
                ),
                'style': OperationalStyle.OPPORTUNISTIC
            }
        ]
        
        self.experiences_created = 0
    
    async def run_demo(self):
        """Run the complete cognitive modeling demonstration"""
        print("🧠 Archangel Cognitive Modeling System Demo")
        print("=" * 60)
        
        # Initialize agents
        await self.initialize_demo_agents()
        
        # Demonstrate memory decay
        await self.demonstrate_memory_decay()
        
        # Demonstrate personality effects
        await self.demonstrate_personality_effects()
        
        # Demonstrate behavior fuzzing
        await self.demonstrate_behavior_fuzzing()
        
        # Demonstrate learning and adaptation
        await self.demonstrate_learning_adaptation()
        
        # Show system statistics
        await self.show_system_statistics()
        
        print("\n✅ Cognitive modeling demonstration complete!")
    
    async def initialize_demo_agents(self):
        """Initialize all demo agents with cognitive modeling"""
        print("\n🚀 Initializing Demo Agents")
        print("-" * 40)
        
        for agent_info in self.demo_agents:
            agent_id = agent_info['id']
            config = agent_info['config']
            style = agent_info['style']
            
            # Initialize cognitive modeling
            await self.cognitive_system.initialize_agent_cognition(agent_id, config, style)
            
            # Get and display personality
            personality = self.cognitive_system.personality_system.get_personality(agent_id)
            
            print(f"\n👤 Agent: {config.name} ({agent_id})")
            print(f"   Role: {config.role.value}")
            print(f"   Team: {config.team.value}")
            print(f"   Style: {style.value}")
            print(f"   Key Traits:")
            
            # Show top 3 personality traits
            sorted_traits = sorted(
                personality.traits.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for trait, value in sorted_traits:
                print(f"     {trait.value}: {value:.2f}")
    
    async def demonstrate_memory_decay(self):
        """Demonstrate memory decay over time"""
        print("\n🧠 Memory Decay Demonstration")
        print("-" * 40)
        
        agent_id = 'red_recon_001'
        
        # Create experiences at different time points
        experiences = []
        time_points = [
            (timedelta(minutes=5), "Recent reconnaissance"),
            (timedelta(hours=6), "Morning scan results"),
            (timedelta(days=2), "Previous vulnerability assessment"),
            (timedelta(days=7), "Week-old network mapping"),
            (timedelta(days=14), "Old penetration test"),
            (timedelta(days=30), "Ancient security audit")
        ]
        
        print(f"\n📊 Creating experiences for {agent_id}:")
        
        for age, description in time_points:
            experience = self.create_test_experience(
                agent_id,
                description,
                datetime.now() - age,
                success=random.choice([True, False]),
                importance=random.uniform(0.3, 0.9)
            )
            experiences.append((experience, description, age))
            
            # Calculate memory strength
            strength = self.cognitive_system.get_memory_strength(agent_id, experience)
            
            print(f"   {description:30} | Age: {str(age):12} | Strength: {strength:.3f}")
        
        # Demonstrate memory access effects
        print(f"\n🔄 Demonstrating memory access effects:")
        
        # Access an old memory multiple times
        old_experience = experiences[3][0]  # Week-old experience
        original_strength = self.cognitive_system.get_memory_strength(agent_id, old_experience)
        
        print(f"   Original strength: {original_strength:.3f}")
        
        # Access it several times
        for i in range(3):
            self.cognitive_system.access_memory(agent_id, old_experience.experience_id)
            await asyncio.sleep(0.1)  # Small delay
        
        new_strength = self.cognitive_system.get_memory_strength(agent_id, old_experience)
        print(f"   After 3 accesses: {new_strength:.3f}")
        print(f"   Improvement: {((new_strength - original_strength) / original_strength * 100):+.1f}%")
    
    async def demonstrate_personality_effects(self):
        """Demonstrate how personality affects behavior"""
        print("\n🎭 Personality Effects Demonstration")
        print("-" * 40)
        
        # Compare decision making between different personality types
        decision_scenario = {
            'attack_vulnerable_service': 0.7,
            'gather_more_intelligence': 0.6,
            'maintain_stealth': 0.8,
            'coordinate_with_team': 0.5,
            'wait_for_opportunity': 0.4
        }
        
        print("\n🎯 Decision Scenario: Discovered vulnerable service")
        print("Base decision scores:")
        for decision, score in decision_scenario.items():
            print(f"   {decision:25} | {score:.2f}")
        
        print("\n👥 Agent Responses (with personality influence):")
        
        for agent_info in self.demo_agents:
            agent_id = agent_info['id']
            config = agent_info['config']
            
            # Get behavior modifiers
            modifiers = self.cognitive_system.personality_system.get_behavior_modifiers(agent_id)
            
            # Apply personality influence to decisions
            influenced_decisions = {}
            for decision, base_score in decision_scenario.items():
                influenced_score = base_score
                
                # Apply relevant modifiers
                if 'attack' in decision:
                    influenced_score *= (0.5 + modifiers.get('aggression_level', 0.5))
                elif 'stealth' in decision:
                    influenced_score *= (0.5 + modifiers.get('stealth_preference', 0.5))
                elif 'intelligence' in decision:
                    influenced_score *= (0.5 + modifiers.get('exploration_tendency', 0.5))
                elif 'coordinate' in decision:
                    influenced_score *= (0.5 + modifiers.get('collaboration_preference', 0.5))
                elif 'wait' in decision:
                    influenced_score *= (0.5 + (1.0 - modifiers.get('aggression_level', 0.5)))
                
                influenced_decisions[decision] = min(1.0, influenced_score)
            
            # Find preferred action
            preferred_action = max(influenced_decisions.items(), key=lambda x: x[1])
            
            print(f"\n   {config.name} ({config.role.value}):")
            print(f"     Preferred: {preferred_action[0]} (score: {preferred_action[1]:.2f})")
            print(f"     Style influence: {config.role.value} -> {agent_info['style'].value}")
    
    async def demonstrate_behavior_fuzzing(self):
        """Demonstrate behavior fuzzing effects"""
        print("\n🎲 Behavior Fuzzing Demonstration")
        print("-" * 40)
        
        agent_id = 'red_exploit_001'
        
        # Base decision scores
        base_decisions = {
            'exploit_immediately': 0.8,
            'gather_credentials': 0.6,
            'lateral_movement': 0.7,
            'establish_persistence': 0.5
        }
        
        # Base action parameters
        base_parameters = {
            'timeout': 30.0,
            'retry_count': 3,
            'stealth_delay': 5.0,
            'payload_size': 1024
        }
        
        print(f"\n🎯 Testing fuzzing effects on {agent_id}")
        print("\nBase decisions:")
        for decision, score in base_decisions.items():
            print(f"   {decision:20} | {score:.3f}")
        
        print("\nBase parameters:")
        for param, value in base_parameters.items():
            print(f"   {param:15} | {value}")
        
        # Test under different stress conditions
        stress_levels = [0.1, 0.5, 0.9]
        
        for stress in stress_levels:
            print(f"\n🌡️  Stress Level: {stress:.1f}")
            
            # Update cognitive state
            self.cognitive_system.update_cognitive_state(
                agent_id,
                stress_delta=stress - self.cognitive_system.cognitive_states[agent_id].stress_level
            )
            
            # Apply cognitive effects
            fuzzed_decisions, fuzzed_params = self.cognitive_system.apply_cognitive_effects(
                agent_id,
                base_decisions.copy(),
                base_parameters.copy()
            )
            
            print("   Fuzzed decisions:")
            for decision, (base_score, fuzzed_score) in zip(base_decisions.keys(), 
                                                           zip(base_decisions.values(), fuzzed_decisions.values())):
                change = fuzzed_score - base_score
                print(f"     {decision:20} | {fuzzed_score:.3f} ({change:+.3f})")
            
            print("   Fuzzed parameters:")
            for param in ['timeout', 'stealth_delay']:
                if param in fuzzed_params:
                    base_val = base_parameters[param]
                    fuzzed_val = fuzzed_params[param]
                    change_pct = ((fuzzed_val - base_val) / base_val) * 100
                    print(f"     {param:15} | {fuzzed_val:.2f} ({change_pct:+.1f}%)")
    
    async def demonstrate_learning_adaptation(self):
        """Demonstrate learning and personality adaptation"""
        print("\n📚 Learning and Adaptation Demonstration")
        print("-" * 40)
        
        agent_id = 'blue_soc_001'
        
        # Get initial personality
        initial_personality = self.cognitive_system.personality_system.get_personality(agent_id)
        initial_traits = initial_personality.traits.copy()
        
        print(f"\n🧠 Agent: {agent_id}")
        print("Initial personality traits:")
        for trait, value in initial_traits.items():
            print(f"   {trait.value:15} | {value:.3f}")
        
        # Simulate a series of experiences
        print(f"\n🎯 Simulating learning experiences...")
        
        experience_types = [
            ("Successful threat detection", True, 0.8, ["Quick response prevented breach"]),
            ("Missed subtle attack", False, 0.3, ["Need better pattern recognition"]),
            ("Effective team coordination", True, 0.7, ["Collaboration led to success"]),
            ("False positive alert", False, 0.4, ["Tuning required for accuracy"]),
            ("Advanced persistent threat found", True, 0.9, ["Persistence paid off", "Deep analysis worked"]),
            ("Overwhelmed by alert volume", False, 0.2, ["Need better prioritization"]),
            ("Successful incident response", True, 0.8, ["Methodical approach worked"]),
            ("Missed zero-day exploit", False, 0.1, ["Adaptability needed"])
        ]
        
        for i, (description, success, confidence, lessons) in enumerate(experience_types):
            print(f"   Experience {i+1}: {description}")
            
            # Create experience
            experience = self.create_test_experience(
                agent_id,
                description,
                datetime.now() - timedelta(hours=i),
                success=success,
                confidence=confidence,
                lessons=lessons
            )
            
            # Learn from experience
            self.cognitive_system.learn_from_experience(agent_id, experience)
            
            # Show cognitive state changes
            cognitive_state = self.cognitive_system.cognitive_states[agent_id]
            print(f"     Success: {success} | Confidence: {cognitive_state.confidence:.3f} | Stress: {cognitive_state.stress_level:.3f}")
        
        # Show personality changes
        final_personality = self.cognitive_system.personality_system.get_personality(agent_id)
        final_traits = final_personality.traits
        
        print(f"\n📈 Personality adaptation results:")
        for trait in PersonalityTrait:
            initial_val = initial_traits[trait]
            final_val = final_traits[trait]
            change = final_val - initial_val
            change_pct = (change / initial_val) * 100 if initial_val > 0 else 0
            
            if abs(change) > 0.01:  # Only show significant changes
                print(f"   {trait.value:15} | {initial_val:.3f} -> {final_val:.3f} ({change_pct:+.1f}%)")
    
    async def show_system_statistics(self):
        """Show comprehensive system statistics"""
        print("\n📊 System Statistics")
        print("-" * 40)
        
        stats = self.cognitive_system.get_system_statistics()
        
        print(f"\n🔢 Overall Metrics:")
        print(f"   Total agents: {stats['total_agents']}")
        print(f"   Memory accesses: {stats['memory_accesses']}")
        print(f"   Fuzzing enabled: {stats['fuzzing_enabled']}")
        
        if 'average_metrics' in stats:
            avg_metrics = stats['average_metrics']
            print(f"\n📈 Average Cognitive Metrics:")
            print(f"   Stress level: {avg_metrics['stress_level']:.3f}")
            print(f"   Fatigue: {avg_metrics['fatigue']:.3f}")
            print(f"   Confidence: {avg_metrics['confidence']:.3f}")
            print(f"   Attention span: {avg_metrics['attention_span']:.3f}")
        
        if 'personality_distribution' in stats:
            print(f"\n🎭 Personality Distribution:")
            for style, count in stats['personality_distribution'].items():
                print(f"   {style:12} | {count} agents")
        
        # Show individual agent profiles
        print(f"\n👥 Individual Agent Profiles:")
        for agent_info in self.demo_agents:
            agent_id = agent_info['id']
            profile = self.cognitive_system.get_agent_profile(agent_id)
            
            print(f"\n   {profile['agent_id']}:")
            print(f"     Style: {profile['personality']['operational_style']}")
            print(f"     Confidence: {profile['cognitive_state']['confidence']:.3f}")
            print(f"     Stress: {profile['cognitive_state']['stress_level']:.3f}")
            print(f"     Attention: {profile['cognitive_state']['attention_span']:.3f}")
    
    def create_test_experience(self, 
                             agent_id: str,
                             description: str,
                             timestamp: datetime,
                             success: bool = True,
                             confidence: float = 0.7,
                             importance: float = 0.5,
                             lessons: List[str] = None) -> Experience:
        """Create a test experience for demonstration"""
        self.experiences_created += 1
        
        # Create mock action and outcome
        action = ActionPlan(
            primary_action=description,
            action_type="test_action",
            target="test_target",
            parameters={},
            expected_outcome="Test outcome",
            success_criteria=["Test criteria"],
            fallback_actions=[],
            estimated_duration=30.0,
            risk_level="medium"
        )
        
        outcome = ActionResult(
            action_id=f"action_{self.experiences_created}",
            action_type="test_action",
            success=success,
            outcome=description,
            data={},
            duration=30.0,
            errors=[] if success else ["Test error"],
            side_effects=[],
            confidence=confidence,
            timestamp=timestamp
        )
        
        # Adjust lessons based on importance
        if lessons is None:
            if importance > 0.7:
                lessons = [f"Important lesson from {description}"]
            elif importance > 0.4:
                lessons = [f"Lesson from {description}"]
            else:
                lessons = []
        
        # Create experience
        experience = Experience(
            experience_id=f"exp_{self.experiences_created}",
            agent_id=agent_id,
            timestamp=timestamp,
            context=None,  # Mock context
            action_taken=action,
            reasoning=None,  # Mock reasoning
            outcome=outcome,
            success=success,
            lessons_learned=lessons,
            mitre_attack_mapping=["T1001"] if importance > 0.6 else [],
            confidence_score=confidence
        )
        
        return experience

async def main():
    """Run the cognitive modeling demonstration"""
    demo = CognitiveModelingDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())