#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Continuous Learning System Demo
Demonstrates continuous learning with human-in-the-loop feedback and knowledge distillation
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import system components
from agents.continuous_learning import (
    ContinuousLearningSystem, LearningPolicy, LearningPolicyType, 
    ModelUpdateStrategy
)
from agents.human_in_the_loop import (
    HumanInTheLoopInterface, HumanFeedback, FeedbackType, 
    ValidationStatus, Priority
)
from agents.knowledge_distillation import (
    KnowledgeDistillationPipeline, DistillationType, RelevanceLevel
)
from agents.learning_system import LearningSystem, LearningConfig
from agents.base_agent import Experience, Action, Team, Role
from memory.memory_manager import MemoryManager


class ContinuousLearningDemo:
    """Demo class for continuous learning system"""
    
    def __init__(self):
        self.continuous_learning = None
        self.human_interface = None
        self.distillation_pipeline = None
        self.learning_system = None
        self.memory_manager = None
        
        # Demo agents
        self.demo_agents = ["red_recon", "red_exploit", "blue_soc", "blue_firewall"]
        
        # Simulation state
        self.simulation_running = False
        self.demo_experiences = []
        self.demo_feedback = []
    
    async def initialize(self):
        """Initialize the demo system"""
        try:
            logger.info("🚀 Initializing Continuous Learning System Demo")
            
            # Initialize core components
            logger.info("📚 Setting up learning system...")
            self.learning_system = LearningSystem(LearningConfig())
            await self.learning_system.initialize()
            
            logger.info("🧠 Setting up memory manager...")
            self.memory_manager = MemoryManager()
            await self.memory_manager.initialize()
            
            logger.info("🤖 Setting up human interface...")
            self.human_interface = HumanInTheLoopInterface(self.learning_system)
            await self.human_interface.initialize()
            
            logger.info("🔬 Setting up knowledge distillation...")
            self.distillation_pipeline = KnowledgeDistillationPipeline()
            await self.distillation_pipeline.initialize()
            
            logger.info("🔄 Setting up continuous learning system...")
            self.continuous_learning = ContinuousLearningSystem(
                learning_system=self.learning_system,
                human_interface=self.human_interface,
                distillation_pipeline=self.distillation_pipeline,
                memory_manager=self.memory_manager
            )
            await self.continuous_learning.initialize()
            
            # Register demo agents with different learning policies
            await self._register_demo_agents()
            
            logger.info("✅ Demo system initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize demo system: {e}")
            raise
    
    async def _register_demo_agents(self):
        """Register demo agents with different learning policies"""
        logger.info("👥 Registering demo agents...")
        
        # Red Team Recon Agent - Conservative learning
        conservative_policy = LearningPolicy(
            policy_id="conservative_recon",
            policy_type=LearningPolicyType.CONSERVATIVE,
            update_strategy=ModelUpdateStrategy.HUMAN_APPROVED,
            learning_rate=0.005,
            exploration_rate=0.05,
            human_approval_required=True,
            feedback_threshold=5
        )
        await self.continuous_learning.register_agent("red_recon", custom_policy=conservative_policy)
        
        # Red Team Exploit Agent - Aggressive learning
        aggressive_policy = LearningPolicy(
            policy_id="aggressive_exploit",
            policy_type=LearningPolicyType.AGGRESSIVE,
            update_strategy=ModelUpdateStrategy.THRESHOLD_BASED,
            learning_rate=0.02,
            exploration_rate=0.3,
            feedback_threshold=3
        )
        await self.continuous_learning.register_agent("red_exploit", custom_policy=aggressive_policy)
        
        # Blue Team SOC Agent - Balanced learning
        await self.continuous_learning.register_agent("blue_soc", LearningPolicyType.BALANCED)
        
        # Blue Team Firewall Agent - Human-guided learning
        human_guided_policy = LearningPolicy(
            policy_id="human_guided_firewall",
            policy_type=LearningPolicyType.HUMAN_GUIDED,
            update_strategy=ModelUpdateStrategy.HUMAN_APPROVED,
            learning_rate=0.01,
            exploration_rate=0.1,
            human_approval_required=True,
            feedback_threshold=2
        )
        await self.continuous_learning.register_agent("blue_firewall", custom_policy=human_guided_policy)
        
        logger.info(f"✅ Registered {len(self.demo_agents)} demo agents")
    
    async def run_demo(self):
        """Run the complete demo"""
        try:
            logger.info("🎬 Starting Continuous Learning Demo")
            
            # Phase 1: Generate initial experiences
            await self._phase_1_generate_experiences()
            
            # Phase 2: Simulate human feedback
            await self._phase_2_human_feedback()
            
            # Phase 3: Demonstrate knowledge distillation
            await self._phase_3_knowledge_distillation()
            
            # Phase 4: Show learning updates
            await self._phase_4_learning_updates()
            
            # Phase 5: Demonstrate human validation workflow
            await self._phase_5_human_validation()
            
            # Phase 6: Show system statistics and insights
            await self._phase_6_system_insights()
            
            logger.info("🎉 Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
    
    async def _phase_1_generate_experiences(self):
        """Phase 1: Generate diverse agent experiences"""
        logger.info("\n" + "="*60)
        logger.info("📊 PHASE 1: Generating Agent Experiences")
        logger.info("="*60)
        
        experience_templates = {
            "red_recon": [
                ("network_scan", {"target": "192.168.1.0/24"}, 0.8),
                ("port_scan", {"target": "192.168.1.10", "ports": "1-1000"}, 0.7),
                ("service_enum", {"target": "192.168.1.10", "service": "ssh"}, 0.6),
                ("vulnerability_scan", {"target": "192.168.1.10"}, 0.5),
                ("dns_enum", {"domain": "target.local"}, 0.9)
            ],
            "red_exploit": [
                ("sql_injection", {"target": "webapp.target.local"}, 0.4),
                ("buffer_overflow", {"target": "192.168.1.10", "service": "ftp"}, 0.3),
                ("privilege_escalation", {"method": "sudo_exploit"}, 0.6),
                ("lateral_movement", {"target": "192.168.1.20"}, 0.5),
                ("persistence", {"method": "cron_job"}, 0.7)
            ],
            "blue_soc": [
                ("alert_analysis", {"alert_id": "IDS-001"}, 0.8),
                ("incident_creation", {"severity": "high"}, 0.9),
                ("threat_hunting", {"ioc": "suspicious_ip"}, 0.6),
                ("log_correlation", {"timeframe": "1h"}, 0.7),
                ("forensic_analysis", {"artifact": "memory_dump"}, 0.5)
            ],
            "blue_firewall": [
                ("rule_creation", {"action": "block", "src": "malicious_ip"}, 0.8),
                ("traffic_analysis", {"protocol": "tcp", "port": 443}, 0.9),
                ("policy_update", {"rule_set": "web_protection"}, 0.7),
                ("anomaly_detection", {"threshold": "95%"}, 0.6),
                ("rule_optimization", {"performance": "high"}, 0.5)
            ]
        }
        
        for agent_id in self.demo_agents:
            logger.info(f"🤖 Generating experiences for {agent_id}...")
            
            templates = experience_templates[agent_id]
            for i in range(15):  # Generate 15 experiences per agent
                template = random.choice(templates)
                action_type, parameters, base_success_rate = template
                
                # Add some randomness to success rate
                success_rate = base_success_rate + random.uniform(-0.2, 0.2)
                success = random.random() < max(0.1, min(0.9, success_rate))
                
                action = Action(
                    action_id=f"{agent_id}_{action_type}_{i}",
                    action_type=action_type,
                    parameters=parameters,
                    timestamp=datetime.now()
                )
                
                experience = Experience(
                    experience_id=f"exp_{agent_id}_{i}",
                    agent_id=agent_id,
                    timestamp=datetime.now(),
                    action_taken=action,
                    success=success,
                    reasoning=f"Automated {action_type} based on current objectives",
                    outcome="Success" if success else "Failed"
                )
                
                await self.continuous_learning.add_experience(agent_id, experience)
                self.demo_experiences.append(experience)
            
            logger.info(f"✅ Generated 15 experiences for {agent_id}")
        
        logger.info(f"📈 Total experiences generated: {len(self.demo_experiences)}")
    
    async def _phase_2_human_feedback(self):
        """Phase 2: Simulate human feedback on agent performance"""
        logger.info("\n" + "="*60)
        logger.info("👤 PHASE 2: Simulating Human Feedback")
        logger.info("="*60)
        
        feedback_scenarios = [
            # Performance ratings
            ("red_recon", FeedbackType.PERFORMANCE_RATING, {
                "performance_rating": 0.8,
                "comments": "Excellent reconnaissance techniques, very thorough"
            }),
            ("red_exploit", FeedbackType.PERFORMANCE_RATING, {
                "performance_rating": 0.6,
                "comments": "Good exploitation attempts but needs better stealth"
            }),
            ("blue_soc", FeedbackType.PERFORMANCE_RATING, {
                "performance_rating": 0.9,
                "comments": "Outstanding threat detection and response time"
            }),
            ("blue_firewall", FeedbackType.PERFORMANCE_RATING, {
                "performance_rating": 0.7,
                "comments": "Solid rule creation but could optimize performance"
            }),
            
            # Behavior tagging
            ("red_recon", FeedbackType.BEHAVIOR_TAGGING, {
                "action_id": "red_recon_network_scan_0",
                "behavior_tags": ["thorough", "stealthy", "patient"],
                "correctness_score": 0.9,
                "comments": "Perfect scanning approach"
            }),
            ("red_exploit", FeedbackType.BEHAVIOR_TAGGING, {
                "action_id": "red_exploit_sql_injection_0",
                "behavior_tags": ["aggressive", "noisy", "effective"],
                "correctness_score": 0.4,
                "comments": "Too aggressive, easily detected"
            }),
            ("blue_soc", FeedbackType.BEHAVIOR_TAGGING, {
                "action_id": "blue_soc_alert_analysis_0",
                "behavior_tags": ["analytical", "thorough", "fast"],
                "correctness_score": 0.95,
                "comments": "Excellent analysis methodology"
            }),
            
            # Strategy corrections
            ("red_exploit", FeedbackType.STRATEGY_CORRECTION, {
                "strategy_modifications": {
                    "stealth_priority": 0.8,
                    "speed_priority": 0.3,
                    "noise_tolerance": 0.2
                },
                "reasoning_corrections": [
                    "Prioritize stealth over speed",
                    "Reduce detection probability",
                    "Use more subtle exploitation techniques"
                ],
                "comments": "Agent needs to be more careful about detection"
            }),
            ("blue_firewall", FeedbackType.STRATEGY_CORRECTION, {
                "strategy_modifications": {
                    "rule_complexity": 0.6,
                    "performance_weight": 0.8,
                    "false_positive_tolerance": 0.1
                },
                "reasoning_corrections": [
                    "Simplify rules for better performance",
                    "Reduce false positive rate",
                    "Focus on high-impact threats"
                ],
                "comments": "Balance security with performance"
            })
        ]
        
        for agent_id, feedback_type, feedback_data in feedback_scenarios:
            logger.info(f"💬 Adding {feedback_type.value} feedback for {agent_id}")
            
            feedback_id = await self.human_interface.provide_feedback(
                agent_id=agent_id,
                feedback_type=feedback_type,
                reviewer_id="security_expert_1",
                **feedback_data
            )
            
            # Also add to continuous learning system
            feedback = HumanFeedback(
                feedback_id=feedback_id,
                agent_id=agent_id,
                feedback_type=feedback_type,
                timestamp=datetime.now(),
                reviewer_id="security_expert_1",
                **feedback_data
            )
            
            await self.continuous_learning.add_human_feedback(agent_id, feedback)
            self.demo_feedback.append(feedback)
        
        logger.info(f"✅ Added {len(feedback_scenarios)} feedback items")
    
    async def _phase_3_knowledge_distillation(self):
        """Phase 3: Demonstrate knowledge distillation"""
        logger.info("\n" + "="*60)
        logger.info("🔬 PHASE 3: Knowledge Distillation")
        logger.info("="*60)
        
        for agent_id in self.demo_agents:
            logger.info(f"🧪 Running knowledge distillation for {agent_id}...")
            
            # Get agent experiences
            agent_experiences = [exp for exp in self.demo_experiences if exp.agent_id == agent_id]
            agent_feedback = [fb for fb in self.demo_feedback if fb.agent_id == agent_id]
            
            # Run different types of distillation
            distillation_types = [
                DistillationType.BEHAVIOR_PRUNING,
                DistillationType.STRATEGY_COMPRESSION,
                DistillationType.RELEVANCE_SCORING
            ]
            
            for distill_type in distillation_types:
                logger.info(f"  🔍 Running {distill_type.value}...")
                
                result = await self.distillation_pipeline.distill_agent_knowledge(
                    agent_id=agent_id,
                    experiences=agent_experiences,
                    distillation_type=distill_type,
                    human_feedback=agent_feedback
                )
                
                logger.info(f"    📊 Input: {result.input_experiences} experiences")
                logger.info(f"    📉 Output: {result.output_experiences} experiences")
                logger.info(f"    🗜️  Compression: {result.compression_ratio:.2f}")
                logger.info(f"    ⭐ Quality: {result.quality_score:.2f}")
                logger.info(f"    🚫 Pruned: {len(result.pruned_behaviors)} behaviors")
                logger.info(f"    🔍 Patterns: {len(result.extracted_patterns)} found")
                
                if result.pruned_behaviors:
                    logger.info(f"    🗑️  Removed behaviors: {', '.join(result.pruned_behaviors[:3])}...")
        
        logger.info("✅ Knowledge distillation completed for all agents")
    
    async def _phase_4_learning_updates(self):
        """Phase 4: Demonstrate learning updates"""
        logger.info("\n" + "="*60)
        logger.info("📈 PHASE 4: Learning Updates")
        logger.info("="*60)
        
        for agent_id in self.demo_agents:
            logger.info(f"🔄 Triggering learning update for {agent_id}...")
            
            policy = self.continuous_learning.learning_policies[agent_id]
            logger.info(f"  📋 Policy: {policy.policy_type.value}")
            logger.info(f"  🎯 Strategy: {policy.update_strategy.value}")
            logger.info(f"  🧠 Learning rate: {policy.learning_rate}")
            
            # Request learning update
            update_id = await self.continuous_learning.request_learning_update(
                agent_id=agent_id,
                trigger="demo_phase_4",
                immediate=True
            )
            
            # Get update results
            updates = self.continuous_learning.update_history[agent_id]
            if updates:
                latest_update = updates[-1]
                logger.info(f"  ✅ Update {update_id} completed")
                logger.info(f"  📊 Success: {latest_update.success}")
                logger.info(f"  💬 Feedback incorporated: {len(latest_update.feedback_incorporated)}")
                
                if latest_update.model_changes:
                    logger.info(f"  🔧 Model changes: {latest_update.model_changes}")
                
                if latest_update.previous_performance and latest_update.updated_performance:
                    prev_success = latest_update.previous_performance.success_rate
                    new_success = latest_update.updated_performance.success_rate
                    improvement = new_success - prev_success
                    logger.info(f"  📈 Performance change: {improvement:+.3f}")
        
        logger.info("✅ Learning updates completed for all agents")
    
    async def _phase_5_human_validation(self):
        """Phase 5: Demonstrate human validation workflow"""
        logger.info("\n" + "="*60)
        logger.info("✋ PHASE 5: Human Validation Workflow")
        logger.info("="*60)
        
        validation_scenarios = [
            {
                "agent_id": "red_exploit",
                "action": Action(
                    action_id="critical_exploit_attempt",
                    action_type="privilege_escalation",
                    parameters={"target": "domain_controller", "method": "zerologon"},
                    timestamp=datetime.now()
                ),
                "context": {"risk_level": "critical", "target_criticality": "high"},
                "reasoning": "Attempting to exploit critical domain controller vulnerability",
                "confidence": 0.7,
                "expected_status": ValidationStatus.APPROVED
            },
            {
                "agent_id": "blue_firewall",
                "action": Action(
                    action_id="emergency_block_rule",
                    action_type="emergency_block",
                    parameters={"src_ip": "192.168.1.100", "reason": "malware_c2"},
                    timestamp=datetime.now()
                ),
                "context": {"urgency": "high", "impact": "network_wide"},
                "reasoning": "Blocking suspected malware C2 communication",
                "confidence": 0.9,
                "expected_status": ValidationStatus.APPROVED
            },
            {
                "agent_id": "red_recon",
                "action": Action(
                    action_id="aggressive_scan",
                    action_type="comprehensive_scan",
                    parameters={"target": "entire_network", "intensity": "maximum"},
                    timestamp=datetime.now()
                ),
                "context": {"stealth": "low", "detection_risk": "high"},
                "reasoning": "Comprehensive network reconnaissance",
                "confidence": 0.4,
                "expected_status": ValidationStatus.MODIFIED
            }
        ]
        
        for scenario in validation_scenarios:
            logger.info(f"🔍 Requesting validation for {scenario['agent_id']}...")
            
            # Request validation
            request_id = await self.human_interface.request_action_validation(
                agent_id=scenario["agent_id"],
                action=scenario["action"],
                context=scenario["context"],
                reasoning=scenario["reasoning"],
                confidence_score=scenario["confidence"]
            )
            
            logger.info(f"  📝 Validation request: {request_id}")
            logger.info(f"  🎯 Action: {scenario['action'].action_type}")
            logger.info(f"  🎲 Confidence: {scenario['confidence']}")
            
            # Simulate human review
            await asyncio.sleep(0.1)  # Simulate review time
            
            # Provide validation response
            modifications = {}
            comments = ""
            
            if scenario["expected_status"] == ValidationStatus.APPROVED:
                comments = "Approved - action is appropriate for current situation"
            elif scenario["expected_status"] == ValidationStatus.MODIFIED:
                modifications = {"stealth_mode": True, "intensity": "low"}
                comments = "Approved with modifications - reduce detection risk"
            
            success = await self.human_interface.validate_action(
                request_id=request_id,
                reviewer_id="security_analyst_1",
                status=scenario["expected_status"],
                comments=comments,
                modifications=modifications
            )
            
            logger.info(f"  ✅ Validation result: {scenario['expected_status'].value}")
            if modifications:
                logger.info(f"  🔧 Modifications: {modifications}")
        
        # Show pending validations
        pending = await self.human_interface.get_pending_validations()
        logger.info(f"📋 Pending validations: {len(pending)}")
        
        logger.info("✅ Human validation workflow demonstrated")
    
    async def _phase_6_system_insights(self):
        """Phase 6: Show system statistics and insights"""
        logger.info("\n" + "="*60)
        logger.info("📊 PHASE 6: System Insights & Statistics")
        logger.info("="*60)
        
        # Feedback statistics
        feedback_stats = await self.human_interface.get_feedback_statistics()
        logger.info("👤 Human Feedback Statistics:")
        logger.info(f"  📊 Total feedback: {feedback_stats['total_feedback']}")
        logger.info(f"  ✅ Total validations: {feedback_stats['total_validations']}")
        logger.info(f"  ⏳ Pending validations: {feedback_stats['pending_validations']}")
        logger.info(f"  🤖 Active agents: {feedback_stats['active_agents']}")
        logger.info(f"  🔧 Total corrections: {feedback_stats['total_corrections']}")
        
        # Agent-specific insights
        logger.info("\n🤖 Agent Learning Insights:")
        for agent_id in self.demo_agents:
            policy = self.continuous_learning.learning_policies[agent_id]
            experiences = len(self.continuous_learning.experience_buffer[agent_id])
            feedback = len(self.continuous_learning.feedback_buffer[agent_id])
            updates = len(self.continuous_learning.update_history[agent_id])
            
            logger.info(f"\n  🎯 {agent_id}:")
            logger.info(f"    📋 Policy: {policy.policy_type.value}")
            logger.info(f"    📊 Experiences: {experiences}")
            logger.info(f"    💬 Feedback: {feedback}")
            logger.info(f"    🔄 Updates: {updates}")
            logger.info(f"    🧠 Learning rate: {policy.learning_rate}")
            logger.info(f"    🔍 Exploration rate: {policy.exploration_rate}")
        
        # Learning effectiveness metrics
        logger.info("\n📈 Learning Effectiveness:")
        total_experiences = sum(len(self.continuous_learning.experience_buffer[agent_id]) 
                              for agent_id in self.demo_agents)
        total_feedback = sum(len(self.continuous_learning.feedback_buffer[agent_id]) 
                           for agent_id in self.demo_agents)
        total_updates = sum(len(self.continuous_learning.update_history[agent_id]) 
                          for agent_id in self.demo_agents)
        
        feedback_ratio = total_feedback / total_experiences if total_experiences > 0 else 0
        update_ratio = total_updates / total_experiences if total_experiences > 0 else 0
        
        logger.info(f"  📊 Total experiences: {total_experiences}")
        logger.info(f"  💬 Feedback ratio: {feedback_ratio:.2%}")
        logger.info(f"  🔄 Update ratio: {update_ratio:.2%}")
        
        # Distillation insights
        logger.info("\n🔬 Knowledge Distillation Insights:")
        for agent_id in self.demo_agents:
            if agent_id in self.distillation_pipeline.distillation_history:
                history = self.distillation_pipeline.distillation_history[agent_id]
                if history:
                    latest = history[-1]
                    logger.info(f"  🧪 {agent_id}:")
                    logger.info(f"    🗜️  Compression: {latest.compression_ratio:.2f}")
                    logger.info(f"    ⭐ Quality: {latest.quality_score:.2f}")
                    logger.info(f"    🚫 Pruned behaviors: {len(latest.pruned_behaviors)}")
        
        logger.info("\n✨ Demo insights generated successfully!")


async def main():
    """Main demo function"""
    demo = ContinuousLearningDemo()
    
    try:
        # Initialize the demo system
        await demo.initialize()
        
        # Run the complete demo
        await demo.run_demo()
        
        print("\n" + "="*80)
        print("🎉 CONTINUOUS LEARNING SYSTEM DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("="*80)
        print("\nKey Features Demonstrated:")
        print("✅ Continuous learning with multiple policy types")
        print("✅ Human-in-the-loop feedback integration")
        print("✅ Knowledge distillation with behavior pruning")
        print("✅ Automated learning updates and model management")
        print("✅ Human validation workflow for critical actions")
        print("✅ Comprehensive system insights and analytics")
        print("\nThe system successfully integrates:")
        print("🔄 Continuous learning loops")
        print("👤 Human feedback and validation")
        print("🔬 Knowledge distillation pipelines")
        print("📊 Performance tracking and improvement")
        print("🛡️  Safety mechanisms and oversight")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())