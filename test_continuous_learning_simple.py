#!/usr/bin/env python3
"""
Simple test for continuous learning and human-in-the-loop systems
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_continuous_learning_integration():
    """Test the integration of continuous learning components"""
    try:
        logger.info("🧪 Testing Continuous Learning Integration")
        
        # Import components
        from agents.continuous_learning import ContinuousLearningSystem, LearningPolicyType
        from agents.human_in_the_loop import HumanInTheLoopInterface, FeedbackType
        from agents.knowledge_distillation import KnowledgeDistillationPipeline, DistillationType
        from agents.base_agent import Experience, Action
        
        # Initialize components
        logger.info("📚 Initializing components...")
        
        human_interface = HumanInTheLoopInterface()
        await human_interface.initialize()
        
        distillation_pipeline = KnowledgeDistillationPipeline()
        await distillation_pipeline.initialize()
        
        continuous_learning = ContinuousLearningSystem(
            human_interface=human_interface,
            distillation_pipeline=distillation_pipeline
        )
        await continuous_learning.initialize()
        
        # Test agent registration
        logger.info("🤖 Testing agent registration...")
        agent_id = "test_agent"
        await continuous_learning.register_agent(agent_id, LearningPolicyType.BALANCED)
        
        assert agent_id in continuous_learning.learning_policies
        assert agent_id in continuous_learning.experience_buffer
        logger.info("✅ Agent registration successful")
        
        # Test experience addition
        logger.info("📊 Testing experience addition...")
        action = Action(
            action_id="test_action",
            action_type="scan",
            parameters={"target": "192.168.1.1"},
            timestamp=datetime.now()
        )
        
        experience = Experience(
            experience_id="test_exp",
            agent_id=agent_id,
            timestamp=datetime.now(),
            action_taken=action,
            success=True,
            reasoning="Test scan",
            outcome="Success"
        )
        
        await continuous_learning.add_experience(agent_id, experience)
        assert len(continuous_learning.experience_buffer[agent_id]) == 1
        logger.info("✅ Experience addition successful")
        
        # Test human feedback
        logger.info("👤 Testing human feedback...")
        feedback_id = await human_interface.provide_feedback(
            agent_id=agent_id,
            feedback_type=FeedbackType.PERFORMANCE_RATING,
            reviewer_id="test_human",
            performance_rating=0.8,
            comments="Good performance"
        )
        
        assert feedback_id is not None
        assert agent_id in human_interface.feedback_history
        logger.info("✅ Human feedback successful")
        
        # Test knowledge distillation
        logger.info("🔬 Testing knowledge distillation...")
        experiences = [experience] * 5  # Create multiple experiences
        
        result = await distillation_pipeline.distill_agent_knowledge(
            agent_id=agent_id,
            experiences=experiences,
            distillation_type=DistillationType.BEHAVIOR_PRUNING
        )
        
        assert result.agent_id == agent_id
        assert result.input_experiences == 5
        assert 0.0 <= result.quality_score <= 1.0
        logger.info("✅ Knowledge distillation successful")
        
        # Test learning update
        logger.info("🔄 Testing learning update...")
        update_id = await continuous_learning.request_learning_update(
            agent_id=agent_id,
            trigger="test",
            immediate=True
        )
        
        assert update_id is not None
        assert len(continuous_learning.update_history[agent_id]) == 1
        logger.info("✅ Learning update successful")
        
        # Test action validation
        logger.info("✋ Testing action validation...")
        validation_action = Action(
            action_id="validation_test",
            action_type="exploit",
            parameters={"target": "test_server"},
            timestamp=datetime.now()
        )
        
        request_id = await human_interface.request_action_validation(
            agent_id=agent_id,
            action=validation_action,
            context={"risk": "medium"},
            reasoning="Test validation",
            confidence_score=0.6
        )
        
        assert request_id is not None
        logger.info("✅ Action validation request successful")
        
        # Get system statistics
        logger.info("📊 Testing system statistics...")
        stats = await human_interface.get_feedback_statistics()
        
        assert "total_feedback" in stats
        assert stats["active_agents"] >= 1
        logger.info("✅ System statistics successful")
        
        logger.info("🎉 All continuous learning tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def test_learning_policies():
    """Test different learning policy types"""
    try:
        logger.info("🧪 Testing Learning Policies")
        
        from agents.continuous_learning import (
            ContinuousLearningSystem, LearningPolicy, LearningPolicyType, 
            ModelUpdateStrategy
        )
        
        continuous_learning = ContinuousLearningSystem()
        await continuous_learning.initialize()
        
        # Test different policy types
        policy_tests = [
            ("conservative_agent", LearningPolicyType.CONSERVATIVE),
            ("aggressive_agent", LearningPolicyType.AGGRESSIVE),
            ("balanced_agent", LearningPolicyType.BALANCED),
            ("human_guided_agent", LearningPolicyType.HUMAN_GUIDED)
        ]
        
        for agent_id, policy_type in policy_tests:
            logger.info(f"🎯 Testing {policy_type.value} policy...")
            
            await continuous_learning.register_agent(agent_id, policy_type)
            
            policy = continuous_learning.learning_policies[agent_id]
            assert policy.policy_type == policy_type
            
            # Verify policy-specific settings
            if policy_type == LearningPolicyType.CONSERVATIVE:
                assert policy.learning_rate <= 0.01
            elif policy_type == LearningPolicyType.AGGRESSIVE:
                assert policy.learning_rate >= 0.015
            
            logger.info(f"✅ {policy_type.value} policy configured correctly")
        
        # Test custom policy
        logger.info("🔧 Testing custom policy...")
        custom_policy = LearningPolicy(
            policy_id="custom_test",
            policy_type=LearningPolicyType.BALANCED,
            update_strategy=ModelUpdateStrategy.IMMEDIATE,
            learning_rate=0.025,
            exploration_rate=0.4
        )
        
        await continuous_learning.register_agent("custom_agent", custom_policy=custom_policy)
        
        stored_policy = continuous_learning.learning_policies["custom_agent"]
        assert stored_policy.learning_rate == 0.025
        assert stored_policy.exploration_rate == 0.4
        
        logger.info("✅ Custom policy test successful")
        logger.info("🎉 All learning policy tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Learning policy test failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("🚀 Starting Continuous Learning System Tests")
    
    tests = [
        ("Integration Test", test_continuous_learning_integration),
        ("Learning Policies Test", test_learning_policies)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name}")
        logger.info('='*50)
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Continuous learning system is working correctly.")
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)