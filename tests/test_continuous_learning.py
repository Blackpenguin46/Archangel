#!/usr/bin/env python3
"""
Tests for Continuous Learning and Human-in-the-Loop Systems
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

# Import the modules to test
from agents.continuous_learning import (
    ContinuousLearningSystem, LearningPolicy, LearningPolicyType, 
    ModelUpdateStrategy, LearningUpdate
)
from agents.human_in_the_loop import (
    HumanInTheLoopInterface, HumanFeedback, FeedbackType, 
    ValidationStatus, ValidationRequest, Priority
)
from agents.knowledge_distillation import (
    KnowledgeDistillationPipeline, DistillationType, BehaviorPattern,
    RelevanceLevel, DistillationResult, BehaviorAnalyzer
)
from agents.base_agent import Experience, Action, Team, Role
from agents.learning_system import LearningSystem, LearningConfig, PerformanceMetrics


class TestContinuousLearningSystem:
    """Test the continuous learning system"""
    
    @pytest.fixture
    async def learning_system(self):
        """Create a mock learning system"""
        system = Mock(spec=LearningSystem)
        system.initialize = AsyncMock()
        system.add_experience_to_replay_buffer = AsyncMock()
        system.get_agent_performance_history = AsyncMock(return_value=[])
        system._calculate_agent_performance = AsyncMock(return_value=PerformanceMetrics(
            agent_id="test_agent",
            success_rate=0.8,
            average_confidence=0.7,
            decision_speed=2.5,
            learning_rate=0.01,
            timestamp=datetime.now()
        ))
        return system
    
    @pytest.fixture
    async def human_interface(self):
        """Create a mock human interface"""
        interface = Mock(spec=HumanInTheLoopInterface)
        interface.initialize = AsyncMock()
        interface.register_validation_callback = Mock()
        return interface
    
    @pytest.fixture
    async def distillation_pipeline(self):
        """Create a mock distillation pipeline"""
        pipeline = Mock(spec=KnowledgeDistillationPipeline)
        pipeline.initialize = AsyncMock()
        pipeline.distill_agent_knowledge = AsyncMock(return_value=DistillationResult(
            distillation_id="test_distill",
            agent_id="test_agent",
            distillation_type=DistillationType.BEHAVIOR_PRUNING,
            input_experiences=100,
            output_experiences=80,
            pruned_behaviors=["bad_behavior_1", "bad_behavior_2"],
            extracted_patterns=[],
            compression_ratio=0.8,
            quality_score=0.75,
            processing_time=1.5,
            timestamp=datetime.now()
        ))
        return pipeline
    
    @pytest.fixture
    async def continuous_learning_system(self, learning_system, human_interface, distillation_pipeline):
        """Create a continuous learning system for testing"""
        system = ContinuousLearningSystem(
            learning_system=learning_system,
            human_interface=human_interface,
            distillation_pipeline=distillation_pipeline
        )
        await system.initialize()
        return system
    
    @pytest.mark.asyncio
    async def test_initialization(self, continuous_learning_system):
        """Test system initialization"""
        assert continuous_learning_system.initialized
        assert continuous_learning_system.learning_system is not None
        assert continuous_learning_system.human_interface is not None
        assert continuous_learning_system.distillation_pipeline is not None
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, continuous_learning_system):
        """Test agent registration for continuous learning"""
        agent_id = "test_agent_1"
        
        # Register agent with default policy
        await continuous_learning_system.register_agent(agent_id)
        
        assert agent_id in continuous_learning_system.learning_policies
        assert agent_id in continuous_learning_system.feedback_buffer
        assert agent_id in continuous_learning_system.experience_buffer
        assert agent_id in continuous_learning_system.update_history
        assert agent_id in continuous_learning_system.learning_tasks
        
        policy = continuous_learning_system.learning_policies[agent_id]
        assert policy.policy_type == LearningPolicyType.BALANCED
    
    @pytest.mark.asyncio
    async def test_custom_policy_registration(self, continuous_learning_system):
        """Test agent registration with custom policy"""
        agent_id = "test_agent_2"
        
        custom_policy = LearningPolicy(
            policy_id="custom_policy",
            policy_type=LearningPolicyType.AGGRESSIVE,
            update_strategy=ModelUpdateStrategy.IMMEDIATE,
            learning_rate=0.05,
            exploration_rate=0.3
        )
        
        await continuous_learning_system.register_agent(agent_id, custom_policy=custom_policy)
        
        stored_policy = continuous_learning_system.learning_policies[agent_id]
        assert stored_policy.policy_type == LearningPolicyType.AGGRESSIVE
        assert stored_policy.learning_rate == 0.05
        assert stored_policy.exploration_rate == 0.3
    
    @pytest.mark.asyncio
    async def test_experience_addition(self, continuous_learning_system):
        """Test adding experiences for learning"""
        agent_id = "test_agent"
        await continuous_learning_system.register_agent(agent_id)
        
        # Create test experience
        action = Action(
            action_id="test_action",
            action_type="reconnaissance",
            parameters={"target": "192.168.1.1"},
            timestamp=datetime.now()
        )
        
        experience = Experience(
            experience_id="exp_1",
            agent_id=agent_id,
            timestamp=datetime.now(),
            action_taken=action,
            success=True,
            reasoning="Test reasoning",
            outcome="Successful scan"
        )
        
        # Add experience
        await continuous_learning_system.add_experience(agent_id, experience)
        
        # Verify experience was added
        assert len(continuous_learning_system.experience_buffer[agent_id]) == 1
        assert continuous_learning_system.experience_buffer[agent_id][0] == experience
        
        # Verify learning system was called
        continuous_learning_system.learning_system.add_experience_to_replay_buffer.assert_called_once_with(experience)
    
    @pytest.mark.asyncio
    async def test_human_feedback_addition(self, continuous_learning_system):
        """Test adding human feedback"""
        agent_id = "test_agent"
        await continuous_learning_system.register_agent(agent_id)
        
        feedback = HumanFeedback(
            feedback_id="feedback_1",
            agent_id=agent_id,
            feedback_type=FeedbackType.PERFORMANCE_RATING,
            timestamp=datetime.now(),
            reviewer_id="human_1",
            performance_rating=0.8,
            comments="Good performance"
        )
        
        await continuous_learning_system.add_human_feedback(agent_id, feedback)
        
        assert len(continuous_learning_system.feedback_buffer[agent_id]) == 1
        assert continuous_learning_system.feedback_buffer[agent_id][0] == feedback
    
    @pytest.mark.asyncio
    async def test_learning_update_request(self, continuous_learning_system):
        """Test requesting learning updates"""
        agent_id = "test_agent"
        await continuous_learning_system.register_agent(agent_id)
        
        # Request immediate update
        update_id = await continuous_learning_system.request_learning_update(
            agent_id, "manual", immediate=True
        )
        
        assert update_id is not None
        assert len(continuous_learning_system.update_history[agent_id]) == 1
        
        # Request queued update
        update_id_2 = await continuous_learning_system.request_learning_update(
            agent_id, "scheduled", immediate=False
        )
        
        assert update_id_2 is not None
        assert len(continuous_learning_system.pending_updates[agent_id]) == 1
    
    @pytest.mark.asyncio
    async def test_knowledge_distillation_trigger(self, continuous_learning_system):
        """Test knowledge distillation triggering"""
        agent_id = "test_agent"
        await continuous_learning_system.register_agent(agent_id)
        
        # Add enough experiences to trigger distillation
        policy = continuous_learning_system.learning_policies[agent_id]
        for i in range(policy.distillation_frequency):
            experience = Experience(
                experience_id=f"exp_{i}",
                agent_id=agent_id,
                timestamp=datetime.now(),
                action_taken=Action(
                    action_id=f"action_{i}",
                    action_type="test",
                    parameters={},
                    timestamp=datetime.now()
                ),
                success=True,
                reasoning="Test",
                outcome="Success"
            )
            await continuous_learning_system.add_experience(agent_id, experience)
        
        # Verify distillation was called
        continuous_learning_system.distillation_pipeline.distill_agent_knowledge.assert_called()


class TestHumanInTheLoopInterface:
    """Test the human-in-the-loop interface"""
    
    @pytest.fixture
    async def human_interface(self):
        """Create a human interface for testing"""
        interface = HumanInTheLoopInterface()
        await interface.initialize()
        return interface
    
    @pytest.mark.asyncio
    async def test_initialization(self, human_interface):
        """Test interface initialization"""
        assert human_interface.initialized
        assert isinstance(human_interface.validation_queue, asyncio.Queue)
        assert isinstance(human_interface.feedback_queue, asyncio.Queue)
    
    @pytest.mark.asyncio
    async def test_action_validation_request(self, human_interface):
        """Test requesting action validation"""
        action = Action(
            action_id="test_action",
            action_type="exploit",
            parameters={"target": "vulnerable_service"},
            timestamp=datetime.now()
        )
        
        request_id = await human_interface.request_action_validation(
            agent_id="test_agent",
            action=action,
            context={"risk_level": "high"},
            reasoning="Attempting to exploit vulnerability",
            confidence_score=0.6
        )
        
        assert request_id is not None
        assert request_id in human_interface.pending_validations
        
        request = human_interface.pending_validations[request_id]
        assert request.agent_id == "test_agent"
        assert request.action == action
        assert request.confidence_score == 0.6
    
    @pytest.mark.asyncio
    async def test_auto_approval(self, human_interface):
        """Test auto-approval of high-confidence actions"""
        action = Action(
            action_id="safe_action",
            action_type="scan",
            parameters={"target": "192.168.1.1"},
            timestamp=datetime.now()
        )
        
        request_id = await human_interface.request_action_validation(
            agent_id="test_agent",
            action=action,
            context={"risk_level": "low"},
            reasoning="Safe network scan",
            confidence_score=0.95  # High confidence
        )
        
        # Should be auto-approved
        assert request_id not in human_interface.pending_validations
    
    @pytest.mark.asyncio
    async def test_feedback_provision(self, human_interface):
        """Test providing human feedback"""
        feedback_id = await human_interface.provide_feedback(
            agent_id="test_agent",
            feedback_type=FeedbackType.BEHAVIOR_TAGGING,
            reviewer_id="human_1",
            behavior_tags=["aggressive", "effective"],
            correctness_score=0.8,
            comments="Good tactical approach"
        )
        
        assert feedback_id is not None
        assert "test_agent" in human_interface.feedback_history
        assert len(human_interface.feedback_history["test_agent"]) == 1
        
        feedback = human_interface.feedback_history["test_agent"][0]
        assert feedback.feedback_type == FeedbackType.BEHAVIOR_TAGGING
        assert feedback.behavior_tags == ["aggressive", "effective"]
        assert feedback.correctness_score == 0.8
    
    @pytest.mark.asyncio
    async def test_action_validation(self, human_interface):
        """Test validating pending actions"""
        # First create a validation request
        action = Action(
            action_id="test_action",
            action_type="exploit",
            parameters={},
            timestamp=datetime.now()
        )
        
        request_id = await human_interface.request_action_validation(
            agent_id="test_agent",
            action=action,
            context={},
            reasoning="Test",
            confidence_score=0.5  # Low confidence to avoid auto-approval
        )
        
        # Now validate it
        success = await human_interface.validate_action(
            request_id=request_id,
            reviewer_id="human_1",
            status=ValidationStatus.APPROVED,
            comments="Approved after review"
        )
        
        assert success
        assert request_id not in human_interface.pending_validations
        assert "test_agent" in human_interface.feedback_history
    
    @pytest.mark.asyncio
    async def test_behavior_tagging(self, human_interface):
        """Test behavior tagging functionality"""
        feedback_id = await human_interface.tag_behavior(
            agent_id="test_agent",
            action_id="action_1",
            tags=["stealthy", "patient", "effective"],
            correctness_score=0.9,
            reviewer_id="expert_1",
            comments="Excellent stealth approach"
        )
        
        assert feedback_id is not None
        feedback = human_interface.feedback_history["test_agent"][0]
        assert feedback.behavior_tags == ["stealthy", "patient", "effective"]
        assert feedback.correctness_score == 0.9
    
    @pytest.mark.asyncio
    async def test_strategy_correction(self, human_interface):
        """Test strategy correction functionality"""
        original_strategy = {"aggression": 0.8, "stealth": 0.2}
        corrected_strategy = {"aggression": 0.4, "stealth": 0.6}
        
        correction_id = await human_interface.correct_strategy(
            agent_id="test_agent",
            original_strategy=original_strategy,
            corrected_strategy=corrected_strategy,
            reviewer_id="expert_1",
            reasoning=["Too aggressive", "Need more stealth"],
            confidence=0.9
        )
        
        assert correction_id is not None
        assert "test_agent" in human_interface.learning_corrections
        
        correction = human_interface.learning_corrections["test_agent"][0]
        assert correction.original_behavior == original_strategy
        assert correction.corrected_behavior == corrected_strategy
    
    @pytest.mark.asyncio
    async def test_pending_validations_query(self, human_interface):
        """Test querying pending validations"""
        # Create some validation requests
        for i in range(3):
            action = Action(
                action_id=f"action_{i}",
                action_type="test",
                parameters={},
                timestamp=datetime.now()
            )
            
            await human_interface.request_action_validation(
                agent_id=f"agent_{i}",
                action=action,
                context={},
                reasoning="Test",
                confidence_score=0.5  # Low confidence to avoid auto-approval
            )
        
        pending = await human_interface.get_pending_validations()
        assert len(pending) == 3
        
        # Test filtering by reviewer
        pending_filtered = await human_interface.get_pending_validations("reviewer_1")
        assert len(pending_filtered) == 3  # All should be unassigned high-priority
    
    @pytest.mark.asyncio
    async def test_feedback_statistics(self, human_interface):
        """Test feedback statistics generation"""
        # Add some feedback
        await human_interface.provide_feedback(
            agent_id="agent_1",
            feedback_type=FeedbackType.PERFORMANCE_RATING,
            reviewer_id="human_1",
            performance_rating=0.8
        )
        
        await human_interface.provide_feedback(
            agent_id="agent_2",
            feedback_type=FeedbackType.BEHAVIOR_TAGGING,
            reviewer_id="human_1",
            behavior_tags=["good"]
        )
        
        stats = await human_interface.get_feedback_statistics()
        
        assert stats["total_feedback"] == 2
        assert stats["active_agents"] == 2
        assert "validation_breakdown" in stats


class TestKnowledgeDistillationPipeline:
    """Test the knowledge distillation pipeline"""
    
    @pytest.fixture
    async def distillation_pipeline(self):
        """Create a distillation pipeline for testing"""
        pipeline = KnowledgeDistillationPipeline()
        await pipeline.initialize()
        return pipeline
    
    @pytest.fixture
    def sample_experiences(self):
        """Create sample experiences for testing"""
        experiences = []
        for i in range(20):
            action = Action(
                action_id=f"action_{i}",
                action_type="scan" if i % 2 == 0 else "exploit",
                parameters={"target": f"192.168.1.{i}"},
                timestamp=datetime.now()
            )
            
            experience = Experience(
                experience_id=f"exp_{i}",
                agent_id="test_agent",
                timestamp=datetime.now(),
                action_taken=action,
                success=i % 3 != 0,  # 2/3 success rate
                reasoning=f"Reasoning for action {i}",
                outcome=f"Outcome {i}"
            )
            experiences.append(experience)
        
        return experiences
    
    @pytest.mark.asyncio
    async def test_initialization(self, distillation_pipeline):
        """Test pipeline initialization"""
        assert distillation_pipeline.initialized
        assert distillation_pipeline.behavior_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_behavior_pruning(self, distillation_pipeline, sample_experiences):
        """Test behavior pruning distillation"""
        result = await distillation_pipeline.distill_agent_knowledge(
            agent_id="test_agent",
            experiences=sample_experiences,
            distillation_type=DistillationType.BEHAVIOR_PRUNING
        )
        
        assert result.distillation_type == DistillationType.BEHAVIOR_PRUNING
        assert result.input_experiences == len(sample_experiences)
        assert result.output_experiences <= result.input_experiences
        assert result.compression_ratio <= 1.0
        assert 0.0 <= result.quality_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_strategy_compression(self, distillation_pipeline, sample_experiences):
        """Test strategy compression distillation"""
        result = await distillation_pipeline.distill_agent_knowledge(
            agent_id="test_agent",
            experiences=sample_experiences,
            distillation_type=DistillationType.STRATEGY_COMPRESSION
        )
        
        assert result.distillation_type == DistillationType.STRATEGY_COMPRESSION
        assert result.compression_ratio <= 1.0
        assert len(result.extracted_patterns) >= 0
    
    @pytest.mark.asyncio
    async def test_experience_filtering(self, distillation_pipeline, sample_experiences):
        """Test experience filtering distillation"""
        result = await distillation_pipeline.distill_agent_knowledge(
            agent_id="test_agent",
            experiences=sample_experiences,
            distillation_type=DistillationType.EXPERIENCE_FILTERING
        )
        
        assert result.distillation_type == DistillationType.EXPERIENCE_FILTERING
        assert result.output_experiences <= result.input_experiences
    
    @pytest.mark.asyncio
    async def test_pattern_extraction(self, distillation_pipeline, sample_experiences):
        """Test pattern extraction distillation"""
        result = await distillation_pipeline.distill_agent_knowledge(
            agent_id="test_agent",
            experiences=sample_experiences,
            distillation_type=DistillationType.PATTERN_EXTRACTION
        )
        
        assert result.distillation_type == DistillationType.PATTERN_EXTRACTION
        assert isinstance(result.extracted_patterns, list)
    
    @pytest.mark.asyncio
    async def test_relevance_scoring(self, distillation_pipeline, sample_experiences):
        """Test relevance scoring distillation"""
        result = await distillation_pipeline.distill_agent_knowledge(
            agent_id="test_agent",
            experiences=sample_experiences,
            distillation_type=DistillationType.RELEVANCE_SCORING
        )
        
        assert result.distillation_type == DistillationType.RELEVANCE_SCORING
        assert result.compression_ratio == 1.0  # No compression in scoring
        
        # Check that patterns have confidence scores
        for pattern in result.extracted_patterns:
            assert hasattr(pattern, 'confidence_score')
            assert 0.0 <= pattern.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_human_feedback_integration(self, distillation_pipeline, sample_experiences):
        """Test integration of human feedback in distillation"""
        # Create human feedback
        human_feedback = [
            HumanFeedback(
                feedback_id="feedback_1",
                agent_id="test_agent",
                feedback_type=FeedbackType.BEHAVIOR_TAGGING,
                timestamp=datetime.now(),
                reviewer_id="human_1",
                action_id="action_0",
                correctness_score=0.9,
                behavior_tags=["excellent"]
            ),
            HumanFeedback(
                feedback_id="feedback_2",
                agent_id="test_agent",
                feedback_type=FeedbackType.BEHAVIOR_TAGGING,
                timestamp=datetime.now(),
                reviewer_id="human_1",
                action_id="action_1",
                correctness_score=0.2,
                behavior_tags=["poor"]
            )
        ]
        
        result = await distillation_pipeline.distill_agent_knowledge(
            agent_id="test_agent",
            experiences=sample_experiences,
            distillation_type=DistillationType.BEHAVIOR_PRUNING,
            human_feedback=human_feedback
        )
        
        # Should have pruned the poor behavior
        assert "action_1" in result.pruned_behaviors or result.output_experiences < result.input_experiences


class TestBehaviorAnalyzer:
    """Test the behavior analyzer"""
    
    @pytest.fixture
    def behavior_analyzer(self):
        """Create a behavior analyzer for testing"""
        return BehaviorAnalyzer()
    
    @pytest.fixture
    def sample_experiences(self):
        """Create sample experiences with patterns"""
        experiences = []
        
        # Create repeated patterns
        for i in range(15):
            if i < 5:
                # Successful scan pattern
                action = Action(
                    action_id=f"scan_{i}",
                    action_type="scan",
                    parameters={"target": "192.168.1.1"},
                    timestamp=datetime.now()
                )
                success = True
            elif i < 10:
                # Failed exploit pattern
                action = Action(
                    action_id=f"exploit_{i}",
                    action_type="exploit",
                    parameters={"target": "192.168.1.1"},
                    timestamp=datetime.now()
                )
                success = False
            else:
                # Mixed persistence pattern
                action = Action(
                    action_id=f"persist_{i}",
                    action_type="persistence",
                    parameters={"method": "backdoor"},
                    timestamp=datetime.now()
                )
                success = i % 2 == 0
            
            experience = Experience(
                experience_id=f"exp_{i}",
                agent_id="test_agent",
                timestamp=datetime.now(),
                action_taken=action,
                success=success,
                reasoning=f"Reasoning {i}",
                outcome=f"Outcome {i}"
            )
            experiences.append(experience)
        
        return experiences
    
    @pytest.mark.asyncio
    async def test_pattern_analysis(self, behavior_analyzer, sample_experiences):
        """Test behavior pattern analysis"""
        patterns = await behavior_analyzer.analyze_behavior_patterns(sample_experiences)
        
        assert len(patterns) >= 2  # Should find at least scan and exploit patterns
        
        # Check pattern properties
        for pattern in patterns:
            assert pattern.frequency >= 3  # Minimum frequency threshold
            assert 0.0 <= pattern.success_rate <= 1.0
            assert pattern.relevance_level in RelevanceLevel
            assert len(pattern.action_sequence) > 0
            assert 0.0 <= pattern.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_relevance_assessment(self, behavior_analyzer):
        """Test relevance level assessment"""
        # Test high success, high frequency -> CRITICAL
        relevance = behavior_analyzer._assess_relevance(0.9, 15, [])
        assert relevance == RelevanceLevel.CRITICAL
        
        # Test medium success, medium frequency -> IMPORTANT
        relevance = behavior_analyzer._assess_relevance(0.7, 8, [])
        assert relevance == RelevanceLevel.IMPORTANT
        
        # Test low success -> HARMFUL
        relevance = behavior_analyzer._assess_relevance(0.1, 10, [])
        assert relevance == RelevanceLevel.HARMFUL
    
    def test_confidence_calculation(self, behavior_analyzer):
        """Test pattern confidence calculation"""
        # High frequency, high success, low diversity -> high confidence
        confidence = behavior_analyzer._calculate_pattern_confidence(20, 0.9, 2)
        assert confidence > 0.7
        
        # Low frequency, low success, high diversity -> low confidence
        confidence = behavior_analyzer._calculate_pattern_confidence(3, 0.3, 10)
        assert confidence < 0.5


class TestIntegrationScenarios:
    """Test integration scenarios between components"""
    
    @pytest.fixture
    async def integrated_system(self):
        """Create an integrated system for testing"""
        # Create mock components
        learning_system = Mock(spec=LearningSystem)
        learning_system.initialize = AsyncMock()
        learning_system.add_experience_to_replay_buffer = AsyncMock()
        learning_system.get_agent_performance_history = AsyncMock(return_value=[])
        learning_system._calculate_agent_performance = AsyncMock(return_value=PerformanceMetrics(
            agent_id="test_agent",
            success_rate=0.8,
            average_confidence=0.7,
            decision_speed=2.5,
            learning_rate=0.01,
            timestamp=datetime.now()
        ))
        
        # Create real components
        human_interface = HumanInTheLoopInterface(learning_system)
        distillation_pipeline = KnowledgeDistillationPipeline()
        continuous_learning = ContinuousLearningSystem(
            learning_system=learning_system,
            human_interface=human_interface,
            distillation_pipeline=distillation_pipeline
        )
        
        # Initialize all components
        await human_interface.initialize()
        await distillation_pipeline.initialize()
        await continuous_learning.initialize()
        
        return {
            'continuous_learning': continuous_learning,
            'human_interface': human_interface,
            'distillation_pipeline': distillation_pipeline,
            'learning_system': learning_system
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_learning_cycle(self, integrated_system):
        """Test complete learning cycle with human feedback"""
        continuous_learning = integrated_system['continuous_learning']
        human_interface = integrated_system['human_interface']
        
        agent_id = "test_agent"
        
        # 1. Register agent
        await continuous_learning.register_agent(agent_id)
        
        # 2. Add experiences
        for i in range(10):
            action = Action(
                action_id=f"action_{i}",
                action_type="scan",
                parameters={"target": f"192.168.1.{i}"},
                timestamp=datetime.now()
            )
            
            experience = Experience(
                experience_id=f"exp_{i}",
                agent_id=agent_id,
                timestamp=datetime.now(),
                action_taken=action,
                success=i % 2 == 0,
                reasoning=f"Scan target {i}",
                outcome=f"Result {i}"
            )
            
            await continuous_learning.add_experience(agent_id, experience)
        
        # 3. Add human feedback
        feedback = HumanFeedback(
            feedback_id="feedback_1",
            agent_id=agent_id,
            feedback_type=FeedbackType.PERFORMANCE_RATING,
            timestamp=datetime.now(),
            reviewer_id="human_1",
            performance_rating=0.8,
            comments="Good scanning technique"
        )
        
        await continuous_learning.add_human_feedback(agent_id, feedback)
        
        # 4. Request learning update
        update_id = await continuous_learning.request_learning_update(
            agent_id, "integration_test", immediate=True
        )
        
        # 5. Verify update was processed
        assert update_id is not None
        assert len(continuous_learning.update_history[agent_id]) == 1
        
        update = continuous_learning.update_history[agent_id][0]
        assert update.agent_id == agent_id
        assert update.trigger == "integration_test"
        assert len(update.feedback_incorporated) == 1
    
    @pytest.mark.asyncio
    async def test_human_validation_workflow(self, integrated_system):
        """Test human validation workflow"""
        human_interface = integrated_system['human_interface']
        
        # 1. Request action validation
        action = Action(
            action_id="risky_action",
            action_type="exploit",
            parameters={"target": "critical_server"},
            timestamp=datetime.now()
        )
        
        request_id = await human_interface.request_action_validation(
            agent_id="test_agent",
            action=action,
            context={"risk_level": "high"},
            reasoning="Attempting critical exploit",
            confidence_score=0.6
        )
        
        # 2. Validate the action
        success = await human_interface.validate_action(
            request_id=request_id,
            reviewer_id="security_expert",
            status=ValidationStatus.MODIFIED,
            comments="Approved with modifications",
            modifications={"safety_check": True}
        )
        
        assert success
        
        # 3. Verify feedback was recorded
        feedback_history = await human_interface.get_agent_feedback_history("test_agent")
        assert len(feedback_history) == 1
        
        feedback = feedback_history[0]
        assert feedback.validation_status == ValidationStatus.MODIFIED
        assert feedback.strategy_modifications == {"safety_check": True}
    
    @pytest.mark.asyncio
    async def test_knowledge_distillation_with_feedback(self, integrated_system):
        """Test knowledge distillation with human feedback integration"""
        distillation_pipeline = integrated_system['distillation_pipeline']
        
        # Create experiences
        experiences = []
        for i in range(20):
            action = Action(
                action_id=f"action_{i}",
                action_type="scan" if i % 2 == 0 else "exploit",
                parameters={"target": f"192.168.1.{i}"},
                timestamp=datetime.now()
            )
            
            experience = Experience(
                experience_id=f"exp_{i}",
                agent_id="test_agent",
                timestamp=datetime.now(),
                action_taken=action,
                success=i % 3 != 0,
                reasoning=f"Action {i}",
                outcome=f"Outcome {i}"
            )
            experiences.append(experience)
        
        # Create human feedback
        human_feedback = [
            HumanFeedback(
                feedback_id="feedback_1",
                agent_id="test_agent",
                feedback_type=FeedbackType.BEHAVIOR_TAGGING,
                timestamp=datetime.now(),
                reviewer_id="expert",
                action_id="action_0",
                correctness_score=0.9,
                behavior_tags=["excellent", "stealthy"]
            ),
            HumanFeedback(
                feedback_id="feedback_2",
                agent_id="test_agent",
                feedback_type=FeedbackType.BEHAVIOR_TAGGING,
                timestamp=datetime.now(),
                reviewer_id="expert",
                action_id="action_1",
                correctness_score=0.2,
                behavior_tags=["poor", "noisy"]
            )
        ]
        
        # Perform distillation
        result = await distillation_pipeline.distill_agent_knowledge(
            agent_id="test_agent",
            experiences=experiences,
            distillation_type=DistillationType.BEHAVIOR_PRUNING,
            human_feedback=human_feedback
        )
        
        # Verify results
        assert result.input_experiences == 20
        assert result.quality_score > 0.0
        assert len(result.extracted_patterns) >= 0
        
        # Should have incorporated human feedback
        assert result.output_experiences <= result.input_experiences


if __name__ == "__main__":
    pytest.main([__file__, "-v"])