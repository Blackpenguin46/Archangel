#!/usr/bin/env python3
"""
Tests for the Experience Replay System
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from agents.experience_replay import (
    ExperienceReplaySystem, ReplayConfig, ExperienceMetadata, KnowledgeDistillation,
    ReplayBatch, PrioritizedReplayBuffer, ReplayMode, KnowledgeType
)
from agents.base_agent import Experience, Team, Role
from memory.memory_manager import MemoryManager

class TestPrioritizedReplayBuffer:
    """Test the prioritized replay buffer"""
    
    def test_buffer_initialization(self):
        """Test buffer initialization"""
        buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6)
        
        assert buffer.max_size == 100
        assert buffer.alpha == 0.6
        assert buffer.size() == 0
        assert len(buffer.buffer) == 0
    
    def test_add_experience(self):
        """Test adding experiences to buffer"""
        buffer = PrioritizedReplayBuffer(max_size=10, alpha=0.6)
        
        # Create mock experience and metadata
        experience = Mock(spec=Experience)
        experience.experience_id = "exp1"
        
        metadata = ExperienceMetadata(
            priority=0.8,
            temporal_relevance=0.7,
            strategic_value=0.6,
            learning_potential=0.9,
            diversity_score=0.5,
            success_impact=0.8,
            replay_count=0,
            last_replayed=None,
            knowledge_extracted=False
        )
        
        buffer.add(experience, metadata)
        
        assert buffer.size() == 1
        assert buffer.buffer[0] == experience
        assert buffer.metadata[0] == metadata
        assert buffer.priorities[0] == 0.8
    
    def test_buffer_overflow(self):
        """Test buffer overflow behavior"""
        buffer = PrioritizedReplayBuffer(max_size=2, alpha=0.6)
        
        # Add 3 experiences
        experiences = []
        for i in range(3):
            exp = Mock(spec=Experience)
            exp.experience_id = f"exp{i}"
            
            metadata = ExperienceMetadata(
                priority=0.5 + i * 0.1,
                temporal_relevance=0.5,
                strategic_value=0.5,
                learning_potential=0.5,
                diversity_score=0.5,
                success_impact=0.5,
                replay_count=0,
                last_replayed=None,
                knowledge_extracted=False
            )
            
            experiences.append(exp)
            buffer.add(exp, metadata)
        
        # Buffer should only contain 2 experiences
        assert buffer.size() == 2
        # Should contain exp1 and exp2 (exp0 was overwritten)
        assert buffer.buffer[0] == experiences[1]
        assert buffer.buffer[1] == experiences[2]
    
    def test_sample_experiences(self):
        """Test sampling experiences from buffer"""
        buffer = PrioritizedReplayBuffer(max_size=10, alpha=0.6)
        
        # Add experiences with different priorities
        experiences = []
        for i in range(5):
            exp = Mock(spec=Experience)
            exp.experience_id = f"exp{i}"
            
            metadata = ExperienceMetadata(
                priority=0.2 + i * 0.2,  # Increasing priorities
                temporal_relevance=0.5,
                strategic_value=0.5,
                learning_potential=0.5,
                diversity_score=0.5,
                success_impact=0.5,
                replay_count=0,
                last_replayed=None,
                knowledge_extracted=False
            )
            
            experiences.append(exp)
            buffer.add(exp, metadata)
        
        # Sample batch
        sampled_exp, sampled_meta, weights, indices = buffer.sample(3, beta=0.4)
        
        assert len(sampled_exp) == 3
        assert len(sampled_meta) == 3
        assert len(weights) == 3
        assert len(indices) == 3
        
        # All weights should be positive
        assert all(w > 0 for w in weights)
        
        # All indices should be valid
        assert all(0 <= idx < buffer.size() for idx in indices)
    
    def test_update_priorities(self):
        """Test updating priorities"""
        buffer = PrioritizedReplayBuffer(max_size=10, alpha=0.6)
        
        # Add experience
        exp = Mock(spec=Experience)
        metadata = ExperienceMetadata(
            priority=0.5,
            temporal_relevance=0.5,
            strategic_value=0.5,
            learning_potential=0.5,
            diversity_score=0.5,
            success_impact=0.5,
            replay_count=0,
            last_replayed=None,
            knowledge_extracted=False
        )
        
        buffer.add(exp, metadata)
        
        # Update priority
        buffer.update_priorities([0], [0.9])
        
        assert buffer.priorities[0] == 0.9
    
    def test_get_max_priority(self):
        """Test getting maximum priority"""
        buffer = PrioritizedReplayBuffer(max_size=10, alpha=0.6)
        
        # Empty buffer should return 1.0
        assert buffer.get_max_priority() == 1.0
        
        # Add experiences with different priorities
        priorities = [0.3, 0.8, 0.5]
        for i, priority in enumerate(priorities):
            exp = Mock(spec=Experience)
            metadata = ExperienceMetadata(
                priority=priority,
                temporal_relevance=0.5,
                strategic_value=0.5,
                learning_potential=0.5,
                diversity_score=0.5,
                success_impact=0.5,
                replay_count=0,
                last_replayed=None,
                knowledge_extracted=False
            )
            buffer.add(exp, metadata)
        
        assert buffer.get_max_priority() == 0.8

class TestExperienceReplaySystem:
    """Test the main experience replay system"""
    
    @pytest.fixture
    async def replay_system(self):
        """Create a replay system for testing"""
        config = ReplayConfig(
            buffer_size=100,
            batch_size=10,
            replay_frequency=1,  # Fast for testing
            min_replay_size=5
        )
        
        # Mock memory manager
        memory_manager = Mock(spec=MemoryManager)
        memory_manager.initialize = AsyncMock()
        
        system = ExperienceReplaySystem(config, memory_manager)
        await system.initialize()
        
        yield system
        
        await system.shutdown()
    
    @pytest.fixture
    def mock_experiences(self):
        """Create mock experiences for testing"""
        experiences = []
        
        for i in range(10):
            exp = Mock(spec=Experience)
            exp.experience_id = f"exp_{i}"
            exp.timestamp = datetime.now() - timedelta(hours=i)
            exp.success = i % 2 == 0  # Alternate success/failure
            exp.confidence_score = 0.5 + (i % 5) * 0.1
            exp.lessons_learned = [f"lesson_{i}"] if i % 3 == 0 else []
            exp.mitre_attack_mapping = [f"T{1000 + i}"] if i % 4 == 0 else []
            
            experiences.append(exp)
        
        return experiences
    
    @pytest.mark.asyncio
    async def test_replay_system_initialization(self, replay_system):
        """Test replay system initialization"""
        assert replay_system.initialized
        assert replay_system.running
        assert replay_system.config is not None
        assert replay_system.replay_buffer is not None
    
    @pytest.mark.asyncio
    async def test_add_experience(self, replay_system, mock_experiences):
        """Test adding experience to replay system"""
        experience = mock_experiences[0]
        
        initial_size = replay_system.replay_buffer.size()
        
        await replay_system.add_experience(experience)
        
        assert replay_system.replay_buffer.size() == initial_size + 1
    
    @pytest.mark.asyncio
    async def test_add_experience_with_priority(self, replay_system, mock_experiences):
        """Test adding experience with custom priority"""
        experience = mock_experiences[0]
        custom_priority = 0.9
        
        await replay_system.add_experience(experience, custom_priority)
        
        # Check that priority was used
        assert replay_system.replay_buffer.priorities[0] == custom_priority
    
    @pytest.mark.asyncio
    async def test_calculate_experience_priority(self, replay_system, mock_experiences):
        """Test experience priority calculation"""
        experience = mock_experiences[0]
        
        priority = await replay_system._calculate_experience_priority(experience)
        
        assert 0.01 <= priority <= 10.0
        assert isinstance(priority, float)
    
    @pytest.mark.asyncio
    async def test_sample_replay_batch_prioritized(self, replay_system, mock_experiences):
        """Test sampling prioritized replay batch"""
        # Add experiences
        for exp in mock_experiences[:5]:
            await replay_system.add_experience(exp)
        
        batch = await replay_system.sample_replay_batch(
            batch_size=3,
            replay_mode=ReplayMode.PRIORITIZED
        )
        
        assert isinstance(batch, ReplayBatch)
        assert len(batch.experiences) == 3
        assert len(batch.batch_metadata) == 3
        assert batch.replay_mode == ReplayMode.PRIORITIZED
        assert batch.batch_id is not None
    
    @pytest.mark.asyncio
    async def test_sample_replay_batch_random(self, replay_system, mock_experiences):
        """Test sampling random replay batch"""
        # Add experiences
        for exp in mock_experiences[:5]:
            await replay_system.add_experience(exp)
        
        batch = await replay_system.sample_replay_batch(
            batch_size=3,
            replay_mode=ReplayMode.RANDOM
        )
        
        assert isinstance(batch, ReplayBatch)
        assert len(batch.experiences) == 3
        assert batch.replay_mode == ReplayMode.RANDOM
    
    @pytest.mark.asyncio
    async def test_sample_replay_batch_temporal(self, replay_system, mock_experiences):
        """Test sampling temporal replay batch"""
        # Add experiences
        for exp in mock_experiences[:5]:
            await replay_system.add_experience(exp)
        
        batch = await replay_system.sample_replay_batch(
            batch_size=3,
            replay_mode=ReplayMode.TEMPORAL
        )
        
        assert isinstance(batch, ReplayBatch)
        assert len(batch.experiences) == 3
        assert batch.replay_mode == ReplayMode.TEMPORAL
    
    @pytest.mark.asyncio
    async def test_sample_replay_batch_strategic(self, replay_system, mock_experiences):
        """Test sampling strategic replay batch"""
        # Add experiences
        for exp in mock_experiences[:5]:
            await replay_system.add_experience(exp)
        
        learning_objectives = ["improve_reconnaissance", "enhance_defense"]
        
        batch = await replay_system.sample_replay_batch(
            batch_size=3,
            replay_mode=ReplayMode.STRATEGIC,
            learning_objectives=learning_objectives
        )
        
        assert isinstance(batch, ReplayBatch)
        assert len(batch.experiences) == 3
        assert batch.replay_mode == ReplayMode.STRATEGIC
        assert batch.learning_objectives == learning_objectives
    
    @pytest.mark.asyncio
    async def test_process_replay_batch(self, replay_system, mock_experiences):
        """Test processing replay batch"""
        # Add experiences
        for exp in mock_experiences[:5]:
            await replay_system.add_experience(exp)
        
        # Sample batch
        batch = await replay_system.sample_replay_batch(batch_size=3)
        
        # Process batch
        results = await replay_system.process_replay_batch(batch)
        
        assert results['success'] is True
        assert results['batch_id'] == batch.batch_id
        assert results['processed_experiences'] == len(batch.experiences)
        assert 'learning_outcomes' in results
        assert 'knowledge_extracted' in results
        assert 'pattern_discoveries' in results
        assert results['processing_time'] > 0
    
    @pytest.mark.asyncio
    async def test_distill_tactical_knowledge(self, replay_system, mock_experiences):
        """Test tactical knowledge distillation"""
        experiences = mock_experiences[:5]
        
        knowledge = await replay_system.distill_tactical_knowledge(
            experiences, KnowledgeType.TACTICAL
        )
        
        if knowledge:  # May return None if no patterns found
            assert isinstance(knowledge, KnowledgeDistillation)
            assert knowledge.knowledge_type == KnowledgeType.TACTICAL
            assert len(knowledge.source_experiences) == len(experiences)
            assert 0.0 <= knowledge.confidence_score <= 1.0
            assert knowledge.knowledge_id in replay_system.distilled_knowledge
    
    @pytest.mark.asyncio
    async def test_extract_learning_from_experience(self, replay_system, mock_experiences):
        """Test learning extraction from experience"""
        experience = mock_experiences[0]
        
        metadata = ExperienceMetadata(
            priority=0.8,
            temporal_relevance=0.7,
            strategic_value=0.6,
            learning_potential=0.9,
            diversity_score=0.5,
            success_impact=0.8,
            replay_count=0,
            last_replayed=None,
            knowledge_extracted=False
        )
        
        learning_outcome = await replay_system._extract_learning_from_experience(
            experience, metadata, ["improve_tactics"]
        )
        
        if learning_outcome:  # May return None if no learning extracted
            assert 'experience_id' in learning_outcome
            assert 'learning_type' in learning_outcome
            assert 'insights' in learning_outcome
            assert 'patterns' in learning_outcome
            assert 'confidence' in learning_outcome
    
    @pytest.mark.asyncio
    async def test_extract_knowledge_patterns(self, replay_system, mock_experiences):
        """Test knowledge pattern extraction"""
        experience = mock_experiences[0]
        
        patterns = await replay_system._extract_knowledge_patterns(experience)
        
        if patterns:  # May return None if no patterns found
            assert 'pattern_id' in patterns
            assert 'pattern_type' in patterns
            assert 'extracted_patterns' in patterns
            assert 'confidence' in patterns
    
    @pytest.mark.asyncio
    async def test_discover_patterns(self, replay_system, mock_experiences):
        """Test pattern discovery across experiences"""
        experience = mock_experiences[0]
        batch_experiences = mock_experiences[:5]
        
        patterns = await replay_system._discover_patterns(experience, batch_experiences)
        
        assert isinstance(patterns, list)
        # May be empty if no patterns discovered
    
    def test_calculate_temporal_relevance(self, replay_system, mock_experiences):
        """Test temporal relevance calculation"""
        experience = mock_experiences[0]
        
        relevance = replay_system._calculate_temporal_relevance(experience)
        
        assert 0.0 <= relevance <= 1.0
        assert isinstance(relevance, float)
    
    def test_calculate_strategic_value(self, replay_system, mock_experiences):
        """Test strategic value calculation"""
        experience = mock_experiences[0]
        
        value = replay_system._calculate_strategic_value(experience)
        
        assert 0.0 <= value <= 1.0
        assert isinstance(value, float)
    
    def test_calculate_learning_potential(self, replay_system, mock_experiences):
        """Test learning potential calculation"""
        experience = mock_experiences[0]
        
        potential = replay_system._calculate_learning_potential(experience)
        
        assert 0.0 <= potential <= 1.0
        assert isinstance(potential, float)
    
    def test_calculate_success_impact(self, replay_system, mock_experiences):
        """Test success impact calculation"""
        successful_exp = mock_experiences[0]  # Even indices are successful
        failed_exp = mock_experiences[1]      # Odd indices are failed
        
        success_impact = replay_system._calculate_success_impact(successful_exp)
        failure_impact = replay_system._calculate_success_impact(failed_exp)
        
        assert 0.0 <= success_impact <= 1.0
        assert 0.0 <= failure_impact <= 1.0
        assert isinstance(success_impact, float)
        assert isinstance(failure_impact, float)
    
    def test_calculate_batch_diversity(self, replay_system, mock_experiences):
        """Test batch diversity calculation"""
        batch_experiences = mock_experiences[:5]
        
        diversity = replay_system._calculate_batch_diversity(batch_experiences)
        
        assert 0.0 <= diversity <= 1.0
        assert isinstance(diversity, float)
    
    def test_calculate_batch_quality(self, replay_system, mock_experiences):
        """Test batch quality calculation"""
        batch_experiences = mock_experiences[:3]
        
        # Create metadata for experiences
        metadata = []
        for i, exp in enumerate(batch_experiences):
            meta = ExperienceMetadata(
                priority=0.5 + i * 0.1,
                temporal_relevance=0.6,
                strategic_value=0.7,
                learning_potential=0.8,
                diversity_score=0.5,
                success_impact=0.6,
                replay_count=0,
                last_replayed=None,
                knowledge_extracted=False
            )
            metadata.append(meta)
        
        quality = replay_system._calculate_batch_quality(batch_experiences, metadata)
        
        assert 0.0 <= quality <= 10.0  # Can be higher than 1.0 due to priority scaling
        assert isinstance(quality, float)
    
    @pytest.mark.asyncio
    async def test_get_replay_statistics(self, replay_system, mock_experiences):
        """Test getting replay statistics"""
        # Add some experiences
        for exp in mock_experiences[:3]:
            await replay_system.add_experience(exp)
        
        stats = await replay_system.get_replay_statistics()
        
        assert "replay_system" in stats
        assert "statistics" in stats
        assert "recent_replays" in stats
        
        replay_info = stats["replay_system"]
        assert replay_info["initialized"] is True
        assert replay_info["running"] is True
        assert replay_info["buffer_size"] == 3
        assert "config" in replay_info

class TestExperienceMetadata:
    """Test experience metadata functionality"""
    
    def test_metadata_creation(self):
        """Test creating experience metadata"""
        metadata = ExperienceMetadata(
            priority=0.8,
            temporal_relevance=0.7,
            strategic_value=0.6,
            learning_potential=0.9,
            diversity_score=0.5,
            success_impact=0.8,
            replay_count=0,
            last_replayed=None,
            knowledge_extracted=False
        )
        
        assert metadata.priority == 0.8
        assert metadata.temporal_relevance == 0.7
        assert metadata.strategic_value == 0.6
        assert metadata.learning_potential == 0.9
        assert metadata.diversity_score == 0.5
        assert metadata.success_impact == 0.8
        assert metadata.replay_count == 0
        assert metadata.last_replayed is None
        assert metadata.knowledge_extracted is False
    
    def test_metadata_update(self):
        """Test updating metadata"""
        metadata = ExperienceMetadata(
            priority=0.5,
            temporal_relevance=0.5,
            strategic_value=0.5,
            learning_potential=0.5,
            diversity_score=0.5,
            success_impact=0.5,
            replay_count=0,
            last_replayed=None,
            knowledge_extracted=False
        )
        
        # Update replay count and timestamp
        metadata.replay_count += 1
        metadata.last_replayed = datetime.now()
        metadata.knowledge_extracted = True
        
        assert metadata.replay_count == 1
        assert metadata.last_replayed is not None
        assert metadata.knowledge_extracted is True

class TestKnowledgeDistillation:
    """Test knowledge distillation functionality"""
    
    def test_knowledge_distillation_creation(self):
        """Test creating knowledge distillation"""
        knowledge = KnowledgeDistillation(
            knowledge_id="knowledge_1",
            knowledge_type=KnowledgeType.TACTICAL,
            source_experiences=["exp_1", "exp_2", "exp_3"],
            distilled_patterns={"pattern1": "value1", "pattern2": "value2"},
            confidence_score=0.85,
            applicability_scope={"teams": ["red"], "roles": ["recon"]},
            validation_results={"test1": "passed"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            usage_count=0,
            effectiveness_score=0.0
        )
        
        assert knowledge.knowledge_id == "knowledge_1"
        assert knowledge.knowledge_type == KnowledgeType.TACTICAL
        assert len(knowledge.source_experiences) == 3
        assert "pattern1" in knowledge.distilled_patterns
        assert knowledge.confidence_score == 0.85
        assert knowledge.usage_count == 0
    
    def test_knowledge_distillation_update(self):
        """Test updating knowledge distillation"""
        knowledge = KnowledgeDistillation(
            knowledge_id="knowledge_1",
            knowledge_type=KnowledgeType.STRATEGIC,
            source_experiences=["exp_1"],
            distilled_patterns={},
            confidence_score=0.5,
            applicability_scope={},
            validation_results={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            usage_count=0,
            effectiveness_score=0.0
        )
        
        # Update usage and effectiveness
        knowledge.usage_count += 1
        knowledge.effectiveness_score = 0.7
        knowledge.updated_at = datetime.now()
        
        assert knowledge.usage_count == 1
        assert knowledge.effectiveness_score == 0.7

class TestReplayBatch:
    """Test replay batch functionality"""
    
    def test_replay_batch_creation(self):
        """Test creating replay batch"""
        experiences = [Mock(spec=Experience) for _ in range(3)]
        metadata = [Mock(spec=ExperienceMetadata) for _ in range(3)]
        
        batch = ReplayBatch(
            batch_id="batch_1",
            experiences=experiences,
            batch_metadata=metadata,
            replay_mode=ReplayMode.PRIORITIZED,
            learning_objectives=["objective1", "objective2"],
            expected_outcomes={"outcome1": 0.8},
            timestamp=datetime.now()
        )
        
        assert batch.batch_id == "batch_1"
        assert len(batch.experiences) == 3
        assert len(batch.batch_metadata) == 3
        assert batch.replay_mode == ReplayMode.PRIORITIZED
        assert len(batch.learning_objectives) == 2
        assert "outcome1" in batch.expected_outcomes

class TestIntegration:
    """Integration tests for experience replay system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_replay_cycle(self):
        """Test complete replay cycle"""
        # Create replay system
        config = ReplayConfig(
            buffer_size=20,
            batch_size=5,
            min_replay_size=3,
            replay_frequency=1
        )
        
        memory_manager = Mock(spec=MemoryManager)
        memory_manager.initialize = AsyncMock()
        
        replay_system = ExperienceReplaySystem(config, memory_manager)
        await replay_system.initialize()
        
        try:
            # Add experiences
            experiences = []
            for i in range(10):
                exp = Mock(spec=Experience)
                exp.experience_id = f"exp_{i}"
                exp.timestamp = datetime.now() - timedelta(hours=i)
                exp.success = i % 2 == 0
                exp.confidence_score = 0.5 + (i % 5) * 0.1
                exp.lessons_learned = [f"lesson_{i}"] if i % 3 == 0 else []
                exp.mitre_attack_mapping = [f"T{1000 + i}"] if i % 4 == 0 else []
                
                experiences.append(exp)
                await replay_system.add_experience(exp)
            
            # Sample batch
            batch = await replay_system.sample_replay_batch(batch_size=5)
            
            assert len(batch.experiences) == 5
            assert len(batch.batch_metadata) == 5
            
            # Process batch
            results = await replay_system.process_replay_batch(batch)
            
            assert results['success'] is True
            assert results['processed_experiences'] == 5
            
            # Distill knowledge
            knowledge = await replay_system.distill_tactical_knowledge(
                experiences[:5], KnowledgeType.TACTICAL
            )
            
            # Get statistics
            stats = await replay_system.get_replay_statistics()
            
            assert stats["replay_system"]["buffer_size"] == 10
            assert stats["statistics"]["total_replays"] >= 1
            
        finally:
            await replay_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_priority_based_sampling(self):
        """Test that higher priority experiences are sampled more frequently"""
        config = ReplayConfig(buffer_size=100, batch_size=10)
        memory_manager = Mock(spec=MemoryManager)
        memory_manager.initialize = AsyncMock()
        
        replay_system = ExperienceReplaySystem(config, memory_manager)
        await replay_system.initialize()
        
        try:
            # Add experiences with different priorities
            high_priority_exp = Mock(spec=Experience)
            high_priority_exp.experience_id = "high_priority"
            high_priority_exp.timestamp = datetime.now()
            high_priority_exp.success = True
            high_priority_exp.confidence_score = 0.9
            high_priority_exp.lessons_learned = ["important_lesson"]
            high_priority_exp.mitre_attack_mapping = ["T1001"]
            
            low_priority_exp = Mock(spec=Experience)
            low_priority_exp.experience_id = "low_priority"
            low_priority_exp.timestamp = datetime.now() - timedelta(days=1)
            low_priority_exp.success = False
            low_priority_exp.confidence_score = 0.3
            low_priority_exp.lessons_learned = []
            low_priority_exp.mitre_attack_mapping = []
            
            # Add experiences
            await replay_system.add_experience(high_priority_exp)
            await replay_system.add_experience(low_priority_exp)
            
            # Sample multiple batches and count occurrences
            high_priority_count = 0
            low_priority_count = 0
            
            for _ in range(20):  # Sample 20 batches
                batch = await replay_system.sample_replay_batch(
                    batch_size=1, replay_mode=ReplayMode.PRIORITIZED
                )
                
                if batch.experiences:
                    if batch.experiences[0].experience_id == "high_priority":
                        high_priority_count += 1
                    elif batch.experiences[0].experience_id == "low_priority":
                        low_priority_count += 1
            
            # High priority experience should be sampled more frequently
            assert high_priority_count > low_priority_count
            
        finally:
            await replay_system.shutdown()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])