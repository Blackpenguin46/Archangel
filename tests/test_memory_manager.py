#!/usr/bin/env python3
"""
Tests for Memory Manager functionality
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil

from memory.memory_manager import (
    MemoryManager, MemoryConfig, create_memory_manager
)
from memory.vector_memory import VectorDBType
from agents.base_agent import Experience, Team, Role

@pytest.fixture
def temp_dirs():
    """Create temporary directories"""
    vector_dir = tempfile.mkdtemp()
    knowledge_dir = tempfile.mkdtemp()
    yield vector_dir, knowledge_dir
    shutil.rmtree(vector_dir)
    shutil.rmtree(knowledge_dir)

@pytest.fixture
def memory_config(temp_dirs):
    """Create test memory configuration"""
    vector_dir, knowledge_dir = temp_dirs
    return MemoryConfig(
        vector_db_type=VectorDBType.MEMORY,  # Use in-memory for testing
        vector_db_path=vector_dir,
        knowledge_base_path=knowledge_dir,
        embedding_model="test-model",
        max_memory_age=timedelta(days=1),
        cleanup_interval=timedelta(seconds=1),  # Fast cleanup for testing
        clustering_enabled=False  # Disable for testing
    )

@pytest.fixture
def memory_manager(memory_config):
    """Create test memory manager"""
    return MemoryManager(memory_config)

@pytest.fixture
def sample_experience():
    """Create sample experience"""
    return Experience(
        experience_id="exp_001",
        agent_id="agent_001",
        timestamp=datetime.now(),
        context=Mock(),
        action_taken=Mock(primary_action="test_action", action_type="reconnaissance"),
        reasoning=Mock(),
        outcome=Mock(success=True, outcome="successful reconnaissance"),
        success=True,
        lessons_learned=["Reconnaissance was effective", "Target responded to scanning"],
        mitre_attack_mapping=["T1595"],
        confidence_score=0.85
    )

@pytest.mark.asyncio
async def test_memory_manager_initialization(memory_manager):
    """Test memory manager initialization"""
    assert not memory_manager.initialized
    assert not memory_manager.running
    
    await memory_manager.initialize()
    
    assert memory_manager.initialized
    assert memory_manager.running
    assert memory_manager.vector_memory.initialized
    assert memory_manager.knowledge_base.initialized

@pytest.mark.asyncio
async def test_store_agent_experience(memory_manager, sample_experience):
    """Test storing agent experience"""
    await memory_manager.initialize()
    
    experience_id = await memory_manager.store_agent_experience("agent_001", sample_experience)
    
    assert experience_id == sample_experience.experience_id
    assert memory_manager.stats['experiences_stored'] == 1

@pytest.mark.asyncio
async def test_retrieve_similar_experiences(memory_manager, sample_experience):
    """Test retrieving similar experiences"""
    await memory_manager.initialize()
    
    # Store experience
    await memory_manager.store_agent_experience("agent_001", sample_experience)
    
    # Retrieve similar experiences
    results = await memory_manager.retrieve_similar_experiences(
        query_text="reconnaissance action",
        agent_id="agent_001",
        max_results=5
    )
    
    assert len(results) > 0
    assert memory_manager.stats['queries_processed'] == 1

@pytest.mark.asyncio
async def test_get_tactical_knowledge(memory_manager):
    """Test getting tactical knowledge"""
    await memory_manager.initialize()
    
    # Get tactical knowledge
    knowledge = await memory_manager.get_tactical_knowledge(
        query="reconnaissance",
        knowledge_type="both"
    )
    
    assert "attack_patterns" in knowledge
    assert "defense_strategies" in knowledge
    assert isinstance(knowledge["attack_patterns"], list)
    assert isinstance(knowledge["defense_strategies"], list)

@pytest.mark.asyncio
async def test_cluster_agent_memories(memory_manager, sample_experience):
    """Test clustering agent memories"""
    await memory_manager.initialize()
    
    # Store multiple experiences
    for i in range(3):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            agent_id="agent_001",
            timestamp=datetime.now(),
            context=Mock(),
            action_taken=Mock(primary_action=f"action_{i}", action_type="test"),
            reasoning=Mock(),
            outcome=Mock(success=True, outcome="success"),
            success=True,
            lessons_learned=[f"Lesson {i}"],
            mitre_attack_mapping=["T1595"],
            confidence_score=0.8
        )
        await memory_manager.store_agent_experience("agent_001", exp)
    
    # Cluster memories
    clusters = await memory_manager.cluster_agent_memories("agent_001")
    
    assert isinstance(clusters, list)
    # May be empty if clustering threshold not met

@pytest.mark.asyncio
async def test_get_lessons_learned(memory_manager, sample_experience):
    """Test getting lessons learned"""
    await memory_manager.initialize()
    
    # Store experience (which should extract lessons)
    await memory_manager.store_agent_experience("agent_001", sample_experience)
    
    # Get lessons learned
    lessons = await memory_manager.get_lessons_learned("general")
    
    assert isinstance(lessons, list)
    # May be empty depending on lesson extraction

@pytest.mark.asyncio
async def test_update_agent_knowledge(memory_manager):
    """Test updating agent knowledge"""
    await memory_manager.initialize()
    
    knowledge_update = {
        "type": "tactical_update",
        "technique": "port_scanning",
        "effectiveness": 0.9,
        "notes": "Highly effective against target type"
    }
    
    await memory_manager.update_agent_knowledge("agent_001", knowledge_update)
    
    assert memory_manager.stats['knowledge_updates'] == 1

@pytest.mark.asyncio
async def test_search_cross_system(memory_manager, sample_experience):
    """Test cross-system search"""
    await memory_manager.initialize()
    
    # Store experience
    await memory_manager.store_agent_experience("agent_001", sample_experience)
    
    # Search across systems
    results = await memory_manager.search_cross_system("reconnaissance")
    
    assert "experiences" in results
    assert "knowledge" in results
    assert "query" in results
    assert results["query"] == "reconnaissance"

@pytest.mark.asyncio
async def test_get_agent_memory_profile(memory_manager, sample_experience):
    """Test getting agent memory profile"""
    await memory_manager.initialize()
    
    # Store experience
    await memory_manager.store_agent_experience("agent_001", sample_experience)
    
    # Get memory profile
    profile = await memory_manager.get_agent_memory_profile("agent_001")
    
    assert "agent_id" in profile
    assert "total_experiences" in profile
    assert "success_rate" in profile
    assert "average_confidence" in profile
    assert profile["agent_id"] == "agent_001"

@pytest.mark.asyncio
async def test_memory_statistics(memory_manager, sample_experience):
    """Test memory statistics"""
    await memory_manager.initialize()
    
    await memory_manager.store_agent_experience("agent_001", sample_experience)
    
    stats = memory_manager.get_memory_statistics()
    
    assert "memory_manager" in stats
    assert "vector_memory" in stats
    assert "knowledge_base" in stats
    assert stats["memory_manager"]["initialized"] is True
    assert stats["memory_manager"]["running"] is True

def test_memory_config_creation():
    """Test memory configuration creation"""
    config = MemoryConfig(
        vector_db_type=VectorDBType.CHROMADB,
        vector_db_path="./test_vector",
        knowledge_base_path="./test_kb",
        embedding_model="test-model",
        max_memory_age=timedelta(days=7),
        cleanup_interval=timedelta(hours=12),
        clustering_enabled=True,
        clustering_interval=timedelta(hours=6)
    )
    
    assert config.vector_db_type == VectorDBType.CHROMADB
    assert config.vector_db_path == "./test_vector"
    assert config.knowledge_base_path == "./test_kb"
    assert config.embedding_model == "test-model"
    assert config.max_memory_age == timedelta(days=7)
    assert config.cleanup_interval == timedelta(hours=12)
    assert config.clustering_enabled is True
    assert config.clustering_interval == timedelta(hours=6)

def test_memory_config_defaults():
    """Test memory configuration defaults"""
    config = MemoryConfig()
    
    assert config.vector_db_type == VectorDBType.CHROMADB
    assert config.vector_db_path == "./memory_db"
    assert config.knowledge_base_path == "./knowledge_base"
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.max_memory_age == timedelta(days=30)
    assert config.cleanup_interval == timedelta(hours=6)
    assert config.clustering_enabled is True
    assert config.clustering_interval == timedelta(hours=12)

@pytest.mark.asyncio
async def test_memory_manager_shutdown(memory_manager):
    """Test memory manager shutdown"""
    await memory_manager.initialize()
    assert memory_manager.initialized
    assert memory_manager.running
    
    await memory_manager.shutdown()
    
    assert not memory_manager.initialized
    assert not memory_manager.running

def test_create_memory_manager_factory():
    """Test memory manager factory function"""
    config = MemoryConfig(vector_db_type=VectorDBType.MEMORY)
    manager = create_memory_manager(config)
    
    assert isinstance(manager, MemoryManager)
    assert manager.config.vector_db_type == VectorDBType.MEMORY

def test_create_memory_manager_factory_default():
    """Test memory manager factory with default config"""
    manager = create_memory_manager()
    
    assert isinstance(manager, MemoryManager)
    assert manager.config.vector_db_type == VectorDBType.CHROMADB

@pytest.mark.asyncio
async def test_lesson_to_dict_conversion(memory_manager):
    """Test lesson to dictionary conversion"""
    await memory_manager.initialize()
    
    # Create mock lesson
    lesson = Mock()
    lesson.lesson_id = "lesson_001"
    lesson.scenario_type = "test"
    lesson.lesson_text = "Test lesson"
    lesson.category = "testing"
    lesson.importance_score = 0.8
    lesson.applicable_roles = ["tester"]
    lesson.mitre_techniques = ["T1595"]
    lesson.created_at = datetime.now()
    lesson.validated = True
    lesson.metadata = {"test": True}
    
    lesson_dict = memory_manager._lesson_to_dict(lesson)
    
    assert lesson_dict["lesson_id"] == "lesson_001"
    assert lesson_dict["scenario_type"] == "test"
    assert lesson_dict["lesson_text"] == "Test lesson"
    assert lesson_dict["validated"] is True

@pytest.mark.asyncio
async def test_extract_knowledge_from_experience(memory_manager, sample_experience):
    """Test knowledge extraction from experience"""
    await memory_manager.initialize()
    
    # This should extract lessons learned
    await memory_manager._extract_knowledge_from_experience(sample_experience)
    
    # Check if lessons were added to knowledge base
    lessons = await memory_manager.knowledge_base.get_lessons_learned("general")
    
    # May be empty depending on implementation
    assert isinstance(lessons, list)

@pytest.mark.asyncio
async def test_background_cleanup_disabled(memory_config):
    """Test memory manager with cleanup disabled"""
    # Disable cleanup
    memory_config.cleanup_interval = None
    
    manager = MemoryManager(memory_config)
    await manager.initialize()
    
    # Should not have cleanup task
    assert manager.cleanup_task is None
    
    await manager.shutdown()

@pytest.mark.asyncio
async def test_background_clustering_disabled(memory_config):
    """Test memory manager with clustering disabled"""
    # Already disabled in fixture
    assert memory_config.clustering_enabled is False
    
    manager = MemoryManager(memory_config)
    await manager.initialize()
    
    # Should not have clustering task
    assert manager.clustering_task is None
    
    await manager.shutdown()

@pytest.mark.asyncio
async def test_role_specific_lessons(memory_manager):
    """Test getting role-specific lessons"""
    await memory_manager.initialize()
    
    lessons = await memory_manager.get_lessons_learned("test_scenario", Role.RECON)
    
    assert isinstance(lessons, list)
    # May be empty in test environment

@pytest.mark.asyncio
async def test_tactical_knowledge_filtering(memory_manager):
    """Test tactical knowledge with different filters"""
    await memory_manager.initialize()
    
    # Test attack-only knowledge
    attack_knowledge = await memory_manager.get_tactical_knowledge(
        query="reconnaissance",
        knowledge_type="attack"
    )
    
    assert "attack_patterns" in attack_knowledge
    assert "defense_strategies" not in attack_knowledge
    
    # Test defense-only knowledge
    defense_knowledge = await memory_manager.get_tactical_knowledge(
        query="monitoring",
        knowledge_type="defense"
    )
    
    assert "defense_strategies" in defense_knowledge
    assert "attack_patterns" not in defense_knowledge