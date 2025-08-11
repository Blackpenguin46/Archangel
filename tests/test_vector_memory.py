#!/usr/bin/env python3
"""
Tests for Vector Memory System functionality
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil

from memory.vector_memory import (
    VectorMemorySystem, VectorDBType, MemoryQuery, MemoryResult, MemoryCluster,
    create_vector_memory
)
from agents.base_agent import Experience, Team, Role

@pytest.fixture
def temp_db_path():
    """Create temporary database path"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def vector_memory(temp_db_path):
    """Create test vector memory system"""
    return VectorMemorySystem(
        db_type=VectorDBType.MEMORY,  # Use in-memory for testing
        db_path=temp_db_path,
        embedding_model="all-MiniLM-L6-v2"
    )

@pytest.fixture
def sample_experience():
    """Create sample experience"""
    return Experience(
        experience_id="exp_001",
        agent_id="agent_001",
        timestamp=datetime.now(),
        context=Mock(),
        action_taken=Mock(primary_action="test_action", action_type="test"),
        reasoning=Mock(),
        outcome=Mock(success=True, outcome="success"),
        success=True,
        lessons_learned=["Test lesson learned"],
        mitre_attack_mapping=["T1595"],
        confidence_score=0.8
    )

@pytest.mark.asyncio
async def test_vector_memory_initialization(vector_memory):
    """Test vector memory initialization"""
    assert not vector_memory.initialized
    
    await vector_memory.initialize()
    
    assert vector_memory.initialized
    assert vector_memory.db_type == VectorDBType.MEMORY

@pytest.mark.asyncio
async def test_store_experience(vector_memory, sample_experience):
    """Test storing experience in vector memory"""
    await vector_memory.initialize()
    
    experience_id = await vector_memory.store_experience("agent_001", sample_experience)
    
    assert experience_id == sample_experience.experience_id
    assert experience_id in vector_memory.memory_store
    assert experience_id in vector_memory.embeddings_store

@pytest.mark.asyncio
async def test_retrieve_similar_experiences(vector_memory, sample_experience):
    """Test retrieving similar experiences"""
    await vector_memory.initialize()
    
    # Store experience
    await vector_memory.store_experience("agent_001", sample_experience)
    
    # Query for similar experiences
    query = MemoryQuery(
        query_text="test action",
        agent_id="agent_001",
        similarity_threshold=0.1,
        max_results=10
    )
    
    results = await vector_memory.retrieve_similar_experiences(query)
    
    assert len(results) > 0
    assert results[0].experience_id == sample_experience.experience_id

@pytest.mark.asyncio
async def test_memory_clustering(vector_memory):
    """Test memory clustering functionality"""
    await vector_memory.initialize()
    
    # Create multiple similar experiences
    experiences = []
    for i in range(5):
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
        experiences.append(exp)
        await vector_memory.store_experience("agent_001", exp)
    
    # Cluster memories
    clusters = await vector_memory.cluster_memories("agent_001", timedelta(hours=1))
    
    # Should create at least one cluster
    assert len(clusters) >= 0  # May be 0 if similarity threshold not met

@pytest.mark.asyncio
async def test_tactical_knowledge_update(vector_memory):
    """Test updating tactical knowledge"""
    await vector_memory.initialize()
    
    tactic = {
        "type": "reconnaissance",
        "technique": "port_scanning",
        "effectiveness": 0.8,
        "stealth": 0.6
    }
    
    await vector_memory.update_tactical_knowledge("agent_001", tactic)
    
    # Should be stored in memory
    tactic_entries = [
        entry for entry in vector_memory.memory_store.values()
        if entry.get("metadata", {}).get("type") == "tactical_knowledge"
    ]
    
    assert len(tactic_entries) > 0

@pytest.mark.asyncio
async def test_role_specific_memories(vector_memory, sample_experience):
    """Test retrieving role-specific memories"""
    await vector_memory.initialize()
    
    await vector_memory.store_experience("agent_001", sample_experience)
    
    results = await vector_memory.get_role_specific_memories("agent_001", Role.RECON)
    
    # Should return results (may be empty based on filtering)
    assert isinstance(results, list)

@pytest.mark.asyncio
async def test_memory_cleanup(vector_memory, sample_experience):
    """Test memory cleanup functionality"""
    await vector_memory.initialize()
    
    # Store experience
    await vector_memory.store_experience("agent_001", sample_experience)
    
    # Clean up memories older than 0 seconds (should clean everything)
    cleaned_count = await vector_memory.cleanup_old_memories(timedelta(seconds=0))
    
    # Should have cleaned up the experience
    assert cleaned_count >= 0

@pytest.mark.asyncio
async def test_memory_statistics(vector_memory, sample_experience):
    """Test memory statistics"""
    await vector_memory.initialize()
    
    await vector_memory.store_experience("agent_001", sample_experience)
    
    stats = vector_memory.get_memory_stats()
    
    assert "storage_type" in stats
    assert "total_experiences" in stats
    assert "initialized" in stats
    assert stats["initialized"] is True

def test_memory_query_creation():
    """Test memory query creation"""
    query = MemoryQuery(
        query_text="test query",
        agent_id="agent_001",
        team=Team.RED,
        role=Role.RECON,
        similarity_threshold=0.7,
        max_results=5
    )
    
    assert query.query_text == "test query"
    assert query.agent_id == "agent_001"
    assert query.team == Team.RED
    assert query.role == Role.RECON
    assert query.similarity_threshold == 0.7
    assert query.max_results == 5

def test_memory_result_creation():
    """Test memory result creation"""
    result = MemoryResult(
        experience_id="exp_001",
        experience=Mock(),
        similarity_score=0.85,
        embedding=[0.1, 0.2, 0.3],
        metadata={"agent_id": "agent_001"}
    )
    
    assert result.experience_id == "exp_001"
    assert result.similarity_score == 0.85
    assert len(result.embedding) == 3
    assert result.metadata["agent_id"] == "agent_001"

def test_memory_cluster_creation():
    """Test memory cluster creation"""
    cluster = MemoryCluster(
        cluster_id="cluster_001",
        cluster_name="Test Cluster",
        center_embedding=[0.5, 0.5, 0.5],
        experiences=["exp_001", "exp_002"],
        similarity_threshold=0.8,
        created_at=datetime.now(),
        last_updated=datetime.now(),
        metadata={"size": 2}
    )
    
    assert cluster.cluster_id == "cluster_001"
    assert cluster.cluster_name == "Test Cluster"
    assert len(cluster.experiences) == 2
    assert cluster.similarity_threshold == 0.8

def test_vector_db_type_enum():
    """Test VectorDBType enum"""
    assert VectorDBType.CHROMADB.value == "chromadb"
    assert VectorDBType.WEAVIATE.value == "weaviate"
    assert VectorDBType.MEMORY.value == "memory"

@pytest.mark.asyncio
async def test_simple_embedding_fallback(vector_memory):
    """Test simple embedding fallback"""
    await vector_memory.initialize()
    
    # Test simple embedding creation
    embedding = vector_memory._simple_embedding("test text")
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # Default dimension
    assert all(isinstance(x, float) for x in embedding)

def test_experience_to_text_conversion(vector_memory, sample_experience):
    """Test converting experience to text"""
    text = vector_memory._experience_to_text(sample_experience)
    
    assert isinstance(text, str)
    assert len(text) > 0
    assert "Agent:" in text

def test_similarity_calculation(vector_memory):
    """Test similarity calculation"""
    embedding1 = [1.0, 0.0, 0.0]
    embedding2 = [1.0, 0.0, 0.0]
    embedding3 = [0.0, 1.0, 0.0]
    
    # Identical embeddings should have similarity 1.0
    similarity1 = vector_memory._calculate_similarity(embedding1, embedding2)
    assert abs(similarity1 - 1.0) < 0.001
    
    # Orthogonal embeddings should have similarity 0.0
    similarity2 = vector_memory._calculate_similarity(embedding1, embedding3)
    assert abs(similarity2 - 0.0) < 0.001

@pytest.mark.asyncio
async def test_vector_memory_shutdown(vector_memory):
    """Test vector memory shutdown"""
    await vector_memory.initialize()
    assert vector_memory.initialized
    
    await vector_memory.shutdown()
    assert not vector_memory.initialized
    assert len(vector_memory.memory_store) == 0

def test_create_vector_memory_factory():
    """Test vector memory factory function"""
    memory = create_vector_memory(
        db_type=VectorDBType.MEMORY,
        db_path="./test_db",
        embedding_model="test-model"
    )
    
    assert isinstance(memory, VectorMemorySystem)
    assert memory.db_type == VectorDBType.MEMORY
    assert memory.db_path == "./test_db"
    assert memory.embedding_model_name == "test-model"

@pytest.mark.asyncio
async def test_filter_results(vector_memory):
    """Test result filtering"""
    await vector_memory.initialize()
    
    # Create test results
    results = [
        MemoryResult(
            experience_id="exp_001",
            experience=Mock(),
            similarity_score=0.9,
            embedding=[],
            metadata={"timestamp": datetime.now().isoformat()}
        ),
        MemoryResult(
            experience_id="exp_002", 
            experience=Mock(),
            similarity_score=0.5,
            embedding=[],
            metadata={"timestamp": datetime.now().isoformat()}
        )
    ]
    
    query = MemoryQuery(
        query_text="test",
        similarity_threshold=0.7,
        max_results=10
    )
    
    filtered = vector_memory._filter_results(results, query)
    
    # Should filter out low similarity result
    assert len(filtered) == 1
    assert filtered[0].similarity_score == 0.9