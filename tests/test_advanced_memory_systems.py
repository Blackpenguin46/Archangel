#!/usr/bin/env python3
"""
Tests for Advanced Memory Systems - Clustering, Retrieval, and Optimization
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil
import json
import gzip
import pickle

from memory.temporal_clustering import (
    TemporalMemoryClusterer, TemporalClusteringConfig, TemporalClusterType, TemporalCluster
)
from memory.semantic_annotation import (
    SemanticMemoryAnnotator, SemanticCategorizationConfig, AnnotationType, SemanticTag
)
from memory.context_aware_retrieval import (
    ContextAwareMemoryRetriever, RetrievalContext, RelevanceScoreType
)
from memory.memory_optimization import (
    MemoryOptimizer, MemoryOptimizationConfig, OptimizationType, GarbageCollectionStrategy
)
from agents.base_agent import Experience, Team, Role

@pytest.fixture
def sample_experiences():
    """Create sample experiences for testing"""
    experiences = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(20):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            agent_id=f"agent_{i % 3}",  # 3 different agents
            timestamp=base_time + timedelta(hours=i * 2),
            context=Mock(),
            action_taken=Mock(
                primary_action=f"action_{i % 5}",
                action_type=["reconnaissance", "exploitation", "persistence"][i % 3]
            ),
            reasoning=Mock(),
            outcome=Mock(success=(i % 3 != 0), outcome="test_outcome"),
            success=(i % 3 != 0),
            lessons_learned=[f"Lesson {i}", f"Insight {i}"] if i % 2 == 0 else [],
            mitre_attack_mapping=[f"T{1000 + i}"] if i % 4 == 0 else [],
            confidence_score=0.3 + (i % 7) * 0.1
        )
        experiences.append(exp)
    
    return experiences

@pytest.fixture
def temp_dirs():
    """Create temporary directories"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
async def temporal_clusterer():
    """Create temporal clusterer for testing"""
    config = TemporalClusteringConfig(
        min_cluster_size=3,
        similarity_threshold=0.7,
        session_gap_threshold=timedelta(hours=2)
    )
    clusterer = TemporalMemoryClusterer(config)
    await clusterer.initialize()
    return clusterer

@pytest.fixture
async def semantic_annotator():
    """Create semantic annotator for testing"""
    config = SemanticCategorizationConfig(
        min_confidence_threshold=0.5,
        enable_automatic_tagging=True
    )
    annotator = SemanticMemoryAnnotator(config)
    await annotator.initialize()
    return annotator

@pytest.fixture
async def context_retriever():
    """Create context-aware retriever for testing"""
    retriever = ContextAwareMemoryRetriever()
    await retriever.initialize()
    return retriever

@pytest.fixture
async def memory_optimizer():
    """Create memory optimizer for testing"""
    config = MemoryOptimizationConfig(
        max_total_memory_size=1024 * 1024,  # 1MB for testing
        compression_age_threshold=timedelta(days=1),
        archive_age_threshold=timedelta(days=7)
    )
    optimizer = MemoryOptimizer(config)
    await optimizer.initialize({})  # Empty memory store for testing
    return optimizer

class TestTemporalClustering:
    """Test temporal clustering algorithms"""
    
    @pytest.mark.asyncio
    async def test_session_based_clustering(self, temporal_clusterer, sample_experiences):
        """Test session-based clustering"""
        clusters = await temporal_clusterer.cluster_agent_experiences_temporal(
            "agent_001", 
            sample_experiences[:10], 
            TemporalClusterType.SESSION_BASED
        )
        
        assert isinstance(clusters, list)
        assert len(clusters) > 0
        
        for cluster in clusters:
            assert isinstance(cluster, TemporalCluster)
            assert cluster.cluster_type == TemporalClusterType.SESSION_BASED
            assert len(cluster.experience_ids) >= temporal_clusterer.config.min_cluster_size
            assert cluster.start_time <= cluster.end_time
    
    @pytest.mark.asyncio
    async def test_time_window_clustering(self, temporal_clusterer, sample_experiences):
        """Test time window clustering"""
        clusters = await temporal_clusterer.cluster_agent_experiences_temporal(
            "agent_001", 
            sample_experiences, 
            TemporalClusterType.TIME_WINDOW
        )
        
        assert isinstance(clusters, list)
        
        for cluster in clusters:
            assert cluster.cluster_type == TemporalClusterType.TIME_WINDOW
            assert cluster.duration <= temporal_clusterer.config.default_window_size
    
    @pytest.mark.asyncio
    async def test_adaptive_window_clustering(self, temporal_clusterer, sample_experiences):
        """Test adaptive window clustering"""
        clusters = await temporal_clusterer.cluster_agent_experiences_temporal(
            "agent_001", 
            sample_experiences, 
            TemporalClusterType.ADAPTIVE_WINDOW
        )
        
        assert isinstance(clusters, list)
        
        for cluster in clusters:
            assert cluster.cluster_type == TemporalClusterType.ADAPTIVE_WINDOW
    
    @pytest.mark.asyncio
    async def test_event_driven_clustering(self, temporal_clusterer, sample_experiences):
        """Test event-driven clustering"""
        clusters = await temporal_clusterer.cluster_agent_experiences_temporal(
            "agent_001", 
            sample_experiences, 
            TemporalClusterType.EVENT_DRIVEN
        )
        
        assert isinstance(clusters, list)
        
        for cluster in clusters:
            assert cluster.cluster_type == TemporalClusterType.EVENT_DRIVEN
    
    @pytest.mark.asyncio
    async def test_learning_phase_clustering(self, temporal_clusterer, sample_experiences):
        """Test learning phase clustering"""
        clusters = await temporal_clusterer.cluster_agent_experiences_temporal(
            "agent_001", 
            sample_experiences, 
            TemporalClusterType.LEARNING_PHASE
        )
        
        assert isinstance(clusters, list)
        
        for cluster in clusters:
            assert cluster.cluster_type == TemporalClusterType.LEARNING_PHASE
    
    @pytest.mark.asyncio
    async def test_cluster_analytics(self, temporal_clusterer, sample_experiences):
        """Test cluster analytics"""
        await temporal_clusterer.cluster_agent_experiences_temporal(
            "agent_001", 
            sample_experiences, 
            TemporalClusterType.SESSION_BASED
        )
        
        analytics = temporal_clusterer.get_cluster_analytics("agent_001")
        
        assert "agent_id" in analytics
        assert "total_clusters" in analytics
        assert "total_experiences" in analytics
        assert analytics["agent_id"] == "agent_001"
    
    @pytest.mark.asyncio
    async def test_clustering_statistics(self, temporal_clusterer, sample_experiences):
        """Test clustering statistics"""
        await temporal_clusterer.cluster_agent_experiences_temporal(
            "agent_001", 
            sample_experiences, 
            TemporalClusterType.SESSION_BASED
        )
        
        stats = temporal_clusterer.get_clustering_statistics()
        
        assert "total_clusters" in stats
        assert "agents_with_clusters" in stats
        assert "cluster_type_distribution" in stats
    
    @pytest.mark.asyncio
    async def test_empty_experiences(self, temporal_clusterer):
        """Test clustering with no experiences"""
        clusters = await temporal_clusterer.cluster_agent_experiences_temporal(
            "agent_001", 
            [], 
            TemporalClusterType.SESSION_BASED
        )
        
        assert clusters == []
    
    @pytest.mark.asyncio
    async def test_insufficient_experiences(self, temporal_clusterer, sample_experiences):
        """Test clustering with insufficient experiences"""
        clusters = await temporal_clusterer.cluster_agent_experiences_temporal(
            "agent_001", 
            sample_experiences[:2],  # Only 2 experiences
            TemporalClusterType.SESSION_BASED
        )
        
        # Should handle gracefully with insufficient data
        assert isinstance(clusters, list)

class TestSemanticAnnotation:
    """Test semantic annotation system"""
    
    @pytest.mark.asyncio
    async def test_single_memory_annotation(self, semantic_annotator, sample_experiences):
        """Test annotating a single memory"""
        experience = sample_experiences[0]
        
        annotations = await semantic_annotator.annotate_memory(
            "memory_001", 
            experience, 
            [AnnotationType.TACTICAL, AnnotationType.PERFORMANCE]
        )
        
        assert isinstance(annotations, list)
        assert len(annotations) <= 2  # Should have tactical and performance annotations
        
        for annotation in annotations:
            assert annotation.annotation_type in [AnnotationType.TACTICAL, AnnotationType.PERFORMANCE]
            assert annotation.confidence_score >= 0.0
            assert annotation.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_batch_annotation(self, semantic_annotator, sample_experiences):
        """Test batch annotation of memories"""
        memories = [(f"memory_{i}", exp) for i, exp in enumerate(sample_experiences[:5])]
        
        results = await semantic_annotator.batch_annotate_memories(memories)
        
        assert isinstance(results, dict)
        assert len(results) == 5
        
        for memory_id, annotations in results.items():
            assert isinstance(annotations, list)
            assert memory_id.startswith("memory_")
    
    @pytest.mark.asyncio
    async def test_tactical_annotation(self, semantic_annotator, sample_experiences):
        """Test tactical annotation specifically"""
        experience = sample_experiences[0]
        experience.mitre_attack_mapping = ["T1595", "T1590"]
        
        annotations = await semantic_annotator.annotate_memory(
            "memory_001", 
            experience, 
            [AnnotationType.TACTICAL]
        )
        
        tactical_annotations = [a for a in annotations if a.annotation_type == AnnotationType.TACTICAL]
        assert len(tactical_annotations) > 0
        
        tactical = tactical_annotations[0]
        assert len(tactical.concepts) > 0
        assert len(tactical.tags) > 0
    
    @pytest.mark.asyncio
    async def test_semantic_tag_search(self, semantic_annotator, sample_experiences):
        """Test searching by semantic tags"""
        # Annotate some memories
        for i, exp in enumerate(sample_experiences[:3]):
            await semantic_annotator.annotate_memory(f"memory_{i}", exp)
        
        # Search by tags
        memory_ids = semantic_annotator.search_by_semantic_tags([SemanticTag.RECONNAISSANCE])
        
        assert isinstance(memory_ids, list)
    
    @pytest.mark.asyncio
    async def test_annotation_statistics(self, semantic_annotator, sample_experiences):
        """Test annotation statistics"""
        # Annotate some memories
        for i, exp in enumerate(sample_experiences[:3]):
            await semantic_annotator.annotate_memory(f"memory_{i}", exp)
        
        stats = semantic_annotator.get_annotation_statistics()
        
        assert "total_annotations" in stats
        assert "annotations_by_type" in stats
        assert "annotated_memories" in stats
    
    @pytest.mark.asyncio
    async def test_annotation_validation(self, semantic_annotator, sample_experiences):
        """Test annotation validation"""
        experience = sample_experiences[0]
        
        # Create annotation with low confidence
        with patch.object(semantic_annotator, '_validate_annotation', return_value=False):
            annotations = await semantic_annotator.annotate_memory("memory_001", experience)
            # Should return empty list if validation fails
            assert len(annotations) == 0

class TestContextAwareRetrieval:
    """Test context-aware memory retrieval"""
    
    @pytest.mark.asyncio
    async def test_basic_contextual_retrieval(self, context_retriever):
        """Test basic contextual retrieval"""
        # Mock vector memory for testing
        mock_vector_memory = Mock()
        mock_vector_memory.retrieve_similar_experiences = AsyncMock(return_value=[])
        context_retriever.vector_memory = mock_vector_memory
        
        context = RetrievalContext(
            query_text="reconnaissance scan",
            agent_id="agent_001",
            max_results=10
        )
        
        results = await context_retriever.retrieve_contextual_memories(context)
        
        assert isinstance(results, list)
        assert len(results) <= context.max_results
    
    @pytest.mark.asyncio
    async def test_relevance_scoring(self, context_retriever):
        """Test multi-dimensional relevance scoring"""
        # Create mock memory results
        mock_experience = Mock()
        mock_experience.timestamp = datetime.now()
        mock_experience.agent_id = "agent_001"
        mock_experience.success = True
        mock_experience.confidence_score = 0.8
        mock_experience.lessons_learned = ["Test lesson"]
        mock_experience.mitre_attack_mapping = ["T1595"]
        
        mock_result = Mock()
        mock_result.experience_id = "exp_001"
        mock_result.experience = mock_experience
        mock_result.embedding = [0.1] * 384
        
        context = RetrievalContext(
            query_text="reconnaissance",
            agent_id="agent_001",
            score_weights={
                RelevanceScoreType.SEMANTIC: 0.4,
                RelevanceScoreType.TEMPORAL: 0.3,
                RelevanceScoreType.PERFORMANCE: 0.3
            }
        )
        
        # Test individual scoring components
        semantic_score = await context_retriever._calculate_semantic_relevance(
            context.query_text, mock_experience, mock_result.embedding
        )
        assert 0.0 <= semantic_score <= 1.0
        
        temporal_score = await context_retriever._calculate_temporal_relevance(
            mock_experience, context
        )
        assert 0.0 <= temporal_score <= 1.0
        
        performance_score = await context_retriever._calculate_performance_relevance(
            mock_experience, context
        )
        assert 0.0 <= performance_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_contextual_filtering(self, context_retriever):
        """Test contextual filtering"""
        # Create test candidates
        candidates = []
        for i in range(5):
            exp = Mock()
            exp.timestamp = datetime.now() - timedelta(days=i)
            exp.agent_id = f"agent_{i % 2}"
            exp.success = (i % 2 == 0)
            exp.confidence_score = 0.5 + i * 0.1
            
            candidates.append((f"memory_{i}", exp, [0.1] * 384))
        
        context = RetrievalContext(
            agent_id="agent_0",
            success_only=True,
            required_confidence=0.6
        )
        
        filtered = await context_retriever._apply_contextual_filters(candidates, context)
        
        # Should filter based on agent_id, success, and confidence
        assert len(filtered) <= len(candidates)
        
        for memory_id, exp, embedding in filtered:
            assert exp.agent_id == "agent_0" or context.cross_team_learning
            if context.success_only:
                assert exp.success
            assert exp.confidence_score >= context.required_confidence
    
    @pytest.mark.asyncio
    async def test_diversity_selection(self, context_retriever):
        """Test diversity-aware result selection"""
        # Create mock results with varying characteristics
        results = []
        for i in range(10):
            mock_exp = Mock()
            mock_exp.timestamp = datetime.now() - timedelta(days=i)
            mock_exp.agent_id = f"agent_{i % 3}"
            mock_exp.success = (i % 2 == 0)
            mock_exp.confidence_score = 0.5 + (i % 5) * 0.1
            
            mock_result = Mock()
            mock_result.experience = mock_exp
            mock_result.relevance_score = Mock()
            mock_result.relevance_score.total_score = 0.9 - i * 0.05
            
            results.append(mock_result)
        
        context = RetrievalContext(
            max_results=5,
            diversity_factor=0.5
        )
        
        diverse_results = await context_retriever._apply_diversity_selection(results, context)
        
        assert len(diverse_results) <= context.max_results
        assert len(diverse_results) <= len(results)
    
    @pytest.mark.asyncio
    async def test_retrieval_analytics(self, context_retriever):
        """Test retrieval analytics tracking"""
        # Perform some mock retrievals to generate analytics
        context_retriever.retrieval_analytics["total_retrievals"] = 5
        context_retriever.retrieval_analytics["avg_relevance_score"] = 0.75
        context_retriever.retrieval_analytics["retrieval_latency"] = [0.1, 0.2, 0.15, 0.18, 0.12]
        
        analytics = context_retriever.get_retrieval_analytics()
        
        assert "total_retrievals" in analytics
        assert "avg_relevance_score" in analytics
        assert "avg_latency" in analytics
        assert analytics["total_retrievals"] == 5
        assert analytics["avg_relevance_score"] == 0.75

class TestMemoryOptimization:
    """Test memory optimization and garbage collection"""
    
    @pytest.mark.asyncio
    async def test_memory_metrics_initialization(self, memory_optimizer):
        """Test memory metrics initialization"""
        # Add some test memories
        test_memories = {
            "memory_001": {"data": "test1", "size": 100},
            "memory_002": {"data": "test2", "size": 200},
        }
        
        memory_optimizer.memory_store = test_memories
        await memory_optimizer._initialize_memory_metrics()
        
        assert len(memory_optimizer.memory_metrics) == 2
        assert "memory_001" in memory_optimizer.memory_metrics
        assert "memory_002" in memory_optimizer.memory_metrics
    
    @pytest.mark.asyncio
    async def test_compression_optimization(self, memory_optimizer):
        """Test memory compression"""
        # Add test memories
        test_data = {"large_data": "x" * 1000, "timestamp": datetime.now().isoformat()}
        memory_optimizer.memory_store["memory_001"] = test_data
        
        # Initialize metrics
        await memory_optimizer._initialize_memory_metrics()
        
        # Set memory as old enough for compression
        memory_optimizer.memory_metrics["memory_001"].age_days = 2
        
        result = await memory_optimizer._perform_compression()
        
        assert isinstance(result.bytes_saved, int)
        assert result.bytes_saved >= 0
        assert result.operation_type == OptimizationType.COMPRESSION
    
    @pytest.mark.asyncio
    async def test_deduplication_optimization(self, memory_optimizer):
        """Test memory deduplication"""
        # Add duplicate memories
        test_data = {"data": "duplicate_content", "id": 1}
        memory_optimizer.memory_store["memory_001"] = test_data
        memory_optimizer.memory_store["memory_002"] = test_data.copy()  # Exact duplicate
        
        await memory_optimizer._initialize_memory_metrics()
        
        result = await memory_optimizer._perform_deduplication()
        
        assert result.operation_type == OptimizationType.DEDUPLICATION
        assert result.success_rate >= 0.0
    
    @pytest.mark.asyncio
    async def test_garbage_collection_strategies(self, memory_optimizer):
        """Test different garbage collection strategies"""
        # Add test memories with different characteristics
        old_time = datetime.now() - timedelta(days=400)  # Very old
        recent_time = datetime.now() - timedelta(days=1)  # Recent
        
        memory_optimizer.memory_store = {
            "old_memory": {"data": "old", "timestamp": old_time.isoformat()},
            "recent_memory": {"data": "recent", "timestamp": recent_time.isoformat()}
        }
        
        await memory_optimizer._initialize_memory_metrics()
        
        # Set up metrics for GC testing
        memory_optimizer.memory_metrics["old_memory"].age_days = 400
        memory_optimizer.memory_metrics["old_memory"].access_frequency = 0.001  # Very low
        memory_optimizer.memory_metrics["recent_memory"].age_days = 1
        memory_optimizer.memory_metrics["recent_memory"].access_frequency = 0.5  # High
        
        # Test time-based GC
        time_candidates = await memory_optimizer._identify_gc_candidates(
            GarbageCollectionStrategy.TIME_BASED
        )
        assert "old_memory" in time_candidates
        assert "recent_memory" not in time_candidates
        
        # Test usage-based GC
        usage_candidates = await memory_optimizer._identify_gc_candidates(
            GarbageCollectionStrategy.USAGE_BASED
        )
        # Should include old memory with low usage
        
        # Test hybrid GC
        hybrid_candidates = await memory_optimizer._identify_gc_candidates(
            GarbageCollectionStrategy.HYBRID
        )
        assert isinstance(hybrid_candidates, list)
    
    @pytest.mark.asyncio
    async def test_memory_value_calculation(self, memory_optimizer):
        """Test memory value calculation"""
        # Create test metrics
        high_value_metrics = memory_optimizer.memory_metrics.get("test", type(memory_optimizer.memory_metrics)("test_memory", 1000))
        high_value_metrics.importance_score = 0.9
        high_value_metrics.utility_score = 0.8
        high_value_metrics.learning_value = 0.7
        high_value_metrics.access_frequency = 0.5
        high_value_metrics.age_days = 30
        
        low_value_metrics = memory_optimizer.memory_metrics.get("test2", type(memory_optimizer.memory_metrics)("test_memory2", 1000))
        low_value_metrics.importance_score = 0.1
        low_value_metrics.utility_score = 0.2
        low_value_metrics.learning_value = 0.1
        low_value_metrics.access_frequency = 0.01
        low_value_metrics.age_days = 300
        
        high_value = memory_optimizer._calculate_memory_value(high_value_metrics)
        low_value = memory_optimizer._calculate_memory_value(low_value_metrics)
        
        assert 0.0 <= high_value <= 1.0
        assert 0.0 <= low_value <= 1.0
        assert high_value > low_value
    
    @pytest.mark.asyncio
    async def test_safe_memory_deletion(self, memory_optimizer):
        """Test safe memory deletion"""
        # Add test memory
        memory_optimizer.memory_store["test_memory"] = {"data": "test"}
        await memory_optimizer._initialize_memory_metrics()
        
        # Test deletion
        result = await memory_optimizer._safe_delete_memory("test_memory")
        assert result is True
        assert "test_memory" not in memory_optimizer.memory_store
        
        # Test deletion of non-existent memory
        result = await memory_optimizer._safe_delete_memory("non_existent")
        assert result is True  # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_memory_compression_and_decompression(self, memory_optimizer):
        """Test memory compression and decompression"""
        test_data = {
            "large_text": "This is a test string that should compress well " * 100,
            "numbers": list(range(1000)),
            "metadata": {"timestamp": datetime.now().isoformat()}
        }
        
        # Test compression
        compressed = await memory_optimizer._compress_memory_data(test_data)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        
        # Test that compression actually reduces size for repetitive data
        original_size = len(pickle.dumps(test_data))
        assert len(compressed) < original_size
    
    @pytest.mark.asyncio
    async def test_comprehensive_optimization(self, memory_optimizer):
        """Test comprehensive memory optimization"""
        # Add various test memories
        memory_optimizer.memory_store = {
            f"memory_{i}": {
                "data": f"test_data_{i}" * (i + 1),
                "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                "importance": 1.0 - (i * 0.1)
            }
            for i in range(10)
        }
        
        await memory_optimizer._initialize_memory_metrics()
        
        # Run comprehensive optimization
        results = await memory_optimizer.optimize_memory_system()
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert hasattr(result, 'operation_type')
            assert hasattr(result, 'bytes_saved')
            assert hasattr(result, 'success_rate')
            assert 0.0 <= result.success_rate <= 1.0
    
    @pytest.mark.asyncio
    async def test_optimization_statistics(self, memory_optimizer):
        """Test optimization statistics"""
        # Add some test data
        memory_optimizer.memory_store["test"] = {"data": "test"}
        await memory_optimizer._initialize_memory_metrics()
        
        # Record some optimization history
        from memory.memory_optimization import OptimizationResult
        mock_result = OptimizationResult(
            operation_type=OptimizationType.COMPRESSION,
            memories_processed=5,
            bytes_saved=1024,
            time_taken=1.5,
            success_rate=0.8
        )
        memory_optimizer.optimization_history.append(mock_result)
        
        stats = memory_optimizer.get_optimization_statistics()
        
        assert "memory_metrics" in stats
        assert "optimization_history" in stats
        assert "performance_metrics" in stats
        assert stats["optimization_history"]["total_operations"] == 1
        assert stats["optimization_history"]["total_bytes_saved"] == 1024

class TestMemorySystemIntegration:
    """Test integration between memory system components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_memory_workflow(self, sample_experiences, temp_dirs):
        """Test complete memory workflow from storage to retrieval"""
        # Initialize all components
        temporal_clusterer = TemporalMemoryClusterer()
        await temporal_clusterer.initialize()
        
        semantic_annotator = SemanticMemoryAnnotator()
        await semantic_annotator.initialize()
        
        context_retriever = ContextAwareMemoryRetriever()
        await context_retriever.initialize()
        
        # 1. Cluster experiences temporally
        clusters = await temporal_clusterer.cluster_agent_experiences_temporal(
            "agent_001",
            sample_experiences[:10],
            TemporalClusterType.SESSION_BASED
        )
        
        assert len(clusters) > 0
        
        # 2. Annotate experiences semantically
        annotations_results = {}
        for i, exp in enumerate(sample_experiences[:5]):
            annotations = await semantic_annotator.annotate_memory(f"memory_{i}", exp)
            annotations_results[f"memory_{i}"] = annotations
        
        assert len(annotations_results) == 5
        
        # 3. Perform contextual retrieval
        context = RetrievalContext(
            query_text="reconnaissance scan",
            agent_id="agent_001",
            max_results=5
        )
        
        # Mock the retrieval since we don't have actual vector storage
        context_retriever.semantic_annotator = semantic_annotator
        results = await context_retriever.retrieve_contextual_memories(context)
        
        # Should handle empty results gracefully
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_memory_lifecycle_management(self, sample_experiences):
        """Test complete memory lifecycle from creation to deletion"""
        # Initialize optimizer
        config = MemoryOptimizationConfig(
            max_total_memory_size=10000,  # Small limit for testing
            archive_age_threshold=timedelta(days=1),
            deletion_age_threshold=timedelta(days=3)
        )
        optimizer = MemoryOptimizer(config)
        
        # Create memory store with sample data
        memory_store = {}
        for i, exp in enumerate(sample_experiences[:5]):
            memory_store[f"memory_{i}"] = {
                "experience": exp,
                "created_at": exp.timestamp.isoformat(),
                "size": len(str(exp)) * 2  # Approximate size
            }
        
        await optimizer.initialize(memory_store)
        
        # Test optimization lifecycle
        results = await optimizer.optimize_memory_system([
            OptimizationType.COMPRESSION,
            OptimizationType.ARCHIVAL,
            OptimizationType.DEDUPLICATION
        ])
        
        assert len(results) == 3  # One result per optimization type
        
        # Test garbage collection
        gc_result = await optimizer.garbage_collect(GarbageCollectionStrategy.HYBRID)
        assert isinstance(gc_result.bytes_saved, int)
        
        # Get final statistics
        stats = optimizer.get_optimization_statistics()
        assert "memory_metrics" in stats
        assert "optimization_history" in stats
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, sample_experiences):
        """Test system performance with larger datasets"""
        # Create larger dataset
        large_dataset = []
        for i in range(100):
            exp = Experience(
                experience_id=f"exp_{i:04d}",
                agent_id=f"agent_{i % 10}",
                timestamp=datetime.now() - timedelta(hours=i),
                context=Mock(),
                action_taken=Mock(primary_action=f"action_{i}", action_type="test"),
                reasoning=Mock(),
                outcome=Mock(success=(i % 2 == 0), outcome="test"),
                success=(i % 2 == 0),
                lessons_learned=[f"Lesson {i}"],
                mitre_attack_mapping=[],
                confidence_score=0.5 + (i % 10) * 0.05
            )
            large_dataset.append(exp)
        
        # Test temporal clustering performance
        clusterer = TemporalMemoryClusterer()
        await clusterer.initialize()
        
        start_time = datetime.now()
        clusters = await clusterer.cluster_agent_experiences_temporal(
            "agent_001",
            large_dataset,
            TemporalClusterType.ADAPTIVE_WINDOW
        )
        clustering_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert clustering_time < 10.0  # 10 seconds max
        assert len(clusters) >= 0
        
        # Test semantic annotation performance
        annotator = SemanticMemoryAnnotator()
        await annotator.initialize()
        
        start_time = datetime.now()
        batch_memories = [(f"memory_{i}", exp) for i, exp in enumerate(large_dataset[:20])]
        annotation_results = await annotator.batch_annotate_memories(batch_memories)
        annotation_time = (datetime.now() - start_time).total_seconds()
        
        assert annotation_time < 15.0  # 15 seconds max for 20 memories
        assert len(annotation_results) <= 20
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, sample_experiences):
        """Test error handling and system recovery"""
        # Test temporal clusterer with invalid data
        clusterer = TemporalMemoryClusterer()
        await clusterer.initialize()
        
        # Test with empty experiences
        clusters = await clusterer.cluster_agent_experiences_temporal("agent_001", [])
        assert clusters == []
        
        # Test with malformed experiences
        bad_experience = Mock()
        bad_experience.timestamp = "not_a_date"  # Invalid timestamp
        
        clusters = await clusterer.cluster_agent_experiences_temporal(
            "agent_001", 
            [bad_experience]
        )
        # Should handle gracefully
        assert isinstance(clusters, list)
        
        # Test semantic annotator with invalid data
        annotator = SemanticMemoryAnnotator()
        await annotator.initialize()
        
        annotations = await annotator.annotate_memory("test", bad_experience)
        # Should handle gracefully and return empty or valid annotations
        assert isinstance(annotations, list)
        
        # Test optimizer with corrupted memory store
        optimizer = MemoryOptimizer()
        corrupted_store = {"bad_memory": None}  # None value should be handled
        await optimizer.initialize(corrupted_store)
        
        # Should initialize without crashing
        assert optimizer.initialized
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, sample_experiences):
        """Test concurrent memory operations"""
        # Initialize components
        clusterer = TemporalMemoryClusterer()
        await clusterer.initialize()
        
        annotator = SemanticMemoryAnnotator()
        await annotator.initialize()
        
        # Create concurrent tasks
        clustering_task = clusterer.cluster_agent_experiences_temporal(
            "agent_001",
            sample_experiences[:10],
            TemporalClusterType.SESSION_BASED
        )
        
        annotation_tasks = [
            annotator.annotate_memory(f"memory_{i}", exp)
            for i, exp in enumerate(sample_experiences[:5])
        ]
        
        # Run concurrently
        results = await asyncio.gather(
            clustering_task,
            *annotation_tasks,
            return_exceptions=True
        )
        
        # Check that all operations completed
        assert len(results) == 6  # 1 clustering + 5 annotations
        
        # Check clustering result
        clusters = results[0]
        assert isinstance(clusters, list)
        
        # Check annotation results
        for result in results[1:]:
            assert isinstance(result, list) or isinstance(result, Exception)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])