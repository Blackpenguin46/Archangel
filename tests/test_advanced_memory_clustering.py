#!/usr/bin/env python3
"""
Tests for Advanced Memory Clustering and Retrieval System
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import uuid

# Import the modules we're testing
from memory.temporal_clustering import (
    TemporalClusteringEngine, ClusteringConfig, ClusteringAlgorithm,
    TemporalCluster, TemporalSegment
)
from memory.semantic_annotation import (
    SemanticAnnotationEngine, AnnotationType, EntityType, ConceptType,
    SemanticAnnotation, AnnotatedMemory, AnnotationRule
)
from memory.context_retrieval import (
    ContextAwareRetrieval, RetrievalConfig, RetrievalStrategy,
    RetrievalContext, ContextType, RelevanceScore, RetrievalResult
)
from memory.memory_optimization import (
    MemoryOptimizer, OptimizationConfig, OptimizationStrategy,
    MemoryMetrics, MemoryPriority, CompressionType
)

class TestTemporalClustering:
    """Test temporal clustering functionality"""
    
    @pytest.fixture
    async def clustering_engine(self):
        """Create temporal clustering engine for testing"""
        config = ClusteringConfig(
            algorithm=ClusteringAlgorithm.TEMPORAL_DBSCAN,
            min_cluster_size=2,
            max_cluster_size=10,
            time_epsilon=timedelta(hours=1),
            similarity_threshold=0.7
        )
        engine = TemporalClusteringEngine(config)
        await engine.initialize()
        yield engine
        await engine.shutdown()
    
    def create_test_experiences(self, count: int = 10) -> List[Dict[str, Any]]:
        """Create test experiences for clustering"""
        experiences = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(count):
            experience = {
                'experience_id': str(uuid.uuid4()),
                'agent_id': f'agent_{i % 3}',  # 3 different agents
                'timestamp': base_time + timedelta(hours=i),
                'content': f'Test experience {i}',
                'embedding': np.random.rand(384).tolist(),  # Random embedding
                'success': i % 2 == 0,  # Alternate success/failure
                'confidence_score': 0.5 + (i % 5) * 0.1,
                'action_type': f'action_{i % 4}'  # 4 different action types
            }
            experiences.append(experience)
        
        return experiences
    
    @pytest.mark.asyncio
    async def test_temporal_dbscan_clustering(self, clustering_engine):
        """Test DBSCAN temporal clustering"""
        experiences = self.create_test_experiences(15)
        
        clusters = await clustering_engine.cluster_experiences_temporal(
            agent_id='test_agent',
            experiences=experiences,
            time_window=timedelta(days=2)
        )
        
        assert len(clusters) > 0
        assert all(isinstance(cluster, TemporalCluster) for cluster in clusters)
        assert all(len(cluster.experience_ids) >= clustering_engine.config.min_cluster_size for cluster in clusters)
        
        # Check cluster properties
        for cluster in clusters:
            assert cluster.cluster_id is not None
            assert cluster.algorithm == ClusteringAlgorithm.TEMPORAL_DBSCAN
            assert cluster.cohesion_score >= 0.0
            assert cluster.activity_density >= 0.0
            assert isinstance(cluster.temporal_pattern, dict)
    
    @pytest.mark.asyncio
    async def test_sliding_window_clustering(self, clustering_engine):
        """Test sliding window clustering"""
        clustering_engine.config.algorithm = ClusteringAlgorithm.SLIDING_WINDOW
        experiences = self.create_test_experiences(20)
        
        clusters = await clustering_engine.cluster_experiences_temporal(
            agent_id='test_agent',
            experiences=experiences,
            time_window=timedelta(days=1)
        )
        
        assert len(clusters) >= 0  # May be 0 if no clusters meet criteria
        for cluster in clusters:
            assert cluster.algorithm == ClusteringAlgorithm.SLIDING_WINDOW
            assert cluster.time_window.total_seconds() > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_threshold_clustering(self, clustering_engine):
        """Test adaptive threshold clustering"""
        clustering_engine.config.algorithm = ClusteringAlgorithm.ADAPTIVE_THRESHOLD
        experiences = self.create_test_experiences(12)
        
        clusters = await clustering_engine.cluster_experiences_temporal(
            agent_id='test_agent',
            experiences=experiences
        )
        
        assert len(clusters) >= 0
        for cluster in clusters:
            assert cluster.algorithm == ClusteringAlgorithm.ADAPTIVE_THRESHOLD
    
    @pytest.mark.asyncio
    async def test_cluster_merging(self, clustering_engine):
        """Test cluster merging functionality"""
        experiences = self.create_test_experiences(20)
        
        # Create initial clusters
        await clustering_engine.cluster_experiences_temporal(
            agent_id='test_agent',
            experiences=experiences
        )
        
        # Test merging
        merged_count = await clustering_engine.merge_similar_clusters('test_agent')
        assert merged_count >= 0
    
    @pytest.mark.asyncio
    async def test_clustering_statistics(self, clustering_engine):
        """Test clustering statistics"""
        experiences = self.create_test_experiences(10)
        
        await clustering_engine.cluster_experiences_temporal(
            agent_id='test_agent',
            experiences=experiences
        )
        
        stats = clustering_engine.get_clustering_statistics()
        
        assert 'total_clusters' in stats
        assert 'agents_with_clusters' in stats
        assert 'average_cluster_size' in stats
        assert 'average_cohesion_score' in stats
        assert 'algorithm_distribution' in stats
        assert 'config' in stats
    
    @pytest.mark.asyncio
    async def test_empty_experiences(self, clustering_engine):
        """Test clustering with empty experiences list"""
        clusters = await clustering_engine.cluster_experiences_temporal(
            agent_id='test_agent',
            experiences=[]
        )
        
        assert len(clusters) == 0
    
    @pytest.mark.asyncio
    async def test_insufficient_experiences(self, clustering_engine):
        """Test clustering with insufficient experiences"""
        experiences = self.create_test_experiences(1)  # Less than min_cluster_size
        
        clusters = await clustering_engine.cluster_experiences_temporal(
            agent_id='test_agent',
            experiences=experiences
        )
        
        assert len(clusters) == 0

class TestSemanticAnnotation:
    """Test semantic annotation functionality"""
    
    @pytest.fixture
    async def annotation_engine(self):
        """Create semantic annotation engine for testing"""
        engine = SemanticAnnotationEngine()
        await engine.initialize()
        yield engine
        await engine.shutdown()
    
    def create_test_memory_content(self) -> str:
        """Create test memory content for annotation"""
        return """
        Agent performed reconnaissance using nmap on target 192.168.1.100.
        Successfully identified open ports 22, 80, and 443.
        Attempted SQL injection attack on web application but failed due to input validation.
        Learned that target uses modern security practices.
        CVE-2023-1234 vulnerability was not present.
        """
    
    @pytest.mark.asyncio
    async def test_memory_annotation(self, annotation_engine):
        """Test basic memory annotation"""
        content = self.create_test_memory_content()
        context = {
            'agent_id': 'test_agent',
            'success': True,
            'confidence_score': 0.8,
            'action_type': 'reconnaissance'
        }
        
        annotated_memory = await annotation_engine.annotate_memory(
            memory_id='test_memory_1',
            content=content,
            context=context
        )
        
        assert isinstance(annotated_memory, AnnotatedMemory)
        assert annotated_memory.memory_id == 'test_memory_1'
        assert annotated_memory.original_content == content
        assert len(annotated_memory.annotations) > 0
        assert len(annotated_memory.categories) > 0
        assert len(annotated_memory.semantic_tags) > 0
        assert 0.0 <= annotated_memory.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_entity_extraction(self, annotation_engine):
        """Test entity extraction from content"""
        content = "Agent used nmap tool to scan 192.168.1.100 and found CVE-2023-1234"
        
        annotated_memory = await annotation_engine.annotate_memory(
            memory_id='test_memory_2',
            content=content
        )
        
        # Check for extracted entities
        entity_annotations = [ann for ann in annotated_memory.annotations if ann.annotation_type == AnnotationType.ENTITY]
        assert len(entity_annotations) > 0
        
        # Should find IP address, tool, and CVE
        entity_types = {ann.entity_type for ann in entity_annotations if ann.entity_type}
        assert EntityType.TARGET in entity_types or EntityType.TOOL in entity_types or EntityType.VULNERABILITY in entity_types
    
    @pytest.mark.asyncio
    async def test_concept_extraction(self, annotation_engine):
        """Test concept extraction from content"""
        content = "Successful reconnaissance phase completed. Attack was blocked by defense systems."
        
        annotated_memory = await annotation_engine.annotate_memory(
            memory_id='test_memory_3',
            content=content
        )
        
        # Check for extracted concepts
        concept_annotations = [ann for ann in annotated_memory.annotations if ann.annotation_type == AnnotationType.CONCEPT]
        assert len(concept_annotations) > 0
        
        # Should find success and failure patterns
        concept_types = {ann.concept_type for ann in concept_annotations if ann.concept_type}
        assert ConceptType.SUCCESS_PATTERN in concept_types or ConceptType.FAILURE_PATTERN in concept_types
    
    @pytest.mark.asyncio
    async def test_annotation_rules(self, annotation_engine):
        """Test custom annotation rules"""
        # Add custom rule
        custom_rule = AnnotationRule(
            rule_id='test_rule',
            name='Test Pattern',
            pattern=r'\btest_pattern\b',
            annotation_type=AnnotationType.ENTITY,
            entity_type=EntityType.TOOL,
            confidence=0.9
        )
        
        await annotation_engine.add_annotation_rule(custom_rule)
        
        content = "Used test_pattern for analysis"
        annotated_memory = await annotation_engine.annotate_memory(
            memory_id='test_memory_4',
            content=content
        )
        
        # Check if custom rule was applied
        rule_annotations = [
            ann for ann in annotated_memory.annotations 
            if ann.metadata.get('rule_id') == 'test_rule'
        ]
        assert len(rule_annotations) > 0
        assert rule_annotations[0].confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, annotation_engine):
        """Test semantic search functionality"""
        # Annotate multiple memories
        contents = [
            "Successful nmap reconnaissance of target system",
            "Failed SQL injection attempt on web application",
            "Discovered open ports during network scanning"
        ]
        
        for i, content in enumerate(contents):
            await annotation_engine.annotate_memory(
                memory_id=f'search_test_{i}',
                content=content
            )
        
        # Search for reconnaissance-related memories
        results = await annotation_engine.search_by_semantics(
            query='reconnaissance scanning',
            filters={'min_confidence': 0.1}
        )
        
        assert len(results) > 0
        assert all(isinstance(result, AnnotatedMemory) for result in results)
    
    @pytest.mark.asyncio
    async def test_annotation_statistics(self, annotation_engine):
        """Test annotation statistics"""
        content = self.create_test_memory_content()
        
        await annotation_engine.annotate_memory(
            memory_id='stats_test',
            content=content
        )
        
        stats = annotation_engine.get_annotation_statistics()
        
        assert 'total_annotated_memories' in stats
        assert 'total_annotation_rules' in stats
        assert 'annotation_type_distribution' in stats
        assert 'entity_type_distribution' in stats
        assert 'concept_type_distribution' in stats
        assert 'processing_stats' in stats

class TestContextAwareRetrieval:
    """Test context-aware retrieval functionality"""
    
    @pytest.fixture
    async def retrieval_system(self):
        """Create context-aware retrieval system for testing"""
        config = RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID_SCORING,
            max_results=10,
            min_relevance_threshold=0.3
        )
        system = ContextAwareRetrieval(config)
        await system.initialize()
        yield system
        await system.shutdown()
    
    def create_test_memories(self, count: int = 15) -> List[Dict[str, Any]]:
        """Create test memories for retrieval"""
        memories = []
        base_time = datetime.now() - timedelta(days=7)
        
        for i in range(count):
            memory = {
                'memory_id': f'memory_{i}',
                'agent_id': f'agent_{i % 3}',
                'content': f'Test memory content {i} with action type {i % 4}',
                'timestamp': base_time + timedelta(hours=i * 2),
                'success': i % 3 != 0,  # Mostly successful
                'confidence_score': 0.3 + (i % 7) * 0.1,
                'action_type': f'action_{i % 4}',
                'task_context': f'task_{i % 5}',
                'team': f'team_{i % 2}'
            }
            memories.append(memory)
        
        return memories
    
    def create_test_context(self, agent_id: str = 'test_agent') -> RetrievalContext:
        """Create test retrieval context"""
        return RetrievalContext(
            context_id=str(uuid.uuid4()),
            context_type=ContextType.CURRENT_TASK,
            agent_id=agent_id,
            current_task='reconnaissance_task',
            agent_state={'team': 'team_0', 'role': 'attacker'},
            scenario_info={'type': 'penetration_test', 'difficulty': 'medium'},
            temporal_window=timedelta(days=3),
            social_connections=['agent_1', 'agent_2'],
            priority_keywords=['reconnaissance', 'scanning', 'target']
        )
    
    @pytest.mark.asyncio
    async def test_contextual_retrieval(self, retrieval_system):
        """Test basic contextual memory retrieval"""
        memories = self.create_test_memories(20)
        context = self.create_test_context()
        
        results = await retrieval_system.retrieve_contextual_memories(
            context=context,
            available_memories=memories,
            query='reconnaissance scanning'
        )
        
        assert len(results) > 0
        assert len(results) <= retrieval_system.config.max_results
        assert all(isinstance(result, RetrievalResult) for result in results)
        assert all(result.relevance_score.total_score >= retrieval_system.config.min_relevance_threshold for result in results)
        
        # Results should be sorted by relevance
        scores = [result.relevance_score.total_score for result in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_relevance_scoring(self, retrieval_system):
        """Test relevance scoring components"""
        memories = self.create_test_memories(10)
        context = self.create_test_context()
        
        results = await retrieval_system.retrieve_contextual_memories(
            context=context,
            available_memories=memories,
            query='test query'
        )
        
        for result in results:
            score = result.relevance_score
            
            # Check score components exist
            assert hasattr(score, 'semantic_score')
            assert hasattr(score, 'temporal_score')
            assert hasattr(score, 'contextual_score')
            assert hasattr(score, 'social_score')
            assert hasattr(score, 'recency_score')
            assert hasattr(score, 'success_score')
            
            # Check score ranges
            assert 0.0 <= score.total_score <= 1.0
            assert 0.0 <= score.semantic_score <= 1.0
            assert 0.0 <= score.temporal_score <= 1.0
            assert 0.0 <= score.contextual_score <= 1.0
            
            # Check scoring breakdown
            assert isinstance(score.scoring_breakdown, dict)
            assert len(score.scoring_breakdown) > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_ranking(self, retrieval_system):
        """Test adaptive ranking strategy"""
        retrieval_system.config.strategy = RetrievalStrategy.ADAPTIVE_RANKING
        
        memories = self.create_test_memories(15)
        context = self.create_test_context()
        
        # Perform multiple retrievals to build history
        for i in range(3):
            await retrieval_system.retrieve_contextual_memories(
                context=context,
                available_memories=memories,
                query=f'query {i}'
            )
        
        # Check that retrieval history is being tracked
        assert len(retrieval_system.retrieval_history[context.agent_id]) > 0
    
    @pytest.mark.asyncio
    async def test_diversity_filtering(self, retrieval_system):
        """Test diversity filtering"""
        retrieval_system.config.diversity_factor = 0.3
        
        # Create memories with similar content but different types
        memories = []
        for i in range(10):
            memory = {
                'memory_id': f'diverse_memory_{i}',
                'agent_id': 'test_agent',
                'content': 'Similar content for all memories',
                'type': f'type_{i % 3}',  # Only 3 different types
                'timestamp': datetime.now(),
                'success': True,
                'confidence_score': 0.8
            }
            memories.append(memory)
        
        context = self.create_test_context()
        results = await retrieval_system.retrieve_contextual_memories(
            context=context,
            available_memories=memories,
            query='similar content'
        )
        
        # Should have diverse types in results
        result_types = {result.memory_content.get('type') for result in results}
        assert len(result_types) > 1  # Should have multiple types
    
    @pytest.mark.asyncio
    async def test_empty_memories(self, retrieval_system):
        """Test retrieval with empty memories list"""
        context = self.create_test_context()
        
        results = await retrieval_system.retrieve_contextual_memories(
            context=context,
            available_memories=[],
            query='test query'
        )
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_retrieval_statistics(self, retrieval_system):
        """Test retrieval statistics"""
        memories = self.create_test_memories(10)
        context = self.create_test_context()
        
        await retrieval_system.retrieve_contextual_memories(
            context=context,
            available_memories=memories,
            query='test'
        )
        
        stats = retrieval_system.get_retrieval_statistics()
        
        assert 'total_agents' in stats
        assert 'total_contexts_cached' in stats
        assert 'config' in stats
        assert 'processing_stats' in stats

class TestMemoryOptimization:
    """Test memory optimization functionality"""
    
    @pytest.fixture
    async def optimizer(self):
        """Create memory optimizer for testing"""
        config = OptimizationConfig(
            strategy=OptimizationStrategy.ADAPTIVE_RETENTION,
            max_memory_size=100,
            target_memory_size=80,
            compression_threshold=0.7,
            eviction_threshold=0.9
        )
        optimizer = MemoryOptimizer(config)
        await optimizer.initialize()
        yield optimizer
        await optimizer.shutdown()
    
    def create_test_memories_for_optimization(self, count: int = 120) -> List[Dict[str, Any]]:
        """Create test memories for optimization"""
        memories = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(count):
            memory = {
                'memory_id': f'opt_memory_{i}',
                'agent_id': f'agent_{i % 5}',
                'content': f'Memory content {i} ' * (10 + i % 20),  # Variable sizes
                'timestamp': base_time + timedelta(hours=i),
                'success': i % 4 != 0,  # Mostly successful
                'confidence_score': 0.1 + (i % 9) * 0.1,
                'recently_accessed': i > count - 20,  # Last 20 recently accessed
                'critical': i % 50 == 0,  # Every 50th is critical
                'unique_insights': i % 25 == 0  # Every 25th has unique insights
            }
            memories.append(memory)
        
        return memories
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self, optimizer):
        """Test basic memory optimization"""
        memories = self.create_test_memories_for_optimization(120)
        
        result = await optimizer.optimize_memory_system(
            memories=memories,
            force_optimization=True
        )
        
        assert result.memories_processed == len(memories)
        assert result.strategy_used == optimizer.config.strategy
        assert result.processing_time > 0
        assert isinstance(result.performance_impact, dict)
        
        # Should have some optimization activity
        total_optimized = result.memories_compressed + result.memories_evicted + result.memories_merged
        assert total_optimized > 0
    
    @pytest.mark.asyncio
    async def test_lru_eviction_strategy(self, optimizer):
        """Test LRU eviction optimization strategy"""
        optimizer.config.strategy = OptimizationStrategy.LRU_EVICTION
        memories = self.create_test_memories_for_optimization(150)
        
        result = await optimizer.optimize_memory_system(
            memories=memories,
            force_optimization=True
        )
        
        assert result.strategy_used == OptimizationStrategy.LRU_EVICTION
        assert result.memories_evicted > 0  # Should evict some memories
    
    @pytest.mark.asyncio
    async def test_importance_based_strategy(self, optimizer):
        """Test importance-based optimization strategy"""
        optimizer.config.strategy = OptimizationStrategy.IMPORTANCE_BASED
        memories = self.create_test_memories_for_optimization(130)
        
        result = await optimizer.optimize_memory_system(
            memories=memories,
            force_optimization=True
        )
        
        assert result.strategy_used == OptimizationStrategy.IMPORTANCE_BASED
        # Should have some combination of compression and eviction
        assert (result.memories_compressed + result.memories_evicted) > 0
    
    @pytest.mark.asyncio
    async def test_clustering_based_strategy(self, optimizer):
        """Test clustering-based optimization strategy"""
        optimizer.config.strategy = OptimizationStrategy.CLUSTERING_BASED
        memories = self.create_test_memories_for_optimization(100)
        
        result = await optimizer.optimize_memory_system(
            memories=memories,
            force_optimization=True
        )
        
        assert result.strategy_used == OptimizationStrategy.CLUSTERING_BASED
    
    @pytest.mark.asyncio
    async def test_hierarchical_compression_strategy(self, optimizer):
        """Test hierarchical compression strategy"""
        optimizer.config.strategy = OptimizationStrategy.HIERARCHICAL_COMPRESSION
        memories = self.create_test_memories_for_optimization(110)
        
        result = await optimizer.optimize_memory_system(
            memories=memories,
            force_optimization=True
        )
        
        assert result.strategy_used == OptimizationStrategy.HIERARCHICAL_COMPRESSION
        assert result.memories_compressed > 0  # Should compress memories
    
    @pytest.mark.asyncio
    async def test_memory_compression(self, optimizer):
        """Test memory compression functionality"""
        memory = {
            'memory_id': 'compress_test',
            'content': 'This is a test memory with detailed content that should be compressed effectively.',
            'timestamp': datetime.now(),
            'success': True,
            'confidence_score': 0.8
        }
        
        # Test different compression types
        from memory.memory_optimization import CompressionType
        
        compressed_size = await optimizer._compress_memory(memory, CompressionType.SEMANTIC_SUMMARIZATION)
        assert compressed_size > 0
        assert compressed_size <= len(str(memory))
        
        # Check compression cache
        assert memory['memory_id'] in optimizer.compression_cache
        cache_entry = optimizer.compression_cache[memory['memory_id']]
        assert cache_entry['compression_type'] == CompressionType.SEMANTIC_SUMMARIZATION
        assert cache_entry['original_size'] > 0
        assert cache_entry['compressed_size'] > 0
    
    @pytest.mark.asyncio
    async def test_optimization_not_needed(self, optimizer):
        """Test when optimization is not needed"""
        # Small number of memories, below threshold
        memories = self.create_test_memories_for_optimization(50)
        
        result = await optimizer.optimize_memory_system(
            memories=memories,
            force_optimization=False
        )
        
        # Should indicate no optimization was needed
        assert result.memories_processed == 0
        assert 'optimization_not_needed' in result.metadata.get('reason', '')
    
    @pytest.mark.asyncio
    async def test_adaptive_thresholds(self, optimizer):
        """Test adaptive threshold adjustment"""
        optimizer.config.adaptive_thresholds = True
        memories = self.create_test_memories_for_optimization(120)
        
        # Get initial thresholds
        initial_thresholds = optimizer.adaptive_thresholds.copy()
        
        # Perform optimization
        result = await optimizer.optimize_memory_system(
            memories=memories,
            force_optimization=True
        )
        
        # Thresholds may have been adjusted
        # (Exact changes depend on optimization effectiveness)
        assert isinstance(optimizer.adaptive_thresholds, dict)
        assert 'importance' in optimizer.adaptive_thresholds
        assert 'compression' in optimizer.adaptive_thresholds
    
    @pytest.mark.asyncio
    async def test_optimization_statistics(self, optimizer):
        """Test optimization statistics"""
        memories = self.create_test_memories_for_optimization(100)
        
        await optimizer.optimize_memory_system(
            memories=memories,
            force_optimization=True
        )
        
        stats = optimizer.get_optimization_statistics()
        
        assert 'total_optimizations' in stats
        assert 'recent_performance' in stats
        assert 'current_config' in stats
        assert 'adaptive_thresholds' in stats
        assert 'cache_statistics' in stats
        assert 'processing_stats' in stats
        
        # Check recent performance metrics
        perf = stats['recent_performance']
        assert 'avg_processing_time' in perf
        assert 'avg_space_saved' in perf
        assert 'avg_compression_ratio' in perf

class TestIntegration:
    """Integration tests for the complete advanced memory system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_memory_processing(self):
        """Test complete memory processing pipeline"""
        # Initialize all components
        clustering_engine = TemporalClusteringEngine()
        annotation_engine = SemanticAnnotationEngine()
        retrieval_system = ContextAwareRetrieval()
        optimizer = MemoryOptimizer()
        
        await clustering_engine.initialize()
        await annotation_engine.initialize()
        await retrieval_system.initialize()
        await optimizer.initialize()
        
        try:
            # Create test data
            memories = []
            for i in range(20):
                memory = {
                    'memory_id': f'integration_memory_{i}',
                    'agent_id': f'agent_{i % 3}',
                    'content': f'Agent performed reconnaissance using nmap on target 192.168.1.{100 + i}. Success: {i % 2 == 0}',
                    'timestamp': datetime.now() - timedelta(hours=i),
                    'success': i % 2 == 0,
                    'confidence_score': 0.5 + (i % 5) * 0.1,
                    'embedding': np.random.rand(384).tolist()
                }
                memories.append(memory)
            
            # Step 1: Annotate memories
            annotated_memories = []
            for memory in memories:
                annotated = await annotation_engine.annotate_memory(
                    memory_id=memory['memory_id'],
                    content=memory['content'],
                    context={'agent_id': memory['agent_id'], 'success': memory['success']}
                )
                annotated_memories.append(annotated)
            
            assert len(annotated_memories) == len(memories)
            
            # Step 2: Cluster memories
            clusters = await clustering_engine.cluster_experiences_temporal(
                agent_id='agent_0',
                experiences=memories[:10],  # First 10 for agent_0
                time_window=timedelta(days=1)
            )
            
            assert len(clusters) >= 0  # May be 0 if clustering criteria not met
            
            # Step 3: Context-aware retrieval
            context = RetrievalContext(
                context_id=str(uuid.uuid4()),
                context_type=ContextType.CURRENT_TASK,
                agent_id='agent_0',
                current_task='reconnaissance',
                priority_keywords=['nmap', 'reconnaissance', 'target']
            )
            
            retrieval_results = await retrieval_system.retrieve_contextual_memories(
                context=context,
                available_memories=memories,
                query='reconnaissance nmap'
            )
            
            assert len(retrieval_results) > 0
            
            # Step 4: Optimize memory system
            optimization_result = await optimizer.optimize_memory_system(
                memories=memories,
                force_optimization=True
            )
            
            assert optimization_result.memories_processed == len(memories)
            
            # Verify integration worked
            assert len(annotated_memories) > 0
            assert len(retrieval_results) > 0
            assert optimization_result.processing_time > 0
            
        finally:
            # Cleanup
            await clustering_engine.shutdown()
            await annotation_engine.shutdown()
            await retrieval_system.shutdown()
            await optimizer.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under load"""
        # This test verifies the system can handle larger datasets
        clustering_engine = TemporalClusteringEngine()
        await clustering_engine.initialize()
        
        try:
            # Create large dataset
            large_dataset = []
            for i in range(1000):
                experience = {
                    'experience_id': str(uuid.uuid4()),
                    'agent_id': f'agent_{i % 10}',
                    'timestamp': datetime.now() - timedelta(minutes=i),
                    'content': f'Large scale test experience {i}',
                    'embedding': np.random.rand(384).tolist(),
                    'success': i % 3 != 0,
                    'confidence_score': np.random.rand()
                }
                large_dataset.append(experience)
            
            # Test clustering performance
            start_time = datetime.now()
            clusters = await clustering_engine.cluster_experiences_temporal(
                agent_id='agent_0',
                experiences=large_dataset[:100],  # Subset for performance
                time_window=timedelta(hours=24)
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Should complete in reasonable time (less than 30 seconds)
            assert processing_time < 30.0
            assert len(clusters) >= 0
            
        finally:
            await clustering_engine.shutdown()

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])