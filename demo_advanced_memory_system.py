#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Advanced Memory System Demo
Comprehensive demonstration of advanced memory clustering and retrieval capabilities
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid

# Import advanced memory system components
from memory.temporal_clustering import (
    TemporalClusteringEngine, ClusteringConfig, ClusteringAlgorithm
)
from memory.semantic_annotation import (
    SemanticAnnotationEngine, AnnotationType, EntityType, ConceptType
)
from memory.context_retrieval import (
    ContextAwareRetrieval, RetrievalConfig, RetrievalStrategy,
    RetrievalContext, ContextType
)
from memory.memory_optimization import (
    MemoryOptimizer, OptimizationConfig, OptimizationStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedMemorySystemDemo:
    """
    Comprehensive demo of the advanced memory clustering and retrieval system.
    
    Demonstrates:
    - Temporal clustering algorithms
    - Semantic annotation and categorization
    - Context-aware memory retrieval
    - Memory optimization and garbage collection
    - Integration between all components
    """
    
    def __init__(self):
        self.clustering_engine = None
        self.annotation_engine = None
        self.retrieval_system = None
        self.optimizer = None
        
        # Demo data
        self.demo_memories = []
        self.demo_agents = ['red_team_1', 'red_team_2', 'blue_team_1', 'blue_team_2', 'coordinator']
        
    async def initialize_systems(self):
        """Initialize all memory system components"""
        logger.info("Initializing advanced memory system components...")
        
        # Initialize temporal clustering
        clustering_config = ClusteringConfig(
            algorithm=ClusteringAlgorithm.TEMPORAL_DBSCAN,
            min_cluster_size=3,
            max_cluster_size=20,
            time_epsilon=timedelta(hours=2),
            similarity_threshold=0.7,
            max_clusters_per_agent=15
        )
        self.clustering_engine = TemporalClusteringEngine(clustering_config)
        await self.clustering_engine.initialize()
        
        # Initialize semantic annotation
        self.annotation_engine = SemanticAnnotationEngine()
        await self.annotation_engine.initialize()
        
        # Initialize context-aware retrieval
        retrieval_config = RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID_SCORING,
            max_results=15,
            min_relevance_threshold=0.3,
            semantic_weight=0.3,
            temporal_weight=0.2,
            contextual_weight=0.25,
            social_weight=0.15,
            recency_weight=0.1,
            diversity_factor=0.2,
            adaptive_scoring_enabled=True
        )
        self.retrieval_system = ContextAwareRetrieval(retrieval_config)
        await self.retrieval_system.initialize()
        
        # Initialize memory optimizer
        optimization_config = OptimizationConfig(
            strategy=OptimizationStrategy.ADAPTIVE_RETENTION,
            max_memory_size=1000,
            target_memory_size=800,
            compression_threshold=0.6,
            eviction_threshold=0.85,
            compression_enabled=True,
            adaptive_thresholds=True
        )
        self.optimizer = MemoryOptimizer(optimization_config)
        await self.optimizer.initialize()
        
        logger.info("All memory system components initialized successfully")
    
    def generate_demo_memories(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate realistic demo memories for testing"""
        logger.info(f"Generating {count} demo memories...")
        
        memories = []
        base_time = datetime.now() - timedelta(days=14)
        
        # Memory templates for different scenarios
        memory_templates = [
            {
                'type': 'reconnaissance',
                'content_template': 'Agent {agent} performed {action} on target {target}. Discovered {findings}. Success: {success}',
                'actions': ['nmap scan', 'port enumeration', 'service fingerprinting', 'vulnerability scan'],
                'findings': ['open ports 22,80,443', 'Apache 2.4.41 server', 'SSH service', 'potential SQLi endpoint'],
                'success_rate': 0.8
            },
            {
                'type': 'exploitation',
                'content_template': 'Agent {agent} attempted {action} against {target}. Result: {result}. Technique: {technique}',
                'actions': ['SQL injection', 'buffer overflow', 'privilege escalation', 'lateral movement'],
                'results': ['gained shell access', 'escalated privileges', 'access denied', 'connection timeout'],
                'techniques': ['T1190', 'T1068', 'T1021', 'T1055'],
                'success_rate': 0.6
            },
            {
                'type': 'defense',
                'content_template': 'Blue team {agent} detected {threat} from {source}. Action taken: {action}. Status: {status}',
                'threats': ['port scan', 'malicious payload', 'unauthorized access', 'data exfiltration'],
                'actions': ['blocked IP', 'quarantined system', 'updated firewall rules', 'alerted SOC'],
                'statuses': ['contained', 'investigating', 'resolved', 'escalated'],
                'success_rate': 0.9
            },
            {
                'type': 'learning',
                'content_template': 'Agent {agent} learned: {lesson}. Context: {context}. Confidence: {confidence}',
                'lessons': [
                    'Target uses WAF protection',
                    'SSH keys required for access',
                    'Network segmentation detected',
                    'IDS monitoring active'
                ],
                'contexts': ['failed attack attempt', 'successful reconnaissance', 'defense analysis', 'post-exploitation'],
                'success_rate': 0.7
            }
        ]
        
        for i in range(count):
            template = np.random.choice(memory_templates)
            agent = np.random.choice(self.demo_agents)
            
            # Generate memory content
            if template['type'] == 'reconnaissance':
                content = template['content_template'].format(
                    agent=agent,
                    action=np.random.choice(template['actions']),
                    target=f"192.168.1.{100 + i % 50}",
                    findings=np.random.choice(template['findings']),
                    success=np.random.random() < template['success_rate']
                )
            elif template['type'] == 'exploitation':
                content = template['content_template'].format(
                    agent=agent,
                    action=np.random.choice(template['actions']),
                    target=f"target_system_{i % 20}",
                    result=np.random.choice(template['results']),
                    technique=np.random.choice(template['techniques'])
                )
            elif template['type'] == 'defense':
                content = template['content_template'].format(
                    agent=agent,
                    threat=np.random.choice(template['threats']),
                    source=f"192.168.1.{200 + i % 30}",
                    action=np.random.choice(template['actions']),
                    status=np.random.choice(template['statuses'])
                )
            else:  # learning
                content = template['content_template'].format(
                    agent=agent,
                    lesson=np.random.choice(template['lessons']),
                    context=np.random.choice(template['contexts']),
                    confidence=f"{0.5 + np.random.random() * 0.5:.2f}"
                )
            
            # Create memory object
            memory = {
                'memory_id': str(uuid.uuid4()),
                'agent_id': agent,
                'content': content,
                'type': template['type'],
                'timestamp': base_time + timedelta(
                    hours=i * 2 + np.random.randint(-60, 60)  # Add some randomness
                ),
                'success': np.random.random() < template['success_rate'],
                'confidence_score': 0.3 + np.random.random() * 0.7,
                'embedding': np.random.rand(384).tolist(),  # Simulated embedding
                'action_type': template['type'],
                'task_context': f"task_{template['type']}_{i % 10}",
                'team': 'red_team' if 'red_team' in agent else 'blue_team' if 'blue_team' in agent else 'coordination',
                'scenario_info': {
                    'type': 'penetration_test',
                    'difficulty': np.random.choice(['easy', 'medium', 'hard']),
                    'environment': np.random.choice(['corporate', 'cloud', 'industrial'])
                },
                'lessons_learned': [
                    f"Lesson from memory {i}",
                    f"Insight about {template['type']} operations"
                ] if np.random.random() < 0.3 else [],
                'mitre_attack_mapping': [np.random.choice(['T1190', 'T1068', 'T1021', 'T1055', 'T1595'])] if np.random.random() < 0.4 else []
            }
            
            memories.append(memory)
        
        self.demo_memories = memories
        logger.info(f"Generated {len(memories)} demo memories")
        return memories
    
    async def demonstrate_semantic_annotation(self):
        """Demonstrate semantic annotation capabilities"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING SEMANTIC ANNOTATION")
        logger.info("="*60)
        
        # Select sample memories for annotation
        sample_memories = self.demo_memories[:10]
        annotated_memories = []
        
        for memory in sample_memories:
            logger.info(f"\nAnnotating memory from {memory['agent_id']}:")
            logger.info(f"Content: {memory['content'][:100]}...")
            
            # Annotate memory
            annotated = await self.annotation_engine.annotate_memory(
                memory_id=memory['memory_id'],
                content=memory['content'],
                context={
                    'agent_id': memory['agent_id'],
                    'success': memory['success'],
                    'confidence_score': memory['confidence_score'],
                    'action_type': memory['action_type'],
                    'team': memory['team']
                }
            )
            
            annotated_memories.append(annotated)
            
            # Display annotation results
            logger.info(f"Annotations found: {len(annotated.annotations)}")
            logger.info(f"Categories: {annotated.categories}")
            logger.info(f"Semantic tags: {list(annotated.semantic_tags)[:5]}...")  # First 5 tags
            logger.info(f"Confidence score: {annotated.confidence_score:.3f}")
            
            # Show specific annotations
            for ann in annotated.annotations[:3]:  # First 3 annotations
                logger.info(f"  - {ann.annotation_type.value}: '{ann.content}' (confidence: {ann.confidence:.2f})")
        
        # Demonstrate semantic search
        logger.info(f"\nDemonstrating semantic search...")
        search_results = await self.annotation_engine.search_by_semantics(
            query='reconnaissance scanning target',
            filters={'min_confidence': 0.5}
        )
        
        logger.info(f"Found {len(search_results)} memories matching 'reconnaissance scanning target'")
        for result in search_results[:3]:
            logger.info(f"  - Memory {result.memory_id}: {result.original_content[:80]}...")
        
        # Show annotation statistics
        stats = self.annotation_engine.get_annotation_statistics()
        logger.info(f"\nAnnotation Statistics:")
        logger.info(f"  - Total annotated memories: {stats['total_annotated_memories']}")
        logger.info(f"  - Active annotation rules: {stats['active_rules']}")
        logger.info(f"  - Entity types found: {list(stats['entity_type_distribution'].keys())}")
        logger.info(f"  - Concept types found: {list(stats['concept_type_distribution'].keys())}")
        
        return annotated_memories
    
    async def demonstrate_temporal_clustering(self):
        """Demonstrate temporal clustering capabilities"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING TEMPORAL CLUSTERING")
        logger.info("="*60)
        
        # Test different clustering algorithms
        algorithms = [
            ClusteringAlgorithm.TEMPORAL_DBSCAN,
            ClusteringAlgorithm.SLIDING_WINDOW,
            ClusteringAlgorithm.ADAPTIVE_THRESHOLD
        ]
        
        all_clusters = {}
        
        for algorithm in algorithms:
            logger.info(f"\nTesting {algorithm.value} clustering...")
            
            # Update algorithm
            self.clustering_engine.config.algorithm = algorithm
            
            # Cluster memories for each agent
            for agent_id in self.demo_agents[:3]:  # Test first 3 agents
                agent_memories = [m for m in self.demo_memories if m['agent_id'] == agent_id]
                
                if len(agent_memories) < 3:
                    continue
                
                clusters = await self.clustering_engine.cluster_experiences_temporal(
                    agent_id=agent_id,
                    experiences=agent_memories,
                    time_window=timedelta(days=7)
                )
                
                all_clusters[f"{agent_id}_{algorithm.value}"] = clusters
                
                logger.info(f"  Agent {agent_id}: {len(clusters)} clusters from {len(agent_memories)} memories")
                
                for i, cluster in enumerate(clusters[:2]):  # Show first 2 clusters
                    logger.info(f"    Cluster {i+1}: {len(cluster.experience_ids)} experiences, "
                              f"cohesion: {cluster.cohesion_score:.3f}, "
                              f"activity density: {cluster.activity_density:.2f}")
                    logger.info(f"    Time span: {cluster.start_time.strftime('%H:%M')} - "
                              f"{cluster.end_time.strftime('%H:%M')}")
        
        # Demonstrate cluster merging
        logger.info(f"\nDemonstrating cluster merging...")
        for agent_id in self.demo_agents[:2]:
            merged_count = await self.clustering_engine.merge_similar_clusters(agent_id)
            logger.info(f"  Agent {agent_id}: merged {merged_count} similar clusters")
        
        # Show clustering statistics
        stats = self.clustering_engine.get_clustering_statistics()
        logger.info(f"\nClustering Statistics:")
        logger.info(f"  - Total clusters: {stats['total_clusters']}")
        logger.info(f"  - Agents with clusters: {stats['agents_with_clusters']}")
        logger.info(f"  - Average cluster size: {stats['average_cluster_size']:.2f}")
        logger.info(f"  - Average cohesion score: {stats['average_cohesion_score']:.3f}")
        logger.info(f"  - Algorithm distribution: {stats['algorithm_distribution']}")
        
        return all_clusters
    
    async def demonstrate_context_aware_retrieval(self):
        """Demonstrate context-aware memory retrieval"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING CONTEXT-AWARE RETRIEVAL")
        logger.info("="*60)
        
        # Test different retrieval scenarios
        test_scenarios = [
            {
                'name': 'Red Team Reconnaissance',
                'context': RetrievalContext(
                    context_id=str(uuid.uuid4()),
                    context_type=ContextType.CURRENT_TASK,
                    agent_id='red_team_1',
                    current_task='reconnaissance_phase',
                    agent_state={'team': 'red_team', 'role': 'attacker', 'experience_level': 'expert'},
                    scenario_info={'type': 'penetration_test', 'difficulty': 'medium', 'environment': 'corporate'},
                    temporal_window=timedelta(days=3),
                    social_connections=['red_team_2', 'coordinator'],
                    priority_keywords=['reconnaissance', 'scanning', 'enumeration', 'target'],
                    environmental_factors={'network_type': 'corporate', 'security_level': 'medium'}
                ),
                'query': 'reconnaissance scanning target enumeration'
            },
            {
                'name': 'Blue Team Defense',
                'context': RetrievalContext(
                    context_id=str(uuid.uuid4()),
                    context_type=ContextType.SCENARIO_CONTEXT,
                    agent_id='blue_team_1',
                    current_task='threat_detection',
                    agent_state={'team': 'blue_team', 'role': 'defender', 'alert_level': 'high'},
                    scenario_info={'type': 'incident_response', 'difficulty': 'hard', 'environment': 'cloud'},
                    temporal_window=timedelta(hours=6),
                    social_connections=['blue_team_2'],
                    priority_keywords=['detection', 'blocking', 'threat', 'malicious'],
                    environmental_factors={'threat_level': 'high', 'response_time': 'critical'}
                ),
                'query': 'threat detection malicious activity blocking'
            },
            {
                'name': 'Learning from Failures',
                'context': RetrievalContext(
                    context_id=str(uuid.uuid4()),
                    context_type=ContextType.AGENT_STATE,
                    agent_id='red_team_2',
                    current_task='post_analysis',
                    agent_state={'team': 'red_team', 'recent_failures': 3, 'learning_mode': True},
                    scenario_info={'type': 'training', 'difficulty': 'medium'},
                    temporal_window=timedelta(days=1),
                    priority_keywords=['failed', 'blocked', 'denied', 'lesson', 'learning'],
                    environmental_factors={'learning_focus': 'failure_analysis'}
                ),
                'query': 'failed attempts lessons learned blocked'
            }
        ]
        
        retrieval_results = {}
        
        for scenario in test_scenarios:
            logger.info(f"\nScenario: {scenario['name']}")
            logger.info(f"Agent: {scenario['context'].agent_id}")
            logger.info(f"Query: '{scenario['query']}'")
            logger.info(f"Context: {scenario['context'].current_task}")
            
            # Perform retrieval
            results = await self.retrieval_system.retrieve_contextual_memories(
                context=scenario['context'],
                available_memories=self.demo_memories,
                query=scenario['query']
            )
            
            retrieval_results[scenario['name']] = results
            
            logger.info(f"Retrieved {len(results)} relevant memories")
            
            # Show top results
            for i, result in enumerate(results[:3]):
                score = result.relevance_score
                logger.info(f"  Result {i+1} (score: {score.total_score:.3f}):")
                logger.info(f"    Memory: {result.memory_content['content'][:80]}...")
                logger.info(f"    Reason: {result.retrieval_reason}")
                logger.info(f"    Breakdown: semantic={score.semantic_score:.2f}, "
                          f"temporal={score.temporal_score:.2f}, "
                          f"contextual={score.contextual_score:.2f}")
        
        # Test adaptive ranking
        logger.info(f"\nTesting adaptive ranking...")
        self.retrieval_system.config.strategy = RetrievalStrategy.ADAPTIVE_RANKING
        
        # Perform multiple retrievals to build history
        test_context = test_scenarios[0]['context']
        for i in range(3):
            await self.retrieval_system.retrieve_contextual_memories(
                context=test_context,
                available_memories=self.demo_memories,
                query=f"test query {i}"
            )
        
        logger.info(f"Built retrieval history for adaptive learning")
        
        # Show retrieval statistics
        stats = self.retrieval_system.get_retrieval_statistics()
        logger.info(f"\nRetrieval Statistics:")
        logger.info(f"  - Total agents: {stats['total_agents']}")
        logger.info(f"  - Contexts cached: {stats['total_contexts_cached']}")
        logger.info(f"  - Adaptive agents: {stats['adaptive_agents']}")
        logger.info(f"  - Strategy: {stats['config']['strategy']}")
        
        return retrieval_results
    
    async def demonstrate_memory_optimization(self):
        """Demonstrate memory optimization and garbage collection"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING MEMORY OPTIMIZATION")
        logger.info("="*60)
        
        # Create a larger dataset for optimization testing
        large_dataset = self.generate_demo_memories(150)
        
        # Test different optimization strategies
        strategies = [
            OptimizationStrategy.IMPORTANCE_BASED,
            OptimizationStrategy.LRU_EVICTION,
            OptimizationStrategy.ADAPTIVE_RETENTION,
            OptimizationStrategy.HIERARCHICAL_COMPRESSION
        ]
        
        optimization_results = {}
        
        for strategy in strategies:
            logger.info(f"\nTesting {strategy.value} optimization...")
            
            # Update strategy
            self.optimizer.config.strategy = strategy
            
            # Perform optimization
            result = await self.optimizer.optimize_memory_system(
                memories=large_dataset,
                force_optimization=True
            )
            
            optimization_results[strategy.value] = result
            
            logger.info(f"  Processed: {result.memories_processed} memories")
            logger.info(f"  Compressed: {result.memories_compressed}")
            logger.info(f"  Evicted: {result.memories_evicted}")
            logger.info(f"  Merged: {result.memories_merged}")
            logger.info(f"  Space saved: {result.space_saved} bytes")
            logger.info(f"  Processing time: {result.processing_time:.3f} seconds")
            logger.info(f"  Space efficiency: {result.performance_impact.get('space_efficiency', 0):.3f}")
        
        # Demonstrate compression
        logger.info(f"\nDemonstrating memory compression...")
        test_memory = large_dataset[0]
        
        from memory.memory_optimization import CompressionType
        compression_types = [
            CompressionType.SEMANTIC_SUMMARIZATION,
            CompressionType.TEMPORAL_AGGREGATION,
            CompressionType.PATTERN_EXTRACTION
        ]
        
        original_size = len(str(test_memory))
        logger.info(f"Original memory size: {original_size} bytes")
        
        for comp_type in compression_types:
            compressed_size = await self.optimizer._compress_memory(test_memory, comp_type)
            compression_ratio = compressed_size / original_size
            logger.info(f"  {comp_type.value}: {compressed_size} bytes (ratio: {compression_ratio:.2f})")
        
        # Show optimization statistics
        stats = self.optimizer.get_optimization_statistics()
        logger.info(f"\nOptimization Statistics:")
        logger.info(f"  - Total optimizations: {stats['total_optimizations']}")
        logger.info(f"  - Average processing time: {stats['recent_performance']['avg_processing_time']:.3f}s")
        logger.info(f"  - Average space saved: {stats['recent_performance']['avg_space_saved']:.0f} bytes")
        logger.info(f"  - Average compression ratio: {stats['recent_performance']['avg_compression_ratio']:.3f}")
        logger.info(f"  - Current strategy: {stats['current_config']['strategy']}")
        logger.info(f"  - Adaptive thresholds: {stats['adaptive_thresholds']}")
        
        return optimization_results
    
    async def demonstrate_integration(self):
        """Demonstrate integration between all components"""
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING SYSTEM INTEGRATION")
        logger.info("="*60)
        
        # Select a subset of memories for integration demo
        integration_memories = self.demo_memories[:50]
        
        logger.info(f"Processing {len(integration_memories)} memories through complete pipeline...")
        
        # Step 1: Semantic annotation
        logger.info("\nStep 1: Semantic annotation...")
        annotated_count = 0
        for memory in integration_memories:
            await self.annotation_engine.annotate_memory(
                memory_id=memory['memory_id'],
                content=memory['content'],
                context={'agent_id': memory['agent_id'], 'success': memory['success']}
            )
            annotated_count += 1
        
        logger.info(f"Annotated {annotated_count} memories")
        
        # Step 2: Temporal clustering
        logger.info("\nStep 2: Temporal clustering...")
        total_clusters = 0
        for agent_id in self.demo_agents:
            agent_memories = [m for m in integration_memories if m['agent_id'] == agent_id]
            if len(agent_memories) >= 3:
                clusters = await self.clustering_engine.cluster_experiences_temporal(
                    agent_id=agent_id,
                    experiences=agent_memories,
                    time_window=timedelta(days=5)
                )
                total_clusters += len(clusters)
        
        logger.info(f"Created {total_clusters} temporal clusters")
        
        # Step 3: Context-aware retrieval
        logger.info("\nStep 3: Context-aware retrieval...")
        test_context = RetrievalContext(
            context_id=str(uuid.uuid4()),
            context_type=ContextType.CURRENT_TASK,
            agent_id='red_team_1',
            current_task='comprehensive_analysis',
            priority_keywords=['reconnaissance', 'exploitation', 'success']
        )
        
        retrieval_results = await self.retrieval_system.retrieve_contextual_memories(
            context=test_context,
            available_memories=integration_memories,
            query='successful reconnaissance and exploitation'
        )
        
        logger.info(f"Retrieved {len(retrieval_results)} contextually relevant memories")
        
        # Step 4: Memory optimization
        logger.info("\nStep 4: Memory optimization...")
        optimization_result = await self.optimizer.optimize_memory_system(
            memories=integration_memories,
            force_optimization=True
        )
        
        logger.info(f"Optimized {optimization_result.memories_processed} memories")
        logger.info(f"Compression/eviction: {optimization_result.memories_compressed + optimization_result.memories_evicted}")
        
        # Integration summary
        logger.info(f"\nIntegration Summary:")
        logger.info(f"  - Memories processed: {len(integration_memories)}")
        logger.info(f"  - Annotations created: {annotated_count}")
        logger.info(f"  - Clusters formed: {total_clusters}")
        logger.info(f"  - Relevant retrievals: {len(retrieval_results)}")
        logger.info(f"  - Optimization savings: {optimization_result.space_saved} bytes")
        
        # Performance metrics
        annotation_stats = self.annotation_engine.get_annotation_statistics()
        clustering_stats = self.clustering_engine.get_clustering_statistics()
        retrieval_stats = self.retrieval_system.get_retrieval_statistics()
        optimization_stats = self.optimizer.get_optimization_statistics()
        
        logger.info(f"\nSystem Performance Metrics:")
        logger.info(f"  - Annotation confidence: {annotation_stats.get('processing_stats', {}).get('memories_annotated', 0) / len(integration_memories):.2f}")
        logger.info(f"  - Clustering efficiency: {clustering_stats.get('average_cohesion_score', 0):.3f}")
        logger.info(f"  - Retrieval relevance: {sum(r.relevance_score.total_score for r in retrieval_results) / len(retrieval_results):.3f}" if retrieval_results else "N/A")
        logger.info(f"  - Optimization efficiency: {optimization_result.performance_impact.get('space_efficiency', 0):.3f}")
        
        return {
            'annotated_memories': annotated_count,
            'clusters_created': total_clusters,
            'retrieval_results': len(retrieval_results),
            'optimization_result': optimization_result
        }
    
    async def run_performance_benchmark(self):
        """Run performance benchmarks on the system"""
        logger.info("\n" + "="*60)
        logger.info("RUNNING PERFORMANCE BENCHMARKS")
        logger.info("="*60)
        
        # Generate larger dataset for benchmarking
        benchmark_memories = self.generate_demo_memories(500)
        
        # Benchmark annotation performance
        logger.info("\nBenchmarking annotation performance...")
        start_time = datetime.now()
        
        annotation_count = 0
        for memory in benchmark_memories[:100]:  # Test with 100 memories
            await self.annotation_engine.annotate_memory(
                memory_id=memory['memory_id'],
                content=memory['content'],
                context={'agent_id': memory['agent_id']}
            )
            annotation_count += 1
        
        annotation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Annotated {annotation_count} memories in {annotation_time:.2f} seconds")
        logger.info(f"Average annotation time: {annotation_time / annotation_count:.3f} seconds per memory")
        
        # Benchmark clustering performance
        logger.info("\nBenchmarking clustering performance...")
        start_time = datetime.now()
        
        cluster_count = 0
        for agent_id in self.demo_agents:
            agent_memories = [m for m in benchmark_memories if m['agent_id'] == agent_id][:50]  # Limit for performance
            if len(agent_memories) >= 3:
                clusters = await self.clustering_engine.cluster_experiences_temporal(
                    agent_id=agent_id,
                    experiences=agent_memories,
                    time_window=timedelta(days=7)
                )
                cluster_count += len(clusters)
        
        clustering_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Created {cluster_count} clusters in {clustering_time:.2f} seconds")
        
        # Benchmark retrieval performance
        logger.info("\nBenchmarking retrieval performance...")
        test_context = RetrievalContext(
            context_id=str(uuid.uuid4()),
            context_type=ContextType.CURRENT_TASK,
            agent_id='red_team_1',
            current_task='benchmark_test'
        )
        
        start_time = datetime.now()
        retrieval_results = await self.retrieval_system.retrieve_contextual_memories(
            context=test_context,
            available_memories=benchmark_memories,
            query='benchmark test query'
        )
        retrieval_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Retrieved {len(retrieval_results)} memories in {retrieval_time:.3f} seconds")
        logger.info(f"Retrieval rate: {len(benchmark_memories) / retrieval_time:.0f} memories/second processed")
        
        # Benchmark optimization performance
        logger.info("\nBenchmarking optimization performance...")
        start_time = datetime.now()
        
        optimization_result = await self.optimizer.optimize_memory_system(
            memories=benchmark_memories,
            force_optimization=True
        )
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Optimized {optimization_result.memories_processed} memories in {optimization_time:.2f} seconds")
        logger.info(f"Optimization rate: {optimization_result.memories_processed / optimization_time:.0f} memories/second")
        
        # Overall performance summary
        logger.info(f"\nPerformance Summary:")
        logger.info(f"  - Annotation: {annotation_count / annotation_time:.1f} memories/second")
        logger.info(f"  - Clustering: {cluster_count / clustering_time:.1f} clusters/second")
        logger.info(f"  - Retrieval: {len(benchmark_memories) / retrieval_time:.0f} memories/second processed")
        logger.info(f"  - Optimization: {optimization_result.memories_processed / optimization_time:.0f} memories/second")
        
        return {
            'annotation_rate': annotation_count / annotation_time,
            'clustering_time': clustering_time,
            'retrieval_rate': len(benchmark_memories) / retrieval_time,
            'optimization_rate': optimization_result.memories_processed / optimization_time
        }
    
    async def cleanup(self):
        """Clean up all system components"""
        logger.info("\nCleaning up system components...")
        
        if self.clustering_engine:
            await self.clustering_engine.shutdown()
        if self.annotation_engine:
            await self.annotation_engine.shutdown()
        if self.retrieval_system:
            await self.retrieval_system.shutdown()
        if self.optimizer:
            await self.optimizer.shutdown()
        
        logger.info("Cleanup completed")
    
    async def run_complete_demo(self):
        """Run the complete advanced memory system demonstration"""
        try:
            logger.info("Starting Advanced Memory System Demonstration")
            logger.info("="*80)
            
            # Initialize systems
            await self.initialize_systems()
            
            # Generate demo data
            self.generate_demo_memories(100)
            
            # Run demonstrations
            await self.demonstrate_semantic_annotation()
            await self.demonstrate_temporal_clustering()
            await self.demonstrate_context_aware_retrieval()
            await self.demonstrate_memory_optimization()
            await self.demonstrate_integration()
            
            # Run performance benchmarks
            await self.run_performance_benchmark()
            
            logger.info("\n" + "="*80)
            logger.info("ADVANCED MEMORY SYSTEM DEMONSTRATION COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise
        finally:
            await self.cleanup()

async def main():
    """Main demo function"""
    demo = AdvancedMemorySystemDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    asyncio.run(main())