#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Memory Optimization
Advanced memory optimization and garbage collection mechanisms
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import uuid
from collections import defaultdict, Counter
import heapq

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Memory optimization strategies"""
    LRU_EVICTION = "lru_eviction"
    IMPORTANCE_BASED = "importance_based"
    CLUSTERING_BASED = "clustering_based"
    ADAPTIVE_RETENTION = "adaptive_retention"
    HIERARCHICAL_COMPRESSION = "hierarchical_compression"

class CompressionType(Enum):
    """Types of memory compression"""
    SEMANTIC_SUMMARIZATION = "semantic_summarization"
    TEMPORAL_AGGREGATION = "temporal_aggregation"
    PATTERN_EXTRACTION = "pattern_extraction"
    LOSSY_COMPRESSION = "lossy_compression"
    LOSSLESS_COMPRESSION = "lossless_compression"

class MemoryPriority(Enum):
    """Memory priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DISPOSABLE = "disposable"

@dataclass
class MemoryMetrics:
    """Metrics for memory optimization decisions"""
    memory_id: str
    access_frequency: int
    last_accessed: datetime
    creation_time: datetime
    importance_score: float
    compression_ratio: float
    storage_size: int
    retrieval_count: int
    success_rate: float
    cluster_membership: Optional[str]
    priority_level: MemoryPriority
    metadata: Dict[str, Any]

@dataclass
class OptimizationConfig:
    """Configuration for memory optimization"""
    strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE_RETENTION
    max_memory_size: int = 100000  # Maximum number of memories
    target_memory_size: int = 80000  # Target size after optimization
    compression_threshold: float = 0.7  # When to compress memories
    eviction_threshold: float = 0.9  # When to start eviction
    min_importance_threshold: float = 0.1  # Minimum importance to keep
    max_age_days: int = 90  # Maximum age before consideration for removal
    compression_enabled: bool = True
    clustering_enabled: bool = True
    adaptive_thresholds: bool = True
    garbage_collection_interval: timedelta = timedelta(hours=6)
    optimization_batch_size: int = 1000

@dataclass
class OptimizationResult:
    """Result of memory optimization operation"""
    operation_id: str
    strategy_used: OptimizationStrategy
    memories_processed: int
    memories_compressed: int
    memories_evicted: int
    memories_merged: int
    space_saved: int
    processing_time: float
    optimization_timestamp: datetime
    performance_impact: Dict[str, float]
    metadata: Dict[str, Any]

class MemoryOptimizer:
    """
    Advanced memory optimization and garbage collection system.
    
    Features:
    - Multiple optimization strategies
    - Intelligent memory compression
    - Adaptive retention policies
    - Cluster-aware optimization
    - Performance-aware garbage collection
    - Memory usage analytics
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.memory_metrics: Dict[str, MemoryMetrics] = {}
        self.optimization_history: List[OptimizationResult] = []
        self.compression_cache: Dict[str, Any] = {}
        
        # Adaptive parameters
        self.adaptive_thresholds: Dict[str, float] = {}
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Background tasks
        self.gc_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'total_optimizations': 0,
            'memories_compressed': 0,
            'memories_evicted': 0,
            'space_saved': 0,
            'gc_runs': 0,
            'performance_improvements': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.running = False
    
    async def initialize(self) -> None:
        """Initialize the memory optimizer"""
        try:
            self.logger.info("Initializing memory optimizer")
            
            # Initialize adaptive thresholds
            await self._initialize_adaptive_thresholds()
            
            # Start background tasks
            if self.config.garbage_collection_interval:
                self.gc_task = asyncio.create_task(self._garbage_collection_loop())
            
            self.initialized = True
            self.running = True
            self.logger.info("Memory optimizer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory optimizer: {e}")
            raise
    
    async def optimize_memory_system(self, 
                                   memories: List[Dict[str, Any]],
                                   force_optimization: bool = False) -> OptimizationResult:
        """
        Optimize memory system using configured strategy
        
        Args:
            memories: List of memories to optimize
            force_optimization: Force optimization regardless of thresholds
            
        Returns:
            OptimizationResult: Results of optimization operation
        """
        try:
            start_time = datetime.now()
            operation_id = str(uuid.uuid4())
            
            # Update memory metrics
            await self._update_memory_metrics(memories)
            
            # Check if optimization is needed
            if not force_optimization and not await self._should_optimize(memories):
                return OptimizationResult(
                    operation_id=operation_id,
                    strategy_used=self.config.strategy,
                    memories_processed=0,
                    memories_compressed=0,
                    memories_evicted=0,
                    memories_merged=0,
                    space_saved=0,
                    processing_time=0.0,
                    optimization_timestamp=start_time,
                    performance_impact={},
                    metadata={'reason': 'optimization_not_needed'}
                )
            
            # Apply optimization strategy
            if self.config.strategy == OptimizationStrategy.LRU_EVICTION:
                result = await self._lru_eviction_optimization(memories, operation_id, start_time)
            elif self.config.strategy == OptimizationStrategy.IMPORTANCE_BASED:
                result = await self._importance_based_optimization(memories, operation_id, start_time)
            elif self.config.strategy == OptimizationStrategy.CLUSTERING_BASED:
                result = await self._clustering_based_optimization(memories, operation_id, start_time)
            elif self.config.strategy == OptimizationStrategy.ADAPTIVE_RETENTION:
                result = await self._adaptive_retention_optimization(memories, operation_id, start_time)
            elif self.config.strategy == OptimizationStrategy.HIERARCHICAL_COMPRESSION:
                result = await self._hierarchical_compression_optimization(memories, operation_id, start_time)
            else:
                result = await self._adaptive_retention_optimization(memories, operation_id, start_time)
            
            # Store optimization result
            self.optimization_history.append(result)
            
            # Update statistics
            self.stats['total_optimizations'] += 1
            self.stats['memories_compressed'] += result.memories_compressed
            self.stats['memories_evicted'] += result.memories_evicted
            self.stats['space_saved'] += result.space_saved
            
            # Update adaptive parameters
            if self.config.adaptive_thresholds:
                await self._update_adaptive_thresholds(result)
            
            self.logger.info(f"Memory optimization completed: {result.memories_processed} processed, "
                           f"{result.memories_compressed} compressed, {result.memories_evicted} evicted")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize memory system: {e}")
            raise
    
    async def _update_memory_metrics(self, memories: List[Dict[str, Any]]) -> None:
        """Update metrics for all memories"""
        try:
            for memory in memories:
                memory_id = memory.get('memory_id', str(uuid.uuid4()))
                
                # Get or create metrics
                if memory_id not in self.memory_metrics:
                    self.memory_metrics[memory_id] = MemoryMetrics(
                        memory_id=memory_id,
                        access_frequency=0,
                        last_accessed=datetime.now(),
                        creation_time=memory.get('created_at', datetime.now()),
                        importance_score=0.5,
                        compression_ratio=1.0,
                        storage_size=len(str(memory)),
                        retrieval_count=0,
                        success_rate=0.5,
                        cluster_membership=None,
                        priority_level=MemoryPriority.MEDIUM,
                        metadata={}
                    )
                
                metrics = self.memory_metrics[memory_id]
                
                # Update metrics
                metrics.importance_score = await self._calculate_importance_score(memory)
                metrics.priority_level = await self._determine_priority_level(memory, metrics)
                metrics.storage_size = len(str(memory))
                
                # Update access patterns (simplified)
                if memory.get('recently_accessed', False):
                    metrics.access_frequency += 1
                    metrics.last_accessed = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Failed to update memory metrics: {e}")
    
    async def _should_optimize(self, memories: List[Dict[str, Any]]) -> bool:
        """Determine if optimization is needed"""
        try:
            current_size = len(memories)
            
            # Size-based optimization
            if current_size >= self.config.max_memory_size:
                return True
            
            # Threshold-based optimization
            utilization = current_size / self.config.max_memory_size
            if utilization >= self.config.eviction_threshold:
                return True
            
            # Performance-based optimization
            if await self._is_performance_degraded():
                return True
            
            # Time-based optimization
            last_optimization = self._get_last_optimization_time()
            if last_optimization and (datetime.now() - last_optimization).days >= 7:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to determine if optimization needed: {e}")
            return False
    
    async def _lru_eviction_optimization(self, memories: List[Dict[str, Any]], operation_id: str, start_time: datetime) -> OptimizationResult:
        """LRU-based memory eviction optimization"""
        try:
            memories_processed = len(memories)
            memories_evicted = 0
            memories_compressed = 0
            memories_merged = 0
            space_saved = 0
            
            if len(memories) <= self.config.target_memory_size:
                return self._create_optimization_result(
                    operation_id, start_time, memories_processed, memories_compressed,
                    memories_evicted, memories_merged, space_saved
                )
            
            # Sort by last access time (LRU first)
            sorted_memories = []
            for memory in memories:
                memory_id = memory.get('memory_id', str(uuid.uuid4()))
                metrics = self.memory_metrics.get(memory_id)
                last_accessed = metrics.last_accessed if metrics else datetime.now()
                sorted_memories.append((last_accessed, memory))
            
            sorted_memories.sort(key=lambda x: x[0])
            
            # Evict oldest memories
            memories_to_evict = len(memories) - self.config.target_memory_size
            
            for i in range(memories_to_evict):
                memory = sorted_memories[i][1]
                memory_id = memory.get('memory_id')
                
                # Check if memory is critical
                metrics = self.memory_metrics.get(memory_id)
                if metrics and metrics.priority_level == MemoryPriority.CRITICAL:
                    continue
                
                # Calculate space saved
                space_saved += len(str(memory))
                memories_evicted += 1
            
            return self._create_optimization_result(
                operation_id, start_time, memories_processed, memories_compressed,
                memories_evicted, memories_merged, space_saved
            )
            
        except Exception as e:
            self.logger.error(f"Failed LRU eviction optimization: {e}")
            raise
    
    async def _importance_based_optimization(self, memories: List[Dict[str, Any]], operation_id: str, start_time: datetime) -> OptimizationResult:
        """Importance-based memory optimization"""
        try:
            memories_processed = len(memories)
            memories_evicted = 0
            memories_compressed = 0
            memories_merged = 0
            space_saved = 0
            
            # Sort by importance score
            scored_memories = []
            for memory in memories:
                memory_id = memory.get('memory_id', str(uuid.uuid4()))
                metrics = self.memory_metrics.get(memory_id)
                importance = metrics.importance_score if metrics else 0.5
                scored_memories.append((importance, memory))
            
            scored_memories.sort(key=lambda x: x[0])
            
            # Process low-importance memories
            for importance, memory in scored_memories:
                if importance < self.config.min_importance_threshold:
                    # Evict very low importance memories
                    space_saved += len(str(memory))
                    memories_evicted += 1
                elif importance < self.config.compression_threshold and self.config.compression_enabled:
                    # Compress medium-low importance memories
                    compressed_size = await self._compress_memory(memory)
                    space_saved += len(str(memory)) - compressed_size
                    memories_compressed += 1
                
                # Stop if we've reached target size
                if (memories_processed - memories_evicted) <= self.config.target_memory_size:
                    break
            
            return self._create_optimization_result(
                operation_id, start_time, memories_processed, memories_compressed,
                memories_evicted, memories_merged, space_saved
            )
            
        except Exception as e:
            self.logger.error(f"Failed importance-based optimization: {e}")
            raise
    
    async def _clustering_based_optimization(self, memories: List[Dict[str, Any]], operation_id: str, start_time: datetime) -> OptimizationResult:
        """Clustering-based memory optimization"""
        try:
            memories_processed = len(memories)
            memories_evicted = 0
            memories_compressed = 0
            memories_merged = 0
            space_saved = 0
            
            # Group memories by clusters
            clusters = await self._group_memories_by_cluster(memories)
            
            for cluster_id, cluster_memories in clusters.items():
                if len(cluster_memories) > 1:
                    # Merge similar memories in cluster
                    merged_memory = await self._merge_cluster_memories(cluster_memories)
                    if merged_memory:
                        original_size = sum(len(str(mem)) for mem in cluster_memories)
                        merged_size = len(str(merged_memory))
                        space_saved += original_size - merged_size
                        memories_merged += len(cluster_memories) - 1
                
                # Compress cluster if it's large
                if len(cluster_memories) > 10:
                    for memory in cluster_memories[:5]:  # Compress oldest 5
                        compressed_size = await self._compress_memory(memory)
                        space_saved += len(str(memory)) - compressed_size
                        memories_compressed += 1
            
            return self._create_optimization_result(
                operation_id, start_time, memories_processed, memories_compressed,
                memories_evicted, memories_merged, space_saved
            )
            
        except Exception as e:
            self.logger.error(f"Failed clustering-based optimization: {e}")
            raise
    
    async def _adaptive_retention_optimization(self, memories: List[Dict[str, Any]], operation_id: str, start_time: datetime) -> OptimizationResult:
        """Adaptive retention-based optimization"""
        try:
            memories_processed = len(memories)
            memories_evicted = 0
            memories_compressed = 0
            memories_merged = 0
            space_saved = 0
            
            # Calculate adaptive thresholds
            adaptive_importance_threshold = self.adaptive_thresholds.get('importance', self.config.min_importance_threshold)
            adaptive_compression_threshold = self.adaptive_thresholds.get('compression', self.config.compression_threshold)
            
            # Score and categorize memories
            for memory in memories:
                memory_id = memory.get('memory_id', str(uuid.uuid4()))
                metrics = self.memory_metrics.get(memory_id)
                
                if not metrics:
                    continue
                
                # Calculate retention score
                retention_score = await self._calculate_retention_score(memory, metrics)
                
                if retention_score < adaptive_importance_threshold:
                    # Evict low-retention memories
                    space_saved += len(str(memory))
                    memories_evicted += 1
                elif retention_score < adaptive_compression_threshold and self.config.compression_enabled:
                    # Compress medium-retention memories
                    compressed_size = await self._compress_memory(memory)
                    space_saved += len(str(memory)) - compressed_size
                    memories_compressed += 1
                
                # Stop if target reached
                if (memories_processed - memories_evicted) <= self.config.target_memory_size:
                    break
            
            return self._create_optimization_result(
                operation_id, start_time, memories_processed, memories_compressed,
                memories_evicted, memories_merged, space_saved
            )
            
        except Exception as e:
            self.logger.error(f"Failed adaptive retention optimization: {e}")
            raise
    
    async def _hierarchical_compression_optimization(self, memories: List[Dict[str, Any]], operation_id: str, start_time: datetime) -> OptimizationResult:
        """Hierarchical compression-based optimization"""
        try:
            memories_processed = len(memories)
            memories_evicted = 0
            memories_compressed = 0
            memories_merged = 0
            space_saved = 0
            
            # Create hierarchy based on importance and age
            hierarchy_levels = await self._create_memory_hierarchy(memories)
            
            # Apply different compression strategies per level
            for level, level_memories in hierarchy_levels.items():
                if level == 'critical':
                    # Keep critical memories uncompressed
                    continue
                elif level == 'high':
                    # Light compression for high-importance memories
                    for memory in level_memories:
                        compressed_size = await self._compress_memory(memory, CompressionType.LOSSLESS_COMPRESSION)
                        space_saved += len(str(memory)) - compressed_size
                        memories_compressed += 1
                elif level == 'medium':
                    # Medium compression
                    for memory in level_memories:
                        compressed_size = await self._compress_memory(memory, CompressionType.SEMANTIC_SUMMARIZATION)
                        space_saved += len(str(memory)) - compressed_size
                        memories_compressed += 1
                elif level == 'low':
                    # Heavy compression or eviction
                    for memory in level_memories:
                        if await self._should_evict_memory(memory):
                            space_saved += len(str(memory))
                            memories_evicted += 1
                        else:
                            compressed_size = await self._compress_memory(memory, CompressionType.LOSSY_COMPRESSION)
                            space_saved += len(str(memory)) - compressed_size
                            memories_compressed += 1
            
            return self._create_optimization_result(
                operation_id, start_time, memories_processed, memories_compressed,
                memories_evicted, memories_merged, space_saved
            )
            
        except Exception as e:
            self.logger.error(f"Failed hierarchical compression optimization: {e}")
            raise
    
    async def _calculate_importance_score(self, memory: Dict[str, Any]) -> float:
        """Calculate importance score for a memory"""
        try:
            score = 0.0
            factors = 0
            
            # Success rate factor
            if memory.get('success') is not None:
                score += 1.0 if memory['success'] else 0.3
                factors += 1
            
            # Confidence factor
            confidence = memory.get('confidence_score', 0.5)
            score += confidence
            factors += 1
            
            # Recency factor
            timestamp = memory.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                days_old = (datetime.now() - timestamp).days
                recency_score = max(0.1, 1.0 - (days_old / 30))  # Decay over 30 days
                score += recency_score
                factors += 1
            
            # Learning value factor
            if memory.get('lessons_learned'):
                score += 0.8
                factors += 1
            
            # Uniqueness factor (simplified)
            if memory.get('unique_insights', False):
                score += 0.6
                factors += 1
            
            return score / factors if factors > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Failed to calculate importance score: {e}")
            return 0.5
    
    async def _determine_priority_level(self, memory: Dict[str, Any], metrics: MemoryMetrics) -> MemoryPriority:
        """Determine priority level for a memory"""
        try:
            importance = metrics.importance_score
            access_frequency = metrics.access_frequency
            
            # Critical memories
            if (importance > 0.9 or 
                access_frequency > 50 or 
                memory.get('critical', False)):
                return MemoryPriority.CRITICAL
            
            # High priority
            elif importance > 0.7 or access_frequency > 20:
                return MemoryPriority.HIGH
            
            # Medium priority
            elif importance > 0.4 or access_frequency > 5:
                return MemoryPriority.MEDIUM
            
            # Low priority
            elif importance > 0.2:
                return MemoryPriority.LOW
            
            # Disposable
            else:
                return MemoryPriority.DISPOSABLE
                
        except Exception as e:
            self.logger.error(f"Failed to determine priority level: {e}")
            return MemoryPriority.MEDIUM
    
    async def _calculate_retention_score(self, memory: Dict[str, Any], metrics: MemoryMetrics) -> float:
        """Calculate retention score for adaptive optimization"""
        try:
            # Base importance score
            retention_score = metrics.importance_score
            
            # Access pattern boost
            if metrics.access_frequency > 0:
                access_boost = min(0.3, metrics.access_frequency / 100)
                retention_score += access_boost
            
            # Recency boost
            days_since_access = (datetime.now() - metrics.last_accessed).days
            recency_boost = max(0.0, 0.2 - (days_since_access / 30))
            retention_score += recency_boost
            
            # Success rate boost
            if metrics.success_rate > 0.7:
                retention_score += 0.1
            
            # Cluster membership boost
            if metrics.cluster_membership:
                retention_score += 0.05
            
            return min(1.0, retention_score)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate retention score: {e}")
            return 0.5
    
    async def _compress_memory(self, memory: Dict[str, Any], compression_type: CompressionType = CompressionType.SEMANTIC_SUMMARIZATION) -> int:
        """Compress a memory and return compressed size"""
        try:
            original_size = len(str(memory))
            
            if compression_type == CompressionType.SEMANTIC_SUMMARIZATION:
                # Summarize content while preserving key information
                compressed_memory = await self._semantic_summarization(memory)
                compressed_size = len(str(compressed_memory))
            elif compression_type == CompressionType.TEMPORAL_AGGREGATION:
                # Aggregate temporal information
                compressed_memory = await self._temporal_aggregation(memory)
                compressed_size = len(str(compressed_memory))
            elif compression_type == CompressionType.PATTERN_EXTRACTION:
                # Extract and store only patterns
                compressed_memory = await self._pattern_extraction(memory)
                compressed_size = len(str(compressed_memory))
            elif compression_type == CompressionType.LOSSLESS_COMPRESSION:
                # Simple lossless compression (placeholder)
                compressed_size = int(original_size * 0.7)  # Assume 30% compression
            else:
                # Lossy compression
                compressed_size = int(original_size * 0.5)  # Assume 50% compression
            
            # Cache compressed version
            memory_id = memory.get('memory_id', str(uuid.uuid4()))
            self.compression_cache[memory_id] = {
                'compressed_memory': compressed_memory if 'compressed_memory' in locals() else memory,
                'compression_type': compression_type,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compressed_size / original_size
            }
            
            return compressed_size
            
        except Exception as e:
            self.logger.error(f"Failed to compress memory: {e}")
            return len(str(memory))
    
    async def _semantic_summarization(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic summarization of memory"""
        try:
            # Simplified semantic summarization
            summarized = {
                'memory_id': memory.get('memory_id'),
                'summary': memory.get('content', '')[:200] + '...',  # First 200 chars
                'key_outcomes': memory.get('lessons_learned', []),
                'success': memory.get('success'),
                'confidence_score': memory.get('confidence_score'),
                'timestamp': memory.get('timestamp'),
                'compressed': True,
                'compression_type': 'semantic_summarization'
            }
            
            return summarized
            
        except Exception as e:
            self.logger.error(f"Failed semantic summarization: {e}")
            return memory
    
    async def _temporal_aggregation(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Perform temporal aggregation of memory"""
        try:
            # Simplified temporal aggregation
            aggregated = {
                'memory_id': memory.get('memory_id'),
                'time_period': memory.get('timestamp'),
                'activity_summary': f"Activity during {memory.get('timestamp')}",
                'outcome_summary': memory.get('success', False),
                'key_metrics': {
                    'confidence': memory.get('confidence_score', 0.5),
                    'success': memory.get('success', False)
                },
                'compressed': True,
                'compression_type': 'temporal_aggregation'
            }
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Failed temporal aggregation: {e}")
            return memory
    
    async def _pattern_extraction(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patterns from memory"""
        try:
            # Simplified pattern extraction
            patterns = {
                'memory_id': memory.get('memory_id'),
                'success_pattern': memory.get('success', False),
                'action_pattern': memory.get('action_type', 'unknown'),
                'outcome_pattern': 'success' if memory.get('success') else 'failure',
                'confidence_pattern': 'high' if memory.get('confidence_score', 0) > 0.7 else 'low',
                'compressed': True,
                'compression_type': 'pattern_extraction'
            }
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed pattern extraction: {e}")
            return memory
    
    async def _group_memories_by_cluster(self, memories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group memories by cluster membership"""
        try:
            clusters = defaultdict(list)
            
            for memory in memories:
                memory_id = memory.get('memory_id', str(uuid.uuid4()))
                metrics = self.memory_metrics.get(memory_id)
                
                if metrics and metrics.cluster_membership:
                    clusters[metrics.cluster_membership].append(memory)
                else:
                    clusters['unclustered'].append(memory)
            
            return dict(clusters)
            
        except Exception as e:
            self.logger.error(f"Failed to group memories by cluster: {e}")
            return {'unclustered': memories}
    
    async def _merge_cluster_memories(self, cluster_memories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Merge similar memories in a cluster"""
        try:
            if len(cluster_memories) < 2:
                return None
            
            # Simple merging strategy - combine similar memories
            merged = {
                'memory_id': str(uuid.uuid4()),
                'merged_from': [mem.get('memory_id') for mem in cluster_memories],
                'content': f"Merged from {len(cluster_memories)} similar memories",
                'success_rate': sum(mem.get('success', 0) for mem in cluster_memories) / len(cluster_memories),
                'confidence_score': sum(mem.get('confidence_score', 0.5) for mem in cluster_memories) / len(cluster_memories),
                'timestamp': max(mem.get('timestamp', datetime.now()) for mem in cluster_memories),
                'merged': True,
                'merge_count': len(cluster_memories)
            }
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Failed to merge cluster memories: {e}")
            return None
    
    async def _create_memory_hierarchy(self, memories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Create memory hierarchy for hierarchical compression"""
        try:
            hierarchy = {
                'critical': [],
                'high': [],
                'medium': [],
                'low': [],
                'disposable': []
            }
            
            for memory in memories:
                memory_id = memory.get('memory_id', str(uuid.uuid4()))
                metrics = self.memory_metrics.get(memory_id)
                
                if metrics:
                    priority = metrics.priority_level
                    hierarchy[priority.value].append(memory)
                else:
                    hierarchy['medium'].append(memory)
            
            return hierarchy
            
        except Exception as e:
            self.logger.error(f"Failed to create memory hierarchy: {e}")
            return {'medium': memories}
    
    async def _should_evict_memory(self, memory: Dict[str, Any]) -> bool:
        """Determine if a memory should be evicted"""
        try:
            memory_id = memory.get('memory_id', str(uuid.uuid4()))
            metrics = self.memory_metrics.get(memory_id)
            
            if not metrics:
                return True  # Evict if no metrics available
            
            # Never evict critical memories
            if metrics.priority_level == MemoryPriority.CRITICAL:
                return False
            
            # Evict disposable memories
            if metrics.priority_level == MemoryPriority.DISPOSABLE:
                return True
            
            # Evict very old, unused memories
            days_since_access = (datetime.now() - metrics.last_accessed).days
            if days_since_access > self.config.max_age_days and metrics.access_frequency == 0:
                return True
            
            # Evict low-importance, low-access memories
            if metrics.importance_score < 0.2 and metrics.access_frequency < 2:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to determine if memory should be evicted: {e}")
            return False
    
    async def _is_performance_degraded(self) -> bool:
        """Check if system performance is degraded"""
        try:
            # Check recent performance metrics
            recent_metrics = self.performance_metrics.get('retrieval_time', [])
            if len(recent_metrics) < 10:
                return False
            
            # Compare recent performance to baseline
            recent_avg = np.mean(recent_metrics[-10:])
            baseline_avg = np.mean(recent_metrics[:-10]) if len(recent_metrics) > 20 else recent_avg
            
            # Performance degraded if recent average is 50% worse
            return recent_avg > baseline_avg * 1.5
            
        except Exception as e:
            self.logger.error(f"Failed to check performance degradation: {e}")
            return False
    
    def _get_last_optimization_time(self) -> Optional[datetime]:
        """Get timestamp of last optimization"""
        try:
            if self.optimization_history:
                return self.optimization_history[-1].optimization_timestamp
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get last optimization time: {e}")
            return None
    
    def _create_optimization_result(self, operation_id: str, start_time: datetime, 
                                  memories_processed: int, memories_compressed: int,
                                  memories_evicted: int, memories_merged: int, 
                                  space_saved: int) -> OptimizationResult:
        """Create optimization result object"""
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                operation_id=operation_id,
                strategy_used=self.config.strategy,
                memories_processed=memories_processed,
                memories_compressed=memories_compressed,
                memories_evicted=memories_evicted,
                memories_merged=memories_merged,
                space_saved=space_saved,
                processing_time=processing_time,
                optimization_timestamp=datetime.now(),
                performance_impact={
                    'processing_time': processing_time,
                    'space_efficiency': space_saved / max(memories_processed, 1),
                    'compression_ratio': memories_compressed / max(memories_processed, 1)
                },
                metadata={
                    'config_used': asdict(self.config),
                    'adaptive_thresholds': self.adaptive_thresholds.copy()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create optimization result: {e}")
            raise
    
    async def _initialize_adaptive_thresholds(self) -> None:
        """Initialize adaptive thresholds"""
        try:
            self.adaptive_thresholds = {
                'importance': self.config.min_importance_threshold,
                'compression': self.config.compression_threshold,
                'eviction': self.config.eviction_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize adaptive thresholds: {e}")
    
    async def _update_adaptive_thresholds(self, result: OptimizationResult) -> None:
        """Update adaptive thresholds based on optimization results"""
        try:
            # Adjust thresholds based on optimization effectiveness
            effectiveness = result.space_saved / max(result.memories_processed, 1)
            
            if effectiveness > 0.3:  # Good optimization
                # Slightly relax thresholds
                self.adaptive_thresholds['importance'] *= 0.95
                self.adaptive_thresholds['compression'] *= 0.98
            elif effectiveness < 0.1:  # Poor optimization
                # Tighten thresholds
                self.adaptive_thresholds['importance'] *= 1.05
                self.adaptive_thresholds['compression'] *= 1.02
            
            # Keep thresholds within reasonable bounds
            self.adaptive_thresholds['importance'] = max(0.05, min(0.5, self.adaptive_thresholds['importance']))
            self.adaptive_thresholds['compression'] = max(0.3, min(0.9, self.adaptive_thresholds['compression']))
            
        except Exception as e:
            self.logger.error(f"Failed to update adaptive thresholds: {e}")
    
    async def _garbage_collection_loop(self) -> None:
        """Background garbage collection loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config.garbage_collection_interval.total_seconds())
                
                if not self.running:
                    break
                
                # Perform lightweight garbage collection
                await self._perform_garbage_collection()
                
                self.stats['gc_runs'] += 1
                
            except Exception as e:
                self.logger.error(f"Error in garbage collection loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_garbage_collection(self) -> None:
        """Perform garbage collection operations"""
        try:
            # Clean up old optimization history
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-50:]
            
            # Clean up old compression cache entries
            if len(self.compression_cache) > 1000:
                # Remove oldest entries
                cache_items = list(self.compression_cache.items())
                self.compression_cache = dict(cache_items[-500:])
            
            # Clean up old performance metrics
            for metric_name, values in self.performance_metrics.items():
                if len(values) > 1000:
                    self.performance_metrics[metric_name] = values[-500:]
            
            self.logger.debug("Garbage collection completed")
            
        except Exception as e:
            self.logger.error(f"Failed to perform garbage collection: {e}")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get memory optimization statistics"""
        try:
            # Calculate recent optimization performance
            recent_optimizations = self.optimization_history[-10:] if self.optimization_history else []
            
            avg_processing_time = 0.0
            avg_space_saved = 0.0
            avg_compression_ratio = 0.0
            
            if recent_optimizations:
                avg_processing_time = np.mean([opt.processing_time for opt in recent_optimizations])
                avg_space_saved = np.mean([opt.space_saved for opt in recent_optimizations])
                avg_compression_ratio = np.mean([
                    opt.memories_compressed / max(opt.memories_processed, 1) 
                    for opt in recent_optimizations
                ])
            
            return {
                'total_optimizations': len(self.optimization_history),
                'recent_performance': {
                    'avg_processing_time': avg_processing_time,
                    'avg_space_saved': avg_space_saved,
                    'avg_compression_ratio': avg_compression_ratio
                },
                'current_config': {
                    'strategy': self.config.strategy.value,
                    'max_memory_size': self.config.max_memory_size,
                    'target_memory_size': self.config.target_memory_size,
                    'compression_enabled': self.config.compression_enabled
                },
                'adaptive_thresholds': self.adaptive_thresholds.copy(),
                'cache_statistics': {
                    'compression_cache_size': len(self.compression_cache),
                    'memory_metrics_tracked': len(self.memory_metrics)
                },
                'processing_stats': self.stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization statistics: {e}")
            return {'error': str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown the memory optimizer"""
        try:
            self.logger.info("Shutting down memory optimizer")
            self.running = False
            
            # Cancel background tasks
            if self.gc_task:
                self.gc_task.cancel()
                try:
                    await self.gc_task
                except asyncio.CancelledError:
                    pass
            
            if self.optimization_task:
                self.optimization_task.cancel()
                try:
                    await self.optimization_task
                except asyncio.CancelledError:
                    pass
            
            # Clear data structures
            self.memory_metrics.clear()
            self.compression_cache.clear()
            self.adaptive_thresholds.clear()
            self.performance_metrics.clear()
            
            self.initialized = False
            self.logger.info("Memory optimizer shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during memory optimizer shutdown: {e}")

# Factory function
def create_memory_optimizer(config: OptimizationConfig = None) -> MemoryOptimizer:
    """Create a memory optimizer instance"""
    return MemoryOptimizer(config or OptimizationConfig())