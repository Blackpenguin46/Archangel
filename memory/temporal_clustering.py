#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Temporal Clustering
Advanced temporal clustering algorithms for experience segmentation
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)

class ClusteringAlgorithm(Enum):
    """Supported clustering algorithms"""
    TEMPORAL_DBSCAN = "temporal_dbscan"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    HIERARCHICAL_TEMPORAL = "hierarchical_temporal"

@dataclass
class TemporalCluster:
    """Temporal cluster with time-based segmentation"""
    cluster_id: str
    algorithm: ClusteringAlgorithm
    time_window: timedelta
    start_time: datetime
    end_time: datetime
    experience_ids: List[str]
    center_embedding: List[float]
    temporal_pattern: Dict[str, Any]
    cohesion_score: float
    activity_density: float
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any]

@dataclass
class TemporalSegment:
    """Time-based segment of experiences"""
    segment_id: str
    start_time: datetime
    end_time: datetime
    experience_count: int
    activity_level: str  # low, medium, high
    dominant_actions: List[str]
    success_rate: float
    confidence_trend: str  # increasing, decreasing, stable
    metadata: Dict[str, Any]

@dataclass
class ClusteringConfig:
    """Configuration for temporal clustering"""
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.TEMPORAL_DBSCAN
    min_cluster_size: int = 3
    max_cluster_size: int = 50
    time_epsilon: timedelta = timedelta(hours=2)
    similarity_threshold: float = 0.7
    sliding_window_size: timedelta = timedelta(hours=6)
    adaptive_threshold_factor: float = 0.1
    max_clusters_per_agent: int = 20
    cluster_merge_threshold: float = 0.8

class TemporalClusteringEngine:
    """
    Advanced temporal clustering engine for experience segmentation.
    
    Features:
    - Multiple clustering algorithms (DBSCAN, sliding window, adaptive)
    - Time-aware similarity calculation
    - Activity pattern recognition
    - Cluster quality metrics
    - Automatic parameter tuning
    """
    
    def __init__(self, config: ClusteringConfig = None):
        self.config = config or ClusteringConfig()
        self.clusters: Dict[str, TemporalCluster] = {}
        self.segments: Dict[str, TemporalSegment] = {}
        self.agent_clusters: Dict[str, List[str]] = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the temporal clustering engine"""
        try:
            self.logger.info("Initializing temporal clustering engine")
            self.initialized = True
            self.logger.info("Temporal clustering engine initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize temporal clustering engine: {e}")
            raise
    
    async def cluster_experiences_temporal(self, 
                                        agent_id: str,
                                        experiences: List[Dict[str, Any]],
                                        time_window: timedelta = None) -> List[TemporalCluster]:
        """
        Cluster experiences using temporal algorithms
        
        Args:
            agent_id: Agent ID for clustering
            experiences: List of experience data with embeddings
            time_window: Time window for clustering
            
        Returns:
            List[TemporalCluster]: Generated temporal clusters
        """
        try:
            if not experiences:
                return []
            
            time_window = time_window or timedelta(days=7)
            
            # Filter experiences by time window
            end_time = datetime.now()
            start_time = end_time - time_window
            
            filtered_experiences = [
                exp for exp in experiences
                if start_time <= exp.get('timestamp', datetime.now()) <= end_time
            ]
            
            if len(filtered_experiences) < self.config.min_cluster_size:
                return []
            
            # Apply selected clustering algorithm
            if self.config.algorithm == ClusteringAlgorithm.TEMPORAL_DBSCAN:
                clusters = await self._temporal_dbscan_clustering(agent_id, filtered_experiences)
            elif self.config.algorithm == ClusteringAlgorithm.SLIDING_WINDOW:
                clusters = await self._sliding_window_clustering(agent_id, filtered_experiences)
            elif self.config.algorithm == ClusteringAlgorithm.ADAPTIVE_THRESHOLD:
                clusters = await self._adaptive_threshold_clustering(agent_id, filtered_experiences)
            elif self.config.algorithm == ClusteringAlgorithm.HIERARCHICAL_TEMPORAL:
                clusters = await self._hierarchical_temporal_clustering(agent_id, filtered_experiences)
            else:
                clusters = await self._temporal_dbscan_clustering(agent_id, filtered_experiences)
            
            # Store clusters
            for cluster in clusters:
                self.clusters[cluster.cluster_id] = cluster
                self.agent_clusters[agent_id].append(cluster.cluster_id)
            
            # Limit clusters per agent
            await self._limit_agent_clusters(agent_id)
            
            self.logger.info(f"Created {len(clusters)} temporal clusters for agent {agent_id}")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Failed to cluster experiences temporally: {e}")
            return []
    
    async def _temporal_dbscan_clustering(self, agent_id: str, experiences: List[Dict[str, Any]]) -> List[TemporalCluster]:
        """DBSCAN clustering with temporal awareness"""
        try:
            clusters = []
            visited = set()
            noise = set()
            
            for i, exp in enumerate(experiences):
                if i in visited:
                    continue
                
                # Find temporal neighbors
                neighbors = await self._find_temporal_neighbors(i, experiences)
                
                if len(neighbors) < self.config.min_cluster_size:
                    noise.add(i)
                    continue
                
                # Create new cluster
                cluster_experiences = []
                cluster_embeddings = []
                cluster_queue = list(neighbors)
                visited.update(neighbors)
                
                while cluster_queue:
                    current_idx = cluster_queue.pop(0)
                    current_exp = experiences[current_idx]
                    
                    cluster_experiences.append(current_exp)
                    if 'embedding' in current_exp:
                        cluster_embeddings.append(current_exp['embedding'])
                    
                    # Find neighbors of current point
                    current_neighbors = await self._find_temporal_neighbors(current_idx, experiences)
                    
                    if len(current_neighbors) >= self.config.min_cluster_size:
                        for neighbor_idx in current_neighbors:
                            if neighbor_idx not in visited:
                                cluster_queue.append(neighbor_idx)
                                visited.add(neighbor_idx)
                
                # Create cluster if valid size
                if len(cluster_experiences) >= self.config.min_cluster_size:
                    cluster = await self._create_temporal_cluster(
                        agent_id, cluster_experiences, cluster_embeddings, 
                        ClusteringAlgorithm.TEMPORAL_DBSCAN
                    )
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Failed in temporal DBSCAN clustering: {e}")
            return []
    
    async def _sliding_window_clustering(self, agent_id: str, experiences: List[Dict[str, Any]]) -> List[TemporalCluster]:
        """Sliding window clustering algorithm"""
        try:
            clusters = []
            
            # Sort experiences by timestamp
            sorted_experiences = sorted(experiences, key=lambda x: x.get('timestamp', datetime.now()))
            
            window_size = self.config.sliding_window_size
            current_time = sorted_experiences[0].get('timestamp', datetime.now())
            end_time = sorted_experiences[-1].get('timestamp', datetime.now())
            
            while current_time <= end_time:
                window_end = current_time + window_size
                
                # Get experiences in current window
                window_experiences = [
                    exp for exp in sorted_experiences
                    if current_time <= exp.get('timestamp', datetime.now()) <= window_end
                ]
                
                if len(window_experiences) >= self.config.min_cluster_size:
                    # Check semantic similarity within window
                    similar_groups = await self._group_by_similarity(window_experiences)
                    
                    for group in similar_groups:
                        if len(group) >= self.config.min_cluster_size:
                            embeddings = [exp.get('embedding', []) for exp in group if exp.get('embedding')]
                            cluster = await self._create_temporal_cluster(
                                agent_id, group, embeddings, 
                                ClusteringAlgorithm.SLIDING_WINDOW
                            )
                            clusters.append(cluster)
                
                # Move window forward
                current_time += window_size / 2  # 50% overlap
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Failed in sliding window clustering: {e}")
            return []
    
    async def _adaptive_threshold_clustering(self, agent_id: str, experiences: List[Dict[str, Any]]) -> List[TemporalCluster]:
        """Adaptive threshold clustering with dynamic similarity adjustment"""
        try:
            clusters = []
            
            # Calculate adaptive threshold based on data distribution
            similarities = []
            for i in range(len(experiences)):
                for j in range(i + 1, len(experiences)):
                    sim = self._calculate_temporal_similarity(experiences[i], experiences[j])
                    similarities.append(sim)
            
            if not similarities:
                return []
            
            # Adaptive threshold based on mean and std
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            adaptive_threshold = mean_sim + (self.config.adaptive_threshold_factor * std_sim)
            
            # Group experiences by adaptive similarity
            used_indices = set()
            
            for i, exp in enumerate(experiences):
                if i in used_indices:
                    continue
                
                cluster_experiences = [exp]
                cluster_indices = {i}
                
                for j, other_exp in enumerate(experiences):
                    if j in used_indices or j == i:
                        continue
                    
                    similarity = self._calculate_temporal_similarity(exp, other_exp)
                    if similarity >= adaptive_threshold:
                        cluster_experiences.append(other_exp)
                        cluster_indices.add(j)
                
                if len(cluster_experiences) >= self.config.min_cluster_size:
                    used_indices.update(cluster_indices)
                    embeddings = [exp.get('embedding', []) for exp in cluster_experiences if exp.get('embedding')]
                    cluster = await self._create_temporal_cluster(
                        agent_id, cluster_experiences, embeddings,
                        ClusteringAlgorithm.ADAPTIVE_THRESHOLD
                    )
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Failed in adaptive threshold clustering: {e}")
            return []
    
    async def _hierarchical_temporal_clustering(self, agent_id: str, experiences: List[Dict[str, Any]]) -> List[TemporalCluster]:
        """Hierarchical clustering with temporal constraints"""
        try:
            clusters = []
            
            # Create initial clusters (single experiences)
            current_clusters = [[exp] for exp in experiences]
            
            # Merge clusters hierarchically
            while len(current_clusters) > 1:
                best_merge = None
                best_similarity = 0.0
                
                # Find best cluster pair to merge
                for i in range(len(current_clusters)):
                    for j in range(i + 1, len(current_clusters)):
                        similarity = await self._calculate_cluster_similarity(
                            current_clusters[i], current_clusters[j]
                        )
                        
                        if similarity > best_similarity and similarity >= self.config.similarity_threshold:
                            best_similarity = similarity
                            best_merge = (i, j)
                
                if best_merge is None:
                    break
                
                # Merge best clusters
                i, j = best_merge
                merged_cluster = current_clusters[i] + current_clusters[j]
                
                # Check temporal constraints
                if await self._validate_temporal_constraints(merged_cluster):
                    # Remove original clusters and add merged
                    new_clusters = []
                    for k, cluster in enumerate(current_clusters):
                        if k != i and k != j:
                            new_clusters.append(cluster)
                    new_clusters.append(merged_cluster)
                    current_clusters = new_clusters
                else:
                    break
            
            # Convert to TemporalCluster objects
            for cluster_experiences in current_clusters:
                if len(cluster_experiences) >= self.config.min_cluster_size:
                    embeddings = [exp.get('embedding', []) for exp in cluster_experiences if exp.get('embedding')]
                    cluster = await self._create_temporal_cluster(
                        agent_id, cluster_experiences, embeddings,
                        ClusteringAlgorithm.HIERARCHICAL_TEMPORAL
                    )
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Failed in hierarchical temporal clustering: {e}")
            return []
    
    async def _find_temporal_neighbors(self, exp_idx: int, experiences: List[Dict[str, Any]]) -> List[int]:
        """Find temporal neighbors for DBSCAN"""
        try:
            neighbors = []
            base_exp = experiences[exp_idx]
            base_time = base_exp.get('timestamp', datetime.now())
            
            for i, exp in enumerate(experiences):
                if i == exp_idx:
                    continue
                
                exp_time = exp.get('timestamp', datetime.now())
                time_diff = abs((exp_time - base_time).total_seconds())
                
                # Check temporal proximity
                if time_diff <= self.config.time_epsilon.total_seconds():
                    # Check semantic similarity
                    similarity = self._calculate_temporal_similarity(base_exp, exp)
                    if similarity >= self.config.similarity_threshold:
                        neighbors.append(i)
            
            return neighbors
            
        except Exception as e:
            self.logger.error(f"Failed to find temporal neighbors: {e}")
            return []
    
    async def _group_by_similarity(self, experiences: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group experiences by semantic similarity"""
        try:
            groups = []
            used_indices = set()
            
            for i, exp in enumerate(experiences):
                if i in used_indices:
                    continue
                
                group = [exp]
                group_indices = {i}
                
                for j, other_exp in enumerate(experiences):
                    if j in used_indices or j == i:
                        continue
                    
                    similarity = self._calculate_temporal_similarity(exp, other_exp)
                    if similarity >= self.config.similarity_threshold:
                        group.append(other_exp)
                        group_indices.add(j)
                
                used_indices.update(group_indices)
                groups.append(group)
            
            return groups
            
        except Exception as e:
            self.logger.error(f"Failed to group by similarity: {e}")
            return []
    
    async def _calculate_cluster_similarity(self, cluster1: List[Dict[str, Any]], cluster2: List[Dict[str, Any]]) -> float:
        """Calculate similarity between two clusters"""
        try:
            similarities = []
            
            for exp1 in cluster1:
                for exp2 in cluster2:
                    sim = self._calculate_temporal_similarity(exp1, exp2)
                    similarities.append(sim)
            
            if not similarities:
                return 0.0
            
            # Use average linkage
            return np.mean(similarities)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cluster similarity: {e}")
            return 0.0
    
    def _calculate_temporal_similarity(self, exp1: Dict[str, Any], exp2: Dict[str, Any]) -> float:
        """Calculate temporal-aware similarity between experiences"""
        try:
            # Semantic similarity from embeddings
            semantic_sim = 0.0
            if exp1.get('embedding') and exp2.get('embedding'):
                semantic_sim = self._cosine_similarity(exp1['embedding'], exp2['embedding'])
            
            # Temporal proximity factor
            time1 = exp1.get('timestamp', datetime.now())
            time2 = exp2.get('timestamp', datetime.now())
            time_diff = abs((time1 - time2).total_seconds())
            max_time_diff = self.config.time_epsilon.total_seconds()
            
            temporal_factor = max(0.0, 1.0 - (time_diff / max_time_diff)) if max_time_diff > 0 else 1.0
            
            # Context similarity (action types, success, etc.)
            context_sim = 0.0
            context_factors = 0
            
            if exp1.get('success') is not None and exp2.get('success') is not None:
                context_sim += 1.0 if exp1['success'] == exp2['success'] else 0.0
                context_factors += 1
            
            if exp1.get('action_type') and exp2.get('action_type'):
                context_sim += 1.0 if exp1['action_type'] == exp2['action_type'] else 0.0
                context_factors += 1
            
            if context_factors > 0:
                context_sim /= context_factors
            
            # Weighted combination
            total_similarity = (0.6 * semantic_sim + 0.3 * temporal_factor + 0.1 * context_sim)
            
            return total_similarity
            
        except Exception as e:
            self.logger.error(f"Failed to calculate temporal similarity: {e}")
            return 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        try:
            if not vec1 or not vec2:
                return 0.0
            
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    async def _validate_temporal_constraints(self, cluster_experiences: List[Dict[str, Any]]) -> bool:
        """Validate temporal constraints for cluster"""
        try:
            if len(cluster_experiences) < 2:
                return True
            
            # Check cluster size limits
            if len(cluster_experiences) > self.config.max_cluster_size:
                return False
            
            # Check temporal span
            timestamps = [exp.get('timestamp', datetime.now()) for exp in cluster_experiences]
            min_time = min(timestamps)
            max_time = max(timestamps)
            time_span = max_time - min_time
            
            # Cluster should not span too long
            max_span = self.config.time_epsilon * 10  # Allow 10x epsilon for cluster span
            if time_span > max_span:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate temporal constraints: {e}")
            return False
    
    async def _create_temporal_cluster(self, 
                                     agent_id: str,
                                     experiences: List[Dict[str, Any]],
                                     embeddings: List[List[float]],
                                     algorithm: ClusteringAlgorithm) -> TemporalCluster:
        """Create a temporal cluster from experiences"""
        try:
            # Calculate cluster properties
            timestamps = [exp.get('timestamp', datetime.now()) for exp in experiences]
            start_time = min(timestamps)
            end_time = max(timestamps)
            time_window = end_time - start_time
            
            # Calculate center embedding
            center_embedding = []
            if embeddings:
                center_embedding = np.mean(embeddings, axis=0).tolist()
            
            # Calculate temporal pattern
            temporal_pattern = await self._analyze_temporal_pattern(experiences)
            
            # Calculate cohesion score
            cohesion_score = await self._calculate_cohesion_score(experiences, embeddings)
            
            # Calculate activity density
            activity_density = len(experiences) / max(time_window.total_seconds() / 3600, 1)  # experiences per hour
            
            cluster = TemporalCluster(
                cluster_id=str(uuid.uuid4()),
                algorithm=algorithm,
                time_window=time_window,
                start_time=start_time,
                end_time=end_time,
                experience_ids=[exp.get('experience_id', str(uuid.uuid4())) for exp in experiences],
                center_embedding=center_embedding,
                temporal_pattern=temporal_pattern,
                cohesion_score=cohesion_score,
                activity_density=activity_density,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                metadata={
                    'agent_id': agent_id,
                    'experience_count': len(experiences),
                    'algorithm': algorithm.value
                }
            )
            
            return cluster
            
        except Exception as e:
            self.logger.error(f"Failed to create temporal cluster: {e}")
            raise
    
    async def _analyze_temporal_pattern(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in experiences"""
        try:
            pattern = {
                'activity_distribution': {},
                'success_trend': [],
                'action_frequency': {},
                'peak_hours': []
            }
            
            # Analyze activity distribution by hour
            hour_counts = defaultdict(int)
            success_by_hour = defaultdict(list)
            action_counts = defaultdict(int)
            
            for exp in experiences:
                timestamp = exp.get('timestamp', datetime.now())
                hour = timestamp.hour
                hour_counts[hour] += 1
                
                success = exp.get('success', False)
                success_by_hour[hour].append(success)
                
                action_type = exp.get('action_type', 'unknown')
                action_counts[action_type] += 1
            
            # Convert to pattern data
            pattern['activity_distribution'] = dict(hour_counts)
            pattern['action_frequency'] = dict(action_counts)
            
            # Find peak hours
            if hour_counts:
                max_activity = max(hour_counts.values())
                pattern['peak_hours'] = [hour for hour, count in hour_counts.items() if count >= max_activity * 0.8]
            
            # Calculate success trend
            for hour in sorted(success_by_hour.keys()):
                successes = success_by_hour[hour]
                success_rate = sum(successes) / len(successes) if successes else 0.0
                pattern['success_trend'].append({'hour': hour, 'success_rate': success_rate})
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Failed to analyze temporal pattern: {e}")
            return {}
    
    async def _calculate_cohesion_score(self, experiences: List[Dict[str, Any]], embeddings: List[List[float]]) -> float:
        """Calculate cluster cohesion score"""
        try:
            if len(embeddings) < 2:
                return 1.0
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = self._cosine_similarity(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            # Return average similarity as cohesion score
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cohesion score: {e}")
            return 0.0
    
    async def _limit_agent_clusters(self, agent_id: str) -> None:
        """Limit number of clusters per agent"""
        try:
            agent_cluster_ids = self.agent_clusters[agent_id]
            
            if len(agent_cluster_ids) <= self.config.max_clusters_per_agent:
                return
            
            # Sort clusters by quality (cohesion score and recency)
            cluster_scores = []
            for cluster_id in agent_cluster_ids:
                cluster = self.clusters.get(cluster_id)
                if cluster:
                    # Score based on cohesion and recency
                    recency_score = 1.0 / max((datetime.now() - cluster.last_updated).days + 1, 1)
                    total_score = cluster.cohesion_score * 0.7 + recency_score * 0.3
                    cluster_scores.append((cluster_id, total_score))
            
            # Keep top clusters
            cluster_scores.sort(key=lambda x: x[1], reverse=True)
            clusters_to_keep = cluster_scores[:self.config.max_clusters_per_agent]
            clusters_to_remove = cluster_scores[self.config.max_clusters_per_agent:]
            
            # Remove low-quality clusters
            for cluster_id, _ in clusters_to_remove:
                if cluster_id in self.clusters:
                    del self.clusters[cluster_id]
                if cluster_id in agent_cluster_ids:
                    agent_cluster_ids.remove(cluster_id)
            
            self.logger.debug(f"Limited agent {agent_id} to {len(clusters_to_keep)} clusters")
            
        except Exception as e:
            self.logger.error(f"Failed to limit agent clusters: {e}")
    
    async def get_agent_temporal_clusters(self, agent_id: str) -> List[TemporalCluster]:
        """Get all temporal clusters for an agent"""
        try:
            cluster_ids = self.agent_clusters.get(agent_id, [])
            clusters = [self.clusters[cid] for cid in cluster_ids if cid in self.clusters]
            
            # Sort by recency
            clusters.sort(key=lambda x: x.last_updated, reverse=True)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Failed to get agent temporal clusters: {e}")
            return []
    
    async def merge_similar_clusters(self, agent_id: str) -> int:
        """Merge similar clusters for an agent"""
        try:
            clusters = await self.get_agent_temporal_clusters(agent_id)
            merged_count = 0
            
            i = 0
            while i < len(clusters):
                j = i + 1
                while j < len(clusters):
                    cluster1 = clusters[i]
                    cluster2 = clusters[j]
                    
                    # Calculate cluster similarity
                    similarity = self._calculate_cluster_center_similarity(cluster1, cluster2)
                    
                    if similarity >= self.config.cluster_merge_threshold:
                        # Merge clusters
                        merged_cluster = await self._merge_clusters(cluster1, cluster2)
                        
                        # Update storage
                        self.clusters[merged_cluster.cluster_id] = merged_cluster
                        
                        # Remove old clusters
                        if cluster1.cluster_id in self.clusters:
                            del self.clusters[cluster1.cluster_id]
                        if cluster2.cluster_id in self.clusters:
                            del self.clusters[cluster2.cluster_id]
                        
                        # Update agent cluster list
                        agent_cluster_ids = self.agent_clusters[agent_id]
                        if cluster1.cluster_id in agent_cluster_ids:
                            agent_cluster_ids.remove(cluster1.cluster_id)
                        if cluster2.cluster_id in agent_cluster_ids:
                            agent_cluster_ids.remove(cluster2.cluster_id)
                        agent_cluster_ids.append(merged_cluster.cluster_id)
                        
                        # Update clusters list
                        clusters[i] = merged_cluster
                        clusters.pop(j)
                        merged_count += 1
                        
                        # Don't increment j since we removed an element
                        continue
                    
                    j += 1
                i += 1
            
            self.logger.info(f"Merged {merged_count} clusters for agent {agent_id}")
            return merged_count
            
        except Exception as e:
            self.logger.error(f"Failed to merge similar clusters: {e}")
            return 0
    
    def _calculate_cluster_center_similarity(self, cluster1: TemporalCluster, cluster2: TemporalCluster) -> float:
        """Calculate similarity between cluster centers"""
        try:
            if not cluster1.center_embedding or not cluster2.center_embedding:
                return 0.0
            
            return self._cosine_similarity(cluster1.center_embedding, cluster2.center_embedding)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cluster center similarity: {e}")
            return 0.0
    
    async def _merge_clusters(self, cluster1: TemporalCluster, cluster2: TemporalCluster) -> TemporalCluster:
        """Merge two clusters into one"""
        try:
            # Combine experience IDs
            combined_experience_ids = cluster1.experience_ids + cluster2.experience_ids
            
            # Calculate new center embedding
            if cluster1.center_embedding and cluster2.center_embedding:
                new_center = np.mean([cluster1.center_embedding, cluster2.center_embedding], axis=0).tolist()
            else:
                new_center = cluster1.center_embedding or cluster2.center_embedding or []
            
            # Calculate new time window
            new_start_time = min(cluster1.start_time, cluster2.start_time)
            new_end_time = max(cluster1.end_time, cluster2.end_time)
            new_time_window = new_end_time - new_start_time
            
            # Merge temporal patterns
            merged_pattern = self._merge_temporal_patterns(cluster1.temporal_pattern, cluster2.temporal_pattern)
            
            # Calculate new cohesion score (average)
            new_cohesion = (cluster1.cohesion_score + cluster2.cohesion_score) / 2
            
            # Calculate new activity density
            total_experiences = len(combined_experience_ids)
            time_span_hours = max(new_time_window.total_seconds() / 3600, 1)
            new_activity_density = total_experiences / time_span_hours
            
            merged_cluster = TemporalCluster(
                cluster_id=str(uuid.uuid4()),
                algorithm=cluster1.algorithm,  # Use first cluster's algorithm
                time_window=new_time_window,
                start_time=new_start_time,
                end_time=new_end_time,
                experience_ids=combined_experience_ids,
                center_embedding=new_center,
                temporal_pattern=merged_pattern,
                cohesion_score=new_cohesion,
                activity_density=new_activity_density,
                created_at=min(cluster1.created_at, cluster2.created_at),
                last_updated=datetime.now(),
                metadata={
                    'agent_id': cluster1.metadata.get('agent_id'),
                    'experience_count': len(combined_experience_ids),
                    'merged_from': [cluster1.cluster_id, cluster2.cluster_id],
                    'algorithm': cluster1.algorithm.value
                }
            )
            
            return merged_cluster
            
        except Exception as e:
            self.logger.error(f"Failed to merge clusters: {e}")
            raise
    
    def _merge_temporal_patterns(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge temporal patterns from two clusters"""
        try:
            merged_pattern = {
                'activity_distribution': {},
                'success_trend': [],
                'action_frequency': {},
                'peak_hours': []
            }
            
            # Merge activity distribution
            for hour, count in pattern1.get('activity_distribution', {}).items():
                merged_pattern['activity_distribution'][hour] = count
            for hour, count in pattern2.get('activity_distribution', {}).items():
                merged_pattern['activity_distribution'][hour] = merged_pattern['activity_distribution'].get(hour, 0) + count
            
            # Merge action frequency
            for action, count in pattern1.get('action_frequency', {}).items():
                merged_pattern['action_frequency'][action] = count
            for action, count in pattern2.get('action_frequency', {}).items():
                merged_pattern['action_frequency'][action] = merged_pattern['action_frequency'].get(action, 0) + count
            
            # Merge peak hours (union)
            peak_hours = set(pattern1.get('peak_hours', []) + pattern2.get('peak_hours', []))
            merged_pattern['peak_hours'] = list(peak_hours)
            
            # Merge success trends (combine and average)
            # This is simplified - in practice would need more sophisticated merging
            merged_pattern['success_trend'] = pattern1.get('success_trend', []) + pattern2.get('success_trend', [])
            
            return merged_pattern
            
        except Exception as e:
            self.logger.error(f"Failed to merge temporal patterns: {e}")
            return {}
    
    def get_clustering_statistics(self) -> Dict[str, Any]:
        """Get clustering engine statistics"""
        try:
            total_clusters = len(self.clusters)
            agents_with_clusters = len(self.agent_clusters)
            
            # Calculate average cluster size
            cluster_sizes = [len(cluster.experience_ids) for cluster in self.clusters.values()]
            avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0
            
            # Calculate average cohesion
            cohesion_scores = [cluster.cohesion_score for cluster in self.clusters.values()]
            avg_cohesion = np.mean(cohesion_scores) if cohesion_scores else 0.0
            
            # Algorithm distribution
            algorithm_counts = defaultdict(int)
            for cluster in self.clusters.values():
                algorithm_counts[cluster.algorithm.value] += 1
            
            return {
                'total_clusters': total_clusters,
                'agents_with_clusters': agents_with_clusters,
                'average_cluster_size': avg_cluster_size,
                'average_cohesion_score': avg_cohesion,
                'algorithm_distribution': dict(algorithm_counts),
                'config': {
                    'algorithm': self.config.algorithm.value,
                    'min_cluster_size': self.config.min_cluster_size,
                    'similarity_threshold': self.config.similarity_threshold,
                    'time_epsilon_hours': self.config.time_epsilon.total_seconds() / 3600
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get clustering statistics: {e}")
            return {'error': str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown the temporal clustering engine"""
        try:
            self.logger.info("Shutting down temporal clustering engine")
            
            # Clear data structures
            self.clusters.clear()
            self.segments.clear()
            self.agent_clusters.clear()
            
            self.initialized = False
            self.logger.info("Temporal clustering engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during temporal clustering engine shutdown: {e}")

# Factory function
def create_temporal_clustering_engine(config: ClusteringConfig = None) -> TemporalClusteringEngine:
    """Create a temporal clustering engine instance"""
    return TemporalClusteringEngine(config or ClusteringConfig())