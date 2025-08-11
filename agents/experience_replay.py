#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Experience Replay System
Experience replay and tactical knowledge distillation for agent learning
"""

import asyncio
import logging
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict

from .base_agent import Experience, Team, Role, ActionResult, ActionPlan
from memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class ReplayMode(Enum):
    """Experience replay modes"""
    RANDOM = "random"
    PRIORITIZED = "prioritized"
    TEMPORAL = "temporal"
    STRATEGIC = "strategic"
    ADVERSARIAL = "adversarial"

class KnowledgeType(Enum):
    """Types of knowledge for distillation"""
    TACTICAL = "tactical"
    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    BEHAVIORAL = "behavioral"
    COLLABORATIVE = "collaborative"

@dataclass
class ReplayConfig:
    """Configuration for experience replay system"""
    buffer_size: int = 50000
    batch_size: int = 64
    replay_frequency: int = 10
    prioritization_alpha: float = 0.6
    importance_sampling_beta: float = 0.4
    temporal_window: timedelta = timedelta(hours=24)
    knowledge_distillation_rate: float = 0.1
    min_replay_size: int = 1000
    max_replay_age: timedelta = timedelta(days=7)
    diversity_threshold: float = 0.3
    success_bias: float = 0.7

@dataclass
class ExperienceMetadata:
    """Metadata for experience prioritization"""
    priority: float
    temporal_relevance: float
    strategic_value: float
    learning_potential: float
    diversity_score: float
    success_impact: float
    replay_count: int
    last_replayed: Optional[datetime]
    knowledge_extracted: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeDistillation:
    """Distilled knowledge from experiences"""
    knowledge_id: str
    knowledge_type: KnowledgeType
    source_experiences: List[str]
    distilled_patterns: Dict[str, Any]
    confidence_score: float
    applicability_scope: Dict[str, Any]
    validation_results: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    usage_count: int
    effectiveness_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReplayBatch:
    """Batch of experiences for replay learning"""
    batch_id: str
    experiences: List[Experience]
    batch_metadata: List[ExperienceMetadata]
    replay_mode: ReplayMode
    learning_objectives: List[str]
    expected_outcomes: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    
    def __init__(self, max_size: int, alpha: float = 0.6):
        self.max_size = max_size
        self.alpha = alpha  # Prioritization exponent
        
        self.buffer = []
        self.priorities = []
        self.metadata = []
        self.position = 0
        
        # Sum tree for efficient sampling
        self._sum_tree = [0.0] * (2 * max_size)
        self._min_tree = [float('inf')] * (2 * max_size)
        
    def add(self, experience: Experience, metadata: ExperienceMetadata) -> None:
        """Add experience with priority"""
        priority = metadata.priority
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
            self.metadata.append(metadata)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            self.metadata[self.position] = metadata
        
        # Update sum tree
        self._update_tree(self.position, priority)
        
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], List[ExperienceMetadata], List[float], List[int]]:
        """Sample batch with importance sampling weights"""
        if len(self.buffer) == 0:
            return [], [], [], []
        
        batch_size = min(batch_size, len(self.buffer))
        
        # Sample indices based on priorities
        indices = []
        weights = []
        
        total_priority = self._sum_tree[1]  # Root of sum tree
        segment_size = total_priority / batch_size
        
        for i in range(batch_size):
            # Sample from segment
            segment_start = segment_size * i
            segment_end = segment_size * (i + 1)
            sample_value = np.random.uniform(segment_start, segment_end)
            
            # Find index in sum tree
            idx = self._find_index(sample_value)
            indices.append(idx)
            
            # Calculate importance sampling weight
            priority = self.priorities[idx]
            prob = priority / total_priority
            weight = (len(self.buffer) * prob) ** (-beta)
            weights.append(weight)
        
        # Normalize weights
        max_weight = max(weights) if weights else 1.0
        weights = [w / max_weight for w in weights]
        
        # Get experiences and metadata
        experiences = [self.buffer[i] for i in indices]
        metadata = [self.metadata[i] for i in indices]
        
        return experiences, metadata, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.buffer):
                self.priorities[idx] = priority
                self._update_tree(idx, priority)
    
    def _update_tree(self, idx: int, priority: float) -> None:
        """Update sum tree with new priority"""
        tree_idx = idx + self.max_size
        
        # Update sum tree
        delta = priority - self._sum_tree[tree_idx]
        self._sum_tree[tree_idx] = priority
        
        # Propagate changes up the tree
        while tree_idx > 1:
            tree_idx //= 2
            self._sum_tree[tree_idx] += delta
        
        # Update min tree
        tree_idx = idx + self.max_size
        self._min_tree[tree_idx] = priority
        
        while tree_idx > 1:
            tree_idx //= 2
            self._min_tree[tree_idx] = min(
                self._min_tree[2 * tree_idx],
                self._min_tree[2 * tree_idx + 1]
            )
    
    def _find_index(self, value: float) -> int:
        """Find index in sum tree for given value"""
        idx = 1  # Start at root
        
        while idx < self.max_size:
            left_child = 2 * idx
            right_child = left_child + 1
            
            if value <= self._sum_tree[left_child]:
                idx = left_child
            else:
                value -= self._sum_tree[left_child]
                idx = right_child
        
        return idx - self.max_size
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def get_max_priority(self) -> float:
        """Get maximum priority in buffer"""
        return max(self.priorities) if self.priorities else 1.0

class ExperienceReplaySystem:
    """
    Experience replay system for tactical knowledge distillation and learning
    """
    
    def __init__(self, config: ReplayConfig = None, memory_manager: MemoryManager = None):
        self.config = config or ReplayConfig()
        self.memory_manager = memory_manager
        
        # Replay buffers
        self.replay_buffer = PrioritizedReplayBuffer(
            self.config.buffer_size,
            self.config.prioritization_alpha
        )
        
        # Knowledge distillation
        self.distilled_knowledge = {}  # knowledge_id -> KnowledgeDistillation
        self.knowledge_patterns = defaultdict(list)  # pattern_type -> List[patterns]
        
        # Replay tracking
        self.replay_history = []
        self.replay_statistics = {
            'total_replays': 0,
            'successful_replays': 0,
            'knowledge_extractions': 0,
            'pattern_discoveries': 0,
            'learning_improvements': 0
        }
        
        # Background tasks
        self.replay_task = None
        self.knowledge_distillation_task = None
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.running = False
    
    async def initialize(self) -> None:
        """Initialize the experience replay system"""
        try:
            self.logger.info("Initializing experience replay system")
            
            # Initialize memory manager if not provided
            if self.memory_manager is None:
                from memory.memory_manager import create_memory_manager
                self.memory_manager = create_memory_manager()
                await self.memory_manager.initialize()
            
            # Start background tasks
            self.replay_task = asyncio.create_task(self._replay_loop())
            self.knowledge_distillation_task = asyncio.create_task(self._knowledge_distillation_loop())
            
            self.initialized = True
            self.running = True
            
            self.logger.info("Experience replay system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize experience replay system: {e}")
            raise
    
    async def add_experience(self, experience: Experience, priority: float = None) -> None:
        """
        Add experience to replay buffer
        
        Args:
            experience: Experience to add
            priority: Optional priority (calculated if not provided)
        """
        try:
            # Calculate priority if not provided
            if priority is None:
                priority = await self._calculate_experience_priority(experience)
            
            # Create metadata
            metadata = ExperienceMetadata(
                priority=priority,
                temporal_relevance=self._calculate_temporal_relevance(experience),
                strategic_value=self._calculate_strategic_value(experience),
                learning_potential=self._calculate_learning_potential(experience),
                diversity_score=self._calculate_diversity_score(experience),
                success_impact=self._calculate_success_impact(experience),
                replay_count=0,
                last_replayed=None,
                knowledge_extracted=False
            )
            
            # Add to buffer
            self.replay_buffer.add(experience, metadata)
            
            self.logger.debug(f"Added experience {experience.experience_id} with priority {priority:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to add experience to replay buffer: {e}")
    
    async def sample_replay_batch(self, 
                                batch_size: int = None,
                                replay_mode: ReplayMode = ReplayMode.PRIORITIZED,
                                learning_objectives: List[str] = None) -> ReplayBatch:
        """
        Sample a batch of experiences for replay learning
        
        Args:
            batch_size: Size of batch to sample
            replay_mode: Mode for sampling experiences
            learning_objectives: Specific learning objectives
            
        Returns:
            ReplayBatch: Batch of experiences for learning
        """
        try:
            batch_size = batch_size or self.config.batch_size
            learning_objectives = learning_objectives or []
            
            if replay_mode == ReplayMode.PRIORITIZED:
                experiences, metadata, weights, indices = self.replay_buffer.sample(
                    batch_size, self.config.importance_sampling_beta
                )
            elif replay_mode == ReplayMode.RANDOM:
                experiences, metadata = await self._sample_random_batch(batch_size)
                weights = [1.0] * len(experiences)
                indices = list(range(len(experiences)))
            elif replay_mode == ReplayMode.TEMPORAL:
                experiences, metadata = await self._sample_temporal_batch(batch_size)
                weights = [1.0] * len(experiences)
                indices = list(range(len(experiences)))
            elif replay_mode == ReplayMode.STRATEGIC:
                experiences, metadata = await self._sample_strategic_batch(batch_size, learning_objectives)
                weights = [1.0] * len(experiences)
                indices = list(range(len(experiences)))
            else:
                # Fallback to prioritized
                experiences, metadata, weights, indices = self.replay_buffer.sample(
                    batch_size, self.config.importance_sampling_beta
                )
            
            # Create replay batch
            batch = ReplayBatch(
                batch_id=str(uuid.uuid4()),
                experiences=experiences,
                batch_metadata=metadata,
                replay_mode=replay_mode,
                learning_objectives=learning_objectives,
                expected_outcomes=await self._predict_batch_outcomes(experiences, learning_objectives),
                timestamp=datetime.now(),
                metadata={
                    'sampling_weights': weights,
                    'buffer_indices': indices,
                    'batch_diversity': self._calculate_batch_diversity(experiences),
                    'batch_quality': self._calculate_batch_quality(experiences, metadata)
                }
            )
            
            # Update replay counts
            for meta in metadata:
                meta.replay_count += 1
                meta.last_replayed = datetime.now()
            
            self.logger.debug(f"Sampled replay batch {batch.batch_id} with {len(experiences)} experiences")
            return batch
            
        except Exception as e:
            self.logger.error(f"Failed to sample replay batch: {e}")
            return ReplayBatch(
                batch_id=str(uuid.uuid4()),
                experiences=[],
                batch_metadata=[],
                replay_mode=replay_mode,
                learning_objectives=learning_objectives,
                expected_outcomes={},
                timestamp=datetime.now()
            )
    
    async def process_replay_batch(self, batch: ReplayBatch) -> Dict[str, Any]:
        """
        Process a replay batch for learning
        
        Args:
            batch: Replay batch to process
            
        Returns:
            Dict containing learning results
        """
        try:
            results = {
                'batch_id': batch.batch_id,
                'processed_experiences': len(batch.experiences),
                'learning_outcomes': [],
                'knowledge_extracted': [],
                'pattern_discoveries': [],
                'performance_improvements': {},
                'processing_time': 0.0,
                'success': False
            }
            
            start_time = datetime.now()
            
            # Process each experience in the batch
            for i, (experience, metadata) in enumerate(zip(batch.experiences, batch.batch_metadata)):
                try:
                    # Extract learning from experience
                    learning_outcome = await self._extract_learning_from_experience(
                        experience, metadata, batch.learning_objectives
                    )
                    
                    if learning_outcome:
                        results['learning_outcomes'].append(learning_outcome)
                    
                    # Extract knowledge patterns
                    if not metadata.knowledge_extracted:
                        knowledge = await self._extract_knowledge_patterns(experience)
                        if knowledge:
                            results['knowledge_extracted'].append(knowledge)
                            metadata.knowledge_extracted = True
                    
                    # Discover new patterns
                    patterns = await self._discover_patterns(experience, batch.experiences)
                    if patterns:
                        results['pattern_discoveries'].extend(patterns)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process experience {experience.experience_id}: {e}")
            
            # Calculate batch-level insights
            batch_insights = await self._analyze_batch_insights(batch, results)
            results['batch_insights'] = batch_insights
            
            # Update statistics
            self.replay_statistics['total_replays'] += 1
            if results['learning_outcomes']:
                self.replay_statistics['successful_replays'] += 1
            self.replay_statistics['knowledge_extractions'] += len(results['knowledge_extracted'])
            self.replay_statistics['pattern_discoveries'] += len(results['pattern_discoveries'])
            
            # Store replay history
            self.replay_history.append({
                'batch_id': batch.batch_id,
                'timestamp': batch.timestamp,
                'results': results,
                'replay_mode': batch.replay_mode.value
            })
            
            # Keep only recent history
            if len(self.replay_history) > 1000:
                self.replay_history = self.replay_history[-1000:]
            
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
            results['success'] = True
            
            self.logger.info(f"Processed replay batch {batch.batch_id}: {len(results['learning_outcomes'])} outcomes, {len(results['knowledge_extracted'])} knowledge extractions")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process replay batch: {e}")
            return {'error': str(e), 'success': False}
    
    async def distill_tactical_knowledge(self, 
                                       experiences: List[Experience],
                                       knowledge_type: KnowledgeType = KnowledgeType.TACTICAL) -> Optional[KnowledgeDistillation]:
        """
        Distill tactical knowledge from a set of experiences
        
        Args:
            experiences: List of experiences to analyze
            knowledge_type: Type of knowledge to distill
            
        Returns:
            KnowledgeDistillation: Distilled knowledge or None
        """
        try:
            if not experiences:
                return None
            
            # Analyze experiences for patterns
            patterns = await self._analyze_experience_patterns(experiences, knowledge_type)
            
            if not patterns:
                return None
            
            # Create knowledge distillation
            knowledge = KnowledgeDistillation(
                knowledge_id=str(uuid.uuid4()),
                knowledge_type=knowledge_type,
                source_experiences=[exp.experience_id for exp in experiences],
                distilled_patterns=patterns,
                confidence_score=self._calculate_pattern_confidence(patterns, experiences),
                applicability_scope=self._determine_applicability_scope(patterns, experiences),
                validation_results={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                usage_count=0,
                effectiveness_score=0.0
            )
            
            # Store distilled knowledge
            self.distilled_knowledge[knowledge.knowledge_id] = knowledge
            
            # Update knowledge patterns
            pattern_type = f"{knowledge_type.value}_patterns"
            self.knowledge_patterns[pattern_type].append(patterns)
            
            self.logger.info(f"Distilled {knowledge_type.value} knowledge {knowledge.knowledge_id} from {len(experiences)} experiences")
            
            return knowledge
            
        except Exception as e:
            self.logger.error(f"Failed to distill tactical knowledge: {e}")
            return None
    
    async def _calculate_experience_priority(self, experience: Experience) -> float:
        """Calculate priority for an experience"""
        try:
            priority = 1.0  # Base priority
            
            # Success bias
            if experience.success:
                priority *= (1.0 + self.config.success_bias)
            
            # Confidence factor
            priority *= experience.confidence_score
            
            # Temporal relevance (newer experiences have higher priority)
            age = datetime.now() - experience.timestamp
            temporal_factor = max(0.1, 1.0 - (age.total_seconds() / self.config.temporal_window.total_seconds()))
            priority *= temporal_factor
            
            # Learning potential (experiences with lessons learned)
            if experience.lessons_learned:
                priority *= (1.0 + len(experience.lessons_learned) * 0.1)
            
            # MITRE ATT&CK mapping (tactical relevance)
            if experience.mitre_attack_mapping:
                priority *= (1.0 + len(experience.mitre_attack_mapping) * 0.05)
            
            return max(0.01, min(10.0, priority))  # Clamp between 0.01 and 10.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate experience priority: {e}")
            return 1.0
    
    def _calculate_temporal_relevance(self, experience: Experience) -> float:
        """Calculate temporal relevance of experience"""
        age = datetime.now() - experience.timestamp
        max_age = self.config.max_replay_age.total_seconds()
        return max(0.0, 1.0 - (age.total_seconds() / max_age))
    
    def _calculate_strategic_value(self, experience: Experience) -> float:
        """Calculate strategic value of experience"""
        value = 0.5  # Base value
        
        # Success contributes to strategic value
        if experience.success:
            value += 0.3
        
        # Lessons learned indicate strategic value
        if experience.lessons_learned:
            value += len(experience.lessons_learned) * 0.1
        
        # MITRE ATT&CK mapping indicates tactical/strategic relevance
        if experience.mitre_attack_mapping:
            value += len(experience.mitre_attack_mapping) * 0.05
        
        return min(1.0, value)
    
    def _calculate_learning_potential(self, experience: Experience) -> float:
        """Calculate learning potential of experience"""
        potential = 0.5  # Base potential
        
        # Failed experiences often have high learning potential
        if not experience.success:
            potential += 0.2
        
        # Experiences with explicit lessons learned
        if experience.lessons_learned:
            potential += 0.3
        
        # High confidence experiences are good for learning
        potential += experience.confidence_score * 0.2
        
        return min(1.0, potential)
    
    def _calculate_diversity_score(self, experience: Experience) -> float:
        """Calculate diversity score compared to existing experiences"""
        # This would compare against existing experiences in buffer
        # For now, return a placeholder value
        return 0.5
    
    def _calculate_success_impact(self, experience: Experience) -> float:
        """Calculate impact of experience on success"""
        if experience.success:
            return experience.confidence_score
        else:
            # Failed experiences can still have learning impact
            return 0.3 * experience.confidence_score
    
    async def _sample_random_batch(self, batch_size: int) -> Tuple[List[Experience], List[ExperienceMetadata]]:
        """Sample random batch of experiences"""
        if self.replay_buffer.size() == 0:
            return [], []
        
        indices = np.random.choice(self.replay_buffer.size(), min(batch_size, self.replay_buffer.size()), replace=False)
        experiences = [self.replay_buffer.buffer[i] for i in indices]
        metadata = [self.replay_buffer.metadata[i] for i in indices]
        
        return experiences, metadata
    
    async def _sample_temporal_batch(self, batch_size: int) -> Tuple[List[Experience], List[ExperienceMetadata]]:
        """Sample batch based on temporal relevance"""
        if self.replay_buffer.size() == 0:
            return [], []
        
        # Sort by temporal relevance
        indexed_metadata = [(i, meta) for i, meta in enumerate(self.replay_buffer.metadata)]
        indexed_metadata.sort(key=lambda x: x[1].temporal_relevance, reverse=True)
        
        # Take top temporal experiences
        selected_indices = [idx for idx, _ in indexed_metadata[:batch_size]]
        experiences = [self.replay_buffer.buffer[i] for i in selected_indices]
        metadata = [self.replay_buffer.metadata[i] for i in selected_indices]
        
        return experiences, metadata
    
    async def _sample_strategic_batch(self, batch_size: int, learning_objectives: List[str]) -> Tuple[List[Experience], List[ExperienceMetadata]]:
        """Sample batch based on strategic value and learning objectives"""
        if self.replay_buffer.size() == 0:
            return [], []
        
        # Score experiences based on strategic value and objective alignment
        scored_experiences = []
        
        for i, (exp, meta) in enumerate(zip(self.replay_buffer.buffer, self.replay_buffer.metadata)):
            score = meta.strategic_value
            
            # Boost score if experience aligns with learning objectives
            for objective in learning_objectives:
                if objective.lower() in str(exp.lessons_learned).lower():
                    score += 0.2
                if hasattr(exp, 'action_taken') and exp.action_taken and objective.lower() in str(exp.action_taken).lower():
                    score += 0.1
            
            scored_experiences.append((i, score))
        
        # Sort by score and select top experiences
        scored_experiences.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in scored_experiences[:batch_size]]
        
        experiences = [self.replay_buffer.buffer[i] for i in selected_indices]
        metadata = [self.replay_buffer.metadata[i] for i in selected_indices]
        
        return experiences, metadata
    
    async def _predict_batch_outcomes(self, experiences: List[Experience], learning_objectives: List[str]) -> Dict[str, Any]:
        """Predict expected outcomes from processing a batch"""
        return {
            'expected_learning_outcomes': len(experiences) * 0.7,  # Expect 70% to yield learning
            'expected_knowledge_extractions': len(experiences) * 0.3,  # Expect 30% to yield knowledge
            'expected_pattern_discoveries': max(1, len(experiences) // 10),  # Expect patterns from groups
            'alignment_with_objectives': len(learning_objectives) * 0.5
        }
    
    def _calculate_batch_diversity(self, experiences: List[Experience]) -> float:
        """Calculate diversity within a batch"""
        if not experiences:
            return 0.0
        
        # Calculate diversity based on different factors
        teams = set()
        roles = set()
        action_types = set()
        
        for exp in experiences:
            if hasattr(exp, 'context') and exp.context:
                # Extract team/role information if available
                pass
            
            if hasattr(exp, 'action_taken') and exp.action_taken:
                action_types.add(str(exp.action_taken))
        
        # Normalize diversity score
        max_diversity = len(experiences)
        actual_diversity = len(action_types)
        
        return min(1.0, actual_diversity / max_diversity) if max_diversity > 0 else 0.0
    
    def _calculate_batch_quality(self, experiences: List[Experience], metadata: List[ExperienceMetadata]) -> float:
        """Calculate overall quality of a batch"""
        if not experiences or not metadata:
            return 0.0
        
        # Average various quality metrics
        avg_priority = sum(meta.priority for meta in metadata) / len(metadata)
        avg_strategic_value = sum(meta.strategic_value for meta in metadata) / len(metadata)
        avg_learning_potential = sum(meta.learning_potential for meta in metadata) / len(metadata)
        
        return (avg_priority + avg_strategic_value + avg_learning_potential) / 3.0
    
    async def _extract_learning_from_experience(self, 
                                              experience: Experience,
                                              metadata: ExperienceMetadata,
                                              learning_objectives: List[str]) -> Optional[Dict[str, Any]]:
        """Extract learning outcomes from an experience"""
        try:
            learning_outcome = {
                'experience_id': experience.experience_id,
                'learning_type': 'tactical',
                'insights': [],
                'patterns': [],
                'recommendations': [],
                'confidence': experience.confidence_score
            }
            
            # Extract explicit lessons learned
            if experience.lessons_learned:
                learning_outcome['insights'].extend(experience.lessons_learned)
            
            # Analyze success/failure patterns
            if experience.success:
                learning_outcome['patterns'].append({
                    'type': 'success_pattern',
                    'context': 'successful_action',
                    'factors': ['high_confidence', 'good_timing']
                })
            else:
                learning_outcome['patterns'].append({
                    'type': 'failure_pattern',
                    'context': 'failed_action',
                    'factors': ['low_confidence', 'poor_timing']
                })
            
            # Generate recommendations
            if not experience.success and experience.lessons_learned:
                learning_outcome['recommendations'].append(
                    f"Avoid similar actions in comparable contexts: {experience.lessons_learned[0] if experience.lessons_learned else 'Unknown'}"
                )
            
            return learning_outcome if learning_outcome['insights'] or learning_outcome['patterns'] else None
            
        except Exception as e:
            self.logger.error(f"Failed to extract learning from experience: {e}")
            return None
    
    async def _extract_knowledge_patterns(self, experience: Experience) -> Optional[Dict[str, Any]]:
        """Extract knowledge patterns from an experience"""
        try:
            patterns = {
                'pattern_id': str(uuid.uuid4()),
                'pattern_type': 'tactical',
                'extracted_patterns': [],
                'confidence': experience.confidence_score,
                'applicability': 'general'
            }
            
            # Extract action-outcome patterns
            if hasattr(experience, 'action_taken') and experience.action_taken:
                patterns['extracted_patterns'].append({
                    'type': 'action_outcome',
                    'action': str(experience.action_taken),
                    'outcome': experience.success,
                    'context_factors': []
                })
            
            # Extract MITRE ATT&CK patterns
            if experience.mitre_attack_mapping:
                patterns['extracted_patterns'].append({
                    'type': 'mitre_mapping',
                    'techniques': experience.mitre_attack_mapping,
                    'success_rate': 1.0 if experience.success else 0.0
                })
            
            return patterns if patterns['extracted_patterns'] else None
            
        except Exception as e:
            self.logger.error(f"Failed to extract knowledge patterns: {e}")
            return None
    
    async def _discover_patterns(self, experience: Experience, batch_experiences: List[Experience]) -> List[Dict[str, Any]]:
        """Discover patterns across multiple experiences"""
        patterns = []
        
        try:
            # Look for common success/failure patterns
            similar_experiences = [
                exp for exp in batch_experiences
                if exp.experience_id != experience.experience_id and exp.success == experience.success
            ]
            
            if len(similar_experiences) >= 2:
                patterns.append({
                    'pattern_type': 'outcome_correlation',
                    'pattern_description': f"Found {len(similar_experiences)} similar {'successful' if experience.success else 'failed'} experiences",
                    'confidence': min(1.0, len(similar_experiences) / 10.0),
                    'supporting_experiences': [exp.experience_id for exp in similar_experiences[:5]]
                })
            
        except Exception as e:
            self.logger.error(f"Failed to discover patterns: {e}")
        
        return patterns
    
    async def _analyze_batch_insights(self, batch: ReplayBatch, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze insights from batch processing"""
        insights = {
            'batch_effectiveness': 0.0,
            'learning_quality': 0.0,
            'knowledge_density': 0.0,
            'pattern_richness': 0.0,
            'recommendations': []
        }
        
        try:
            # Calculate batch effectiveness
            if batch.experiences:
                insights['batch_effectiveness'] = len(results['learning_outcomes']) / len(batch.experiences)
            
            # Calculate learning quality
            if results['learning_outcomes']:
                avg_confidence = sum(
                    outcome.get('confidence', 0.0) for outcome in results['learning_outcomes']
                ) / len(results['learning_outcomes'])
                insights['learning_quality'] = avg_confidence
            
            # Calculate knowledge density
            insights['knowledge_density'] = len(results['knowledge_extracted']) / max(1, len(batch.experiences))
            
            # Calculate pattern richness
            insights['pattern_richness'] = len(results['pattern_discoveries']) / max(1, len(batch.experiences))
            
            # Generate recommendations
            if insights['batch_effectiveness'] < 0.5:
                insights['recommendations'].append("Consider adjusting sampling strategy for better learning outcomes")
            
            if insights['knowledge_density'] < 0.3:
                insights['recommendations'].append("Focus on experiences with higher knowledge extraction potential")
            
        except Exception as e:
            self.logger.error(f"Failed to analyze batch insights: {e}")
        
        return insights
    
    async def _analyze_experience_patterns(self, experiences: List[Experience], knowledge_type: KnowledgeType) -> Dict[str, Any]:
        """Analyze patterns across multiple experiences"""
        patterns = {
            'common_actions': {},
            'success_factors': [],
            'failure_factors': [],
            'temporal_patterns': [],
            'contextual_patterns': []
        }
        
        try:
            # Analyze common actions
            action_counts = {}
            for exp in experiences:
                if hasattr(exp, 'action_taken') and exp.action_taken:
                    action_str = str(exp.action_taken)
                    action_counts[action_str] = action_counts.get(action_str, 0) + 1
            
            patterns['common_actions'] = action_counts
            
            # Analyze success/failure factors
            successful_experiences = [exp for exp in experiences if exp.success]
            failed_experiences = [exp for exp in experiences if not exp.success]
            
            if successful_experiences:
                patterns['success_factors'] = [
                    'high_confidence' if sum(exp.confidence_score for exp in successful_experiences) / len(successful_experiences) > 0.7 else 'variable_confidence',
                    'good_timing' if len(successful_experiences) > len(failed_experiences) else 'timing_issues'
                ]
            
            if failed_experiences:
                patterns['failure_factors'] = [
                    'low_confidence' if sum(exp.confidence_score for exp in failed_experiences) / len(failed_experiences) < 0.5 else 'confidence_not_factor',
                    'poor_execution' if len(failed_experiences) > len(successful_experiences) else 'execution_adequate'
                ]
            
        except Exception as e:
            self.logger.error(f"Failed to analyze experience patterns: {e}")
        
        return patterns
    
    def _calculate_pattern_confidence(self, patterns: Dict[str, Any], experiences: List[Experience]) -> float:
        """Calculate confidence in discovered patterns"""
        try:
            # Base confidence on number of supporting experiences
            base_confidence = min(1.0, len(experiences) / 10.0)
            
            # Adjust based on pattern consistency
            if 'common_actions' in patterns:
                action_consistency = len(patterns['common_actions']) / max(1, len(experiences))
                base_confidence *= (1.0 + action_consistency) / 2.0
            
            return max(0.1, min(1.0, base_confidence))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate pattern confidence: {e}")
            return 0.5
    
    def _determine_applicability_scope(self, patterns: Dict[str, Any], experiences: List[Experience]) -> Dict[str, Any]:
        """Determine the scope of applicability for patterns"""
        scope = {
            'teams': set(),
            'roles': set(),
            'scenarios': set(),
            'time_periods': [],
            'confidence_ranges': []
        }
        
        try:
            for exp in experiences:
                # Extract team/role information if available in context
                if hasattr(exp, 'context') and exp.context:
                    # This would extract actual team/role information
                    pass
                
                # Time period
                scope['time_periods'].append(exp.timestamp)
                
                # Confidence range
                scope['confidence_ranges'].append(exp.confidence_score)
            
            # Convert sets to lists for JSON serialization
            return {
                'teams': list(scope['teams']),
                'roles': list(scope['roles']),
                'scenarios': list(scope['scenarios']),
                'time_range': {
                    'start': min(scope['time_periods']).isoformat() if scope['time_periods'] else None,
                    'end': max(scope['time_periods']).isoformat() if scope['time_periods'] else None
                },
                'confidence_range': {
                    'min': min(scope['confidence_ranges']) if scope['confidence_ranges'] else 0.0,
                    'max': max(scope['confidence_ranges']) if scope['confidence_ranges'] else 1.0,
                    'avg': sum(scope['confidence_ranges']) / len(scope['confidence_ranges']) if scope['confidence_ranges'] else 0.5
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to determine applicability scope: {e}")
            return scope
    
    async def _replay_loop(self) -> None:
        """Background replay processing loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config.replay_frequency)
                
                if not self.running:
                    break
                
                # Check if we have enough experiences to replay
                if self.replay_buffer.size() < self.config.min_replay_size:
                    continue
                
                # Sample and process a batch
                batch = await self.sample_replay_batch()
                if batch.experiences:
                    results = await self.process_replay_batch(batch)
                    
                    if results.get('success', False):
                        self.logger.debug(f"Processed replay batch: {len(results.get('learning_outcomes', []))} outcomes")
                
            except Exception as e:
                self.logger.error(f"Error in replay loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _knowledge_distillation_loop(self) -> None:
        """Background knowledge distillation loop"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if not self.running:
                    break
                
                # Collect experiences for knowledge distillation
                if self.replay_buffer.size() >= 50:  # Need sufficient experiences
                    # Sample experiences for distillation
                    experiences = []
                    for i in range(min(50, self.replay_buffer.size())):
                        experiences.append(self.replay_buffer.buffer[i])
                    
                    # Distill knowledge
                    knowledge = await self.distill_tactical_knowledge(experiences, KnowledgeType.TACTICAL)
                    
                    if knowledge:
                        self.logger.info(f"Distilled knowledge {knowledge.knowledge_id}")
                
            except Exception as e:
                self.logger.error(f"Error in knowledge distillation loop: {e}")
                await asyncio.sleep(300)
    
    async def get_replay_statistics(self) -> Dict[str, Any]:
        """Get comprehensive replay system statistics"""
        try:
            return {
                "replay_system": {
                    "initialized": self.initialized,
                    "running": self.running,
                    "buffer_size": self.replay_buffer.size(),
                    "max_buffer_size": self.config.buffer_size,
                    "distilled_knowledge_count": len(self.distilled_knowledge),
                    "knowledge_patterns": {k: len(v) for k, v in self.knowledge_patterns.items()},
                    "config": {
                        "batch_size": self.config.batch_size,
                        "replay_frequency": self.config.replay_frequency,
                        "prioritization_alpha": self.config.prioritization_alpha,
                        "importance_sampling_beta": self.config.importance_sampling_beta
                    }
                },
                "statistics": self.replay_statistics,
                "recent_replays": self.replay_history[-10:] if self.replay_history else []
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get replay statistics: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown the experience replay system"""
        try:
            self.logger.info("Shutting down experience replay system")
            self.running = False
            
            # Cancel background tasks
            if self.replay_task:
                self.replay_task.cancel()
                try:
                    await self.replay_task
                except asyncio.CancelledError:
                    pass
            
            if self.knowledge_distillation_task:
                self.knowledge_distillation_task.cancel()
                try:
                    await self.knowledge_distillation_task
                except asyncio.CancelledError:
                    pass
            
            self.initialized = False
            self.logger.info("Experience replay system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during replay system shutdown: {e}")

# Factory function
def create_experience_replay_system(config: ReplayConfig = None, memory_manager: MemoryManager = None) -> ExperienceReplaySystem:
    """Create an experience replay system instance"""
    return ExperienceReplaySystem(config or ReplayConfig(), memory_manager)