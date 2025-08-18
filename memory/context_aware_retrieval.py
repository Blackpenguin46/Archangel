#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Context-Aware Memory Retrieval
Advanced retrieval system with multi-dimensional relevance scoring
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
from collections import defaultdict
import json
import math

from agents.base_agent import Experience, Team, Role
from .semantic_annotation import SemanticAnnotation, SemanticTag, AnnotationType
from .temporal_clustering import TemporalCluster

logger = logging.getLogger(__name__)

class RelevanceScoreType(Enum):
    """Types of relevance scoring"""
    SEMANTIC = "semantic"           # Semantic similarity
    TEMPORAL = "temporal"           # Temporal relevance
    CONTEXTUAL = "contextual"       # Contextual match
    PERFORMANCE = "performance"     # Performance-based relevance
    STRATEGIC = "strategic"         # Strategic importance
    COLLABORATIVE = "collaborative" # Team/collaboration relevance
    LEARNING = "learning"           # Learning progression relevance
    RECENCY = "recency"            # Recency-based scoring

@dataclass
class RetrievalContext:
    """Context information for memory retrieval"""
    # Query information
    query_text: str = ""
    query_type: str = "general"  # general, tactical, strategic, learning, etc.
    
    # Agent context
    agent_id: Optional[str] = None
    agent_role: Optional[Role] = None
    agent_team: Optional[Team] = None
    agent_skill_level: float = 0.5  # 0.0 = novice, 1.0 = expert
    
    # Scenario context
    current_scenario: Optional[str] = None
    scenario_phase: Optional[str] = None  # reconnaissance, exploitation, etc.
    target_environment: Optional[str] = None
    urgency_level: float = 0.5  # 0.0 = low urgency, 1.0 = critical
    
    # Temporal context
    query_timestamp: datetime = field(default_factory=datetime.now)
    time_window: Optional[Tuple[datetime, datetime]] = None
    temporal_weight: float = 0.3  # Weight for temporal relevance
    
    # Performance context
    required_confidence: float = 0.5
    success_only: bool = False
    learning_focus: bool = False
    
    # Collaboration context
    team_context: bool = False
    similar_agents_only: bool = False
    cross_team_learning: bool = True
    
    # Retrieval preferences
    max_results: int = 20
    min_relevance_score: float = 0.3
    diversity_factor: float = 0.2  # 0.0 = no diversity, 1.0 = max diversity
    include_annotations: bool = True
    
    # Scoring weights
    score_weights: Dict[RelevanceScoreType, float] = field(default_factory=lambda: {
        RelevanceScoreType.SEMANTIC: 0.25,
        RelevanceScoreType.TEMPORAL: 0.15,
        RelevanceScoreType.CONTEXTUAL: 0.20,
        RelevanceScoreType.PERFORMANCE: 0.15,
        RelevanceScoreType.STRATEGIC: 0.10,
        RelevanceScoreType.COLLABORATIVE: 0.05,
        RelevanceScoreType.LEARNING: 0.05,
        RelevanceScoreType.RECENCY: 0.05
    })

@dataclass
class RelevanceScore:
    """Detailed relevance scoring information"""
    total_score: float
    component_scores: Dict[RelevanceScoreType, float] = field(default_factory=dict)
    confidence: float = 0.0
    explanation: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextualMemoryResult:
    """Enhanced memory result with contextual relevance"""
    memory_id: str
    experience: Experience
    relevance_score: RelevanceScore
    
    # Context matching
    context_match_factors: List[str] = field(default_factory=list)
    semantic_annotations: List[SemanticAnnotation] = field(default_factory=list)
    temporal_cluster: Optional[TemporalCluster] = None
    
    # Learning and development
    skill_relevance: Dict[str, float] = field(default_factory=dict)
    learning_potential: float = 0.0
    
    # Collaboration aspects
    team_relevance: float = 0.0
    collaboration_insights: List[str] = field(default_factory=list)
    
    # Metadata
    retrieval_timestamp: datetime = field(default_factory=datetime.now)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

class ContextAwareMemoryRetriever:
    """
    Advanced context-aware memory retrieval system with multi-dimensional relevance scoring.
    
    Features:
    - Multi-dimensional relevance scoring (semantic, temporal, contextual, etc.)
    - Context-aware filtering and ranking
    - Learning progression awareness
    - Team and collaboration context
    - Performance-based relevance
    - Strategic importance weighting
    - Diversity-aware result selection
    """
    
    def __init__(self):
        # Memory storage references (would be injected in real implementation)
        self.vector_memory = None
        self.semantic_annotator = None
        self.temporal_clusterer = None
        
        # Caching for performance
        self.embedding_cache: Dict[str, List[float]] = {}
        self.score_cache: Dict[str, Dict[str, float]] = {}
        
        # Analytics and optimization
        self.retrieval_analytics = {
            "total_retrievals": 0,
            "avg_relevance_score": 0.0,
            "retrieval_latency": [],
            "context_types": defaultdict(int),
            "score_distributions": defaultdict(list)
        }
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self, vector_memory=None, semantic_annotator=None, temporal_clusterer=None) -> None:
        """Initialize the context-aware retrieval system"""
        try:
            self.logger.info("Initializing context-aware memory retrieval system")
            
            # Set component references
            self.vector_memory = vector_memory
            self.semantic_annotator = semantic_annotator
            self.temporal_clusterer = temporal_clusterer
            
            # Initialize scoring models and caches
            await self._initialize_scoring_models()
            await self._load_retrieval_cache()
            
            self.initialized = True
            self.logger.info("Context-aware retrieval system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize context-aware retrieval: {e}")
            raise
    
    async def retrieve_contextual_memories(self, 
                                         context: RetrievalContext) -> List[ContextualMemoryResult]:
        """
        Retrieve memories using advanced contextual relevance scoring
        
        Args:
            context: Retrieval context with query and constraints
            
        Returns:
            List[ContextualMemoryResult]: Ranked contextual memory results
        """
        try:
            start_time = datetime.now()
            self.logger.debug(f"Starting contextual memory retrieval for query: {context.query_text[:50]}...")
            
            # Phase 1: Initial candidate retrieval
            initial_candidates = await self._get_initial_candidates(context)
            self.logger.debug(f"Retrieved {len(initial_candidates)} initial candidates")
            
            # Phase 2: Context-aware filtering
            filtered_candidates = await self._apply_contextual_filters(initial_candidates, context)
            self.logger.debug(f"Filtered to {len(filtered_candidates)} candidates")
            
            # Phase 3: Multi-dimensional relevance scoring
            scored_results = await self._score_relevance(filtered_candidates, context)
            self.logger.debug(f"Scored {len(scored_results)} candidates")
            
            # Phase 4: Diversity-aware ranking and selection
            final_results = await self._rank_and_select(scored_results, context)
            self.logger.debug(f"Selected {len(final_results)} final results")
            
            # Phase 5: Enhancement with contextual metadata
            enhanced_results = await self._enhance_with_context(final_results, context)
            
            # Record analytics
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            await self._record_retrieval_analytics(context, enhanced_results, latency)
            
            self.logger.info(f"Contextual retrieval completed in {latency:.3f}s, returned {len(enhanced_results)} results")
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve contextual memories: {e}")
            return []
    
    async def _get_initial_candidates(self, context: RetrievalContext) -> List[Tuple[str, Experience, List[float]]]:
        """Get initial candidate memories using basic similarity search"""
        try:
            candidates = []
            
            # Vector similarity search if available
            if self.vector_memory and context.query_text:
                from .vector_memory import MemoryQuery
                
                vector_query = MemoryQuery(
                    query_text=context.query_text,
                    agent_id=context.agent_id,
                    similarity_threshold=0.1,  # Low threshold for initial retrieval
                    max_results=context.max_results * 3,  # Get more candidates for filtering
                    include_metadata=True
                )
                
                vector_results = await self.vector_memory.retrieve_similar_experiences(vector_query)
                
                for result in vector_results:
                    candidates.append((result.experience_id, result.experience, result.embedding))
            
            # Semantic tag-based retrieval if semantic annotator available
            if self.semantic_annotator and context.query_type != "general":
                relevant_tags = await self._map_query_to_semantic_tags(context)
                semantic_memory_ids = self.semantic_annotator.search_by_semantic_tags(relevant_tags)
                
                # Add semantic candidates (would need to fetch full experiences)
                for memory_id in semantic_memory_ids[:context.max_results]:
                    if not any(memory_id == candidate[0] for candidate in candidates):
                        # In real implementation, would fetch experience from storage
                        # For now, create placeholder
                        pass
            
            # Temporal clustering based retrieval
            if self.temporal_clusterer and context.time_window:
                # Get memories from relevant temporal clusters
                # Implementation would depend on temporal clusterer interface
                pass
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Failed to get initial candidates: {e}")
            return []
    
    async def _apply_contextual_filters(self, 
                                      candidates: List[Tuple[str, Experience, List[float]]], 
                                      context: RetrievalContext) -> List[Tuple[str, Experience, List[float]]]:
        """Apply contextual filters to candidate memories"""
        try:
            filtered = []
            
            for memory_id, experience, embedding in candidates:
                # Time window filter
                if context.time_window:
                    start_time, end_time = context.time_window
                    if not (start_time <= experience.timestamp <= end_time):
                        continue
                
                # Agent filter
                if context.agent_id and experience.agent_id != context.agent_id:
                    if not context.cross_team_learning:
                        continue
                
                # Role filter
                if context.agent_role and hasattr(experience, 'role'):
                    if experience.role != context.agent_role and not context.cross_team_learning:
                        continue
                
                # Success filter
                if context.success_only and not experience.success:
                    continue
                
                # Confidence filter
                if experience.confidence_score < context.required_confidence:
                    continue
                
                # Team context filter
                if context.team_context and hasattr(experience, 'team_context'):
                    if not getattr(experience, 'team_context', False):
                        continue
                
                # Scenario filter
                if context.current_scenario:
                    # Check if experience is relevant to current scenario
                    scenario_relevance = await self._check_scenario_relevance(experience, context.current_scenario)
                    if scenario_relevance < 0.3:
                        continue
                
                filtered.append((memory_id, experience, embedding))
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Failed to apply contextual filters: {e}")
            return candidates
    
    async def _score_relevance(self, 
                             candidates: List[Tuple[str, Experience, List[float]]], 
                             context: RetrievalContext) -> List[ContextualMemoryResult]:
        """Score candidates using multi-dimensional relevance scoring"""
        try:
            scored_results = []
            
            for memory_id, experience, embedding in candidates:
                # Calculate component scores
                component_scores = {}
                explanations = []
                
                # Semantic relevance
                if RelevanceScoreType.SEMANTIC in context.score_weights:
                    semantic_score = await self._calculate_semantic_relevance(
                        context.query_text, experience, embedding
                    )
                    component_scores[RelevanceScoreType.SEMANTIC] = semantic_score
                    if semantic_score > 0.7:
                        explanations.append(f"High semantic similarity ({semantic_score:.2f})")
                
                # Temporal relevance
                if RelevanceScoreType.TEMPORAL in context.score_weights:
                    temporal_score = await self._calculate_temporal_relevance(experience, context)
                    component_scores[RelevanceScoreType.TEMPORAL] = temporal_score
                    if temporal_score > 0.7:
                        explanations.append(f"High temporal relevance ({temporal_score:.2f})")
                
                # Contextual relevance
                if RelevanceScoreType.CONTEXTUAL in context.score_weights:
                    contextual_score = await self._calculate_contextual_relevance(experience, context)
                    component_scores[RelevanceScoreType.CONTEXTUAL] = contextual_score
                    if contextual_score > 0.7:
                        explanations.append(f"Strong contextual match ({contextual_score:.2f})")
                
                # Performance relevance
                if RelevanceScoreType.PERFORMANCE in context.score_weights:
                    performance_score = await self._calculate_performance_relevance(experience, context)
                    component_scores[RelevanceScoreType.PERFORMANCE] = performance_score
                
                # Strategic relevance
                if RelevanceScoreType.STRATEGIC in context.score_weights:
                    strategic_score = await self._calculate_strategic_relevance(experience, context)
                    component_scores[RelevanceScoreType.STRATEGIC] = strategic_score
                
                # Collaborative relevance
                if RelevanceScoreType.COLLABORATIVE in context.score_weights:
                    collaborative_score = await self._calculate_collaborative_relevance(experience, context)
                    component_scores[RelevanceScoreType.COLLABORATIVE] = collaborative_score
                
                # Learning relevance
                if RelevanceScoreType.LEARNING in context.score_weights:
                    learning_score = await self._calculate_learning_relevance(experience, context)
                    component_scores[RelevanceScoreType.LEARNING] = learning_score
                
                # Recency relevance
                if RelevanceScoreType.RECENCY in context.score_weights:
                    recency_score = await self._calculate_recency_relevance(experience, context)
                    component_scores[RelevanceScoreType.RECENCY] = recency_score
                
                # Calculate weighted total score
                total_score = 0.0
                for score_type, weight in context.score_weights.items():
                    if score_type in component_scores:
                        total_score += component_scores[score_type] * weight
                
                # Create relevance score object
                relevance_score = RelevanceScore(
                    total_score=total_score,
                    component_scores=component_scores,
                    confidence=min(component_scores.values()) if component_scores else 0.0,
                    explanation=explanations,
                    metadata={
                        "scoring_timestamp": datetime.now().isoformat(),
                        "context_type": context.query_type,
                        "weights_used": dict(context.score_weights)
                    }
                )
                
                # Create contextual result
                result = ContextualMemoryResult(
                    memory_id=memory_id,
                    experience=experience,
                    relevance_score=relevance_score
                )
                
                scored_results.append(result)
            
            return scored_results
            
        except Exception as e:
            self.logger.error(f"Failed to score relevance: {e}")
            return []
    
    async def _calculate_semantic_relevance(self, query: str, experience: Experience, embedding: List[float]) -> float:
        """Calculate semantic similarity relevance"""
        try:
            if not query or not embedding:
                return 0.0
            
            # In real implementation, would use proper semantic similarity
            # For now, use simplified text matching
            experience_text = f"{experience.lessons_learned} {experience.mitre_attack_mapping}"
            
            query_words = set(query.lower().split())
            experience_words = set(str(experience_text).lower().split())
            
            if not query_words:
                return 0.0
            
            intersection = query_words.intersection(experience_words)
            semantic_score = len(intersection) / len(query_words)
            
            return min(1.0, semantic_score)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate semantic relevance: {e}")
            return 0.0
    
    async def _calculate_temporal_relevance(self, experience: Experience, context: RetrievalContext) -> float:
        """Calculate temporal relevance based on recency and temporal patterns"""
        try:
            now = context.query_timestamp
            time_diff = now - experience.timestamp
            
            # Recency component (exponential decay)
            days_old = time_diff.days
            recency_score = math.exp(-days_old / 30.0)  # Decay over 30 days
            
            # Time-of-day similarity
            query_hour = now.hour
            exp_hour = experience.timestamp.hour
            hour_diff = min(abs(query_hour - exp_hour), 24 - abs(query_hour - exp_hour))
            time_of_day_score = 1.0 - (hour_diff / 12.0)
            
            # Day-of-week similarity
            query_dow = now.weekday()
            exp_dow = experience.timestamp.weekday()
            dow_score = 1.0 if query_dow == exp_dow else 0.7 if abs(query_dow - exp_dow) <= 1 else 0.3
            
            # Combined temporal score
            temporal_score = (recency_score * 0.6 + time_of_day_score * 0.2 + dow_score * 0.2)
            
            return min(1.0, temporal_score)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate temporal relevance: {e}")
            return 0.0
    
    async def _calculate_contextual_relevance(self, experience: Experience, context: RetrievalContext) -> float:
        """Calculate contextual relevance based on scenario and environment"""
        try:
            contextual_score = 0.0
            factors = 0
            
            # Scenario relevance
            if context.current_scenario:
                scenario_score = await self._check_scenario_relevance(experience, context.current_scenario)
                contextual_score += scenario_score
                factors += 1
            
            # Role relevance
            if context.agent_role and hasattr(experience, 'role'):
                role_match = 1.0 if experience.role == context.agent_role else 0.5
                contextual_score += role_match
                factors += 1
            
            # Team relevance
            if context.agent_team and hasattr(experience, 'team'):
                team_match = 1.0 if experience.team == context.agent_team else 0.3
                contextual_score += team_match
                factors += 1
            
            # Environment relevance
            if context.target_environment:
                # Simplified environment matching
                env_score = 0.5  # Default score
                contextual_score += env_score
                factors += 1
            
            return contextual_score / factors if factors > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Failed to calculate contextual relevance: {e}")
            return 0.0
    
    async def _calculate_performance_relevance(self, experience: Experience, context: RetrievalContext) -> float:
        """Calculate performance-based relevance"""
        try:
            # Success weighting
            success_score = 1.0 if experience.success else 0.3
            
            # Confidence alignment with required confidence
            confidence_diff = abs(experience.confidence_score - context.required_confidence)
            confidence_score = 1.0 - confidence_diff
            
            # Skill level alignment
            skill_alignment = 1.0 - abs(experience.confidence_score - context.agent_skill_level)
            
            # Combined performance score
            performance_score = (success_score * 0.4 + confidence_score * 0.4 + skill_alignment * 0.2)
            
            return min(1.0, max(0.0, performance_score))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance relevance: {e}")
            return 0.0
    
    async def _calculate_strategic_relevance(self, experience: Experience, context: RetrievalContext) -> float:
        """Calculate strategic importance relevance"""
        try:
            strategic_score = 0.0
            
            # Lesson quality (more lessons = more strategic value)
            if experience.lessons_learned:
                lesson_score = min(1.0, len(experience.lessons_learned) / 3.0)
                strategic_score += lesson_score * 0.4
            
            # MITRE mapping (indicates tactical sophistication)
            if experience.mitre_attack_mapping:
                mitre_score = min(1.0, len(experience.mitre_attack_mapping) / 5.0)
                strategic_score += mitre_score * 0.3
            
            # Experience confidence (high confidence experiences more strategic)
            strategic_score += experience.confidence_score * 0.3
            
            return min(1.0, strategic_score)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate strategic relevance: {e}")
            return 0.0
    
    async def _calculate_collaborative_relevance(self, experience: Experience, context: RetrievalContext) -> float:
        """Calculate collaboration-based relevance"""
        try:
            if not context.team_context:
                return 0.5  # Neutral if not team context
            
            # Team operation experience
            team_score = 0.0
            if hasattr(experience, 'team_context') and getattr(experience, 'team_context', False):
                team_score = 1.0
            else:
                team_score = 0.3  # Solo experiences still valuable
            
            # Cross-team learning
            if context.cross_team_learning:
                if hasattr(experience, 'team') and experience.team != context.agent_team:
                    team_score *= 0.8  # Slight penalty for different team
            
            return team_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate collaborative relevance: {e}")
            return 0.0
    
    async def _calculate_learning_relevance(self, experience: Experience, context: RetrievalContext) -> float:
        """Calculate learning progression relevance"""
        try:
            if not context.learning_focus:
                return 0.5  # Neutral if not learning focused
            
            learning_score = 0.0
            
            # Lesson richness
            if experience.lessons_learned:
                learning_score += min(1.0, len(experience.lessons_learned) / 2.0) * 0.5
            
            # Failure experiences valuable for learning
            if not experience.success:
                learning_score += 0.3
            
            # Confidence progression opportunities
            if experience.confidence_score < context.agent_skill_level:
                learning_score += 0.2  # Learn from more confident experiences
            
            return min(1.0, learning_score)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate learning relevance: {e}")
            return 0.0
    
    async def _calculate_recency_relevance(self, experience: Experience, context: RetrievalContext) -> float:
        """Calculate recency-based relevance"""
        try:
            time_diff = context.query_timestamp - experience.timestamp
            days_old = time_diff.days
            
            # Exponential decay over time
            recency_score = math.exp(-days_old / 14.0)  # 14-day half-life
            
            # Urgency factor
            if context.urgency_level > 0.7:
                # High urgency prefers very recent experiences
                recency_score = math.exp(-days_old / 7.0)  # 7-day half-life for urgent queries
            
            return min(1.0, recency_score)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate recency relevance: {e}")
            return 0.0
    
    async def _rank_and_select(self, 
                             scored_results: List[ContextualMemoryResult], 
                             context: RetrievalContext) -> List[ContextualMemoryResult]:
        """Rank results and apply diversity-aware selection"""
        try:
            # Filter by minimum relevance score
            filtered_results = [
                result for result in scored_results
                if result.relevance_score.total_score >= context.min_relevance_score
            ]
            
            # Sort by relevance score
            filtered_results.sort(key=lambda r: r.relevance_score.total_score, reverse=True)
            
            # Apply diversity if requested
            if context.diversity_factor > 0.0:
                filtered_results = await self._apply_diversity_selection(filtered_results, context)
            
            # Limit results
            return filtered_results[:context.max_results]
            
        except Exception as e:
            self.logger.error(f"Failed to rank and select results: {e}")
            return scored_results[:context.max_results]
    
    async def _apply_diversity_selection(self, 
                                       results: List[ContextualMemoryResult], 
                                       context: RetrievalContext) -> List[ContextualMemoryResult]:
        """Apply diversity-aware selection to avoid redundant results"""
        try:
            if len(results) <= context.max_results:
                return results
            
            selected = []
            remaining = results.copy()
            
            # Always include top result
            if remaining:
                selected.append(remaining.pop(0))
            
            # Select diverse results
            while len(selected) < context.max_results and remaining:
                best_candidate = None
                best_score = -1.0
                
                for candidate in remaining:
                    # Calculate diversity score
                    diversity_score = await self._calculate_diversity_score(candidate, selected)
                    
                    # Combined score: relevance + diversity
                    combined_score = (
                        candidate.relevance_score.total_score * (1.0 - context.diversity_factor) +
                        diversity_score * context.diversity_factor
                    )
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = candidate
                
                if best_candidate:
                    selected.append(best_candidate)
                    remaining.remove(best_candidate)
                else:
                    break
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Failed to apply diversity selection: {e}")
            return results[:context.max_results]
    
    async def _calculate_diversity_score(self, 
                                       candidate: ContextualMemoryResult, 
                                       selected: List[ContextualMemoryResult]) -> float:
        """Calculate diversity score for a candidate against selected results"""
        try:
            if not selected:
                return 1.0
            
            diversity_scores = []
            
            for selected_result in selected:
                # Time diversity
                time_diff = abs((candidate.experience.timestamp - selected_result.experience.timestamp).days)
                time_diversity = min(1.0, time_diff / 30.0)  # Different if >30 days apart
                
                # Agent diversity
                agent_diversity = 0.0 if candidate.experience.agent_id == selected_result.experience.agent_id else 1.0
                
                # Success pattern diversity
                success_diversity = 0.0 if candidate.experience.success == selected_result.experience.success else 0.5
                
                # Confidence diversity
                conf_diff = abs(candidate.experience.confidence_score - selected_result.experience.confidence_score)
                confidence_diversity = min(1.0, conf_diff / 0.5)
                
                # Combined diversity for this pair
                pair_diversity = (time_diversity * 0.3 + agent_diversity * 0.3 + 
                                success_diversity * 0.2 + confidence_diversity * 0.2)
                diversity_scores.append(pair_diversity)
            
            # Return minimum diversity (most similar to any selected result)
            return min(diversity_scores)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate diversity score: {e}")
            return 0.5
    
    async def _enhance_with_context(self, 
                                  results: List[ContextualMemoryResult], 
                                  context: RetrievalContext) -> List[ContextualMemoryResult]:
        """Enhance results with additional contextual metadata"""
        try:
            enhanced_results = []
            
            for result in results:
                # Add semantic annotations if available
                if self.semantic_annotator and context.include_annotations:
                    annotations = self.semantic_annotator.get_memory_annotations(result.memory_id)
                    result.semantic_annotations = annotations
                
                # Add context match factors
                result.context_match_factors = await self._identify_context_matches(result, context)
                
                # Calculate skill relevance
                result.skill_relevance = await self._calculate_skill_relevance(result, context)
                
                # Calculate learning potential
                result.learning_potential = await self._calculate_learning_potential(result, context)
                
                # Add collaboration insights
                result.collaboration_insights = await self._extract_collaboration_insights(result, context)
                
                # Add processing metadata
                result.processing_metadata = {
                    "retrieval_context_type": context.query_type,
                    "score_components": len(result.relevance_score.component_scores),
                    "enhancement_timestamp": datetime.now().isoformat()
                }
                
                enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Failed to enhance with context: {e}")
            return results
    
    async def _identify_context_matches(self, 
                                      result: ContextualMemoryResult, 
                                      context: RetrievalContext) -> List[str]:
        """Identify specific context matching factors"""
        try:
            matches = []
            
            # Scenario match
            if context.current_scenario:
                scenario_relevance = await self._check_scenario_relevance(
                    result.experience, context.current_scenario
                )
                if scenario_relevance > 0.7:
                    matches.append(f"scenario_match_{context.current_scenario}")
            
            # Role match
            if context.agent_role and hasattr(result.experience, 'role'):
                if result.experience.role == context.agent_role:
                    matches.append(f"role_match_{context.agent_role.value}")
            
            # Success pattern match
            if result.experience.success and context.success_only:
                matches.append("success_pattern_match")
            
            # Confidence level match
            conf_diff = abs(result.experience.confidence_score - context.required_confidence)
            if conf_diff < 0.2:
                matches.append("confidence_level_match")
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Failed to identify context matches: {e}")
            return []
    
    async def _calculate_skill_relevance(self, 
                                       result: ContextualMemoryResult, 
                                       context: RetrievalContext) -> Dict[str, float]:
        """Calculate relevance to specific skills"""
        try:
            skill_relevance = {}
            
            # Extract skills from experience (simplified)
            if hasattr(result.experience, 'action_taken') and result.experience.action_taken:
                action_type = getattr(result.experience.action_taken, 'action_type', '')
                if action_type:
                    skill_relevance[action_type] = result.experience.confidence_score
            
            # MITRE techniques as skills
            if result.experience.mitre_attack_mapping:
                for technique in result.experience.mitre_attack_mapping:
                    skill_relevance[f"mitre_{technique}"] = result.experience.confidence_score
            
            return skill_relevance
            
        except Exception as e:
            self.logger.error(f"Failed to calculate skill relevance: {e}")
            return {}
    
    async def _calculate_learning_potential(self, 
                                          result: ContextualMemoryResult, 
                                          context: RetrievalContext) -> float:
        """Calculate learning potential of this memory for the agent"""
        try:
            learning_potential = 0.0
            
            # Rich lessons increase learning potential
            if result.experience.lessons_learned:
                learning_potential += min(1.0, len(result.experience.lessons_learned) / 3.0) * 0.4
            
            # Confidence gap indicates learning opportunity
            skill_gap = abs(result.experience.confidence_score - context.agent_skill_level)
            if result.experience.confidence_score > context.agent_skill_level:
                learning_potential += skill_gap * 0.4  # Learn from more skilled experiences
            
            # Failure experiences valuable for learning
            if not result.experience.success:
                learning_potential += 0.2
            
            return min(1.0, learning_potential)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate learning potential: {e}")
            return 0.0
    
    async def _extract_collaboration_insights(self, 
                                            result: ContextualMemoryResult, 
                                            context: RetrievalContext) -> List[str]:
        """Extract collaboration insights from the memory"""
        try:
            insights = []
            
            # Team operation insights
            if hasattr(result.experience, 'team_context') and getattr(result.experience, 'team_context', False):
                insights.append("team_operation_experience")
            
            # Cross-team learning insights
            if (hasattr(result.experience, 'team') and 
                context.agent_team and 
                result.experience.team != context.agent_team):
                insights.append(f"cross_team_learning_from_{result.experience.team.value}")
            
            # Collaborative lessons
            if result.experience.lessons_learned:
                for lesson in result.experience.lessons_learned:
                    if any(term in lesson.lower() for term in ["team", "coordinate", "collaborate", "communicate"]):
                        insights.append("collaborative_lesson_available")
                        break
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to extract collaboration insights: {e}")
            return []
    
    async def _check_scenario_relevance(self, experience: Experience, scenario: str) -> float:
        """Check relevance of experience to specific scenario"""
        try:
            # Simplified scenario matching (would be more sophisticated in production)
            relevance_score = 0.5  # Default neutral relevance
            
            # Check if scenario keywords appear in experience data
            scenario_lower = scenario.lower()
            
            # Check lessons learned
            if experience.lessons_learned:
                for lesson in experience.lessons_learned:
                    if any(term in lesson.lower() for term in scenario_lower.split()):
                        relevance_score += 0.2
            
            # Check MITRE mappings
            if experience.mitre_attack_mapping:
                # Would map scenario to expected MITRE techniques
                relevance_score += 0.1
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            self.logger.error(f"Failed to check scenario relevance: {e}")
            return 0.5
    
    async def _map_query_to_semantic_tags(self, context: RetrievalContext) -> List[SemanticTag]:
        """Map query context to relevant semantic tags"""
        try:
            tags = []
            
            query_lower = context.query_text.lower()
            
            # Map query terms to semantic tags
            if any(term in query_lower for term in ["recon", "scan", "discover"]):
                tags.append(SemanticTag.RECONNAISSANCE)
            
            if any(term in query_lower for term in ["exploit", "attack", "penetrat"]):
                tags.append(SemanticTag.EXPLOITATION)
            
            if any(term in query_lower for term in ["persist", "maintain"]):
                tags.append(SemanticTag.PERSISTENCE)
            
            if any(term in query_lower for term in ["lateral", "movement", "pivot"]):
                tags.append(SemanticTag.LATERAL_MOVEMENT)
            
            # Map query type to tags
            if context.query_type == "tactical":
                tags.extend([SemanticTag.TECHNICAL_SKILL, SemanticTag.PROBLEM_SOLVING])
            elif context.query_type == "learning":
                tags.extend([SemanticTag.LEARNING_AGILITY, SemanticTag.ADAPTATION])
            
            return tags
            
        except Exception as e:
            self.logger.error(f"Failed to map query to semantic tags: {e}")
            return []
    
    async def _record_retrieval_analytics(self, 
                                        context: RetrievalContext, 
                                        results: List[ContextualMemoryResult], 
                                        latency: float) -> None:
        """Record analytics for retrieval operation"""
        try:
            self.retrieval_analytics["total_retrievals"] += 1
            self.retrieval_analytics["retrieval_latency"].append(latency)
            self.retrieval_analytics["context_types"][context.query_type] += 1
            
            if results:
                avg_score = sum(r.relevance_score.total_score for r in results) / len(results)
                self.retrieval_analytics["avg_relevance_score"] = (
                    (self.retrieval_analytics["avg_relevance_score"] * (self.retrieval_analytics["total_retrievals"] - 1) + avg_score) /
                    self.retrieval_analytics["total_retrievals"]
                )
                
                for result in results:
                    self.retrieval_analytics["score_distributions"]["total_scores"].append(result.relevance_score.total_score)
            
        except Exception as e:
            self.logger.error(f"Failed to record retrieval analytics: {e}")
    
    async def _initialize_scoring_models(self) -> None:
        """Initialize scoring models and algorithms"""
        try:
            # Initialize caching structures
            self.embedding_cache.clear()
            self.score_cache.clear()
            
            self.logger.debug("Initialized scoring models")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize scoring models: {e}")
    
    async def _load_retrieval_cache(self) -> None:
        """Load retrieval cache from storage"""
        try:
            # In production, would load from persistent storage
            self.logger.debug("Loaded retrieval cache")
            
        except Exception as e:
            self.logger.error(f"Failed to load retrieval cache: {e}")
    
    def get_retrieval_analytics(self) -> Dict[str, Any]:
        """Get comprehensive retrieval analytics"""
        try:
            analytics = dict(self.retrieval_analytics)
            
            # Calculate derived metrics
            if analytics["retrieval_latency"]:
                analytics["avg_latency"] = sum(analytics["retrieval_latency"]) / len(analytics["retrieval_latency"])
                analytics["max_latency"] = max(analytics["retrieval_latency"])
                analytics["min_latency"] = min(analytics["retrieval_latency"])
            
            # Score distribution stats
            if analytics["score_distributions"]["total_scores"]:
                scores = analytics["score_distributions"]["total_scores"]
                analytics["score_stats"] = {
                    "mean": sum(scores) / len(scores),
                    "max": max(scores),
                    "min": min(scores)
                }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to get retrieval analytics: {e}")
            return {}

# Factory function
def create_context_aware_retriever() -> ContextAwareMemoryRetriever:
    """Create a context-aware memory retriever instance"""
    return ContextAwareMemoryRetriever()