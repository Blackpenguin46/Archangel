#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Context-Aware Memory Retrieval
Advanced context-aware memory retrieval with relevance scoring
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """Memory retrieval strategies"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TEMPORAL_PROXIMITY = "temporal_proximity"
    CONTEXTUAL_RELEVANCE = "contextual_relevance"
    HYBRID_SCORING = "hybrid_scoring"
    ADAPTIVE_RANKING = "adaptive_ranking"

class ContextType(Enum):
    """Types of context for retrieval"""
    CURRENT_TASK = "current_task"
    AGENT_STATE = "agent_state"
    SCENARIO_CONTEXT = "scenario_context"
    TEMPORAL_CONTEXT = "temporal_context"
    SOCIAL_CONTEXT = "social_context"
    ENVIRONMENTAL_CONTEXT = "environmental_context"

@dataclass
class RetrievalContext:
    """Context information for memory retrieval"""
    context_id: str
    context_type: ContextType
    agent_id: str
    current_task: Optional[str] = None
    agent_state: Dict[str, Any] = None
    scenario_info: Dict[str, Any] = None
    temporal_window: timedelta = None
    social_connections: List[str] = None
    environmental_factors: Dict[str, Any] = None
    priority_keywords: List[str] = None
    exclusion_filters: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.agent_state is None:
            self.agent_state = {}
        if self.scenario_info is None:
            self.scenario_info = {}
        if self.social_connections is None:
            self.social_connections = []
        if self.environmental_factors is None:
            self.environmental_factors = {}
        if self.priority_keywords is None:
            self.priority_keywords = []
        if self.exclusion_filters is None:
            self.exclusion_filters = {}
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class RelevanceScore:
    """Relevance scoring breakdown"""
    memory_id: str
    total_score: float
    semantic_score: float
    temporal_score: float
    contextual_score: float
    social_score: float
    recency_score: float
    success_score: float
    confidence_score: float
    scoring_breakdown: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class RetrievalResult:
    """Result from context-aware retrieval"""
    memory_id: str
    memory_content: Dict[str, Any]
    relevance_score: RelevanceScore
    retrieval_reason: str
    context_match: Dict[str, Any]
    retrieved_at: datetime

@dataclass
class RetrievalConfig:
    """Configuration for context-aware retrieval"""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_SCORING
    max_results: int = 20
    min_relevance_threshold: float = 0.3
    semantic_weight: float = 0.3
    temporal_weight: float = 0.2
    contextual_weight: float = 0.25
    social_weight: float = 0.1
    recency_weight: float = 0.1
    success_weight: float = 0.05
    temporal_decay_factor: float = 0.1
    context_expansion_enabled: bool = True
    adaptive_scoring_enabled: bool = True
    diversity_factor: float = 0.2

class ContextAwareRetrieval:
    """
    Advanced context-aware memory retrieval system.
    
    Features:
    - Multi-dimensional relevance scoring
    - Context-sensitive ranking
    - Adaptive retrieval strategies
    - Temporal and social context integration
    - Diversity-aware result selection
    - Real-time context adaptation
    """
    
    def __init__(self, config: RetrievalConfig = None):
        self.config = config or RetrievalConfig()
        self.retrieval_history: Dict[str, List[RetrievalResult]] = defaultdict(list)
        self.context_cache: Dict[str, RetrievalContext] = {}
        self.scoring_models: Dict[str, Any] = {}
        
        # Adaptive scoring parameters
        self.agent_preferences: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.context_patterns: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Statistics
        self.stats = {
            'retrievals_performed': 0,
            'contexts_processed': 0,
            'adaptive_adjustments': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the context-aware retrieval system"""
        try:
            self.logger.info("Initializing context-aware retrieval system")
            
            # Initialize scoring models
            await self._initialize_scoring_models()
            
            self.initialized = True
            self.logger.info("Context-aware retrieval system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize context-aware retrieval: {e}")
            raise
    
    async def retrieve_contextual_memories(self, 
                                         context: RetrievalContext,
                                         available_memories: List[Dict[str, Any]],
                                         query: str = "") -> List[RetrievalResult]:
        """
        Retrieve memories based on context with relevance scoring
        
        Args:
            context: Retrieval context information
            available_memories: Pool of available memories
            query: Optional text query
            
        Returns:
            List[RetrievalResult]: Ranked retrieval results
        """
        try:
            if not available_memories:
                return []
            
            # Cache context for future use
            self.context_cache[context.context_id] = context
            
            # Calculate relevance scores for all memories
            scored_memories = []
            for memory in available_memories:
                relevance_score = await self._calculate_relevance_score(memory, context, query)
                
                if relevance_score.total_score >= self.config.min_relevance_threshold:
                    retrieval_result = RetrievalResult(
                        memory_id=memory.get('memory_id', str(uuid.uuid4())),
                        memory_content=memory,
                        relevance_score=relevance_score,
                        retrieval_reason=await self._generate_retrieval_reason(relevance_score),
                        context_match=await self._analyze_context_match(memory, context),
                        retrieved_at=datetime.now()
                    )
                    scored_memories.append(retrieval_result)
            
            # Apply retrieval strategy
            if self.config.strategy == RetrievalStrategy.ADAPTIVE_RANKING:
                results = await self._adaptive_ranking(scored_memories, context)
            elif self.config.strategy == RetrievalStrategy.HYBRID_SCORING:
                results = await self._hybrid_scoring_ranking(scored_memories, context)
            else:
                results = sorted(scored_memories, key=lambda x: x.relevance_score.total_score, reverse=True)
            
            # Apply diversity filtering
            if self.config.diversity_factor > 0:
                results = await self._apply_diversity_filtering(results, context)
            
            # Limit results
            results = results[:self.config.max_results]
            
            # Store retrieval history
            self.retrieval_history[context.agent_id].extend(results)
            
            # Update adaptive parameters
            if self.config.adaptive_scoring_enabled:
                await self._update_adaptive_parameters(context, results)
            
            # Update statistics
            self.stats['retrievals_performed'] += 1
            self.stats['contexts_processed'] += 1
            
            self.logger.debug(f"Retrieved {len(results)} contextual memories for agent {context.agent_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve contextual memories: {e}")
            return []
    
    async def _calculate_relevance_score(self, memory: Dict[str, Any], context: RetrievalContext, query: str) -> RelevanceScore:
        """Calculate comprehensive relevance score for a memory"""
        try:
            # Initialize score components
            semantic_score = 0.0
            temporal_score = 0.0
            contextual_score = 0.0
            social_score = 0.0
            recency_score = 0.0
            success_score = 0.0
            confidence_score = 0.0
            
            # Semantic similarity score
            if query:
                semantic_score = await self._calculate_semantic_similarity(memory, query)
            
            # Temporal relevance score
            temporal_score = await self._calculate_temporal_relevance(memory, context)
            
            # Contextual relevance score
            contextual_score = await self._calculate_contextual_relevance(memory, context)
            
            # Social relevance score
            social_score = await self._calculate_social_relevance(memory, context)
            
            # Recency score
            recency_score = await self._calculate_recency_score(memory)
            
            # Success pattern score
            success_score = await self._calculate_success_score(memory, context)
            
            # Confidence score
            confidence_score = memory.get('confidence_score', 0.5)
            
            # Apply adaptive weights if available
            weights = await self._get_adaptive_weights(context.agent_id)
            
            # Calculate weighted total score
            total_score = (
                weights.get('semantic', self.config.semantic_weight) * semantic_score +
                weights.get('temporal', self.config.temporal_weight) * temporal_score +
                weights.get('contextual', self.config.contextual_weight) * contextual_score +
                weights.get('social', self.config.social_weight) * social_score +
                weights.get('recency', self.config.recency_weight) * recency_score +
                weights.get('success', self.config.success_weight) * success_score
            )
            
            # Apply confidence boost
            total_score = total_score * (0.5 + 0.5 * confidence_score)
            
            scoring_breakdown = {
                'semantic': semantic_score,
                'temporal': temporal_score,
                'contextual': contextual_score,
                'social': social_score,
                'recency': recency_score,
                'success': success_score,
                'confidence': confidence_score,
                'weights_used': weights
            }
            
            relevance_score = RelevanceScore(
                memory_id=memory.get('memory_id', str(uuid.uuid4())),
                total_score=total_score,
                semantic_score=semantic_score,
                temporal_score=temporal_score,
                contextual_score=contextual_score,
                social_score=social_score,
                recency_score=recency_score,
                success_score=success_score,
                confidence_score=confidence_score,
                scoring_breakdown=scoring_breakdown,
                metadata={
                    'memory_type': memory.get('type', 'unknown'),
                    'agent_id': memory.get('agent_id'),
                    'timestamp': memory.get('timestamp')
                }
            )
            
            return relevance_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate relevance score: {e}")
            return RelevanceScore(
                memory_id=memory.get('memory_id', str(uuid.uuid4())),
                total_score=0.0,
                semantic_score=0.0,
                temporal_score=0.0,
                contextual_score=0.0,
                social_score=0.0,
                recency_score=0.0,
                success_score=0.0,
                confidence_score=0.0,
                scoring_breakdown={},
                metadata={}
            )
    
    async def _calculate_semantic_similarity(self, memory: Dict[str, Any], query: str) -> float:
        """Calculate semantic similarity between memory and query"""
        try:
            if not query:
                return 0.0
            
            # Get memory content
            memory_text = memory.get('content', '')
            if isinstance(memory_text, dict):
                memory_text = str(memory_text)
            
            # Simple keyword matching (could be enhanced with embeddings)
            query_words = set(query.lower().split())
            memory_words = set(memory_text.lower().split())
            
            if not query_words or not memory_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(memory_words))
            union = len(query_words.union(memory_words))
            
            similarity = intersection / union if union > 0 else 0.0
            
            # Boost for exact phrase matches
            if query.lower() in memory_text.lower():
                similarity = min(1.0, similarity + 0.3)
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Failed to calculate semantic similarity: {e}")
            return 0.0
    
    async def _calculate_temporal_relevance(self, memory: Dict[str, Any], context: RetrievalContext) -> float:
        """Calculate temporal relevance of memory to current context"""
        try:
            memory_timestamp = memory.get('timestamp')
            if not memory_timestamp:
                return 0.5  # Neutral score for unknown timestamps
            
            if isinstance(memory_timestamp, str):
                memory_timestamp = datetime.fromisoformat(memory_timestamp)
            
            current_time = datetime.now()
            time_diff = current_time - memory_timestamp
            
            # Apply temporal window if specified
            if context.temporal_window:
                if time_diff <= context.temporal_window:
                    # Within window - high relevance
                    window_ratio = time_diff.total_seconds() / context.temporal_window.total_seconds()
                    return 1.0 - (window_ratio * 0.3)  # Slight decay within window
                else:
                    # Outside window - apply decay
                    excess_time = time_diff - context.temporal_window
                    decay_factor = np.exp(-excess_time.total_seconds() / (24 * 3600 * 7))  # Weekly decay
                    return 0.3 * decay_factor
            else:
                # No specific window - general recency scoring
                days_old = time_diff.total_seconds() / (24 * 3600)
                return np.exp(-days_old * self.config.temporal_decay_factor)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate temporal relevance: {e}")
            return 0.5
    
    async def _calculate_contextual_relevance(self, memory: Dict[str, Any], context: RetrievalContext) -> float:
        """Calculate contextual relevance based on task and scenario"""
        try:
            relevance_score = 0.0
            factors_count = 0
            
            # Task relevance
            if context.current_task and memory.get('task_context'):
                task_similarity = await self._calculate_task_similarity(
                    context.current_task, memory.get('task_context')
                )
                relevance_score += task_similarity
                factors_count += 1
            
            # Agent state relevance
            if context.agent_state and memory.get('agent_state'):
                state_similarity = await self._calculate_state_similarity(
                    context.agent_state, memory.get('agent_state')
                )
                relevance_score += state_similarity
                factors_count += 1
            
            # Scenario relevance
            if context.scenario_info and memory.get('scenario_info'):
                scenario_similarity = await self._calculate_scenario_similarity(
                    context.scenario_info, memory.get('scenario_info')
                )
                relevance_score += scenario_similarity
                factors_count += 1
            
            # Environmental factors
            if context.environmental_factors and memory.get('environmental_context'):
                env_similarity = await self._calculate_environmental_similarity(
                    context.environmental_factors, memory.get('environmental_context')
                )
                relevance_score += env_similarity
                factors_count += 1
            
            # Priority keywords
            if context.priority_keywords:
                keyword_score = await self._calculate_keyword_relevance(
                    memory, context.priority_keywords
                )
                relevance_score += keyword_score
                factors_count += 1
            
            return relevance_score / factors_count if factors_count > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Failed to calculate contextual relevance: {e}")
            return 0.5
    
    async def _calculate_social_relevance(self, memory: Dict[str, Any], context: RetrievalContext) -> float:
        """Calculate social relevance based on agent connections"""
        try:
            if not context.social_connections:
                return 0.5  # Neutral score if no social context
            
            memory_agent = memory.get('agent_id')
            if not memory_agent:
                return 0.5
            
            # Direct connection
            if memory_agent in context.social_connections:
                return 1.0
            
            # Same team or group (if available in memory)
            memory_team = memory.get('team')
            context_team = context.agent_state.get('team') if context.agent_state else None
            
            if memory_team and context_team and memory_team == context_team:
                return 0.8
            
            # Collaborative history (simplified)
            if memory.get('collaborative', False):
                return 0.6
            
            return 0.3  # Low but not zero for unrelated agents
            
        except Exception as e:
            self.logger.error(f"Failed to calculate social relevance: {e}")
            return 0.5
    
    async def _calculate_recency_score(self, memory: Dict[str, Any]) -> float:
        """Calculate recency score for memory"""
        try:
            memory_timestamp = memory.get('timestamp')
            if not memory_timestamp:
                return 0.5
            
            if isinstance(memory_timestamp, str):
                memory_timestamp = datetime.fromisoformat(memory_timestamp)
            
            current_time = datetime.now()
            time_diff = current_time - memory_timestamp
            hours_old = time_diff.total_seconds() / 3600
            
            # Exponential decay with configurable rate
            return np.exp(-hours_old / 168)  # Half-life of 1 week
            
        except Exception as e:
            self.logger.error(f"Failed to calculate recency score: {e}")
            return 0.5
    
    async def _calculate_success_score(self, memory: Dict[str, Any], context: RetrievalContext) -> float:
        """Calculate success pattern score"""
        try:
            memory_success = memory.get('success', None)
            if memory_success is None:
                return 0.5
            
            # Boost successful memories
            if memory_success:
                return 1.0
            else:
                # Failed memories can still be valuable for learning
                return 0.3
            
        except Exception as e:
            self.logger.error(f"Failed to calculate success score: {e}")
            return 0.5
    
    async def _calculate_task_similarity(self, current_task: str, memory_task: str) -> float:
        """Calculate similarity between current task and memory task"""
        try:
            if not current_task or not memory_task:
                return 0.0
            
            # Simple string similarity (could be enhanced)
            current_words = set(current_task.lower().split())
            memory_words = set(memory_task.lower().split())
            
            if not current_words or not memory_words:
                return 0.0
            
            intersection = len(current_words.intersection(memory_words))
            union = len(current_words.union(memory_words))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate task similarity: {e}")
            return 0.0
    
    async def _calculate_state_similarity(self, current_state: Dict[str, Any], memory_state: Dict[str, Any]) -> float:
        """Calculate similarity between agent states"""
        try:
            if not current_state or not memory_state:
                return 0.0
            
            # Compare key state attributes
            similarity_scores = []
            
            for key in current_state.keys():
                if key in memory_state:
                    current_val = current_state[key]
                    memory_val = memory_state[key]
                    
                    if current_val == memory_val:
                        similarity_scores.append(1.0)
                    elif isinstance(current_val, (int, float)) and isinstance(memory_val, (int, float)):
                        # Numerical similarity
                        max_val = max(abs(current_val), abs(memory_val), 1)
                        diff = abs(current_val - memory_val)
                        similarity_scores.append(1.0 - (diff / max_val))
                    else:
                        similarity_scores.append(0.0)
            
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate state similarity: {e}")
            return 0.0
    
    async def _calculate_scenario_similarity(self, current_scenario: Dict[str, Any], memory_scenario: Dict[str, Any]) -> float:
        """Calculate similarity between scenarios"""
        try:
            if not current_scenario or not memory_scenario:
                return 0.0
            
            # Compare scenario attributes
            similarity_factors = []
            
            # Scenario type
            if current_scenario.get('type') == memory_scenario.get('type'):
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.0)
            
            # Difficulty level
            current_difficulty = current_scenario.get('difficulty', 'medium')
            memory_difficulty = memory_scenario.get('difficulty', 'medium')
            if current_difficulty == memory_difficulty:
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.3)
            
            # Environment
            current_env = current_scenario.get('environment', '')
            memory_env = memory_scenario.get('environment', '')
            if current_env == memory_env:
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.0)
            
            return np.mean(similarity_factors) if similarity_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate scenario similarity: {e}")
            return 0.0
    
    async def _calculate_environmental_similarity(self, current_env: Dict[str, Any], memory_env: Dict[str, Any]) -> float:
        """Calculate similarity between environmental factors"""
        try:
            if not current_env or not memory_env:
                return 0.0
            
            # Simple key-value matching
            matching_keys = 0
            total_keys = len(set(current_env.keys()).union(set(memory_env.keys())))
            
            for key in current_env.keys():
                if key in memory_env and current_env[key] == memory_env[key]:
                    matching_keys += 1
            
            return matching_keys / total_keys if total_keys > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate environmental similarity: {e}")
            return 0.0
    
    async def _calculate_keyword_relevance(self, memory: Dict[str, Any], keywords: List[str]) -> float:
        """Calculate relevance based on priority keywords"""
        try:
            if not keywords:
                return 0.5
            
            memory_text = str(memory.get('content', ''))
            memory_text_lower = memory_text.lower()
            
            keyword_matches = 0
            for keyword in keywords:
                if keyword.lower() in memory_text_lower:
                    keyword_matches += 1
            
            return keyword_matches / len(keywords)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate keyword relevance: {e}")
            return 0.0
    
    async def _get_adaptive_weights(self, agent_id: str) -> Dict[str, float]:
        """Get adaptive weights for an agent"""
        try:
            if agent_id in self.agent_preferences:
                return self.agent_preferences[agent_id]
            else:
                # Return default weights
                return {
                    'semantic': self.config.semantic_weight,
                    'temporal': self.config.temporal_weight,
                    'contextual': self.config.contextual_weight,
                    'social': self.config.social_weight,
                    'recency': self.config.recency_weight,
                    'success': self.config.success_weight
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get adaptive weights: {e}")
            return {}
    
    async def _adaptive_ranking(self, scored_memories: List[RetrievalResult], context: RetrievalContext) -> List[RetrievalResult]:
        """Apply adaptive ranking based on agent history"""
        try:
            # Get agent's retrieval history
            agent_history = self.retrieval_history.get(context.agent_id, [])
            
            if not agent_history:
                # No history - use standard ranking
                return sorted(scored_memories, key=lambda x: x.relevance_score.total_score, reverse=True)
            
            # Analyze successful retrievals
            successful_patterns = await self._analyze_successful_patterns(agent_history)
            
            # Adjust scores based on patterns
            for result in scored_memories:
                pattern_boost = await self._calculate_pattern_boost(result, successful_patterns)
                result.relevance_score.total_score *= (1.0 + pattern_boost)
            
            # Sort by adjusted scores
            return sorted(scored_memories, key=lambda x: x.relevance_score.total_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to apply adaptive ranking: {e}")
            return scored_memories
    
    async def _hybrid_scoring_ranking(self, scored_memories: List[RetrievalResult], context: RetrievalContext) -> List[RetrievalResult]:
        """Apply hybrid scoring with multiple ranking factors"""
        try:
            # Sort by total score first
            sorted_memories = sorted(scored_memories, key=lambda x: x.relevance_score.total_score, reverse=True)
            
            # Apply secondary ranking factors
            for i, result in enumerate(sorted_memories):
                # Position penalty (slight preference for top results)
                position_factor = 1.0 - (i * 0.01)
                
                # Diversity bonus (prefer different types of memories)
                diversity_bonus = await self._calculate_diversity_bonus(result, sorted_memories[:i])
                
                # Final adjustment
                result.relevance_score.total_score *= position_factor * (1.0 + diversity_bonus)
            
            # Re-sort after adjustments
            return sorted(sorted_memories, key=lambda x: x.relevance_score.total_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to apply hybrid scoring ranking: {e}")
            return scored_memories
    
    async def _apply_diversity_filtering(self, results: List[RetrievalResult], context: RetrievalContext) -> List[RetrievalResult]:
        """Apply diversity filtering to avoid redundant results"""
        try:
            if self.config.diversity_factor <= 0 or len(results) <= 1:
                return results
            
            diverse_results = []
            seen_types = set()
            seen_agents = set()
            
            for result in results:
                memory = result.memory_content
                memory_type = memory.get('type', 'unknown')
                memory_agent = memory.get('agent_id', 'unknown')
                
                # Calculate diversity score
                diversity_score = 1.0
                
                # Type diversity
                if memory_type in seen_types:
                    diversity_score *= (1.0 - self.config.diversity_factor)
                else:
                    seen_types.add(memory_type)
                
                # Agent diversity
                if memory_agent in seen_agents:
                    diversity_score *= (1.0 - self.config.diversity_factor * 0.5)
                else:
                    seen_agents.add(memory_agent)
                
                # Apply diversity adjustment
                result.relevance_score.total_score *= diversity_score
                diverse_results.append(result)
            
            # Re-sort after diversity adjustment
            return sorted(diverse_results, key=lambda x: x.relevance_score.total_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to apply diversity filtering: {e}")
            return results
    
    async def _analyze_successful_patterns(self, history: List[RetrievalResult]) -> Dict[str, float]:
        """Analyze patterns in successful retrievals"""
        try:
            patterns = defaultdict(float)
            
            # Analyze recent successful retrievals (simplified)
            recent_history = history[-50:]  # Last 50 retrievals
            
            for result in recent_history:
                # Assume higher scores indicate more successful retrievals
                if result.relevance_score.total_score > 0.7:
                    # Extract patterns
                    memory_type = result.memory_content.get('type', 'unknown')
                    patterns[f"type:{memory_type}"] += 0.1
                    
                    memory_agent = result.memory_content.get('agent_id', 'unknown')
                    patterns[f"agent:{memory_agent}"] += 0.05
                    
                    # Scoring component patterns
                    breakdown = result.relevance_score.scoring_breakdown
                    for component, score in breakdown.items():
                        if score > 0.8:
                            patterns[f"high_{component}"] += 0.1
            
            return dict(patterns)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze successful patterns: {e}")
            return {}
    
    async def _calculate_pattern_boost(self, result: RetrievalResult, patterns: Dict[str, float]) -> float:
        """Calculate boost based on successful patterns"""
        try:
            boost = 0.0
            
            memory = result.memory_content
            memory_type = memory.get('type', 'unknown')
            memory_agent = memory.get('agent_id', 'unknown')
            
            # Apply pattern boosts
            boost += patterns.get(f"type:{memory_type}", 0.0)
            boost += patterns.get(f"agent:{memory_agent}", 0.0)
            
            # Component-based boosts
            breakdown = result.relevance_score.scoring_breakdown
            for component, score in breakdown.items():
                if score > 0.8:
                    boost += patterns.get(f"high_{component}", 0.0)
            
            return min(boost, 0.5)  # Cap boost at 50%
            
        except Exception as e:
            self.logger.error(f"Failed to calculate pattern boost: {e}")
            return 0.0
    
    async def _calculate_diversity_bonus(self, result: RetrievalResult, previous_results: List[RetrievalResult]) -> float:
        """Calculate diversity bonus for result"""
        try:
            if not previous_results:
                return 0.0
            
            memory = result.memory_content
            memory_type = memory.get('type', 'unknown')
            memory_agent = memory.get('agent_id', 'unknown')
            
            # Check for type diversity
            previous_types = {r.memory_content.get('type', 'unknown') for r in previous_results}
            type_bonus = 0.1 if memory_type not in previous_types else 0.0
            
            # Check for agent diversity
            previous_agents = {r.memory_content.get('agent_id', 'unknown') for r in previous_results}
            agent_bonus = 0.05 if memory_agent not in previous_agents else 0.0
            
            return type_bonus + agent_bonus
            
        except Exception as e:
            self.logger.error(f"Failed to calculate diversity bonus: {e}")
            return 0.0
    
    async def _update_adaptive_parameters(self, context: RetrievalContext, results: List[RetrievalResult]) -> None:
        """Update adaptive parameters based on retrieval results"""
        try:
            if not self.config.adaptive_scoring_enabled:
                return
            
            agent_id = context.agent_id
            
            # Analyze result quality (simplified)
            if results:
                avg_score = np.mean([r.relevance_score.total_score for r in results])
                
                # Adjust weights based on performance
                if agent_id not in self.agent_preferences:
                    self.agent_preferences[agent_id] = {
                        'semantic': self.config.semantic_weight,
                        'temporal': self.config.temporal_weight,
                        'contextual': self.config.contextual_weight,
                        'social': self.config.social_weight,
                        'recency': self.config.recency_weight,
                        'success': self.config.success_weight
                    }
                
                # Simple adaptive adjustment (could be more sophisticated)
                if avg_score > 0.8:
                    # Good performance - slight reinforcement
                    for component in ['semantic', 'contextual', 'temporal']:
                        top_result = results[0]
                        component_score = getattr(top_result.relevance_score, f"{component}_score")
                        if component_score > 0.8:
                            self.agent_preferences[agent_id][component] *= 1.05
                elif avg_score < 0.4:
                    # Poor performance - adjust weights
                    self.agent_preferences[agent_id]['contextual'] *= 1.1
                    self.agent_preferences[agent_id]['semantic'] *= 0.95
                
                # Normalize weights
                total_weight = sum(self.agent_preferences[agent_id].values())
                for key in self.agent_preferences[agent_id]:
                    self.agent_preferences[agent_id][key] /= total_weight
                
                self.stats['adaptive_adjustments'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to update adaptive parameters: {e}")
    
    async def _generate_retrieval_reason(self, relevance_score: RelevanceScore) -> str:
        """Generate human-readable reason for retrieval"""
        try:
            reasons = []
            
            # Identify top scoring components
            breakdown = relevance_score.scoring_breakdown
            top_components = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:2]
            
            for component, score in top_components:
                if score > 0.7:
                    if component == 'semantic':
                        reasons.append("high semantic similarity")
                    elif component == 'temporal':
                        reasons.append("temporal relevance")
                    elif component == 'contextual':
                        reasons.append("contextual match")
                    elif component == 'social':
                        reasons.append("social connection")
                    elif component == 'recency':
                        reasons.append("recent activity")
                    elif component == 'success':
                        reasons.append("success pattern")
            
            if not reasons:
                reasons.append("general relevance")
            
            return f"Retrieved due to {', '.join(reasons)}"
            
        except Exception as e:
            self.logger.error(f"Failed to generate retrieval reason: {e}")
            return "Retrieved based on relevance scoring"
    
    async def _analyze_context_match(self, memory: Dict[str, Any], context: RetrievalContext) -> Dict[str, Any]:
        """Analyze how memory matches the retrieval context"""
        try:
            match_analysis = {
                'task_match': False,
                'agent_match': False,
                'temporal_match': False,
                'scenario_match': False,
                'keyword_matches': []
            }
            
            # Task match
            if context.current_task and memory.get('task_context'):
                match_analysis['task_match'] = context.current_task.lower() in memory.get('task_context', '').lower()
            
            # Agent match
            if memory.get('agent_id') == context.agent_id:
                match_analysis['agent_match'] = True
            
            # Temporal match
            if context.temporal_window and memory.get('timestamp'):
                memory_time = memory.get('timestamp')
                if isinstance(memory_time, str):
                    memory_time = datetime.fromisoformat(memory_time)
                time_diff = datetime.now() - memory_time
                match_analysis['temporal_match'] = time_diff <= context.temporal_window
            
            # Keyword matches
            if context.priority_keywords:
                memory_text = str(memory.get('content', ''))
                for keyword in context.priority_keywords:
                    if keyword.lower() in memory_text.lower():
                        match_analysis['keyword_matches'].append(keyword)
            
            return match_analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze context match: {e}")
            return {}
    
    async def _initialize_scoring_models(self) -> None:
        """Initialize scoring models and parameters"""
        try:
            # Initialize default scoring models (placeholder for ML models)
            self.scoring_models = {
                'semantic_model': None,  # Could be a trained embedding model
                'temporal_model': None,  # Could be a time-series model
                'contextual_model': None  # Could be a context classification model
            }
            
            self.logger.debug("Initialized scoring models")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize scoring models: {e}")
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        try:
            # Calculate agent-specific statistics
            agent_stats = {}
            for agent_id, history in self.retrieval_history.items():
                if history:
                    avg_score = np.mean([r.relevance_score.total_score for r in history])
                    agent_stats[agent_id] = {
                        'total_retrievals': len(history),
                        'average_score': avg_score,
                        'recent_retrievals': len([r for r in history if (datetime.now() - r.retrieved_at).days <= 1])
                    }
            
            return {
                'total_agents': len(self.retrieval_history),
                'total_contexts_cached': len(self.context_cache),
                'adaptive_agents': len(self.agent_preferences),
                'config': {
                    'strategy': self.config.strategy.value,
                    'max_results': self.config.max_results,
                    'min_threshold': self.config.min_relevance_threshold,
                    'diversity_factor': self.config.diversity_factor
                },
                'agent_statistics': agent_stats,
                'processing_stats': self.stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get retrieval statistics: {e}")
            return {'error': str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown the context-aware retrieval system"""
        try:
            self.logger.info("Shutting down context-aware retrieval system")
            
            # Clear data structures
            self.retrieval_history.clear()
            self.context_cache.clear()
            self.scoring_models.clear()
            self.agent_preferences.clear()
            self.context_patterns.clear()
            
            self.initialized = False
            self.logger.info("Context-aware retrieval system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during context-aware retrieval shutdown: {e}")

# Factory function
def create_context_aware_retrieval(config: RetrievalConfig = None) -> ContextAwareRetrieval:
    """Create a context-aware retrieval system instance"""
    return ContextAwareRetrieval(config or RetrievalConfig())