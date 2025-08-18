#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Knowledge Distillation Pipeline
Knowledge distillation with irrelevant behavior pruning for agent learning
"""

import asyncio
import logging
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict

from .base_agent import Experience, Team, Role, ActionResult
from .learning_system import PerformanceMetrics, StrategyUpdate
from .human_in_the_loop import HumanFeedback, FeedbackType

logger = logging.getLogger(__name__)

class DistillationType(Enum):
    """Types of knowledge distillation"""
    BEHAVIOR_PRUNING = "behavior_pruning"
    STRATEGY_COMPRESSION = "strategy_compression"
    EXPERIENCE_FILTERING = "experience_filtering"
    PATTERN_EXTRACTION = "pattern_extraction"
    RELEVANCE_SCORING = "relevance_scoring"

class RelevanceLevel(Enum):
    """Relevance levels for behavior classification"""
    CRITICAL = "critical"      # Essential behaviors
    IMPORTANT = "important"    # Useful behaviors
    NEUTRAL = "neutral"        # Neither helpful nor harmful
    IRRELEVANT = "irrelevant"  # Unnecessary behaviors
    HARMFUL = "harmful"        # Counterproductive behaviors

@dataclass
class BehaviorPattern:
    """Identified behavior pattern for distillation"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    success_rate: float
    relevance_level: RelevanceLevel
    context_conditions: List[str]
    action_sequence: List[str]
    outcomes: List[str]
    confidence_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DistillationRule:
    """Rule for knowledge distillation and behavior pruning"""
    rule_id: str
    rule_type: DistillationType
    condition: str
    action: str
    priority: int
    confidence_threshold: float
    success_rate_threshold: float
    frequency_threshold: int
    created_by: str
    timestamp: datetime
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DistillationResult:
    """Result of knowledge distillation process"""
    distillation_id: str
    agent_id: str
    distillation_type: DistillationType
    input_experiences: int
    output_experiences: int
    pruned_behaviors: List[str]
    extracted_patterns: List[BehaviorPattern]
    compression_ratio: float
    quality_score: float
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class BehaviorAnalyzer:
    """Analyzes agent behaviors to identify patterns and relevance"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.relevance_cache = {}
        self.logger = logging.getLogger(__name__)
    
    async def analyze_behavior_patterns(self, experiences: List[Experience]) -> List[BehaviorPattern]:
        """Analyze experiences to identify behavior patterns"""
        try:
            patterns = []
            
            # Group experiences by action type and context
            action_groups = defaultdict(list)
            for exp in experiences:
                if hasattr(exp, 'action_taken') and exp.action_taken:
                    action_key = self._get_action_key(exp)
                    action_groups[action_key].append(exp)
            
            # Analyze each group for patterns
            for action_key, group_experiences in action_groups.items():
                if len(group_experiences) >= 3:  # Minimum frequency for pattern
                    pattern = await self._extract_pattern(action_key, group_experiences)
                    if pattern:
                        patterns.append(pattern)
            
            # Sort patterns by relevance and frequency
            patterns.sort(key=lambda p: (p.relevance_level.value, -p.frequency, -p.success_rate))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to analyze behavior patterns: {e}")
            return []
    
    def _get_action_key(self, experience: Experience) -> str:
        """Generate a key for grouping similar actions"""
        action_str = str(experience.action_taken) if experience.action_taken else "unknown"
        context_str = ""
        
        if hasattr(experience, 'context') and experience.context:
            # Extract key context elements
            context_elements = []
            if hasattr(experience.context, 'phase'):
                context_elements.append(f"phase:{experience.context.phase}")
            if hasattr(experience.context, 'target_type'):
                context_elements.append(f"target:{experience.context.target_type}")
            context_str = "|".join(context_elements)
        
        return f"{action_str}#{context_str}"
    
    async def _extract_pattern(self, action_key: str, experiences: List[Experience]) -> Optional[BehaviorPattern]:
        """Extract a behavior pattern from grouped experiences"""
        try:
            if not experiences:
                return None
            
            # Calculate pattern statistics
            frequency = len(experiences)
            success_count = sum(1 for exp in experiences if exp.success)
            success_rate = success_count / frequency if frequency > 0 else 0.0
            
            # Extract action sequence
            action_sequence = []
            for exp in experiences[:5]:  # Sample first 5
                if exp.action_taken:
                    action_sequence.append(str(exp.action_taken))
            
            # Extract outcomes
            outcomes = []
            for exp in experiences:
                if hasattr(exp, 'outcome') and exp.outcome:
                    outcomes.append(str(exp.outcome))
            
            # Determine relevance level
            relevance_level = self._assess_relevance(success_rate, frequency, experiences)
            
            # Extract context conditions
            context_conditions = self._extract_context_conditions(experiences)
            
            # Calculate confidence score
            confidence_score = self._calculate_pattern_confidence(
                frequency, success_rate, len(set(action_sequence))
            )
            
            pattern = BehaviorPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type=action_key.split('#')[0],
                description=f"Pattern for {action_key} with {frequency} occurrences",
                frequency=frequency,
                success_rate=success_rate,
                relevance_level=relevance_level,
                context_conditions=context_conditions,
                action_sequence=list(set(action_sequence)),  # Unique actions
                outcomes=list(set(outcomes)),  # Unique outcomes
                confidence_score=confidence_score,
                timestamp=datetime.now()
            )
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Failed to extract pattern: {e}")
            return None
    
    def _assess_relevance(self, success_rate: float, frequency: int, experiences: List[Experience]) -> RelevanceLevel:
        """Assess the relevance level of a behavior pattern"""
        # High success rate and frequency = critical or important
        if success_rate >= 0.8 and frequency >= 10:
            return RelevanceLevel.CRITICAL
        elif success_rate >= 0.6 and frequency >= 5:
            return RelevanceLevel.IMPORTANT
        elif success_rate >= 0.4:
            return RelevanceLevel.NEUTRAL
        elif success_rate >= 0.2:
            return RelevanceLevel.IRRELEVANT
        else:
            return RelevanceLevel.HARMFUL
    
    def _extract_context_conditions(self, experiences: List[Experience]) -> List[str]:
        """Extract common context conditions from experiences"""
        conditions = []
        
        # Analyze context patterns
        context_elements = defaultdict(int)
        for exp in experiences:
            if hasattr(exp, 'context') and exp.context:
                # Extract context attributes
                for attr, value in vars(exp.context).items():
                    if value is not None:
                        context_elements[f"{attr}:{value}"] += 1
        
        # Keep conditions that appear in majority of experiences
        threshold = len(experiences) * 0.6
        for condition, count in context_elements.items():
            if count >= threshold:
                conditions.append(condition)
        
        return conditions
    
    def _calculate_pattern_confidence(self, frequency: int, success_rate: float, diversity: int) -> float:
        """Calculate confidence score for a pattern"""
        # Normalize frequency (log scale)
        freq_score = min(1.0, np.log(frequency + 1) / np.log(100))
        
        # Success rate score
        success_score = success_rate
        
        # Diversity penalty (too much diversity reduces confidence)
        diversity_score = max(0.1, 1.0 - (diversity - 1) * 0.1)
        
        # Weighted combination
        confidence = (freq_score * 0.3 + success_score * 0.5 + diversity_score * 0.2)
        
        return min(1.0, max(0.0, confidence))

class KnowledgeDistillationPipeline:
    """
    Pipeline for distilling agent knowledge and pruning irrelevant behaviors
    """
    
    def __init__(self):
        self.behavior_analyzer = BehaviorAnalyzer()
        self.distillation_rules = {}  # rule_id -> DistillationRule
        self.distillation_history = {}  # agent_id -> List[DistillationResult]
        
        # Configuration
        self.min_experience_count = 50
        self.relevance_threshold = RelevanceLevel.NEUTRAL
        self.compression_target = 0.7  # Target 70% compression
        self.quality_threshold = 0.6
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the knowledge distillation pipeline"""
        try:
            self.logger.info("Initializing knowledge distillation pipeline")
            
            # Load default distillation rules
            await self._load_default_rules()
            
            self.initialized = True
            self.logger.info("Knowledge distillation pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge distillation pipeline: {e}")
            raise
    
    async def distill_agent_knowledge(self, 
                                    agent_id: str,
                                    experiences: List[Experience],
                                    distillation_type: DistillationType = DistillationType.BEHAVIOR_PRUNING,
                                    human_feedback: List[HumanFeedback] = None) -> DistillationResult:
        """
        Distill knowledge from agent experiences
        
        Args:
            agent_id: ID of the agent
            experiences: List of agent experiences
            distillation_type: Type of distillation to perform
            human_feedback: Optional human feedback to guide distillation
            
        Returns:
            DistillationResult: Results of the distillation process
        """
        try:
            start_time = datetime.now()
            distillation_id = str(uuid.uuid4())
            
            self.logger.info(f"Starting {distillation_type.value} for agent {agent_id} with {len(experiences)} experiences")
            
            # Analyze behavior patterns
            patterns = await self.behavior_analyzer.analyze_behavior_patterns(experiences)
            
            # Apply distillation based on type
            if distillation_type == DistillationType.BEHAVIOR_PRUNING:
                result = await self._prune_irrelevant_behaviors(
                    agent_id, experiences, patterns, human_feedback
                )
            elif distillation_type == DistillationType.STRATEGY_COMPRESSION:
                result = await self._compress_strategies(
                    agent_id, experiences, patterns, human_feedback
                )
            elif distillation_type == DistillationType.EXPERIENCE_FILTERING:
                result = await self._filter_experiences(
                    agent_id, experiences, patterns, human_feedback
                )
            elif distillation_type == DistillationType.PATTERN_EXTRACTION:
                result = await self._extract_key_patterns(
                    agent_id, experiences, patterns, human_feedback
                )
            else:
                result = await self._score_relevance(
                    agent_id, experiences, patterns, human_feedback
                )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            result.distillation_id = distillation_id
            result.timestamp = datetime.now()
            
            # Store result
            if agent_id not in self.distillation_history:
                self.distillation_history[agent_id] = []
            self.distillation_history[agent_id].append(result)
            
            self.logger.info(f"Completed distillation for agent {agent_id}: {result.compression_ratio:.2f} compression ratio")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to distill agent knowledge: {e}")
            raise
    
    async def _prune_irrelevant_behaviors(self, 
                                        agent_id: str,
                                        experiences: List[Experience],
                                        patterns: List[BehaviorPattern],
                                        human_feedback: List[HumanFeedback] = None) -> DistillationResult:
        """Prune irrelevant and harmful behaviors"""
        try:
            pruned_behaviors = []
            filtered_experiences = []
            
            # Create relevance map from patterns
            relevance_map = {}
            for pattern in patterns:
                for action in pattern.action_sequence:
                    relevance_map[action] = pattern.relevance_level
            
            # Incorporate human feedback
            if human_feedback:
                for feedback in human_feedback:
                    if feedback.feedback_type == FeedbackType.BEHAVIOR_TAGGING:
                        if feedback.correctness_score is not None and feedback.correctness_score < 0.3:
                            # Mark low-scored behaviors as harmful
                            if feedback.action_id:
                                relevance_map[feedback.action_id] = RelevanceLevel.HARMFUL
            
            # Filter experiences based on relevance
            for exp in experiences:
                action_str = str(exp.action_taken) if exp.action_taken else "unknown"
                relevance = relevance_map.get(action_str, RelevanceLevel.NEUTRAL)
                
                # Keep critical, important, and neutral behaviors
                if relevance in [RelevanceLevel.CRITICAL, RelevanceLevel.IMPORTANT, RelevanceLevel.NEUTRAL]:
                    filtered_experiences.append(exp)
                else:
                    pruned_behaviors.append(action_str)
            
            # Calculate metrics
            compression_ratio = len(filtered_experiences) / len(experiences) if experiences else 1.0
            quality_score = self._calculate_quality_score(filtered_experiences, patterns)
            
            # Extract relevant patterns
            relevant_patterns = [p for p in patterns if p.relevance_level != RelevanceLevel.HARMFUL]
            
            result = DistillationResult(
                distillation_id="",  # Will be set by caller
                agent_id=agent_id,
                distillation_type=DistillationType.BEHAVIOR_PRUNING,
                input_experiences=len(experiences),
                output_experiences=len(filtered_experiences),
                pruned_behaviors=list(set(pruned_behaviors)),
                extracted_patterns=relevant_patterns,
                compression_ratio=compression_ratio,
                quality_score=quality_score,
                processing_time=0.0,  # Will be set by caller
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to prune irrelevant behaviors: {e}")
            raise
    
    async def _compress_strategies(self, 
                                 agent_id: str,
                                 experiences: List[Experience],
                                 patterns: List[BehaviorPattern],
                                 human_feedback: List[HumanFeedback] = None) -> DistillationResult:
        """Compress strategies by removing redundant patterns"""
        try:
            # Group similar patterns
            pattern_groups = self._group_similar_patterns(patterns)
            
            # Select best pattern from each group
            compressed_patterns = []
            for group in pattern_groups:
                # Sort by success rate and frequency
                group.sort(key=lambda p: (p.success_rate, p.frequency), reverse=True)
                compressed_patterns.append(group[0])  # Keep the best
            
            # Filter experiences based on compressed patterns
            filtered_experiences = []
            pattern_actions = set()
            for pattern in compressed_patterns:
                pattern_actions.update(pattern.action_sequence)
            
            for exp in experiences:
                action_str = str(exp.action_taken) if exp.action_taken else "unknown"
                if action_str in pattern_actions or exp.success:  # Keep successful experiences
                    filtered_experiences.append(exp)
            
            # Calculate metrics
            compression_ratio = len(compressed_patterns) / len(patterns) if patterns else 1.0
            quality_score = self._calculate_quality_score(filtered_experiences, compressed_patterns)
            
            result = DistillationResult(
                distillation_id="",
                agent_id=agent_id,
                distillation_type=DistillationType.STRATEGY_COMPRESSION,
                input_experiences=len(experiences),
                output_experiences=len(filtered_experiences),
                pruned_behaviors=[],
                extracted_patterns=compressed_patterns,
                compression_ratio=compression_ratio,
                quality_score=quality_score,
                processing_time=0.0,
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to compress strategies: {e}")
            raise
    
    async def _filter_experiences(self, 
                                agent_id: str,
                                experiences: List[Experience],
                                patterns: List[BehaviorPattern],
                                human_feedback: List[HumanFeedback] = None) -> DistillationResult:
        """Filter experiences to keep only the most relevant ones"""
        try:
            # Score experiences based on multiple criteria
            scored_experiences = []
            
            for exp in experiences:
                score = self._score_experience_relevance(exp, patterns, human_feedback)
                scored_experiences.append((exp, score))
            
            # Sort by score and keep top experiences
            scored_experiences.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top percentage based on compression target
            keep_count = int(len(experiences) * self.compression_target)
            filtered_experiences = [exp for exp, score in scored_experiences[:keep_count]]
            
            # Calculate metrics
            compression_ratio = len(filtered_experiences) / len(experiences) if experiences else 1.0
            quality_score = self._calculate_quality_score(filtered_experiences, patterns)
            
            result = DistillationResult(
                distillation_id="",
                agent_id=agent_id,
                distillation_type=DistillationType.EXPERIENCE_FILTERING,
                input_experiences=len(experiences),
                output_experiences=len(filtered_experiences),
                pruned_behaviors=[],
                extracted_patterns=patterns,
                compression_ratio=compression_ratio,
                quality_score=quality_score,
                processing_time=0.0,
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to filter experiences: {e}")
            raise
    
    async def _extract_key_patterns(self, 
                                  agent_id: str,
                                  experiences: List[Experience],
                                  patterns: List[BehaviorPattern],
                                  human_feedback: List[HumanFeedback] = None) -> DistillationResult:
        """Extract only the most important patterns"""
        try:
            # Filter patterns by relevance and quality
            key_patterns = []
            
            for pattern in patterns:
                # Keep critical and important patterns
                if pattern.relevance_level in [RelevanceLevel.CRITICAL, RelevanceLevel.IMPORTANT]:
                    key_patterns.append(pattern)
                # Keep high-performing neutral patterns
                elif (pattern.relevance_level == RelevanceLevel.NEUTRAL and 
                      pattern.success_rate >= 0.7 and 
                      pattern.frequency >= 5):
                    key_patterns.append(pattern)
            
            # Incorporate human feedback
            if human_feedback:
                human_approved_patterns = set()
                for feedback in human_feedback:
                    if (feedback.feedback_type == FeedbackType.BEHAVIOR_TAGGING and
                        feedback.correctness_score is not None and
                        feedback.correctness_score >= 0.7):
                        human_approved_patterns.add(feedback.action_id)
                
                # Add patterns that match human-approved actions
                for pattern in patterns:
                    if any(action in human_approved_patterns for action in pattern.action_sequence):
                        if pattern not in key_patterns:
                            key_patterns.append(pattern)
            
            # Calculate metrics
            compression_ratio = len(key_patterns) / len(patterns) if patterns else 1.0
            quality_score = np.mean([p.success_rate for p in key_patterns]) if key_patterns else 0.0
            
            result = DistillationResult(
                distillation_id="",
                agent_id=agent_id,
                distillation_type=DistillationType.PATTERN_EXTRACTION,
                input_experiences=len(experiences),
                output_experiences=len(experiences),  # No experience filtering
                pruned_behaviors=[],
                extracted_patterns=key_patterns,
                compression_ratio=compression_ratio,
                quality_score=quality_score,
                processing_time=0.0,
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to extract key patterns: {e}")
            raise
    
    async def _score_relevance(self, 
                             agent_id: str,
                             experiences: List[Experience],
                             patterns: List[BehaviorPattern],
                             human_feedback: List[HumanFeedback] = None) -> DistillationResult:
        """Score the relevance of all behaviors and patterns"""
        try:
            # Score each pattern
            for pattern in patterns:
                # Base score from success rate and frequency
                base_score = pattern.success_rate * 0.7 + min(1.0, pattern.frequency / 20) * 0.3
                
                # Adjust based on relevance level
                relevance_multiplier = {
                    RelevanceLevel.CRITICAL: 1.2,
                    RelevanceLevel.IMPORTANT: 1.0,
                    RelevanceLevel.NEUTRAL: 0.8,
                    RelevanceLevel.IRRELEVANT: 0.4,
                    RelevanceLevel.HARMFUL: 0.1
                }
                
                pattern.confidence_score = base_score * relevance_multiplier[pattern.relevance_level]
            
            # Incorporate human feedback scores
            if human_feedback:
                feedback_scores = {}
                for feedback in human_feedback:
                    if feedback.correctness_score is not None:
                        feedback_scores[feedback.action_id] = feedback.correctness_score
                
                # Adjust pattern scores based on human feedback
                for pattern in patterns:
                    for action in pattern.action_sequence:
                        if action in feedback_scores:
                            human_score = feedback_scores[action]
                            # Blend with existing score
                            pattern.confidence_score = (pattern.confidence_score * 0.7 + human_score * 0.3)
            
            # Calculate overall quality
            quality_score = np.mean([p.confidence_score for p in patterns]) if patterns else 0.0
            
            result = DistillationResult(
                distillation_id="",
                agent_id=agent_id,
                distillation_type=DistillationType.RELEVANCE_SCORING,
                input_experiences=len(experiences),
                output_experiences=len(experiences),
                pruned_behaviors=[],
                extracted_patterns=patterns,
                compression_ratio=1.0,  # No compression in scoring
                quality_score=quality_score,
                processing_time=0.0,
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to score relevance: {e}")
            raise
    
    def _group_similar_patterns(self, patterns: List[BehaviorPattern]) -> List[List[BehaviorPattern]]:
        """Group similar patterns together"""
        groups = []
        used_patterns = set()
        
        for i, pattern1 in enumerate(patterns):
            if i in used_patterns:
                continue
            
            group = [pattern1]
            used_patterns.add(i)
            
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                if j in used_patterns:
                    continue
                
                # Check similarity
                if self._patterns_similar(pattern1, pattern2):
                    group.append(pattern2)
                    used_patterns.add(j)
            
            groups.append(group)
        
        return groups
    
    def _patterns_similar(self, pattern1: BehaviorPattern, pattern2: BehaviorPattern) -> bool:
        """Check if two patterns are similar"""
        # Same pattern type
        if pattern1.pattern_type != pattern2.pattern_type:
            return False
        
        # Similar action sequences
        actions1 = set(pattern1.action_sequence)
        actions2 = set(pattern2.action_sequence)
        
        if not actions1 or not actions2:
            return False
        
        overlap = len(actions1 & actions2)
        union = len(actions1 | actions2)
        
        similarity = overlap / union if union > 0 else 0.0
        
        return similarity >= 0.7  # 70% similarity threshold
    
    def _score_experience_relevance(self, 
                                  experience: Experience,
                                  patterns: List[BehaviorPattern],
                                  human_feedback: List[HumanFeedback] = None) -> float:
        """Score the relevance of an individual experience"""
        score = 0.0
        
        # Base score from success
        if experience.success:
            score += 0.5
        
        # Score from pattern matching
        action_str = str(experience.action_taken) if experience.action_taken else "unknown"
        
        for pattern in patterns:
            if action_str in pattern.action_sequence:
                # Add score based on pattern relevance and success rate
                pattern_score = pattern.success_rate * 0.3
                
                if pattern.relevance_level == RelevanceLevel.CRITICAL:
                    pattern_score *= 1.5
                elif pattern.relevance_level == RelevanceLevel.IMPORTANT:
                    pattern_score *= 1.2
                elif pattern.relevance_level == RelevanceLevel.HARMFUL:
                    pattern_score *= 0.1
                
                score += pattern_score
                break  # Only count first matching pattern
        
        # Score from human feedback
        if human_feedback:
            for feedback in human_feedback:
                if (feedback.action_id == experience.experience_id and
                    feedback.correctness_score is not None):
                    score += feedback.correctness_score * 0.4
        
        # Score from confidence
        if hasattr(experience, 'confidence_score') and experience.confidence_score:
            score += experience.confidence_score * 0.2
        
        return min(1.0, score)
    
    def _calculate_quality_score(self, 
                               experiences: List[Experience],
                               patterns: List[BehaviorPattern]) -> float:
        """Calculate quality score for distilled knowledge"""
        if not experiences and not patterns:
            return 0.0
        
        quality_components = []
        
        # Experience quality
        if experiences:
            success_rate = sum(1 for exp in experiences if exp.success) / len(experiences)
            quality_components.append(success_rate)
        
        # Pattern quality
        if patterns:
            avg_pattern_success = np.mean([p.success_rate for p in patterns])
            avg_pattern_confidence = np.mean([p.confidence_score for p in patterns])
            quality_components.extend([avg_pattern_success, avg_pattern_confidence])
        
        return np.mean(quality_components) if quality_components else 0.0
    
    async def _load_default_rules(self) -> None:
        """Load default distillation rules"""
        try:
            default_rules = [
                DistillationRule(
                    rule_id="prune_low_success",
                    rule_type=DistillationType.BEHAVIOR_PRUNING,
                    condition="success_rate < 0.2 AND frequency >= 5",
                    action="mark_as_harmful",
                    priority=1,
                    confidence_threshold=0.8,
                    success_rate_threshold=0.2,
                    frequency_threshold=5,
                    created_by="system",
                    timestamp=datetime.now()
                ),
                DistillationRule(
                    rule_id="keep_high_success",
                    rule_type=DistillationType.PATTERN_EXTRACTION,
                    condition="success_rate >= 0.8 AND frequency >= 3",
                    action="mark_as_critical",
                    priority=1,
                    confidence_threshold=0.9,
                    success_rate_threshold=0.8,
                    frequency_threshold=3,
                    created_by="system",
                    timestamp=datetime.now()
                ),
                DistillationRule(
                    rule_id="compress_similar",
                    rule_type=DistillationType.STRATEGY_COMPRESSION,
                    condition="similarity >= 0.7",
                    action="merge_patterns",
                    priority=2,
                    confidence_threshold=0.7,
                    success_rate_threshold=0.5,
                    frequency_threshold=2,
                    created_by="system",
                    timestamp=datetime.now()
                )
            ]
            
            for rule in default_rules:
                self.distillation_rules[rule.rule_id] = rule
            
            self.logger.info(f"Loaded {len(default_rules)} default distillation rules")
            
        except Exception as e:
            self.logger.error(f"Failed to load default rules: {e}")
    
    # Query methods
    async def get_distillation_history(self, agent_id: str) -> List[DistillationResult]:
        """Get distillation history for an agent"""
        return self.distillation_history.get(agent_id, [])
    
    async def get_behavior_patterns(self, agent_id: str) -> List[BehaviorPattern]:
        """Get extracted behavior patterns for an agent"""
        patterns = []
        
        if agent_id in self.distillation_history:
            for result in self.distillation_history[agent_id]:
                patterns.extend(result.extracted_patterns)
        
        return patterns
    
    async def add_distillation_rule(self, rule: DistillationRule) -> None:
        """Add a new distillation rule"""
        self.distillation_rules[rule.rule_id] = rule
        self.logger.info(f"Added distillation rule: {rule.rule_id}")
    
    async def remove_distillation_rule(self, rule_id: str) -> bool:
        """Remove a distillation rule"""
        if rule_id in self.distillation_rules:
            del self.distillation_rules[rule_id]
            self.logger.info(f"Removed distillation rule: {rule_id}")
            return True
        return False
    
    async def get_distillation_statistics(self) -> Dict[str, Any]:
        """Get statistics about knowledge distillation"""
        total_distillations = sum(len(history) for history in self.distillation_history.values())
        
        if total_distillations == 0:
            return {
                "total_distillations": 0,
                "average_compression_ratio": 0.0,
                "average_quality_score": 0.0,
                "active_agents": 0,
                "total_patterns": 0,
                "active_rules": len([r for r in self.distillation_rules.values() if r.active])
            }
        
        all_results = []
        for history in self.distillation_history.values():
            all_results.extend(history)
        
        avg_compression = np.mean([r.compression_ratio for r in all_results])
        avg_quality = np.mean([r.quality_score for r in all_results])
        total_patterns = sum(len(r.extracted_patterns) for r in all_results)
        
        return {
            "total_distillations": total_distillations,
            "average_compression_ratio": avg_compression,
            "average_quality_score": avg_quality,
            "active_agents": len(self.distillation_history),
            "total_patterns": total_patterns,
            "active_rules": len([r for r in self.distillation_rules.values() if r.active])
        }

# Factory function
def create_knowledge_distillation_pipeline() -> KnowledgeDistillationPipeline:
    """Create and return a knowledge distillation pipeline"""
    return KnowledgeDistillationPipeline()