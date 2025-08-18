#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Semantic Annotation System
Advanced semantic annotation for memory categorization and enrichment
"""

import asyncio
import logging
import json
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import uuid
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class AnnotationType(Enum):
    """Types of semantic annotations"""
    ENTITY = "entity"
    CONCEPT = "concept"
    RELATIONSHIP = "relationship"
    SENTIMENT = "sentiment"
    INTENT = "intent"
    CONTEXT = "context"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"

class EntityType(Enum):
    """Entity types for annotation"""
    AGENT = "agent"
    TARGET = "target"
    TOOL = "tool"
    TECHNIQUE = "technique"
    VULNERABILITY = "vulnerability"
    ASSET = "asset"
    CREDENTIAL = "credential"
    NETWORK = "network"
    FILE = "file"
    PROCESS = "process"

class ConceptType(Enum):
    """Concept types for categorization"""
    ATTACK_PHASE = "attack_phase"
    DEFENSE_STRATEGY = "defense_strategy"
    SUCCESS_PATTERN = "success_pattern"
    FAILURE_PATTERN = "failure_pattern"
    LEARNING_OUTCOME = "learning_outcome"
    TACTICAL_KNOWLEDGE = "tactical_knowledge"
    OPERATIONAL_CONTEXT = "operational_context"

@dataclass
class SemanticAnnotation:
    """Semantic annotation for memory content"""
    annotation_id: str
    annotation_type: AnnotationType
    entity_type: Optional[EntityType] = None
    concept_type: Optional[ConceptType] = None
    text_span: Tuple[int, int] = (0, 0)  # Start and end positions
    content: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class AnnotatedMemory:
    """Memory with semantic annotations"""
    memory_id: str
    original_content: str
    annotations: List[SemanticAnnotation]
    categories: List[str]
    semantic_tags: Set[str]
    confidence_score: float
    processing_timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class AnnotationRule:
    """Rule for automatic annotation"""
    rule_id: str
    name: str
    pattern: str  # Regex pattern
    annotation_type: AnnotationType
    entity_type: Optional[EntityType] = None
    concept_type: Optional[ConceptType] = None
    confidence: float = 0.8
    active: bool = True
    metadata: Dict[str, Any] = None

class SemanticAnnotationEngine:
    """
    Advanced semantic annotation engine for memory categorization.
    
    Features:
    - Multi-type annotation (entities, concepts, relationships)
    - Pattern-based automatic annotation
    - Context-aware categorization
    - Confidence scoring and validation
    - Semantic tag generation
    - Memory enrichment and indexing
    """
    
    def __init__(self):
        self.annotation_rules: Dict[str, AnnotationRule] = {}
        self.annotated_memories: Dict[str, AnnotatedMemory] = {}
        self.category_taxonomy: Dict[str, List[str]] = {}
        self.semantic_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.stats = {
            'memories_annotated': 0,
            'annotations_created': 0,
            'categories_assigned': 0,
            'rules_applied': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the semantic annotation engine"""
        try:
            self.logger.info("Initializing semantic annotation engine")
            
            # Initialize default annotation rules
            await self._initialize_default_rules()
            
            # Initialize category taxonomy
            await self._initialize_category_taxonomy()
            
            self.initialized = True
            self.logger.info("Semantic annotation engine initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize semantic annotation engine: {e}")
            raise
    
    async def annotate_memory(self, memory_id: str, content: str, context: Dict[str, Any] = None) -> AnnotatedMemory:
        """
        Annotate memory content with semantic information
        
        Args:
            memory_id: Unique memory identifier
            content: Memory content to annotate
            context: Additional context for annotation
            
        Returns:
            AnnotatedMemory: Annotated memory object
        """
        try:
            context = context or {}
            
            # Extract annotations
            annotations = await self._extract_annotations(content, context)
            
            # Generate categories
            categories = await self._generate_categories(content, annotations, context)
            
            # Generate semantic tags
            semantic_tags = await self._generate_semantic_tags(content, annotations, categories)
            
            # Calculate overall confidence
            confidence_score = await self._calculate_annotation_confidence(annotations)
            
            # Create annotated memory
            annotated_memory = AnnotatedMemory(
                memory_id=memory_id,
                original_content=content,
                annotations=annotations,
                categories=categories,
                semantic_tags=semantic_tags,
                confidence_score=confidence_score,
                processing_timestamp=datetime.now(),
                metadata={
                    'context': context,
                    'annotation_count': len(annotations),
                    'category_count': len(categories),
                    'tag_count': len(semantic_tags)
                }
            )
            
            # Store annotated memory
            self.annotated_memories[memory_id] = annotated_memory
            
            # Update semantic index
            await self._update_semantic_index(memory_id, annotated_memory)
            
            # Update statistics
            self.stats['memories_annotated'] += 1
            self.stats['annotations_created'] += len(annotations)
            self.stats['categories_assigned'] += len(categories)
            
            self.logger.debug(f"Annotated memory {memory_id} with {len(annotations)} annotations")
            return annotated_memory
            
        except Exception as e:
            self.logger.error(f"Failed to annotate memory: {e}")
            raise
    
    async def _extract_annotations(self, content: str, context: Dict[str, Any]) -> List[SemanticAnnotation]:
        """Extract semantic annotations from content"""
        try:
            annotations = []
            
            # Apply annotation rules
            for rule in self.annotation_rules.values():
                if not rule.active:
                    continue
                
                rule_annotations = await self._apply_annotation_rule(content, rule, context)
                annotations.extend(rule_annotations)
                
                if rule_annotations:
                    self.stats['rules_applied'] += 1
            
            # Extract entities
            entity_annotations = await self._extract_entities(content, context)
            annotations.extend(entity_annotations)
            
            # Extract concepts
            concept_annotations = await self._extract_concepts(content, context)
            annotations.extend(concept_annotations)
            
            # Extract relationships
            relationship_annotations = await self._extract_relationships(content, annotations, context)
            annotations.extend(relationship_annotations)
            
            # Remove duplicates and merge overlapping annotations
            annotations = await self._merge_overlapping_annotations(annotations)
            
            return annotations
            
        except Exception as e:
            self.logger.error(f"Failed to extract annotations: {e}")
            return []
    
    async def _apply_annotation_rule(self, content: str, rule: AnnotationRule, context: Dict[str, Any]) -> List[SemanticAnnotation]:
        """Apply a single annotation rule to content"""
        try:
            annotations = []
            
            # Find pattern matches
            matches = re.finditer(rule.pattern, content, re.IGNORECASE)
            
            for match in matches:
                annotation = SemanticAnnotation(
                    annotation_id=str(uuid.uuid4()),
                    annotation_type=rule.annotation_type,
                    entity_type=rule.entity_type,
                    concept_type=rule.concept_type,
                    text_span=(match.start(), match.end()),
                    content=match.group(),
                    confidence=rule.confidence,
                    metadata={
                        'rule_id': rule.rule_id,
                        'rule_name': rule.name,
                        'pattern': rule.pattern,
                        'context': context
                    }
                )
                annotations.append(annotation)
            
            return annotations
            
        except Exception as e:
            self.logger.error(f"Failed to apply annotation rule {rule.rule_id}: {e}")
            return []
    
    async def _extract_entities(self, content: str, context: Dict[str, Any]) -> List[SemanticAnnotation]:
        """Extract entity annotations from content"""
        try:
            annotations = []
            
            # Define entity patterns
            entity_patterns = {
                EntityType.AGENT: [
                    r'\b(agent|bot|user)[-_]?\w*\b',
                    r'\b\w*[-_]?(agent|bot)\b'
                ],
                EntityType.TARGET: [
                    r'\b(target|victim|host|server|system)\s+\w+\b',
                    r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'  # IP addresses
                ],
                EntityType.TOOL: [
                    r'\b(nmap|sqlmap|metasploit|burp|wireshark|nessus)\b',
                    r'\b\w+\.(exe|py|sh|bat)\b'
                ],
                EntityType.TECHNIQUE: [
                    r'\b(T\d{4}(\.\d{3})?)\b',  # MITRE ATT&CK technique IDs
                    r'\b(sql\s+injection|xss|csrf|rce|lfi|rfi)\b'
                ],
                EntityType.VULNERABILITY: [
                    r'\b(CVE-\d{4}-\d{4,})\b',
                    r'\b(buffer\s+overflow|privilege\s+escalation)\b'
                ],
                EntityType.CREDENTIAL: [
                    r'\b(username|password|token|key|credential)\s*[:=]\s*\w+\b',
                    r'\b(admin|root|administrator)\b'
                ],
                EntityType.NETWORK: [
                    r'\b(port\s+\d+|tcp|udp|http|https|ssh|ftp)\b',
                    r'\b\w+\.(com|org|net|edu|gov)\b'
                ],
                EntityType.FILE: [
                    r'\b\w+\.(txt|log|conf|cfg|ini|xml|json)\b',
                    r'\b(/[^/\s]+)+\b'  # Unix paths
                ]
            }
            
            # Extract entities using patterns
            for entity_type, patterns in entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        annotation = SemanticAnnotation(
                            annotation_id=str(uuid.uuid4()),
                            annotation_type=AnnotationType.ENTITY,
                            entity_type=entity_type,
                            text_span=(match.start(), match.end()),
                            content=match.group(),
                            confidence=0.7,
                            metadata={
                                'extraction_method': 'pattern_matching',
                                'pattern': pattern,
                                'context': context
                            }
                        )
                        annotations.append(annotation)
            
            return annotations
            
        except Exception as e:
            self.logger.error(f"Failed to extract entities: {e}")
            return []
    
    async def _extract_concepts(self, content: str, context: Dict[str, Any]) -> List[SemanticAnnotation]:
        """Extract concept annotations from content"""
        try:
            annotations = []
            
            # Define concept patterns
            concept_patterns = {
                ConceptType.ATTACK_PHASE: [
                    r'\b(reconnaissance|recon|scanning|enumeration)\b',
                    r'\b(exploitation|exploit|attack|breach)\b',
                    r'\b(persistence|backdoor|implant)\b',
                    r'\b(privilege\s+escalation|privesc)\b',
                    r'\b(lateral\s+movement|pivoting)\b',
                    r'\b(exfiltration|data\s+theft|extraction)\b'
                ],
                ConceptType.DEFENSE_STRATEGY: [
                    r'\b(detection|monitoring|alerting)\b',
                    r'\b(prevention|blocking|filtering)\b',
                    r'\b(containment|isolation|quarantine)\b',
                    r'\b(response|mitigation|remediation)\b',
                    r'\b(forensics|investigation|analysis)\b'
                ],
                ConceptType.SUCCESS_PATTERN: [
                    r'\b(successful|succeeded|achieved|accomplished)\b',
                    r'\b(gained\s+access|compromised|breached)\b',
                    r'\b(executed|completed|finished)\b'
                ],
                ConceptType.FAILURE_PATTERN: [
                    r'\b(failed|unsuccessful|blocked|denied)\b',
                    r'\b(error|exception|timeout|refused)\b',
                    r'\b(detected|caught|prevented)\b'
                ],
                ConceptType.LEARNING_OUTCOME: [
                    r'\b(learned|discovered|found|identified)\b',
                    r'\b(lesson|insight|knowledge|understanding)\b',
                    r'\b(improved|optimized|enhanced)\b'
                ]
            }
            
            # Extract concepts using patterns
            for concept_type, patterns in concept_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        annotation = SemanticAnnotation(
                            annotation_id=str(uuid.uuid4()),
                            annotation_type=AnnotationType.CONCEPT,
                            concept_type=concept_type,
                            text_span=(match.start(), match.end()),
                            content=match.group(),
                            confidence=0.6,
                            metadata={
                                'extraction_method': 'concept_matching',
                                'pattern': pattern,
                                'context': context
                            }
                        )
                        annotations.append(annotation)
            
            return annotations
            
        except Exception as e:
            self.logger.error(f"Failed to extract concepts: {e}")
            return []
    
    async def _extract_relationships(self, content: str, existing_annotations: List[SemanticAnnotation], context: Dict[str, Any]) -> List[SemanticAnnotation]:
        """Extract relationship annotations between entities and concepts"""
        try:
            annotations = []
            
            # Define relationship patterns
            relationship_patterns = [
                (r'(\w+)\s+(uses|utilizes|employs)\s+(\w+)', 'uses'),
                (r'(\w+)\s+(targets|attacks|exploits)\s+(\w+)', 'targets'),
                (r'(\w+)\s+(detects|identifies|finds)\s+(\w+)', 'detects'),
                (r'(\w+)\s+(blocks|prevents|stops)\s+(\w+)', 'blocks'),
                (r'(\w+)\s+(leads\s+to|results\s+in|causes)\s+(\w+)', 'causes'),
                (r'(\w+)\s+(requires|needs|depends\s+on)\s+(\w+)', 'requires')
            ]
            
            # Extract relationships
            for pattern, relation_type in relationship_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    annotation = SemanticAnnotation(
                        annotation_id=str(uuid.uuid4()),
                        annotation_type=AnnotationType.RELATIONSHIP,
                        text_span=(match.start(), match.end()),
                        content=match.group(),
                        confidence=0.5,
                        metadata={
                            'relation_type': relation_type,
                            'subject': match.group(1),
                            'object': match.group(3),
                            'extraction_method': 'relationship_pattern',
                            'context': context
                        }
                    )
                    annotations.append(annotation)
            
            return annotations
            
        except Exception as e:
            self.logger.error(f"Failed to extract relationships: {e}")
            return []
    
    async def _merge_overlapping_annotations(self, annotations: List[SemanticAnnotation]) -> List[SemanticAnnotation]:
        """Merge overlapping annotations"""
        try:
            if not annotations:
                return []
            
            # Sort by start position
            sorted_annotations = sorted(annotations, key=lambda x: x.text_span[0])
            merged = []
            
            current = sorted_annotations[0]
            
            for next_annotation in sorted_annotations[1:]:
                # Check for overlap
                if (current.text_span[1] > next_annotation.text_span[0] and 
                    current.annotation_type == next_annotation.annotation_type):
                    
                    # Merge annotations - keep the one with higher confidence
                    if next_annotation.confidence > current.confidence:
                        current = next_annotation
                    # Extend span if needed
                    current.text_span = (
                        min(current.text_span[0], next_annotation.text_span[0]),
                        max(current.text_span[1], next_annotation.text_span[1])
                    )
                else:
                    merged.append(current)
                    current = next_annotation
            
            merged.append(current)
            return merged
            
        except Exception as e:
            self.logger.error(f"Failed to merge overlapping annotations: {e}")
            return annotations
    
    async def _generate_categories(self, content: str, annotations: List[SemanticAnnotation], context: Dict[str, Any]) -> List[str]:
        """Generate categories for memory content"""
        try:
            categories = set()
            
            # Category based on annotations
            for annotation in annotations:
                if annotation.entity_type:
                    categories.add(f"entity_{annotation.entity_type.value}")
                if annotation.concept_type:
                    categories.add(f"concept_{annotation.concept_type.value}")
            
            # Context-based categories
            if context.get('success'):
                categories.add('successful_operation')
            else:
                categories.add('failed_operation')
            
            if context.get('team'):
                categories.add(f"team_{context['team']}")
            
            if context.get('action_type'):
                categories.add(f"action_{context['action_type']}")
            
            # Content-based categories
            content_lower = content.lower()
            
            if any(word in content_lower for word in ['attack', 'exploit', 'breach', 'compromise']):
                categories.add('offensive_operation')
            
            if any(word in content_lower for word in ['defend', 'block', 'prevent', 'detect']):
                categories.add('defensive_operation')
            
            if any(word in content_lower for word in ['learn', 'discover', 'insight', 'knowledge']):
                categories.add('learning_experience')
            
            # Confidence-based categories
            if context.get('confidence_score', 0) > 0.8:
                categories.add('high_confidence')
            elif context.get('confidence_score', 0) < 0.3:
                categories.add('low_confidence')
            
            return list(categories)
            
        except Exception as e:
            self.logger.error(f"Failed to generate categories: {e}")
            return []
    
    async def _generate_semantic_tags(self, content: str, annotations: List[SemanticAnnotation], categories: List[str]) -> Set[str]:
        """Generate semantic tags for indexing and search"""
        try:
            tags = set()
            
            # Tags from annotations
            for annotation in annotations:
                # Add annotation content as tag
                tag_content = annotation.content.lower().strip()
                if len(tag_content) > 2:  # Minimum tag length
                    tags.add(tag_content)
                
                # Add type-based tags
                if annotation.entity_type:
                    tags.add(annotation.entity_type.value)
                if annotation.concept_type:
                    tags.add(annotation.concept_type.value)
            
            # Tags from categories
            for category in categories:
                tags.add(category)
            
            # Extract key terms from content
            key_terms = await self._extract_key_terms(content)
            tags.update(key_terms)
            
            # Filter out common words and short tags
            filtered_tags = {
                tag for tag in tags 
                if len(tag) > 2 and tag not in self._get_stop_words()
            }
            
            return filtered_tags
            
        except Exception as e:
            self.logger.error(f"Failed to generate semantic tags: {e}")
            return set()
    
    async def _extract_key_terms(self, content: str) -> Set[str]:
        """Extract key terms from content"""
        try:
            # Simple key term extraction (could be enhanced with NLP libraries)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            
            # Count word frequency
            word_counts = Counter(words)
            
            # Get top terms (excluding stop words)
            stop_words = self._get_stop_words()
            key_terms = {
                word for word, count in word_counts.most_common(10)
                if word not in stop_words and count > 1
            }
            
            return key_terms
            
        except Exception as e:
            self.logger.error(f"Failed to extract key terms: {e}")
            return set()
    
    def _get_stop_words(self) -> Set[str]:
        """Get common stop words to filter out"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those', 'i', 'me',
            'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'whose', 'this', 'that', 'these', 'those', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'shall'
        }
    
    async def _calculate_annotation_confidence(self, annotations: List[SemanticAnnotation]) -> float:
        """Calculate overall confidence score for annotations"""
        try:
            if not annotations:
                return 0.0
            
            # Weight annotations by type
            type_weights = {
                AnnotationType.ENTITY: 1.0,
                AnnotationType.CONCEPT: 0.8,
                AnnotationType.RELATIONSHIP: 0.6,
                AnnotationType.SENTIMENT: 0.4,
                AnnotationType.INTENT: 0.7,
                AnnotationType.CONTEXT: 0.5
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for annotation in annotations:
                weight = type_weights.get(annotation.annotation_type, 0.5)
                weighted_sum += annotation.confidence * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate annotation confidence: {e}")
            return 0.0
    
    async def _update_semantic_index(self, memory_id: str, annotated_memory: AnnotatedMemory) -> None:
        """Update semantic index for fast retrieval"""
        try:
            # Index by categories
            for category in annotated_memory.categories:
                self.semantic_index[f"category:{category}"].add(memory_id)
            
            # Index by semantic tags
            for tag in annotated_memory.semantic_tags:
                self.semantic_index[f"tag:{tag}"].add(memory_id)
            
            # Index by annotation types
            for annotation in annotated_memory.annotations:
                self.semantic_index[f"annotation:{annotation.annotation_type.value}"].add(memory_id)
                
                if annotation.entity_type:
                    self.semantic_index[f"entity:{annotation.entity_type.value}"].add(memory_id)
                
                if annotation.concept_type:
                    self.semantic_index[f"concept:{annotation.concept_type.value}"].add(memory_id)
            
        except Exception as e:
            self.logger.error(f"Failed to update semantic index: {e}")
    
    async def search_by_semantics(self, query: str, filters: Dict[str, Any] = None) -> List[AnnotatedMemory]:
        """Search annotated memories by semantic criteria"""
        try:
            filters = filters or {}
            matching_memory_ids = set()
            
            # Search by query terms
            query_terms = query.lower().split()
            for term in query_terms:
                # Search in tags
                tag_matches = self.semantic_index.get(f"tag:{term}", set())
                matching_memory_ids.update(tag_matches)
                
                # Search in categories
                category_matches = self.semantic_index.get(f"category:{term}", set())
                matching_memory_ids.update(category_matches)
            
            # Apply filters
            if filters.get('annotation_type'):
                annotation_matches = self.semantic_index.get(f"annotation:{filters['annotation_type']}", set())
                matching_memory_ids = matching_memory_ids.intersection(annotation_matches) if matching_memory_ids else annotation_matches
            
            if filters.get('entity_type'):
                entity_matches = self.semantic_index.get(f"entity:{filters['entity_type']}", set())
                matching_memory_ids = matching_memory_ids.intersection(entity_matches) if matching_memory_ids else entity_matches
            
            if filters.get('concept_type'):
                concept_matches = self.semantic_index.get(f"concept:{filters['concept_type']}", set())
                matching_memory_ids = matching_memory_ids.intersection(concept_matches) if matching_memory_ids else concept_matches
            
            # Get annotated memories
            results = []
            for memory_id in matching_memory_ids:
                if memory_id in self.annotated_memories:
                    memory = self.annotated_memories[memory_id]
                    
                    # Apply confidence filter
                    if filters.get('min_confidence'):
                        if memory.confidence_score < filters['min_confidence']:
                            continue
                    
                    results.append(memory)
            
            # Sort by confidence score
            results.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search by semantics: {e}")
            return []
    
    async def get_memory_categories(self, memory_id: str) -> List[str]:
        """Get categories for a specific memory"""
        try:
            annotated_memory = self.annotated_memories.get(memory_id)
            return annotated_memory.categories if annotated_memory else []
            
        except Exception as e:
            self.logger.error(f"Failed to get memory categories: {e}")
            return []
    
    async def get_semantic_tags(self, memory_id: str) -> Set[str]:
        """Get semantic tags for a specific memory"""
        try:
            annotated_memory = self.annotated_memories.get(memory_id)
            return annotated_memory.semantic_tags if annotated_memory else set()
            
        except Exception as e:
            self.logger.error(f"Failed to get semantic tags: {e}")
            return set()
    
    async def add_annotation_rule(self, rule: AnnotationRule) -> None:
        """Add a new annotation rule"""
        try:
            self.annotation_rules[rule.rule_id] = rule
            self.logger.debug(f"Added annotation rule: {rule.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to add annotation rule: {e}")
    
    async def remove_annotation_rule(self, rule_id: str) -> bool:
        """Remove an annotation rule"""
        try:
            if rule_id in self.annotation_rules:
                del self.annotation_rules[rule_id]
                self.logger.debug(f"Removed annotation rule: {rule_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove annotation rule: {e}")
            return False
    
    async def _initialize_default_rules(self) -> None:
        """Initialize default annotation rules"""
        try:
            default_rules = [
                AnnotationRule(
                    rule_id="mitre_technique",
                    name="MITRE ATT&CK Technique",
                    pattern=r'\bT\d{4}(\.\d{3})?\b',
                    annotation_type=AnnotationType.ENTITY,
                    entity_type=EntityType.TECHNIQUE,
                    confidence=0.9
                ),
                AnnotationRule(
                    rule_id="cve_vulnerability",
                    name="CVE Vulnerability",
                    pattern=r'\bCVE-\d{4}-\d{4,}\b',
                    annotation_type=AnnotationType.ENTITY,
                    entity_type=EntityType.VULNERABILITY,
                    confidence=0.95
                ),
                AnnotationRule(
                    rule_id="ip_address",
                    name="IP Address",
                    pattern=r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
                    annotation_type=AnnotationType.ENTITY,
                    entity_type=EntityType.TARGET,
                    confidence=0.8
                ),
                AnnotationRule(
                    rule_id="success_indicator",
                    name="Success Indicator",
                    pattern=r'\b(successful|succeeded|achieved|completed)\b',
                    annotation_type=AnnotationType.CONCEPT,
                    concept_type=ConceptType.SUCCESS_PATTERN,
                    confidence=0.7
                ),
                AnnotationRule(
                    rule_id="failure_indicator",
                    name="Failure Indicator",
                    pattern=r'\b(failed|unsuccessful|blocked|denied|error)\b',
                    annotation_type=AnnotationType.CONCEPT,
                    concept_type=ConceptType.FAILURE_PATTERN,
                    confidence=0.7
                )
            ]
            
            for rule in default_rules:
                self.annotation_rules[rule.rule_id] = rule
            
            self.logger.info(f"Initialized {len(default_rules)} default annotation rules")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default rules: {e}")
    
    async def _initialize_category_taxonomy(self) -> None:
        """Initialize category taxonomy"""
        try:
            self.category_taxonomy = {
                'operations': ['offensive_operation', 'defensive_operation', 'reconnaissance_operation'],
                'outcomes': ['successful_operation', 'failed_operation', 'partial_success'],
                'learning': ['learning_experience', 'tactical_knowledge', 'strategic_insight'],
                'entities': ['entity_agent', 'entity_target', 'entity_tool', 'entity_technique'],
                'concepts': ['concept_attack_phase', 'concept_defense_strategy', 'concept_success_pattern'],
                'confidence': ['high_confidence', 'medium_confidence', 'low_confidence']
            }
            
            self.logger.debug("Initialized category taxonomy")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize category taxonomy: {e}")
    
    def get_annotation_statistics(self) -> Dict[str, Any]:
        """Get annotation engine statistics"""
        try:
            # Calculate annotation type distribution
            annotation_type_counts = defaultdict(int)
            entity_type_counts = defaultdict(int)
            concept_type_counts = defaultdict(int)
            
            for memory in self.annotated_memories.values():
                for annotation in memory.annotations:
                    annotation_type_counts[annotation.annotation_type.value] += 1
                    if annotation.entity_type:
                        entity_type_counts[annotation.entity_type.value] += 1
                    if annotation.concept_type:
                        concept_type_counts[annotation.concept_type.value] += 1
            
            # Calculate category distribution
            category_counts = defaultdict(int)
            for memory in self.annotated_memories.values():
                for category in memory.categories:
                    category_counts[category] += 1
            
            return {
                'total_annotated_memories': len(self.annotated_memories),
                'total_annotation_rules': len(self.annotation_rules),
                'active_rules': sum(1 for rule in self.annotation_rules.values() if rule.active),
                'annotation_type_distribution': dict(annotation_type_counts),
                'entity_type_distribution': dict(entity_type_counts),
                'concept_type_distribution': dict(concept_type_counts),
                'category_distribution': dict(category_counts),
                'semantic_index_size': len(self.semantic_index),
                'processing_stats': self.stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get annotation statistics: {e}")
            return {'error': str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown the semantic annotation engine"""
        try:
            self.logger.info("Shutting down semantic annotation engine")
            
            # Clear data structures
            self.annotation_rules.clear()
            self.annotated_memories.clear()
            self.category_taxonomy.clear()
            self.semantic_index.clear()
            
            self.initialized = False
            self.logger.info("Semantic annotation engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during semantic annotation engine shutdown: {e}")

# Factory function
def create_semantic_annotation_engine() -> SemanticAnnotationEngine:
    """Create a semantic annotation engine instance"""
    return SemanticAnnotationEngine()