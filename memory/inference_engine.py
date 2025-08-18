#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Inference Engine
Knowledge graph inference and relationship modeling
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from pathlib import Path
import networkx as nx
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class InferenceType(Enum):
    """Types of inference operations"""
    TRANSITIVE = "transitive"
    SIMILARITY = "similarity"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    STATISTICAL = "statistical"
    LOGICAL = "logical"

class ConfidenceLevel(Enum):
    """Confidence levels for inferences"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

@dataclass
class InferenceRule:
    """Rule for knowledge inference"""
    rule_id: str
    name: str
    description: str
    inference_type: InferenceType
    conditions: List[Dict[str, Any]]
    conclusions: List[Dict[str, Any]]
    confidence_threshold: float
    priority: int
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class InferenceResult:
    """Result of an inference operation"""
    result_id: str
    rule_id: str
    inference_type: InferenceType
    source_entities: List[str]
    inferred_entity: Optional[str]
    inferred_relation: Optional[str]
    confidence_score: float
    confidence_level: ConfidenceLevel
    evidence: List[str]
    properties: Dict[str, Any]
    created_at: datetime = None
    validated: Optional[bool] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class KnowledgePattern:
    """Pattern in the knowledge graph"""
    pattern_id: str
    name: str
    description: str
    pattern_type: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    frequency: int
    confidence: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class InferenceEngine:
    """
    Inference engine for knowledge graph reasoning.
    
    Features:
    - Rule-based inference for knowledge derivation
    - Graph pattern matching and discovery
    - Confidence propagation and uncertainty handling
    - Temporal reasoning for time-based relationships
    - Statistical inference from simulation outcomes
    """
    
    def __init__(self, data_dir: str = "./knowledge_base/inference"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Inference components
        self.inference_rules: Dict[str, InferenceRule] = {}
        self.inference_results: Dict[str, InferenceResult] = {}
        self.knowledge_patterns: Dict[str, KnowledgePattern] = {}
        
        # Knowledge graph reference
        self.knowledge_graph: Optional[nx.MultiDiGraph] = None
        
        # Inference statistics
        self.inference_stats = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'by_type': defaultdict(int),
            'by_confidence': defaultdict(int)
        }
        
        # File paths
        self.rules_file = self.data_dir / "inference_rules.json"
        self.results_file = self.data_dir / "inference_results.json"
        self.patterns_file = self.data_dir / "knowledge_patterns.json"
        self.stats_file = self.data_dir / "inference_stats.json"
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self, knowledge_graph: nx.MultiDiGraph) -> None:
        """Initialize the inference engine"""
        try:
            self.logger.info("Initializing inference engine")
            
            self.knowledge_graph = knowledge_graph
            
            # Load existing data
            await self._load_inference_rules()
            await self._load_inference_results()
            await self._load_knowledge_patterns()
            await self._load_inference_stats()
            
            # Initialize default rules if empty
            if not self.inference_rules:
                await self._initialize_default_rules()
            
            self.initialized = True
            self.logger.info(f"Inference engine initialized with {len(self.inference_rules)} rules")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize inference engine: {e}")
            raise
    
    async def add_inference_rule(self, rule: InferenceRule) -> None:
        """Add an inference rule"""
        try:
            self.inference_rules[rule.rule_id] = rule
            await self._save_inference_rules()
            
            self.logger.debug(f"Added inference rule: {rule.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to add inference rule: {e}")
            raise
    
    async def run_inference(self, rule_ids: Optional[List[str]] = None) -> List[InferenceResult]:
        """Run inference using specified rules or all enabled rules"""
        try:
            if not self.knowledge_graph:
                raise ValueError("Knowledge graph not initialized")
            
            results = []
            
            # Determine which rules to run
            rules_to_run = []
            if rule_ids:
                rules_to_run = [self.inference_rules[rid] for rid in rule_ids if rid in self.inference_rules]
            else:
                rules_to_run = [rule for rule in self.inference_rules.values() if rule.enabled]
            
            # Sort rules by priority
            rules_to_run.sort(key=lambda x: x.priority, reverse=True)
            
            for rule in rules_to_run:
                try:
                    rule_results = await self._apply_inference_rule(rule)
                    results.extend(rule_results)
                    
                    # Update statistics
                    self.inference_stats['by_type'][rule.inference_type.value] += len(rule_results)
                    
                except Exception as e:
                    self.logger.error(f"Failed to apply rule {rule.name}: {e}")
                    self.inference_stats['failed_inferences'] += 1
            
            # Update overall statistics
            self.inference_stats['total_inferences'] += len(results)
            self.inference_stats['successful_inferences'] += len(results)
            
            # Save results
            for result in results:
                self.inference_results[result.result_id] = result
                self.inference_stats['by_confidence'][result.confidence_level.value] += 1
            
            await self._save_inference_results()
            await self._save_inference_stats()
            
            self.logger.info(f"Inference completed: {len(results)} new inferences generated")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to run inference: {e}")
            return []
    
    async def discover_patterns(self, min_frequency: int = 3, min_confidence: float = 0.6) -> List[KnowledgePattern]:
        """Discover patterns in the knowledge graph"""
        try:
            if not self.knowledge_graph:
                raise ValueError("Knowledge graph not initialized")
            
            patterns = []
            
            # Find common subgraph patterns
            patterns.extend(await self._find_path_patterns(min_frequency))
            patterns.extend(await self._find_star_patterns(min_frequency))
            patterns.extend(await self._find_triangle_patterns(min_frequency))
            
            # Filter by confidence
            patterns = [p for p in patterns if p.confidence >= min_confidence]
            
            # Save discovered patterns
            for pattern in patterns:
                self.knowledge_patterns[pattern.pattern_id] = pattern
            
            await self._save_knowledge_patterns()
            
            self.logger.info(f"Discovered {len(patterns)} knowledge patterns")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to discover patterns: {e}")
            return []
    
    async def validate_inference(self, result_id: str, validation: bool, evidence: Optional[str] = None) -> None:
        """Validate an inference result"""
        try:
            if result_id not in self.inference_results:
                raise ValueError(f"Inference result {result_id} not found")
            
            result = self.inference_results[result_id]
            result.validated = validation
            
            if evidence:
                result.evidence.append(f"Validation: {evidence}")
            
            # Adjust confidence based on validation
            if validation:
                result.confidence_score = min(1.0, result.confidence_score + 0.1)
            else:
                result.confidence_score = max(0.0, result.confidence_score - 0.2)
            
            # Update confidence level
            result.confidence_level = self._score_to_confidence_level(result.confidence_score)
            
            await self._save_inference_results()
            
            self.logger.debug(f"Validated inference {result_id}: {validation}")
            
        except Exception as e:
            self.logger.error(f"Failed to validate inference: {e}")
    
    async def get_inference_explanation(self, result_id: str) -> Dict[str, Any]:
        """Get explanation for an inference result"""
        try:
            if result_id not in self.inference_results:
                raise ValueError(f"Inference result {result_id} not found")
            
            result = self.inference_results[result_id]
            rule = self.inference_rules.get(result.rule_id)
            
            explanation = {
                'result_id': result_id,
                'inference_type': result.inference_type.value,
                'confidence_score': result.confidence_score,
                'confidence_level': result.confidence_level.value,
                'evidence': result.evidence,
                'rule_used': {
                    'name': rule.name if rule else 'Unknown',
                    'description': rule.description if rule else 'Rule not found'
                },
                'source_entities': result.source_entities,
                'reasoning_chain': await self._build_reasoning_chain(result),
                'validation_status': result.validated
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Failed to get inference explanation: {e}")
            return {}
    
    async def propagate_confidence(self, start_node: str, max_depth: int = 3) -> Dict[str, float]:
        """Propagate confidence scores through the knowledge graph"""
        try:
            if not self.knowledge_graph or start_node not in self.knowledge_graph:
                return {}
            
            confidence_scores = {start_node: 1.0}
            visited = set()
            queue = deque([(start_node, 1.0, 0)])
            
            while queue:
                node, confidence, depth = queue.popleft()
                
                if node in visited or depth >= max_depth:
                    continue
                
                visited.add(node)
                
                # Propagate to neighbors
                for neighbor in self.knowledge_graph.neighbors(node):
                    edge_data = self.knowledge_graph.get_edge_data(node, neighbor)
                    
                    # Calculate propagated confidence
                    edge_confidence = 1.0
                    if edge_data:
                        for key, data in edge_data.items():
                            edge_confidence = min(edge_confidence, data.get('confidence', 1.0))
                    
                    propagated_confidence = confidence * edge_confidence * 0.9  # Decay factor
                    
                    if neighbor not in confidence_scores or propagated_confidence > confidence_scores[neighbor]:
                        confidence_scores[neighbor] = propagated_confidence
                        queue.append((neighbor, propagated_confidence, depth + 1))
            
            return confidence_scores
            
        except Exception as e:
            self.logger.error(f"Failed to propagate confidence: {e}")
            return {}
    
    async def find_shortest_reasoning_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest reasoning path between two entities"""
        try:
            if not self.knowledge_graph:
                return None
            
            try:
                path = nx.shortest_path(self.knowledge_graph, source, target)
                return path
            except nx.NetworkXNoPath:
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to find reasoning path: {e}")
            return None
    
    async def get_inference_statistics(self) -> Dict[str, Any]:
        """Get inference engine statistics"""
        try:
            stats = dict(self.inference_stats)
            
            # Add additional statistics
            stats.update({
                'total_rules': len(self.inference_rules),
                'enabled_rules': len([r for r in self.inference_rules.values() if r.enabled]),
                'total_results': len(self.inference_results),
                'validated_results': len([r for r in self.inference_results.values() if r.validated is True]),
                'unvalidated_results': len([r for r in self.inference_results.values() if r.validated is None]),
                'rejected_results': len([r for r in self.inference_results.values() if r.validated is False]),
                'knowledge_patterns': len(self.knowledge_patterns),
                'success_rate': (stats['successful_inferences'] / max(stats['total_inferences'], 1)) * 100
            })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get inference statistics: {e}")
            return {}
    
    async def _apply_inference_rule(self, rule: InferenceRule) -> List[InferenceResult]:
        """Apply a specific inference rule"""
        try:
            results = []
            
            if rule.inference_type == InferenceType.TRANSITIVE:
                results = await self._apply_transitive_rule(rule)
            elif rule.inference_type == InferenceType.SIMILARITY:
                results = await self._apply_similarity_rule(rule)
            elif rule.inference_type == InferenceType.CAUSAL:
                results = await self._apply_causal_rule(rule)
            elif rule.inference_type == InferenceType.TEMPORAL:
                results = await self._apply_temporal_rule(rule)
            elif rule.inference_type == InferenceType.STATISTICAL:
                results = await self._apply_statistical_rule(rule)
            elif rule.inference_type == InferenceType.LOGICAL:
                results = await self._apply_logical_rule(rule)
            
            # Filter by confidence threshold
            results = [r for r in results if r.confidence_score >= rule.confidence_threshold]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to apply inference rule: {e}")
            return []
    
    async def _apply_transitive_rule(self, rule: InferenceRule) -> List[InferenceResult]:
        """Apply transitive inference rule (if A->B and B->C, then A->C)"""
        try:
            results = []
            
            # Find transitive relationships
            for node in self.knowledge_graph.nodes():
                # Get outgoing edges
                for neighbor1 in self.knowledge_graph.neighbors(node):
                    for neighbor2 in self.knowledge_graph.neighbors(neighbor1):
                        if neighbor2 != node and not self.knowledge_graph.has_edge(node, neighbor2):
                            # Check if this satisfies rule conditions
                            if await self._check_rule_conditions(rule, [node, neighbor1, neighbor2]):
                                confidence = await self._calculate_transitive_confidence(node, neighbor1, neighbor2)
                                
                                result = InferenceResult(
                                    result_id=str(uuid.uuid4()),
                                    rule_id=rule.rule_id,
                                    inference_type=InferenceType.TRANSITIVE,
                                    source_entities=[node, neighbor1, neighbor2],
                                    inferred_entity=None,
                                    inferred_relation=f"transitive_{node}_{neighbor2}",
                                    confidence_score=confidence,
                                    confidence_level=self._score_to_confidence_level(confidence),
                                    evidence=[f"Transitive relationship: {node} -> {neighbor1} -> {neighbor2}"],
                                    properties={"path": [node, neighbor1, neighbor2]}
                                )
                                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to apply transitive rule: {e}")
            return []
    
    async def _apply_similarity_rule(self, rule: InferenceRule) -> List[InferenceResult]:
        """Apply similarity inference rule"""
        try:
            results = []
            
            # Find similar entities based on properties
            nodes = list(self.knowledge_graph.nodes(data=True))
            
            for i, (node1, data1) in enumerate(nodes):
                for j, (node2, data2) in enumerate(nodes[i+1:], i+1):
                    similarity = await self._calculate_node_similarity(data1, data2)
                    
                    if similarity >= 0.7:  # High similarity threshold
                        confidence = similarity * 0.8  # Adjust confidence
                        
                        result = InferenceResult(
                            result_id=str(uuid.uuid4()),
                            rule_id=rule.rule_id,
                            inference_type=InferenceType.SIMILARITY,
                            source_entities=[node1, node2],
                            inferred_entity=None,
                            inferred_relation=f"similar_{node1}_{node2}",
                            confidence_score=confidence,
                            confidence_level=self._score_to_confidence_level(confidence),
                            evidence=[f"High similarity score: {similarity:.2f}"],
                            properties={"similarity_score": similarity}
                        )
                        results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to apply similarity rule: {e}")
            return []
    
    async def _apply_causal_rule(self, rule: InferenceRule) -> List[InferenceResult]:
        """Apply causal inference rule"""
        try:
            results = []
            
            # Look for causal patterns in the graph
            # This is a simplified implementation
            for node in self.knowledge_graph.nodes():
                predecessors = list(self.knowledge_graph.predecessors(node))
                successors = list(self.knowledge_graph.successors(node))
                
                if len(predecessors) > 0 and len(successors) > 0:
                    # Potential causal relationship
                    for pred in predecessors:
                        for succ in successors:
                            if not self.knowledge_graph.has_edge(pred, succ):
                                confidence = 0.6  # Base causal confidence
                                
                                result = InferenceResult(
                                    result_id=str(uuid.uuid4()),
                                    rule_id=rule.rule_id,
                                    inference_type=InferenceType.CAUSAL,
                                    source_entities=[pred, node, succ],
                                    inferred_entity=None,
                                    inferred_relation=f"causal_{pred}_{succ}",
                                    confidence_score=confidence,
                                    confidence_level=self._score_to_confidence_level(confidence),
                                    evidence=[f"Causal chain: {pred} -> {node} -> {succ}"],
                                    properties={"mediator": node}
                                )
                                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to apply causal rule: {e}")
            return []
    
    async def _apply_temporal_rule(self, rule: InferenceRule) -> List[InferenceResult]:
        """Apply temporal inference rule"""
        try:
            results = []
            
            # Analyze temporal patterns in the graph
            # This would require timestamp data on edges/nodes
            # Simplified implementation for now
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to apply temporal rule: {e}")
            return []
    
    async def _apply_statistical_rule(self, rule: InferenceRule) -> List[InferenceResult]:
        """Apply statistical inference rule"""
        try:
            results = []
            
            # Statistical analysis of graph properties
            # This would analyze patterns in simulation outcomes
            # Simplified implementation for now
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to apply statistical rule: {e}")
            return []
    
    async def _apply_logical_rule(self, rule: InferenceRule) -> List[InferenceResult]:
        """Apply logical inference rule"""
        try:
            results = []
            
            # Logical reasoning based on rule conditions
            # This would implement formal logic rules
            # Simplified implementation for now
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to apply logical rule: {e}")
            return []
    
    async def _check_rule_conditions(self, rule: InferenceRule, entities: List[str]) -> bool:
        """Check if rule conditions are satisfied"""
        try:
            # Simplified condition checking
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check rule conditions: {e}")
            return False
    
    async def _calculate_transitive_confidence(self, node1: str, node2: str, node3: str) -> float:
        """Calculate confidence for transitive relationship"""
        try:
            # Get edge confidences
            edge1_conf = 1.0
            edge2_conf = 1.0
            
            if self.knowledge_graph.has_edge(node1, node2):
                edge_data = self.knowledge_graph.get_edge_data(node1, node2)
                if edge_data:
                    edge1_conf = min(data.get('confidence', 1.0) for data in edge_data.values())
            
            if self.knowledge_graph.has_edge(node2, node3):
                edge_data = self.knowledge_graph.get_edge_data(node2, node3)
                if edge_data:
                    edge2_conf = min(data.get('confidence', 1.0) for data in edge_data.values())
            
            # Transitive confidence is the product with decay
            return edge1_conf * edge2_conf * 0.8
            
        except Exception as e:
            self.logger.error(f"Failed to calculate transitive confidence: {e}")
            return 0.5
    
    async def _calculate_node_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """Calculate similarity between two nodes"""
        try:
            # Simple similarity based on common properties
            keys1 = set(data1.keys())
            keys2 = set(data2.keys())
            
            common_keys = keys1.intersection(keys2)
            all_keys = keys1.union(keys2)
            
            if not all_keys:
                return 0.0
            
            # Jaccard similarity for keys
            key_similarity = len(common_keys) / len(all_keys)
            
            # Value similarity for common keys
            value_similarity = 0.0
            if common_keys:
                matches = 0
                for key in common_keys:
                    if data1[key] == data2[key]:
                        matches += 1
                value_similarity = matches / len(common_keys)
            
            return (key_similarity + value_similarity) / 2
            
        except Exception as e:
            self.logger.error(f"Failed to calculate node similarity: {e}")
            return 0.0
    
    def _score_to_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level"""
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.7:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    async def _build_reasoning_chain(self, result: InferenceResult) -> List[Dict[str, Any]]:
        """Build reasoning chain for an inference result"""
        try:
            chain = []
            
            # Add source entities
            for entity in result.source_entities:
                if self.knowledge_graph and entity in self.knowledge_graph:
                    node_data = self.knowledge_graph.nodes[entity]
                    chain.append({
                        'type': 'entity',
                        'id': entity,
                        'name': node_data.get('name', entity),
                        'properties': dict(node_data)
                    })
            
            # Add inference step
            chain.append({
                'type': 'inference',
                'inference_type': result.inference_type.value,
                'confidence': result.confidence_score,
                'evidence': result.evidence
            })
            
            return chain
            
        except Exception as e:
            self.logger.error(f"Failed to build reasoning chain: {e}")
            return []
    
    async def _find_path_patterns(self, min_frequency: int) -> List[KnowledgePattern]:
        """Find common path patterns in the graph"""
        try:
            patterns = []
            path_counts = defaultdict(int)
            
            # Count 2-hop and 3-hop paths
            for node in self.knowledge_graph.nodes():
                for path_length in [2, 3]:
                    for target in self.knowledge_graph.nodes():
                        if node != target:
                            try:
                                paths = list(nx.all_simple_paths(
                                    self.knowledge_graph, node, target, cutoff=path_length
                                ))
                                for path in paths:
                                    if len(path) == path_length + 1:
                                        # Create pattern signature
                                        pattern_sig = tuple(
                                            self.knowledge_graph.nodes[n].get('entity_type', 'unknown')
                                            for n in path
                                        )
                                        path_counts[pattern_sig] += 1
                            except nx.NetworkXNoPath:
                                continue
            
            # Create patterns for frequent paths
            for pattern_sig, count in path_counts.items():
                if count >= min_frequency:
                    pattern = KnowledgePattern(
                        pattern_id=str(uuid.uuid4()),
                        name=f"Path Pattern: {' -> '.join(pattern_sig)}",
                        description=f"Common path pattern with {count} occurrences",
                        pattern_type="path",
                        nodes=[{"type": node_type} for node_type in pattern_sig],
                        edges=[{"type": "path_edge"} for _ in range(len(pattern_sig) - 1)],
                        frequency=count,
                        confidence=min(1.0, count / 10.0)  # Normalize frequency to confidence
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find path patterns: {e}")
            return []
    
    async def _find_star_patterns(self, min_frequency: int) -> List[KnowledgePattern]:
        """Find star patterns (hub nodes) in the graph"""
        try:
            patterns = []
            
            # Find nodes with high degree (potential hubs)
            for node in self.knowledge_graph.nodes():
                degree = self.knowledge_graph.degree(node)
                if degree >= min_frequency:
                    node_data = self.knowledge_graph.nodes[node]
                    
                    pattern = KnowledgePattern(
                        pattern_id=str(uuid.uuid4()),
                        name=f"Star Pattern: {node_data.get('name', node)}",
                        description=f"Hub node with {degree} connections",
                        pattern_type="star",
                        nodes=[{"id": node, "type": "hub"}] + [
                            {"type": "spoke"} for _ in range(degree)
                        ],
                        edges=[{"type": "hub_edge"} for _ in range(degree)],
                        frequency=degree,
                        confidence=min(1.0, degree / 20.0)
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find star patterns: {e}")
            return []
    
    async def _find_triangle_patterns(self, min_frequency: int) -> List[KnowledgePattern]:
        """Find triangle patterns in the graph"""
        try:
            patterns = []
            triangle_types = defaultdict(int)
            
            # Find triangles
            for triangle in nx.enumerate_all_cliques(self.knowledge_graph.to_undirected()):
                if len(triangle) == 3:
                    # Get node types
                    node_types = tuple(sorted([
                        self.knowledge_graph.nodes[node].get('entity_type', 'unknown')
                        for node in triangle
                    ]))
                    triangle_types[node_types] += 1
            
            # Create patterns for frequent triangles
            for triangle_type, count in triangle_types.items():
                if count >= min_frequency:
                    pattern = KnowledgePattern(
                        pattern_id=str(uuid.uuid4()),
                        name=f"Triangle Pattern: {' - '.join(triangle_type)}",
                        description=f"Triangle pattern with {count} occurrences",
                        pattern_type="triangle",
                        nodes=[{"type": node_type} for node_type in triangle_type],
                        edges=[{"type": "triangle_edge"} for _ in range(3)],
                        frequency=count,
                        confidence=min(1.0, count / 5.0)
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find triangle patterns: {e}")
            return []
    
    async def _initialize_default_rules(self) -> None:
        """Initialize default inference rules"""
        try:
            # Transitive rule
            transitive_rule = InferenceRule(
                rule_id=str(uuid.uuid4()),
                name="Transitive Relationship Rule",
                description="If A relates to B and B relates to C, then A may relate to C",
                inference_type=InferenceType.TRANSITIVE,
                conditions=[
                    {"type": "path_exists", "length": 2},
                    {"type": "no_direct_connection"}
                ],
                conclusions=[
                    {"type": "create_relation", "relation_type": "transitive"}
                ],
                confidence_threshold=0.5,
                priority=1
            )
            
            # Similarity rule
            similarity_rule = InferenceRule(
                rule_id=str(uuid.uuid4()),
                name="Entity Similarity Rule",
                description="Entities with similar properties may have similar relationships",
                inference_type=InferenceType.SIMILARITY,
                conditions=[
                    {"type": "property_similarity", "threshold": 0.7}
                ],
                conclusions=[
                    {"type": "create_relation", "relation_type": "similar_to"}
                ],
                confidence_threshold=0.6,
                priority=2
            )
            
            # Add rules
            await self.add_inference_rule(transitive_rule)
            await self.add_inference_rule(similarity_rule)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default rules: {e}")
    
    async def _load_inference_rules(self) -> None:
        """Load inference rules from file"""
        try:
            if self.rules_file.exists():
                with open(self.rules_file, 'r') as f:
                    data = json.load(f)
                    for rule_data in data:
                        rule = InferenceRule(**rule_data)
                        rule.inference_type = InferenceType(rule.inference_type)
                        rule.created_at = datetime.fromisoformat(rule.created_at)
                        self.inference_rules[rule.rule_id] = rule
                
                self.logger.debug(f"Loaded {len(self.inference_rules)} inference rules")
                
        except Exception as e:
            self.logger.error(f"Failed to load inference rules: {e}")
    
    async def _save_inference_rules(self) -> None:
        """Save inference rules to file"""
        try:
            data = []
            for rule in self.inference_rules.values():
                rule_dict = asdict(rule)
                rule_dict['inference_type'] = rule.inference_type.value
                rule_dict['created_at'] = rule.created_at.isoformat()
                data.append(rule_dict)
            
            with open(self.rules_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save inference rules: {e}")
    
    async def _load_inference_results(self) -> None:
        """Load inference results from file"""
        try:
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    for result_data in data:
                        result = InferenceResult(**result_data)
                        result.inference_type = InferenceType(result.inference_type)
                        result.confidence_level = ConfidenceLevel(result.confidence_level)
                        result.created_at = datetime.fromisoformat(result.created_at)
                        self.inference_results[result.result_id] = result
                
                self.logger.debug(f"Loaded {len(self.inference_results)} inference results")
                
        except Exception as e:
            self.logger.error(f"Failed to load inference results: {e}")
    
    async def _save_inference_results(self) -> None:
        """Save inference results to file"""
        try:
            data = []
            for result in self.inference_results.values():
                result_dict = asdict(result)
                result_dict['inference_type'] = result.inference_type.value
                result_dict['confidence_level'] = result.confidence_level.value
                result_dict['created_at'] = result.created_at.isoformat()
                data.append(result_dict)
            
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save inference results: {e}")
    
    async def _load_knowledge_patterns(self) -> None:
        """Load knowledge patterns from file"""
        try:
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                    for pattern_data in data:
                        pattern = KnowledgePattern(**pattern_data)
                        pattern.created_at = datetime.fromisoformat(pattern.created_at)
                        self.knowledge_patterns[pattern.pattern_id] = pattern
                
                self.logger.debug(f"Loaded {len(self.knowledge_patterns)} knowledge patterns")
                
        except Exception as e:
            self.logger.error(f"Failed to load knowledge patterns: {e}")
    
    async def _save_knowledge_patterns(self) -> None:
        """Save knowledge patterns to file"""
        try:
            data = []
            for pattern in self.knowledge_patterns.values():
                pattern_dict = asdict(pattern)
                pattern_dict['created_at'] = pattern.created_at.isoformat()
                data.append(pattern_dict)
            
            with open(self.patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save knowledge patterns: {e}")
    
    async def _load_inference_stats(self) -> None:
        """Load inference statistics from file"""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, 'r') as f:
                    self.inference_stats.update(json.load(f))
                
        except Exception as e:
            self.logger.error(f"Failed to load inference stats: {e}")
    
    async def _save_inference_stats(self) -> None:
        """Save inference statistics to file"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(dict(self.inference_stats), f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save inference stats: {e}")