#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Ontology Manager
Domain-specific ontology for threat classification and semantic mapping
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Ontology entity types"""
    THREAT_ACTOR = "threat_actor"
    ATTACK_TECHNIQUE = "attack_technique"
    DEFENSE_TECHNIQUE = "defense_technique"
    VULNERABILITY = "vulnerability"
    ASSET = "asset"
    INDICATOR = "indicator"
    MITIGATION = "mitigation"
    TACTIC = "tactic"
    TOOL = "tool"
    MALWARE = "malware"

class RelationType(Enum):
    """Ontology relationship types"""
    USES = "uses"
    MITIGATES = "mitigates"
    DETECTS = "detects"
    EXPLOITS = "exploits"
    TARGETS = "targets"
    IMPLEMENTS = "implements"
    COUNTERS = "counters"
    ENABLES = "enables"
    REQUIRES = "requires"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    DERIVED_FROM = "derived_from"

@dataclass
class OntologyEntity:
    """Ontology entity with semantic properties"""
    entity_id: str
    entity_type: EntityType
    name: str
    description: str
    properties: Dict[str, Any]
    mitre_id: Optional[str] = None
    d3fend_id: Optional[str] = None
    confidence_score: float = 1.0
    created_at: datetime = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class OntologyRelation:
    """Ontology relationship between entities"""
    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: RelationType
    properties: Dict[str, Any]
    confidence_score: float = 1.0
    evidence: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class SemanticMapping:
    """Semantic mapping between frameworks"""
    mapping_id: str
    source_framework: str
    target_framework: str
    source_id: str
    target_id: str
    mapping_type: str
    confidence_score: float
    properties: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class InferenceRule:
    """Inference rule for knowledge derivation"""
    rule_id: str
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    conclusions: List[Dict[str, Any]]
    confidence_threshold: float
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class OntologyManager:
    """
    Ontology manager for cybersecurity domain knowledge.
    
    Features:
    - Domain-specific ontology for threat classification
    - Semantic entity mapping to MITRE ATT&CK and D3FEND
    - Knowledge graph with relationship modeling
    - Inference capabilities for knowledge derivation
    - Automated ontology updates from simulation outcomes
    """
    
    def __init__(self, data_dir: str = "./knowledge_base/ontology"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Core ontology storage
        self.entities: Dict[str, OntologyEntity] = {}
        self.relations: Dict[str, OntologyRelation] = {}
        self.mappings: Dict[str, SemanticMapping] = {}
        self.inference_rules: Dict[str, InferenceRule] = {}
        
        # Knowledge graph
        self.knowledge_graph = nx.MultiDiGraph()
        
        # Semantic indices
        self.entity_by_type: Dict[EntityType, Set[str]] = defaultdict(set)
        self.entity_by_mitre_id: Dict[str, str] = {}
        self.entity_by_d3fend_id: Dict[str, str] = {}
        self.relations_by_type: Dict[RelationType, Set[str]] = defaultdict(set)
        
        # File paths
        self.entities_file = self.data_dir / "entities.json"
        self.relations_file = self.data_dir / "relations.json"
        self.mappings_file = self.data_dir / "mappings.json"
        self.inference_rules_file = self.data_dir / "inference_rules.json"
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the ontology manager"""
        try:
            self.logger.info("Initializing ontology manager")
            
            # Load existing data
            await self._load_entities()
            await self._load_relations()
            await self._load_mappings()
            await self._load_inference_rules()
            
            # Initialize default ontology if empty
            if not self.entities:
                await self._initialize_default_ontology()
            
            # Build knowledge graph
            await self._build_knowledge_graph()
            
            # Build semantic indices
            await self._build_semantic_indices()
            
            self.initialized = True
            self.logger.info(f"Ontology manager initialized with {len(self.entities)} entities and {len(self.relations)} relations")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ontology manager: {e}")
            raise
    
    async def add_entity(self, entity: OntologyEntity) -> None:
        """Add an entity to the ontology"""
        try:
            self.entities[entity.entity_id] = entity
            
            # Update indices
            self.entity_by_type[entity.entity_type].add(entity.entity_id)
            if entity.mitre_id:
                self.entity_by_mitre_id[entity.mitre_id] = entity.entity_id
            if entity.d3fend_id:
                self.entity_by_d3fend_id[entity.d3fend_id] = entity.entity_id
            
            # Add to knowledge graph
            node_attrs = {
                'entity_type': entity.entity_type.value,
                'name': entity.name,
                'mitre_id': entity.mitre_id,
                'd3fend_id': entity.d3fend_id
            }
            
            # Add properties, avoiding conflicts with existing attributes
            for key, value in entity.properties.items():
                if key not in node_attrs:
                    node_attrs[key] = value
            
            self.knowledge_graph.add_node(entity.entity_id, **node_attrs)
            
            await self._save_entities()
            
            self.logger.debug(f"Added entity: {entity.name} ({entity.entity_type.value})")
            
        except Exception as e:
            self.logger.error(f"Failed to add entity: {e}")
            raise
    
    async def add_relation(self, relation: OntologyRelation) -> None:
        """Add a relation to the ontology"""
        try:
            # Validate entities exist
            if relation.source_entity not in self.entities:
                raise ValueError(f"Source entity {relation.source_entity} not found")
            if relation.target_entity not in self.entities:
                raise ValueError(f"Target entity {relation.target_entity} not found")
            
            self.relations[relation.relation_id] = relation
            
            # Update indices
            self.relations_by_type[relation.relation_type].add(relation.relation_id)
            
            # Add to knowledge graph
            edge_attrs = {
                'relation_type': relation.relation_type.value,
                'confidence': relation.confidence_score
            }
            
            # Add properties, avoiding conflicts with existing attributes
            for key, value in relation.properties.items():
                if key not in edge_attrs:
                    edge_attrs[key] = value
            
            self.knowledge_graph.add_edge(
                relation.source_entity,
                relation.target_entity,
                key=relation.relation_id,
                **edge_attrs
            )
            
            await self._save_relations()
            
            self.logger.debug(f"Added relation: {relation.source_entity} -> {relation.target_entity} ({relation.relation_type.value})")
            
        except Exception as e:
            self.logger.error(f"Failed to add relation: {e}")
            raise
    
    async def add_semantic_mapping(self, mapping: SemanticMapping) -> None:
        """Add a semantic mapping between frameworks"""
        try:
            self.mappings[mapping.mapping_id] = mapping
            await self._save_mappings()
            
            self.logger.debug(f"Added mapping: {mapping.source_framework}:{mapping.source_id} -> {mapping.target_framework}:{mapping.target_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add semantic mapping: {e}")
            raise
    
    async def get_entity_by_mitre_id(self, mitre_id: str) -> Optional[OntologyEntity]:
        """Get entity by MITRE ATT&CK ID"""
        try:
            entity_id = self.entity_by_mitre_id.get(mitre_id)
            if entity_id:
                return self.entities.get(entity_id)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get entity by MITRE ID: {e}")
            return None
    
    async def get_entity_by_d3fend_id(self, d3fend_id: str) -> Optional[OntologyEntity]:
        """Get entity by D3FEND ID"""
        try:
            entity_id = self.entity_by_d3fend_id.get(d3fend_id)
            if entity_id:
                return self.entities.get(entity_id)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get entity by D3FEND ID: {e}")
            return None
    
    async def get_entities_by_type(self, entity_type: EntityType) -> List[OntologyEntity]:
        """Get all entities of a specific type"""
        try:
            entity_ids = self.entity_by_type.get(entity_type, set())
            return [self.entities[entity_id] for entity_id in entity_ids if entity_id in self.entities]
            
        except Exception as e:
            self.logger.error(f"Failed to get entities by type: {e}")
            return []
    
    async def get_related_entities(self, entity_id: str, relation_types: Optional[List[RelationType]] = None, max_depth: int = 1) -> List[Tuple[OntologyEntity, List[OntologyRelation]]]:
        """Get entities related to a given entity"""
        try:
            if entity_id not in self.entities:
                return []
            
            related = []
            visited = set()
            
            def _traverse(current_id: str, depth: int, path: List[OntologyRelation]):
                if depth > max_depth or current_id in visited:
                    return
                
                visited.add(current_id)
                
                # Get outgoing relations
                for relation in self.relations.values():
                    if relation.source_entity == current_id:
                        if not relation_types or relation.relation_type in relation_types:
                            target_entity = self.entities.get(relation.target_entity)
                            if target_entity:
                                new_path = path + [relation]
                                related.append((target_entity, new_path))
                                
                                if depth < max_depth:
                                    _traverse(relation.target_entity, depth + 1, new_path)
            
            _traverse(entity_id, 0, [])
            return related
            
        except Exception as e:
            self.logger.error(f"Failed to get related entities: {e}")
            return []
    
    async def find_semantic_mappings(self, source_framework: str, source_id: str) -> List[SemanticMapping]:
        """Find semantic mappings for a given framework entity"""
        try:
            mappings = []
            for mapping in self.mappings.values():
                if mapping.source_framework == source_framework and mapping.source_id == source_id:
                    mappings.append(mapping)
            
            # Sort by confidence score
            mappings.sort(key=lambda x: x.confidence_score, reverse=True)
            return mappings
            
        except Exception as e:
            self.logger.error(f"Failed to find semantic mappings: {e}")
            return []
    
    async def infer_knowledge(self) -> List[Dict[str, Any]]:
        """Apply inference rules to derive new knowledge"""
        try:
            inferences = []
            
            for rule in self.inference_rules.values():
                if not rule.enabled:
                    continue
                
                # Apply rule conditions
                matches = await self._apply_rule_conditions(rule)
                
                for match in matches:
                    # Check confidence threshold
                    if match.get('confidence', 0.0) >= rule.confidence_threshold:
                        # Apply conclusions
                        conclusions = await self._apply_rule_conclusions(rule, match)
                        inferences.extend(conclusions)
            
            return inferences
            
        except Exception as e:
            self.logger.error(f"Failed to infer knowledge: {e}")
            return []
    
    async def update_from_simulation_outcome(self, outcome: Dict[str, Any]) -> None:
        """Update ontology based on simulation outcomes"""
        try:
            # Extract entities and relations from simulation outcome
            if 'attack_technique' in outcome:
                await self._process_attack_outcome(outcome)
            
            if 'defense_action' in outcome:
                await self._process_defense_outcome(outcome)
            
            if 'new_relationships' in outcome:
                await self._process_relationship_updates(outcome['new_relationships'])
            
            # Update confidence scores based on success/failure
            await self._update_confidence_scores(outcome)
            
            self.logger.debug("Updated ontology from simulation outcome")
            
        except Exception as e:
            self.logger.error(f"Failed to update from simulation outcome: {e}")
    
    async def get_ontology_statistics(self) -> Dict[str, Any]:
        """Get ontology statistics"""
        try:
            stats = {
                'entities': {
                    'total': len(self.entities),
                    'by_type': {entity_type.value: len(entity_ids) for entity_type, entity_ids in self.entity_by_type.items()}
                },
                'relations': {
                    'total': len(self.relations),
                    'by_type': {relation_type.value: len(relation_ids) for relation_type, relation_ids in self.relations_by_type.items()}
                },
                'mappings': {
                    'total': len(self.mappings),
                    'frameworks': list(set(mapping.source_framework for mapping in self.mappings.values()) | 
                                     set(mapping.target_framework for mapping in self.mappings.values()))
                },
                'inference_rules': {
                    'total': len(self.inference_rules),
                    'enabled': len([rule for rule in self.inference_rules.values() if rule.enabled])
                },
                'knowledge_graph': {
                    'nodes': self.knowledge_graph.number_of_nodes(),
                    'edges': self.knowledge_graph.number_of_edges(),
                    'density': nx.density(self.knowledge_graph) if self.knowledge_graph.number_of_nodes() > 0 else 0.0
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get ontology statistics: {e}")
            return {}
    
    async def export_knowledge_graph(self, format: str = "json") -> Dict[str, Any]:
        """Export knowledge graph in specified format"""
        try:
            if format == "json":
                return {
                    'nodes': [
                        {
                            'id': node_id,
                            **self.knowledge_graph.nodes[node_id]
                        }
                        for node_id in self.knowledge_graph.nodes()
                    ],
                    'edges': [
                        {
                            'source': source,
                            'target': target,
                            'key': key,
                            **edge_data
                        }
                        for source, target, key, edge_data in self.knowledge_graph.edges(keys=True, data=True)
                    ]
                }
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export knowledge graph: {e}")
            return {}
    
    async def _load_entities(self) -> None:
        """Load entities from file"""
        try:
            if self.entities_file.exists():
                with open(self.entities_file, 'r') as f:
                    data = json.load(f)
                    for entity_data in data:
                        entity = OntologyEntity(**entity_data)
                        entity.entity_type = EntityType(entity.entity_type)
                        entity.created_at = datetime.fromisoformat(entity.created_at)
                        entity.last_updated = datetime.fromisoformat(entity.last_updated)
                        self.entities[entity.entity_id] = entity
                
                self.logger.debug(f"Loaded {len(self.entities)} entities")
                
        except Exception as e:
            self.logger.error(f"Failed to load entities: {e}")
    
    async def _save_entities(self) -> None:
        """Save entities to file"""
        try:
            data = []
            for entity in self.entities.values():
                entity_dict = asdict(entity)
                entity_dict['entity_type'] = entity.entity_type.value
                entity_dict['created_at'] = entity.created_at.isoformat()
                entity_dict['last_updated'] = entity.last_updated.isoformat()
                data.append(entity_dict)
            
            with open(self.entities_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save entities: {e}")
    
    async def _load_relations(self) -> None:
        """Load relations from file"""
        try:
            if self.relations_file.exists():
                with open(self.relations_file, 'r') as f:
                    data = json.load(f)
                    for relation_data in data:
                        relation = OntologyRelation(**relation_data)
                        relation.relation_type = RelationType(relation.relation_type)
                        relation.created_at = datetime.fromisoformat(relation.created_at)
                        if relation.evidence is None:
                            relation.evidence = []
                        self.relations[relation.relation_id] = relation
                
                self.logger.debug(f"Loaded {len(self.relations)} relations")
                
        except Exception as e:
            self.logger.error(f"Failed to load relations: {e}")
    
    async def _save_relations(self) -> None:
        """Save relations to file"""
        try:
            data = []
            for relation in self.relations.values():
                relation_dict = asdict(relation)
                relation_dict['relation_type'] = relation.relation_type.value
                relation_dict['created_at'] = relation.created_at.isoformat()
                data.append(relation_dict)
            
            with open(self.relations_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save relations: {e}")
    
    async def _load_mappings(self) -> None:
        """Load semantic mappings from file"""
        try:
            if self.mappings_file.exists():
                with open(self.mappings_file, 'r') as f:
                    data = json.load(f)
                    for mapping_data in data:
                        mapping = SemanticMapping(**mapping_data)
                        mapping.created_at = datetime.fromisoformat(mapping.created_at)
                        self.mappings[mapping.mapping_id] = mapping
                
                self.logger.debug(f"Loaded {len(self.mappings)} mappings")
                
        except Exception as e:
            self.logger.error(f"Failed to load mappings: {e}")
    
    async def _save_mappings(self) -> None:
        """Save semantic mappings to file"""
        try:
            data = []
            for mapping in self.mappings.values():
                mapping_dict = asdict(mapping)
                mapping_dict['created_at'] = mapping.created_at.isoformat()
                data.append(mapping_dict)
            
            with open(self.mappings_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save mappings: {e}")
    
    async def _load_inference_rules(self) -> None:
        """Load inference rules from file"""
        try:
            if self.inference_rules_file.exists():
                with open(self.inference_rules_file, 'r') as f:
                    data = json.load(f)
                    for rule_data in data:
                        rule = InferenceRule(**rule_data)
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
                rule_dict['created_at'] = rule.created_at.isoformat()
                data.append(rule_dict)
            
            with open(self.inference_rules_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save inference rules: {e}")
    
    async def _initialize_default_ontology(self) -> None:
        """Initialize with default ontology entities and relations"""
        try:
            # Create default entities
            await self._create_default_entities()
            
            # Create default relations
            await self._create_default_relations()
            
            # Create default semantic mappings
            await self._create_default_mappings()
            
            # Create default inference rules
            await self._create_default_inference_rules()
            
            self.logger.info("Initialized default ontology")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default ontology: {e}")
    
    async def _create_default_entities(self) -> None:
        """Create default ontology entities"""
        try:
            # MITRE ATT&CK Tactics
            tactics = [
                ("reconnaissance", "Reconnaissance", "The adversary is trying to gather information they can use to plan future operations."),
                ("initial-access", "Initial Access", "The adversary is trying to get into your network."),
                ("execution", "Execution", "The adversary is trying to run malicious code."),
                ("persistence", "Persistence", "The adversary is trying to maintain their foothold."),
                ("privilege-escalation", "Privilege Escalation", "The adversary is trying to gain higher-level permissions."),
                ("defense-evasion", "Defense Evasion", "The adversary is trying to avoid being detected."),
                ("credential-access", "Credential Access", "The adversary is trying to steal account names and passwords."),
                ("discovery", "Discovery", "The adversary is trying to figure out your environment."),
                ("lateral-movement", "Lateral Movement", "The adversary is trying to move through your environment."),
                ("collection", "Collection", "The adversary is trying to gather data of interest to their goal."),
                ("command-and-control", "Command and Control", "The adversary is trying to communicate with compromised systems."),
                ("exfiltration", "Exfiltration", "The adversary is trying to steal data."),
                ("impact", "Impact", "The adversary is trying to manipulate, interrupt, or destroy your systems and data.")
            ]
            
            for tactic_id, name, description in tactics:
                entity = OntologyEntity(
                    entity_id=str(uuid.uuid4()),
                    entity_type=EntityType.TACTIC,
                    name=name,
                    description=description,
                    properties={"tactic_id": tactic_id, "framework": "mitre_attack"},
                    mitre_id=tactic_id
                )
                await self.add_entity(entity)
            
            # Sample attack techniques
            techniques = [
                ("T1595", "Active Scanning", "Adversaries may execute active reconnaissance scans to gather information.", "reconnaissance"),
                ("T1190", "Exploit Public-Facing Application", "Adversaries may attempt to exploit weaknesses in Internet-facing applications.", "initial-access"),
                ("T1059", "Command and Scripting Interpreter", "Adversaries may abuse command and script interpreters to execute commands.", "execution")
            ]
            
            for tech_id, name, description, tactic in techniques:
                entity = OntologyEntity(
                    entity_id=str(uuid.uuid4()),
                    entity_type=EntityType.ATTACK_TECHNIQUE,
                    name=name,
                    description=description,
                    properties={"technique_id": tech_id, "tactic": tactic, "framework": "mitre_attack"},
                    mitre_id=tech_id
                )
                await self.add_entity(entity)
            
            # Sample defense techniques (D3FEND)
            defense_techniques = [
                ("D3-NTA", "Network Traffic Analysis", "Analyzing network traffic to detect malicious activity."),
                ("D3-SYMON", "System Monitoring", "Monitoring system activities for signs of compromise."),
                ("D3-AL", "Application Logging", "Logging application events for security analysis.")
            ]
            
            for def_id, name, description in defense_techniques:
                entity = OntologyEntity(
                    entity_id=str(uuid.uuid4()),
                    entity_type=EntityType.DEFENSE_TECHNIQUE,
                    name=name,
                    description=description,
                    properties={"technique_id": def_id, "framework": "d3fend"},
                    d3fend_id=def_id
                )
                await self.add_entity(entity)
            
        except Exception as e:
            self.logger.error(f"Failed to create default entities: {e}")
    
    async def _create_default_relations(self) -> None:
        """Create default ontology relations"""
        try:
            # Find entities for creating relations
            recon_tactic = None
            active_scanning = None
            network_analysis = None
            
            for entity in self.entities.values():
                if entity.mitre_id == "reconnaissance":
                    recon_tactic = entity
                elif entity.mitre_id == "T1595":
                    active_scanning = entity
                elif entity.d3fend_id == "D3-NTA":
                    network_analysis = entity
            
            # Create relations
            if recon_tactic and active_scanning:
                relation = OntologyRelation(
                    relation_id=str(uuid.uuid4()),
                    source_entity=active_scanning.entity_id,
                    target_entity=recon_tactic.entity_id,
                    relation_type=RelationType.PART_OF,
                    properties={"description": "Active Scanning is part of Reconnaissance tactic"}
                )
                await self.add_relation(relation)
            
            if active_scanning and network_analysis:
                relation = OntologyRelation(
                    relation_id=str(uuid.uuid4()),
                    source_entity=network_analysis.entity_id,
                    target_entity=active_scanning.entity_id,
                    relation_type=RelationType.DETECTS,
                    properties={"description": "Network Traffic Analysis can detect Active Scanning"}
                )
                await self.add_relation(relation)
            
        except Exception as e:
            self.logger.error(f"Failed to create default relations: {e}")
    
    async def _create_default_mappings(self) -> None:
        """Create default semantic mappings"""
        try:
            # MITRE ATT&CK to D3FEND mappings
            mappings = [
                ("mitre_attack", "T1595", "d3fend", "D3-NTA", "detection", 0.8),
                ("mitre_attack", "T1190", "d3fend", "D3-AL", "detection", 0.7),
                ("mitre_attack", "T1059", "d3fend", "D3-SYMON", "detection", 0.9)
            ]
            
            for source_fw, source_id, target_fw, target_id, mapping_type, confidence in mappings:
                mapping = SemanticMapping(
                    mapping_id=str(uuid.uuid4()),
                    source_framework=source_fw,
                    target_framework=target_fw,
                    source_id=source_id,
                    target_id=target_id,
                    mapping_type=mapping_type,
                    confidence_score=confidence,
                    properties={"auto_generated": True}
                )
                await self.add_semantic_mapping(mapping)
            
        except Exception as e:
            self.logger.error(f"Failed to create default mappings: {e}")
    
    async def _create_default_inference_rules(self) -> None:
        """Create default inference rules"""
        try:
            # Rule: If technique A is detected by defense B, and A is similar to C, then B might detect C
            rule = InferenceRule(
                rule_id=str(uuid.uuid4()),
                name="Transitive Detection Rule",
                description="If a defense technique detects an attack technique, and that attack technique is similar to another, the defense might also detect the similar technique.",
                conditions=[
                    {"relation_type": "detects", "confidence_min": 0.7},
                    {"relation_type": "similar_to", "confidence_min": 0.6}
                ],
                conclusions=[
                    {"relation_type": "detects", "confidence_factor": 0.8}
                ],
                confidence_threshold=0.5
            )
            
            self.inference_rules[rule.rule_id] = rule
            await self._save_inference_rules()
            
        except Exception as e:
            self.logger.error(f"Failed to create default inference rules: {e}")
    
    async def _build_knowledge_graph(self) -> None:
        """Build NetworkX knowledge graph from entities and relations"""
        try:
            self.knowledge_graph.clear()
            
            # Add nodes
            for entity in self.entities.values():
                node_attrs = {
                    'entity_type': entity.entity_type.value,
                    'name': entity.name,
                    'mitre_id': entity.mitre_id,
                    'd3fend_id': entity.d3fend_id
                }
                
                # Add properties, avoiding conflicts with existing attributes
                for key, value in entity.properties.items():
                    if key not in node_attrs:
                        node_attrs[key] = value
                
                self.knowledge_graph.add_node(entity.entity_id, **node_attrs)
            
            # Add edges
            for relation in self.relations.values():
                if (relation.source_entity in self.entities and 
                    relation.target_entity in self.entities):
                    edge_attrs = {
                        'relation_type': relation.relation_type.value,
                        'confidence': relation.confidence_score
                    }
                    
                    # Add properties, avoiding conflicts with existing attributes
                    for key, value in relation.properties.items():
                        if key not in edge_attrs:
                            edge_attrs[key] = value
                    
                    self.knowledge_graph.add_edge(
                        relation.source_entity,
                        relation.target_entity,
                        key=relation.relation_id,
                        **edge_attrs
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to build knowledge graph: {e}")
    
    async def _build_semantic_indices(self) -> None:
        """Build semantic indices for fast lookup"""
        try:
            # Clear existing indices
            self.entity_by_type.clear()
            self.entity_by_mitre_id.clear()
            self.entity_by_d3fend_id.clear()
            self.relations_by_type.clear()
            
            # Build entity indices
            for entity_id, entity in self.entities.items():
                self.entity_by_type[entity.entity_type].add(entity_id)
                if entity.mitre_id:
                    self.entity_by_mitre_id[entity.mitre_id] = entity_id
                if entity.d3fend_id:
                    self.entity_by_d3fend_id[entity.d3fend_id] = entity_id
            
            # Build relation indices
            for relation_id, relation in self.relations.items():
                self.relations_by_type[relation.relation_type].add(relation_id)
            
        except Exception as e:
            self.logger.error(f"Failed to build semantic indices: {e}")
    
    async def _apply_rule_conditions(self, rule: InferenceRule) -> List[Dict[str, Any]]:
        """Apply rule conditions to find matches"""
        try:
            matches = []
            # Simplified rule application - in practice this would be more sophisticated
            return matches
            
        except Exception as e:
            self.logger.error(f"Failed to apply rule conditions: {e}")
            return []
    
    async def _apply_rule_conclusions(self, rule: InferenceRule, match: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply rule conclusions to generate new knowledge"""
        try:
            conclusions = []
            # Simplified conclusion application
            return conclusions
            
        except Exception as e:
            self.logger.error(f"Failed to apply rule conclusions: {e}")
            return []
    
    async def _process_attack_outcome(self, outcome: Dict[str, Any]) -> None:
        """Process attack outcome to update ontology"""
        try:
            technique_id = outcome.get('attack_technique')
            success = outcome.get('success', False)
            
            # Update confidence scores based on success
            entity = await self.get_entity_by_mitre_id(technique_id)
            if entity and 'success_rate' in entity.properties:
                current_rate = entity.properties['success_rate']
                # Simple update - in practice would use more sophisticated learning
                if success:
                    entity.properties['success_rate'] = min(1.0, current_rate + 0.1)
                else:
                    entity.properties['success_rate'] = max(0.0, current_rate - 0.05)
                
                entity.last_updated = datetime.now()
                await self._save_entities()
            
        except Exception as e:
            self.logger.error(f"Failed to process attack outcome: {e}")
    
    async def _process_defense_outcome(self, outcome: Dict[str, Any]) -> None:
        """Process defense outcome to update ontology"""
        try:
            defense_action = outcome.get('defense_action')
            effectiveness = outcome.get('effectiveness', 0.5)
            
            # Find corresponding defense entity and update effectiveness
            for entity in self.entities.values():
                if (entity.entity_type == EntityType.DEFENSE_TECHNIQUE and 
                    defense_action in entity.name.lower()):
                    if 'effectiveness_score' in entity.properties:
                        current_score = entity.properties['effectiveness_score']
                        # Weighted average update
                        entity.properties['effectiveness_score'] = (current_score * 0.9 + effectiveness * 0.1)
                        entity.last_updated = datetime.now()
                        await self._save_entities()
                    break
            
        except Exception as e:
            self.logger.error(f"Failed to process defense outcome: {e}")
    
    async def _process_relationship_updates(self, relationships: List[Dict[str, Any]]) -> None:
        """Process relationship updates from simulation"""
        try:
            for rel_data in relationships:
                source_id = rel_data.get('source')
                target_id = rel_data.get('target')
                rel_type = rel_data.get('type')
                confidence = rel_data.get('confidence', 0.5)
                
                # Check if relation already exists
                existing_relation = None
                for relation in self.relations.values():
                    if (relation.source_entity == source_id and 
                        relation.target_entity == target_id and 
                        relation.relation_type.value == rel_type):
                        existing_relation = relation
                        break
                
                if existing_relation:
                    # Update confidence
                    existing_relation.confidence_score = (existing_relation.confidence_score + confidence) / 2
                else:
                    # Create new relation
                    relation = OntologyRelation(
                        relation_id=str(uuid.uuid4()),
                        source_entity=source_id,
                        target_entity=target_id,
                        relation_type=RelationType(rel_type),
                        properties={"auto_generated": True},
                        confidence_score=confidence
                    )
                    await self.add_relation(relation)
            
        except Exception as e:
            self.logger.error(f"Failed to process relationship updates: {e}")
    
    async def _update_confidence_scores(self, outcome: Dict[str, Any]) -> None:
        """Update confidence scores based on simulation outcome"""
        try:
            # Update entity and relation confidence scores based on validation
            success = outcome.get('success', False)
            entities_involved = outcome.get('entities', [])
            
            for entity_id in entities_involved:
                if entity_id in self.entities:
                    entity = self.entities[entity_id]
                    # Adjust confidence based on successful validation
                    if success:
                        entity.confidence_score = min(1.0, entity.confidence_score + 0.05)
                    else:
                        entity.confidence_score = max(0.1, entity.confidence_score - 0.02)
                    
                    entity.last_updated = datetime.now()
            
            await self._save_entities()
            
        except Exception as e:
            self.logger.error(f"Failed to update confidence scores: {e}")