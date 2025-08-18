#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Semantic Mapper
Semantic entity mapping to MITRE ATT&CK and D3FEND frameworks
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
# import requests  # Commented out - not used in current implementation
# from urllib.parse import urljoin  # Commented out - not used in current implementation

logger = logging.getLogger(__name__)

class FrameworkType(Enum):
    """Supported cybersecurity frameworks"""
    MITRE_ATTACK = "mitre_attack"
    D3FEND = "d3fend"
    NIST_CSF = "nist_csf"
    CIS_CONTROLS = "cis_controls"
    CAPEC = "capec"

class MappingConfidence(Enum):
    """Mapping confidence levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

@dataclass
class FrameworkEntity:
    """Entity from a cybersecurity framework"""
    framework: FrameworkType
    entity_id: str
    name: str
    description: str
    category: str
    properties: Dict[str, Any]
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class SemanticMapping:
    """Semantic mapping between framework entities"""
    mapping_id: str
    source_framework: FrameworkType
    source_entity_id: str
    target_framework: FrameworkType
    target_entity_id: str
    mapping_type: str
    confidence: MappingConfidence
    confidence_score: float
    evidence: List[str]
    properties: Dict[str, Any]
    created_at: datetime = None
    validated: bool = False
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class MappingRule:
    """Rule for automatic semantic mapping"""
    rule_id: str
    name: str
    description: str
    source_framework: FrameworkType
    target_framework: FrameworkType
    conditions: List[Dict[str, Any]]
    mapping_logic: Dict[str, Any]
    confidence_threshold: float
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class SemanticMapper:
    """
    Semantic mapper for cybersecurity frameworks.
    
    Features:
    - MITRE ATT&CK framework integration
    - D3FEND framework integration
    - Automatic semantic mapping generation
    - Confidence scoring for mappings
    - Validation and refinement of mappings
    """
    
    def __init__(self, data_dir: str = "./knowledge_base/semantic_mappings"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Framework data storage
        self.framework_entities: Dict[FrameworkType, Dict[str, FrameworkEntity]] = {
            framework: {} for framework in FrameworkType
        }
        
        # Semantic mappings
        self.mappings: Dict[str, SemanticMapping] = {}
        self.mapping_rules: Dict[str, MappingRule] = {}
        
        # Mapping indices for fast lookup
        self.mappings_by_source: Dict[Tuple[FrameworkType, str], List[str]] = {}
        self.mappings_by_target: Dict[Tuple[FrameworkType, str], List[str]] = {}
        
        # File paths
        self.mitre_entities_file = self.data_dir / "mitre_entities.json"
        self.d3fend_entities_file = self.data_dir / "d3fend_entities.json"
        self.mappings_file = self.data_dir / "semantic_mappings.json"
        self.mapping_rules_file = self.data_dir / "mapping_rules.json"
        
        # Framework URLs for updates
        self.framework_urls = {
            FrameworkType.MITRE_ATTACK: "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
            FrameworkType.D3FEND: "https://d3fend.mitre.org/api/ontology/d3fend.json"
        }
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the semantic mapper"""
        try:
            self.logger.info("Initializing semantic mapper")
            
            # Load existing data
            await self._load_framework_entities()
            await self._load_mappings()
            await self._load_mapping_rules()
            
            # Initialize default data if empty
            if not self.framework_entities[FrameworkType.MITRE_ATTACK]:
                await self._initialize_default_mitre_data()
            
            if not self.framework_entities[FrameworkType.D3FEND]:
                await self._initialize_default_d3fend_data()
            
            if not self.mapping_rules:
                await self._initialize_default_mapping_rules()
            
            # Build mapping indices
            await self._build_mapping_indices()
            
            self.initialized = True
            self.logger.info("Semantic mapper initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize semantic mapper: {e}")
            raise
    
    async def add_framework_entity(self, entity: FrameworkEntity) -> None:
        """Add a framework entity"""
        try:
            self.framework_entities[entity.framework][entity.entity_id] = entity
            await self._save_framework_entities(entity.framework)
            
            self.logger.debug(f"Added {entity.framework.value} entity: {entity.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to add framework entity: {e}")
            raise
    
    async def add_semantic_mapping(self, mapping: SemanticMapping) -> None:
        """Add a semantic mapping"""
        try:
            self.mappings[mapping.mapping_id] = mapping
            await self._save_mappings()
            await self._update_mapping_indices(mapping)
            
            self.logger.debug(f"Added mapping: {mapping.source_framework.value}:{mapping.source_entity_id} -> {mapping.target_framework.value}:{mapping.target_entity_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add semantic mapping: {e}")
            raise
    
    async def find_mappings(self, framework: FrameworkType, entity_id: str, direction: str = "both") -> List[SemanticMapping]:
        """Find semantic mappings for a framework entity"""
        try:
            mappings = []
            
            if direction in ["source", "both"]:
                # Find mappings where this entity is the source
                mapping_ids = self.mappings_by_source.get((framework, entity_id), [])
                mappings.extend([self.mappings[mid] for mid in mapping_ids if mid in self.mappings])
            
            if direction in ["target", "both"]:
                # Find mappings where this entity is the target
                mapping_ids = self.mappings_by_target.get((framework, entity_id), [])
                mappings.extend([self.mappings[mid] for mid in mapping_ids if mid in self.mappings])
            
            # Sort by confidence score
            mappings.sort(key=lambda x: x.confidence_score, reverse=True)
            return mappings
            
        except Exception as e:
            self.logger.error(f"Failed to find mappings: {e}")
            return []
    
    async def generate_automatic_mappings(self, source_framework: FrameworkType, target_framework: FrameworkType) -> List[SemanticMapping]:
        """Generate automatic semantic mappings using rules"""
        try:
            new_mappings = []
            
            # Get applicable mapping rules
            applicable_rules = [
                rule for rule in self.mapping_rules.values()
                if (rule.source_framework == source_framework and 
                    rule.target_framework == target_framework and 
                    rule.enabled)
            ]
            
            source_entities = self.framework_entities[source_framework]
            target_entities = self.framework_entities[target_framework]
            
            for rule in applicable_rules:
                for source_id, source_entity in source_entities.items():
                    for target_id, target_entity in target_entities.items():
                        # Check if mapping already exists
                        existing_mappings = await self.find_mappings(source_framework, source_id, "source")
                        if any(m.target_framework == target_framework and m.target_entity_id == target_id 
                               for m in existing_mappings):
                            continue
                        
                        # Apply mapping rule
                        mapping_result = await self._apply_mapping_rule(rule, source_entity, target_entity)
                        if mapping_result and mapping_result['confidence'] >= rule.confidence_threshold:
                            mapping = SemanticMapping(
                                mapping_id=str(uuid.uuid4()),
                                source_framework=source_framework,
                                source_entity_id=source_id,
                                target_framework=target_framework,
                                target_entity_id=target_id,
                                mapping_type=mapping_result['type'],
                                confidence=self._score_to_confidence(mapping_result['confidence']),
                                confidence_score=mapping_result['confidence'],
                                evidence=mapping_result.get('evidence', []),
                                properties={"auto_generated": True, "rule_id": rule.rule_id}
                            )
                            new_mappings.append(mapping)
            
            # Add new mappings
            for mapping in new_mappings:
                await self.add_semantic_mapping(mapping)
            
            self.logger.info(f"Generated {len(new_mappings)} automatic mappings from {source_framework.value} to {target_framework.value}")
            return new_mappings
            
        except Exception as e:
            self.logger.error(f"Failed to generate automatic mappings: {e}")
            return []
    
    async def validate_mapping(self, mapping_id: str, validation_result: bool, evidence: Optional[str] = None) -> None:
        """Validate a semantic mapping"""
        try:
            if mapping_id not in self.mappings:
                raise ValueError(f"Mapping {mapping_id} not found")
            
            mapping = self.mappings[mapping_id]
            mapping.validated = validation_result
            
            if evidence:
                mapping.evidence.append(evidence)
            
            # Adjust confidence based on validation
            if validation_result:
                mapping.confidence_score = min(1.0, mapping.confidence_score + 0.1)
                if mapping.confidence_score >= 0.8:
                    mapping.confidence = MappingConfidence.HIGH
                elif mapping.confidence_score >= 0.6:
                    mapping.confidence = MappingConfidence.MEDIUM
            else:
                mapping.confidence_score = max(0.0, mapping.confidence_score - 0.2)
                if mapping.confidence_score < 0.3:
                    mapping.confidence = MappingConfidence.UNCERTAIN
                elif mapping.confidence_score < 0.6:
                    mapping.confidence = MappingConfidence.LOW
            
            await self._save_mappings()
            
            self.logger.debug(f"Validated mapping {mapping_id}: {validation_result}")
            
        except Exception as e:
            self.logger.error(f"Failed to validate mapping: {e}")
    
    async def update_framework_data(self, framework: FrameworkType, force_update: bool = False) -> None:
        """Update framework data from official sources"""
        try:
            if framework not in self.framework_urls:
                self.logger.warning(f"No update URL configured for {framework.value}")
                return
            
            # Check if update is needed
            if not force_update:
                last_update = await self._get_last_framework_update(framework)
                if last_update and (datetime.now() - last_update).days < 7:
                    self.logger.debug(f"Framework {framework.value} updated recently, skipping")
                    return
            
            self.logger.info(f"Updating {framework.value} framework data")
            
            if framework == FrameworkType.MITRE_ATTACK:
                await self._update_mitre_attack_data()
            elif framework == FrameworkType.D3FEND:
                await self._update_d3fend_data()
            
            await self._save_framework_update_timestamp(framework)
            
        except Exception as e:
            self.logger.error(f"Failed to update framework data: {e}")
    
    async def get_mapping_statistics(self) -> Dict[str, Any]:
        """Get semantic mapping statistics"""
        try:
            stats = {
                'total_mappings': len(self.mappings),
                'by_confidence': {
                    confidence.value: len([m for m in self.mappings.values() if m.confidence == confidence])
                    for confidence in MappingConfidence
                },
                'by_framework_pair': {},
                'validated_mappings': len([m for m in self.mappings.values() if m.validated]),
                'auto_generated': len([m for m in self.mappings.values() if m.properties.get('auto_generated', False)]),
                'framework_entities': {
                    framework.value: len(entities)
                    for framework, entities in self.framework_entities.items()
                }
            }
            
            # Count mappings by framework pair
            for mapping in self.mappings.values():
                pair_key = f"{mapping.source_framework.value} -> {mapping.target_framework.value}"
                stats['by_framework_pair'][pair_key] = stats['by_framework_pair'].get(pair_key, 0) + 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get mapping statistics: {e}")
            return {}
    
    async def export_mappings(self, format: str = "json", framework_filter: Optional[List[FrameworkType]] = None) -> Dict[str, Any]:
        """Export semantic mappings in specified format"""
        try:
            mappings_to_export = []
            
            for mapping in self.mappings.values():
                if framework_filter:
                    if (mapping.source_framework not in framework_filter and 
                        mapping.target_framework not in framework_filter):
                        continue
                
                mapping_dict = asdict(mapping)
                mapping_dict['source_framework'] = mapping.source_framework.value
                mapping_dict['target_framework'] = mapping.target_framework.value
                mapping_dict['confidence'] = mapping.confidence.value
                mapping_dict['created_at'] = mapping.created_at.isoformat()
                mappings_to_export.append(mapping_dict)
            
            if format == "json":
                return {
                    'mappings': mappings_to_export,
                    'metadata': {
                        'export_timestamp': datetime.now().isoformat(),
                        'total_mappings': len(mappings_to_export),
                        'frameworks_included': [f.value for f in framework_filter] if framework_filter else "all"
                    }
                }
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export mappings: {e}")
            return {}
    
    async def _load_framework_entities(self) -> None:
        """Load framework entities from files"""
        try:
            # Load MITRE ATT&CK entities
            if self.mitre_entities_file.exists():
                with open(self.mitre_entities_file, 'r') as f:
                    data = json.load(f)
                    for entity_data in data:
                        entity = FrameworkEntity(**entity_data)
                        entity.framework = FrameworkType(entity.framework)
                        entity.last_updated = datetime.fromisoformat(entity.last_updated)
                        self.framework_entities[FrameworkType.MITRE_ATTACK][entity.entity_id] = entity
            
            # Load D3FEND entities
            if self.d3fend_entities_file.exists():
                with open(self.d3fend_entities_file, 'r') as f:
                    data = json.load(f)
                    for entity_data in data:
                        entity = FrameworkEntity(**entity_data)
                        entity.framework = FrameworkType(entity.framework)
                        entity.last_updated = datetime.fromisoformat(entity.last_updated)
                        self.framework_entities[FrameworkType.D3FEND][entity.entity_id] = entity
            
            total_entities = sum(len(entities) for entities in self.framework_entities.values())
            self.logger.debug(f"Loaded {total_entities} framework entities")
            
        except Exception as e:
            self.logger.error(f"Failed to load framework entities: {e}")
    
    async def _save_framework_entities(self, framework: FrameworkType) -> None:
        """Save framework entities to file"""
        try:
            entities = self.framework_entities[framework]
            data = []
            
            for entity in entities.values():
                entity_dict = asdict(entity)
                entity_dict['framework'] = entity.framework.value
                entity_dict['last_updated'] = entity.last_updated.isoformat()
                data.append(entity_dict)
            
            if framework == FrameworkType.MITRE_ATTACK:
                filepath = self.mitre_entities_file
            elif framework == FrameworkType.D3FEND:
                filepath = self.d3fend_entities_file
            else:
                filepath = self.data_dir / f"{framework.value}_entities.json"
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save framework entities: {e}")
    
    async def _load_mappings(self) -> None:
        """Load semantic mappings from file"""
        try:
            if self.mappings_file.exists():
                with open(self.mappings_file, 'r') as f:
                    data = json.load(f)
                    for mapping_data in data:
                        mapping = SemanticMapping(**mapping_data)
                        mapping.source_framework = FrameworkType(mapping.source_framework)
                        mapping.target_framework = FrameworkType(mapping.target_framework)
                        mapping.confidence = MappingConfidence(mapping.confidence)
                        mapping.created_at = datetime.fromisoformat(mapping.created_at)
                        self.mappings[mapping.mapping_id] = mapping
                
                self.logger.debug(f"Loaded {len(self.mappings)} semantic mappings")
                
        except Exception as e:
            self.logger.error(f"Failed to load mappings: {e}")
    
    async def _save_mappings(self) -> None:
        """Save semantic mappings to file"""
        try:
            data = []
            for mapping in self.mappings.values():
                mapping_dict = asdict(mapping)
                mapping_dict['source_framework'] = mapping.source_framework.value
                mapping_dict['target_framework'] = mapping.target_framework.value
                mapping_dict['confidence'] = mapping.confidence.value
                mapping_dict['created_at'] = mapping.created_at.isoformat()
                data.append(mapping_dict)
            
            with open(self.mappings_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save mappings: {e}")
    
    async def _load_mapping_rules(self) -> None:
        """Load mapping rules from file"""
        try:
            if self.mapping_rules_file.exists():
                with open(self.mapping_rules_file, 'r') as f:
                    data = json.load(f)
                    for rule_data in data:
                        rule = MappingRule(**rule_data)
                        rule.source_framework = FrameworkType(rule.source_framework)
                        rule.target_framework = FrameworkType(rule.target_framework)
                        rule.created_at = datetime.fromisoformat(rule.created_at)
                        self.mapping_rules[rule.rule_id] = rule
                
                self.logger.debug(f"Loaded {len(self.mapping_rules)} mapping rules")
                
        except Exception as e:
            self.logger.error(f"Failed to load mapping rules: {e}")
    
    async def _save_mapping_rules(self) -> None:
        """Save mapping rules to file"""
        try:
            data = []
            for rule in self.mapping_rules.values():
                rule_dict = asdict(rule)
                rule_dict['source_framework'] = rule.source_framework.value
                rule_dict['target_framework'] = rule.target_framework.value
                rule_dict['created_at'] = rule.created_at.isoformat()
                data.append(rule_dict)
            
            with open(self.mapping_rules_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save mapping rules: {e}")
    
    async def _initialize_default_mitre_data(self) -> None:
        """Initialize default MITRE ATT&CK data"""
        try:
            # Sample MITRE ATT&CK techniques
            default_techniques = [
                {
                    "entity_id": "T1595",
                    "name": "Active Scanning",
                    "description": "Adversaries may execute active reconnaissance scans to gather information that can be used during targeting.",
                    "category": "reconnaissance",
                    "properties": {
                        "tactics": ["reconnaissance"],
                        "platforms": ["PRE"],
                        "data_sources": ["Network Traffic: Network Traffic Flow"],
                        "mitigations": ["M1056: Pre-compromise"]
                    }
                },
                {
                    "entity_id": "T1190",
                    "name": "Exploit Public-Facing Application",
                    "description": "Adversaries may attempt to take advantage of a weakness in an Internet-facing computer or program using software, data, or commands in order to cause unintended or unanticipated behavior.",
                    "category": "initial-access",
                    "properties": {
                        "tactics": ["initial-access"],
                        "platforms": ["Linux", "Windows", "macOS", "Network"],
                        "data_sources": ["Application Log: Application Log Content", "Network Traffic: Network Traffic Content"],
                        "mitigations": ["M1048: Application Isolation and Sandboxing", "M1030: Network Segmentation"]
                    }
                },
                {
                    "entity_id": "T1059",
                    "name": "Command and Scripting Interpreter",
                    "description": "Adversaries may abuse command and script interpreters to execute commands, scripts, or binaries.",
                    "category": "execution",
                    "properties": {
                        "tactics": ["execution"],
                        "platforms": ["Linux", "macOS", "Windows"],
                        "data_sources": ["Command: Command Execution", "Process: Process Creation"],
                        "mitigations": ["M1038: Execution Prevention", "M1042: Disable or Remove Feature or Program"]
                    }
                }
            ]
            
            for tech_data in default_techniques:
                entity = FrameworkEntity(
                    framework=FrameworkType.MITRE_ATTACK,
                    **tech_data
                )
                await self.add_framework_entity(entity)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default MITRE data: {e}")
    
    async def _initialize_default_d3fend_data(self) -> None:
        """Initialize default D3FEND data"""
        try:
            # Sample D3FEND techniques
            default_techniques = [
                {
                    "entity_id": "D3-NTA",
                    "name": "Network Traffic Analysis",
                    "description": "Analyzing network traffic to detect malicious activity and policy violations.",
                    "category": "detect",
                    "properties": {
                        "definition": "The analysis of network traffic in order to detect anomalies and threats.",
                        "synonyms": ["Network Monitoring", "Traffic Analysis"],
                        "related_offensive_techniques": ["T1595", "T1046"]
                    }
                },
                {
                    "entity_id": "D3-SYMON",
                    "name": "System Monitoring",
                    "description": "Monitoring system activities and processes for signs of compromise.",
                    "category": "detect",
                    "properties": {
                        "definition": "The monitoring of system activities to detect malicious behavior.",
                        "synonyms": ["Host Monitoring", "Endpoint Monitoring"],
                        "related_offensive_techniques": ["T1059", "T1055"]
                    }
                },
                {
                    "entity_id": "D3-AL",
                    "name": "Application Logging",
                    "description": "Logging application events and activities for security analysis.",
                    "category": "detect",
                    "properties": {
                        "definition": "The logging of application events for security monitoring and analysis.",
                        "synonyms": ["App Logging", "Software Logging"],
                        "related_offensive_techniques": ["T1190", "T1068"]
                    }
                }
            ]
            
            for tech_data in default_techniques:
                entity = FrameworkEntity(
                    framework=FrameworkType.D3FEND,
                    **tech_data
                )
                await self.add_framework_entity(entity)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default D3FEND data: {e}")
    
    async def _initialize_default_mapping_rules(self) -> None:
        """Initialize default mapping rules"""
        try:
            # Rule for mapping MITRE ATT&CK to D3FEND based on keywords
            keyword_rule = MappingRule(
                rule_id=str(uuid.uuid4()),
                name="Keyword-based MITRE to D3FEND Mapping",
                description="Maps MITRE ATT&CK techniques to D3FEND techniques based on keyword similarity",
                source_framework=FrameworkType.MITRE_ATTACK,
                target_framework=FrameworkType.D3FEND,
                conditions=[
                    {"type": "keyword_similarity", "threshold": 0.3},
                    {"type": "category_match", "categories": ["network", "system", "application"]}
                ],
                mapping_logic={
                    "type": "detection",
                    "confidence_calculation": "keyword_similarity * 0.7 + category_bonus * 0.3"
                },
                confidence_threshold=0.5
            )
            
            self.mapping_rules[keyword_rule.rule_id] = keyword_rule
            await self._save_mapping_rules()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default mapping rules: {e}")
    
    async def _build_mapping_indices(self) -> None:
        """Build mapping indices for fast lookup"""
        try:
            self.mappings_by_source.clear()
            self.mappings_by_target.clear()
            
            for mapping_id, mapping in self.mappings.items():
                # Source index
                source_key = (mapping.source_framework, mapping.source_entity_id)
                if source_key not in self.mappings_by_source:
                    self.mappings_by_source[source_key] = []
                self.mappings_by_source[source_key].append(mapping_id)
                
                # Target index
                target_key = (mapping.target_framework, mapping.target_entity_id)
                if target_key not in self.mappings_by_target:
                    self.mappings_by_target[target_key] = []
                self.mappings_by_target[target_key].append(mapping_id)
            
        except Exception as e:
            self.logger.error(f"Failed to build mapping indices: {e}")
    
    async def _update_mapping_indices(self, mapping: SemanticMapping) -> None:
        """Update mapping indices for a new mapping"""
        try:
            # Update source index
            source_key = (mapping.source_framework, mapping.source_entity_id)
            if source_key not in self.mappings_by_source:
                self.mappings_by_source[source_key] = []
            self.mappings_by_source[source_key].append(mapping.mapping_id)
            
            # Update target index
            target_key = (mapping.target_framework, mapping.target_entity_id)
            if target_key not in self.mappings_by_target:
                self.mappings_by_target[target_key] = []
            self.mappings_by_target[target_key].append(mapping.mapping_id)
            
        except Exception as e:
            self.logger.error(f"Failed to update mapping indices: {e}")
    
    async def _apply_mapping_rule(self, rule: MappingRule, source_entity: FrameworkEntity, target_entity: FrameworkEntity) -> Optional[Dict[str, Any]]:
        """Apply a mapping rule to determine if entities should be mapped"""
        try:
            confidence = 0.0
            evidence = []
            
            # Apply conditions
            for condition in rule.conditions:
                if condition['type'] == 'keyword_similarity':
                    similarity = self._calculate_keyword_similarity(source_entity, target_entity)
                    if similarity >= condition['threshold']:
                        confidence += similarity * 0.5
                        evidence.append(f"Keyword similarity: {similarity:.2f}")
                
                elif condition['type'] == 'category_match':
                    if self._check_category_match(source_entity, target_entity, condition['categories']):
                        confidence += 0.3
                        evidence.append("Category match found")
            
            if confidence > 0:
                return {
                    'type': rule.mapping_logic.get('type', 'related'),
                    'confidence': min(1.0, confidence),
                    'evidence': evidence
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to apply mapping rule: {e}")
            return None
    
    def _calculate_keyword_similarity(self, entity1: FrameworkEntity, entity2: FrameworkEntity) -> float:
        """Calculate keyword similarity between two entities"""
        try:
            # Simple keyword-based similarity
            text1 = f"{entity1.name} {entity1.description}".lower()
            text2 = f"{entity2.name} {entity2.description}".lower()
            
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            # Remove common words
            common_words = {'the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'by', 'from', 'a', 'an'}
            words1 -= common_words
            words2 -= common_words
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate keyword similarity: {e}")
            return 0.0
    
    def _check_category_match(self, entity1: FrameworkEntity, entity2: FrameworkEntity, categories: List[str]) -> bool:
        """Check if entities have matching categories"""
        try:
            category1 = entity1.category.lower()
            category2 = entity2.category.lower()
            
            for category in categories:
                if category.lower() in category1 or category.lower() in category2:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check category match: {e}")
            return False
    
    def _score_to_confidence(self, score: float) -> MappingConfidence:
        """Convert numeric score to confidence level"""
        if score >= 0.8:
            return MappingConfidence.HIGH
        elif score >= 0.6:
            return MappingConfidence.MEDIUM
        elif score >= 0.3:
            return MappingConfidence.LOW
        else:
            return MappingConfidence.UNCERTAIN
    
    async def _get_last_framework_update(self, framework: FrameworkType) -> Optional[datetime]:
        """Get last framework update timestamp"""
        try:
            timestamp_file = self.data_dir / f"{framework.value}_last_update.txt"
            if timestamp_file.exists():
                with open(timestamp_file, 'r') as f:
                    timestamp_str = f.read().strip()
                    return datetime.fromisoformat(timestamp_str)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get last framework update: {e}")
            return None
    
    async def _save_framework_update_timestamp(self, framework: FrameworkType) -> None:
        """Save framework update timestamp"""
        try:
            timestamp_file = self.data_dir / f"{framework.value}_last_update.txt"
            with open(timestamp_file, 'w') as f:
                f.write(datetime.now().isoformat())
                
        except Exception as e:
            self.logger.error(f"Failed to save framework update timestamp: {e}")
    
    async def _update_mitre_attack_data(self) -> None:
        """Update MITRE ATT&CK data from official source"""
        try:
            # In a real implementation, this would fetch from the MITRE ATT&CK STIX data
            # For now, we'll just log that an update would occur
            self.logger.info("MITRE ATT&CK data update would be performed here")
            
        except Exception as e:
            self.logger.error(f"Failed to update MITRE ATT&CK data: {e}")
    
    async def _update_d3fend_data(self) -> None:
        """Update D3FEND data from official source"""
        try:
            # In a real implementation, this would fetch from the D3FEND API
            # For now, we'll just log that an update would occur
            self.logger.info("D3FEND data update would be performed here")
            
        except Exception as e:
            self.logger.error(f"Failed to update D3FEND data: {e}")