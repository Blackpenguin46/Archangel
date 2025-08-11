#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Knowledge Base
MITRE ATT&CK integration and tactical knowledge storage
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

logger = logging.getLogger(__name__)

class TacticType(Enum):
    """MITRE ATT&CK Tactic Types"""
    RECONNAISSANCE = "reconnaissance"
    RESOURCE_DEVELOPMENT = "resource-development"
    INITIAL_ACCESS = "initial-access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege-escalation"
    DEFENSE_EVASION = "defense-evasion"
    CREDENTIAL_ACCESS = "credential-access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral-movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command-and-control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"

class DefenseTactic(Enum):
    """Defensive Tactics"""
    DETECT = "detect"
    DENY = "deny"
    DISRUPT = "disrupt"
    DEGRADE = "degrade"
    DECEIVE = "deceive"
    CONTAIN = "contain"

@dataclass
class MitreAttackInfo:
    """MITRE ATT&CK technique information"""
    technique_id: str
    technique_name: str
    tactic: TacticType
    description: str
    platforms: List[str]
    data_sources: List[str]
    mitigations: List[str]
    detection_methods: List[str]
    references: List[str]
    last_updated: datetime

@dataclass
class AttackPattern:
    """Attack pattern with MITRE mapping"""
    pattern_id: str
    name: str
    description: str
    mitre_techniques: List[str]
    tactics: List[TacticType]
    indicators: List[str]
    countermeasures: List[str]
    success_rate: float
    difficulty_level: str
    created_at: datetime
    last_used: Optional[datetime]
    metadata: Dict[str, Any]

@dataclass
class DefenseStrategy:
    """Defense strategy with D3FEND mapping"""
    strategy_id: str
    name: str
    description: str
    defense_tactics: List[DefenseTactic]
    mitre_mitigations: List[str]
    effectiveness_score: float
    implementation_complexity: str
    tools_required: List[str]
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any]

@dataclass
class TTP:
    """Tactics, Techniques, and Procedures"""
    ttp_id: str
    name: str
    tactic: TacticType
    techniques: List[str]
    procedures: List[str]
    mitre_mapping: List[str]
    success_indicators: List[str]
    failure_indicators: List[str]
    prerequisites: List[str]
    artifacts: List[str]
    created_at: datetime

@dataclass
class Lesson:
    """Learned lesson from scenarios"""
    lesson_id: str
    scenario_type: str
    lesson_text: str
    category: str
    importance_score: float
    applicable_roles: List[str]
    mitre_techniques: List[str]
    created_at: datetime
    validated: bool
    metadata: Dict[str, Any]

class KnowledgeGraph:
    """Knowledge graph representation"""
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.node_types: Set[str] = set()
        self.edge_types: Set[str] = set()

class KnowledgeBase:
    """
    Knowledge base for storing and retrieving cybersecurity knowledge.
    
    Features:
    - MITRE ATT&CK framework integration
    - Attack pattern storage and retrieval
    - Defense strategy management
    - TTP (Tactics, Techniques, Procedures) mapping
    - Lessons learned storage
    - Knowledge graph generation
    """
    
    def __init__(self, data_dir: str = "./knowledge_base"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Storage
        self.attack_patterns: Dict[str, AttackPattern] = {}
        self.defense_strategies: Dict[str, DefenseStrategy] = {}
        self.mitre_techniques: Dict[str, MitreAttackInfo] = {}
        self.ttps: Dict[str, TTP] = {}
        self.lessons: Dict[str, Lesson] = {}
        
        # Knowledge graph
        self.knowledge_graph = KnowledgeGraph()
        
        # File paths
        self.attack_patterns_file = self.data_dir / "attack_patterns.json"
        self.defense_strategies_file = self.data_dir / "defense_strategies.json"
        self.mitre_data_file = self.data_dir / "mitre_attack.json"
        self.ttps_file = self.data_dir / "ttps.json"
        self.lessons_file = self.data_dir / "lessons.json"
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the knowledge base"""
        try:
            self.logger.info("Initializing knowledge base")
            
            # Load existing data
            await self._load_attack_patterns()
            await self._load_defense_strategies()
            await self._load_mitre_data()
            await self._load_ttps()
            await self._load_lessons()
            
            # Initialize with default data if empty
            if not self.mitre_techniques:
                await self._initialize_default_mitre_data()
            
            if not self.attack_patterns:
                await self._initialize_default_attack_patterns()
            
            if not self.defense_strategies:
                await self._initialize_default_defense_strategies()
            
            # Build knowledge graph
            await self._build_knowledge_graph()
            
            self.initialized = True
            self.logger.info("Knowledge base initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge base: {e}")
            raise
    
    async def store_attack_pattern(self, pattern: AttackPattern) -> None:
        """Store an attack pattern"""
        try:
            self.attack_patterns[pattern.pattern_id] = pattern
            await self._save_attack_patterns()
            await self._update_knowledge_graph()
            
            self.logger.debug(f"Stored attack pattern: {pattern.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store attack pattern: {e}")
            raise
    
    async def store_defense_strategy(self, strategy: DefenseStrategy) -> None:
        """Store a defense strategy"""
        try:
            self.defense_strategies[strategy.strategy_id] = strategy
            await self._save_defense_strategies()
            await self._update_knowledge_graph()
            
            self.logger.debug(f"Stored defense strategy: {strategy.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to store defense strategy: {e}")
            raise
    
    async def query_mitre_attack(self, technique_id: str) -> Optional[MitreAttackInfo]:
        """Query MITRE ATT&CK information by technique ID"""
        try:
            return self.mitre_techniques.get(technique_id)
            
        except Exception as e:
            self.logger.error(f"Failed to query MITRE ATT&CK: {e}")
            return None
    
    async def update_ttp_mapping(self, action: str, ttp: TTP) -> None:
        """Update TTP mapping for an action"""
        try:
            # Create or update TTP
            self.ttps[ttp.ttp_id] = ttp
            await self._save_ttps()
            
            self.logger.debug(f"Updated TTP mapping for action: {action}")
            
        except Exception as e:
            self.logger.error(f"Failed to update TTP mapping: {e}")
    
    async def get_lessons_learned(self, scenario_type: str) -> List[Lesson]:
        """Get lessons learned for a scenario type"""
        try:
            lessons = [
                lesson for lesson in self.lessons.values()
                if lesson.scenario_type == scenario_type
            ]
            
            # Sort by importance score
            lessons.sort(key=lambda x: x.importance_score, reverse=True)
            
            return lessons
            
        except Exception as e:
            self.logger.error(f"Failed to get lessons learned: {e}")
            return []
    
    async def add_lesson_learned(self, lesson: Lesson) -> None:
        """Add a new lesson learned"""
        try:
            self.lessons[lesson.lesson_id] = lesson
            await self._save_lessons()
            
            self.logger.debug(f"Added lesson learned: {lesson.lesson_text[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Failed to add lesson learned: {e}")
    
    async def generate_knowledge_graph(self) -> KnowledgeGraph:
        """Generate knowledge graph from stored data"""
        try:
            await self._build_knowledge_graph()
            return self.knowledge_graph
            
        except Exception as e:
            self.logger.error(f"Failed to generate knowledge graph: {e}")
            return KnowledgeGraph()
    
    async def search_attack_patterns(self, query: str, tactics: Optional[List[TacticType]] = None) -> List[AttackPattern]:
        """Search attack patterns by query and tactics"""
        try:
            results = []
            query_lower = query.lower()
            
            for pattern in self.attack_patterns.values():
                # Text search
                if (query_lower in pattern.name.lower() or 
                    query_lower in pattern.description.lower()):
                    
                    # Tactic filter
                    if tactics:
                        if any(tactic in pattern.tactics for tactic in tactics):
                            results.append(pattern)
                    else:
                        results.append(pattern)
            
            # Sort by success rate
            results.sort(key=lambda x: x.success_rate, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search attack patterns: {e}")
            return []
    
    async def search_defense_strategies(self, query: str, tactics: Optional[List[DefenseTactic]] = None) -> List[DefenseStrategy]:
        """Search defense strategies by query and tactics"""
        try:
            results = []
            query_lower = query.lower()
            
            for strategy in self.defense_strategies.values():
                # Text search
                if (query_lower in strategy.name.lower() or 
                    query_lower in strategy.description.lower()):
                    
                    # Tactic filter
                    if tactics:
                        if any(tactic in strategy.defense_tactics for tactic in tactics):
                            results.append(strategy)
                    else:
                        results.append(strategy)
            
            # Sort by effectiveness score
            results.sort(key=lambda x: x.effectiveness_score, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search defense strategies: {e}")
            return []
    
    async def get_related_techniques(self, technique_id: str) -> List[MitreAttackInfo]:
        """Get techniques related to a given technique"""
        try:
            base_technique = self.mitre_techniques.get(technique_id)
            if not base_technique:
                return []
            
            related = []
            for tech_id, technique in self.mitre_techniques.items():
                if tech_id != technique_id and technique.tactic == base_technique.tactic:
                    related.append(technique)
            
            return related
            
        except Exception as e:
            self.logger.error(f"Failed to get related techniques: {e}")
            return []
    
    async def _load_attack_patterns(self) -> None:
        """Load attack patterns from file"""
        try:
            if self.attack_patterns_file.exists():
                with open(self.attack_patterns_file, 'r') as f:
                    data = json.load(f)
                    for pattern_data in data:
                        pattern = AttackPattern(**pattern_data)
                        # Convert string dates back to datetime
                        pattern.created_at = datetime.fromisoformat(pattern.created_at)
                        if pattern.last_used:
                            pattern.last_used = datetime.fromisoformat(pattern.last_used)
                        # Convert string tactics back to enums
                        pattern.tactics = [TacticType(t) for t in pattern.tactics]
                        self.attack_patterns[pattern.pattern_id] = pattern
                
                self.logger.debug(f"Loaded {len(self.attack_patterns)} attack patterns")
                
        except Exception as e:
            self.logger.error(f"Failed to load attack patterns: {e}")
    
    async def _save_attack_patterns(self) -> None:
        """Save attack patterns to file"""
        try:
            data = []
            for pattern in self.attack_patterns.values():
                pattern_dict = asdict(pattern)
                # Convert datetime to string
                pattern_dict['created_at'] = pattern.created_at.isoformat()
                if pattern.last_used:
                    pattern_dict['last_used'] = pattern.last_used.isoformat()
                # Convert enums to strings
                pattern_dict['tactics'] = [t.value for t in pattern.tactics]
                data.append(pattern_dict)
            
            with open(self.attack_patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save attack patterns: {e}")
    
    async def _load_defense_strategies(self) -> None:
        """Load defense strategies from file"""
        try:
            if self.defense_strategies_file.exists():
                with open(self.defense_strategies_file, 'r') as f:
                    data = json.load(f)
                    for strategy_data in data:
                        strategy = DefenseStrategy(**strategy_data)
                        # Convert string dates back to datetime
                        strategy.created_at = datetime.fromisoformat(strategy.created_at)
                        strategy.last_updated = datetime.fromisoformat(strategy.last_updated)
                        # Convert string tactics back to enums
                        strategy.defense_tactics = [DefenseTactic(t) for t in strategy.defense_tactics]
                        self.defense_strategies[strategy.strategy_id] = strategy
                
                self.logger.debug(f"Loaded {len(self.defense_strategies)} defense strategies")
                
        except Exception as e:
            self.logger.error(f"Failed to load defense strategies: {e}")
    
    async def _save_defense_strategies(self) -> None:
        """Save defense strategies to file"""
        try:
            data = []
            for strategy in self.defense_strategies.values():
                strategy_dict = asdict(strategy)
                # Convert datetime to string
                strategy_dict['created_at'] = strategy.created_at.isoformat()
                strategy_dict['last_updated'] = strategy.last_updated.isoformat()
                # Convert enums to strings
                strategy_dict['defense_tactics'] = [t.value for t in strategy.defense_tactics]
                data.append(strategy_dict)
            
            with open(self.defense_strategies_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save defense strategies: {e}")
    
    async def _load_mitre_data(self) -> None:
        """Load MITRE ATT&CK data from file"""
        try:
            if self.mitre_data_file.exists():
                with open(self.mitre_data_file, 'r') as f:
                    data = json.load(f)
                    for tech_data in data:
                        technique = MitreAttackInfo(**tech_data)
                        # Convert string dates and enums
                        technique.last_updated = datetime.fromisoformat(technique.last_updated)
                        technique.tactic = TacticType(technique.tactic)
                        self.mitre_techniques[technique.technique_id] = technique
                
                self.logger.debug(f"Loaded {len(self.mitre_techniques)} MITRE techniques")
                
        except Exception as e:
            self.logger.error(f"Failed to load MITRE data: {e}")
    
    async def _load_ttps(self) -> None:
        """Load TTPs from file"""
        try:
            if self.ttps_file.exists():
                with open(self.ttps_file, 'r') as f:
                    data = json.load(f)
                    for ttp_data in data:
                        ttp = TTP(**ttp_data)
                        ttp.created_at = datetime.fromisoformat(ttp.created_at)
                        ttp.tactic = TacticType(ttp.tactic)
                        self.ttps[ttp.ttp_id] = ttp
                
                self.logger.debug(f"Loaded {len(self.ttps)} TTPs")
                
        except Exception as e:
            self.logger.error(f"Failed to load TTPs: {e}")
    
    async def _save_ttps(self) -> None:
        """Save TTPs to file"""
        try:
            data = []
            for ttp in self.ttps.values():
                ttp_dict = asdict(ttp)
                ttp_dict['created_at'] = ttp.created_at.isoformat()
                ttp_dict['tactic'] = ttp.tactic.value
                data.append(ttp_dict)
            
            with open(self.ttps_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save TTPs: {e}")
    
    async def _load_lessons(self) -> None:
        """Load lessons learned from file"""
        try:
            if self.lessons_file.exists():
                with open(self.lessons_file, 'r') as f:
                    data = json.load(f)
                    for lesson_data in data:
                        lesson = Lesson(**lesson_data)
                        lesson.created_at = datetime.fromisoformat(lesson.created_at)
                        self.lessons[lesson.lesson_id] = lesson
                
                self.logger.debug(f"Loaded {len(self.lessons)} lessons")
                
        except Exception as e:
            self.logger.error(f"Failed to load lessons: {e}")
    
    async def _save_lessons(self) -> None:
        """Save lessons learned to file"""
        try:
            data = []
            for lesson in self.lessons.values():
                lesson_dict = asdict(lesson)
                lesson_dict['created_at'] = lesson.created_at.isoformat()
                data.append(lesson_dict)
            
            with open(self.lessons_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save lessons: {e}")
    
    async def _initialize_default_mitre_data(self) -> None:
        """Initialize with default MITRE ATT&CK data"""
        try:
            # Sample MITRE techniques
            default_techniques = [
                MitreAttackInfo(
                    technique_id="T1595",
                    technique_name="Active Scanning",
                    tactic=TacticType.RECONNAISSANCE,
                    description="Adversaries may execute active reconnaissance scans to gather information that can be used during targeting.",
                    platforms=["PRE"],
                    data_sources=["Network Traffic"],
                    mitigations=["M1056"],
                    detection_methods=["Monitor network traffic for scanning patterns"],
                    references=["https://attack.mitre.org/techniques/T1595/"],
                    last_updated=datetime.now()
                ),
                MitreAttackInfo(
                    technique_id="T1190",
                    technique_name="Exploit Public-Facing Application",
                    tactic=TacticType.INITIAL_ACCESS,
                    description="Adversaries may attempt to take advantage of a weakness in an Internet-facing computer or program using software, data, or commands in order to cause unintended or unanticipated behavior.",
                    platforms=["Linux", "Windows", "macOS"],
                    data_sources=["Application Log", "Network Traffic"],
                    mitigations=["M1048", "M1030"],
                    detection_methods=["Monitor application logs for exploitation attempts"],
                    references=["https://attack.mitre.org/techniques/T1190/"],
                    last_updated=datetime.now()
                )
            ]
            
            for technique in default_techniques:
                self.mitre_techniques[technique.technique_id] = technique
            
            # Save to file
            await self._save_mitre_data()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default MITRE data: {e}")
    
    async def _save_mitre_data(self) -> None:
        """Save MITRE data to file"""
        try:
            data = []
            for technique in self.mitre_techniques.values():
                tech_dict = asdict(technique)
                tech_dict['last_updated'] = technique.last_updated.isoformat()
                tech_dict['tactic'] = technique.tactic.value
                data.append(tech_dict)
            
            with open(self.mitre_data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save MITRE data: {e}")
    
    async def _initialize_default_attack_patterns(self) -> None:
        """Initialize with default attack patterns"""
        try:
            default_patterns = [
                AttackPattern(
                    pattern_id=str(uuid.uuid4()),
                    name="Web Application Reconnaissance",
                    description="Systematic reconnaissance of web applications to identify vulnerabilities",
                    mitre_techniques=["T1595", "T1590"],
                    tactics=[TacticType.RECONNAISSANCE],
                    indicators=["Port scanning", "Directory enumeration", "Technology fingerprinting"],
                    countermeasures=["Rate limiting", "Web application firewall", "Intrusion detection"],
                    success_rate=0.8,
                    difficulty_level="low",
                    created_at=datetime.now(),
                    last_used=None,
                    metadata={"category": "web_application", "stealth": "medium"}
                ),
                AttackPattern(
                    pattern_id=str(uuid.uuid4()),
                    name="SQL Injection Attack",
                    description="Exploitation of SQL injection vulnerabilities in web applications",
                    mitre_techniques=["T1190"],
                    tactics=[TacticType.INITIAL_ACCESS],
                    indicators=["SQL error messages", "Unusual database queries", "Data exfiltration"],
                    countermeasures=["Input validation", "Parameterized queries", "WAF rules"],
                    success_rate=0.6,
                    difficulty_level="medium",
                    created_at=datetime.now(),
                    last_used=None,
                    metadata={"category": "web_application", "impact": "high"}
                )
            ]
            
            for pattern in default_patterns:
                self.attack_patterns[pattern.pattern_id] = pattern
            
            await self._save_attack_patterns()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default attack patterns: {e}")
    
    async def _initialize_default_defense_strategies(self) -> None:
        """Initialize with default defense strategies"""
        try:
            default_strategies = [
                DefenseStrategy(
                    strategy_id=str(uuid.uuid4()),
                    name="Web Application Protection",
                    description="Comprehensive protection strategy for web applications",
                    defense_tactics=[DefenseTactic.DETECT, DefenseTactic.DENY],
                    mitre_mitigations=["M1048", "M1030"],
                    effectiveness_score=0.8,
                    implementation_complexity="medium",
                    tools_required=["WAF", "IDS", "Log analysis"],
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    metadata={"category": "web_security", "priority": "high"}
                ),
                DefenseStrategy(
                    strategy_id=str(uuid.uuid4()),
                    name="Network Monitoring",
                    description="Continuous network monitoring and anomaly detection",
                    defense_tactics=[DefenseTactic.DETECT, DefenseTactic.CONTAIN],
                    mitre_mitigations=["M1031", "M1037"],
                    effectiveness_score=0.7,
                    implementation_complexity="high",
                    tools_required=["SIEM", "Network monitoring", "Threat intelligence"],
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    metadata={"category": "network_security", "priority": "high"}
                )
            ]
            
            for strategy in default_strategies:
                self.defense_strategies[strategy.strategy_id] = strategy
            
            await self._save_defense_strategies()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default defense strategies: {e}")
    
    async def _build_knowledge_graph(self) -> None:
        """Build knowledge graph from stored data"""
        try:
            # Clear existing graph
            self.knowledge_graph = KnowledgeGraph()
            
            # Add attack pattern nodes
            for pattern in self.attack_patterns.values():
                self.knowledge_graph.nodes[pattern.pattern_id] = {
                    "type": "attack_pattern",
                    "name": pattern.name,
                    "tactics": [t.value for t in pattern.tactics],
                    "success_rate": pattern.success_rate
                }
                self.knowledge_graph.node_types.add("attack_pattern")
            
            # Add defense strategy nodes
            for strategy in self.defense_strategies.values():
                self.knowledge_graph.nodes[strategy.strategy_id] = {
                    "type": "defense_strategy",
                    "name": strategy.name,
                    "tactics": [t.value for t in strategy.defense_tactics],
                    "effectiveness": strategy.effectiveness_score
                }
                self.knowledge_graph.node_types.add("defense_strategy")
            
            # Add MITRE technique nodes
            for technique in self.mitre_techniques.values():
                self.knowledge_graph.nodes[technique.technique_id] = {
                    "type": "mitre_technique",
                    "name": technique.technique_name,
                    "tactic": technique.tactic.value,
                    "platforms": technique.platforms
                }
                self.knowledge_graph.node_types.add("mitre_technique")
            
            # Add edges (relationships)
            await self._add_knowledge_graph_edges()
            
        except Exception as e:
            self.logger.error(f"Failed to build knowledge graph: {e}")
    
    async def _add_knowledge_graph_edges(self) -> None:
        """Add edges to knowledge graph"""
        try:
            # Connect attack patterns to MITRE techniques
            for pattern in self.attack_patterns.values():
                for technique_id in pattern.mitre_techniques:
                    if technique_id in self.mitre_techniques:
                        edge = {
                            "source": pattern.pattern_id,
                            "target": technique_id,
                            "type": "uses_technique",
                            "weight": 1.0
                        }
                        self.knowledge_graph.edges.append(edge)
                        self.knowledge_graph.edge_types.add("uses_technique")
            
            # Connect defense strategies to MITRE mitigations
            for strategy in self.defense_strategies.values():
                for mitigation_id in strategy.mitre_mitigations:
                    # In a real implementation, we'd have mitigation nodes
                    # For now, just track the relationship
                    pass
            
        except Exception as e:
            self.logger.error(f"Failed to add knowledge graph edges: {e}")
    
    async def _update_knowledge_graph(self) -> None:
        """Update knowledge graph after data changes"""
        try:
            await self._build_knowledge_graph()
            
        except Exception as e:
            self.logger.error(f"Failed to update knowledge graph: {e}")
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            return {
                "attack_patterns": len(self.attack_patterns),
                "defense_strategies": len(self.defense_strategies),
                "mitre_techniques": len(self.mitre_techniques),
                "ttps": len(self.ttps),
                "lessons_learned": len(self.lessons),
                "knowledge_graph_nodes": len(self.knowledge_graph.nodes),
                "knowledge_graph_edges": len(self.knowledge_graph.edges),
                "initialized": self.initialized
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get knowledge base stats: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown the knowledge base"""
        try:
            self.logger.info("Shutting down knowledge base")
            
            # Save all data
            await self._save_attack_patterns()
            await self._save_defense_strategies()
            await self._save_mitre_data()
            await self._save_ttps()
            await self._save_lessons()
            
            # Clear memory
            self.attack_patterns.clear()
            self.defense_strategies.clear()
            self.mitre_techniques.clear()
            self.ttps.clear()
            self.lessons.clear()
            
            self.initialized = False
            self.logger.info("Knowledge base shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during knowledge base shutdown: {e}")

# Factory function
def create_knowledge_base(data_dir: str = "./knowledge_base") -> KnowledgeBase:
    """Create a knowledge base instance"""
    return KnowledgeBase(data_dir=data_dir)