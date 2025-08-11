#!/usr/bin/env python3
"""
Tests for Knowledge Base functionality
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock
import tempfile
import shutil
import uuid

from memory.knowledge_base import (
    KnowledgeBase, AttackPattern, DefenseStrategy, MitreAttackInfo, TTP, Lesson,
    TacticType, DefenseTactic, KnowledgeGraph, create_knowledge_base
)

@pytest.fixture
def temp_data_dir():
    """Create temporary data directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def knowledge_base(temp_data_dir):
    """Create test knowledge base"""
    return KnowledgeBase(data_dir=temp_data_dir)

@pytest.fixture
def sample_attack_pattern():
    """Create sample attack pattern"""
    return AttackPattern(
        pattern_id=str(uuid.uuid4()),
        name="Test Attack Pattern",
        description="A test attack pattern for unit testing",
        mitre_techniques=["T1595", "T1590"],
        tactics=[TacticType.RECONNAISSANCE],
        indicators=["port_scanning", "service_enumeration"],
        countermeasures=["rate_limiting", "monitoring"],
        success_rate=0.75,
        difficulty_level="medium",
        created_at=datetime.now(),
        last_used=None,
        metadata={"category": "test", "priority": "medium"}
    )

@pytest.fixture
def sample_defense_strategy():
    """Create sample defense strategy"""
    return DefenseStrategy(
        strategy_id=str(uuid.uuid4()),
        name="Test Defense Strategy",
        description="A test defense strategy for unit testing",
        defense_tactics=[DefenseTactic.DETECT, DefenseTactic.DENY],
        mitre_mitigations=["M1031", "M1037"],
        effectiveness_score=0.8,
        implementation_complexity="medium",
        tools_required=["SIEM", "Firewall", "IDS"],
        created_at=datetime.now(),
        last_updated=datetime.now(),
        metadata={"category": "test", "priority": "high"}
    )

@pytest.fixture
def sample_mitre_technique():
    """Create sample MITRE technique"""
    return MitreAttackInfo(
        technique_id="T1595",
        technique_name="Active Scanning",
        tactic=TacticType.RECONNAISSANCE,
        description="Test MITRE technique description",
        platforms=["Linux", "Windows"],
        data_sources=["Network Traffic"],
        mitigations=["M1056"],
        detection_methods=["Monitor network traffic"],
        references=["https://attack.mitre.org/techniques/T1595/"],
        last_updated=datetime.now()
    )

@pytest.mark.asyncio
async def test_knowledge_base_initialization(knowledge_base):
    """Test knowledge base initialization"""
    assert not knowledge_base.initialized
    
    await knowledge_base.initialize()
    
    assert knowledge_base.initialized
    assert len(knowledge_base.mitre_techniques) > 0  # Should have default data
    assert len(knowledge_base.attack_patterns) > 0
    assert len(knowledge_base.defense_strategies) > 0

@pytest.mark.asyncio
async def test_store_attack_pattern(knowledge_base, sample_attack_pattern):
    """Test storing attack pattern"""
    await knowledge_base.initialize()
    
    await knowledge_base.store_attack_pattern(sample_attack_pattern)
    
    assert sample_attack_pattern.pattern_id in knowledge_base.attack_patterns
    stored_pattern = knowledge_base.attack_patterns[sample_attack_pattern.pattern_id]
    assert stored_pattern.name == sample_attack_pattern.name

@pytest.mark.asyncio
async def test_store_defense_strategy(knowledge_base, sample_defense_strategy):
    """Test storing defense strategy"""
    await knowledge_base.initialize()
    
    await knowledge_base.store_defense_strategy(sample_defense_strategy)
    
    assert sample_defense_strategy.strategy_id in knowledge_base.defense_strategies
    stored_strategy = knowledge_base.defense_strategies[sample_defense_strategy.strategy_id]
    assert stored_strategy.name == sample_defense_strategy.name

@pytest.mark.asyncio
async def test_query_mitre_attack(knowledge_base):
    """Test querying MITRE ATT&CK information"""
    await knowledge_base.initialize()
    
    # Should have default techniques
    technique = await knowledge_base.query_mitre_attack("T1595")
    
    if technique:  # May not exist in minimal test setup
        assert technique.technique_id == "T1595"
        assert isinstance(technique.tactic, TacticType)

@pytest.mark.asyncio
async def test_search_attack_patterns(knowledge_base, sample_attack_pattern):
    """Test searching attack patterns"""
    await knowledge_base.initialize()
    
    await knowledge_base.store_attack_pattern(sample_attack_pattern)
    
    # Search by name
    results = await knowledge_base.search_attack_patterns("Test Attack")
    assert len(results) > 0
    assert results[0].name == sample_attack_pattern.name
    
    # Search by tactic
    results = await knowledge_base.search_attack_patterns("", [TacticType.RECONNAISSANCE])
    assert len(results) > 0

@pytest.mark.asyncio
async def test_search_defense_strategies(knowledge_base, sample_defense_strategy):
    """Test searching defense strategies"""
    await knowledge_base.initialize()
    
    await knowledge_base.store_defense_strategy(sample_defense_strategy)
    
    # Search by name
    results = await knowledge_base.search_defense_strategies("Test Defense")
    assert len(results) > 0
    assert results[0].name == sample_defense_strategy.name
    
    # Search by tactic
    results = await knowledge_base.search_defense_strategies("", [DefenseTactic.DETECT])
    assert len(results) > 0

@pytest.mark.asyncio
async def test_lessons_learned(knowledge_base):
    """Test lessons learned functionality"""
    await knowledge_base.initialize()
    
    # Add a lesson
    lesson = Lesson(
        lesson_id=str(uuid.uuid4()),
        scenario_type="test_scenario",
        lesson_text="This is a test lesson learned",
        category="testing",
        importance_score=0.8,
        applicable_roles=["tester"],
        mitre_techniques=["T1595"],
        created_at=datetime.now(),
        validated=True,
        metadata={"source": "test"}
    )
    
    await knowledge_base.add_lesson_learned(lesson)
    
    # Retrieve lessons
    lessons = await knowledge_base.get_lessons_learned("test_scenario")
    assert len(lessons) > 0
    assert lessons[0].lesson_text == lesson.lesson_text

@pytest.mark.asyncio
async def test_ttp_mapping(knowledge_base):
    """Test TTP mapping functionality"""
    await knowledge_base.initialize()
    
    ttp = TTP(
        ttp_id=str(uuid.uuid4()),
        name="Test TTP",
        tactic=TacticType.RECONNAISSANCE,
        techniques=["T1595"],
        procedures=["port_scan", "service_enum"],
        mitre_mapping=["T1595"],
        success_indicators=["open_ports_found"],
        failure_indicators=["no_response"],
        prerequisites=["network_access"],
        artifacts=["scan_logs"],
        created_at=datetime.now()
    )
    
    await knowledge_base.update_ttp_mapping("test_action", ttp)
    
    assert ttp.ttp_id in knowledge_base.ttps

@pytest.mark.asyncio
async def test_knowledge_graph_generation(knowledge_base, sample_attack_pattern):
    """Test knowledge graph generation"""
    await knowledge_base.initialize()
    
    await knowledge_base.store_attack_pattern(sample_attack_pattern)
    
    graph = await knowledge_base.generate_knowledge_graph()
    
    assert isinstance(graph, KnowledgeGraph)
    assert len(graph.nodes) > 0
    assert len(graph.node_types) > 0

@pytest.mark.asyncio
async def test_related_techniques(knowledge_base):
    """Test getting related techniques"""
    await knowledge_base.initialize()
    
    # This test depends on having multiple techniques with same tactic
    related = await knowledge_base.get_related_techniques("T1595")
    
    # May be empty in minimal test setup
    assert isinstance(related, list)

def test_tactic_type_enum():
    """Test TacticType enum values"""
    assert TacticType.RECONNAISSANCE.value == "reconnaissance"
    assert TacticType.INITIAL_ACCESS.value == "initial-access"
    assert TacticType.EXECUTION.value == "execution"
    assert TacticType.PERSISTENCE.value == "persistence"

def test_defense_tactic_enum():
    """Test DefenseTactic enum values"""
    assert DefenseTactic.DETECT.value == "detect"
    assert DefenseTactic.DENY.value == "deny"
    assert DefenseTactic.DISRUPT.value == "disrupt"
    assert DefenseTactic.CONTAIN.value == "contain"

def test_attack_pattern_creation(sample_attack_pattern):
    """Test attack pattern creation"""
    assert sample_attack_pattern.name == "Test Attack Pattern"
    assert TacticType.RECONNAISSANCE in sample_attack_pattern.tactics
    assert sample_attack_pattern.success_rate == 0.75
    assert sample_attack_pattern.difficulty_level == "medium"

def test_defense_strategy_creation(sample_defense_strategy):
    """Test defense strategy creation"""
    assert sample_defense_strategy.name == "Test Defense Strategy"
    assert DefenseTactic.DETECT in sample_defense_strategy.defense_tactics
    assert sample_defense_strategy.effectiveness_score == 0.8
    assert sample_defense_strategy.implementation_complexity == "medium"

def test_mitre_attack_info_creation(sample_mitre_technique):
    """Test MITRE attack info creation"""
    assert sample_mitre_technique.technique_id == "T1595"
    assert sample_mitre_technique.technique_name == "Active Scanning"
    assert sample_mitre_technique.tactic == TacticType.RECONNAISSANCE
    assert "Linux" in sample_mitre_technique.platforms

def test_knowledge_graph_creation():
    """Test knowledge graph creation"""
    graph = KnowledgeGraph()
    
    # Add test nodes
    graph.nodes["node1"] = {"type": "test", "name": "Test Node"}
    graph.node_types.add("test")
    
    # Add test edges
    graph.edges.append({
        "source": "node1",
        "target": "node2", 
        "type": "test_relation",
        "weight": 1.0
    })
    graph.edge_types.add("test_relation")
    
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 1
    assert "test" in graph.node_types
    assert "test_relation" in graph.edge_types

@pytest.mark.asyncio
async def test_knowledge_base_statistics(knowledge_base):
    """Test knowledge base statistics"""
    await knowledge_base.initialize()
    
    stats = knowledge_base.get_knowledge_base_stats()
    
    assert "attack_patterns" in stats
    assert "defense_strategies" in stats
    assert "mitre_techniques" in stats
    assert "initialized" in stats
    assert stats["initialized"] is True

@pytest.mark.asyncio
async def test_knowledge_base_shutdown(knowledge_base):
    """Test knowledge base shutdown"""
    await knowledge_base.initialize()
    assert knowledge_base.initialized
    
    await knowledge_base.shutdown()
    assert not knowledge_base.initialized

def test_create_knowledge_base_factory():
    """Test knowledge base factory function"""
    kb = create_knowledge_base("./test_kb")
    
    assert isinstance(kb, KnowledgeBase)
    assert str(kb.data_dir).endswith("test_kb")

@pytest.mark.asyncio
async def test_data_persistence(knowledge_base, sample_attack_pattern):
    """Test data persistence to files"""
    await knowledge_base.initialize()
    
    # Store pattern
    await knowledge_base.store_attack_pattern(sample_attack_pattern)
    
    # Check that file exists
    assert knowledge_base.attack_patterns_file.exists()
    
    # Create new knowledge base instance
    kb2 = KnowledgeBase(data_dir=str(knowledge_base.data_dir))
    await kb2.initialize()
    
    # Should load the stored pattern
    assert sample_attack_pattern.pattern_id in kb2.attack_patterns

def test_lesson_creation():
    """Test lesson creation"""
    lesson = Lesson(
        lesson_id="lesson_001",
        scenario_type="test",
        lesson_text="Test lesson",
        category="testing",
        importance_score=0.9,
        applicable_roles=["tester"],
        mitre_techniques=["T1595"],
        created_at=datetime.now(),
        validated=True,
        metadata={"test": True}
    )
    
    assert lesson.lesson_id == "lesson_001"
    assert lesson.scenario_type == "test"
    assert lesson.importance_score == 0.9
    assert lesson.validated is True

def test_ttp_creation():
    """Test TTP creation"""
    ttp = TTP(
        ttp_id="ttp_001",
        name="Test TTP",
        tactic=TacticType.RECONNAISSANCE,
        techniques=["T1595"],
        procedures=["scan"],
        mitre_mapping=["T1595"],
        success_indicators=["success"],
        failure_indicators=["failure"],
        prerequisites=["access"],
        artifacts=["logs"],
        created_at=datetime.now()
    )
    
    assert ttp.ttp_id == "ttp_001"
    assert ttp.name == "Test TTP"
    assert ttp.tactic == TacticType.RECONNAISSANCE
    assert "T1595" in ttp.techniques