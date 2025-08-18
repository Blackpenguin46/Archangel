#!/usr/bin/env python3
"""
Simple test for the ontology system without external dependencies
"""

import asyncio
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

# Import the ontology modules
from memory.ontology_manager import (
    OntologyManager, OntologyEntity, OntologyRelation, SemanticMapping,
    EntityType, RelationType
)

async def test_ontology_basic():
    """Basic test of ontology functionality"""
    print("🧪 Testing Ontology Manager Basic Functionality")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize ontology manager
        manager = OntologyManager(data_dir=temp_dir)
        await manager.initialize()
        
        print("✅ Ontology manager initialized")
        
        # Test adding entities
        entity = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.ATTACK_TECHNIQUE,
            name="Test Attack Technique",
            description="A test attack technique for validation",
            properties={"test": True},
            mitre_id="T9999"
        )
        
        await manager.add_entity(entity)
        print("✅ Entity added successfully")
        
        # Test retrieving entity
        retrieved = await manager.get_entity_by_mitre_id("T9999")
        assert retrieved is not None
        assert retrieved.name == "Test Attack Technique"
        print("✅ Entity retrieval works")
        
        # Test getting entities by type
        attack_entities = await manager.get_entities_by_type(EntityType.ATTACK_TECHNIQUE)
        assert len(attack_entities) > 0
        print("✅ Entity type filtering works")
        
        # Test statistics
        stats = await manager.get_ontology_statistics()
        assert stats['entities']['total'] > 0
        print("✅ Statistics generation works")
        
        print("🎉 All basic ontology tests passed!")

async def test_ontology_relations():
    """Test ontology relations functionality"""
    print("\n🧪 Testing Ontology Relations")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = OntologyManager(data_dir=temp_dir)
        await manager.initialize()
        
        # Create two entities
        entity1 = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.ATTACK_TECHNIQUE,
            name="Source Technique",
            description="Source technique",
            properties={}
        )
        
        entity2 = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.DEFENSE_TECHNIQUE,
            name="Target Technique",
            description="Target technique",
            properties={}
        )
        
        await manager.add_entity(entity1)
        await manager.add_entity(entity2)
        print("✅ Created test entities")
        
        # Create relation
        relation = OntologyRelation(
            relation_id=str(uuid.uuid4()),
            source_entity=entity1.entity_id,
            target_entity=entity2.entity_id,
            relation_type=RelationType.COUNTERS,
            properties={"test_relation": True}
        )
        
        await manager.add_relation(relation)
        print("✅ Created relation")
        
        # Test getting related entities
        related = await manager.get_related_entities(entity1.entity_id)
        assert len(related) > 0
        print("✅ Related entities retrieval works")
        
        # Test knowledge graph
        assert manager.knowledge_graph.has_edge(entity1.entity_id, entity2.entity_id)
        print("✅ Knowledge graph updated correctly")
        
        print("🎉 All relation tests passed!")

async def test_semantic_mapping():
    """Test semantic mapping functionality"""
    print("\n🧪 Testing Semantic Mapping")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        from memory.semantic_mapper import SemanticMapper, FrameworkEntity, FrameworkType, MappingConfidence
        
        mapper = SemanticMapper(data_dir=temp_dir)
        await mapper.initialize()
        print("✅ Semantic mapper initialized")
        
        # Add framework entity
        entity = FrameworkEntity(
            framework=FrameworkType.MITRE_ATTACK,
            entity_id="T9999",
            name="Test Technique",
            description="A test technique",
            category="test",
            properties={"test": True}
        )
        
        await mapper.add_framework_entity(entity)
        print("✅ Framework entity added")
        
        # Test statistics
        stats = await mapper.get_mapping_statistics()
        assert 'total_mappings' in stats
        print("✅ Mapping statistics work")
        
        print("🎉 All semantic mapping tests passed!")

async def test_ontology_updater():
    """Test ontology updater functionality"""
    print("\n🧪 Testing Ontology Updater")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        from memory.ontology_updater import OntologyUpdater, SimulationOutcome
        
        updater = OntologyUpdater(data_dir=temp_dir)
        await updater.initialize()
        print("✅ Ontology updater initialized")
        
        # Create simulation outcome
        outcome = SimulationOutcome(
            outcome_id=str(uuid.uuid4()),
            scenario_id="test_scenario",
            agent_id="test_agent",
            action_taken="network_scan",
            technique_used="T1595",
            success=True,
            effectiveness_score=0.8,
            duration=30.0,
            detected=False,
            detection_time=None,
            countermeasures_triggered=[],
            artifacts_created=["scan_results.txt"],
            metadata={}
        )
        
        updates = await updater.process_simulation_outcome(outcome)
        print("✅ Simulation outcome processed")
        
        # Test learning insights
        insights = await updater.get_learning_insights()
        assert 'statistics' in insights
        print("✅ Learning insights generated")
        
        print("🎉 All ontology updater tests passed!")

async def main():
    """Run all tests"""
    print("🚀 Starting Ontology System Tests\n")
    
    try:
        await test_ontology_basic()
        await test_ontology_relations()
        await test_semantic_mapping()
        await test_ontology_updater()
        
        print("\n🎉 ALL TESTS PASSED! Ontology system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())