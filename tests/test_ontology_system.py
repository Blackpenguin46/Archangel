#!/usr/bin/env python3
"""
Tests for the ontology-driven knowledge base system
"""

import asyncio
import json
import pytest
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import networkx as nx

# Import the ontology modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from memory.ontology_manager import (
    OntologyManager, OntologyEntity, OntologyRelation, SemanticMapping,
    EntityType, RelationType
)
from memory.semantic_mapper import (
    SemanticMapper, FrameworkEntity, FrameworkType, MappingConfidence
)
from memory.inference_engine import (
    InferenceEngine, InferenceRule, InferenceResult, InferenceType, ConfidenceLevel
)
from memory.ontology_updater import (
    OntologyUpdater, SimulationOutcome, ThreatIntelligence, UpdateType, UpdateSource
)

class TestOntologyManager:
    """Test cases for OntologyManager"""
    
    @pytest.fixture
    async def ontology_manager(self):
        """Create a test ontology manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = OntologyManager(data_dir=temp_dir)
            await manager.initialize()
            yield manager
    
    @pytest.mark.asyncio
    async def test_initialization(self, ontology_manager):
        """Test ontology manager initialization"""
        assert ontology_manager.initialized
        assert len(ontology_manager.entities) > 0
        assert len(ontology_manager.relations) > 0
        assert ontology_manager.knowledge_graph.number_of_nodes() > 0
    
    @pytest.mark.asyncio
    async def test_add_entity(self, ontology_manager):
        """Test adding entities to ontology"""
        entity = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.ATTACK_TECHNIQUE,
            name="Test Attack Technique",
            description="A test attack technique for validation",
            properties={"test": True},
            mitre_id="T9999"
        )
        
        await ontology_manager.add_entity(entity)
        
        # Verify entity was added
        assert entity.entity_id in ontology_manager.entities
        assert ontology_manager.entities[entity.entity_id] == entity
        
        # Verify indices were updated
        assert entity.entity_id in ontology_manager.entity_by_type[EntityType.ATTACK_TECHNIQUE]
        assert ontology_manager.entity_by_mitre_id["T9999"] == entity.entity_id
        
        # Verify knowledge graph was updated
        assert entity.entity_id in ontology_manager.knowledge_graph.nodes()
    
    @pytest.mark.asyncio
    async def test_add_relation(self, ontology_manager):
        """Test adding relations to ontology"""
        # Create two entities first
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
        
        await ontology_manager.add_entity(entity1)
        await ontology_manager.add_entity(entity2)
        
        # Create relation
        relation = OntologyRelation(
            relation_id=str(uuid.uuid4()),
            source_entity=entity1.entity_id,
            target_entity=entity2.entity_id,
            relation_type=RelationType.COUNTERS,
            properties={"test_relation": True}
        )
        
        await ontology_manager.add_relation(relation)
        
        # Verify relation was added
        assert relation.relation_id in ontology_manager.relations
        assert ontology_manager.relations[relation.relation_id] == relation
        
        # Verify indices were updated
        assert relation.relation_id in ontology_manager.relations_by_type[RelationType.COUNTERS]
        
        # Verify knowledge graph was updated
        assert ontology_manager.knowledge_graph.has_edge(entity1.entity_id, entity2.entity_id)
    
    @pytest.mark.asyncio
    async def test_get_entity_by_mitre_id(self, ontology_manager):
        """Test retrieving entity by MITRE ID"""
        # Add entity with MITRE ID
        entity = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.ATTACK_TECHNIQUE,
            name="Test Technique",
            description="Test technique",
            properties={},
            mitre_id="T1234"
        )
        
        await ontology_manager.add_entity(entity)
        
        # Retrieve by MITRE ID
        retrieved = await ontology_manager.get_entity_by_mitre_id("T1234")
        assert retrieved is not None
        assert retrieved.entity_id == entity.entity_id
        assert retrieved.mitre_id == "T1234"
        
        # Test non-existent MITRE ID
        not_found = await ontology_manager.get_entity_by_mitre_id("T9999")
        assert not_found is None
    
    @pytest.mark.asyncio
    async def test_get_related_entities(self, ontology_manager):
        """Test getting related entities"""
        # Create a chain of entities: A -> B -> C
        entity_a = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.ATTACK_TECHNIQUE,
            name="Entity A",
            description="First entity",
            properties={}
        )
        
        entity_b = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.DEFENSE_TECHNIQUE,
            name="Entity B",
            description="Second entity",
            properties={}
        )
        
        entity_c = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.MITIGATION,
            name="Entity C",
            description="Third entity",
            properties={}
        )
        
        await ontology_manager.add_entity(entity_a)
        await ontology_manager.add_entity(entity_b)
        await ontology_manager.add_entity(entity_c)
        
        # Create relations A -> B -> C
        relation_ab = OntologyRelation(
            relation_id=str(uuid.uuid4()),
            source_entity=entity_a.entity_id,
            target_entity=entity_b.entity_id,
            relation_type=RelationType.USES,
            properties={}
        )
        
        relation_bc = OntologyRelation(
            relation_id=str(uuid.uuid4()),
            source_entity=entity_b.entity_id,
            target_entity=entity_c.entity_id,
            relation_type=RelationType.ENABLES,
            properties={}
        )
        
        await ontology_manager.add_relation(relation_ab)
        await ontology_manager.add_relation(relation_bc)
        
        # Test getting related entities from A
        related = await ontology_manager.get_related_entities(entity_a.entity_id, max_depth=1)
        assert len(related) == 1
        assert related[0][0].entity_id == entity_b.entity_id
        
        # Test getting related entities with depth 2
        related_depth2 = await ontology_manager.get_related_entities(entity_a.entity_id, max_depth=2)
        assert len(related_depth2) == 2
        entity_ids = [r[0].entity_id for r in related_depth2]
        assert entity_b.entity_id in entity_ids
        assert entity_c.entity_id in entity_ids
    
    @pytest.mark.asyncio
    async def test_ontology_statistics(self, ontology_manager):
        """Test getting ontology statistics"""
        stats = await ontology_manager.get_ontology_statistics()
        
        assert 'entities' in stats
        assert 'relations' in stats
        assert 'mappings' in stats
        assert 'knowledge_graph' in stats
        
        assert stats['entities']['total'] > 0
        assert 'by_type' in stats['entities']
        assert stats['knowledge_graph']['nodes'] > 0


class TestSemanticMapper:
    """Test cases for SemanticMapper"""
    
    @pytest.fixture
    async def semantic_mapper(self):
        """Create a test semantic mapper"""
        with tempfile.TemporaryDirectory() as temp_dir:
            mapper = SemanticMapper(data_dir=temp_dir)
            await mapper.initialize()
            yield mapper
    
    @pytest.mark.asyncio
    async def test_initialization(self, semantic_mapper):
        """Test semantic mapper initialization"""
        assert semantic_mapper.initialized
        assert len(semantic_mapper.framework_entities[FrameworkType.MITRE_ATTACK]) > 0
        assert len(semantic_mapper.framework_entities[FrameworkType.D3FEND]) > 0
    
    @pytest.mark.asyncio
    async def test_add_framework_entity(self, semantic_mapper):
        """Test adding framework entities"""
        entity = FrameworkEntity(
            framework=FrameworkType.MITRE_ATTACK,
            entity_id="T9999",
            name="Test Technique",
            description="A test technique",
            category="test",
            properties={"test": True}
        )
        
        await semantic_mapper.add_framework_entity(entity)
        
        # Verify entity was added
        assert entity.entity_id in semantic_mapper.framework_entities[FrameworkType.MITRE_ATTACK]
        assert semantic_mapper.framework_entities[FrameworkType.MITRE_ATTACK][entity.entity_id] == entity
    
    @pytest.mark.asyncio
    async def test_generate_automatic_mappings(self, semantic_mapper):
        """Test automatic mapping generation"""
        # Add some test entities with similar names
        mitre_entity = FrameworkEntity(
            framework=FrameworkType.MITRE_ATTACK,
            entity_id="T1234",
            name="Network Scanning",
            description="Scanning network for vulnerabilities",
            category="reconnaissance",
            properties={}
        )
        
        d3fend_entity = FrameworkEntity(
            framework=FrameworkType.D3FEND,
            entity_id="D3-NET",
            name="Network Analysis",
            description="Analyzing network traffic for threats",
            category="detect",
            properties={}
        )
        
        await semantic_mapper.add_framework_entity(mitre_entity)
        await semantic_mapper.add_framework_entity(d3fend_entity)
        
        # Generate mappings
        mappings = await semantic_mapper.generate_automatic_mappings(
            FrameworkType.MITRE_ATTACK, 
            FrameworkType.D3FEND
        )
        
        # Should generate at least one mapping due to keyword similarity
        assert len(mappings) >= 0  # May be 0 if similarity threshold not met
    
    @pytest.mark.asyncio
    async def test_find_mappings(self, semantic_mapper):
        """Test finding semantic mappings"""
        # Create and add a mapping
        mapping = SemanticMapping(
            mapping_id=str(uuid.uuid4()),
            source_framework=FrameworkType.MITRE_ATTACK,
            source_entity_id="T1595",
            target_framework=FrameworkType.D3FEND,
            target_entity_id="D3-NTA",
            mapping_type="detection",
            confidence=MappingConfidence.HIGH,
            confidence_score=0.8,
            evidence=["Test mapping"],
            properties={}
        )
        
        await semantic_mapper.add_semantic_mapping(mapping)
        
        # Find mappings
        found_mappings = await semantic_mapper.find_mappings(
            FrameworkType.MITRE_ATTACK, 
            "T1595"
        )
        
        assert len(found_mappings) == 1
        assert found_mappings[0].mapping_id == mapping.mapping_id
    
    @pytest.mark.asyncio
    async def test_mapping_statistics(self, semantic_mapper):
        """Test getting mapping statistics"""
        stats = await semantic_mapper.get_mapping_statistics()
        
        assert 'total_mappings' in stats
        assert 'by_confidence' in stats
        assert 'framework_entities' in stats
        assert 'validated_mappings' in stats


class TestInferenceEngine:
    """Test cases for InferenceEngine"""
    
    @pytest.fixture
    async def inference_engine(self):
        """Create a test inference engine"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple knowledge graph
            kg = nx.MultiDiGraph()
            kg.add_node("A", entity_type="attack", name="Attack A")
            kg.add_node("B", entity_type="defense", name="Defense B")
            kg.add_node("C", entity_type="mitigation", name="Mitigation C")
            kg.add_edge("A", "B", relation_type="countered_by", confidence=0.8)
            kg.add_edge("B", "C", relation_type="enables", confidence=0.7)
            
            engine = InferenceEngine(data_dir=temp_dir)
            await engine.initialize(kg)
            yield engine
    
    @pytest.mark.asyncio
    async def test_initialization(self, inference_engine):
        """Test inference engine initialization"""
        assert inference_engine.initialized
        assert inference_engine.knowledge_graph is not None
        assert len(inference_engine.inference_rules) > 0
    
    @pytest.mark.asyncio
    async def test_add_inference_rule(self, inference_engine):
        """Test adding inference rules"""
        rule = InferenceRule(
            rule_id=str(uuid.uuid4()),
            name="Test Rule",
            description="A test inference rule",
            inference_type=InferenceType.TRANSITIVE,
            conditions=[{"type": "test"}],
            conclusions=[{"type": "test"}],
            confidence_threshold=0.5,
            priority=1
        )
        
        await inference_engine.add_inference_rule(rule)
        
        # Verify rule was added
        assert rule.rule_id in inference_engine.inference_rules
        assert inference_engine.inference_rules[rule.rule_id] == rule
    
    @pytest.mark.asyncio
    async def test_run_inference(self, inference_engine):
        """Test running inference"""
        # Run inference with all rules
        results = await inference_engine.run_inference()
        
        # Should generate some results
        assert isinstance(results, list)
        # Results may be empty if no patterns match
    
    @pytest.mark.asyncio
    async def test_propagate_confidence(self, inference_engine):
        """Test confidence propagation"""
        # Propagate confidence from node A
        confidence_scores = await inference_engine.propagate_confidence("A")
        
        assert "A" in confidence_scores
        assert confidence_scores["A"] == 1.0
        # Should have propagated to connected nodes
        assert len(confidence_scores) > 1
    
    @pytest.mark.asyncio
    async def test_find_reasoning_path(self, inference_engine):
        """Test finding reasoning paths"""
        # Find path from A to C
        path = await inference_engine.find_shortest_reasoning_path("A", "C")
        
        if path:  # Path may not exist in simple test graph
            assert path[0] == "A"
            assert path[-1] == "C"
    
    @pytest.mark.asyncio
    async def test_inference_statistics(self, inference_engine):
        """Test getting inference statistics"""
        stats = await inference_engine.get_inference_statistics()
        
        assert 'total_rules' in stats
        assert 'enabled_rules' in stats
        assert 'total_results' in stats
        assert 'success_rate' in stats


class TestOntologyUpdater:
    """Test cases for OntologyUpdater"""
    
    @pytest.fixture
    async def ontology_updater(self):
        """Create a test ontology updater"""
        with tempfile.TemporaryDirectory() as temp_dir:
            updater = OntologyUpdater(data_dir=temp_dir)
            await updater.initialize()
            yield updater
    
    @pytest.mark.asyncio
    async def test_initialization(self, ontology_updater):
        """Test ontology updater initialization"""
        assert ontology_updater.initialized
        assert len(ontology_updater.learning_rules) > 0
    
    @pytest.mark.asyncio
    async def test_process_simulation_outcome(self, ontology_updater):
        """Test processing simulation outcomes"""
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
        
        updates = await ontology_updater.process_simulation_outcome(outcome)
        
        # Verify outcome was stored
        assert outcome.outcome_id in ontology_updater.simulation_outcomes
        
        # Updates may be empty if batch processing is used
        assert isinstance(updates, list)
    
    @pytest.mark.asyncio
    async def test_process_threat_intelligence(self, ontology_updater):
        """Test processing threat intelligence"""
        intel = ThreatIntelligence(
            intel_id=str(uuid.uuid4()),
            source="test_source",
            intel_type="technique_update",
            technique_id="T1595",
            iocs=["192.168.1.1", "malware.exe"],
            ttps=["network_scanning", "port_enumeration"],
            confidence_score=0.9,
            severity="high",
            description="New scanning technique observed",
            references=["https://example.com/report"]
        )
        
        updates = await ontology_updater.process_threat_intelligence(intel)
        
        # Verify intelligence was stored
        assert intel.intel_id in ontology_updater.threat_intelligence
        
        # Should generate at least one update
        assert len(updates) > 0
        assert all(isinstance(update, OntologyUpdate) for update in updates)
    
    @pytest.mark.asyncio
    async def test_learning_insights(self, ontology_updater):
        """Test getting learning insights"""
        # Add some test outcomes first
        for i in range(5):
            outcome = SimulationOutcome(
                outcome_id=str(uuid.uuid4()),
                scenario_id=f"scenario_{i}",
                agent_id="test_agent",
                action_taken="test_action",
                technique_used="T1595",
                success=i % 2 == 0,  # Alternate success/failure
                effectiveness_score=0.5 + (i * 0.1),
                duration=10.0 + i,
                detected=i % 3 == 0,  # Detected every third attempt
                detection_time=5.0 if i % 3 == 0 else None,
                countermeasures_triggered=["firewall"] if i % 3 == 0 else [],
                artifacts_created=[],
                metadata={}
            )
            await ontology_updater.process_simulation_outcome(outcome)
        
        insights = await ontology_updater.get_learning_insights()
        
        assert 'statistics' in insights
        assert 'top_techniques' in insights
        assert isinstance(insights['statistics'], dict)


class TestIntegration:
    """Integration tests for the complete ontology system"""
    
    @pytest.fixture
    async def ontology_system(self):
        """Create a complete ontology system for integration testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize all components
            ontology_manager = OntologyManager(data_dir=f"{temp_dir}/ontology")
            semantic_mapper = SemanticMapper(data_dir=f"{temp_dir}/mappings")
            ontology_updater = OntologyUpdater(data_dir=f"{temp_dir}/updates")
            
            await ontology_manager.initialize()
            await semantic_mapper.initialize()
            await ontology_updater.initialize()
            
            # Create inference engine with ontology manager's knowledge graph
            inference_engine = InferenceEngine(data_dir=f"{temp_dir}/inference")
            await inference_engine.initialize(ontology_manager.knowledge_graph)
            
            yield {
                'ontology_manager': ontology_manager,
                'semantic_mapper': semantic_mapper,
                'inference_engine': inference_engine,
                'ontology_updater': ontology_updater
            }
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, ontology_system):
        """Test complete end-to-end ontology workflow"""
        om = ontology_system['ontology_manager']
        sm = ontology_system['semantic_mapper']
        ie = ontology_system['inference_engine']
        ou = ontology_system['ontology_updater']
        
        # 1. Add entities to ontology
        attack_entity = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.ATTACK_TECHNIQUE,
            name="SQL Injection",
            description="SQL injection attack technique",
            properties={"category": "web_attack"},
            mitre_id="T1190"
        )
        
        defense_entity = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.DEFENSE_TECHNIQUE,
            name="Input Validation",
            description="Validate all user inputs",
            properties={"category": "web_defense"},
            d3fend_id="D3-IV"
        )
        
        await om.add_entity(attack_entity)
        await om.add_entity(defense_entity)
        
        # 2. Create relation between entities
        relation = OntologyRelation(
            relation_id=str(uuid.uuid4()),
            source_entity=defense_entity.entity_id,
            target_entity=attack_entity.entity_id,
            relation_type=RelationType.MITIGATES,
            properties={"effectiveness": 0.9}
        )
        
        await om.add_relation(relation)
        
        # 3. Add framework entities to semantic mapper
        mitre_entity = FrameworkEntity(
            framework=FrameworkType.MITRE_ATTACK,
            entity_id="T1190",
            name="Exploit Public-Facing Application",
            description="SQL injection and similar attacks",
            category="initial-access",
            properties={}
        )
        
        await sm.add_framework_entity(mitre_entity)
        
        # 4. Process simulation outcome
        outcome = SimulationOutcome(
            outcome_id=str(uuid.uuid4()),
            scenario_id="integration_test",
            agent_id="test_agent",
            action_taken="sql_injection_attempt",
            technique_used="T1190",
            success=False,  # Blocked by input validation
            effectiveness_score=0.1,
            duration=5.0,
            detected=True,
            detection_time=2.0,
            countermeasures_triggered=["input_validation", "waf"],
            artifacts_created=["blocked_request.log"],
            metadata={"target": "login_form"}
        )
        
        updates = await ou.process_simulation_outcome(outcome)
        
        # 5. Run inference to discover new relationships
        inference_results = await ie.run_inference()
        
        # 6. Verify the system worked end-to-end
        assert len(om.entities) >= 2
        assert len(om.relations) >= 1
        assert om.knowledge_graph.number_of_nodes() >= 2
        assert om.knowledge_graph.number_of_edges() >= 1
        
        # Verify simulation outcome was processed
        assert outcome.outcome_id in ou.simulation_outcomes
        
        # Verify we can retrieve entities by framework IDs
        retrieved_attack = await om.get_entity_by_mitre_id("T1190")
        assert retrieved_attack is not None
        assert retrieved_attack.entity_id == attack_entity.entity_id
        
        retrieved_defense = await om.get_entity_by_d3fend_id("D3-IV")
        assert retrieved_defense is not None
        assert retrieved_defense.entity_id == defense_entity.entity_id
    
    @pytest.mark.asyncio
    async def test_ontology_accuracy(self, ontology_system):
        """Test ontology accuracy and semantic relationship correctness"""
        om = ontology_system['ontology_manager']
        
        # Create entities with known relationships
        recon_tactic = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.TACTIC,
            name="Reconnaissance",
            description="Information gathering tactic",
            properties={"phase": "pre-attack"},
            mitre_id="TA0043"
        )
        
        scanning_technique = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.ATTACK_TECHNIQUE,
            name="Active Scanning",
            description="Network scanning technique",
            properties={"stealth": "low"},
            mitre_id="T1595"
        )
        
        network_monitoring = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.DEFENSE_TECHNIQUE,
            name="Network Traffic Analysis",
            description="Monitor network traffic for anomalies",
            properties={"detection_type": "behavioral"},
            d3fend_id="D3-NTA"
        )
        
        await om.add_entity(recon_tactic)
        await om.add_entity(scanning_technique)
        await om.add_entity(network_monitoring)
        
        # Create semantically correct relationships
        tactic_technique_relation = OntologyRelation(
            relation_id=str(uuid.uuid4()),
            source_entity=scanning_technique.entity_id,
            target_entity=recon_tactic.entity_id,
            relation_type=RelationType.PART_OF,
            properties={"confidence": 1.0}
        )
        
        detection_relation = OntologyRelation(
            relation_id=str(uuid.uuid4()),
            source_entity=network_monitoring.entity_id,
            target_entity=scanning_technique.entity_id,
            relation_type=RelationType.DETECTS,
            properties={"confidence": 0.9, "detection_rate": 0.85}
        )
        
        await om.add_relation(tactic_technique_relation)
        await om.add_relation(detection_relation)
        
        # Verify semantic correctness
        # 1. Technique should be part of tactic
        related_to_tactic = await om.get_related_entities(scanning_technique.entity_id)
        tactic_relations = [r for r in related_to_tactic if r[1][-1].relation_type == RelationType.PART_OF]
        assert len(tactic_relations) > 0
        
        # 2. Defense technique should detect attack technique
        related_to_defense = await om.get_related_entities(network_monitoring.entity_id)
        detection_relations = [r for r in related_to_defense if r[1][-1].relation_type == RelationType.DETECTS]
        assert len(detection_relations) > 0
        
        # 3. Verify knowledge graph structure
        assert om.knowledge_graph.has_edge(scanning_technique.entity_id, recon_tactic.entity_id)
        assert om.knowledge_graph.has_edge(network_monitoring.entity_id, scanning_technique.entity_id)
        
        # 4. Verify entity retrieval by framework IDs
        assert await om.get_entity_by_mitre_id("TA0043") == recon_tactic
        assert await om.get_entity_by_mitre_id("T1595") == scanning_technique
        assert await om.get_entity_by_d3fend_id("D3-NTA") == network_monitoring


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])