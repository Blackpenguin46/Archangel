#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Ontology System Demo
Demonstrates the ontology-driven knowledge base with semantic mapping
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

# Import ontology modules
from memory.ontology_manager import (
    OntologyManager, OntologyEntity, OntologyRelation, 
    EntityType, RelationType
)
from memory.semantic_mapper import (
    SemanticMapper, FrameworkEntity, FrameworkType, 
    MappingConfidence, SemanticMapping
)
from memory.inference_engine import (
    InferenceEngine, InferenceRule, InferenceType
)
from memory.ontology_updater import (
    OntologyUpdater, SimulationOutcome, ThreatIntelligence
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OntologySystemDemo:
    """Demonstration of the complete ontology system"""
    
    def __init__(self):
        self.ontology_manager = None
        self.semantic_mapper = None
        self.inference_engine = None
        self.ontology_updater = None
    
    async def initialize(self):
        """Initialize all ontology system components"""
        logger.info("🚀 Initializing Ontology System Demo")
        
        # Initialize components
        self.ontology_manager = OntologyManager("./demo_kb/ontology")
        self.semantic_mapper = SemanticMapper("./demo_kb/mappings")
        self.ontology_updater = OntologyUpdater("./demo_kb/updates")
        
        await self.ontology_manager.initialize()
        await self.semantic_mapper.initialize()
        await self.ontology_updater.initialize()
        
        # Initialize inference engine with knowledge graph
        self.inference_engine = InferenceEngine("./demo_kb/inference")
        await self.inference_engine.initialize(self.ontology_manager.knowledge_graph)
        
        logger.info("✅ All components initialized successfully")
    
    async def demonstrate_ontology_creation(self):
        """Demonstrate creating ontology entities and relationships"""
        logger.info("\n📊 Demonstrating Ontology Creation")
        
        # Create MITRE ATT&CK entities
        logger.info("Creating MITRE ATT&CK entities...")
        
        # Tactics
        recon_tactic = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.TACTIC,
            name="Reconnaissance",
            description="The adversary is trying to gather information they can use to plan future operations.",
            properties={
                "phase": "pre-attack",
                "mitre_id": "TA0043"
            },
            mitre_id="TA0043"
        )
        
        initial_access_tactic = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.TACTIC,
            name="Initial Access",
            description="The adversary is trying to get into your network.",
            properties={
                "phase": "attack",
                "mitre_id": "TA0001"
            },
            mitre_id="TA0001"
        )
        
        # Techniques
        active_scanning = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.ATTACK_TECHNIQUE,
            name="Active Scanning",
            description="Adversaries may execute active reconnaissance scans to gather information.",
            properties={
                "stealth_level": "low",
                "detection_difficulty": "easy",
                "platforms": ["PRE"],
                "data_sources": ["Network Traffic"]
            },
            mitre_id="T1595"
        )
        
        exploit_public_app = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.ATTACK_TECHNIQUE,
            name="Exploit Public-Facing Application",
            description="Adversaries may attempt to exploit weaknesses in Internet-facing applications.",
            properties={
                "stealth_level": "medium",
                "detection_difficulty": "medium",
                "platforms": ["Linux", "Windows", "macOS"],
                "data_sources": ["Application Log", "Network Traffic"]
            },
            mitre_id="T1190"
        )
        
        # D3FEND entities
        logger.info("Creating D3FEND defense entities...")
        
        network_analysis = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.DEFENSE_TECHNIQUE,
            name="Network Traffic Analysis",
            description="Analyzing network traffic to detect malicious activity.",
            properties={
                "detection_type": "behavioral",
                "effectiveness": 0.85,
                "false_positive_rate": 0.05
            },
            d3fend_id="D3-NTA"
        )
        
        app_hardening = OntologyEntity(
            entity_id=str(uuid.uuid4()),
            entity_type=EntityType.DEFENSE_TECHNIQUE,
            name="Application Hardening",
            description="Hardening applications against exploitation attempts.",
            properties={
                "prevention_type": "proactive",
                "effectiveness": 0.90,
                "implementation_complexity": "medium"
            },
            d3fend_id="D3-AH"
        )
        
        # Add entities to ontology
        entities = [
            recon_tactic, initial_access_tactic, active_scanning, 
            exploit_public_app, network_analysis, app_hardening
        ]
        
        for entity in entities:
            await self.ontology_manager.add_entity(entity)
            logger.info(f"  ➕ Added {entity.entity_type.value}: {entity.name}")
        
        # Create relationships
        logger.info("Creating semantic relationships...")
        
        # Technique -> Tactic relationships
        scanning_recon_rel = OntologyRelation(
            relation_id=str(uuid.uuid4()),
            source_entity=active_scanning.entity_id,
            target_entity=recon_tactic.entity_id,
            relation_type=RelationType.PART_OF,
            properties={"confidence": 1.0}
        )
        
        exploit_initial_rel = OntologyRelation(
            relation_id=str(uuid.uuid4()),
            source_entity=exploit_public_app.entity_id,
            target_entity=initial_access_tactic.entity_id,
            relation_type=RelationType.PART_OF,
            properties={"confidence": 1.0}
        )
        
        # Defense -> Attack relationships
        network_detects_scanning = OntologyRelation(
            relation_id=str(uuid.uuid4()),
            source_entity=network_analysis.entity_id,
            target_entity=active_scanning.entity_id,
            relation_type=RelationType.DETECTS,
            properties={"confidence": 0.9, "detection_rate": 0.85}
        )
        
        hardening_mitigates_exploit = OntologyRelation(
            relation_id=str(uuid.uuid4()),
            source_entity=app_hardening.entity_id,
            target_entity=exploit_public_app.entity_id,
            relation_type=RelationType.MITIGATES,
            properties={"confidence": 0.95, "mitigation_rate": 0.90}
        )
        
        relations = [
            scanning_recon_rel, exploit_initial_rel,
            network_detects_scanning, hardening_mitigates_exploit
        ]
        
        for relation in relations:
            await self.ontology_manager.add_relation(relation)
            source_name = next(e.name for e in entities if e.entity_id == relation.source_entity)
            target_name = next(e.name for e in entities if e.entity_id == relation.target_entity)
            logger.info(f"  🔗 {source_name} {relation.relation_type.value} {target_name}")
        
        # Display ontology statistics
        stats = await self.ontology_manager.get_ontology_statistics()
        logger.info(f"\n📈 Ontology Statistics:")
        logger.info(f"  Total Entities: {stats['entities']['total']}")
        logger.info(f"  Total Relations: {stats['relations']['total']}")
        logger.info(f"  Knowledge Graph Nodes: {stats['knowledge_graph']['nodes']}")
        logger.info(f"  Knowledge Graph Edges: {stats['knowledge_graph']['edges']}")
    
    async def demonstrate_semantic_mapping(self):
        """Demonstrate semantic mapping between frameworks"""
        logger.info("\n🔄 Demonstrating Semantic Mapping")
        
        # Add framework entities to semantic mapper
        logger.info("Adding framework entities...")
        
        mitre_entities = [
            FrameworkEntity(
                framework=FrameworkType.MITRE_ATTACK,
                entity_id="T1595",
                name="Active Scanning",
                description="Adversaries may execute active reconnaissance scans",
                category="reconnaissance",
                properties={
                    "tactics": ["reconnaissance"],
                    "platforms": ["PRE"],
                    "detection_difficulty": "easy"
                }
            ),
            FrameworkEntity(
                framework=FrameworkType.MITRE_ATTACK,
                entity_id="T1190",
                name="Exploit Public-Facing Application",
                description="Adversaries may exploit weaknesses in Internet-facing applications",
                category="initial-access",
                properties={
                    "tactics": ["initial-access"],
                    "platforms": ["Linux", "Windows", "macOS"],
                    "detection_difficulty": "medium"
                }
            )
        ]
        
        d3fend_entities = [
            FrameworkEntity(
                framework=FrameworkType.D3FEND,
                entity_id="D3-NTA",
                name="Network Traffic Analysis",
                description="Analyzing network traffic to detect malicious activity",
                category="detect",
                properties={
                    "detection_type": "behavioral",
                    "related_offensive_techniques": ["T1595", "T1046"]
                }
            ),
            FrameworkEntity(
                framework=FrameworkType.D3FEND,
                entity_id="D3-AH",
                name="Application Hardening",
                description="Hardening applications against exploitation",
                category="harden",
                properties={
                    "prevention_type": "proactive",
                    "related_offensive_techniques": ["T1190", "T1068"]
                }
            )
        ]
        
        # Add entities to semantic mapper
        for entity in mitre_entities + d3fend_entities:
            await self.semantic_mapper.add_framework_entity(entity)
            logger.info(f"  ➕ Added {entity.framework.value}: {entity.name}")
        
        # Generate automatic mappings
        logger.info("Generating automatic semantic mappings...")
        mappings = await self.semantic_mapper.generate_automatic_mappings(
            FrameworkType.MITRE_ATTACK,
            FrameworkType.D3FEND
        )
        
        logger.info(f"  🤖 Generated {len(mappings)} automatic mappings")
        
        # Create manual high-confidence mappings
        logger.info("Creating manual semantic mappings...")
        
        manual_mappings = [
            SemanticMapping(
                mapping_id=str(uuid.uuid4()),
                source_framework=FrameworkType.MITRE_ATTACK,
                source_entity_id="T1595",
                target_framework=FrameworkType.D3FEND,
                target_entity_id="D3-NTA",
                mapping_type="detection",
                confidence=MappingConfidence.HIGH,
                confidence_score=0.9,
                evidence=[
                    "Network Traffic Analysis directly detects Active Scanning",
                    "Both techniques involve network traffic monitoring"
                ],
                properties={"detection_effectiveness": 0.85}
            ),
            SemanticMapping(
                mapping_id=str(uuid.uuid4()),
                source_framework=FrameworkType.MITRE_ATTACK,
                source_entity_id="T1190",
                target_framework=FrameworkType.D3FEND,
                target_entity_id="D3-AH",
                mapping_type="mitigation",
                confidence=MappingConfidence.HIGH,
                confidence_score=0.95,
                evidence=[
                    "Application Hardening directly mitigates public-facing application exploits",
                    "Proactive defense against application vulnerabilities"
                ],
                properties={"mitigation_effectiveness": 0.90}
            )
        ]
        
        for mapping in manual_mappings:
            await self.semantic_mapper.add_semantic_mapping(mapping)
            logger.info(f"  🔗 Mapped {mapping.source_entity_id} -> {mapping.target_entity_id} ({mapping.mapping_type})")
        
        # Display mapping statistics
        stats = await self.semantic_mapper.get_mapping_statistics()
        logger.info(f"\n📊 Semantic Mapping Statistics:")
        logger.info(f"  Total Mappings: {stats['total_mappings']}")
        logger.info(f"  High Confidence: {stats['by_confidence'].get('high', 0)}")
        logger.info(f"  Medium Confidence: {stats['by_confidence'].get('medium', 0)}")
        logger.info(f"  Framework Entities: {sum(stats['framework_entities'].values())}")
    
    async def demonstrate_inference_engine(self):
        """Demonstrate knowledge inference and pattern discovery"""
        logger.info("\n🧠 Demonstrating Inference Engine")
        
        # Run inference to discover new relationships
        logger.info("Running inference engine...")
        inference_results = await self.inference_engine.run_inference()
        
        logger.info(f"  🔍 Generated {len(inference_results)} inferences")
        
        for i, result in enumerate(inference_results[:3]):  # Show first 3 results
            logger.info(f"  {i+1}. {result.inference_type.value} inference:")
            logger.info(f"     Confidence: {result.confidence_score:.2f}")
            logger.info(f"     Evidence: {', '.join(result.evidence[:2])}")
        
        # Discover knowledge patterns
        logger.info("Discovering knowledge patterns...")
        patterns = await self.inference_engine.discover_patterns(min_frequency=2)
        
        logger.info(f"  📋 Discovered {len(patterns)} knowledge patterns")
        
        for i, pattern in enumerate(patterns[:2]):  # Show first 2 patterns
            logger.info(f"  {i+1}. {pattern.pattern_type} pattern: {pattern.name}")
            logger.info(f"     Frequency: {pattern.frequency}, Confidence: {pattern.confidence:.2f}")
        
        # Demonstrate confidence propagation
        logger.info("Demonstrating confidence propagation...")
        
        # Get a sample entity for propagation
        sample_entities = list(self.ontology_manager.entities.keys())
        if sample_entities:
            sample_entity = sample_entities[0]
            confidence_scores = await self.inference_engine.propagate_confidence(sample_entity)
            
            logger.info(f"  🌊 Propagated confidence from {sample_entity}:")
            for entity_id, score in list(confidence_scores.items())[:3]:
                entity_name = self.ontology_manager.entities.get(entity_id, {}).name or entity_id
                logger.info(f"     {entity_name}: {score:.3f}")
        
        # Get inference statistics
        stats = await self.inference_engine.get_inference_statistics()
        logger.info(f"\n📈 Inference Statistics:")
        logger.info(f"  Total Rules: {stats['total_rules']}")
        logger.info(f"  Enabled Rules: {stats['enabled_rules']}")
        logger.info(f"  Total Results: {stats['total_results']}")
        logger.info(f"  Success Rate: {stats['success_rate']:.1f}%")
    
    async def demonstrate_ontology_learning(self):
        """Demonstrate learning from simulation outcomes"""
        logger.info("\n📚 Demonstrating Ontology Learning")
        
        # Simulate various attack scenarios
        logger.info("Processing simulation outcomes...")
        
        scenarios = [
            {
                "name": "Successful Network Scan",
                "technique": "T1595",
                "success": True,
                "effectiveness": 0.8,
                "detected": False,
                "countermeasures": []
            },
            {
                "name": "Blocked Network Scan",
                "technique": "T1595",
                "success": False,
                "effectiveness": 0.2,
                "detected": True,
                "countermeasures": ["network_monitoring", "firewall"]
            },
            {
                "name": "Web Application Exploit",
                "technique": "T1190",
                "success": True,
                "effectiveness": 0.9,
                "detected": False,
                "countermeasures": []
            },
            {
                "name": "Mitigated Web Exploit",
                "technique": "T1190",
                "success": False,
                "effectiveness": 0.1,
                "detected": True,
                "countermeasures": ["application_hardening", "waf"]
            },
            {
                "name": "Stealthy Reconnaissance",
                "technique": "T1595",
                "success": True,
                "effectiveness": 0.7,
                "detected": False,
                "countermeasures": []
            }
        ]
        
        all_updates = []
        
        for scenario in scenarios:
            outcome = SimulationOutcome(
                outcome_id=str(uuid.uuid4()),
                scenario_id=f"demo_{scenario['name'].lower().replace(' ', '_')}",
                agent_id="demo_agent",
                action_taken=scenario['name'],
                technique_used=scenario['technique'],
                success=scenario['success'],
                effectiveness_score=scenario['effectiveness'],
                duration=30.0,
                detected=scenario['detected'],
                detection_time=5.0 if scenario['detected'] else None,
                countermeasures_triggered=scenario['countermeasures'],
                artifacts_created=[f"{scenario['name'].lower()}.log"],
                metadata={"demo": True}
            )
            
            updates = await self.ontology_updater.process_simulation_outcome(outcome)
            all_updates.extend(updates)
            
            status = "✅ Success" if scenario['success'] else "❌ Blocked"
            detection = "🔍 Detected" if scenario['detected'] else "👻 Undetected"
            logger.info(f"  {status} {detection} - {scenario['name']} (Effectiveness: {scenario['effectiveness']:.1f})")
        
        # Process threat intelligence
        logger.info("Processing threat intelligence...")
        
        intel_reports = [
            ThreatIntelligence(
                intel_id=str(uuid.uuid4()),
                source="demo_threat_feed",
                intel_type="technique_update",
                technique_id="T1595",
                iocs=["192.168.1.100", "scanner.exe"],
                ttps=["port_scanning", "service_enumeration"],
                confidence_score=0.9,
                severity="medium",
                description="New scanning tools observed in the wild",
                references=["https://demo.threat-intel.com/report1"]
            ),
            ThreatIntelligence(
                intel_id=str(uuid.uuid4()),
                source="demo_security_vendor",
                intel_type="vulnerability_report",
                technique_id="T1190",
                iocs=["exploit.php", "malicious.js"],
                ttps=["sql_injection", "xss"],
                confidence_score=0.95,
                severity="high",
                description="Critical web application vulnerabilities being exploited",
                references=["https://demo.vendor.com/advisory"]
            )
        ]
        
        for intel in intel_reports:
            intel_updates = await self.ontology_updater.process_threat_intelligence(intel)
            all_updates.extend(intel_updates)
            logger.info(f"  📡 Processed {intel.intel_type} from {intel.source}")
        
        logger.info(f"  🔄 Generated {len(all_updates)} ontology updates")
        
        # Get learning insights
        insights = await self.ontology_updater.get_learning_insights()
        
        logger.info(f"\n🎯 Learning Insights:")
        logger.info(f"  Outcomes Processed: {insights['statistics']['total_outcomes_processed']}")
        logger.info(f"  Updates Generated: {insights['statistics']['total_updates_generated']}")
        
        if insights['top_techniques']:
            logger.info(f"  Top Technique: {insights['top_techniques'][0]['technique']} "
                       f"(Success Rate: {insights['top_techniques'][0]['success_rate']:.1%})")
    
    async def demonstrate_knowledge_export(self):
        """Demonstrate knowledge export and visualization"""
        logger.info("\n📤 Demonstrating Knowledge Export")
        
        # Export ontology statistics
        ontology_stats = await self.ontology_manager.get_ontology_statistics()
        
        # Export knowledge graph
        knowledge_graph = await self.ontology_manager.export_knowledge_graph()
        
        # Export semantic mappings
        mappings = await self.semantic_mapper.export_mappings()
        
        # Export inference statistics
        inference_stats = await self.inference_engine.get_inference_statistics()
        
        # Create comprehensive export
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "ontology_statistics": ontology_stats,
            "knowledge_graph": knowledge_graph,
            "semantic_mappings": mappings,
            "inference_statistics": inference_stats,
            "metadata": {
                "demo_version": "1.0",
                "components": ["ontology_manager", "semantic_mapper", "inference_engine", "ontology_updater"]
            }
        }
        
        # Save export
        export_file = Path("demo_ontology_export.json")
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"  💾 Exported complete ontology system to {export_file}")
        logger.info(f"  📊 Export contains {len(knowledge_graph['nodes'])} nodes and {len(knowledge_graph['edges'])} edges")
        logger.info(f"  🔗 Export contains {len(mappings['mappings'])} semantic mappings")
    
    async def run_complete_demo(self):
        """Run the complete ontology system demonstration"""
        try:
            await self.initialize()
            
            await self.demonstrate_ontology_creation()
            await self.demonstrate_semantic_mapping()
            await self.demonstrate_inference_engine()
            await self.demonstrate_ontology_learning()
            await self.demonstrate_knowledge_export()
            
            logger.info("\n🎉 Ontology System Demo Completed Successfully!")
            logger.info("\nKey Features Demonstrated:")
            logger.info("  ✅ Domain-specific ontology creation")
            logger.info("  ✅ MITRE ATT&CK and D3FEND integration")
            logger.info("  ✅ Semantic entity mapping")
            logger.info("  ✅ Knowledge graph inference")
            logger.info("  ✅ Automated learning from simulation outcomes")
            logger.info("  ✅ Threat intelligence integration")
            logger.info("  ✅ Knowledge export and visualization")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise

async def main():
    """Main demo function"""
    demo = OntologySystemDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    asyncio.run(main())