"""
Demonstration of Control and Data Plane Separation

This script demonstrates the separation between control plane (agent decision-making)
and data plane (environment state management) in the Archangel system.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any

from agents.control_plane import (
    get_control_plane,
    shutdown_control_plane,
    DecisionType
)
from agents.data_plane import (
    get_data_plane,
    shutdown_data_plane
)
from agents.plane_coordinator import (
    get_plane_coordinator,
    shutdown_plane_coordinator,
    AgentPlaneAdapter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlanesSeparationDemo:
    """Demonstrates control and data plane separation"""
    
    def __init__(self):
        self.control_plane = None
        self.data_plane = None
        self.coordinator = None
        self.agents = {}
    
    async def initialize(self):
        """Initialize all planes and coordination"""
        logger.info("Initializing plane separation demonstration...")
        
        # Initialize planes
        self.control_plane = await get_control_plane()
        self.data_plane = await get_data_plane()
        self.coordinator = await get_plane_coordinator()
        
        logger.info("All planes initialized successfully")
    
    async def shutdown(self):
        """Shutdown all components"""
        logger.info("Shutting down plane separation demonstration...")
        
        await shutdown_control_plane()
        await shutdown_data_plane()
        await shutdown_plane_coordinator()
        
        logger.info("Shutdown complete")
    
    async def demonstrate_control_plane_isolation(self):
        """Demonstrate control plane operates independently"""
        logger.info("\n=== CONTROL PLANE ISOLATION DEMONSTRATION ===")
        
        # Register agents with control plane
        agent_ids = ["red_team_agent", "blue_team_agent", "coordinator_agent"]
        
        for agent_id in agent_ids:
            logger.info(f"Registering agent: {agent_id}")
            
            # Register with control plane
            decision_engine = self.control_plane.register_agent(agent_id)
            await decision_engine.initialize()
            
            # Store for later use
            self.agents[agent_id] = decision_engine
            
            logger.info(f"Agent {agent_id} registered and initialized")
        
        # Demonstrate decision making in control plane
        logger.info("\nDemonstrating isolated decision making...")
        
        for agent_id, decision_engine in self.agents.items():
            logger.info(f"\nAgent {agent_id} making decisions:")
            
            # Make different types of decisions
            for decision_type in [DecisionType.TACTICAL, DecisionType.STRATEGIC]:
                context = {
                    "agent_role": agent_id.split("_")[0],
                    "scenario": "demonstration",
                    "phase": "testing",
                    "objectives": [f"demonstrate_{decision_type.value}_decision"]
                }
                
                decision = await decision_engine.make_decision(
                    decision_type=decision_type,
                    context=context,
                    constraints=["demonstration_only", "no_real_impact"]
                )
                
                logger.info(f"  {decision_type.value.upper()} Decision:")
                logger.info(f"    Action: {decision.action}")
                logger.info(f"    Confidence: {decision.confidence:.2f}")
                logger.info(f"    Reasoning: {decision.reasoning}")
        
        # Demonstrate coordination within control plane
        logger.info("\nDemonstrating control plane coordination...")
        
        coordination_manager = self.control_plane.get_coordination_manager()
        
        request_id = await coordination_manager.request_coordination(
            initiator_agent_id="red_team_agent",
            target_agents=["blue_team_agent"],
            coordination_type="adversarial_engagement",
            payload={
                "engagement_type": "reconnaissance_vs_detection",
                "scenario": "demonstration"
            },
            priority=8
        )
        
        logger.info(f"Coordination request created: {request_id}")
        
        # Wait for coordination processing
        await asyncio.sleep(0.5)
        
        status = coordination_manager.get_coordination_status(request_id)
        logger.info(f"Coordination status: {status}")
        
        # Show control plane metrics
        metrics = self.control_plane.get_overall_metrics()
        logger.info(f"\nControl Plane Metrics:")
        logger.info(f"  Active Agents: {metrics.active_agents}")
        logger.info(f"  Total Decisions: {metrics.total_decisions}")
        logger.info(f"  Decisions/Second: {metrics.decisions_per_second:.2f}")
        logger.info(f"  Coordination Success Rate: {metrics.coordination_success_rate:.2f}")
    
    async def demonstrate_data_plane_isolation(self):
        """Demonstrate data plane operates independently"""
        logger.info("\n=== DATA PLANE ISOLATION DEMONSTRATION ===")
        
        # Get data plane components
        state_manager = self.data_plane.get_state_manager()
        sim_executor = self.data_plane.get_simulation_executor()
        
        # Create mock enterprise environment
        logger.info("Creating mock enterprise environment...")
        
        # Create network infrastructure entities
        entities = {}
        
        # DMZ servers
        dmz_entities = [
            ("web_server", {"type": "web_server", "service": "nginx", "port": 80, "vulnerable": True}),
            ("mail_server", {"type": "mail_server", "service": "postfix", "port": 25, "config": "default"}),
            ("dns_server", {"type": "dns_server", "service": "bind9", "port": 53, "zone_transfer": True})
        ]
        
        for entity_name, properties in dmz_entities:
            entity_id = state_manager.create_entity(
                entity_type="server",
                properties=properties,
                position={"zone": "dmz", "x": len(entities) * 100, "y": 100}
            )
            entities[entity_name] = entity_id
            logger.info(f"Created {entity_name}: {entity_id}")
        
        # Internal network entities
        internal_entities = [
            ("file_server", {"type": "file_server", "service": "smb", "shares": ["public", "admin"], "permissions": "weak"}),
            ("database", {"type": "database", "service": "mysql", "version": "5.7", "root_password": "admin123"}),
            ("domain_controller", {"type": "domain_controller", "service": "active_directory", "domain": "corp.local"})
        ]
        
        for entity_name, properties in internal_entities:
            entity_id = state_manager.create_entity(
                entity_type="server",
                properties=properties,
                position={"zone": "internal", "x": len(entities) * 100, "y": 200}
            )
            entities[entity_name] = entity_id
            logger.info(f"Created {entity_name}: {entity_id}")
        
        # Create network relationships
        logger.info("\nCreating network relationships...")
        
        # DMZ to Internal connections
        state_manager.create_relationship(
            entities["web_server"], entities["database"], "connects_to"
        )
        state_manager.create_relationship(
            entities["mail_server"], entities["domain_controller"], "authenticates_with"
        )
        
        logger.info("Network relationships established")
        
        # Demonstrate environment queries
        logger.info("\nDemonstrating environment state queries...")
        
        # Query all servers
        all_servers = state_manager.query_entities(entity_type="server")
        logger.info(f"Total servers in environment: {len(all_servers)}")
        
        # Query by zone
        dmz_servers = state_manager.query_entities(
            entity_type="server",
            properties_filter={"zone": "dmz"}
        )
        logger.info(f"DMZ servers: {len(dmz_servers)}")
        
        # Query vulnerable services
        vulnerable_servers = []
        for server in all_servers:
            if server.properties.get("vulnerable") or server.properties.get("root_password"):
                vulnerable_servers.append(server)
        
        logger.info(f"Potentially vulnerable servers: {len(vulnerable_servers)}")
        
        # Demonstrate simulation execution
        logger.info("\nDemonstrating simulation execution...")
        
        # Start simulation
        await sim_executor.start_simulation(time_scale=5.0)
        logger.info("Simulation started with 5x time scale")
        
        initial_time = sim_executor.simulation_time
        logger.info(f"Initial simulation time: {initial_time:.2f}")
        
        # Let simulation run
        await asyncio.sleep(1.0)
        
        current_time = sim_executor.simulation_time
        logger.info(f"Simulation time after 1 second: {current_time:.2f}")
        logger.info(f"Time progression: {current_time - initial_time:.2f} simulation seconds")
        
        # Demonstrate state changes during simulation
        logger.info("\nSimulating environment changes...")
        
        # Simulate attack progression
        state_manager.update_entity(
            entities["web_server"],
            properties={
                **state_manager.get_entity(entities["web_server"]).properties,
                "compromised": True,
                "compromise_time": current_time
            }
        )
        logger.info("Web server marked as compromised")
        
        # Simulate defensive response
        state_manager.update_entity(
            entities["web_server"],
            properties={
                **state_manager.get_entity(entities["web_server"]).properties,
                "firewall_rules_updated": True,
                "monitoring_enabled": True
            }
        )
        logger.info("Defensive measures applied to web server")
        
        # Stop simulation
        await sim_executor.stop_simulation()
        logger.info("Simulation stopped")
        
        # Create snapshot
        snapshot_id = self.data_plane.create_snapshot()
        logger.info(f"Environment snapshot created: {snapshot_id}")
        
        # Show data plane metrics
        metrics = self.data_plane.get_overall_metrics()
        logger.info(f"\nData Plane Metrics:")
        logger.info(f"  Status: {metrics['status']}")
        logger.info(f"  Entities: {metrics['state_manager']['entities_count']}")
        logger.info(f"  State Changes: {metrics['state_manager']['state_changes']}")
        logger.info(f"  Queries/Second: {metrics['state_manager']['queries_per_second']:.2f}")
    
    async def demonstrate_plane_coordination(self):
        """Demonstrate coordination between planes"""
        logger.info("\n=== PLANE COORDINATION DEMONSTRATION ===")
        
        # Create agent adapters that bridge both planes
        adapters = {}
        
        for agent_id in ["red_recon_agent", "blue_soc_agent"]:
            logger.info(f"Creating plane adapter for {agent_id}")
            
            # Register with control plane
            decision_engine = self.control_plane.register_agent(agent_id)
            await decision_engine.initialize()
            
            # Create adapter for cross-plane operations
            adapter = AgentPlaneAdapter(agent_id)
            await adapter.initialize()
            
            adapters[agent_id] = (decision_engine, adapter)
            logger.info(f"Agent {agent_id} ready for cross-plane operations")
        
        # Demonstrate Red Team reconnaissance through coordination
        logger.info("\nDemonstrating Red Team reconnaissance via plane coordination...")
        
        red_engine, red_adapter = adapters["red_recon_agent"]
        
        # Red agent makes decision to perform reconnaissance
        recon_decision = await red_engine.make_decision(
            decision_type=DecisionType.TACTICAL,
            context={
                "phase": "reconnaissance",
                "target": "enterprise_network",
                "objective": "identify_attack_surface"
            }
        )
        
        logger.info(f"Red Team Decision: {recon_decision.action}")
        logger.info(f"Reasoning: {recon_decision.reasoning}")
        
        # Execute reconnaissance through data plane
        logger.info("Executing reconnaissance through data plane...")
        
        # Query environment for targets
        servers = await red_adapter.find_entities(entity_type="server")
        logger.info(f"Red Team discovered {len(servers)} servers")
        
        # Analyze discovered targets
        vulnerable_targets = []
        for server in servers:
            properties = server.get('properties', {})
            if (properties.get('vulnerable') or 
                properties.get('root_password') or 
                properties.get('zone') == 'dmz'):
                vulnerable_targets.append(server)
        
        logger.info(f"Red Team identified {len(vulnerable_targets)} potential targets")
        
        # Demonstrate Blue Team detection through coordination
        logger.info("\nDemonstrating Blue Team detection via plane coordination...")
        
        blue_engine, blue_adapter = adapters["blue_soc_agent"]
        
        # Blue agent makes decision to monitor for threats
        detection_decision = await blue_engine.make_decision(
            decision_type=DecisionType.TACTICAL,
            context={
                "phase": "monitoring",
                "focus": "anomaly_detection",
                "objective": "identify_reconnaissance_activity"
            }
        )
        
        logger.info(f"Blue Team Decision: {detection_decision.action}")
        logger.info(f"Reasoning: {detection_decision.reasoning}")
        
        # Monitor environment through data plane
        logger.info("Blue Team monitoring environment through data plane...")
        
        # Check for compromised systems
        compromised_systems = await blue_adapter.find_entities(
            entity_type="server",
            compromised=True
        )
        
        if compromised_systems:
            logger.info(f"Blue Team detected {len(compromised_systems)} compromised systems")
            
            # Create incident response entity
            incident_id = await blue_adapter.create_entity(
                entity_type="security_incident",
                properties={
                    "type": "compromise_detected",
                    "severity": "high",
                    "affected_systems": len(compromised_systems),
                    "detection_time": datetime.now().isoformat(),
                    "status": "investigating"
                }
            )
            
            logger.info(f"Security incident created: {incident_id}")
        else:
            logger.info("Blue Team: No compromised systems detected")
        
        # Demonstrate coordination between teams
        logger.info("\nDemonstrating inter-team coordination...")
        
        coordination_manager = self.control_plane.get_coordination_manager()
        
        # Red team coordinates attack escalation
        red_coord_request = await coordination_manager.request_coordination(
            initiator_agent_id="red_recon_agent",
            target_agents=["red_exploit_agent"],  # Hypothetical exploit agent
            coordination_type="attack_escalation",
            payload={
                "targets": [server['entity_id'] for server in vulnerable_targets[:2]],
                "reconnaissance_complete": True,
                "priority_targets": ["web_server", "database"]
            }
        )
        
        logger.info(f"Red Team coordination request: {red_coord_request}")
        
        # Blue team coordinates response
        blue_coord_request = await coordination_manager.request_coordination(
            initiator_agent_id="blue_soc_agent",
            target_agents=["blue_firewall_agent"],  # Hypothetical firewall agent
            coordination_type="threat_response",
            payload={
                "threat_level": "high",
                "response_type": "containment",
                "affected_systems": [s['entity_id'] for s in compromised_systems]
            }
        )
        
        logger.info(f"Blue Team coordination request: {blue_coord_request}")
        
        # Wait for coordination processing
        await asyncio.sleep(0.5)
        
        # Check coordination results
        red_status = coordination_manager.get_coordination_status(red_coord_request)
        blue_status = coordination_manager.get_coordination_status(blue_coord_request)
        
        logger.info(f"Red Team coordination status: {red_status}")
        logger.info(f"Blue Team coordination status: {blue_status}")
        
        # Show coordination metrics
        coord_metrics = coordination_manager.get_metrics()
        logger.info(f"\nCoordination Metrics:")
        logger.info(f"  Active Agents: {coord_metrics['active_agents']}")
        logger.info(f"  Coordination Requests: {coord_metrics['coordination_requests']}")
        logger.info(f"  Successful Coordinations: {coord_metrics['successful_coordinations']}")
        logger.info(f"  Average Response Time: {coord_metrics['average_response_time']:.3f}s")
    
    async def demonstrate_isolation_benefits(self):
        """Demonstrate benefits of plane separation"""
        logger.info("\n=== ISOLATION BENEFITS DEMONSTRATION ===")
        
        logger.info("1. SCALABILITY BENEFITS:")
        logger.info("   - Control plane can scale decision-making independently")
        logger.info("   - Data plane can scale environment simulation independently")
        logger.info("   - Each plane optimized for its specific workload")
        
        logger.info("\n2. RELIABILITY BENEFITS:")
        logger.info("   - Control plane failures don't affect environment state")
        logger.info("   - Data plane issues don't impact agent decision-making")
        logger.info("   - Isolated recovery and fault tolerance")
        
        logger.info("\n3. SECURITY BENEFITS:")
        logger.info("   - Agents cannot directly manipulate environment state")
        logger.info("   - All environment access is mediated and logged")
        logger.info("   - Clear audit trail of cross-plane interactions")
        
        logger.info("\n4. PERFORMANCE BENEFITS:")
        logger.info("   - Decision-making optimized for reasoning workloads")
        logger.info("   - Environment simulation optimized for state management")
        logger.info("   - Reduced contention and improved throughput")
        
        # Demonstrate performance isolation
        logger.info("\nDemonstrating performance isolation...")
        
        # Stress test control plane
        start_time = time.time()
        
        decision_tasks = []
        for agent_id, decision_engine in self.agents.items():
            for i in range(10):
                task = decision_engine.make_decision(
                    decision_type=DecisionType.TACTICAL,
                    context={"stress_test": True, "iteration": i}
                )
                decision_tasks.append(task)
        
        await asyncio.gather(*decision_tasks)
        
        control_time = time.time() - start_time
        logger.info(f"Control plane: {len(decision_tasks)} decisions in {control_time:.2f}s")
        
        # Stress test data plane
        start_time = time.time()
        
        state_manager = self.data_plane.get_state_manager()
        
        entity_tasks = []
        for i in range(50):
            entity_id = state_manager.create_entity(
                entity_type="stress_test_entity",
                properties={"iteration": i, "stress_test": True}
            )
            entity_tasks.append(entity_id)
        
        data_time = time.time() - start_time
        logger.info(f"Data plane: {len(entity_tasks)} entities in {data_time:.2f}s")
        
        logger.info(f"Performance isolation maintained - operations completed independently")
    
    async def run_demonstration(self):
        """Run the complete demonstration"""
        try:
            await self.initialize()
            
            logger.info("Starting Control and Data Plane Separation Demonstration")
            logger.info("=" * 60)
            
            await self.demonstrate_control_plane_isolation()
            await self.demonstrate_data_plane_isolation()
            await self.demonstrate_plane_coordination()
            await self.demonstrate_isolation_benefits()
            
            logger.info("\n" + "=" * 60)
            logger.info("Plane Separation Demonstration Complete!")
            logger.info("\nKey Takeaways:")
            logger.info("- Control and data planes operate independently")
            logger.info("- Coordination enables necessary cross-plane communication")
            logger.info("- Isolation provides scalability, reliability, and security benefits")
            logger.info("- Architecture supports distributed, fault-tolerant execution")
            
        except Exception as e:
            logger.error(f"Demonstration failed: {e}")
            raise
        finally:
            await self.shutdown()


async def main():
    """Main demonstration function"""
    demo = PlanesSeparationDemo()
    await demo.run_demonstration()


if __name__ == "__main__":
    asyncio.run(main())