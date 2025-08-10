#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Agent Framework Demo
Demonstrates the foundational multi-agent coordination system
"""

import asyncio
import logging
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent, AgentConfig, Team, Role, EnvironmentState, ReasoningResult, ActionPlan, ActionResult
from agents.communication import MessageBus, create_team_message, CoordinationType, Priority
from agents.coordinator import LangGraphCoordinator, Scenario, Phase

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemoReconAgent(BaseAgent):
    """Demo Red Team reconnaissance agent"""
    
    async def reason_about_situation(self, state: EnvironmentState) -> ReasoningResult:
        """Analyze situation for reconnaissance opportunities"""
        return ReasoningResult(
            situation_assessment=f"Analyzing network topology with {len(state.active_services)} active services",
            threat_analysis="Low defensive presence detected",
            opportunity_identification=["network_scanning", "service_enumeration", "vulnerability_discovery"],
            risk_assessment={"detection_risk": 0.3, "success_probability": 0.8},
            recommended_actions=["perform_network_scan", "enumerate_services"],
            confidence_score=0.85,
            reasoning_chain=[
                "Observed network topology",
                "Identified active services",
                "Assessed defensive measures",
                "Calculated risk/reward ratio"
            ],
            alternatives_considered=["passive_reconnaissance", "active_scanning", "social_engineering"]
        )
    
    async def plan_actions(self, reasoning: ReasoningResult) -> ActionPlan:
        """Plan reconnaissance actions"""
        return ActionPlan(
            primary_action="network_scan",
            action_type="reconnaissance",
            target="192.168.1.0/24",
            parameters={
                "scan_type": "stealth",
                "ports": "1-1000",
                "timing": "slow"
            },
            expected_outcome="Discover active hosts and services",
            success_criteria=["hosts_discovered", "services_identified"],
            fallback_actions=["passive_dns_enumeration", "osint_gathering"],
            estimated_duration=30.0,
            risk_level="low"
        )
    
    async def execute_action(self, action: ActionPlan) -> ActionResult:
        """Execute reconnaissance action"""
        # Simulate reconnaissance execution
        await asyncio.sleep(2)  # Simulate work
        
        return ActionResult(
            action_id=f"recon_{datetime.now().timestamp()}",
            action_type=action.action_type,
            success=True,
            outcome="Discovered 5 active hosts with web services",
            data={
                "hosts_found": ["192.168.1.10", "192.168.1.20", "192.168.1.30"],
                "services": ["http:80", "https:443", "ssh:22"],
                "vulnerabilities": ["outdated_apache", "weak_ssh_config"]
            },
            duration=2.0,
            errors=[],
            side_effects=["network_traffic_generated"],
            confidence=0.9,
            timestamp=datetime.now()
        )

class DemoSOCAgent(BaseAgent):
    """Demo Blue Team SOC analyst agent"""
    
    async def reason_about_situation(self, state: EnvironmentState) -> ReasoningResult:
        """Analyze situation for security threats"""
        return ReasoningResult(
            situation_assessment=f"Monitoring {len(state.security_alerts)} security alerts",
            threat_analysis="Suspicious network scanning activity detected",
            opportunity_identification=["investigate_alerts", "correlate_events", "implement_countermeasures"],
            risk_assessment={"threat_level": 0.6, "containment_probability": 0.7},
            recommended_actions=["investigate_scanning", "enhance_monitoring"],
            confidence_score=0.75,
            reasoning_chain=[
                "Analyzed security alerts",
                "Correlated network events",
                "Assessed threat indicators",
                "Determined response priority"
            ],
            alternatives_considered=["immediate_blocking", "continued_monitoring", "threat_hunting"]
        )
    
    async def plan_actions(self, reasoning: ReasoningResult) -> ActionPlan:
        """Plan defensive actions"""
        return ActionPlan(
            primary_action="investigate_alerts",
            action_type="investigation",
            target="security_alerts",
            parameters={
                "alert_types": ["network_scan", "suspicious_traffic"],
                "time_window": "last_hour",
                "correlation_rules": "enabled"
            },
            expected_outcome="Identify and classify threats",
            success_criteria=["threats_identified", "risk_assessed"],
            fallback_actions=["escalate_to_tier2", "implement_blocking"],
            estimated_duration=15.0,
            risk_level="low"
        )
    
    async def execute_action(self, action: ActionPlan) -> ActionResult:
        """Execute defensive action"""
        # Simulate investigation
        await asyncio.sleep(1.5)
        
        return ActionResult(
            action_id=f"soc_{datetime.now().timestamp()}",
            action_type=action.action_type,
            success=True,
            outcome="Identified reconnaissance activity from external source",
            data={
                "threat_type": "reconnaissance",
                "source_ip": "external",
                "confidence": 0.8,
                "recommended_action": "monitor_and_log"
            },
            duration=1.5,
            errors=[],
            side_effects=["alert_generated", "log_entry_created"],
            confidence=0.8,
            timestamp=datetime.now()
        )

async def demo_agent_framework():
    """Demonstrate the agent framework capabilities"""
    logger.info("🚀 Starting Archangel Agent Framework Demo")
    
    try:
        # Initialize message bus
        logger.info("📡 Initializing secure message bus...")
        message_bus = MessageBus(use_encryption=False)  # Disable encryption for demo
        await message_bus.initialize()
        await message_bus.start_message_processing()
        
        # Initialize coordinator
        logger.info("🎯 Initializing LangGraph coordinator...")
        coordinator = LangGraphCoordinator(message_bus)
        await coordinator.initialize()
        
        # Create demo agents
        logger.info("🤖 Creating autonomous agents...")
        
        # Red Team Recon Agent
        recon_config = AgentConfig(
            agent_id="recon_001",
            team=Team.RED,
            role=Role.RECON,
            name="Demo Recon Agent",
            description="Autonomous reconnaissance agent",
            tools=["nmap", "dirb", "nikto"],
            objectives=["discover_targets", "identify_vulnerabilities"]
        )
        recon_agent = DemoReconAgent(recon_config)
        await recon_agent.initialize()
        
        # Blue Team SOC Agent
        soc_config = AgentConfig(
            agent_id="soc_001",
            team=Team.BLUE,
            role=Role.SOC_ANALYST,
            name="Demo SOC Agent",
            description="Autonomous SOC analyst agent",
            tools=["siem", "ids", "firewall"],
            objectives=["detect_threats", "investigate_incidents"]
        )
        soc_agent = DemoSOCAgent(soc_config)
        await soc_agent.initialize()
        
        # Register agents with coordinator
        logger.info("📋 Registering agents with coordinator...")
        await coordinator.register_agent(recon_agent, ["network_scanning", "vulnerability_assessment"])
        await coordinator.register_agent(soc_agent, ["threat_detection", "incident_response"])
        
        # Create demo scenario
        logger.info("📜 Creating demo scenario...")
        scenario = Scenario(
            scenario_id="demo_001",
            name="Basic Red vs Blue Demo",
            description="Demonstration of autonomous agent coordination",
            duration=timedelta(minutes=5),
            phases=[Phase.RECONNAISSANCE, Phase.DEFENSE],
            objectives={
                Team.RED: ["discover_network_topology", "identify_vulnerabilities"],
                Team.BLUE: ["detect_reconnaissance", "assess_threats"]
            },
            constraints=["no_destructive_actions", "stealth_required"],
            success_criteria={
                Team.RED: ["successful_reconnaissance"],
                Team.BLUE: ["threat_detected"]
            },
            environment_config={"network": "192.168.1.0/24", "services": ["web", "ssh"]}
        )
        
        # Demonstrate agent reasoning and coordination
        logger.info("🧠 Demonstrating agent reasoning...")
        
        # Red Team Agent Cycle
        logger.info("🔴 Red Team Agent - Reconnaissance Phase")
        red_state = await recon_agent.perceive_environment()
        red_reasoning = await recon_agent.reason_about_situation(red_state)
        red_plan = await recon_agent.plan_actions(red_reasoning)
        red_result = await recon_agent.execute_action(red_plan)
        await recon_agent.learn_from_outcome(red_plan, red_result)
        
        logger.info(f"   🎯 Action: {red_plan.primary_action}")
        logger.info(f"   📊 Success: {red_result.success}")
        logger.info(f"   📈 Confidence: {red_result.confidence}")
        
        # Blue Team Agent Cycle
        logger.info("🔵 Blue Team Agent - Defense Phase")
        blue_state = await soc_agent.perceive_environment()
        blue_reasoning = await soc_agent.reason_about_situation(blue_state)
        blue_plan = await soc_agent.plan_actions(blue_reasoning)
        blue_result = await soc_agent.execute_action(blue_plan)
        await soc_agent.learn_from_outcome(blue_plan, blue_result)
        
        logger.info(f"   🎯 Action: {blue_plan.primary_action}")
        logger.info(f"   📊 Success: {blue_result.success}")
        logger.info(f"   📈 Confidence: {blue_result.confidence}")
        
        # Demonstrate team communication
        logger.info("💬 Demonstrating team communication...")
        
        # Red team coordination message
        red_message = create_team_message(
            sender_id=recon_agent.agent_id,
            team="red",
            coordination_type=CoordinationType.ATTACK_PLAN,
            content={
                "phase": "reconnaissance_complete",
                "findings": red_result.data,
                "next_phase": "exploitation"
            },
            priority=Priority.HIGH
        )
        
        await message_bus.broadcast_to_team("red", red_message)
        logger.info("   📤 Red team coordination message sent")
        
        # Blue team coordination message
        blue_message = create_team_message(
            sender_id=soc_agent.agent_id,
            team="blue",
            coordination_type=CoordinationType.DEFENSE_STRATEGY,
            content={
                "threat_detected": True,
                "threat_type": "reconnaissance",
                "recommended_response": "enhanced_monitoring"
            },
            priority=Priority.HIGH
        )
        
        await message_bus.broadcast_to_team("blue", blue_message)
        logger.info("   📤 Blue team coordination message sent")
        
        # Show coordination status
        logger.info("📊 Coordination Status:")
        status = coordinator.get_coordination_status()
        logger.info(f"   🎮 Registered Agents: {status['registered_agents']}")
        logger.info(f"   🔴 Red Team: {status['team_assignments']['red']} agents")
        logger.info(f"   🔵 Blue Team: {status['team_assignments']['blue']} agents")
        logger.info(f"   📈 Coordination Events: {status['metrics']['agent_coordination_events']}")
        
        # Show agent performance
        logger.info("🏆 Agent Performance:")
        red_status = await recon_agent.get_status()
        blue_status = await soc_agent.get_status()
        
        logger.info(f"   🔴 {red_status['name']}: {red_status['performance_metrics']['successful_actions']} successful actions")
        logger.info(f"   🔵 {blue_status['name']}: {blue_status['performance_metrics']['successful_actions']} successful actions")
        
        # Demonstrate scenario execution (simplified)
        logger.info("🎬 Starting scenario execution...")
        workflow_id = await coordinator.start_scenario(scenario)
        logger.info(f"   🆔 Workflow ID: {workflow_id}")
        logger.info(f"   📍 Current Phase: {coordinator.current_phase.value}")
        
        # Wait a bit to show the system running
        await asyncio.sleep(3)
        
        logger.info("✅ Demo completed successfully!")
        
        # Show final statistics
        logger.info("📈 Final Statistics:")
        bus_stats = message_bus.get_statistics()
        logger.info(f"   📨 Messages Sent: {bus_stats['messages_sent']}")
        logger.info(f"   📬 Messages Received: {bus_stats['messages_received']}")
        logger.info(f"   🔐 Trusted Agents: {bus_stats['trusted_agents']}")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up...")
        try:
            await coordinator.shutdown()
            await message_bus.shutdown()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

async def main():
    """Main demo entry point"""
    print("🛡️ ARCHANGEL AUTONOMOUS AI EVOLUTION")
    print("=" * 50)
    print("Multi-Agent Coordination Framework Demo")
    print("=" * 50)
    print()
    
    try:
        await demo_agent_framework()
        print()
        print("🎉 Demo completed successfully!")
        print("This demonstrates the foundational framework for:")
        print("  • Autonomous agent reasoning and decision-making")
        print("  • Secure inter-agent communication")
        print("  • Multi-agent coordination and workflow management")
        print("  • Red vs Blue team simulation capabilities")
        print()
        print("Next steps: Implement vector memory, LLM integration, and mock enterprise environment")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))