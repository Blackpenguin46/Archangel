#!/usr/bin/env python3
"""
Demo of LLM reasoning and behavior tree integration
Shows how the integrated system works for autonomous agent decision-making
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from agents.llm_interface import LLMConfig, LLMProvider
from agents.reasoning_integration import IntegratedReasoningSystem, ReasoningContext
from agents.planning import WorldState
from agents.base_agent import BaseAgent, AgentConfig, Team, Role, EnvironmentState

class DemoAgent(BaseAgent):
    """Demo agent that uses the integrated reasoning system"""
    
    async def execute_action(self, action):
        """Execute an action (simulated)"""
        print(f"🎯 Executing action: {action.primary_action}")
        print(f"   Type: {action.action_type}")
        print(f"   Target: {action.target}")
        print(f"   Risk Level: {action.risk_level}")
        
        # Simulate action execution
        await asyncio.sleep(0.5)
        
        # Update world state based on action
        if hasattr(self, 'world_state'):
            if action.primary_action.lower() == "network scan":
                self.world_state.set_fact("network_mapped", True)
                self.world_state.set_fact("services_discovered", True)
            elif "exploit" in action.primary_action.lower():
                self.world_state.set_fact("system_compromised", True)
            elif "monitor" in action.primary_action.lower():
                self.world_state.set_fact("network_monitored", True)
                self.world_state.set_fact("alerts_generated", True)
        
        return {
            "action_id": f"action_{datetime.now().timestamp()}",
            "action_type": action.action_type,
            "success": True,
            "outcome": f"Successfully executed {action.primary_action}",
            "data": {"simulated": True},
            "duration": 0.5,
            "errors": [],
            "side_effects": [],
            "confidence": 0.9,
            "timestamp": datetime.now()
        }

async def demo_red_team_agent():
    """Demonstrate Red Team agent reasoning"""
    print("\n🔴 RED TEAM AGENT DEMO")
    print("=" * 50)
    
    # Create Red Team agent
    config = AgentConfig(
        agent_id="red_agent_001",
        team=Team.RED,
        role=Role.RECON,
        name="Red Team Recon Agent",
        description="Autonomous reconnaissance agent"
    )
    
    agent = DemoAgent(config)
    
    # Mock the reasoning system to avoid actual LLM calls
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4-turbo",
        api_key="demo-key"
    )
    
    agent.reasoning_system = IntegratedReasoningSystem(llm_config)
    agent.world_state = WorldState(facts={
        "network_access": True,
        "network_mapped": False,
        "services_discovered": False,
        "vulnerabilities_identified": False,
        "system_compromised": False
    })
    
    # Mock LLM responses for realistic Red Team behavior
    agent.reasoning_system.llm_engine.reason_about_situation = AsyncMock(return_value={
        "situation_assessment": "Network is accessible with no immediate defensive measures detected. Environment appears suitable for reconnaissance activities.",
        "threat_analysis": "Low threat environment with minimal monitoring. No active defensive agents detected in the immediate vicinity.",
        "opportunity_identification": [
            "Network scanning to map topology",
            "Service enumeration on discovered hosts",
            "Vulnerability assessment of exposed services"
        ],
        "risk_assessment": {
            "detection": 0.2,
            "success": 0.85,
            "impact": 0.3
        },
        "recommended_actions": [
            "network_scan",
            "service_enumeration",
            "vulnerability_assessment"
        ],
        "confidence_score": 0.85,
        "reasoning_chain": [
            "Analyzed network accessibility",
            "Assessed defensive posture",
            "Identified reconnaissance opportunities",
            "Evaluated risk vs reward"
        ],
        "alternatives_considered": [
            "Direct exploitation attempt",
            "Social engineering approach",
            "Physical access attempt"
        ]
    })
    
    # Initialize agent
    await agent.initialize()
    
    # Create environment state
    env_state = EnvironmentState(
        timestamp=datetime.now(),
        network_topology={
            "subnets": ["192.168.1.0/24", "10.0.0.0/8"],
            "gateways": ["192.168.1.1", "10.0.0.1"]
        },
        active_services=[
            {"ip": "192.168.1.10", "port": 80, "service": "http", "version": "Apache 2.4.41"},
            {"ip": "192.168.1.20", "port": 22, "service": "ssh", "version": "OpenSSH 7.4"},
            {"ip": "10.0.0.50", "port": 3306, "service": "mysql", "version": "MySQL 5.7.30"}
        ],
        security_alerts=[],
        system_logs=[],
        agent_positions={"red_agent_001": {"status": "active", "last_seen": datetime.now()}},
        threat_level="low"
    )
    
    print(f"🤖 Agent: {agent.name}")
    print(f"📍 Team: {agent.team.value.upper()}")
    print(f"🎭 Role: {agent.role.value}")
    print(f"🌐 Network Access: {agent.world_state.has_fact('network_access', True)}")
    
    # Perform reasoning
    print("\n🧠 REASONING PHASE")
    reasoning_result = await agent.reason_about_situation(env_state)
    
    print(f"📊 Situation: {reasoning_result.situation_assessment}")
    print(f"⚠️  Threat Analysis: {reasoning_result.threat_analysis}")
    print(f"🎯 Opportunities: {', '.join(reasoning_result.opportunity_identification[:2])}")
    print(f"📈 Confidence: {reasoning_result.confidence_score:.2f}")
    
    # Plan actions
    print("\n📋 PLANNING PHASE")
    action_plan = await agent.plan_actions(reasoning_result)
    
    print(f"🎯 Primary Action: {action_plan.primary_action}")
    print(f"🏷️  Action Type: {action_plan.action_type}")
    print(f"⏱️  Estimated Duration: {action_plan.estimated_duration}s")
    print(f"⚠️  Risk Level: {action_plan.risk_level}")
    
    # Execute action
    print("\n⚡ EXECUTION PHASE")
    action_result = await agent.execute_action(action_plan)
    
    print(f"✅ Success: {action_result['success']}")
    print(f"📝 Outcome: {action_result['outcome']}")
    print(f"⏱️  Duration: {action_result['duration']}s")
    
    # Learn from outcome
    await agent.learn_from_outcome(action_plan, action_result)
    
    print(f"🧠 Experience stored. Total experiences: {len(agent.experiences)}")
    print(f"📊 Success rate: {agent.performance_metrics['successful_actions']}/{agent.performance_metrics['actions_taken']}")

async def demo_blue_team_agent():
    """Demonstrate Blue Team agent reasoning"""
    print("\n🔵 BLUE TEAM AGENT DEMO")
    print("=" * 50)
    
    # Create Blue Team agent
    config = AgentConfig(
        agent_id="blue_agent_001",
        team=Team.BLUE,
        role=Role.SOC_ANALYST,
        name="Blue Team SOC Analyst",
        description="Autonomous security monitoring agent"
    )
    
    agent = DemoAgent(config)
    
    # Mock the reasoning system
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4-turbo",
        api_key="demo-key"
    )
    
    agent.reasoning_system = IntegratedReasoningSystem(llm_config)
    agent.world_state = WorldState(facts={
        "monitoring_enabled": True,
        "network_monitored": False,
        "alerts_generated": False,
        "threats_identified": False,
        "incident_contained": False
    })
    
    # Mock LLM responses for realistic Blue Team behavior
    agent.reasoning_system.llm_engine.reason_about_situation = AsyncMock(return_value={
        "situation_assessment": "Suspicious network activity detected. Multiple reconnaissance attempts observed from unknown source. Immediate investigation required.",
        "threat_analysis": "Potential advanced persistent threat (APT) conducting systematic network reconnaissance. Attack patterns consistent with initial compromise phase.",
        "opportunity_identification": [
            "Deploy additional network monitoring",
            "Correlate logs for attack patterns",
            "Implement traffic blocking rules",
            "Initiate incident response procedures"
        ],
        "risk_assessment": {
            "compromise_risk": 0.7,
            "data_loss_risk": 0.4,
            "service_disruption": 0.3
        },
        "recommended_actions": [
            "enhanced_monitoring",
            "log_correlation",
            "traffic_analysis",
            "incident_response"
        ],
        "confidence_score": 0.78,
        "reasoning_chain": [
            "Detected anomalous network traffic",
            "Correlated with known attack patterns",
            "Assessed threat severity and impact",
            "Prioritized defensive responses"
        ],
        "alternatives_considered": [
            "Immediate network isolation",
            "Passive monitoring continuation",
            "Proactive threat hunting"
        ]
    })
    
    # Initialize agent
    await agent.initialize()
    
    # Create environment state with security alerts
    env_state = EnvironmentState(
        timestamp=datetime.now(),
        network_topology={
            "subnets": ["192.168.1.0/24", "10.0.0.0/8"],
            "security_zones": ["DMZ", "Internal", "Management"]
        },
        active_services=[
            {"ip": "192.168.1.100", "port": 443, "service": "https", "status": "monitoring"},
            {"ip": "192.168.1.101", "port": 514, "service": "syslog", "status": "active"}
        ],
        security_alerts=[
            {
                "id": "alert_001",
                "severity": "medium",
                "type": "network_scan",
                "source": "192.168.1.50",
                "description": "Port scan detected from internal host",
                "timestamp": datetime.now()
            },
            {
                "id": "alert_002", 
                "severity": "high",
                "type": "suspicious_traffic",
                "source": "unknown",
                "description": "Unusual outbound connections detected",
                "timestamp": datetime.now()
            }
        ],
        system_logs=[
            {"level": "warning", "message": "Multiple failed login attempts"},
            {"level": "info", "message": "Network scan activity detected"}
        ],
        agent_positions={"blue_agent_001": {"status": "monitoring", "last_seen": datetime.now()}},
        threat_level="medium"
    )
    
    print(f"🤖 Agent: {agent.name}")
    print(f"📍 Team: {agent.team.value.upper()}")
    print(f"🎭 Role: {agent.role.value}")
    print(f"🚨 Active Alerts: {len(env_state.security_alerts)}")
    print(f"📊 Threat Level: {env_state.threat_level.upper()}")
    
    # Perform reasoning
    print("\n🧠 REASONING PHASE")
    reasoning_result = await agent.reason_about_situation(env_state)
    
    print(f"📊 Situation: {reasoning_result.situation_assessment}")
    print(f"⚠️  Threat Analysis: {reasoning_result.threat_analysis}")
    print(f"🛡️  Defensive Opportunities: {', '.join(reasoning_result.opportunity_identification[:2])}")
    print(f"📈 Confidence: {reasoning_result.confidence_score:.2f}")
    
    # Plan actions
    print("\n📋 PLANNING PHASE")
    action_plan = await agent.plan_actions(reasoning_result)
    
    print(f"🎯 Primary Action: {action_plan.primary_action}")
    print(f"🏷️  Action Type: {action_plan.action_type}")
    print(f"⏱️  Estimated Duration: {action_plan.estimated_duration}s")
    print(f"⚠️  Risk Level: {action_plan.risk_level}")
    
    # Execute action
    print("\n⚡ EXECUTION PHASE")
    action_result = await agent.execute_action(action_plan)
    
    print(f"✅ Success: {action_result['success']}")
    print(f"📝 Outcome: {action_result['outcome']}")
    print(f"⏱️  Duration: {action_result['duration']}s")
    
    # Learn from outcome
    await agent.learn_from_outcome(action_plan, action_result)
    
    print(f"🧠 Experience stored. Total experiences: {len(agent.experiences)}")
    print(f"📊 Success rate: {agent.performance_metrics['successful_actions']}/{agent.performance_metrics['actions_taken']}")

async def demo_behavior_tree_execution():
    """Demonstrate behavior tree execution"""
    print("\n🌳 BEHAVIOR TREE EXECUTION DEMO")
    print("=" * 50)
    
    from agents.behavior_tree import BehaviorTreeBuilder, ExecutionContext
    from agents.reasoning_integration import LLMBehaviorTreeIntegration
    
    # Create mock LLM engine
    mock_llm_engine = Mock()
    mock_llm_engine.reason_about_situation = AsyncMock(return_value={
        "situation_assessment": "Target network identified with multiple entry points",
        "recommended_actions": ["reconnaissance", "vulnerability_scan"],
        "confidence_score": 0.9
    })
    
    # Create behavior tree integration
    integration = LLMBehaviorTreeIntegration(mock_llm_engine)
    
    # Create Red Team behavior tree
    red_tree = integration.create_reasoning_tree("red", "recon")
    
    print(f"🌳 Created behavior tree: {red_tree.name}")
    print(f"📊 Tree structure: {json.dumps(red_tree.get_tree_structure(), indent=2)}")
    
    # Create execution context
    context = ExecutionContext(
        agent_id="demo_agent",
        environment_state={
            "network_topology": {"subnets": ["192.168.1.0/24"]},
            "active_services": [{"ip": "192.168.1.10", "port": 80}],
            "security_alerts": [],
            "threat_level": "low"
        },
        agent_memory={},
        available_tools=["nmap", "nessus"],
        constraints=["simulation_only"],
        objectives=["map_network"]
    )
    
    # Set up blackboard
    context.blackboard["team"] = "red"
    context.blackboard["role"] = "recon"
    context.blackboard["world_state"] = {
        "network_access": True,
        "network_mapped": False,
        "services_discovered": False
    }
    
    print("\n⚡ Executing behavior tree...")
    result = await red_tree.execute(context)
    
    print(f"✅ Execution result: {result.status.value}")
    print(f"📝 Message: {result.message}")
    print(f"⏱️  Execution time: {result.execution_time:.2f}s")
    print(f"🔄 Children executed: {len(result.children_results)}")
    
    # Show updated world state
    updated_state = context.blackboard.get("world_state", {})
    print(f"🌍 Updated world state: {json.dumps(updated_state, indent=2)}")

async def demo_planning_system():
    """Demonstrate planning system"""
    print("\n📋 PLANNING SYSTEM DEMO")
    print("=" * 50)
    
    from agents.planning import PlanningEngine, WorldState, Goal
    
    # Create planning engine
    engine = PlanningEngine()
    
    print("🏭 Planning engine initialized")
    print(f"📚 Available actions: {len(engine.action_library.actions)}")
    
    # Create initial world state for Red Team
    initial_state = engine.get_world_state_template("red")
    print(f"🌍 Initial world state: {json.dumps(initial_state.facts, indent=2)}")
    
    # Create goal
    goal = engine.create_goal_template("red", "compromise")
    print(f"🎯 Goal: {goal.name}")
    print(f"📋 Goal conditions: {json.dumps(goal.conditions, indent=2)}")
    
    # Create plan
    print("\n🔄 Creating plan...")
    plan = await engine.create_plan(initial_state, goal)
    
    if plan:
        print(f"✅ Plan created successfully!")
        print(f"📊 Plan ID: {plan.plan_id}")
        print(f"🎯 Goal: {plan.goal.name}")
        print(f"📝 Actions: {len(plan.actions)}")
        print(f"💰 Total cost: {plan.total_cost}")
        print(f"⏱️  Estimated duration: {plan.estimated_duration}s")
        
        print("\n📋 Action sequence:")
        for i, action in enumerate(plan.actions, 1):
            print(f"  {i}. {action.name} ({action.action_type.value})")
            print(f"     Cost: {action.cost}, Duration: {action.duration}s")
        
        # Execute plan
        print("\n⚡ Executing plan...")
        result = await engine.execute_plan(plan, initial_state)
        
        print(f"✅ Execution success: {result['success']}")
        print(f"🎯 Goal satisfied: {result['goal_satisfied']}")
        print(f"📊 Actions executed: {len(result['executed_actions'])}")
        print(f"⏱️  Total execution time: {result['execution_time']:.2f}s")
        
    else:
        print("❌ Plan creation failed")

async def main():
    """Run all demos"""
    print("🚀 LLM REASONING & BEHAVIOR TREE INTEGRATION DEMO")
    print("=" * 60)
    print("This demo shows how autonomous agents use LLM reasoning,")
    print("behavior trees, and planning systems for decision-making.")
    print("=" * 60)
    
    try:
        await demo_red_team_agent()
        await demo_blue_team_agent()
        await demo_behavior_tree_execution()
        await demo_planning_system()
        
        print("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The integrated reasoning system demonstrates:")
        print("✓ LLM-powered situation analysis")
        print("✓ Structured decision-making with behavior trees")
        print("✓ Strategic planning with GOAP")
        print("✓ Safety validation and constraint enforcement")
        print("✓ Learning from action outcomes")
        print("✓ Team-specific reasoning patterns")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())