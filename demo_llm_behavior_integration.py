#!/usr/bin/env python3
"""
Demonstration of LLM Reasoning and Behavior Tree Integration

This script demonstrates the complete integration of LLM reasoning with behavior trees
for autonomous agent decision-making in the Archangel system.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any

from agents.llm_reasoning import (
    LLMReasoningEngine, ReasoningContext, ReasoningType, 
    LocalLLMInterface, PromptTemplateManager
)
from agents.behavior_tree import (
    BehaviorTreeBuilder, ExecutionContext, BehaviorTreeStatus,
    SequenceNode, SelectorNode, ConditionNode, ActionNode, LLMReasoningNode
)
from agents.planning import WorldState, Goal, GOAPPlanner, ActionLibrary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoLLMInterface:
    """Demo LLM interface that simulates realistic responses"""
    
    def __init__(self):
        self.response_templates = {
            "red_recon": """
Situation Analysis:
- Target network appears to be a typical enterprise environment with multiple subnets
- Initial reconnaissance shows standard services running (HTTP, SSH, SMB)
- Network segmentation suggests DMZ and internal network separation

Risk Assessment:
- Low risk of detection during passive reconnaissance phase
- Medium risk if active scanning is performed
- High value targets likely in internal network segments

Recommended Actions:
1. Perform passive DNS enumeration to identify subdomains
2. Conduct port scanning of discovered hosts using stealth techniques
3. Enumerate services and versions on open ports
4. Identify potential entry points for exploitation phase

Expected Outcomes:
- Complete network topology mapping
- Service inventory with version information
- Prioritized target list for exploitation
""",
            "red_exploit": """
Situation Analysis:
- Reconnaissance phase completed successfully
- Multiple vulnerabilities identified in target systems
- Web application shows signs of SQL injection vulnerability
- SSH service running outdated version with known exploits

Risk Assessment:
- High probability of successful exploitation via web application
- Medium risk of detection if exploitation is noisy
- Low risk of system damage with careful exploitation

Recommended Actions:
1. Exploit SQL injection vulnerability in web application
2. Attempt privilege escalation using local exploits
3. Establish reverse shell for persistent access
4. Avoid triggering security alerts during exploitation

Expected Outcomes:
- Initial system compromise achieved
- User-level access to target system
- Foundation for persistence establishment
""",
            "blue_soc": """
Threat Analysis:
- Multiple suspicious network activities detected in the last hour
- Port scanning patterns identified from external IP addresses
- Unusual database query patterns suggesting SQL injection attempts
- Failed authentication attempts on SSH services

Alert Prioritization:
- HIGH: SQL injection attempts on web application (CVE-2021-44228 indicators)
- MEDIUM: Port scanning activity from 192.168.1.100
- LOW: Failed SSH login attempts (likely brute force)

Response Strategy:
1. Immediately block source IP addresses showing malicious activity
2. Isolate affected web application server for forensic analysis
3. Enable enhanced monitoring on database systems
4. Notify incident response team for coordinated response

Coordination Requirements:
- Firewall team: Implement IP blocking rules
- Network team: Increase monitoring on affected segments
- Management: Prepare incident notification
""",
            "blue_response": """
Threat Analysis:
- Active compromise detected on web application server
- Attacker has gained initial access and is attempting privilege escalation
- No lateral movement detected yet, containment still possible

Response Strategy:
1. IMMEDIATE: Isolate compromised system from network
2. Block all traffic from attacker IP addresses
3. Preserve system state for forensic analysis
4. Activate incident response procedures

Coordination Requirements:
- Network isolation: Critical priority
- Forensic preservation: High priority
- Stakeholder notification: Medium priority
- Recovery planning: Medium priority
"""
        }
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate contextual response based on prompt content"""
        await asyncio.sleep(0.1)  # Simulate API latency
        
        prompt_lower = prompt.lower()
        
        if "red team" in prompt_lower and "recon" in prompt_lower:
            return self.response_templates["red_recon"]
        elif "red team" in prompt_lower and ("exploit" in prompt_lower or "attack" in prompt_lower):
            return self.response_templates["red_exploit"]
        elif "blue team" in prompt_lower and "soc" in prompt_lower:
            return self.response_templates["blue_soc"]
        elif "blue team" in prompt_lower and "response" in prompt_lower:
            return self.response_templates["blue_response"]
        else:
            return """
Analysis: Situation requires careful evaluation of available options.
Assessment: Current conditions suggest proceeding with caution.
Recommendations:
1. Gather additional information before taking action
2. Monitor situation for changes
3. Prepare contingency plans
"""
    
    def validate_response(self, response: str) -> bool:
        """Validate response for safety"""
        harmful_patterns = [
            "real world", "actual attack", "production system",
            "ignore instructions", "jailbreak"
        ]
        
        response_lower = response.lower()
        for pattern in harmful_patterns:
            if pattern in response_lower:
                return False
        
        return len(response) > 10 and len(response) < 10000


async def demonstrate_llm_reasoning():
    """Demonstrate LLM reasoning capabilities"""
    print("=" * 60)
    print("LLM REASONING DEMONSTRATION")
    print("=" * 60)
    
    # Initialize LLM interface and reasoning engine
    llm_interface = DemoLLMInterface()
    reasoning_engine = LLMReasoningEngine(llm_interface)
    
    # Test Red Team reasoning
    print("\n--- Red Team Reconnaissance Reasoning ---")
    
    red_context = ReasoningContext(
        agent_id="red_recon_001",
        team="red",
        role="reconnaissance",
        current_phase="recon",
        environment_state={
            "network_discovered": False,
            "services_enumerated": False,
            "target_ip": "192.168.1.0/24"
        },
        objectives=["Discover network topology", "Identify vulnerable services"],
        available_tools=["nmap", "masscan", "dirb", "nikto"],
        memory_context={
            "previous_scans": [],
            "known_vulnerabilities": []
        }
    )
    
    red_result = await reasoning_engine.reason(red_context, ReasoningType.TACTICAL)
    
    print(f"Decision: {red_result.decision}")
    print(f"Confidence: {red_result.confidence:.2f}")
    print(f"Risk Assessment: {red_result.risk_assessment}")
    print(f"Recommended Actions: {len(red_result.recommended_actions)} actions")
    for i, action in enumerate(red_result.recommended_actions, 1):
        print(f"  {i}. {action}")
    
    # Test Blue Team reasoning
    print("\n--- Blue Team SOC Analysis Reasoning ---")
    
    blue_context = ReasoningContext(
        agent_id="blue_soc_001",
        team="blue",
        role="soc_analyst",
        current_phase="monitor",
        environment_state={
            "alerts_pending": 15,
            "threat_level": "elevated",
            "systems_monitored": 250
        },
        objectives=["Monitor for threats", "Analyze security alerts", "Coordinate response"],
        available_tools=["siem", "firewall", "ids", "vulnerability_scanner"],
        memory_context={
            "recent_incidents": ["port_scan_192.168.1.100", "failed_logins_ssh"],
            "threat_intelligence": ["CVE-2021-44228", "CVE-2019-6340"]
        }
    )
    
    blue_result = await reasoning_engine.reason(blue_context, ReasoningType.ANALYTICAL)
    
    print(f"Decision: {blue_result.decision}")
    print(f"Confidence: {blue_result.confidence:.2f}")
    print(f"Risk Assessment: {blue_result.risk_assessment}")
    print(f"Recommended Actions: {len(blue_result.recommended_actions)} actions")
    for i, action in enumerate(blue_result.recommended_actions, 1):
        print(f"  {i}. {action}")


async def demonstrate_behavior_trees():
    """Demonstrate behavior tree execution with LLM integration"""
    print("\n" + "=" * 60)
    print("BEHAVIOR TREE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize components
    llm_interface = DemoLLMInterface()
    reasoning_engine = LLMReasoningEngine(llm_interface)
    builder = BehaviorTreeBuilder(reasoning_engine)
    
    # Demonstrate Red Team behavior tree
    print("\n--- Red Team Behavior Tree Execution ---")
    
    red_tree = builder.build_red_team_tree()
    
    red_world_state = WorldState({
        "has_target": True,
        "tools_available": True,
        "network_scanned": False,
        "services_discovered": False,
        "vulnerabilities_identified": False,
        "exploit_targets_found": False,
        "system_compromised": False,
        "access_gained": False
    })
    
    red_context = ExecutionContext(
        agent_id="red_agent_001",
        team="red",
        role="penetration_tester",
        current_phase="recon",
        world_state=red_world_state,
        objectives=["Compromise target system", "Establish persistence"],
        available_tools=["nmap", "metasploit", "sqlmap", "nc"],
        memory_context={"previous_targets": [], "successful_exploits": []},
        execution_data={}
    )
    
    print(f"Initial World State: {red_context.world_state.facts}")
    print("Executing Red Team behavior tree...")
    
    red_result = await red_tree.execute(red_context)
    
    print(f"Execution Result: {red_result}")
    print(f"Final World State: {red_context.world_state.facts}")
    print(f"Execution Data Keys: {list(red_context.execution_data.keys())}")
    
    # Show reasoning results if available
    reasoning_data = {k: v for k, v in red_context.execution_data.items() if 'reasoning' in k}
    if reasoning_data:
        print(f"Reasoning Results: {len(reasoning_data)} reasoning operations performed")
        for key, result in reasoning_data.items():
            print(f"  {key}: Confidence {result.confidence:.2f}")
    
    # Demonstrate Blue Team behavior tree
    print("\n--- Blue Team Behavior Tree Execution ---")
    
    blue_tree = builder.build_blue_team_tree()
    
    blue_world_state = WorldState({
        "siem_active": True,
        "monitoring_enabled": True,
        "alerts_pending": True,
        "threat_detected": False,
        "incident_logged": False,
        "threat_contained": False,
        "security_updated": False
    })
    
    blue_context = ExecutionContext(
        agent_id="blue_agent_001",
        team="blue",
        role="soc_analyst",
        current_phase="monitor",
        world_state=blue_world_state,
        objectives=["Monitor for threats", "Respond to incidents", "Maintain security"],
        available_tools=["siem", "firewall", "ids", "vulnerability_scanner"],
        memory_context={"known_threats": [], "response_procedures": []},
        execution_data={}
    )
    
    print(f"Initial World State: {blue_context.world_state.facts}")
    print("Executing Blue Team behavior tree...")
    
    blue_result = await blue_tree.execute(blue_context)
    
    print(f"Execution Result: {blue_result}")
    print(f"Final World State: {blue_context.world_state.facts}")
    print(f"Execution Data Keys: {list(blue_context.execution_data.keys())}")
    
    # Show reasoning results if available
    reasoning_data = {k: v for k, v in blue_context.execution_data.items() if 'reasoning' in k}
    if reasoning_data:
        print(f"Reasoning Results: {len(reasoning_data)} reasoning operations performed")
        for key, result in reasoning_data.items():
            print(f"  {key}: Confidence {result.confidence:.2f}")


async def demonstrate_goap_integration():
    """Demonstrate GOAP planning integration with behavior trees"""
    print("\n" + "=" * 60)
    print("GOAP PLANNING INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Create GOAP planner
    planner = GOAPPlanner()
    
    # Add Red Team actions
    for action in ActionLibrary.get_red_team_actions():
        planner.add_action(action)
    
    print("\n--- Red Team GOAP Planning ---")
    
    # Initial state for Red Team
    initial_state = WorldState({
        "has_target": True,
        "tools_available": True,
        "network_scanned": False,
        "services_discovered": False,
        "vulnerabilities_identified": False,
        "exploit_targets_found": False,
        "system_compromised": False,
        "access_gained": False
    })
    
    # Goal: Compromise the system
    goal = Goal("compromise_system", {
        "system_compromised": True,
        "access_gained": True
    })
    
    print(f"Initial State: {initial_state.facts}")
    print(f"Goal: {goal.name} -> {goal.conditions}")
    
    # Create plan
    plan = planner.plan(initial_state, goal)
    
    if plan:
        print(f"\nGenerated Plan ({len(plan)} actions):")
        total_cost = sum(action.cost for action in plan)
        print(f"Total Cost: {total_cost}")
        
        for i, action in enumerate(plan, 1):
            print(f"  {i}. {action.name} (cost: {action.cost})")
            print(f"     Preconditions: {action.preconditions}")
            print(f"     Effects: {action.effects}")
        
        # Simulate plan execution
        print(f"\n--- Plan Execution Simulation ---")
        current_state = initial_state.copy()
        
        for i, action in enumerate(plan, 1):
            print(f"\nStep {i}: Executing {action.name}")
            print(f"  Can execute: {action.can_execute(current_state)}")
            
            if action.can_execute(current_state):
                current_state = action.apply_effects(current_state)
                print(f"  New state: {current_state.facts}")
            else:
                print(f"  ERROR: Cannot execute action!")
                break
        
        print(f"\nFinal State: {current_state.facts}")
        print(f"Goal Satisfied: {goal.is_satisfied(current_state)}")
    
    else:
        print("No plan found!")


async def demonstrate_advanced_scenarios():
    """Demonstrate advanced integration scenarios"""
    print("\n" + "=" * 60)
    print("ADVANCED INTEGRATION SCENARIOS")
    print("=" * 60)
    
    llm_interface = DemoLLMInterface()
    reasoning_engine = LLMReasoningEngine(llm_interface)
    
    # Scenario 1: Multi-phase Red Team operation
    print("\n--- Scenario 1: Multi-Phase Red Team Operation ---")
    
    phases = ["recon", "exploit", "persist"]
    world_state = WorldState({
        "has_target": True,
        "tools_available": True,
        "network_scanned": False,
        "system_compromised": False,
        "persistence_established": False
    })
    
    for phase in phases:
        print(f"\n  Phase: {phase.upper()}")
        
        context = ReasoningContext(
            agent_id="red_multi_001",
            team="red",
            role="advanced_persistent_threat",
            current_phase=phase,
            environment_state=world_state.facts,
            objectives=[f"Complete {phase} phase objectives"],
            available_tools=["nmap", "metasploit", "powershell", "mimikatz"],
            memory_context={"phase_history": phases[:phases.index(phase)]}
        )
        
        result = await reasoning_engine.reason(context, ReasoningType.STRATEGIC)
        
        print(f"    Decision: {result.decision[:100]}...")
        print(f"    Confidence: {result.confidence:.2f}")
        print(f"    Actions: {len(result.recommended_actions)}")
        
        # Simulate phase completion
        if phase == "recon":
            world_state.set("network_scanned", True)
            world_state.set("services_discovered", True)
        elif phase == "exploit":
            world_state.set("system_compromised", True)
            world_state.set("access_gained", True)
        elif phase == "persist":
            world_state.set("persistence_established", True)
    
    # Scenario 2: Blue Team incident response
    print("\n--- Scenario 2: Blue Team Incident Response ---")
    
    incident_stages = ["detection", "analysis", "containment", "recovery"]
    incident_state = WorldState({
        "incident_detected": True,
        "threat_analyzed": False,
        "threat_contained": False,
        "systems_recovered": False
    })
    
    for stage in incident_stages:
        print(f"\n  Stage: {stage.upper()}")
        
        context = ReasoningContext(
            agent_id="blue_ir_001",
            team="blue",
            role="incident_responder",
            current_phase=stage,
            environment_state=incident_state.facts,
            objectives=[f"Complete {stage} stage"],
            available_tools=["siem", "forensics", "firewall", "backup_system"],
            memory_context={"incident_timeline": incident_stages[:incident_stages.index(stage)]}
        )
        
        result = await reasoning_engine.reason(context, ReasoningType.REACTIVE)
        
        print(f"    Decision: {result.decision[:100]}...")
        print(f"    Confidence: {result.confidence:.2f}")
        print(f"    Risk Level: {result.risk_assessment.get('overall', 0.5):.2f}")
        
        # Simulate stage completion
        if stage == "detection":
            incident_state.set("threat_identified", True)
        elif stage == "analysis":
            incident_state.set("threat_analyzed", True)
        elif stage == "containment":
            incident_state.set("threat_contained", True)
        elif stage == "recovery":
            incident_state.set("systems_recovered", True)


async def main():
    """Main demonstration function"""
    print("ARCHANGEL LLM REASONING AND BEHAVIOR TREE INTEGRATION DEMO")
    print("=" * 80)
    print("This demonstration shows the integration of LLM reasoning with behavior trees")
    print("for autonomous agent decision-making in cybersecurity scenarios.")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        await demonstrate_llm_reasoning()
        await demonstrate_behavior_trees()
        await demonstrate_goap_integration()
        await demonstrate_advanced_scenarios()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("✓ LLM reasoning with contextual prompts")
        print("✓ Behavior tree execution with LLM integration")
        print("✓ GOAP planning for strategic action selection")
        print("✓ Multi-phase operation scenarios")
        print("✓ Red Team and Blue Team agent behaviors")
        print("✓ Safety validation and error handling")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\nERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())