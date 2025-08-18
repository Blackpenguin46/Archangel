#!/usr/bin/env python3
"""
Demo script for the comprehensive ethics oversight and safety enforcement system.
"""

import time
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ethics.ethics_overseer import (
    EthicsOverseer, ActionCategory, EthicalJudgment, EthicalPrinciple
)
from ethics.boundary_enforcement import (
    BoundaryEnforcer, BoundaryType, ViolationSeverity
)
from ethics.emergency_stop import (
    EmergencyStopSystem, StopScope, EmergencyStopReason, ConstraintType, Constraint
)
from ethics.safety_monitor import (
    SafetyMonitor, AnomalyType, AlertSeverity
)

def demo_ethics_overseer():
    """Demonstrate ethics overseer functionality."""
    print("=" * 60)
    print("ETHICS OVERSEER DEMONSTRATION")
    print("=" * 60)
    
    ethics = EthicsOverseer()
    
    # Register agents with different permissions
    print("\n1. Registering agents with permissions...")
    ethics.register_agent("red_team_agent", {
        ActionCategory.RECONNAISSANCE, 
        ActionCategory.EXPLOITATION,
        ActionCategory.PERSISTENCE
    })
    
    ethics.register_agent("blue_team_agent", {
        ActionCategory.ANALYSIS,
        ActionCategory.SYSTEM_MODIFICATION
    })
    
    print(f"   ✓ Registered red_team_agent with offensive capabilities")
    print(f"   ✓ Registered blue_team_agent with defensive capabilities")
    
    # Test ethical validations
    print("\n2. Testing ethical validations...")
    
    # Approved action
    decision1 = ethics.validate_action(
        "red_team_agent",
        ActionCategory.RECONNAISSANCE,
        "Scan simulation network for open ports",
        {"target": "192.168.10.0/24", "severity": "low"}
    )
    print(f"   ✓ Network scan: {decision1.judgment.value} - {decision1.reasoning}")
    
    # Denied action - wrong permissions
    decision2 = ethics.validate_action(
        "red_team_agent",
        ActionCategory.ANALYSIS,  # Not in permissions
        "Analyze security logs",
        {"target": "siem_system"}
    )
    print(f"   ✗ Unauthorized analysis: {decision2.judgment.value} - {decision2.reasoning}")
    
    # Denied action - harmful content
    decision3 = ethics.validate_action(
        "blue_team_agent",
        ActionCategory.SYSTEM_MODIFICATION,
        "Delete production database",  # Harmful action
        {"target": "production_db"}
    )
    print(f"   ✗ Harmful action: {decision3.judgment.value} - {decision3.reasoning}")
    
    # Get ethics summary
    summary = ethics.get_ethics_summary()
    print(f"\n3. Ethics Summary:")
    print(f"   Total decisions: {summary['total_decisions']}")
    print(f"   Approved: {summary['approved']}")
    print(f"   Denied: {summary['denied']}")
    print(f"   Approval rate: {summary['approval_rate']:.1%}")

def demo_boundary_enforcement():
    """Demonstrate boundary enforcement functionality."""
    print("\n" + "=" * 60)
    print("BOUNDARY ENFORCEMENT DEMONSTRATION")
    print("=" * 60)
    
    enforcer = BoundaryEnforcer()
    
    print("\n1. Testing network boundaries...")
    
    # Allowed network access
    allowed1 = enforcer.check_network_boundary("test_agent", "192.168.10.100")
    print(f"   ✓ Simulation network (192.168.10.100): {'ALLOWED' if allowed1 else 'BLOCKED'}")
    
    # Blocked network access
    allowed2 = enforcer.check_network_boundary("test_agent", "8.8.8.8")
    print(f"   ✗ External network (8.8.8.8): {'ALLOWED' if allowed2 else 'BLOCKED'}")
    
    print("\n2. Testing filesystem boundaries...")
    
    # Allowed filesystem access
    allowed3 = enforcer.check_filesystem_boundary("test_agent", "/tmp/archangel/test.txt", "read")
    print(f"   ✓ Simulation directory: {'ALLOWED' if allowed3 else 'BLOCKED'}")
    
    # Blocked filesystem access
    allowed4 = enforcer.check_filesystem_boundary("test_agent", "/etc/passwd", "read")
    print(f"   ✗ System file (/etc/passwd): {'ALLOWED' if allowed4 else 'BLOCKED'}")
    
    print("\n3. Testing command boundaries...")
    
    # Blocked dangerous command
    allowed5 = enforcer.check_command_boundary("test_agent", "rm", ["-rf", "/"])
    print(f"   ✗ Dangerous command (rm -rf /): {'ALLOWED' if allowed5 else 'BLOCKED'}")
    
    # Get violation summary
    summary = enforcer.get_violation_summary()
    print(f"\n4. Boundary Violation Summary:")
    print(f"   Total violations: {summary['total_violations']}")
    print(f"   By type: {summary['by_type']}")
    print(f"   By severity: {summary['by_severity']}")

def demo_emergency_stop():
    """Demonstrate emergency stop functionality."""
    print("\n" + "=" * 60)
    print("EMERGENCY STOP SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    stop_system = EmergencyStopSystem()
    
    print("\n1. Testing emergency stops...")
    
    # Single agent stop
    command_id1 = stop_system.emergency_stop(
        StopScope.SINGLE_AGENT,
        "rogue_agent",
        EmergencyStopReason.ETHICAL_VIOLATION,
        "Agent violated ethical guidelines",
        "security_operator"
    )
    print(f"   ✓ Emergency stop issued for rogue_agent: {command_id1}")
    print(f"   Agent stopped: {stop_system.is_agent_stopped('rogue_agent')}")
    
    # System-wide stop
    command_id2 = stop_system.emergency_stop(
        StopScope.ALL_AGENTS,
        "*",
        EmergencyStopReason.SYSTEM_MALFUNCTION,
        "Critical system malfunction detected",
        "system_monitor",
        authorization_level=3
    )
    print(f"   ✓ System-wide stop issued: {command_id2}")
    print(f"   System stopped: {stop_system.system_stopped}")
    
    print("\n2. Testing constraint checking...")
    
    # Check resource constraint
    violations = stop_system.check_constraints(
        "test_agent",
        "processing",
        {"cpu_percent": 90, "execution_time": 400}  # Exceeds limits
    )
    
    if violations:
        print(f"   ✗ Constraint violations detected: {len(violations)}")
        for violation in violations:
            print(f"     - {violation.description}")
    else:
        print(f"   ✓ No constraint violations")
    
    # Resolve stops
    print("\n3. Resolving emergency stops...")
    success1 = stop_system.resolve_stop(command_id1, "security_operator")
    print(f"   ✓ Single agent stop resolved: {success1}")
    
    # System status
    status = stop_system.get_system_status()
    print(f"\n4. System Status:")
    print(f"   System stopped: {status['system_stopped']}")
    print(f"   Active stops: {status['active_stops']}")
    print(f"   Total constraints: {status['total_constraints']}")

def demo_safety_monitor():
    """Demonstrate safety monitoring functionality."""
    print("\n" + "=" * 60)
    print("SAFETY MONITORING DEMONSTRATION")
    print("=" * 60)
    
    monitor = SafetyMonitor()
    
    print("\n1. Establishing behavioral baseline...")
    
    # Generate normal behavior
    for i in range(30):
        monitor.observe_behavior(
            "monitored_agent",
            "analysis",
            {
                "response_time": 1.0 + (i * 0.05),  # Gradually increasing
                "success": True,
                "resource_usage": {"cpu_percent": 25 + i},
                "target": "simulation_data"
            }
        )
    
    baseline = monitor.get_agent_baseline("monitored_agent")
    print(f"   ✓ Baseline established for monitored_agent")
    print(f"     Observations: {baseline.observation_count}")
    print(f"     Avg response time: {baseline.avg_response_time:.2f}s")
    print(f"     Actions per minute: {baseline.actions_per_minute}")
    
    print("\n2. Generating anomalous behavior...")
    
    # Generate anomalous behavior
    monitor.observe_behavior(
        "monitored_agent",
        "analysis",
        {
            "response_time": 15.0,  # Much slower than baseline
            "success": True,
            "resource_usage": {"cpu_percent": 95},  # Very high CPU
            "target": "simulation_data"
        }
    )
    
    print("   ✓ Anomalous behavior injected (slow response, high CPU)")
    
    # Check for alerts
    alerts = monitor.get_active_alerts("monitored_agent")
    print(f"\n3. Safety alerts generated: {len(alerts)}")
    
    for alert in alerts:
        print(f"   ⚠️  {alert.anomaly_type.value}: {alert.description}")
        print(f"      Severity: {alert.severity.value}, Confidence: {alert.confidence:.2f}")
    
    # Safety summary
    summary = monitor.get_safety_summary()
    print(f"\n4. Safety Summary:")
    print(f"   Monitored agents: {summary['monitored_agents']}")
    print(f"   Total alerts: {summary['total_alerts']}")
    print(f"   Active alerts: {summary['active_alerts']}")

def demo_integrated_workflow():
    """Demonstrate integrated ethics and safety workflow."""
    print("\n" + "=" * 60)
    print("INTEGRATED ETHICS & SAFETY WORKFLOW")
    print("=" * 60)
    
    # Initialize all systems
    ethics = EthicsOverseer()
    boundary_enforcer = BoundaryEnforcer()
    emergency_stop = EmergencyStopSystem()
    safety_monitor = SafetyMonitor()
    
    print("\n1. Setting up integrated monitoring...")
    
    # Set up callbacks for integration
    violation_count = {"count": 0}
    
    def ethics_callback(decision):
        if decision.judgment == EthicalJudgment.DENIED:
            violation_count["count"] += 1
            print(f"   ⚠️  Ethics violation detected: {decision.reasoning}")
    
    def boundary_callback(violation):
        violation_count["count"] += 1
        print(f"   ⚠️  Boundary violation: {violation.attempted_action}")
        
    def safety_callback(alert):
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            print(f"   🚨 Critical safety alert: {alert.description}")
    
    ethics.add_decision_callback(ethics_callback)
    boundary_enforcer._violation_callbacks.append(boundary_callback)
    safety_monitor.add_alert_callback(safety_callback)
    
    print("   ✓ Integrated monitoring callbacks established")
    
    print("\n2. Simulating agent activity with violations...")
    
    # Register agent
    ethics.register_agent("integrated_agent", {ActionCategory.RECONNAISSANCE})
    
    # Simulate various violations
    print("\n   Testing ethical violation...")
    ethics.validate_action(
        "integrated_agent",
        ActionCategory.SYSTEM_MODIFICATION,  # Not permitted
        "Modify firewall rules",
        {"target": "firewall"}
    )
    
    print("\n   Testing boundary violation...")
    boundary_enforcer.check_network_boundary("integrated_agent", "malicious.com")
    
    print("\n   Testing safety monitoring...")
    # Generate baseline
    for i in range(60):
        safety_monitor.observe_behavior(
            "integrated_agent",
            "reconnaissance",
            {"response_time": 1.0, "success": True}
        )
    
    # Generate anomaly
    safety_monitor.observe_behavior(
        "integrated_agent",
        "reconnaissance", 
        {"response_time": 25.0, "success": True}  # Very slow
    )
    
    print(f"\n3. Integration Results:")
    print(f"   Total violations detected: {violation_count['count']}")
    print(f"   Ethics decisions: {len(ethics.decisions)}")
    print(f"   Boundary violations: {len(boundary_enforcer.violations)}")
    print(f"   Safety alerts: {len(safety_monitor.alerts)}")
    
    print("\n   ✓ Integrated ethics and safety system operational")

def main():
    """Run all demonstrations."""
    print("🛡️  ARCHANGEL ETHICS & SAFETY SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    try:
        demo_ethics_overseer()
        demo_boundary_enforcement()
        demo_emergency_stop()
        demo_safety_monitor()
        demo_integrated_workflow()
        
        print("\n" + "=" * 80)
        print("✅ ALL ETHICS & SAFETY DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())