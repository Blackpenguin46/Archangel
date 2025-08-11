#!/usr/bin/env python3
"""
Demonstration of enhanced inter-agent communication and team coordination
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.communication import (
    MessageBus, AgentMessage, TeamMessage, IntelligenceMessage, AlertMessage, ResponseMessage,
    MessageType, Priority, CoordinationType, IntelligenceType, AlertType, ResponseAction,
    create_intelligence_message, create_team_message, create_alert_message, create_response_message
)
from datetime import datetime

async def demonstrate_red_team_coordination():
    """Demonstrate Red Team intelligence sharing and coordination"""
    print("🔴 RED TEAM COORDINATION DEMONSTRATION")
    print("=" * 50)
    
    # Create message bus
    message_bus = MessageBus(
        bind_address="tcp://localhost:5562",
        use_encryption=False
    )
    
    # Register Red Team agents
    agents = ["red_recon", "red_exploit", "red_persistence", "red_exfiltration"]
    for agent in agents:
        message_bus.register_agent(agent, "red")
    
    print(f"✓ Registered {len(agents)} Red Team agents")
    
    # Mock publish method to avoid ZMQ
    async def mock_publish(topic, message):
        message_bus._log_communication(message, topic)
        print(f"  📡 Published to {topic}: {message.message_type.value}")
        return True
    
    message_bus.publish_message = mock_publish
    
    # 1. Reconnaissance agent shares target intelligence
    print("\n1. Reconnaissance Phase - Intelligence Sharing")
    await message_bus.share_intelligence(
        sender_id="red_recon",
        team="red",
        intelligence_type=IntelligenceType.TARGET_INFO,
        target_info={
            "ip": "192.168.1.100",
            "hostname": "web-server-01",
            "os": "Ubuntu 20.04",
            "services": ["ssh:22", "http:80", "https:443"]
        },
        confidence_level=0.9,
        content={
            "scan_results": "3 open ports discovered",
            "vulnerabilities": ["outdated_ssh", "weak_ssl_config"],
            "next_steps": "attempt_ssh_bruteforce"
        }
    )
    
    # 2. Coordinate attack plan
    print("\n2. Attack Planning - Team Coordination")
    await message_bus.coordinate_team_action(
        sender_id="red_recon",
        team="red",
        coordination_type=CoordinationType.ATTACK_PLAN,
        action_details={
            "phase": "initial_access",
            "primary_target": "192.168.1.100:22",
            "assigned_agents": ["red_exploit"],
            "timeline": "immediate",
            "fallback_targets": ["192.168.1.100:80"]
        },
        priority=Priority.HIGH
    )
    
    # 3. Exploit agent shares success
    print("\n3. Exploitation Phase - Success Intelligence")
    await message_bus.share_intelligence(
        sender_id="red_exploit",
        team="red",
        intelligence_type=IntelligenceType.EXPLOIT_SUCCESS,
        target_info={
            "ip": "192.168.1.100",
            "access_method": "ssh_bruteforce",
            "credentials": "admin:password123"
        },
        confidence_level=1.0,
        content={
            "status": "initial_access_achieved",
            "privileges": "user_level",
            "next_phase": "privilege_escalation"
        }
    )
    
    # 4. Coordinate persistence
    print("\n4. Persistence Phase - Resource Coordination")
    await message_bus.coordinate_team_action(
        sender_id="red_exploit",
        team="red",
        coordination_type=CoordinationType.RESOURCE_REQUEST,
        action_details={
            "resource_type": "persistence_agent",
            "target_system": "192.168.1.100",
            "access_credentials": "admin:password123",
            "persistence_method": "ssh_key_implant"
        },
        priority=Priority.HIGH
    )
    
    # Display Red Team statistics
    print("\n📊 RED TEAM STATISTICS")
    stats = message_bus.get_statistics()
    print(f"  • Agents registered: {stats['red_team_agents']}")
    print(f"  • Intelligence shared: {stats['intelligence_shared']}")
    print(f"  • Messages sent: {stats['messages_sent']}")
    
    # Show intelligence gathered
    intel_list = message_bus.get_team_intelligence("red")
    print(f"  • Intelligence items: {len(intel_list)}")
    for i, intel in enumerate(intel_list, 1):
        print(f"    {i}. {intel['intelligence_type']} (confidence: {intel['confidence_level']})")
    
    return message_bus

async def demonstrate_blue_team_coordination(message_bus):
    """Demonstrate Blue Team alert and response coordination"""
    print("\n\n🔵 BLUE TEAM COORDINATION DEMONSTRATION")
    print("=" * 50)
    
    # Register Blue Team agents
    agents = ["blue_soc", "blue_firewall", "blue_siem", "blue_incident_response"]
    for agent in agents:
        message_bus.register_agent(agent, "blue")
    
    print(f"✓ Registered {len(agents)} Blue Team agents")
    
    # 1. SOC analyst detects suspicious activity
    print("\n1. Detection Phase - Security Alert")
    await message_bus.send_alert(
        sender_id="blue_soc",
        alert_type=AlertType.SUSPICIOUS_ACTIVITY,
        severity="high",
        source_system="SIEM",
        affected_assets=["web-server-01"],
        indicators={
            "src_ip": "unknown",
            "dst_ip": "192.168.1.100",
            "dst_port": 22,
            "activity": "multiple_failed_ssh_attempts",
            "count": 50,
            "time_window": "5_minutes"
        }
    )
    
    # 2. SIEM correlates and escalates
    print("\n2. Correlation Phase - Escalated Alert")
    await message_bus.send_alert(
        sender_id="blue_siem",
        alert_type=AlertType.INTRUSION_DETECTED,
        severity="critical",
        source_system="Correlation_Engine",
        affected_assets=["web-server-01"],
        indicators={
            "attack_pattern": "ssh_bruteforce_successful",
            "compromised_account": "admin",
            "login_time": datetime.now().isoformat(),
            "source_analysis": "likely_automated_attack"
        }
    )
    
    # 3. Coordinate immediate response
    print("\n3. Response Phase - Immediate Actions")
    await message_bus.coordinate_response(
        sender_id="blue_incident_response",
        response_action=ResponseAction.ISOLATE_HOST,
        target_assets=["web-server-01"],
        execution_priority=3,
        requires_approval=False
    )
    
    await message_bus.coordinate_response(
        sender_id="blue_firewall",
        response_action=ResponseAction.BLOCK_IP,
        target_assets=["perimeter_firewall"],
        execution_priority=2,
        requires_approval=False
    )
    
    # 4. Coordinate investigation
    print("\n4. Investigation Phase - Team Coordination")
    await message_bus.coordinate_team_action(
        sender_id="blue_incident_response",
        team="blue",
        coordination_type=CoordinationType.DEFENSE_STRATEGY,
        action_details={
            "incident_id": "INC-2024-001",
            "response_team": ["blue_soc", "blue_siem", "blue_incident_response"],
            "investigation_priority": "critical",
            "containment_status": "in_progress",
            "next_steps": ["forensic_analysis", "credential_reset", "system_hardening"]
        },
        priority=Priority.CRITICAL
    )
    
    # 5. Reset compromised credentials
    print("\n5. Recovery Phase - Credential Reset")
    await message_bus.coordinate_response(
        sender_id="blue_incident_response",
        response_action=ResponseAction.RESET_CREDENTIALS,
        target_assets=["web-server-01", "domain_controller"],
        execution_priority=1,
        requires_approval=True
    )
    
    # Display Blue Team statistics
    print("\n📊 BLUE TEAM STATISTICS")
    stats = message_bus.get_statistics()
    print(f"  • Agents registered: {stats['blue_team_agents']}")
    print(f"  • Alerts generated: {stats['alerts_generated']}")
    print(f"  • Responses coordinated: {stats['responses_coordinated']}")
    
    # Show coordination activities
    coord_list = message_bus.get_team_coordination("blue")
    print(f"  • Coordination activities: {len(coord_list)}")
    for i, coord in enumerate(coord_list, 1):
        print(f"    {i}. {coord['coordination_type']} (priority: {coord['priority']})")

def demonstrate_cross_team_monitoring(message_bus):
    """Demonstrate cross-team communication monitoring"""
    print("\n\n🔍 CROSS-TEAM COMMUNICATION MONITORING")
    print("=" * 50)
    
    # Get communication statistics
    stats = message_bus.get_statistics()
    print(f"✓ Total messages logged: {stats['communication_log_entries']}")
    print(f"✓ Cross-team messages: {stats['cross_team_messages_logged']}")
    print(f"✓ Red team messages: {stats['red_team_messages']}")
    print(f"✓ Blue team messages: {stats['blue_team_messages']}")
    
    # Show recent communication log
    print("\n📋 RECENT COMMUNICATION LOG")
    comm_log = message_bus.get_communication_log(limit=5)
    for i, entry in enumerate(comm_log, 1):
        sender_team = message_bus._get_agent_team(entry['sender_id'])
        print(f"  {i}. [{entry['timestamp'][:19]}] {sender_team.upper() if sender_team else 'UNKNOWN'} "
              f"{entry['sender_id']} -> {entry['message_type']} (priority: {entry['priority']})")
    
    # Show cross-team communications
    cross_team = message_bus.get_cross_team_communications(limit=3)
    if cross_team:
        print("\n⚠️  CROSS-TEAM COMMUNICATIONS DETECTED")
        for i, entry in enumerate(cross_team, 1):
            sender_team = message_bus._get_agent_team(entry['sender_id'])
            recipient_team = message_bus._get_agent_team(entry['recipient_id'])
            print(f"  {i}. {sender_team.upper() if sender_team else 'UNKNOWN'} -> "
                  f"{recipient_team.upper() if recipient_team else 'UNKNOWN'}: {entry['message_type']}")
    else:
        print("\n✓ No unauthorized cross-team communications detected")

async def demonstrate_security_features(message_bus):
    """Demonstrate security features"""
    print("\n\n🔒 SECURITY FEATURES DEMONSTRATION")
    print("=" * 50)
    
    # Test message validation
    print("1. Message Validation")
    valid_msg = AgentMessage(
        message_id="security_test_001",
        sender_id="test_agent",
        recipient_id="test_recipient",
        message_type=MessageType.STATUS,
        content={"test": "data"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    is_valid = message_bus.validate_message(valid_msg)
    print(f"  ✓ Valid message validation: {is_valid}")
    
    # Test signature generation
    print("\n2. Message Integrity")
    signature = message_bus._generate_signature(valid_msg)
    print(f"  ✓ Message signature generated: {signature[:20]}...")
    
    # Test encryption (mock mode)
    print("\n3. Message Encryption")
    original_content = valid_msg.content.copy()
    encrypted_msg = await message_bus._encrypt_message(valid_msg)
    print(f"  ✓ Message encrypted: {encrypted_msg.encrypted}")
    print(f"  ✓ Signature added: {encrypted_msg.signature is not None}")
    
    # Test decryption
    decrypted_msg = await message_bus._decrypt_message(encrypted_msg)
    print(f"  ✓ Message decrypted: {not decrypted_msg.encrypted}")
    
    # Show security statistics
    print("\n📊 SECURITY STATISTICS")
    stats = message_bus.get_statistics()
    print(f"  • Trusted agents: {stats['trusted_agents']}")
    print(f"  • Validation failures: {stats['validation_failures']}")
    print(f"  • Encryption failures: {stats['encryption_failures']}")

async def main():
    """Main demonstration"""
    print("🚀 ARCHANGEL AUTONOMOUS AI EVOLUTION")
    print("Inter-Agent Communication & Team Coordination Demo")
    print("=" * 60)
    
    try:
        # Demonstrate Red Team coordination
        message_bus = await demonstrate_red_team_coordination()
        
        # Demonstrate Blue Team coordination
        await demonstrate_blue_team_coordination(message_bus)
        
        # Demonstrate monitoring capabilities
        demonstrate_cross_team_monitoring(message_bus)
        
        # Demonstrate security features
        await demonstrate_security_features(message_bus)
        
        print("\n" + "=" * 60)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("\nKey Features Demonstrated:")
        print("  ✓ Red Team intelligence sharing and coordination")
        print("  ✓ Blue Team alert generation and response coordination")
        print("  ✓ Cross-team communication monitoring and logging")
        print("  ✓ Message validation, encryption, and integrity checking")
        print("  ✓ Team-specific channels and statistics")
        print("  ✓ Comprehensive audit trails and forensic capabilities")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)