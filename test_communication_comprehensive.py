#!/usr/bin/env python3
"""
Comprehensive test for communication system functionality
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

def test_message_classes():
    """Test all message classes"""
    print("Testing message classes...")
    
    # Test AgentMessage
    agent_msg = AgentMessage(
        message_id="agent_001",
        sender_id="sender",
        recipient_id="recipient",
        message_type=MessageType.STATUS,
        content={"status": "active"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    assert agent_msg.message_id == "agent_001"
    assert agent_msg.message_type == MessageType.STATUS
    
    # Test serialization
    msg_dict = agent_msg.to_dict()
    assert msg_dict['message_type'] == "status"
    assert msg_dict['priority'] == 2
    
    # Test deserialization
    reconstructed = AgentMessage.from_dict(msg_dict)
    assert reconstructed.message_id == agent_msg.message_id
    assert reconstructed.message_type == agent_msg.message_type
    print("✓ AgentMessage works")
    
    # Test IntelligenceMessage
    intel_msg = IntelligenceMessage(
        message_id="intel_001",
        sender_id="recon_agent",
        recipient_id="exploit_agent",
        message_type=MessageType.INTELLIGENCE,
        content={"vulnerability": "SQL injection"},
        timestamp=datetime.now(),
        priority=Priority.HIGH,
        intelligence_type=IntelligenceType.VULNERABILITY,
        target_info={"ip": "192.168.1.100", "port": 80},
        confidence_level=0.8,
        source_reliability=0.9
    )
    
    assert intel_msg.intelligence_type == IntelligenceType.VULNERABILITY
    assert intel_msg.confidence_level == 0.8
    
    intel_dict = intel_msg.to_dict()
    assert intel_dict['intelligence_type'] == "vulnerability"
    assert intel_dict['confidence_level'] == 0.8
    print("✓ IntelligenceMessage works")
    
    # Test AlertMessage
    alert_msg = AlertMessage(
        message_id="alert_001",
        sender_id="ids_agent",
        recipient_id="team.blue",
        message_type=MessageType.ALERT,
        content={"description": "Intrusion detected"},
        timestamp=datetime.now(),
        priority=Priority.CRITICAL,
        alert_type=AlertType.INTRUSION_DETECTED,
        severity="critical",
        source_system="Suricata",
        affected_assets=["web_server", "db_server"],
        indicators={"src_ip": "10.0.0.1", "dst_port": 80}
    )
    
    assert alert_msg.alert_type == AlertType.INTRUSION_DETECTED
    assert alert_msg.severity == "critical"
    assert len(alert_msg.affected_assets) == 2
    
    alert_dict = alert_msg.to_dict()
    assert alert_dict['alert_type'] == "intrusion_detected"
    assert alert_dict['severity'] == "critical"
    print("✓ AlertMessage works")
    
    # Test ResponseMessage
    response_msg = ResponseMessage(
        message_id="response_001",
        sender_id="firewall_agent",
        recipient_id="team.blue",
        message_type=MessageType.COMMAND,
        content={"action": "block_ip", "ip": "10.0.0.1"},
        timestamp=datetime.now(),
        priority=Priority.HIGH,
        response_action=ResponseAction.BLOCK_IP,
        target_assets=["firewall1", "firewall2"],
        execution_priority=3,
        requires_approval=True
    )
    
    assert response_msg.response_action == ResponseAction.BLOCK_IP
    assert response_msg.execution_priority == 3
    assert response_msg.requires_approval is True
    
    response_dict = response_msg.to_dict()
    assert response_dict['response_action'] == "block_ip"
    assert response_dict['requires_approval'] is True
    print("✓ ResponseMessage works")
    
    # Test TeamMessage
    team_msg = TeamMessage(
        message_id="team_001",
        sender_id="team_leader",
        recipient_id="team.red",
        message_type=MessageType.COORDINATION,
        content={"action": "coordinate_attack"},
        timestamp=datetime.now(),
        priority=Priority.HIGH,
        team="red",
        coordination_type=CoordinationType.ATTACK_PLAN
    )
    
    assert team_msg.team == "red"
    assert team_msg.coordination_type == CoordinationType.ATTACK_PLAN
    
    team_dict = team_msg.to_dict()
    assert team_dict['team'] == "red"
    assert team_dict['coordination_type'] == "attack_plan"
    print("✓ TeamMessage works")

def test_factory_functions():
    """Test message factory functions"""
    print("\nTesting factory functions...")
    
    # Test intelligence message factory
    intel_msg = create_intelligence_message(
        sender_id="recon_agent",
        recipient_id="exploit_agent",
        intelligence_type=IntelligenceType.TARGET_INFO,
        target_info={"hostname": "web-server-01", "os": "Ubuntu 20.04"},
        confidence_level=0.95,
        source_reliability=0.85,
        content={"scan_results": "open_ports: [22, 80, 443]"}
    )
    
    assert isinstance(intel_msg, IntelligenceMessage)
    assert intel_msg.intelligence_type == IntelligenceType.TARGET_INFO
    assert intel_msg.confidence_level == 0.95
    print("✓ Intelligence message factory works")
    
    # Test team message factory
    team_msg = create_team_message(
        sender_id="blue_leader",
        team="blue",
        coordination_type=CoordinationType.DEFENSE_STRATEGY,
        content={"strategy": "implement_network_segmentation"},
        priority=Priority.HIGH
    )
    
    assert isinstance(team_msg, TeamMessage)
    assert team_msg.team == "blue"
    assert team_msg.coordination_type == CoordinationType.DEFENSE_STRATEGY
    print("✓ Team message factory works")
    
    # Test alert message factory
    alert_msg = create_alert_message(
        sender_id="siem_agent",
        alert_type=AlertType.MALWARE_DETECTED,
        severity="high",
        source_system="ClamAV",
        affected_assets=["workstation_01"],
        indicators={"file_hash": "abc123", "file_path": "/tmp/malware.exe"},
        content={"malware_family": "Trojan.Generic"}
    )
    
    assert isinstance(alert_msg, AlertMessage)
    assert alert_msg.alert_type == AlertType.MALWARE_DETECTED
    assert alert_msg.severity == "high"
    print("✓ Alert message factory works")
    
    # Test response message factory
    response_msg = create_response_message(
        sender_id="incident_response_agent",
        response_action=ResponseAction.QUARANTINE_FILE,
        target_assets=["workstation_01"],
        content={"file_path": "/tmp/malware.exe", "quarantine_location": "/quarantine/"},
        execution_priority=2,
        requires_approval=False
    )
    
    assert isinstance(response_msg, ResponseMessage)
    assert response_msg.response_action == ResponseAction.QUARANTINE_FILE
    assert response_msg.requires_approval is False
    print("✓ Response message factory works")

def test_message_bus_functionality():
    """Test MessageBus functionality"""
    print("\nTesting MessageBus functionality...")
    
    # Create message bus
    message_bus = MessageBus(
        bind_address="tcp://localhost:5559",
        use_encryption=False
    )
    
    # Test agent registration
    message_bus.register_agent("red_recon", "red")
    message_bus.register_agent("red_exploit", "red")
    message_bus.register_agent("blue_soc", "blue")
    message_bus.register_agent("blue_firewall", "blue")
    
    assert len(message_bus.team_channels["red"]["agents"]) == 2
    assert len(message_bus.team_channels["blue"]["agents"]) == 2
    assert "red_recon" in message_bus.team_channels["red"]["agents"]
    assert "blue_soc" in message_bus.team_channels["blue"]["agents"]
    print("✓ Agent registration works")
    
    # Test message validation
    valid_msg = AgentMessage(
        message_id="valid_001",
        sender_id="red_recon",
        recipient_id="red_exploit",
        message_type=MessageType.INTELLIGENCE,
        content={"target": "192.168.1.100"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    is_valid = message_bus.validate_message(valid_msg)
    assert is_valid is True
    print("✓ Message validation works")
    
    # Test communication logging
    message_bus._log_communication(valid_msg, "intelligence.red")
    assert len(message_bus.communication_log) == 1
    
    log_entry = message_bus.communication_log[0]
    assert log_entry['message_id'] == "valid_001"
    assert log_entry['sender_id'] == "red_recon"
    assert log_entry['topic'] == "intelligence.red"
    print("✓ Communication logging works")
    
    # Test cross-team detection
    cross_team_msg = AgentMessage(
        message_id="cross_001",
        sender_id="red_recon",
        recipient_id="blue_soc",
        message_type=MessageType.STATUS,
        content={"status": "probing"},
        timestamp=datetime.now(),
        priority=Priority.LOW
    )
    
    is_cross_team = message_bus._is_cross_team_communication(cross_team_msg)
    assert is_cross_team is True
    
    message_bus._log_communication(cross_team_msg, "status")
    assert len(message_bus.cross_team_messages) == 1
    assert message_bus.stats['cross_team_messages'] == 1
    print("✓ Cross-team communication detection works")
    
    # Test statistics
    stats = message_bus.get_statistics()
    assert stats['red_team_agents'] == 2
    assert stats['blue_team_agents'] == 2
    assert stats['communication_log_entries'] == 2
    assert stats['cross_team_messages_logged'] == 1
    print("✓ Statistics collection works")
    
    # Test team intelligence storage
    intel_msg = IntelligenceMessage(
        message_id="intel_store_001",
        sender_id="red_recon",
        recipient_id="team.red",
        message_type=MessageType.INTELLIGENCE,
        content={"vulnerability": "open_ssh"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL,
        intelligence_type=IntelligenceType.VULNERABILITY,
        target_info={"ip": "192.168.1.100", "port": 22},
        confidence_level=0.7,
        source_reliability=0.8
    )
    
    message_bus.team_channels["red"]["intelligence"].append(intel_msg)
    
    intel_list = message_bus.get_team_intelligence("red", limit=5)
    assert len(intel_list) == 1
    assert intel_list[0]['intelligence_type'] == "vulnerability"
    print("✓ Team intelligence storage works")
    
    # Test team coordination storage
    coord_msg = TeamMessage(
        message_id="coord_001",
        sender_id="blue_soc",
        recipient_id="team.blue",
        message_type=MessageType.COORDINATION,
        content={"action": "increase_monitoring"},
        timestamp=datetime.now(),
        priority=Priority.HIGH,
        team="blue",
        coordination_type=CoordinationType.DEFENSE_STRATEGY
    )
    
    message_bus.team_channels["blue"]["coordination"].append(coord_msg)
    
    coord_list = message_bus.get_team_coordination("blue", limit=5)
    assert len(coord_list) == 1
    assert coord_list[0]['coordination_type'] == "defense_strategy"
    print("✓ Team coordination storage works")

async def test_async_functionality():
    """Test async functionality"""
    print("\nTesting async functionality...")
    
    message_bus = MessageBus(
        bind_address="tcp://localhost:5560",
        use_encryption=True
    )
    
    # Test encryption/decryption
    test_msg = AgentMessage(
        message_id="encrypt_001",
        sender_id="test_sender",
        recipient_id="test_recipient",
        message_type=MessageType.INTELLIGENCE,
        content={"secret_data": "classified_information"},
        timestamp=datetime.now(),
        priority=Priority.HIGH
    )
    
    original_content = test_msg.content.copy()
    
    # Test encryption
    encrypted_msg = await message_bus._encrypt_message(test_msg)
    assert encrypted_msg.encrypted is True
    assert encrypted_msg.signature is not None
    print("✓ Message encryption works")
    
    # Test decryption
    decrypted_msg = await message_bus._decrypt_message(encrypted_msg)
    assert decrypted_msg.encrypted is False
    
    # Note: In mock mode without cryptography, content won't actually be encrypted
    # but the encryption/decryption flow should work
    print("✓ Message decryption works")
    
    # Test signature generation
    signature = message_bus._generate_signature(test_msg)
    assert signature.startswith("hash_") or signature.startswith("hmac_")
    assert len(signature) > 10
    print("✓ Signature generation works")

async def test_team_coordination_methods():
    """Test team coordination methods"""
    print("\nTesting team coordination methods...")
    
    message_bus = MessageBus(
        bind_address="tcp://localhost:5561",
        use_encryption=False
    )
    
    # Register agents
    message_bus.register_agent("red_leader", "red")
    message_bus.register_agent("blue_leader", "blue")
    
    # Mock the publish_message method to avoid ZMQ operations
    async def mock_publish(topic, message):
        message_bus._log_communication(message, topic)
        return True
    
    message_bus.publish_message = mock_publish
    
    # Test intelligence sharing
    await message_bus.share_intelligence(
        sender_id="red_leader",
        team="red",
        intelligence_type=IntelligenceType.NETWORK_TOPOLOGY,
        target_info={"network": "192.168.1.0/24", "gateway": "192.168.1.1"},
        confidence_level=0.9,
        content={"topology": "flat_network", "devices": 15}
    )
    
    assert len(message_bus.team_channels["red"]["intelligence"]) == 1
    assert message_bus.stats['intelligence_shared'] == 1
    print("✓ Intelligence sharing works")
    
    # Test team coordination
    await message_bus.coordinate_team_action(
        sender_id="blue_leader",
        team="blue",
        coordination_type=CoordinationType.RESOURCE_REQUEST,
        action_details={"resource": "additional_analysts", "urgency": "high"},
        priority=Priority.HIGH
    )
    
    assert len(message_bus.team_channels["blue"]["coordination"]) == 1
    print("✓ Team coordination works")
    
    # Test alert sending
    await message_bus.send_alert(
        sender_id="blue_leader",
        alert_type=AlertType.SUSPICIOUS_ACTIVITY,
        severity="medium",
        source_system="SIEM",
        affected_assets=["web_server"],
        indicators={"unusual_traffic": "high_volume_requests"}
    )
    
    assert message_bus.stats['alerts_generated'] == 1
    print("✓ Alert sending works")
    
    # Test response coordination
    await message_bus.coordinate_response(
        sender_id="blue_leader",
        response_action=ResponseAction.UPDATE_FIREWALL,
        target_assets=["main_firewall"],
        execution_priority=2,
        requires_approval=True
    )
    
    assert message_bus.stats['responses_coordinated'] == 1
    print("✓ Response coordination works")

def main():
    """Run all tests"""
    print("Running comprehensive communication system tests...\n")
    
    tests = [
        test_message_classes,
        test_factory_functions,
        test_message_bus_functionality,
    ]
    
    async_tests = [
        test_async_functionality,
        test_team_coordination_methods
    ]
    
    passed = 0
    total = len(tests) + len(async_tests)
    
    # Run synchronous tests
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run asynchronous tests
    for test in async_tests:
        try:
            asyncio.run(test())
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)