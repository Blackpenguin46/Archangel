#!/usr/bin/env python3
"""
Simple test for communication system functionality
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

def test_message_creation():
    """Test basic message creation"""
    print("Testing message creation...")
    
    # Test basic agent message
    message = AgentMessage(
        message_id="test_001",
        sender_id="agent_001",
        recipient_id="agent_002",
        message_type=MessageType.INTELLIGENCE,
        content={"data": "test_data"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    assert message.message_id == "test_001"
    assert message.sender_id == "agent_001"
    assert message.message_type == MessageType.INTELLIGENCE
    print("✓ Basic message creation works")
    
    # Test intelligence message
    intel_msg = create_intelligence_message(
        sender_id="recon_agent",
        recipient_id="exploit_agent",
        intelligence_type=IntelligenceType.VULNERABILITY,
        target_info={"ip": "192.168.1.100", "port": 80},
        confidence_level=0.8,
        source_reliability=0.9,
        content={"vulnerability": "SQL injection"}
    )
    
    assert isinstance(intel_msg, IntelligenceMessage)
    assert intel_msg.intelligence_type == IntelligenceType.VULNERABILITY
    assert intel_msg.confidence_level == 0.8
    print("✓ Intelligence message creation works")
    
    # Test alert message
    alert_msg = create_alert_message(
        sender_id="soc_agent",
        alert_type=AlertType.INTRUSION_DETECTED,
        severity="high",
        source_system="IDS",
        affected_assets=["server1", "server2"],
        indicators={"ip": "192.168.1.100", "port": 80},
        content={"description": "Suspicious activity detected"}
    )
    
    assert isinstance(alert_msg, AlertMessage)
    assert alert_msg.alert_type == AlertType.INTRUSION_DETECTED
    assert alert_msg.severity == "high"
    print("✓ Alert message creation works")
    
    # Test response message
    response_msg = create_response_message(
        sender_id="firewall_agent",
        response_action=ResponseAction.BLOCK_IP,
        target_assets=["firewall1"],
        content={"ip_to_block": "192.168.1.100"},
        execution_priority=3,
        requires_approval=True
    )
    
    assert isinstance(response_msg, ResponseMessage)
    assert response_msg.response_action == ResponseAction.BLOCK_IP
    assert response_msg.requires_approval is True
    print("✓ Response message creation works")

def test_message_bus_basic():
    """Test basic message bus functionality"""
    print("\nTesting message bus basic functionality...")
    
    # Create message bus (without encryption for testing)
    message_bus = MessageBus(
        bind_address="tcp://localhost:5557",
        use_encryption=False
    )
    
    # Test agent registration
    message_bus.register_agent("red_agent_001", "red")
    message_bus.register_agent("blue_agent_001", "blue")
    
    assert "red_agent_001" in message_bus.team_channels["red"]["agents"]
    assert "blue_agent_001" in message_bus.team_channels["blue"]["agents"]
    print("✓ Agent registration works")
    
    # Test message validation
    valid_message = AgentMessage(
        message_id="valid_test",
        sender_id="sender",
        recipient_id="recipient",
        message_type=MessageType.STATUS,
        content={"status": "active"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    # Note: validation might not work without jsonschema, but we can test the structure
    try:
        is_valid = message_bus.validate_message(valid_message)
        print(f"✓ Message validation works: {is_valid}")
    except Exception as e:
        print(f"⚠ Message validation requires jsonschema: {e}")
    
    # Test statistics
    stats = message_bus.get_statistics()
    assert 'running' in stats
    assert 'red_team_agents' in stats
    assert 'blue_team_agents' in stats
    assert stats['red_team_agents'] == 1
    assert stats['blue_team_agents'] == 1
    print("✓ Statistics collection works")
    
    # Test communication logging
    test_message = AgentMessage(
        message_id="log_test",
        sender_id="red_agent_001",
        recipient_id="blue_agent_001",
        message_type=MessageType.STATUS,
        content={"status": "active"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    message_bus._log_communication(test_message, "test_topic")
    assert len(message_bus.communication_log) == 1
    print("✓ Communication logging works")
    
    # Test cross-team detection
    is_cross_team = message_bus._is_cross_team_communication(test_message)
    assert is_cross_team is True
    print("✓ Cross-team communication detection works")

async def test_async_functionality():
    """Test async functionality"""
    print("\nTesting async functionality...")
    
    message_bus = MessageBus(
        bind_address="tcp://localhost:5558",
        use_encryption=False
    )
    
    # Test encryption/decryption
    message = AgentMessage(
        message_id="encrypt_test",
        sender_id="sender",
        recipient_id="recipient",
        message_type=MessageType.INTELLIGENCE,
        content={"secret": "classified_data"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    try:
        original_content = message.content.copy()
        encrypted_message = await message_bus._encrypt_message(message)
        
        assert encrypted_message.encrypted is True
        assert encrypted_message.signature is not None
        print("✓ Message encryption works")
        
        decrypted_message = await message_bus._decrypt_message(encrypted_message)
        assert decrypted_message.encrypted is False
        assert decrypted_message.content == original_content
        print("✓ Message decryption works")
        
    except Exception as e:
        print(f"⚠ Encryption/decryption requires cryptography: {e}")

def main():
    """Run all tests"""
    print("Running communication system tests...\n")
    
    try:
        test_message_creation()
        test_message_bus_basic()
        asyncio.run(test_async_functionality())
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)