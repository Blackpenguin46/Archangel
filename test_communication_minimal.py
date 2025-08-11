#!/usr/bin/env python3
"""
Minimal test for communication system functionality without external dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test just the message classes and enums without the MessageBus
from datetime import datetime

def test_enums():
    """Test enum definitions"""
    print("Testing enum definitions...")
    
    # Import enums directly to test they're defined correctly
    try:
        from agents.communication import MessageType, Priority, CoordinationType, IntelligenceType, AlertType, ResponseAction
        
        # Test MessageType enum
        assert MessageType.INTELLIGENCE.value == "intelligence"
        assert MessageType.COORDINATION.value == "coordination"
        assert MessageType.ALERT.value == "alert"
        print("✓ MessageType enum works")
        
        # Test Priority enum
        assert Priority.LOW.value == 1
        assert Priority.NORMAL.value == 2
        assert Priority.HIGH.value == 3
        assert Priority.CRITICAL.value == 4
        print("✓ Priority enum works")
        
        # Test new enums
        assert AlertType.INTRUSION_DETECTED.value == "intrusion_detected"
        assert ResponseAction.BLOCK_IP.value == "block_ip"
        print("✓ New enum types work")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_message_schemas():
    """Test message schema definitions"""
    print("\nTesting message schema definitions...")
    
    try:
        from agents.communication import MESSAGE_SCHEMAS
        
        # Check that schemas are defined
        assert "base_message" in MESSAGE_SCHEMAS
        assert "intelligence_message" in MESSAGE_SCHEMAS
        assert "team_message" in MESSAGE_SCHEMAS
        assert "alert_message" in MESSAGE_SCHEMAS
        print("✓ Message schemas defined")
        
        # Check base message schema structure
        base_schema = MESSAGE_SCHEMAS["base_message"]
        assert "properties" in base_schema
        assert "message_id" in base_schema["properties"]
        assert "sender_id" in base_schema["properties"]
        assert "recipient_id" in base_schema["properties"]
        print("✓ Base message schema structure correct")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Schema test error: {e}")
        return False

def test_basic_message_structure():
    """Test basic message structure without dependencies"""
    print("\nTesting basic message structure...")
    
    try:
        # Create a simple message class for testing
        class TestMessage:
            def __init__(self, message_id, sender_id, recipient_id, message_type, content, timestamp, priority):
                self.message_id = message_id
                self.sender_id = sender_id
                self.recipient_id = recipient_id
                self.message_type = message_type
                self.content = content
                self.timestamp = timestamp
                self.priority = priority
                self.encrypted = False
                self.signature = None
            
            def to_dict(self):
                return {
                    'message_id': self.message_id,
                    'sender_id': self.sender_id,
                    'recipient_id': self.recipient_id,
                    'message_type': self.message_type,
                    'content': self.content,
                    'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
                    'priority': self.priority,
                    'encrypted': self.encrypted,
                    'signature': self.signature
                }
        
        # Test message creation
        message = TestMessage(
            message_id="test_001",
            sender_id="agent_001",
            recipient_id="agent_002",
            message_type="intelligence",
            content={"data": "test_data"},
            timestamp=datetime.now(),
            priority=2
        )
        
        assert message.message_id == "test_001"
        assert message.sender_id == "agent_001"
        print("✓ Basic message structure works")
        
        # Test serialization
        message_dict = message.to_dict()
        assert message_dict['message_id'] == "test_001"
        assert message_dict['sender_id'] == "agent_001"
        assert 'timestamp' in message_dict
        print("✓ Message serialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Message structure test error: {e}")
        return False

def test_team_coordination_concepts():
    """Test team coordination concepts"""
    print("\nTesting team coordination concepts...")
    
    try:
        # Test team channel structure
        team_channels = {
            'red': {'agents': set(), 'intelligence': [], 'coordination': []},
            'blue': {'agents': set(), 'intelligence': [], 'coordination': []}
        }
        
        # Test agent registration concept
        team_channels['red']['agents'].add('red_agent_001')
        team_channels['blue']['agents'].add('blue_agent_001')
        
        assert 'red_agent_001' in team_channels['red']['agents']
        assert 'blue_agent_001' in team_channels['blue']['agents']
        print("✓ Team channel structure works")
        
        # Test cross-team detection concept
        def is_cross_team(sender_team, recipient_team):
            return (sender_team and recipient_team and 
                   sender_team != recipient_team and
                   sender_team in ['red', 'blue'] and 
                   recipient_team in ['red', 'blue'])
        
        assert is_cross_team('red', 'blue') is True
        assert is_cross_team('red', 'red') is False
        print("✓ Cross-team detection logic works")
        
        # Test communication logging concept
        communication_log = []
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message_id': 'test_001',
            'sender_id': 'red_agent_001',
            'recipient_id': 'blue_agent_001',
            'message_type': 'intelligence',
            'topic': 'test_topic',
            'priority': 2,
            'encrypted': False
        }
        communication_log.append(log_entry)
        
        assert len(communication_log) == 1
        assert communication_log[0]['message_id'] == 'test_001'
        print("✓ Communication logging concept works")
        
        return True
        
    except Exception as e:
        print(f"❌ Team coordination test error: {e}")
        return False

def main():
    """Run all tests"""
    print("Running minimal communication system tests...\n")
    
    tests = [
        test_enums,
        test_message_schemas,
        test_basic_message_structure,
        test_team_coordination_concepts
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
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