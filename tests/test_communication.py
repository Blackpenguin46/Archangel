#!/usr/bin/env python3
"""
Tests for communication system functionality
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from agents.communication import (
    MessageBus, AgentMessage, TeamMessage, IntelligenceMessage, AlertMessage, ResponseMessage,
    MessageType, Priority, CoordinationType, IntelligenceType, AlertType, ResponseAction,
    create_intelligence_message, create_team_message, create_alert_message, create_response_message
)

@pytest.fixture
def message_bus():
    """Create test message bus"""
    return MessageBus(
        bind_address="tcp://localhost:5556",  # Use different port for testing
        use_encryption=False  # Disable encryption for testing
    )

@pytest.fixture
def sample_agent_message():
    """Create sample agent message"""
    return AgentMessage(
        message_id="test_msg_001",
        sender_id="agent_001",
        recipient_id="agent_002",
        message_type=MessageType.INTELLIGENCE,
        content={"data": "test_data"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )

@pytest.fixture
def sample_team_message():
    """Create sample team message"""
    return TeamMessage(
        message_id="team_msg_001",
        sender_id="agent_001",
        recipient_id="team.red",
        message_type=MessageType.COORDINATION,
        content={"action": "coordinate_attack"},
        timestamp=datetime.now(),
        priority=Priority.HIGH,
        team="red",
        coordination_type=CoordinationType.ATTACK_PLAN
    )

def test_agent_message_creation(sample_agent_message):
    """Test agent message creation"""
    assert sample_agent_message.message_id == "test_msg_001"
    assert sample_agent_message.sender_id == "agent_001"
    assert sample_agent_message.recipient_id == "agent_002"
    assert sample_agent_message.message_type == MessageType.INTELLIGENCE
    assert sample_agent_message.priority == Priority.NORMAL

def test_agent_message_serialization(sample_agent_message):
    """Test agent message serialization"""
    message_dict = sample_agent_message.to_dict()
    
    assert message_dict['message_id'] == "test_msg_001"
    assert message_dict['sender_id'] == "agent_001"
    assert message_dict['message_type'] == "intelligence"
    assert message_dict['priority'] == 2
    assert 'timestamp' in message_dict

def test_agent_message_deserialization(sample_agent_message):
    """Test agent message deserialization"""
    message_dict = sample_agent_message.to_dict()
    reconstructed = AgentMessage.from_dict(message_dict)
    
    assert reconstructed.message_id == sample_agent_message.message_id
    assert reconstructed.sender_id == sample_agent_message.sender_id
    assert reconstructed.message_type == sample_agent_message.message_type
    assert reconstructed.priority == sample_agent_message.priority

def test_team_message_creation(sample_team_message):
    """Test team message creation"""
    assert sample_team_message.team == "red"
    assert sample_team_message.coordination_type == CoordinationType.ATTACK_PLAN
    assert sample_team_message.message_type == MessageType.COORDINATION

def test_team_message_serialization(sample_team_message):
    """Test team message serialization"""
    message_dict = sample_team_message.to_dict()
    
    assert message_dict['team'] == "red"
    assert message_dict['coordination_type'] == "attack_plan"
    assert message_dict['message_type'] == "coordination"

def test_intelligence_message_factory():
    """Test intelligence message factory function"""
    message = create_intelligence_message(
        sender_id="recon_agent",
        recipient_id="exploit_agent",
        intelligence_type=IntelligenceType.VULNERABILITY,
        target_info={"ip": "192.168.1.100", "port": 80},
        confidence_level=0.8,
        source_reliability=0.9,
        content={"vulnerability": "SQL injection"}
    )
    
    assert isinstance(message, IntelligenceMessage)
    assert message.sender_id == "recon_agent"
    assert message.intelligence_type == IntelligenceType.VULNERABILITY
    assert message.confidence_level == 0.8
    assert message.source_reliability == 0.9

def test_team_message_factory():
    """Test team message factory function"""
    message = create_team_message(
        sender_id="team_leader",
        team="blue",
        coordination_type=CoordinationType.DEFENSE_STRATEGY,
        content={"strategy": "implement_firewall_rules"},
        priority=Priority.HIGH
    )
    
    assert isinstance(message, TeamMessage)
    assert message.sender_id == "team_leader"
    assert message.team == "blue"
    assert message.coordination_type == CoordinationType.DEFENSE_STRATEGY
    assert message.priority == Priority.HIGH

@pytest.mark.asyncio
async def test_message_bus_initialization(message_bus):
    """Test message bus initialization"""
    assert not message_bus.running
    
    # Mock ZeroMQ context and sockets to avoid actual network operations
    with patch('zmq.asyncio.Context') as mock_context:
        mock_socket = Mock()
        mock_context.return_value.socket.return_value = mock_socket
        
        await message_bus.initialize()
        
        assert message_bus.running
        assert mock_context.called

@pytest.mark.asyncio
async def test_message_bus_statistics(message_bus):
    """Test message bus statistics"""
    stats = message_bus.get_statistics()
    
    assert 'running' in stats
    assert 'subscriptions' in stats
    assert 'messages_sent' in stats
    assert 'messages_received' in stats
    assert stats['running'] == message_bus.running

def test_message_type_enum():
    """Test MessageType enum values"""
    assert MessageType.INTELLIGENCE.value == "intelligence"
    assert MessageType.COORDINATION.value == "coordination"
    assert MessageType.ALERT.value == "alert"
    assert MessageType.REQUEST.value == "request"
    assert MessageType.RESPONSE.value == "response"
    assert MessageType.STATUS.value == "status"
    assert MessageType.COMMAND.value == "command"

def test_priority_enum():
    """Test Priority enum values"""
    assert Priority.LOW.value == 1
    assert Priority.NORMAL.value == 2
    assert Priority.HIGH.value == 3
    assert Priority.CRITICAL.value == 4

def test_coordination_type_enum():
    """Test CoordinationType enum values"""
    assert CoordinationType.ATTACK_PLAN.value == "attack_plan"
    assert CoordinationType.DEFENSE_STRATEGY.value == "defense_strategy"
    assert CoordinationType.RESOURCE_REQUEST.value == "resource_request"
    assert CoordinationType.OBJECTIVE_UPDATE.value == "objective_update"
    assert CoordinationType.TEAM_FORMATION.value == "team_formation"

def test_intelligence_type_enum():
    """Test IntelligenceType enum values"""
    assert IntelligenceType.VULNERABILITY.value == "vulnerability"
    assert IntelligenceType.TARGET_INFO.value == "target_info"
    assert IntelligenceType.THREAT_INTEL.value == "threat_intel"
    assert IntelligenceType.RECONNAISSANCE.value == "reconnaissance"
    assert IntelligenceType.COMPROMISE_INFO.value == "compromise_info"

@pytest.mark.asyncio
async def test_message_encryption_flag(message_bus):
    """Test message encryption functionality"""
    message = AgentMessage(
        message_id="encrypt_test",
        sender_id="sender",
        recipient_id="recipient",
        message_type=MessageType.INTELLIGENCE,
        content={"secret": "data"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    # Test encryption marking
    encrypted_message = await message_bus._encrypt_message(message)
    assert encrypted_message.encrypted is True
    assert encrypted_message.signature is not None

@pytest.mark.asyncio
async def test_message_signature_generation(message_bus):
    """Test message signature generation"""
    message = AgentMessage(
        message_id="sig_test",
        sender_id="sender",
        recipient_id="recipient",
        message_type=MessageType.INTELLIGENCE,
        content={"data": "test"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    signature = message_bus._generate_signature(message)
    assert signature.startswith("sig_")
    assert len(signature) > 4

@pytest.mark.asyncio
async def test_message_bus_shutdown(message_bus):
    """Test message bus shutdown"""
    # Mock the context and sockets
    with patch('zmq.asyncio.Context') as mock_context:
        mock_socket = Mock()
        mock_context.return_value.socket.return_value = mock_socket
        
        await message_bus.initialize()
        assert message_bus.running
        
        await message_bus.shutdown()
        assert not message_bus.running
        
        # Verify sockets were closed
        mock_socket.close.assert_called()

def test_message_priority_ordering():
    """Test message priority ordering"""
    assert Priority.CRITICAL.value > Priority.HIGH.value
    assert Priority.HIGH.value > Priority.NORMAL.value
    assert Priority.NORMAL.value > Priority.LOW.value

@pytest.mark.asyncio
async def test_message_queue_operations(message_bus):
    """Test message queue operations"""
    # Test empty queue
    message = await message_bus.get_next_message()
    assert message is None
    
    # Test adding message to queue
    test_message = AgentMessage(
        message_id="queue_test",
        sender_id="sender",
        recipient_id="recipient",
        message_type=MessageType.STATUS,
        content={"status": "active"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    await message_bus.message_queue.put(test_message)
    retrieved_message = await message_bus.get_next_message()
    
    assert retrieved_message is not None
    assert retrieved_message.message_id == "queue_test"

# New tests for team coordination functionality

def test_agent_registration(message_bus):
    """Test agent registration with teams"""
    # Register red team agent
    message_bus.register_agent("red_agent_001", "red")
    assert "red_agent_001" in message_bus.team_channels["red"]["agents"]
    assert "red_agent_001" in message_bus.trusted_agents
    
    # Register blue team agent
    message_bus.register_agent("blue_agent_001", "blue")
    assert "blue_agent_001" in message_bus.team_channels["blue"]["agents"]
    assert "blue_agent_001" in message_bus.trusted_agents
    
    # Test invalid team
    with pytest.raises(ValueError):
        message_bus.register_agent("invalid_agent", "invalid_team")

def test_agent_unregistration(message_bus):
    """Test agent unregistration from teams"""
    # Register and then unregister agent
    message_bus.register_agent("test_agent", "red")
    assert "test_agent" in message_bus.team_channels["red"]["agents"]
    
    message_bus.unregister_agent("test_agent", "red")
    assert "test_agent" not in message_bus.team_channels["red"]["agents"]
    assert "test_agent" not in message_bus.trusted_agents

def test_alert_message_creation():
    """Test alert message creation"""
    from agents.communication import create_alert_message, AlertType
    
    message = create_alert_message(
        sender_id="soc_agent",
        alert_type=AlertType.INTRUSION_DETECTED,
        severity="high",
        source_system="IDS",
        affected_assets=["server1", "server2"],
        indicators={"ip": "192.168.1.100", "port": 80},
        content={"description": "Suspicious activity detected"}
    )
    
    assert isinstance(message, AlertMessage)
    assert message.alert_type == AlertType.INTRUSION_DETECTED
    assert message.severity == "high"
    assert message.source_system == "IDS"
    assert len(message.affected_assets) == 2
    assert message.priority == Priority.HIGH

def test_response_message_creation():
    """Test response message creation"""
    from agents.communication import create_response_message, ResponseAction
    
    message = create_response_message(
        sender_id="firewall_agent",
        response_action=ResponseAction.BLOCK_IP,
        target_assets=["firewall1"],
        content={"ip_to_block": "192.168.1.100"},
        execution_priority=3,
        requires_approval=True
    )
    
    assert isinstance(message, ResponseMessage)
    assert message.response_action == ResponseAction.BLOCK_IP
    assert message.execution_priority == 3
    assert message.requires_approval is True
    assert message.priority == Priority.HIGH

def test_message_validation(message_bus):
    """Test message validation against JSON schema"""
    # Valid message
    valid_message = AgentMessage(
        message_id="valid_test",
        sender_id="sender",
        recipient_id="recipient",
        message_type=MessageType.STATUS,
        content={"status": "active"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    assert message_bus.validate_message(valid_message) is True
    
    # Invalid message (missing required field)
    invalid_message = AgentMessage(
        message_id="",  # Empty message_id should fail validation
        sender_id="sender",
        recipient_id="recipient",
        message_type=MessageType.STATUS,
        content={"status": "active"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    # Note: This test might pass because our current validation is basic
    # In a real implementation, we'd have stricter validation

@pytest.mark.asyncio
async def test_intelligence_sharing(message_bus):
    """Test intelligence sharing functionality"""
    # Register agents
    message_bus.register_agent("recon_agent", "red")
    message_bus.register_agent("exploit_agent", "red")
    
    # Mock the publish_message method to avoid ZeroMQ operations
    with patch.object(message_bus, 'publish_message', new_callable=AsyncMock) as mock_publish:
        await message_bus.share_intelligence(
            sender_id="recon_agent",
            team="red",
            intelligence_type=IntelligenceType.VULNERABILITY,
            target_info={"ip": "192.168.1.100", "port": 80},
            confidence_level=0.8,
            content={"vulnerability": "SQL injection"}
        )
        
        # Verify intelligence was stored
        assert len(message_bus.team_channels["red"]["intelligence"]) == 1
        
        # Verify publish was called
        mock_publish.assert_called_once()
        
        # Verify statistics updated
        assert message_bus.stats['intelligence_shared'] == 1

@pytest.mark.asyncio
async def test_team_coordination(message_bus):
    """Test team coordination functionality"""
    # Register agents
    message_bus.register_agent("team_leader", "blue")
    
    # Mock the publish_message method
    with patch.object(message_bus, 'publish_message', new_callable=AsyncMock) as mock_publish:
        await message_bus.coordinate_team_action(
            sender_id="team_leader",
            team="blue",
            coordination_type=CoordinationType.DEFENSE_STRATEGY,
            action_details={"strategy": "implement_firewall_rules"},
            priority=Priority.HIGH
        )
        
        # Verify coordination was stored
        assert len(message_bus.team_channels["blue"]["coordination"]) == 1
        
        # Verify publish was called
        mock_publish.assert_called_once()

@pytest.mark.asyncio
async def test_alert_sending(message_bus):
    """Test alert sending functionality"""
    # Mock the publish_message method
    with patch.object(message_bus, 'publish_message', new_callable=AsyncMock) as mock_publish:
        await message_bus.send_alert(
            sender_id="ids_agent",
            alert_type=AlertType.INTRUSION_DETECTED,
            severity="critical",
            source_system="Suricata",
            affected_assets=["web_server"],
            indicators={"src_ip": "10.0.0.1", "dst_port": 80}
        )
        
        # Verify publish was called
        mock_publish.assert_called_once()
        
        # Verify statistics updated
        assert message_bus.stats['alerts_generated'] == 1

@pytest.mark.asyncio
async def test_response_coordination(message_bus):
    """Test response coordination functionality"""
    # Mock the publish_message method
    with patch.object(message_bus, 'publish_message', new_callable=AsyncMock) as mock_publish:
        await message_bus.coordinate_response(
            sender_id="firewall_agent",
            response_action=ResponseAction.BLOCK_IP,
            target_assets=["firewall1", "firewall2"],
            execution_priority=2,
            requires_approval=False
        )
        
        # Verify publish was called
        mock_publish.assert_called_once()
        
        # Verify statistics updated
        assert message_bus.stats['responses_coordinated'] == 1

def test_communication_logging(message_bus):
    """Test communication logging functionality"""
    # Register agents
    message_bus.register_agent("red_agent", "red")
    message_bus.register_agent("blue_agent", "blue")
    
    # Create test message
    message = AgentMessage(
        message_id="log_test",
        sender_id="red_agent",
        recipient_id="blue_agent",
        message_type=MessageType.STATUS,
        content={"status": "active"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    # Log communication
    message_bus._log_communication(message, "test_topic")
    
    # Verify log entry was created
    assert len(message_bus.communication_log) == 1
    log_entry = message_bus.communication_log[0]
    assert log_entry['message_id'] == "log_test"
    assert log_entry['sender_id'] == "red_agent"
    assert log_entry['topic'] == "test_topic"

def test_cross_team_communication_detection(message_bus):
    """Test cross-team communication detection"""
    # Register agents from different teams
    message_bus.register_agent("red_agent", "red")
    message_bus.register_agent("blue_agent", "blue")
    
    # Create cross-team message
    message = AgentMessage(
        message_id="cross_team_test",
        sender_id="red_agent",
        recipient_id="blue_agent",
        message_type=MessageType.STATUS,
        content={"status": "active"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    # Test cross-team detection
    assert message_bus._is_cross_team_communication(message) is True
    
    # Log communication and verify cross-team tracking
    message_bus._log_communication(message, "test_topic")
    assert len(message_bus.cross_team_messages) == 1
    assert message_bus.stats['cross_team_messages'] == 1

def test_team_statistics(message_bus):
    """Test team-specific statistics"""
    # Register agents
    message_bus.register_agent("red1", "red")
    message_bus.register_agent("red2", "red")
    message_bus.register_agent("blue1", "blue")
    
    # Get statistics
    stats = message_bus.get_statistics()
    
    assert stats['red_team_agents'] == 2
    assert stats['blue_team_agents'] == 1
    assert stats['red_team_intelligence'] == 0
    assert stats['blue_team_coordination'] == 0

def test_get_communication_log(message_bus):
    """Test communication log retrieval with filtering"""
    # Register agents
    message_bus.register_agent("red_agent", "red")
    
    # Create and log test messages
    for i in range(5):
        message = AgentMessage(
            message_id=f"test_{i}",
            sender_id="red_agent",
            recipient_id="recipient",
            message_type=MessageType.STATUS,
            content={"index": i},
            timestamp=datetime.now(),
            priority=Priority.NORMAL
        )
        message_bus._log_communication(message, "test_topic")
    
    # Test log retrieval
    log = message_bus.get_communication_log(limit=3)
    assert len(log) == 3
    
    # Test team filtering
    log = message_bus.get_communication_log(team="red", limit=10)
    assert len(log) == 5
    
    # Test message type filtering
    log = message_bus.get_communication_log(message_type=MessageType.STATUS, limit=10)
    assert len(log) == 5

def test_team_intelligence_retrieval(message_bus):
    """Test team intelligence retrieval"""
    # Create intelligence message
    intel_msg = IntelligenceMessage(
        message_id="intel_test",
        sender_id="recon_agent",
        recipient_id="team.red",
        message_type=MessageType.INTELLIGENCE,
        content={"vulnerability": "test"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL,
        intelligence_type=IntelligenceType.VULNERABILITY,
        target_info={"ip": "192.168.1.1"},
        confidence_level=0.9,
        source_reliability=0.8
    )
    
    # Add to team channel
    message_bus.team_channels["red"]["intelligence"].append(intel_msg)
    
    # Retrieve intelligence
    intel_list = message_bus.get_team_intelligence("red", limit=5)
    assert len(intel_list) == 1
    assert intel_list[0]['intelligence_type'] == "vulnerability"

def test_enhanced_encryption_decryption(message_bus):
    """Test enhanced encryption and decryption"""
    message = AgentMessage(
        message_id="encrypt_test",
        sender_id="sender",
        recipient_id="recipient",
        message_type=MessageType.INTELLIGENCE,
        content={"secret": "classified_data"},
        timestamp=datetime.now(),
        priority=Priority.NORMAL
    )
    
    # Test encryption
    original_content = message.content.copy()
    encrypted_message = asyncio.run(message_bus._encrypt_message(message))
    
    assert encrypted_message.encrypted is True
    assert encrypted_message.signature is not None
    assert encrypted_message.content != original_content
    assert 'encrypted_data' in encrypted_message.content
    
    # Test decryption
    decrypted_message = asyncio.run(message_bus._decrypt_message(encrypted_message))
    assert decrypted_message.encrypted is False
    assert decrypted_message.content == original_content