#!/usr/bin/env python3
"""
Tests for communication system functionality
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from agents.communication import (
    MessageBus, AgentMessage, TeamMessage, IntelligenceMessage,
    MessageType, Priority, CoordinationType, IntelligenceType,
    create_intelligence_message, create_team_message
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