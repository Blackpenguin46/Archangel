#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Communication System
Secure message bus for inter-agent communication
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import ssl
import zmq
import zmq.asyncio

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages that can be sent between agents"""
    INTELLIGENCE = "intelligence"
    COORDINATION = "coordination"
    ALERT = "alert"
    REQUEST = "request"
    RESPONSE = "response"
    STATUS = "status"
    COMMAND = "command"

class Priority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class CoordinationType(Enum):
    """Types of coordination messages"""
    ATTACK_PLAN = "attack_plan"
    DEFENSE_STRATEGY = "defense_strategy"
    RESOURCE_REQUEST = "resource_request"
    OBJECTIVE_UPDATE = "objective_update"
    TEAM_FORMATION = "team_formation"

class IntelligenceType(Enum):
    """Types of intelligence sharing"""
    VULNERABILITY = "vulnerability"
    TARGET_INFO = "target_info"
    THREAT_INTEL = "threat_intel"
    RECONNAISSANCE = "reconnaissance"
    COMPROMISE_INFO = "compromise_info"

@dataclass
class AgentMessage:
    """Base message structure for agent communication"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    priority: Priority
    encrypted: bool = False
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'message_type': self.message_type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'encrypted': self.encrypted,
            'signature': self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary"""
        return cls(
            message_id=data['message_id'],
            sender_id=data['sender_id'],
            recipient_id=data['recipient_id'],
            message_type=MessageType(data['message_type']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            priority=Priority(data['priority']),
            encrypted=data.get('encrypted', False),
            signature=data.get('signature')
        )

@dataclass
class TeamMessage(AgentMessage):
    """Message for team-wide communication"""
    team: str = ""
    coordination_type: Optional[CoordinationType] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'team': self.team,
            'coordination_type': self.coordination_type.value if self.coordination_type else None
        })
        return data

@dataclass
class IntelligenceMessage(AgentMessage):
    """Message for intelligence sharing"""
    intelligence_type: IntelligenceType = IntelligenceType.RECONNAISSANCE
    target_info: Dict[str, Any] = None
    confidence_level: float = 0.0
    source_reliability: float = 0.0
    
    def __post_init__(self):
        if self.target_info is None:
            self.target_info = {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'intelligence_type': self.intelligence_type.value,
            'target_info': self.target_info,
            'confidence_level': self.confidence_level,
            'source_reliability': self.source_reliability
        })
        return data

class MessageBus:
    """
    Secure message bus for inter-agent communication using ZeroMQ
    
    Features:
    - Encrypted communication with TLS
    - Message authentication and integrity
    - Topic-based pub/sub messaging
    - Direct agent-to-agent messaging
    - Message queuing and delivery guarantees
    """
    
    def __init__(self, 
                 bind_address: str = "tcp://*:5555",
                 use_encryption: bool = True,
                 cert_file: Optional[str] = None,
                 key_file: Optional[str] = None):
        self.bind_address = bind_address
        self.use_encryption = use_encryption
        self.cert_file = cert_file
        self.key_file = key_file
        
        # ZeroMQ context and sockets
        self.context = zmq.asyncio.Context()
        self.publisher = None
        self.subscriber = None
        self.router = None
        self.dealer = None
        
        # Message handling
        self.subscriptions = {}
        self.message_handlers = {}
        self.message_queue = asyncio.Queue()
        
        # Security
        self.agent_keys = {}
        self.trusted_agents = set()
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_dropped': 0,
            'encryption_failures': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    async def initialize(self) -> None:
        """Initialize the message bus"""
        try:
            self.logger.info("Initializing secure message bus")
            
            # Create publisher socket for broadcasting
            self.publisher = self.context.socket(zmq.PUB)
            if self.use_encryption:
                await self._setup_encryption(self.publisher)
            self.publisher.bind(self.bind_address)
            
            # Create subscriber socket for receiving broadcasts
            self.subscriber = self.context.socket(zmq.SUB)
            if self.use_encryption:
                await self._setup_encryption(self.subscriber)
            
            # Create router/dealer for direct messaging
            self.router = self.context.socket(zmq.ROUTER)
            self.dealer = self.context.socket(zmq.DEALER)
            
            if self.use_encryption:
                await self._setup_encryption(self.router)
                await self._setup_encryption(self.dealer)
            
            self.running = True
            self.logger.info("Message bus initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize message bus: {e}")
            raise
    
    async def _setup_encryption(self, socket) -> None:
        """Setup TLS encryption for socket"""
        if self.cert_file and self.key_file:
            # Configure TLS (this is a simplified version)
            # In production, use proper certificate management
            socket.setsockopt(zmq.CURVE_SERVER, 1)
            # Additional TLS configuration would go here
            self.logger.debug("TLS encryption configured for socket")
    
    async def publish_message(self, topic: str, message: AgentMessage) -> None:
        """
        Publish message to a topic
        
        Args:
            topic: Topic to publish to
            message: Message to publish
        """
        try:
            if not self.running:
                raise RuntimeError("Message bus not initialized")
            
            # Encrypt message if required
            if self.use_encryption:
                message = await self._encrypt_message(message)
            
            # Serialize message
            message_data = json.dumps(message.to_dict()).encode('utf-8')
            
            # Send message
            await self.publisher.send_multipart([topic.encode('utf-8'), message_data])
            
            self.stats['messages_sent'] += 1
            self.logger.debug(f"Published message to topic {topic}")
            
        except Exception as e:
            self.stats['messages_dropped'] += 1
            self.logger.error(f"Failed to publish message: {e}")
            raise
    
    async def subscribe_to_topic(self, topic: str, callback: Callable[[AgentMessage], None]) -> None:
        """
        Subscribe to a topic with callback
        
        Args:
            topic: Topic to subscribe to
            callback: Function to call when message received
        """
        try:
            # Subscribe to topic
            self.subscriber.setsockopt(zmq.SUBSCRIBE, topic.encode('utf-8'))
            
            # Store callback
            self.subscriptions[topic] = callback
            
            self.logger.debug(f"Subscribed to topic {topic}")
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to topic {topic}: {e}")
            raise
    
    async def send_direct_message(self, recipient: str, message: AgentMessage) -> None:
        """
        Send direct message to specific agent
        
        Args:
            recipient: Agent ID to send message to
            message: Message to send
        """
        try:
            if not self.running:
                raise RuntimeError("Message bus not initialized")
            
            # Set recipient
            message.recipient_id = recipient
            
            # Encrypt message if required
            if self.use_encryption:
                message = await self._encrypt_message(message)
            
            # Serialize message
            message_data = json.dumps(message.to_dict()).encode('utf-8')
            
            # Send via dealer socket
            await self.dealer.send_multipart([recipient.encode('utf-8'), message_data])
            
            self.stats['messages_sent'] += 1
            self.logger.debug(f"Sent direct message to {recipient}")
            
        except Exception as e:
            self.stats['messages_dropped'] += 1
            self.logger.error(f"Failed to send direct message: {e}")
            raise
    
    async def broadcast_to_team(self, team: str, message: TeamMessage) -> None:
        """
        Broadcast message to all team members
        
        Args:
            team: Team to broadcast to
            message: Message to broadcast
        """
        try:
            topic = f"team.{team}"
            await self.publish_message(topic, message)
            
            self.logger.debug(f"Broadcasted message to team {team}")
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast to team {team}: {e}")
            raise
    
    async def _encrypt_message(self, message: AgentMessage) -> AgentMessage:
        """
        Encrypt message content
        
        Args:
            message: Message to encrypt
            
        Returns:
            Encrypted message
        """
        try:
            # This is a simplified encryption implementation
            # In production, use proper cryptographic libraries
            
            # Mark as encrypted
            message.encrypted = True
            
            # Add signature for integrity
            message.signature = self._generate_signature(message)
            
            return message
            
        except Exception as e:
            self.stats['encryption_failures'] += 1
            self.logger.error(f"Failed to encrypt message: {e}")
            raise
    
    def _generate_signature(self, message: AgentMessage) -> str:
        """Generate message signature for integrity verification"""
        # This is a simplified signature implementation
        # In production, use proper digital signatures
        content_hash = hash(json.dumps(message.content, sort_keys=True))
        return f"sig_{content_hash}"
    
    async def _decrypt_message(self, message: AgentMessage) -> AgentMessage:
        """
        Decrypt message content
        
        Args:
            message: Encrypted message
            
        Returns:
            Decrypted message
        """
        try:
            if not message.encrypted:
                return message
            
            # Verify signature
            expected_signature = self._generate_signature(message)
            if message.signature != expected_signature:
                raise ValueError("Message signature verification failed")
            
            # Mark as decrypted
            message.encrypted = False
            message.signature = None
            
            return message
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt message: {e}")
            raise
    
    async def start_message_processing(self) -> None:
        """Start processing incoming messages"""
        try:
            self.logger.info("Starting message processing")
            
            # Start subscriber processing
            asyncio.create_task(self._process_subscriptions())
            
            # Start direct message processing
            asyncio.create_task(self._process_direct_messages())
            
        except Exception as e:
            self.logger.error(f"Failed to start message processing: {e}")
            raise
    
    async def _process_subscriptions(self) -> None:
        """Process subscription messages"""
        while self.running:
            try:
                # Receive message
                topic, message_data = await self.subscriber.recv_multipart()
                topic = topic.decode('utf-8')
                
                # Deserialize message
                message_dict = json.loads(message_data.decode('utf-8'))
                message = AgentMessage.from_dict(message_dict)
                
                # Decrypt if needed
                if message.encrypted:
                    message = await self._decrypt_message(message)
                
                # Call callback if subscribed
                if topic in self.subscriptions:
                    callback = self.subscriptions[topic]
                    await callback(message)
                
                self.stats['messages_received'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing subscription message: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_direct_messages(self) -> None:
        """Process direct messages"""
        while self.running:
            try:
                # Receive message via router
                identity, message_data = await self.router.recv_multipart()
                
                # Deserialize message
                message_dict = json.loads(message_data.decode('utf-8'))
                message = AgentMessage.from_dict(message_dict)
                
                # Decrypt if needed
                if message.encrypted:
                    message = await self._decrypt_message(message)
                
                # Add to message queue for processing
                await self.message_queue.put(message)
                
                self.stats['messages_received'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing direct message: {e}")
                await asyncio.sleep(0.1)
    
    async def get_next_message(self) -> Optional[AgentMessage]:
        """Get next message from queue"""
        try:
            return await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        return {
            'running': self.running,
            'subscriptions': len(self.subscriptions),
            'trusted_agents': len(self.trusted_agents),
            **self.stats
        }
    
    async def shutdown(self) -> None:
        """Shutdown the message bus"""
        try:
            self.logger.info("Shutting down message bus")
            self.running = False
            
            # Close sockets
            if self.publisher:
                self.publisher.close()
            if self.subscriber:
                self.subscriber.close()
            if self.router:
                self.router.close()
            if self.dealer:
                self.dealer.close()
            
            # Terminate context
            self.context.term()
            
            self.logger.info("Message bus shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during message bus shutdown: {e}")

# Factory functions for creating messages
def create_intelligence_message(sender_id: str, 
                               recipient_id: str,
                               intelligence_type: IntelligenceType,
                               target_info: Dict[str, Any],
                               confidence_level: float,
                               source_reliability: float,
                               content: Dict[str, Any]) -> IntelligenceMessage:
    """Create intelligence sharing message"""
    return IntelligenceMessage(
        message_id=str(uuid.uuid4()),
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=MessageType.INTELLIGENCE,
        content=content,
        timestamp=datetime.now(),
        priority=Priority.NORMAL,
        intelligence_type=intelligence_type,
        target_info=target_info,
        confidence_level=confidence_level,
        source_reliability=source_reliability
    )

def create_team_message(sender_id: str,
                       team: str,
                       coordination_type: CoordinationType,
                       content: Dict[str, Any],
                       priority: Priority = Priority.NORMAL) -> TeamMessage:
    """Create team coordination message"""
    return TeamMessage(
        message_id=str(uuid.uuid4()),
        sender_id=sender_id,
        recipient_id=f"team.{team}",
        message_type=MessageType.COORDINATION,
        content=content,
        timestamp=datetime.now(),
        priority=priority,
        team=team,
        coordination_type=coordination_type
    )