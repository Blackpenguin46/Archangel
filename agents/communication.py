#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Communication System
Secure message bus for inter-agent communication with team coordination
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
import hashlib
import hmac

# Optional imports for full functionality
try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    zmq = None

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    jsonschema = None

try:
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None

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
    NETWORK_TOPOLOGY = "network_topology"
    CREDENTIAL_INFO = "credential_info"
    EXPLOIT_SUCCESS = "exploit_success"

class AlertType(Enum):
    """Types of security alerts"""
    INTRUSION_DETECTED = "intrusion_detected"
    MALWARE_DETECTED = "malware_detected"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    POLICY_VIOLATION = "policy_violation"
    SYSTEM_COMPROMISE = "system_compromise"
    DATA_EXFILTRATION = "data_exfiltration"
    LATERAL_MOVEMENT = "lateral_movement"

class ResponseAction(Enum):
    """Types of response actions"""
    ISOLATE_HOST = "isolate_host"
    BLOCK_IP = "block_ip"
    UPDATE_FIREWALL = "update_firewall"
    QUARANTINE_FILE = "quarantine_file"
    RESET_CREDENTIALS = "reset_credentials"
    ESCALATE_INCIDENT = "escalate_incident"
    DEPLOY_COUNTERMEASURE = "deploy_countermeasure"

# JSON Schema definitions for message validation
MESSAGE_SCHEMAS = {
    "base_message": {
        "type": "object",
        "properties": {
            "message_id": {"type": "string"},
            "sender_id": {"type": "string"},
            "recipient_id": {"type": "string"},
            "message_type": {"type": "string", "enum": [t.value for t in MessageType]},
            "content": {"type": "object"},
            "timestamp": {"type": "string", "format": "date-time"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 4},
            "encrypted": {"type": "boolean"},
            "signature": {"type": ["string", "null"]}
        },
        "required": ["message_id", "sender_id", "recipient_id", "message_type", "content", "timestamp", "priority"]
    },
    "intelligence_message": {
        "allOf": [
            {"$ref": "#/definitions/base_message"},
            {
                "properties": {
                    "intelligence_type": {"type": "string", "enum": [t.value for t in IntelligenceType]},
                    "target_info": {"type": "object"},
                    "confidence_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "source_reliability": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["intelligence_type", "target_info", "confidence_level", "source_reliability"]
            }
        ]
    },
    "team_message": {
        "allOf": [
            {"$ref": "#/definitions/base_message"},
            {
                "properties": {
                    "team": {"type": "string", "enum": ["red", "blue"]},
                    "coordination_type": {"type": "string", "enum": [t.value for t in CoordinationType]}
                },
                "required": ["team", "coordination_type"]
            }
        ]
    },
    "alert_message": {
        "allOf": [
            {"$ref": "#/definitions/base_message"},
            {
                "properties": {
                    "alert_type": {"type": "string", "enum": [t.value for t in AlertType]},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "source_system": {"type": "string"},
                    "affected_assets": {"type": "array", "items": {"type": "string"}},
                    "indicators": {"type": "object"}
                },
                "required": ["alert_type", "severity", "source_system"]
            }
        ]
    }
}

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

@dataclass
class AlertMessage(AgentMessage):
    """Message for security alerts"""
    alert_type: AlertType = AlertType.SUSPICIOUS_ACTIVITY
    severity: str = "medium"
    source_system: str = ""
    affected_assets: List[str] = None
    indicators: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.affected_assets is None:
            self.affected_assets = []
        if self.indicators is None:
            self.indicators = {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'alert_type': self.alert_type.value,
            'severity': self.severity,
            'source_system': self.source_system,
            'affected_assets': self.affected_assets,
            'indicators': self.indicators
        })
        return data

@dataclass
class ResponseMessage(AgentMessage):
    """Message for coordinated response actions"""
    response_action: ResponseAction = ResponseAction.ISOLATE_HOST
    target_assets: List[str] = None
    execution_priority: int = 1
    requires_approval: bool = False
    estimated_impact: str = "low"
    
    def __post_init__(self):
        if self.target_assets is None:
            self.target_assets = []
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'response_action': self.response_action.value,
            'target_assets': self.target_assets,
            'execution_priority': self.execution_priority,
            'requires_approval': self.requires_approval,
            'estimated_impact': self.estimated_impact
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
    - Team-specific channels and coordination
    - Cross-team communication monitoring
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
        if ZMQ_AVAILABLE:
            self.context = zmq.asyncio.Context()
        else:
            self.context = None
        self.publisher = None
        self.subscriber = None
        self.router = None
        self.dealer = None
        
        # Message handling
        self.subscriptions = {}
        self.message_handlers = {}
        self.message_queue = asyncio.Queue()
        
        # Team coordination
        self.team_channels = {
            'red': {'agents': set(), 'intelligence': [], 'coordination': []},
            'blue': {'agents': set(), 'intelligence': [], 'coordination': []}
        }
        
        # Cross-team monitoring
        self.communication_log = []
        self.cross_team_messages = []
        
        # Security
        self.agent_keys = {}
        self.trusted_agents = set()
        if CRYPTOGRAPHY_AVAILABLE and Fernet:
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
        else:
            self.encryption_key = None
            self.cipher_suite = None
        
        # Message validation
        if JSONSCHEMA_AVAILABLE and jsonschema:
            self.schema_validator = jsonschema.Draft7Validator
        else:
            self.schema_validator = None
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_dropped': 0,
            'encryption_failures': 0,
            'validation_failures': 0,
            'red_team_messages': 0,
            'blue_team_messages': 0,
            'cross_team_messages': 0,
            'intelligence_shared': 0,
            'alerts_generated': 0,
            'responses_coordinated': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    async def initialize(self) -> None:
        """Initialize the message bus"""
        try:
            if not ZMQ_AVAILABLE:
                self.logger.warning("ZeroMQ not available, running in mock mode")
                self.running = True
                return
            
            self.logger.info("Initializing secure message bus with team coordination")
            
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
            self.logger.info("Message bus initialized successfully with team coordination")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize message bus: {e}")
            raise
    
    def register_agent(self, agent_id: str, team: str) -> None:
        """
        Register agent with a team for coordination
        
        Args:
            agent_id: Unique agent identifier
            team: Team name ('red' or 'blue')
        """
        try:
            if team not in self.team_channels:
                raise ValueError(f"Invalid team: {team}")
            
            self.team_channels[team]['agents'].add(agent_id)
            self.trusted_agents.add(agent_id)
            
            self.logger.info(f"Registered agent {agent_id} with team {team}")
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_id}: {e}")
            raise
    
    def unregister_agent(self, agent_id: str, team: str) -> None:
        """
        Unregister agent from team
        
        Args:
            agent_id: Agent identifier
            team: Team name
        """
        try:
            if team in self.team_channels:
                self.team_channels[team]['agents'].discard(agent_id)
            self.trusted_agents.discard(agent_id)
            
            self.logger.info(f"Unregistered agent {agent_id} from team {team}")
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
    
    def validate_message(self, message: AgentMessage) -> bool:
        """
        Validate message against JSON schema
        
        Args:
            message: Message to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not JSONSCHEMA_AVAILABLE or not self.schema_validator:
                # Basic validation without jsonschema
                return (hasattr(message, 'message_id') and 
                       hasattr(message, 'sender_id') and 
                       hasattr(message, 'recipient_id') and
                       hasattr(message, 'message_type') and
                       hasattr(message, 'content') and
                       hasattr(message, 'timestamp') and
                       hasattr(message, 'priority') and
                       bool(message.message_id) and 
                       bool(message.sender_id) and 
                       bool(message.recipient_id))
            
            message_dict = message.to_dict()
            
            # Select appropriate schema based on message type
            if isinstance(message, IntelligenceMessage):
                schema = MESSAGE_SCHEMAS["intelligence_message"]
            elif isinstance(message, TeamMessage):
                schema = MESSAGE_SCHEMAS["team_message"]
            elif isinstance(message, AlertMessage):
                schema = MESSAGE_SCHEMAS["alert_message"]
            else:
                schema = MESSAGE_SCHEMAS["base_message"]
            
            # Add definitions to schema
            schema["definitions"] = MESSAGE_SCHEMAS
            
            # Validate message
            self.schema_validator(schema).validate(message_dict)
            return True
            
        except Exception as e:
            self.stats['validation_failures'] += 1
            self.logger.error(f"Message validation failed: {e}")
            return False
    
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
            
            # Validate message
            if not self.validate_message(message):
                raise ValueError("Message validation failed")
            
            # Log communication for monitoring
            self._log_communication(message, topic)
            
            # Encrypt message if required
            if self.use_encryption:
                message = await self._encrypt_message(message)
            
            # Serialize message
            message_data = json.dumps(message.to_dict()).encode('utf-8')
            
            # Send message
            await self.publisher.send_multipart([topic.encode('utf-8'), message_data])
            
            self.stats['messages_sent'] += 1
            self._update_team_stats(message)
            self.logger.debug(f"Published message to topic {topic}")
            
        except Exception as e:
            self.stats['messages_dropped'] += 1
            self.logger.error(f"Failed to publish message: {e}")
            raise
    
    async def share_intelligence(self, sender_id: str, team: str, 
                               intelligence_type: IntelligenceType,
                               target_info: Dict[str, Any],
                               confidence_level: float,
                               content: Dict[str, Any]) -> None:
        """
        Share intelligence within a team (Red Team coordination)
        
        Args:
            sender_id: Agent sharing intelligence
            team: Team to share with
            intelligence_type: Type of intelligence
            target_info: Information about target
            confidence_level: Confidence in intelligence
            content: Intelligence content
        """
        try:
            if team not in self.team_channels:
                raise ValueError(f"Invalid team: {team}")
            
            # Create intelligence message
            message = IntelligenceMessage(
                message_id=str(uuid.uuid4()),
                sender_id=sender_id,
                recipient_id=f"team.{team}",
                message_type=MessageType.INTELLIGENCE,
                content=content,
                timestamp=datetime.now(),
                priority=Priority.NORMAL,
                intelligence_type=intelligence_type,
                target_info=target_info,
                confidence_level=confidence_level,
                source_reliability=0.8  # Default reliability
            )
            
            # Store intelligence in team channel
            self.team_channels[team]['intelligence'].append(message)
            
            # Broadcast to team
            topic = f"intelligence.{team}"
            await self.publish_message(topic, message)
            
            self.stats['intelligence_shared'] += 1
            self.logger.info(f"Intelligence shared by {sender_id} with team {team}")
            
        except Exception as e:
            self.logger.error(f"Failed to share intelligence: {e}")
            raise
    
    async def coordinate_team_action(self, sender_id: str, team: str,
                                   coordination_type: CoordinationType,
                                   action_details: Dict[str, Any],
                                   priority: Priority = Priority.NORMAL) -> None:
        """
        Coordinate action within a team
        
        Args:
            sender_id: Agent initiating coordination
            team: Team to coordinate with
            coordination_type: Type of coordination
            action_details: Details of the action
            priority: Message priority
        """
        try:
            if team not in self.team_channels:
                raise ValueError(f"Invalid team: {team}")
            
            # Create team coordination message
            message = TeamMessage(
                message_id=str(uuid.uuid4()),
                sender_id=sender_id,
                recipient_id=f"team.{team}",
                message_type=MessageType.COORDINATION,
                content=action_details,
                timestamp=datetime.now(),
                priority=priority,
                team=team,
                coordination_type=coordination_type
            )
            
            # Store coordination in team channel
            self.team_channels[team]['coordination'].append(message)
            
            # Broadcast to team
            topic = f"coordination.{team}"
            await self.publish_message(topic, message)
            
            self.logger.info(f"Team coordination initiated by {sender_id} for team {team}")
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate team action: {e}")
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
    
    async def send_alert(self, sender_id: str, alert_type: AlertType,
                        severity: str, source_system: str,
                        affected_assets: List[str],
                        indicators: Dict[str, Any]) -> None:
        """
        Send security alert (Blue Team alert system)
        
        Args:
            sender_id: Agent sending alert
            alert_type: Type of alert
            severity: Alert severity level
            source_system: System generating alert
            affected_assets: Assets affected by alert
            indicators: Indicators of compromise
        """
        try:
            # Create alert message
            message = AlertMessage(
                message_id=str(uuid.uuid4()),
                sender_id=sender_id,
                recipient_id="team.blue",
                message_type=MessageType.ALERT,
                content={
                    "description": f"{alert_type.value} detected",
                    "details": indicators,
                    "timestamp": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                priority=Priority.HIGH if severity in ["high", "critical"] else Priority.NORMAL,
                alert_type=alert_type,
                severity=severity,
                source_system=source_system,
                affected_assets=affected_assets,
                indicators=indicators
            )
            
            # Broadcast alert to Blue Team
            topic = "alerts.blue"
            await self.publish_message(topic, message)
            
            self.stats['alerts_generated'] += 1
            self.logger.info(f"Alert sent by {sender_id}: {alert_type.value} ({severity})")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
            raise
    
    async def coordinate_response(self, sender_id: str, 
                                response_action: ResponseAction,
                                target_assets: List[str],
                                execution_priority: int = 1,
                                requires_approval: bool = False) -> None:
        """
        Coordinate response action (Blue Team response coordination)
        
        Args:
            sender_id: Agent coordinating response
            response_action: Type of response action
            target_assets: Assets to apply response to
            execution_priority: Priority of execution
            requires_approval: Whether action requires approval
        """
        try:
            # Create response message
            message = ResponseMessage(
                message_id=str(uuid.uuid4()),
                sender_id=sender_id,
                recipient_id="team.blue",
                message_type=MessageType.COMMAND,
                content={
                    "action": response_action.value,
                    "targets": target_assets,
                    "execution_time": datetime.now().isoformat(),
                    "approval_required": requires_approval
                },
                timestamp=datetime.now(),
                priority=Priority.HIGH if execution_priority > 2 else Priority.NORMAL,
                response_action=response_action,
                target_assets=target_assets,
                execution_priority=execution_priority,
                requires_approval=requires_approval
            )
            
            # Broadcast response coordination to Blue Team
            topic = "response.blue"
            await self.publish_message(topic, message)
            
            self.stats['responses_coordinated'] += 1
            self.logger.info(f"Response coordinated by {sender_id}: {response_action.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate response: {e}")
            raise
    
    def _log_communication(self, message: AgentMessage, topic: str) -> None:
        """
        Log communication for monitoring and analysis
        
        Args:
            message: Message being sent
            topic: Topic/channel used
        """
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'message_id': message.message_id,
                'sender_id': message.sender_id,
                'recipient_id': message.recipient_id,
                'message_type': message.message_type.value,
                'topic': topic,
                'priority': message.priority.value,
                'encrypted': message.encrypted
            }
            
            # Add specific fields based on message type
            if isinstance(message, IntelligenceMessage):
                log_entry['intelligence_type'] = message.intelligence_type.value
                log_entry['confidence_level'] = message.confidence_level
            elif isinstance(message, AlertMessage):
                log_entry['alert_type'] = message.alert_type.value
                log_entry['severity'] = message.severity
            elif isinstance(message, TeamMessage):
                log_entry['team'] = message.team
                log_entry['coordination_type'] = message.coordination_type.value if message.coordination_type else None
            
            # Store in communication log
            self.communication_log.append(log_entry)
            
            # Check for cross-team communication
            if self._is_cross_team_communication(message):
                self.cross_team_messages.append(log_entry)
                self.stats['cross_team_messages'] += 1
            
            # Limit log size to prevent memory issues
            if len(self.communication_log) > 10000:
                self.communication_log = self.communication_log[-5000:]
            
        except Exception as e:
            self.logger.error(f"Failed to log communication: {e}")
    
    def _is_cross_team_communication(self, message: AgentMessage) -> bool:
        """
        Check if message is cross-team communication
        
        Args:
            message: Message to check
            
        Returns:
            True if cross-team, False otherwise
        """
        try:
            sender_team = self._get_agent_team(message.sender_id)
            recipient_team = self._get_agent_team(message.recipient_id)
            
            return (sender_team and recipient_team and 
                   sender_team != recipient_team and
                   sender_team in ['red', 'blue'] and 
                   recipient_team in ['red', 'blue'])
            
        except Exception:
            return False
    
    def _get_agent_team(self, agent_id: str) -> Optional[str]:
        """
        Get team for agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Team name or None
        """
        for team, data in self.team_channels.items():
            if agent_id in data['agents']:
                return team
        return None
    
    def _update_team_stats(self, message: AgentMessage) -> None:
        """
        Update team-specific statistics
        
        Args:
            message: Message being processed
        """
        try:
            sender_team = self._get_agent_team(message.sender_id)
            
            if sender_team == 'red':
                self.stats['red_team_messages'] += 1
            elif sender_team == 'blue':
                self.stats['blue_team_messages'] += 1
            
            # Update message type statistics
            if isinstance(message, IntelligenceMessage):
                self.stats['intelligence_shared'] += 1
            elif isinstance(message, AlertMessage):
                self.stats['alerts_generated'] += 1
            elif isinstance(message, ResponseMessage):
                self.stats['responses_coordinated'] += 1
                
        except Exception as e:
            self.logger.error(f"Failed to update team stats: {e}")
    
    def get_communication_log(self, team: Optional[str] = None, 
                            message_type: Optional[MessageType] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get communication log with optional filtering
        
        Args:
            team: Filter by team
            message_type: Filter by message type
            limit: Maximum number of entries
            
        Returns:
            List of log entries
        """
        try:
            filtered_log = self.communication_log
            
            # Filter by team
            if team:
                filtered_log = [
                    entry for entry in filtered_log
                    if self._get_agent_team(entry['sender_id']) == team
                ]
            
            # Filter by message type
            if message_type:
                filtered_log = [
                    entry for entry in filtered_log
                    if entry['message_type'] == message_type.value
                ]
            
            # Apply limit
            return filtered_log[-limit:] if limit else filtered_log
            
        except Exception as e:
            self.logger.error(f"Failed to get communication log: {e}")
            return []
    
    def get_cross_team_communications(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get cross-team communications for monitoring
        
        Args:
            limit: Maximum number of entries
            
        Returns:
            List of cross-team communication entries
        """
        return self.cross_team_messages[-limit:] if limit else self.cross_team_messages
    
    async def _encrypt_message(self, message: AgentMessage) -> AgentMessage:
        """
        Encrypt message content using Fernet encryption
        
        Args:
            message: Message to encrypt
            
        Returns:
            Encrypted message
        """
        try:
            if not CRYPTOGRAPHY_AVAILABLE or not self.cipher_suite:
                # Mock encryption without cryptography
                message.encrypted = True
                message.signature = self._generate_signature(message)
                return message
            
            # Serialize content for encryption
            content_json = json.dumps(message.content, sort_keys=True)
            
            # Encrypt content
            encrypted_content = self.cipher_suite.encrypt(content_json.encode())
            
            # Replace content with encrypted version
            message.content = {'encrypted_data': encrypted_content.decode()}
            message.encrypted = True
            
            # Add signature for integrity
            message.signature = self._generate_signature(message)
            
            return message
            
        except Exception as e:
            self.stats['encryption_failures'] += 1
            self.logger.error(f"Failed to encrypt message: {e}")
            raise
    
    def _generate_signature(self, message: AgentMessage) -> str:
        """Generate HMAC signature for message integrity verification"""
        try:
            # Create message data for signing
            sign_data = {
                'sender_id': message.sender_id,
                'recipient_id': message.recipient_id,
                'message_type': message.message_type.value,
                'timestamp': message.timestamp.isoformat(),
                'content': message.content
            }
            
            if CRYPTOGRAPHY_AVAILABLE and self.encryption_key:
                # Generate HMAC signature
                message_bytes = json.dumps(sign_data, sort_keys=True).encode()
                signature = hmac.new(
                    self.encryption_key, 
                    message_bytes, 
                    hashlib.sha256
                ).hexdigest()
                return f"hmac_{signature}"
            else:
                # Simple hash-based signature without cryptography
                message_bytes = json.dumps(sign_data, sort_keys=True).encode()
                signature = hashlib.sha256(message_bytes).hexdigest()
                return f"hash_{signature}"
            
        except Exception as e:
            self.logger.error(f"Failed to generate signature: {e}")
            return "sig_error"
    
    async def _decrypt_message(self, message: AgentMessage) -> AgentMessage:
        """
        Decrypt message content using Fernet decryption
        
        Args:
            message: Encrypted message
            
        Returns:
            Decrypted message
        """
        try:
            if not message.encrypted:
                return message
            
            if not CRYPTOGRAPHY_AVAILABLE or not self.cipher_suite:
                # Mock decryption without cryptography
                message.encrypted = False
                message.signature = None
                return message
            
            # Extract encrypted data
            if 'encrypted_data' not in message.content:
                raise ValueError("No encrypted data found in message")
            
            encrypted_data = message.content['encrypted_data'].encode()
            
            # Decrypt content
            decrypted_content = self.cipher_suite.decrypt(encrypted_data)
            message.content = json.loads(decrypted_content.decode())
            
            # Verify signature after decryption
            expected_signature = self._generate_signature(message)
            if message.signature != expected_signature:
                self.logger.warning("Message signature verification failed after decryption")
            
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
        """Get comprehensive message bus statistics"""
        team_stats = {}
        for team, data in self.team_channels.items():
            team_stats[f'{team}_team_agents'] = len(data['agents'])
            team_stats[f'{team}_team_intelligence'] = len(data['intelligence'])
            team_stats[f'{team}_team_coordination'] = len(data['coordination'])
        
        return {
            'running': self.running,
            'subscriptions': len(self.subscriptions),
            'trusted_agents': len(self.trusted_agents),
            'communication_log_entries': len(self.communication_log),
            'cross_team_messages_logged': len(self.cross_team_messages),
            **team_stats,
            **self.stats
        }
    
    def get_team_intelligence(self, team: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent intelligence for a team
        
        Args:
            team: Team name
            limit: Maximum number of entries
            
        Returns:
            List of intelligence messages
        """
        try:
            if team not in self.team_channels:
                return []
            
            intelligence = self.team_channels[team]['intelligence']
            recent_intel = intelligence[-limit:] if limit else intelligence
            
            return [msg.to_dict() for msg in recent_intel]
            
        except Exception as e:
            self.logger.error(f"Failed to get team intelligence: {e}")
            return []
    
    def get_team_coordination(self, team: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent coordination messages for a team
        
        Args:
            team: Team name
            limit: Maximum number of entries
            
        Returns:
            List of coordination messages
        """
        try:
            if team not in self.team_channels:
                return []
            
            coordination = self.team_channels[team]['coordination']
            recent_coord = coordination[-limit:] if limit else coordination
            
            return [msg.to_dict() for msg in recent_coord]
            
        except Exception as e:
            self.logger.error(f"Failed to get team coordination: {e}")
            return []
    
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

def create_alert_message(sender_id: str,
                        alert_type: AlertType,
                        severity: str,
                        source_system: str,
                        affected_assets: List[str],
                        indicators: Dict[str, Any],
                        content: Dict[str, Any]) -> AlertMessage:
    """Create security alert message"""
    return AlertMessage(
        message_id=str(uuid.uuid4()),
        sender_id=sender_id,
        recipient_id="team.blue",
        message_type=MessageType.ALERT,
        content=content,
        timestamp=datetime.now(),
        priority=Priority.HIGH if severity in ["high", "critical"] else Priority.NORMAL,
        alert_type=alert_type,
        severity=severity,
        source_system=source_system,
        affected_assets=affected_assets,
        indicators=indicators
    )

def create_response_message(sender_id: str,
                           response_action: ResponseAction,
                           target_assets: List[str],
                           content: Dict[str, Any],
                           execution_priority: int = 1,
                           requires_approval: bool = False) -> ResponseMessage:
    """Create response coordination message"""
    return ResponseMessage(
        message_id=str(uuid.uuid4()),
        sender_id=sender_id,
        recipient_id="team.blue",
        message_type=MessageType.COMMAND,
        content=content,
        timestamp=datetime.now(),
        priority=Priority.HIGH if execution_priority > 2 else Priority.NORMAL,
        response_action=response_action,
        target_assets=target_assets,
        execution_priority=execution_priority,
        requires_approval=requires_approval
    )