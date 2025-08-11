#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Blue Team Agents
Defensive cybersecurity agents for threat detection, response, and mitigation
"""

import asyncio
import logging
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field

from .base_agent import (
    Agent, Experience, ActionTaken, ActionOutcome, 
    EnvironmentContext, Team, Role
)
from .communication import CommunicationBus, Message, MessageType
from memory.vector_memory import VectorMemorySystem
from memory.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident ticket status"""
    NEW = "new"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class SecurityAlert:
    """Security alert data structure"""
    alert_id: str
    source: str
    alert_type: str
    severity: AlertSeverity
    timestamp: datetime
    description: str
    indicators: Dict[str, Any]
    raw_data: Dict[str, Any]
    confidence_score: float
    false_positive_likelihood: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'source': self.source,
            'alert_type': self.alert_type,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'indicators': self.indicators,
            'raw_data': self.raw_data,
            'confidence_score': self.confidence_score,
            'false_positive_likelihood': self.false_positive_likelihood
        }


@dataclass
class IncidentTicket:
    """Incident ticket data structure"""
    ticket_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: IncidentStatus
    assigned_to: Optional[str]
    created_by: str
    created_at: datetime
    updated_at: datetime
    related_alerts: List[str]
    indicators_of_compromise: List[str]
    response_actions: List[str]
    resolution_notes: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'ticket_id': self.ticket_id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'status': self.status.value,
            'assigned_to': self.assigned_to,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'related_alerts': self.related_alerts,
            'indicators_of_compromise': self.indicators_of_compromise,
            'response_actions': self.response_actions,
            'resolution_notes': self.resolution_notes
        }


class SOCAnalystAgent(Agent):
    """
    SOC Analyst Agent for alert monitoring and incident management.
    
    Capabilities:
    - Real-time alert monitoring and triage
    - Incident ticket creation and management
    - Alert correlation and pattern recognition
    - False positive identification
    - Threat intelligence integration
    - Escalation and notification management
    """
    
    def __init__(self, 
                 agent_id: str,
                 communication_bus: CommunicationBus,
                 memory_system: VectorMemorySystem,
                 knowledge_base: KnowledgeBase,
                 alert_sources: List[str] = None):
        super().__init__(
            agent_id=agent_id,
            team=Team.BLUE,
            role=Role.SOC_ANALYST,
            communication_bus=communication_bus,
            memory_system=memory_system
        )
        
        self.knowledge_base = knowledge_base
        self.alert_sources = alert_sources or [
            "SIEM", "IDS", "IPS", "EDR", "Firewall", "AV", "DLP"
        ]
        
        # Alert management
        self.active_alerts: Dict[str, SecurityAlert] = {}
        self.processed_alerts: Dict[str, SecurityAlert] = {}
        self.incident_tickets: Dict[str, IncidentTicket] = {}
        
        # Alert correlation
        self.correlation_rules: List[Dict[str, Any]] = []
        self.alert_patterns: Dict[str, List[str]] = {}
        
        # Performance metrics
        self.metrics = {
            'alerts_processed': 0,
            'incidents_created': 0,
            'false_positives_identified': 0,
            'mean_time_to_triage': 0.0,
            'escalations_sent': 0
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """Initialize the SOC Analyst agent"""
        await super().initialize()
        
        # Load correlation rules
        await self._load_correlation_rules()
        
        # Start alert monitoring
        self.alert_monitor_task = asyncio.create_task(self._monitor_alerts())
        
        self.logger.info(f"SOC Analyst Agent {self.agent_id} initialized")
    
    async def _load_correlation_rules(self) -> None:
        """Load alert correlation rules"""
        # Default correlation rules
        self.correlation_rules = [
            {
                'name': 'Multiple Failed Logins',
                'pattern': ['failed_login', 'failed_login', 'failed_login'],
                'timeframe': timedelta(minutes=5),
                'severity': AlertSeverity.MEDIUM,
                'description': 'Multiple failed login attempts detected'
            },
            {
                'name': 'Lateral Movement Pattern',
                'pattern': ['successful_login', 'process_creation', 'network_connection'],
                'timeframe': timedelta(minutes=10),
                'severity': AlertSeverity.HIGH,
                'description': 'Potential lateral movement activity'
            },
            {
                'name': 'Data Exfiltration Pattern',
                'pattern': ['file_access', 'data_compression', 'network_upload'],
                'timeframe': timedelta(minutes=15),
                'severity': AlertSeverity.CRITICAL,
                'description': 'Potential data exfiltration activity'
            }
        ]
    
    async def _monitor_alerts(self) -> None:
        """Monitor incoming security alerts"""
        while self.active:
            try:
                # Get messages from communication bus
                messages = await self.communication_bus.get_messages(self.agent_id)
                
                for message in messages:
                    if message.message_type == MessageType.SECURITY_ALERT:
                        await self._process_security_alert(message)
                    elif message.message_type == MessageType.INTELLIGENCE_REPORT:
                        await self._process_intelligence_report(message)
                
                # Perform periodic correlation analysis
                await self._correlate_alerts()
                
                # Check for stale alerts
                await self._cleanup_stale_alerts()
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _process_security_alert(self, message: Message) -> None:
        """Process incoming security alert"""
        try:
            alert_data = message.content
            
            # Create SecurityAlert object
            alert = SecurityAlert(
                alert_id=alert_data.get('alert_id', str(uuid.uuid4())),
                source=alert_data.get('source', 'unknown'),
                alert_type=alert_data.get('alert_type', 'generic'),
                severity=AlertSeverity(alert_data.get('severity', 'medium')),
                timestamp=datetime.fromisoformat(alert_data.get('timestamp', datetime.now().isoformat())),
                description=alert_data.get('description', ''),
                indicators=alert_data.get('indicators', {}),
                raw_data=alert_data.get('raw_data', {}),
                confidence_score=alert_data.get('confidence_score', 0.5),
                false_positive_likelihood=alert_data.get('false_positive_likelihood', 0.0)
            )
            
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            
            # Perform initial triage
            triage_result = await self._triage_alert(alert)
            
            # Create incident ticket if necessary
            if triage_result['create_incident']:
                await self._create_incident_ticket(alert, triage_result)
            
            # Update metrics
            self.metrics['alerts_processed'] += 1
            
            self.logger.info(f"Processed alert {alert.alert_id} from {alert.source}")
            
        except Exception as e:
            self.logger.error(f"Error processing security alert: {e}")
    
    async def _triage_alert(self, alert: SecurityAlert) -> Dict[str, Any]:
        """Perform alert triage and analysis"""
        triage_result = {
            'create_incident': False,
            'escalate': False,
            'false_positive': False,
            'confidence': 0.0,
            'reasoning': '',
            'recommended_actions': []
        }
        
        try:
            # Check for false positive indicators
            fp_score = await self._calculate_false_positive_score(alert)
            
            if fp_score > 0.8:
                triage_result['false_positive'] = True
                triage_result['reasoning'] = f"High false positive likelihood: {fp_score:.2f}"
                self.metrics['false_positives_identified'] += 1
                return triage_result
            
            # Analyze alert severity and indicators
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                triage_result['create_incident'] = True
                triage_result['escalate'] = True
            elif alert.severity == AlertSeverity.MEDIUM and alert.confidence_score > 0.7:
                triage_result['create_incident'] = True
            
            # Check for known attack patterns
            pattern_match = await self._check_attack_patterns(alert)
            if pattern_match:
                triage_result['create_incident'] = True
                triage_result['confidence'] = pattern_match['confidence']
                triage_result['reasoning'] = f"Matches known attack pattern: {pattern_match['pattern_name']}"
            
            # Generate recommended actions
            triage_result['recommended_actions'] = await self._generate_response_actions(alert)
            
            return triage_result
            
        except Exception as e:
            self.logger.error(f"Error in alert triage: {e}")
            # Default to creating incident for safety
            triage_result['create_incident'] = True
            triage_result['reasoning'] = f"Triage error, defaulting to incident creation: {e}"
            return triage_result
    
    async def _calculate_false_positive_score(self, alert: SecurityAlert) -> float:
        """Calculate false positive likelihood score"""
        try:
            # Query memory for similar alerts
            similar_experiences = await self.memory_system.retrieve_similar_experiences(
                query=f"{alert.alert_type} {alert.source} {alert.description}",
                agent_id=self.agent_id,
                limit=10,
                similarity_threshold=0.6
            )
            
            if not similar_experiences:
                return alert.false_positive_likelihood
            
            # Calculate FP score based on historical data
            fp_count = 0
            total_count = len(similar_experiences)
            
            for exp in similar_experiences:
                if exp.metadata.get('false_positive', False):
                    fp_count += 1
            
            historical_fp_rate = fp_count / total_count if total_count > 0 else 0.0
            
            # Combine with alert's inherent FP likelihood
            combined_score = (historical_fp_rate + alert.false_positive_likelihood) / 2
            
            return min(combined_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating false positive score: {e}")
            return alert.false_positive_likelihood
    
    async def _check_attack_patterns(self, alert: SecurityAlert) -> Optional[Dict[str, Any]]:
        """Check alert against known attack patterns"""
        try:
            # Query knowledge base for attack patterns
            patterns = await self.knowledge_base.query_attack_patterns(
                indicators=list(alert.indicators.keys()),
                alert_type=alert.alert_type
            )
            
            best_match = None
            best_confidence = 0.0
            
            for pattern in patterns:
                # Calculate pattern match confidence
                confidence = self._calculate_pattern_confidence(alert, pattern)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = {
                        'pattern_name': pattern.get('name', 'unknown'),
                        'pattern_id': pattern.get('pattern_id', 'unknown'),
                        'confidence': confidence,
                        'mitre_mapping': pattern.get('mitre_mapping', [])
                    }
            
            return best_match if best_confidence > 0.6 else None
            
        except Exception as e:
            self.logger.error(f"Error checking attack patterns: {e}")
            return None
    
    def _calculate_pattern_confidence(self, alert: SecurityAlert, pattern: Dict[str, Any]) -> float:
        """Calculate confidence score for pattern match"""
        try:
            # Simple pattern matching based on indicators
            pattern_indicators = set(pattern.get('indicators', []))
            alert_indicators = set(alert.indicators.keys())
            
            if not pattern_indicators:
                return 0.0
            
            # Calculate overlap
            overlap = len(pattern_indicators.intersection(alert_indicators))
            confidence = overlap / len(pattern_indicators)
            
            # Boost confidence for exact alert type match
            if pattern.get('alert_type') == alert.alert_type:
                confidence += 0.2
            
            # Boost confidence for severity match
            if pattern.get('severity') == alert.severity.value:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern confidence: {e}")
            return 0.0
    
    async def _generate_response_actions(self, alert: SecurityAlert) -> List[str]:
        """Generate recommended response actions"""
        actions = []
        
        try:
            # Basic actions based on alert type
            if alert.alert_type == 'malware_detection':
                actions.extend([
                    'isolate_affected_host',
                    'run_full_antivirus_scan',
                    'collect_malware_sample'
                ])
            elif alert.alert_type == 'network_intrusion':
                actions.extend([
                    'block_source_ip',
                    'analyze_network_traffic',
                    'check_for_lateral_movement'
                ])
            elif alert.alert_type == 'failed_login':
                actions.extend([
                    'check_account_status',
                    'review_login_patterns',
                    'consider_account_lockout'
                ])
            elif alert.alert_type == 'data_exfiltration':
                actions.extend([
                    'block_data_transfer',
                    'identify_data_accessed',
                    'notify_data_owner',
                    'escalate_to_management'
                ])
            
            # Add severity-based actions
            if alert.severity == AlertSeverity.CRITICAL:
                actions.extend([
                    'immediate_escalation',
                    'activate_incident_response_team',
                    'notify_management'
                ])
            elif alert.severity == AlertSeverity.HIGH:
                actions.extend([
                    'escalate_to_senior_analyst',
                    'begin_investigation'
                ])
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error generating response actions: {e}")
            return ['manual_investigation_required']
    
    async def _create_incident_ticket(self, alert: SecurityAlert, triage_result: Dict[str, Any]) -> str:
        """Create incident ticket from alert"""
        try:
            ticket_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{len(self.incident_tickets) + 1:04d}"
            
            # Generate ticket title and description
            title = f"{alert.alert_type.replace('_', ' ').title()} - {alert.source}"
            description = f"""
Alert ID: {alert.alert_id}
Source: {alert.source}
Severity: {alert.severity.value}
Confidence: {alert.confidence_score:.2f}

Description: {alert.description}

Indicators:
{json.dumps(alert.indicators, indent=2)}

Triage Analysis:
{triage_result['reasoning']}

Recommended Actions:
{chr(10).join(f"- {action}" for action in triage_result['recommended_actions'])}
            """.strip()
            
            # Create incident ticket
            ticket = IncidentTicket(
                ticket_id=ticket_id,
                title=title,
                description=description,
                severity=alert.severity,
                status=IncidentStatus.NEW,
                assigned_to=None,
                created_by=self.agent_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                related_alerts=[alert.alert_id],
                indicators_of_compromise=list(alert.indicators.keys()),
                response_actions=triage_result['recommended_actions'],
                resolution_notes=None
            )
            
            # Store ticket
            self.incident_tickets[ticket_id] = ticket
            
            # Send notification
            await self._send_incident_notification(ticket, triage_result['escalate'])
            
            # Update metrics
            self.metrics['incidents_created'] += 1
            if triage_result['escalate']:
                self.metrics['escalations_sent'] += 1
            
            self.logger.info(f"Created incident ticket {ticket_id} for alert {alert.alert_id}")
            
            return ticket_id
            
        except Exception as e:
            self.logger.error(f"Error creating incident ticket: {e}")
            raise
    
    async def _send_incident_notification(self, ticket: IncidentTicket, escalate: bool = False) -> None:
        """Send incident notification"""
        try:
            # Determine recipients based on severity and escalation
            recipients = ["soc_manager"]
            
            if escalate or ticket.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                recipients.extend(["incident_response_team", "security_manager"])
            
            if ticket.severity == AlertSeverity.CRITICAL:
                recipients.extend(["ciso", "management"])
            
            # Create notification message
            notification = Message(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id="broadcast",
                message_type=MessageType.INCIDENT_NOTIFICATION,
                content={
                    'ticket_id': ticket.ticket_id,
                    'title': ticket.title,
                    'severity': ticket.severity.value,
                    'status': ticket.status.value,
                    'created_at': ticket.created_at.isoformat(),
                    'indicators': ticket.indicators_of_compromise,
                    'recommended_actions': ticket.response_actions,
                    'escalate': escalate,
                    'recipients': recipients
                },
                timestamp=datetime.now()
            )
            
            await self.communication_bus.send_message(notification)
            
            self.logger.info(f"Sent incident notification for ticket {ticket.ticket_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending incident notification: {e}")
    
    async def _correlate_alerts(self) -> None:
        """Perform alert correlation analysis"""
        try:
            if len(self.active_alerts) < 2:
                return
            
            # Check correlation rules
            for rule in self.correlation_rules:
                correlated_alerts = await self._find_correlated_alerts(rule)
                
                if correlated_alerts:
                    await self._create_correlation_incident(rule, correlated_alerts)
            
        except Exception as e:
            self.logger.error(f"Error in alert correlation: {e}")
    
    async def _find_correlated_alerts(self, rule: Dict[str, Any]) -> List[SecurityAlert]:
        """Find alerts matching correlation rule"""
        try:
            pattern = rule['pattern']
            timeframe = rule['timeframe']
            current_time = datetime.now()
            
            # Get alerts within timeframe
            recent_alerts = [
                alert for alert in self.active_alerts.values()
                if current_time - alert.timestamp <= timeframe
            ]
            
            # Sort by timestamp
            recent_alerts.sort(key=lambda x: x.timestamp)
            
            # Look for pattern matches
            correlated_alerts = []
            pattern_index = 0
            
            for alert in recent_alerts:
                if pattern_index < len(pattern) and alert.alert_type == pattern[pattern_index]:
                    correlated_alerts.append(alert)
                    pattern_index += 1
                    
                    if pattern_index == len(pattern):
                        # Found complete pattern
                        return correlated_alerts
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error finding correlated alerts: {e}")
            return []
    
    async def _create_correlation_incident(self, rule: Dict[str, Any], alerts: List[SecurityAlert]) -> None:
        """Create incident from correlated alerts"""
        try:
            ticket_id = f"COR-{datetime.now().strftime('%Y%m%d')}-{len(self.incident_tickets) + 1:04d}"
            
            # Combine alert information
            alert_ids = [alert.alert_id for alert in alerts]
            all_indicators = set()
            for alert in alerts:
                all_indicators.update(alert.indicators.keys())
            
            description = f"""
Correlation Rule: {rule['name']}
Pattern: {' -> '.join(rule['pattern'])}
Timeframe: {rule['timeframe']}

Correlated Alerts:
{chr(10).join(f"- {alert.alert_id}: {alert.description}" for alert in alerts)}

Combined Indicators:
{chr(10).join(f"- {indicator}" for indicator in all_indicators)}
            """.strip()
            
            # Create correlation incident
            ticket = IncidentTicket(
                ticket_id=ticket_id,
                title=f"Correlated Activity: {rule['name']}",
                description=description,
                severity=rule['severity'],
                status=IncidentStatus.NEW,
                assigned_to=None,
                created_by=self.agent_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                related_alerts=alert_ids,
                indicators_of_compromise=list(all_indicators),
                response_actions=[
                    'investigate_correlated_activity',
                    'analyze_attack_timeline',
                    'assess_impact_scope'
                ],
                resolution_notes=None
            )
            
            self.incident_tickets[ticket_id] = ticket
            
            # Send high-priority notification
            await self._send_incident_notification(ticket, escalate=True)
            
            self.logger.info(f"Created correlation incident {ticket_id} from {len(alerts)} alerts")
            
        except Exception as e:
            self.logger.error(f"Error creating correlation incident: {e}")
    
    async def _cleanup_stale_alerts(self) -> None:
        """Clean up old processed alerts"""
        try:
            current_time = datetime.now()
            stale_threshold = timedelta(hours=24)
            
            stale_alerts = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if current_time - alert.timestamp > stale_threshold
            ]
            
            for alert_id in stale_alerts:
                alert = self.active_alerts.pop(alert_id)
                self.processed_alerts[alert_id] = alert
            
            if stale_alerts:
                self.logger.debug(f"Cleaned up {len(stale_alerts)} stale alerts")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up stale alerts: {e}")
    
    async def make_decision(self, environment_context: EnvironmentContext) -> ActionTaken:
        """Make SOC analyst decision based on current environment"""
        try:
            # Analyze current alert queue
            high_priority_alerts = [
                alert for alert in self.active_alerts.values()
                if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            ]
            
            if high_priority_alerts:
                # Focus on highest priority alert
                alert = max(high_priority_alerts, key=lambda x: (x.severity.value, x.confidence_score))
                
                action = ActionTaken(
                    action_id=str(uuid.uuid4()),
                    primary_action="investigate_alert",
                    action_type="incident_response",
                    target=alert.alert_id,
                    parameters={
                        'alert_type': alert.alert_type,
                        'severity': alert.severity.value,
                        'source': alert.source,
                        'indicators': alert.indicators
                    },
                    confidence_score=0.8,
                    reasoning=f"Investigating high-priority {alert.severity.value} alert from {alert.source}"
                )
            else:
                # Proactive monitoring
                action = ActionTaken(
                    action_id=str(uuid.uuid4()),
                    primary_action="monitor_alerts",
                    action_type="monitoring",
                    target="alert_queue",
                    parameters={
                        'active_alerts': len(self.active_alerts),
                        'monitoring_sources': self.alert_sources
                    },
                    confidence_score=0.6,
                    reasoning="Proactive alert monitoring and queue management"
                )
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error in SOC analyst decision making: {e}")
            
            # Fallback action
            return ActionTaken(
                action_id=str(uuid.uuid4()),
                primary_action="error_recovery",
                action_type="system_maintenance",
                target="self",
                parameters={'error': str(e)},
                confidence_score=0.3,
                reasoning=f"Error recovery due to: {e}"
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get SOC analyst performance metrics"""
        return {
            'agent_id': self.agent_id,
            'active_alerts': len(self.active_alerts),
            'processed_alerts': len(self.processed_alerts),
            'active_incidents': len([t for t in self.incident_tickets.values() 
                                   if t.status != IncidentStatus.CLOSED]),
            'total_incidents': len(self.incident_tickets),
            **self.metrics
        }
    
    async def shutdown(self) -> None:
        """Shutdown the SOC Analyst agent"""
        try:
            if hasattr(self, 'alert_monitor_task'):
                self.alert_monitor_task.cancel()
                try:
                    await self.alert_monitor_task
                except asyncio.CancelledError:
                    pass
            
            await super().shutdown()
            self.logger.info(f"SOC Analyst Agent {self.agent_id} shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during SOC analyst shutdown: {e}")

class FirewallRuleAction(Enum):
    """Firewall rule actions"""
    ALLOW = "allow"
    DENY = "deny"
    DROP = "drop"
    REJECT = "reject"


class FirewallRuleProtocol(Enum):
    """Firewall rule protocols"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    ANY = "any"


@dataclass
class FirewallRule:
    """Firewall rule data structure"""
    rule_id: str
    name: str
    action: FirewallRuleAction
    protocol: FirewallRuleProtocol
    source_ip: str
    source_port: str
    destination_ip: str
    destination_port: str
    direction: str  # inbound/outbound
    priority: int
    description: str
    created_by: str
    created_at: datetime
    expires_at: Optional[datetime]
    active: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'action': self.action.value,
            'protocol': self.protocol.value,
            'source_ip': self.source_ip,
            'source_port': self.source_port,
            'destination_ip': self.destination_ip,
            'destination_port': self.destination_port,
            'direction': self.direction,
            'priority': self.priority,
            'description': self.description,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'active': self.active
        }


class FirewallConfiguratorAgent(Agent):
    """
    Firewall Configurator Agent for dynamic rule generation and deployment.
    
    Capabilities:
    - Dynamic firewall rule generation based on threats
    - Automated rule deployment and management
    - Rule optimization and conflict resolution
    - Temporary and permanent rule creation
    - Rule expiration and cleanup
    - Integration with threat intelligence
    """
    
    def __init__(self, 
                 agent_id: str,
                 communication_bus: CommunicationBus,
                 memory_system: VectorMemorySystem,
                 knowledge_base: KnowledgeBase,
                 firewall_systems: List[str] = None):
        super().__init__(
            agent_id=agent_id,
            team=Team.BLUE,
            role=Role.FIREWALL_ADMIN,
            communication_bus=communication_bus,
            memory_system=memory_system
        )
        
        self.knowledge_base = knowledge_base
        self.firewall_systems = firewall_systems or [
            "perimeter_firewall", "internal_firewall", "host_firewall"
        ]
        
        # Rule management
        self.active_rules: Dict[str, FirewallRule] = {}
        self.rule_templates: Dict[str, Dict[str, Any]] = {}
        self.deployment_queue: List[str] = []
        
        # Rule optimization
        self.rule_conflicts: List[Dict[str, Any]] = []
        self.optimization_suggestions: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.metrics = {
            'rules_created': 0,
            'rules_deployed': 0,
            'rules_expired': 0,
            'conflicts_resolved': 0,
            'blocked_threats': 0,
            'false_positives': 0
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """Initialize the Firewall Configurator agent"""
        await super().initialize()
        
        # Load rule templates
        await self._load_rule_templates()
        
        # Start rule management tasks
        self.rule_manager_task = asyncio.create_task(self._manage_rules())
        self.deployment_task = asyncio.create_task(self._deploy_rules())
        
        self.logger.info(f"Firewall Configurator Agent {self.agent_id} initialized")
    
    async def _load_rule_templates(self) -> None:
        """Load firewall rule templates"""
        self.rule_templates = {
            'block_malicious_ip': {
                'name': 'Block Malicious IP',
                'action': FirewallRuleAction.DENY,
                'protocol': FirewallRuleProtocol.ANY,
                'source_port': 'any',
                'destination_ip': 'any',
                'destination_port': 'any',
                'direction': 'inbound',
                'priority': 100,
                'description': 'Block traffic from malicious IP address'
            },
            'block_malware_c2': {
                'name': 'Block C2 Communication',
                'action': FirewallRuleAction.DROP,
                'protocol': FirewallRuleProtocol.TCP,
                'source_ip': 'any',
                'source_port': 'any',
                'destination_port': 'any',
                'direction': 'outbound',
                'priority': 90,
                'description': 'Block command and control communication'
            },
            'limit_failed_logins': {
                'name': 'Rate Limit Failed Logins',
                'action': FirewallRuleAction.REJECT,
                'protocol': FirewallRuleProtocol.TCP,
                'source_port': 'any',
                'destination_ip': 'any',
                'destination_port': '22,3389',
                'direction': 'inbound',
                'priority': 80,
                'description': 'Rate limit authentication attempts'
            },
            'block_data_exfiltration': {
                'name': 'Block Data Exfiltration',
                'action': FirewallRuleAction.DENY,
                'protocol': FirewallRuleProtocol.ANY,
                'source_ip': 'any',
                'source_port': 'any',
                'destination_port': '80,443,21,22',
                'direction': 'outbound',
                'priority': 95,
                'description': 'Block potential data exfiltration channels'
            }
        }
    
    async def _manage_rules(self) -> None:
        """Manage firewall rules lifecycle"""
        while self.active:
            try:
                # Process incoming messages
                messages = await self.communication_bus.get_messages(self.agent_id)
                
                for message in messages:
                    if message.message_type == MessageType.THREAT_DETECTED:
                        await self._handle_threat_detection(message)
                    elif message.message_type == MessageType.INCIDENT_NOTIFICATION:
                        await self._handle_incident_notification(message)
                    elif message.message_type == MessageType.INTELLIGENCE_REPORT:
                        await self._handle_intelligence_report(message)
                
                # Check for expired rules
                await self._cleanup_expired_rules()
                
                # Optimize rules
                await self._optimize_rules()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in rule management: {e}")
                await asyncio.sleep(10)
    
    async def _handle_threat_detection(self, message: Message) -> None:
        """Handle threat detection and create appropriate firewall rules"""
        try:
            threat_data = message.content
            threat_type = threat_data.get('threat_type', 'unknown')
            
            # Generate rules based on threat type
            rules = await self._generate_threat_response_rules(threat_data)
            
            for rule in rules:
                await self._create_firewall_rule(rule)
            
            self.logger.info(f"Created {len(rules)} rules for threat type: {threat_type}")
            
        except Exception as e:
            self.logger.error(f"Error handling threat detection: {e}")
    
    async def _generate_threat_response_rules(self, threat_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate firewall rules based on threat data"""
        rules = []
        threat_type = threat_data.get('threat_type', 'unknown')
        indicators = threat_data.get('indicators', {})
        
        try:
            # Generate rules based on threat type
            if threat_type == 'malicious_ip':
                malicious_ips = indicators.get('ip_addresses', [])
                for ip in malicious_ips:
                    rule_data = self.rule_templates['block_malicious_ip'].copy()
                    rule_data['source_ip'] = ip
                    rule_data['name'] = f"Block Malicious IP {ip}"
                    rule_data['expires_at'] = datetime.now() + timedelta(hours=24)
                    rules.append(rule_data)
            
            elif threat_type == 'c2_communication':
                c2_domains = indicators.get('domains', [])
                c2_ips = indicators.get('ip_addresses', [])
                
                for domain in c2_domains:
                    rule_data = self.rule_templates['block_malware_c2'].copy()
                    rule_data['destination_ip'] = domain
                    rule_data['name'] = f"Block C2 Domain {domain}"
                    rule_data['expires_at'] = datetime.now() + timedelta(days=7)
                    rules.append(rule_data)
                
                for ip in c2_ips:
                    rule_data = self.rule_templates['block_malware_c2'].copy()
                    rule_data['destination_ip'] = ip
                    rule_data['name'] = f"Block C2 IP {ip}"
                    rule_data['expires_at'] = datetime.now() + timedelta(days=7)
                    rules.append(rule_data)
            
            elif threat_type == 'brute_force_attack':
                source_ips = indicators.get('source_ips', [])
                target_ports = indicators.get('target_ports', ['22', '3389'])
                
                for ip in source_ips:
                    rule_data = self.rule_templates['limit_failed_logins'].copy()
                    rule_data['source_ip'] = ip
                    rule_data['destination_port'] = ','.join(target_ports)
                    rule_data['name'] = f"Block Brute Force from {ip}"
                    rule_data['expires_at'] = datetime.now() + timedelta(hours=12)
                    rules.append(rule_data)
            
            elif threat_type == 'data_exfiltration':
                suspicious_ips = indicators.get('destination_ips', [])
                
                for ip in suspicious_ips:
                    rule_data = self.rule_templates['block_data_exfiltration'].copy()
                    rule_data['destination_ip'] = ip
                    rule_data['name'] = f"Block Exfiltration to {ip}"
                    rule_data['expires_at'] = datetime.now() + timedelta(hours=6)
                    rules.append(rule_data)
            
            return rules
            
        except Exception as e:
            self.logger.error(f"Error generating threat response rules: {e}")
            return []
    
    async def _create_firewall_rule(self, rule_data: Dict[str, Any]) -> str:
        """Create a new firewall rule"""
        try:
            rule_id = f"FW-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
            
            # Create FirewallRule object
            rule = FirewallRule(
                rule_id=rule_id,
                name=rule_data['name'],
                action=rule_data['action'],
                protocol=rule_data['protocol'],
                source_ip=rule_data.get('source_ip', 'any'),
                source_port=rule_data.get('source_port', 'any'),
                destination_ip=rule_data.get('destination_ip', 'any'),
                destination_port=rule_data.get('destination_port', 'any'),
                direction=rule_data.get('direction', 'inbound'),
                priority=rule_data.get('priority', 50),
                description=rule_data.get('description', ''),
                created_by=self.agent_id,
                created_at=datetime.now(),
                expires_at=rule_data.get('expires_at'),
                active=True
            )
            
            # Check for conflicts
            conflicts = await self._check_rule_conflicts(rule)
            if conflicts:
                await self._resolve_rule_conflicts(rule, conflicts)
            
            # Store rule
            self.active_rules[rule_id] = rule
            
            # Queue for deployment
            self.deployment_queue.append(rule_id)
            
            # Update metrics
            self.metrics['rules_created'] += 1
            
            self.logger.info(f"Created firewall rule {rule_id}: {rule.name}")
            
            return rule_id
            
        except Exception as e:
            self.logger.error(f"Error creating firewall rule: {e}")
            raise
    
    async def _check_rule_conflicts(self, new_rule: FirewallRule) -> List[Dict[str, Any]]:
        """Check for conflicts with existing rules"""
        conflicts = []
        
        try:
            for existing_rule in self.active_rules.values():
                if not existing_rule.active:
                    continue
                
                # Check for overlapping conditions
                conflict_score = self._calculate_rule_overlap(new_rule, existing_rule)
                
                if conflict_score > 0.8:  # High overlap threshold
                    conflicts.append({
                        'existing_rule': existing_rule,
                        'conflict_type': 'high_overlap',
                        'conflict_score': conflict_score,
                        'resolution': 'merge_or_prioritize'
                    })
                elif conflict_score > 0.5:  # Medium overlap
                    conflicts.append({
                        'existing_rule': existing_rule,
                        'conflict_type': 'medium_overlap',
                        'conflict_score': conflict_score,
                        'resolution': 'adjust_priority'
                    })
            
            return conflicts
            
        except Exception as e:
            self.logger.error(f"Error checking rule conflicts: {e}")
            return []
    
    def _calculate_rule_overlap(self, rule1: FirewallRule, rule2: FirewallRule) -> float:
        """Calculate overlap score between two firewall rules"""
        try:
            overlap_score = 0.0
            
            # Check IP address overlap
            if (rule1.source_ip == rule2.source_ip or 
                rule1.source_ip == 'any' or rule2.source_ip == 'any'):
                overlap_score += 0.2
            
            if (rule1.destination_ip == rule2.destination_ip or 
                rule1.destination_ip == 'any' or rule2.destination_ip == 'any'):
                overlap_score += 0.2
            
            # Check port overlap
            if (rule1.source_port == rule2.source_port or 
                rule1.source_port == 'any' or rule2.source_port == 'any'):
                overlap_score += 0.15
            
            if (rule1.destination_port == rule2.destination_port or 
                rule1.destination_port == 'any' or rule2.destination_port == 'any'):
                overlap_score += 0.15
            
            # Check protocol overlap
            if (rule1.protocol == rule2.protocol or 
                rule1.protocol == FirewallRuleProtocol.ANY or 
                rule2.protocol == FirewallRuleProtocol.ANY):
                overlap_score += 0.1
            
            # Check direction overlap
            if rule1.direction == rule2.direction:
                overlap_score += 0.1
            
            # Check action conflict
            if rule1.action != rule2.action:
                overlap_score += 0.1  # Conflicting actions increase overlap concern
            
            return min(overlap_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating rule overlap: {e}")
            return 0.0
    
    async def _resolve_rule_conflicts(self, new_rule: FirewallRule, conflicts: List[Dict[str, Any]]) -> None:
        """Resolve conflicts between firewall rules"""
        try:
            for conflict in conflicts:
                existing_rule = conflict['existing_rule']
                conflict_type = conflict['conflict_type']
                
                if conflict_type == 'high_overlap':
                    # Check if new rule is more specific or has higher priority
                    if new_rule.priority > existing_rule.priority:
                        # Deactivate existing rule
                        existing_rule.active = False
                        self.logger.info(f"Deactivated conflicting rule {existing_rule.rule_id}")
                    else:
                        # Adjust new rule priority
                        new_rule.priority = existing_rule.priority + 1
                        self.logger.info(f"Adjusted new rule priority to {new_rule.priority}")
                
                elif conflict_type == 'medium_overlap':
                    # Adjust priority to ensure proper ordering
                    if new_rule.action == FirewallRuleAction.DENY and existing_rule.action == FirewallRuleAction.ALLOW:
                        new_rule.priority = max(new_rule.priority, existing_rule.priority + 1)
                    elif new_rule.action == FirewallRuleAction.ALLOW and existing_rule.action == FirewallRuleAction.DENY:
                        new_rule.priority = min(new_rule.priority, existing_rule.priority - 1)
            
            self.metrics['conflicts_resolved'] += len(conflicts)
            
        except Exception as e:
            self.logger.error(f"Error resolving rule conflicts: {e}")
    
    async def _deploy_rules(self) -> None:
        """Deploy queued firewall rules"""
        while self.active:
            try:
                if self.deployment_queue:
                    rule_id = self.deployment_queue.pop(0)
                    
                    if rule_id in self.active_rules:
                        rule = self.active_rules[rule_id]
                        success = await self._deploy_rule_to_firewalls(rule)
                        
                        if success:
                            self.metrics['rules_deployed'] += 1
                            self.logger.info(f"Successfully deployed rule {rule_id}")
                        else:
                            self.logger.error(f"Failed to deploy rule {rule_id}")
                
                await asyncio.sleep(2)  # Deploy every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error in rule deployment: {e}")
                await asyncio.sleep(5)
    
    async def _deploy_rule_to_firewalls(self, rule: FirewallRule) -> bool:
        """Deploy rule to firewall systems"""
        try:
            deployment_success = True
            
            for firewall_system in self.firewall_systems:
                # Simulate firewall deployment
                # In real implementation, this would use firewall APIs
                deployment_result = await self._simulate_firewall_deployment(firewall_system, rule)
                
                if not deployment_result:
                    deployment_success = False
                    self.logger.error(f"Failed to deploy rule {rule.rule_id} to {firewall_system}")
            
            if deployment_success:
                # Send deployment notification
                await self._send_deployment_notification(rule)
            
            return deployment_success
            
        except Exception as e:
            self.logger.error(f"Error deploying rule to firewalls: {e}")
            return False
    
    async def _simulate_firewall_deployment(self, firewall_system: str, rule: FirewallRule) -> bool:
        """Simulate firewall rule deployment"""
        try:
            # Simulate deployment delay and potential failures
            await asyncio.sleep(0.5)
            
            # 95% success rate for simulation
            import random
            return random.random() > 0.05
            
        except Exception as e:
            self.logger.error(f"Error in simulated deployment: {e}")
            return False
    
    async def _send_deployment_notification(self, rule: FirewallRule) -> None:
        """Send rule deployment notification"""
        try:
            notification = Message(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id="broadcast",
                message_type=MessageType.RULE_DEPLOYED,
                content={
                    'rule_id': rule.rule_id,
                    'rule_name': rule.name,
                    'action': rule.action.value,
                    'target': f"{rule.source_ip}:{rule.source_port} -> {rule.destination_ip}:{rule.destination_port}",
                    'firewall_systems': self.firewall_systems,
                    'deployed_at': datetime.now().isoformat()
                },
                timestamp=datetime.now()
            )
            
            await self.communication_bus.send_message(notification)
            
        except Exception as e:
            self.logger.error(f"Error sending deployment notification: {e}")
    
    async def _cleanup_expired_rules(self) -> None:
        """Clean up expired firewall rules"""
        try:
            current_time = datetime.now()
            expired_rules = []
            
            for rule_id, rule in self.active_rules.items():
                if rule.expires_at and current_time >= rule.expires_at:
                    expired_rules.append(rule_id)
            
            for rule_id in expired_rules:
                rule = self.active_rules[rule_id]
                rule.active = False
                
                # Remove from firewall systems
                await self._remove_rule_from_firewalls(rule)
                
                self.metrics['rules_expired'] += 1
                self.logger.info(f"Expired rule {rule_id}: {rule.name}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired rules: {e}")
    
    async def _remove_rule_from_firewalls(self, rule: FirewallRule) -> None:
        """Remove rule from firewall systems"""
        try:
            for firewall_system in self.firewall_systems:
                # Simulate rule removal
                await self._simulate_firewall_removal(firewall_system, rule)
            
            # Send removal notification
            removal_notification = Message(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id="broadcast",
                message_type=MessageType.RULE_REMOVED,
                content={
                    'rule_id': rule.rule_id,
                    'rule_name': rule.name,
                    'reason': 'expired',
                    'removed_at': datetime.now().isoformat()
                },
                timestamp=datetime.now()
            )
            
            await self.communication_bus.send_message(removal_notification)
            
        except Exception as e:
            self.logger.error(f"Error removing rule from firewalls: {e}")
    
    async def _simulate_firewall_removal(self, firewall_system: str, rule: FirewallRule) -> bool:
        """Simulate firewall rule removal"""
        try:
            await asyncio.sleep(0.3)
            return True
            
        except Exception as e:
            self.logger.error(f"Error in simulated removal: {e}")
            return False
    
    async def _optimize_rules(self) -> None:
        """Optimize firewall rules for performance"""
        try:
            # Simple optimization: consolidate similar rules
            rule_groups = self._group_similar_rules()
            
            for group in rule_groups:
                if len(group) > 3:  # Consolidate if more than 3 similar rules
                    consolidated_rule = await self._consolidate_rules(group)
                    if consolidated_rule:
                        # Deactivate original rules
                        for rule in group:
                            rule.active = False
                        
                        # Add consolidated rule
                        self.active_rules[consolidated_rule.rule_id] = consolidated_rule
                        self.deployment_queue.append(consolidated_rule.rule_id)
            
        except Exception as e:
            self.logger.error(f"Error optimizing rules: {e}")
    
    def _group_similar_rules(self) -> List[List[FirewallRule]]:
        """Group similar firewall rules for optimization"""
        try:
            groups = []
            processed_rules = set()
            
            for rule1 in self.active_rules.values():
                if not rule1.active or rule1.rule_id in processed_rules:
                    continue
                
                similar_group = [rule1]
                processed_rules.add(rule1.rule_id)
                
                for rule2 in self.active_rules.values():
                    if (not rule2.active or rule2.rule_id in processed_rules or 
                        rule1.rule_id == rule2.rule_id):
                        continue
                    
                    # Check similarity
                    if (rule1.action == rule2.action and 
                        rule1.protocol == rule2.protocol and
                        rule1.direction == rule2.direction and
                        rule1.destination_port == rule2.destination_port):
                        similar_group.append(rule2)
                        processed_rules.add(rule2.rule_id)
                
                if len(similar_group) > 1:
                    groups.append(similar_group)
            
            return groups
            
        except Exception as e:
            self.logger.error(f"Error grouping similar rules: {e}")
            return []
    
    async def _consolidate_rules(self, rules: List[FirewallRule]) -> Optional[FirewallRule]:
        """Consolidate multiple similar rules into one"""
        try:
            if not rules:
                return None
            
            # Use first rule as template
            template = rules[0]
            
            # Collect all source IPs
            source_ips = set()
            for rule in rules:
                if rule.source_ip != 'any':
                    source_ips.add(rule.source_ip)
            
            # Create consolidated rule
            consolidated_rule_id = f"CONS-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
            
            consolidated_rule = FirewallRule(
                rule_id=consolidated_rule_id,
                name=f"Consolidated: {template.name}",
                action=template.action,
                protocol=template.protocol,
                source_ip=','.join(source_ips) if source_ips else 'any',
                source_port=template.source_port,
                destination_ip=template.destination_ip,
                destination_port=template.destination_port,
                direction=template.direction,
                priority=template.priority,
                description=f"Consolidated from {len(rules)} similar rules",
                created_by=self.agent_id,
                created_at=datetime.now(),
                expires_at=max((r.expires_at for r in rules if r.expires_at), default=None),
                active=True
            )
            
            self.logger.info(f"Consolidated {len(rules)} rules into {consolidated_rule_id}")
            
            return consolidated_rule
            
        except Exception as e:
            self.logger.error(f"Error consolidating rules: {e}")
            return None
    
    async def make_decision(self, environment_context: EnvironmentContext) -> ActionTaken:
        """Make firewall configuration decision"""
        try:
            # Check for pending threats requiring firewall response
            if self.deployment_queue:
                pending_rule = self.active_rules.get(self.deployment_queue[0])
                if pending_rule:
                    action = ActionTaken(
                        action_id=str(uuid.uuid4()),
                        primary_action="deploy_firewall_rule",
                        action_type="defensive_configuration",
                        target=pending_rule.rule_id,
                        parameters={
                            'rule_name': pending_rule.name,
                            'action': pending_rule.action.value,
                            'target': f"{pending_rule.source_ip} -> {pending_rule.destination_ip}",
                            'priority': pending_rule.priority
                        },
                        confidence_score=0.8,
                        reasoning=f"Deploying firewall rule: {pending_rule.name}"
                    )
                    return action
            
            # Check for optimization opportunities
            active_rule_count = len([r for r in self.active_rules.values() if r.active])
            if active_rule_count > 100:  # Too many rules
                action = ActionTaken(
                    action_id=str(uuid.uuid4()),
                    primary_action="optimize_firewall_rules",
                    action_type="system_optimization",
                    target="firewall_ruleset",
                    parameters={
                        'active_rules': active_rule_count,
                        'optimization_type': 'consolidation'
                    },
                    confidence_score=0.7,
                    reasoning="Optimizing firewall rules for performance"
                )
                return action
            
            # Default monitoring action
            action = ActionTaken(
                action_id=str(uuid.uuid4()),
                primary_action="monitor_firewall_rules",
                action_type="monitoring",
                target="firewall_systems",
                parameters={
                    'active_rules': active_rule_count,
                    'systems': self.firewall_systems
                },
                confidence_score=0.6,
                reasoning="Monitoring firewall rule status and performance"
            )
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error in firewall decision making: {e}")
            
            return ActionTaken(
                action_id=str(uuid.uuid4()),
                primary_action="error_recovery",
                action_type="system_maintenance",
                target="self",
                parameters={'error': str(e)},
                confidence_score=0.3,
                reasoning=f"Error recovery due to: {e}"
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get firewall configurator performance metrics"""
        active_rule_count = len([r for r in self.active_rules.values() if r.active])
        
        return {
            'agent_id': self.agent_id,
            'active_rules': active_rule_count,
            'total_rules': len(self.active_rules),
            'pending_deployments': len(self.deployment_queue),
            'firewall_systems': len(self.firewall_systems),
            **self.metrics
        }
    
    async def shutdown(self) -> None:
        """Shutdown the Firewall Configurator agent"""
        try:
            if hasattr(self, 'rule_manager_task'):
                self.rule_manager_task.cancel()
                try:
                    await self.rule_manager_task
                except asyncio.CancelledError:
                    pass
            
            if hasattr(self, 'deployment_task'):
                self.deployment_task.cancel()
                try:
                    await self.deployment_task
                except asyncio.CancelledError:
                    pass
            
            await super().shutdown()
            self.logger.info(f"Firewall Configurator Agent {self.agent_id} shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during firewall configurator shutdown: {e}")

class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ThreatLevel(Enum):
    """Threat intelligence levels"""
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Log entry data structure"""
    log_id: str
    timestamp: datetime
    source: str
    log_level: LogLevel
    message: str
    raw_data: Dict[str, Any]
    parsed_fields: Dict[str, Any]
    correlation_id: Optional[str]
    threat_indicators: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'log_id': self.log_id,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'log_level': self.log_level.value,
            'message': self.message,
            'raw_data': self.raw_data,
            'parsed_fields': self.parsed_fields,
            'correlation_id': self.correlation_id,
            'threat_indicators': self.threat_indicators
        }


@dataclass
class ThreatIntelligence:
    """Threat intelligence data structure"""
    intel_id: str
    source: str
    intel_type: str
    threat_level: ThreatLevel
    indicators: Dict[str, List[str]]
    description: str
    confidence_score: float
    created_at: datetime
    expires_at: Optional[datetime]
    mitre_mapping: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'intel_id': self.intel_id,
            'source': self.source,
            'intel_type': self.intel_type,
            'threat_level': self.threat_level.value,
            'indicators': self.indicators,
            'description': self.description,
            'confidence_score': self.confidence_score,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'mitre_mapping': self.mitre_mapping
        }


@dataclass
class CorrelationResult:
    """Log correlation result"""
    correlation_id: str
    related_logs: List[str]
    correlation_type: str
    confidence_score: float
    threat_level: ThreatLevel
    timeline: List[Dict[str, Any]]
    indicators: List[str]
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'correlation_id': self.correlation_id,
            'related_logs': self.related_logs,
            'correlation_type': self.correlation_type,
            'confidence_score': self.confidence_score,
            'threat_level': self.threat_level.value,
            'timeline': self.timeline,
            'indicators': self.indicators,
            'description': self.description
        }


class SIEMIntegratorAgent(Agent):
    """
    SIEM Integrator Agent for log correlation and threat intelligence analysis.
    
    Capabilities:
    - Real-time log ingestion and parsing
    - Advanced log correlation and pattern detection
    - Threat intelligence integration and enrichment
    - Automated threat hunting and analysis
    - Timeline reconstruction and forensic analysis
    - Custom rule creation and management
    """
    
    def __init__(self, 
                 agent_id: str,
                 communication_bus: CommunicationBus,
                 memory_system: VectorMemorySystem,
                 knowledge_base: KnowledgeBase,
                 log_sources: List[str] = None,
                 threat_intel_feeds: List[str] = None):
        super().__init__(
            agent_id=agent_id,
            team=Team.BLUE,
            role=Role.SIEM_ANALYST,
            communication_bus=communication_bus,
            memory_system=memory_system
        )
        
        self.knowledge_base = knowledge_base
        self.log_sources = log_sources or [
            "windows_events", "linux_syslog", "firewall_logs", 
            "web_server_logs", "database_logs", "network_logs"
        ]
        self.threat_intel_feeds = threat_intel_feeds or [
            "misp", "otx", "virustotal", "internal_intel"
        ]
        
        # Log management
        self.log_buffer: Dict[str, LogEntry] = {}
        self.correlation_results: Dict[str, CorrelationResult] = {}
        self.threat_intelligence: Dict[str, ThreatIntelligence] = {}
        
        # Correlation rules and patterns
        self.correlation_rules: List[Dict[str, Any]] = []
        self.threat_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.metrics = {
            'logs_processed': 0,
            'correlations_found': 0,
            'threats_detected': 0,
            'intel_enrichments': 0,
            'false_positives': 0,
            'processing_time_avg': 0.0
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """Initialize the SIEM Integrator agent"""
        await super().initialize()
        
        # Load correlation rules and threat patterns
        await self._load_correlation_rules()
        await self._load_threat_patterns()
        
        # Start processing tasks
        self.log_processor_task = asyncio.create_task(self._process_logs())
        self.correlation_task = asyncio.create_task(self._correlate_logs())
        self.intel_updater_task = asyncio.create_task(self._update_threat_intelligence())
        
        self.logger.info(f"SIEM Integrator Agent {self.agent_id} initialized")
    
    async def _load_correlation_rules(self) -> None:
        """Load log correlation rules"""
        self.correlation_rules = [
            {
                'name': 'Failed Login Sequence',
                'pattern': [
                    {'event_type': 'failed_login', 'timeframe': timedelta(minutes=5)},
                    {'event_type': 'failed_login', 'same_user': True},
                    {'event_type': 'failed_login', 'same_user': True}
                ],
                'threat_level': ThreatLevel.MEDIUM,
                'description': 'Multiple failed login attempts from same user'
            },
            {
                'name': 'Lateral Movement Pattern',
                'pattern': [
                    {'event_type': 'successful_login', 'timeframe': timedelta(minutes=10)},
                    {'event_type': 'process_creation', 'same_host': False},
                    {'event_type': 'network_connection', 'internal_target': True}
                ],
                'threat_level': ThreatLevel.HIGH,
                'description': 'Potential lateral movement activity detected'
            },
            {
                'name': 'Data Exfiltration Pattern',
                'pattern': [
                    {'event_type': 'file_access', 'timeframe': timedelta(minutes=15)},
                    {'event_type': 'data_compression', 'large_files': True},
                    {'event_type': 'network_upload', 'external_destination': True}
                ],
                'threat_level': ThreatLevel.CRITICAL,
                'description': 'Potential data exfiltration activity detected'
            },
            {
                'name': 'Privilege Escalation Pattern',
                'pattern': [
                    {'event_type': 'process_creation', 'timeframe': timedelta(minutes=5)},
                    {'event_type': 'privilege_change', 'elevation': True},
                    {'event_type': 'sensitive_file_access', 'admin_files': True}
                ],
                'threat_level': ThreatLevel.HIGH,
                'description': 'Potential privilege escalation detected'
            }
        ]
    
    async def _load_threat_patterns(self) -> None:
        """Load threat detection patterns"""
        self.threat_patterns = {
            'malware_execution': {
                'indicators': ['suspicious_process', 'unsigned_binary', 'network_callback'],
                'confidence_threshold': 0.7,
                'threat_level': ThreatLevel.HIGH
            },
            'command_injection': {
                'indicators': ['web_request', 'command_execution', 'system_call'],
                'confidence_threshold': 0.8,
                'threat_level': ThreatLevel.HIGH
            },
            'sql_injection': {
                'indicators': ['database_error', 'suspicious_query', 'data_extraction'],
                'confidence_threshold': 0.75,
                'threat_level': ThreatLevel.MEDIUM
            },
            'insider_threat': {
                'indicators': ['unusual_access_pattern', 'off_hours_activity', 'data_download'],
                'confidence_threshold': 0.6,
                'threat_level': ThreatLevel.MEDIUM
            }
        }
    
    async def _process_logs(self) -> None:
        """Process incoming log entries"""
        while self.active:
            try:
                # Get messages from communication bus
                messages = await self.communication_bus.get_messages(self.agent_id)
                
                for message in messages:
                    if message.message_type == MessageType.LOG_ENTRY:
                        await self._process_log_entry(message)
                    elif message.message_type == MessageType.THREAT_INTELLIGENCE:
                        await self._process_threat_intelligence(message)
                
                # Clean up old logs
                await self._cleanup_old_logs()
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                self.logger.error(f"Error in log processing: {e}")
                await asyncio.sleep(5)
    
    async def _process_log_entry(self, message: Message) -> None:
        """Process individual log entry"""
        try:
            start_time = datetime.now()
            
            log_data = message.content
            
            # Create LogEntry object
            log_entry = LogEntry(
                log_id=log_data.get('log_id', str(uuid.uuid4())),
                timestamp=datetime.fromisoformat(log_data.get('timestamp', datetime.now().isoformat())),
                source=log_data.get('source', 'unknown'),
                log_level=LogLevel(log_data.get('log_level', 'info')),
                message=log_data.get('message', ''),
                raw_data=log_data.get('raw_data', {}),
                parsed_fields=log_data.get('parsed_fields', {}),
                correlation_id=None,
                threat_indicators=[]
            )
            
            # Parse and enrich log entry
            await self._parse_log_entry(log_entry)
            await self._enrich_with_threat_intelligence(log_entry)
            
            # Store log entry
            self.log_buffer[log_entry.log_id] = log_entry
            
            # Update metrics
            self.metrics['logs_processed'] += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics['processing_time_avg'] = (
                (self.metrics['processing_time_avg'] * (self.metrics['logs_processed'] - 1) + processing_time) /
                self.metrics['logs_processed']
            )
            
            self.logger.debug(f"Processed log entry {log_entry.log_id} from {log_entry.source}")
            
        except Exception as e:
            self.logger.error(f"Error processing log entry: {e}")
    
    async def _parse_log_entry(self, log_entry: LogEntry) -> None:
        """Parse log entry to extract structured fields"""
        try:
            # Basic parsing based on log source
            if log_entry.source == 'windows_events':
                await self._parse_windows_event(log_entry)
            elif log_entry.source == 'linux_syslog':
                await self._parse_linux_syslog(log_entry)
            elif log_entry.source == 'firewall_logs':
                await self._parse_firewall_log(log_entry)
            elif log_entry.source == 'web_server_logs':
                await self._parse_web_server_log(log_entry)
            
            # Extract common threat indicators
            await self._extract_threat_indicators(log_entry)
            
        except Exception as e:
            self.logger.error(f"Error parsing log entry: {e}")
    
    async def _parse_windows_event(self, log_entry: LogEntry) -> None:
        """Parse Windows event log"""
        try:
            raw_data = log_entry.raw_data
            
            # Extract common Windows event fields
            log_entry.parsed_fields.update({
                'event_id': raw_data.get('EventID'),
                'computer_name': raw_data.get('Computer'),
                'user_name': raw_data.get('UserName'),
                'process_name': raw_data.get('ProcessName'),
                'source_ip': raw_data.get('SourceIP'),
                'target_user': raw_data.get('TargetUserName')
            })
            
            # Identify event types
            event_id = raw_data.get('EventID')
            if event_id == 4624:
                log_entry.parsed_fields['event_type'] = 'successful_login'
            elif event_id == 4625:
                log_entry.parsed_fields['event_type'] = 'failed_login'
            elif event_id == 4688:
                log_entry.parsed_fields['event_type'] = 'process_creation'
            elif event_id == 4648:
                log_entry.parsed_fields['event_type'] = 'explicit_logon'
            
        except Exception as e:
            self.logger.error(f"Error parsing Windows event: {e}")
    
    async def _parse_linux_syslog(self, log_entry: LogEntry) -> None:
        """Parse Linux syslog entry"""
        try:
            message = log_entry.message
            
            # Extract common syslog fields
            if 'sshd' in message:
                if 'Failed password' in message:
                    log_entry.parsed_fields['event_type'] = 'failed_login'
                elif 'Accepted password' in message:
                    log_entry.parsed_fields['event_type'] = 'successful_login'
                
                # Extract IP address
                import re
                ip_match = re.search(r'from (\d+\.\d+\.\d+\.\d+)', message)
                if ip_match:
                    log_entry.parsed_fields['source_ip'] = ip_match.group(1)
            
            elif 'sudo' in message:
                log_entry.parsed_fields['event_type'] = 'privilege_escalation'
                
                # Extract user and command
                user_match = re.search(r'(\w+) : TTY=', message)
                if user_match:
                    log_entry.parsed_fields['user_name'] = user_match.group(1)
            
        except Exception as e:
            self.logger.error(f"Error parsing Linux syslog: {e}")
    
    async def _parse_firewall_log(self, log_entry: LogEntry) -> None:
        """Parse firewall log entry"""
        try:
            raw_data = log_entry.raw_data
            
            log_entry.parsed_fields.update({
                'action': raw_data.get('action'),
                'source_ip': raw_data.get('src_ip'),
                'destination_ip': raw_data.get('dst_ip'),
                'source_port': raw_data.get('src_port'),
                'destination_port': raw_data.get('dst_port'),
                'protocol': raw_data.get('protocol'),
                'bytes_transferred': raw_data.get('bytes')
            })
            
            # Determine event type
            action = raw_data.get('action', '').lower()
            if action in ['block', 'deny', 'drop']:
                log_entry.parsed_fields['event_type'] = 'blocked_connection'
            elif action in ['allow', 'accept']:
                log_entry.parsed_fields['event_type'] = 'allowed_connection'
            
        except Exception as e:
            self.logger.error(f"Error parsing firewall log: {e}")
    
    async def _parse_web_server_log(self, log_entry: LogEntry) -> None:
        """Parse web server log entry"""
        try:
            raw_data = log_entry.raw_data
            
            log_entry.parsed_fields.update({
                'client_ip': raw_data.get('client_ip'),
                'method': raw_data.get('method'),
                'url': raw_data.get('url'),
                'status_code': raw_data.get('status_code'),
                'user_agent': raw_data.get('user_agent'),
                'referer': raw_data.get('referer'),
                'response_size': raw_data.get('response_size')
            })
            
            # Identify suspicious patterns
            url = raw_data.get('url', '')
            status_code = raw_data.get('status_code', 0)
            
            if status_code >= 400:
                log_entry.parsed_fields['event_type'] = 'web_error'
            elif any(pattern in url.lower() for pattern in ['admin', 'login', 'auth']):
                log_entry.parsed_fields['event_type'] = 'web_authentication'
            else:
                log_entry.parsed_fields['event_type'] = 'web_request'
            
        except Exception as e:
            self.logger.error(f"Error parsing web server log: {e}")
    
    async def _extract_threat_indicators(self, log_entry: LogEntry) -> None:
        """Extract threat indicators from log entry"""
        try:
            indicators = []
            
            # Check for suspicious IPs
            source_ip = log_entry.parsed_fields.get('source_ip')
            if source_ip and await self._is_suspicious_ip(source_ip):
                indicators.append(f"suspicious_ip:{source_ip}")
            
            # Check for suspicious processes
            process_name = log_entry.parsed_fields.get('process_name')
            if process_name and await self._is_suspicious_process(process_name):
                indicators.append(f"suspicious_process:{process_name}")
            
            # Check for suspicious URLs
            url = log_entry.parsed_fields.get('url')
            if url and await self._is_suspicious_url(url):
                indicators.append(f"suspicious_url:{url}")
            
            # Check for failed authentication patterns
            event_type = log_entry.parsed_fields.get('event_type')
            if event_type == 'failed_login':
                indicators.append("failed_authentication")
            
            log_entry.threat_indicators = indicators
            
        except Exception as e:
            self.logger.error(f"Error extracting threat indicators: {e}")
    
    async def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious"""
        try:
            # Check against threat intelligence
            for intel in self.threat_intelligence.values():
                if ip_address in intel.indicators.get('ip_addresses', []):
                    return True
            
            # Check for private IP ranges (simplified)
            if ip_address.startswith(('10.', '192.168.', '172.')):
                return False  # Internal IPs are generally not suspicious
            
            # Additional checks could include:
            # - Geolocation analysis
            # - Reputation databases
            # - Historical behavior analysis
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking suspicious IP: {e}")
            return False
    
    async def _is_suspicious_process(self, process_name: str) -> bool:
        """Check if process name is suspicious"""
        try:
            suspicious_processes = [
                'powershell.exe', 'cmd.exe', 'wscript.exe', 'cscript.exe',
                'regsvr32.exe', 'rundll32.exe', 'mshta.exe'
            ]
            
            return process_name.lower() in suspicious_processes
            
        except Exception as e:
            self.logger.error(f"Error checking suspicious process: {e}")
            return False
    
    async def _is_suspicious_url(self, url: str) -> bool:
        """Check if URL is suspicious"""
        try:
            suspicious_patterns = [
                'admin', 'login', 'auth', 'config', 'backup',
                'shell', 'cmd', 'exec', 'eval', 'system'
            ]
            
            url_lower = url.lower()
            return any(pattern in url_lower for pattern in suspicious_patterns)
            
        except Exception as e:
            self.logger.error(f"Error checking suspicious URL: {e}")
            return False
    
    async def _enrich_with_threat_intelligence(self, log_entry: LogEntry) -> None:
        """Enrich log entry with threat intelligence"""
        try:
            enrichments = 0
            
            # Check indicators against threat intelligence
            for intel in self.threat_intelligence.values():
                # Check IP addresses
                source_ip = log_entry.parsed_fields.get('source_ip')
                if source_ip and source_ip in intel.indicators.get('ip_addresses', []):
                    log_entry.threat_indicators.append(f"threat_intel_ip:{intel.intel_id}")
                    enrichments += 1
                
                # Check domains
                url = log_entry.parsed_fields.get('url', '')
                for domain in intel.indicators.get('domains', []):
                    if domain in url:
                        log_entry.threat_indicators.append(f"threat_intel_domain:{intel.intel_id}")
                        enrichments += 1
                
                # Check file hashes
                file_hash = log_entry.parsed_fields.get('file_hash')
                if file_hash and file_hash in intel.indicators.get('file_hashes', []):
                    log_entry.threat_indicators.append(f"threat_intel_hash:{intel.intel_id}")
                    enrichments += 1
            
            if enrichments > 0:
                self.metrics['intel_enrichments'] += enrichments
                self.logger.debug(f"Enriched log {log_entry.log_id} with {enrichments} threat intel matches")
            
        except Exception as e:
            self.logger.error(f"Error enriching with threat intelligence: {e}")
    
    async def _correlate_logs(self) -> None:
        """Perform log correlation analysis"""
        while self.active:
            try:
                # Run correlation rules
                for rule in self.correlation_rules:
                    correlations = await self._find_correlations(rule)
                    
                    for correlation in correlations:
                        await self._create_correlation_result(rule, correlation)
                
                # Clean up old correlations
                await self._cleanup_old_correlations()
                
                await asyncio.sleep(10)  # Correlate every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in log correlation: {e}")
                await asyncio.sleep(30)
    
    async def _find_correlations(self, rule: Dict[str, Any]) -> List[List[LogEntry]]:
        """Find log correlations based on rule"""
        try:
            correlations = []
            pattern = rule['pattern']
            
            if not pattern:
                return correlations
            
            # Get recent logs within timeframe
            timeframe = pattern[0].get('timeframe', timedelta(minutes=10))
            current_time = datetime.now()
            
            recent_logs = [
                log for log in self.log_buffer.values()
                if current_time - log.timestamp <= timeframe
            ]
            
            # Sort by timestamp
            recent_logs.sort(key=lambda x: x.timestamp)
            
            # Look for pattern matches
            for i, log1 in enumerate(recent_logs):
                if not self._matches_pattern_step(log1, pattern[0]):
                    continue
                
                correlation_group = [log1]
                last_log = log1
                
                # Try to match subsequent pattern steps
                for step_idx in range(1, len(pattern)):
                    step_pattern = pattern[step_idx]
                    
                    # Look for matching log within timeframe
                    for j in range(i + 1, len(recent_logs)):
                        log2 = recent_logs[j]
                        
                        if (log2.timestamp - last_log.timestamp <= step_pattern.get('timeframe', timedelta(minutes=5)) and
                            self._matches_pattern_step(log2, step_pattern) and
                            self._matches_correlation_constraints(last_log, log2, step_pattern)):
                            
                            correlation_group.append(log2)
                            last_log = log2
                            break
                
                # Check if we found a complete pattern
                if len(correlation_group) == len(pattern):
                    correlations.append(correlation_group)
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error finding correlations: {e}")
            return []
    
    def _matches_pattern_step(self, log_entry: LogEntry, pattern_step: Dict[str, Any]) -> bool:
        """Check if log entry matches pattern step"""
        try:
            event_type = pattern_step.get('event_type')
            if event_type and log_entry.parsed_fields.get('event_type') != event_type:
                return False
            
            # Additional pattern matching logic could be added here
            return True
            
        except Exception as e:
            self.logger.error(f"Error matching pattern step: {e}")
            return False
    
    def _matches_correlation_constraints(self, log1: LogEntry, log2: LogEntry, pattern_step: Dict[str, Any]) -> bool:
        """Check if logs match correlation constraints"""
        try:
            # Same user constraint
            if pattern_step.get('same_user'):
                user1 = log1.parsed_fields.get('user_name')
                user2 = log2.parsed_fields.get('user_name')
                if user1 != user2:
                    return False
            
            # Same host constraint
            if pattern_step.get('same_host'):
                host1 = log1.parsed_fields.get('computer_name')
                host2 = log2.parsed_fields.get('computer_name')
                if host1 != host2:
                    return False
            
            # Different host constraint
            if pattern_step.get('same_host') is False:
                host1 = log1.parsed_fields.get('computer_name')
                host2 = log2.parsed_fields.get('computer_name')
                if host1 == host2:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking correlation constraints: {e}")
            return False
    
    async def _create_correlation_result(self, rule: Dict[str, Any], correlation_logs: List[LogEntry]) -> None:
        """Create correlation result from matched logs"""
        try:
            correlation_id = f"CORR-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
            
            # Build timeline
            timeline = []
            for log in correlation_logs:
                timeline.append({
                    'timestamp': log.timestamp.isoformat(),
                    'log_id': log.log_id,
                    'source': log.source,
                    'event_type': log.parsed_fields.get('event_type', 'unknown'),
                    'description': log.message[:100]  # Truncate for timeline
                })
            
            # Collect all indicators
            all_indicators = set()
            for log in correlation_logs:
                all_indicators.update(log.threat_indicators)
            
            # Calculate confidence score
            confidence_score = self._calculate_correlation_confidence(rule, correlation_logs)
            
            # Create correlation result
            correlation_result = CorrelationResult(
                correlation_id=correlation_id,
                related_logs=[log.log_id for log in correlation_logs],
                correlation_type=rule['name'],
                confidence_score=confidence_score,
                threat_level=rule['threat_level'],
                timeline=timeline,
                indicators=list(all_indicators),
                description=rule['description']
            )
            
            # Store correlation result
            self.correlation_results[correlation_id] = correlation_result
            
            # Update correlation IDs in logs
            for log in correlation_logs:
                log.correlation_id = correlation_id
            
            # Send correlation alert
            await self._send_correlation_alert(correlation_result)
            
            # Update metrics
            self.metrics['correlations_found'] += 1
            if correlation_result.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self.metrics['threats_detected'] += 1
            
            self.logger.info(f"Created correlation {correlation_id}: {rule['name']}")
            
        except Exception as e:
            self.logger.error(f"Error creating correlation result: {e}")
    
    def _calculate_correlation_confidence(self, rule: Dict[str, Any], logs: List[LogEntry]) -> float:
        """Calculate confidence score for correlation"""
        try:
            base_confidence = 0.7  # Base confidence for pattern match
            
            # Boost confidence for threat indicators
            total_indicators = sum(len(log.threat_indicators) for log in logs)
            indicator_boost = min(total_indicators * 0.05, 0.2)
            
            # Boost confidence for tight timing
            if len(logs) > 1:
                time_span = (logs[-1].timestamp - logs[0].timestamp).total_seconds()
                if time_span < 60:  # Very tight timing
                    timing_boost = 0.1
                elif time_span < 300:  # Moderate timing
                    timing_boost = 0.05
                else:
                    timing_boost = 0.0
            else:
                timing_boost = 0.0
            
            confidence = base_confidence + indicator_boost + timing_boost
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation confidence: {e}")
            return 0.5
    
    async def _send_correlation_alert(self, correlation_result: CorrelationResult) -> None:
        """Send correlation alert"""
        try:
            alert_message = Message(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id="broadcast",
                message_type=MessageType.CORRELATION_ALERT,
                content={
                    'correlation_id': correlation_result.correlation_id,
                    'correlation_type': correlation_result.correlation_type,
                    'threat_level': correlation_result.threat_level.value,
                    'confidence_score': correlation_result.confidence_score,
                    'related_logs': correlation_result.related_logs,
                    'timeline': correlation_result.timeline,
                    'indicators': correlation_result.indicators,
                    'description': correlation_result.description
                },
                timestamp=datetime.now()
            )
            
            await self.communication_bus.send_message(alert_message)
            
            self.logger.info(f"Sent correlation alert for {correlation_result.correlation_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending correlation alert: {e}")
    
    async def make_decision(self, environment_context: EnvironmentContext) -> ActionTaken:
        """Make SIEM analysis decision"""
        try:
            # Check for high-priority correlations
            high_priority_correlations = [
                corr for corr in self.correlation_results.values()
                if corr.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            ]
            
            if high_priority_correlations:
                # Focus on highest priority correlation
                correlation = max(high_priority_correlations, 
                                key=lambda x: (x.threat_level.value, x.confidence_score))
                
                action = ActionTaken(
                    action_id=str(uuid.uuid4()),
                    primary_action="investigate_correlation",
                    action_type="threat_analysis",
                    target=correlation.correlation_id,
                    parameters={
                        'correlation_type': correlation.correlation_type,
                        'threat_level': correlation.threat_level.value,
                        'confidence': correlation.confidence_score,
                        'indicators': correlation.indicators
                    },
                    confidence_score=correlation.confidence_score,
                    reasoning=f"Investigating {correlation.threat_level.value} threat correlation: {correlation.correlation_type}"
                )
            else:
                # Proactive log analysis
                recent_log_count = len([
                    log for log in self.log_buffer.values()
                    if datetime.now() - log.timestamp <= timedelta(minutes=10)
                ])
                
                action = ActionTaken(
                    action_id=str(uuid.uuid4()),
                    primary_action="analyze_logs",
                    action_type="monitoring",
                    target="log_stream",
                    parameters={
                        'recent_logs': recent_log_count,
                        'log_sources': self.log_sources,
                        'correlation_rules': len(self.correlation_rules)
                    },
                    confidence_score=0.6,
                    reasoning="Proactive log analysis and correlation monitoring"
                )
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error in SIEM decision making: {e}")
            
            return ActionTaken(
                action_id=str(uuid.uuid4()),
                primary_action="error_recovery",
                action_type="system_maintenance",
                target="self",
                parameters={'error': str(e)},
                confidence_score=0.3,
                reasoning=f"Error recovery due to: {e}"
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get SIEM integrator performance metrics"""
        return {
            'agent_id': self.agent_id,
            'log_buffer_size': len(self.log_buffer),
            'active_correlations': len(self.correlation_results),
            'threat_intel_feeds': len(self.threat_intelligence),
            'log_sources': len(self.log_sources),
            **self.metrics
        }
    
    async def shutdown(self) -> None:
        """Shutdown the SIEM Integrator agent"""
        try:
            # Cancel all tasks
            tasks = ['log_processor_task', 'correlation_task', 'intel_updater_task']
            for task_name in tasks:
                if hasattr(self, task_name):
                    task = getattr(self, task_name)
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            await super().shutdown()
            self.logger.info(f"SIEM Integrator Agent {self.agent_id} shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during SIEM integrator shutdown: {e}")
    
    # Additional helper methods for threat intelligence management
    async def _update_threat_intelligence(self) -> None:
        """Update threat intelligence feeds"""
        while self.active:
            try:
                # Simulate threat intelligence updates
                # In real implementation, this would fetch from actual feeds
                await self._fetch_threat_intelligence_updates()
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error updating threat intelligence: {e}")
                await asyncio.sleep(600)  # Retry after 10 minutes
    
    async def _fetch_threat_intelligence_updates(self) -> None:
        """Fetch threat intelligence updates from feeds"""
        try:
            # Simulate fetching updates
            # This would integrate with real threat intel feeds
            pass
            
        except Exception as e:
            self.logger.error(f"Error fetching threat intelligence: {e}")
    
    async def _cleanup_old_logs(self) -> None:
        """Clean up old log entries"""
        try:
            current_time = datetime.now()
            retention_period = timedelta(hours=24)
            
            old_logs = [
                log_id for log_id, log in self.log_buffer.items()
                if current_time - log.timestamp > retention_period
            ]
            
            for log_id in old_logs:
                del self.log_buffer[log_id]
            
            if old_logs:
                self.logger.debug(f"Cleaned up {len(old_logs)} old log entries")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old logs: {e}")
    
    async def _cleanup_old_correlations(self) -> None:
        """Clean up old correlation results"""
        try:
            current_time = datetime.now()
            retention_period = timedelta(days=7)
            
            old_correlations = [
                corr_id for corr_id, corr in self.correlation_results.items()
                if current_time - corr.timeline[0]['timestamp'] > retention_period
            ]
            
            for corr_id in old_correlations:
                del self.correlation_results[corr_id]
            
            if old_correlations:
                self.logger.debug(f"Cleaned up {len(old_correlations)} old correlations")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old correlations: {e}")
class Co
mplianceFramework(Enum):
    """Compliance frameworks"""
    NIST_CSF = "nist_csf"
    ISO_27001 = "iso_27001"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    SOX = "sox"


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNKNOWN = "unknown"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class ComplianceControl:
    """Compliance control definition"""
    control_id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirements: List[str]
    evidence_types: List[str]
    automated_checks: List[str]
    manual_checks: List[str]
    risk_level: str
    frequency: str  # daily, weekly, monthly, quarterly, annually
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'control_id': self.control_id,
            'framework': self.framework.value,
            'title': self.title,
            'description': self.description,
            'requirements': self.requirements,
            'evidence_types': self.evidence_types,
            'automated_checks': self.automated_checks,
            'manual_checks': self.manual_checks,
            'risk_level': self.risk_level,
            'frequency': self.frequency
        }


@dataclass
class ComplianceAssessment:
    """Compliance assessment result"""
    assessment_id: str
    control_id: str
    framework: ComplianceFramework
    status: ComplianceStatus
    score: float  # 0.0 to 1.0
    findings: List[str]
    evidence: List[Dict[str, Any]]
    recommendations: List[str]
    assessed_by: str
    assessed_at: datetime
    next_assessment: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'assessment_id': self.assessment_id,
            'control_id': self.control_id,
            'framework': self.framework.value,
            'status': self.status.value,
            'score': self.score,
            'findings': self.findings,
            'evidence': self.evidence,
            'recommendations': self.recommendations,
            'assessed_by': self.assessed_by,
            'assessed_at': self.assessed_at.isoformat(),
            'next_assessment': self.next_assessment.isoformat()
        }


@dataclass
class ComplianceReport:
    """Compliance report"""
    report_id: str
    framework: ComplianceFramework
    report_type: str
    period_start: datetime
    period_end: datetime
    overall_score: float
    control_assessments: List[str]
    summary: Dict[str, Any]
    recommendations: List[str]
    generated_by: str
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'report_id': self.report_id,
            'framework': self.framework.value,
            'report_type': self.report_type,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'overall_score': self.overall_score,
            'control_assessments': self.control_assessments,
            'summary': self.summary,
            'recommendations': self.recommendations,
            'generated_by': self.generated_by,
            'generated_at': self.generated_at.isoformat()
        }


class ComplianceAuditorAgent(Agent):
    """
    Compliance Auditor Agent for policy alignment checking and reporting.
    
    Capabilities:
    - Automated compliance control assessment
    - Policy alignment verification
    - Evidence collection and validation
    - Compliance gap analysis
    - Automated reporting and dashboards
    - Risk assessment and prioritization
    - Remediation tracking and management
    """
    
    def __init__(self, 
                 agent_id: str,
                 communication_bus: CommunicationBus,
                 memory_system: VectorMemorySystem,
                 knowledge_base: KnowledgeBase,
                 frameworks: List[ComplianceFramework] = None):
        super().__init__(
            agent_id=agent_id,
            team=Team.BLUE,
            role=Role.COMPLIANCE_AUDITOR,
            communication_bus=communication_bus,
            memory_system=memory_system
        )
        
        self.knowledge_base = knowledge_base
        self.frameworks = frameworks or [
            ComplianceFramework.NIST_CSF,
            ComplianceFramework.ISO_27001,
            ComplianceFramework.SOC2
        ]
        
        # Compliance management
        self.compliance_controls: Dict[str, ComplianceControl] = {}
        self.assessments: Dict[str, ComplianceAssessment] = {}
        self.reports: Dict[str, ComplianceReport] = {}
        
        # Assessment scheduling
        self.assessment_schedule: Dict[str, datetime] = {}
        self.pending_assessments: List[str] = []
        
        # Performance metrics
        self.metrics = {
            'controls_assessed': 0,
            'compliance_violations': 0,
            'reports_generated': 0,
            'evidence_collected': 0,
            'recommendations_made': 0,
            'average_compliance_score': 0.0
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """Initialize the Compliance Auditor agent"""
        await super().initialize()
        
        # Load compliance controls
        await self._load_compliance_controls()
        
        # Initialize assessment schedule
        await self._initialize_assessment_schedule()
        
        # Start compliance monitoring tasks
        self.assessment_task = asyncio.create_task(self._run_assessments())
        self.monitoring_task = asyncio.create_task(self._monitor_compliance())
        self.reporting_task = asyncio.create_task(self._generate_reports())
        
        self.logger.info(f"Compliance Auditor Agent {self.agent_id} initialized")
    
    async def _load_compliance_controls(self) -> None:
        """Load compliance controls for configured frameworks"""
        try:
            # NIST Cybersecurity Framework controls
            if ComplianceFramework.NIST_CSF in self.frameworks:
                await self._load_nist_csf_controls()
            
            # ISO 27001 controls
            if ComplianceFramework.ISO_27001 in self.frameworks:
                await self._load_iso27001_controls()
            
            # SOC 2 controls
            if ComplianceFramework.SOC2 in self.frameworks:
                await self._load_soc2_controls()
            
            self.logger.info(f"Loaded {len(self.compliance_controls)} compliance controls")
            
        except Exception as e:
            self.logger.error(f"Error loading compliance controls: {e}")
    
    async def _load_nist_csf_controls(self) -> None:
        """Load NIST Cybersecurity Framework controls"""
        nist_controls = [
            {
                'control_id': 'ID.AM-1',
                'title': 'Physical devices and systems within the organization are inventoried',
                'description': 'Maintain an accurate inventory of physical devices and systems',
                'requirements': [
                    'Maintain asset inventory',
                    'Regular inventory updates',
                    'Asset classification'
                ],
                'evidence_types': ['asset_inventory', 'inventory_reports'],
                'automated_checks': ['asset_discovery_scan', 'inventory_validation'],
                'manual_checks': ['physical_verification'],
                'risk_level': 'medium',
                'frequency': 'monthly'
            },
            {
                'control_id': 'PR.AC-1',
                'title': 'Identities and credentials are issued, managed, verified, revoked, and audited',
                'description': 'Manage user identities and access credentials throughout their lifecycle',
                'requirements': [
                    'Identity management system',
                    'Credential lifecycle management',
                    'Regular access reviews'
                ],
                'evidence_types': ['access_logs', 'user_accounts', 'access_reviews'],
                'automated_checks': ['user_account_audit', 'access_review_validation'],
                'manual_checks': ['access_certification'],
                'risk_level': 'high',
                'frequency': 'monthly'
            },
            {
                'control_id': 'DE.CM-1',
                'title': 'The network is monitored to detect potential cybersecurity events',
                'description': 'Implement network monitoring to detect security events',
                'requirements': [
                    'Network monitoring tools',
                    'Security event detection',
                    'Alert management'
                ],
                'evidence_types': ['network_logs', 'security_alerts', 'monitoring_reports'],
                'automated_checks': ['network_monitoring_status', 'alert_validation'],
                'manual_checks': ['monitoring_effectiveness_review'],
                'risk_level': 'high',
                'frequency': 'daily'
            }
        ]
        
        for control_data in nist_controls:
            control = ComplianceControl(
                control_id=control_data['control_id'],
                framework=ComplianceFramework.NIST_CSF,
                title=control_data['title'],
                description=control_data['description'],
                requirements=control_data['requirements'],
                evidence_types=control_data['evidence_types'],
                automated_checks=control_data['automated_checks'],
                manual_checks=control_data['manual_checks'],
                risk_level=control_data['risk_level'],
                frequency=control_data['frequency']
            )
            self.compliance_controls[control.control_id] = control
    
    async def _load_iso27001_controls(self) -> None:
        """Load ISO 27001 controls"""
        iso_controls = [
            {
                'control_id': 'A.9.1.1',
                'title': 'Access control policy',
                'description': 'An access control policy shall be established, documented and reviewed',
                'requirements': [
                    'Documented access control policy',
                    'Regular policy reviews',
                    'Management approval'
                ],
                'evidence_types': ['policy_documents', 'review_records', 'approval_records'],
                'automated_checks': ['policy_version_check'],
                'manual_checks': ['policy_review', 'approval_verification'],
                'risk_level': 'medium',
                'frequency': 'annually'
            },
            {
                'control_id': 'A.12.6.1',
                'title': 'Management of technical vulnerabilities',
                'description': 'Information about technical vulnerabilities shall be obtained in a timely fashion',
                'requirements': [
                    'Vulnerability management process',
                    'Regular vulnerability scanning',
                    'Timely patching'
                ],
                'evidence_types': ['vulnerability_scans', 'patch_reports', 'remediation_records'],
                'automated_checks': ['vulnerability_scan_status', 'patch_compliance'],
                'manual_checks': ['vulnerability_assessment_review'],
                'risk_level': 'high',
                'frequency': 'weekly'
            }
        ]
        
        for control_data in iso_controls:
            control = ComplianceControl(
                control_id=control_data['control_id'],
                framework=ComplianceFramework.ISO_27001,
                title=control_data['title'],
                description=control_data['description'],
                requirements=control_data['requirements'],
                evidence_types=control_data['evidence_types'],
                automated_checks=control_data['automated_checks'],
                manual_checks=control_data['manual_checks'],
                risk_level=control_data['risk_level'],
                frequency=control_data['frequency']
            )
            self.compliance_controls[control.control_id] = control
    
    async def _load_soc2_controls(self) -> None:
        """Load SOC 2 controls"""
        soc2_controls = [
            {
                'control_id': 'CC6.1',
                'title': 'Logical and Physical Access Controls',
                'description': 'The entity implements logical and physical access controls',
                'requirements': [
                    'Access control implementation',
                    'Physical security measures',
                    'Logical access restrictions'
                ],
                'evidence_types': ['access_controls', 'physical_security', 'access_logs'],
                'automated_checks': ['access_control_validation', 'security_system_status'],
                'manual_checks': ['physical_security_inspection'],
                'risk_level': 'high',
                'frequency': 'monthly'
            },
            {
                'control_id': 'CC7.1',
                'title': 'System Monitoring',
                'description': 'The entity monitors system components and the operation of controls',
                'requirements': [
                    'System monitoring implementation',
                    'Control operation monitoring',
                    'Alert management'
                ],
                'evidence_types': ['monitoring_logs', 'system_alerts', 'control_reports'],
                'automated_checks': ['monitoring_system_status', 'alert_validation'],
                'manual_checks': ['monitoring_effectiveness_review'],
                'risk_level': 'medium',
                'frequency': 'daily'
            }
        ]
        
        for control_data in soc2_controls:
            control = ComplianceControl(
                control_id=control_data['control_id'],
                framework=ComplianceFramework.SOC2,
                title=control_data['title'],
                description=control_data['description'],
                requirements=control_data['requirements'],
                evidence_types=control_data['evidence_types'],
                automated_checks=control_data['automated_checks'],
                manual_checks=control_data['manual_checks'],
                risk_level=control_data['risk_level'],
                frequency=control_data['frequency']
            )
            self.compliance_controls[control.control_id] = control
    
    async def _initialize_assessment_schedule(self) -> None:
        """Initialize assessment schedule for all controls"""
        try:
            current_time = datetime.now()
            
            for control_id, control in self.compliance_controls.items():
                # Calculate next assessment time based on frequency
                if control.frequency == 'daily':
                    next_assessment = current_time + timedelta(days=1)
                elif control.frequency == 'weekly':
                    next_assessment = current_time + timedelta(weeks=1)
                elif control.frequency == 'monthly':
                    next_assessment = current_time + timedelta(days=30)
                elif control.frequency == 'quarterly':
                    next_assessment = current_time + timedelta(days=90)
                elif control.frequency == 'annually':
                    next_assessment = current_time + timedelta(days=365)
                else:
                    next_assessment = current_time + timedelta(days=30)  # Default to monthly
                
                self.assessment_schedule[control_id] = next_assessment
            
            self.logger.info(f"Initialized assessment schedule for {len(self.assessment_schedule)} controls")
            
        except Exception as e:
            self.logger.error(f"Error initializing assessment schedule: {e}")
    
    async def _run_assessments(self) -> None:
        """Run scheduled compliance assessments"""
        while self.active:
            try:
                current_time = datetime.now()
                
                # Check for due assessments
                due_assessments = [
                    control_id for control_id, next_time in self.assessment_schedule.items()
                    if current_time >= next_time
                ]
                
                # Run due assessments
                for control_id in due_assessments:
                    await self._assess_control(control_id)
                    
                    # Update next assessment time
                    control = self.compliance_controls[control_id]
                    if control.frequency == 'daily':
                        self.assessment_schedule[control_id] = current_time + timedelta(days=1)
                    elif control.frequency == 'weekly':
                        self.assessment_schedule[control_id] = current_time + timedelta(weeks=1)
                    elif control.frequency == 'monthly':
                        self.assessment_schedule[control_id] = current_time + timedelta(days=30)
                    elif control.frequency == 'quarterly':
                        self.assessment_schedule[control_id] = current_time + timedelta(days=90)
                    elif control.frequency == 'annually':
                        self.assessment_schedule[control_id] = current_time + timedelta(days=365)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in assessment runner: {e}")
                await asyncio.sleep(1800)  # Retry after 30 minutes
    
    async def _assess_control(self, control_id: str) -> str:
        """Assess a specific compliance control"""
        try:
            control = self.compliance_controls.get(control_id)
            if not control:
                raise ValueError(f"Control {control_id} not found")
            
            assessment_id = f"ASSESS-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
            
            # Run automated checks
            automated_results = await self._run_automated_checks(control)
            
            # Collect evidence
            evidence = await self._collect_evidence(control)
            
            # Calculate compliance score
            score, status = await self._calculate_compliance_score(control, automated_results, evidence)
            
            # Generate findings and recommendations
            findings = await self._generate_findings(control, automated_results, evidence)
            recommendations = await self._generate_recommendations(control, findings)
            
            # Create assessment
            assessment = ComplianceAssessment(
                assessment_id=assessment_id,
                control_id=control_id,
                framework=control.framework,
                status=status,
                score=score,
                findings=findings,
                evidence=evidence,
                recommendations=recommendations,
                assessed_by=self.agent_id,
                assessed_at=datetime.now(),
                next_assessment=self.assessment_schedule.get(control_id, datetime.now() + timedelta(days=30))
            )
            
            # Store assessment
            self.assessments[assessment_id] = assessment
            
            # Send assessment notification
            await self._send_assessment_notification(assessment)
            
            # Update metrics
            self.metrics['controls_assessed'] += 1
            if status == ComplianceStatus.NON_COMPLIANT:
                self.metrics['compliance_violations'] += 1
            
            # Update average compliance score
            total_score = sum(a.score for a in self.assessments.values())
            self.metrics['average_compliance_score'] = total_score / len(self.assessments)
            
            self.logger.info(f"Completed assessment {assessment_id} for control {control_id}: {status.value}")
            
            return assessment_id
            
        except Exception as e:
            self.logger.error(f"Error assessing control {control_id}: {e}")
            raise
    
    async def _run_automated_checks(self, control: ComplianceControl) -> Dict[str, Any]:
        """Run automated checks for a control"""
        try:
            results = {}
            
            for check in control.automated_checks:
                # Simulate automated check execution
                # In real implementation, these would be actual system checks
                result = await self._execute_automated_check(check)
                results[check] = result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running automated checks: {e}")
            return {}
    
    async def _execute_automated_check(self, check_name: str) -> Dict[str, Any]:
        """Execute a specific automated check"""
        try:
            # Simulate different types of automated checks
            if check_name == 'asset_discovery_scan':
                return {
                    'status': 'passed',
                    'details': 'Asset inventory is up to date',
                    'score': 0.9,
                    'evidence': ['asset_inventory.json', 'scan_results.xml']
                }
            elif check_name == 'user_account_audit':
                return {
                    'status': 'failed',
                    'details': '5 inactive accounts found',
                    'score': 0.7,
                    'evidence': ['user_audit_report.csv'],
                    'issues': ['inactive_accounts', 'missing_reviews']
                }
            elif check_name == 'vulnerability_scan_status':
                return {
                    'status': 'passed',
                    'details': 'Vulnerability scans running on schedule',
                    'score': 0.95,
                    'evidence': ['scan_schedule.json', 'scan_results.xml']
                }
            elif check_name == 'network_monitoring_status':
                return {
                    'status': 'passed',
                    'details': 'Network monitoring active and alerting',
                    'score': 0.85,
                    'evidence': ['monitoring_config.json', 'alert_logs.txt']
                }
            else:
                # Default check result
                return {
                    'status': 'unknown',
                    'details': f'Check {check_name} not implemented',
                    'score': 0.5,
                    'evidence': []
                }
            
        except Exception as e:
            self.logger.error(f"Error executing automated check {check_name}: {e}")
            return {
                'status': 'error',
                'details': f'Check failed with error: {e}',
                'score': 0.0,
                'evidence': []
            }
    
    async def _collect_evidence(self, control: ComplianceControl) -> List[Dict[str, Any]]:
        """Collect evidence for compliance control"""
        try:
            evidence = []
            
            for evidence_type in control.evidence_types:
                evidence_item = await self._collect_evidence_type(evidence_type)
                if evidence_item:
                    evidence.append(evidence_item)
            
            self.metrics['evidence_collected'] += len(evidence)
            return evidence
            
        except Exception as e:
            self.logger.error(f"Error collecting evidence: {e}")
            return []
    
    async def _collect_evidence_type(self, evidence_type: str) -> Optional[Dict[str, Any]]:
        """Collect specific type of evidence"""
        try:
            # Simulate evidence collection
            if evidence_type == 'asset_inventory':
                return {
                    'type': 'asset_inventory',
                    'source': 'asset_management_system',
                    'collected_at': datetime.now().isoformat(),
                    'data': {
                        'total_assets': 150,
                        'last_updated': '2024-01-15',
                        'coverage': 0.95
                    }
                }
            elif evidence_type == 'access_logs':
                return {
                    'type': 'access_logs',
                    'source': 'identity_management_system',
                    'collected_at': datetime.now().isoformat(),
                    'data': {
                        'log_entries': 1250,
                        'date_range': '2024-01-01 to 2024-01-15',
                        'anomalies_detected': 3
                    }
                }
            elif evidence_type == 'vulnerability_scans':
                return {
                    'type': 'vulnerability_scans',
                    'source': 'vulnerability_scanner',
                    'collected_at': datetime.now().isoformat(),
                    'data': {
                        'last_scan': '2024-01-14',
                        'vulnerabilities_found': 12,
                        'critical_vulnerabilities': 2
                    }
                }
            else:
                return {
                    'type': evidence_type,
                    'source': 'unknown',
                    'collected_at': datetime.now().isoformat(),
                    'data': {'status': 'evidence_type_not_implemented'}
                }
            
        except Exception as e:
            self.logger.error(f"Error collecting evidence type {evidence_type}: {e}")
            return None
    
    async def _calculate_compliance_score(self, 
                                        control: ComplianceControl, 
                                        automated_results: Dict[str, Any], 
                                        evidence: List[Dict[str, Any]]) -> Tuple[float, ComplianceStatus]:
        """Calculate compliance score and status"""
        try:
            # Calculate score based on automated checks
            automated_scores = [
                result.get('score', 0.0) for result in automated_results.values()
                if isinstance(result, dict)
            ]
            
            if automated_scores:
                automated_score = sum(automated_scores) / len(automated_scores)
            else:
                automated_score = 0.5  # Default if no automated checks
            
            # Adjust score based on evidence quality
            evidence_score = min(len(evidence) / len(control.evidence_types), 1.0) if control.evidence_types else 1.0
            
            # Calculate overall score
            overall_score = (automated_score * 0.7) + (evidence_score * 0.3)
            
            # Determine status
            if overall_score >= 0.9:
                status = ComplianceStatus.COMPLIANT
            elif overall_score >= 0.7:
                status = ComplianceStatus.PARTIALLY_COMPLIANT
            elif overall_score >= 0.5:
                status = ComplianceStatus.REQUIRES_REVIEW
            else:
                status = ComplianceStatus.NON_COMPLIANT
            
            return overall_score, status
            
        except Exception as e:
            self.logger.error(f"Error calculating compliance score: {e}")
            return 0.0, ComplianceStatus.UNKNOWN
    
    async def _generate_findings(self, 
                               control: ComplianceControl, 
                               automated_results: Dict[str, Any], 
                               evidence: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance findings"""
        try:
            findings = []
            
            # Analyze automated check results
            for check_name, result in automated_results.items():
                if isinstance(result, dict):
                    if result.get('status') == 'failed':
                        findings.append(f"Automated check '{check_name}' failed: {result.get('details', 'No details')}")
                    elif result.get('issues'):
                        for issue in result['issues']:
                            findings.append(f"Issue found in '{check_name}': {issue}")
            
            # Analyze evidence gaps
            missing_evidence = set(control.evidence_types) - set(e.get('type', '') for e in evidence)
            for missing in missing_evidence:
                findings.append(f"Missing evidence type: {missing}")
            
            # Check evidence quality
            for evidence_item in evidence:
                if evidence_item.get('data', {}).get('status') == 'evidence_type_not_implemented':
                    findings.append(f"Evidence collection not implemented for: {evidence_item.get('type')}")
            
            return findings
            
        except Exception as e:
            self.logger.error(f"Error generating findings: {e}")
            return [f"Error generating findings: {e}"]
    
    async def _generate_recommendations(self, control: ComplianceControl, findings: List[str]) -> List[str]:
        """Generate compliance recommendations"""
        try:
            recommendations = []
            
            # Generate recommendations based on findings
            for finding in findings:
                if 'failed' in finding.lower():
                    recommendations.append(f"Investigate and remediate the failed check: {finding}")
                elif 'missing evidence' in finding.lower():
                    recommendations.append(f"Implement evidence collection for: {finding}")
                elif 'inactive accounts' in finding.lower():
                    recommendations.append("Review and disable inactive user accounts")
                elif 'vulnerability' in finding.lower():
                    recommendations.append("Prioritize vulnerability remediation based on risk assessment")
            
            # Add general recommendations based on control type
            if control.risk_level == 'high':
                recommendations.append("Consider implementing additional monitoring for this high-risk control")
            
            if not recommendations:
                recommendations.append("Continue monitoring and maintain current compliance posture")
            
            self.metrics['recommendations_made'] += len(recommendations)
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations - manual review required"]
    
    async def _send_assessment_notification(self, assessment: ComplianceAssessment) -> None:
        """Send assessment completion notification"""
        try:
            notification = Message(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id="broadcast",
                message_type=MessageType.COMPLIANCE_ASSESSMENT,
                content={
                    'assessment_id': assessment.assessment_id,
                    'control_id': assessment.control_id,
                    'framework': assessment.framework.value,
                    'status': assessment.status.value,
                    'score': assessment.score,
                    'findings_count': len(assessment.findings),
                    'recommendations_count': len(assessment.recommendations),
                    'assessed_at': assessment.assessed_at.isoformat()
                },
                timestamp=datetime.now()
            )
            
            await self.communication_bus.send_message(notification)
            
            self.logger.info(f"Sent assessment notification for {assessment.assessment_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending assessment notification: {e}")
    
    async def _monitor_compliance(self) -> None:
        """Monitor overall compliance status"""
        while self.active:
            try:
                # Check for compliance violations
                violations = [
                    assessment for assessment in self.assessments.values()
                    if assessment.status == ComplianceStatus.NON_COMPLIANT
                ]
                
                if violations:
                    await self._handle_compliance_violations(violations)
                
                # Check for overdue assessments
                overdue = await self._check_overdue_assessments()
                if overdue:
                    await self._handle_overdue_assessments(overdue)
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in compliance monitoring: {e}")
                await asyncio.sleep(3600)  # Retry after 1 hour
    
    async def _handle_compliance_violations(self, violations: List[ComplianceAssessment]) -> None:
        """Handle compliance violations"""
        try:
            for violation in violations:
                # Send violation alert
                alert = Message(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    recipient_id="broadcast",
                    message_type=MessageType.COMPLIANCE_VIOLATION,
                    content={
                        'assessment_id': violation.assessment_id,
                        'control_id': violation.control_id,
                        'framework': violation.framework.value,
                        'score': violation.score,
                        'findings': violation.findings,
                        'recommendations': violation.recommendations
                    },
                    timestamp=datetime.now()
                )
                
                await self.communication_bus.send_message(alert)
            
            self.logger.warning(f"Handled {len(violations)} compliance violations")
            
        except Exception as e:
            self.logger.error(f"Error handling compliance violations: {e}")
    
    async def _check_overdue_assessments(self) -> List[str]:
        """Check for overdue assessments"""
        try:
            current_time = datetime.now()
            overdue = []
            
            for control_id, next_time in self.assessment_schedule.items():
                if current_time > next_time + timedelta(days=1):  # 1 day grace period
                    overdue.append(control_id)
            
            return overdue
            
        except Exception as e:
            self.logger.error(f"Error checking overdue assessments: {e}")
            return []
    
    async def _handle_overdue_assessments(self, overdue: List[str]) -> None:
        """Handle overdue assessments"""
        try:
            for control_id in overdue:
                # Add to pending assessments
                if control_id not in self.pending_assessments:
                    self.pending_assessments.append(control_id)
                
                # Send overdue notification
                notification = Message(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    recipient_id="broadcast",
                    message_type=MessageType.ASSESSMENT_OVERDUE,
                    content={
                        'control_id': control_id,
                        'framework': self.compliance_controls[control_id].framework.value,
                        'scheduled_time': self.assessment_schedule[control_id].isoformat(),
                        'overdue_days': (datetime.now() - self.assessment_schedule[control_id]).days
                    },
                    timestamp=datetime.now()
                )
                
                await self.communication_bus.send_message(notification)
            
            self.logger.warning(f"Identified {len(overdue)} overdue assessments")
            
        except Exception as e:
            self.logger.error(f"Error handling overdue assessments: {e}")
    
    async def _generate_reports(self) -> None:
        """Generate compliance reports"""
        while self.active:
            try:
                # Generate monthly reports
                if datetime.now().day == 1:  # First day of month
                    for framework in self.frameworks:
                        await self._generate_compliance_report(framework, 'monthly')
                
                # Generate quarterly reports
                if datetime.now().month % 3 == 1 and datetime.now().day == 1:
                    for framework in self.frameworks:
                        await self._generate_compliance_report(framework, 'quarterly')
                
                await asyncio.sleep(86400)  # Check daily
                
            except Exception as e:
                self.logger.error(f"Error in report generation: {e}")
                await asyncio.sleep(86400)
    
    async def _generate_compliance_report(self, framework: ComplianceFramework, report_type: str) -> str:
        """Generate compliance report for framework"""
        try:
            report_id = f"RPT-{framework.value.upper()}-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
            
            # Get assessments for framework
            framework_assessments = [
                assessment for assessment in self.assessments.values()
                if assessment.framework == framework
            ]
            
            if not framework_assessments:
                self.logger.warning(f"No assessments found for framework {framework.value}")
                return report_id
            
            # Calculate overall score
            overall_score = sum(a.score for a in framework_assessments) / len(framework_assessments)
            
            # Generate summary
            summary = {
                'total_controls': len([c for c in self.compliance_controls.values() if c.framework == framework]),
                'assessed_controls': len(framework_assessments),
                'compliant_controls': len([a for a in framework_assessments if a.status == ComplianceStatus.COMPLIANT]),
                'non_compliant_controls': len([a for a in framework_assessments if a.status == ComplianceStatus.NON_COMPLIANT]),
                'partially_compliant_controls': len([a for a in framework_assessments if a.status == ComplianceStatus.PARTIALLY_COMPLIANT]),
                'overall_score': overall_score,
                'compliance_percentage': (len([a for a in framework_assessments if a.status == ComplianceStatus.COMPLIANT]) / len(framework_assessments)) * 100
            }
            
            # Generate recommendations
            all_recommendations = []
            for assessment in framework_assessments:
                if assessment.status != ComplianceStatus.COMPLIANT:
                    all_recommendations.extend(assessment.recommendations)
            
            # Remove duplicates
            unique_recommendations = list(set(all_recommendations))
            
            # Create report
            if report_type == 'monthly':
                period_start = datetime.now().replace(day=1) - timedelta(days=30)
                period_end = datetime.now().replace(day=1) - timedelta(days=1)
            else:  # quarterly
                period_start = datetime.now().replace(day=1) - timedelta(days=90)
                period_end = datetime.now().replace(day=1) - timedelta(days=1)
            
            report = ComplianceReport(
                report_id=report_id,
                framework=framework,
                report_type=report_type,
                period_start=period_start,
                period_end=period_end,
                overall_score=overall_score,
                control_assessments=[a.assessment_id for a in framework_assessments],
                summary=summary,
                recommendations=unique_recommendations,
                generated_by=self.agent_id,
                generated_at=datetime.now()
            )
            
            # Store report
            self.reports[report_id] = report
            
            # Send report notification
            await self._send_report_notification(report)
            
            # Update metrics
            self.metrics['reports_generated'] += 1
            
            self.logger.info(f"Generated {report_type} compliance report {report_id} for {framework.value}")
            
            return report_id
            
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            raise
    
    async def _send_report_notification(self, report: ComplianceReport) -> None:
        """Send report generation notification"""
        try:
            notification = Message(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id="broadcast",
                message_type=MessageType.COMPLIANCE_REPORT,
                content={
                    'report_id': report.report_id,
                    'framework': report.framework.value,
                    'report_type': report.report_type,
                    'overall_score': report.overall_score,
                    'summary': report.summary,
                    'recommendations_count': len(report.recommendations),
                    'generated_at': report.generated_at.isoformat()
                },
                timestamp=datetime.now()
            )
            
            await self.communication_bus.send_message(notification)
            
            self.logger.info(f"Sent report notification for {report.report_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending report notification: {e}")
    
    async def make_decision(self, environment_context: EnvironmentContext) -> ActionTaken:
        """Make compliance auditor decision"""
        try:
            # Check for pending assessments
            if self.pending_assessments:
                control_id = self.pending_assessments[0]
                control = self.compliance_controls.get(control_id)
                
                if control:
                    action = ActionTaken(
                        action_id=str(uuid.uuid4()),
                        primary_action="assess_compliance_control",
                        action_type="compliance_assessment",
                        target=control_id,
                        parameters={
                            'control_title': control.title,
                            'framework': control.framework.value,
                            'risk_level': control.risk_level,
                            'frequency': control.frequency
                        },
                        confidence_score=0.9,
                        reasoning=f"Assessing overdue compliance control: {control.title}"
                    )
                    
                    # Remove from pending list
                    self.pending_assessments.remove(control_id)
                    return action
            
            # Check for compliance violations requiring attention
            recent_violations = [
                assessment for assessment in self.assessments.values()
                if (assessment.status == ComplianceStatus.NON_COMPLIANT and
                    datetime.now() - assessment.assessed_at <= timedelta(days=1))
            ]
            
            if recent_violations:
                violation = recent_violations[0]
                action = ActionTaken(
                    action_id=str(uuid.uuid4()),
                    primary_action="remediate_compliance_violation",
                    action_type="compliance_remediation",
                    target=violation.control_id,
                    parameters={
                        'assessment_id': violation.assessment_id,
                        'framework': violation.framework.value,
                        'score': violation.score,
                        'findings_count': len(violation.findings)
                    },
                    confidence_score=0.8,
                    reasoning=f"Addressing compliance violation in control {violation.control_id}"
                )
                return action
            
            # Default monitoring action
            action = ActionTaken(
                action_id=str(uuid.uuid4()),
                primary_action="monitor_compliance_status",
                action_type="monitoring",
                target="compliance_framework",
                parameters={
                    'frameworks': [f.value for f in self.frameworks],
                    'total_controls': len(self.compliance_controls),
                    'recent_assessments': len([a for a in self.assessments.values() 
                                             if datetime.now() - a.assessed_at <= timedelta(days=7)])
                },
                confidence_score=0.6,
                reasoning="Monitoring overall compliance status and assessment schedule"
            )
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error in compliance auditor decision making: {e}")
            
            return ActionTaken(
                action_id=str(uuid.uuid4()),
                primary_action="error_recovery",
                action_type="system_maintenance",
                target="self",
                parameters={'error': str(e)},
                confidence_score=0.3,
                reasoning=f"Error recovery due to: {e}"
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get compliance auditor performance metrics"""
        return {
            'agent_id': self.agent_id,
            'total_controls': len(self.compliance_controls),
            'total_assessments': len(self.assessments),
            'pending_assessments': len(self.pending_assessments),
            'total_reports': len(self.reports),
            'frameworks': [f.value for f in self.frameworks],
            **self.metrics
        }
    
    async def shutdown(self) -> None:
        """Shutdown the Compliance Auditor agent"""
        try:
            # Cancel all tasks
            tasks = ['assessment_task', 'monitoring_task', 'reporting_task']
            for task_name in tasks:
                if hasattr(self, task_name):
                    task = getattr(self, task_name)
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            await super().shutdown()
            self.logger.info(f"Compliance Auditor Agent {self.agent_id} shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during compliance auditor shutdown: {e}")