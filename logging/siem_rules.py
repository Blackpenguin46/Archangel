#!/usr/bin/env python3
"""
Custom SIEM Rules Engine for Attack Pattern Detection
Advanced rule engine for detecting sophisticated attack patterns and threats
"""

import re
import json
import yaml
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Pattern, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
from pathlib import Path

# Import from our log parser
from .log_parser import ParsedEvent, EventType, SeverityLevel, CorrelatedEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of SIEM rules."""
    THRESHOLD = "threshold"
    SEQUENCE = "sequence"
    STATISTICAL = "statistical"
    PATTERN_MATCH = "pattern_match"
    BEHAVIORAL = "behavioral"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"


class ThreatCategory(Enum):
    """Categories of security threats."""
    AUTHENTICATION_ATTACK = "authentication_attack"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    MALWARE = "malware"
    INSIDER_THREAT = "insider_threat"
    NETWORK_ATTACK = "network_attack"
    WEB_ATTACK = "web_attack"
    DENIAL_OF_SERVICE = "denial_of_service"
    RECONNAISSANCE = "reconnaissance"
    LATERAL_MOVEMENT = "lateral_movement"


@dataclass
class RuleCondition:
    """Individual condition within a SIEM rule."""
    field: str
    operator: str  # eq, neq, gt, lt, gte, lte, contains, regex, in
    value: Union[str, int, float, List[str]]
    case_sensitive: bool = True


@dataclass
class SIEMRule:
    """Complete SIEM rule definition."""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    threat_category: ThreatCategory
    severity: SeverityLevel
    conditions: List[RuleCondition]
    time_window: int = 300  # seconds
    threshold: int = 1
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    mitre_tactics: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)
    false_positive_filters: List[RuleCondition] = field(default_factory=list)
    created_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RuleMatch:
    """Result of a rule match."""
    rule_id: str
    rule_name: str
    matched_events: List[ParsedEvent]
    confidence: float  # 0.0 to 1.0
    threat_score: int  # 1 to 100
    indicators: Dict[str, Any]
    remediation_steps: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AttackPatternLibrary:
    """Library of known attack patterns and signatures."""
    
    def __init__(self):
        self.patterns = self._initialize_attack_patterns()
    
    def _initialize_attack_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive attack pattern definitions."""
        return {
            'brute_force_ssh': {
                'name': 'SSH Brute Force Attack',
                'description': 'Multiple failed SSH authentication attempts',
                'indicators': [
                    'high frequency of failed SSH logins',
                    'same source IP with different usernames',
                    'dictionary attack patterns'
                ],
                'mitre_technique': 'T1110.001',
                'severity': SeverityLevel.HIGH
            },
            'credential_stuffing': {
                'name': 'Credential Stuffing Attack',
                'description': 'Automated login attempts using leaked credentials',
                'indicators': [
                    'failed logins from multiple IPs',
                    'same username across different sources',
                    'high velocity authentication attempts'
                ],
                'mitre_technique': 'T1110.004',
                'severity': SeverityLevel.HIGH
            },
            'privilege_escalation_sudo': {
                'name': 'Sudo Privilege Escalation',
                'description': 'Suspicious sudo command execution patterns',
                'indicators': [
                    'unusual sudo commands',
                    'privilege escalation after authentication',
                    'access to sensitive files'
                ],
                'mitre_technique': 'T1548.003',
                'severity': SeverityLevel.CRITICAL
            },
            'lateral_movement_ssh': {
                'name': 'SSH Lateral Movement',
                'description': 'SSH connections indicating lateral movement',
                'indicators': [
                    'SSH connections between internal hosts',
                    'rapid succession of connections',
                    'unusual user account usage'
                ],
                'mitre_technique': 'T1021.004',
                'severity': SeverityLevel.HIGH
            },
            'data_exfiltration_web': {
                'name': 'Web-based Data Exfiltration',
                'description': 'Suspicious web requests indicating data theft',
                'indicators': [
                    'large response sizes',
                    'unusual file access patterns',
                    'automated tool user agents'
                ],
                'mitre_technique': 'T1041',
                'severity': SeverityLevel.CRITICAL
            },
            'sql_injection': {
                'name': 'SQL Injection Attack',
                'description': 'Web requests containing SQL injection patterns',
                'indicators': [
                    'SQL keywords in parameters',
                    'union select statements',
                    'error-based injection attempts'
                ],
                'mitre_technique': 'T1190',
                'severity': SeverityLevel.HIGH
            },
            'xss_attack': {
                'name': 'Cross-Site Scripting (XSS)',
                'description': 'Malicious script injection attempts',
                'indicators': [
                    'script tags in parameters',
                    'javascript event handlers',
                    'encoded malicious payloads'
                ],
                'mitre_technique': 'T1059.007',
                'severity': SeverityLevel.MEDIUM
            },
            'command_injection': {
                'name': 'Command Injection',
                'description': 'OS command injection in web parameters',
                'indicators': [
                    'shell metacharacters',
                    'command chaining operators',
                    'system command names'
                ],
                'mitre_technique': 'T1059.004',
                'severity': SeverityLevel.HIGH
            },
            'directory_traversal': {
                'name': 'Directory Traversal',
                'description': 'Path traversal attempts to access restricted files',
                'indicators': [
                    '../ sequences in paths',
                    'encoded path traversal',
                    'access to system files'
                ],
                'mitre_technique': 'T1083',
                'severity': SeverityLevel.MEDIUM
            },
            'reconnaissance_scan': {
                'name': 'Network Reconnaissance',
                'description': 'Network scanning and enumeration activities',
                'indicators': [
                    'port scanning patterns',
                    'vulnerability scanner signatures',
                    'service enumeration attempts'
                ],
                'mitre_technique': 'T1046',
                'severity': SeverityLevel.MEDIUM
            }
        }


class SIEMRuleEngine:
    """Advanced SIEM rule engine for attack detection."""
    
    def __init__(self):
        self.rules = []
        self.event_buffer = deque(maxlen=10000)
        self.rule_cache = {}
        self.pattern_library = AttackPatternLibrary()
        self.statistics = defaultdict(int)
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """Initialize comprehensive set of default SIEM rules."""
        default_rules = [
            # Authentication Attack Rules
            SIEMRule(
                rule_id="AUTH_001",
                name="SSH Brute Force Detection",
                description="Multiple failed SSH authentication attempts from same source",
                rule_type=RuleType.THRESHOLD,
                threat_category=ThreatCategory.AUTHENTICATION_ATTACK,
                severity=SeverityLevel.HIGH,
                conditions=[
                    RuleCondition("event_type", "eq", "authentication"),
                    RuleCondition("message", "contains", "Failed"),
                    RuleCondition("message", "contains", "ssh")
                ],
                time_window=300,
                threshold=5,
                tags=["brute_force", "ssh", "authentication"],
                mitre_tactics=["TA0006"],  # Credential Access
                mitre_techniques=["T1110.001"]  # Password Brute Force
            ),
            
            SIEMRule(
                rule_id="AUTH_002",
                name="Multiple Failed Web Logins",
                description="Rapid failed login attempts on web application",
                rule_type=RuleType.THRESHOLD,
                threat_category=ThreatCategory.AUTHENTICATION_ATTACK,
                severity=SeverityLevel.MEDIUM,
                conditions=[
                    RuleCondition("status", "eq", "401"),
                    RuleCondition("resource", "contains", "login")
                ],
                time_window=180,
                threshold=10,
                tags=["web_attack", "brute_force", "login"],
                mitre_techniques=["T1110.003"]
            ),
            
            # Privilege Escalation Rules
            SIEMRule(
                rule_id="PRIV_001",
                name="Suspicious Sudo Usage",
                description="Unusual sudo command patterns indicating privilege escalation",
                rule_type=RuleType.PATTERN_MATCH,
                threat_category=ThreatCategory.PRIVILEGE_ESCALATION,
                severity=SeverityLevel.HIGH,
                conditions=[
                    RuleCondition("message", "contains", "sudo"),
                    RuleCondition("message", "regex", r"(bash|sh|nc|wget|curl|python)"),
                ],
                time_window=600,
                threshold=1,
                tags=["privilege_escalation", "sudo", "shell"],
                mitre_techniques=["T1548.003"]
            ),
            
            SIEMRule(
                rule_id="PRIV_002",
                name="Root Shell Spawning",
                description="Detection of root shell spawning activities",
                rule_type=RuleType.PATTERN_MATCH,
                threat_category=ThreatCategory.PRIVILEGE_ESCALATION,
                severity=SeverityLevel.CRITICAL,
                conditions=[
                    RuleCondition("user", "eq", "root"),
                    RuleCondition("message", "regex", r"(/bin/bash|/bin/sh|su\s+root)")
                ],
                time_window=60,
                threshold=1,
                tags=["privilege_escalation", "root", "shell"],
                mitre_techniques=["T1548"]
            ),
            
            # Web Attack Rules
            SIEMRule(
                rule_id="WEB_001",
                name="SQL Injection Attempt",
                description="Detection of SQL injection patterns in web requests",
                rule_type=RuleType.PATTERN_MATCH,
                threat_category=ThreatCategory.WEB_ATTACK,
                severity=SeverityLevel.HIGH,
                conditions=[
                    RuleCondition("resource", "regex", 
                                r"(union\s+select|' or |' and |--|\*\*|/\*|\*/|0x[0-9a-f]+)")
                ],
                time_window=300,
                threshold=1,
                tags=["web_attack", "sql_injection", "database"],
                mitre_techniques=["T1190"]
            ),
            
            SIEMRule(
                rule_id="WEB_002",
                name="XSS Attack Detection",
                description="Cross-site scripting attempt detection",
                rule_type=RuleType.PATTERN_MATCH,
                threat_category=ThreatCategory.WEB_ATTACK,
                severity=SeverityLevel.MEDIUM,
                conditions=[
                    RuleCondition("resource", "regex", 
                                r"(<script|javascript:|onload=|onerror=|alert\(|prompt\()")
                ],
                time_window=300,
                threshold=1,
                tags=["web_attack", "xss", "javascript"],
                mitre_techniques=["T1059.007"]
            ),
            
            SIEMRule(
                rule_id="WEB_003",
                name="Directory Traversal Attempt",
                description="Path traversal attack detection",
                rule_type=RuleType.PATTERN_MATCH,
                threat_category=ThreatCategory.WEB_ATTACK,
                severity=SeverityLevel.MEDIUM,
                conditions=[
                    RuleCondition("resource", "regex", r"(\.\./|\.\.\\|%2e%2e%2f|%252e%252e%252f)")
                ],
                time_window=300,
                threshold=1,
                tags=["web_attack", "directory_traversal", "path"],
                mitre_techniques=["T1083"]
            ),
            
            # Network Attack Rules
            SIEMRule(
                rule_id="NET_001",
                name="Port Scanning Detection",
                description="Network port scanning activity detection",
                rule_type=RuleType.THRESHOLD,
                threat_category=ThreatCategory.RECONNAISSANCE,
                severity=SeverityLevel.MEDIUM,
                conditions=[
                    RuleCondition("event_type", "eq", "network_access"),
                    RuleCondition("status", "in", ["403", "404", "503"])
                ],
                time_window=120,
                threshold=20,
                tags=["reconnaissance", "port_scan", "network"],
                mitre_techniques=["T1046"]
            ),
            
            # Behavioral Rules
            SIEMRule(
                rule_id="BEH_001",
                name="Unusual Login Time",
                description="Login attempts outside normal business hours",
                rule_type=RuleType.BEHAVIORAL,
                threat_category=ThreatCategory.INSIDER_THREAT,
                severity=SeverityLevel.LOW,
                conditions=[
                    RuleCondition("event_type", "eq", "authentication"),
                    RuleCondition("message", "contains", "Accepted")
                ],
                time_window=3600,
                threshold=1,
                tags=["behavioral", "off_hours", "authentication"],
                mitre_techniques=["T1078"]
            ),
            
            # Anomaly Detection Rules
            SIEMRule(
                rule_id="ANOM_001",
                name="Error Rate Spike",
                description="Unusual spike in error events",
                rule_type=RuleType.STATISTICAL,
                threat_category=ThreatCategory.DENIAL_OF_SERVICE,
                severity=SeverityLevel.MEDIUM,
                conditions=[
                    RuleCondition("event_type", "eq", "error")
                ],
                time_window=300,
                threshold=50,
                tags=["anomaly", "error_spike", "availability"],
                mitre_techniques=["T1499"]
            )
        ]
        
        self.rules.extend(default_rules)
    
    def add_rule(self, rule: SIEMRule) -> None:
        """Add a custom SIEM rule."""
        self.rules.append(rule)
        logger.info(f"Added SIEM rule: {rule.rule_id} - {rule.name}")
    
    def load_rules_from_file(self, file_path: str) -> None:
        """Load SIEM rules from YAML or JSON file."""
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"Rule file not found: {file_path}")
            return
        
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            for rule_data in data.get('rules', []):
                rule = self._parse_rule_data(rule_data)
                if rule:
                    self.add_rule(rule)
                    
        except Exception as e:
            logger.error(f"Error loading rules from {file_path}: {e}")
    
    def _parse_rule_data(self, rule_data: Dict[str, Any]) -> Optional[SIEMRule]:
        """Parse rule data from file into SIEMRule object."""
        try:
            conditions = []
            for cond_data in rule_data.get('conditions', []):
                condition = RuleCondition(
                    field=cond_data['field'],
                    operator=cond_data['operator'],
                    value=cond_data['value'],
                    case_sensitive=cond_data.get('case_sensitive', True)
                )
                conditions.append(condition)
            
            rule = SIEMRule(
                rule_id=rule_data['rule_id'],
                name=rule_data['name'],
                description=rule_data['description'],
                rule_type=RuleType(rule_data['rule_type']),
                threat_category=ThreatCategory(rule_data['threat_category']),
                severity=SeverityLevel[rule_data['severity']],
                conditions=conditions,
                time_window=rule_data.get('time_window', 300),
                threshold=rule_data.get('threshold', 1),
                enabled=rule_data.get('enabled', True),
                tags=rule_data.get('tags', []),
                mitre_techniques=rule_data.get('mitre_techniques', [])
            )
            
            return rule
            
        except Exception as e:
            logger.error(f"Error parsing rule data: {e}")
            return None
    
    def evaluate_event(self, event: ParsedEvent) -> List[RuleMatch]:
        """Evaluate a single event against all SIEM rules."""
        self.event_buffer.append(event)
        matches = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            match = self._evaluate_rule(rule, event)
            if match:
                matches.append(match)
                self.statistics[f"rule_{rule.rule_id}_matches"] += 1
                logger.warning(f"SIEM Rule Match: {rule.name} (Confidence: {match.confidence:.2f})")
        
        return matches
    
    def _evaluate_rule(self, rule: SIEMRule, new_event: ParsedEvent) -> Optional[RuleMatch]:
        """Evaluate a specific rule against current events."""
        if rule.rule_type == RuleType.THRESHOLD:
            return self._evaluate_threshold_rule(rule, new_event)
        elif rule.rule_type == RuleType.PATTERN_MATCH:
            return self._evaluate_pattern_rule(rule, new_event)
        elif rule.rule_type == RuleType.BEHAVIORAL:
            return self._evaluate_behavioral_rule(rule, new_event)
        elif rule.rule_type == RuleType.STATISTICAL:
            return self._evaluate_statistical_rule(rule, new_event)
        elif rule.rule_type == RuleType.SEQUENCE:
            return self._evaluate_sequence_rule(rule, new_event)
        
        return None
    
    def _evaluate_threshold_rule(self, rule: SIEMRule, new_event: ParsedEvent) -> Optional[RuleMatch]:
        """Evaluate threshold-based rules."""
        # Get events within time window
        cutoff_time = new_event.timestamp - timedelta(seconds=rule.time_window)
        relevant_events = [
            event for event in self.event_buffer
            if event.timestamp >= cutoff_time and self._event_matches_conditions(event, rule.conditions)
        ]
        
        if len(relevant_events) >= rule.threshold:
            confidence = min(1.0, len(relevant_events) / (rule.threshold * 2))
            return self._create_rule_match(rule, relevant_events, confidence)
        
        return None
    
    def _evaluate_pattern_rule(self, rule: SIEMRule, event: ParsedEvent) -> Optional[RuleMatch]:
        """Evaluate pattern matching rules."""
        if self._event_matches_conditions(event, rule.conditions):
            # Check false positive filters
            if self._event_matches_conditions(event, rule.false_positive_filters):
                return None
            
            confidence = 0.8  # High confidence for direct pattern matches
            return self._create_rule_match(rule, [event], confidence)
        
        return None
    
    def _evaluate_behavioral_rule(self, rule: SIEMRule, event: ParsedEvent) -> Optional[RuleMatch]:
        """Evaluate behavioral analysis rules."""
        if not self._event_matches_conditions(event, rule.conditions):
            return None
        
        # Check for behavioral anomalies
        confidence = 0.5  # Lower confidence for behavioral rules
        
        # Time-based behavioral analysis
        if rule.rule_id == "BEH_001":  # Unusual login time
            hour = event.timestamp.hour
            if hour < 6 or hour > 22:  # Outside business hours
                confidence = 0.7
                return self._create_rule_match(rule, [event], confidence)
        
        return None
    
    def _evaluate_statistical_rule(self, rule: SIEMRule, new_event: ParsedEvent) -> Optional[RuleMatch]:
        """Evaluate statistical anomaly rules."""
        # Get events within time window
        cutoff_time = new_event.timestamp - timedelta(seconds=rule.time_window)
        relevant_events = [
            event for event in self.event_buffer
            if event.timestamp >= cutoff_time and self._event_matches_conditions(event, rule.conditions)
        ]
        
        # Calculate statistical threshold
        historical_count = len(relevant_events)
        if historical_count >= rule.threshold:
            # Calculate z-score or other statistical measures
            confidence = min(1.0, historical_count / (rule.threshold * 3))
            return self._create_rule_match(rule, relevant_events, confidence)
        
        return None
    
    def _evaluate_sequence_rule(self, rule: SIEMRule, new_event: ParsedEvent) -> Optional[RuleMatch]:
        """Evaluate sequence-based rules."""
        # This would implement sequence detection logic
        # For now, return None as it requires more complex implementation
        return None
    
    def _event_matches_conditions(self, event: ParsedEvent, conditions: List[RuleCondition]) -> bool:
        """Check if an event matches all specified conditions."""
        for condition in conditions:
            if not self._evaluate_condition(event, condition):
                return False
        return True
    
    def _evaluate_condition(self, event: ParsedEvent, condition: RuleCondition) -> bool:
        """Evaluate a single condition against an event."""
        # Get the field value from the event
        value = self._get_event_field(event, condition.field)
        if value is None:
            return False
        
        # Convert to string for pattern matching
        if not isinstance(value, str):
            value = str(value)
        
        # Apply case sensitivity
        if not condition.case_sensitive:
            value = value.lower()
            if isinstance(condition.value, str):
                condition.value = condition.value.lower()
        
        # Evaluate based on operator
        if condition.operator == "eq":
            return value == condition.value
        elif condition.operator == "neq":
            return value != condition.value
        elif condition.operator == "contains":
            return str(condition.value) in value
        elif condition.operator == "regex":
            pattern = re.compile(str(condition.value), re.IGNORECASE if not condition.case_sensitive else 0)
            return bool(pattern.search(value))
        elif condition.operator == "in":
            return value in condition.value
        elif condition.operator in ["gt", "gte", "lt", "lte"]:
            try:
                num_value = float(value)
                condition_value = float(condition.value)
                if condition.operator == "gt":
                    return num_value > condition_value
                elif condition.operator == "gte":
                    return num_value >= condition_value
                elif condition.operator == "lt":
                    return num_value < condition_value
                elif condition.operator == "lte":
                    return num_value <= condition_value
            except (ValueError, TypeError):
                return False
        
        return False
    
    def _get_event_field(self, event: ParsedEvent, field_name: str) -> Any:
        """Get a field value from an event."""
        if hasattr(event, field_name):
            value = getattr(event, field_name)
            if hasattr(value, 'value'):  # Handle enum values
                return value.value
            return value
        elif field_name in event.extracted_fields:
            return event.extracted_fields[field_name]
        
        return None
    
    def _create_rule_match(self, rule: SIEMRule, events: List[ParsedEvent], confidence: float) -> RuleMatch:
        """Create a RuleMatch object for a triggered rule."""
        # Calculate threat score
        threat_score = int(rule.severity.value * 20 * confidence)  # Scale to 1-100
        
        # Extract indicators
        indicators = self._extract_indicators(rule, events)
        
        # Generate remediation steps
        remediation_steps = self._generate_remediation_steps(rule, events)
        
        return RuleMatch(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            matched_events=events,
            confidence=confidence,
            threat_score=threat_score,
            indicators=indicators,
            remediation_steps=remediation_steps
        )
    
    def _extract_indicators(self, rule: SIEMRule, events: List[ParsedEvent]) -> Dict[str, Any]:
        """Extract threat indicators from matched events."""
        indicators = {
            'event_count': len(events),
            'time_span': (events[-1].timestamp - events[0].timestamp).total_seconds() if len(events) > 1 else 0,
            'threat_category': rule.threat_category.value,
            'mitre_techniques': rule.mitre_techniques,
            'source_ips': list(set(e.source_ip for e in events if e.source_ip)),
            'users': list(set(e.user for e in events if e.user)),
            'resources': list(set(e.resource for e in events if e.resource))
        }
        
        return indicators
    
    def _generate_remediation_steps(self, rule: SIEMRule, events: List[ParsedEvent]) -> List[str]:
        """Generate remediation steps based on rule type and matched events."""
        steps = []
        
        if rule.threat_category == ThreatCategory.AUTHENTICATION_ATTACK:
            steps.extend([
                "Block source IP addresses temporarily",
                "Review authentication logs for patterns",
                "Consider implementing account lockout policies",
                "Enable two-factor authentication"
            ])
        elif rule.threat_category == ThreatCategory.WEB_ATTACK:
            steps.extend([
                "Block malicious requests at WAF level",
                "Review application input validation",
                "Update web application security controls",
                "Monitor for additional attack attempts"
            ])
        elif rule.threat_category == ThreatCategory.PRIVILEGE_ESCALATION:
            steps.extend([
                "Investigate user account activity",
                "Review sudo access permissions",
                "Monitor privileged operations closely",
                "Consider revoking elevated access"
            ])
        
        # Add general steps
        steps.extend([
            "Create incident ticket for investigation",
            "Notify security team immediately",
            "Preserve logs for forensic analysis"
        ])
        
        return steps
    
    def export_rules_to_file(self, file_path: str, format_type: str = 'yaml') -> None:
        """Export current rules to a file."""
        rules_data = {
            'rules': []
        }
        
        for rule in self.rules:
            rule_dict = {
                'rule_id': rule.rule_id,
                'name': rule.name,
                'description': rule.description,
                'rule_type': rule.rule_type.value,
                'threat_category': rule.threat_category.value,
                'severity': rule.severity.name,
                'conditions': [
                    {
                        'field': cond.field,
                        'operator': cond.operator,
                        'value': cond.value,
                        'case_sensitive': cond.case_sensitive
                    }
                    for cond in rule.conditions
                ],
                'time_window': rule.time_window,
                'threshold': rule.threshold,
                'enabled': rule.enabled,
                'tags': rule.tags,
                'mitre_techniques': rule.mitre_techniques
            }
            rules_data['rules'].append(rule_dict)
        
        with open(file_path, 'w') as f:
            if format_type.lower() == 'yaml':
                yaml.dump(rules_data, f, default_flow_style=False, indent=2)
            else:
                json.dump(rules_data, f, indent=2)
        
        logger.info(f"Exported {len(self.rules)} rules to {file_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rule engine statistics."""
        enabled_rules = sum(1 for rule in self.rules if rule.enabled)
        
        stats = {
            'total_rules': len(self.rules),
            'enabled_rules': enabled_rules,
            'disabled_rules': len(self.rules) - enabled_rules,
            'events_processed': len(self.event_buffer),
            'rule_matches': dict(self.statistics),
            'threat_categories': list(set(rule.threat_category.value for rule in self.rules))
        }
        
        return stats


def main():
    """Demonstration of SIEM rules engine."""
    # Initialize the rule engine
    engine = SIEMRuleEngine()
    
    print("SIEM Rules Engine Demonstration")
    print("=" * 50)
    print(f"Loaded {len(engine.rules)} default rules")
    
    # Show statistics
    stats = engine.get_statistics()
    print(f"Enabled rules: {stats['enabled_rules']}")
    print(f"Threat categories: {', '.join(stats['threat_categories'])}")
    
    # Test with sample events (would normally come from log parser)
    from .log_parser import ParsedEvent, EventType, SeverityLevel
    
    # Simulate SSH brute force attack
    sample_events = []
    for i in range(6):
        event = ParsedEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.AUTHENTICATION,
            severity=SeverityLevel.MEDIUM,
            source_ip="192.168.1.100",
            user=f"user{i}",
            message="Failed password for user from 192.168.1.100 port 22 ssh2",
            raw_log=f"Failed ssh login attempt {i}"
        )
        sample_events.append(event)
    
    # Evaluate events
    print("\nEvaluating sample events...")
    all_matches = []
    for event in sample_events:
        matches = engine.evaluate_event(event)
        all_matches.extend(matches)
    
    # Display results
    print(f"\nDetected {len(all_matches)} rule matches:")
    for match in all_matches:
        print(f"- Rule: {match.rule_name}")
        print(f"  Confidence: {match.confidence:.2f}")
        print(f"  Threat Score: {match.threat_score}/100")
        print(f"  Events: {len(match.matched_events)}")
        print(f"  Indicators: {match.indicators}")
        print()


if __name__ == "__main__":
    main()