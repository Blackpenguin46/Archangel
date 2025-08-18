#!/usr/bin/env python3
"""
Advanced Log Parsing and Event Correlation for Security Analysis
Intelligent log analysis with pattern recognition and security event correlation
"""

import re
import json
import hashlib
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Pattern, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of security events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK_ACCESS = "network_access"
    FILE_ACCESS = "file_access"
    PROCESS_EXECUTION = "process_execution"
    SYSTEM_CHANGE = "system_change"
    ANOMALY = "anomaly"
    ATTACK = "attack"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class SeverityLevel(Enum):
    """Security event severity levels."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1


@dataclass
class ParsedEvent:
    """Structured representation of a parsed log event."""
    timestamp: datetime
    event_type: EventType
    severity: SeverityLevel
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    user: Optional[str] = None
    action: Optional[str] = None
    resource: Optional[str] = None
    status: Optional[str] = None
    message: str = ""
    raw_log: str = ""
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default="")
    
    def __post_init__(self):
        if not self.event_id:
            # Generate unique event ID based on content
            content = f"{self.timestamp}{self.message}{self.raw_log}"
            self.event_id = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class CorrelationRule:
    """Rule for correlating related security events."""
    name: str
    description: str
    event_types: List[EventType]
    time_window: int  # seconds
    threshold: int  # minimum events to trigger
    severity: SeverityLevel
    pattern: Dict[str, Any]
    action: str


@dataclass
class CorrelatedEvent:
    """Group of correlated security events."""
    correlation_id: str
    rule_name: str
    events: List[ParsedEvent]
    severity: SeverityLevel
    start_time: datetime
    end_time: datetime
    summary: str
    indicators: Dict[str, Any] = field(default_factory=dict)


class LogPatternLibrary:
    """Library of log parsing patterns for different systems."""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, Pattern]]:
        """Initialize comprehensive log parsing patterns."""
        return {
            'syslog': {
                'auth_success': re.compile(
                    r'(?P<timestamp>\w+\s+\d+\s+\d+:\d+:\d+)\s+'
                    r'(?P<host>\S+)\s+sshd\[\d+\]:\s+'
                    r'Accepted\s+(?P<method>\w+)\s+for\s+(?P<user>\w+)\s+'
                    r'from\s+(?P<source_ip>[\d.]+)\s+port\s+(?P<port>\d+)'
                ),
                'auth_failure': re.compile(
                    r'(?P<timestamp>\w+\s+\d+\s+\d+:\d+:\d+)\s+'
                    r'(?P<host>\S+)\s+sshd\[\d+\]:\s+'
                    r'Failed\s+(?P<method>\w+)\s+for\s+(?P<user>\w+)\s+'
                    r'from\s+(?P<source_ip>[\d.]+)\s+port\s+(?P<port>\d+)'
                ),
                'sudo_command': re.compile(
                    r'(?P<timestamp>\w+\s+\d+\s+\d+:\d+:\d+)\s+'
                    r'(?P<host>\S+)\s+sudo:\s+'
                    r'(?P<user>\w+)\s+:\s+TTY=(?P<tty>\S+)\s+;\s+'
                    r'PWD=(?P<pwd>\S+)\s+;\s+USER=(?P<target_user>\w+)\s+;\s+'
                    r'COMMAND=(?P<command>.*)'
                )
            },
            'apache': {
                'access_log': re.compile(
                    r'(?P<source_ip>[\d.]+)\s+-\s+-\s+'
                    r'\[(?P<timestamp>[^\]]+)\]\s+'
                    r'"(?P<method>\w+)\s+(?P<path>\S+)\s+HTTP/[\d.]+"\s+'
                    r'(?P<status>\d+)\s+(?P<size>\d+)\s+'
                    r'"(?P<referer>[^"]*)"\s+"(?P<user_agent>[^"]*)"'
                ),
                'error_log': re.compile(
                    r'\[(?P<timestamp>[^\]]+)\]\s+'
                    r'\[(?P<level>\w+)\]\s+'
                    r'\[pid\s+(?P<pid>\d+)\]\s+'
                    r'(?P<message>.*)'
                )
            },
            'nginx': {
                'access_log': re.compile(
                    r'(?P<source_ip>[\d.]+)\s+-\s+(?P<user>\S+)\s+'
                    r'\[(?P<timestamp>[^\]]+)\]\s+'
                    r'"(?P<method>\w+)\s+(?P<path>\S+)\s+HTTP/[\d.]+"\s+'
                    r'(?P<status>\d+)\s+(?P<size>\d+)\s+'
                    r'"(?P<referer>[^"]*)"\s+"(?P<user_agent>[^"]*)"'
                )
            },
            'application': {
                'json_log': re.compile(
                    r'(?P<json_data>\{.*\})'
                ),
                'structured_log': re.compile(
                    r'(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\d.Z+-]*)\s+'
                    r'\[(?P<level>\w+)\]\s+'
                    r'(?P<component>\w+):\s+'
                    r'(?P<message>.*)'
                )
            },
            'security': {
                'failed_login': re.compile(
                    r'(?P<timestamp>.*?)\s+'
                    r'.*?failed.*?login.*?'
                    r'(?P<user>\w+).*?'
                    r'from\s+(?P<source_ip>[\d.]+)',
                    re.IGNORECASE
                ),
                'privilege_escalation': re.compile(
                    r'(?P<timestamp>.*?)\s+'
                    r'.*?(?P<user>\w+).*?'
                    r'(?:sudo|su|elevated|admin|root)',
                    re.IGNORECASE
                ),
                'file_access': re.compile(
                    r'(?P<timestamp>.*?)\s+'
                    r'.*?(?P<action>read|write|delete|modify).*?'
                    r'file.*?(?P<path>/\S+)',
                    re.IGNORECASE
                )
            }
        }


class LogParser:
    """Advanced log parser with pattern matching and field extraction."""
    
    def __init__(self):
        self.pattern_library = LogPatternLibrary()
        self.custom_patterns = {}
    
    def add_custom_pattern(self, source_type: str, pattern_name: str, pattern: str) -> None:
        """Add a custom parsing pattern."""
        if source_type not in self.custom_patterns:
            self.custom_patterns[source_type] = {}
        self.custom_patterns[source_type][pattern_name] = re.compile(pattern)
    
    def parse_log_entry(self, log_line: str, source_type: str = 'auto') -> Optional[ParsedEvent]:
        """Parse a single log entry into a structured event."""
        if source_type == 'auto':
            source_type = self._detect_log_type(log_line)
        
        # Try patterns for the detected source type
        patterns = self.pattern_library.patterns.get(source_type, {})
        custom_patterns = self.custom_patterns.get(source_type, {})
        all_patterns = {**patterns, **custom_patterns}
        
        for pattern_name, pattern in all_patterns.items():
            match = pattern.search(log_line)
            if match:
                return self._create_parsed_event(match, pattern_name, log_line)
        
        # If no specific pattern matches, try generic parsing
        return self._generic_parse(log_line)
    
    def _detect_log_type(self, log_line: str) -> str:
        """Automatically detect the type of log entry."""
        line_lower = log_line.lower()
        
        # Check for common log formats
        if 'sshd' in line_lower or 'sudo' in line_lower:
            return 'syslog'
        elif '"GET' in log_line or '"POST' in log_line:
            if 'apache' in line_lower:
                return 'apache'
            else:
                return 'nginx'
        elif log_line.strip().startswith('{') and log_line.strip().endswith('}'):
            return 'application'
        elif any(term in line_lower for term in ['failed', 'error', 'denied', 'unauthorized']):
            return 'security'
        else:
            return 'application'
    
    def _create_parsed_event(self, match: re.Match, pattern_name: str, raw_log: str) -> ParsedEvent:
        """Create a ParsedEvent from a regex match."""
        fields = match.groupdict()
        
        # Parse timestamp
        timestamp = self._parse_timestamp(fields.get('timestamp', ''))
        
        # Determine event type and severity
        event_type, severity = self._classify_event(pattern_name, fields, raw_log)
        
        return ParsedEvent(
            timestamp=timestamp,
            event_type=event_type,
            severity=severity,
            source_ip=fields.get('source_ip'),
            destination_ip=fields.get('destination_ip'),
            user=fields.get('user'),
            action=fields.get('action') or fields.get('method'),
            resource=fields.get('resource') or fields.get('path'),
            status=fields.get('status'),
            message=fields.get('message', ''),
            raw_log=raw_log,
            extracted_fields=fields
        )
    
    def _generic_parse(self, log_line: str) -> ParsedEvent:
        """Generic parsing for unrecognized log formats."""
        # Extract timestamp if present
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\d.Z+-]*',
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            r'\w+\s+\d+\s+\d{2}:\d{2}:\d{2}'
        ]
        
        timestamp = datetime.now(timezone.utc)
        for pattern in timestamp_patterns:
            match = re.search(pattern, log_line)
            if match:
                timestamp = self._parse_timestamp(match.group())
                break
        
        # Classify as error, warning, or info based on content
        event_type = EventType.INFO
        severity = SeverityLevel.INFO
        
        line_lower = log_line.lower()
        if any(term in line_lower for term in ['error', 'fail', 'denied', 'unauthorized']):
            event_type = EventType.ERROR
            severity = SeverityLevel.HIGH
        elif any(term in line_lower for term in ['warn', 'warning']):
            event_type = EventType.WARNING
            severity = SeverityLevel.MEDIUM
        
        return ParsedEvent(
            timestamp=timestamp,
            event_type=event_type,
            severity=severity,
            message=log_line.strip(),
            raw_log=log_line
        )
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse various timestamp formats."""
        if not timestamp_str:
            return datetime.now(timezone.utc)
        
        # Common timestamp formats
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%b %d %H:%M:%S',
            '%d/%b/%Y:%H:%M:%S %z'
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        
        # If all parsing fails, return current time
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return datetime.now(timezone.utc)
    
    def _classify_event(self, pattern_name: str, fields: Dict[str, Any], raw_log: str) -> Tuple[EventType, SeverityLevel]:
        """Classify the event type and severity based on pattern and content."""
        classification_map = {
            'auth_success': (EventType.AUTHENTICATION, SeverityLevel.INFO),
            'auth_failure': (EventType.AUTHENTICATION, SeverityLevel.HIGH),
            'sudo_command': (EventType.AUTHORIZATION, SeverityLevel.MEDIUM),
            'failed_login': (EventType.AUTHENTICATION, SeverityLevel.HIGH),
            'privilege_escalation': (EventType.AUTHORIZATION, SeverityLevel.HIGH),
            'file_access': (EventType.FILE_ACCESS, SeverityLevel.MEDIUM),
            'access_log': (EventType.NETWORK_ACCESS, SeverityLevel.INFO),
            'error_log': (EventType.ERROR, SeverityLevel.MEDIUM)
        }
        
        event_type, severity = classification_map.get(pattern_name, (EventType.INFO, SeverityLevel.INFO))
        
        # Adjust severity based on content
        raw_lower = raw_log.lower()
        if any(term in raw_lower for term in ['attack', 'intrusion', 'breach', 'exploit']):
            severity = SeverityLevel.CRITICAL
            event_type = EventType.ATTACK
        elif any(term in raw_lower for term in ['anomaly', 'suspicious', 'unusual']):
            severity = SeverityLevel.HIGH
            event_type = EventType.ANOMALY
        
        # Adjust severity based on HTTP status codes
        if 'status' in fields:
            status_code = fields['status']
            if status_code in ['401', '403']:
                severity = SeverityLevel.HIGH
            elif status_code.startswith('5'):
                severity = SeverityLevel.MEDIUM
        
        return event_type, severity


class EventCorrelator:
    """Advanced event correlation engine for security analysis."""
    
    def __init__(self, max_events: int = 10000):
        self.events = deque(maxlen=max_events)
        self.correlation_rules = []
        self.correlation_cache = {}
        self.active_correlations = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default correlation rules for common attack patterns."""
        default_rules = [
            CorrelationRule(
                name="Brute Force Attack",
                description="Multiple failed authentication attempts from same source",
                event_types=[EventType.AUTHENTICATION],
                time_window=300,  # 5 minutes
                threshold=5,
                severity=SeverityLevel.HIGH,
                pattern={'status': 'failed', 'group_by': 'source_ip'},
                action="alert"
            ),
            CorrelationRule(
                name="Privilege Escalation Chain",
                description="Authentication followed by privilege escalation",
                event_types=[EventType.AUTHENTICATION, EventType.AUTHORIZATION],
                time_window=600,  # 10 minutes
                threshold=2,
                severity=SeverityLevel.CRITICAL,
                pattern={'sequence': True, 'group_by': 'user'},
                action="block"
            ),
            CorrelationRule(
                name="Suspicious File Access",
                description="Multiple sensitive file access attempts",
                event_types=[EventType.FILE_ACCESS],
                time_window=900,  # 15 minutes
                threshold=10,
                severity=SeverityLevel.MEDIUM,
                pattern={'resource_pattern': '/etc/|/var/log/|/root/', 'group_by': 'user'},
                action="monitor"
            ),
            CorrelationRule(
                name="Network Scanning",
                description="Multiple connection attempts to different ports",
                event_types=[EventType.NETWORK_ACCESS],
                time_window=180,  # 3 minutes
                threshold=20,
                severity=SeverityLevel.HIGH,
                pattern={'group_by': 'source_ip', 'diverse_targets': True},
                action="alert"
            ),
            CorrelationRule(
                name="Error Spike",
                description="Unusual spike in error events",
                event_types=[EventType.ERROR],
                time_window=300,  # 5 minutes
                threshold=50,
                severity=SeverityLevel.MEDIUM,
                pattern={'group_by': 'component'},
                action="investigate"
            )
        ]
        
        self.correlation_rules.extend(default_rules)
    
    def add_rule(self, rule: CorrelationRule) -> None:
        """Add a custom correlation rule."""
        self.correlation_rules.append(rule)
    
    def add_event(self, event: ParsedEvent) -> List[CorrelatedEvent]:
        """Add an event and check for correlations."""
        self.events.append(event)
        correlations = []
        
        # Check each correlation rule
        for rule in self.correlation_rules:
            if event.event_type in rule.event_types:
                correlation = self._check_rule(rule, event)
                if correlation:
                    correlations.append(correlation)
        
        return correlations
    
    def _check_rule(self, rule: CorrelationRule, new_event: ParsedEvent) -> Optional[CorrelatedEvent]:
        """Check if a rule is triggered by the new event."""
        # Get events within the time window
        cutoff_time = new_event.timestamp - timedelta(seconds=rule.time_window)
        relevant_events = [
            event for event in self.events
            if event.timestamp >= cutoff_time and event.event_type in rule.event_types
        ]
        
        if len(relevant_events) < rule.threshold:
            return None
        
        # Apply rule-specific pattern matching
        pattern = rule.pattern
        
        if 'group_by' in pattern:
            grouped_events = self._group_events(relevant_events, pattern['group_by'])
            
            for group_key, events in grouped_events.items():
                if len(events) >= rule.threshold:
                    if self._matches_pattern(events, pattern):
                        return self._create_correlation(rule, events)
        
        elif 'sequence' in pattern and pattern['sequence']:
            # Check for event sequences
            if self._check_sequence(relevant_events, rule.event_types):
                return self._create_correlation(rule, relevant_events)
        
        else:
            # Simple threshold-based correlation
            if len(relevant_events) >= rule.threshold:
                return self._create_correlation(rule, relevant_events)
        
        return None
    
    def _group_events(self, events: List[ParsedEvent], group_by: str) -> Dict[str, List[ParsedEvent]]:
        """Group events by a specified field."""
        grouped = defaultdict(list)
        
        for event in events:
            if group_by == 'source_ip':
                key = event.source_ip or 'unknown'
            elif group_by == 'user':
                key = event.user or 'unknown'
            elif group_by == 'component':
                key = event.extracted_fields.get('component', 'unknown')
            else:
                key = getattr(event, group_by, 'unknown')
            
            grouped[key].append(event)
        
        return grouped
    
    def _matches_pattern(self, events: List[ParsedEvent], pattern: Dict[str, Any]) -> bool:
        """Check if events match the specified pattern."""
        if 'status' in pattern:
            status_matches = sum(1 for event in events if event.status == pattern['status'])
            if status_matches < len(events) * 0.8:  # 80% threshold
                return False
        
        if 'resource_pattern' in pattern:
            import re
            resource_pattern = re.compile(pattern['resource_pattern'])
            matching_resources = sum(
                1 for event in events
                if event.resource and resource_pattern.search(event.resource)
            )
            if matching_resources < len(events) * 0.5:  # 50% threshold
                return False
        
        if 'diverse_targets' in pattern and pattern['diverse_targets']:
            # Check for diverse target resources/IPs
            targets = set()
            for event in events:
                target = event.destination_ip or event.resource or ''
                if target:
                    targets.add(target)
            
            if len(targets) < min(10, len(events) * 0.5):
                return False
        
        return True
    
    def _check_sequence(self, events: List[ParsedEvent], event_types: List[EventType]) -> bool:
        """Check if events follow the expected sequence."""
        if len(events) < len(event_types):
            return False
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Look for the sequence in the events
        type_index = 0
        for event in sorted_events:
            if event.event_type == event_types[type_index]:
                type_index += 1
                if type_index >= len(event_types):
                    return True
        
        return False
    
    def _create_correlation(self, rule: CorrelationRule, events: List[ParsedEvent]) -> CorrelatedEvent:
        """Create a correlated event from matching events."""
        events = sorted(events, key=lambda e: e.timestamp)
        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        
        # Generate correlation ID
        correlation_id = hashlib.sha256(
            f"{rule.name}_{start_time}_{end_time}_{len(events)}".encode()
        ).hexdigest()[:16]
        
        # Generate summary
        summary = self._generate_summary(rule, events)
        
        # Extract indicators
        indicators = self._extract_indicators(events)
        
        return CorrelatedEvent(
            correlation_id=correlation_id,
            rule_name=rule.name,
            events=events,
            severity=rule.severity,
            start_time=start_time,
            end_time=end_time,
            summary=summary,
            indicators=indicators
        )
    
    def _generate_summary(self, rule: CorrelationRule, events: List[ParsedEvent]) -> str:
        """Generate a human-readable summary of the correlation."""
        event_count = len(events)
        time_span = (events[-1].timestamp - events[0].timestamp).total_seconds()
        
        # Extract common elements
        sources = set(event.source_ip for event in events if event.source_ip)
        users = set(event.user for event in events if event.user)
        
        summary_parts = [
            f"{rule.description}:",
            f"{event_count} events over {time_span:.0f} seconds"
        ]
        
        if sources:
            summary_parts.append(f"Sources: {', '.join(list(sources)[:3])}")
            if len(sources) > 3:
                summary_parts[-1] += f" and {len(sources) - 3} more"
        
        if users:
            summary_parts.append(f"Users: {', '.join(list(users)[:3])}")
            if len(users) > 3:
                summary_parts[-1] += f" and {len(users) - 3} more"
        
        return " | ".join(summary_parts)
    
    def _extract_indicators(self, events: List[ParsedEvent]) -> Dict[str, Any]:
        """Extract security indicators from correlated events."""
        indicators = {
            'event_count': len(events),
            'unique_sources': len(set(e.source_ip for e in events if e.source_ip)),
            'unique_users': len(set(e.user for e in events if e.user)),
            'time_span': (events[-1].timestamp - events[0].timestamp).total_seconds(),
            'severity_distribution': {},
            'event_types': list(set(e.event_type.value for e in events))
        }
        
        # Count severity levels
        for event in events:
            severity = event.severity.name
            indicators['severity_distribution'][severity] = \
                indicators['severity_distribution'].get(severity, 0) + 1
        
        # Extract common patterns
        if len(events) >= 3:
            # Calculate event frequency
            total_time = indicators['time_span']
            if total_time > 0:
                indicators['events_per_minute'] = (len(events) / total_time) * 60
        
        return indicators


class SecurityAnalyzer:
    """High-level security analysis using parsed and correlated events."""
    
    def __init__(self):
        self.parser = LogParser()
        self.correlator = EventCorrelator()
        self.threat_indicators = {}
        self.baseline_metrics = {}
    
    def analyze_log_stream(self, log_lines: List[str], source_type: str = 'auto') -> Dict[str, Any]:
        """Analyze a stream of log lines for security threats."""
        parsed_events = []
        correlations = []
        
        # Parse all log entries
        for line in log_lines:
            if line.strip():
                event = self.parser.parse_log_entry(line, source_type)
                if event:
                    parsed_events.append(event)
                    # Check for correlations
                    new_correlations = self.correlator.add_event(event)
                    correlations.extend(new_correlations)
        
        # Analyze results
        analysis = {
            'total_events': len(parsed_events),
            'event_types': self._count_by_type(parsed_events),
            'severity_distribution': self._count_by_severity(parsed_events),
            'correlations': len(correlations),
            'high_severity_events': [e for e in parsed_events if e.severity.value >= 4],
            'correlated_events': correlations,
            'threat_score': self._calculate_threat_score(parsed_events, correlations),
            'recommendations': self._generate_recommendations(parsed_events, correlations)
        }
        
        return analysis
    
    def _count_by_type(self, events: List[ParsedEvent]) -> Dict[str, int]:
        """Count events by type."""
        counts = defaultdict(int)
        for event in events:
            counts[event.event_type.value] += 1
        return dict(counts)
    
    def _count_by_severity(self, events: List[ParsedEvent]) -> Dict[str, int]:
        """Count events by severity."""
        counts = defaultdict(int)
        for event in events:
            counts[event.severity.name] += 1
        return dict(counts)
    
    def _calculate_threat_score(self, events: List[ParsedEvent], correlations: List[CorrelatedEvent]) -> float:
        """Calculate an overall threat score (0-100)."""
        score = 0.0
        
        # Base score from event severities
        for event in events:
            score += event.severity.value
        
        # Bonus for correlations
        for correlation in correlations:
            score += correlation.severity.value * 5  # Correlations are more significant
        
        # Normalize to 0-100 scale
        if events:
            max_possible = len(events) * 5 + len(correlations) * 25
            score = min(100, (score / max_possible) * 100)
        
        return round(score, 2)
    
    def _generate_recommendations(self, events: List[ParsedEvent], 
                                correlations: List[CorrelatedEvent]) -> List[str]:
        """Generate security recommendations based on analysis."""
        recommendations = []
        
        # Check for common security issues
        auth_failures = sum(1 for e in events if e.event_type == EventType.AUTHENTICATION and 
                          ('fail' in e.message.lower() or e.status == 'failed'))
        
        if auth_failures > 10:
            recommendations.append(
                f"High number of authentication failures ({auth_failures}). "
                "Consider implementing account lockout policies."
            )
        
        # Check for privilege escalation
        priv_events = [e for e in events if e.event_type == EventType.AUTHORIZATION]
        if len(priv_events) > 5:
            recommendations.append(
                "Multiple privilege escalation events detected. "
                "Review sudo access and monitor privileged operations."
            )
        
        # Check correlations
        for correlation in correlations:
            if correlation.severity.value >= 4:
                recommendations.append(
                    f"Critical correlation detected: {correlation.rule_name}. "
                    f"Immediate investigation recommended."
                )
        
        # Check for anomalies
        anomaly_events = [e for e in events if e.event_type == EventType.ANOMALY]
        if anomaly_events:
            recommendations.append(
                f"{len(anomaly_events)} anomalous events detected. "
                "Review system behavior and investigate outliers."
            )
        
        return recommendations


def main():
    """Demonstration of log parsing and correlation capabilities."""
    analyzer = SecurityAnalyzer()
    
    # Sample log entries for testing
    sample_logs = [
        "Jan 15 10:30:15 server sshd[1234]: Failed password for root from 192.168.1.100 port 22",
        "Jan 15 10:30:20 server sshd[1235]: Failed password for admin from 192.168.1.100 port 22",
        "Jan 15 10:30:25 server sshd[1236]: Failed password for user from 192.168.1.100 port 22",
        "Jan 15 10:30:30 server sshd[1237]: Failed password for test from 192.168.1.100 port 22",
        "Jan 15 10:30:35 server sshd[1238]: Failed password for guest from 192.168.1.100 port 22",
        "Jan 15 10:31:00 server sshd[1239]: Accepted password for root from 192.168.1.100 port 22",
        "Jan 15 10:31:10 server sudo: root : TTY=pts/0 ; PWD=/root ; USER=root ; COMMAND=/bin/cat /etc/shadow",
        '192.168.1.200 - - [15/Jan/2024:10:32:00 +0000] "GET /admin HTTP/1.1" 401 0',
        '192.168.1.200 - - [15/Jan/2024:10:32:05 +0000] "GET /admin HTTP/1.1" 401 0',
        '192.168.1.200 - - [15/Jan/2024:10:32:10 +0000] "GET /login HTTP/1.1" 200 1234'
    ]
    
    print("Security Log Analysis Demonstration")
    print("=" * 50)
    
    # Analyze the sample logs
    analysis = analyzer.analyze_log_stream(sample_logs)
    
    print(f"Total Events: {analysis['total_events']}")
    print(f"Event Types: {analysis['event_types']}")
    print(f"Severity Distribution: {analysis['severity_distribution']}")
    print(f"Correlations Found: {analysis['correlations']}")
    print(f"Threat Score: {analysis['threat_score']}/100")
    print("\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"- {rec}")
    
    print("\nCorrelated Events:")
    for correlation in analysis['correlated_events']:
        print(f"- {correlation.rule_name}: {correlation.summary}")


if __name__ == "__main__":
    main()