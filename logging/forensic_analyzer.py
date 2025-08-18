#!/usr/bin/env python3
"""
Log Retention and Forensic Analysis System
Advanced forensic analysis capabilities with intelligent log retention and investigation tools
"""

import os
import json
import gzip
import sqlite3
import hashlib
import tarfile
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import logging
from pathlib import Path
import pickle
import threading
import queue
import time

# Import from our other modules
from .log_parser import ParsedEvent, EventType, SeverityLevel
from .siem_rules import RuleMatch, ThreatCategory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetentionPolicy(Enum):
    """Log retention policies."""
    CRITICAL = "critical"      # 7 years
    HIGH = "high"             # 3 years  
    MEDIUM = "medium"         # 1 year
    LOW = "low"              # 90 days
    TEMPORARY = "temporary"   # 30 days


class ForensicEventType(Enum):
    """Types of forensic events."""
    LOGIN_PATTERN = "login_pattern"
    FILE_ACCESS = "file_access"
    NETWORK_ACTIVITY = "network_activity"
    PRIVILEGE_CHANGE = "privilege_change"
    SYSTEM_CHANGE = "system_change"
    ANOMALY = "anomaly"


@dataclass
class RetentionConfig:
    """Configuration for log retention."""
    policy: RetentionPolicy
    retention_days: int
    compression_enabled: bool = True
    encryption_enabled: bool = False
    archival_storage: Optional[str] = None
    integrity_checks: bool = True


@dataclass
class ForensicTimeline:
    """Timeline of events for forensic analysis."""
    start_time: datetime
    end_time: datetime
    events: List[ParsedEvent]
    analysis_type: str
    confidence_score: float
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForensicCase:
    """Forensic investigation case."""
    case_id: str
    name: str
    description: str
    created_by: str
    created_date: datetime
    status: str  # open, investigating, closed
    severity: SeverityLevel
    events: List[ParsedEvent] = field(default_factory=list)
    timelines: List[ForensicTimeline] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class LogRetentionManager:
    """Manages log retention policies and archival."""
    
    def __init__(self, storage_path: str = "/var/log/archangel/archives"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize retention policies
        self.retention_policies = {
            RetentionPolicy.CRITICAL: RetentionConfig(
                policy=RetentionPolicy.CRITICAL,
                retention_days=2555,  # 7 years
                compression_enabled=True,
                encryption_enabled=True,
                integrity_checks=True
            ),
            RetentionPolicy.HIGH: RetentionConfig(
                policy=RetentionPolicy.HIGH,
                retention_days=1095,  # 3 years
                compression_enabled=True,
                encryption_enabled=True,
                integrity_checks=True
            ),
            RetentionPolicy.MEDIUM: RetentionConfig(
                policy=RetentionPolicy.MEDIUM,
                retention_days=365,  # 1 year
                compression_enabled=True,
                encryption_enabled=False,
                integrity_checks=True
            ),
            RetentionPolicy.LOW: RetentionConfig(
                policy=RetentionPolicy.LOW,
                retention_days=90,
                compression_enabled=True,
                encryption_enabled=False,
                integrity_checks=False
            ),
            RetentionPolicy.TEMPORARY: RetentionConfig(
                policy=RetentionPolicy.TEMPORARY,
                retention_days=30,
                compression_enabled=False,
                encryption_enabled=False,
                integrity_checks=False
            )
        }
        
        self.db_path = self.storage_path / "retention_metadata.db"
        self._initialize_database()
        
        # Background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self.cleanup_thread.start()
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for retention metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archived_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    archive_path TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    retention_policy TEXT NOT NULL,
                    created_date TIMESTAMP NOT NULL,
                    expiry_date TIMESTAMP NOT NULL,
                    compressed BOOLEAN DEFAULT FALSE,
                    encrypted BOOLEAN DEFAULT FALSE,
                    checksum TEXT,
                    size_bytes INTEGER,
                    event_count INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retention_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    description TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    affected_archives INTEGER,
                    details TEXT
                )
            """)
            
            conn.commit()
    
    def archive_logs(self, events: List[ParsedEvent], 
                    policy: RetentionPolicy = RetentionPolicy.MEDIUM,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Archive a batch of log events."""
        config = self.retention_policies[policy]
        
        # Create archive filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_name = f"logs_{policy.value}_{timestamp}"
        
        if config.compression_enabled:
            archive_path = self.storage_path / f"{archive_name}.tar.gz"
        else:
            archive_path = self.storage_path / f"{archive_name}.json"
        
        # Prepare data for archival
        archive_data = {
            'metadata': {
                'created_date': datetime.now(timezone.utc).isoformat(),
                'retention_policy': policy.value,
                'event_count': len(events),
                'custom_metadata': metadata or {}
            },
            'events': []
        }
        
        # Serialize events
        for event in events:
            event_data = {
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'severity': event.severity.name,
                'source_ip': event.source_ip,
                'destination_ip': event.destination_ip,
                'user': event.user,
                'action': event.action,
                'resource': event.resource,
                'status': event.status,
                'message': event.message,
                'raw_log': event.raw_log,
                'extracted_fields': event.extracted_fields,
                'event_id': event.event_id
            }
            archive_data['events'].append(event_data)
        
        # Write archive
        if config.compression_enabled:
            self._write_compressed_archive(archive_path, archive_data)
        else:
            with open(archive_path, 'w') as f:
                json.dump(archive_data, f, indent=2)
        
        # Calculate checksum
        checksum = self._calculate_file_checksum(archive_path)
        
        # Store metadata in database
        expiry_date = datetime.now(timezone.utc) + timedelta(days=config.retention_days)
        file_size = archive_path.stat().st_size
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO archived_logs 
                (archive_path, original_path, retention_policy, created_date, 
                 expiry_date, compressed, encrypted, checksum, size_bytes, event_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(archive_path), str(archive_path),  # For now, both are the same
                policy.value, datetime.now(timezone.utc), expiry_date,
                config.compression_enabled, config.encryption_enabled,
                checksum, file_size, len(events)
            ))
            
            # Log retention event
            conn.execute("""
                INSERT INTO retention_events
                (event_type, description, timestamp, affected_archives, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                'archive_created',
                f'Created archive with {len(events)} events',
                datetime.now(timezone.utc),
                1,
                json.dumps({'archive_path': str(archive_path), 'policy': policy.value})
            ))
            
            conn.commit()
        
        logger.info(f"Archived {len(events)} events to {archive_path}")
        return str(archive_path)
    
    def _write_compressed_archive(self, archive_path: Path, data: Dict[str, Any]) -> None:
        """Write compressed archive file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(data, temp_file, indent=2)
            temp_path = temp_file.name
        
        try:
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(temp_path, arcname='logs.json')
        finally:
            os.unlink(temp_path)
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def retrieve_archived_logs(self, archive_path: str) -> List[ParsedEvent]:
        """Retrieve logs from an archive."""
        archive_path = Path(archive_path)
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        
        # Verify integrity
        if not self._verify_archive_integrity(archive_path):
            raise ValueError(f"Archive integrity check failed: {archive_path}")
        
        # Load archive data
        if archive_path.suffix == '.gz':
            data = self._load_compressed_archive(archive_path)
        else:
            with open(archive_path, 'r') as f:
                data = json.load(f)
        
        # Reconstruct events
        events = []
        for event_data in data['events']:
            event = ParsedEvent(
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                event_type=EventType(event_data['event_type']),
                severity=SeverityLevel[event_data['severity']],
                source_ip=event_data['source_ip'],
                destination_ip=event_data['destination_ip'],
                user=event_data['user'],
                action=event_data['action'],
                resource=event_data['resource'],
                status=event_data['status'],
                message=event_data['message'],
                raw_log=event_data['raw_log'],
                extracted_fields=event_data['extracted_fields'],
                event_id=event_data['event_id']
            )
            events.append(event)
        
        return events
    
    def _load_compressed_archive(self, archive_path: Path) -> Dict[str, Any]:
        """Load data from compressed archive."""
        with tarfile.open(archive_path, 'r:gz') as tar:
            json_file = tar.extractfile('logs.json')
            if json_file:
                return json.load(json_file)
        raise ValueError(f"Invalid archive format: {archive_path}")
    
    def _verify_archive_integrity(self, archive_path: Path) -> bool:
        """Verify archive integrity using stored checksum."""
        current_checksum = self._calculate_file_checksum(archive_path)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT checksum FROM archived_logs WHERE archive_path = ?",
                (str(archive_path),)
            )
            result = cursor.fetchone()
            
            if result:
                stored_checksum = result[0]
                return current_checksum == stored_checksum
        
        return False  # No checksum found
    
    def _background_cleanup(self) -> None:
        """Background thread for cleaning up expired archives."""
        while True:
            try:
                self._cleanup_expired_archives()
                time.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _cleanup_expired_archives(self) -> None:
        """Clean up expired archives."""
        current_time = datetime.now(timezone.utc)
        
        with sqlite3.connect(self.db_path) as conn:
            # Find expired archives
            cursor = conn.execute("""
                SELECT id, archive_path FROM archived_logs 
                WHERE expiry_date < ?
            """, (current_time,))
            
            expired_archives = cursor.fetchall()
            
            for archive_id, archive_path in expired_archives:
                try:
                    # Delete the archive file
                    if Path(archive_path).exists():
                        os.unlink(archive_path)
                    
                    # Remove from database
                    conn.execute("DELETE FROM archived_logs WHERE id = ?", (archive_id,))
                    
                    # Log cleanup event
                    conn.execute("""
                        INSERT INTO retention_events
                        (event_type, description, timestamp, affected_archives, details)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        'archive_expired',
                        f'Deleted expired archive',
                        current_time,
                        1,
                        json.dumps({'archive_path': archive_path})
                    ))
                    
                    logger.info(f"Cleaned up expired archive: {archive_path}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning up archive {archive_path}: {e}")
            
            conn.commit()


class ForensicAnalyzer:
    """Advanced forensic analysis engine for security investigations."""
    
    def __init__(self, retention_manager: LogRetentionManager):
        self.retention_manager = retention_manager
        self.active_cases = {}
        self.analysis_cache = {}
        
        # Forensic analysis patterns
        self.attack_patterns = self._initialize_attack_patterns()
        
    def _initialize_attack_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize forensic attack patterns."""
        return {
            'lateral_movement': {
                'description': 'Detect lateral movement patterns',
                'indicators': [
                    'SSH connections between internal hosts',
                    'Credential reuse across systems',
                    'Privilege escalation after authentication'
                ],
                'time_window': 3600,  # 1 hour
                'confidence_threshold': 0.7
            },
            'data_exfiltration': {
                'description': 'Detect data exfiltration attempts',
                'indicators': [
                    'Large data transfers',
                    'Unusual file access patterns',
                    'Off-hours activity'
                ],
                'time_window': 7200,  # 2 hours
                'confidence_threshold': 0.6
            },
            'insider_threat': {
                'description': 'Detect insider threat activities',
                'indicators': [
                    'Access to sensitive data',
                    'Unusual working hours',
                    'Privilege abuse'
                ],
                'time_window': 86400,  # 24 hours
                'confidence_threshold': 0.5
            },
            'advanced_persistent_threat': {
                'description': 'Detect APT-style attacks',
                'indicators': [
                    'Long-term low-level activity',
                    'Multiple attack vectors',
                    'Stealthy reconnaissance'
                ],
                'time_window': 604800,  # 1 week
                'confidence_threshold': 0.8
            }
        }
    
    def create_forensic_case(self, name: str, description: str, 
                           created_by: str, severity: SeverityLevel = SeverityLevel.MEDIUM) -> str:
        """Create a new forensic investigation case."""
        case_id = hashlib.sha256(
            f"{name}_{description}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        case = ForensicCase(
            case_id=case_id,
            name=name,
            description=description,
            created_by=created_by,
            created_date=datetime.now(timezone.utc),
            status="open",
            severity=severity
        )
        
        self.active_cases[case_id] = case
        logger.info(f"Created forensic case: {case_id} - {name}")
        
        return case_id
    
    def add_events_to_case(self, case_id: str, events: List[ParsedEvent]) -> None:
        """Add events to a forensic case."""
        if case_id not in self.active_cases:
            raise ValueError(f"Case not found: {case_id}")
        
        case = self.active_cases[case_id]
        case.events.extend(events)
        
        logger.info(f"Added {len(events)} events to case {case_id}")
    
    def analyze_timeline(self, case_id: str, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> ForensicTimeline:
        """Analyze timeline of events in a forensic case."""
        if case_id not in self.active_cases:
            raise ValueError(f"Case not found: {case_id}")
        
        case = self.active_cases[case_id]
        events = case.events
        
        # Filter by time range if specified
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        if not events:
            raise ValueError("No events found in specified time range")
        
        # Sort events by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        # Analyze patterns
        confidence_score = self._calculate_timeline_confidence(events)
        artifacts = self._extract_timeline_artifacts(events)
        
        timeline = ForensicTimeline(
            start_time=events[0].timestamp,
            end_time=events[-1].timestamp,
            events=events,
            analysis_type="temporal_analysis",
            confidence_score=confidence_score,
            artifacts=artifacts
        )
        
        case.timelines.append(timeline)
        return timeline
    
    def _calculate_timeline_confidence(self, events: List[ParsedEvent]) -> float:
        """Calculate confidence score for timeline analysis."""
        if not events:
            return 0.0
        
        # Factors affecting confidence
        factors = {
            'event_count': min(1.0, len(events) / 100),  # More events = higher confidence
            'time_consistency': self._check_time_consistency(events),
            'source_diversity': self._calculate_source_diversity(events),
            'severity_distribution': self._analyze_severity_distribution(events)
        }
        
        # Weighted average
        weights = {'event_count': 0.2, 'time_consistency': 0.3, 
                  'source_diversity': 0.3, 'severity_distribution': 0.2}
        
        confidence = sum(factors[key] * weights[key] for key in factors)
        return min(1.0, confidence)
    
    def _check_time_consistency(self, events: List[ParsedEvent]) -> float:
        """Check temporal consistency of events."""
        if len(events) < 2:
            return 1.0
        
        # Calculate time intervals between events
        intervals = []
        for i in range(1, len(events)):
            interval = (events[i].timestamp - events[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        # Analyze interval consistency
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        
        # Lower variance = higher consistency
        consistency = 1.0 / (1.0 + variance / (avg_interval ** 2))
        return min(1.0, consistency)
    
    def _calculate_source_diversity(self, events: List[ParsedEvent]) -> float:
        """Calculate diversity of event sources."""
        sources = set()
        for event in events:
            if event.source_ip:
                sources.add(event.source_ip)
            if event.user:
                sources.add(event.user)
        
        # More diverse sources = higher score (up to a point)
        diversity = min(1.0, len(sources) / 10)
        return diversity
    
    def _analyze_severity_distribution(self, events: List[ParsedEvent]) -> float:
        """Analyze distribution of event severities."""
        severity_counts = Counter(event.severity for event in events)
        
        # Higher proportion of high-severity events = higher score
        high_severity = (severity_counts[SeverityLevel.CRITICAL] + 
                        severity_counts[SeverityLevel.HIGH])
        
        if len(events) == 0:
            return 0.0
        
        return min(1.0, high_severity / len(events))
    
    def _extract_timeline_artifacts(self, events: List[ParsedEvent]) -> Dict[str, Any]:
        """Extract artifacts from timeline analysis."""
        artifacts = {
            'unique_sources': list(set(e.source_ip for e in events if e.source_ip)),
            'unique_users': list(set(e.user for e in events if e.user)),
            'event_types': list(set(e.event_type.value for e in events)),
            'time_span_hours': (events[-1].timestamp - events[0].timestamp).total_seconds() / 3600,
            'peak_activity_hour': self._find_peak_activity_hour(events),
            'suspicious_patterns': self._identify_suspicious_patterns(events)
        }
        
        return artifacts
    
    def _find_peak_activity_hour(self, events: List[ParsedEvent]) -> int:
        """Find the hour with the highest activity."""
        hour_counts = Counter(event.timestamp.hour for event in events)
        if hour_counts:
            return hour_counts.most_common(1)[0][0]
        return 0
    
    def _identify_suspicious_patterns(self, events: List[ParsedEvent]) -> List[str]:
        """Identify suspicious patterns in the timeline."""
        patterns = []
        
        # Check for unusual time patterns
        hours = [event.timestamp.hour for event in events]
        night_events = sum(1 for hour in hours if hour < 6 or hour > 22)
        if night_events > len(events) * 0.3:
            patterns.append("High proportion of off-hours activity")
        
        # Check for rapid-fire events
        intervals = []
        for i in range(1, len(events)):
            interval = (events[i].timestamp - events[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        rapid_events = sum(1 for interval in intervals if interval < 5)
        if rapid_events > len(intervals) * 0.5:
            patterns.append("Rapid succession of events (possible automation)")
        
        # Check for privilege escalation patterns
        auth_events = [e for e in events if e.event_type == EventType.AUTHENTICATION]
        authz_events = [e for e in events if e.event_type == EventType.AUTHORIZATION]
        
        if auth_events and authz_events:
            patterns.append("Authentication followed by authorization events")
        
        return patterns
    
    def detect_attack_patterns(self, case_id: str) -> Dict[str, Any]:
        """Detect known attack patterns in a forensic case."""
        if case_id not in self.active_cases:
            raise ValueError(f"Case not found: {case_id}")
        
        case = self.active_cases[case_id]
        events = case.events
        
        detected_patterns = {}
        
        for pattern_name, pattern_config in self.attack_patterns.items():
            confidence = self._analyze_attack_pattern(events, pattern_config)
            
            if confidence >= pattern_config['confidence_threshold']:
                detected_patterns[pattern_name] = {
                    'confidence': confidence,
                    'description': pattern_config['description'],
                    'indicators': pattern_config['indicators']
                }
        
        return detected_patterns
    
    def _analyze_attack_pattern(self, events: List[ParsedEvent], 
                              pattern_config: Dict[str, Any]) -> float:
        """Analyze events for a specific attack pattern."""
        # This is a simplified pattern analysis
        # In a real implementation, this would be much more sophisticated
        
        confidence = 0.0
        indicators = pattern_config['indicators']
        
        # Count matching indicators
        matches = 0
        for indicator in indicators:
            if self._check_indicator(events, indicator):
                matches += 1
        
        # Calculate confidence based on indicator matches
        confidence = matches / len(indicators)
        
        return confidence
    
    def _check_indicator(self, events: List[ParsedEvent], indicator: str) -> bool:
        """Check if an indicator is present in the events."""
        indicator_lower = indicator.lower()
        
        for event in events:
            # Check various event fields
            fields_to_check = [
                event.message.lower(),
                event.raw_log.lower(),
                str(event.event_type.value).lower(),
                str(event.action or '').lower(),
                str(event.resource or '').lower()
            ]
            
            for field in fields_to_check:
                if any(term in field for term in indicator_lower.split()):
                    return True
        
        return False
    
    def generate_forensic_report(self, case_id: str) -> Dict[str, Any]:
        """Generate comprehensive forensic report."""
        if case_id not in self.active_cases:
            raise ValueError(f"Case not found: {case_id}")
        
        case = self.active_cases[case_id]
        
        # Detect attack patterns
        attack_patterns = self.detect_attack_patterns(case_id)
        
        # Analyze timelines
        timeline_summaries = []
        for timeline in case.timelines:
            timeline_summaries.append({
                'start_time': timeline.start_time.isoformat(),
                'end_time': timeline.end_time.isoformat(),
                'event_count': len(timeline.events),
                'confidence': timeline.confidence_score,
                'artifacts': timeline.artifacts
            })
        
        # Generate summary statistics
        event_stats = self._generate_event_statistics(case.events)
        
        report = {
            'case_info': {
                'case_id': case.case_id,
                'name': case.name,
                'description': case.description,
                'created_by': case.created_by,
                'created_date': case.created_date.isoformat(),
                'status': case.status,
                'severity': case.severity.name
            },
            'event_statistics': event_stats,
            'attack_patterns': attack_patterns,
            'timelines': timeline_summaries,
            'findings': case.findings,
            'recommendations': case.recommendations,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
        return report
    
    def _generate_event_statistics(self, events: List[ParsedEvent]) -> Dict[str, Any]:
        """Generate statistical summary of events."""
        if not events:
            return {}
        
        stats = {
            'total_events': len(events),
            'time_range': {
                'start': min(e.timestamp for e in events).isoformat(),
                'end': max(e.timestamp for e in events).isoformat(),
                'span_hours': (max(e.timestamp for e in events) - 
                             min(e.timestamp for e in events)).total_seconds() / 3600
            },
            'event_types': dict(Counter(e.event_type.value for e in events)),
            'severity_distribution': dict(Counter(e.severity.name for e in events)),
            'unique_sources': len(set(e.source_ip for e in events if e.source_ip)),
            'unique_users': len(set(e.user for e in events if e.user)),
            'unique_resources': len(set(e.resource for e in events if e.resource))
        }
        
        return stats


def main():
    """Demonstration of forensic analysis capabilities."""
    # Initialize components
    retention_manager = LogRetentionManager()
    forensic_analyzer = ForensicAnalyzer(retention_manager)
    
    print("Forensic Analysis System Demonstration")
    print("=" * 50)
    
    # Create a sample forensic case
    case_id = forensic_analyzer.create_forensic_case(
        name="Suspected Brute Force Attack",
        description="Investigation of suspicious authentication patterns",
        created_by="analyst@archangel.local",
        severity=SeverityLevel.HIGH
    )
    
    print(f"Created forensic case: {case_id}")
    
    # Simulate some events for analysis
    from .log_parser import ParsedEvent, EventType
    
    sample_events = []
    base_time = datetime.now(timezone.utc)
    
    for i in range(10):
        event = ParsedEvent(
            timestamp=base_time + timedelta(minutes=i),
            event_type=EventType.AUTHENTICATION,
            severity=SeverityLevel.HIGH,
            source_ip="192.168.1.100",
            user=f"user{i % 3}",
            message=f"Failed authentication attempt {i}",
            raw_log=f"Auth failure log {i}"
        )
        sample_events.append(event)
    
    # Add events to case
    forensic_analyzer.add_events_to_case(case_id, sample_events)
    
    # Analyze timeline
    timeline = forensic_analyzer.analyze_timeline(case_id)
    print(f"Timeline confidence: {timeline.confidence_score:.2f}")
    print(f"Timeline artifacts: {timeline.artifacts}")
    
    # Detect attack patterns
    patterns = forensic_analyzer.detect_attack_patterns(case_id)
    print(f"Detected patterns: {list(patterns.keys())}")
    
    # Generate report
    report = forensic_analyzer.generate_forensic_report(case_id)
    print(f"Report generated with {report['event_statistics']['total_events']} events")
    
    # Demonstrate archival
    archive_path = retention_manager.archive_logs(
        sample_events, 
        RetentionPolicy.HIGH,
        metadata={'case_id': case_id, 'investigation': 'brute_force'}
    )
    print(f"Events archived to: {archive_path}")


if __name__ == "__main__":
    main()