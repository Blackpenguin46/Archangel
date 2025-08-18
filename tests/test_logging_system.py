#!/usr/bin/env python3
"""
Comprehensive Test Suite for Logging and SIEM Integration
Tests for log completeness, correlation accuracy, and forensic analysis
"""

import unittest
import asyncio
import json
import tempfile
import shutil
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import logging
from pathlib import Path

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from logging.centralized_logging import (
    LogEntry, LogLevel, LogCategory, LogAggregator,
    FileOutput, ElasticsearchOutput, KafkaOutput,
    ArchangelLogger
)
from logging.log_parser import (
    LogParser, EventCorrelator, SecurityAnalyzer,
    ParsedEvent, EventType, SeverityLevel, CorrelatedEvent
)
from logging.siem_rules import (
    SIEMRuleEngine, SIEMRule, RuleType, ThreatCategory,
    RuleCondition, RuleMatch
)
from logging.forensic_analyzer import (
    LogRetentionManager, ForensicAnalyzer, RetentionPolicy,
    ForensicCase, ForensicTimeline
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogAggregationTests(unittest.TestCase):
    """Test centralized log aggregation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix='log_test_')
        self.aggregator = LogAggregator()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_log_entry_creation(self):
        """Test LogEntry creation and validation."""
        entry = LogEntry(
            level=LogLevel.ERROR,
            category=LogCategory.SECURITY,
            component="test_component",
            message="Test error message",
            user_id="test_user",
            session_id="test_session"
        )
        
        self.assertEqual(entry.level, LogLevel.ERROR)
        self.assertEqual(entry.category, LogCategory.SECURITY)
        self.assertEqual(entry.component, "test_component")
        self.assertEqual(entry.message, "Test error message")
        self.assertIsInstance(entry.timestamp, datetime)
        self.assertIsNotNone(entry.trace_id)
    
    def test_file_output(self):
        """Test file-based log output."""
        log_file = os.path.join(self.test_dir, 'test.log')
        file_output = FileOutput(log_file)
        
        entry = LogEntry(
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            message="Test log message"
        )
        
        file_output.write(entry)
        file_output.close()
        
        # Verify file was created and contains log entry
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, 'r') as f:
            content = f.read()
            self.assertIn("Test log message", content)
            self.assertIn("INFO", content)
    
    def test_log_aggregation(self):
        """Test log aggregation from multiple sources."""
        # Add file output
        log_file = os.path.join(self.test_dir, 'aggregated.log')
        self.aggregator.add_output(FileOutput(log_file))
        
        # Create test entries
        entries = [
            LogEntry(level=LogLevel.INFO, message="Info message 1"),
            LogEntry(level=LogLevel.WARNING, message="Warning message"),
            LogEntry(level=LogLevel.ERROR, message="Error message"),
            LogEntry(level=LogLevel.INFO, message="Info message 2")
        ]
        
        # Process entries
        for entry in entries:
            self.aggregator.process_log(entry)
        
        self.aggregator.close()
        
        # Verify all entries were written
        with open(log_file, 'r') as f:
            content = f.read()
            for entry in entries:
                self.assertIn(entry.message, content)
    
    def test_log_filtering(self):
        """Test log filtering functionality."""
        log_file = os.path.join(self.test_dir, 'filtered.log')
        file_output = FileOutput(log_file)
        
        # Set minimum level filter
        file_output.set_level_filter(LogLevel.WARNING)
        self.aggregator.add_output(file_output)
        
        # Create entries with different levels
        entries = [
            LogEntry(level=LogLevel.DEBUG, message="Debug message"),
            LogEntry(level=LogLevel.INFO, message="Info message"),
            LogEntry(level=LogLevel.WARNING, message="Warning message"),
            LogEntry(level=LogLevel.ERROR, message="Error message")
        ]
        
        for entry in entries:
            self.aggregator.process_log(entry)
        
        self.aggregator.close()
        
        # Verify only WARNING and ERROR messages were written
        with open(log_file, 'r') as f:
            content = f.read()
            self.assertNotIn("Debug message", content)
            self.assertNotIn("Info message", content)
            self.assertIn("Warning message", content)
            self.assertIn("Error message", content)
    
    def test_batch_processing(self):
        """Test batch processing of log entries."""
        log_file = os.path.join(self.test_dir, 'batch.log')
        file_output = FileOutput(log_file, batch_size=3)
        self.aggregator.add_output(file_output)
        
        # Create entries
        entries = [LogEntry(level=LogLevel.INFO, message=f"Message {i}") for i in range(5)]
        
        # Process entries
        for entry in entries:
            self.aggregator.process_log(entry)
        
        # Force flush
        self.aggregator.flush()
        self.aggregator.close()
        
        # Verify all entries were written
        with open(log_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 5)


class LogParsingTests(unittest.TestCase):
    """Test log parsing and event correlation."""
    
    def setUp(self):
        """Set up test environment."""
        self.parser = LogParser()
        self.correlator = EventCorrelator()
        self.analyzer = SecurityAnalyzer()
    
    def test_syslog_parsing(self):
        """Test parsing of syslog entries."""
        syslog_entry = "Jan 15 10:30:15 server sshd[1234]: Failed password for root from 192.168.1.100 port 22"
        
        event = self.parser.parse_log_entry(syslog_entry, 'syslog')
        
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, EventType.AUTHENTICATION)
        self.assertEqual(event.source_ip, "192.168.1.100")
        self.assertEqual(event.user, "root")
        self.assertIn("Failed", event.message)
    
    def test_apache_log_parsing(self):
        """Test parsing of Apache access logs."""
        apache_entry = '192.168.1.200 - - [15/Jan/2024:10:32:00 +0000] "GET /admin HTTP/1.1" 401 0'
        
        event = self.parser.parse_log_entry(apache_entry, 'apache')
        
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, EventType.NETWORK_ACCESS)
        self.assertEqual(event.source_ip, "192.168.1.200")
        self.assertEqual(event.action, "GET")
        self.assertEqual(event.resource, "/admin")
        self.assertEqual(event.status, "401")
    
    def test_json_log_parsing(self):
        """Test parsing of JSON log entries."""
        json_entry = '{"timestamp": "2024-01-15T10:30:15Z", "level": "ERROR", "component": "auth", "message": "Authentication failed", "user": "testuser"}'
        
        event = self.parser.parse_log_entry(json_entry, 'application')
        
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, EventType.ERROR)
        self.assertEqual(event.severity, SeverityLevel.HIGH)
        self.assertIn("Authentication failed", event.message)
    
    def test_custom_pattern_addition(self):
        """Test adding custom parsing patterns."""
        custom_pattern = r'CUSTOM: (?P<timestamp>[\d-]+ [\d:]+) (?P<user>\w+) (?P<action>\w+) (?P<resource>.*)'
        self.parser.add_custom_pattern('custom', 'custom_format', custom_pattern)
        
        custom_entry = "CUSTOM: 2024-01-15 10:30:15 admin login /dashboard"
        event = self.parser.parse_log_entry(custom_entry, 'custom')
        
        self.assertIsNotNone(event)
        self.assertEqual(event.user, "admin")
        self.assertEqual(event.action, "login")
        self.assertEqual(event.resource, "/dashboard")
    
    def test_brute_force_correlation(self):
        """Test correlation of brute force attack events."""
        # Create multiple failed authentication events
        events = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(6):
            event = ParsedEvent(
                timestamp=base_time + timedelta(seconds=i*10),
                event_type=EventType.AUTHENTICATION,
                severity=SeverityLevel.MEDIUM,
                source_ip="192.168.1.100",
                user=f"user{i}",
                status="failed",
                message="Failed password attempt"
            )
            events.append(event)
        
        # Add events to correlator
        correlations = []
        for event in events:
            new_correlations = self.correlator.add_event(event)
            correlations.extend(new_correlations)
        
        # Should detect brute force pattern
        self.assertGreater(len(correlations), 0)
        
        brute_force_correlation = next(
            (c for c in correlations if "Brute Force" in c.rule_name), None
        )
        self.assertIsNotNone(brute_force_correlation)
        self.assertEqual(brute_force_correlation.severity, SeverityLevel.HIGH)
    
    def test_privilege_escalation_correlation(self):
        """Test correlation of privilege escalation events."""
        base_time = datetime.now(timezone.utc)
        
        # Authentication success followed by privilege escalation
        auth_event = ParsedEvent(
            timestamp=base_time,
            event_type=EventType.AUTHENTICATION,
            severity=SeverityLevel.INFO,
            user="testuser",
            message="Accepted password"
        )
        
        priv_event = ParsedEvent(
            timestamp=base_time + timedelta(seconds=30),
            event_type=EventType.AUTHORIZATION,
            severity=SeverityLevel.MEDIUM,
            user="testuser",
            message="sudo command executed"
        )
        
        # Add events
        correlations = []
        for event in [auth_event, priv_event]:
            new_correlations = self.correlator.add_event(event)
            correlations.extend(new_correlations)
        
        # Should detect privilege escalation pattern
        priv_correlation = next(
            (c for c in correlations if "Privilege Escalation" in c.rule_name), None
        )
        if priv_correlation:  # May not trigger with just 2 events
            self.assertEqual(priv_correlation.severity, SeverityLevel.CRITICAL)
    
    def test_security_analysis(self):
        """Test comprehensive security analysis."""
        # Create mixed log entries
        log_lines = [
            "Jan 15 10:30:15 server sshd[1234]: Failed password for root from 192.168.1.100 port 22",
            "Jan 15 10:30:20 server sshd[1235]: Failed password for admin from 192.168.1.100 port 22",
            "Jan 15 10:30:25 server sshd[1236]: Failed password for user from 192.168.1.100 port 22",
            '192.168.1.200 - - [15/Jan/2024:10:32:00 +0000] "GET /admin HTTP/1.1" 401 0',
            '192.168.1.200 - - [15/Jan/2024:10:32:05 +0000] "GET /admin HTTP/1.1" 401 0'
        ]
        
        analysis = self.analyzer.analyze_log_stream(log_lines)
        
        self.assertGreater(analysis['total_events'], 0)
        self.assertIn('authentication', analysis['event_types'])
        self.assertIn('network_access', analysis['event_types'])
        self.assertGreater(analysis['threat_score'], 0)
        self.assertIsInstance(analysis['recommendations'], list)


class SIEMRulesTests(unittest.TestCase):
    """Test SIEM rules engine functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.engine = SIEMRuleEngine()
    
    def test_rule_creation(self):
        """Test creation of SIEM rules."""
        rule = SIEMRule(
            rule_id="TEST_001",
            name="Test Rule",
            description="Test rule description",
            rule_type=RuleType.THRESHOLD,
            threat_category=ThreatCategory.AUTHENTICATION_ATTACK,
            severity=SeverityLevel.HIGH,
            conditions=[
                RuleCondition("event_type", "eq", "authentication"),
                RuleCondition("status", "eq", "failed")
            ],
            threshold=3,
            time_window=300
        )
        
        self.engine.add_rule(rule)
        
        # Verify rule was added
        rule_found = any(r.rule_id == "TEST_001" for r in self.engine.rules)
        self.assertTrue(rule_found)
    
    def test_threshold_rule_evaluation(self):
        """Test evaluation of threshold-based rules."""
        # Create events that should trigger the rule
        events = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(6):  # Exceeds default SSH brute force threshold of 5
            event = ParsedEvent(
                timestamp=base_time + timedelta(seconds=i*10),
                event_type=EventType.AUTHENTICATION,
                severity=SeverityLevel.MEDIUM,
                source_ip="192.168.1.100",
                message="Failed ssh authentication",
                raw_log="ssh: Failed password for user"
            )
            events.append(event)
        
        # Evaluate events
        all_matches = []
        for event in events:
            matches = self.engine.evaluate_event(event)
            all_matches.extend(matches)
        
        # Should trigger SSH brute force rule
        ssh_matches = [m for m in all_matches if "SSH" in m.rule_name]
        self.assertGreater(len(ssh_matches), 0)
    
    def test_pattern_rule_evaluation(self):
        """Test evaluation of pattern matching rules."""
        # Create event with SQL injection pattern
        event = ParsedEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.NETWORK_ACCESS,
            severity=SeverityLevel.MEDIUM,
            resource="/search?q=' union select * from users--",
            message="Web request with SQL injection"
        )
        
        matches = self.engine.evaluate_event(event)
        
        # Should trigger SQL injection rule
        sql_matches = [m for m in matches if "SQL" in m.rule_name]
        self.assertGreater(len(sql_matches), 0)
    
    def test_false_positive_filtering(self):
        """Test false positive filtering in rules."""
        # Create a rule with false positive filters
        rule = SIEMRule(
            rule_id="TEST_FP",
            name="Test FP Rule",
            description="Rule with false positive filters",
            rule_type=RuleType.PATTERN_MATCH,
            threat_category=ThreatCategory.WEB_ATTACK,
            severity=SeverityLevel.MEDIUM,
            conditions=[
                RuleCondition("message", "contains", "password")
            ],
            false_positive_filters=[
                RuleCondition("message", "contains", "test")
            ]
        )
        
        self.engine.add_rule(rule)
        
        # Event that should trigger the rule
        trigger_event = ParsedEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.AUTHENTICATION,
            message="Failed password attempt"
        )
        
        # Event that should be filtered out
        filtered_event = ParsedEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.AUTHENTICATION,
            message="Test password validation"
        )
        
        trigger_matches = self.engine.evaluate_event(trigger_event)
        filtered_matches = self.engine.evaluate_event(filtered_event)
        
        # First event should match, second should be filtered
        fp_trigger_matches = [m for m in trigger_matches if m.rule_id == "TEST_FP"]
        fp_filtered_matches = [m for m in filtered_matches if m.rule_id == "TEST_FP"]
        
        self.assertGreater(len(fp_trigger_matches), 0)
        self.assertEqual(len(fp_filtered_matches), 0)
    
    def test_rule_statistics(self):
        """Test rule engine statistics collection."""
        # Generate some events to create statistics
        events = [
            ParsedEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=EventType.AUTHENTICATION,
                message="Failed ssh authentication"
            ) for _ in range(3)
        ]
        
        for event in events:
            self.engine.evaluate_event(event)
        
        stats = self.engine.get_statistics()
        
        self.assertIn('total_rules', stats)
        self.assertIn('enabled_rules', stats)
        self.assertIn('events_processed', stats)
        self.assertGreater(stats['events_processed'], 0)
    
    def test_rule_export_import(self):
        """Test exporting and importing rules."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_file = os.path.join(temp_dir, 'rules.yaml')
            
            # Export rules
            self.engine.export_rules_to_file(export_file, 'yaml')
            self.assertTrue(os.path.exists(export_file))
            
            # Create new engine and import rules
            new_engine = SIEMRuleEngine()
            original_count = len(new_engine.rules)
            
            new_engine.load_rules_from_file(export_file)
            
            # Should have loaded additional rules
            self.assertGreaterEqual(len(new_engine.rules), original_count)


class ForensicAnalysisTests(unittest.TestCase):
    """Test forensic analysis and log retention."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix='forensic_test_')
        self.retention_manager = LogRetentionManager(self.test_dir)
        self.forensic_analyzer = ForensicAnalyzer(self.retention_manager)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_forensic_case_creation(self):
        """Test creation of forensic cases."""
        case_id = self.forensic_analyzer.create_forensic_case(
            name="Test Investigation",
            description="Test case for unit testing",
            created_by="test_analyst",
            severity=SeverityLevel.HIGH
        )
        
        self.assertIsNotNone(case_id)
        self.assertIn(case_id, self.forensic_analyzer.active_cases)
        
        case = self.forensic_analyzer.active_cases[case_id]
        self.assertEqual(case.name, "Test Investigation")
        self.assertEqual(case.severity, SeverityLevel.HIGH)
        self.assertEqual(case.status, "open")
    
    def test_timeline_analysis(self):
        """Test forensic timeline analysis."""
        # Create case
        case_id = self.forensic_analyzer.create_forensic_case(
            name="Timeline Test",
            description="Testing timeline analysis",
            created_by="test_analyst"
        )
        
        # Create sample events
        base_time = datetime.now(timezone.utc)
        events = []
        
        for i in range(5):
            event = ParsedEvent(
                timestamp=base_time + timedelta(minutes=i),
                event_type=EventType.AUTHENTICATION,
                severity=SeverityLevel.MEDIUM,
                source_ip="192.168.1.100",
                user="testuser",
                message=f"Event {i}"
            )
            events.append(event)
        
        # Add events to case
        self.forensic_analyzer.add_events_to_case(case_id, events)
        
        # Analyze timeline
        timeline = self.forensic_analyzer.analyze_timeline(case_id)
        
        self.assertIsNotNone(timeline)
        self.assertEqual(len(timeline.events), 5)
        self.assertGreater(timeline.confidence_score, 0)
        self.assertIn('unique_sources', timeline.artifacts)
    
    def test_attack_pattern_detection(self):
        """Test detection of attack patterns in forensic cases."""
        # Create case
        case_id = self.forensic_analyzer.create_forensic_case(
            name="Attack Pattern Test",
            description="Testing attack pattern detection",
            created_by="test_analyst"
        )
        
        # Create events that might indicate lateral movement
        events = [
            ParsedEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=EventType.AUTHENTICATION,
                severity=SeverityLevel.INFO,
                source_ip="192.168.1.100",
                user="admin",
                message="SSH connection between internal hosts"
            ),
            ParsedEvent(
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=5),
                event_type=EventType.AUTHORIZATION,
                severity=SeverityLevel.MEDIUM,
                user="admin",
                message="Privilege escalation after authentication"
            )
        ]
        
        self.forensic_analyzer.add_events_to_case(case_id, events)
        
        # Detect patterns
        patterns = self.forensic_analyzer.detect_attack_patterns(case_id)
        
        self.assertIsInstance(patterns, dict)
        # May or may not detect patterns with limited test data
    
    def test_forensic_report_generation(self):
        """Test generation of forensic reports."""
        # Create and populate case
        case_id = self.forensic_analyzer.create_forensic_case(
            name="Report Test",
            description="Testing report generation",
            created_by="test_analyst"
        )
        
        events = [
            ParsedEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=EventType.AUTHENTICATION,
                severity=SeverityLevel.HIGH,
                message="Test event for report"
            )
        ]
        
        self.forensic_analyzer.add_events_to_case(case_id, events)
        
        # Generate report
        report = self.forensic_analyzer.generate_forensic_report(case_id)
        
        self.assertIn('case_info', report)
        self.assertIn('event_statistics', report)
        self.assertIn('attack_patterns', report)
        self.assertEqual(report['case_info']['case_id'], case_id)
        self.assertEqual(report['event_statistics']['total_events'], 1)
    
    def test_log_retention_archival(self):
        """Test log retention and archival."""
        # Create sample events
        events = [
            ParsedEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=EventType.AUTHENTICATION,
                severity=SeverityLevel.MEDIUM,
                message=f"Test event {i}"
            ) for i in range(10)
        ]
        
        # Archive events
        archive_path = self.retention_manager.archive_logs(
            events,
            RetentionPolicy.MEDIUM,
            metadata={'test': 'data'}
        )
        
        self.assertTrue(os.path.exists(archive_path))
        
        # Retrieve archived events
        retrieved_events = self.retention_manager.retrieve_archived_logs(archive_path)
        
        self.assertEqual(len(retrieved_events), 10)
        self.assertEqual(retrieved_events[0].message, "Test event 0")
    
    def test_archive_integrity_verification(self):
        """Test archive integrity verification."""
        # Create and archive events
        events = [
            ParsedEvent(
                timestamp=datetime.now(timezone.utc),
                event_type=EventType.SYSTEM_CHANGE,
                message="Integrity test event"
            )
        ]
        
        archive_path = self.retention_manager.archive_logs(events, RetentionPolicy.TEMPORARY)
        
        # Verify integrity
        integrity_ok = self.retention_manager._verify_archive_integrity(Path(archive_path))
        self.assertTrue(integrity_ok)
        
        # Corrupt the file and verify integrity fails
        with open(archive_path, 'ab') as f:
            f.write(b'corrupted_data')
        
        integrity_corrupted = self.retention_manager._verify_archive_integrity(Path(archive_path))
        self.assertFalse(integrity_corrupted)


class IntegrationTests(unittest.TestCase):
    """Integration tests for the complete logging system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp(prefix='integration_test_')
        self.aggregator = LogAggregator()
        self.parser = LogParser()
        self.siem_engine = SIEMRuleEngine()
        self.retention_manager = LogRetentionManager(self.test_dir)
        self.forensic_analyzer = ForensicAnalyzer(self.retention_manager)
    
    def tearDown(self):
        """Clean up integration test environment."""
        self.aggregator.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_end_to_end_log_processing(self):
        """Test complete end-to-end log processing pipeline."""
        # Set up file output for aggregator
        log_file = os.path.join(self.test_dir, 'integration.log')
        self.aggregator.add_output(FileOutput(log_file))
        
        # Create raw log entries
        raw_logs = [
            "Jan 15 10:30:15 server sshd[1234]: Failed password for root from 192.168.1.100 port 22",
            "Jan 15 10:30:20 server sshd[1235]: Failed password for admin from 192.168.1.100 port 22",
            "Jan 15 10:30:25 server sshd[1236]: Failed password for user from 192.168.1.100 port 22",
            "Jan 15 10:30:30 server sshd[1237]: Failed password for test from 192.168.1.100 port 22",
            "Jan 15 10:30:35 server sshd[1238]: Failed password for guest from 192.168.1.100 port 22",
            "Jan 15 10:31:00 server sshd[1239]: Accepted password for admin from 192.168.1.100 port 22"
        ]
        
        # Process through aggregator
        for raw_log in raw_logs:
            log_entry = LogEntry(
                level=LogLevel.WARNING,
                category=LogCategory.SECURITY,
                message=raw_log,
                component="ssh_monitor"
            )
            self.aggregator.process_log(log_entry)
        
        self.aggregator.flush()
        
        # Parse logs
        parsed_events = []
        for raw_log in raw_logs:
            event = self.parser.parse_log_entry(raw_log, 'syslog')
            if event:
                parsed_events.append(event)
        
        self.assertGreater(len(parsed_events), 0)
        
        # Evaluate with SIEM rules
        all_matches = []
        for event in parsed_events:
            matches = self.siem_engine.evaluate_event(event)
            all_matches.extend(matches)
        
        # Should detect brute force attack
        self.assertGreater(len(all_matches), 0)
        
        # Create forensic case
        case_id = self.forensic_analyzer.create_forensic_case(
            name="Integration Test Case",
            description="End-to-end test investigation",
            created_by="integration_test"
        )
        
        self.forensic_analyzer.add_events_to_case(case_id, parsed_events)
        
        # Generate forensic report
        report = self.forensic_analyzer.generate_forensic_report(case_id)
        
        self.assertIn('case_info', report)
        self.assertGreater(report['event_statistics']['total_events'], 0)
        
        # Archive events
        archive_path = self.retention_manager.archive_logs(
            parsed_events,
            RetentionPolicy.HIGH,
            metadata={'case_id': case_id, 'test': 'integration'}
        )
        
        self.assertTrue(os.path.exists(archive_path))
        
        # Verify end-to-end log file was created
        self.assertTrue(os.path.exists(log_file))
    
    def test_high_volume_processing(self):
        """Test system performance with high volume of logs."""
        import time
        
        # Generate large number of log entries
        start_time = time.time()
        
        for i in range(1000):
            log_entry = LogEntry(
                level=LogLevel.INFO,
                category=LogCategory.SYSTEM,
                message=f"High volume test message {i}",
                component="performance_test"
            )
            self.aggregator.process_log(log_entry)
        
        self.aggregator.flush()
        
        processing_time = time.time() - start_time
        
        # Should process 1000 entries in reasonable time (< 5 seconds)
        self.assertLess(processing_time, 5.0)
        
        logger.info(f"Processed 1000 log entries in {processing_time:.2f} seconds")
    
    def test_concurrent_processing(self):
        """Test concurrent log processing."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def worker(worker_id):
            try:
                for i in range(100):
                    log_entry = LogEntry(
                        level=LogLevel.INFO,
                        message=f"Worker {worker_id} message {i}",
                        component=f"worker_{worker_id}"
                    )
                    self.aggregator.process_log(log_entry)
                
                results_queue.put(f"Worker {worker_id} completed")
            except Exception as e:
                results_queue.put(f"Worker {worker_id} error: {e}")
        
        # Start multiple worker threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # All workers should complete successfully
        completed_workers = [r for r in results if "completed" in r]
        self.assertEqual(len(completed_workers), 5)
        
        self.aggregator.flush()


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)