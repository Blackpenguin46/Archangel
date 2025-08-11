#!/usr/bin/env python3
"""
Tests for Archangel Autonomous AI Evolution - Audit and Replay System
"""

import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import the audit and replay system
from agents.audit_replay import (
    AuditReplaySystem, AuditEvent, ReplayEvent, AuditSession,
    AuditEventType, ReplayEventType, IntegrityLevel,
    AuditDatabase, CryptographicIntegrity, SessionReplayEngine,
    create_audit_replay_system
)

class TestAuditEvent(unittest.TestCase):
    """Test AuditEvent data structure"""
    
    def test_audit_event_creation(self):
        """Test creating an audit event"""
        event = AuditEvent(
            event_id="test_event_001",
            timestamp=datetime.now(),
            event_type=AuditEventType.AGENT_DECISION,
            agent_id="test_agent",
            session_id="test_session",
            sequence_number=1,
            event_data={"test": "data"},
            prompt="Test prompt",
            reasoning="Test reasoning",
            action_taken="test_action",
            confidence_score=0.85
        )
        
        self.assertEqual(event.event_id, "test_event_001")
        self.assertEqual(event.event_type, AuditEventType.AGENT_DECISION)
        self.assertEqual(event.agent_id, "test_agent")
        self.assertEqual(event.confidence_score, 0.85)
        self.assertEqual(len(event.tags), 0)
        self.assertEqual(len(event.related_events), 0)
    
    def test_audit_event_with_tags(self):
        """Test audit event with tags and related events"""
        event = AuditEvent(
            event_id="test_event_002",
            timestamp=datetime.now(),
            event_type=AuditEventType.LLM_PROMPT,
            agent_id="test_agent",
            session_id="test_session",
            sequence_number=2,
            event_data={"model": "gpt-4"},
            tags=["llm", "reasoning"],
            related_events=["test_event_003"]
        )
        
        self.assertEqual(len(event.tags), 2)
        self.assertIn("llm", event.tags)
        self.assertIn("reasoning", event.tags)
        self.assertEqual(len(event.related_events), 1)
        self.assertIn("test_event_003", event.related_events)

class TestReplayEvent(unittest.TestCase):
    """Test ReplayEvent data structure"""
    
    def test_replay_event_creation(self):
        """Test creating a replay event"""
        event = ReplayEvent(
            event_id="replay_event_001",
            timestamp=datetime.now(),
            event_type=ReplayEventType.DECISION_POINT,
            agent_id="test_agent",
            session_id="test_session",
            sequence_number=1,
            pre_state={"state": "before"},
            post_state={"state": "after"},
            event_details={"action": "decision_made"}
        )
        
        self.assertEqual(event.event_id, "replay_event_001")
        self.assertEqual(event.event_type, ReplayEventType.DECISION_POINT)
        self.assertTrue(event.can_replay)
        self.assertIsNone(event.replay_duration)

class TestCryptographicIntegrity(unittest.TestCase):
    """Test cryptographic integrity verification"""
    
    def setUp(self):
        self.integrity = CryptographicIntegrity()
    
    def test_hash_calculation(self):
        """Test SHA-256 hash calculation"""
        data = "test data for hashing"
        hash1 = self.integrity.calculate_hash(data)
        hash2 = self.integrity.calculate_hash(data)
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA-256 produces 64-character hex string
    
    def test_hmac_calculation(self):
        """Test HMAC calculation"""
        data = "test data for hmac"
        
        # HMAC requires a key
        if self.integrity.hmac_key:
            hmac1 = self.integrity.calculate_hmac(data)
            hmac2 = self.integrity.calculate_hmac(data)
            
            self.assertEqual(hmac1, hmac2)
            self.assertEqual(len(hmac1), 64)  # HMAC-SHA256 produces 64-character hex string
    
    def test_integrity_addition_hash(self):
        """Test adding hash integrity to audit event"""
        event = AuditEvent(
            event_id="test_event_001",
            timestamp=datetime.now(),
            event_type=AuditEventType.AGENT_DECISION,
            agent_id="test_agent",
            session_id="test_session",
            sequence_number=1,
            event_data={"test": "data"},
            integrity_level=IntegrityLevel.HASH
        )
        
        self.integrity.add_integrity(event)
        self.assertIsNotNone(event.checksum)
        self.assertEqual(len(event.checksum), 64)
    
    def test_integrity_verification_hash(self):
        """Test verifying hash integrity"""
        event = AuditEvent(
            event_id="test_event_001",
            timestamp=datetime.now(),
            event_type=AuditEventType.AGENT_DECISION,
            agent_id="test_agent",
            session_id="test_session",
            sequence_number=1,
            event_data={"test": "data"},
            integrity_level=IntegrityLevel.HASH
        )
        
        self.integrity.add_integrity(event)
        self.assertTrue(self.integrity.verify_integrity(event))
        
        # Tamper with event
        event.event_data["test"] = "tampered"
        self.assertFalse(self.integrity.verify_integrity(event))
    
    @patch('agents.audit_replay.CRYPTOGRAPHY_AVAILABLE', True)
    def test_key_generation(self):
        """Test RSA key pair generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                private_key_path, public_key_path = self.integrity.generate_keys(temp_dir)
                
                self.assertTrue(Path(private_key_path).exists())
                self.assertTrue(Path(public_key_path).exists())
                
                # Verify keys are loaded
                self.assertIsNotNone(self.integrity.private_key)
                self.assertIsNotNone(self.integrity.public_key)
                
            except RuntimeError:
                # Skip if cryptography not available
                self.skipTest("Cryptography library not available")

class TestAuditDatabase(unittest.TestCase):
    """Test audit database functionality"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.database = AuditDatabase(self.db_path)
    
    def tearDown(self):
        if hasattr(self, 'database') and self.database.connection:
            self.database.connection.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    async def test_database_initialization(self):
        """Test database initialization"""
        await self.database.initialize()
        
        # Check that tables exist
        cursor = self.database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['audit_events', 'replay_events', 'audit_sessions', 'audit_search']
        for table in expected_tables:
            self.assertIn(table, tables)
    
    async def test_store_audit_event(self):
        """Test storing audit events"""
        await self.database.initialize()
        
        event = AuditEvent(
            event_id="test_event_001",
            timestamp=datetime.now(),
            event_type=AuditEventType.AGENT_DECISION,
            agent_id="test_agent",
            session_id="test_session",
            sequence_number=1,
            event_data={"test": "data"},
            prompt="Test prompt",
            reasoning="Test reasoning",
            tags=["test", "audit"]
        )
        
        await self.database.store_audit_event(event)
        
        # Verify event was stored
        cursor = self.database.connection.execute(
            "SELECT * FROM audit_events WHERE event_id = ?",
            (event.event_id,)
        )
        row = cursor.fetchone()
        
        self.assertIsNotNone(row)
        self.assertEqual(row['event_id'], event.event_id)
        self.assertEqual(row['agent_id'], event.agent_id)
        self.assertEqual(row['prompt'], event.prompt)
    
    async def test_store_replay_event(self):
        """Test storing replay events"""
        await self.database.initialize()
        
        event = ReplayEvent(
            event_id="replay_event_001",
            timestamp=datetime.now(),
            event_type=ReplayEventType.DECISION_POINT,
            agent_id="test_agent",
            session_id="test_session",
            sequence_number=1,
            pre_state={"state": "before"},
            post_state={"state": "after"},
            event_details={"action": "decision_made"}
        )
        
        await self.database.store_replay_event(event)
        
        # Verify event was stored
        cursor = self.database.connection.execute(
            "SELECT * FROM replay_events WHERE event_id = ?",
            (event.event_id,)
        )
        row = cursor.fetchone()
        
        self.assertIsNotNone(row)
        self.assertEqual(row['event_id'], event.event_id)
        self.assertEqual(row['agent_id'], event.agent_id)
    
    async def test_search_events(self):
        """Test searching audit events"""
        await self.database.initialize()
        
        # Store multiple events
        events = []
        for i in range(5):
            event = AuditEvent(
                event_id=f"test_event_{i:03d}",
                timestamp=datetime.now(),
                event_type=AuditEventType.AGENT_DECISION,
                agent_id=f"agent_{i % 2}",  # Two different agents
                session_id="test_session",
                sequence_number=i,
                event_data={"test": f"data_{i}"},
                prompt=f"Test prompt {i}",
                reasoning=f"Test reasoning {i}",
                tags=["test", "search"]
            )
            events.append(event)
            await self.database.store_audit_event(event)
        
        # Search all events
        results = await self.database.search_events(query="")
        self.assertEqual(len(results), 5)
        
        # Search by agent
        results = await self.database.search_events(query="", agent_id="agent_0")
        self.assertEqual(len(results), 3)  # Events 0, 2, 4
        
        # Search by session
        results = await self.database.search_events(query="", session_id="test_session")
        self.assertEqual(len(results), 5)
        
        # Search by event type
        results = await self.database.search_events(
            query="", 
            event_type=AuditEventType.AGENT_DECISION
        )
        self.assertEqual(len(results), 5)
    
    async def test_get_session_timeline(self):
        """Test getting complete session timeline"""
        await self.database.initialize()
        
        session_id = "timeline_test_session"
        
        # Store audit events
        for i in range(3):
            event = AuditEvent(
                event_id=f"audit_event_{i:03d}",
                timestamp=datetime.now(),
                event_type=AuditEventType.AGENT_DECISION,
                agent_id="test_agent",
                session_id=session_id,
                sequence_number=i,
                event_data={"test": f"data_{i}"}
            )
            await self.database.store_audit_event(event)
        
        # Store replay events
        for i in range(2):
            event = ReplayEvent(
                event_id=f"replay_event_{i:03d}",
                timestamp=datetime.now(),
                event_type=ReplayEventType.DECISION_POINT,
                agent_id="test_agent",
                session_id=session_id,
                sequence_number=i + 10,  # Different sequence numbers
                pre_state={"state": "before"},
                post_state={"state": "after"},
                event_details={"action": f"decision_{i}"}
            )
            await self.database.store_replay_event(event)
        
        # Get timeline
        timeline = await self.database.get_session_timeline(session_id)
        
        self.assertEqual(len(timeline), 5)  # 3 audit + 2 replay events
        
        # Verify events are sorted by timestamp
        timestamps = [event.timestamp for event in timeline]
        self.assertEqual(timestamps, sorted(timestamps))

class TestSessionReplayEngine(unittest.TestCase):
    """Test session replay engine"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.database = AuditDatabase(self.db_path)
        self.replay_engine = SessionReplayEngine(self.database)
    
    def tearDown(self):
        if hasattr(self, 'database') and self.database.connection:
            self.database.connection.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    async def test_create_replay_session(self):
        """Test creating a replay session"""
        await self.database.initialize()
        
        session_id = "original_session"
        
        # Store some events
        for i in range(3):
            event = AuditEvent(
                event_id=f"event_{i:03d}",
                timestamp=datetime.now(),
                event_type=AuditEventType.AGENT_DECISION,
                agent_id="test_agent",
                session_id=session_id,
                sequence_number=i,
                event_data={"test": f"data_{i}"}
            )
            await self.database.store_audit_event(event)
        
        # Create replay session
        replay_id = await self.replay_engine.create_replay_session(session_id)
        
        self.assertIsNotNone(replay_id)
        self.assertIn(replay_id, self.replay_engine.replay_sessions)
        
        # Check replay session data
        replay_session = self.replay_engine.replay_sessions[replay_id]
        self.assertEqual(replay_session['original_session_id'], session_id)
        self.assertEqual(len(replay_session['timeline']), 3)
        self.assertEqual(replay_session['current_position'], 0)
    
    async def test_replay_step(self):
        """Test stepping through replay"""
        await self.database.initialize()
        
        session_id = "step_test_session"
        
        # Store events
        events = []
        for i in range(3):
            event = AuditEvent(
                event_id=f"event_{i:03d}",
                timestamp=datetime.now(),
                event_type=AuditEventType.AGENT_DECISION,
                agent_id="test_agent",
                session_id=session_id,
                sequence_number=i,
                event_data={"test": f"data_{i}"}
            )
            events.append(event)
            await self.database.store_audit_event(event)
        
        # Create replay session
        replay_id = await self.replay_engine.create_replay_session(session_id)
        
        # Step through replay
        for i in range(3):
            event = await self.replay_engine.replay_step(replay_id)
            self.assertIsNotNone(event)
            self.assertEqual(event.event_id, f"event_{i:03d}")
        
        # Should return None when replay is complete
        event = await self.replay_engine.replay_step(replay_id)
        self.assertIsNone(event)
    
    async def test_replay_to_timestamp(self):
        """Test replaying to specific timestamp"""
        await self.database.initialize()
        
        session_id = "timestamp_test_session"
        base_time = datetime.now()
        
        # Store events with specific timestamps
        for i in range(5):
            event = AuditEvent(
                event_id=f"event_{i:03d}",
                timestamp=base_time + timedelta(seconds=i),
                event_type=AuditEventType.AGENT_DECISION,
                agent_id="test_agent",
                session_id=session_id,
                sequence_number=i,
                event_data={"test": f"data_{i}"}
            )
            await self.database.store_audit_event(event)
        
        # Create replay session
        replay_id = await self.replay_engine.create_replay_session(session_id)
        
        # Replay to timestamp (should get first 3 events)
        target_time = base_time + timedelta(seconds=2.5)
        events = await self.replay_engine.replay_to_timestamp(replay_id, target_time)
        
        self.assertEqual(len(events), 3)  # Events 0, 1, 2
        self.assertEqual(events[0].event_id, "event_000")
        self.assertEqual(events[2].event_id, "event_002")
    
    def test_replay_controls(self):
        """Test replay control functions"""
        replay_id = "test_replay"
        
        # Create mock replay session
        self.replay_engine.replay_sessions[replay_id] = {
            'original_session_id': 'test_session',
            'timeline': [],
            'current_position': 0,
            'replay_speed': 1.0,
            'paused': False,
            'created_at': datetime.now()
        }
        
        # Test speed control
        self.replay_engine.set_replay_speed(replay_id, 2.0)
        self.assertEqual(self.replay_engine.replay_sessions[replay_id]['replay_speed'], 2.0)
        
        # Test pause/resume
        self.replay_engine.pause_replay(replay_id)
        self.assertTrue(self.replay_engine.replay_sessions[replay_id]['paused'])
        
        self.replay_engine.resume_replay(replay_id)
        self.assertFalse(self.replay_engine.replay_sessions[replay_id]['paused'])
        
        # Test status
        status = self.replay_engine.get_replay_status(replay_id)
        self.assertEqual(status['replay_id'], replay_id)
        self.assertEqual(status['replay_speed'], 2.0)
        self.assertFalse(status['paused'])

class TestAuditReplaySystem(unittest.TestCase):
    """Test complete audit and replay system"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.system = create_audit_replay_system(
            db_path=self.db_path,
            integrity_level=IntegrityLevel.HASH
        )
    
    def tearDown(self):
        if hasattr(self, 'system') and self.system.database.connection:
            self.system.database.connection.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    async def test_system_initialization(self):
        """Test system initialization"""
        await self.system.initialize()
        
        # Check database is initialized
        self.assertIsNotNone(self.system.database.connection)
        
        # Check integrity system is set up
        self.assertIsNotNone(self.system.integrity)
        
        # Check replay engine is available
        self.assertIsNotNone(self.system.replay_engine)
    
    async def test_audit_session_lifecycle(self):
        """Test complete audit session lifecycle"""
        await self.system.initialize()
        
        session_id = "lifecycle_test_session"
        
        # Start session
        await self.system.start_audit_session(
            session_id=session_id,
            scenario_id="test_scenario",
            participants=["agent1", "agent2"],
            objectives=["test objective"]
        )
        
        self.assertIn(session_id, self.system.active_sessions)
        self.assertIn(session_id, self.system.sequence_counters)
        
        # Log some events
        event_id = await self.system.log_agent_decision(
            session_id=session_id,
            agent_id="agent1",
            prompt="Test prompt",
            context={"test": "context"},
            reasoning="Test reasoning",
            action_taken="test_action",
            action_parameters={"param": "value"},
            confidence_score=0.85,
            tags=["test"]
        )
        
        self.assertIsNotNone(event_id)
        
        # Log LLM interaction
        prompt_id, response_id = await self.system.log_llm_interaction(
            session_id=session_id,
            agent_id="agent1",
            prompt="LLM prompt",
            response="LLM response",
            model_name="gpt-4",
            context={"model": "test"}
        )
        
        self.assertIsNotNone(prompt_id)
        self.assertIsNotNone(response_id)
        
        # End session
        await self.system.end_audit_session(session_id)
        
        self.assertNotIn(session_id, self.system.active_sessions)
        self.assertNotIn(session_id, self.system.sequence_counters)
    
    async def test_search_functionality(self):
        """Test search functionality"""
        await self.system.initialize()
        
        session_id = "search_test_session"
        
        # Start session
        await self.system.start_audit_session(
            session_id=session_id,
            scenario_id="test_scenario",
            participants=["agent1"]
        )
        
        # Log events with different content
        await self.system.log_agent_decision(
            session_id=session_id,
            agent_id="agent1",
            prompt="Network scan prompt",
            context={"target": "192.168.1.0/24"},
            reasoning="Need to scan network",
            action_taken="network_scan",
            action_parameters={"target": "192.168.1.0/24"},
            confidence_score=0.9,
            tags=["reconnaissance", "network"]
        )
        
        await self.system.log_agent_decision(
            session_id=session_id,
            agent_id="agent1",
            prompt="Exploit vulnerability prompt",
            context={"vulnerability": "CVE-2023-1234"},
            reasoning="Exploit found vulnerability",
            action_taken="exploit",
            action_parameters={"cve": "CVE-2023-1234"},
            confidence_score=0.75,
            tags=["exploitation", "vulnerability"]
        )
        
        # Search for network-related events
        events = await self.system.search_audit_events(
            query="network",
            session_id=session_id
        )
        
        # Should find at least one event
        self.assertGreater(len(events), 0)
        
        # Search by agent
        events = await self.system.search_audit_events(
            query="",
            agent_id="agent1",
            session_id=session_id
        )
        
        self.assertEqual(len(events), 2)
        
        # Search by event type
        events = await self.system.search_audit_events(
            query="",
            event_type=AuditEventType.AGENT_DECISION,
            session_id=session_id
        )
        
        self.assertEqual(len(events), 2)
        
        await self.system.end_audit_session(session_id)
    
    async def test_replay_functionality(self):
        """Test replay functionality"""
        await self.system.initialize()
        
        session_id = "replay_test_session"
        
        # Start session and log events
        await self.system.start_audit_session(
            session_id=session_id,
            scenario_id="test_scenario",
            participants=["agent1"]
        )
        
        # Log multiple events
        for i in range(3):
            await self.system.log_agent_decision(
                session_id=session_id,
                agent_id="agent1",
                prompt=f"Test prompt {i}",
                context={"step": i},
                reasoning=f"Test reasoning {i}",
                action_taken=f"test_action_{i}",
                action_parameters={"step": i},
                confidence_score=0.8 + i * 0.05
            )
        
        await self.system.end_audit_session(session_id)
        
        # Create replay
        replay_id = await self.system.create_session_replay(session_id)
        self.assertIsNotNone(replay_id)
        
        # Test replay status
        status = self.system.replay_engine.get_replay_status(replay_id)
        self.assertEqual(status['original_session_id'], session_id)
        self.assertEqual(status['total_events'], 3)
        self.assertEqual(status['current_position'], 0)
    
    async def test_integrity_verification(self):
        """Test integrity verification"""
        await self.system.initialize()
        
        session_id = "integrity_test_session"
        
        # Start session
        await self.system.start_audit_session(
            session_id=session_id,
            scenario_id="test_scenario",
            participants=["agent1"]
        )
        
        # Log event
        event_id = await self.system.log_agent_decision(
            session_id=session_id,
            agent_id="agent1",
            prompt="Test prompt",
            context={"test": "context"},
            reasoning="Test reasoning",
            action_taken="test_action",
            action_parameters={"param": "value"},
            confidence_score=0.85
        )
        
        # Search for the event
        events = await self.system.search_audit_events(
            query="",
            session_id=session_id
        )
        
        self.assertEqual(len(events), 1)
        event = events[0]
        
        # Verify integrity
        self.assertTrue(self.system.integrity.verify_integrity(event))
        
        # Tamper with event and verify it fails
        event.event_data["test"] = "tampered"
        self.assertFalse(self.system.integrity.verify_integrity(event))
        
        await self.system.end_audit_session(session_id)
    
    async def test_system_statistics(self):
        """Test system statistics"""
        await self.system.initialize()
        
        session_id = "stats_test_session"
        
        # Start session and log events
        await self.system.start_audit_session(
            session_id=session_id,
            scenario_id="test_scenario",
            participants=["agent1"]
        )
        
        await self.system.log_agent_decision(
            session_id=session_id,
            agent_id="agent1",
            prompt="Test prompt",
            context={"test": "context"},
            reasoning="Test reasoning",
            action_taken="test_action",
            action_parameters={"param": "value"},
            confidence_score=0.85
        )
        
        await self.system.end_audit_session(session_id)
        
        # Get statistics
        stats = await self.system.get_system_stats()
        
        self.assertIn('system_stats', stats)
        self.assertIn('database_stats', stats)
        self.assertIn('integrity_stats', stats)
        
        # Check some specific stats
        self.assertGreater(stats['system_stats']['events_logged'], 0)
        self.assertGreater(stats['system_stats']['sessions_created'], 0)
        self.assertGreater(stats['database_stats']['total_audit_events'], 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
    
    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    async def test_complete_audit_workflow(self):
        """Test complete audit workflow from start to finish"""
        # Initialize system
        system = create_audit_replay_system(
            db_path=self.db_path,
            integrity_level=IntegrityLevel.HMAC
        )
        await system.initialize()
        
        session_id = "complete_workflow_test"
        
        try:
            # Start audit session
            await system.start_audit_session(
                session_id=session_id,
                scenario_id="red_vs_blue_simulation",
                participants=["red_agent_1", "blue_agent_1"],
                session_type="adversarial_simulation",
                objectives=[
                    "Test red team reconnaissance",
                    "Test blue team detection",
                    "Validate audit logging"
                ]
            )
            
            # Simulate red team activity
            await system.log_agent_decision(
                session_id=session_id,
                agent_id="red_agent_1",
                prompt="Analyze target network for entry points",
                context={
                    "target_network": "192.168.1.0/24",
                    "available_tools": ["nmap", "masscan"],
                    "time_constraint": "30 minutes"
                },
                reasoning="Network reconnaissance is the first step in penetration testing. I need to identify open ports and services to find potential attack vectors.",
                action_taken="network_reconnaissance",
                action_parameters={
                    "target": "192.168.1.0/24",
                    "scan_type": "tcp_syn",
                    "port_range": "1-65535",
                    "timing": "aggressive"
                },
                confidence_score=0.92,
                execution_result={
                    "ports_discovered": [22, 80, 443, 3389],
                    "services_identified": ["ssh", "http", "https", "rdp"],
                    "scan_duration": "45 seconds",
                    "hosts_alive": 15
                },
                tags=["reconnaissance", "network_scan", "red_team"]
            )
            
            # Log LLM interaction for red team
            await system.log_llm_interaction(
                session_id=session_id,
                agent_id="red_agent_1",
                prompt="Based on the network scan results showing ports 22, 80, 443, and 3389 open, what would be the most effective attack vector?",
                response="Given the open ports, I recommend focusing on port 22 (SSH) as it often has weak credentials or misconfigurations. Port 3389 (RDP) is also a high-value target but may have more monitoring. The web services on ports 80 and 443 should be enumerated for vulnerabilities.",
                model_name="gpt-4-turbo",
                context={
                    "scan_results": {
                        "ports": [22, 80, 443, 3389],
                        "services": ["ssh", "http", "https", "rdp"]
                    },
                    "attack_phase": "initial_access"
                },
                tags=["llm_reasoning", "attack_planning", "red_team"]
            )
            
            # Simulate blue team detection
            await system.log_agent_decision(
                session_id=session_id,
                agent_id="blue_agent_1",
                prompt="Analyze network traffic anomalies detected by IDS",
                context={
                    "alert_source": "Suricata IDS",
                    "alert_type": "Port Scan Detected",
                    "source_ip": "192.168.1.100",
                    "target_range": "192.168.1.0/24",
                    "alert_severity": "medium"
                },
                reasoning="The IDS has detected a port scan from an internal IP address. This could indicate lateral movement or reconnaissance activity. I need to investigate the source host and determine if this is legitimate or malicious activity.",
                action_taken="incident_investigation",
                action_parameters={
                    "source_ip": "192.168.1.100",
                    "investigation_type": "network_forensics",
                    "priority": "medium",
                    "assigned_analyst": "blue_agent_1"
                },
                confidence_score=0.78,
                execution_result={
                    "host_identified": True,
                    "user_context": "admin_user",
                    "process_analysis": "nmap process detected",
                    "recommendation": "isolate_host_for_analysis"
                },
                tags=["detection", "incident_response", "blue_team"]
            )
            
            # Log system event
            system_event_id = str(__import__('uuid').uuid4())
            await system.database.store_audit_event(AuditEvent(
                event_id=system_event_id,
                timestamp=datetime.now(),
                event_type=AuditEventType.SYSTEM_EVENT,
                agent_id="system",
                session_id=session_id,
                sequence_number=system.sequence_counters[session_id],
                event_data={
                    "event_type": "phase_transition",
                    "from_phase": "reconnaissance",
                    "to_phase": "initial_access",
                    "trigger": "scan_completion"
                },
                tags=["system", "phase_transition"]
            ))
            system.sequence_counters[session_id] += 1
            
            # End session
            await system.end_audit_session(session_id)
            
            # Verify session was properly logged
            events = await system.search_audit_events(
                query="",
                session_id=session_id,
                limit=100
            )
            
            self.assertGreaterEqual(len(events), 4)  # At least 4 events logged
            
            # Verify different event types are present
            event_types = {event.event_type for event in events}
            self.assertIn(AuditEventType.AGENT_DECISION, event_types)
            self.assertIn(AuditEventType.LLM_PROMPT, event_types)
            self.assertIn(AuditEventType.LLM_RESPONSE, event_types)
            
            # Verify integrity of all events
            for event in events:
                self.assertTrue(system.integrity.verify_integrity(event))
            
            # Create and test replay
            replay_id = await system.create_session_replay(session_id)
            self.assertIsNotNone(replay_id)
            
            # Test replay functionality
            replay_status = system.replay_engine.get_replay_status(replay_id)
            self.assertEqual(replay_status['original_session_id'], session_id)
            self.assertGreater(replay_status['total_events'], 0)
            
            # Step through replay
            events_replayed = []
            while True:
                event = await system.replay_engine.replay_step(replay_id)
                if event is None:
                    break
                events_replayed.append(event)
            
            self.assertGreater(len(events_replayed), 0)
            
            # Verify search functionality
            network_events = await system.search_audit_events(
                query="network",
                session_id=session_id
            )
            self.assertGreater(len(network_events), 0)
            
            # Search by agent
            red_events = await system.search_audit_events(
                query="",
                agent_id="red_agent_1",
                session_id=session_id
            )
            blue_events = await system.search_audit_events(
                query="",
                agent_id="blue_agent_1",
                session_id=session_id
            )
            
            self.assertGreater(len(red_events), 0)
            self.assertGreater(len(blue_events), 0)
            
            # Get system statistics
            stats = await system.get_system_stats()
            self.assertGreater(stats['system_stats']['events_logged'], 0)
            self.assertGreater(stats['system_stats']['sessions_created'], 0)
            self.assertGreater(stats['database_stats']['total_audit_events'], 0)
            
            print(f"✓ Complete audit workflow test passed")
            print(f"  Events logged: {stats['system_stats']['events_logged']}")
            print(f"  Sessions created: {stats['system_stats']['sessions_created']}")
            print(f"  Replays created: {stats['system_stats']['replays_created']}")
            
        finally:
            # Clean up
            if system.database.connection:
                system.database.connection.close()

# Async test runner
class AsyncTestRunner:
    """Helper class to run async tests"""
    
    @staticmethod
    def run_async_test(test_func):
        """Run an async test function"""
        return asyncio.run(test_func())

# Test suite for async tests
def create_async_test_suite():
    """Create test suite with async tests"""
    suite = unittest.TestSuite()
    
    # Database tests
    db_test = TestAuditDatabase()
    suite.addTest(AsyncTestRunner.run_async_test(db_test.test_database_initialization))
    suite.addTest(AsyncTestRunner.run_async_test(db_test.test_store_audit_event))
    suite.addTest(AsyncTestRunner.run_async_test(db_test.test_store_replay_event))
    suite.addTest(AsyncTestRunner.run_async_test(db_test.test_search_events))
    suite.addTest(AsyncTestRunner.run_async_test(db_test.test_get_session_timeline))
    
    # Replay engine tests
    replay_test = TestSessionReplayEngine()
    suite.addTest(AsyncTestRunner.run_async_test(replay_test.test_create_replay_session))
    suite.addTest(AsyncTestRunner.run_async_test(replay_test.test_replay_step))
    suite.addTest(AsyncTestRunner.run_async_test(replay_test.test_replay_to_timestamp))
    
    # System tests
    system_test = TestAuditReplaySystem()
    suite.addTest(AsyncTestRunner.run_async_test(system_test.test_system_initialization))
    suite.addTest(AsyncTestRunner.run_async_test(system_test.test_audit_session_lifecycle))
    suite.addTest(AsyncTestRunner.run_async_test(system_test.test_search_functionality))
    suite.addTest(AsyncTestRunner.run_async_test(system_test.test_replay_functionality))
    suite.addTest(AsyncTestRunner.run_async_test(system_test.test_integrity_verification))
    suite.addTest(AsyncTestRunner.run_async_test(system_test.test_system_statistics))
    
    # Integration tests
    integration_test = TestIntegration()
    suite.addTest(AsyncTestRunner.run_async_test(integration_test.test_complete_audit_workflow))
    
    return suite

if __name__ == "__main__":
    # Run synchronous tests
    print("Running synchronous tests...")
    sync_loader = unittest.TestLoader()
    sync_suite = unittest.TestSuite()
    
    # Add synchronous test classes
    sync_suite.addTests(sync_loader.loadTestsFromTestCase(TestAuditEvent))
    sync_suite.addTests(sync_loader.loadTestsFromTestCase(TestReplayEvent))
    sync_suite.addTests(sync_loader.loadTestsFromTestCase(TestCryptographicIntegrity))
    
    sync_runner = unittest.TextTestRunner(verbosity=2)
    sync_result = sync_runner.run(sync_suite)
    
    # Run asynchronous tests
    print("\nRunning asynchronous tests...")
    
    async def run_async_tests():
        """Run all async tests"""
        test_classes = [
            TestAuditDatabase,
            TestSessionReplayEngine, 
            TestAuditReplaySystem,
            TestIntegration
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_class in test_classes:
            print(f"\n--- {test_class.__name__} ---")
            test_instance = test_class()
            
            # Get all test methods
            test_methods = [method for method in dir(test_instance) 
                          if method.startswith('test_') and callable(getattr(test_instance, method))]
            
            for method_name in test_methods:
                total_tests += 1
                try:
                    test_instance.setUp()
                    test_method = getattr(test_instance, method_name)
                    await test_method()
                    test_instance.tearDown()
                    print(f"✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"✗ {method_name}: {e}")
                    try:
                        test_instance.tearDown()
                    except:
                        pass
        
        print(f"\nAsync Tests Summary: {passed_tests}/{total_tests} passed")
        return passed_tests == total_tests
    
    async_success = asyncio.run(run_async_tests())
    
    # Overall result
    overall_success = sync_result.wasSuccessful() and async_success
    print(f"\nOverall Test Result: {'PASSED' if overall_success else 'FAILED'}")
    
    exit(0 if overall_success else 1)