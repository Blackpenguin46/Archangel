#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Audit and Replay System
Comprehensive audit logging and session replay for forensic analysis
"""

import asyncio
import json
import logging
import hashlib
import hmac
import sqlite3
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import pickle
import gzip
import threading
from contextlib import asynccontextmanager

# Optional imports for enhanced functionality
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    AGENT_DECISION = "agent_decision"
    LLM_PROMPT = "llm_prompt"
    LLM_RESPONSE = "llm_response"
    ACTION_EXECUTION = "action_execution"
    COMMUNICATION = "communication"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    ERROR_EVENT = "error_event"
    PHASE_TRANSITION = "phase_transition"
    OBJECTIVE_UPDATE = "objective_update"

class ReplayEventType(Enum):
    """Types of replay events"""
    DECISION_POINT = "decision_point"
    ACTION_RESULT = "action_result"
    STATE_CHANGE = "state_change"
    COMMUNICATION_EVENT = "communication_event"
    ENVIRONMENT_UPDATE = "environment_update"

class IntegrityLevel(Enum):
    """Cryptographic integrity levels"""
    NONE = "none"
    HASH = "hash"
    HMAC = "hmac"
    DIGITAL_SIGNATURE = "digital_signature"

@dataclass
class AuditEvent:
    """Individual audit event with full context"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    agent_id: str
    session_id: str
    sequence_number: int
    
    # Core event data
    event_data: Dict[str, Any]
    
    # Decision context (for agent decisions)
    prompt: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    confidence_score: Optional[float] = None
    
    # Execution details
    action_taken: Optional[str] = None
    action_parameters: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None
    
    # Metadata
    tags: List[str] = None
    related_events: List[str] = None
    
    # Integrity
    integrity_level: IntegrityLevel = IntegrityLevel.HASH
    checksum: Optional[str] = None
    signature: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.related_events is None:
            self.related_events = []

@dataclass
class ReplayEvent:
    """Event for session replay with state information"""
    event_id: str
    timestamp: datetime
    event_type: ReplayEventType
    agent_id: str
    session_id: str
    sequence_number: int
    
    # State information
    pre_state: Dict[str, Any]
    post_state: Dict[str, Any]
    
    # Event details
    event_details: Dict[str, Any]
    
    # Replay metadata
    replay_duration: Optional[float] = None
    can_replay: bool = True
    replay_notes: Optional[str] = None

@dataclass
class AuditSession:
    """Complete audit session with metadata"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    scenario_id: str
    participants: List[str]
    
    # Session metadata
    session_type: str
    objectives: List[str]
    success_criteria: Dict[str, Any]
    
    # Audit statistics
    total_events: int = 0
    decision_points: int = 0
    actions_executed: int = 0
    communications: int = 0
    errors: int = 0
    
    # Integrity
    session_hash: Optional[str] = None
    verified: bool = False

class AuditDatabase:
    """SQLite-based audit database with full-text search"""
    
    def __init__(self, db_path: str = "audit.db"):
        self.db_path = db_path
        self.connection = None
        self.lock = threading.Lock()
        
    async def initialize(self) -> None:
        """Initialize the audit database"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            # Enable full-text search
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            await self._create_tables()
            await self._create_indexes()
            
            logger.info(f"Audit database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
            raise
    
    async def _create_tables(self) -> None:
        """Create database tables"""
        # Audit events table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                event_data TEXT NOT NULL,
                prompt TEXT,
                context TEXT,
                reasoning TEXT,
                confidence_score REAL,
                action_taken TEXT,
                action_parameters TEXT,
                execution_result TEXT,
                tags TEXT,
                related_events TEXT,
                integrity_level TEXT NOT NULL,
                checksum TEXT,
                signature TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Replay events table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS replay_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                pre_state TEXT NOT NULL,
                post_state TEXT NOT NULL,
                event_details TEXT NOT NULL,
                replay_duration REAL,
                can_replay BOOLEAN DEFAULT TRUE,
                replay_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Audit sessions table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS audit_sessions (
                session_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                scenario_id TEXT NOT NULL,
                participants TEXT NOT NULL,
                session_type TEXT NOT NULL,
                objectives TEXT NOT NULL,
                success_criteria TEXT NOT NULL,
                total_events INTEGER DEFAULT 0,
                decision_points INTEGER DEFAULT 0,
                actions_executed INTEGER DEFAULT 0,
                communications INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                session_hash TEXT,
                verified BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Full-text search virtual table
        self.connection.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS audit_search USING fts5(
                event_id,
                event_type,
                agent_id,
                session_id,
                event_data,
                prompt,
                reasoning,
                action_taken,
                tags,
                content='audit_events',
                content_rowid='rowid'
            )
        """)
        
        self.connection.commit()
    
    async def _create_indexes(self) -> None:
        """Create database indexes for performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_events(agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_events(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_replay_timestamp ON replay_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_replay_session ON replay_events(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_session_start ON audit_sessions(start_time)"
        ]
        
        for index_sql in indexes:
            self.connection.execute(index_sql)
        
        self.connection.commit()
    
    async def store_audit_event(self, event: AuditEvent) -> None:
        """Store audit event in database"""
        try:
            with self.lock:
                self.connection.execute("""
                    INSERT INTO audit_events (
                        event_id, timestamp, event_type, agent_id, session_id,
                        sequence_number, event_data, prompt, context, reasoning,
                        confidence_score, action_taken, action_parameters,
                        execution_result, tags, related_events, integrity_level,
                        checksum, signature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.agent_id,
                    event.session_id,
                    event.sequence_number,
                    json.dumps(event.event_data),
                    event.prompt,
                    json.dumps(event.context) if event.context else None,
                    event.reasoning,
                    event.confidence_score,
                    event.action_taken,
                    json.dumps(event.action_parameters) if event.action_parameters else None,
                    json.dumps(event.execution_result) if event.execution_result else None,
                    json.dumps(event.tags),
                    json.dumps(event.related_events),
                    event.integrity_level.value,
                    event.checksum,
                    event.signature
                ))
                
                # Update full-text search
                self.connection.execute("""
                    INSERT INTO audit_search (
                        event_id, event_type, agent_id, session_id,
                        event_data, prompt, reasoning, action_taken, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type.value,
                    event.agent_id,
                    event.session_id,
                    json.dumps(event.event_data),
                    event.prompt or "",
                    event.reasoning or "",
                    event.action_taken or "",
                    json.dumps(event.tags)
                ))
                
                self.connection.commit()
                
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
            raise
    
    async def store_replay_event(self, event: ReplayEvent) -> None:
        """Store replay event in database"""
        try:
            with self.lock:
                self.connection.execute("""
                    INSERT INTO replay_events (
                        event_id, timestamp, event_type, agent_id, session_id,
                        sequence_number, pre_state, post_state, event_details,
                        replay_duration, can_replay, replay_notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.agent_id,
                    event.session_id,
                    event.sequence_number,
                    json.dumps(event.pre_state),
                    json.dumps(event.post_state),
                    json.dumps(event.event_details),
                    event.replay_duration,
                    event.can_replay,
                    event.replay_notes
                ))
                
                self.connection.commit()
                
        except Exception as e:
            logger.error(f"Failed to store replay event: {e}")
            raise
    
    async def search_events(self, 
                          query: str,
                          session_id: Optional[str] = None,
                          agent_id: Optional[str] = None,
                          event_type: Optional[AuditEventType] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 100) -> List[AuditEvent]:
        """Search audit events with filters"""
        try:
            # Build search query
            sql_parts = ["SELECT * FROM audit_events WHERE 1=1"]
            params = []
            
            # Full-text search
            if query:
                sql_parts.append("AND event_id IN (SELECT event_id FROM audit_search WHERE audit_search MATCH ?)")
                params.append(query)
            
            # Filters
            if session_id:
                sql_parts.append("AND session_id = ?")
                params.append(session_id)
            
            if agent_id:
                sql_parts.append("AND agent_id = ?")
                params.append(agent_id)
            
            if event_type:
                sql_parts.append("AND event_type = ?")
                params.append(event_type.value)
            
            if start_time:
                sql_parts.append("AND timestamp >= ?")
                params.append(start_time.isoformat())
            
            if end_time:
                sql_parts.append("AND timestamp <= ?")
                params.append(end_time.isoformat())
            
            sql_parts.append("ORDER BY timestamp DESC LIMIT ?")
            params.append(limit)
            
            sql = " ".join(sql_parts)
            
            with self.lock:
                cursor = self.connection.execute(sql, params)
                rows = cursor.fetchall()
            
            # Convert to AuditEvent objects
            events = []
            for row in rows:
                event = AuditEvent(
                    event_id=row['event_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    event_type=AuditEventType(row['event_type']),
                    agent_id=row['agent_id'],
                    session_id=row['session_id'],
                    sequence_number=row['sequence_number'],
                    event_data=json.loads(row['event_data']),
                    prompt=row['prompt'],
                    context=json.loads(row['context']) if row['context'] else None,
                    reasoning=row['reasoning'],
                    confidence_score=row['confidence_score'],
                    action_taken=row['action_taken'],
                    action_parameters=json.loads(row['action_parameters']) if row['action_parameters'] else None,
                    execution_result=json.loads(row['execution_result']) if row['execution_result'] else None,
                    tags=json.loads(row['tags']),
                    related_events=json.loads(row['related_events']),
                    integrity_level=IntegrityLevel(row['integrity_level']),
                    checksum=row['checksum'],
                    signature=row['signature']
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to search events: {e}")
            raise
    
    async def get_session_timeline(self, session_id: str) -> List[Union[AuditEvent, ReplayEvent]]:
        """Get complete timeline for a session"""
        try:
            # Get audit events
            audit_events = await self.search_events(
                query="",
                session_id=session_id,
                limit=10000
            )
            
            # Get replay events
            with self.lock:
                cursor = self.connection.execute("""
                    SELECT * FROM replay_events 
                    WHERE session_id = ? 
                    ORDER BY timestamp
                """, (session_id,))
                replay_rows = cursor.fetchall()
            
            replay_events = []
            for row in replay_rows:
                event = ReplayEvent(
                    event_id=row['event_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    event_type=ReplayEventType(row['event_type']),
                    agent_id=row['agent_id'],
                    session_id=row['session_id'],
                    sequence_number=row['sequence_number'],
                    pre_state=json.loads(row['pre_state']),
                    post_state=json.loads(row['post_state']),
                    event_details=json.loads(row['event_details']),
                    replay_duration=row['replay_duration'],
                    can_replay=bool(row['can_replay']),
                    replay_notes=row['replay_notes']
                )
                replay_events.append(event)
            
            # Combine and sort by timestamp
            all_events = audit_events + replay_events
            all_events.sort(key=lambda x: x.timestamp)
            
            return all_events
            
        except Exception as e:
            logger.error(f"Failed to get session timeline: {e}")
            raise

class CryptographicIntegrity:
    """Cryptographic integrity verification for audit events"""
    
    def __init__(self, private_key_path: Optional[str] = None, public_key_path: Optional[str] = None):
        self.private_key = None
        self.public_key = None
        self.hmac_key = None
        
        if CRYPTOGRAPHY_AVAILABLE:
            if private_key_path and Path(private_key_path).exists():
                with open(private_key_path, 'rb') as f:
                    self.private_key = load_pem_private_key(f.read(), password=None)
            
            if public_key_path and Path(public_key_path).exists():
                with open(public_key_path, 'rb') as f:
                    self.public_key = load_pem_public_key(f.read())
        
        # Always generate HMAC key for basic integrity
        self.hmac_key = b"audit_integrity_key_" + uuid.uuid4().bytes
    
    def generate_keys(self, key_dir: str = "keys") -> Tuple[str, str]:
        """Generate RSA key pair for digital signatures"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        key_path = Path(key_dir)
        key_path.mkdir(exist_ok=True)
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Generate public key
        public_key = private_key.public_key()
        
        # Save private key
        private_key_path = key_path / "audit_private.pem"
        with open(private_key_path, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Save public key
        public_key_path = key_path / "audit_public.pem"
        with open(public_key_path, 'wb') as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
        
        self.private_key = private_key
        self.public_key = public_key
        
        return str(private_key_path), str(public_key_path)
    
    def calculate_hash(self, data: str) -> str:
        """Calculate SHA-256 hash of data"""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def calculate_hmac(self, data: str) -> str:
        """Calculate HMAC-SHA256 of data"""
        if not self.hmac_key:
            raise RuntimeError("HMAC key not available")
        
        return hmac.new(
            self.hmac_key,
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def sign_data(self, data: str) -> str:
        """Create digital signature of data"""
        if not CRYPTOGRAPHY_AVAILABLE or not self.private_key:
            raise RuntimeError("Digital signature not available")
        
        signature = self.private_key.sign(
            data.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature.hex()
    
    def verify_signature(self, data: str, signature: str) -> bool:
        """Verify digital signature"""
        if not CRYPTOGRAPHY_AVAILABLE or not self.public_key:
            return False
        
        try:
            self.public_key.verify(
                bytes.fromhex(signature),
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def add_integrity(self, event: AuditEvent) -> None:
        """Add integrity verification to audit event"""
        # Create data string for integrity calculation
        data_parts = [
            event.event_id,
            event.timestamp.isoformat(),
            event.event_type.value,
            event.agent_id,
            event.session_id,
            str(event.sequence_number),
            json.dumps(event.event_data, sort_keys=True)
        ]
        
        if event.prompt:
            data_parts.append(event.prompt)
        if event.reasoning:
            data_parts.append(event.reasoning)
        if event.action_taken:
            data_parts.append(event.action_taken)
        
        data_string = "|".join(data_parts)
        
        # Apply integrity based on level
        if event.integrity_level == IntegrityLevel.HASH:
            event.checksum = self.calculate_hash(data_string)
        elif event.integrity_level == IntegrityLevel.HMAC:
            event.checksum = self.calculate_hmac(data_string)
        elif event.integrity_level == IntegrityLevel.DIGITAL_SIGNATURE:
            event.signature = self.sign_data(data_string)
            event.checksum = self.calculate_hash(data_string)
    
    def verify_integrity(self, event: AuditEvent) -> bool:
        """Verify integrity of audit event"""
        # Recreate data string
        data_parts = [
            event.event_id,
            event.timestamp.isoformat(),
            event.event_type.value,
            event.agent_id,
            event.session_id,
            str(event.sequence_number),
            json.dumps(event.event_data, sort_keys=True)
        ]
        
        if event.prompt:
            data_parts.append(event.prompt)
        if event.reasoning:
            data_parts.append(event.reasoning)
        if event.action_taken:
            data_parts.append(event.action_taken)
        
        data_string = "|".join(data_parts)
        
        # Verify based on integrity level
        if event.integrity_level == IntegrityLevel.HASH:
            return event.checksum == self.calculate_hash(data_string)
        elif event.integrity_level == IntegrityLevel.HMAC:
            return event.checksum == self.calculate_hmac(data_string)
        elif event.integrity_level == IntegrityLevel.DIGITAL_SIGNATURE:
            hash_valid = event.checksum == self.calculate_hash(data_string)
            sig_valid = self.verify_signature(data_string, event.signature) if event.signature else False
            return hash_valid and sig_valid
        
        return True  # No integrity verification

class SessionReplayEngine:
    """Engine for replaying audit sessions"""
    
    def __init__(self, database: AuditDatabase):
        self.database = database
        self.replay_sessions = {}
        self.replay_speed = 1.0  # Normal speed
    
    async def create_replay_session(self, session_id: str, replay_id: Optional[str] = None) -> str:
        """Create a new replay session"""
        if replay_id is None:
            replay_id = f"replay_{uuid.uuid4().hex[:8]}"
        
        # Get session timeline
        timeline = await self.database.get_session_timeline(session_id)
        
        self.replay_sessions[replay_id] = {
            'original_session_id': session_id,
            'timeline': timeline,
            'current_position': 0,
            'replay_speed': self.replay_speed,
            'paused': False,
            'created_at': datetime.now()
        }
        
        logger.info(f"Created replay session {replay_id} for session {session_id}")
        return replay_id
    
    async def replay_step(self, replay_id: str) -> Optional[Union[AuditEvent, ReplayEvent]]:
        """Execute one step of replay"""
        if replay_id not in self.replay_sessions:
            raise ValueError(f"Replay session {replay_id} not found")
        
        session = self.replay_sessions[replay_id]
        timeline = session['timeline']
        position = session['current_position']
        
        if position >= len(timeline):
            return None  # Replay complete
        
        current_event = timeline[position]
        session['current_position'] += 1
        
        logger.debug(f"Replay {replay_id}: Step {position + 1}/{len(timeline)}")
        return current_event
    
    async def replay_to_timestamp(self, replay_id: str, target_time: datetime) -> List[Union[AuditEvent, ReplayEvent]]:
        """Replay events up to a specific timestamp"""
        if replay_id not in self.replay_sessions:
            raise ValueError(f"Replay session {replay_id} not found")
        
        session = self.replay_sessions[replay_id]
        timeline = session['timeline']
        events = []
        
        while session['current_position'] < len(timeline):
            event = timeline[session['current_position']]
            if event.timestamp > target_time:
                break
            
            events.append(event)
            session['current_position'] += 1
        
        return events
    
    async def replay_full_session(self, replay_id: str) -> List[Union[AuditEvent, ReplayEvent]]:
        """Replay entire session"""
        if replay_id not in self.replay_sessions:
            raise ValueError(f"Replay session {replay_id} not found")
        
        session = self.replay_sessions[replay_id]
        timeline = session['timeline']
        session['current_position'] = len(timeline)
        
        return timeline
    
    def set_replay_speed(self, replay_id: str, speed: float) -> None:
        """Set replay speed (1.0 = normal, 2.0 = 2x, 0.5 = half speed)"""
        if replay_id in self.replay_sessions:
            self.replay_sessions[replay_id]['replay_speed'] = speed
    
    def pause_replay(self, replay_id: str) -> None:
        """Pause replay session"""
        if replay_id in self.replay_sessions:
            self.replay_sessions[replay_id]['paused'] = True
    
    def resume_replay(self, replay_id: str) -> None:
        """Resume replay session"""
        if replay_id in self.replay_sessions:
            self.replay_sessions[replay_id]['paused'] = False
    
    def get_replay_status(self, replay_id: str) -> Dict[str, Any]:
        """Get current replay status"""
        if replay_id not in self.replay_sessions:
            return {}
        
        session = self.replay_sessions[replay_id]
        timeline = session['timeline']
        
        return {
            'replay_id': replay_id,
            'original_session_id': session['original_session_id'],
            'total_events': len(timeline),
            'current_position': session['current_position'],
            'progress_percent': (session['current_position'] / len(timeline)) * 100 if timeline else 0,
            'replay_speed': session['replay_speed'],
            'paused': session['paused'],
            'created_at': session['created_at'].isoformat()
        }

class AuditReplaySystem:
    """Main audit and replay system coordinator"""
    
    def __init__(self, 
                 db_path: str = "audit.db",
                 integrity_level: IntegrityLevel = IntegrityLevel.HMAC,
                 private_key_path: Optional[str] = None,
                 public_key_path: Optional[str] = None):
        self.database = AuditDatabase(db_path)
        self.integrity = CryptographicIntegrity(private_key_path, public_key_path)
        self.replay_engine = SessionReplayEngine(self.database)
        self.default_integrity_level = integrity_level
        
        # Session management
        self.active_sessions = {}
        self.sequence_counters = {}
        
        # Statistics
        self.stats = {
            'events_logged': 0,
            'sessions_created': 0,
            'replays_created': 0,
            'integrity_violations': 0,
            'search_queries': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the audit and replay system"""
        try:
            await self.database.initialize()
            
            # Generate keys if using digital signatures and keys don't exist
            if (self.default_integrity_level == IntegrityLevel.DIGITAL_SIGNATURE and 
                not self.integrity.private_key and CRYPTOGRAPHY_AVAILABLE):
                try:
                    private_key_path, public_key_path = self.integrity.generate_keys()
                    self.logger.info(f"Generated audit keys: {private_key_path}, {public_key_path}")
                except Exception as e:
                    self.logger.warning(f"Could not generate digital signature keys: {e}")
                    self.logger.info("Falling back to HMAC integrity")
                    self.default_integrity_level = IntegrityLevel.HMAC
            
            self.logger.info("Audit and replay system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audit system: {e}")
            raise
    
    async def start_audit_session(self, 
                                session_id: str,
                                scenario_id: str,
                                participants: List[str],
                                session_type: str = "simulation",
                                objectives: List[str] = None) -> None:
        """Start a new audit session"""
        try:
            if objectives is None:
                objectives = []
            
            session = AuditSession(
                session_id=session_id,
                start_time=datetime.now(),
                end_time=None,
                scenario_id=scenario_id,
                participants=participants,
                session_type=session_type,
                objectives=objectives,
                success_criteria={}
            )
            
            self.active_sessions[session_id] = session
            self.sequence_counters[session_id] = 0
            
            # Store session in database
            with self.database.lock:
                self.database.connection.execute("""
                    INSERT INTO audit_sessions (
                        session_id, start_time, scenario_id, participants,
                        session_type, objectives, success_criteria
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.start_time.isoformat(),
                    session.scenario_id,
                    json.dumps(session.participants),
                    session.session_type,
                    json.dumps(session.objectives),
                    json.dumps(session.success_criteria)
                ))
                self.database.connection.commit()
            
            self.stats['sessions_created'] += 1
            self.logger.info(f"Started audit session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start audit session: {e}")
            raise
    
    async def log_agent_decision(self,
                               session_id: str,
                               agent_id: str,
                               prompt: str,
                               context: Dict[str, Any],
                               reasoning: str,
                               action_taken: str,
                               action_parameters: Dict[str, Any],
                               confidence_score: float,
                               execution_result: Dict[str, Any] = None,
                               tags: List[str] = None) -> str:
        """Log an agent decision with full context"""
        try:
            event_id = str(uuid.uuid4())
            
            if session_id not in self.sequence_counters:
                self.sequence_counters[session_id] = 0
            
            sequence_number = self.sequence_counters[session_id]
            self.sequence_counters[session_id] += 1
            
            event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                event_type=AuditEventType.AGENT_DECISION,
                agent_id=agent_id,
                session_id=session_id,
                sequence_number=sequence_number,
                event_data={
                    'decision_type': 'agent_action',
                    'context_summary': context.get('summary', 'No summary'),
                    'confidence_score': confidence_score
                },
                prompt=prompt,
                context=context,
                reasoning=reasoning,
                confidence_score=confidence_score,
                action_taken=action_taken,
                action_parameters=action_parameters,
                execution_result=execution_result or {},
                tags=tags or [],
                integrity_level=self.default_integrity_level
            )
            
            # Add integrity verification
            self.integrity.add_integrity(event)
            
            # Store in database
            await self.database.store_audit_event(event)
            
            # Update session statistics
            if session_id in self.active_sessions:
                self.active_sessions[session_id].total_events += 1
                self.active_sessions[session_id].decision_points += 1
                if execution_result:
                    self.active_sessions[session_id].actions_executed += 1
            
            self.stats['events_logged'] += 1
            self.logger.debug(f"Logged agent decision: {event_id}")
            
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log agent decision: {e}")
            raise
    
    async def log_llm_interaction(self,
                                session_id: str,
                                agent_id: str,
                                prompt: str,
                                response: str,
                                model_name: str,
                                context: Dict[str, Any] = None,
                                tags: List[str] = None) -> Tuple[str, str]:
        """Log LLM prompt and response as separate events"""
        try:
            prompt_event_id = str(uuid.uuid4())
            response_event_id = str(uuid.uuid4())
            
            if session_id not in self.sequence_counters:
                self.sequence_counters[session_id] = 0
            
            # Log prompt event
            prompt_sequence = self.sequence_counters[session_id]
            self.sequence_counters[session_id] += 1
            
            prompt_event = AuditEvent(
                event_id=prompt_event_id,
                timestamp=datetime.now(),
                event_type=AuditEventType.LLM_PROMPT,
                agent_id=agent_id,
                session_id=session_id,
                sequence_number=prompt_sequence,
                event_data={
                    'model_name': model_name,
                    'prompt_length': len(prompt),
                    'context_keys': list(context.keys()) if context else []
                },
                prompt=prompt,
                context=context or {},
                tags=tags or [],
                related_events=[response_event_id],
                integrity_level=self.default_integrity_level
            )
            
            # Log response event
            response_sequence = self.sequence_counters[session_id]
            self.sequence_counters[session_id] += 1
            
            response_event = AuditEvent(
                event_id=response_event_id,
                timestamp=datetime.now(),
                event_type=AuditEventType.LLM_RESPONSE,
                agent_id=agent_id,
                session_id=session_id,
                sequence_number=response_sequence,
                event_data={
                    'model_name': model_name,
                    'response_length': len(response),
                    'response_content': response
                },
                context=context or {},
                tags=tags or [],
                related_events=[prompt_event_id],
                integrity_level=self.default_integrity_level
            )
            
            # Add integrity verification
            self.integrity.add_integrity(prompt_event)
            self.integrity.add_integrity(response_event)
            
            # Store in database
            await self.database.store_audit_event(prompt_event)
            await self.database.store_audit_event(response_event)
            
            # Update session statistics
            if session_id in self.active_sessions:
                self.active_sessions[session_id].total_events += 2
            
            self.stats['events_logged'] += 2
            self.logger.debug(f"Logged LLM interaction: {prompt_event_id}, {response_event_id}")
            
            return prompt_event_id, response_event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log LLM interaction: {e}")
            raise
    
    async def end_audit_session(self, session_id: str) -> None:
        """End an audit session and calculate final hash"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            session.end_time = datetime.now()
            
            # Calculate session hash from all events
            events = await self.database.search_events(
                query="",
                session_id=session_id,
                limit=10000
            )
            
            # Create session hash from all event checksums
            event_hashes = [event.checksum for event in events if event.checksum]
            session_hash_data = "|".join(sorted(event_hashes))
            session.session_hash = self.integrity.calculate_hash(session_hash_data)
            session.verified = True
            
            # Update database
            with self.database.lock:
                self.database.connection.execute("""
                    UPDATE audit_sessions SET
                        end_time = ?,
                        total_events = ?,
                        decision_points = ?,
                        actions_executed = ?,
                        communications = ?,
                        errors = ?,
                        session_hash = ?,
                        verified = ?
                    WHERE session_id = ?
                """, (
                    session.end_time.isoformat(),
                    session.total_events,
                    session.decision_points,
                    session.actions_executed,
                    session.communications,
                    session.errors,
                    session.session_hash,
                    session.verified,
                    session_id
                ))
                self.database.connection.commit()
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            del self.sequence_counters[session_id]
            
            self.logger.info(f"Ended audit session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to end audit session: {e}")
            raise
    
    async def search_audit_events(self, 
                                query: str,
                                session_id: Optional[str] = None,
                                agent_id: Optional[str] = None,
                                event_type: Optional[AuditEventType] = None,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None,
                                limit: int = 100) -> List[AuditEvent]:
        """Search audit events with full-text search and filters"""
        try:
            self.stats['search_queries'] += 1
            
            events = await self.database.search_events(
                query=query,
                session_id=session_id,
                agent_id=agent_id,
                event_type=event_type,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            # Verify integrity of returned events
            verified_events = []
            for event in events:
                if self.integrity.verify_integrity(event):
                    verified_events.append(event)
                else:
                    self.stats['integrity_violations'] += 1
                    self.logger.warning(f"Integrity violation detected for event {event.event_id}")
            
            return verified_events
            
        except Exception as e:
            self.logger.error(f"Failed to search audit events: {e}")
            raise
    
    async def create_session_replay(self, session_id: str) -> str:
        """Create a replay session for forensic analysis"""
        try:
            replay_id = await self.replay_engine.create_replay_session(session_id)
            self.stats['replays_created'] += 1
            
            self.logger.info(f"Created replay session: {replay_id}")
            return replay_id
            
        except Exception as e:
            self.logger.error(f"Failed to create session replay: {e}")
            raise
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            # Database statistics
            with self.database.lock:
                cursor = self.database.connection.execute("SELECT COUNT(*) FROM audit_events")
                total_events = cursor.fetchone()[0]
                
                cursor = self.database.connection.execute("SELECT COUNT(*) FROM audit_sessions")
                total_sessions = cursor.fetchone()[0]
                
                cursor = self.database.connection.execute("SELECT COUNT(*) FROM replay_events")
                total_replay_events = cursor.fetchone()[0]
            
            return {
                'system_stats': self.stats,
                'database_stats': {
                    'total_audit_events': total_events,
                    'total_sessions': total_sessions,
                    'total_replay_events': total_replay_events,
                    'active_sessions': len(self.active_sessions),
                    'active_replays': len(self.replay_engine.replay_sessions)
                },
                'integrity_stats': {
                    'default_level': self.default_integrity_level.value,
                    'digital_signatures_available': bool(self.integrity.private_key),
                    'hmac_available': bool(self.integrity.hmac_key)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system stats: {e}")
            raise

# Factory functions for easy instantiation
def create_audit_replay_system(db_path: str = "audit.db",
                             integrity_level: IntegrityLevel = IntegrityLevel.HMAC,
                             private_key_path: Optional[str] = None,
                             public_key_path: Optional[str] = None) -> AuditReplaySystem:
    """Create and return an AuditReplaySystem instance"""
    return AuditReplaySystem(
        db_path=db_path,
        integrity_level=integrity_level,
        private_key_path=private_key_path,
        public_key_path=public_key_path
    )

if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize system
        audit_system = create_audit_replay_system()
        await audit_system.initialize()
        
        # Start session
        await audit_system.start_audit_session(
            session_id="test_session_001",
            scenario_id="red_vs_blue_basic",
            participants=["red_agent_1", "blue_agent_1"],
            objectives=["Test audit logging"]
        )
        
        # Log some events
        await audit_system.log_agent_decision(
            session_id="test_session_001",
            agent_id="red_agent_1",
            prompt="Analyze target network for vulnerabilities",
            context={"target": "192.168.1.0/24", "tools": ["nmap"]},
            reasoning="Network scan will reveal open ports and services",
            action_taken="network_scan",
            action_parameters={"target": "192.168.1.0/24", "scan_type": "tcp"},
            confidence_score=0.85,
            execution_result={"ports_found": [22, 80, 443], "services": ["ssh", "http", "https"]},
            tags=["reconnaissance", "network_scan"]
        )
        
        # Search events
        events = await audit_system.search_audit_events(
            query="network scan",
            session_id="test_session_001"
        )
        
        print(f"Found {len(events)} events")
        
        # Create replay
        replay_id = await audit_system.create_session_replay("test_session_001")
        print(f"Created replay: {replay_id}")
        
        # End session
        await audit_system.end_audit_session("test_session_001")
        
        # Get stats
        stats = await audit_system.get_system_stats()
        print(f"System stats: {stats}")
    
    asyncio.run(main())