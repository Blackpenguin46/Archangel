#!/usr/bin/env python3
"""
Comprehensive Logging System for Archangel AI vs AI
Stores detailed logs for analysis and review
"""

import logging
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import uuid

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, timedelta)):
        return obj.isoformat()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    if hasattr(obj, '_asdict'):
        return obj._asdict()
    return str(obj)

def safe_json_dumps(obj, **kwargs):
    """Safe JSON dumps that handles datetime and other objects"""
    return json.dumps(obj, default=json_serial, **kwargs)

@dataclass
class AIDecision:
    """Represents a single AI decision"""
    agent_id: str
    agent_type: str  # 'red' or 'blue'
    decision_type: str
    reasoning_text: str
    confidence_score: float
    input_state: Dict[str, Any]
    output_action: Dict[str, Any]
    timestamp: datetime
    execution_success: bool
    learned_from: List[str]  # What this decision learned from
    
@dataclass
class LearningEvent:
    """Represents AI learning from experience"""
    agent_id: str
    event_type: str  # 'success', 'failure', 'adaptation'
    trigger: str
    old_behavior: str
    new_behavior: str
    confidence_change: float
    timestamp: datetime
    
@dataclass
class SystemEvent:
    """General system events"""
    event_type: str
    description: str
    affected_systems: List[str]
    severity: str
    timestamp: datetime
    metadata: Dict[str, Any]

class ArchangelLogger:
    """Advanced logging system for AI vs AI analysis"""
    
    def __init__(self, session_id: str, log_dir: str = "logs"):
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize databases
        self.db_path = self.log_dir / f"archangel_session_{session_id}.db"
        self.init_database()
        
        # Initialize file loggers
        self.setup_file_loggers()
        
        # In-memory stores for analysis
        self.ai_decisions: List[AIDecision] = []
        self.learning_events: List[LearningEvent] = []
        self.system_events: List[SystemEvent] = []
        
        # Performance tracking
        self.red_team_performance = []
        self.blue_team_performance = []
        self.adaptation_metrics = {}
        
        self.logger = logging.getLogger(f"archangel_logger_{session_id}")
        self.logger.info(f"🗂️ Archangel logging system initialized for session {session_id}")
    
    def init_database(self):
        """Initialize SQLite database for structured logging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # AI Decisions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                agent_id TEXT,
                agent_type TEXT,
                decision_type TEXT,
                reasoning_text TEXT,
                confidence_score REAL,
                input_state TEXT,
                output_action TEXT,
                timestamp TEXT,
                execution_success BOOLEAN,
                learned_from TEXT
            )
        """)
        
        # Learning Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                agent_id TEXT,
                event_type TEXT,
                trigger_event TEXT,
                old_behavior TEXT,
                new_behavior TEXT,
                confidence_change REAL,
                timestamp TEXT
            )
        """)
        
        # System Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                event_type TEXT,
                description TEXT,
                affected_systems TEXT,
                severity TEXT,
                timestamp TEXT,
                metadata TEXT
            )
        """)
        
        # Performance Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                agent_id TEXT,
                metric_type TEXT,
                metric_value REAL,
                timestamp TEXT,
                context TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def setup_file_loggers(self):
        """Setup file-based loggers for different categories"""
        session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # AI Reasoning log
        self.ai_reasoning_logger = logging.getLogger(f"ai_reasoning_{self.session_id}")
        ai_handler = logging.FileHandler(
            self.log_dir / f"ai_reasoning_{session_time}.log"
        )
        ai_handler.setFormatter(logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(message)s'
        ))
        self.ai_reasoning_logger.addHandler(ai_handler)
        self.ai_reasoning_logger.setLevel(logging.DEBUG)
        
        # Learning Evolution log
        self.learning_logger = logging.getLogger(f"learning_{self.session_id}")
        learning_handler = logging.FileHandler(
            self.log_dir / f"learning_evolution_{session_time}.log"
        )
        learning_handler.setFormatter(logging.Formatter(
            '%(asctime)s - [LEARNING] - %(message)s'
        ))
        self.learning_logger.addHandler(learning_handler)
        self.learning_logger.setLevel(logging.INFO)
        
        # System Events log
        self.system_logger = logging.getLogger(f"system_{self.session_id}")
        system_handler = logging.FileHandler(
            self.log_dir / f"system_events_{session_time}.log"
        )
        system_handler.setFormatter(logging.Formatter(
            '%(asctime)s - [SYSTEM] - %(message)s'
        ))
        self.system_logger.addHandler(system_handler)
        self.system_logger.setLevel(logging.INFO)
    
    def log_ai_decision(self, decision: AIDecision):
        """Log an AI decision with full reasoning"""
        # Store in memory
        self.ai_decisions.append(decision)
        
        # Log to file
        self.ai_reasoning_logger.info(
            f"🤖 {decision.agent_type.upper()} AGENT {decision.agent_id} - {decision.decision_type}\n"
            f"   Reasoning: {decision.reasoning_text}\n"
            f"   Confidence: {decision.confidence_score:.2f}\n"
            f"   Action: {decision.output_action}\n"
            f"   Success: {decision.execution_success}\n"
            f"   Learned From: {', '.join(decision.learned_from) if decision.learned_from else 'None'}"
        )
        
        # Store in database
        self._store_ai_decision_db(decision)
    
    def log_learning_event(self, learning: LearningEvent):
        """Log AI learning and adaptation"""
        # Store in memory
        self.learning_events.append(learning)
        
        # Log to file
        self.learning_logger.info(
            f"🧠 AGENT {learning.agent_id} LEARNED - {learning.event_type.upper()}\n"
            f"   Trigger: {learning.trigger}\n"
            f"   Old Behavior: {learning.old_behavior}\n"
            f"   New Behavior: {learning.new_behavior}\n"
            f"   Confidence Change: {learning.confidence_change:+.2f}"
        )
        
        # Store in database
        self._store_learning_event_db(learning)
    
    def log_system_event(self, event: SystemEvent):
        """Log system-level events"""
        # Store in memory
        self.system_events.append(event)
        
        # Log to file
        self.system_logger.info(
            f"⚡ {event.severity.upper()} - {event.event_type}\n"
            f"   Description: {event.description}\n"
            f"   Affected Systems: {', '.join(event.affected_systems)}\n"
            f"   Metadata: {safe_json_dumps(event.metadata, indent=2)}"
        )
        
        # Store in database
        self._store_system_event_db(event)
    
    def track_performance(self, agent_id: str, metric_type: str, value: float, context: str = ""):
        """Track agent performance over time"""
        timestamp = datetime.now()
        
        # Store performance data
        if agent_id.startswith('red'):
            self.red_team_performance.append({
                'agent_id': agent_id,
                'metric_type': metric_type,
                'value': value,
                'timestamp': timestamp,
                'context': context
            })
        elif agent_id.startswith('blue'):
            self.blue_team_performance.append({
                'agent_id': agent_id,
                'metric_type': metric_type,
                'value': value,
                'timestamp': timestamp,
                'context': context
            })
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO performance_metrics 
            (id, session_id, agent_id, metric_type, metric_value, timestamp, context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), self.session_id, agent_id, metric_type, 
            value, timestamp.isoformat(), context
        ))
        conn.commit()
        conn.close()
    
    def analyze_learning_progression(self) -> Dict[str, Any]:
        """Analyze how AI agents are learning and improving"""
        analysis = {
            'total_decisions': len(self.ai_decisions),
            'total_learning_events': len(self.learning_events),
            'red_team_adaptation': self._analyze_team_adaptation('red'),
            'blue_team_adaptation': self._analyze_team_adaptation('blue'),
            'cross_team_learning': self._analyze_cross_team_learning(),
            'emergent_strategies': self._identify_emergent_strategies(),
            'learning_velocity': self._calculate_learning_velocity()
        }
        
        return analysis
    
    def _analyze_team_adaptation(self, team_type: str) -> Dict[str, Any]:
        """Analyze adaptation patterns for a team"""
        team_decisions = [d for d in self.ai_decisions if d.agent_type == team_type]
        team_learning = [l for l in self.learning_events if l.agent_id.startswith(team_type)]
        
        if not team_decisions:
            return {'adaptation_rate': 0, 'strategy_evolution': 'none'}
        
        # Calculate adaptation metrics
        recent_decisions = team_decisions[-10:] if len(team_decisions) >= 10 else team_decisions
        avg_confidence = sum(d.confidence_score for d in recent_decisions) / len(recent_decisions)
        
        adaptation_rate = len(team_learning) / len(team_decisions) if team_decisions else 0
        
        # Identify strategy patterns
        decision_types = [d.decision_type for d in recent_decisions]
        strategy_diversity = len(set(decision_types)) / len(decision_types) if decision_types else 0
        
        return {
            'total_decisions': len(team_decisions),
            'total_adaptations': len(team_learning),
            'adaptation_rate': adaptation_rate,
            'average_confidence': avg_confidence,
            'strategy_diversity': strategy_diversity,
            'recent_behaviors': decision_types[-5:] if decision_types else []
        }
    
    def _analyze_cross_team_learning(self) -> Dict[str, Any]:
        """Analyze how teams learn from each other"""
        cross_learning_events = []
        
        for learning in self.learning_events:
            if 'opponent' in learning.trigger.lower() or 'counter' in learning.trigger.lower():
                cross_learning_events.append(learning)
        
        return {
            'cross_learning_events': len(cross_learning_events),
            'adaptation_triggers': [l.trigger for l in cross_learning_events[-5:]],
            'competitive_evolution': len(cross_learning_events) > 0
        }
    
    def _identify_emergent_strategies(self) -> List[Dict[str, Any]]:
        """Identify emergent strategies not explicitly programmed"""
        strategies = []
        
        # Look for novel decision patterns
        decision_sequences = {}
        for i in range(len(self.ai_decisions) - 2):
            sequence = (
                self.ai_decisions[i].decision_type,
                self.ai_decisions[i+1].decision_type,
                self.ai_decisions[i+2].decision_type
            )
            decision_sequences[sequence] = decision_sequences.get(sequence, 0) + 1
        
        # Find rare but effective sequences
        for sequence, count in decision_sequences.items():
            if count >= 2:  # Repeated pattern
                strategies.append({
                    'pattern': ' → '.join(sequence),
                    'frequency': count,
                    'emergence_type': 'sequential_strategy'
                })
        
        return strategies[:5]  # Top 5 emergent strategies
    
    def _calculate_learning_velocity(self) -> float:
        """Calculate how fast the AI is learning"""
        if len(self.learning_events) < 2:
            return 0.0
        
        # Learning events per hour
        first_event = self.learning_events[0].timestamp
        last_event = self.learning_events[-1].timestamp
        duration_hours = (last_event - first_event).total_seconds() / 3600
        
        if duration_hours == 0:
            return 0.0
        
        return len(self.learning_events) / duration_hours
    
    def generate_session_report(self) -> str:
        """Generate comprehensive session report"""
        try:
            analysis = self.analyze_learning_progression()
        except Exception as e:
            self.logger.warning(f"Error analyzing learning progression: {e}")
            # Provide fallback analysis
            analysis = {
                'total_decisions': len(self.ai_decisions),
                'total_learning_events': len(self.learning_events),
                'red_team_adaptation': {'total_decisions': 0, 'total_adaptations': 0, 'adaptation_rate': 0, 'average_confidence': 0, 'strategy_diversity': 0},
                'blue_team_adaptation': {'total_decisions': 0, 'total_adaptations': 0, 'adaptation_rate': 0, 'average_confidence': 0, 'strategy_diversity': 0},
                'cross_team_learning': {'cross_learning_events': 0, 'competitive_evolution': False},
                'emergent_strategies': [],
                'learning_velocity': 0.0
            }
        
        # Safely access analysis values with defaults
        total_decisions = analysis.get('total_decisions', 0)
        total_learning = analysis.get('total_learning_events', 0)
        learning_velocity = analysis.get('learning_velocity', 0.0)
        
        red_adaptation = analysis.get('red_team_adaptation', {})
        blue_adaptation = analysis.get('blue_team_adaptation', {})
        cross_learning = analysis.get('cross_team_learning', {})
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🗂️ ARCHANGEL AI vs AI SESSION REPORT                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Session ID: {self.session_id}
║ Duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
║ Database: {self.db_path}
╠══════════════════════════════════════════════════════════════════════════════╣
║                           🤖 AI DECISION ANALYSIS                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Total AI Decisions: {total_decisions}
║ Learning Events: {total_learning}
║ Learning Velocity: {learning_velocity:.2f} events/hour
║
║ 🔴 RED TEAM PERFORMANCE:
║   • Decisions Made: {red_adaptation.get('total_decisions', 0)}
║   • Adaptations: {red_adaptation.get('total_adaptations', 0)}
║   • Adaptation Rate: {red_adaptation.get('adaptation_rate', 0):.2%}
║   • Avg Confidence: {red_adaptation.get('average_confidence', 0):.2f}
║   • Strategy Diversity: {red_adaptation.get('strategy_diversity', 0):.2f}
║
║ 🔵 BLUE TEAM PERFORMANCE:
║   • Decisions Made: {blue_adaptation.get('total_decisions', 0)}
║   • Adaptations: {blue_adaptation.get('total_adaptations', 0)}
║   • Adaptation Rate: {blue_adaptation.get('adaptation_rate', 0):.2%}
║   • Avg Confidence: {blue_adaptation.get('average_confidence', 0):.2f}
║   • Strategy Diversity: {blue_adaptation.get('strategy_diversity', 0):.2f}
║
║ 🧠 CROSS-TEAM LEARNING:
║   • Cross-Learning Events: {cross_learning.get('cross_learning_events', 0)}
║   • Competitive Evolution: {cross_learning.get('competitive_evolution', False)}
║
║ ⚡ EMERGENT STRATEGIES:
"""
        
        emergent_strategies = analysis.get('emergent_strategies', [])
        for strategy in emergent_strategies:
            if isinstance(strategy, dict):
                pattern = strategy.get('pattern', 'Unknown strategy')
                frequency = strategy.get('frequency', 1)
                report += f"║   • {pattern} (x{frequency})\n"
            else:
                report += f"║   • {str(strategy)}\n"
        
        report += f"""║
╠══════════════════════════════════════════════════════════════════════════════╣
║                              📊 LOG FILES                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ • AI Reasoning: logs/ai_reasoning_*.log
║ • Learning Evolution: logs/learning_evolution_*.log  
║ • System Events: logs/system_events_*.log
║ • Database: {self.db_path}
║
║ 🔍 REVIEW COMMANDS:
║   tail -f logs/ai_reasoning_*.log     # Watch AI decisions live
║   grep "LEARNED" logs/learning_*.log  # See learning events
║   sqlite3 {self.db_path}              # Query structured data
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        return report
    
    def _store_ai_decision_db(self, decision: AIDecision):
        """Store AI decision in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ai_decisions 
            (id, session_id, agent_id, agent_type, decision_type, reasoning_text, 
             confidence_score, input_state, output_action, timestamp, execution_success, learned_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), self.session_id, decision.agent_id, decision.agent_type,
            decision.decision_type, decision.reasoning_text, decision.confidence_score,
            safe_json_dumps(decision.input_state), safe_json_dumps(decision.output_action),
            decision.timestamp.isoformat(), decision.execution_success,
            safe_json_dumps(decision.learned_from)
        ))
        conn.commit()
        conn.close()
    
    def _store_learning_event_db(self, learning: LearningEvent):
        """Store learning event in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO learning_events
            (id, session_id, agent_id, event_type, trigger_event, old_behavior, 
             new_behavior, confidence_change, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), self.session_id, learning.agent_id, learning.event_type,
            learning.trigger, learning.old_behavior, learning.new_behavior,
            learning.confidence_change, learning.timestamp.isoformat()
        ))
        conn.commit()
        conn.close()
    
    def _store_system_event_db(self, event: SystemEvent):
        """Store system event in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO system_events
            (id, session_id, event_type, description, affected_systems, severity, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), self.session_id, event.event_type, event.description,
            safe_json_dumps(event.affected_systems), event.severity,
            event.timestamp.isoformat(), safe_json_dumps(event.metadata)
        ))
        conn.commit()
        conn.close()

# Convenience functions for easy logging
def create_ai_decision(agent_id: str, agent_type: str, decision_type: str, 
                      reasoning: str, confidence: float, input_state: Dict,
                      output_action: Dict, success: bool, learned_from: List[str] = None) -> AIDecision:
    """Create an AI decision record"""
    return AIDecision(
        agent_id=agent_id,
        agent_type=agent_type,
        decision_type=decision_type,
        reasoning_text=reasoning,
        confidence_score=confidence,
        input_state=input_state,
        output_action=output_action,
        timestamp=datetime.now(),
        execution_success=success,
        learned_from=learned_from or []
    )

def create_learning_event(agent_id: str, event_type: str, trigger: str,
                         old_behavior: str, new_behavior: str, confidence_change: float) -> LearningEvent:
    """Create a learning event record"""
    return LearningEvent(
        agent_id=agent_id,
        event_type=event_type,
        trigger=trigger,
        old_behavior=old_behavior,
        new_behavior=new_behavior,
        confidence_change=confidence_change,
        timestamp=datetime.now()
    )

def create_system_event(event_type: str, description: str, affected_systems: List[str],
                       severity: str, metadata: Dict[str, Any] = None) -> SystemEvent:
    """Create a system event record"""
    return SystemEvent(
        event_type=event_type,
        description=description,
        affected_systems=affected_systems,
        severity=severity,
        timestamp=datetime.now(),
        metadata=metadata or {}
    )