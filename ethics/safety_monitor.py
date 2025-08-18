"""
Safety monitoring with anomaly detection for agent behavior.
"""

import time
import threading
import logging
import statistics
import math
from typing import Dict, List, Optional, Callable, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import hashlib

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of behavioral anomalies."""
    FREQUENCY_ANOMALY = "frequency_anomaly"      # Unusual action frequency
    PATTERN_ANOMALY = "pattern_anomaly"          # Unexpected behavior patterns
    RESOURCE_ANOMALY = "resource_anomaly"        # Abnormal resource usage
    COMMUNICATION_ANOMALY = "communication_anomaly"  # Unusual communication patterns
    TIMING_ANOMALY = "timing_anomaly"            # Unexpected timing patterns
    CAPABILITY_ANOMALY = "capability_anomaly"    # Trying unauthorized actions
    PERFORMANCE_ANOMALY = "performance_anomaly"   # Performance degradation/enhancement

class AlertSeverity(Enum):
    """Severity levels for safety alerts."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class BehaviorBaseline:
    """Baseline behavior profile for an agent."""
    agent_id: str
    actions_per_minute: float = 0.0
    avg_response_time: float = 0.0
    common_actions: Dict[str, int] = field(default_factory=dict)
    communication_partners: Set[str] = field(default_factory=set)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    activity_periods: List[Tuple[int, int]] = field(default_factory=list)  # (start_hour, end_hour)
    success_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)
    observation_count: int = 0

@dataclass
class SafetyAlert:
    """Safety monitoring alert."""
    alert_id: str
    agent_id: str
    anomaly_type: AnomalyType
    severity: AlertSeverity
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    baseline_value: Optional[float] = None
    observed_value: Optional[float] = None
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: str = ""

class AnomalyDetector:
    """
    Statistical anomaly detector for agent behavior.
    
    Uses multiple detection methods:
    - Statistical outlier detection (Z-score, IQR)
    - Moving average analysis
    - Pattern recognition
    - Frequency analysis
    """
    
    def __init__(self, window_size: int = 100):
        """Initialize anomaly detector.
        
        Args:
            window_size: Size of sliding window for analysis
        """
        self.window_size = window_size
        self.z_score_threshold = 3.0  # Standard deviations for outlier detection
        self.iqr_multiplier = 1.5     # IQR multiplier for outlier detection
        
    def detect_frequency_anomaly(self, current_frequency: float, 
                                historical_frequencies: List[float]) -> Tuple[bool, float]:
        """Detect frequency anomalies using statistical methods.
        
        Args:
            current_frequency: Current frequency value
            historical_frequencies: Historical frequency values
            
        Returns:
            Tuple of (is_anomaly, confidence)
        """
        if len(historical_frequencies) < 10:
            return False, 0.0
            
        mean_freq = statistics.mean(historical_frequencies)
        std_freq = statistics.stdev(historical_frequencies) if len(historical_frequencies) > 1 else 1.0
        
        if std_freq == 0:
            std_freq = 0.1  # Avoid division by zero
            
        # Z-score analysis
        z_score = abs(current_frequency - mean_freq) / std_freq
        z_anomaly = z_score > self.z_score_threshold
        
        # IQR analysis
        q1 = statistics.quantiles(historical_frequencies, n=4)[0]
        q3 = statistics.quantiles(historical_frequencies, n=4)[2]
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        iqr_anomaly = current_frequency < lower_bound or current_frequency > upper_bound
        
        # Combine detections
        is_anomaly = z_anomaly or iqr_anomaly
        confidence = min(1.0, z_score / self.z_score_threshold) if is_anomaly else 0.0
        
        return is_anomaly, confidence
        
    def detect_pattern_anomaly(self, current_pattern: List[str],
                              historical_patterns: List[List[str]]) -> Tuple[bool, float]:
        """Detect pattern anomalies in action sequences.
        
        Args:
            current_pattern: Current action pattern
            historical_patterns: Historical action patterns
            
        Returns:
            Tuple of (is_anomaly, confidence)
        """
        if len(historical_patterns) < 5:
            return False, 0.0
            
        # Calculate pattern similarity scores
        similarities = []
        for hist_pattern in historical_patterns:
            similarity = self._calculate_pattern_similarity(current_pattern, hist_pattern)
            similarities.append(similarity)
            
        avg_similarity = statistics.mean(similarities)
        
        # If current pattern is significantly different from historical patterns
        is_anomaly = avg_similarity < 0.3  # Less than 30% similarity
        confidence = 1.0 - avg_similarity if is_anomaly else 0.0
        
        return is_anomaly, confidence
        
    def detect_timing_anomaly(self, current_timing: float,
                             historical_timings: List[float]) -> Tuple[bool, float]:
        """Detect timing anomalies.
        
        Args:
            current_timing: Current timing value
            historical_timings: Historical timing values
            
        Returns:
            Tuple of (is_anomaly, confidence)
        """
        if len(historical_timings) < 10:
            return False, 0.0
            
        # Use frequency anomaly detection for timing data
        return self.detect_frequency_anomaly(current_timing, historical_timings)
        
    def _calculate_pattern_similarity(self, pattern1: List[str], pattern2: List[str]) -> float:
        """Calculate similarity between two action patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score between 0 and 1
        """
        if not pattern1 or not pattern2:
            return 0.0
            
        # Simple Jaccard similarity
        set1 = set(pattern1)
        set2 = set(pattern2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

class SafetyMonitor:
    """
    Comprehensive safety monitoring system with anomaly detection.
    
    Features:
    - Behavioral baseline establishment
    - Real-time anomaly detection
    - Multi-dimensional safety analysis
    - Alert generation and management
    - Automatic threat response integration
    """
    
    def __init__(self):
        """Initialize safety monitor."""
        self.baselines: Dict[str, BehaviorBaseline] = {}
        self.alerts: List[SafetyAlert] = []
        self.behavior_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        self._lock = threading.RLock()
        self.detector = AnomalyDetector()
        
        # Monitoring parameters
        self.baseline_window = 1000  # Number of observations to establish baseline
        self.min_observations = 50   # Minimum observations before anomaly detection
        self.alert_cooldown = 300    # Seconds between similar alerts
        
        # Last alert times to prevent spam
        self._last_alert_times: Dict[Tuple[str, AnomalyType], float] = {}
        
        # Callbacks
        self._alert_callbacks: List[Callable[[SafetyAlert], None]] = []
        
    def observe_behavior(self, agent_id: str, action_type: str, 
                        action_data: Dict[str, Any]):
        """Observe and record agent behavior.
        
        Args:
            agent_id: ID of the agent
            action_type: Type of action performed
            action_data: Data about the action
        """
        current_time = time.time()
        
        with self._lock:
            # Initialize baseline if needed
            if agent_id not in self.baselines:
                self.baselines[agent_id] = BehaviorBaseline(agent_id=agent_id)
                
            baseline = self.baselines[agent_id]
            
            # Record behavior observation
            observation = {
                "timestamp": current_time,
                "action_type": action_type,
                "response_time": action_data.get("response_time", 0.0),
                "success": action_data.get("success", True),
                "resource_usage": action_data.get("resource_usage", {}),
                "communication_target": action_data.get("target"),
                "metadata": action_data
            }
            
            self.behavior_history[agent_id].append(observation)
            
            # Update baseline
            self._update_baseline(baseline, observation)
            
            # Check for anomalies if we have sufficient data
            if baseline.observation_count >= self.min_observations:
                self._check_anomalies(agent_id, observation)
                
    def get_agent_baseline(self, agent_id: str) -> Optional[BehaviorBaseline]:
        """Get behavioral baseline for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            BehaviorBaseline or None if not found
        """
        with self._lock:
            return self.baselines.get(agent_id)
            
    def get_active_alerts(self, agent_id: Optional[str] = None, 
                         severity: Optional[AlertSeverity] = None) -> List[SafetyAlert]:
        """Get active safety alerts.
        
        Args:
            agent_id: Optional agent ID filter
            severity: Optional severity filter
            
        Returns:
            List of matching active alerts
        """
        with self._lock:
            alerts = [a for a in self.alerts if not a.resolved]
            
            if agent_id:
                alerts = [a for a in alerts if a.agent_id == agent_id]
                
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
                
            return alerts
            
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge a safety alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: Who acknowledged the alert
        """
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Safety alert {alert_id} acknowledged by {acknowledged_by}")
                    break
                    
    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: str):
        """Resolve a safety alert.
        
        Args:
            alert_id: ID of the alert to resolve
            resolved_by: Who resolved the alert
            resolution_notes: Notes about the resolution
        """
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolution_notes = f"Resolved by {resolved_by}: {resolution_notes}"
                    logger.info(f"Safety alert {alert_id} resolved: {resolution_notes}")
                    break
                    
    def get_safety_summary(self) -> Dict[str, Any]:
        """Get safety monitoring summary.
        
        Returns:
            Dictionary with safety statistics
        """
        with self._lock:
            total_alerts = len(self.alerts)
            active_alerts = len([a for a in self.alerts if not a.resolved])
            
            by_severity = defaultdict(int)
            by_type = defaultdict(int)
            
            for alert in self.alerts:
                by_severity[alert.severity.value] += 1
                by_type[alert.anomaly_type.value] += 1
                
            return {
                "monitored_agents": len(self.baselines),
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "alerts_by_severity": dict(by_severity),
                "alerts_by_type": dict(by_type),
                "recent_alerts": len([a for a in self.alerts 
                                    if time.time() - a.timestamp < 3600])
            }
            
    def add_alert_callback(self, callback: Callable[[SafetyAlert], None]):
        """Add callback for safety alerts.
        
        Args:
            callback: Function to call when alerts are generated
        """
        self._alert_callbacks.append(callback)
        
    def _update_baseline(self, baseline: BehaviorBaseline, observation: Dict[str, Any]):
        """Update behavioral baseline with new observation.
        
        Args:
            baseline: Baseline to update
            observation: New observation
        """
        baseline.observation_count += 1
        baseline.last_updated = time.time()
        
        # Update action frequency (actions per minute)
        time_window = 60.0  # 1 minute
        recent_actions = [obs for obs in self.behavior_history[baseline.agent_id]
                         if time.time() - obs["timestamp"] <= time_window]
        baseline.actions_per_minute = len(recent_actions)
        
        # Update average response time
        response_times = [obs["response_time"] for obs in self.behavior_history[baseline.agent_id]
                         if obs["response_time"] > 0]
        if response_times:
            baseline.avg_response_time = statistics.mean(response_times[-100:])  # Last 100
            
        # Update common actions
        action_type = observation["action_type"]
        if action_type in baseline.common_actions:
            baseline.common_actions[action_type] += 1
        else:
            baseline.common_actions[action_type] = 1
            
        # Update communication partners
        target = observation.get("communication_target")
        if target:
            baseline.communication_partners.add(target)
            
        # Update success rate
        recent_observations = list(self.behavior_history[baseline.agent_id])[-100:]
        recent_successes = [obs["success"] for obs in recent_observations]
        if recent_successes:
            baseline.success_rate = sum(recent_successes) / len(recent_successes)
            
    def _check_anomalies(self, agent_id: str, observation: Dict[str, Any]):
        """Check for anomalies in the observation.
        
        Args:
            agent_id: ID of the agent
            observation: Current observation
        """
        baseline = self.baselines[agent_id]
        history = list(self.behavior_history[agent_id])
        
        # Check frequency anomalies
        self._check_frequency_anomaly(agent_id, baseline, history)
        
        # Check pattern anomalies
        self._check_pattern_anomaly(agent_id, baseline, history)
        
        # Check timing anomalies
        self._check_timing_anomaly(agent_id, baseline, observation, history)
        
        # Check resource anomalies
        self._check_resource_anomaly(agent_id, baseline, observation, history)
        
        # Check communication anomalies
        self._check_communication_anomaly(agent_id, baseline, observation, history)
        
    def _check_frequency_anomaly(self, agent_id: str, baseline: BehaviorBaseline, 
                               history: List[Dict[str, Any]]):
        """Check for frequency anomalies.
        
        Args:
            agent_id: Agent ID
            baseline: Agent baseline
            history: Behavior history
        """
        if len(history) < 50:
            return
            
        # Calculate historical frequencies
        time_window = 60.0
        current_time = time.time()
        
        historical_frequencies = []
        for i in range(0, len(history) - 10, 10):  # Sample every 10 observations
            window_start = history[i]["timestamp"]
            window_actions = [obs for obs in history[i:i+50]
                            if window_start <= obs["timestamp"] <= window_start + time_window]
            freq = len(window_actions) / (time_window / 60.0)  # Actions per minute
            historical_frequencies.append(freq)
            
        if historical_frequencies:
            is_anomaly, confidence = self.detector.detect_frequency_anomaly(
                baseline.actions_per_minute, historical_frequencies)
                
            if is_anomaly and self._should_create_alert(agent_id, AnomalyType.FREQUENCY_ANOMALY):
                severity = AlertSeverity.HIGH if confidence > 0.8 else AlertSeverity.MEDIUM
                self._create_alert(
                    agent_id, AnomalyType.FREQUENCY_ANOMALY, severity,
                    f"Unusual action frequency: {baseline.actions_per_minute:.1f} actions/min",
                    {"current_frequency": baseline.actions_per_minute,
                     "historical_mean": statistics.mean(historical_frequencies)},
                    statistics.mean(historical_frequencies),
                    baseline.actions_per_minute,
                    confidence
                )
                
    def _check_pattern_anomaly(self, agent_id: str, baseline: BehaviorBaseline,
                             history: List[Dict[str, Any]]):
        """Check for pattern anomalies.
        
        Args:
            agent_id: Agent ID
            baseline: Agent baseline  
            history: Behavior history
        """
        if len(history) < 100:
            return
            
        # Get recent action pattern
        recent_actions = [obs["action_type"] for obs in history[-20:]]
        
        # Get historical patterns
        historical_patterns = []
        for i in range(0, len(history) - 40, 20):
            pattern = [obs["action_type"] for obs in history[i:i+20]]
            historical_patterns.append(pattern)
            
        if historical_patterns:
            is_anomaly, confidence = self.detector.detect_pattern_anomaly(
                recent_actions, historical_patterns)
                
            if is_anomaly and self._should_create_alert(agent_id, AnomalyType.PATTERN_ANOMALY):
                severity = AlertSeverity.MEDIUM if confidence > 0.7 else AlertSeverity.LOW
                self._create_alert(
                    agent_id, AnomalyType.PATTERN_ANOMALY, severity,
                    f"Unusual behavior pattern detected",
                    {"recent_pattern": recent_actions[:10],  # First 10 for brevity
                     "pattern_similarity": 1.0 - confidence},
                    confidence=confidence
                )
                
    def _check_timing_anomaly(self, agent_id: str, baseline: BehaviorBaseline,
                            observation: Dict[str, Any], history: List[Dict[str, Any]]):
        """Check for timing anomalies.
        
        Args:
            agent_id: Agent ID
            baseline: Agent baseline
            observation: Current observation
            history: Behavior history
        """
        response_time = observation.get("response_time", 0)
        if response_time <= 0:
            return
            
        historical_times = [obs["response_time"] for obs in history[-100:]
                          if obs["response_time"] > 0]
        
        if len(historical_times) >= 10:
            is_anomaly, confidence = self.detector.detect_timing_anomaly(
                response_time, historical_times)
                
            if is_anomaly and self._should_create_alert(agent_id, AnomalyType.TIMING_ANOMALY):
                severity = AlertSeverity.HIGH if response_time > baseline.avg_response_time * 5 else AlertSeverity.MEDIUM
                self._create_alert(
                    agent_id, AnomalyType.TIMING_ANOMALY, severity,
                    f"Unusual response time: {response_time:.2f}s vs avg {baseline.avg_response_time:.2f}s",
                    {"response_time": response_time, "baseline_avg": baseline.avg_response_time},
                    baseline.avg_response_time,
                    response_time,
                    confidence
                )
                
    def _check_resource_anomaly(self, agent_id: str, baseline: BehaviorBaseline,
                              observation: Dict[str, Any], history: List[Dict[str, Any]]):
        """Check for resource usage anomalies.
        
        Args:
            agent_id: Agent ID
            baseline: Agent baseline
            observation: Current observation
            history: Behavior history
        """
        resource_usage = observation.get("resource_usage", {})
        if not resource_usage:
            return
            
        # Check CPU usage anomaly
        cpu_usage = resource_usage.get("cpu_percent", 0)
        if cpu_usage > 0:
            historical_cpu = [obs.get("resource_usage", {}).get("cpu_percent", 0)
                            for obs in history[-100:] if obs.get("resource_usage", {}).get("cpu_percent", 0) > 0]
            
            if len(historical_cpu) >= 10:
                is_anomaly, confidence = self.detector.detect_frequency_anomaly(cpu_usage, historical_cpu)
                
                if is_anomaly and cpu_usage > statistics.mean(historical_cpu) * 2:
                    if self._should_create_alert(agent_id, AnomalyType.RESOURCE_ANOMALY):
                        severity = AlertSeverity.HIGH if cpu_usage > 80 else AlertSeverity.MEDIUM
                        self._create_alert(
                            agent_id, AnomalyType.RESOURCE_ANOMALY, severity,
                            f"Unusual CPU usage: {cpu_usage:.1f}% vs avg {statistics.mean(historical_cpu):.1f}%",
                            {"cpu_usage": cpu_usage, "historical_avg": statistics.mean(historical_cpu)},
                            statistics.mean(historical_cpu),
                            cpu_usage,
                            confidence
                        )
                        
    def _check_communication_anomaly(self, agent_id: str, baseline: BehaviorBaseline,
                                   observation: Dict[str, Any], history: List[Dict[str, Any]]):
        """Check for communication anomalies.
        
        Args:
            agent_id: Agent ID
            baseline: Agent baseline
            observation: Current observation
            history: Behavior history
        """
        target = observation.get("communication_target")
        if not target:
            return
            
        # Check if communicating with new/unusual targets
        if target not in baseline.communication_partners:
            # This is a new communication partner
            if len(baseline.communication_partners) > 5:  # Only alert if agent has established patterns
                if self._should_create_alert(agent_id, AnomalyType.COMMUNICATION_ANOMALY):
                    self._create_alert(
                        agent_id, AnomalyType.COMMUNICATION_ANOMALY, AlertSeverity.LOW,
                        f"Communication with new target: {target}",
                        {"target": target, "known_partners": len(baseline.communication_partners)},
                        confidence=0.6
                    )
                    
    def _should_create_alert(self, agent_id: str, anomaly_type: AnomalyType) -> bool:
        """Check if alert should be created based on cooldown.
        
        Args:
            agent_id: Agent ID
            anomaly_type: Type of anomaly
            
        Returns:
            True if alert should be created
        """
        key = (agent_id, anomaly_type)
        last_alert = self._last_alert_times.get(key, 0)
        
        if time.time() - last_alert < self.alert_cooldown:
            return False
            
        self._last_alert_times[key] = time.time()
        return True
        
    def _create_alert(self, agent_id: str, anomaly_type: AnomalyType, 
                     severity: AlertSeverity, description: str,
                     evidence: Dict[str, Any], baseline_value: Optional[float] = None,
                     observed_value: Optional[float] = None, confidence: float = 0.0):
        """Create a safety alert.
        
        Args:
            agent_id: Agent ID
            anomaly_type: Type of anomaly
            severity: Alert severity
            description: Alert description
            evidence: Evidence data
            baseline_value: Baseline value for comparison
            observed_value: Observed value
            confidence: Confidence in anomaly detection
        """
        alert_id = hashlib.md5(f"{agent_id}_{anomaly_type.value}_{time.time()}".encode()).hexdigest()[:12]
        
        alert = SafetyAlert(
            alert_id=alert_id,
            agent_id=agent_id,
            anomaly_type=anomaly_type,
            severity=severity,
            description=description,
            evidence=evidence,
            baseline_value=baseline_value,
            observed_value=observed_value,
            confidence=confidence
        )
        
        with self._lock:
            self.alerts.append(alert)
            
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
                
        logger.warning(f"Safety alert created: {alert_id} - {description}")

# Global safety monitor instance
safety_monitor = SafetyMonitor()