"""
System and agent performance profiling capabilities.
"""

import time
import psutil
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import cProfile
import pstats
import io
from contextlib import contextmanager
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceProfile:
    """Performance profile data structure."""
    timestamp: datetime
    duration: float
    cpu_usage: float
    memory_usage: float
    network_io: Dict[str, int]
    disk_io: Dict[str, int]
    function_stats: Optional[Dict[str, Any]] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentPerformanceProfile:
    """Agent-specific performance profile."""
    agent_id: str
    decision_time: float
    memory_retrieval_time: float
    action_execution_time: float
    communication_time: float
    total_response_time: float
    memory_usage: float
    cache_hit_rate: float
    error_count: int = 0

class SystemProfiler:
    """System-wide performance profiler."""
    
    def __init__(self):
        self.profiles: List[PerformanceProfile] = []
        self.is_profiling = False
        self.profile_interval = 1.0  # seconds
        self._profiling_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
    def start_profiling(self, interval: float = 1.0) -> None:
        """Start continuous system profiling."""
        if self.is_profiling:
            logger.warning("Profiling already in progress")
            return
            
        self.profile_interval = interval
        self.is_profiling = True
        self._stop_event.clear()
        
        self._profiling_thread = threading.Thread(
            target=self._profiling_loop,
            daemon=True
        )
        self._profiling_thread.start()
        logger.info(f"Started system profiling with {interval}s interval")
        
    def stop_profiling(self) -> None:
        """Stop continuous profiling."""
        if not self.is_profiling:
            return
            
        self.is_profiling = False
        self._stop_event.set()
        
        if self._profiling_thread:
            self._profiling_thread.join(timeout=5.0)
            
        logger.info("Stopped system profiling")
        
    def _profiling_loop(self) -> None:
        """Main profiling loop."""
        while not self._stop_event.wait(self.profile_interval):
            try:
                profile = self._capture_system_profile()
                self.profiles.append(profile)
                
                # Keep only last 1000 profiles to prevent memory bloat
                if len(self.profiles) > 1000:
                    self.profiles = self.profiles[-1000:]
                    
            except Exception as e:
                logger.error(f"Error during profiling: {e}")
                
    def _capture_system_profile(self) -> PerformanceProfile:
        """Capture current system performance metrics."""
        start_time = time.time()
        
        # CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
        # Disk I/O
        disk_io_counters = psutil.disk_io_counters()
        disk_io = {
            'read_bytes': disk_io_counters.read_bytes,
            'write_bytes': disk_io_counters.write_bytes,
            'read_count': disk_io_counters.read_count,
            'write_count': disk_io_counters.write_count
        } if disk_io_counters else {}
        
        duration = time.time() - start_time
        
        return PerformanceProfile(
            timestamp=datetime.now(),
            duration=duration,
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            network_io=network_io,
            disk_io=disk_io
        )
        
    @contextmanager
    def profile_function(self, function_name: str):
        """Context manager for profiling specific functions."""
        profiler = cProfile.Profile()
        start_time = time.time()
        
        try:
            profiler.enable()
            yield
        finally:
            profiler.disable()
            duration = time.time() - start_time
            
            # Capture function statistics
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            # Create profile with function stats
            profile = self._capture_system_profile()
            profile.function_stats = {
                'function_name': function_name,
                'duration': duration,
                'stats': stats_stream.getvalue()
            }
            
            self.profiles.append(profile)
            logger.info(f"Profiled function '{function_name}' - Duration: {duration:.3f}s")
            
    def get_performance_summary(self, 
                              time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get performance summary for specified time window."""
        if not self.profiles:
            return {}
            
        # Filter profiles by time window
        profiles = self.profiles
        if time_window:
            cutoff_time = datetime.now() - time_window
            profiles = [p for p in profiles if p.timestamp >= cutoff_time]
            
        if not profiles:
            return {}
            
        # Calculate statistics
        cpu_values = [p.cpu_usage for p in profiles]
        memory_values = [p.memory_usage for p in profiles]
        
        return {
            'profile_count': len(profiles),
            'time_range': {
                'start': profiles[0].timestamp.isoformat(),
                'end': profiles[-1].timestamp.isoformat()
            },
            'cpu_usage': {
                'avg': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'memory_usage': {
                'avg': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values)
            },
            'network_io': {
                'total_bytes_sent': sum(p.network_io.get('bytes_sent', 0) for p in profiles),
                'total_bytes_recv': sum(p.network_io.get('bytes_recv', 0) for p in profiles)
            }
        }
        
    def export_profiles(self, filepath: str) -> None:
        """Export profiles to JSON file."""
        export_data = []
        for profile in self.profiles:
            data = {
                'timestamp': profile.timestamp.isoformat(),
                'duration': profile.duration,
                'cpu_usage': profile.cpu_usage,
                'memory_usage': profile.memory_usage,
                'network_io': profile.network_io,
                'disk_io': profile.disk_io,
                'custom_metrics': profile.custom_metrics
            }
            if profile.function_stats:
                data['function_stats'] = profile.function_stats
            export_data.append(data)
            
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported {len(export_data)} profiles to {filepath}")

class AgentProfiler:
    """Agent-specific performance profiler."""
    
    def __init__(self):
        self.agent_profiles: Dict[str, List[AgentPerformanceProfile]] = {}
        self._active_sessions: Dict[str, Dict[str, float]] = {}
        
    def start_agent_session(self, agent_id: str) -> None:
        """Start profiling session for an agent."""
        self._active_sessions[agent_id] = {
            'session_start': time.time(),
            'decision_start': None,
            'memory_retrieval_start': None,
            'action_execution_start': None,
            'communication_start': None
        }
        
    def mark_decision_start(self, agent_id: str) -> None:
        """Mark start of decision-making phase."""
        if agent_id in self._active_sessions:
            self._active_sessions[agent_id]['decision_start'] = time.time()
            
    def mark_decision_end(self, agent_id: str) -> None:
        """Mark end of decision-making phase."""
        if agent_id in self._active_sessions and self._active_sessions[agent_id]['decision_start']:
            session = self._active_sessions[agent_id]
            session['decision_time'] = time.time() - session['decision_start']
            
    def mark_memory_retrieval_start(self, agent_id: str) -> None:
        """Mark start of memory retrieval."""
        if agent_id in self._active_sessions:
            self._active_sessions[agent_id]['memory_retrieval_start'] = time.time()
            
    def mark_memory_retrieval_end(self, agent_id: str) -> None:
        """Mark end of memory retrieval."""
        if agent_id in self._active_sessions and self._active_sessions[agent_id]['memory_retrieval_start']:
            session = self._active_sessions[agent_id]
            session['memory_retrieval_time'] = time.time() - session['memory_retrieval_start']
            
    def mark_action_execution_start(self, agent_id: str) -> None:
        """Mark start of action execution."""
        if agent_id in self._active_sessions:
            self._active_sessions[agent_id]['action_execution_start'] = time.time()
            
    def mark_action_execution_end(self, agent_id: str) -> None:
        """Mark end of action execution."""
        if agent_id in self._active_sessions and self._active_sessions[agent_id]['action_execution_start']:
            session = self._active_sessions[agent_id]
            session['action_execution_time'] = time.time() - session['action_execution_start']
            
    def mark_communication_start(self, agent_id: str) -> None:
        """Mark start of communication."""
        if agent_id in self._active_sessions:
            self._active_sessions[agent_id]['communication_start'] = time.time()
            
    def mark_communication_end(self, agent_id: str) -> None:
        """Mark end of communication."""
        if agent_id in self._active_sessions and self._active_sessions[agent_id]['communication_start']:
            session = self._active_sessions[agent_id]
            session['communication_time'] = time.time() - session['communication_start']
            
    def end_agent_session(self, agent_id: str, 
                         memory_usage: float = 0.0,
                         cache_hit_rate: float = 0.0,
                         error_count: int = 0) -> AgentPerformanceProfile:
        """End profiling session and create performance profile."""
        if agent_id not in self._active_sessions:
            raise ValueError(f"No active session for agent {agent_id}")
            
        session = self._active_sessions[agent_id]
        total_response_time = time.time() - session['session_start']
        
        profile = AgentPerformanceProfile(
            agent_id=agent_id,
            decision_time=session.get('decision_time', 0.0),
            memory_retrieval_time=session.get('memory_retrieval_time', 0.0),
            action_execution_time=session.get('action_execution_time', 0.0),
            communication_time=session.get('communication_time', 0.0),
            total_response_time=total_response_time,
            memory_usage=memory_usage,
            cache_hit_rate=cache_hit_rate,
            error_count=error_count
        )
        
        # Store profile
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = []
        self.agent_profiles[agent_id].append(profile)
        
        # Keep only last 100 profiles per agent
        if len(self.agent_profiles[agent_id]) > 100:
            self.agent_profiles[agent_id] = self.agent_profiles[agent_id][-100:]
            
        # Clean up session
        del self._active_sessions[agent_id]
        
        return profile
        
    def get_agent_performance_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get performance summary for specific agent."""
        if agent_id not in self.agent_profiles or not self.agent_profiles[agent_id]:
            return {}
            
        profiles = self.agent_profiles[agent_id]
        
        # Calculate averages
        avg_decision_time = sum(p.decision_time for p in profiles) / len(profiles)
        avg_memory_retrieval_time = sum(p.memory_retrieval_time for p in profiles) / len(profiles)
        avg_action_execution_time = sum(p.action_execution_time for p in profiles) / len(profiles)
        avg_communication_time = sum(p.communication_time for p in profiles) / len(profiles)
        avg_total_response_time = sum(p.total_response_time for p in profiles) / len(profiles)
        avg_cache_hit_rate = sum(p.cache_hit_rate for p in profiles) / len(profiles)
        
        return {
            'agent_id': agent_id,
            'profile_count': len(profiles),
            'avg_decision_time': avg_decision_time,
            'avg_memory_retrieval_time': avg_memory_retrieval_time,
            'avg_action_execution_time': avg_action_execution_time,
            'avg_communication_time': avg_communication_time,
            'avg_total_response_time': avg_total_response_time,
            'avg_cache_hit_rate': avg_cache_hit_rate,
            'total_errors': sum(p.error_count for p in profiles),
            'performance_trend': self._calculate_performance_trend(profiles)
        }
        
    def _calculate_performance_trend(self, profiles: List[AgentPerformanceProfile]) -> str:
        """Calculate performance trend (improving, degrading, stable)."""
        if len(profiles) < 10:
            return "insufficient_data"
            
        # Compare recent vs older performance
        recent_profiles = profiles[-10:]
        older_profiles = profiles[-20:-10] if len(profiles) >= 20 else profiles[:-10]
        
        recent_avg = sum(p.total_response_time for p in recent_profiles) / len(recent_profiles)
        older_avg = sum(p.total_response_time for p in older_profiles) / len(older_profiles)
        
        improvement_threshold = 0.1  # 10% improvement threshold
        
        if recent_avg < older_avg * (1 - improvement_threshold):
            return "improving"
        elif recent_avg > older_avg * (1 + improvement_threshold):
            return "degrading"
        else:
            return "stable"
            
    def get_all_agents_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for all agents."""
        return {
            agent_id: self.get_agent_performance_summary(agent_id)
            for agent_id in self.agent_profiles.keys()
        }

# Decorator for automatic function profiling
def profile_function(profiler: SystemProfiler, function_name: Optional[str] = None):
    """Decorator for automatic function profiling."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            name = function_name or f"{func.__module__}.{func.__name__}"
            with profiler.profile_function(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Context manager for agent profiling
@contextmanager
def profile_agent_operation(profiler: AgentProfiler, agent_id: str, operation: str):
    """Context manager for profiling agent operations."""
    if operation == "decision":
        profiler.mark_decision_start(agent_id)
        try:
            yield
        finally:
            profiler.mark_decision_end(agent_id)
    elif operation == "memory_retrieval":
        profiler.mark_memory_retrieval_start(agent_id)
        try:
            yield
        finally:
            profiler.mark_memory_retrieval_end(agent_id)
    elif operation == "action_execution":
        profiler.mark_action_execution_start(agent_id)
        try:
            yield
        finally:
            profiler.mark_action_execution_end(agent_id)
    elif operation == "communication":
        profiler.mark_communication_start(agent_id)
        try:
            yield
        finally:
            profiler.mark_communication_end(agent_id)
    else:
        yield