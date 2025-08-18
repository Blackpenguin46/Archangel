#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Feature Flag System
Production-ready feature flag implementation with runtime control and A/B testing
"""

import logging
import json
import yaml
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any, Union, Callable, Protocol
from enum import Enum
from pathlib import Path
import threading
import time
import asyncio
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class FlagState(Enum):
    """Feature flag states"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    TESTING = "testing"
    CANARY = "canary"
    DEPRECATED = "deprecated"
    KILL_SWITCH = "kill_switch"

class RolloutType(Enum):
    """Rollout type strategies"""
    PERCENTAGE = "percentage"
    USER_ID = "user_id"
    ENVIRONMENT = "environment"
    TIME_BASED = "time_based"
    CUSTOM = "custom"

class FlagEvaluationContext:
    """Context for feature flag evaluation"""
    def __init__(self, 
                 user_id: Optional[str] = None,
                 environment: str = "development",
                 timestamp: Optional[datetime] = None,
                 custom_attributes: Optional[Dict[str, Any]] = None):
        self.user_id = user_id
        self.environment = environment
        self.timestamp = timestamp or datetime.now()
        self.custom_attributes = custom_attributes or {}
        self.session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        data = f"{self.user_id}_{self.environment}_{self.timestamp.isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:8]

@dataclass
class RolloutRule:
    """Rule for feature flag rollout"""
    rule_id: str
    rule_type: RolloutType
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Percentage rollout
    percentage: float = 0.0
    
    # User-based rollout
    user_whitelist: Set[str] = field(default_factory=set)
    user_blacklist: Set[str] = field(default_factory=set)
    
    # Environment rollout
    environments: Set[str] = field(default_factory=set)
    
    # Time-based rollout
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Custom evaluation function
    custom_evaluator: Optional[str] = None
    
    def evaluate(self, context: FlagEvaluationContext) -> bool:
        """Evaluate if rule matches the given context"""
        try:
            if self.rule_type == RolloutType.PERCENTAGE:
                return self._evaluate_percentage(context)
            elif self.rule_type == RolloutType.USER_ID:
                return self._evaluate_user_id(context)
            elif self.rule_type == RolloutType.ENVIRONMENT:
                return self._evaluate_environment(context)
            elif self.rule_type == RolloutType.TIME_BASED:
                return self._evaluate_time_based(context)
            elif self.rule_type == RolloutType.CUSTOM:
                return self._evaluate_custom(context)
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating rollout rule {self.rule_id}: {e}")
            return False
    
    def _evaluate_percentage(self, context: FlagEvaluationContext) -> bool:
        """Evaluate percentage-based rollout"""
        if self.percentage <= 0:
            return False
        if self.percentage >= 100:
            return True
        
        # Use consistent hashing for stable rollout
        hash_input = f"{self.rule_id}_{context.user_id or context.session_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        user_percentage = (hash_value % 10000) / 100.0  # 0-99.99
        
        return user_percentage < self.percentage
    
    def _evaluate_user_id(self, context: FlagEvaluationContext) -> bool:
        """Evaluate user ID-based rollout"""
        if not context.user_id:
            return False
        
        if context.user_id in self.user_blacklist:
            return False
        
        if self.user_whitelist and context.user_id not in self.user_whitelist:
            return False
        
        return True
    
    def _evaluate_environment(self, context: FlagEvaluationContext) -> bool:
        """Evaluate environment-based rollout"""
        return context.environment in self.environments
    
    def _evaluate_time_based(self, context: FlagEvaluationContext) -> bool:
        """Evaluate time-based rollout"""
        now = context.timestamp
        
        if self.start_time and now < self.start_time:
            return False
        
        if self.end_time and now > self.end_time:
            return False
        
        return True
    
    def _evaluate_custom(self, context: FlagEvaluationContext) -> bool:
        """Evaluate custom rollout logic"""
        # In production, this would execute registered custom functions
        # For now, return based on custom conditions
        return self.conditions.get("enabled", False)

@dataclass
class FeatureFlagDefinition:
    """Complete feature flag definition"""
    flag_id: str
    name: str
    description: str
    
    # State and rollout
    state: FlagState = FlagState.DISABLED
    rollout_rules: List[RolloutRule] = field(default_factory=list)
    
    # Metadata
    category: str = ""
    tags: Set[str] = field(default_factory=set)
    owner: str = ""
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Default values
    default_enabled: bool = False
    default_value: Any = None
    
    # Monitoring
    metrics_enabled: bool = True
    alert_on_evaluation_errors: bool = True
    
    # Dependencies
    prerequisite_flags: List[str] = field(default_factory=list)
    conflicting_flags: List[str] = field(default_factory=list)
    
    def is_enabled_for_context(self, context: FlagEvaluationContext) -> bool:
        """Evaluate if flag is enabled for given context"""
        try:
            # Check if flag is expired
            if self.expires_at and context.timestamp > self.expires_at:
                return False
            
            # Check flag state
            if self.state == FlagState.DISABLED:
                return False
            elif self.state == FlagState.KILL_SWITCH:
                return False
            elif self.state == FlagState.ENABLED:
                return True
            elif self.state == FlagState.DEPRECATED:
                return self.default_enabled
            
            # For TESTING and CANARY, check rollout rules
            if self.state in [FlagState.TESTING, FlagState.CANARY]:
                return any(rule.evaluate(context) for rule in self.rollout_rules)
            
            return self.default_enabled
            
        except Exception as e:
            logger.error(f"Error evaluating flag {self.flag_id}: {e}")
            return self.default_enabled

class FlagEvaluationResult:
    """Result of feature flag evaluation"""
    def __init__(self, 
                 flag_id: str,
                 enabled: bool,
                 value: Any = None,
                 reason: str = "",
                 evaluation_time: float = 0.0,
                 context: Optional[FlagEvaluationContext] = None):
        self.flag_id = flag_id
        self.enabled = enabled
        self.value = value
        self.reason = reason
        self.evaluation_time = evaluation_time
        self.context = context
        self.evaluated_at = datetime.now()

class FeatureFlagManager:
    """
    Production-ready feature flag management system.
    
    Features:
    - Runtime flag evaluation with multiple rollout strategies
    - A/B testing and gradual rollouts
    - Flag dependencies and conflict resolution
    - Performance monitoring and metrics
    - Hot reloading of flag configurations
    - Thread-safe operations
    - Context-aware evaluation
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.flags: Dict[str, FeatureFlagDefinition] = {}
        self.config_path = config_path
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics and monitoring
        self.evaluation_count = 0
        self.evaluation_times: List[float] = []
        self.error_count = 0
        
        # Hot reloading
        self._last_config_mtime = 0
        self._reload_interval = 60  # seconds
        self._reload_thread = None
        self._shutdown_event = threading.Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Load initial configuration
        if config_path and config_path.exists():
            self.load_configuration(config_path)
        
        # Start configuration reloader
        self.start_config_reloader()
    
    def create_archangel_feature_flags(self) -> None:
        """Create feature flags for Archangel project"""
        
        flags = [
            # Foundation features
            FeatureFlagDefinition(
                flag_id="multi_agent_coordination",
                name="Multi-Agent Coordination",
                description="Enable multi-agent coordination framework",
                state=FlagState.ENABLED,
                category="foundation",
                tags={"mvp", "foundation"},
                owner="platform-team",
                rollout_rules=[
                    RolloutRule(
                        rule_id="coord_env_rollout",
                        rule_type=RolloutType.ENVIRONMENT,
                        environments={"development", "testing", "staging"}
                    )
                ],
                default_enabled=True
            ),
            
            FeatureFlagDefinition(
                flag_id="basic_memory_system",
                name="Basic Memory System",
                description="Enable vector memory and basic knowledge storage",
                state=FlagState.TESTING,
                category="memory",
                tags={"mvp", "memory"},
                owner="ai-team",
                rollout_rules=[
                    RolloutRule(
                        rule_id="memory_canary",
                        rule_type=RolloutType.PERCENTAGE,
                        percentage=25.0
                    ),
                    RolloutRule(
                        rule_id="memory_dev_env",
                        rule_type=RolloutType.ENVIRONMENT,
                        environments={"development", "testing"}
                    )
                ],
                default_enabled=False,
                prerequisite_flags=["multi_agent_coordination"]
            ),
            
            # Agent features
            FeatureFlagDefinition(
                flag_id="red_team_agents",
                name="Red Team Agents",
                description="Enable red team autonomous agents",
                state=FlagState.CANARY,
                category="agents",
                tags={"mvp", "red-team"},
                owner="red-team",
                rollout_rules=[
                    RolloutRule(
                        rule_id="red_team_canary",
                        rule_type=RolloutType.PERCENTAGE,
                        percentage=10.0
                    ),
                    RolloutRule(
                        rule_id="red_team_whitelist",
                        rule_type=RolloutType.USER_ID,
                        user_whitelist={"admin", "red_team_lead", "tester"}
                    )
                ],
                default_enabled=False,
                prerequisite_flags=["multi_agent_coordination", "basic_memory_system"]
            ),
            
            FeatureFlagDefinition(
                flag_id="blue_team_agents",
                name="Blue Team Agents", 
                description="Enable blue team autonomous agents",
                state=FlagState.CANARY,
                category="agents",
                tags={"mvp", "blue-team"},
                owner="blue-team",
                rollout_rules=[
                    RolloutRule(
                        rule_id="blue_team_canary",
                        rule_type=RolloutType.PERCENTAGE,
                        percentage=10.0
                    ),
                    RolloutRule(
                        rule_id="blue_team_time_window",
                        rule_type=RolloutType.TIME_BASED,
                        start_time=datetime.now(),
                        end_time=datetime.now() + timedelta(days=30)
                    )
                ],
                default_enabled=False,
                prerequisite_flags=["multi_agent_coordination", "basic_memory_system"]
            ),
            
            # Environment and game features
            FeatureFlagDefinition(
                flag_id="basic_environment",
                name="Basic Environment",
                description="Enable basic mock enterprise environment",
                state=FlagState.TESTING,
                category="environment", 
                tags={"mvp", "infrastructure"},
                owner="platform-team",
                rollout_rules=[
                    RolloutRule(
                        rule_id="env_gradual_rollout",
                        rule_type=RolloutType.PERCENTAGE,
                        percentage=50.0
                    )
                ],
                default_enabled=False
            ),
            
            FeatureFlagDefinition(
                flag_id="basic_game_loop",
                name="Basic Game Loop",
                description="Enable basic scenario execution and game loop",
                state=FlagState.DISABLED,
                category="game-logic",
                tags={"mvp", "scenarios"},
                owner="game-team",
                default_enabled=False,
                prerequisite_flags=["red_team_agents", "blue_team_agents", "basic_environment"]
            ),
            
            # Advanced features (disabled by default)
            FeatureFlagDefinition(
                flag_id="advanced_reasoning",
                name="Advanced AI Reasoning",
                description="Enable advanced LLM reasoning and behavior trees",
                state=FlagState.DISABLED,
                category="ai",
                tags={"advanced", "reasoning"},
                owner="ai-team",
                default_enabled=False,
                expires_at=datetime.now() + timedelta(days=90)
            ),
            
            FeatureFlagDefinition(
                flag_id="self_play_learning",
                name="Self-Play Learning",
                description="Enable adversarial self-play and learning systems",
                state=FlagState.DISABLED,
                category="learning",
                tags={"advanced", "ml"},
                owner="ml-team",
                default_enabled=False,
                prerequisite_flags=["advanced_reasoning"]
            ),
            
            # Kill switch for emergency
            FeatureFlagDefinition(
                flag_id="emergency_shutdown",
                name="Emergency Shutdown",
                description="Emergency kill switch for all autonomous operations",
                state=FlagState.DISABLED,
                category="safety",
                tags={"kill-switch", "emergency"},
                owner="security-team",
                default_enabled=False,
                conflicting_flags=["red_team_agents", "blue_team_agents"],
                alert_on_evaluation_errors=True
            )
        ]
        
        # Add flags to manager
        with self._lock:
            for flag in flags:
                self.flags[flag.flag_id] = flag
        
        self.logger.info(f"Created {len(flags)} Archangel feature flags")
    
    def is_enabled(self, 
                   flag_id: str, 
                   context: Optional[FlagEvaluationContext] = None,
                   default: bool = False) -> bool:
        """Check if feature flag is enabled for given context"""
        result = self.evaluate_flag(flag_id, context, default)
        return result.enabled
    
    def get_value(self, 
                  flag_id: str,
                  context: Optional[FlagEvaluationContext] = None,
                  default: Any = None) -> Any:
        """Get feature flag value for given context"""
        result = self.evaluate_flag(flag_id, context, default)
        return result.value if result.value is not None else default
    
    def evaluate_flag(self, 
                     flag_id: str,
                     context: Optional[FlagEvaluationContext] = None,
                     default: Any = False) -> FlagEvaluationResult:
        """Comprehensive feature flag evaluation"""
        start_time = time.time()
        
        try:
            with self._lock:
                self.evaluation_count += 1
                
                # Use default context if none provided
                if context is None:
                    context = FlagEvaluationContext()
                
                # Check if flag exists
                if flag_id not in self.flags:
                    return FlagEvaluationResult(
                        flag_id=flag_id,
                        enabled=bool(default),
                        value=default,
                        reason="Flag not found",
                        evaluation_time=time.time() - start_time,
                        context=context
                    )
                
                flag = self.flags[flag_id]
                
                # Check prerequisites
                if flag.prerequisite_flags:
                    for prereq_flag in flag.prerequisite_flags:
                        if not self.is_enabled(prereq_flag, context):
                            return FlagEvaluationResult(
                                flag_id=flag_id,
                                enabled=False,
                                value=default,
                                reason=f"Prerequisite {prereq_flag} not enabled",
                                evaluation_time=time.time() - start_time,
                                context=context
                            )
                
                # Check conflicting flags
                if flag.conflicting_flags:
                    for conflict_flag in flag.conflicting_flags:
                        if self.is_enabled(conflict_flag, context):
                            return FlagEvaluationResult(
                                flag_id=flag_id,
                                enabled=False,
                                value=default,
                                reason=f"Conflicting flag {conflict_flag} is enabled",
                                evaluation_time=time.time() - start_time,
                                context=context
                            )
                
                # Evaluate flag
                enabled = flag.is_enabled_for_context(context)
                value = flag.default_value if enabled else default
                reason = f"State: {flag.state.value}, Rules evaluated"
                
                result = FlagEvaluationResult(
                    flag_id=flag_id,
                    enabled=enabled,
                    value=value,
                    reason=reason,
                    evaluation_time=time.time() - start_time,
                    context=context
                )
                
                # Record metrics
                self.evaluation_times.append(result.evaluation_time)
                if len(self.evaluation_times) > 1000:  # Keep only recent times
                    self.evaluation_times = self.evaluation_times[-1000:]
                
                return result
                
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error evaluating flag {flag_id}: {e}")
            
            return FlagEvaluationResult(
                flag_id=flag_id,
                enabled=bool(default),
                value=default,
                reason=f"Evaluation error: {str(e)}",
                evaluation_time=time.time() - start_time,
                context=context
            )
    
    def update_flag_state(self, flag_id: str, new_state: FlagState) -> bool:
        """Update feature flag state"""
        try:
            with self._lock:
                if flag_id not in self.flags:
                    return False
                
                flag = self.flags[flag_id]
                old_state = flag.state
                flag.state = new_state
                flag.updated_at = datetime.now()
                
                self.logger.info(f"Updated flag {flag_id} state: {old_state.value} -> {new_state.value}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update flag {flag_id} state: {e}")
            return False
    
    def update_rollout_percentage(self, flag_id: str, percentage: float) -> bool:
        """Update percentage rollout for a flag"""
        try:
            with self._lock:
                if flag_id not in self.flags:
                    return False
                
                flag = self.flags[flag_id]
                
                # Find or create percentage rollout rule
                percentage_rule = None
                for rule in flag.rollout_rules:
                    if rule.rule_type == RolloutType.PERCENTAGE:
                        percentage_rule = rule
                        break
                
                if not percentage_rule:
                    percentage_rule = RolloutRule(
                        rule_id=f"{flag_id}_percentage",
                        rule_type=RolloutType.PERCENTAGE
                    )
                    flag.rollout_rules.append(percentage_rule)
                
                percentage_rule.percentage = percentage
                flag.updated_at = datetime.now()
                
                self.logger.info(f"Updated flag {flag_id} rollout percentage to {percentage}%")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update flag {flag_id} percentage: {e}")
            return False
    
    def emergency_disable_all(self, except_flags: Optional[List[str]] = None) -> int:
        """Emergency disable all feature flags except specified ones"""
        except_flags = except_flags or []
        disabled_count = 0
        
        try:
            with self._lock:
                for flag_id, flag in self.flags.items():
                    if flag_id not in except_flags and flag.state != FlagState.KILL_SWITCH:
                        flag.state = FlagState.KILL_SWITCH
                        flag.updated_at = datetime.now()
                        disabled_count += 1
                
                self.logger.critical(f"Emergency disabled {disabled_count} feature flags")
                return disabled_count
                
        except Exception as e:
            self.logger.error(f"Failed to emergency disable flags: {e}")
            return 0
    
    def get_flag_metrics(self) -> Dict[str, Any]:
        """Get feature flag metrics and statistics"""
        with self._lock:
            avg_eval_time = (
                sum(self.evaluation_times) / len(self.evaluation_times)
                if self.evaluation_times else 0
            )
            
            flag_states = {}
            for state in FlagState:
                flag_states[state.value] = sum(
                    1 for flag in self.flags.values() if flag.state == state
                )
            
            return {
                "total_flags": len(self.flags),
                "total_evaluations": self.evaluation_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.evaluation_count, 1),
                "average_evaluation_time_ms": avg_eval_time * 1000,
                "flag_states": flag_states,
                "recent_evaluation_times": self.evaluation_times[-10:] if self.evaluation_times else []
            }
    
    def load_configuration(self, config_path: Path) -> bool:
        """Load feature flag configuration from file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            flags_data = config.get('feature_flags', [])
            loaded_count = 0
            
            with self._lock:
                for flag_data in flags_data:
                    try:
                        # Parse rollout rules
                        rollout_rules = []
                        for rule_data in flag_data.get('rollout_rules', []):
                            rule = RolloutRule(
                                rule_id=rule_data['rule_id'],
                                rule_type=RolloutType(rule_data['rule_type']),
                                percentage=rule_data.get('percentage', 0.0),
                                user_whitelist=set(rule_data.get('user_whitelist', [])),
                                environments=set(rule_data.get('environments', [])),
                                conditions=rule_data.get('conditions', {})
                            )
                            
                            # Parse time-based rules
                            if 'start_time' in rule_data:
                                rule.start_time = datetime.fromisoformat(rule_data['start_time'])
                            if 'end_time' in rule_data:
                                rule.end_time = datetime.fromisoformat(rule_data['end_time'])
                            
                            rollout_rules.append(rule)
                        
                        # Create flag
                        flag = FeatureFlagDefinition(
                            flag_id=flag_data['flag_id'],
                            name=flag_data['name'],
                            description=flag_data['description'],
                            state=FlagState(flag_data.get('state', 'disabled')),
                            rollout_rules=rollout_rules,
                            category=flag_data.get('category', ''),
                            tags=set(flag_data.get('tags', [])),
                            owner=flag_data.get('owner', ''),
                            default_enabled=flag_data.get('default_enabled', False),
                            default_value=flag_data.get('default_value'),
                            prerequisite_flags=flag_data.get('prerequisite_flags', []),
                            conflicting_flags=flag_data.get('conflicting_flags', [])
                        )
                        
                        # Parse expiration
                        if 'expires_at' in flag_data:
                            flag.expires_at = datetime.fromisoformat(flag_data['expires_at'])
                        
                        self.flags[flag.flag_id] = flag
                        loaded_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Failed to load flag {flag_data.get('flag_id', 'unknown')}: {e}")
            
            self._last_config_mtime = config_path.stat().st_mtime
            self.logger.info(f"Loaded {loaded_count} feature flags from {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            return False
    
    def export_configuration(self, output_path: Path) -> bool:
        """Export current feature flag configuration"""
        try:
            config = {
                "feature_flags": [],
                "exported_at": datetime.now().isoformat(),
                "metrics": self.get_flag_metrics()
            }
            
            with self._lock:
                for flag in self.flags.values():
                    flag_data = {
                        "flag_id": flag.flag_id,
                        "name": flag.name,
                        "description": flag.description,
                        "state": flag.state.value,
                        "category": flag.category,
                        "tags": list(flag.tags),
                        "owner": flag.owner,
                        "default_enabled": flag.default_enabled,
                        "default_value": flag.default_value,
                        "prerequisite_flags": flag.prerequisite_flags,
                        "conflicting_flags": flag.conflicting_flags,
                        "rollout_rules": [
                            {
                                "rule_id": rule.rule_id,
                                "rule_type": rule.rule_type.value,
                                "percentage": rule.percentage,
                                "user_whitelist": list(rule.user_whitelist),
                                "environments": list(rule.environments),
                                "conditions": rule.conditions,
                                "start_time": rule.start_time.isoformat() if rule.start_time else None,
                                "end_time": rule.end_time.isoformat() if rule.end_time else None
                            }
                            for rule in flag.rollout_rules
                        ]
                    }
                    
                    if flag.expires_at:
                        flag_data["expires_at"] = flag.expires_at.isoformat()
                    
                    config["feature_flags"].append(flag_data)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            self.logger.info(f"Exported feature flag configuration to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def start_config_reloader(self) -> None:
        """Start background configuration reloader thread"""
        if not self.config_path or self._reload_thread:
            return
        
        def reload_config():
            while not self._shutdown_event.wait(self._reload_interval):
                try:
                    if self.config_path.exists():
                        current_mtime = self.config_path.stat().st_mtime
                        if current_mtime > self._last_config_mtime:
                            self.logger.info("Configuration file changed, reloading...")
                            self.load_configuration(self.config_path)
                except Exception as e:
                    self.logger.error(f"Error in config reloader: {e}")
        
        self._reload_thread = threading.Thread(target=reload_config, daemon=True)
        self._reload_thread.start()
        self.logger.info("Started configuration reloader thread")
    
    def shutdown(self) -> None:
        """Shutdown feature flag manager"""
        if self._shutdown_event:
            self._shutdown_event.set()
        
        if self._reload_thread:
            self._reload_thread.join(timeout=5)
        
        self.logger.info("Feature flag manager shut down")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.shutdown()


# Decorator for feature flag-controlled functions
def feature_flag(flag_id: str, 
                manager: Optional[FeatureFlagManager] = None,
                context: Optional[FlagEvaluationContext] = None,
                default: bool = False):
    """Decorator to control function execution with feature flags"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal manager
            if manager is None:
                # Use global manager if available
                manager = getattr(wrapper, '_global_manager', None)
            
            if manager and manager.is_enabled(flag_id, context, default):
                return func(*args, **kwargs)
            else:
                # Return None or raise NotImplementedError based on use case
                return None
        
        wrapper._flag_id = flag_id
        wrapper._manager = manager
        return wrapper
    return decorator


@contextmanager
def feature_enabled(manager: FeatureFlagManager,
                   flag_id: str,
                   context: Optional[FlagEvaluationContext] = None):
    """Context manager for feature flag checking"""
    enabled = manager.is_enabled(flag_id, context)
    yield enabled


def main():
    """Main function for feature flag demonstration"""
    try:
        print("Archangel Feature Flag System")
        print("=" * 50)
        
        # Initialize feature flag manager
        flag_manager = FeatureFlagManager()
        
        # Create Archangel feature flags
        flag_manager.create_archangel_feature_flags()
        
        print(f"Created {len(flag_manager.flags)} feature flags")
        
        # Show flag metrics
        metrics = flag_manager.get_flag_metrics()
        print(f"\nFlag Metrics:")
        print(f"  Total flags: {metrics['total_flags']}")
        print(f"  Flag states: {metrics['flag_states']}")
        
        # Test flag evaluations
        contexts = [
            FlagEvaluationContext(user_id="admin", environment="development"),
            FlagEvaluationContext(user_id="user1", environment="testing"),
            FlagEvaluationContext(user_id="user2", environment="production")
        ]
        
        test_flags = ["multi_agent_coordination", "red_team_agents", "basic_memory_system"]
        
        print(f"\nFlag Evaluations:")
        for flag_id in test_flags:
            print(f"\n{flag_id}:")
            for ctx in contexts:
                result = flag_manager.evaluate_flag(flag_id, ctx)
                print(f"  {ctx.environment} ({ctx.user_id}): {'✓' if result.enabled else '✗'} - {result.reason}")
        
        # Test flag updates
        print(f"\nTesting Flag Updates:")
        flag_manager.update_rollout_percentage("basic_memory_system", 75.0)
        flag_manager.update_flag_state("red_team_agents", FlagState.TESTING)
        
        # Re-evaluate after updates
        test_ctx = FlagEvaluationContext(user_id="test_user", environment="testing")
        for flag_id in ["basic_memory_system", "red_team_agents"]:
            result = flag_manager.evaluate_flag(flag_id, test_ctx)
            print(f"  {flag_id}: {'✓' if result.enabled else '✗'} (after update)")
        
        # Demonstrate decorator usage
        @feature_flag("multi_agent_coordination", flag_manager)
        def start_coordination():
            return "Agent coordination started"
        
        @feature_flag("advanced_reasoning", flag_manager)
        def advanced_ai_function():
            return "Advanced AI reasoning active"
        
        print(f"\nDecorator Tests:")
        print(f"  Coordination: {start_coordination()}")
        print(f"  Advanced AI: {advanced_ai_function()}")
        
        # Context manager usage
        with feature_enabled(flag_manager, "red_team_agents", test_ctx) as enabled:
            if enabled:
                print(f"  ✓ Red team agents are enabled - starting red team operations")
            else:
                print(f"  ✗ Red team agents are disabled - skipping red team operations")
        
        # Export configuration
        output_path = Path("project_management") / "feature_flags_config.json"
        flag_manager.export_configuration(output_path)
        print(f"\nConfiguration exported to {output_path}")
        
        # Final metrics
        final_metrics = flag_manager.get_flag_metrics()
        print(f"\nFinal Metrics:")
        print(f"  Total evaluations: {final_metrics['total_evaluations']}")
        print(f"  Average evaluation time: {final_metrics['average_evaluation_time_ms']:.2f}ms")
        print(f"  Error rate: {final_metrics['error_rate']:.2%}")
        
        print("\nFeature flag demonstration complete!")
        
    except Exception as e:
        logger.error(f"Feature flag demo failed: {e}")
        raise


if __name__ == "__main__":
    main()