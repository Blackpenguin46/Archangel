"""
Graceful degradation system for partial system failures.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class DegradationLevel(Enum):
    """Degradation levels from full functionality to minimal operation."""
    FULL = "full"                    # Full functionality
    MINOR_DEGRADATION = "minor"      # Minor features disabled
    MODERATE_DEGRADATION = "moderate"  # Significant features disabled
    SEVERE_DEGRADATION = "severe"    # Only core features available
    MINIMAL = "minimal"              # Bare minimum functionality
    EMERGENCY = "emergency"          # Emergency mode only

class ComponentCriticality(Enum):
    """Criticality levels for system components."""
    ESSENTIAL = "essential"      # System cannot function without this
    CRITICAL = "critical"        # Major functionality impacted
    IMPORTANT = "important"      # Noticeable impact
    OPTIONAL = "optional"        # Nice-to-have features
    NON_ESSENTIAL = "non_essential"  # Can be disabled without impact

@dataclass
class DegradationRule:
    """Rules for degrading functionality based on component failures."""
    rule_id: str
    failed_components: Set[str]
    degradation_level: DegradationLevel
    disabled_features: Set[str] = field(default_factory=set)
    fallback_behavior: Optional[str] = None
    description: str = ""
    priority: int = 0  # Higher priority rules override lower priority

@dataclass
class ComponentHealth:
    """Health status for a system component."""
    component_id: str
    is_healthy: bool
    criticality: ComponentCriticality
    last_health_check: float = field(default_factory=time.time)
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeatureConfig:
    """Configuration for a system feature."""
    feature_id: str
    enabled: bool = True
    required_components: Set[str] = field(default_factory=set)
    fallback_implementation: Optional[Callable] = None
    resource_cost: float = 1.0  # Relative cost of this feature
    user_impact: float = 1.0    # Impact on user experience (0-1)

class GracefulDegradation:
    """
    Graceful degradation system that maintains partial functionality during failures.
    
    Features:
    - Multi-level degradation based on component health
    - Rule-based degradation logic
    - Feature disable/enable management
    - Automatic fallback behavior
    - Recovery detection and re-enablement
    - Impact assessment and reporting
    - Resource optimization during degradation
    """
    
    def __init__(self):
        """Initialize graceful degradation system."""
        self.components: Dict[str, ComponentHealth] = {}
        self.features: Dict[str, FeatureConfig] = {}
        self.degradation_rules: List[DegradationRule] = []
        self.current_level = DegradationLevel.FULL
        self.disabled_features: Set[str] = set()
        
        self._lock = threading.RLock()
        self._degradation_callbacks: List[Callable[[DegradationLevel, Set[str]], None]] = []
        self._recovery_callbacks: List[Callable[[DegradationLevel, Set[str]], None]] = []
        
        # Degradation history
        self.degradation_history: List[Dict[str, Any]] = []
        self.max_history_length = 100
        
        # Initialize default degradation rules
        self._initialize_default_rules()
        
    def register_component(self, component_id: str, criticality: ComponentCriticality,
                          metadata: Optional[Dict[str, Any]] = None):
        """Register a component for health monitoring.
        
        Args:
            component_id: Unique identifier for the component
            criticality: How critical this component is to system operation
            metadata: Optional metadata about the component
        """
        with self._lock:
            self.components[component_id] = ComponentHealth(
                component_id=component_id,
                is_healthy=True,
                criticality=criticality,
                metadata=metadata or {}
            )
            logger.info(f"Registered component {component_id} with criticality {criticality.value}")
            
    def register_feature(self, feature_id: str, required_components: Set[str],
                        fallback_implementation: Optional[Callable] = None,
                        resource_cost: float = 1.0, user_impact: float = 1.0):
        """Register a system feature.
        
        Args:
            feature_id: Unique identifier for the feature
            required_components: Components required for this feature
            fallback_implementation: Optional fallback function
            resource_cost: Relative resource cost (0.0-10.0)
            user_impact: Impact on user experience if disabled (0.0-1.0)
        """
        with self._lock:
            self.features[feature_id] = FeatureConfig(
                feature_id=feature_id,
                required_components=required_components,
                fallback_implementation=fallback_implementation,
                resource_cost=resource_cost,
                user_impact=user_impact
            )
            logger.info(f"Registered feature {feature_id} with {len(required_components)} dependencies")
            
    def add_degradation_rule(self, rule: DegradationRule):
        """Add a degradation rule.
        
        Args:
            rule: DegradationRule to add
        """
        with self._lock:
            self.degradation_rules.append(rule)
            # Sort by priority (highest first)
            self.degradation_rules.sort(key=lambda r: r.priority, reverse=True)
            logger.info(f"Added degradation rule: {rule.rule_id}")
            
    def update_component_health(self, component_id: str, is_healthy: bool,
                               metadata: Optional[Dict[str, Any]] = None):
        """Update the health status of a component.
        
        Args:
            component_id: ID of the component
            is_healthy: Whether the component is healthy
            metadata: Optional metadata about the health status
        """
        with self._lock:
            if component_id not in self.components:
                logger.warning(f"Health update for unregistered component: {component_id}")
                return
                
            component = self.components[component_id]
            old_health = component.is_healthy
            
            component.is_healthy = is_healthy
            component.last_health_check = time.time()
            
            if not is_healthy:
                component.failure_count += 1
            else:
                component.failure_count = 0
                
            if metadata:
                component.metadata.update(metadata)
                
            # Trigger degradation evaluation if health changed
            if old_health != is_healthy:
                logger.info(f"Component {component_id} health changed: {old_health} → {is_healthy}")
                self._evaluate_degradation()
                
    def _evaluate_degradation(self):
        """Evaluate current system state and apply appropriate degradation."""
        with self._lock:
            # Get current unhealthy components
            unhealthy_components = {
                comp_id for comp_id, comp in self.components.items()
                if not comp.is_healthy
            }
            
            # Find applicable degradation rules
            applicable_rules = []
            for rule in self.degradation_rules:
                if rule.failed_components.issubset(unhealthy_components):
                    applicable_rules.append(rule)
                    
            # Apply the highest priority rule
            if applicable_rules:
                active_rule = applicable_rules[0]  # Highest priority
                self._apply_degradation_rule(active_rule, unhealthy_components)
            else:
                # No failures or no applicable rules, check for recovery
                if self.current_level != DegradationLevel.FULL:
                    self._attempt_recovery()
                    
    def _apply_degradation_rule(self, rule: DegradationRule, unhealthy_components: Set[str]):
        """Apply a degradation rule.
        
        Args:
            rule: Degradation rule to apply
            unhealthy_components: Set of currently unhealthy components
        """
        old_level = self.current_level
        old_disabled = self.disabled_features.copy()
        
        # Update degradation level
        self.current_level = rule.degradation_level
        
        # Disable features based on component health and rule
        self._update_feature_availability(unhealthy_components)
        
        # Add rule-specific disabled features
        self.disabled_features.update(rule.disabled_features)
        
        # Record degradation event
        self._record_degradation_event(rule, unhealthy_components)
        
        # Notify callbacks if state changed
        if old_level != self.current_level or old_disabled != self.disabled_features:
            logger.warning(f"System degraded to {self.current_level.value} level")
            logger.info(f"Disabled features: {self.disabled_features}")
            
            for callback in self._degradation_callbacks:
                try:
                    callback(self.current_level, self.disabled_features)
                except Exception as e:
                    logger.error(f"Error in degradation callback: {e}")
                    
    def _update_feature_availability(self, unhealthy_components: Set[str]):
        """Update feature availability based on component health.
        
        Args:
            unhealthy_components: Set of unhealthy components
        """
        newly_disabled = set()
        
        for feature_id, feature in self.features.items():
            # Check if all required components are healthy
            required_healthy = feature.required_components - unhealthy_components
            feature_should_be_enabled = len(required_healthy) == len(feature.required_components)
            
            if not feature_should_be_enabled and feature.enabled:
                # Disable feature
                feature.enabled = False
                self.disabled_features.add(feature_id)
                newly_disabled.add(feature_id)
                logger.warning(f"Disabled feature {feature_id} due to unhealthy components: {feature.required_components & unhealthy_components}")
                
        return newly_disabled
        
    def _attempt_recovery(self):
        """Attempt to recover from degraded state."""
        with self._lock:
            # Check if all components are healthy
            all_healthy = all(comp.is_healthy for comp in self.components.values())
            
            if all_healthy:
                old_level = self.current_level
                old_disabled = self.disabled_features.copy()
                
                # Full recovery
                self.current_level = DegradationLevel.FULL
                self.disabled_features.clear()
                
                # Re-enable all features
                for feature in self.features.values():
                    feature.enabled = True
                    
                logger.info(f"System recovered to {self.current_level.value} level")
                
                # Record recovery event
                self._record_recovery_event()
                
                # Notify recovery callbacks
                for callback in self._recovery_callbacks:
                    try:
                        callback(self.current_level, old_disabled)
                    except Exception as e:
                        logger.error(f"Error in recovery callback: {e}")
                        
    def is_feature_enabled(self, feature_id: str) -> bool:
        """Check if a feature is currently enabled.
        
        Args:
            feature_id: ID of the feature to check
            
        Returns:
            True if feature is enabled
        """
        with self._lock:
            feature = self.features.get(feature_id)
            if not feature:
                return False
            return feature.enabled and feature_id not in self.disabled_features
            
    def get_fallback_implementation(self, feature_id: str) -> Optional[Callable]:
        """Get fallback implementation for a disabled feature.
        
        Args:
            feature_id: ID of the feature
            
        Returns:
            Fallback function if available
        """
        with self._lock:
            feature = self.features.get(feature_id)
            return feature.fallback_implementation if feature else None
            
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status.
        
        Returns:
            Dictionary with degradation status information
        """
        with self._lock:
            unhealthy_components = [
                comp_id for comp_id, comp in self.components.items()
                if not comp.is_healthy
            ]
            
            return {
                "current_level": self.current_level.value,
                "disabled_features": list(self.disabled_features),
                "unhealthy_components": unhealthy_components,
                "total_components": len(self.components),
                "healthy_components": len(self.components) - len(unhealthy_components),
                "total_features": len(self.features),
                "enabled_features": sum(1 for f in self.features.values() if f.enabled),
                "last_updated": time.time()
            }
            
    def get_impact_assessment(self) -> Dict[str, Any]:
        """Get assessment of current degradation impact.
        
        Returns:
            Dictionary with impact assessment
        """
        with self._lock:
            total_user_impact = 0.0
            total_resource_savings = 0.0
            
            for feature_id in self.disabled_features:
                feature = self.features.get(feature_id)
                if feature:
                    total_user_impact += feature.user_impact
                    total_resource_savings += feature.resource_cost
                    
            total_possible_impact = sum(f.user_impact for f in self.features.values())
            total_possible_resources = sum(f.resource_cost for f in self.features.values())
            
            return {
                "user_impact_percentage": (total_user_impact / max(total_possible_impact, 1)) * 100,
                "resource_savings_percentage": (total_resource_savings / max(total_possible_resources, 1)) * 100,
                "disabled_feature_count": len(self.disabled_features),
                "degradation_level": self.current_level.value,
                "estimated_recovery_time": self._estimate_recovery_time()
            }
            
    def _estimate_recovery_time(self) -> float:
        """Estimate time for full system recovery.
        
        Returns:
            Estimated recovery time in seconds
        """
        # Simple heuristic based on component failure counts
        max_failures = max((comp.failure_count for comp in self.components.values() if not comp.is_healthy), default=0)
        
        # Base recovery time increases with failure count
        base_time = 60.0  # 1 minute base
        return base_time * (1 + max_failures * 0.5)
        
    def add_degradation_callback(self, callback: Callable[[DegradationLevel, Set[str]], None]):
        """Add callback for degradation events.
        
        Args:
            callback: Function to call when system degrades
        """
        self._degradation_callbacks.append(callback)
        
    def add_recovery_callback(self, callback: Callable[[DegradationLevel, Set[str]], None]):
        """Add callback for recovery events.
        
        Args:
            callback: Function to call when system recovers
        """
        self._recovery_callbacks.append(callback)
        
    def _record_degradation_event(self, rule: DegradationRule, unhealthy_components: Set[str]):
        """Record a degradation event.
        
        Args:
            rule: Applied degradation rule
            unhealthy_components: Components that were unhealthy
        """
        event = {
            "timestamp": time.time(),
            "event_type": "degradation",
            "rule_id": rule.rule_id,
            "degradation_level": rule.degradation_level.value,
            "unhealthy_components": list(unhealthy_components),
            "disabled_features": list(rule.disabled_features),
            "description": rule.description
        }
        
        self.degradation_history.append(event)
        
        # Trim history if needed
        if len(self.degradation_history) > self.max_history_length:
            self.degradation_history = self.degradation_history[-self.max_history_length:]
            
    def _record_recovery_event(self):
        """Record a recovery event."""
        event = {
            "timestamp": time.time(),
            "event_type": "recovery",
            "degradation_level": self.current_level.value,
            "recovered_features": list(self.features.keys()),
            "description": "Full system recovery"
        }
        
        self.degradation_history.append(event)
        
    def _initialize_default_rules(self):
        """Initialize default degradation rules."""
        # Example degradation rules - would be customized for specific system
        
        # Critical component failure
        critical_rule = DegradationRule(
            rule_id="critical_component_failure",
            failed_components=set(),  # Will be populated dynamically
            degradation_level=DegradationLevel.SEVERE_DEGRADATION,
            disabled_features={"advanced_analytics", "background_processing"},
            description="Critical component failed",
            priority=100
        )
        
        # Multiple component failures
        multiple_failures_rule = DegradationRule(
            rule_id="multiple_component_failures", 
            failed_components=set(),  # Will be populated dynamically
            degradation_level=DegradationLevel.MINIMAL,
            disabled_features={"reporting", "monitoring", "analytics"},
            description="Multiple components failed",
            priority=200
        )
        
        # These would be added when components are registered
        # self.add_degradation_rule(critical_rule)

# Global graceful degradation instance
graceful_degradation = GracefulDegradation()