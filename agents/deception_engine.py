#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Advanced Deception Engine
Adaptive deception strategies that evolve based on attacker behavior patterns
"""

import logging
import json
import asyncio
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
from enum import Enum
from pathlib import Path
import hashlib
import numpy as np
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class DeceptionType(Enum):
    """Types of deception strategies"""
    HONEYPOT = "honeypot"
    FALSE_DATA = "false_data" 
    MISDIRECTION = "misdirection"
    DELAY_TACTICS = "delay_tactics"
    FALSE_VULNERABILITY = "false_vulnerability"
    FAKE_NETWORK = "fake_network"
    BREADCRUMB_TRAIL = "breadcrumb_trail"
    FALSE_INTELLIGENCE = "false_intelligence"

class AttackerProfile(Enum):
    """Attacker behavior profiles for adaptive responses"""
    SCRIPT_KIDDIE = "script_kiddie"
    OPPORTUNISTIC = "opportunistic"
    TARGETED_APT = "targeted_apt"
    INSIDER_THREAT = "insider_threat"
    AUTOMATED_SCAN = "automated_scan"
    RECONNAISSANCE = "reconnaissance"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_EXFILTRATION = "data_exfiltration"

class DeceptionEffectiveness(Enum):
    """Effectiveness levels of deception strategies"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

@dataclass
class AttackerBehaviorPattern:
    """Pattern analysis of attacker behavior"""
    pattern_id: str
    attacker_profile: AttackerProfile
    
    # Behavioral indicators
    attack_vectors: List[str] = field(default_factory=list)
    timing_patterns: Dict[str, Any] = field(default_factory=dict)
    target_preferences: List[str] = field(default_factory=list)
    tool_signatures: List[str] = field(default_factory=list)
    
    # Sophistication indicators
    evasion_techniques: List[str] = field(default_factory=list)
    persistence_methods: List[str] = field(default_factory=list)
    communication_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Learning indicators
    adaptation_rate: float = 0.0
    learning_curve: List[float] = field(default_factory=list)
    mistake_patterns: List[str] = field(default_factory=list)
    
    # Metadata
    first_observed: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.5
    observation_count: int = 0

@dataclass
class DeceptionStrategy:
    """Individual deception strategy definition"""
    strategy_id: str
    name: str
    deception_type: DeceptionType
    description: str
    
    # Targeting
    target_profiles: List[AttackerProfile] = field(default_factory=list)
    attack_phase_targets: List[str] = field(default_factory=list)  # recon, exploit, persist, etc.
    
    # Implementation details
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    deployment_time: timedelta = field(default=timedelta(minutes=5))
    duration: timedelta = field(default=timedelta(hours=24))
    
    # Effectiveness tracking
    success_rate: float = 0.0
    detection_delay_avg: float = 0.0  # seconds
    engagement_duration_avg: float = 0.0  # seconds
    
    # Adaptation parameters
    adaptation_triggers: List[str] = field(default_factory=list)
    mutation_parameters: Dict[str, Any] = field(default_factory=dict)
    learning_rate: float = 0.1
    
    # Status
    is_active: bool = False
    deployment_count: int = 0
    last_effectiveness_score: DeceptionEffectiveness = DeceptionEffectiveness.MEDIUM
    
    # Risk management
    risk_level: float = 0.3  # 0.0 = no risk, 1.0 = high risk
    rollback_conditions: List[str] = field(default_factory=list)
    
    def calculate_effectiveness(self, outcomes: List[Dict[str, Any]]) -> DeceptionEffectiveness:
        """Calculate current effectiveness based on recent outcomes"""
        if not outcomes:
            return DeceptionEffectiveness.MEDIUM
        
        # Weighted scoring based on multiple factors
        engagement_score = sum(1 for o in outcomes if o.get('engaged', False)) / len(outcomes)
        delay_score = np.mean([min(o.get('detection_delay', 0), 300) / 300 for o in outcomes])
        success_score = sum(1 for o in outcomes if o.get('successful', False)) / len(outcomes)
        
        composite_score = (engagement_score * 0.4 + delay_score * 0.3 + success_score * 0.3)
        
        if composite_score >= 0.8:
            return DeceptionEffectiveness.VERY_HIGH
        elif composite_score >= 0.6:
            return DeceptionEffectiveness.HIGH
        elif composite_score >= 0.4:
            return DeceptionEffectiveness.MEDIUM
        elif composite_score >= 0.2:
            return DeceptionEffectiveness.LOW
        else:
            return DeceptionEffectiveness.VERY_LOW

@dataclass
class DeceptionDeployment:
    """Active deception deployment instance"""
    deployment_id: str
    strategy_id: str
    
    # Deployment details
    deployed_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    target_environment: str = ""
    
    # Targeting
    target_attacker_patterns: List[str] = field(default_factory=list)
    bait_configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Interaction tracking
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    total_engagements: int = 0
    unique_attackers: Set[str] = field(default_factory=set)
    
    # Real-time adaptation
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    is_active: bool = True
    effectiveness_trend: List[float] = field(default_factory=list)

class AdaptiveDeceptionEngine:
    """
    Advanced deception engine with adaptive strategies based on attacker behavior.
    
    Features:
    - Real-time attacker behavior analysis
    - Dynamic strategy selection and adaptation  
    - Multi-layered deception coordination
    - Effectiveness optimization through machine learning
    - Counter-intelligence operation support
    - Automated honeypot orchestration
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.strategies: Dict[str, DeceptionStrategy] = {}
        self.attacker_patterns: Dict[str, AttackerBehaviorPattern] = {}
        self.active_deployments: Dict[str, DeceptionDeployment] = {}
        
        # Adaptation engine
        self.behavior_analyzer = AttackerBehaviorAnalyzer()
        self.strategy_optimizer = StrategyOptimizer()
        self.deployment_coordinator = DeploymentCoordinator()
        
        # Learning system
        self.effectiveness_history: deque = deque(maxlen=10000)
        self.adaptation_rules: List[Dict[str, Any]] = []
        self.learning_rate = 0.1
        
        # Monitoring and metrics
        self.metrics: Dict[str, Any] = defaultdict(float)
        self.interaction_log: deque = deque(maxlen=50000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        if config_path and config_path.exists():
            self.load_configuration(config_path)
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default deception strategies"""
        
        default_strategies = [
            # Honeypot strategies
            DeceptionStrategy(
                strategy_id="adaptive_honeypot",
                name="Adaptive Honeypot",
                deception_type=DeceptionType.HONEYPOT,
                description="Dynamic honeypot that adapts configuration based on attacker behavior",
                target_profiles=[AttackerProfile.RECONNAISSANCE, AttackerProfile.AUTOMATED_SCAN],
                attack_phase_targets=["reconnaissance", "initial_access"],
                resource_requirements={"cpu": 0.5, "memory": "256MB", "storage": "1GB"},
                deployment_time=timedelta(minutes=2),
                duration=timedelta(hours=48),
                adaptation_triggers=["interaction_pattern_change", "evasion_detected"],
                mutation_parameters={
                    "service_ports": [22, 23, 80, 443, 3389, 5900],
                    "response_delays": [0.1, 0.5, 1.0, 2.0],
                    "vulnerability_signatures": ["ssh_weak_auth", "web_sqli", "rdp_brute"]
                },
                rollback_conditions=["high_resource_usage", "legitimate_user_interaction"]
            ),
            
            # False data strategies  
            DeceptionStrategy(
                strategy_id="false_intelligence_feed",
                name="False Intelligence Feed",
                deception_type=DeceptionType.FALSE_INTELLIGENCE,
                description="Plant false but believable intelligence to mislead attackers",
                target_profiles=[AttackerProfile.TARGETED_APT, AttackerProfile.INSIDER_THREAT],
                attack_phase_targets=["reconnaissance", "persistence"],
                resource_requirements={"cpu": 0.1, "memory": "64MB", "network": "low"},
                deployment_time=timedelta(seconds=30),
                duration=timedelta(days=7),
                adaptation_triggers=["attacker_sophistication_increase", "intelligence_consumed"],
                mutation_parameters={
                    "intelligence_types": ["network_topology", "user_accounts", "system_configs"],
                    "credibility_levels": [0.7, 0.8, 0.9],
                    "update_frequencies": [3600, 7200, 14400]  # seconds
                }
            ),
            
            # Misdirection tactics
            DeceptionStrategy(
                strategy_id="breadcrumb_misdirection",
                name="Breadcrumb Misdirection",
                deception_type=DeceptionType.BREADCRUMB_TRAIL,
                description="Create false trails leading attackers away from real assets",
                target_profiles=[AttackerProfile.LATERAL_MOVEMENT, AttackerProfile.DATA_EXFILTRATION],
                attack_phase_targets=["lateral_movement", "collection"],
                resource_requirements={"cpu": 0.2, "memory": "128MB"},
                deployment_time=timedelta(minutes=1),
                duration=timedelta(hours=12),
                adaptation_triggers=["trail_following_detected", "attacker_backtrack"],
                mutation_parameters={
                    "trail_complexity": [2, 3, 5, 8],  # number of hops
                    "false_target_types": ["database", "file_share", "backup_system"],
                    "credibility_artifacts": ["logs", "configs", "credentials"]
                }
            ),
            
            # Delay tactics
            DeceptionStrategy(
                strategy_id="adaptive_delay_trap",
                name="Adaptive Delay Trap", 
                deception_type=DeceptionType.DELAY_TACTICS,
                description="Dynamic delays and obstacles to slow down attackers",
                target_profiles=[AttackerProfile.AUTOMATED_SCAN, AttackerProfile.OPPORTUNISTIC],
                attack_phase_targets=["reconnaissance", "initial_access", "execution"],
                resource_requirements={"cpu": 0.1, "memory": "32MB"},
                deployment_time=timedelta(seconds=10),
                duration=timedelta(hours=6),
                adaptation_triggers=["scan_rate_change", "tool_change_detected"],
                mutation_parameters={
                    "delay_patterns": ["linear", "exponential", "random", "adaptive"],
                    "max_delays": [5, 10, 30, 60],  # seconds
                    "complexity_levels": [1, 2, 3]
                }
            ),
            
            # False vulnerability exposure
            DeceptionStrategy(
                strategy_id="vulnerability_mirage",
                name="Vulnerability Mirage",
                deception_type=DeceptionType.FALSE_VULNERABILITY,
                description="Present false vulnerabilities to attract and analyze attackers",
                target_profiles=[AttackerProfile.OPPORTUNISTIC, AttackerProfile.SCRIPT_KIDDIE],
                attack_phase_targets=["reconnaissance", "weaponization", "exploitation"],
                resource_requirements={"cpu": 0.3, "memory": "128MB"},
                deployment_time=timedelta(minutes=5),
                duration=timedelta(days=1),
                adaptation_triggers=["exploit_attempt", "vulnerability_scan"],
                mutation_parameters={
                    "vulnerability_types": ["rce", "sqli", "xss", "lfi", "auth_bypass"],
                    "severity_levels": ["critical", "high", "medium"],
                    "patch_dates": ["recent", "old", "very_old"]
                }
            )
        ]
        
        for strategy in default_strategies:
            self.strategies[strategy.strategy_id] = strategy
        
        self.logger.info(f"Initialized {len(default_strategies)} default deception strategies")
    
    async def analyze_attacker_behavior(self, 
                                      interaction_data: Dict[str, Any]) -> AttackerBehaviorPattern:
        """Analyze attacker behavior from interaction data"""
        
        attacker_id = interaction_data.get('attacker_id', 'unknown')
        
        with self._lock:
            if attacker_id not in self.attacker_patterns:
                # Create new pattern
                pattern = AttackerBehaviorPattern(
                    pattern_id=attacker_id,
                    attacker_profile=AttackerProfile.RECONNAISSANCE  # Default
                )
                self.attacker_patterns[attacker_id] = pattern
            else:
                pattern = self.attacker_patterns[attacker_id]
        
        # Update pattern based on new interaction data
        await self.behavior_analyzer.update_pattern(pattern, interaction_data)
        
        # Classify attacker profile based on updated pattern
        new_profile = await self.behavior_analyzer.classify_attacker_profile(pattern)
        if new_profile != pattern.attacker_profile:
            self.logger.info(f"Attacker {attacker_id} profile updated: "
                           f"{pattern.attacker_profile.value} -> {new_profile.value}")
            pattern.attacker_profile = new_profile
        
        pattern.last_updated = datetime.now()
        pattern.observation_count += 1
        
        return pattern
    
    async def select_adaptive_strategy(self, 
                                     attacker_pattern: AttackerBehaviorPattern,
                                     context: Dict[str, Any]) -> Optional[DeceptionStrategy]:
        """Select most appropriate deception strategy for given attacker pattern"""
        
        # Get strategies targeting this attacker profile
        candidate_strategies = [
            s for s in self.strategies.values()
            if attacker_pattern.attacker_profile in s.target_profiles
        ]
        
        if not candidate_strategies:
            # Fallback to general strategies
            candidate_strategies = list(self.strategies.values())
        
        # Score strategies based on multiple factors
        strategy_scores = []
        
        for strategy in candidate_strategies:
            score = await self._calculate_strategy_score(strategy, attacker_pattern, context)
            strategy_scores.append((strategy, score))
        
        # Sort by score descending
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        if strategy_scores:
            selected_strategy = strategy_scores[0][0]
            self.logger.info(f"Selected strategy '{selected_strategy.name}' for attacker "
                           f"{attacker_pattern.pattern_id} (score: {strategy_scores[0][1]:.3f})")
            return selected_strategy
        
        return None
    
    async def _calculate_strategy_score(self, 
                                      strategy: DeceptionStrategy,
                                      attacker_pattern: AttackerBehaviorPattern,
                                      context: Dict[str, Any]) -> float:
        """Calculate effectiveness score for strategy against attacker pattern"""
        
        score = 0.0
        
        # Profile matching score (0.0 - 0.3)
        if attacker_pattern.attacker_profile in strategy.target_profiles:
            score += 0.3
        else:
            score += 0.1  # Partial credit for general applicability
        
        # Historical effectiveness score (0.0 - 0.3)
        effectiveness_value = strategy.last_effectiveness_score.value / 5.0  # Normalize to 0-1
        score += effectiveness_value * 0.3
        
        # Resource availability score (0.0 - 0.2)
        resource_score = self._calculate_resource_availability_score(strategy, context)
        score += resource_score * 0.2
        
        # Novelty/surprise score (0.0 - 0.2) - prefer strategies not recently used
        novelty_score = self._calculate_novelty_score(strategy, attacker_pattern)
        score += novelty_score * 0.2
        
        return score
    
    def _calculate_resource_availability_score(self, 
                                             strategy: DeceptionStrategy, 
                                             context: Dict[str, Any]) -> float:
        """Calculate score based on resource availability"""
        # Simplified resource check - in production would integrate with infrastructure
        available_resources = context.get('available_resources', {})
        required_resources = strategy.resource_requirements
        
        if not required_resources:
            return 1.0
        
        # Check each resource requirement
        resource_scores = []
        for resource, requirement in required_resources.items():
            available = available_resources.get(resource, 1.0)  # Default to available
            
            if isinstance(requirement, (int, float)):
                if available >= requirement:
                    resource_scores.append(1.0)
                else:
                    resource_scores.append(available / requirement)
            else:
                resource_scores.append(0.8)  # Default partial score for string requirements
        
        return np.mean(resource_scores) if resource_scores else 0.8
    
    def _calculate_novelty_score(self, 
                                strategy: DeceptionStrategy,
                                attacker_pattern: AttackerBehaviorPattern) -> float:
        """Calculate novelty score - higher for strategies not recently used against this attacker"""
        
        # Check recent deployments against this attacker
        recent_deployments = [
            d for d in self.active_deployments.values()
            if (d.strategy_id == strategy.strategy_id and 
                attacker_pattern.pattern_id in d.target_attacker_patterns and
                d.deployed_at > datetime.now() - timedelta(hours=24))
        ]
        
        if not recent_deployments:
            return 1.0
        elif len(recent_deployments) == 1:
            return 0.5
        else:
            return 0.1  # Very low novelty if used multiple times recently
    
    async def deploy_strategy(self, 
                            strategy: DeceptionStrategy,
                            target_patterns: List[AttackerBehaviorPattern],
                            environment: str = "default") -> DeceptionDeployment:
        """Deploy deception strategy against specific attacker patterns"""
        
        deployment_id = self._generate_deployment_id(strategy, target_patterns)
        
        deployment = DeceptionDeployment(
            deployment_id=deployment_id,
            strategy_id=strategy.strategy_id,
            expires_at=datetime.now() + strategy.duration,
            target_environment=environment,
            target_attacker_patterns=[p.pattern_id for p in target_patterns],
            current_parameters=strategy.mutation_parameters.copy()
        )
        
        # Configure deployment based on target attacker patterns
        bait_config = await self._generate_bait_configuration(strategy, target_patterns)
        deployment.bait_configuration = bait_config
        
        # Deploy through coordinator
        success = await self.deployment_coordinator.deploy(deployment, strategy)
        
        if success:
            with self._lock:
                self.active_deployments[deployment_id] = deployment
                strategy.is_active = True
                strategy.deployment_count += 1
            
            self.logger.info(f"Successfully deployed strategy '{strategy.name}' "
                           f"as deployment {deployment_id}")
            return deployment
        else:
            raise RuntimeError(f"Failed to deploy strategy '{strategy.name}'")
    
    def _generate_deployment_id(self, 
                               strategy: DeceptionStrategy,
                               target_patterns: List[AttackerBehaviorPattern]) -> str:
        """Generate unique deployment ID"""
        pattern_ids = [p.pattern_id for p in target_patterns]
        data = f"{strategy.strategy_id}_{'-'.join(pattern_ids)}_{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]
    
    async def _generate_bait_configuration(self, 
                                         strategy: DeceptionStrategy,
                                         target_patterns: List[AttackerBehaviorPattern]) -> Dict[str, Any]:
        """Generate bait configuration optimized for target attacker patterns"""
        
        base_config = strategy.mutation_parameters.copy()
        
        # Adapt configuration based on attacker patterns
        for pattern in target_patterns:
            if pattern.attacker_profile == AttackerProfile.SCRIPT_KIDDIE:
                # Simple, obvious vulnerabilities work best
                base_config.update({
                    "vulnerability_complexity": "low",
                    "authentication_strength": "weak",
                    "response_time": "fast"
                })
            
            elif pattern.attacker_profile == AttackerProfile.TARGETED_APT:
                # Sophisticated, believable deception required
                base_config.update({
                    "vulnerability_complexity": "medium",
                    "authentication_strength": "moderate", 
                    "response_time": "realistic",
                    "legitimacy_artifacts": True
                })
            
            elif pattern.attacker_profile == AttackerProfile.AUTOMATED_SCAN:
                # Focus on signature evasion and detection
                base_config.update({
                    "signature_evasion": True,
                    "scan_detection": True,
                    "response_randomization": True
                })
        
        return base_config
    
    async def adapt_strategy(self, 
                           deployment: DeceptionDeployment,
                           interaction_data: Dict[str, Any]) -> bool:
        """Adapt deployed strategy based on attacker interaction"""
        
        strategy = self.strategies.get(deployment.strategy_id)
        if not strategy:
            return False
        
        # Record interaction
        deployment.interactions.append({
            **interaction_data,
            'timestamp': datetime.now(),
            'deployment_id': deployment.deployment_id
        })
        deployment.total_engagements += 1
        
        # Check if adaptation is needed
        adaptation_needed = await self._check_adaptation_triggers(
            strategy, deployment, interaction_data
        )
        
        if adaptation_needed:
            # Generate adapted parameters
            new_parameters = await self._generate_adapted_parameters(
                strategy, deployment, interaction_data
            )
            
            # Apply adaptation
            success = await self.deployment_coordinator.update_deployment(
                deployment, new_parameters
            )
            
            if success:
                deployment.adaptation_history.append({
                    'timestamp': datetime.now(),
                    'trigger': interaction_data.get('adaptation_trigger', 'interaction'),
                    'old_parameters': deployment.current_parameters.copy(),
                    'new_parameters': new_parameters.copy()
                })
                deployment.current_parameters = new_parameters
                
                self.logger.info(f"Adapted deployment {deployment.deployment_id} "
                               f"based on {interaction_data.get('adaptation_trigger', 'interaction')}")
                return True
        
        return False
    
    async def _check_adaptation_triggers(self, 
                                       strategy: DeceptionStrategy,
                                       deployment: DeceptionDeployment,
                                       interaction_data: Dict[str, Any]) -> bool:
        """Check if strategy adaptation should be triggered"""
        
        for trigger in strategy.adaptation_triggers:
            if trigger == "interaction_pattern_change":
                # Check if interaction pattern has changed significantly
                if self._detect_interaction_pattern_change(deployment):
                    return True
            
            elif trigger == "evasion_detected":
                if interaction_data.get('evasion_detected', False):
                    return True
            
            elif trigger == "attacker_sophistication_increase":
                if interaction_data.get('sophistication_increase', False):
                    return True
            
            elif trigger == "effectiveness_decline":
                if self._detect_effectiveness_decline(deployment):
                    return True
        
        return False
    
    def _detect_interaction_pattern_change(self, deployment: DeceptionDeployment) -> bool:
        """Detect significant change in interaction patterns"""
        if len(deployment.interactions) < 10:
            return False
        
        # Simple pattern change detection - in production would use more sophisticated analysis
        recent_interactions = deployment.interactions[-5:]
        older_interactions = deployment.interactions[-10:-5]
        
        recent_types = [i.get('interaction_type', 'unknown') for i in recent_interactions]
        older_types = [i.get('interaction_type', 'unknown') for i in older_interactions]
        
        # Check if interaction types have changed significantly
        recent_set = set(recent_types)
        older_set = set(older_types)
        
        overlap = len(recent_set.intersection(older_set))
        total_unique = len(recent_set.union(older_set))
        
        similarity = overlap / total_unique if total_unique > 0 else 1.0
        
        return similarity < 0.5  # Significant change if less than 50% similarity
    
    def _detect_effectiveness_decline(self, deployment: DeceptionDeployment) -> bool:
        """Detect decline in deployment effectiveness"""
        if len(deployment.effectiveness_trend) < 5:
            return False
        
        # Check if effectiveness is declining over recent measurements
        recent_trend = deployment.effectiveness_trend[-5:]
        return all(recent_trend[i] >= recent_trend[i+1] for i in range(len(recent_trend)-1))
    
    async def _generate_adapted_parameters(self, 
                                         strategy: DeceptionStrategy,
                                         deployment: DeceptionDeployment,
                                         interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new parameters for strategy adaptation"""
        
        current_params = deployment.current_parameters.copy()
        
        # Adapt based on interaction data
        if interaction_data.get('evasion_detected'):
            # Increase complexity and change signatures
            if 'vulnerability_signatures' in current_params:
                # Rotate to different vulnerability signatures
                available_sigs = strategy.mutation_parameters.get('vulnerability_signatures', [])
                current_sigs = current_params.get('vulnerability_signatures', [])
                new_sigs = [sig for sig in available_sigs if sig not in current_sigs]
                if new_sigs:
                    current_params['vulnerability_signatures'] = random.sample(
                        new_sigs, min(len(new_sigs), 2)
                    )
        
        if interaction_data.get('sophistication_increase'):
            # Increase realism and complexity
            if 'credibility_levels' in strategy.mutation_parameters:
                max_credibility = max(strategy.mutation_parameters['credibility_levels'])
                current_params['credibility_level'] = max_credibility
            
            if 'complexity_levels' in strategy.mutation_parameters:
                max_complexity = max(strategy.mutation_parameters['complexity_levels'])
                current_params['complexity_level'] = max_complexity
        
        # Random mutation with learning rate
        for param_name, param_values in strategy.mutation_parameters.items():
            if param_name not in current_params:
                continue
            
            if random.random() < strategy.learning_rate:
                if isinstance(param_values, list):
                    current_params[param_name] = random.choice(param_values)
                elif isinstance(param_values, dict):
                    current_params[param_name] = random.choice(list(param_values.keys()))
        
        return current_params
    
    async def measure_effectiveness(self, deployment_id: str) -> DeceptionEffectiveness:
        """Measure current effectiveness of a deployment"""
        
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.active_deployments[deployment_id]
        
        # Collect recent outcomes for effectiveness calculation
        recent_interactions = deployment.interactions[-20:]  # Last 20 interactions
        
        outcomes = []
        for interaction in recent_interactions:
            outcomes.append({
                'engaged': interaction.get('attacker_engaged', False),
                'detection_delay': interaction.get('detection_delay_seconds', 0),
                'successful': interaction.get('deception_successful', False)
            })
        
        strategy = self.strategies[deployment.strategy_id]
        effectiveness = strategy.calculate_effectiveness(outcomes)
        
        # Update deployment effectiveness trend
        deployment.effectiveness_trend.append(effectiveness.value)
        if len(deployment.effectiveness_trend) > 50:
            deployment.effectiveness_trend = deployment.effectiveness_trend[-50:]
        
        # Update strategy effectiveness
        strategy.last_effectiveness_score = effectiveness
        
        return effectiveness
    
    async def get_deception_metrics(self) -> Dict[str, Any]:
        """Get comprehensive deception system metrics"""
        
        with self._lock:
            total_deployments = len(self.active_deployments)
            total_strategies = len(self.strategies)
            total_attacker_patterns = len(self.attacker_patterns)
        
        # Calculate effectiveness statistics
        effectiveness_scores = []
        for deployment in self.active_deployments.values():
            if deployment.effectiveness_trend:
                effectiveness_scores.append(deployment.effectiveness_trend[-1])
        
        avg_effectiveness = np.mean(effectiveness_scores) if effectiveness_scores else 0
        
        # Calculate engagement statistics
        total_engagements = sum(d.total_engagements for d in self.active_deployments.values())
        unique_attackers = set()
        for d in self.active_deployments.values():
            unique_attackers.update(d.unique_attackers)
        
        return {
            "system_overview": {
                "total_strategies": total_strategies,
                "active_deployments": total_deployments,
                "tracked_attackers": total_attacker_patterns,
                "total_engagements": total_engagements,
                "unique_attackers": len(unique_attackers)
            },
            "effectiveness": {
                "average_effectiveness": avg_effectiveness,
                "effectiveness_distribution": {
                    level.name: effectiveness_scores.count(level.value) 
                    for level in DeceptionEffectiveness
                } if effectiveness_scores else {},
                "adaptation_rate": sum(
                    len(d.adaptation_history) for d in self.active_deployments.values()
                ) / max(total_deployments, 1)
            },
            "strategy_performance": {
                strategy_id: {
                    "deployment_count": strategy.deployment_count,
                    "success_rate": strategy.success_rate,
                    "last_effectiveness": strategy.last_effectiveness_score.name
                }
                for strategy_id, strategy in self.strategies.items()
            },
            "attacker_analysis": {
                profile.name: sum(
                    1 for pattern in self.attacker_patterns.values()
                    if pattern.attacker_profile == profile
                )
                for profile in AttackerProfile
            }
        }
    
    async def export_deception_intelligence(self, output_path: Path) -> None:
        """Export deception intelligence and attacker patterns for analysis"""
        
        intelligence_data = {
            "export_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_strategies": len(self.strategies),
                "active_deployments": len(self.active_deployments),
                "tracked_attackers": len(self.attacker_patterns)
            },
            "attacker_patterns": {
                pattern_id: asdict(pattern) 
                for pattern_id, pattern in self.attacker_patterns.items()
            },
            "deployment_intelligence": {
                deployment_id: {
                    "strategy_id": deployment.strategy_id,
                    "deployment_duration": (datetime.now() - deployment.deployed_at).total_seconds(),
                    "total_engagements": deployment.total_engagements,
                    "adaptation_count": len(deployment.adaptation_history),
                    "effectiveness_trend": deployment.effectiveness_trend,
                    "interaction_summary": {
                        "total_interactions": len(deployment.interactions),
                        "interaction_types": list(set(
                            i.get('interaction_type', 'unknown') 
                            for i in deployment.interactions
                        )),
                        "attacker_tools_observed": list(set(
                            i.get('tool_signature', 'unknown') 
                            for i in deployment.interactions if 'tool_signature' in i
                        ))
                    }
                }
                for deployment_id, deployment in self.active_deployments.items()
            },
            "strategy_effectiveness": {
                strategy_id: {
                    "effectiveness_score": strategy.last_effectiveness_score.name,
                    "deployment_count": strategy.deployment_count,
                    "success_rate": strategy.success_rate,
                    "target_profiles": [p.name for p in strategy.target_profiles]
                }
                for strategy_id, strategy in self.strategies.items()
            },
            "metrics": await self.get_deception_metrics()
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(intelligence_data, f, indent=2, default=str)
        
        self.logger.info(f"Deception intelligence exported to {output_path}")
    
    async def cleanup_expired_deployments(self) -> int:
        """Clean up expired deployments"""
        
        expired_deployments = []
        current_time = datetime.now()
        
        with self._lock:
            for deployment_id, deployment in self.active_deployments.items():
                if deployment.expires_at and current_time > deployment.expires_at:
                    expired_deployments.append(deployment_id)
        
        # Clean up expired deployments
        for deployment_id in expired_deployments:
            deployment = self.active_deployments[deployment_id]
            
            # Cleanup through coordinator
            await self.deployment_coordinator.cleanup(deployment)
            
            # Remove from active deployments
            del self.active_deployments[deployment_id]
            
            # Update strategy status
            strategy = self.strategies.get(deployment.strategy_id)
            if strategy:
                # Check if any other deployments are using this strategy
                other_deployments = [
                    d for d in self.active_deployments.values()
                    if d.strategy_id == deployment.strategy_id
                ]
                if not other_deployments:
                    strategy.is_active = False
        
        if expired_deployments:
            self.logger.info(f"Cleaned up {len(expired_deployments)} expired deployments")
        
        return len(expired_deployments)
    
    async def shutdown(self) -> None:
        """Shutdown deception engine and clean up resources"""
        
        self.logger.info("Shutting down deception engine...")
        
        # Clean up all active deployments
        deployment_ids = list(self.active_deployments.keys())
        for deployment_id in deployment_ids:
            deployment = self.active_deployments[deployment_id]
            await self.deployment_coordinator.cleanup(deployment)
            del self.active_deployments[deployment_id]
        
        # Mark all strategies as inactive
        for strategy in self.strategies.values():
            strategy.is_active = False
        
        self.logger.info("Deception engine shut down complete")


class AttackerBehaviorAnalyzer:
    """Analyzes and classifies attacker behavior patterns"""
    
    async def update_pattern(self, 
                           pattern: AttackerBehaviorPattern,
                           interaction_data: Dict[str, Any]) -> None:
        """Update attacker pattern with new interaction data"""
        
        # Update attack vectors
        if 'attack_vector' in interaction_data:
            vector = interaction_data['attack_vector']
            if vector not in pattern.attack_vectors:
                pattern.attack_vectors.append(vector)
        
        # Update timing patterns  
        timestamp = interaction_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        if 'timing_patterns' not in pattern.timing_patterns:
            pattern.timing_patterns = {'intervals': [], 'activity_hours': []}
        
        pattern.timing_patterns['activity_hours'].append(timestamp.hour)
        
        # Update tool signatures
        if 'tool_signature' in interaction_data:
            tool = interaction_data['tool_signature']
            if tool not in pattern.tool_signatures:
                pattern.tool_signatures.append(tool)
        
        # Update evasion techniques
        if 'evasion_technique' in interaction_data:
            technique = interaction_data['evasion_technique']
            if technique not in pattern.evasion_techniques:
                pattern.evasion_techniques.append(technique)
    
    async def classify_attacker_profile(self, 
                                      pattern: AttackerBehaviorPattern) -> AttackerProfile:
        """Classify attacker profile based on behavior pattern"""
        
        # Simple rule-based classification - in production would use ML
        
        # Check for automated scanning indicators
        if ('nmap' in pattern.tool_signatures or 
            'masscan' in pattern.tool_signatures or
            len(pattern.timing_patterns.get('intervals', [])) > 0):
            return AttackerProfile.AUTOMATED_SCAN
        
        # Check for script kiddie indicators
        if ('sqlmap' in pattern.tool_signatures or
            'metasploit' in pattern.tool_signatures and
            len(pattern.evasion_techniques) == 0):
            return AttackerProfile.SCRIPT_KIDDIE
        
        # Check for APT indicators
        if (len(pattern.evasion_techniques) >= 3 or
            'custom_tool' in pattern.tool_signatures or
            pattern.adaptation_rate > 0.5):
            return AttackerProfile.TARGETED_APT
        
        # Check for insider threat indicators
        if ('internal_tool' in pattern.tool_signatures or
            any('business_hours' in str(h) for h in pattern.timing_patterns.get('activity_hours', []))):
            return AttackerProfile.INSIDER_THREAT
        
        # Default classifications based on attack vectors
        if any(vector in pattern.attack_vectors for vector in ['lateral_movement', 'privilege_escalation']):
            return AttackerProfile.LATERAL_MOVEMENT
        
        if any(vector in pattern.attack_vectors for vector in ['data_collection', 'exfiltration']):
            return AttackerProfile.DATA_EXFILTRATION
        
        # Default to opportunistic
        return AttackerProfile.OPPORTUNISTIC


class StrategyOptimizer:
    """Optimizes deception strategies based on effectiveness data"""
    
    def __init__(self):
        self.optimization_history: deque = deque(maxlen=1000)
    
    async def optimize_strategy_parameters(self, 
                                         strategy: DeceptionStrategy,
                                         effectiveness_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize strategy parameters based on effectiveness data"""
        
        if not effectiveness_data:
            return strategy.mutation_parameters.copy()
        
        # Analyze which parameter combinations were most effective
        parameter_effectiveness = defaultdict(list)
        
        for data_point in effectiveness_data:
            parameters = data_point.get('parameters', {})
            effectiveness = data_point.get('effectiveness_score', 0)
            
            for param_name, param_value in parameters.items():
                parameter_effectiveness[param_name].append((param_value, effectiveness))
        
        # Find optimal parameter values
        optimized_parameters = strategy.mutation_parameters.copy()
        
        for param_name, value_effectiveness_pairs in parameter_effectiveness.items():
            if len(value_effectiveness_pairs) < 2:
                continue
            
            # Group by parameter value and calculate average effectiveness
            value_groups = defaultdict(list)
            for value, effectiveness in value_effectiveness_pairs:
                value_groups[value].append(effectiveness)
            
            # Find value with highest average effectiveness
            best_value = None
            best_avg_effectiveness = -1
            
            for value, effectiveness_scores in value_groups.items():
                avg_effectiveness = np.mean(effectiveness_scores)
                if avg_effectiveness > best_avg_effectiveness:
                    best_avg_effectiveness = avg_effectiveness
                    best_value = value
            
            if best_value is not None and param_name in optimized_parameters:
                # Update to most effective value if it's in the allowed values
                if isinstance(optimized_parameters[param_name], list):
                    if best_value in optimized_parameters[param_name]:
                        # Bias towards the most effective value
                        value_list = optimized_parameters[param_name].copy()
                        value_list = [best_value] * 3 + [v for v in value_list if v != best_value]
                        optimized_parameters[param_name] = value_list
        
        return optimized_parameters


class DeploymentCoordinator:
    """Coordinates deployment and management of deception strategies"""
    
    def __init__(self):
        self.deployment_registry: Dict[str, Dict[str, Any]] = {}
        self.resource_manager = ResourceManager()
    
    async def deploy(self, 
                   deployment: DeceptionDeployment,
                   strategy: DeceptionStrategy) -> bool:
        """Deploy deception strategy"""
        
        try:
            # Check resource availability
            if not await self.resource_manager.check_resources(strategy.resource_requirements):
                logger.warning(f"Insufficient resources for deployment {deployment.deployment_id}")
                return False
            
            # Reserve resources
            await self.resource_manager.reserve_resources(
                deployment.deployment_id, 
                strategy.resource_requirements
            )
            
            # Simulate deployment process
            await asyncio.sleep(0.1)  # Simulated deployment time
            
            # Register deployment
            self.deployment_registry[deployment.deployment_id] = {
                'strategy_id': strategy.strategy_id,
                'deployed_at': deployment.deployed_at,
                'status': 'active',
                'resources': strategy.resource_requirements
            }
            
            logger.info(f"Deployment {deployment.deployment_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy {deployment.deployment_id}: {e}")
            return False
    
    async def update_deployment(self, 
                              deployment: DeceptionDeployment,
                              new_parameters: Dict[str, Any]) -> bool:
        """Update active deployment with new parameters"""
        
        try:
            # Simulate parameter update
            await asyncio.sleep(0.05)
            
            # Update registry
            if deployment.deployment_id in self.deployment_registry:
                self.deployment_registry[deployment.deployment_id]['last_updated'] = datetime.now()
                self.deployment_registry[deployment.deployment_id]['parameters'] = new_parameters
            
            logger.debug(f"Updated deployment {deployment.deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update deployment {deployment.deployment_id}: {e}")
            return False
    
    async def cleanup(self, deployment: DeceptionDeployment) -> None:
        """Cleanup deployment and release resources"""
        
        try:
            # Release resources
            await self.resource_manager.release_resources(deployment.deployment_id)
            
            # Remove from registry
            if deployment.deployment_id in self.deployment_registry:
                del self.deployment_registry[deployment.deployment_id]
            
            logger.debug(f"Cleaned up deployment {deployment.deployment_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup deployment {deployment.deployment_id}: {e}")


class ResourceManager:
    """Manages computational resources for deception deployments"""
    
    def __init__(self):
        self.available_resources = {
            'cpu': 8.0,  # CPU cores
            'memory': '16GB',  # Memory
            'storage': '100GB',  # Storage
            'network': 'high'  # Network bandwidth
        }
        self.reserved_resources: Dict[str, Dict[str, Any]] = {}
    
    async def check_resources(self, requirements: Dict[str, Any]) -> bool:
        """Check if resources are available for deployment"""
        
        # Simplified resource checking
        for resource, requirement in requirements.items():
            if resource == 'cpu':
                total_reserved = sum(
                    r.get('cpu', 0) for r in self.reserved_resources.values()
                )
                if total_reserved + requirement > self.available_resources['cpu']:
                    return False
            # Additional resource checks would be implemented here
        
        return True
    
    async def reserve_resources(self, deployment_id: str, requirements: Dict[str, Any]) -> None:
        """Reserve resources for deployment"""
        self.reserved_resources[deployment_id] = requirements.copy()
    
    async def release_resources(self, deployment_id: str) -> None:
        """Release resources from deployment"""
        if deployment_id in self.reserved_resources:
            del self.reserved_resources[deployment_id]


async def main():
    """Main function for deception engine demonstration"""
    try:
        print("Archangel Advanced Deception Engine")
        print("=" * 50)
        
        # Initialize deception engine
        deception_engine = AdaptiveDeceptionEngine()
        
        # Show available strategies
        print(f"Available Strategies: {len(deception_engine.strategies)}")
        for strategy_id, strategy in deception_engine.strategies.items():
            print(f"  - {strategy.name} ({strategy.deception_type.value})")
            print(f"    Targets: {[p.value for p in strategy.target_profiles]}")
        
        # Simulate attacker interaction
        interaction_data = {
            'attacker_id': 'attacker_001',
            'attack_vector': 'web_scan',
            'tool_signature': 'nmap',
            'timestamp': datetime.now(),
            'source_ip': '192.168.1.100'
        }
        
        # Analyze attacker behavior
        attacker_pattern = await deception_engine.analyze_attacker_behavior(interaction_data)
        print(f"\nAttacker Analysis:")
        print(f"  Profile: {attacker_pattern.attacker_profile.value}")
        print(f"  Attack Vectors: {attacker_pattern.attack_vectors}")
        print(f"  Tools: {attacker_pattern.tool_signatures}")
        
        # Select and deploy strategy
        context = {'available_resources': {'cpu': 4.0, 'memory': '8GB'}}
        selected_strategy = await deception_engine.select_adaptive_strategy(attacker_pattern, context)
        
        if selected_strategy:
            print(f"\nSelected Strategy: {selected_strategy.name}")
            
            # Deploy strategy
            deployment = await deception_engine.deploy_strategy(
                selected_strategy, 
                [attacker_pattern],
                "test_environment"
            )
            print(f"Deployed as: {deployment.deployment_id}")
            
            # Simulate adaptation
            adaptation_data = {
                'evasion_detected': True,
                'sophistication_increase': False,
                'adaptation_trigger': 'evasion_detected'
            }
            
            adapted = await deception_engine.adapt_strategy(deployment, adaptation_data)
            print(f"Strategy Adapted: {adapted}")
            
            # Measure effectiveness
            effectiveness = await deception_engine.measure_effectiveness(deployment.deployment_id)
            print(f"Current Effectiveness: {effectiveness.name}")
        
        # Show system metrics
        metrics = await deception_engine.get_deception_metrics()
        print(f"\nSystem Metrics:")
        print(f"  Active Deployments: {metrics['system_overview']['active_deployments']}")
        print(f"  Tracked Attackers: {metrics['system_overview']['tracked_attackers']}")
        print(f"  Average Effectiveness: {metrics['effectiveness']['average_effectiveness']:.2f}")
        
        # Export intelligence
        output_path = Path("agents") / "deception_intelligence.json"
        await deception_engine.export_deception_intelligence(output_path)
        print(f"\nIntelligence exported to {output_path}")
        
        print("\nDeception engine demonstration complete!")
        
        # Shutdown
        await deception_engine.shutdown()
        
    except Exception as e:
        logger.error(f"Deception engine demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())