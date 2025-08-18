#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - MVP Management and Phased Delivery
System for managing MVP definition, feature rollout, and incremental delivery
"""

import logging
import json
import yaml
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any, Union, Callable
from enum import Enum
from pathlib import Path
import semver
from contextlib import contextmanager

from .critical_path import TaskDefinition, TaskStatus, TaskPriority, DeliveryPhase, CriticalPathAnalyzer

logger = logging.getLogger(__name__)

class FeatureFlag(Enum):
    """Feature flag states"""
    DISABLED = "disabled"
    DEVELOPMENT = "development"
    TESTING = "testing"
    ENABLED = "enabled"
    DEPRECATED = "deprecated"

class ReleaseType(Enum):
    """Release types for semantic versioning"""
    PATCH = "patch"
    MINOR = "minor"  
    MAJOR = "major"
    PRERELEASE = "prerelease"

@dataclass
class Feature:
    """Definition of a system feature"""
    feature_id: str
    name: str
    description: str
    
    # Implementation details
    required_tasks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other features
    
    # Feature flags
    flag_state: FeatureFlag = FeatureFlag.DISABLED
    rollout_percentage: float = 0.0
    
    # Metadata
    category: str = ""
    tags: Set[str] = field(default_factory=set)
    owner: Optional[str] = None
    
    # Quality gates
    acceptance_criteria: List[str] = field(default_factory=list)
    test_coverage_threshold: float = 0.8
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Rollout control
    target_environments: List[str] = field(default_factory=lambda: ["development"])
    rollout_strategy: str = "all_at_once"  # "all_at_once", "gradual", "canary"
    
    # Monitoring
    metrics: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    enabled_at: Optional[datetime] = None
    disabled_at: Optional[datetime] = None

@dataclass 
class ReleaseCandidate:
    """Release candidate definition"""
    version: str
    name: str
    description: str
    
    # Content
    included_features: List[str] = field(default_factory=list)
    included_tasks: List[str] = field(default_factory=list)
    bug_fixes: List[str] = field(default_factory=list)
    
    # Quality gates
    quality_gates: List[str] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    
    # Release metadata
    release_type: ReleaseType = ReleaseType.MINOR
    target_date: Optional[datetime] = None
    
    # Status tracking
    created_at: datetime = field(default_factory=datetime.now)
    released_at: Optional[datetime] = None
    
    # Risk assessment
    risk_level: float = 1.0
    rollback_plan: str = ""

@dataclass
class DeploymentEnvironment:
    """Deployment environment configuration"""
    env_id: str
    name: str
    description: str
    
    # Environment properties
    environment_type: str = "development"  # development, testing, staging, production
    auto_deploy: bool = False
    
    # Feature flag overrides
    feature_overrides: Dict[str, FeatureFlag] = field(default_factory=dict)
    
    # Quality gates
    required_gates: List[str] = field(default_factory=list)
    
    # Monitoring
    monitoring_enabled: bool = True
    alert_channels: List[str] = field(default_factory=list)

class MVPManager:
    """
    MVP Management and Phased Delivery System.
    
    Features:
    - MVP definition and tracking
    - Feature flag management
    - Phased rollout control
    - Release candidate management
    - Quality gate enforcement
    - Environment-specific deployment
    """
    
    def __init__(self):
        self.features: Dict[str, Feature] = {}
        self.environments: Dict[str, DeploymentEnvironment] = {}
        self.release_candidates: Dict[str, ReleaseCandidate] = {}
        self.current_mvp: Optional[str] = None
        self.current_version = "0.1.0"
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default environments
        self._create_default_environments()
    
    def _create_default_environments(self) -> None:
        """Create default deployment environments"""
        environments = [
            DeploymentEnvironment(
                env_id="development",
                name="Development",
                description="Local development environment",
                environment_type="development",
                auto_deploy=True,
                required_gates=["unit_tests"],
                monitoring_enabled=False
            ),
            DeploymentEnvironment(
                env_id="testing",
                name="Testing", 
                description="Automated testing environment",
                environment_type="testing",
                auto_deploy=True,
                required_gates=["unit_tests", "integration_tests"],
                monitoring_enabled=True
            ),
            DeploymentEnvironment(
                env_id="staging",
                name="Staging",
                description="Pre-production staging environment",
                environment_type="staging",
                auto_deploy=False,
                required_gates=["unit_tests", "integration_tests", "security_scan", "performance_tests"],
                monitoring_enabled=True,
                alert_channels=["slack-staging"]
            ),
            DeploymentEnvironment(
                env_id="production",
                name="Production",
                description="Live production environment",
                environment_type="production",
                auto_deploy=False,
                required_gates=[
                    "unit_tests", "integration_tests", "security_scan", 
                    "performance_tests", "load_tests", "manual_approval"
                ],
                monitoring_enabled=True,
                alert_channels=["pagerduty", "slack-production"]
            )
        ]
        
        for env in environments:
            self.environments[env.env_id] = env
    
    def define_archangel_mvp(self) -> str:
        """Define the Archangel MVP with core features"""
        mvp_features = [
            Feature(
                feature_id="multi_agent_coordination",
                name="Multi-Agent Coordination",
                description="Basic multi-agent coordination framework with LangGraph",
                required_tasks=["task_01"],
                category="Foundation",
                tags={"mvp", "foundation", "coordination"},
                acceptance_criteria=[
                    "Agents can be started and stopped",
                    "Basic message passing works",
                    "Agent coordination is logged",
                    "System handles agent failures gracefully"
                ],
                performance_requirements={
                    "max_startup_time": 30,  # seconds
                    "message_latency": 100,   # milliseconds
                    "max_memory_usage": 512   # MB
                },
                target_environments=["development", "testing", "staging"],
                metrics=["agent_count", "message_throughput", "error_rate"],
                alerts=["agent_failure", "high_error_rate"]
            ),
            
            Feature(
                feature_id="basic_memory_system",
                name="Basic Memory System", 
                description="Vector memory and basic knowledge storage",
                required_tasks=["task_02"],
                dependencies=["multi_agent_coordination"],
                category="Memory",
                tags={"mvp", "memory", "knowledge"},
                acceptance_criteria=[
                    "Agents can store and retrieve experiences",
                    "Basic similarity search works",
                    "Memory persistence across restarts",
                    "Memory cleanup and optimization functional"
                ],
                performance_requirements={
                    "search_latency": 200,      # milliseconds
                    "storage_efficiency": 0.8,  # compression ratio
                    "max_memory_size": 1024     # MB
                },
                metrics=["memory_usage", "search_performance", "storage_growth"],
                alerts=["memory_full", "slow_search"]
            ),
            
            Feature(
                feature_id="red_team_agents",
                name="Basic Red Team Agents",
                description="Simplified red team agents with basic attack capabilities", 
                required_tasks=["task_03"],
                dependencies=["multi_agent_coordination", "basic_memory_system"],
                category="Agents",
                tags={"mvp", "red-team", "agents"},
                acceptance_criteria=[
                    "ReconAgent can perform basic network discovery",
                    "ExploitAgent can attempt basic exploits", 
                    "Agents coordinate and share intelligence",
                    "Actions are logged and auditable"
                ],
                performance_requirements={
                    "action_frequency": 10,    # actions per minute
                    "decision_latency": 5000,  # milliseconds
                    "success_rate": 0.3        # 30% minimum success rate
                },
                metrics=["actions_per_minute", "success_rate", "intelligence_shared"],
                alerts=["agent_stuck", "low_success_rate"]
            ),
            
            Feature(
                feature_id="blue_team_agents", 
                name="Basic Blue Team Agents",
                description="Simplified blue team agents with basic defense capabilities",
                required_tasks=["task_04"],
                dependencies=["multi_agent_coordination", "basic_memory_system"],
                category="Agents",
                tags={"mvp", "blue-team", "agents"},
                acceptance_criteria=[
                    "SOCAnalystAgent can detect basic threats",
                    "Agents can coordinate defensive responses",
                    "Basic alerting and incident creation works",
                    "Defense actions are logged"
                ],
                performance_requirements={
                    "detection_latency": 30000,  # milliseconds
                    "alert_accuracy": 0.7,       # 70% accuracy
                    "response_time": 60000        # milliseconds
                },
                metrics=["alerts_generated", "detection_accuracy", "response_time"],
                alerts=["detection_failure", "slow_response"]
            ),
            
            Feature(
                feature_id="basic_environment",
                name="Basic Mock Environment", 
                description="Simplified mock enterprise environment for testing",
                required_tasks=["task_05"],
                category="Environment",
                tags={"mvp", "environment", "infrastructure"},
                acceptance_criteria=[
                    "Basic containerized services running",
                    "Network connectivity between services",
                    "Basic logging and monitoring active",
                    "Environment can be reset/restarted"
                ],
                performance_requirements={
                    "startup_time": 120,       # seconds
                    "service_availability": 0.95,
                    "resource_usage": 2048     # MB total
                },
                metrics=["service_uptime", "resource_usage", "startup_time"],
                alerts=["service_down", "high_resource_usage"]
            ),
            
            Feature(
                feature_id="basic_game_loop",
                name="Basic Game Loop",
                description="Simple scenario execution and phase management",
                required_tasks=["task_10"],
                dependencies=["red_team_agents", "blue_team_agents", "basic_environment"],
                category="Game Logic",
                tags={"mvp", "game-loop", "scenarios"},
                acceptance_criteria=[
                    "Scenarios can be started and stopped",
                    "Basic phase transitions work",
                    "Objectives are tracked",
                    "Game state is persisted"
                ],
                performance_requirements={
                    "phase_transition_time": 10000,  # milliseconds
                    "objective_update_latency": 1000,
                    "concurrent_scenarios": 1
                },
                metrics=["scenario_completion_rate", "phase_duration", "objective_success"],
                alerts=["scenario_stuck", "phase_timeout"]
            ),
            
            Feature(
                feature_id="basic_scoring",
                name="Basic Scoring System",
                description="Simple scoring and evaluation for red/blue team performance",
                required_tasks=["task_11"],
                dependencies=["basic_game_loop"],
                category="Evaluation", 
                tags={"mvp", "scoring", "evaluation"},
                acceptance_criteria=[
                    "Red and blue team scores calculated",
                    "Basic metrics collection works",
                    "Score persistence and history",
                    "Simple reporting available"
                ],
                performance_requirements={
                    "score_calculation_latency": 5000,  # milliseconds
                    "metric_collection_overhead": 0.05,  # 5% overhead
                    "report_generation_time": 30000
                },
                metrics=["scoring_accuracy", "calculation_time", "metric_coverage"],
                alerts=["scoring_error", "metric_collection_failure"]
            ),
            
            Feature(
                feature_id="basic_monitoring",
                name="Basic Monitoring and Logging",
                description="Essential monitoring and logging for system health",
                required_tasks=["task_45"],
                category="Operations",
                tags={"mvp", "monitoring", "logging"},
                acceptance_criteria=[
                    "All components log to centralized system",
                    "Basic health checks work",
                    "Error detection and alerting active",
                    "Logs are searchable and persistent"
                ],
                performance_requirements={
                    "log_ingestion_latency": 1000,   # milliseconds
                    "health_check_frequency": 30,    # seconds
                    "log_retention": 30              # days
                },
                metrics=["log_volume", "error_rate", "health_check_success"],
                alerts=["high_error_rate", "logging_failure", "health_check_failure"]
            )
        ]
        
        # Add features to system
        for feature in mvp_features:
            self.features[feature.feature_id] = feature
        
        # Create MVP release candidate
        mvp_version = "1.0.0-alpha"
        mvp_rc = ReleaseCandidate(
            version=mvp_version,
            name="Archangel MVP Alpha",
            description="Minimum viable product with core autonomous AI security features",
            included_features=[f.feature_id for f in mvp_features],
            included_tasks=list(set(task for f in mvp_features for task in f.required_tasks)),
            quality_gates=[
                "All MVP features functional",
                "Basic red vs blue scenarios work",
                "System stability under normal load",
                "Core security controls operational",
                "Basic monitoring and logging active"
            ],
            release_type=ReleaseType.MAJOR,
            target_date=datetime.now() + timedelta(days=30)
        )
        
        self.release_candidates[mvp_version] = mvp_rc
        self.current_mvp = mvp_version
        
        self.logger.info(f"Archangel MVP defined with {len(mvp_features)} core features")
        return mvp_version
    
    def set_feature_flag(self, 
                        feature_id: str, 
                        flag_state: FeatureFlag,
                        environment: Optional[str] = None,
                        rollout_percentage: float = 0.0) -> None:
        """Set feature flag state"""
        if feature_id not in self.features:
            raise ValueError(f"Feature {feature_id} not found")
        
        feature = self.features[feature_id]
        
        if environment:
            # Set environment-specific override
            if environment not in self.environments:
                raise ValueError(f"Environment {environment} not found")
            
            self.environments[environment].feature_overrides[feature_id] = flag_state
        else:
            # Set global feature flag
            feature.flag_state = flag_state
            feature.rollout_percentage = rollout_percentage
            
            if flag_state == FeatureFlag.ENABLED:
                feature.enabled_at = datetime.now()
            elif flag_state == FeatureFlag.DISABLED:
                feature.disabled_at = datetime.now()
        
        self.logger.info(f"Set feature {feature_id} to {flag_state.value}" + 
                        (f" in {environment}" if environment else " globally"))
    
    def is_feature_enabled(self, feature_id: str, environment: str = "development") -> bool:
        """Check if feature is enabled in given environment"""
        if feature_id not in self.features:
            return False
        
        feature = self.features[feature_id]
        
        # Check environment-specific override first
        if environment in self.environments:
            env_overrides = self.environments[environment].feature_overrides
            if feature_id in env_overrides:
                return env_overrides[feature_id] == FeatureFlag.ENABLED
        
        # Check global feature flag
        if feature.flag_state == FeatureFlag.ENABLED:
            # Check rollout percentage
            return feature.rollout_percentage >= 100.0
        
        return feature.flag_state == FeatureFlag.DEVELOPMENT and environment == "development"
    
    def gradual_rollout(self, 
                       feature_id: str,
                       target_percentage: float,
                       step_size: float = 10.0,
                       step_duration: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Implement gradual feature rollout"""
        if feature_id not in self.features:
            raise ValueError(f"Feature {feature_id} not found")
        
        feature = self.features[feature_id]
        current_percentage = feature.rollout_percentage
        
        rollout_plan = {
            'feature_id': feature_id,
            'current_percentage': current_percentage,
            'target_percentage': target_percentage,
            'step_size': step_size,
            'step_duration_hours': step_duration.total_seconds() / 3600,
            'estimated_completion': datetime.now() + timedelta(
                hours=((target_percentage - current_percentage) / step_size) * 
                      (step_duration.total_seconds() / 3600)
            ),
            'steps': []
        }
        
        # Generate rollout steps
        percentage = current_percentage
        step_time = datetime.now()
        
        while percentage < target_percentage:
            percentage = min(percentage + step_size, target_percentage)
            step_time += step_duration
            
            rollout_plan['steps'].append({
                'percentage': percentage,
                'scheduled_time': step_time,
                'quality_gates': [
                    f"Error rate < 5% for {feature_id}",
                    f"Performance degradation < 10%",
                    f"No critical alerts for {feature_id}"
                ]
            })
        
        self.logger.info(f"Generated gradual rollout plan for {feature_id}: "
                        f"{current_percentage}% -> {target_percentage}% in {len(rollout_plan['steps'])} steps")
        
        return rollout_plan
    
    def check_quality_gates(self, feature_id: str, environment: str = "testing") -> Dict[str, Any]:
        """Check quality gates for feature deployment"""
        if feature_id not in self.features:
            raise ValueError(f"Feature {feature_id} not found")
        
        feature = self.features[feature_id]
        env = self.environments.get(environment)
        
        gate_results = {
            'feature_id': feature_id,
            'environment': environment,
            'overall_passed': True,
            'gate_results': {},
            'recommendations': [],
            'checked_at': datetime.now()
        }
        
        # Check feature-specific acceptance criteria
        for criteria in feature.acceptance_criteria:
            # Simulate quality gate check (in real system, would integrate with CI/CD)
            gate_passed = self._simulate_quality_gate_check(criteria, feature_id, environment)
            gate_results['gate_results'][criteria] = {
                'passed': gate_passed,
                'checked_at': datetime.now()
            }
            
            if not gate_passed:
                gate_results['overall_passed'] = False
                gate_results['recommendations'].append(f"Fix failing criterion: {criteria}")
        
        # Check environment-specific gates
        if env:
            for gate in env.required_gates:
                gate_passed = self._simulate_quality_gate_check(gate, feature_id, environment)
                gate_results['gate_results'][gate] = {
                    'passed': gate_passed,
                    'checked_at': datetime.now()
                }
                
                if not gate_passed:
                    gate_results['overall_passed'] = False
                    gate_results['recommendations'].append(f"Fix failing gate: {gate}")
        
        # Check performance requirements
        for req_name, req_value in feature.performance_requirements.items():
            perf_passed = self._simulate_performance_check(req_name, req_value, feature_id, environment)
            gate_results['gate_results'][f"performance_{req_name}"] = {
                'passed': perf_passed,
                'checked_at': datetime.now()
            }
            
            if not perf_passed:
                gate_results['overall_passed'] = False
                gate_results['recommendations'].append(f"Fix performance issue: {req_name}")
        
        return gate_results
    
    def _simulate_quality_gate_check(self, gate: str, feature_id: str, environment: str) -> bool:
        """Simulate quality gate check (replace with real implementation)"""
        # In real system, this would integrate with actual testing infrastructure
        import random
        
        # Simulate different gate success rates
        success_rates = {
            'unit_tests': 0.9,
            'integration_tests': 0.85,
            'security_scan': 0.8,
            'performance_tests': 0.75,
            'load_tests': 0.7,
            'manual_approval': 0.95
        }
        
        # Use feature and environment to create deterministic but varied results
        random.seed(hash(f"{gate}_{feature_id}_{environment}"))
        success_rate = success_rates.get(gate, 0.8)
        
        return random.random() < success_rate
    
    def _simulate_performance_check(self, req_name: str, req_value: Any, feature_id: str, environment: str) -> bool:
        """Simulate performance requirement check"""
        import random
        
        # Simulate performance check based on requirement type
        random.seed(hash(f"{req_name}_{feature_id}_{environment}"))
        
        # Most performance checks should pass in testing environments
        return random.random() < 0.85
    
    def create_release_candidate(self, 
                                version: str,
                                name: str, 
                                description: str,
                                included_features: List[str],
                                release_type: ReleaseType = ReleaseType.MINOR) -> str:
        """Create a new release candidate"""
        
        # Validate version format
        try:
            semver.VersionInfo.parse(version)
        except ValueError as e:
            raise ValueError(f"Invalid version format: {version}") from e
        
        # Validate features exist
        missing_features = [f for f in included_features if f not in self.features]
        if missing_features:
            raise ValueError(f"Features not found: {missing_features}")
        
        # Collect all required tasks
        included_tasks = []
        for feature_id in included_features:
            feature = self.features[feature_id]
            included_tasks.extend(feature.required_tasks)
        
        # Create release candidate
        rc = ReleaseCandidate(
            version=version,
            name=name,
            description=description,
            included_features=included_features,
            included_tasks=list(set(included_tasks)),
            release_type=release_type,
            quality_gates=[
                "All included features pass quality gates",
                "Integration tests pass",
                "Performance requirements met",
                "Security scan clean",
                "Documentation updated"
            ]
        )
        
        self.release_candidates[version] = rc
        
        self.logger.info(f"Created release candidate {version} with {len(included_features)} features")
        return version
    
    def deploy_to_environment(self, 
                             version: str,
                             environment: str,
                             force: bool = False) -> Dict[str, Any]:
        """Deploy release candidate to environment"""
        if version not in self.release_candidates:
            raise ValueError(f"Release candidate {version} not found")
        
        if environment not in self.environments:
            raise ValueError(f"Environment {environment} not found")
        
        rc = self.release_candidates[version]
        env = self.environments[environment]
        
        deployment_result = {
            'version': version,
            'environment': environment,
            'success': True,
            'deployed_at': datetime.now(),
            'features_deployed': [],
            'quality_gate_results': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check quality gates unless forced
            if not force:
                for feature_id in rc.included_features:
                    gate_result = self.check_quality_gates(feature_id, environment)
                    deployment_result['quality_gate_results'][feature_id] = gate_result
                    
                    if not gate_result['overall_passed']:
                        deployment_result['errors'].append(
                            f"Quality gates failed for feature {feature_id}"
                        )
                        deployment_result['success'] = False
            
            # Deploy features if quality gates pass or forced
            if deployment_result['success'] or force:
                for feature_id in rc.included_features:
                    try:
                        # Enable feature in environment
                        self.set_feature_flag(feature_id, FeatureFlag.ENABLED, environment, 100.0)
                        deployment_result['features_deployed'].append(feature_id)
                        
                    except Exception as e:
                        deployment_result['errors'].append(
                            f"Failed to deploy feature {feature_id}: {str(e)}"
                        )
                        deployment_result['success'] = False
                
                if deployment_result['success']:
                    rc.released_at = datetime.now()
                    self.current_version = version
            
            return deployment_result
            
        except Exception as e:
            deployment_result['success'] = False
            deployment_result['errors'].append(f"Deployment failed: {str(e)}")
            return deployment_result
    
    def get_mvp_status(self) -> Dict[str, Any]:
        """Get current MVP status"""
        if not self.current_mvp:
            return {'mvp_defined': False}
        
        mvp_rc = self.release_candidates[self.current_mvp]
        
        status = {
            'mvp_defined': True,
            'version': self.current_mvp,
            'name': mvp_rc.name,
            'features': {},
            'overall_progress': 0.0,
            'ready_for_release': False,
            'blocking_issues': []
        }
        
        total_features = len(mvp_rc.included_features)
        ready_features = 0
        
        for feature_id in mvp_rc.included_features:
            feature = self.features[feature_id]
            feature_status = {
                'name': feature.name,
                'flag_state': feature.flag_state.value,
                'rollout_percentage': feature.rollout_percentage,
                'quality_gates_passed': False,
                'ready': False
            }
            
            # Check quality gates for staging environment
            try:
                gate_result = self.check_quality_gates(feature_id, "staging")
                feature_status['quality_gates_passed'] = gate_result['overall_passed']
                
                if not gate_result['overall_passed']:
                    status['blocking_issues'].extend([
                        f"{feature_id}: {rec}" for rec in gate_result['recommendations']
                    ])
                
            except Exception as e:
                feature_status['quality_gates_passed'] = False
                status['blocking_issues'].append(f"{feature_id}: Quality gate check failed - {str(e)}")
            
            # Feature is ready if enabled and quality gates pass
            feature_status['ready'] = (
                feature.flag_state in [FeatureFlag.ENABLED, FeatureFlag.TESTING] and
                feature_status['quality_gates_passed']
            )
            
            if feature_status['ready']:
                ready_features += 1
            
            status['features'][feature_id] = feature_status
        
        status['overall_progress'] = ready_features / total_features if total_features > 0 else 0.0
        status['ready_for_release'] = status['overall_progress'] >= 1.0
        
        return status
    
    def export_feature_config(self, output_path: Path) -> None:
        """Export feature configuration for runtime use"""
        try:
            config = {
                'features': {},
                'environments': {},
                'current_version': self.current_version,
                'current_mvp': self.current_mvp,
                'exported_at': datetime.now().isoformat()
            }
            
            # Export feature flags
            for feature_id, feature in self.features.items():
                config['features'][feature_id] = {
                    'name': feature.name,
                    'flag_state': feature.flag_state.value,
                    'rollout_percentage': feature.rollout_percentage,
                    'target_environments': feature.target_environments,
                    'category': feature.category,
                    'tags': list(feature.tags)
                }
            
            # Export environment configurations
            for env_id, env in self.environments.items():
                config['environments'][env_id] = {
                    'name': env.name,
                    'environment_type': env.environment_type,
                    'auto_deploy': env.auto_deploy,
                    'feature_overrides': {
                        k: v.value for k, v in env.feature_overrides.items()
                    },
                    'monitoring_enabled': env.monitoring_enabled
                }
            
            # Write configuration
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            self.logger.info(f"Feature configuration exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export feature config: {e}")
            raise


@contextmanager
def feature_flag(mvp_manager: MVPManager, feature_id: str, environment: str = "development"):
    """Context manager for feature flag checking"""
    if mvp_manager.is_feature_enabled(feature_id, environment):
        yield True
    else:
        yield False


def main():
    """Main function for MVP management demonstration"""
    try:
        print("Archangel MVP Management System")
        print("=" * 50)
        
        # Initialize MVP manager
        mvp_manager = MVPManager()
        
        # Define Archangel MVP
        mvp_version = mvp_manager.define_archangel_mvp()
        print(f"MVP defined: {mvp_version}")
        
        # Show MVP features
        mvp_rc = mvp_manager.release_candidates[mvp_version]
        print(f"\nMVP Features ({len(mvp_rc.included_features)}):")
        for feature_id in mvp_rc.included_features:
            feature = mvp_manager.features[feature_id]
            print(f"  - {feature.name}")
            print(f"    Tasks: {', '.join(feature.required_tasks)}")
            print(f"    State: {feature.flag_state.value}")
        
        # Enable some features for testing
        mvp_manager.set_feature_flag("multi_agent_coordination", FeatureFlag.ENABLED, rollout_percentage=100.0)
        mvp_manager.set_feature_flag("basic_memory_system", FeatureFlag.TESTING, rollout_percentage=50.0)
        mvp_manager.set_feature_flag("red_team_agents", FeatureFlag.DEVELOPMENT)
        
        print(f"\nFeature Flags Updated:")
        for feature_id in ["multi_agent_coordination", "basic_memory_system", "red_team_agents"]:
            enabled = mvp_manager.is_feature_enabled(feature_id, "development")
            print(f"  - {feature_id}: {'✓' if enabled else '✗'} (development)")
        
        # Check MVP status
        mvp_status = mvp_manager.get_mvp_status()
        print(f"\nMVP Status:")
        print(f"  Progress: {mvp_status['overall_progress']:.1%}")
        print(f"  Ready for release: {'Yes' if mvp_status['ready_for_release'] else 'No'}")
        print(f"  Blocking issues: {len(mvp_status['blocking_issues'])}")
        
        # Show gradual rollout plan
        rollout_plan = mvp_manager.gradual_rollout("basic_memory_system", 100.0, 25.0)
        print(f"\nGradual Rollout Plan for {rollout_plan['feature_id']}:")
        print(f"  Current: {rollout_plan['current_percentage']}%")
        print(f"  Target: {rollout_plan['target_percentage']}%")
        print(f"  Steps: {len(rollout_plan['steps'])}")
        print(f"  Completion: {rollout_plan['estimated_completion'].strftime('%Y-%m-%d %H:%M')}")
        
        # Test deployment
        deployment_result = mvp_manager.deploy_to_environment(mvp_version, "testing", force=True)
        print(f"\nDeployment to Testing:")
        print(f"  Success: {'Yes' if deployment_result['success'] else 'No'}")
        print(f"  Features deployed: {len(deployment_result['features_deployed'])}")
        if deployment_result['errors']:
            print(f"  Errors: {len(deployment_result['errors'])}")
        
        # Export configuration
        output_path = Path("project_management") / "feature_config.json"
        mvp_manager.export_feature_config(output_path)
        print(f"\nFeature configuration exported to {output_path}")
        
        # Demonstrate feature flag usage
        print(f"\nFeature Flag Usage Example:")
        with feature_flag(mvp_manager, "multi_agent_coordination", "development") as enabled:
            if enabled:
                print("  ✓ Multi-agent coordination is enabled - proceeding with functionality")
            else:
                print("  ✗ Multi-agent coordination is disabled - skipping functionality")
        
        print("\nMVP management demonstration complete!")
        
    except Exception as e:
        logger.error(f"MVP management demo failed: {e}")
        raise


if __name__ == "__main__":
    main()