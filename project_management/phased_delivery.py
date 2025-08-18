#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Phased Delivery System
Comprehensive system for managing incremental feature rollout and deployment phases
"""

import logging
import json
import yaml
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
from enum import Enum
from pathlib import Path
import calendar
from collections import defaultdict

from .mvp_manager import MVPManager, Feature, FeatureFlag, ReleaseCandidate
from .critical_path import TaskDefinition, TaskStatus, CriticalPathAnalyzer

logger = logging.getLogger(__name__)

class PhaseStatus(Enum):
    """Phase execution status"""
    PLANNED = "planned"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class RolloutStrategy(Enum):
    """Feature rollout strategies"""
    BIG_BANG = "big_bang"           # All at once
    CANARY = "canary"               # Small percentage first
    BLUE_GREEN = "blue_green"       # Switch between environments
    ROLLING = "rolling"             # Gradual percentage increase
    RING = "ring"                   # Deploy to user rings
    GEOGRAPHIC = "geographic"       # Deploy by region/location

@dataclass
class DeliveryMilestone:
    """Delivery milestone definition"""
    milestone_id: str
    name: str
    description: str
    
    # Timing
    planned_date: date
    actual_date: Optional[date] = None
    
    # Requirements
    required_features: List[str] = field(default_factory=list)
    required_tasks: List[str] = field(default_factory=list)
    
    # Success criteria
    success_criteria: List[str] = field(default_factory=list)
    quality_gates: List[str] = field(default_factory=list)
    
    # Status
    status: PhaseStatus = PhaseStatus.PLANNED
    completion_percentage: float = 0.0
    
    # Risks and mitigation
    risk_factors: List[str] = field(default_factory=list)
    mitigation_plans: List[str] = field(default_factory=list)

@dataclass
class DeliveryPhaseDefinition:
    """Definition of a delivery phase"""
    phase_id: str
    name: str
    description: str
    
    # Timing
    start_date: date
    end_date: date
    duration_days: int = field(init=False)
    
    # Content
    included_features: List[str] = field(default_factory=list)
    milestones: List[str] = field(default_factory=list)
    
    # Dependencies
    depends_on_phases: List[str] = field(default_factory=list)
    blocks_phases: List[str] = field(default_factory=list)
    
    # Execution
    rollout_strategy: RolloutStrategy = RolloutStrategy.ROLLING
    target_environments: List[str] = field(default_factory=list)
    
    # Quality and success
    success_criteria: List[str] = field(default_factory=list)
    exit_criteria: List[str] = field(default_factory=list)
    
    # Status tracking
    status: PhaseStatus = PhaseStatus.PLANNED
    actual_start_date: Optional[date] = None
    actual_end_date: Optional[date] = None
    
    # Metrics
    kpis: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields"""
        self.duration_days = (self.end_date - self.start_date).days

@dataclass
class RolloutPlan:
    """Detailed rollout plan for a feature or phase"""
    plan_id: str
    feature_id: str
    strategy: RolloutStrategy
    
    # Rollout configuration
    total_duration: timedelta
    environments: List[str]
    rollout_steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality gates
    quality_gates: List[str] = field(default_factory=list)
    rollback_triggers: List[str] = field(default_factory=list)
    
    # Monitoring
    success_metrics: List[str] = field(default_factory=list)
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Status
    current_step: int = 0
    status: PhaseStatus = PhaseStatus.PLANNED
    
    # Risk management
    rollback_plan: str = ""
    contingency_plans: List[str] = field(default_factory=list)

class PhasedDeliveryManager:
    """
    Comprehensive phased delivery management system.
    
    Features:
    - Multi-phase delivery planning and execution
    - Flexible rollout strategies and timing
    - Milestone tracking and quality gates
    - Risk management and rollback capabilities
    - Progress monitoring and reporting
    - Integration with feature flags and MVP management
    """
    
    def __init__(self, mvp_manager: MVPManager):
        self.mvp_manager = mvp_manager
        self.phases: Dict[str, DeliveryPhaseDefinition] = {}
        self.milestones: Dict[str, DeliveryMilestone] = {}
        self.rollout_plans: Dict[str, RolloutPlan] = {}
        
        self.current_phase: Optional[str] = None
        self.project_start_date = date.today()
        self.project_target_date = date.today() + timedelta(days=365)
        
        self.logger = logging.getLogger(__name__)
    
    def define_archangel_delivery_phases(self) -> None:
        """Define delivery phases for the Archangel project"""
        
        # Phase 1: Foundation (Weeks 1-3)
        foundation_phase = DeliveryPhaseDefinition(
            phase_id="foundation",
            name="Foundation Phase",
            description="Core architecture and basic autonomous agent framework",
            start_date=self.project_start_date,
            end_date=self.project_start_date + timedelta(days=21),
            included_features=[
                "multi_agent_coordination",
                "basic_memory_system", 
                "basic_monitoring"
            ],
            target_environments=["development", "testing"],
            rollout_strategy=RolloutStrategy.ROLLING,
            success_criteria=[
                "All foundation features operational",
                "Basic agent coordination working",
                "Memory system storing and retrieving data",
                "Essential monitoring and logging active"
            ],
            exit_criteria=[
                "Foundation features pass all quality gates",
                "System stable under normal load",
                "Ready for agent development phase"
            ],
            kpis={
                "system_uptime": 0.95,
                "test_coverage": 0.8,
                "defect_density": 2.0  # defects per KLOC
            }
        )
        
        # Phase 2: Agent Development (Weeks 4-6)
        agent_phase = DeliveryPhaseDefinition(
            phase_id="agent_development",
            name="Agent Development Phase", 
            description="Red and Blue team autonomous agents with basic capabilities",
            start_date=foundation_phase.end_date,
            end_date=foundation_phase.end_date + timedelta(days=21),
            included_features=[
                "red_team_agents",
                "blue_team_agents"
            ],
            depends_on_phases=["foundation"],
            target_environments=["development", "testing", "staging"],
            rollout_strategy=RolloutStrategy.CANARY,
            success_criteria=[
                "Red team agents perform basic attacks",
                "Blue team agents detect and respond",
                "Agent coordination and intelligence sharing working",
                "Basic adversarial scenarios executable"
            ],
            exit_criteria=[
                "Agents operate autonomously",
                "Intelligence sharing functional",
                "Ready for environment integration"
            ],
            kpis={
                "agent_success_rate": 0.6,
                "coordination_latency": 5.0,  # seconds
                "intelligence_accuracy": 0.7
            }
        )
        
        # Phase 3: Environment Integration (Weeks 7-9)
        environment_phase = DeliveryPhaseDefinition(
            phase_id="environment_integration",
            name="Environment Integration Phase",
            description="Mock enterprise environment with realistic attack/defense scenarios",
            start_date=agent_phase.end_date,
            end_date=agent_phase.end_date + timedelta(days=21),
            included_features=[
                "basic_environment",
                "basic_game_loop",
                "basic_scoring"
            ],
            depends_on_phases=["foundation", "agent_development"],
            target_environments=["development", "testing", "staging"],
            rollout_strategy=RolloutStrategy.ROLLING,
            success_criteria=[
                "Mock environment operational",
                "Agents interact with environment",
                "Game loop manages scenario phases",
                "Scoring system tracks performance"
            ],
            exit_criteria=[
                "End-to-end scenarios working",
                "Environment realistic and stable",
                "Ready for advanced features"
            ],
            kpis={
                "environment_uptime": 0.95,
                "scenario_completion_rate": 0.8,
                "scoring_accuracy": 0.85
            }
        )
        
        # Phase 4: MVP Release (Week 10)
        mvp_phase = DeliveryPhaseDefinition(
            phase_id="mvp_release",
            name="MVP Release Phase",
            description="Minimum viable product release with core functionality",
            start_date=environment_phase.end_date,
            end_date=environment_phase.end_date + timedelta(days=7),
            included_features=[
                "multi_agent_coordination", "basic_memory_system",
                "red_team_agents", "blue_team_agents", 
                "basic_environment", "basic_game_loop",
                "basic_scoring", "basic_monitoring"
            ],
            depends_on_phases=["foundation", "agent_development", "environment_integration"],
            target_environments=["staging", "production"],
            rollout_strategy=RolloutStrategy.BLUE_GREEN,
            success_criteria=[
                "All MVP features operational in production",
                "Basic red vs blue scenarios working end-to-end",
                "System meets performance requirements",
                "Monitoring and alerting functional"
            ],
            exit_criteria=[
                "MVP deployed to production",
                "User acceptance criteria met",
                "Ready for advanced feature development"
            ],
            kpis={
                "system_availability": 0.99,
                "user_satisfaction": 0.8,
                "performance_sla_met": 0.95
            }
        )
        
        # Phase 5: Advanced Features (Weeks 11-15)
        advanced_phase = DeliveryPhaseDefinition(
            phase_id="advanced_features",
            name="Advanced Features Phase",
            description="Advanced AI features, learning systems, and enhanced capabilities",
            start_date=mvp_phase.end_date,
            end_date=mvp_phase.end_date + timedelta(days=35),
            included_features=[
                "advanced_reasoning", "self_play_learning",
                "deception_technologies", "advanced_monitoring"
            ],
            depends_on_phases=["mvp_release"],
            target_environments=["development", "testing", "staging", "production"],
            rollout_strategy=RolloutStrategy.CANARY,
            success_criteria=[
                "Advanced AI reasoning operational",
                "Self-play learning improving agent performance",
                "Deception technologies effective",
                "Advanced monitoring providing insights"
            ],
            exit_criteria=[
                "Advanced features stable and effective",
                "Learning systems showing improvement",
                "Ready for production scale"
            ],
            kpis={
                "learning_improvement_rate": 0.1,  # 10% per week
                "deception_effectiveness": 0.7,
                "advanced_scenario_success": 0.6
            }
        )
        
        # Phase 6: Production Scale (Weeks 16-20)
        production_phase = DeliveryPhaseDefinition(
            phase_id="production_scale",
            name="Production Scale Phase",
            description="Production deployment, scaling, and optimization",
            start_date=advanced_phase.end_date,
            end_date=advanced_phase.end_date + timedelta(days=35),
            included_features=[
                "horizontal_scaling", "performance_optimization",
                "enterprise_integration", "comprehensive_monitoring"
            ],
            depends_on_phases=["advanced_features"],
            target_environments=["production"],
            rollout_strategy=RolloutStrategy.GEOGRAPHIC,
            success_criteria=[
                "System scales to production load",
                "Performance optimized for efficiency",
                "Enterprise integrations working",
                "Comprehensive monitoring operational"
            ],
            exit_criteria=[
                "Production system stable and scalable",
                "All quality and performance targets met",
                "Ready for general availability"
            ],
            kpis={
                "concurrent_users": 1000,
                "response_time_p95": 2.0,  # seconds
                "cost_per_user": 10.0,     # dollars per month
                "system_efficiency": 0.85
            }
        )
        
        # Add phases to manager
        phases = [
            foundation_phase, agent_phase, environment_phase,
            mvp_phase, advanced_phase, production_phase
        ]
        
        for phase in phases:
            self.phases[phase.phase_id] = phase
        
        # Set current phase
        self.current_phase = "foundation"
        
        self.logger.info(f"Defined {len(phases)} delivery phases for Archangel project")
    
    def create_milestones(self) -> None:
        """Create delivery milestones for tracking progress"""
        
        milestones = [
            DeliveryMilestone(
                milestone_id="m1_foundation_complete",
                name="Foundation Complete",
                description="Core architecture and agent framework operational",
                planned_date=self.project_start_date + timedelta(days=21),
                required_features=["multi_agent_coordination", "basic_memory_system"],
                success_criteria=[
                    "Agent coordination framework working",
                    "Memory system operational", 
                    "Basic monitoring active",
                    "All foundation tests passing"
                ],
                quality_gates=[
                    "Unit tests > 80% coverage",
                    "Integration tests passing",
                    "Performance within requirements"
                ]
            ),
            
            DeliveryMilestone(
                milestone_id="m2_agents_operational",
                name="Agents Operational", 
                description="Red and Blue team agents working autonomously",
                planned_date=self.project_start_date + timedelta(days=42),
                required_features=["red_team_agents", "blue_team_agents"],
                success_criteria=[
                    "Red team agents perform attacks",
                    "Blue team agents detect and respond",
                    "Agent intelligence sharing working",
                    "Basic coordination scenarios successful"
                ],
                quality_gates=[
                    "Agent decision-making tests passing",
                    "Coordination logic validated",
                    "Security boundaries enforced"
                ]
            ),
            
            DeliveryMilestone(
                milestone_id="m3_mvp_ready",
                name="MVP Ready",
                description="Minimum viable product ready for release",
                planned_date=self.project_start_date + timedelta(days=70),
                required_features=[
                    "multi_agent_coordination", "basic_memory_system",
                    "red_team_agents", "blue_team_agents",
                    "basic_environment", "basic_game_loop", "basic_scoring"
                ],
                success_criteria=[
                    "End-to-end scenarios working",
                    "All MVP features operational",
                    "Performance requirements met",
                    "User acceptance criteria satisfied"
                ],
                quality_gates=[
                    "All test suites passing",
                    "Security audit complete",
                    "Performance benchmarks met",
                    "Documentation complete"
                ]
            ),
            
            DeliveryMilestone(
                milestone_id="m4_production_ready",
                name="Production Ready",
                description="System ready for production deployment and scaling",
                planned_date=self.project_start_date + timedelta(days=140),
                required_features=["horizontal_scaling", "enterprise_integration"],
                success_criteria=[
                    "Production deployment successful",
                    "Scaling requirements met",
                    "All integrations working",
                    "Monitoring and alerting operational"
                ],
                quality_gates=[
                    "Load testing passed",
                    "Security penetration testing clean",
                    "Disaster recovery tested",
                    "SLA requirements validated"
                ]
            )
        ]
        
        for milestone in milestones:
            self.milestones[milestone.milestone_id] = milestone
        
        # Assign milestones to phases
        self.phases["foundation"].milestones = ["m1_foundation_complete"]
        self.phases["agent_development"].milestones = ["m2_agents_operational"] 
        self.phases["mvp_release"].milestones = ["m3_mvp_ready"]
        self.phases["production_scale"].milestones = ["m4_production_ready"]
        
        self.logger.info(f"Created {len(milestones)} delivery milestones")
    
    def create_rollout_plans(self) -> None:
        """Create detailed rollout plans for each phase"""
        
        # Foundation Phase - Rolling rollout
        foundation_plan = RolloutPlan(
            plan_id="foundation_rollout",
            feature_id="foundation",
            strategy=RolloutStrategy.ROLLING,
            total_duration=timedelta(days=21),
            environments=["development", "testing"],
            rollout_steps=[
                {
                    "step": 1,
                    "description": "Deploy to development",
                    "duration_days": 7,
                    "environments": ["development"],
                    "percentage": 100,
                    "quality_gates": ["unit_tests", "smoke_tests"]
                },
                {
                    "step": 2, 
                    "description": "Deploy to testing",
                    "duration_days": 7,
                    "environments": ["testing"],
                    "percentage": 100,
                    "quality_gates": ["integration_tests", "regression_tests"]
                },
                {
                    "step": 3,
                    "description": "Stabilization and optimization",
                    "duration_days": 7,
                    "environments": ["development", "testing"],
                    "percentage": 100,
                    "quality_gates": ["performance_tests", "stability_tests"]
                }
            ],
            quality_gates=[
                "All unit tests passing",
                "Integration tests passing", 
                "Performance within baseline",
                "No critical defects"
            ],
            rollback_triggers=[
                "Test failure rate > 5%",
                "Performance degradation > 20%",
                "Critical defects found"
            ],
            success_metrics=[
                "test_pass_rate",
                "deployment_success_rate",
                "system_stability"
            ],
            alert_thresholds={
                "error_rate": 0.05,
                "performance_degradation": 0.20,
                "availability": 0.95
            },
            rollback_plan="Revert to previous stable version, investigate issues, fix and redeploy"
        )
        
        # Agent Development - Canary rollout
        agent_plan = RolloutPlan(
            plan_id="agent_rollout",
            feature_id="agent_development", 
            strategy=RolloutStrategy.CANARY,
            total_duration=timedelta(days=21),
            environments=["development", "testing", "staging"],
            rollout_steps=[
                {
                    "step": 1,
                    "description": "Canary deployment (10%)",
                    "duration_days": 3,
                    "environments": ["development"],
                    "percentage": 10,
                    "quality_gates": ["agent_functionality_tests", "safety_checks"]
                },
                {
                    "step": 2,
                    "description": "Expand canary (50%)",
                    "duration_days": 5,
                    "environments": ["development", "testing"],
                    "percentage": 50,
                    "quality_gates": ["coordination_tests", "intelligence_sharing_tests"]
                },
                {
                    "step": 3,
                    "description": "Full deployment",
                    "duration_days": 7,
                    "environments": ["development", "testing"],
                    "percentage": 100,
                    "quality_gates": ["full_scenario_tests", "stress_tests"]
                },
                {
                    "step": 4,
                    "description": "Staging deployment",
                    "duration_days": 6,
                    "environments": ["staging"],
                    "percentage": 100,
                    "quality_gates": ["pre_production_tests", "security_validation"]
                }
            ],
            quality_gates=[
                "Agent decision-making validated",
                "Coordination mechanisms working",
                "Security boundaries enforced",
                "Performance within requirements"
            ],
            rollback_triggers=[
                "Agent failure rate > 10%",
                "Security boundary violations",
                "Coordination failures > 5%"
            ],
            success_metrics=[
                "agent_success_rate",
                "coordination_effectiveness", 
                "security_compliance"
            ],
            alert_thresholds={
                "agent_failure_rate": 0.10,
                "coordination_latency": 10.0,  # seconds
                "security_violations": 0
            }
        )
        
        # MVP Release - Blue-Green deployment
        mvp_plan = RolloutPlan(
            plan_id="mvp_rollout",
            feature_id="mvp_release",
            strategy=RolloutStrategy.BLUE_GREEN,
            total_duration=timedelta(days=7),
            environments=["staging", "production"],
            rollout_steps=[
                {
                    "step": 1,
                    "description": "Deploy to Green environment",
                    "duration_days": 2,
                    "environments": ["staging-green"],
                    "percentage": 100,
                    "quality_gates": ["full_mvp_tests", "performance_validation"]
                },
                {
                    "step": 2,
                    "description": "Switch traffic to Green",
                    "duration_days": 1,
                    "environments": ["production-green"], 
                    "percentage": 100,
                    "quality_gates": ["production_smoke_tests", "monitoring_validation"]
                },
                {
                    "step": 3,
                    "description": "Monitor and stabilize",
                    "duration_days": 2,
                    "environments": ["production"],
                    "percentage": 100,
                    "quality_gates": ["stability_monitoring", "user_acceptance"]
                },
                {
                    "step": 4,
                    "description": "Decommission Blue",
                    "duration_days": 2,
                    "environments": ["production"],
                    "percentage": 100,
                    "quality_gates": ["rollback_capability_verified"]
                }
            ],
            quality_gates=[
                "All MVP features operational",
                "Performance SLAs met",
                "Security validation complete",
                "User acceptance criteria satisfied"
            ],
            rollback_triggers=[
                "Critical functionality failure",
                "Performance SLA breach",
                "Security incident",
                "User acceptance failure"
            ],
            success_metrics=[
                "system_availability",
                "user_satisfaction",
                "feature_adoption"
            ],
            alert_thresholds={
                "availability": 0.99,
                "response_time": 2.0,  # seconds
                "error_rate": 0.01
            },
            rollback_plan="Switch traffic back to Blue environment, investigate issues, fix in Green"
        )
        
        # Add rollout plans
        rollout_plans = [foundation_plan, agent_plan, mvp_plan]
        for plan in rollout_plans:
            self.rollout_plans[plan.plan_id] = plan
        
        self.logger.info(f"Created {len(rollout_plans)} rollout plans")
    
    def get_current_phase_status(self) -> Dict[str, Any]:
        """Get status of the current active phase"""
        if not self.current_phase or self.current_phase not in self.phases:
            return {"error": "No active phase"}
        
        phase = self.phases[self.current_phase]
        
        # Calculate progress based on features and milestones
        total_features = len(phase.included_features)
        completed_features = 0
        
        for feature_id in phase.included_features:
            if feature_id in self.mvp_manager.features:
                feature = self.mvp_manager.features[feature_id]
                if feature.flag_state in [FeatureFlag.ENABLED, FeatureFlag.TESTING]:
                    completed_features += 1
        
        feature_progress = completed_features / total_features if total_features > 0 else 0.0
        
        # Check milestone progress
        milestone_progress = 0.0
        if phase.milestones:
            completed_milestones = sum(
                1 for ms_id in phase.milestones 
                if ms_id in self.milestones and 
                   self.milestones[ms_id].status == PhaseStatus.COMPLETED
            )
            milestone_progress = completed_milestones / len(phase.milestones)
        
        # Overall progress
        overall_progress = (feature_progress + milestone_progress) / 2
        
        # Calculate time progress
        today = date.today()
        if today < phase.start_date:
            time_progress = 0.0
            days_remaining = (phase.start_date - today).days
            status = "planned"
        elif today > phase.end_date:
            time_progress = 1.0
            days_remaining = 0
            status = "overdue" if overall_progress < 1.0 else "completed"
        else:
            days_elapsed = (today - phase.start_date).days
            time_progress = days_elapsed / phase.duration_days
            days_remaining = (phase.end_date - today).days
            status = "active"
        
        return {
            "phase_id": phase.phase_id,
            "name": phase.name,
            "status": status,
            "overall_progress": overall_progress,
            "feature_progress": feature_progress,
            "milestone_progress": milestone_progress,
            "time_progress": time_progress,
            "days_remaining": days_remaining,
            "start_date": phase.start_date.isoformat(),
            "end_date": phase.end_date.isoformat(),
            "features": {
                "total": total_features,
                "completed": completed_features,
                "remaining": total_features - completed_features
            },
            "milestones": {
                "total": len(phase.milestones),
                "completed": sum(
                    1 for ms_id in phase.milestones 
                    if ms_id in self.milestones and 
                       self.milestones[ms_id].status == PhaseStatus.COMPLETED
                ),
                "details": [
                    {
                        "id": ms_id,
                        "name": self.milestones[ms_id].name if ms_id in self.milestones else "Unknown",
                        "status": self.milestones[ms_id].status.value if ms_id in self.milestones else "unknown"
                    }
                    for ms_id in phase.milestones
                ]
            },
            "kpis": phase.kpis
        }
    
    def get_delivery_roadmap(self) -> Dict[str, Any]:
        """Generate comprehensive delivery roadmap"""
        roadmap = {
            "project_timeline": {
                "start_date": self.project_start_date.isoformat(),
                "target_date": self.project_target_date.isoformat(),
                "current_phase": self.current_phase,
                "total_phases": len(self.phases)
            },
            "phases": [],
            "milestones": [],
            "critical_path": [],
            "risks": [],
            "dependencies": []
        }
        
        # Add phase information
        for phase_id, phase in sorted(self.phases.items(), key=lambda x: x[1].start_date):
            phase_info = {
                "phase_id": phase_id,
                "name": phase.name,
                "start_date": phase.start_date.isoformat(),
                "end_date": phase.end_date.isoformat(),
                "duration_days": phase.duration_days,
                "status": phase.status.value,
                "features": len(phase.included_features),
                "milestones": len(phase.milestones),
                "dependencies": phase.depends_on_phases,
                "rollout_strategy": phase.rollout_strategy.value
            }
            roadmap["phases"].append(phase_info)
        
        # Add milestone information
        for ms_id, milestone in sorted(self.milestones.items(), key=lambda x: x[1].planned_date):
            ms_info = {
                "milestone_id": ms_id,
                "name": milestone.name,
                "planned_date": milestone.planned_date.isoformat(),
                "actual_date": milestone.actual_date.isoformat() if milestone.actual_date else None,
                "status": milestone.status.value,
                "completion_percentage": milestone.completion_percentage,
                "required_features": len(milestone.required_features)
            }
            roadmap["milestones"].append(ms_info)
        
        # Identify critical path
        critical_phases = []
        for phase_id, phase in self.phases.items():
            if not phase.depends_on_phases:  # Start phases
                critical_phases.append({
                    "phase_id": phase_id,
                    "name": phase.name,
                    "duration": phase.duration_days,
                    "end_date": phase.end_date.isoformat()
                })
        
        roadmap["critical_path"] = critical_phases
        
        # Add high-level risks
        roadmap["risks"] = [
            {
                "category": "Timeline",
                "risk": "Phase dependencies may cause delays",
                "impact": "Medium",
                "mitigation": "Buffer time built into schedule, parallel development where possible"
            },
            {
                "category": "Technical",
                "risk": "Agent coordination complexity higher than expected",
                "impact": "High", 
                "mitigation": "Incremental development, extensive testing, fallback mechanisms"
            },
            {
                "category": "Quality",
                "risk": "Security vulnerabilities in autonomous agents",
                "impact": "High",
                "mitigation": "Security-first design, regular audits, sandboxed execution"
            }
        ]
        
        return roadmap
    
    def advance_to_next_phase(self) -> bool:
        """Advance to the next phase if current phase is complete"""
        if not self.current_phase:
            return False
        
        current = self.phases[self.current_phase]
        
        # Check if current phase is complete
        phase_status = self.get_current_phase_status()
        if phase_status["overall_progress"] < 1.0:
            self.logger.warning(f"Cannot advance from {self.current_phase}: only {phase_status['overall_progress']:.1%} complete")
            return False
        
        # Find next phase
        phases_by_start_date = sorted(
            [(k, v) for k, v in self.phases.items()],
            key=lambda x: x[1].start_date
        )
        
        current_index = next(
            (i for i, (k, v) in enumerate(phases_by_start_date) if k == self.current_phase),
            -1
        )
        
        if current_index >= 0 and current_index < len(phases_by_start_date) - 1:
            next_phase_id = phases_by_start_date[current_index + 1][0]
            
            # Mark current phase as completed
            current.status = PhaseStatus.COMPLETED
            current.actual_end_date = date.today()
            
            # Start next phase
            next_phase = self.phases[next_phase_id]
            next_phase.status = PhaseStatus.ACTIVE
            next_phase.actual_start_date = date.today()
            self.current_phase = next_phase_id
            
            self.logger.info(f"Advanced from {current.name} to {next_phase.name}")
            return True
        
        self.logger.info(f"Project complete - no more phases after {current.name}")
        return False
    
    def export_delivery_plan(self, output_path: Path) -> None:
        """Export comprehensive delivery plan"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate roadmap
            roadmap = self.get_delivery_roadmap()
            
            # Add detailed information
            detailed_plan = {
                **roadmap,
                "detailed_phases": {
                    phase_id: asdict(phase) for phase_id, phase in self.phases.items()
                },
                "detailed_milestones": {
                    ms_id: asdict(milestone) for ms_id, milestone in self.milestones.items()
                },
                "rollout_plans": {
                    plan_id: asdict(plan) for plan_id, plan in self.rollout_plans.items()
                },
                "current_status": self.get_current_phase_status(),
                "exported_at": datetime.now().isoformat()
            }
            
            # Export as JSON
            with open(output_path, 'w') as f:
                json.dump(detailed_plan, f, indent=2, default=str)
            
            self.logger.info(f"Delivery plan exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export delivery plan: {e}")
            raise


def main():
    """Main function for phased delivery demonstration"""
    try:
        print("Archangel Phased Delivery Management")
        print("=" * 50)
        
        # Initialize systems
        from .mvp_manager import MVPManager
        mvp_manager = MVPManager()
        mvp_manager.define_archangel_mvp()
        
        delivery_manager = PhasedDeliveryManager(mvp_manager)
        
        # Define delivery phases
        delivery_manager.define_archangel_delivery_phases()
        delivery_manager.create_milestones()
        delivery_manager.create_rollout_plans()
        
        print(f"Delivery phases defined: {len(delivery_manager.phases)}")
        print(f"Milestones created: {len(delivery_manager.milestones)}")
        print(f"Rollout plans: {len(delivery_manager.rollout_plans)}")
        
        # Show current phase status
        current_status = delivery_manager.get_current_phase_status()
        print(f"\nCurrent Phase: {current_status['name']}")
        print(f"Progress: {current_status['overall_progress']:.1%}")
        print(f"Time Progress: {current_status['time_progress']:.1%}")
        print(f"Days Remaining: {current_status['days_remaining']}")
        
        # Show roadmap overview
        roadmap = delivery_manager.get_delivery_roadmap()
        print(f"\nProject Roadmap:")
        print(f"Start: {roadmap['project_timeline']['start_date']}")
        print(f"Target: {roadmap['project_timeline']['target_date']}")
        print(f"Phases: {roadmap['project_timeline']['total_phases']}")
        
        print(f"\nPhases:")
        for phase in roadmap['phases'][:3]:  # Show first 3
            print(f"  - {phase['name']}: {phase['start_date']} to {phase['end_date']} ({phase['duration_days']} days)")
        
        print(f"\nKey Milestones:")
        for milestone in roadmap['milestones']:
            print(f"  - {milestone['name']}: {milestone['planned_date']} ({milestone['status']})")
        
        # Show rollout plan example
        foundation_plan = delivery_manager.rollout_plans['foundation_rollout']
        print(f"\nFoundation Rollout Plan:")
        print(f"Strategy: {foundation_plan.strategy.value}")
        print(f"Duration: {foundation_plan.total_duration.days} days")
        print(f"Steps: {len(foundation_plan.rollout_steps)}")
        
        for step in foundation_plan.rollout_steps:
            print(f"  Step {step['step']}: {step['description']} ({step['duration_days']} days)")
        
        # Export delivery plan
        output_path = Path("project_management") / "delivery_plan.json"
        delivery_manager.export_delivery_plan(output_path)
        print(f"\nDelivery plan exported to {output_path}")
        
        print("\nPhased delivery demonstration complete!")
        
    except Exception as e:
        logger.error(f"Phased delivery demo failed: {e}")
        raise


if __name__ == "__main__":
    main()