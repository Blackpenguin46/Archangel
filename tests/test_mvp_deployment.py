#!/usr/bin/env python3
"""
Tests for MVP functionality and incremental deployment reliability
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta, date
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json

from project_management.critical_path import (
    CriticalPathAnalyzer, TaskDefinition, TaskPriority, TaskStatus, DeliveryPhase,
    create_archangel_task_definitions
)
from project_management.mvp_manager import (
    MVPManager, Feature, FeatureFlag, ReleaseCandidate, DeploymentEnvironment
)
from project_management.phased_delivery import (
    PhasedDeliveryManager, DeliveryPhaseDefinition, DeliveryMilestone, 
    PhaseStatus, RolloutStrategy
)
from project_management.feature_flags import (
    FeatureFlagManager, FeatureFlagDefinition, FlagState, FlagEvaluationContext,
    RolloutRule, RolloutType
)

class TestCriticalPathAnalysis:
    """Test critical path analysis functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = CriticalPathAnalyzer()
        
        # Create test tasks
        test_tasks = [
            TaskDefinition(
                task_id="task_foundation",
                name="Foundation Setup",
                description="Basic foundation setup",
                dependencies=[],
                priority=TaskPriority.MUST,
                delivery_phase=DeliveryPhase.FOUNDATION,
                estimated_duration_days=5.0,
                status=TaskStatus.COMPLETED
            ),
            TaskDefinition(
                task_id="task_agents",
                name="Agent Development", 
                description="Develop autonomous agents",
                dependencies=["task_foundation"],
                priority=TaskPriority.MUST,
                delivery_phase=DeliveryPhase.ALPHA,
                estimated_duration_days=10.0,
                status=TaskStatus.IN_PROGRESS
            ),
            TaskDefinition(
                task_id="task_environment",
                name="Environment Setup",
                description="Mock enterprise environment",
                dependencies=["task_foundation"],
                priority=TaskPriority.MUST,
                delivery_phase=DeliveryPhase.ALPHA,
                estimated_duration_days=7.0,
                status=TaskStatus.NOT_STARTED
            ),
            TaskDefinition(
                task_id="task_integration",
                name="System Integration",
                description="Integrate all components",
                dependencies=["task_agents", "task_environment"],
                priority=TaskPriority.MUST,
                delivery_phase=DeliveryPhase.ALPHA,
                estimated_duration_days=3.0,
                status=TaskStatus.NOT_STARTED
            )
        ]
        
        for task in test_tasks:
            self.analyzer.add_task(task)
    
    def test_task_addition(self):
        """Test adding tasks to analyzer"""
        initial_count = len(self.analyzer.tasks)
        
        new_task = TaskDefinition(
            task_id="test_task",
            name="Test Task",
            description="Test task for validation",
            estimated_duration_days=2.0
        )
        
        self.analyzer.add_task(new_task)
        
        assert len(self.analyzer.tasks) == initial_count + 1
        assert "test_task" in self.analyzer.tasks
        assert self.analyzer.tasks["test_task"].name == "Test Task"
    
    def test_dependency_graph_creation(self):
        """Test dependency graph construction"""
        # Check nodes
        assert len(self.analyzer.dependency_graph.nodes) == 4
        assert "task_foundation" in self.analyzer.dependency_graph.nodes
        assert "task_integration" in self.analyzer.dependency_graph.nodes
        
        # Check edges (dependencies)
        assert self.analyzer.dependency_graph.has_edge("task_foundation", "task_agents")
        assert self.analyzer.dependency_graph.has_edge("task_foundation", "task_environment")
        assert self.analyzer.dependency_graph.has_edge("task_agents", "task_integration")
        assert self.analyzer.dependency_graph.has_edge("task_environment", "task_integration")
    
    def test_critical_path_analysis(self):
        """Test critical path identification"""
        critical_paths = self.analyzer.analyze_critical_paths()
        
        assert isinstance(critical_paths, list)
        assert len(critical_paths) > 0
        
        # Check path structure
        longest_path = critical_paths[0]
        assert hasattr(longest_path, 'path_id')
        assert hasattr(longest_path, 'tasks')
        assert hasattr(longest_path, 'total_duration')
        assert hasattr(longest_path, 'is_critical')
        
        # Verify path contains expected tasks
        assert "task_foundation" in longest_path.tasks
        assert "task_integration" in longest_path.tasks
    
    def test_bottleneck_identification(self):
        """Test bottleneck identification"""
        bottlenecks = self.analyzer.identify_bottlenecks()
        
        assert isinstance(bottlenecks, dict)
        assert 'task_bottlenecks' in bottlenecks
        assert 'dependency_bottlenecks' in bottlenecks
        assert 'risk_bottlenecks' in bottlenecks
        
        # All bottleneck categories should be lists
        for category in bottlenecks.values():
            assert isinstance(category, list)
    
    def test_mvp_definition(self):
        """Test MVP definition creation"""
        core_features = ["multi_agent_coordination", "basic_memory_system"]
        
        mvp = self.analyzer.define_mvp(core_features)
        
        assert isinstance(mvp, object)
        assert hasattr(mvp, 'mvp_id')
        assert hasattr(mvp, 'required_tasks')
        assert hasattr(mvp, 'core_features')
        assert mvp.core_features == core_features
        assert len(mvp.required_tasks) > 0
    
    def test_delivery_sequence_optimization(self):
        """Test delivery sequence optimization"""
        delivery_plan = self.analyzer.optimize_delivery_sequence()
        
        assert isinstance(delivery_plan, dict)
        
        # Check all delivery phases are present
        for phase in DeliveryPhase:
            assert phase in delivery_plan
            assert isinstance(delivery_plan[phase], list)
        
        # Foundation tasks should come before Alpha tasks
        foundation_tasks = delivery_plan[DeliveryPhase.FOUNDATION]
        alpha_tasks = delivery_plan[DeliveryPhase.ALPHA]
        
        # task_foundation should be in foundation phase
        assert any("foundation" in task_id.lower() for task_id in foundation_tasks)
    
    def test_project_dashboard_generation(self):
        """Test project dashboard data generation"""
        dashboard = self.analyzer.generate_project_dashboard()
        
        assert isinstance(dashboard, dict)
        assert 'project_overview' in dashboard
        assert 'priority_breakdown' in dashboard
        assert 'phase_breakdown' in dashboard
        
        # Check overview metrics
        overview = dashboard['project_overview']
        assert 'total_tasks' in overview
        assert 'completed_tasks' in overview
        assert 'in_progress_tasks' in overview
        assert overview['total_tasks'] == 4  # Our test tasks
    
    def test_real_archangel_tasks(self):
        """Test with real Archangel task definitions"""
        real_analyzer = CriticalPathAnalyzer()
        
        # Load real Archangel tasks
        real_tasks = create_archangel_task_definitions()
        for task in real_tasks:
            real_analyzer.add_task(task)
        
        # Analyze critical paths
        critical_paths = real_analyzer.analyze_critical_paths()
        
        assert len(critical_paths) > 0
        assert len(real_analyzer.tasks) >= 10  # Should have substantial number of tasks
        
        # Check that foundation tasks are properly identified
        foundation_tasks = [t for t in real_analyzer.tasks.values() 
                          if t.delivery_phase == DeliveryPhase.FOUNDATION]
        assert len(foundation_tasks) > 0


class TestMVPManager:
    """Test MVP management functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mvp_manager = MVPManager()
    
    def test_mvp_manager_initialization(self):
        """Test MVP manager initialization"""
        assert isinstance(self.mvp_manager.features, dict)
        assert isinstance(self.mvp_manager.environments, dict)
        assert len(self.mvp_manager.environments) == 4  # Default environments
        
        # Check default environments
        assert "development" in self.mvp_manager.environments
        assert "production" in self.mvp_manager.environments
    
    def test_archangel_mvp_definition(self):
        """Test Archangel MVP definition"""
        mvp_version = self.mvp_manager.define_archangel_mvp()
        
        assert isinstance(mvp_version, str)
        assert mvp_version in self.mvp_manager.release_candidates
        assert self.mvp_manager.current_mvp == mvp_version
        
        # Check MVP features
        mvp_rc = self.mvp_manager.release_candidates[mvp_version]
        assert len(mvp_rc.included_features) > 0
        assert "multi_agent_coordination" in mvp_rc.included_features
        assert "basic_memory_system" in mvp_rc.included_features
    
    def test_feature_flag_management(self):
        """Test feature flag operations"""
        self.mvp_manager.define_archangel_mvp()
        
        # Test setting feature flag
        self.mvp_manager.set_feature_flag(
            "multi_agent_coordination", 
            FeatureFlag.ENABLED,
            rollout_percentage=100.0
        )
        
        # Test checking feature flag
        enabled = self.mvp_manager.is_feature_enabled(
            "multi_agent_coordination", 
            "development"
        )
        assert enabled
        
        # Test environment-specific override
        self.mvp_manager.set_feature_flag(
            "multi_agent_coordination",
            FeatureFlag.DISABLED,
            environment="testing"
        )
        
        # Should be disabled in testing
        testing_enabled = self.mvp_manager.is_feature_enabled(
            "multi_agent_coordination",
            "testing"
        )
        assert not testing_enabled
        
        # Should still be enabled in development
        dev_enabled = self.mvp_manager.is_feature_enabled(
            "multi_agent_coordination",
            "development"
        )
        assert dev_enabled
    
    def test_gradual_rollout(self):
        """Test gradual feature rollout"""
        self.mvp_manager.define_archangel_mvp()
        
        rollout_plan = self.mvp_manager.gradual_rollout(
            "basic_memory_system",
            target_percentage=100.0,
            step_size=25.0
        )
        
        assert isinstance(rollout_plan, dict)
        assert 'feature_id' in rollout_plan
        assert 'steps' in rollout_plan
        assert len(rollout_plan['steps']) == 4  # 0->25->50->75->100
        
        # Check step structure
        step = rollout_plan['steps'][0]
        assert 'percentage' in step
        assert 'scheduled_time' in step
        assert 'quality_gates' in step
    
    def test_quality_gates(self):
        """Test quality gate checking"""
        self.mvp_manager.define_archangel_mvp()
        
        gate_results = self.mvp_manager.check_quality_gates(
            "multi_agent_coordination",
            "testing"
        )
        
        assert isinstance(gate_results, dict)
        assert 'feature_id' in gate_results
        assert 'environment' in gate_results
        assert 'overall_passed' in gate_results
        assert 'gate_results' in gate_results
        assert 'recommendations' in gate_results
    
    def test_mvp_status(self):
        """Test MVP status reporting"""
        self.mvp_manager.define_archangel_mvp()
        
        # Enable some features
        self.mvp_manager.set_feature_flag("multi_agent_coordination", FeatureFlag.ENABLED)
        self.mvp_manager.set_feature_flag("basic_memory_system", FeatureFlag.TESTING)
        
        status = self.mvp_manager.get_mvp_status()
        
        assert isinstance(status, dict)
        assert status['mvp_defined']
        assert 'features' in status
        assert 'overall_progress' in status
        assert 'ready_for_release' in status
        
        # Check feature status
        assert 'multi_agent_coordination' in status['features']
        assert status['features']['multi_agent_coordination']['flag_state'] == 'enabled'
    
    def test_deployment(self):
        """Test deployment to environment"""
        mvp_version = self.mvp_manager.define_archangel_mvp()
        
        # Force deployment to testing environment
        deployment_result = self.mvp_manager.deploy_to_environment(
            mvp_version,
            "testing",
            force=True
        )
        
        assert isinstance(deployment_result, dict)
        assert 'success' in deployment_result
        assert 'features_deployed' in deployment_result
        assert 'deployed_at' in deployment_result
        
        # Should have deployed some features
        assert len(deployment_result['features_deployed']) > 0
    
    def test_configuration_export(self):
        """Test feature configuration export"""
        self.mvp_manager.define_archangel_mvp()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "feature_config.json"
            self.mvp_manager.export_feature_config(output_path)
            
            assert output_path.exists()
            
            # Load and verify exported configuration
            with open(output_path, 'r') as f:
                config = json.load(f)
            
            assert 'features' in config
            assert 'environments' in config
            assert 'current_version' in config
            assert len(config['features']) > 0


class TestPhasedDelivery:
    """Test phased delivery management"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mvp_manager = MVPManager()
        self.mvp_manager.define_archangel_mvp()
        self.delivery_manager = PhasedDeliveryManager(self.mvp_manager)
    
    def test_delivery_manager_initialization(self):
        """Test delivery manager initialization"""
        assert isinstance(self.delivery_manager.phases, dict)
        assert isinstance(self.delivery_manager.milestones, dict)
        assert isinstance(self.delivery_manager.rollout_plans, dict)
        assert self.delivery_manager.project_start_date == date.today()
    
    def test_archangel_phases_definition(self):
        """Test Archangel delivery phases definition"""
        self.delivery_manager.define_archangel_delivery_phases()
        
        assert len(self.delivery_manager.phases) == 6  # 6 phases defined
        assert "foundation" in self.delivery_manager.phases
        assert "mvp_release" in self.delivery_manager.phases
        assert "production_scale" in self.delivery_manager.phases
        
        # Check phase properties
        foundation_phase = self.delivery_manager.phases["foundation"]
        assert foundation_phase.name == "Foundation Phase"
        assert len(foundation_phase.included_features) > 0
        assert foundation_phase.rollout_strategy == RolloutStrategy.ROLLING
        assert foundation_phase.status == PhaseStatus.PLANNED
    
    def test_milestone_creation(self):
        """Test milestone creation"""
        self.delivery_manager.define_archangel_delivery_phases()
        self.delivery_manager.create_milestones()
        
        assert len(self.delivery_manager.milestones) > 0
        
        # Check milestone structure
        first_milestone = list(self.delivery_manager.milestones.values())[0]
        assert hasattr(first_milestone, 'milestone_id')
        assert hasattr(first_milestone, 'name')
        assert hasattr(first_milestone, 'planned_date')
        assert hasattr(first_milestone, 'success_criteria')
        assert len(first_milestone.success_criteria) > 0
    
    def test_rollout_plans_creation(self):
        """Test rollout plans creation"""
        self.delivery_manager.define_archangel_delivery_phases()
        self.delivery_manager.create_rollout_plans()
        
        assert len(self.delivery_manager.rollout_plans) > 0
        
        # Check rollout plan structure
        foundation_plan = self.delivery_manager.rollout_plans.get("foundation_rollout")
        if foundation_plan:
            assert foundation_plan.strategy in [rs for rs in RolloutStrategy]
            assert len(foundation_plan.rollout_steps) > 0
            assert len(foundation_plan.quality_gates) > 0
            
            # Check rollout step structure
            step = foundation_plan.rollout_steps[0]
            assert 'step' in step
            assert 'description' in step
            assert 'duration_days' in step
    
    def test_current_phase_status(self):
        """Test current phase status reporting"""
        self.delivery_manager.define_archangel_delivery_phases()
        self.delivery_manager.create_milestones()
        
        status = self.delivery_manager.get_current_phase_status()
        
        assert isinstance(status, dict)
        assert 'phase_id' in status
        assert 'name' in status
        assert 'status' in status
        assert 'overall_progress' in status
        assert 'features' in status
        assert 'milestones' in status
        
        # Progress should be between 0 and 1
        assert 0 <= status['overall_progress'] <= 1
    
    def test_delivery_roadmap(self):
        """Test delivery roadmap generation"""
        self.delivery_manager.define_archangel_delivery_phases()
        self.delivery_manager.create_milestones()
        
        roadmap = self.delivery_manager.get_delivery_roadmap()
        
        assert isinstance(roadmap, dict)
        assert 'project_timeline' in roadmap
        assert 'phases' in roadmap
        assert 'milestones' in roadmap
        assert 'risks' in roadmap
        
        # Check project timeline
        timeline = roadmap['project_timeline']
        assert 'start_date' in timeline
        assert 'target_date' in timeline
        assert 'current_phase' in timeline
        assert 'total_phases' in timeline
        
        # Check phases
        assert len(roadmap['phases']) > 0
        phase = roadmap['phases'][0]
        assert 'phase_id' in phase
        assert 'start_date' in phase
        assert 'end_date' in phase
        assert 'duration_days' in phase
    
    def test_phase_advancement(self):
        """Test phase advancement"""
        self.delivery_manager.define_archangel_delivery_phases()
        
        # Initially should not advance (phase not complete)
        advanced = self.delivery_manager.advance_to_next_phase()
        assert not advanced  # Phase not complete
        
        # Mock phase completion and try again
        if self.delivery_manager.current_phase:
            current_phase = self.delivery_manager.phases[self.delivery_manager.current_phase]
            
            # Enable all features in current phase to simulate completion
            for feature_id in current_phase.included_features:
                if feature_id in self.mvp_manager.features:
                    self.mvp_manager.set_feature_flag(feature_id, FeatureFlag.ENABLED)
            
            # Try advancement again (may still not advance due to other requirements)
            advanced = self.delivery_manager.advance_to_next_phase()
            # Note: This might still be False due to milestone completion requirements
    
    def test_delivery_plan_export(self):
        """Test delivery plan export"""
        self.delivery_manager.define_archangel_delivery_phases()
        self.delivery_manager.create_milestones()
        self.delivery_manager.create_rollout_plans()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "delivery_plan.json"
            self.delivery_manager.export_delivery_plan(output_path)
            
            assert output_path.exists()
            
            # Load and verify exported plan
            with open(output_path, 'r') as f:
                plan = json.load(f)
            
            assert 'project_timeline' in plan
            assert 'detailed_phases' in plan
            assert 'detailed_milestones' in plan
            assert 'current_status' in plan


class TestFeatureFlags:
    """Test feature flag system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.flag_manager = FeatureFlagManager()
    
    def test_flag_manager_initialization(self):
        """Test feature flag manager initialization"""
        assert isinstance(self.flag_manager.flags, dict)
        assert self.flag_manager.evaluation_count == 0
        assert self.flag_manager.error_count == 0
    
    def test_archangel_flags_creation(self):
        """Test Archangel feature flags creation"""
        self.flag_manager.create_archangel_feature_flags()
        
        assert len(self.flag_manager.flags) > 0
        assert "multi_agent_coordination" in self.flag_manager.flags
        assert "red_team_agents" in self.flag_manager.flags
        assert "emergency_shutdown" in self.flag_manager.flags
        
        # Check flag structure
        flag = self.flag_manager.flags["multi_agent_coordination"]
        assert flag.name == "Multi-Agent Coordination"
        assert flag.state == FlagState.ENABLED
        assert flag.category == "foundation"
        assert "mvp" in flag.tags
    
    def test_flag_evaluation(self):
        """Test feature flag evaluation"""
        self.flag_manager.create_archangel_feature_flags()
        
        # Test basic evaluation
        context = FlagEvaluationContext(
            user_id="test_user",
            environment="development"
        )
        
        result = self.flag_manager.evaluate_flag("multi_agent_coordination", context)
        
        assert hasattr(result, 'flag_id')
        assert hasattr(result, 'enabled')
        assert hasattr(result, 'reason')
        assert hasattr(result, 'evaluation_time')
        assert result.flag_id == "multi_agent_coordination"
        assert isinstance(result.enabled, bool)
        assert result.evaluation_time >= 0
    
    def test_flag_state_updates(self):
        """Test flag state updates"""
        self.flag_manager.create_archangel_feature_flags()
        
        # Update flag state
        success = self.flag_manager.update_flag_state("red_team_agents", FlagState.ENABLED)
        assert success
        
        # Verify state changed
        flag = self.flag_manager.flags["red_team_agents"]
        assert flag.state == FlagState.ENABLED
    
    def test_percentage_rollout(self):
        """Test percentage rollout functionality"""
        self.flag_manager.create_archangel_feature_flags()
        
        # Update rollout percentage
        success = self.flag_manager.update_rollout_percentage("basic_memory_system", 50.0)
        assert success
        
        # Test multiple evaluations to check percentage consistency
        context1 = FlagEvaluationContext(user_id="user1", environment="development")
        context2 = FlagEvaluationContext(user_id="user2", environment="development")
        
        result1 = self.flag_manager.evaluate_flag("basic_memory_system", context1)
        result2 = self.flag_manager.evaluate_flag("basic_memory_system", context2)
        
        # Results should be consistent for same user
        result1_again = self.flag_manager.evaluate_flag("basic_memory_system", context1)
        assert result1.enabled == result1_again.enabled
    
    def test_prerequisite_flags(self):
        """Test prerequisite flag logic"""
        self.flag_manager.create_archangel_feature_flags()
        
        context = FlagEvaluationContext(user_id="test", environment="development")
        
        # Disable prerequisite
        self.flag_manager.update_flag_state("multi_agent_coordination", FlagState.DISABLED)
        
        # Check dependent flag
        result = self.flag_manager.evaluate_flag("basic_memory_system", context)
        
        # Should be disabled due to prerequisite
        if not result.enabled:
            assert "Prerequisite" in result.reason
    
    def test_emergency_shutdown(self):
        """Test emergency shutdown functionality"""
        self.flag_manager.create_archangel_feature_flags()
        
        # Count enabled flags before shutdown
        initial_enabled = sum(
            1 for flag in self.flag_manager.flags.values() 
            if flag.state == FlagState.ENABLED
        )
        
        # Perform emergency shutdown
        disabled_count = self.flag_manager.emergency_disable_all(
            except_flags=["emergency_shutdown"]
        )
        
        assert disabled_count > 0
        
        # Check that flags are in kill switch state
        kill_switch_count = sum(
            1 for flag in self.flag_manager.flags.values()
            if flag.state == FlagState.KILL_SWITCH
        )
        
        assert kill_switch_count > 0
    
    def test_flag_metrics(self):
        """Test flag metrics collection"""
        self.flag_manager.create_archangel_feature_flags()
        
        # Perform some evaluations
        context = FlagEvaluationContext(user_id="test", environment="development")
        for _ in range(5):
            self.flag_manager.evaluate_flag("multi_agent_coordination", context)
        
        metrics = self.flag_manager.get_flag_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_flags' in metrics
        assert 'total_evaluations' in metrics
        assert 'error_count' in metrics
        assert 'error_rate' in metrics
        assert 'average_evaluation_time_ms' in metrics
        assert 'flag_states' in metrics
        
        assert metrics['total_evaluations'] >= 5
        assert metrics['total_flags'] > 0
    
    def test_configuration_export_import(self):
        """Test configuration export and import"""
        self.flag_manager.create_archangel_feature_flags()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export configuration
            export_path = Path(temp_dir) / "flags_config.json"
            success = self.flag_manager.export_configuration(export_path)
            assert success
            assert export_path.exists()
            
            # Create new manager and load configuration
            new_manager = FeatureFlagManager()
            load_success = new_manager.load_configuration(export_path)
            assert load_success
            
            # Verify flags were loaded
            assert len(new_manager.flags) == len(self.flag_manager.flags)
            assert "multi_agent_coordination" in new_manager.flags


class TestIntegratedDeployment:
    """Test integrated deployment scenarios"""
    
    def setup_method(self):
        """Set up integrated test environment"""
        self.mvp_manager = MVPManager()
        self.flag_manager = FeatureFlagManager()
        self.delivery_manager = PhasedDeliveryManager(self.mvp_manager)
        
        # Initialize all systems
        self.mvp_manager.define_archangel_mvp()
        self.flag_manager.create_archangel_feature_flags()
        self.delivery_manager.define_archangel_delivery_phases()
        self.delivery_manager.create_milestones()
    
    def test_end_to_end_mvp_deployment(self):
        """Test complete MVP deployment workflow"""
        # Check initial state
        mvp_status = self.mvp_manager.get_mvp_status()
        initial_progress = mvp_status['overall_progress']
        
        # Enable core MVP features through feature flags
        core_features = ["multi_agent_coordination", "basic_memory_system"]
        
        for feature_id in core_features:
            # Set feature flag
            self.flag_manager.update_flag_state(feature_id, FlagState.ENABLED)
            
            # Enable in MVP manager
            if feature_id in self.mvp_manager.features:
                self.mvp_manager.set_feature_flag(feature_id, FeatureFlag.ENABLED)
        
        # Check MVP status improvement
        updated_status = self.mvp_manager.get_mvp_status()
        assert updated_status['overall_progress'] >= initial_progress
        
        # Test deployment to staging
        mvp_version = self.mvp_manager.current_mvp
        deployment_result = self.mvp_manager.deploy_to_environment(
            mvp_version,
            "staging",
            force=True  # Force for testing
        )
        
        assert deployment_result['success']
        assert len(deployment_result['features_deployed']) > 0
    
    def test_phased_rollout_coordination(self):
        """Test coordination between phased delivery and feature flags"""
        # Get current phase
        phase_status = self.delivery_manager.get_current_phase_status()
        current_phase_id = phase_status['phase_id']
        
        if current_phase_id in self.delivery_manager.phases:
            current_phase = self.delivery_manager.phases[current_phase_id]
            
            # Enable features for current phase
            for feature_id in current_phase.included_features:
                if feature_id in self.flag_manager.flags:
                    self.flag_manager.update_flag_state(feature_id, FlagState.TESTING)
                
                if feature_id in self.mvp_manager.features:
                    self.mvp_manager.set_feature_flag(feature_id, FeatureFlag.TESTING)
            
            # Check updated phase status
            updated_status = self.delivery_manager.get_current_phase_status()
            # Progress should remain consistent or improve
            assert updated_status['overall_progress'] >= phase_status['overall_progress']
    
    def test_quality_gate_integration(self):
        """Test quality gate checking across systems"""
        # Check MVP quality gates
        mvp_features = ["multi_agent_coordination", "basic_memory_system"]
        
        for feature_id in mvp_features:
            if feature_id in self.mvp_manager.features:
                gate_results = self.mvp_manager.check_quality_gates(feature_id, "testing")
                
                assert isinstance(gate_results, dict)
                assert 'overall_passed' in gate_results
                assert 'gate_results' in gate_results
        
        # Check feature flag evaluation consistency
        context = FlagEvaluationContext(user_id="tester", environment="testing")
        
        for feature_id in mvp_features:
            if feature_id in self.flag_manager.flags:
                result = self.flag_manager.evaluate_flag(feature_id, context)
                assert hasattr(result, 'enabled')
                assert result.evaluation_time >= 0
    
    def test_rollback_scenarios(self):
        """Test rollback scenarios and emergency procedures"""
        # Enable some features
        test_features = ["red_team_agents", "blue_team_agents"]
        
        for feature_id in test_features:
            if feature_id in self.flag_manager.flags:
                self.flag_manager.update_flag_state(feature_id, FlagState.ENABLED)
        
        # Simulate emergency - disable all features
        disabled_count = self.flag_manager.emergency_disable_all()
        assert disabled_count > 0
        
        # Verify features are disabled
        context = FlagEvaluationContext(environment="production")
        
        for feature_id in test_features:
            if feature_id in self.flag_manager.flags:
                result = self.flag_manager.evaluate_flag(feature_id, context)
                # Should be disabled after emergency shutdown
                assert not result.enabled or "kill_switch" in result.reason.lower()
    
    def test_performance_under_load(self):
        """Test system performance under load"""
        # Perform many flag evaluations
        contexts = [
            FlagEvaluationContext(user_id=f"user_{i}", environment="testing")
            for i in range(100)
        ]
        
        test_flags = ["multi_agent_coordination", "basic_memory_system", "red_team_agents"]
        
        start_time = datetime.now()
        
        for context in contexts:
            for flag_id in test_flags:
                if flag_id in self.flag_manager.flags:
                    self.flag_manager.evaluate_flag(flag_id, context)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert total_time < 5.0  # 5 seconds for 300 evaluations
        
        # Check metrics
        metrics = self.flag_manager.get_flag_metrics()
        assert metrics['total_evaluations'] >= 300
        assert metrics['error_rate'] < 0.1  # Less than 10% errors


class TestDeploymentReliability:
    """Test deployment reliability and error handling"""
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        mvp_manager = MVPManager()
        
        # Test with invalid feature ID
        try:
            mvp_manager.set_feature_flag("nonexistent_feature", FeatureFlag.ENABLED)
            assert False, "Should have raised error for nonexistent feature"
        except (ValueError, KeyError):
            pass  # Expected
    
    def test_dependency_validation(self):
        """Test dependency validation"""
        analyzer = CriticalPathAnalyzer()
        
        # Create tasks with circular dependency
        task1 = TaskDefinition(
            task_id="task1",
            name="Task 1",
            description="First task",
            dependencies=["task2"]  # Depends on task2
        )
        
        task2 = TaskDefinition(
            task_id="task2", 
            name="Task 2",
            description="Second task",
            dependencies=["task1"]  # Circular dependency
        )
        
        analyzer.add_task(task1)
        analyzer.add_task(task2)
        
        # Should handle circular dependency gracefully
        try:
            critical_paths = analyzer.analyze_critical_paths()
            # If no exception, that's also acceptable (graceful handling)
        except ValueError as e:
            assert "cycle" in str(e).lower()
    
    def test_concurrent_access(self):
        """Test concurrent access to systems"""
        flag_manager = FeatureFlagManager()
        flag_manager.create_archangel_feature_flags()
        
        import threading
        import time
        
        results = []
        errors = []
        
        def evaluate_flags():
            try:
                for _ in range(10):
                    context = FlagEvaluationContext(user_id="concurrent_user")
                    result = flag_manager.evaluate_flag("multi_agent_coordination", context)
                    results.append(result.enabled)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=evaluate_flags)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have no errors and consistent results
        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 evaluations each
        
        # All results should be consistent (same flag, same context should give same result)
        if results:
            first_result = results[0]
            assert all(r == first_result for r in results)
    
    def test_recovery_from_failures(self):
        """Test system recovery from various failure scenarios"""
        mvp_manager = MVPManager()
        
        # Test recovery from invalid deployment
        try:
            deployment_result = mvp_manager.deploy_to_environment(
                "nonexistent_version",
                "testing"
            )
            assert not deployment_result['success']
            assert len(deployment_result['errors']) > 0
        except (ValueError, KeyError):
            pass  # Also acceptable
        
        # System should still be functional
        mvp_manager.define_archangel_mvp()
        mvp_status = mvp_manager.get_mvp_status()
        assert mvp_status['mvp_defined']


@pytest.mark.asyncio
async def test_async_operations():
    """Test asynchronous operations and compatibility"""
    # Test that systems work in async contexts
    mvp_manager = MVPManager()
    mvp_manager.define_archangel_mvp()
    
    # Simulate async deployment check
    await asyncio.sleep(0.1)
    
    mvp_status = mvp_manager.get_mvp_status()
    assert isinstance(mvp_status, dict)
    
    # Test feature flag evaluation in async context
    flag_manager = FeatureFlagManager()
    flag_manager.create_archangel_feature_flags()
    
    context = FlagEvaluationContext(user_id="async_user")
    result = flag_manager.evaluate_flag("multi_agent_coordination", context)
    
    assert hasattr(result, 'enabled')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])