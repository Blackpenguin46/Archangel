#!/usr/bin/env python3
"""
Tests for deception effectiveness and counter-intelligence accuracy
Comprehensive test suite for all deception and counter-intelligence components
"""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import random
import numpy as np
from dataclasses import asdict

# Import deception system components
from agents.deception_engine import (
    AdaptiveDeceptionEngine, DeceptionStrategy, DeceptionType, AttackerProfile,
    AttackerBehaviorPattern, DeceptionDeployment, DeceptionEffectiveness
)

from agents.counter_intelligence import (
    CounterIntelligenceOperations, IntelligenceType, DisinformationStrategy,
    CredibilityLevel, FalseIntelligenceAsset, DisinformationCampaign,
    CounterIntelTarget
)

from agents.deception_optimizer import (
    DeceptionEffectivenessOptimizer, EffectivenessMetrics, OptimizationParameters,
    OptimizationExperiment, OptimizationMetric, EffectivenessIndicator
)

from agents.honeypot_orchestrator import (
    AdvancedHoneypotOrchestrator, HoneypotConfiguration, HoneypotType,
    HoneypotProfile, OrchestrationPlan, DeploymentStrategy, HoneypotInstance
)


class TestAdaptiveDeceptionEngine:
    """Test adaptive deception engine functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = AdaptiveDeceptionEngine()
    
    def test_engine_initialization(self):
        """Test deception engine initialization"""
        assert isinstance(self.engine.strategies, dict)
        assert len(self.engine.strategies) > 0
        assert isinstance(self.engine.attacker_patterns, dict)
        assert isinstance(self.engine.active_deployments, dict)
    
    @pytest.mark.asyncio
    async def test_attacker_behavior_analysis(self):
        """Test attacker behavior pattern analysis"""
        interaction_data = {
            'attacker_id': 'test_attacker_001',
            'attack_vector': 'web_scan',
            'tool_signature': 'nmap',
            'timestamp': datetime.now(),
            'source_ip': '192.168.1.100',
            'evasion_technique': 'basic_obfuscation'
        }
        
        pattern = await self.engine.analyze_attacker_behavior(interaction_data)
        
        assert isinstance(pattern, AttackerBehaviorPattern)
        assert pattern.pattern_id == 'test_attacker_001'
        assert 'web_scan' in pattern.attack_vectors
        assert 'nmap' in pattern.tool_signatures
        assert 'basic_obfuscation' in pattern.evasion_techniques
        assert pattern.observation_count == 1
    
    @pytest.mark.asyncio
    async def test_strategy_selection(self):
        """Test adaptive strategy selection"""
        # Create test attacker pattern
        pattern = AttackerBehaviorPattern(
            pattern_id="test_pattern",
            attacker_profile=AttackerProfile.RECONNAISSANCE,
            attack_vectors=["web_scan", "port_scan"],
            tool_signatures=["nmap", "masscan"]
        )
        
        context = {'available_resources': {'cpu': 2.0, 'memory': '4GB'}}
        
        strategy = await self.engine.select_adaptive_strategy(pattern, context)
        
        assert strategy is not None
        assert isinstance(strategy, DeceptionStrategy)
        assert AttackerProfile.RECONNAISSANCE in strategy.target_profiles
    
    @pytest.mark.asyncio
    async def test_strategy_deployment(self):
        """Test strategy deployment"""
        # Get a strategy
        strategy = list(self.engine.strategies.values())[0]
        
        # Create test attacker pattern
        pattern = AttackerBehaviorPattern(
            pattern_id="deploy_test",
            attacker_profile=AttackerProfile.AUTOMATED_SCAN
        )
        
        deployment = await self.engine.deploy_strategy(
            strategy, [pattern], "test_environment"
        )
        
        assert isinstance(deployment, DeceptionDeployment)
        assert deployment.strategy_id == strategy.strategy_id
        assert "deploy_test" in deployment.target_attacker_patterns
        assert deployment.is_active
        assert deployment.deployment_id in self.engine.active_deployments
    
    @pytest.mark.asyncio
    async def test_strategy_adaptation(self):
        """Test strategy adaptation based on interaction"""
        # Deploy a strategy first
        strategy = list(self.engine.strategies.values())[0]
        pattern = AttackerBehaviorPattern(
            pattern_id="adapt_test",
            attacker_profile=AttackerProfile.SCRIPT_KIDDIE
        )
        
        deployment = await self.engine.deploy_strategy(strategy, [pattern])
        
        # Test adaptation
        interaction_data = {
            'attacker_id': 'adapt_test',
            'evasion_detected': True,
            'sophistication_increase': False,
            'adaptation_trigger': 'evasion_detected',
            'timestamp': datetime.now()
        }
        
        adapted = await self.engine.adapt_strategy(deployment, interaction_data)
        
        assert isinstance(adapted, bool)
        assert len(deployment.interactions) > 0
        if adapted:
            assert len(deployment.adaptation_history) > 0
    
    @pytest.mark.asyncio
    async def test_effectiveness_measurement(self):
        """Test effectiveness measurement"""
        # Deploy a strategy
        strategy = list(self.engine.strategies.values())[0]
        pattern = AttackerBehaviorPattern(pattern_id="effectiveness_test")
        deployment = await self.engine.deploy_strategy(strategy, [pattern])
        
        # Add some test interactions
        for i in range(5):
            interaction_data = {
                'attacker_id': f'attacker_{i}',
                'attacker_engaged': random.choice([True, False]),
                'detection_delay_seconds': random.randint(60, 600),
                'deception_successful': random.choice([True, False])
            }
            deployment.interactions.append({
                **interaction_data,
                'timestamp': datetime.now()
            })
        
        effectiveness = await self.engine.measure_effectiveness(deployment.deployment_id)
        
        assert isinstance(effectiveness, DeceptionEffectiveness)
        assert effectiveness.value >= 1 and effectiveness.value <= 5
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection"""
        metrics = await self.engine.get_deception_metrics()
        
        assert isinstance(metrics, dict)
        assert 'system_overview' in metrics
        assert 'effectiveness' in metrics
        assert 'strategy_performance' in metrics
        assert 'attacker_analysis' in metrics
        
        # Check structure
        assert 'total_strategies' in metrics['system_overview']
        assert isinstance(metrics['system_overview']['total_strategies'], int)
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_deployments(self):
        """Test cleanup of expired deployments"""
        # Create an expired deployment
        strategy = list(self.engine.strategies.values())[0]
        pattern = AttackerBehaviorPattern(pattern_id="expired_test")
        deployment = await self.engine.deploy_strategy(strategy, [pattern])
        
        # Manually set expiration to past
        deployment.expires_at = datetime.now() - timedelta(hours=1)
        
        cleaned_count = await self.engine.cleanup_expired_deployments()
        
        assert isinstance(cleaned_count, int)
        assert deployment.deployment_id not in self.engine.active_deployments
    
    @pytest.mark.asyncio
    async def test_intelligence_export(self):
        """Test intelligence export functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_intelligence.json"
            
            await self.engine.export_deception_intelligence(output_path)
            
            assert output_path.exists()
            
            # Validate exported data
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert 'export_metadata' in data
            assert 'attacker_patterns' in data
            assert 'deployment_intelligence' in data
            assert 'strategy_effectiveness' in data
    
    def test_strategy_effectiveness_calculation(self):
        """Test strategy effectiveness calculation"""
        strategy = list(self.engine.strategies.values())[0]
        
        # Test outcomes
        outcomes = [
            {'engaged': True, 'detection_delay': 300, 'successful': True},
            {'engaged': True, 'detection_delay': 150, 'successful': False},
            {'engaged': False, 'detection_delay': 0, 'successful': False},
            {'engaged': True, 'detection_delay': 600, 'successful': True}
        ]
        
        effectiveness = strategy.calculate_effectiveness(outcomes)
        
        assert isinstance(effectiveness, DeceptionEffectiveness)
        assert effectiveness.value >= 1 and effectiveness.value <= 5


class TestCounterIntelligenceOperations:
    """Test counter-intelligence operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.counter_intel = CounterIntelligenceOperations()
    
    def test_system_initialization(self):
        """Test counter-intelligence system initialization"""
        assert isinstance(self.counter_intel.intelligence_assets, dict)
        assert isinstance(self.counter_intel.active_campaigns, dict)
        assert isinstance(self.counter_intel.attacker_profiles, dict)
        assert len(self.counter_intel.intelligence_templates) > 0
    
    @pytest.mark.asyncio
    async def test_false_intelligence_generation(self):
        """Test false intelligence generation"""
        asset = await self.counter_intel.generate_false_intelligence(
            IntelligenceType.NETWORK_TOPOLOGY,
            CredibilityLevel.HIGH,
            "test_target",
            {"organization": "TestCorp"}
        )
        
        assert isinstance(asset, FalseIntelligenceAsset)
        assert asset.intelligence_type == IntelligenceType.NETWORK_TOPOLOGY
        assert asset.credibility_level == CredibilityLevel.HIGH
        assert "test_target" in asset.target_profiles
        assert isinstance(asset.content, dict)
        assert len(asset.believability_factors) > 0
        assert asset.asset_id in self.counter_intel.intelligence_assets
    
    @pytest.mark.asyncio
    async def test_network_topology_generation(self):
        """Test network topology generation"""
        asset = await self.counter_intel.generate_false_intelligence(
            IntelligenceType.NETWORK_TOPOLOGY,
            CredibilityLevel.MEDIUM,
            "network_test"
        )
        
        content = asset.content
        assert 'subnets' in content
        assert 'servers' in content
        assert isinstance(content['subnets'], list)
        assert isinstance(content['servers'], list)
        assert len(content['subnets']) > 0
        assert len(content['servers']) > 0
        
        # Check subnet structure
        subnet = content['subnets'][0]
        assert 'subnet' in subnet
        assert 'vlan_id' in subnet
        assert 'description' in subnet
    
    @pytest.mark.asyncio
    async def test_credential_database_generation(self):
        """Test credential database generation"""
        asset = await self.counter_intel.generate_false_intelligence(
            IntelligenceType.USER_CREDENTIALS,
            CredibilityLevel.HIGH,
            "creds_test"
        )
        
        content = asset.content
        assert 'users' in content
        assert 'groups' in content
        assert isinstance(content['users'], list)
        assert len(content['users']) > 0
        
        # Check user structure
        user = content['users'][0]
        assert 'username' in user
        assert 'account_type' in user
    
    @pytest.mark.asyncio
    async def test_disinformation_campaign_creation(self):
        """Test disinformation campaign creation"""
        campaign = await self.counter_intel.create_disinformation_campaign(
            "Test Campaign",
            DisinformationStrategy.PLANTED_DOCUMENTS,
            [CounterIntelTarget.RECONNAISSANCE_AGENTS],
            "Test narrative for deception",
            14
        )
        
        assert isinstance(campaign, DisinformationCampaign)
        assert campaign.name == "Test Campaign"
        assert campaign.strategy == DisinformationStrategy.PLANTED_DOCUMENTS
        assert CounterIntelTarget.RECONNAISSANCE_AGENTS in campaign.target_operations
        assert campaign.primary_narrative == "Test narrative for deception"
        assert len(campaign.supporting_assets) > 0
        assert len(campaign.deployment_schedule) > 0
        assert campaign.campaign_id in self.counter_intel.active_campaigns
    
    @pytest.mark.asyncio
    async def test_campaign_launch(self):
        """Test campaign launch"""
        # Create campaign
        campaign = await self.counter_intel.create_disinformation_campaign(
            "Launch Test",
            DisinformationStrategy.FALSE_COMMUNICATIONS,
            [CounterIntelTarget.LATERAL_MOVEMENT],
            "Launch test narrative"
        )
        
        # Launch campaign
        success = await self.counter_intel.launch_campaign(campaign.campaign_id)
        
        assert isinstance(success, bool)
        if success:
            campaign = self.counter_intel.active_campaigns[campaign.campaign_id]
            assert campaign.status == "active"
            assert campaign.launched_at is not None
    
    @pytest.mark.asyncio
    async def test_intelligence_consumption_tracking(self):
        """Test intelligence consumption tracking"""
        # Generate an asset
        asset = await self.counter_intel.generate_false_intelligence(
            IntelligenceType.SYSTEM_CONFIGURATION,
            CredibilityLevel.MEDIUM,
            "consumption_test"
        )
        
        # Track consumption
        await self.counter_intel.track_intelligence_consumption(
            asset.asset_id,
            "test_accessor",
            {
                "access_method": "file_share",
                "time_spent_seconds": 120,
                "data_extracted": True,
                "verification_attempts": 1
            }
        )
        
        # Check tracking results
        updated_asset = self.counter_intel.intelligence_assets[asset.asset_id]
        assert updated_asset.access_count == 1
        assert "test_accessor" in updated_asset.unique_accessors
        assert len(updated_asset.consumption_patterns) == 1
        assert updated_asset.last_accessed is not None
    
    @pytest.mark.asyncio
    async def test_attacker_profile_creation(self):
        """Test attacker deception profile creation and updating"""
        # Generate asset and track consumption
        asset = await self.counter_intel.generate_false_intelligence(
            IntelligenceType.VULNERABILITY_DATA,
            CredibilityLevel.LOW,
            "profile_test"
        )
        
        # Track multiple consumptions to build profile
        for i in range(3):
            await self.counter_intel.track_intelligence_consumption(
                asset.asset_id,
                "profile_attacker",
                {
                    "access_method": "web",
                    "time_spent_seconds": 60 + i * 30,
                    "data_extracted": i < 2,  # Extract first two times
                    "verification_attempts": i  # Increasing verification attempts
                }
            )
        
        # Check profile creation
        assert "profile_attacker" in self.counter_intel.attacker_profiles
        profile = self.counter_intel.attacker_profiles["profile_attacker"]
        
        assert profile.attacker_identifier == "profile_attacker"
        assert len(profile.successful_deceptions) > 0
        assert len(profile.information_consumption_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_campaign_adaptation(self):
        """Test campaign adaptation based on feedback"""
        # Create and launch campaign
        campaign = await self.counter_intel.create_disinformation_campaign(
            "Adaptation Test",
            DisinformationStrategy.HONEYPOT_INTELLIGENCE,
            [CounterIntelTarget.DATA_EXFILTRATION],
            "Adaptation test narrative"
        )
        
        await self.counter_intel.launch_campaign(campaign.campaign_id)
        
        # Test adaptation
        feedback_data = {
            "skepticism_detected": True,
            "verification_increase": True,
            "consumption_decline": False,
            "pattern_recognition": False
        }
        
        adapted = await self.counter_intel.adapt_campaign_based_on_feedback(
            campaign.campaign_id, feedback_data
        )
        
        assert isinstance(adapted, bool)
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test counter-intelligence metrics collection"""
        metrics = await self.counter_intel.get_counter_intelligence_metrics()
        
        assert isinstance(metrics, dict)
        assert 'system_overview' in metrics
        assert 'deception_effectiveness' in metrics
        assert 'intelligence_analysis' in metrics
        assert 'attacker_profiling' in metrics
        assert 'campaign_performance' in metrics
        
        # Check metric values
        overview = metrics['system_overview']
        assert 'total_intelligence_assets' in overview
        assert 'active_campaigns' in overview
        assert isinstance(overview['total_intelligence_assets'], int)
    
    @pytest.mark.asyncio
    async def test_report_export(self):
        """Test counter-intelligence report export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "counter_intel_report.json"
            
            await self.counter_intel.export_counter_intelligence_report(output_path)
            
            assert output_path.exists()
            
            # Validate exported data
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert 'report_metadata' in data
            assert 'executive_summary' in data
            assert 'asset_intelligence' in data
            assert 'campaign_analysis' in data
            assert 'attacker_profiling' in data
            assert 'recommendations' in data


class TestDeceptionEffectivenessOptimizer:
    """Test deception effectiveness optimizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = DeceptionEffectivenessOptimizer()
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        assert isinstance(self.optimizer.effectiveness_history, deque)
        assert isinstance(self.optimizer.optimization_experiments, dict)
        assert isinstance(self.optimizer.parameter_sets, dict)
        assert hasattr(self.optimizer, 'effectiveness_predictor')
        assert hasattr(self.optimizer, 'parameter_optimizer')
    
    @pytest.mark.asyncio
    async def test_effectiveness_measurement(self):
        """Test effectiveness measurement"""
        # Create sample interaction data
        interaction_data = [
            {
                "attacker_id": f"attacker_{i}",
                "timestamp": datetime.now() - timedelta(minutes=i*10),
                "engaged": random.choice([True, False]),
                "duration_seconds": random.randint(30, 600),
                "information_extracted": random.choice([True, False]),
                "verification_attempts": random.randint(0, 3),
                "believability_assessment": random.uniform(0.3, 0.9),
                "confusion_indicators": {"uncertainty_score": random.uniform(0.0, 1.0)},
                "misdirection_success": random.choice([True, False]),
                "behavior_changes": [f"change_{random.randint(1, 5)}"]
            }
            for i in range(20)
        ]
        
        metrics = await self.optimizer.measure_deception_effectiveness(
            "test_strategy",
            "test_deployment", 
            interaction_data,
            timedelta(hours=2)
        )
        
        assert isinstance(metrics, EffectivenessMetrics)
        assert metrics.strategy_id == "test_strategy"
        assert metrics.deployment_id == "test_deployment"
        assert 0.0 <= metrics.engagement_rate <= 1.0
        assert 0.0 <= metrics.believability_rating <= 1.0
        assert 0.0 <= metrics.resource_cost_score <= 1.0
        assert metrics.measurement_id in [m.measurement_id for m in self.optimizer.effectiveness_history]
    
    @pytest.mark.asyncio
    async def test_parameter_optimization(self):
        """Test parameter optimization"""
        # Create base parameters
        base_params = OptimizationParameters(
            parameter_set_id="test_base",
            strategy_id="test_strategy",
            credibility_level=0.5,
            complexity_score=0.5,
            visibility_level=0.5,
            verification_difficulty=0.5
        )
        
        # Create effectiveness history
        effectiveness_history = []
        for i in range(5):
            metrics = EffectivenessMetrics(
                measurement_id=f"test_metrics_{i}",
                strategy_id="test_strategy",
                deployment_id=f"deployment_{i}",
                engagement_rate=random.uniform(0.2, 0.8),
                believability_rating=random.uniform(0.4, 0.9),
                resource_cost_score=random.uniform(0.1, 0.7)
            )
            effectiveness_history.append(metrics)
        
        # Optimize parameters
        optimized_params = await self.optimizer.optimize_parameters(
            "test_strategy",
            base_params,
            effectiveness_history
        )
        
        assert isinstance(optimized_params, OptimizationParameters)
        assert optimized_params.strategy_id == "test_strategy"
        assert optimized_params.optimization_generation == base_params.optimization_generation + 1
        assert base_params.parameter_set_id in optimized_params.parent_parameter_sets
    
    @pytest.mark.asyncio
    async def test_optimization_experiment(self):
        """Test optimization experiment creation and execution"""
        # Create parameters
        control_params = OptimizationParameters(
            parameter_set_id="control_test",
            strategy_id="test_strategy",
            credibility_level=0.5,
            visibility_level=0.5
        )
        
        test_params = [
            OptimizationParameters(
                parameter_set_id="test_high_cred",
                strategy_id="test_strategy", 
                credibility_level=0.8,
                visibility_level=0.5
            ),
            OptimizationParameters(
                parameter_set_id="test_high_vis",
                strategy_id="test_strategy",
                credibility_level=0.5,
                visibility_level=0.8
            )
        ]
        
        # Create experiment
        experiment = await self.optimizer.create_optimization_experiment(
            "Credibility vs Visibility Test",
            "test_strategy",
            control_params,
            test_params,
            [OptimizationMetric.ENGAGEMENT_RATE, OptimizationMetric.BELIEVABILITY_SCORE]
        )
        
        assert isinstance(experiment, OptimizationExperiment)
        assert experiment.name == "Credibility vs Visibility Test"
        assert len(experiment.test_parameters) == 2
        assert experiment.experiment_id in self.optimizer.optimization_experiments
        
        # Run experiment
        success = await self.optimizer.run_optimization_experiment(experiment.experiment_id)
        
        assert isinstance(success, bool)
        if success:
            updated_experiment = self.optimizer.optimization_experiments[experiment.experiment_id]
            assert updated_experiment.status == "completed"
            assert updated_experiment.control_results is not None
            assert len(updated_experiment.test_results) > 0
            assert updated_experiment.winning_parameters is not None
    
    @pytest.mark.asyncio
    async def test_optimization_insights(self):
        """Test optimization insights generation"""
        # Add some effectiveness history
        for i in range(10):
            metrics = EffectivenessMetrics(
                measurement_id=f"insight_test_{i}",
                strategy_id="insight_strategy",
                deployment_id=f"deployment_{i}",
                engagement_rate=0.4 + i * 0.05,  # Improving trend
                believability_rating=0.6 + random.uniform(-0.1, 0.1),
                resource_cost_score=0.5 + random.uniform(-0.1, 0.1),
                measurement_timestamp=datetime.now() - timedelta(hours=i)
            )
            self.optimizer.effectiveness_history.append(metrics)
        
        # Get insights
        insights = await self.optimizer.get_optimization_insights("insight_strategy")
        
        assert isinstance(insights, dict)
        assert 'strategy_id' in insights
        assert 'effectiveness_summary' in insights
        assert 'trend_analysis' in insights
        assert 'optimization_opportunities' in insights
        assert 'recommendations' in insights
        
        # Check summary
        summary = insights['effectiveness_summary']
        assert 'measurements_count' in summary
        assert 'average_engagement_rate' in summary
        assert 'trend_direction' in summary
        assert summary['measurements_count'] == 10
    
    @pytest.mark.asyncio
    async def test_report_export(self):
        """Test optimization report export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "optimization_report.json"
            
            await self.optimizer.export_optimization_report(output_path)
            
            assert output_path.exists()
            
            # Validate exported data
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert 'report_metadata' in data
            assert 'system_overview' in data
            assert 'effectiveness_analysis' in data
            assert 'experiment_results' in data
            assert 'optimization_insights' in data


class TestHoneypotOrchestrator:
    """Test honeypot orchestrator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.orchestrator = AdvancedHoneypotOrchestrator()
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        assert isinstance(self.orchestrator.honeypot_instances, dict)
        assert isinstance(self.orchestrator.orchestration_plans, dict)
        assert isinstance(self.orchestrator.deployment_templates, dict)
        assert len(self.orchestrator.deployment_templates) > 0
    
    @pytest.mark.asyncio
    async def test_honeypot_configuration_creation(self):
        """Test honeypot configuration creation"""
        config = await self.orchestrator.create_honeypot_configuration(
            "Test Web Server",
            "web_server_standard",
            {"interaction_complexity": 0.8}
        )
        
        assert isinstance(config, HoneypotConfiguration)
        assert config.name == "Test Web Server"
        assert config.honeypot_type == HoneypotType.WEB_SERVER
        assert len(config.services) > 0
        assert len(config.credentials) > 0
        assert config.interaction_complexity == 0.8
    
    @pytest.mark.asyncio
    async def test_honeypot_deployment(self):
        """Test individual honeypot deployment"""
        # Create configuration
        config = await self.orchestrator.create_honeypot_configuration(
            "Deploy Test",
            "ssh_server_secure"
        )
        
        # Deploy honeypot
        instance = await self.orchestrator.deploy_honeypot(config)
        
        assert isinstance(instance, HoneypotInstance)
        assert instance.configuration == config
        assert instance.status == "running"
        assert instance.deployed_at is not None
        assert instance.actual_ip is not None
        assert instance.instance_id in self.orchestrator.honeypot_instances
    
    @pytest.mark.asyncio
    async def test_orchestration_plan_creation(self):
        """Test orchestration plan creation"""
        plan = await self.orchestrator.create_orchestration_plan(
            "Enterprise Test",
            "enterprise_network",
            4
        )
        
        assert isinstance(plan, OrchestrationPlan)
        assert plan.name == "Enterprise Test"
        assert len(plan.honeypot_configurations) == 4
        assert len(plan.network_topology) > 0
        assert len(plan.interaction_flows) >= 0
        assert plan.plan_id in self.orchestrator.orchestration_plans
    
    @pytest.mark.asyncio
    async def test_plan_deployment(self):
        """Test orchestration plan deployment"""
        # Create plan
        plan = await self.orchestrator.create_orchestration_plan(
            "Deploy Test Plan",
            "web_infrastructure",
            3
        )
        
        # Deploy plan
        result = await self.orchestrator.deploy_orchestration_plan(plan.plan_id)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'honeypot_deployments' in result
        assert 'network_setup' in result
        assert len(result['honeypot_deployments']) == 3
        
        # Check individual deployments
        for deployment_info in result['honeypot_deployments'].values():
            assert 'status' in deployment_info
            if deployment_info['status'] == 'success':
                assert 'instance_id' in deployment_info
                assert 'ip_address' in deployment_info
    
    @pytest.mark.asyncio
    async def test_behavioral_adaptation(self):
        """Test honeypot behavioral adaptation"""
        # Deploy a honeypot
        config = await self.orchestrator.create_honeypot_configuration(
            "Adaptation Test",
            "iot_device_camera"
        )
        instance = await self.orchestrator.deploy_honeypot(config)
        
        # Test adaptation
        interaction_data = {
            "attacker_id": "adaptive_attacker",
            "advanced_technique": True,
            "persistence_attempt": True,
            "tools_used": ["custom_exploit"],
            "duration": 1200
        }
        
        from agents.honeypot_orchestrator import ThreatLevel
        adapted = await self.orchestrator.adapt_honeypot_behavior(
            instance.instance_id,
            interaction_data,
            ThreatLevel.HIGH
        )
        
        assert isinstance(adapted, bool)
        if adapted:
            assert instance.adaptation_count > 0
            assert instance.last_adaptation is not None
    
    @pytest.mark.asyncio
    async def test_orchestration_scaling(self):
        """Test orchestration scaling"""
        # Create and deploy plan
        plan = await self.orchestrator.create_orchestration_plan(
            "Scaling Test",
            "generic",
            2
        )
        await self.orchestrator.deploy_orchestration_plan(plan.plan_id)
        
        # Test scaling up
        scaling_result = await self.orchestrator.scale_orchestration(
            plan.plan_id,
            "scale_up",
            4
        )
        
        assert isinstance(scaling_result, dict)
        assert 'success' in scaling_result
        assert 'changes' in scaling_result
        assert 'current_count' in scaling_result
        assert 'target_count' in scaling_result
        
        if scaling_result['success']:
            assert len(scaling_result['changes']) > 0
            assert scaling_result['target_count'] == 4
    
    @pytest.mark.asyncio
    async def test_honeypot_termination(self):
        """Test honeypot termination"""
        # Deploy honeypot
        config = await self.orchestrator.create_honeypot_configuration(
            "Termination Test",
            "database_enterprise"
        )
        instance = await self.orchestrator.deploy_honeypot(config)
        
        # Terminate honeypot
        success = await self.orchestrator.terminate_honeypot(instance.instance_id)
        
        assert isinstance(success, bool)
        if success:
            assert instance.instance_id not in self.orchestrator.honeypot_instances
    
    @pytest.mark.asyncio
    async def test_status_monitoring(self):
        """Test orchestration status monitoring"""
        # Deploy some honeypots
        for i in range(2):
            config = await self.orchestrator.create_honeypot_configuration(
                f"Status Test {i}",
                "web_server_standard"
            )
            await self.orchestrator.deploy_honeypot(config)
        
        # Get system status
        status = await self.orchestrator.get_orchestration_status()
        
        assert isinstance(status, dict)
        assert 'system_overview' in status
        assert 'performance_metrics' in status
        assert 'health_status' in status
        
        overview = status['system_overview']
        assert 'total_honeypot_instances' in overview
        assert 'running_instances' in overview
        assert overview['running_instances'] >= 2
    
    @pytest.mark.asyncio
    async def test_report_export(self):
        """Test orchestration report export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "orchestration_report.json"
            
            await self.orchestrator.export_orchestration_report(output_path)
            
            assert output_path.exists()
            
            # Validate exported data
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert 'report_metadata' in data
            assert 'system_overview' in data
            assert 'orchestration_plans' in data
            assert 'honeypot_instances' in data
            assert 'deployment_templates' in data
            assert 'performance_analysis' in data


class TestIntegratedDeceptionSystems:
    """Test integration between deception system components"""
    
    def setup_method(self):
        """Set up integrated test environment"""
        self.deception_engine = AdaptiveDeceptionEngine()
        self.counter_intel = CounterIntelligenceOperations()
        self.optimizer = DeceptionEffectivenessOptimizer()
        self.orchestrator = AdvancedHoneypotOrchestrator()
    
    @pytest.mark.asyncio
    async def test_end_to_end_deception_workflow(self):
        """Test complete end-to-end deception workflow"""
        # 1. Analyze attacker behavior
        interaction_data = {
            'attacker_id': 'integration_attacker',
            'attack_vector': 'lateral_movement',
            'tool_signature': 'custom_tool',
            'evasion_technique': 'advanced_obfuscation',
            'timestamp': datetime.now()
        }
        
        pattern = await self.deception_engine.analyze_attacker_behavior(interaction_data)
        
        # 2. Generate false intelligence
        intel_asset = await self.counter_intel.generate_false_intelligence(
            IntelligenceType.NETWORK_TOPOLOGY,
            CredibilityLevel.HIGH,
            pattern.pattern_id
        )
        
        # 3. Select and deploy adaptive strategy
        context = {'available_resources': {'cpu': 4.0, 'memory': '8GB'}}
        strategy = await self.deception_engine.select_adaptive_strategy(pattern, context)
        
        deployment = await self.deception_engine.deploy_strategy(
            strategy, [pattern], "integration_test"
        )
        
        # 4. Deploy coordinated honeypots
        plan = await self.orchestrator.create_orchestration_plan(
            "Integration Test Plan",
            "enterprise_network",
            3
        )
        
        orchestration_result = await self.orchestrator.deploy_orchestration_plan(plan.plan_id)
        
        # 5. Simulate interactions and measure effectiveness
        simulated_interactions = [
            {
                "attacker_id": pattern.pattern_id,
                "engaged": True,
                "duration_seconds": 300,
                "information_extracted": True,
                "believability_assessment": 0.8,
                "misdirection_success": True
            }
            for _ in range(5)
        ]
        
        effectiveness = await self.optimizer.measure_deception_effectiveness(
            strategy.strategy_id,
            deployment.deployment_id,
            simulated_interactions
        )
        
        # Validate integration results
        assert pattern.pattern_id == 'integration_attacker'
        assert intel_asset.intelligence_type == IntelligenceType.NETWORK_TOPOLOGY
        assert deployment.strategy_id == strategy.strategy_id
        assert orchestration_result['success'] or len(orchestration_result['errors']) > 0
        assert effectiveness.engagement_rate >= 0.0
        
        # 6. Test coordinated adaptation
        high_threat_interaction = {
            'attacker_id': pattern.pattern_id,
            'sophistication_increase': True,
            'evasion_detected': True,
            'adaptation_trigger': 'threat_escalation'
        }
        
        # Adapt deception strategy
        strategy_adapted = await self.deception_engine.adapt_strategy(
            deployment, high_threat_interaction
        )
        
        # Adapt counter-intelligence campaign if exists
        if self.counter_intel.active_campaigns:
            campaign_id = list(self.counter_intel.active_campaigns.keys())[0]
            campaign_adapted = await self.counter_intel.adapt_campaign_based_on_feedback(
                campaign_id, {"sophistication_increase": True}
            )
        
        # Adapt honeypot behavior
        if self.orchestrator.honeypot_instances:
            instance_id = list(self.orchestrator.honeypot_instances.keys())[0]
            from agents.honeypot_orchestrator import ThreatLevel
            honeypot_adapted = await self.orchestrator.adapt_honeypot_behavior(
                instance_id, high_threat_interaction, ThreatLevel.HIGH
            )
        
        # Integration should handle coordinated adaptation
        assert isinstance(strategy_adapted, bool)
    
    @pytest.mark.asyncio
    async def test_cross_system_intelligence_sharing(self):
        """Test intelligence sharing between deception systems"""
        # Generate intelligence in counter-intel system
        intel_asset = await self.counter_intel.generate_false_intelligence(
            IntelligenceType.USER_CREDENTIALS,
            CredibilityLevel.MEDIUM,
            "shared_intelligence_test"
        )
        
        # Track consumption
        await self.counter_intel.track_intelligence_consumption(
            intel_asset.asset_id,
            "shared_attacker",
            {"data_extracted": True, "time_spent_seconds": 180}
        )
        
        # Use intelligence for honeypot configuration
        config = await self.orchestrator.create_honeypot_configuration(
            "Intelligence-Driven Honeypot",
            "ssh_server_secure",
            {
                "credentials": intel_asset.content.get("users", [{}])[0] if intel_asset.content.get("users") else {},
                "interaction_complexity": 0.7
            }
        )
        
        # Deploy honeypot with intelligence-based configuration
        instance = await self.orchestrator.deploy_honeypot(config)
        
        # Validate cross-system integration
        assert intel_asset.access_count > 0
        assert "shared_attacker" in intel_asset.unique_accessors
        assert instance.configuration.name == "Intelligence-Driven Honeypot"
        assert instance.status == "running"
    
    @pytest.mark.asyncio 
    async def test_coordinated_optimization(self):
        """Test coordinated optimization across all systems"""
        # Create baseline measurements
        base_interactions = [
            {
                "attacker_id": f"opt_attacker_{i}",
                "engaged": random.choice([True, False]),
                "duration_seconds": random.randint(60, 400),
                "believability_assessment": random.uniform(0.4, 0.8)
            }
            for i in range(10)
        ]
        
        baseline_metrics = await self.optimizer.measure_deception_effectiveness(
            "coordinated_test",
            "baseline_deployment",
            base_interactions
        )
        
        # Create optimization parameters
        base_params = OptimizationParameters(
            parameter_set_id="coordinated_base",
            strategy_id="coordinated_test",
            credibility_level=0.5,
            visibility_level=0.5,
            interaction_depth=0.5
        )
        
        # Run optimization
        optimized_params = await self.optimizer.optimize_parameters(
            "coordinated_test",
            base_params,
            [baseline_metrics]
        )
        
        # Apply optimizations across systems
        # 1. Update deception strategies
        for strategy in self.deception_engine.strategies.values():
            if strategy.strategy_id == "coordinated_test":
                strategy.learning_rate = optimized_params.learning_rate
        
        # 2. Update counter-intelligence credibility
        for asset in self.counter_intel.intelligence_assets.values():
            if optimized_params.credibility_level > 0.7:
                if asset.credibility_level.value < 4:
                    asset.credibility_level = CredibilityLevel.HIGH
        
        # 3. Update honeypot configurations
        for instance in self.orchestrator.honeypot_instances.values():
            instance.configuration.interaction_complexity = optimized_params.interaction_depth
        
        # Validate coordinated optimization
        assert optimized_params.optimization_generation > base_params.optimization_generation
        assert baseline_metrics.measurement_id in [m.measurement_id for m in self.optimizer.effectiveness_history]
    
    @pytest.mark.asyncio
    async def test_system_resilience_and_recovery(self):
        """Test system resilience and recovery capabilities"""
        # Deploy multiple components
        strategy = list(self.deception_engine.strategies.values())[0]
        pattern = AttackerBehaviorPattern(pattern_id="resilience_test")
        deployment = await self.deception_engine.deploy_strategy(strategy, [pattern])
        
        campaign = await self.counter_intel.create_disinformation_campaign(
            "Resilience Test",
            DisinformationStrategy.PLANTED_DOCUMENTS,
            [CounterIntelTarget.RECONNAISSANCE_AGENTS],
            "Resilience test narrative"
        )
        
        plan = await self.orchestrator.create_orchestration_plan(
            "Resilience Plan",
            "enterprise_network",
            2
        )
        await self.orchestrator.deploy_orchestration_plan(plan.plan_id)
        
        # Simulate system stress/failures
        # 1. Expire deployments
        deployment.expires_at = datetime.now() - timedelta(minutes=1)
        
        # 2. Mark instances as unhealthy
        for instance in self.orchestrator.honeypot_instances.values():
            instance.health_status = "unhealthy"
            instance.error_count = 5
        
        # 3. Test cleanup and recovery
        expired_count = await self.deception_engine.cleanup_expired_deployments()
        
        # 4. Test system status after stress
        deception_metrics = await self.deception_engine.get_deception_metrics()
        counter_metrics = await self.counter_intel.get_counter_intelligence_metrics()
        orchestration_status = await self.orchestrator.get_orchestration_status()
        
        # Validate resilience
        assert isinstance(expired_count, int)
        assert isinstance(deception_metrics, dict)
        assert isinstance(counter_metrics, dict)
        assert isinstance(orchestration_status, dict)
        
        # System should continue operating despite failures
        assert 'system_overview' in deception_metrics
        assert 'system_overview' in counter_metrics
        assert 'system_overview' in orchestration_status
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under high load"""
        import time
        
        # Generate high load scenario
        start_time = time.time()
        
        tasks = []
        
        # Concurrent attacker behavior analysis
        for i in range(20):
            interaction = {
                'attacker_id': f'load_test_attacker_{i}',
                'attack_vector': f'vector_{i % 5}',
                'timestamp': datetime.now()
            }
            task = self.deception_engine.analyze_attacker_behavior(interaction)
            tasks.append(task)
        
        # Concurrent intelligence generation
        for i in range(10):
            task = self.counter_intel.generate_false_intelligence(
                IntelligenceType.NETWORK_TOPOLOGY,
                CredibilityLevel.MEDIUM,
                f"load_test_{i}"
            )
            tasks.append(task)
        
        # Concurrent honeypot configuration creation
        for i in range(15):
            task = self.orchestrator.create_honeypot_configuration(
                f"Load Test {i}",
                "web_server_standard"
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Validate performance
        assert total_time < 30.0  # Should complete within 30 seconds
        
        # Count successful operations
        successful_operations = sum(1 for result in results if not isinstance(result, Exception))
        error_rate = (len(results) - successful_operations) / len(results)
        
        assert error_rate < 0.1  # Less than 10% error rate under load
        assert successful_operations > 40  # At least 40 successful operations


# Performance and reliability tests
class TestSystemPerformance:
    """Test system performance and reliability"""
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage remains stable under continuous operation"""
        import psutil
        import gc
        
        deception_engine = AdaptiveDeceptionEngine()
        initial_memory = psutil.Process().memory_info().rss
        
        # Perform many operations
        for i in range(100):
            interaction = {
                'attacker_id': f'memory_test_{i}',
                'attack_vector': 'test_vector',
                'timestamp': datetime.now()
            }
            await deception_engine.analyze_attacker_behavior(interaction)
            
            if i % 10 == 0:
                gc.collect()  # Force garbage collection
        
        final_memory = psutil.Process().memory_info().rss
        memory_growth = (final_memory - initial_memory) / initial_memory
        
        # Memory growth should be reasonable (less than 50%)
        assert memory_growth < 0.5
    
    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self):
        """Test thread safety under concurrent access"""
        import asyncio
        
        counter_intel = CounterIntelligenceOperations()
        
        async def concurrent_operation(operation_id):
            try:
                asset = await counter_intel.generate_false_intelligence(
                    IntelligenceType.SYSTEM_CONFIGURATION,
                    CredibilityLevel.MEDIUM,
                    f"concurrent_{operation_id}"
                )
                
                await counter_intel.track_intelligence_consumption(
                    asset.asset_id,
                    f"concurrent_attacker_{operation_id}",
                    {"data_extracted": True}
                )
                
                return True
            except Exception as e:
                print(f"Concurrent operation {operation_id} failed: {e}")
                return False
        
        # Run multiple concurrent operations
        tasks = [concurrent_operation(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful operations
        successful = sum(1 for result in results if result is True)
        
        # Should have high success rate even under concurrency
        assert successful >= 45  # At least 90% success rate


def test_deception_strategy_configuration():
    """Test deception strategy configuration validation"""
    engine = AdaptiveDeceptionEngine()
    
    # Test strategy properties
    for strategy in engine.strategies.values():
        assert hasattr(strategy, 'strategy_id')
        assert hasattr(strategy, 'name')
        assert hasattr(strategy, 'deception_type')
        assert isinstance(strategy.target_profiles, list)
        assert len(strategy.target_profiles) > 0
        assert isinstance(strategy.success_rate, float)
        assert 0.0 <= strategy.success_rate <= 1.0


def test_intelligence_template_validation():
    """Test intelligence template validation"""
    counter_intel = CounterIntelligenceOperations()
    
    # Test template structure
    for template_id, template in counter_intel.intelligence_templates.items():
        assert 'template_id' in template
        assert 'intelligence_type' in template
        assert 'structure' in template
        assert isinstance(template['structure'], dict)
        if 'believability_factors' in template:
            assert isinstance(template['believability_factors'], list)


def test_honeypot_template_validation():
    """Test honeypot template validation"""
    orchestrator = AdvancedHoneypotOrchestrator()
    
    # Test template structure
    for template_id, template in orchestrator.deployment_templates.items():
        assert 'honeypot_type' in template
        assert 'profile' in template
        if 'services' in template:
            assert isinstance(template['services'], list)
        if 'credentials' in template:
            assert isinstance(template['credentials'], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])