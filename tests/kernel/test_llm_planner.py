#!/usr/bin/env python3
"""
Test suite for LLM Planning Engine

Tests the natural language objective parsing, multi-stage operation planning,
adaptive strategy modification, and constraint handling capabilities.
"""

import asyncio
import pytest
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../opt/archangel'))

from ai.planner import (
    LLMPlanningEngine, ObjectiveParser, StrategyPlanner, AdaptiveStrategyModifier,
    OperationType, OperationPhase, RiskLevel, StealthLevel,
    OperationObjective, OperationConstraints, AdaptationContext
)
from ai.models import AIModelManager, ModelType, InferenceRequest, InferenceResult


class TestObjectiveParser:
    """Test natural language objective parsing"""
    
    def setup_method(self):
        self.parser = ObjectiveParser()
    
    @pytest.mark.asyncio
    async def test_parse_penetration_test_objective(self):
        """Test parsing penetration test objectives"""
        input_text = "Perform a full penetration test of 192.168.1.0/24"
        
        objective = await self.parser.parse_objective(input_text)
        
        assert objective.operation_type == OperationType.PENETRATION_TEST
        assert "192.168.1.0/24" in objective.primary_targets
        assert objective.raw_input == input_text
        assert len(objective.success_criteria) > 0
    
    @pytest.mark.asyncio
    async def test_parse_osint_objective(self):
        """Test parsing OSINT investigation objectives"""
        input_text = "Conduct OSINT investigation on example.com"
        
        objective = await self.parser.parse_objective(input_text)
        
        assert objective.operation_type == OperationType.OSINT_INVESTIGATION
        assert "example.com" in objective.primary_targets
        assert "intelligence" in objective.success_criteria[0].lower()
    
    @pytest.mark.asyncio
    async def test_parse_web_audit_objective(self):
        """Test parsing web application audit objectives"""
        input_text = "Perform web application security audit of https://webapp.example.com"
        
        objective = await self.parser.parse_objective(input_text)
        
        assert objective.operation_type == OperationType.WEB_APPLICATION_AUDIT
        assert any("webapp.example.com" in target for target in objective.primary_targets)
    
    @pytest.mark.asyncio
    async def test_parse_stealth_constraints(self):
        """Test parsing stealth requirements"""
        input_text = "Perform stealthy penetration test of target.com"
        
        objective = await self.parser.parse_objective(input_text)
        
        assert objective.constraints.stealth_requirements == StealthLevel.HIGH
    
    @pytest.mark.asyncio
    async def test_parse_time_constraints(self):
        """Test parsing time constraints"""
        input_text = "Complete vulnerability scan of 10.0.0.0/8 within 2 hours"
        
        objective = await self.parser.parse_objective(input_text)
        
        assert objective.constraints.max_duration_hours == 2
    
    @pytest.mark.asyncio
    async def test_parse_compliance_constraints(self):
        """Test parsing compliance requirements"""
        input_text = "Perform NIST compliant security assessment of infrastructure"
        
        objective = await self.parser.parse_objective(input_text)
        
        assert "NIST" in objective.constraints.compliance_requirements
    
    @pytest.mark.asyncio
    async def test_parse_multiple_targets(self):
        """Test parsing multiple targets"""
        input_text = "Test 192.168.1.1, 192.168.1.2, and example.com"
        
        objective = await self.parser.parse_objective(input_text)
        
        assert len(objective.primary_targets) >= 3
        assert "192.168.1.1" in objective.primary_targets
        assert "192.168.1.2" in objective.primary_targets
        assert "example.com" in objective.primary_targets


class TestStrategyPlanner:
    """Test multi-stage operation planning"""
    
    def setup_method(self):
        # Mock model manager
        self.mock_model_manager = Mock(spec=AIModelManager)
        self.mock_model_manager.infer = AsyncMock()
        
        self.planner = StrategyPlanner(self.mock_model_manager)
    
    @pytest.mark.asyncio
    async def test_create_penetration_test_plan(self):
        """Test creating penetration test operation plan"""
        # Mock LLM responses
        self.mock_model_manager.infer.return_value = InferenceResult(
            output={
                "steps": [
                    {
                        "name": "Network Discovery",
                        "description": "Discover active hosts",
                        "tools_required": ["nmap"],
                        "estimated_duration": "30 minutes",
                        "success_criteria": ["Hosts identified"],
                        "failure_alternatives": ["Try alternative discovery"]
                    }
                ]
            },
            confidence=0.9,
            processing_time_ms=100,
            model_name="test_model"
        )
        
        objective = OperationObjective(
            raw_input="Penetration test of 192.168.1.0/24",
            operation_type=OperationType.PENETRATION_TEST,
            primary_targets=["192.168.1.0/24"],
            secondary_objectives=[],
            success_criteria=["Complete assessment"],
            constraints=OperationConstraints()
        )
        
        plan = await self.planner.create_operation_plan(objective)
        
        assert plan.objective == objective
        assert len(plan.phases) > 0
        assert OperationPhase.RECONNAISSANCE in [p.phase for p in plan.phases]
        assert OperationPhase.SCANNING in [p.phase for p in plan.phases]
        assert OperationPhase.EXPLOITATION in [p.phase for p in plan.phases]
        assert plan.plan_id.startswith("plan_")
    
    @pytest.mark.asyncio
    async def test_create_osint_plan(self):
        """Test creating OSINT investigation plan"""
        self.mock_model_manager.infer.return_value = InferenceResult(
            output="Execute comprehensive OSINT investigation",
            confidence=0.85,
            processing_time_ms=150,
            model_name="test_model"
        )
        
        objective = OperationObjective(
            raw_input="OSINT investigation of example.com",
            operation_type=OperationType.OSINT_INVESTIGATION,
            primary_targets=["example.com"],
            secondary_objectives=["employee_enumeration"],
            success_criteria=["Gather intelligence"],
            constraints=OperationConstraints()
        )
        
        plan = await self.planner.create_operation_plan(objective)
        
        assert OperationPhase.OSINT in [p.phase for p in plan.phases]
        assert OperationPhase.REPORTING in [p.phase for p in plan.phases]
        assert len(plan.phases) == 2  # OSINT operations are simpler
    
    @pytest.mark.asyncio
    async def test_phase_step_generation(self):
        """Test detailed phase step generation"""
        self.mock_model_manager.infer.return_value = InferenceResult(
            output={
                "steps": [
                    {
                        "name": "Port Scanning",
                        "description": "Scan for open ports",
                        "tools_required": ["nmap", "masscan"],
                        "estimated_duration": "45 minutes",
                        "success_criteria": ["Open ports identified"],
                        "failure_alternatives": ["Manual port checking"],
                        "stealth_considerations": ["Rate limiting"],
                        "requires_approval": False
                    },
                    {
                        "name": "Service Enumeration",
                        "description": "Enumerate running services",
                        "tools_required": ["nmap", "banner_grabbing"],
                        "estimated_duration": "30 minutes",
                        "success_criteria": ["Services cataloged"],
                        "failure_alternatives": ["Alternative enumeration"],
                        "stealth_considerations": ["Minimal probes"],
                        "requires_approval": False
                    }
                ]
            },
            confidence=0.88,
            processing_time_ms=200,
            model_name="test_model"
        )
        
        objective = OperationObjective(
            raw_input="Scan network infrastructure",
            operation_type=OperationType.NETWORK_ASSESSMENT,
            primary_targets=["10.0.0.0/24"],
            secondary_objectives=[],
            success_criteria=["Network mapped"],
            constraints=OperationConstraints()
        )
        
        phase_plan = await self.planner._create_phase_plan(OperationPhase.SCANNING, objective)
        
        assert len(phase_plan.steps) == 2
        assert phase_plan.steps[0].name == "Port Scanning"
        assert "nmap" in phase_plan.steps[0].tools_required
        assert phase_plan.steps[1].name == "Service Enumeration"
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self):
        """Test operation risk assessment"""
        objective = OperationObjective(
            raw_input="Test production environment",
            operation_type=OperationType.PENETRATION_TEST,
            primary_targets=["production.example.com"],
            secondary_objectives=[],
            success_criteria=["Assessment complete"],
            constraints=OperationConstraints(stealth_requirements=StealthLevel.MAXIMUM)
        )
        
        risk_assessment = self.planner._perform_risk_assessment(objective, [])
        
        assert risk_assessment["overall_risk_level"] == objective.risk_level.value
        assert "Production environment target" in risk_assessment["risk_factors"]
        assert "Maximum stealth required" in risk_assessment["risk_factors"]
        assert len(risk_assessment["mitigation_strategies"]) > 0


class TestAdaptiveStrategyModifier:
    """Test adaptive strategy modification"""
    
    def setup_method(self):
        self.mock_model_manager = Mock(spec=AIModelManager)
        self.mock_model_manager.infer = AsyncMock()
        
        self.modifier = AdaptiveStrategyModifier(self.mock_model_manager)
    
    @pytest.mark.asyncio
    async def test_adapt_strategy_for_failed_steps(self):
        """Test strategy adaptation when steps fail"""
        # Mock LLM adaptation response
        self.mock_model_manager.infer.return_value = InferenceResult(
            output={
                "adaptation_type": "major",
                "recommended_changes": [
                    {
                        "area": "strategy",
                        "change": "Switch to alternative exploitation methods",
                        "reason": "Primary exploitation failed",
                        "priority": "high"
                    }
                ],
                "risk_assessment": "Medium risk adaptation",
                "expected_impact": "Improved success probability"
            },
            confidence=0.82,
            processing_time_ms=300,
            model_name="test_model"
        )
        
        # Create test plan
        objective = OperationObjective(
            raw_input="Test target system",
            operation_type=OperationType.PENETRATION_TEST,
            primary_targets=["target.com"],
            secondary_objectives=[],
            success_criteria=["Successful exploitation"],
            constraints=OperationConstraints()
        )
        
        from ai.planner import OperationPlan, OperationPhaseplan
        plan = OperationPlan(
            plan_id="test_plan_001",
            objective=objective,
            phases=[],
            overall_strategy="Standard penetration testing approach",
            estimated_total_duration="4 hours",
            risk_assessment={},
            contingency_plans=[],
            resource_requirements={},
            success_metrics=[],
            created_at=datetime.now(),
            last_modified=datetime.now()
        )
        
        # Create adaptation context with failed steps
        context = AdaptationContext(
            current_phase=OperationPhase.EXPLOITATION,
            completed_steps=["recon_01", "scan_01"],
            failed_steps=["exploit_01", "exploit_02"],
            discovered_information={},
            environmental_changes=[],
            new_constraints=[],
            performance_metrics={},
            time_elapsed=2.5,
            remaining_time=1.5
        )
        
        adapted_plan = await self.modifier.adapt_strategy(plan, context)
        
        assert adapted_plan.version == plan.version + 1
        assert "ADAPTED" in adapted_plan.overall_strategy
        assert len(self.modifier.adaptation_history) == 1
    
    @pytest.mark.asyncio
    async def test_no_adaptation_needed(self):
        """Test when no adaptation is needed"""
        objective = OperationObjective(
            raw_input="Simple scan",
            operation_type=OperationType.VULNERABILITY_SCAN,
            primary_targets=["test.com"],
            secondary_objectives=[],
            success_criteria=["Scan complete"],
            constraints=OperationConstraints()
        )
        
        from ai.planner import OperationPlan
        plan = OperationPlan(
            plan_id="test_plan_002",
            objective=objective,
            phases=[],
            overall_strategy="Simple vulnerability scanning",
            estimated_total_duration="1 hour",
            risk_assessment={},
            contingency_plans=[],
            resource_requirements={},
            success_metrics=[],
            created_at=datetime.now(),
            last_modified=datetime.now()
        )
        
        # Context with no issues
        context = AdaptationContext(
            current_phase=OperationPhase.SCANNING,
            completed_steps=["scan_01"],
            failed_steps=[],
            discovered_information={},
            environmental_changes=[],
            new_constraints=[],
            performance_metrics={"success_rate": 0.95},
            time_elapsed=0.5,
            remaining_time=0.5
        )
        
        adapted_plan = await self.modifier.adapt_strategy(plan, context)
        
        # Should return original plan unchanged
        assert adapted_plan.version == plan.version
        assert adapted_plan.overall_strategy == plan.overall_strategy
    
    @pytest.mark.asyncio
    async def test_time_pressure_adaptation(self):
        """Test adaptation under time pressure"""
        self.mock_model_manager.infer.return_value = InferenceResult(
            output={
                "adaptation_type": "critical",
                "recommended_changes": [
                    {
                        "area": "timing",
                        "change": "Compress remaining phases",
                        "reason": "Time running short",
                        "priority": "high"
                    }
                ],
                "risk_assessment": "Time pressure requires prioritization",
                "expected_impact": "Focus on critical objectives"
            },
            confidence=0.75,
            processing_time_ms=150,
            model_name="test_model"
        )
        
        objective = OperationObjective(
            raw_input="Urgent assessment",
            operation_type=OperationType.PENETRATION_TEST,
            primary_targets=["urgent.com"],
            secondary_objectives=[],
            success_criteria=["Critical findings identified"],
            constraints=OperationConstraints(max_duration_hours=2)
        )
        
        from ai.planner import OperationPlan
        plan = OperationPlan(
            plan_id="test_plan_003",
            objective=objective,
            phases=[],
            overall_strategy="Comprehensive assessment",
            estimated_total_duration="4 hours",
            risk_assessment={},
            contingency_plans=[],
            resource_requirements={},
            success_metrics=[],
            created_at=datetime.now(),
            last_modified=datetime.now()
        )
        
        # Context showing time pressure
        context = AdaptationContext(
            current_phase=OperationPhase.SCANNING,
            completed_steps=["recon_01"],
            failed_steps=[],
            discovered_information={},
            environmental_changes=[],
            new_constraints=[],
            performance_metrics={},
            time_elapsed=1.5,  # 1.5 hours elapsed
            remaining_time=0.5  # Only 0.5 hours remaining
        )
        
        adapted_plan = await self.modifier.adapt_strategy(plan, context)
        
        assert adapted_plan.version > plan.version
        assert len(self.modifier.adaptation_history) == 1


class TestLLMPlanningEngine:
    """Test the main LLM Planning Engine"""
    
    def setup_method(self):
        # Mock model manager
        self.mock_model_manager = Mock(spec=AIModelManager)
        self.mock_model_manager.infer = AsyncMock()
        
        self.engine = LLMPlanningEngine(self.mock_model_manager)
    
    @pytest.mark.asyncio
    async def test_parse_and_plan_integration(self):
        """Test complete parse and plan workflow"""
        # Mock LLM responses for planning
        self.mock_model_manager.infer.side_effect = [
            # Phase step generation
            InferenceResult(
                output={
                    "steps": [
                        {
                            "name": "Network Discovery",
                            "description": "Discover network hosts",
                            "tools_required": ["nmap"],
                            "estimated_duration": "30 minutes",
                            "success_criteria": ["Hosts found"],
                            "failure_alternatives": ["Manual discovery"]
                        }
                    ]
                },
                confidence=0.9,
                processing_time_ms=100,
                model_name="test_model"
            ),
            # Overall strategy generation
            InferenceResult(
                output="Execute systematic penetration testing methodology",
                confidence=0.85,
                processing_time_ms=200,
                model_name="test_model"
            )
        ]
        
        natural_input = "Perform penetration test of 192.168.1.100"
        
        plan = await self.engine.parse_and_plan(natural_input)
        
        assert plan.plan_id in self.engine.active_plans
        assert plan.objective.operation_type == OperationType.PENETRATION_TEST
        assert "192.168.1.100" in plan.objective.primary_targets
        assert len(plan.phases) > 0
        assert self.engine.stats["plans_created"] == 1
    
    @pytest.mark.asyncio
    async def test_constraint_handling(self):
        """Test handling of operation constraints"""
        additional_constraints = {
            "stealth_requirements": "maximum",
            "max_duration_hours": 3,
            "prohibited_actions": ["destructive_operations"],
            "compliance_requirements": ["SOX", "HIPAA"]
        }
        
        # Mock LLM responses
        self.mock_model_manager.infer.return_value = InferenceResult(
            output={"steps": []},
            confidence=0.8,
            processing_time_ms=100,
            model_name="test_model"
        )
        
        natural_input = "Security audit of healthcare system"
        
        plan = await self.engine.parse_and_plan(natural_input, additional_constraints)
        
        assert plan.objective.constraints.stealth_requirements == StealthLevel.MAXIMUM
        assert plan.objective.constraints.max_duration_hours == 3
        assert "destructive_operations" in plan.objective.constraints.prohibited_actions
        assert "SOX" in plan.objective.constraints.compliance_requirements
        assert "HIPAA" in plan.objective.constraints.compliance_requirements
    
    @pytest.mark.asyncio
    async def test_plan_adaptation(self):
        """Test plan adaptation functionality"""
        # First create a plan
        self.mock_model_manager.infer.return_value = InferenceResult(
            output={"steps": []},
            confidence=0.8,
            processing_time_ms=100,
            model_name="test_model"
        )
        
        plan = await self.engine.parse_and_plan("Test network security")
        plan_id = plan.plan_id
        
        # Mock adaptation response
        self.mock_model_manager.infer.return_value = InferenceResult(
            output={
                "adaptation_type": "minor",
                "recommended_changes": [
                    {
                        "area": "tools",
                        "change": "Switch to alternative scanning tools",
                        "reason": "Primary tools blocked",
                        "priority": "medium"
                    }
                ],
                "risk_assessment": "Low risk adaptation",
                "expected_impact": "Maintain operation effectiveness"
            },
            confidence=0.78,
            processing_time_ms=250,
            model_name="test_model"
        )
        
        # Adapt the plan
        context_updates = {
            "failed_steps": ["scan_01"],
            "environmental_changes": ["Firewall blocking scans"],
            "time_elapsed": 1.0
        }
        
        adapted_plan = await self.engine.adapt_plan(plan_id, context_updates)
        
        assert adapted_plan is not None
        assert adapted_plan.version > plan.version
        assert self.engine.stats["adaptations_made"] == 1
    
    @pytest.mark.asyncio
    async def test_safety_validation(self):
        """Test operation safety validation"""
        # Test that dangerous operations are flagged
        with pytest.raises(ValueError, match="requires explicit authorization"):
            await self.engine.parse_and_plan("Hack localhost")
    
    @pytest.mark.asyncio
    async def test_plan_completion(self):
        """Test plan completion tracking"""
        # Create a plan
        self.mock_model_manager.infer.return_value = InferenceResult(
            output={"steps": []},
            confidence=0.8,
            processing_time_ms=100,
            model_name="test_model"
        )
        
        plan = await self.engine.parse_and_plan("Security assessment")
        plan_id = plan.plan_id
        
        # Complete the plan successfully
        self.engine.complete_plan(plan_id, success=True)
        
        assert plan_id not in self.engine.active_plans
        assert self.engine.stats["successful_operations"] == 1
        assert self.engine.stats["failed_operations"] == 0
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        stats = self.engine.get_statistics()
        
        assert "plans_created" in stats
        assert "adaptations_made" in stats
        assert "successful_operations" in stats
        assert "failed_operations" in stats
        assert "average_planning_time_ms" in stats


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])