#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Scenario Generation Tests
Comprehensive tests for scenario generation quality and execution reliability
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from scenarios.scenario_templates import (
    ScenarioTemplate, ScenarioTemplateManager, ScenarioInstance,
    ScenarioType, ScenarioCategory, ComplexityLevel, NetworkTopology,
    ScenarioParameter, ScenarioObjective, ScenarioAsset
)
from scenarios.dynamic_generation import (
    DynamicScenarioGenerator, GenerationContext, AgentLearningProfile,
    GenerationType, LearningOutcome, GeneratedScenario
)
from scenarios.difficulty_progression import (
    DifficultyProgressionEngine, ProgressionPath, ProgressionState,
    DifficultyProfile, ProgressionStrategy
)
from scenarios.scenario_validation import (
    ScenarioValidator, ScenarioTester, ValidationLevel, ValidationCategory,
    TestType, TestCase, TestSuite
)


class TestScenarioTemplateManager:
    """Test scenario template management functionality"""
    
    @pytest.fixture
    async def template_manager(self):
        """Create a template manager for testing"""
        manager = ScenarioTemplateManager()
        await manager.initialize()
        return manager
    
    @pytest.fixture
    def sample_template(self):
        """Create a sample scenario template"""
        return ScenarioTemplate(
            template_id=str(uuid.uuid4()),
            name="Test Reconnaissance Scenario",
            description="A test scenario for reconnaissance training",
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.RECONNAISSANCE,
            complexity=ComplexityLevel.INTERMEDIATE,
            estimated_duration=timedelta(hours=2),
            min_participants=1,
            max_participants=5,
            objectives=[
                ScenarioObjective(
                    objective_id="obj1",
                    name="Network Discovery",
                    description="Discover network topology",
                    success_criteria=["Identify at least 5 hosts", "Map network segments"]
                )
            ],
            parameters=[
                ScenarioParameter(
                    name="target_network",
                    parameter_type="string",
                    default_value="192.168.1.0/24",
                    description="Target network range"
                )
            ]
        )
    
    @pytest.mark.asyncio
    async def test_create_template(self, template_manager, sample_template):
        """Test template creation"""
        template_id = await template_manager.create_template(sample_template)
        
        assert template_id == sample_template.template_id
        assert template_id in template_manager.templates
        
        retrieved = await template_manager.get_template(template_id)
        assert retrieved is not None
        assert retrieved.name == sample_template.name
    
    @pytest.mark.asyncio
    async def test_template_validation(self, template_manager, sample_template):
        """Test template validation"""
        validation_result = await template_manager.validate_template(sample_template)
        
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_template_validation(self, template_manager):
        """Test validation of invalid template"""
        invalid_template = ScenarioTemplate(
            template_id=str(uuid.uuid4()),
            name="",  # Invalid: empty name
            description="",  # Invalid: empty description
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.RECONNAISSANCE,
            complexity=ComplexityLevel.INTERMEDIATE
        )
        
        validation_result = await template_manager.validate_template(invalid_template)
        
        assert validation_result["valid"] is False
        assert len(validation_result["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_template_instantiation(self, template_manager, sample_template):
        """Test template instantiation"""
        await template_manager.create_template(sample_template)
        
        parameters = {"target_network": "10.0.0.0/24"}
        participants = ["agent1", "agent2"]
        
        instance = await template_manager.instantiate_template(
            sample_template.template_id,
            parameters,
            participants
        )
        
        assert instance.template_id == sample_template.template_id
        assert instance.parameters == parameters
        assert instance.participants == participants
        assert instance.status == "created"
    
    @pytest.mark.asyncio
    async def test_template_cloning(self, template_manager, sample_template):
        """Test template cloning"""
        await template_manager.create_template(sample_template)
        
        new_name = "Cloned Reconnaissance Scenario"
        modifications = {"complexity": ComplexityLevel.ADVANCED}
        
        cloned_id = await template_manager.clone_template(
            sample_template.template_id,
            new_name,
            modifications
        )
        
        cloned_template = await template_manager.get_template(cloned_id)
        assert cloned_template is not None
        assert cloned_template.name == new_name
        assert cloned_template.complexity == ComplexityLevel.ADVANCED
    
    @pytest.mark.asyncio
    async def test_template_recommendations(self, template_manager, sample_template):
        """Test template recommendations"""
        await template_manager.create_template(sample_template)
        
        agent_profile = {
            "skill_level": 0.6,
            "experience_count": 25,
            "success_rate": 0.75,
            "preferred_categories": ["reconnaissance"]
        }
        
        recommendations = await template_manager.get_template_recommendations(
            agent_profile,
            max_recommendations=3
        )
        
        assert len(recommendations) <= 3
        assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)
        assert all(isinstance(rec[0], ScenarioTemplate) and isinstance(rec[1], float) 
                  for rec in recommendations)


class TestDynamicScenarioGenerator:
    """Test dynamic scenario generation functionality"""
    
    @pytest.fixture
    async def generator(self):
        """Create a scenario generator for testing"""
        generator = DynamicScenarioGenerator()
        await generator.initialize()
        return generator
    
    @pytest.fixture
    def agent_profile(self):
        """Create a sample agent learning profile"""
        return AgentLearningProfile(
            agent_id="test_agent",
            skill_levels={
                "reconnaissance": 0.6,
                "exploitation": 0.4,
                "persistence": 0.3
            },
            success_rates={
                ScenarioCategory.RECONNAISSANCE: 0.7,
                ScenarioCategory.INITIAL_ACCESS: 0.5
            },
            preferred_complexity=ComplexityLevel.INTERMEDIATE,
            identified_weaknesses=["exploitation", "persistence"],
            core_strengths=["reconnaissance"]
        )
    
    @pytest.fixture
    def generation_context(self, agent_profile):
        """Create a generation context for testing"""
        return GenerationContext(
            target_agents=["test_agent"],
            agent_profiles={"test_agent": agent_profile},
            primary_objectives=[LearningOutcome.SKILL_MASTERY],
            generation_type=GenerationType.ADAPTIVE_LEARNING,
            novelty_factor=0.5,
            challenge_factor=0.6
        )
    
    @pytest.mark.asyncio
    async def test_scenario_generation(self, generator, generation_context):
        """Test basic scenario generation"""
        with patch.object(generator, '_create_adaptive_scenario_template', 
                         return_value=AsyncMock()) as mock_create:
            mock_template = ScenarioTemplate(
                template_id=str(uuid.uuid4()),
                name="Generated Test Scenario",
                description="A dynamically generated test scenario",
                scenario_type=ScenarioType.TRAINING,
                category=ScenarioCategory.RECONNAISSANCE,
                complexity=ComplexityLevel.INTERMEDIATE
            )
            mock_create.return_value = mock_template
            
            generated = await generator.generate_scenario(generation_context)
            
            assert isinstance(generated, GeneratedScenario)
            assert generated.scenario == mock_template
            assert generated.generation_type == GenerationType.ADAPTIVE_LEARNING
            assert 0.0 <= generated.confidence_score <= 1.0
            assert 0.0 <= generated.success_probability <= 1.0
    
    @pytest.mark.asyncio
    async def test_agent_profile_update(self, generator, agent_profile):
        """Test agent profile updates based on scenario results"""
        generator.agent_profiles["test_agent"] = agent_profile
        
        scenario_result = {
            "success": True,
            "completion_time": 3600,  # 1 hour
            "confidence": 0.8,
            "category": ScenarioCategory.RECONNAISSANCE,
            "skills_demonstrated": ["reconnaissance", "network_analysis"],
            "lessons_learned": ["Improved scanning techniques"]
        }
        
        await generator.update_agent_profile("test_agent", scenario_result)
        
        updated_profile = generator.agent_profiles["test_agent"]
        
        # Check that success rate was updated
        assert ScenarioCategory.RECONNAISSANCE in updated_profile.success_rates
        
        # Check that skill levels were updated
        assert "reconnaissance" in updated_profile.skill_levels
        assert "network_analysis" in updated_profile.skill_levels
        
        # Check that learning trajectory was updated
        assert len(updated_profile.learning_trajectory) > 0
    
    @pytest.mark.asyncio
    async def test_learning_recommendations(self, generator, agent_profile):
        """Test learning recommendations generation"""
        generator.agent_profiles["test_agent"] = agent_profile
        
        recommendations = await generator.get_learning_recommendations("test_agent")
        
        assert isinstance(recommendations, list)
        
        # Should have recommendations for identified weaknesses
        weakness_recs = [r for r in recommendations if r["type"] == "weakness_targeting"]
        assert len(weakness_recs) > 0
        
        # Check recommendation structure
        for rec in recommendations:
            assert "type" in rec
            assert "recommendation" in rec
            assert "priority" in rec
    
    @pytest.mark.asyncio
    async def test_generation_strategy_selection(self, generator, generation_context):
        """Test generation strategy selection logic"""
        # Test collaborative strategy for multiple agents
        generation_context.target_agents = ["agent1", "agent2"]
        generation_context.collaboration_factor = 0.8
        
        strategy = await generator._select_generation_strategy(generation_context)
        assert strategy == GenerationType.COLLABORATIVE
        
        # Test weakness targeting for agents with major weaknesses
        generation_context.target_agents = ["test_agent"]
        generation_context.collaboration_factor = 0.3
        
        strategy = await generator._select_generation_strategy(generation_context)
        # Should select weakness targeting due to identified weaknesses
        assert strategy in [GenerationType.WEAKNESS_TARGETING, GenerationType.ADAPTIVE_LEARNING]


class TestDifficultyProgression:
    """Test difficulty progression and scaling functionality"""
    
    @pytest.fixture
    async def progression_engine(self):
        """Create a difficulty progression engine for testing"""
        engine = DifficultyProgressionEngine()
        await engine.initialize()
        return engine
    
    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario for difficulty calculation"""
        return ScenarioTemplate(
            template_id=str(uuid.uuid4()),
            name="Test Scenario",
            description="A test scenario",
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.RECONNAISSANCE,
            complexity=ComplexityLevel.INTERMEDIATE,
            estimated_duration=timedelta(hours=2),
            objectives=[
                ScenarioObjective(
                    objective_id="obj1",
                    name="Test Objective",
                    description="A test objective",
                    dependencies=[]
                )
            ]
        )
    
    @pytest.fixture
    def agent_profile(self):
        """Create a sample agent profile"""
        return AgentLearningProfile(
            agent_id="test_agent",
            skill_levels={"reconnaissance": 0.6},
            completed_scenarios=["scenario1", "scenario2"]
        )
    
    @pytest.mark.asyncio
    async def test_difficulty_calculation(self, progression_engine, sample_scenario, agent_profile):
        """Test scenario difficulty calculation"""
        difficulty = await progression_engine.calculate_scenario_difficulty(
            sample_scenario, agent_profile
        )
        
        assert isinstance(difficulty, DifficultyProfile)
        assert 0.0 <= difficulty.overall_difficulty <= 1.0
        assert 0.0 <= difficulty.cognitive_load <= 1.0
        assert 0.0 <= difficulty.stress_level <= 1.0
        
        # Check that all metrics are within valid range
        assert 0.0 <= difficulty.complexity_score <= 1.0
        assert 0.0 <= difficulty.time_pressure <= 1.0
        assert 0.0 <= difficulty.resource_constraints <= 1.0
    
    @pytest.mark.asyncio
    async def test_progression_path_creation(self, progression_engine):
        """Test progression path creation"""
        path = ProgressionPath(
            path_id="test_path",
            name="Test Progression Path",
            description="A test progression path",
            strategy=ProgressionStrategy.MASTERY_BASED,
            target_category=ScenarioCategory.RECONNAISSANCE,
            stages=[
                {"name": "Beginner", "difficulty_increase": 0.0},
                {"name": "Intermediate", "difficulty_increase": 0.2},
                {"name": "Advanced", "difficulty_increase": 0.4}
            ]
        )
        
        path_id = await progression_engine.create_progression_path(path)
        assert path_id == "test_path"
        assert path_id in progression_engine.progression_paths
    
    @pytest.mark.asyncio
    async def test_agent_progression_start(self, progression_engine):
        """Test starting an agent on a progression path"""
        # Create a test progression path first
        path = ProgressionPath(
            path_id="test_path",
            name="Test Path",
            description="Test progression path",
            strategy=ProgressionStrategy.MASTERY_BASED,
            target_category=ScenarioCategory.RECONNAISSANCE
        )
        await progression_engine.create_progression_path(path)
        
        # Start agent progression
        success = await progression_engine.start_agent_progression(
            "test_agent", "test_path"
        )
        
        assert success is True
        assert "test_agent" in progression_engine.agent_progressions
        assert "test_path" in progression_engine.agent_progressions["test_agent"]
    
    @pytest.mark.asyncio
    async def test_progression_update(self, progression_engine):
        """Test progression updates based on scenario results"""
        # Setup progression
        path = ProgressionPath(
            path_id="test_path",
            name="Test Path",
            description="Test progression path",
            strategy=ProgressionStrategy.MASTERY_BASED,
            target_category=ScenarioCategory.RECONNAISSANCE,
            mastery_threshold=0.8,
            min_attempts=3
        )
        await progression_engine.create_progression_path(path)
        await progression_engine.start_agent_progression("test_agent", "test_path")
        
        # Simulate successful scenario results
        for i in range(4):  # 4 successful attempts
            scenario_result = {
                "success": True,
                "completion_time": 1800,
                "score": 85,
                "confidence": 0.8
            }
            
            update_result = await progression_engine.update_progression(
                "test_agent", "test_path", scenario_result
            )
            
            assert update_result["progression_updated"] is True
        
        # Check that progression was updated
        progression = progression_engine.agent_progressions["test_agent"]["test_path"]
        assert progression.stage_attempts == 4
        assert progression.stage_successes == 4
        assert progression.stage_success_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_difficulty_adaptation(self, progression_engine):
        """Test difficulty adaptation based on performance"""
        # Setup progression
        path = ProgressionPath(
            path_id="test_path",
            name="Test Path",
            description="Test progression path",
            strategy=ProgressionStrategy.ADAPTIVE,
            target_category=ScenarioCategory.RECONNAISSANCE
        )
        await progression_engine.create_progression_path(path)
        await progression_engine.start_agent_progression("test_agent", "test_path")
        
        progression = progression_engine.agent_progressions["test_agent"]["test_path"]
        
        # Simulate consistently high performance (should increase difficulty)
        progression.stage_attempts = 5
        progression.stage_successes = 5
        progression.stage_success_rate = 1.0
        
        adaptation = await progression_engine._evaluate_difficulty_adaptation(progression, path)
        
        assert adaptation["adapt"] is True
        assert adaptation["direction"] == "increase"


class TestScenarioValidation:
    """Test scenario validation functionality"""
    
    @pytest.fixture
    async def validator(self):
        """Create a scenario validator for testing"""
        validator = ScenarioValidator()
        await validator.initialize()
        return validator
    
    @pytest.fixture
    def valid_scenario(self):
        """Create a valid scenario for testing"""
        return ScenarioTemplate(
            template_id=str(uuid.uuid4()),
            name="Valid Test Scenario",
            description="A valid test scenario with proper structure",
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.RECONNAISSANCE,
            complexity=ComplexityLevel.INTERMEDIATE,
            estimated_duration=timedelta(hours=2),
            objectives=[
                ScenarioObjective(
                    objective_id="obj1",
                    name="Test Objective",
                    description="A test objective",
                    success_criteria=["Complete task successfully"]
                )
            ],
            parameters=[
                ScenarioParameter(
                    name="test_param",
                    parameter_type="string",
                    default_value="test_value",
                    description="A test parameter"
                )
            ]
        )
    
    @pytest.fixture
    def invalid_scenario(self):
        """Create an invalid scenario for testing"""
        return ScenarioTemplate(
            template_id=str(uuid.uuid4()),
            name="",  # Invalid: empty name
            description="",  # Invalid: empty description
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.RECONNAISSANCE,
            complexity=ComplexityLevel.INTERMEDIATE,
            objectives=[],  # Invalid: no objectives
            parameters=[
                ScenarioParameter(
                    name="",  # Invalid: empty name
                    parameter_type="invalid_type",  # Invalid: bad type
                    default_value="test"
                )
            ]
        )
    
    @pytest.mark.asyncio
    async def test_valid_scenario_validation(self, validator, valid_scenario):
        """Test validation of a valid scenario"""
        result = await validator.validate_scenario(valid_scenario, ValidationLevel.STANDARD)
        
        assert result.valid is True
        assert len(result.failed_checks) == 0
        assert result.validation_level == ValidationLevel.STANDARD
        assert result.checks_performed > 0
    
    @pytest.mark.asyncio
    async def test_invalid_scenario_validation(self, validator, invalid_scenario):
        """Test validation of an invalid scenario"""
        result = await validator.validate_scenario(invalid_scenario, ValidationLevel.STANDARD)
        
        assert result.valid is False
        assert len(result.failed_checks) > 0
        assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_validation_levels(self, validator, valid_scenario):
        """Test different validation levels"""
        # Basic validation
        basic_result = await validator.validate_scenario(valid_scenario, ValidationLevel.BASIC)
        
        # Comprehensive validation
        comprehensive_result = await validator.validate_scenario(
            valid_scenario, ValidationLevel.COMPREHENSIVE
        )
        
        # Comprehensive should perform more checks
        assert comprehensive_result.checks_performed >= basic_result.checks_performed
    
    @pytest.mark.asyncio
    async def test_generated_scenario_validation(self, validator, valid_scenario):
        """Test validation of generated scenarios"""
        generated = GeneratedScenario(
            scenario=valid_scenario,
            generation_rationale="Test generation for validation",
            confidence_score=0.8,
            generation_type=GenerationType.ADAPTIVE_LEARNING,
            target_outcomes=[LearningOutcome.SKILL_MASTERY],
            expected_difficulty=0.6,
            novelty_score=0.5,
            success_probability=0.7,
            learning_potential=0.8,
            engagement_score=0.7
        )
        
        result = await validator.validate_generated_scenario(generated)
        
        assert isinstance(result, ValidationResult)
        # Should pass validation for well-formed generated scenario
        assert result.valid is True
    
    @pytest.mark.asyncio
    async def test_custom_validator_registration(self, validator):
        """Test registration of custom validators"""
        async def custom_validator(scenario, level):
            return {
                "name": "Custom Test Validator",
                "passed": True,
                "warnings": []
            }
        
        await validator.register_custom_validator(
            "custom_test", custom_validator, ValidationCategory.LOGIC
        )
        
        assert "custom_test" in validator.custom_validators
        assert custom_validator in validator.validation_rules[ValidationCategory.LOGIC]


class TestScenarioTester:
    """Test scenario testing framework functionality"""
    
    @pytest.fixture
    async def tester(self):
        """Create a scenario tester for testing"""
        tester = ScenarioTester()
        await tester.initialize()
        return tester
    
    @pytest.fixture
    def sample_test_case(self):
        """Create a sample test case"""
        async def test_function(test_data):
            return test_data.get("expected", True)
        
        return TestCase(
            test_id="test_case_1",
            name="Sample Test Case",
            description="A sample test case for testing",
            test_type=TestType.UNIT_TEST,
            test_function=test_function,
            test_data={"expected": True},
            expected_result=True
        )
    
    @pytest.fixture
    def sample_test_suite(self, sample_test_case):
        """Create a sample test suite"""
        return TestSuite(
            suite_id="test_suite_1",
            name="Sample Test Suite",
            description="A sample test suite for testing",
            test_cases=[sample_test_case]
        )
    
    @pytest.mark.asyncio
    async def test_test_suite_creation(self, tester, sample_test_suite):
        """Test test suite creation"""
        suite_id = await tester.create_test_suite(sample_test_suite)
        
        assert suite_id == sample_test_suite.suite_id
        assert suite_id in tester.test_suites
    
    @pytest.mark.asyncio
    async def test_test_suite_execution(self, tester, sample_test_suite):
        """Test test suite execution"""
        await tester.create_test_suite(sample_test_suite)
        
        result = await tester.run_test_suite(sample_test_suite.suite_id)
        
        assert result["suite_id"] == sample_test_suite.suite_id
        assert result["total_tests"] == 1
        assert result["passed_tests"] == 1
        assert result["success_rate"] == 1.0
        assert len(result["results"]) == 1
    
    @pytest.mark.asyncio
    async def test_failing_test_case(self, tester):
        """Test handling of failing test cases"""
        async def failing_test_function(test_data):
            raise ValueError("Test failure")
        
        failing_test = TestCase(
            test_id="failing_test",
            name="Failing Test",
            description="A test that should fail",
            test_type=TestType.UNIT_TEST,
            test_function=failing_test_function
        )
        
        test_suite = TestSuite(
            suite_id="failing_suite",
            name="Failing Test Suite",
            description="A test suite with failing tests",
            test_cases=[failing_test]
        )
        
        await tester.create_test_suite(test_suite)
        result = await tester.run_test_suite(test_suite.suite_id)
        
        assert result["passed_tests"] == 0
        assert result["failed_tests"] == 1
        assert result["success_rate"] == 0.0


class TestIntegration:
    """Integration tests for the complete scenario system"""
    
    @pytest.fixture
    async def complete_system(self):
        """Create a complete scenario system for integration testing"""
        template_manager = ScenarioTemplateManager()
        generator = DynamicScenarioGenerator(template_manager)
        progression_engine = DifficultyProgressionEngine()
        validator = ScenarioValidator()
        tester = ScenarioTester()
        
        await template_manager.initialize()
        await generator.initialize()
        await progression_engine.initialize()
        await validator.initialize()
        await tester.initialize()
        
        return {
            "template_manager": template_manager,
            "generator": generator,
            "progression_engine": progression_engine,
            "validator": validator,
            "tester": tester
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_scenario_workflow(self, complete_system):
        """Test complete end-to-end scenario workflow"""
        template_manager = complete_system["template_manager"]
        generator = complete_system["generator"]
        validator = complete_system["validator"]
        
        # 1. Create agent profile
        agent_profile = AgentLearningProfile(
            agent_id="integration_test_agent",
            skill_levels={"reconnaissance": 0.5},
            identified_weaknesses=["exploitation"]
        )
        
        # 2. Generate scenario
        context = GenerationContext(
            target_agents=["integration_test_agent"],
            agent_profiles={"integration_test_agent": agent_profile},
            generation_type=GenerationType.WEAKNESS_TARGETING
        )
        
        with patch.object(generator, '_create_weakness_focused_template') as mock_create:
            mock_template = ScenarioTemplate(
                template_id=str(uuid.uuid4()),
                name="Generated Integration Test Scenario",
                description="A scenario generated for integration testing",
                scenario_type=ScenarioType.TRAINING,
                category=ScenarioCategory.INITIAL_ACCESS,
                complexity=ComplexityLevel.INTERMEDIATE,
                objectives=[
                    ScenarioObjective(
                        objective_id="obj1",
                        name="Test Exploitation",
                        description="Practice exploitation techniques",
                        success_criteria=["Successfully exploit vulnerability"]
                    )
                ]
            )
            mock_create.return_value = mock_template
            
            generated_scenario = await generator.generate_scenario(context)
        
        # 3. Validate generated scenario
        validation_result = await validator.validate_generated_scenario(generated_scenario)
        
        # 4. Verify workflow
        assert isinstance(generated_scenario, GeneratedScenario)
        assert generated_scenario.generation_type == GenerationType.WEAKNESS_TARGETING
        assert validation_result.valid is True
        
        # 5. Update agent profile based on results
        scenario_result = {
            "success": True,
            "completion_time": 2400,
            "confidence": 0.7,
            "category": ScenarioCategory.INITIAL_ACCESS,
            "skills_demonstrated": ["exploitation"],
            "lessons_learned": ["Improved exploitation techniques"]
        }
        
        await generator.update_agent_profile("integration_test_agent", scenario_result)
        
        # Verify profile was updated
        updated_profile = generator.agent_profiles["integration_test_agent"]
        assert "exploitation" in updated_profile.skill_levels
        assert len(updated_profile.performance_history) > 0
    
    @pytest.mark.asyncio
    async def test_difficulty_progression_integration(self, complete_system):
        """Test integration of difficulty progression with scenario generation"""
        progression_engine = complete_system["progression_engine"]
        
        # Create and start progression
        path = ProgressionPath(
            path_id="integration_test_path",
            name="Integration Test Path",
            description="A progression path for integration testing",
            strategy=ProgressionStrategy.MASTERY_BASED,
            target_category=ScenarioCategory.RECONNAISSANCE
        )
        
        await progression_engine.create_progression_path(path)
        await progression_engine.start_agent_progression("test_agent", "integration_test_path")
        
        # Get difficulty profile
        difficulty = await progression_engine.get_next_difficulty("test_agent", "integration_test_path")
        
        assert isinstance(difficulty, DifficultyProfile)
        assert 0.0 <= difficulty.overall_difficulty <= 1.0
        
        # Simulate progression updates
        for i in range(3):
            scenario_result = {
                "success": True,
                "completion_time": 1800,
                "score": 80,
                "confidence": 0.8
            }
            
            update_result = await progression_engine.update_progression(
                "test_agent", "integration_test_path", scenario_result
            )
            
            assert update_result["progression_updated"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])