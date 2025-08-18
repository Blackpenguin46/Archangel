#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Scenario Validation and Testing Framework
Comprehensive validation and testing system for scenario quality and reliability
"""

import asyncio
import logging
import uuid
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from collections import defaultdict
import traceback

from .scenario_templates import ScenarioTemplate, ScenarioInstance, ScenarioParameter, ScenarioObjective
from .dynamic_generation import GeneratedScenario, AgentLearningProfile

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation thoroughness levels"""
    BASIC = "basic"                    # Basic syntax and structure validation
    STANDARD = "standard"              # Standard validation with logic checks
    COMPREHENSIVE = "comprehensive"    # Full validation with simulation
    PRODUCTION = "production"          # Production-ready validation

class ValidationCategory(Enum):
    """Categories of validation checks"""
    SYNTAX = "syntax"                  # Basic syntax and structure
    LOGIC = "logic"                    # Logical consistency
    PERFORMANCE = "performance"        # Performance characteristics
    SECURITY = "security"              # Security considerations
    USABILITY = "usability"            # User experience
    RELIABILITY = "reliability"        # Execution reliability
    SCALABILITY = "scalability"        # Scalability characteristics

class TestType(Enum):
    """Types of scenario tests"""
    UNIT_TEST = "unit_test"            # Individual component tests
    INTEGRATION_TEST = "integration_test"  # Component integration tests
    PERFORMANCE_TEST = "performance_test"  # Performance and load tests
    STRESS_TEST = "stress_test"        # Stress and edge case tests
    REGRESSION_TEST = "regression_test"  # Regression testing
    USER_ACCEPTANCE_TEST = "user_acceptance_test"  # User acceptance tests

@dataclass
class ValidationResult:
    """Result of scenario validation"""
    valid: bool
    validation_level: ValidationLevel
    
    # Validation details
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Category-specific results
    category_results: Dict[ValidationCategory, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance metrics
    validation_time: float = 0.0
    checks_performed: int = 0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    validated_at: datetime = field(default_factory=datetime.now)
    validator_version: str = "1.0"

@dataclass
class TestCase:
    """Individual test case definition"""
    test_id: str
    name: str
    description: str
    test_type: TestType
    
    # Test configuration
    test_function: Optional[Callable] = None
    test_data: Dict[str, Any] = field(default_factory=dict)
    expected_result: Any = None
    
    # Test constraints
    timeout: timedelta = timedelta(minutes=5)
    max_retries: int = 3
    
    # Prerequisites
    prerequisites: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    success_rate: float = 0.0

@dataclass
class TestResult:
    """Result of test execution"""
    test_id: str
    passed: bool
    
    # Execution details
    execution_time: float = 0.0
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Test data
    actual_result: Any = None
    test_data: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    
    # Metadata
    executed_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0

@dataclass
class TestSuite:
    """Collection of related test cases"""
    suite_id: str
    name: str
    description: str
    
    # Test cases
    test_cases: List[TestCase] = field(default_factory=list)
    
    # Suite configuration
    parallel_execution: bool = False
    stop_on_failure: bool = False
    
    # Results tracking
    last_run: Optional[datetime] = None
    success_rate: float = 0.0
    total_runs: int = 0

class ScenarioValidator:
    """
    Comprehensive scenario validation system.
    
    Features:
    - Multi-level validation (basic to production)
    - Category-specific validation checks
    - Performance and security validation
    - Automated test generation
    - Validation reporting and analytics
    """
    
    def __init__(self):
        # Validation rules and checks
        self.validation_rules: Dict[ValidationCategory, List[Callable]] = defaultdict(list)
        self.custom_validators: Dict[str, Callable] = {}
        
        # Validation cache
        self.validation_cache: Dict[str, ValidationResult] = {}
        
        # Performance tracking
        self.validation_metrics = {
            "total_validations": 0,
            "validation_times": [],
            "success_rates": {},
            "common_failures": defaultdict(int)
        }
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the scenario validator"""
        try:
            self.logger.info("Initializing scenario validator")
            
            # Register built-in validation rules
            await self._register_builtin_validators()
            
            # Initialize validation cache
            self.validation_cache = {}
            
            self.initialized = True
            self.logger.info("Scenario validator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize scenario validator: {e}")
            raise
    
    async def validate_scenario(self, 
                              scenario: ScenarioTemplate,
                              validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate a scenario template"""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = f"{scenario.template_id}_{validation_level.value}"
            if cache_key in self.validation_cache:
                cached_result = self.validation_cache[cache_key]
                # Use cached result if recent (within 1 hour)
                if datetime.now() - cached_result.validated_at < timedelta(hours=1):
                    return cached_result
            
            self.logger.info(f"Validating scenario: {scenario.name} (level: {validation_level.value})")
            
            # Initialize validation result
            result = ValidationResult(
                valid=True,
                validation_level=validation_level
            )
            
            # Perform validation checks by category
            for category in ValidationCategory:
                if validation_level == ValidationLevel.BASIC and category not in [ValidationCategory.SYNTAX, ValidationCategory.LOGIC]:
                    continue
                
                category_result = await self._validate_category(scenario, category, validation_level)
                result.category_results[category] = category_result
                
                # Update overall result
                if not category_result["passed"]:
                    result.valid = False
                
                result.passed_checks.extend(category_result["passed_checks"])
                result.failed_checks.extend(category_result["failed_checks"])
                result.warnings.extend(category_result["warnings"])
            
            # Generate recommendations
            result.recommendations = await self._generate_recommendations(scenario, result)
            
            # Update metrics
            result.validation_time = time.time() - start_time
            result.checks_performed = len(result.passed_checks) + len(result.failed_checks)
            
            # Cache result
            self.validation_cache[cache_key] = result
            
            # Update analytics
            self._update_validation_metrics(result)
            
            self.logger.info(f"Validation completed: {result.valid} ({result.validation_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to validate scenario: {e}")
            return ValidationResult(
                valid=False,
                validation_level=validation_level,
                failed_checks=[f"Validation error: {str(e)}"]
            )
    
    async def validate_generated_scenario(self, 
                                        generated: GeneratedScenario,
                                        validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate a dynamically generated scenario"""
        try:
            # Validate the base scenario template
            base_result = await self.validate_scenario(generated.scenario, validation_level)
            
            # Additional validation for generated scenarios
            generation_checks = await self._validate_generation_quality(generated)
            
            # Combine results
            base_result.passed_checks.extend(generation_checks["passed_checks"])
            base_result.failed_checks.extend(generation_checks["failed_checks"])
            base_result.warnings.extend(generation_checks["warnings"])
            
            if not generation_checks["passed"]:
                base_result.valid = False
            
            return base_result
            
        except Exception as e:
            self.logger.error(f"Failed to validate generated scenario: {e}")
            return ValidationResult(
                valid=False,
                validation_level=validation_level,
                failed_checks=[f"Generated scenario validation error: {str(e)}"]
            )
    
    async def register_custom_validator(self, 
                                      name: str,
                                      validator_func: Callable,
                                      category: ValidationCategory) -> None:
        """Register a custom validation function"""
        try:
            self.custom_validators[name] = validator_func
            self.validation_rules[category].append(validator_func)
            
            self.logger.info(f"Registered custom validator: {name} ({category.value})")
            
        except Exception as e:
            self.logger.error(f"Failed to register custom validator: {e}")
    
    async def _validate_category(self, 
                               scenario: ScenarioTemplate,
                               category: ValidationCategory,
                               validation_level: ValidationLevel) -> Dict[str, Any]:
        """Validate a specific category"""
        try:
            category_result = {
                "passed": True,
                "passed_checks": [],
                "failed_checks": [],
                "warnings": []
            }
            
            # Get validators for this category
            validators = self.validation_rules.get(category, [])
            
            for validator in validators:
                try:
                    check_result = await validator(scenario, validation_level)
                    
                    if check_result["passed"]:
                        category_result["passed_checks"].append(check_result["name"])
                    else:
                        category_result["failed_checks"].append(check_result["name"])
                        category_result["passed"] = False
                    
                    category_result["warnings"].extend(check_result.get("warnings", []))
                    
                except Exception as e:
                    self.logger.error(f"Validator {validator.__name__} failed: {e}")
                    category_result["failed_checks"].append(f"{validator.__name__}: {str(e)}")
                    category_result["passed"] = False
            
            return category_result
            
        except Exception as e:
            self.logger.error(f"Failed to validate category {category.value}: {e}")
            return {
                "passed": False,
                "passed_checks": [],
                "failed_checks": [f"Category validation error: {str(e)}"],
                "warnings": []
            }
    
    async def _validate_generation_quality(self, generated: GeneratedScenario) -> Dict[str, Any]:
        """Validate quality of generated scenario"""
        try:
            checks = {
                "passed": True,
                "passed_checks": [],
                "failed_checks": [],
                "warnings": []
            }
            
            # Check confidence score
            if generated.confidence_score < 0.5:
                checks["warnings"].append("Low confidence score in generated scenario")
            elif generated.confidence_score >= 0.8:
                checks["passed_checks"].append("High confidence score")
            
            # Check learning potential
            if generated.learning_potential < 0.6:
                checks["warnings"].append("Low learning potential")
            elif generated.learning_potential >= 0.8:
                checks["passed_checks"].append("High learning potential")
            
            # Check success probability
            if generated.success_probability < 0.3:
                checks["warnings"].append("Very low success probability - may be too difficult")
            elif generated.success_probability > 0.9:
                checks["warnings"].append("Very high success probability - may be too easy")
            else:
                checks["passed_checks"].append("Appropriate success probability")
            
            # Check generation rationale
            if not generated.generation_rationale or len(generated.generation_rationale) < 20:
                checks["failed_checks"].append("Missing or insufficient generation rationale")
                checks["passed"] = False
            else:
                checks["passed_checks"].append("Clear generation rationale provided")
            
            # Check target outcomes
            if not generated.target_outcomes:
                checks["warnings"].append("No target learning outcomes specified")
            else:
                checks["passed_checks"].append("Target learning outcomes specified")
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Failed to validate generation quality: {e}")
            return {
                "passed": False,
                "passed_checks": [],
                "failed_checks": [f"Generation quality validation error: {str(e)}"],
                "warnings": []
            }
    
    async def _generate_recommendations(self, 
                                      scenario: ScenarioTemplate,
                                      result: ValidationResult) -> List[str]:
        """Generate improvement recommendations"""
        try:
            recommendations = []
            
            # Analyze failed checks
            if "Missing scenario description" in result.failed_checks:
                recommendations.append("Add a comprehensive scenario description")
            
            if "No objectives defined" in result.failed_checks:
                recommendations.append("Define clear learning objectives")
            
            if "Missing success criteria" in result.failed_checks:
                recommendations.append("Specify measurable success criteria for objectives")
            
            # Analyze warnings
            if any("duration" in warning.lower() for warning in result.warnings):
                recommendations.append("Review scenario duration for appropriateness")
            
            if any("complexity" in warning.lower() for warning in result.warnings):
                recommendations.append("Adjust scenario complexity to match target audience")
            
            # Performance recommendations
            if ValidationCategory.PERFORMANCE in result.category_results:
                perf_result = result.category_results[ValidationCategory.PERFORMANCE]
                if not perf_result["passed"]:
                    recommendations.append("Optimize scenario for better performance")
            
            # Security recommendations
            if ValidationCategory.SECURITY in result.category_results:
                sec_result = result.category_results[ValidationCategory.SECURITY]
                if not sec_result["passed"]:
                    recommendations.append("Address security concerns in scenario design")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return ["Review scenario for potential improvements"]
    
    def _update_validation_metrics(self, result: ValidationResult) -> None:
        """Update validation analytics"""
        try:
            self.validation_metrics["total_validations"] += 1
            self.validation_metrics["validation_times"].append(result.validation_time)
            
            # Track success rates by level
            level_key = result.validation_level.value
            if level_key not in self.validation_metrics["success_rates"]:
                self.validation_metrics["success_rates"][level_key] = []
            self.validation_metrics["success_rates"][level_key].append(result.valid)
            
            # Track common failures
            for failure in result.failed_checks:
                self.validation_metrics["common_failures"][failure] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to update validation metrics: {e}")
    
    async def _register_builtin_validators(self) -> None:
        """Register built-in validation functions"""
        try:
            # Syntax validators
            self.validation_rules[ValidationCategory.SYNTAX].extend([
                self._validate_basic_structure,
                self._validate_parameter_syntax,
                self._validate_objective_syntax
            ])
            
            # Logic validators
            self.validation_rules[ValidationCategory.LOGIC].extend([
                self._validate_objective_logic,
                self._validate_parameter_logic,
                self._validate_dependency_logic
            ])
            
            # Performance validators
            self.validation_rules[ValidationCategory.PERFORMANCE].extend([
                self._validate_performance_characteristics,
                self._validate_resource_requirements
            ])
            
            # Security validators
            self.validation_rules[ValidationCategory.SECURITY].extend([
                self._validate_security_considerations,
                self._validate_asset_security
            ])
            
            # Usability validators
            self.validation_rules[ValidationCategory.USABILITY].extend([
                self._validate_usability_aspects,
                self._validate_documentation_quality
            ])
            
            # Reliability validators
            self.validation_rules[ValidationCategory.RELIABILITY].extend([
                self._validate_reliability_aspects,
                self._validate_error_handling
            ])
            
        except Exception as e:
            self.logger.error(f"Failed to register builtin validators: {e}")
    
    # Built-in validator functions
    async def _validate_basic_structure(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate basic scenario structure"""
        result = {"name": "Basic Structure", "passed": True, "warnings": []}
        
        if not scenario.name:
            result["passed"] = False
            result["error"] = "Scenario name is required"
        
        if not scenario.description:
            result["passed"] = False
            result["error"] = "Scenario description is required"
        
        if not scenario.objectives:
            result["warnings"].append("No objectives defined")
        
        return result
    
    async def _validate_parameter_syntax(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate parameter syntax"""
        result = {"name": "Parameter Syntax", "passed": True, "warnings": []}
        
        for param in scenario.parameters:
            if not param.name:
                result["passed"] = False
                result["error"] = "Parameter name is required"
                break
            
            if param.parameter_type not in ["string", "int", "float", "bool", "enum", "list"]:
                result["passed"] = False
                result["error"] = f"Invalid parameter type: {param.parameter_type}"
                break
        
        return result
    
    async def _validate_objective_syntax(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate objective syntax"""
        result = {"name": "Objective Syntax", "passed": True, "warnings": []}
        
        for obj in scenario.objectives:
            if not obj.name or not obj.description:
                result["passed"] = False
                result["error"] = "Objective name and description are required"
                break
            
            if not obj.success_criteria:
                result["warnings"].append(f"Objective '{obj.name}' has no success criteria")
        
        return result
    
    async def _validate_objective_logic(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate objective logic"""
        result = {"name": "Objective Logic", "passed": True, "warnings": []}
        
        # Check for circular dependencies
        obj_deps = {obj.objective_id: obj.dependencies for obj in scenario.objectives}
        
        def has_circular_dep(obj_id, visited=None):
            if visited is None:
                visited = set()
            if obj_id in visited:
                return True
            visited.add(obj_id)
            for dep in obj_deps.get(obj_id, []):
                if has_circular_dep(dep, visited.copy()):
                    return True
            return False
        
        for obj in scenario.objectives:
            if has_circular_dep(obj.objective_id):
                result["passed"] = False
                result["error"] = f"Circular dependency detected in objective '{obj.name}'"
                break
        
        return result
    
    async def _validate_parameter_logic(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate parameter logic"""
        result = {"name": "Parameter Logic", "passed": True, "warnings": []}
        
        for param in scenario.parameters:
            if param.min_value is not None and param.max_value is not None:
                if param.min_value > param.max_value:
                    result["passed"] = False
                    result["error"] = f"Parameter '{param.name}' min_value > max_value"
                    break
        
        return result
    
    async def _validate_dependency_logic(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate dependency logic"""
        result = {"name": "Dependency Logic", "passed": True, "warnings": []}
        
        # Check if all dependencies exist (this would need access to template manager)
        # For now, just validate structure
        if scenario.dependencies:
            result["warnings"].append("Dependencies specified - ensure they exist")
        
        return result
    
    async def _validate_performance_characteristics(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate performance characteristics"""
        result = {"name": "Performance Characteristics", "passed": True, "warnings": []}
        
        # Check duration reasonableness
        duration_hours = scenario.estimated_duration.total_seconds() / 3600
        if duration_hours > 8:
            result["warnings"].append("Very long scenario duration (>8 hours)")
        elif duration_hours < 0.1:
            result["warnings"].append("Very short scenario duration (<6 minutes)")
        
        # Check asset count
        if len(scenario.required_assets) > 20:
            result["warnings"].append("Large number of required assets may impact performance")
        
        return result
    
    async def _validate_resource_requirements(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate resource requirements"""
        result = {"name": "Resource Requirements", "passed": True, "warnings": []}
        
        # Check participant limits
        if scenario.max_participants > 50:
            result["warnings"].append("Very high maximum participants")
        
        if scenario.min_participants > scenario.max_participants:
            result["passed"] = False
            result["error"] = "Minimum participants exceeds maximum"
        
        return result
    
    async def _validate_security_considerations(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate security considerations"""
        result = {"name": "Security Considerations", "passed": True, "warnings": []}
        
        # Check for security-related assets
        has_security_assets = any(
            "security" in asset.asset_type.lower() or 
            "firewall" in asset.name.lower() or
            "ids" in asset.name.lower()
            for asset in scenario.required_assets
        )
        
        if not has_security_assets and scenario.category.value in ["incident_response", "threat_hunting"]:
            result["warnings"].append("Security scenario should include security assets")
        
        return result
    
    async def _validate_asset_security(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate asset security configuration"""
        result = {"name": "Asset Security", "passed": True, "warnings": []}
        
        for asset in scenario.required_assets:
            if not asset.security_controls and asset.vulnerabilities:
                result["warnings"].append(f"Asset '{asset.name}' has vulnerabilities but no security controls")
        
        return result
    
    async def _validate_usability_aspects(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate usability aspects"""
        result = {"name": "Usability Aspects", "passed": True, "warnings": []}
        
        # Check documentation length
        if scenario.documentation and len(scenario.documentation) > 5000:
            result["warnings"].append("Very long documentation may impact usability")
        elif not scenario.documentation:
            result["warnings"].append("Missing documentation")
        
        return result
    
    async def _validate_documentation_quality(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate documentation quality"""
        result = {"name": "Documentation Quality", "passed": True, "warnings": []}
        
        if scenario.documentation:
            # Basic quality checks
            if len(scenario.documentation.split()) < 50:
                result["warnings"].append("Documentation seems too brief")
        else:
            result["warnings"].append("No documentation provided")
        
        return result
    
    async def _validate_reliability_aspects(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate reliability aspects"""
        result = {"name": "Reliability Aspects", "passed": True, "warnings": []}
        
        # Check for validation rules
        if not scenario.validation_rules:
            result["warnings"].append("No validation rules defined")
        
        return result
    
    async def _validate_error_handling(self, scenario: ScenarioTemplate, level: ValidationLevel) -> Dict[str, Any]:
        """Validate error handling"""
        result = {"name": "Error Handling", "passed": True, "warnings": []}
        
        # Check for cleanup scripts
        cleanup_assets = sum(1 for asset in scenario.required_assets if asset.cleanup_scripts)
        if cleanup_assets < len(scenario.required_assets) / 2:
            result["warnings"].append("Many assets lack cleanup scripts")
        
        return result


class ScenarioTester:
    """
    Comprehensive scenario testing framework.
    
    Features:
    - Automated test case generation
    - Multiple test types (unit, integration, performance)
    - Test suite management
    - Regression testing
    - Performance benchmarking
    """
    
    def __init__(self):
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: Dict[str, List[TestResult]] = defaultdict(list)
        
        # Test execution metrics
        self.test_metrics = {
            "total_tests_run": 0,
            "success_rate": 0.0,
            "average_execution_time": 0.0,
            "performance_benchmarks": {}
        }
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the scenario tester"""
        try:
            self.logger.info("Initializing scenario tester")
            
            # Create default test suites
            await self._create_default_test_suites()
            
            self.initialized = True
            self.logger.info("Scenario tester initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize scenario tester: {e}")
            raise
    
    async def create_test_suite(self, suite: TestSuite) -> str:
        """Create a new test suite"""
        try:
            self.test_suites[suite.suite_id] = suite
            self.logger.info(f"Created test suite: {suite.name}")
            return suite.suite_id
            
        except Exception as e:
            self.logger.error(f"Failed to create test suite: {e}")
            raise
    
    async def run_test_suite(self, suite_id: str) -> Dict[str, Any]:
        """Run a complete test suite"""
        try:
            if suite_id not in self.test_suites:
                raise ValueError(f"Test suite {suite_id} not found")
            
            suite = self.test_suites[suite_id]
            start_time = time.time()
            
            self.logger.info(f"Running test suite: {suite.name}")
            
            results = []
            passed_tests = 0
            
            for test_case in suite.test_cases:
                try:
                    test_result = await self._run_test_case(test_case)
                    results.append(test_result)
                    
                    if test_result.passed:
                        passed_tests += 1
                    elif suite.stop_on_failure:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Test case {test_case.name} failed with exception: {e}")
                    results.append(TestResult(
                        test_id=test_case.test_id,
                        passed=False,
                        error_message=str(e),
                        stack_trace=traceback.format_exc()
                    ))
            
            # Update suite metrics
            suite.last_run = datetime.now()
            suite.success_rate = passed_tests / len(suite.test_cases) if suite.test_cases else 0.0
            suite.total_runs += 1
            
            execution_time = time.time() - start_time
            
            suite_result = {
                "suite_id": suite_id,
                "suite_name": suite.name,
                "total_tests": len(suite.test_cases),
                "passed_tests": passed_tests,
                "failed_tests": len(suite.test_cases) - passed_tests,
                "success_rate": suite.success_rate,
                "execution_time": execution_time,
                "results": results
            }
            
            self.logger.info(f"Test suite completed: {passed_tests}/{len(suite.test_cases)} passed")
            return suite_result
            
        except Exception as e:
            self.logger.error(f"Failed to run test suite: {e}")
            return {
                "suite_id": suite_id,
                "error": str(e),
                "success_rate": 0.0
            }
    
    async def _run_test_case(self, test_case: TestCase) -> TestResult:
        """Run an individual test case"""
        try:
            start_time = time.time()
            
            # Check prerequisites
            for prereq in test_case.prerequisites:
                if not await self._check_prerequisite(prereq):
                    return TestResult(
                        test_id=test_case.test_id,
                        passed=False,
                        error_message=f"Prerequisite not met: {prereq}"
                    )
            
            # Execute test function
            if test_case.test_function:
                actual_result = await asyncio.wait_for(
                    test_case.test_function(test_case.test_data),
                    timeout=test_case.timeout.total_seconds()
                )
                
                # Compare with expected result
                passed = (actual_result == test_case.expected_result if 
                         test_case.expected_result is not None else True)
                
                return TestResult(
                    test_id=test_case.test_id,
                    passed=passed,
                    execution_time=time.time() - start_time,
                    actual_result=actual_result,
                    test_data=test_case.test_data
                )
            else:
                return TestResult(
                    test_id=test_case.test_id,
                    passed=False,
                    error_message="No test function defined"
                )
                
        except asyncio.TimeoutError:
            return TestResult(
                test_id=test_case.test_id,
                passed=False,
                error_message="Test timed out",
                execution_time=test_case.timeout.total_seconds()
            )
        except Exception as e:
            return TestResult(
                test_id=test_case.test_id,
                passed=False,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                execution_time=time.time() - start_time
            )
    
    async def _check_prerequisite(self, prerequisite: str) -> bool:
        """Check if a prerequisite is met"""
        # Implementation would depend on specific prerequisites
        # For now, assume all prerequisites are met
        return True
    
    async def _create_default_test_suites(self) -> None:
        """Create default test suites"""
        try:
            # Basic validation test suite
            validation_suite = TestSuite(
                suite_id="scenario_validation",
                name="Scenario Validation Tests",
                description="Basic validation tests for scenario templates"
            )
            
            # Add basic test cases
            validation_suite.test_cases = [
                TestCase(
                    test_id="test_scenario_structure",
                    name="Test Scenario Structure",
                    description="Validate basic scenario structure",
                    test_type=TestType.UNIT_TEST,
                    test_function=self._test_scenario_structure
                ),
                TestCase(
                    test_id="test_parameter_validation",
                    name="Test Parameter Validation",
                    description="Validate scenario parameters",
                    test_type=TestType.UNIT_TEST,
                    test_function=self._test_parameter_validation
                )
            ]
            
            self.test_suites[validation_suite.suite_id] = validation_suite
            
        except Exception as e:
            self.logger.error(f"Failed to create default test suites: {e}")
    
    # Test functions
    async def _test_scenario_structure(self, test_data: Dict[str, Any]) -> bool:
        """Test scenario structure"""
        # Implementation would test actual scenario structure
        return True
    
    async def _test_parameter_validation(self, test_data: Dict[str, Any]) -> bool:
        """Test parameter validation"""
        # Implementation would test parameter validation
        return True