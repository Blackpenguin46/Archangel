#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Integrated Scenario System
Complete scenario generation, validation, and management system integration
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .scenario_templates import ScenarioTemplateManager, ScenarioTemplate, ScenarioInstance
from .dynamic_generation import DynamicScenarioGenerator, GenerationContext, AgentLearningProfile
from .difficulty_progression import DifficultyProgressionEngine, DifficultyProfile
from .scenario_validation import ScenarioValidator, ScenarioTester, ValidationLevel

logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    """System status states"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class ScenarioRequest:
    """Request for scenario generation"""
    request_id: str
    agent_ids: List[str]
    
    # Generation preferences
    generation_type: Optional[str] = None
    target_category: Optional[str] = None
    complexity_preference: Optional[str] = None
    
    # Constraints
    max_duration: Optional[timedelta] = None
    max_participants: Optional[int] = None
    
    # Context
    learning_objectives: List[str] = field(default_factory=list)
    previous_scenarios: List[str] = field(default_factory=list)
    
    # Metadata
    requested_at: datetime = field(default_factory=datetime.now)
    requested_by: str = "system"
    priority: str = "normal"  # low, normal, high, urgent

@dataclass
class ScenarioResponse:
    """Response to scenario generation request"""
    request_id: str
    success: bool
    
    # Generated content
    scenario: Optional[ScenarioTemplate] = None
    instance: Optional[ScenarioInstance] = None
    difficulty_profile: Optional[DifficultyProfile] = None
    
    # Generation metadata
    generation_time: float = 0.0
    confidence_score: float = 0.0
    validation_passed: bool = False
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)

class IntegratedScenarioSystem:
    """
    Integrated scenario generation and management system.
    
    Features:
    - Unified interface for all scenario operations
    - Automatic validation and quality assurance
    - Performance monitoring and optimization
    - Agent learning integration
    - Comprehensive error handling and recovery
    """
    
    def __init__(self):
        # Core components
        self.template_manager: Optional[ScenarioTemplateManager] = None
        self.generator: Optional[DynamicScenarioGenerator] = None
        self.progression_engine: Optional[DifficultyProgressionEngine] = None
        self.validator: Optional[ScenarioValidator] = None
        self.tester: Optional[ScenarioTester] = None
        
        # System state
        self.status = SystemStatus.INITIALIZING
        self.initialization_time: Optional[datetime] = None
        
        # Request tracking
        self.active_requests: Dict[str, ScenarioRequest] = {}
        self.request_history: List[ScenarioResponse] = []
        
        # Performance metrics
        self.system_metrics = {
            "total_requests": 0,
            "successful_generations": 0,
            "average_generation_time": 0.0,
            "validation_success_rate": 0.0,
            "system_uptime": 0.0
        }
        
        # Configuration
        self.config = {
            "auto_validation": True,
            "validation_level": ValidationLevel.STANDARD,
            "max_concurrent_requests": 10,
            "request_timeout": timedelta(minutes=5),
            "cache_scenarios": True,
            "enable_progression": True
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the integrated scenario system"""
        try:
            self.logger.info("Initializing integrated scenario system")
            self.status = SystemStatus.INITIALIZING
            
            # Initialize core components
            self.template_manager = ScenarioTemplateManager()
            await self.template_manager.initialize()
            
            self.generator = DynamicScenarioGenerator(self.template_manager)
            await self.generator.initialize()
            
            self.progression_engine = DifficultyProgressionEngine()
            await self.progression_engine.initialize()
            
            self.validator = ScenarioValidator()
            await self.validator.initialize()
            
            self.tester = ScenarioTester()
            await self.tester.initialize()
            
            # System ready
            self.status = SystemStatus.READY
            self.initialization_time = datetime.now()
            
            self.logger.info("Integrated scenario system initialized successfully")
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            self.logger.error(f"Failed to initialize scenario system: {e}")
            raise
    
    async def generate_scenario(self, request: ScenarioRequest) -> ScenarioResponse:
        """Generate a scenario based on request"""
        try:
            if self.status != SystemStatus.READY:
                return ScenarioResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message=f"System not ready (status: {self.status.value})"
                )
            
            self.logger.info(f"Processing scenario request: {request.request_id}")
            start_time = datetime.now()
            
            # Track active request
            self.active_requests[request.request_id] = request
            
            try:
                # Build generation context
                context = await self._build_generation_context(request)
                
                # Generate scenario
                generated_scenario = await self.generator.generate_scenario(context)
                
                # Validate scenario if enabled
                validation_passed = True
                warnings = []
                
                if self.config["auto_validation"]:
                    validation_result = await self.validator.validate_generated_scenario(
                        generated_scenario, self.config["validation_level"]
                    )
                    validation_passed = validation_result.valid
                    warnings = validation_result.warnings
                
                # Calculate difficulty profile if progression enabled
                difficulty_profile = None
                if self.config["enable_progression"] and request.agent_ids:
                    # Use first agent for difficulty calculation
                    agent_id = request.agent_ids[0]
                    if agent_id in self.generator.agent_profiles:
                        agent_profile = self.generator.agent_profiles[agent_id]
                        difficulty_profile = await self.progression_engine.calculate_scenario_difficulty(
                            generated_scenario.scenario, agent_profile
                        )
                
                # Create scenario instance
                instance_params = await self._extract_instance_parameters(request, generated_scenario)
                instance = await self.template_manager.instantiate_template(
                    generated_scenario.scenario.template_id,
                    instance_params,
                    request.agent_ids
                )
                
                # Generate recommendations
                recommendations = await self._generate_system_recommendations(
                    request, generated_scenario, validation_passed
                )
                
                # Create response
                generation_time = (datetime.now() - start_time).total_seconds()
                
                response = ScenarioResponse(
                    request_id=request.request_id,
                    success=True,
                    scenario=generated_scenario.scenario,
                    instance=instance,
                    difficulty_profile=difficulty_profile,
                    generation_time=generation_time,
                    confidence_score=generated_scenario.confidence_score,
                    validation_passed=validation_passed,
                    recommendations=recommendations,
                    warnings=warnings
                )
                
                # Update metrics
                await self._update_system_metrics(response)
                
                self.logger.info(f"Successfully generated scenario: {generated_scenario.scenario.name}")
                return response
                
            finally:
                # Clean up active request
                if request.request_id in self.active_requests:
                    del self.active_requests[request.request_id]
                
        except Exception as e:
            self.logger.error(f"Failed to generate scenario: {e}")
            return ScenarioResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
                generation_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def update_agent_performance(self, 
                                     agent_id: str,
                                     scenario_id: str,
                                     performance_data: Dict[str, Any]) -> bool:
        """Update agent performance data"""
        try:
            # Update generator's agent profile
            await self.generator.update_agent_profile(agent_id, performance_data)
            
            # Update progression if enabled
            if self.config["enable_progression"]:
                # Find relevant progression paths
                if agent_id in self.progression_engine.agent_progressions:
                    for path_id in self.progression_engine.agent_progressions[agent_id]:
                        await self.progression_engine.update_progression(
                            agent_id, path_id, performance_data
                        )
            
            self.logger.info(f"Updated performance data for agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update agent performance: {e}")
            return False
    
    async def get_agent_recommendations(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get comprehensive recommendations for an agent"""
        try:
            recommendations = []
            
            # Get learning recommendations from generator
            learning_recs = await self.generator.get_learning_recommendations(agent_id)
            recommendations.extend(learning_recs)
            
            # Get progression recommendations if enabled
            if self.config["enable_progression"]:
                progression_recs = await self.progression_engine.get_progression_recommendations(agent_id)
                recommendations.extend(progression_recs)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get agent recommendations: {e}")
            return []
    
    async def validate_scenario(self, 
                              scenario: ScenarioTemplate,
                              validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, Any]:
        """Validate a scenario template"""
        try:
            validation_result = await self.validator.validate_scenario(scenario, validation_level)
            
            return {
                "valid": validation_result.valid,
                "passed_checks": validation_result.passed_checks,
                "failed_checks": validation_result.failed_checks,
                "warnings": validation_result.warnings,
                "recommendations": validation_result.recommendations,
                "validation_time": validation_result.validation_time
            }
            
        except Exception as e:
            self.logger.error(f"Failed to validate scenario: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def run_system_tests(self) -> Dict[str, Any]:
        """Run comprehensive system tests"""
        try:
            self.logger.info("Running system tests")
            
            # Run validator tests
            validation_suite_result = await self.tester.run_test_suite("scenario_validation")
            
            # Run additional system tests
            system_health = await self._check_system_health()
            
            return {
                "validation_tests": validation_suite_result,
                "system_health": system_health,
                "overall_status": "healthy" if system_health["healthy"] else "issues_detected"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to run system tests: {e}")
            return {
                "error": str(e),
                "overall_status": "error"
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            uptime = (datetime.now() - self.initialization_time).total_seconds() if self.initialization_time else 0
            
            return {
                "status": self.status.value,
                "uptime_seconds": uptime,
                "active_requests": len(self.active_requests),
                "total_requests": self.system_metrics["total_requests"],
                "success_rate": (self.system_metrics["successful_generations"] / 
                               max(1, self.system_metrics["total_requests"])),
                "average_generation_time": self.system_metrics["average_generation_time"],
                "components": {
                    "template_manager": self.template_manager is not None,
                    "generator": self.generator is not None,
                    "progression_engine": self.progression_engine is not None,
                    "validator": self.validator is not None,
                    "tester": self.tester is not None
                },
                "configuration": self.config
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _build_generation_context(self, request: ScenarioRequest) -> GenerationContext:
        """Build generation context from request"""
        try:
            # Get agent profiles
            agent_profiles = {}
            for agent_id in request.agent_ids:
                if agent_id in self.generator.agent_profiles:
                    agent_profiles[agent_id] = self.generator.agent_profiles[agent_id]
                else:
                    # Create default profile for new agent
                    agent_profiles[agent_id] = AgentLearningProfile(agent_id=agent_id)
            
            # Build context
            context = GenerationContext(
                target_agents=request.agent_ids,
                agent_profiles=agent_profiles,
                previous_scenarios=request.previous_scenarios
            )
            
            # Apply request preferences
            if request.generation_type:
                from .dynamic_generation import GenerationType
                context.generation_type = GenerationType(request.generation_type)
            
            if request.target_category:
                from .scenario_templates import ScenarioCategory
                context.allowed_categories = [ScenarioCategory(request.target_category)]
            
            if request.max_duration:
                context.time_constraint = request.max_duration
            
            # Set learning objectives
            if request.learning_objectives:
                from .dynamic_generation import LearningOutcome
                context.primary_objectives = [
                    LearningOutcome(obj) for obj in request.learning_objectives
                    if obj in [outcome.value for outcome in LearningOutcome]
                ]
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to build generation context: {e}")
            raise
    
    async def _extract_instance_parameters(self, 
                                         request: ScenarioRequest,
                                         generated_scenario) -> Dict[str, Any]:
        """Extract parameters for scenario instantiation"""
        try:
            parameters = {}
            
            # Use default parameter values from template
            for param in generated_scenario.scenario.parameters:
                parameters[param.name] = param.default_value
            
            # Apply any request-specific overrides
            # (This could be extended to accept parameter overrides in the request)
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"Failed to extract instance parameters: {e}")
            return {}
    
    async def _generate_system_recommendations(self, 
                                             request: ScenarioRequest,
                                             generated_scenario,
                                             validation_passed: bool) -> List[str]:
        """Generate system-level recommendations"""
        try:
            recommendations = []
            
            # Validation-based recommendations
            if not validation_passed:
                recommendations.append("Review scenario validation issues before execution")
            
            # Difficulty-based recommendations
            if generated_scenario.expected_difficulty > 0.8:
                recommendations.append("High difficulty scenario - consider providing additional support")
            elif generated_scenario.expected_difficulty < 0.3:
                recommendations.append("Low difficulty scenario - consider increasing challenge level")
            
            # Success probability recommendations
            if generated_scenario.success_probability < 0.4:
                recommendations.append("Low success probability - consider adjusting difficulty or providing hints")
            
            # Learning potential recommendations
            if generated_scenario.learning_potential > 0.8:
                recommendations.append("High learning potential - excellent choice for skill development")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    async def _update_system_metrics(self, response: ScenarioResponse) -> None:
        """Update system performance metrics"""
        try:
            self.system_metrics["total_requests"] += 1
            
            if response.success:
                self.system_metrics["successful_generations"] += 1
            
            # Update average generation time
            current_avg = self.system_metrics["average_generation_time"]
            total_requests = self.system_metrics["total_requests"]
            
            self.system_metrics["average_generation_time"] = (
                (current_avg * (total_requests - 1) + response.generation_time) / total_requests
            )
            
            # Update validation success rate
            if response.validation_passed:
                # Calculate validation success rate
                successful_validations = sum(1 for r in self.request_history if r.validation_passed)
                self.system_metrics["validation_success_rate"] = successful_validations / total_requests
            
            # Store response in history (keep last 1000)
            self.request_history.append(response)
            if len(self.request_history) > 1000:
                self.request_history = self.request_history[-1000:]
            
        except Exception as e:
            self.logger.error(f"Failed to update system metrics: {e}")
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        try:
            health_status = {
                "healthy": True,
                "issues": [],
                "component_status": {}
            }
            
            # Check component initialization
            components = {
                "template_manager": self.template_manager,
                "generator": self.generator,
                "progression_engine": self.progression_engine,
                "validator": self.validator,
                "tester": self.tester
            }
            
            for name, component in components.items():
                if component is None:
                    health_status["healthy"] = False
                    health_status["issues"].append(f"{name} not initialized")
                    health_status["component_status"][name] = "not_initialized"
                elif hasattr(component, 'initialized') and not component.initialized:
                    health_status["healthy"] = False
                    health_status["issues"].append(f"{name} not properly initialized")
                    health_status["component_status"][name] = "initialization_failed"
                else:
                    health_status["component_status"][name] = "healthy"
            
            # Check system metrics
            if self.system_metrics["total_requests"] > 0:
                success_rate = (self.system_metrics["successful_generations"] / 
                              self.system_metrics["total_requests"])
                if success_rate < 0.8:
                    health_status["issues"].append(f"Low success rate: {success_rate:.2%}")
                
                if self.system_metrics["average_generation_time"] > 30:  # 30 seconds
                    health_status["issues"].append("High average generation time")
            
            # Check active requests
            if len(self.active_requests) >= self.config["max_concurrent_requests"]:
                health_status["issues"].append("Maximum concurrent requests reached")
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Failed to check system health: {e}")
            return {
                "healthy": False,
                "issues": [f"Health check failed: {str(e)}"],
                "component_status": {}
            }