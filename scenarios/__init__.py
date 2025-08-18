#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Scenario Generation System
Comprehensive scenario generation, validation, and management system
"""

from .scenario_templates import (
    ScenarioTemplate,
    ScenarioTemplateManager,
    ScenarioInstance,
    ScenarioType,
    ScenarioCategory,
    ComplexityLevel,
    NetworkTopology,
    ScenarioParameter,
    ScenarioObjective,
    ScenarioAsset
)

from .dynamic_generation import (
    DynamicScenarioGenerator,
    GenerationContext,
    AgentLearningProfile,
    GenerationType,
    LearningOutcome,
    GeneratedScenario
)

from .difficulty_progression import (
    DifficultyProgressionEngine,
    ProgressionPath,
    ProgressionState,
    DifficultyProfile,
    ProgressionStrategy,
    DifficultyMetric
)

from .scenario_validation import (
    ScenarioValidator,
    ScenarioTester,
    ValidationLevel,
    ValidationCategory,
    ValidationResult,
    TestType,
    TestCase,
    TestSuite,
    TestResult
)

__all__ = [
    # Template system
    'ScenarioTemplate',
    'ScenarioTemplateManager',
    'ScenarioInstance',
    'ScenarioType',
    'ScenarioCategory',
    'ComplexityLevel',
    'NetworkTopology',
    'ScenarioParameter',
    'ScenarioObjective',
    'ScenarioAsset',
    
    # Dynamic generation
    'DynamicScenarioGenerator',
    'GenerationContext',
    'AgentLearningProfile',
    'GenerationType',
    'LearningOutcome',
    'GeneratedScenario',
    
    # Difficulty progression
    'DifficultyProgressionEngine',
    'ProgressionPath',
    'ProgressionState',
    'DifficultyProfile',
    'ProgressionStrategy',
    'DifficultyMetric',
    
    # Validation and testing
    'ScenarioValidator',
    'ScenarioTester',
    'ValidationLevel',
    'ValidationCategory',
    'ValidationResult',
    'TestType',
    'TestCase',
    'TestSuite',
    'TestResult'
]

__version__ = "1.0.0"
__author__ = "Archangel System"
__description__ = "Comprehensive scenario generation and management system for autonomous AI training"