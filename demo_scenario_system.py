#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Scenario System Demo
Demonstration of the complete scenario generation and management system
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta

from scenarios.scenario_system import IntegratedScenarioSystem, ScenarioRequest
from scenarios.scenario_templates import ScenarioType, ScenarioCategory, ComplexityLevel
from scenarios.dynamic_generation import GenerationType, LearningOutcome
from scenarios.difficulty_progression import ProgressionStrategy
from scenarios.scenario_validation import ValidationLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_basic_scenario_generation():
    """Demonstrate basic scenario generation"""
    print("\n" + "="*60)
    print("DEMO: Basic Scenario Generation")
    print("="*60)
    
    # Initialize the integrated scenario system
    system = IntegratedScenarioSystem()
    await system.initialize()
    
    # Create a scenario request
    request = ScenarioRequest(
        request_id=str(uuid.uuid4()),
        agent_ids=["demo_agent_1"],
        generation_type=GenerationType.ADAPTIVE_LEARNING.value,
        target_category=ScenarioCategory.RECONNAISSANCE.value,
        complexity_preference=ComplexityLevel.INTERMEDIATE.value,
        learning_objectives=[LearningOutcome.SKILL_MASTERY.value],
        requested_by="demo_user"
    )
    
    print(f"Generating scenario for agent: {request.agent_ids[0]}")
    print(f"Target category: {request.target_category}")
    print(f"Generation type: {request.generation_type}")
    
    # Generate scenario
    response = await system.generate_scenario(request)
    
    if response.success:
        print(f"\n✅ Scenario generated successfully!")
        print(f"   Name: {response.scenario.name}")
        print(f"   Description: {response.scenario.description}")
        print(f"   Complexity: {response.scenario.complexity.value}")
        print(f"   Duration: {response.scenario.estimated_duration}")
        print(f"   Objectives: {len(response.scenario.objectives)}")
        print(f"   Generation time: {response.generation_time:.2f}s")
        print(f"   Confidence score: {response.confidence_score:.2f}")
        print(f"   Validation passed: {response.validation_passed}")
        
        if response.difficulty_profile:
            print(f"   Overall difficulty: {response.difficulty_profile.overall_difficulty:.2f}")
            print(f"   Cognitive load: {response.difficulty_profile.cognitive_load:.2f}")
            print(f"   Stress level: {response.difficulty_profile.stress_level:.2f}")
        
        if response.recommendations:
            print(f"   Recommendations:")
            for rec in response.recommendations:
                print(f"     - {rec}")
    else:
        print(f"❌ Scenario generation failed: {response.error_message}")
    
    return system, response

async def demo_agent_learning_progression():
    """Demonstrate agent learning progression"""
    print("\n" + "="*60)
    print("DEMO: Agent Learning Progression")
    print("="*60)
    
    system = IntegratedScenarioSystem()
    await system.initialize()
    
    agent_id = "learning_agent"
    
    # Simulate multiple scenario completions with learning progression
    scenarios_completed = 0
    
    for i in range(5):
        print(f"\n--- Scenario {i+1} ---")
        
        # Generate scenario
        request = ScenarioRequest(
            request_id=str(uuid.uuid4()),
            agent_ids=[agent_id],
            generation_type=GenerationType.ADAPTIVE_LEARNING.value,
            target_category=ScenarioCategory.RECONNAISSANCE.value
        )
        
        response = await system.generate_scenario(request)
        
        if response.success:
            scenarios_completed += 1
            print(f"Generated: {response.scenario.name}")
            print(f"Difficulty: {response.difficulty_profile.overall_difficulty:.2f}" if response.difficulty_profile else "N/A")
            
            # Simulate scenario completion with improving performance
            success_rate = min(0.9, 0.4 + (i * 0.15))  # Gradually improving
            completion_time = max(1800, 3600 - (i * 300))  # Getting faster
            
            performance_data = {
                "success": True if success_rate > 0.5 else False,
                "completion_time": completion_time,
                "confidence": success_rate,
                "category": ScenarioCategory.RECONNAISSANCE,
                "skills_demonstrated": ["reconnaissance", "network_analysis"],
                "lessons_learned": [f"Lesson from scenario {i+1}"]
            }
            
            # Update agent performance
            await system.update_agent_performance(
                agent_id, response.instance.instance_id, performance_data
            )
            
            print(f"Performance: Success={performance_data['success']}, "
                  f"Time={completion_time}s, Confidence={success_rate:.2f}")
        
        # Get updated recommendations
        recommendations = await system.get_agent_recommendations(agent_id)
        if recommendations:
            print("Updated recommendations:")
            for rec in recommendations[:2]:  # Show top 2
                print(f"  - {rec.get('recommendation', rec.get('type', 'Unknown'))}")
    
    print(f"\n📈 Learning progression completed: {scenarios_completed} scenarios")

async def demo_difficulty_progression():
    """Demonstrate difficulty progression system"""
    print("\n" + "="*60)
    print("DEMO: Difficulty Progression System")
    print("="*60)
    
    system = IntegratedScenarioSystem()
    await system.initialize()
    
    # Create a progression path
    from scenarios.difficulty_progression import ProgressionPath
    
    progression_path = ProgressionPath(
        path_id="demo_progression",
        name="Demo Red Team Progression",
        description="Demonstration progression path for red team skills",
        strategy=ProgressionStrategy.MASTERY_BASED,
        target_category=ScenarioCategory.RECONNAISSANCE,
        stages=[
            {"name": "Basic Recon", "difficulty_increase": 0.0},
            {"name": "Advanced Recon", "difficulty_increase": 0.2},
            {"name": "Expert Recon", "difficulty_increase": 0.4}
        ],
        mastery_threshold=0.8,
        min_attempts=3
    )
    
    # Add progression path to system
    await system.progression_engine.create_progression_path(progression_path)
    
    agent_id = "progression_agent"
    
    # Start agent on progression path
    await system.progression_engine.start_agent_progression(agent_id, "demo_progression")
    
    print(f"Started agent {agent_id} on progression path: {progression_path.name}")
    
    # Simulate progression through stages
    for stage in range(3):
        print(f"\n--- Stage {stage + 1}: {progression_path.stages[stage]['name']} ---")
        
        # Get current difficulty
        difficulty = await system.progression_engine.get_next_difficulty(agent_id, "demo_progression")
        print(f"Current difficulty: {difficulty.overall_difficulty:.2f}")
        
        # Simulate multiple attempts at current stage
        for attempt in range(4):
            # Generate scenario at current difficulty
            request = ScenarioRequest(
                request_id=str(uuid.uuid4()),
                agent_ids=[agent_id],
                generation_type=GenerationType.PROGRESSION_BASED.value
            )
            
            response = await system.generate_scenario(request)
            
            if response.success:
                # Simulate performance (improving over attempts)
                success_prob = min(0.95, 0.6 + (attempt * 0.1))
                success = success_prob > 0.7
                
                performance_data = {
                    "success": success,
                    "completion_time": 2400 - (attempt * 200),
                    "confidence": success_prob,
                    "score": int(success_prob * 100)
                }
                
                # Update progression
                progression_result = await system.progression_engine.update_progression(
                    agent_id, "demo_progression", performance_data
                )
                
                print(f"  Attempt {attempt + 1}: Success={success}, "
                      f"Confidence={success_prob:.2f}")
                
                if progression_result.get("stage_advanced"):
                    print(f"  🎉 Advanced to next stage!")
                    break
        
        # Check if we've completed all stages
        progression_state = system.progression_engine.agent_progressions[agent_id]["demo_progression"]
        if progression_state.current_stage >= len(progression_path.stages) - 1:
            print(f"\n🏆 Progression path completed!")
            break

async def demo_scenario_validation():
    """Demonstrate scenario validation system"""
    print("\n" + "="*60)
    print("DEMO: Scenario Validation System")
    print("="*60)
    
    system = IntegratedScenarioSystem()
    await system.initialize()
    
    # Create a test scenario for validation
    from scenarios.scenario_templates import ScenarioTemplate, ScenarioObjective, ScenarioParameter
    
    test_scenario = ScenarioTemplate(
        template_id=str(uuid.uuid4()),
        name="Test Validation Scenario",
        description="A scenario created to test the validation system",
        scenario_type=ScenarioType.TRAINING,
        category=ScenarioCategory.RECONNAISSANCE,
        complexity=ComplexityLevel.INTERMEDIATE,
        estimated_duration=timedelta(hours=2),
        objectives=[
            ScenarioObjective(
                objective_id="obj1",
                name="Network Discovery",
                description="Discover network topology and services",
                success_criteria=["Identify at least 5 hosts", "Map network segments"]
            )
        ],
        parameters=[
            ScenarioParameter(
                name="target_network",
                parameter_type="string",
                default_value="192.168.1.0/24",
                description="Target network range for reconnaissance"
            )
        ]
    )
    
    print("Validating scenario with different validation levels...")
    
    # Test different validation levels
    validation_levels = [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]
    
    for level in validation_levels:
        print(f"\n--- {level.value.upper()} Validation ---")
        
        validation_result = await system.validate_scenario(test_scenario, level)
        
        print(f"Valid: {validation_result['valid']}")
        print(f"Checks performed: {len(validation_result['passed_checks']) + len(validation_result['failed_checks'])}")
        print(f"Passed checks: {len(validation_result['passed_checks'])}")
        print(f"Failed checks: {len(validation_result['failed_checks'])}")
        print(f"Warnings: {len(validation_result['warnings'])}")
        print(f"Validation time: {validation_result['validation_time']:.3f}s")
        
        if validation_result['failed_checks']:
            print("Failed checks:")
            for check in validation_result['failed_checks'][:3]:  # Show first 3
                print(f"  - {check}")
        
        if validation_result['recommendations']:
            print("Recommendations:")
            for rec in validation_result['recommendations'][:2]:  # Show first 2
                print(f"  - {rec}")

async def demo_system_testing():
    """Demonstrate system testing capabilities"""
    print("\n" + "="*60)
    print("DEMO: System Testing")
    print("="*60)
    
    system = IntegratedScenarioSystem()
    await system.initialize()
    
    print("Running comprehensive system tests...")
    
    # Run system tests
    test_results = await system.run_system_tests()
    
    print(f"\nOverall status: {test_results['overall_status']}")
    
    if 'validation_tests' in test_results:
        validation_results = test_results['validation_tests']
        print(f"\nValidation tests:")
        print(f"  Total tests: {validation_results.get('total_tests', 0)}")
        print(f"  Passed: {validation_results.get('passed_tests', 0)}")
        print(f"  Failed: {validation_results.get('failed_tests', 0)}")
        print(f"  Success rate: {validation_results.get('success_rate', 0):.2%}")
    
    if 'system_health' in test_results:
        health = test_results['system_health']
        print(f"\nSystem health: {'Healthy' if health.get('healthy') else 'Issues detected'}")
        
        if health.get('issues'):
            print("Issues:")
            for issue in health['issues']:
                print(f"  - {issue}")
        
        if health.get('component_status'):
            print("Component status:")
            for component, status in health['component_status'].items():
                print(f"  {component}: {status}")

async def demo_system_status():
    """Demonstrate system status monitoring"""
    print("\n" + "="*60)
    print("DEMO: System Status Monitoring")
    print("="*60)
    
    system = IntegratedScenarioSystem()
    await system.initialize()
    
    # Generate a few scenarios to populate metrics
    for i in range(3):
        request = ScenarioRequest(
            request_id=str(uuid.uuid4()),
            agent_ids=[f"status_demo_agent_{i}"],
            generation_type=GenerationType.ADAPTIVE_LEARNING.value
        )
        await system.generate_scenario(request)
    
    # Get system status
    status = await system.get_system_status()
    
    print(f"System Status: {status['status']}")
    print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
    print(f"Active requests: {status['active_requests']}")
    print(f"Total requests: {status['total_requests']}")
    print(f"Success rate: {status['success_rate']:.2%}")
    print(f"Average generation time: {status['average_generation_time']:.2f}s")
    
    print("\nComponent Status:")
    for component, initialized in status['components'].items():
        print(f"  {component}: {'✅ Initialized' if initialized else '❌ Not initialized'}")
    
    print("\nConfiguration:")
    for key, value in status['configuration'].items():
        print(f"  {key}: {value}")

async def main():
    """Run all demonstrations"""
    print("🚀 Archangel Scenario System Demonstration")
    print("=" * 80)
    
    try:
        # Run all demos
        await demo_basic_scenario_generation()
        await demo_agent_learning_progression()
        await demo_difficulty_progression()
        await demo_scenario_validation()
        await demo_system_testing()
        await demo_system_status()
        
        print("\n" + "="*80)
        print("✅ All demonstrations completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Demo execution failed")

if __name__ == "__main__":
    asyncio.run(main())