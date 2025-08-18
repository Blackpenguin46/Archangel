#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Scenario DSL Demo
Demonstration of the Domain-Specific Language for scenario creation
"""

import logging
import sys
from datetime import datetime, timedelta

from scenarios.scenario_dsl import ScenarioDSL, DSLTemplateLibrary, create_dsl_example
from scenarios.dsl_parser import DSLParser, ValidationLevel
from scenarios.visual_editor import VisualScenarioEditor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_basic_dsl():
    """Demonstrate basic DSL functionality"""
    print("=" * 60)
    print("BASIC DSL FUNCTIONALITY DEMO")
    print("=" * 60)
    
    # Create DSL instance
    dsl = ScenarioDSL()
    
    # Simple scenario example
    simple_dsl = '''
scenario("Basic Phishing Exercise",
    type=ScenarioType.TRAINING,
    category=ScenarioCategory.INITIAL_ACCESS,
    complexity=ComplexityLevel.BEGINNER,
    duration=timedelta(hours=1)
)

description("A basic phishing awareness training exercise")

parameter("target_count", "int", 
    default=10, 
    min_value=5, 
    max_value=50,
    description="Number of employees to target"
)

parameter("email_template", "enum",
    allowed_values=["urgent_security", "it_support", "hr_notice"],
    default="urgent_security",
    description="Phishing email template to use"
)

objective("Send Phishing Emails",
    "Successfully deliver phishing emails to targets",
    type="primary",
    points=50,
    success_criteria=["emails_sent >= target_count"]
)

objective("Track Click Rate",
    "Monitor employee click-through rates",
    type="primary",
    points=75,
    success_criteria=["click_rate_measured", "results_documented"]
)

asset("Email Server", "vm",
    configuration={
        "os": "ubuntu",
        "service": "postfix",
        "domain": "company.local"
    },
    security_controls=["spam_filter", "dkim"]
)

asset("Phishing Landing Page", "service",
    configuration={
        "web_server": "nginx",
        "ssl_enabled": True
    },
    security_controls=["https_only"]
)

validation_rule("No real credentials collected")
validation_rule("All activities logged for training purposes")

tags(["phishing", "awareness", "training", "beginner"])

documentation("""
This scenario provides basic phishing awareness training.
Employees receive simulated phishing emails and their responses
are tracked for training purposes. No real credentials are collected.
""")
    '''
    
    try:
        print("Parsing simple DSL scenario...")
        scenario = dsl.parse_scenario(simple_dsl)
        
        print(f"✓ Successfully created scenario: {scenario.name}")
        print(f"  Type: {scenario.scenario_type.value}")
        print(f"  Category: {scenario.category.value}")
        print(f"  Complexity: {scenario.complexity.value}")
        print(f"  Duration: {scenario.estimated_duration}")
        print(f"  Parameters: {len(scenario.parameters)}")
        print(f"  Objectives: {len(scenario.objectives)}")
        print(f"  Assets: {len(scenario.required_assets)}")
        print(f"  Tags: {', '.join(scenario.tags)}")
        
        # Show parameters
        print("\nParameters:")
        for param in scenario.parameters:
            print(f"  - {param.name} ({param.parameter_type}): {param.description}")
        
        # Show objectives
        print("\nObjectives:")
        for obj in scenario.objectives:
            print(f"  - {obj.name}: {obj.description} ({obj.points} points)")
        
        # Show assets
        print("\nAssets:")
        for asset in scenario.required_assets:
            print(f"  - {asset.name} ({asset.asset_type})")
        
    except Exception as e:
        print(f"✗ Failed to parse scenario: {e}")
        return False
    
    return True

def demo_dsl_validation():
    """Demonstrate DSL validation capabilities"""
    print("\n" + "=" * 60)
    print("DSL VALIDATION DEMO")
    print("=" * 60)
    
    parser = DSLParser()
    
    # Test cases with different types of errors
    test_cases = [
        {
            "name": "Valid Scenario",
            "code": '''
scenario("Valid Test")
description("This is valid")
objective("Test", "Test objective")
            ''',
            "expected_valid": True
        },
        {
            "name": "Syntax Error",
            "code": '''
scenario("Syntax Error Test"
# Missing closing parenthesis
            ''',
            "expected_valid": False
        },
        {
            "name": "Semantic Error",
            "code": '''
# No scenario definition
parameter("test", "string")
            ''',
            "expected_valid": False
        },
        {
            "name": "Logic Error",
            "code": '''
scenario("Logic Error Test")
parameter("duplicate", "string")
parameter("duplicate", "int")  # Duplicate parameter
            ''',
            "expected_valid": False
        },
        {
            "name": "Security Issue",
            "code": '''
scenario("Security Test")
exec("dangerous_code")  # Dangerous function
            ''',
            "expected_valid": False
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print("-" * 40)
        
        try:
            result = parser.validate(test_case['code'], ValidationLevel.COMPREHENSIVE)
            
            print(f"Valid: {result.valid}")
            print(f"Validation time: {result.validation_time:.3f}s")
            print(f"Complexity score: {result.complexity_score:.1f}")
            
            if result.syntax_errors:
                print(f"Syntax errors: {len(result.syntax_errors)}")
                for error in result.syntax_errors[:3]:  # Show first 3
                    print(f"  - {error}")
            
            if result.semantic_errors:
                print(f"Semantic errors: {len(result.semantic_errors)}")
                for error in result.semantic_errors[:3]:
                    print(f"  - {error}")
            
            if result.logic_errors:
                print(f"Logic errors: {len(result.logic_errors)}")
                for error in result.logic_errors[:3]:
                    print(f"  - {error}")
            
            if result.warnings:
                print(f"Warnings: {len(result.warnings)}")
                for warning in result.warnings[:3]:
                    print(f"  - {warning}")
            
            if result.suggestions:
                print(f"Suggestions: {len(result.suggestions)}")
                for suggestion in result.suggestions[:3]:
                    print(f"  - {suggestion}")
            
            # Check if result matches expectation
            if result.valid == test_case['expected_valid']:
                print("✓ Validation result as expected")
            else:
                print("✗ Unexpected validation result")
                
        except Exception as e:
            print(f"✗ Validation failed: {e}")

def demo_template_library():
    """Demonstrate template library functionality"""
    print("\n" + "=" * 60)
    print("TEMPLATE LIBRARY DEMO")
    print("=" * 60)
    
    # Create library and load defaults
    library = DSLTemplateLibrary()
    library.create_default_library()
    
    print("Available Components:")
    components = library.list_components()
    for component in components:
        print(f"  - {component['name']}: {component['description']}")
        print(f"    Usage count: {component['usage_count']}")
    
    print("\nAvailable Patterns:")
    patterns = library.list_patterns()
    for pattern in patterns:
        print(f"  - {pattern['name']}: {pattern['description']}")
        print(f"    Usage count: {pattern['usage_count']}")
    
    # Demonstrate component usage
    print("\nUsing web_server component:")
    web_server_code = library.get_component("web_server")
    if web_server_code:
        print("Component code:")
        print(web_server_code)
        
        # Create scenario using component
        dsl = ScenarioDSL()
        scenario_with_component = f'''
scenario("Component Demo")
description("Demonstrating component usage")

{web_server_code}
        '''
        
        try:
            scenario = dsl.parse_scenario(scenario_with_component)
            print(f"\n✓ Successfully created scenario with component")
            print(f"  Assets: {len(scenario.required_assets)}")
            if scenario.required_assets:
                asset = scenario.required_assets[0]
                print(f"  Asset name: {asset.name}")
                print(f"  Asset type: {asset.asset_type}")
        except Exception as e:
            print(f"✗ Failed to use component: {e}")
    
    # Demonstrate pattern usage
    print("\nUsing phishing_campaign pattern:")
    pattern_code = library.get_pattern("phishing_campaign")
    if pattern_code:
        try:
            scenario = dsl.parse_scenario(pattern_code)
            print(f"✓ Successfully created scenario from pattern")
            print(f"  Name: {scenario.name}")
            print(f"  Objectives: {len(scenario.objectives)}")
            print(f"  Parameters: {len(scenario.parameters)}")
        except Exception as e:
            print(f"✗ Failed to use pattern: {e}")

def demo_complex_scenario():
    """Demonstrate complex scenario creation"""
    print("\n" + "=" * 60)
    print("COMPLEX SCENARIO DEMO")
    print("=" * 60)
    
    # Use the built-in complex example
    complex_dsl = create_dsl_example()
    
    print("Parsing complex APT simulation scenario...")
    print(f"DSL code length: {len(complex_dsl)} characters")
    print(f"DSL code lines: {len(complex_dsl.split('\\n'))}")
    
    try:
        dsl = ScenarioDSL()
        parser = DSLParser()
        
        # Parse the scenario
        start_time = datetime.now()
        parse_result = parser.parse(complex_dsl, ValidationLevel.COMPREHENSIVE)
        end_time = datetime.now()
        
        parse_time = (end_time - start_time).total_seconds()
        
        if parse_result.success:
            scenario = parse_result.scenario
            
            print(f"✓ Successfully parsed complex scenario in {parse_time:.3f}s")
            print(f"  Name: {scenario.name}")
            print(f"  Type: {scenario.scenario_type.value}")
            print(f"  Category: {scenario.category.value}")
            print(f"  Complexity: {scenario.complexity.value}")
            print(f"  Duration: {scenario.estimated_duration}")
            print(f"  Parameters: {len(scenario.parameters)}")
            print(f"  Objectives: {len(scenario.objectives)}")
            print(f"  Assets: {len(scenario.required_assets)}")
            print(f"  Validation rules: {len(scenario.validation_rules)}")
            
            # Show some objectives
            print("\\nSample Objectives:")
            for i, obj in enumerate(scenario.objectives[:5]):  # Show first 5
                prereqs = f" (requires: {', '.join(obj.prerequisites)})" if obj.prerequisites else ""
                print(f"  {i+1}. {obj.name}: {obj.points} points{prereqs}")
            
            if len(scenario.objectives) > 5:
                print(f"  ... and {len(scenario.objectives) - 5} more objectives")
            
            # Show some assets
            print("\\nSample Assets:")
            for i, asset in enumerate(scenario.required_assets[:3]):  # Show first 3
                vulns = f" (vulns: {len(asset.vulnerabilities)})" if asset.vulnerabilities else ""
                print(f"  {i+1}. {asset.name} ({asset.asset_type}){vulns}")
            
            if len(scenario.required_assets) > 3:
                print(f"  ... and {len(scenario.required_assets) - 3} more assets")
            
        else:
            print(f"✗ Failed to parse complex scenario")
            print(f"  Errors: {len(parse_result.errors)}")
            for error in parse_result.errors[:5]:
                print(f"    - {error}")
                
    except Exception as e:
        print(f"✗ Exception during complex scenario parsing: {e}")

def demo_visual_editor():
    """Demonstrate visual editor (if GUI available)"""
    print("\n" + "=" * 60)
    print("VISUAL EDITOR DEMO")
    print("=" * 60)
    
    try:
        import tkinter as tk
        
        print("Visual editor is available!")
        print("The visual editor provides:")
        print("  - Drag-and-drop component placement")
        print("  - Property editing panels")
        print("  - Real-time DSL code generation")
        print("  - Visual validation feedback")
        print("  - Template library integration")
        print("  - Export to DSL and JSON formats")
        
        response = input("\\nWould you like to launch the visual editor? (y/n): ")
        if response.lower().startswith('y'):
            print("Launching visual editor...")
            editor = VisualScenarioEditor()
            editor.run()
        else:
            print("Visual editor demo skipped.")
            
    except ImportError:
        print("Visual editor not available (tkinter not installed)")
        print("The visual editor would provide a drag-and-drop interface")
        print("for non-technical users to create scenarios visually.")

def main():
    """Main demo function"""
    print("Archangel Scenario DSL Demo")
    print("=" * 60)
    print("This demo showcases the Domain-Specific Language (DSL)")
    print("for creating cybersecurity training scenarios.")
    print()
    
    try:
        # Run all demos
        success = True
        
        success &= demo_basic_dsl()
        demo_dsl_validation()
        demo_template_library()
        demo_complex_scenario()
        demo_visual_editor()
        
        print("\\n" + "=" * 60)
        if success:
            print("✓ All demos completed successfully!")
        else:
            print("✗ Some demos encountered issues")
        print("=" * 60)
        
        print("\\nDSL Features Demonstrated:")
        print("  ✓ Intuitive Python-based syntax")
        print("  ✓ Comprehensive validation engine")
        print("  ✓ Multi-level error checking")
        print("  ✓ Template library system")
        print("  ✓ Complex scenario support")
        print("  ✓ Visual editor interface")
        print("  ✓ Performance analysis")
        print("  ✓ Security validation")
        
        print("\\nNext Steps:")
        print("  1. Create your own scenarios using the DSL")
        print("  2. Use the template library for common patterns")
        print("  3. Validate scenarios before deployment")
        print("  4. Try the visual editor for easier creation")
        print("  5. Extend the DSL with custom functions")
        
    except KeyboardInterrupt:
        print("\\n\\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\\n✗ Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())