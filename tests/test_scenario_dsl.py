#!/usr/bin/env python3
"""
Tests for Scenario DSL and authoring tools
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scenarios.scenario_dsl import (
    ScenarioDSL, DSLError, DSLSyntaxError, DSLValidationError, DSLRuntimeError,
    DSLContext, DSLTemplateLibrary, create_dsl_example
)
from scenarios.dsl_parser import DSLParser, ValidationLevel, ValidationResult, ParseResult
from scenarios.scenario_templates import ScenarioType, ScenarioCategory, ComplexityLevel

class TestScenarioDSL:
    """Test the core DSL functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.dsl = ScenarioDSL()
    
    def test_basic_scenario_creation(self):
        """Test basic scenario creation"""
        dsl_code = '''
scenario("Test Scenario",
    type=ScenarioType.TRAINING,
    category=ScenarioCategory.RECONNAISSANCE,
    complexity=ComplexityLevel.BEGINNER
)

description("A simple test scenario")
        '''
        
        scenario = self.dsl.parse_scenario(dsl_code)
        
        assert scenario is not None
        assert scenario.name == "Test Scenario"
        assert scenario.scenario_type == ScenarioType.TRAINING
        assert scenario.category == ScenarioCategory.RECONNAISSANCE
        assert scenario.complexity == ComplexityLevel.BEGINNER
        assert scenario.description == "A simple test scenario"
    
    def test_parameter_definition(self):
        """Test parameter definition"""
        dsl_code = '''
scenario("Test Scenario")

parameter("target_count", "int", 
    default=10, 
    min_value=1, 
    max_value=100,
    description="Number of targets"
)

parameter("attack_type", "enum",
    allowed_values=["phishing", "malware", "social"],
    default="phishing",
    description="Type of attack to simulate"
)
        '''
        
        scenario = self.dsl.parse_scenario(dsl_code)
        
        assert len(scenario.parameters) == 2
        
        # Check integer parameter
        int_param = next(p for p in scenario.parameters if p.name == "target_count")
        assert int_param.parameter_type == "int"
        assert int_param.default_value == 10
        assert int_param.min_value == 1
        assert int_param.max_value == 100
        
        # Check enum parameter
        enum_param = next(p for p in scenario.parameters if p.name == "attack_type")
        assert enum_param.parameter_type == "enum"
        assert enum_param.allowed_values == ["phishing", "malware", "social"]
        assert enum_param.default_value == "phishing"
    
    def test_objective_definition(self):
        """Test objective definition"""
        dsl_code = '''
scenario("Test Scenario")

objective("Primary Goal", "Complete the primary objective",
    type="primary",
    points=100,
    time_limit=timedelta(minutes=30),
    success_criteria=["goal_achieved", "no_detection"],
    hints=["Check the logs", "Look for anomalies"]
)
        '''
        
        scenario = self.dsl.parse_scenario(dsl_code)
        
        assert len(scenario.objectives) == 1
        
        objective = scenario.objectives[0]
        assert objective.name == "Primary Goal"
        assert objective.description == "Complete the primary objective"
        assert objective.objective_type == "primary"
        assert objective.points == 100
        assert objective.time_limit == timedelta(minutes=30)
        assert objective.success_criteria == ["goal_achieved", "no_detection"]
        assert objective.hints == ["Check the logs", "Look for anomalies"]
    
    def test_asset_definition(self):
        """Test asset definition"""
        dsl_code = '''
scenario("Test Scenario")

asset("Web Server", "vm",
    configuration={
        "os": "ubuntu",
        "services": ["apache2", "mysql"]
    },
    vulnerabilities=["CVE-2021-44228"],
    security_controls=["firewall", "ids"]
)
        '''
        
        scenario = self.dsl.parse_scenario(dsl_code)
        
        assert len(scenario.required_assets) == 1
        
        asset = scenario.required_assets[0]
        assert asset.name == "Web Server"
        assert asset.asset_type == "vm"
        assert asset.configuration == {"os": "ubuntu", "services": ["apache2", "mysql"]}
        assert asset.vulnerabilities == ["CVE-2021-44228"]
        assert asset.security_controls == ["firewall", "ids"]
    
    def test_validation_rules(self):
        """Test validation rule definition"""
        dsl_code = '''
scenario("Test Scenario")

validation_rule("No permanent damage allowed")
validation_rule("All actions must be logged")
        '''
        
        scenario = self.dsl.parse_scenario(dsl_code)
        
        assert len(scenario.validation_rules) == 2
        assert "No permanent damage allowed" in scenario.validation_rules
        assert "All actions must be logged" in scenario.validation_rules
    
    def test_dsl_validation_success(self):
        """Test successful DSL validation"""
        dsl_code = '''
scenario("Valid Scenario",
    type=ScenarioType.TRAINING,
    category=ScenarioCategory.RECONNAISSANCE
)

description("A valid test scenario")

parameter("test_param", "string", default="test")

objective("Test Objective", "Test objective description")
        '''
        
        validation_result = self.dsl.validate_dsl(dsl_code)
        
        assert validation_result["valid"] is True
        assert validation_result["syntax_valid"] is True
        assert validation_result["semantic_valid"] is True
        assert len(validation_result["errors"]) == 0
    
    def test_dsl_syntax_error(self):
        """Test DSL syntax error detection"""
        dsl_code = '''
scenario("Invalid Scenario"
    # Missing closing parenthesis
        '''
        
        validation_result = self.dsl.validate_dsl(dsl_code)
        
        assert validation_result["valid"] is False
        assert validation_result["syntax_valid"] is False
        assert len(validation_result["errors"]) > 0
    
    def test_dsl_semantic_error(self):
        """Test DSL semantic error detection"""
        dsl_code = '''
# No scenario definition - should cause semantic error
parameter("test_param", "string", default="test")
        '''
        
        validation_result = self.dsl.validate_dsl(dsl_code)
        
        assert validation_result["valid"] is False
        assert len(validation_result["errors"]) > 0
    
    def test_dangerous_function_detection(self):
        """Test detection of dangerous functions"""
        dsl_code = '''
scenario("Dangerous Scenario")

# This should be detected as dangerous
exec("malicious_code")
        '''
        
        with pytest.raises(DSLError):
            self.dsl.parse_scenario(dsl_code)
    
    def test_variable_usage(self):
        """Test variable definition and usage"""
        dsl_code = '''
variable("server_count", 5)

scenario("Variable Test")

parameter("servers", "int", default=server_count)
        '''
        
        scenario = self.dsl.parse_scenario(dsl_code)
        
        # Check that variable was used
        param = scenario.parameters[0]
        assert param.default_value == 5
    
    def test_complex_scenario_example(self):
        """Test parsing of complex scenario example"""
        dsl_code = create_dsl_example()
        
        scenario = self.dsl.parse_scenario(dsl_code)
        
        assert scenario is not None
        assert scenario.name == "Advanced Persistent Threat Simulation"
        assert scenario.scenario_type == ScenarioType.LIVE_EXERCISE
        assert scenario.category == ScenarioCategory.FULL_CAMPAIGN
        assert scenario.complexity == ComplexityLevel.ADVANCED
        assert len(scenario.parameters) > 0
        assert len(scenario.objectives) > 0
        assert len(scenario.required_assets) > 0

class TestDSLParser:
    """Test the DSL parser and validation engine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.parser = DSLParser()
    
    def test_basic_parsing(self):
        """Test basic DSL parsing"""
        dsl_code = '''
scenario("Parser Test",
    type=ScenarioType.TRAINING
)

description("Test scenario for parser")
        '''
        
        result = self.parser.parse(dsl_code)
        
        assert result.success is True
        assert result.scenario is not None
        assert result.scenario.name == "Parser Test"
        assert len(result.errors) == 0
    
    def test_validation_levels(self):
        """Test different validation levels"""
        dsl_code = '''
scenario("Validation Test")
description("Test scenario")
        '''
        
        # Basic validation
        basic_result = self.parser.validate(dsl_code, ValidationLevel.BASIC)
        assert basic_result.valid is True
        
        # Standard validation
        standard_result = self.parser.validate(dsl_code, ValidationLevel.STANDARD)
        assert standard_result.valid is True
        
        # Strict validation
        strict_result = self.parser.validate(dsl_code, ValidationLevel.STRICT)
        assert strict_result.valid is True
        
        # Comprehensive validation
        comprehensive_result = self.parser.validate(dsl_code, ValidationLevel.COMPREHENSIVE)
        assert comprehensive_result.valid is True
        assert comprehensive_result.complexity_score >= 0
    
    def test_syntax_error_detection(self):
        """Test syntax error detection with line numbers"""
        dsl_code = '''
scenario("Syntax Error Test"
# Missing closing parenthesis on line above
description("Test")
        '''
        
        result = self.parser.validate(dsl_code, ValidationLevel.BASIC)
        
        assert result.valid is False
        assert len(result.syntax_errors) > 0
    
    def test_semantic_validation(self):
        """Test semantic validation"""
        dsl_code = '''
scenario("Semantic Test")

# Duplicate parameter names - should be detected
parameter("test_param", "string", default="value1")
parameter("test_param", "int", default=42)
        '''
        
        result = self.parser.validate(dsl_code, ValidationLevel.STANDARD)
        
        # Should detect duplicate parameter names
        assert len(result.logic_errors) > 0 or len(result.warnings) > 0
    
    def test_dependency_analysis(self):
        """Test dependency analysis"""
        dsl_code = '''
scenario("Dependency Test")

objective("First", "First objective")

objective("Second", "Second objective",
    prerequisites=["First"]
)

objective("Third", "Third objective", 
    prerequisites=["Second"]
)
        '''
        
        result = self.parser.validate(dsl_code, ValidationLevel.COMPREHENSIVE)
        
        assert result.valid is True
        assert "dependency_analysis" in result.dependency_analysis
        assert len(result.dependency_analysis["dependencies"]) >= 0
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection"""
        dsl_code = '''
scenario("Circular Dependency Test")

objective("A", "Objective A", dependencies=["B"])
objective("B", "Objective B", dependencies=["A"])
        '''
        
        result = self.parser.validate(dsl_code, ValidationLevel.COMPREHENSIVE)
        
        # Should detect circular dependency
        assert len(result.logic_errors) > 0
        assert result.valid is False
    
    def test_performance_analysis(self):
        """Test performance analysis"""
        # Create a complex scenario
        dsl_code = '''
scenario("Performance Test")
        ''' + '\n'.join([
            f'parameter("param_{i}", "int", default={i})' 
            for i in range(20)
        ]) + '\n' + '\n'.join([
            f'objective("obj_{i}", "Objective {i}")' 
            for i in range(15)
        ])
        
        result = self.parser.validate(dsl_code, ValidationLevel.COMPREHENSIVE)
        
        assert result.complexity_score > 0
        # Should warn about many parameters/objectives
        assert len(result.warnings) > 0
    
    def test_security_validation(self):
        """Test security validation"""
        dsl_code = '''
scenario("Security Test")

# This should trigger security warnings
exec("dangerous_code")
open("file.txt", "w")
        '''
        
        result = self.parser.validate(dsl_code, ValidationLevel.STRICT)
        
        assert result.valid is False
        assert len(result.syntax_errors) > 0  # Should detect dangerous functions
    
    def test_suggestion_generation(self):
        """Test suggestion generation"""
        dsl_code = '''
scenario("Minimal Scenario")
        '''
        
        result = self.parser.validate(dsl_code, ValidationLevel.COMPREHENSIVE)
        
        # Should suggest adding description, documentation, etc.
        assert len(result.suggestions) > 0

class TestDSLTemplateLibrary:
    """Test the DSL template library"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.library = DSLTemplateLibrary()
        self.library.create_default_library()
    
    def test_component_registration(self):
        """Test component registration"""
        component_code = '''
asset("Test Asset", "vm",
    configuration={"os": "ubuntu"}
)
        '''
        
        self.library.register_component(
            "test_asset",
            component_code,
            "Test asset component"
        )
        
        assert "test_asset" in self.library.components
        retrieved_code = self.library.get_component("test_asset")
        assert retrieved_code == component_code
    
    def test_pattern_registration(self):
        """Test pattern registration"""
        pattern_code = '''
scenario("Test Pattern")
objective("Test Objective", "Test description")
        '''
        
        self.library.register_pattern(
            "test_pattern",
            pattern_code,
            "Test pattern"
        )
        
        assert "test_pattern" in self.library.patterns
        retrieved_code = self.library.get_pattern("test_pattern")
        assert retrieved_code == pattern_code
    
    def test_default_library_creation(self):
        """Test default library creation"""
        components = self.library.list_components()
        patterns = self.library.list_patterns()
        
        assert len(components) > 0
        assert len(patterns) > 0
        
        # Check for expected default components
        component_names = [c["name"] for c in components]
        assert "web_server" in component_names
        assert "database_server" in component_names
        
        # Check for expected default patterns
        pattern_names = [p["name"] for p in patterns]
        assert "phishing_campaign" in pattern_names
        assert "incident_response" in pattern_names
    
    def test_usage_tracking(self):
        """Test usage tracking"""
        # Get a component multiple times
        self.library.get_component("web_server")
        self.library.get_component("web_server")
        
        components = self.library.list_components()
        web_server_component = next(c for c in components if c["name"] == "web_server")
        
        assert web_server_component["usage_count"] == 2

class TestDSLIntegration:
    """Integration tests for the complete DSL system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.dsl = ScenarioDSL()
        self.parser = DSLParser()
        self.library = DSLTemplateLibrary()
        self.library.create_default_library()
    
    def test_end_to_end_scenario_creation(self):
        """Test complete scenario creation workflow"""
        # Create a scenario using DSL
        dsl_code = '''
scenario("Integration Test Scenario",
    type=ScenarioType.TRAINING,
    category=ScenarioCategory.PHISHING,
    complexity=ComplexityLevel.INTERMEDIATE,
    duration=timedelta(hours=2)
)

description("Complete integration test scenario")

parameter("target_count", "int", default=20, min_value=5, max_value=100)
parameter("email_template", "enum", 
    allowed_values=["urgent", "it_support", "hr"],
    default="urgent"
)

objective("Send Phishing Emails", 
    "Successfully deliver phishing emails",
    type="primary",
    points=50,
    success_criteria=["emails_sent >= target_count"]
)

objective("Collect Credentials",
    "Harvest user credentials",
    type="primary", 
    points=100,
    prerequisites=["Send Phishing Emails"],
    success_criteria=["credentials_collected > 0"]
)

asset("Email Server", "vm",
    configuration={"os": "ubuntu", "service": "postfix"},
    vulnerabilities=["open_relay"],
    security_controls=["spam_filter"]
)

asset("Web Server", "vm",
    configuration={"os": "ubuntu", "service": "apache2"},
    vulnerabilities=["outdated_php"],
    security_controls=["mod_security"]
)

validation_rule("No real emails sent")
validation_rule("All activities logged")

tags(["phishing", "social_engineering", "training"])

documentation("""
This scenario simulates a phishing campaign for training purposes.
Participants learn to identify and respond to phishing attempts.
""")
        '''
        
        # Parse and validate
        parse_result = self.parser.parse(dsl_code, ValidationLevel.COMPREHENSIVE)
        
        assert parse_result.success is True
        assert parse_result.scenario is not None
        
        scenario = parse_result.scenario
        
        # Verify scenario properties
        assert scenario.name == "Integration Test Scenario"
        assert scenario.scenario_type == ScenarioType.TRAINING
        assert scenario.category == ScenarioCategory.PHISHING
        assert scenario.complexity == ComplexityLevel.INTERMEDIATE
        assert scenario.estimated_duration == timedelta(hours=2)
        
        # Verify parameters
        assert len(scenario.parameters) == 2
        param_names = [p.name for p in scenario.parameters]
        assert "target_count" in param_names
        assert "email_template" in param_names
        
        # Verify objectives
        assert len(scenario.objectives) == 2
        obj_names = [o.name for o in scenario.objectives]
        assert "Send Phishing Emails" in obj_names
        assert "Collect Credentials" in obj_names
        
        # Verify assets
        assert len(scenario.required_assets) == 2
        asset_names = [a.name for a in scenario.required_assets]
        assert "Email Server" in asset_names
        assert "Web Server" in asset_names
        
        # Verify validation rules
        assert len(scenario.validation_rules) == 2
        
        # Verify tags
        assert "phishing" in scenario.tags
        assert "social_engineering" in scenario.tags
        
        # Verify documentation
        assert "phishing campaign" in scenario.documentation.lower()
    
    def test_template_library_integration(self):
        """Test integration with template library"""
        # Get a component from library
        web_server_code = self.library.get_component("web_server")
        assert web_server_code is not None
        
        # Create scenario using library component
        dsl_code = f'''
scenario("Library Integration Test")

{web_server_code}
        '''
        
        scenario = self.dsl.parse_scenario(dsl_code)
        
        assert scenario is not None
        assert len(scenario.required_assets) == 1
        assert scenario.required_assets[0].name == "Web Server"
    
    def test_error_recovery(self):
        """Test error recovery and reporting"""
        # Create DSL with multiple errors
        dsl_code = '''
scenario("Error Test"
# Missing closing parenthesis

parameter("duplicate", "string")
parameter("duplicate", "int")  # Duplicate parameter

objective("Bad Objective", "Description",
    prerequisites=["NonExistent"]  # Missing prerequisite
)

exec("dangerous")  # Dangerous function
        '''
        
        # Should handle multiple errors gracefully
        validation_result = self.parser.validate(dsl_code, ValidationLevel.COMPREHENSIVE)
        
        assert validation_result.valid is False
        assert len(validation_result.syntax_errors) > 0
        assert len(validation_result.logic_errors) > 0
    
    def test_performance_with_large_scenario(self):
        """Test performance with large scenarios"""
        # Generate a large scenario
        large_dsl = '''
scenario("Large Scenario Test",
    type=ScenarioType.TRAINING,
    complexity=ComplexityLevel.ADVANCED
)

description("Large scenario for performance testing")
        '''
        
        # Add many parameters
        for i in range(50):
            large_dsl += f'\nparameter("param_{i}", "int", default={i})'
        
        # Add many objectives
        for i in range(30):
            large_dsl += f'\nobjective("Objective {i}", "Description {i}", points={i*10})'
        
        # Add many assets
        for i in range(25):
            large_dsl += f'\nasset("Asset {i}", "vm", configuration={{"id": {i}}})'
        
        # Parse and validate
        start_time = datetime.now()
        result = self.parser.parse(large_dsl, ValidationLevel.COMPREHENSIVE)
        end_time = datetime.now()
        
        parse_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert parse_time < 5.0  # 5 seconds max
        assert result.success is True
        assert result.scenario is not None
        
        # Verify all components were parsed
        assert len(result.scenario.parameters) == 50
        assert len(result.scenario.objectives) == 30
        assert len(result.scenario.required_assets) == 25

class TestDSLExamples:
    """Test various DSL examples and use cases"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.dsl = ScenarioDSL()
        self.parser = DSLParser()
    
    def test_red_team_scenario(self):
        """Test Red Team scenario example"""
        dsl_code = '''
scenario("Red Team Exercise",
    type=ScenarioType.LIVE_EXERCISE,
    category=ScenarioCategory.FULL_CAMPAIGN,
    complexity=ComplexityLevel.EXPERT
)

description("Advanced Red Team penetration testing exercise")

parameter("target_network", "string", default="192.168.1.0/24")
parameter("time_limit", "int", default=480, description="Time limit in minutes")

objective("Reconnaissance", 
    "Gather intelligence on target network",
    type="primary",
    points=75,
    time_limit=timedelta(hours=1)
)

objective("Initial Access",
    "Gain initial foothold in target network", 
    type="primary",
    points=100,
    prerequisites=["Reconnaissance"]
)

objective("Privilege Escalation",
    "Escalate privileges to administrator level",
    type="primary",
    points=125,
    prerequisites=["Initial Access"]
)

objective("Lateral Movement",
    "Move laterally to additional systems",
    type="primary", 
    points=150,
    prerequisites=["Privilege Escalation"]
)

objective("Data Exfiltration",
    "Locate and exfiltrate sensitive data",
    type="secondary",
    points=100
)

asset("Domain Controller", "vm",
    configuration={"os": "windows_server", "role": "domain_controller"},
    vulnerabilities=["zerologon", "weak_passwords"],
    security_controls=["windows_defender", "audit_logging"]
)

asset("Web Server", "vm",
    configuration={"os": "ubuntu", "services": ["apache2", "mysql"]},
    vulnerabilities=["sql_injection", "outdated_cms"],
    security_controls=["mod_security", "fail2ban"]
)

validation_rule("No permanent damage to systems")
validation_rule("All activities must be logged")

tags(["red_team", "penetration_testing", "advanced"])
        '''
        
        scenario = self.dsl.parse_scenario(dsl_code)
        
        assert scenario.name == "Red Team Exercise"
        assert scenario.complexity == ComplexityLevel.EXPERT
        assert len(scenario.objectives) == 5
        assert len(scenario.required_assets) == 2
        
        # Check objective dependencies
        initial_access = next(o for o in scenario.objectives if o.name == "Initial Access")
        assert "Reconnaissance" in initial_access.prerequisites
    
    def test_blue_team_scenario(self):
        """Test Blue Team scenario example"""
        dsl_code = '''
scenario("Blue Team Defense Exercise",
    type=ScenarioType.TRAINING,
    category=ScenarioCategory.INCIDENT_RESPONSE,
    complexity=ComplexityLevel.ADVANCED
)

description("Blue Team incident response and threat hunting exercise")

parameter("incident_type", "enum",
    allowed_values=["malware", "data_breach", "insider_threat"],
    default="malware"
)

objective("Threat Detection",
    "Detect and identify security threats",
    type="primary",
    points=100,
    time_limit=timedelta(minutes=30)
)

objective("Incident Response",
    "Execute incident response procedures",
    type="primary",
    points=125,
    prerequisites=["Threat Detection"]
)

objective("Threat Containment", 
    "Contain and isolate threats",
    type="primary",
    points=150,
    prerequisites=["Incident Response"]
)

objective("Forensic Analysis",
    "Perform digital forensics investigation",
    type="secondary",
    points=100
)

asset("SIEM System", "service",
    configuration={"type": "splunk", "data_sources": ["windows_logs", "network_logs"]},
    security_controls=["access_control", "encryption"]
)

asset("SOC Workstation", "vm",
    configuration={"os": "windows_10", "tools": ["wireshark", "volatility"]},
    security_controls=["endpoint_protection", "privilege_management"]
)

validation_rule("Follow incident response procedures")
validation_rule("Maintain chain of custody for evidence")

tags(["blue_team", "incident_response", "soc"])
        '''
        
        scenario = self.dsl.parse_scenario(dsl_code)
        
        assert scenario.category == ScenarioCategory.INCIDENT_RESPONSE
        assert len(scenario.objectives) == 4
        
        # Check for SIEM asset
        siem_asset = next(a for a in scenario.required_assets if a.name == "SIEM System")
        assert siem_asset.asset_type == "service"

if __name__ == "__main__":
    pytest.main([__file__])