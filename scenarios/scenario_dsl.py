#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Scenario Domain-Specific Language (DSL)
Python-based DSL for intuitive scenario scripting and configuration
"""

import ast
import logging
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable, Type
from enum import Enum
import inspect
import json

from .scenario_templates import (
    ScenarioTemplate, ScenarioParameter, ScenarioObjective, ScenarioAsset,
    ScenarioType, ScenarioCategory, ComplexityLevel, NetworkTopology
)
from agents.base_agent import Team, Role

logger = logging.getLogger(__name__)

class DSLError(Exception):
    """Base exception for DSL-related errors"""
    pass

class DSLSyntaxError(DSLError):
    """Syntax error in DSL code"""
    pass

class DSLValidationError(DSLError):
    """Validation error in DSL configuration"""
    pass

class DSLRuntimeError(DSLError):
    """Runtime error during DSL execution"""
    pass

@dataclass
class DSLContext:
    """Context for DSL execution"""
    variables: Dict[str, Any] = field(default_factory=dict)
    functions: Dict[str, Callable] = field(default_factory=dict)
    templates: Dict[str, ScenarioTemplate] = field(default_factory=dict)
    current_scenario: Optional[ScenarioTemplate] = None
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class ScenarioDSL:
    """
    Domain-Specific Language for scenario creation and configuration.
    
    Provides a Python-based DSL that allows intuitive scenario scripting
    with built-in validation and error checking.
    """
    
    def __init__(self):
        self.context = DSLContext()
        self.builtin_functions = self._create_builtin_functions()
        self.logger = logging.getLogger(__name__)
        
        # Register built-in functions
        self.context.functions.update(self.builtin_functions)
    
    def parse_scenario(self, dsl_code: str) -> ScenarioTemplate:
        """Parse DSL code and return a scenario template"""
        try:
            # Reset context for new scenario
            self.context = DSLContext()
            self.context.functions.update(self.builtin_functions)
            
            # Parse and validate syntax
            tree = self._parse_syntax(dsl_code)
            
            # Execute DSL code
            self._execute_dsl(tree)
            
            # Validate and return scenario
            if not self.context.current_scenario:
                raise DSLRuntimeError("No scenario was defined in the DSL code")
            
            return self.context.current_scenario
            
        except Exception as e:
            self.logger.error(f"Failed to parse scenario DSL: {e}")
            raise DSLError(f"DSL parsing failed: {str(e)}") from e
    
    def validate_dsl(self, dsl_code: str) -> Dict[str, Any]:
        """Validate DSL code without executing it"""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "syntax_valid": False,
                "semantic_valid": False
            }
            
            # Syntax validation
            try:
                tree = self._parse_syntax(dsl_code)
                validation_result["syntax_valid"] = True
            except SyntaxError as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Syntax error: {str(e)}")
                return validation_result
            
            # Semantic validation
            try:
                # Create a copy of context for validation
                original_context = self.context
                self.context = DSLContext()
                self.context.functions.update(self.builtin_functions)
                
                # Execute in validation mode
                self._execute_dsl(tree, validation_mode=True)
                
                validation_result["semantic_valid"] = True
                validation_result["errors"].extend(self.context.validation_errors)
                validation_result["warnings"].extend(self.context.warnings)
                
                if self.context.validation_errors:
                    validation_result["valid"] = False
                
                # Restore original context
                self.context = original_context
                
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Semantic error: {str(e)}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Failed to validate DSL: {e}")
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "syntax_valid": False,
                "semantic_valid": False
            }
    
    def _parse_syntax(self, dsl_code: str) -> ast.AST:
        """Parse DSL code syntax"""
        try:
            # Parse Python AST
            tree = ast.parse(dsl_code)
            
            # Validate AST structure
            self._validate_ast_structure(tree)
            
            return tree
            
        except SyntaxError as e:
            raise DSLSyntaxError(f"Invalid Python syntax: {str(e)}") from e
        except Exception as e:
            raise DSLSyntaxError(f"AST parsing failed: {str(e)}") from e
    
    def _validate_ast_structure(self, tree: ast.AST) -> None:
        """Validate AST structure for DSL compliance"""
        # Check for dangerous operations
        dangerous_nodes = []
        
        for node in ast.walk(tree):
            # Prevent imports (except allowed ones)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ['datetime', 'timedelta']:
                        dangerous_nodes.append(f"Import not allowed: {alias.name}")
            
            # Prevent dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', 'open', '__import__']:
                        dangerous_nodes.append(f"Dangerous function call: {node.func.id}")
        
        if dangerous_nodes:
            raise DSLSyntaxError(f"Dangerous operations detected: {', '.join(dangerous_nodes)}")
    
    def _execute_dsl(self, tree: ast.AST, validation_mode: bool = False) -> None:
        """Execute DSL AST"""
        try:
            # Create execution namespace
            namespace = {
                '__builtins__': {
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'abs': abs,
                    'round': round,
                },
                # DSL functions
                **self.context.functions,
                # DSL variables
                **self.context.variables,
                # Enums and classes
                'ScenarioType': ScenarioType,
                'ScenarioCategory': ScenarioCategory,
                'ComplexityLevel': ComplexityLevel,
                'NetworkTopology': NetworkTopology,
                'Team': Team,
                'Role': Role,
                'timedelta': timedelta,
                'datetime': datetime,
            }
            
            # Execute the code
            exec(compile(tree, '<dsl>', 'exec'), namespace)
            
            # Update context with any new variables
            for key, value in namespace.items():
                if key not in self.context.functions and not key.startswith('__'):
                    self.context.variables[key] = value
            
        except Exception as e:
            if validation_mode:
                self.context.validation_errors.append(f"Execution error: {str(e)}")
            else:
                raise DSLRuntimeError(f"DSL execution failed: {str(e)}") from e
    
    def _create_builtin_functions(self) -> Dict[str, Callable]:
        """Create built-in DSL functions"""
        return {
            'scenario': self._dsl_scenario,
            'parameter': self._dsl_parameter,
            'objective': self._dsl_objective,
            'asset': self._dsl_asset,
            'network': self._dsl_network,
            'team': self._dsl_team,
            'duration': self._dsl_duration,
            'tags': self._dsl_tags,
            'description': self._dsl_description,
            'documentation': self._dsl_documentation,
            'validation_rule': self._dsl_validation_rule,
            'prerequisite': self._dsl_prerequisite,
            'dependency': self._dsl_dependency,
            'include_template': self._dsl_include_template,
            'extend_template': self._dsl_extend_template,
            'variable': self._dsl_variable,
            'condition': self._dsl_condition,
            'loop': self._dsl_loop,
            'template_function': self._dsl_template_function,
        }
    
    def _dsl_scenario(self, name: str, **kwargs) -> None:
        """DSL function to define a scenario"""
        try:
            # Create new scenario template
            scenario = ScenarioTemplate(
                template_id=str(uuid.uuid4()),
                name=name,
                description=kwargs.get('description', ''),
                scenario_type=kwargs.get('type', ScenarioType.TRAINING),
                category=kwargs.get('category', ScenarioCategory.FULL_CAMPAIGN),
                complexity=kwargs.get('complexity', ComplexityLevel.INTERMEDIATE),
                network_topology=kwargs.get('network_topology', NetworkTopology.SIMPLE_NETWORK),
                estimated_duration=kwargs.get('duration', timedelta(hours=2)),
                min_participants=kwargs.get('min_participants', 1),
                max_participants=kwargs.get('max_participants', 10),
                tags=set(kwargs.get('tags', [])),
                version=kwargs.get('version', '1.0'),
                author=kwargs.get('author', 'DSL Generated')
            )
            
            self.context.current_scenario = scenario
            
        except Exception as e:
            raise DSLRuntimeError(f"Failed to create scenario: {str(e)}") from e
    
    def _dsl_parameter(self, name: str, param_type: str, **kwargs) -> None:
        """DSL function to add a parameter"""
        if not self.context.current_scenario:
            raise DSLRuntimeError("No scenario defined. Use scenario() first.")
        
        try:
            parameter = ScenarioParameter(
                name=name,
                parameter_type=param_type,
                default_value=kwargs.get('default'),
                min_value=kwargs.get('min_value'),
                max_value=kwargs.get('max_value'),
                allowed_values=kwargs.get('allowed_values'),
                required=kwargs.get('required', True),
                description=kwargs.get('description', ''),
                examples=kwargs.get('examples', []),
                depends_on=kwargs.get('depends_on', []),
                conditional_logic=kwargs.get('conditional_logic')
            )
            
            self.context.current_scenario.parameters.append(parameter)
            
        except Exception as e:
            raise DSLRuntimeError(f"Failed to add parameter: {str(e)}") from e
    
    def _dsl_objective(self, name: str, description: str, **kwargs) -> None:
        """DSL function to add an objective"""
        if not self.context.current_scenario:
            raise DSLRuntimeError("No scenario defined. Use scenario() first.")
        
        try:
            objective = ScenarioObjective(
                objective_id=str(uuid.uuid4()),
                name=name,
                description=description,
                objective_type=kwargs.get('type', 'primary'),
                difficulty=kwargs.get('difficulty', ComplexityLevel.INTERMEDIATE),
                success_criteria=kwargs.get('success_criteria', []),
                points=kwargs.get('points', 100),
                time_limit=kwargs.get('time_limit'),
                prerequisites=kwargs.get('prerequisites', []),
                dependencies=kwargs.get('dependencies', []),
                hints=kwargs.get('hints', []),
                guidance=kwargs.get('guidance', ''),
                validation_rules=kwargs.get('validation_rules', [])
            )
            
            self.context.current_scenario.objectives.append(objective)
            
        except Exception as e:
            raise DSLRuntimeError(f"Failed to add objective: {str(e)}") from e
    
    def _dsl_asset(self, name: str, asset_type: str, **kwargs) -> None:
        """DSL function to add an asset"""
        if not self.context.current_scenario:
            raise DSLRuntimeError("No scenario defined. Use scenario() first.")
        
        try:
            asset = ScenarioAsset(
                asset_id=str(uuid.uuid4()),
                asset_type=asset_type,
                name=name,
                configuration=kwargs.get('configuration', {}),
                initial_state=kwargs.get('initial_state', {}),
                vulnerabilities=kwargs.get('vulnerabilities', []),
                security_controls=kwargs.get('security_controls', []),
                monitoring_enabled=kwargs.get('monitoring_enabled', True),
                logging_level=kwargs.get('logging_level', 'standard'),
                startup_scripts=kwargs.get('startup_scripts', []),
                cleanup_scripts=kwargs.get('cleanup_scripts', [])
            )
            
            self.context.current_scenario.required_assets.append(asset)
            
        except Exception as e:
            raise DSLRuntimeError(f"Failed to add asset: {str(e)}") from e
    
    def _dsl_network(self, topology: NetworkTopology) -> None:
        """DSL function to set network topology"""
        if not self.context.current_scenario:
            raise DSLRuntimeError("No scenario defined. Use scenario() first.")
        
        self.context.current_scenario.network_topology = topology
    
    def _dsl_team(self, teams: List[Team]) -> None:
        """DSL function to set recommended teams"""
        if not self.context.current_scenario:
            raise DSLRuntimeError("No scenario defined. Use scenario() first.")
        
        self.context.current_scenario.recommended_teams = teams
    
    def _dsl_duration(self, duration: timedelta) -> None:
        """DSL function to set scenario duration"""
        if not self.context.current_scenario:
            raise DSLRuntimeError("No scenario defined. Use scenario() first.")
        
        self.context.current_scenario.estimated_duration = duration
    
    def _dsl_tags(self, tags: List[str]) -> None:
        """DSL function to set scenario tags"""
        if not self.context.current_scenario:
            raise DSLRuntimeError("No scenario defined. Use scenario() first.")
        
        self.context.current_scenario.tags.update(tags)
    
    def _dsl_description(self, description: str) -> None:
        """DSL function to set scenario description"""
        if not self.context.current_scenario:
            raise DSLRuntimeError("No scenario defined. Use scenario() first.")
        
        self.context.current_scenario.description = description
    
    def _dsl_documentation(self, documentation: str) -> None:
        """DSL function to set scenario documentation"""
        if not self.context.current_scenario:
            raise DSLRuntimeError("No scenario defined. Use scenario() first.")
        
        self.context.current_scenario.documentation = documentation
    
    def _dsl_validation_rule(self, rule: str) -> None:
        """DSL function to add validation rule"""
        if not self.context.current_scenario:
            raise DSLRuntimeError("No scenario defined. Use scenario() first.")
        
        self.context.current_scenario.validation_rules.append(rule)
    
    def _dsl_prerequisite(self, prerequisite: str) -> None:
        """DSL function to add prerequisite"""
        if not self.context.current_scenario:
            raise DSLRuntimeError("No scenario defined. Use scenario() first.")
        
        self.context.current_scenario.prerequisites.append(prerequisite)
    
    def _dsl_dependency(self, dependency: str) -> None:
        """DSL function to add dependency"""
        if not self.context.current_scenario:
            raise DSLRuntimeError("No scenario defined. Use scenario() first.")
        
        self.context.current_scenario.dependencies.append(dependency)
    
    def _dsl_include_template(self, template_id: str) -> None:
        """DSL function to include another template"""
        if template_id in self.context.templates:
            template = self.context.templates[template_id]
            # Merge template components into current scenario
            if self.context.current_scenario:
                self.context.current_scenario.parameters.extend(template.parameters)
                self.context.current_scenario.objectives.extend(template.objectives)
                self.context.current_scenario.required_assets.extend(template.required_assets)
    
    def _dsl_extend_template(self, template_id: str) -> None:
        """DSL function to extend another template"""
        if template_id in self.context.templates:
            template = self.context.templates[template_id]
            # Use template as base for current scenario
            if not self.context.current_scenario:
                self.context.current_scenario = ScenarioTemplate(
                    template_id=str(uuid.uuid4()),
                    name=template.name + " (Extended)",
                    description=template.description,
                    scenario_type=template.scenario_type,
                    category=template.category,
                    complexity=template.complexity,
                    network_topology=template.network_topology,
                    estimated_duration=template.estimated_duration,
                    min_participants=template.min_participants,
                    max_participants=template.max_participants,
                    recommended_teams=template.recommended_teams.copy(),
                    required_roles=template.required_roles.copy(),
                    required_assets=[asset for asset in template.required_assets],
                    objectives=[obj for obj in template.objectives],
                    parameters=[param for param in template.parameters],
                    tags=template.tags.copy()
                )
    
    def _dsl_variable(self, name: str, value: Any) -> None:
        """DSL function to set a variable"""
        self.context.variables[name] = value
    
    def _dsl_condition(self, condition: str, then_block: Callable, else_block: Callable = None) -> None:
        """DSL function for conditional execution"""
        try:
            # Evaluate condition in current context
            result = eval(condition, {"__builtins__": {}}, self.context.variables)
            
            if result:
                then_block()
            elif else_block:
                else_block()
                
        except Exception as e:
            raise DSLRuntimeError(f"Condition evaluation failed: {str(e)}") from e
    
    def _dsl_loop(self, iterable: Any, loop_block: Callable) -> None:
        """DSL function for loop execution"""
        try:
            for item in iterable:
                self.context.variables['_item'] = item
                loop_block()
                
        except Exception as e:
            raise DSLRuntimeError(f"Loop execution failed: {str(e)}") from e
    
    def _dsl_template_function(self, name: str, func: Callable) -> None:
        """DSL function to define a custom template function"""
        self.context.functions[name] = func

class DSLTemplateLibrary:
    """Library of reusable DSL components and patterns"""
    
    def __init__(self):
        self.components = {}
        self.patterns = {}
        self.snippets = {}
        self.logger = logging.getLogger(__name__)
    
    def register_component(self, name: str, dsl_code: str, description: str = "") -> None:
        """Register a reusable DSL component"""
        self.components[name] = {
            'code': dsl_code,
            'description': description,
            'created_at': datetime.now(),
            'usage_count': 0
        }
    
    def register_pattern(self, name: str, pattern_code: str, description: str = "") -> None:
        """Register a reusable DSL pattern"""
        self.patterns[name] = {
            'code': pattern_code,
            'description': description,
            'created_at': datetime.now(),
            'usage_count': 0
        }
    
    def get_component(self, name: str) -> Optional[str]:
        """Get a component by name"""
        if name in self.components:
            self.components[name]['usage_count'] += 1
            return self.components[name]['code']
        return None
    
    def get_pattern(self, name: str) -> Optional[str]:
        """Get a pattern by name"""
        if name in self.patterns:
            self.patterns[name]['usage_count'] += 1
            return self.patterns[name]['code']
        return None
    
    def list_components(self) -> List[Dict[str, Any]]:
        """List all available components"""
        return [
            {
                'name': name,
                'description': info['description'],
                'usage_count': info['usage_count']
            }
            for name, info in self.components.items()
        ]
    
    def list_patterns(self) -> List[Dict[str, Any]]:
        """List all available patterns"""
        return [
            {
                'name': name,
                'description': info['description'],
                'usage_count': info['usage_count']
            }
            for name, info in self.patterns.items()
        ]
    
    def create_default_library(self) -> None:
        """Create default library with common components and patterns"""
        # Common components
        self.register_component(
            'web_server',
            '''
asset("Web Server", "vm",
    configuration={
        "os": "ubuntu",
        "services": ["apache2", "mysql"],
        "ports": [80, 443, 22]
    },
    vulnerabilities=["CVE-2021-44228", "weak_passwords"],
    security_controls=["firewall", "fail2ban"]
)
            '''.strip(),
            "Standard web server with common vulnerabilities"
        )
        
        self.register_component(
            'database_server',
            '''
asset("Database Server", "vm",
    configuration={
        "os": "ubuntu",
        "services": ["mysql"],
        "ports": [3306, 22]
    },
    vulnerabilities=["default_credentials", "unencrypted_traffic"],
    security_controls=["access_control", "audit_logging"]
)
            '''.strip(),
            "Database server with typical misconfigurations"
        )
        
        # Common patterns
        self.register_pattern(
            'phishing_campaign',
            '''
scenario("Phishing Campaign Exercise",
    type=ScenarioType.TRAINING,
    category=ScenarioCategory.INITIAL_ACCESS,
    complexity=ComplexityLevel.INTERMEDIATE
)

description("Simulate a phishing campaign targeting employee credentials")

objective("Send Phishing Emails", 
    "Successfully deliver phishing emails to target users",
    type="primary",
    points=50,
    success_criteria=["emails_delivered > 10", "click_rate > 0.1"]
)

objective("Harvest Credentials",
    "Collect user credentials from phishing landing page", 
    type="primary",
    points=100,
    success_criteria=["credentials_collected > 5"]
)

parameter("target_count", "int", default=20, min_value=5, max_value=100,
    description="Number of target users to send phishing emails to")

parameter("email_template", "enum", 
    allowed_values=["urgent_security", "it_support", "hr_notice"],
    default="urgent_security",
    description="Phishing email template to use")
            '''.strip(),
            "Complete phishing campaign scenario pattern"
        )
        
        self.register_pattern(
            'incident_response',
            '''
scenario("Incident Response Exercise",
    type=ScenarioType.TRAINING,
    category=ScenarioCategory.INCIDENT_RESPONSE,
    complexity=ComplexityLevel.ADVANCED
)

description("Practice incident response procedures for a security breach")

objective("Detect Incident",
    "Identify and classify the security incident",
    type="primary",
    points=75,
    time_limit=timedelta(minutes=30),
    success_criteria=["incident_detected", "classification_correct"]
)

objective("Contain Threat",
    "Implement containment measures to prevent spread",
    type="primary", 
    points=100,
    success_criteria=["threat_contained", "systems_isolated"]
)

objective("Eradicate Threat",
    "Remove threat from all affected systems",
    type="primary",
    points=125,
    success_criteria=["threat_removed", "systems_clean"]
)

parameter("incident_type", "enum",
    allowed_values=["malware", "data_breach", "insider_threat", "ddos"],
    default="malware",
    description="Type of security incident to simulate")

parameter("severity_level", "enum",
    allowed_values=["low", "medium", "high", "critical"],
    default="medium", 
    description="Severity level of the incident")
            '''.strip(),
            "Comprehensive incident response exercise pattern"
        )

def create_dsl_example() -> str:
    """Create an example DSL scenario"""
    return '''
# Red Team vs Blue Team Exercise
scenario("Advanced Persistent Threat Simulation",
    type=ScenarioType.LIVE_EXERCISE,
    category=ScenarioCategory.FULL_CAMPAIGN,
    complexity=ComplexityLevel.ADVANCED,
    duration=timedelta(hours=4)
)

description("""
Simulate an Advanced Persistent Threat (APT) campaign where Red Team
attempts to establish persistence in a corporate network while Blue Team
defends and responds to the attack.
""")

# Network topology
network(NetworkTopology.ENTERPRISE_NETWORK)

# Teams
team([Team.RED_TEAM, Team.BLUE_TEAM])

# Parameters
parameter("target_network", "string", 
    default="192.168.1.0/24",
    description="Target network range for the exercise")

parameter("attack_duration", "int",
    default=240, min_value=60, max_value=480,
    description="Maximum attack duration in minutes")

parameter("stealth_mode", "bool",
    default=True,
    description="Enable stealth mode for Red Team operations")

# Assets
asset("Domain Controller", "vm",
    configuration={
        "os": "windows_server_2019",
        "services": ["active_directory", "dns", "dhcp"],
        "ip": "192.168.1.10"
    },
    vulnerabilities=["zerologon", "weak_admin_password"],
    security_controls=["windows_defender", "audit_logging"]
)

asset("Web Server", "vm", 
    configuration={
        "os": "ubuntu_20.04",
        "services": ["apache2", "mysql", "wordpress"],
        "ip": "192.168.1.20"
    },
    vulnerabilities=["outdated_wordpress", "sql_injection"],
    security_controls=["fail2ban", "mod_security"]
)

asset("Employee Workstation", "vm",
    configuration={
        "os": "windows_10",
        "services": ["rdp", "smb"],
        "ip": "192.168.1.100"
    },
    vulnerabilities=["unpatched_office", "weak_user_password"],
    security_controls=["windows_defender", "user_account_control"]
)

# Red Team Objectives
objective("Initial Access",
    "Gain initial foothold in the target network",
    type="primary",
    points=100,
    time_limit=timedelta(minutes=60),
    success_criteria=["shell_obtained", "network_access_confirmed"],
    hints=["Check web applications for vulnerabilities", "Consider phishing attacks"]
)

objective("Privilege Escalation", 
    "Escalate privileges to administrator/root level",
    type="primary",
    points=150,
    prerequisites=["Initial Access"],
    success_criteria=["admin_privileges_obtained"],
    hints=["Look for local privilege escalation vulnerabilities"]
)

objective("Lateral Movement",
    "Move laterally to at least 2 additional systems",
    type="primary", 
    points=200,
    prerequisites=["Privilege Escalation"],
    success_criteria=["lateral_movement_count >= 2"],
    hints=["Use credential harvesting", "Exploit trust relationships"]
)

objective("Persistence",
    "Establish persistent access mechanisms",
    type="primary",
    points=175,
    success_criteria=["backdoor_installed", "persistence_verified"],
    hints=["Consider scheduled tasks", "Service installations"]
)

objective("Data Exfiltration",
    "Locate and exfiltrate sensitive data",
    type="secondary",
    points=125,
    success_criteria=["sensitive_data_found", "data_exfiltrated"],
    hints=["Look for databases", "Check file shares"]
)

# Blue Team Objectives  
objective("Threat Detection",
    "Detect and alert on Red Team activities",
    type="primary",
    points=100,
    time_limit=timedelta(minutes=90),
    success_criteria=["threats_detected >= 3", "false_positive_rate < 0.2"],
    hints=["Monitor network traffic", "Analyze system logs"]
)

objective("Incident Response",
    "Respond to detected threats appropriately", 
    type="primary",
    points=150,
    prerequisites=["Threat Detection"],
    success_criteria=["incidents_created", "response_initiated"],
    hints=["Follow incident response procedures", "Document all actions"]
)

objective("Threat Containment",
    "Contain identified threats to prevent spread",
    type="primary",
    points=175,
    success_criteria=["threats_contained", "spread_prevented"],
    hints=["Isolate affected systems", "Block malicious traffic"]
)

objective("Forensic Analysis",
    "Perform forensic analysis on compromised systems",
    type="secondary", 
    points=125,
    success_criteria=["forensics_completed", "attack_timeline_created"],
    hints=["Preserve evidence", "Analyze artifacts"]
)

# Validation rules
validation_rule("Red Team must not cause permanent damage")
validation_rule("Blue Team must document all actions")
validation_rule("Exercise must complete within time limit")

# Tags
tags(["apt", "red_vs_blue", "enterprise", "advanced"])

# Documentation
documentation("""
This scenario simulates a realistic APT campaign in an enterprise environment.
Red Team should focus on stealth and persistence while Blue Team practices
detection and response capabilities.

Success is measured by:
- Red Team: Achieving objectives while maintaining stealth
- Blue Team: Detecting and responding to threats effectively

The exercise includes realistic network topology with common enterprise
assets and vulnerabilities that mirror real-world environments.
""")
'''