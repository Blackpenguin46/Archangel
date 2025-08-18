#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - DSL Parser and Validation Engine
Advanced parsing and validation system for scenario DSL
"""

import ast
import logging
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from enum import Enum
import json

from .scenario_dsl import ScenarioDSL, DSLError, DSLSyntaxError, DSLValidationError
from .scenario_templates import ScenarioTemplate, ScenarioParameter, ScenarioObjective

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"           # Basic syntax and structure validation
    STANDARD = "standard"     # Standard validation with semantic checks
    STRICT = "strict"         # Strict validation with advanced checks
    COMPREHENSIVE = "comprehensive"  # Full validation with optimization suggestions

@dataclass
class ValidationResult:
    """Result of DSL validation"""
    valid: bool
    level: ValidationLevel
    
    # Validation details
    syntax_errors: List[str] = field(default_factory=list)
    semantic_errors: List[str] = field(default_factory=list)
    logic_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Performance metrics
    validation_time: float = 0.0
    complexity_score: float = 0.0
    
    # Detailed analysis
    ast_analysis: Dict[str, Any] = field(default_factory=dict)
    dependency_analysis: Dict[str, Any] = field(default_factory=dict)
    security_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    validated_at: datetime = field(default_factory=datetime.now)
    validator_version: str = "1.0"

@dataclass
class ParseResult:
    """Result of DSL parsing"""
    success: bool
    scenario: Optional[ScenarioTemplate] = None
    
    # Parse details
    parse_time: float = 0.0
    ast_nodes: int = 0
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    parsed_at: datetime = field(default_factory=datetime.now)
    parser_version: str = "1.0"

class DSLParser:
    """
    Advanced DSL parser with comprehensive validation and error reporting.
    
    Features:
    - Multi-level validation (basic to comprehensive)
    - Detailed error reporting with line numbers and suggestions
    - Performance analysis and optimization recommendations
    - Security validation for DSL code
    - Dependency analysis and circular dependency detection
    """
    
    def __init__(self):
        self.dsl = ScenarioDSL()
        self.validation_rules = self._create_validation_rules()
        self.security_rules = self._create_security_rules()
        self.logger = logging.getLogger(__name__)
    
    def parse(self, dsl_code: str, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ParseResult:
        """Parse DSL code with specified validation level"""
        start_time = datetime.now()
        
        try:
            # Validate first
            validation_result = self.validate(dsl_code, validation_level)
            
            if not validation_result.valid:
                return ParseResult(
                    success=False,
                    errors=validation_result.syntax_errors + validation_result.semantic_errors,
                    warnings=validation_result.warnings,
                    parse_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Parse scenario
            scenario = self.dsl.parse_scenario(dsl_code)
            
            # Count AST nodes for complexity analysis
            tree = ast.parse(dsl_code)
            ast_nodes = len(list(ast.walk(tree)))
            
            return ParseResult(
                success=True,
                scenario=scenario,
                parse_time=(datetime.now() - start_time).total_seconds(),
                ast_nodes=ast_nodes,
                warnings=validation_result.warnings
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse DSL: {e}")
            return ParseResult(
                success=False,
                errors=[str(e)],
                parse_time=(datetime.now() - start_time).total_seconds()
            )
    
    def validate(self, dsl_code: str, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Comprehensive DSL validation"""
        start_time = datetime.now()
        
        result = ValidationResult(valid=True, level=level)
        
        try:
            # Basic syntax validation
            result = self._validate_syntax(dsl_code, result)
            
            if level == ValidationLevel.BASIC:
                result.validation_time = (datetime.now() - start_time).total_seconds()
                return result
            
            # Semantic validation
            result = self._validate_semantics(dsl_code, result)
            
            if level == ValidationLevel.STANDARD:
                result.validation_time = (datetime.now() - start_time).total_seconds()
                return result
            
            # Strict validation
            result = self._validate_logic(dsl_code, result)
            result = self._validate_security(dsl_code, result)
            
            if level == ValidationLevel.STRICT:
                result.validation_time = (datetime.now() - start_time).total_seconds()
                return result
            
            # Comprehensive validation
            result = self._analyze_dependencies(dsl_code, result)
            result = self._analyze_performance(dsl_code, result)
            result = self._generate_suggestions(dsl_code, result)
            
            result.validation_time = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            result.valid = False
            result.syntax_errors.append(f"Validation error: {str(e)}")
            result.validation_time = (datetime.now() - start_time).total_seconds()
            return result
    
    def _validate_syntax(self, dsl_code: str, result: ValidationResult) -> ValidationResult:
        """Validate DSL syntax"""
        try:
            # Parse AST
            tree = ast.parse(dsl_code)
            
            # Store AST analysis
            result.ast_analysis = {
                'total_nodes': len(list(ast.walk(tree))),
                'function_calls': len([n for n in ast.walk(tree) if isinstance(n, ast.Call)]),
                'assignments': len([n for n in ast.walk(tree) if isinstance(n, ast.Assign)]),
                'imports': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
            }
            
            # Check for dangerous constructs
            dangerous_nodes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', 'open', '__import__', 'compile']:
                        dangerous_nodes.append(f"Line {node.lineno}: Dangerous function '{node.func.id}'")
                
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name not in ['datetime', 'timedelta']:
                                dangerous_nodes.append(f"Line {node.lineno}: Restricted import '{alias.name}'")
            
            if dangerous_nodes:
                result.valid = False
                result.syntax_errors.extend(dangerous_nodes)
            
        except SyntaxError as e:
            result.valid = False
            result.syntax_errors.append(f"Line {e.lineno}: {e.msg}")
        except Exception as e:
            result.valid = False
            result.syntax_errors.append(f"Syntax validation failed: {str(e)}")
        
        return result
    
    def _validate_semantics(self, dsl_code: str, result: ValidationResult) -> ValidationResult:
        """Validate DSL semantics"""
        try:
            # Use DSL's built-in validation
            dsl_validation = self.dsl.validate_dsl(dsl_code)
            
            if not dsl_validation['valid']:
                result.valid = False
                result.semantic_errors.extend(dsl_validation['errors'])
            
            result.warnings.extend(dsl_validation['warnings'])
            
            # Additional semantic checks
            lines = dsl_code.split('\n')
            
            # Check for scenario definition
            scenario_defined = any('scenario(' in line for line in lines)
            if not scenario_defined:
                result.semantic_errors.append("No scenario definition found")
                result.valid = False
            
            # Check for objectives
            objectives_defined = any('objective(' in line for line in lines)
            if not objectives_defined:
                result.warnings.append("No objectives defined")
            
            # Check for assets
            assets_defined = any('asset(' in line for line in lines)
            if not assets_defined:
                result.warnings.append("No assets defined")
            
            # Check for balanced quotes and parentheses
            quote_balance = self._check_quote_balance(dsl_code)
            if not quote_balance['balanced']:
                result.valid = False
                result.semantic_errors.append(f"Unbalanced quotes: {quote_balance['error']}")
            
            paren_balance = self._check_parentheses_balance(dsl_code)
            if not paren_balance['balanced']:
                result.valid = False
                result.semantic_errors.append(f"Unbalanced parentheses: {paren_balance['error']}")
            
        except Exception as e:
            result.valid = False
            result.semantic_errors.append(f"Semantic validation failed: {str(e)}")
        
        return result
    
    def _validate_logic(self, dsl_code: str, result: ValidationResult) -> ValidationResult:
        """Validate DSL logic and consistency"""
        try:
            lines = dsl_code.split('\n')
            
            # Check for logical inconsistencies
            parameters = []
            objectives = []
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Extract parameter definitions
                if line.startswith('parameter('):
                    param_match = re.search(r'parameter\s*\(\s*["\']([^"\']+)["\']', line)
                    if param_match:
                        parameters.append((param_match.group(1), i))
                
                # Extract objective definitions
                if line.startswith('objective('):
                    obj_match = re.search(r'objective\s*\(\s*["\']([^"\']+)["\']', line)
                    if obj_match:
                        objectives.append((obj_match.group(1), i))
            
            # Check for duplicate parameters
            param_names = [p[0] for p in parameters]
            duplicates = set([name for name in param_names if param_names.count(name) > 1])
            for dup in duplicates:
                lines_with_dup = [str(line) for name, line in parameters if name == dup]
                result.logic_errors.append(f"Duplicate parameter '{dup}' on lines: {', '.join(lines_with_dup)}")
            
            # Check for duplicate objectives
            obj_names = [o[0] for o in objectives]
            duplicates = set([name for name in obj_names if obj_names.count(name) > 1])
            for dup in duplicates:
                lines_with_dup = [str(line) for name, line in objectives if name == dup]
                result.logic_errors.append(f"Duplicate objective '{dup}' on lines: {', '.join(lines_with_dup)}")
            
            # Check for unreferenced parameters
            param_usage = {}
            for param_name, _ in parameters:
                usage_count = sum(1 for line in lines if param_name in line and not line.strip().startswith('parameter('))
                param_usage[param_name] = usage_count
            
            for param_name, usage_count in param_usage.items():
                if usage_count == 0:
                    result.warnings.append(f"Parameter '{param_name}' is defined but never used")
            
            if result.logic_errors:
                result.valid = False
            
        except Exception as e:
            result.logic_errors.append(f"Logic validation failed: {str(e)}")
        
        return result
    
    def _validate_security(self, dsl_code: str, result: ValidationResult) -> ValidationResult:
        """Validate DSL for security issues"""
        try:
            security_issues = []
            
            # Check against security rules
            for rule_name, rule_func in self.security_rules.items():
                issues = rule_func(dsl_code)
                if issues:
                    security_issues.extend([f"{rule_name}: {issue}" for issue in issues])
            
            result.security_analysis = {
                'issues_found': len(security_issues),
                'issues': security_issues,
                'security_score': max(0, 100 - len(security_issues) * 10)
            }
            
            if security_issues:
                result.warnings.extend(security_issues)
            
        except Exception as e:
            result.security_analysis = {'error': str(e)}
        
        return result
    
    def _analyze_dependencies(self, dsl_code: str, result: ValidationResult) -> ValidationResult:
        """Analyze dependencies and detect circular references"""
        try:
            dependencies = {}
            prerequisites = {}
            
            lines = dsl_code.split('\n')
            current_objective = None
            
            for line in lines:
                line = line.strip()
                
                # Track current objective
                obj_match = re.search(r'objective\s*\(\s*["\']([^"\']+)["\']', line)
                if obj_match:
                    current_objective = obj_match.group(1)
                    dependencies[current_objective] = []
                    prerequisites[current_objective] = []
                
                # Extract dependencies
                if 'dependencies=' in line and current_objective:
                    dep_match = re.search(r'dependencies\s*=\s*\[([^\]]+)\]', line)
                    if dep_match:
                        deps = [d.strip().strip('"\'') for d in dep_match.group(1).split(',')]
                        dependencies[current_objective].extend(deps)
                
                # Extract prerequisites
                if 'prerequisites=' in line and current_objective:
                    prereq_match = re.search(r'prerequisites\s*=\s*\[([^\]]+)\]', line)
                    if prereq_match:
                        prereqs = [p.strip().strip('"\'') for p in prereq_match.group(1).split(',')]
                        prerequisites[current_objective].extend(prereqs)
            
            # Check for circular dependencies
            circular_deps = self._detect_circular_dependencies(dependencies)
            if circular_deps:
                result.logic_errors.extend([f"Circular dependency detected: {' -> '.join(cycle)}" for cycle in circular_deps])
                result.valid = False
            
            # Check for missing dependencies
            all_objectives = set(dependencies.keys())
            missing_deps = []
            
            for obj, deps in dependencies.items():
                for dep in deps:
                    if dep not in all_objectives:
                        missing_deps.append(f"Objective '{obj}' depends on undefined objective '{dep}'")
            
            if missing_deps:
                result.logic_errors.extend(missing_deps)
                result.valid = False
            
            result.dependency_analysis = {
                'total_objectives': len(dependencies),
                'dependencies': dependencies,
                'prerequisites': prerequisites,
                'circular_dependencies': circular_deps,
                'missing_dependencies': missing_deps
            }
            
        except Exception as e:
            result.dependency_analysis = {'error': str(e)}
        
        return result
    
    def _analyze_performance(self, dsl_code: str, result: ValidationResult) -> ValidationResult:
        """Analyze DSL performance characteristics"""
        try:
            lines = dsl_code.split('\n')
            
            # Calculate complexity metrics
            complexity_factors = {
                'lines_of_code': len([l for l in lines if l.strip()]),
                'function_calls': len([l for l in lines if '(' in l and ')' in l]),
                'parameters': len([l for l in lines if 'parameter(' in l]),
                'objectives': len([l for l in lines if 'objective(' in l]),
                'assets': len([l for l in lines if 'asset(' in l]),
                'conditions': len([l for l in lines if 'condition(' in l]),
                'loops': len([l for l in lines if 'loop(' in l])
            }
            
            # Calculate complexity score (0-100)
            complexity_score = min(100, sum([
                complexity_factors['lines_of_code'] * 0.1,
                complexity_factors['function_calls'] * 0.5,
                complexity_factors['parameters'] * 2,
                complexity_factors['objectives'] * 3,
                complexity_factors['assets'] * 2,
                complexity_factors['conditions'] * 5,
                complexity_factors['loops'] * 10
            ]))
            
            result.complexity_score = complexity_score
            
            # Performance warnings
            if complexity_factors['lines_of_code'] > 500:
                result.warnings.append("Large DSL file (>500 lines) may impact parsing performance")
            
            if complexity_factors['objectives'] > 20:
                result.warnings.append("Many objectives (>20) may impact scenario execution")
            
            if complexity_factors['assets'] > 50:
                result.warnings.append("Many assets (>50) may impact resource usage")
            
        except Exception as e:
            result.complexity_score = 0
            result.warnings.append(f"Performance analysis failed: {str(e)}")
        
        return result
    
    def _generate_suggestions(self, dsl_code: str, result: ValidationResult) -> ValidationResult:
        """Generate optimization and improvement suggestions"""
        try:
            suggestions = []
            
            lines = dsl_code.split('\n')
            
            # Check for missing documentation
            has_description = any('description(' in line for line in lines)
            has_documentation = any('documentation(' in line for line in lines)
            
            if not has_description:
                suggestions.append("Add a description() call to document the scenario purpose")
            
            if not has_documentation:
                suggestions.append("Add documentation() for detailed scenario information")
            
            # Check for missing tags
            has_tags = any('tags(' in line for line in lines)
            if not has_tags:
                suggestions.append("Add tags() to improve scenario discoverability")
            
            # Check for hardcoded values
            hardcoded_ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', dsl_code)
            if hardcoded_ips:
                suggestions.append("Consider parameterizing hardcoded IP addresses")
            
            # Check for missing validation rules
            has_validation = any('validation_rule(' in line for line in lines)
            if not has_validation:
                suggestions.append("Add validation_rule() calls to ensure scenario integrity")
            
            # Performance suggestions
            if result.complexity_score > 70:
                suggestions.append("Consider breaking complex scenario into smaller components")
            
            # Security suggestions
            if result.security_analysis.get('security_score', 100) < 80:
                suggestions.append("Review security analysis warnings and address issues")
            
            result.suggestions = suggestions
            
        except Exception as e:
            result.suggestions = [f"Suggestion generation failed: {str(e)}"]
        
        return result
    
    def _check_quote_balance(self, code: str) -> Dict[str, Any]:
        """Check if quotes are balanced in the code"""
        single_quotes = code.count("'") - code.count("\\'")
        double_quotes = code.count('"') - code.count('\\"')
        
        return {
            'balanced': single_quotes % 2 == 0 and double_quotes % 2 == 0,
            'error': f"Unbalanced quotes: {single_quotes % 2} single, {double_quotes % 2} double"
        }
    
    def _check_parentheses_balance(self, code: str) -> Dict[str, Any]:
        """Check if parentheses are balanced in the code"""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for i, char in enumerate(code):
            if char in pairs:
                stack.append((char, i))
            elif char in pairs.values():
                if not stack:
                    return {'balanced': False, 'error': f"Unmatched closing '{char}' at position {i}"}
                
                open_char, open_pos = stack.pop()
                if pairs[open_char] != char:
                    return {'balanced': False, 'error': f"Mismatched '{open_char}' at {open_pos} and '{char}' at {i}"}
        
        if stack:
            open_char, open_pos = stack[-1]
            return {'balanced': False, 'error': f"Unmatched opening '{open_char}' at position {open_pos}"}
        
        return {'balanced': True, 'error': None}
    
    def _detect_circular_dependencies(self, dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies using DFS"""
        def dfs(node, path, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in dependencies.get(node, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor, path, visited, rec_stack)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            path.pop()
            rec_stack.remove(node)
            return None
        
        visited = set()
        cycles = []
        
        for node in dependencies:
            if node not in visited:
                cycle = dfs(node, [], visited, set())
                if cycle:
                    cycles.append(cycle)
        
        return cycles
    
    def _create_validation_rules(self) -> Dict[str, Callable]:
        """Create validation rules"""
        return {
            'required_scenario': lambda code: [] if 'scenario(' in code else ['No scenario definition found'],
            'balanced_quotes': lambda code: [] if self._check_quote_balance(code)['balanced'] else ['Unbalanced quotes'],
            'balanced_parentheses': lambda code: [] if self._check_parentheses_balance(code)['balanced'] else ['Unbalanced parentheses']
        }
    
    def _create_security_rules(self) -> Dict[str, Callable]:
        """Create security validation rules"""
        def check_dangerous_functions(code):
            dangerous = ['exec', 'eval', 'open', '__import__', 'compile']
            issues = []
            for func in dangerous:
                if f'{func}(' in code:
                    issues.append(f"Dangerous function '{func}' detected")
            return issues
        
        def check_file_operations(code):
            file_ops = ['open(', 'file(', 'with open']
            issues = []
            for op in file_ops:
                if op in code:
                    issues.append(f"File operation '{op}' detected - ensure it's necessary")
            return issues
        
        def check_network_operations(code):
            network_ops = ['socket', 'urllib', 'requests', 'http']
            issues = []
            for op in network_ops:
                if op in code:
                    issues.append(f"Network operation '{op}' detected - review for security")
            return issues
        
        return {
            'dangerous_functions': check_dangerous_functions,
            'file_operations': check_file_operations,
            'network_operations': check_network_operations
        }

class DSLFormatter:
    """DSL code formatter for consistent styling"""
    
    def __init__(self):
        self.indent_size = 4
        self.logger = logging.getLogger(__name__)
    
    def format(self, dsl_code: str) -> str:
        """Format DSL code with consistent styling"""
        try:
            lines = dsl_code.split('\n')
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                
                if not stripped or stripped.startswith('#'):
                    formatted_lines.append(stripped)
                    continue
                
                # Adjust indent level
                if stripped.endswith(':'):
                    formatted_lines.append(' ' * (indent_level * self.indent_size) + stripped)
                    indent_level += 1
                elif stripped in [')', ']', '}']:
                    indent_level = max(0, indent_level - 1)
                    formatted_lines.append(' ' * (indent_level * self.indent_size) + stripped)
                else:
                    formatted_lines.append(' ' * (indent_level * self.indent_size) + stripped)
            
            return '\n'.join(formatted_lines)
            
        except Exception as e:
            self.logger.error(f"Failed to format DSL code: {e}")
            return dsl_code  # Return original if formatting fails