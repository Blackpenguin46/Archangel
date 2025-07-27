"""
LLM Planning Engine for Archangel

This module implements the LLM-based planning engine that provides autonomous
operation planning, natural language objective parsing, and adaptive strategy
modification for complex security operations.
"""

import asyncio
import logging
import json
import re
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib

from .models import AIModelManager, ModelType, InferenceRequest, InferenceResult

logger = logging.getLogger(__name__)


class OperationPhase(Enum):
    """Phases of security operations"""
    RECONNAISSANCE = "reconnaissance"
    SCANNING = "scanning"
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    REPORTING = "reporting"
    OSINT = "osint"
    WEB_AUDIT = "web_audit"
    NETWORK_COMPROMISE = "network_compromise"


class OperationType(Enum):
    """Types of security operations"""
    PENETRATION_TEST = "penetration_test"
    OSINT_INVESTIGATION = "osint_investigation"
    WEB_APPLICATION_AUDIT = "web_application_audit"
    NETWORK_ASSESSMENT = "network_assessment"
    VULNERABILITY_SCAN = "vulnerability_scan"
    EXPLOIT_DEVELOPMENT = "exploit_development"


class RiskLevel(Enum):
    """Risk levels for operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StealthLevel(Enum):
    """Stealth levels for operations"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class OperationConstraints:
    """Constraints and limitations for operations"""
    scope_boundaries: List[str] = field(default_factory=list)
    prohibited_actions: List[str] = field(default_factory=list)
    time_windows: List[Dict[str, str]] = field(default_factory=list)
    stealth_requirements: StealthLevel = StealthLevel.MEDIUM
    max_duration_hours: Optional[int] = None
    authorized_targets: List[str] = field(default_factory=list)
    excluded_targets: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    documentation_standards: List[str] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    approval_required_actions: List[str] = field(default_factory=list)


@dataclass
class OperationObjective:
    """Parsed operation objective"""
    raw_input: str
    operation_type: OperationType
    primary_targets: List[str]
    secondary_objectives: List[str]
    success_criteria: List[str]
    constraints: OperationConstraints
    priority: int = 1
    estimated_duration: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM


@dataclass
class OperationPhaseStep:
    """Individual step within an operation phase"""
    step_id: str
    name: str
    description: str
    tools_required: List[str]
    estimated_duration: str
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    failure_alternatives: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    requires_approval: bool = False
    stealth_considerations: List[str] = field(default_factory=list)


@dataclass
class OperationPhaseplan:
    """Plan for a specific operation phase"""
    phase: OperationPhase
    name: str
    description: str
    steps: List[OperationPhaseStep]
    estimated_duration: str
    success_criteria: List[str]
    failure_handling: List[str]
    dependencies: List[OperationPhase] = field(default_factory=list)
    parallel_execution: bool = False


@dataclass
class OperationPlan:
    """Complete operation plan"""
    plan_id: str
    objective: OperationObjective
    phases: List[OperationPhaseplan]
    overall_strategy: str
    estimated_total_duration: str
    risk_assessment: Dict[str, Any]
    contingency_plans: List[Dict[str, Any]]
    resource_requirements: Dict[str, Any]
    success_metrics: List[str]
    created_at: datetime
    last_modified: datetime
    version: int = 1


@dataclass
class AdaptationContext:
    """Context for adaptive strategy modification"""
    current_phase: OperationPhase
    completed_steps: List[str]
    failed_steps: List[str]
    discovered_information: Dict[str, Any]
    environmental_changes: List[str]
    new_constraints: List[str]
    performance_metrics: Dict[str, float]
    time_elapsed: float
    remaining_time: Optional[float]


class ObjectiveParser:
    """Parses natural language objectives into structured operation objectives"""
    
    # Pattern matching for different operation types
    OPERATION_PATTERNS = {
        OperationType.PENETRATION_TEST: [
            r"penetration test|pentest|pen test",
            r"security assessment|security audit",
            r"vulnerability assessment",
            r"red team|red teaming"
        ],
        OperationType.OSINT_INVESTIGATION: [
            r"osint|open source intelligence",
            r"reconnaissance|recon",
            r"information gathering",
            r"intelligence gathering"
        ],
        OperationType.WEB_APPLICATION_AUDIT: [
            r"web application|web app",
            r"application security|app security",
            r"web security audit",
            r"web vulnerability"
        ],
        OperationType.NETWORK_ASSESSMENT: [
            r"network assessment|network audit",
            r"network security|network scan",
            r"infrastructure assessment"
        ],
        OperationType.VULNERABILITY_SCAN: [
            r"vulnerability scan|vuln scan",
            r"security scan",
            r"automated scan"
        ],
        OperationType.EXPLOIT_DEVELOPMENT: [
            r"exploit development|exploit dev",
            r"proof of concept|poc",
            r"weaponize|weaponization"
        ]
    }
    
    # Target extraction patterns
    TARGET_PATTERNS = [
        r"https?://([a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})?)",  # URLs
        r"(?:of|against|on)\s+([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",  # Domain names
        r"(?:of|against|on)\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?)",  # IP addresses/CIDR
        r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?)",  # Standalone IP/CIDR
        r"([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",  # Standalone domain names
        r"(?:test|scan|audit)\s+([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",  # Test/scan domain
        r"(?:test|scan|audit)\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?)",  # Test/scan IP
        r"(?:hack|attack|exploit)\s+([a-zA-Z0-9.-]+)",  # Attack targets including localhost
        r"\b(localhost|127\.0\.0\.1)\b",  # Localhost specifically
    ]
    
    # Constraint extraction patterns
    CONSTRAINT_PATTERNS = {
        'stealth': r"stealth|stealthy|quiet|covert|undetected",
        'time_limit': r"within\s+(\d+)\s+(hours?|days?|minutes?)",
        'scope_limit': r"limited to|restricted to|only\s+([^,]+)",
        'no_damage': r"no damage|non-destructive|safe|read-only",
        'compliance': r"([a-zA-Z]{3,})\s+compliant|compliant\s+with\s+([a-zA-Z]{3,})|following\s+([a-zA-Z]{3,})",
    }
    
    def __init__(self):
        self.compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        for op_type, patterns in self.OPERATION_PATTERNS.items():
            self.compiled_patterns[op_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    async def parse_objective(self, raw_input: str) -> OperationObjective:
        """Parse natural language input into structured objective"""
        try:
            logger.info(f"Parsing objective: {raw_input}")
            
            # Determine operation type
            operation_type = self._identify_operation_type(raw_input)
            
            # Extract targets
            targets = self._extract_targets(raw_input)
            
            # Extract constraints
            constraints = self._extract_constraints(raw_input)
            
            # Extract secondary objectives
            secondary_objectives = self._extract_secondary_objectives(raw_input, operation_type)
            
            # Generate success criteria
            success_criteria = self._generate_success_criteria(operation_type, targets)
            
            # Estimate risk level
            risk_level = self._assess_risk_level(raw_input, constraints)
            
            objective = OperationObjective(
                raw_input=raw_input,
                operation_type=operation_type,
                primary_targets=targets,
                secondary_objectives=secondary_objectives,
                success_criteria=success_criteria,
                constraints=constraints,
                risk_level=risk_level
            )
            
            logger.info(f"Parsed objective: {operation_type.value} targeting {len(targets)} targets")
            return objective
            
        except Exception as e:
            logger.error(f"Failed to parse objective: {e}")
            raise
    
    def _identify_operation_type(self, text: str) -> OperationType:
        """Identify the type of operation from text"""
        text_lower = text.lower()
        
        # Score each operation type based on pattern matches
        scores = {}
        for op_type, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(text_lower)
                score += len(matches)
            scores[op_type] = score
        
        # Return the highest scoring operation type
        if scores:
            best_match = max(scores.items(), key=lambda x: x[1])
            if best_match[1] > 0:
                return best_match[0]
        
        # Default to penetration test if no clear match
        return OperationType.PENETRATION_TEST
    
    def _extract_targets(self, text: str) -> List[str]:
        """Extract target information from text"""
        targets = []
        
        for pattern in self.TARGET_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            targets.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_targets = []
        for target in targets:
            if target not in seen:
                seen.add(target)
                unique_targets.append(target)
        
        return unique_targets
    
    def _extract_constraints(self, text: str) -> OperationConstraints:
        """Extract operation constraints from text"""
        constraints = OperationConstraints()
        text_lower = text.lower()
        
        # Check for stealth requirements
        if re.search(self.CONSTRAINT_PATTERNS['stealth'], text_lower):
            constraints.stealth_requirements = StealthLevel.HIGH
        
        # Extract time limits
        time_matches = re.findall(self.CONSTRAINT_PATTERNS['time_limit'], text_lower)
        if time_matches:
            duration, unit = time_matches[0]
            if unit.startswith('hour'):
                constraints.max_duration_hours = int(duration)
            elif unit.startswith('day'):
                constraints.max_duration_hours = int(duration) * 24
        
        # Extract scope limitations
        scope_matches = re.findall(self.CONSTRAINT_PATTERNS['scope_limit'], text_lower)
        if scope_matches:
            constraints.scope_boundaries.extend(scope_matches)
        
        # Check for damage restrictions
        if re.search(self.CONSTRAINT_PATTERNS['no_damage'], text_lower):
            constraints.prohibited_actions.extend([
                "destructive_operations",
                "data_modification",
                "service_disruption"
            ])
        
        # Extract compliance requirements
        compliance_matches = re.findall(self.CONSTRAINT_PATTERNS['compliance'], text_lower)
        if compliance_matches:
            # Flatten the tuple results and filter out empty strings, convert to uppercase
            flattened = [match.upper() for group in compliance_matches for match in group if match]
            constraints.compliance_requirements.extend(flattened)
        
        return constraints
    
    def _extract_secondary_objectives(self, text: str, operation_type: OperationType) -> List[str]:
        """Extract secondary objectives based on operation type"""
        secondary_objectives = []
        text_lower = text.lower()
        
        # Common secondary objectives by operation type
        if operation_type == OperationType.PENETRATION_TEST:
            if "privilege escalation" in text_lower or "escalate" in text_lower:
                secondary_objectives.append("privilege_escalation")
            if "lateral movement" in text_lower or "pivot" in text_lower:
                secondary_objectives.append("lateral_movement")
            if "persistence" in text_lower:
                secondary_objectives.append("establish_persistence")
        
        elif operation_type == OperationType.OSINT_INVESTIGATION:
            if "employee" in text_lower or "staff" in text_lower:
                secondary_objectives.append("employee_enumeration")
            if "email" in text_lower:
                secondary_objectives.append("email_harvesting")
            if "breach" in text_lower or "leak" in text_lower:
                secondary_objectives.append("breach_data_analysis")
        
        return secondary_objectives
    
    def _generate_success_criteria(self, operation_type: OperationType, targets: List[str]) -> List[str]:
        """Generate success criteria based on operation type"""
        criteria = []
        
        if operation_type == OperationType.PENETRATION_TEST:
            criteria.extend([
                "Complete reconnaissance phase",
                "Identify and validate vulnerabilities",
                "Demonstrate successful exploitation",
                "Generate comprehensive report"
            ])
            if len(targets) > 1:
                criteria.append(f"Assess all {len(targets)} specified targets")
        
        elif operation_type == OperationType.OSINT_INVESTIGATION:
            criteria.extend([
                "Gather comprehensive target intelligence",
                "Identify potential attack vectors",
                "Correlate findings across sources",
                "Provide actionable intelligence report"
            ])
        
        elif operation_type == OperationType.WEB_APPLICATION_AUDIT:
            criteria.extend([
                "Map application attack surface",
                "Test for OWASP Top 10 vulnerabilities",
                "Validate security controls",
                "Provide remediation recommendations"
            ])
        
        return criteria
    
    def _assess_risk_level(self, text: str, constraints: OperationConstraints) -> RiskLevel:
        """Assess risk level based on text and constraints"""
        text_lower = text.lower()
        risk_indicators = {
            'high': ['production', 'live', 'critical', 'exploit'],
            'medium': ['test', 'staging', 'development'],
            'low': ['scan', 'reconnaissance', 'passive']
        }
        
        # Check for risk indicators
        for level, indicators in risk_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                if level == 'high':
                    return RiskLevel.HIGH
                elif level == 'medium':
                    return RiskLevel.MEDIUM
                elif level == 'low':
                    return RiskLevel.LOW
        
        # Consider constraints
        if constraints.stealth_requirements == StealthLevel.MAXIMUM:
            return RiskLevel.HIGH
        
        if "no damage" in constraints.prohibited_actions:
            return RiskLevel.LOW
        
        return RiskLevel.MEDIUM


class StrategyPlanner:
    """Plans multi-stage operations with adaptive strategy modification"""
    
    def __init__(self, model_manager: AIModelManager):
        self.model_manager = model_manager
        self.phase_templates = self._load_phase_templates()
        self.strategy_cache = {}
    
    def _load_phase_templates(self) -> Dict[OperationPhase, Dict[str, Any]]:
        """Load templates for different operation phases"""
        return {
            OperationPhase.RECONNAISSANCE: {
                "name": "Reconnaissance",
                "description": "Gather initial information about targets",
                "typical_tools": ["nmap", "dig", "whois", "shodan"],
                "typical_duration": "30-60 minutes",
                "stealth_considerations": ["Rate limiting", "Source IP rotation"]
            },
            OperationPhase.SCANNING: {
                "name": "Vulnerability Scanning",
                "description": "Identify vulnerabilities and services",
                "typical_tools": ["nmap", "masscan", "nikto", "dirb"],
                "typical_duration": "1-2 hours",
                "stealth_considerations": ["Scan timing", "Payload obfuscation"]
            },
            OperationPhase.EXPLOITATION: {
                "name": "Exploitation",
                "description": "Exploit identified vulnerabilities",
                "typical_tools": ["metasploit", "sqlmap", "custom_exploits"],
                "typical_duration": "2-4 hours",
                "stealth_considerations": ["Payload encoding", "Traffic mimicry"]
            },
            OperationPhase.POST_EXPLOITATION: {
                "name": "Post-Exploitation",
                "description": "Escalate privileges and establish persistence",
                "typical_tools": ["mimikatz", "bloodhound", "crackmapexec"],
                "typical_duration": "1-3 hours",
                "stealth_considerations": ["Living off the land", "Memory-only execution"]
            },
            OperationPhase.REPORTING: {
                "name": "Reporting",
                "description": "Generate comprehensive security report",
                "typical_tools": ["report_generator", "evidence_compiler"],
                "typical_duration": "1-2 hours",
                "stealth_considerations": ["Data exfiltration methods"]
            }
        }
    
    async def create_operation_plan(self, objective: OperationObjective) -> OperationPlan:
        """Create a comprehensive operation plan"""
        try:
            logger.info(f"Creating operation plan for {objective.operation_type.value}")
            
            # Generate plan ID
            plan_id = self._generate_plan_id(objective)
            
            # Determine phases based on operation type
            phases = self._determine_phases(objective)
            
            # Create detailed phase plans
            phase_plans = []
            for phase in phases:
                phase_plan = await self._create_phase_plan(phase, objective)
                phase_plans.append(phase_plan)
            
            # Generate overall strategy using LLM
            overall_strategy = await self._generate_overall_strategy(objective, phase_plans)
            
            # Estimate total duration
            total_duration = self._estimate_total_duration(phase_plans)
            
            # Perform risk assessment
            risk_assessment = self._perform_risk_assessment(objective, phase_plans)
            
            # Generate contingency plans
            contingency_plans = await self._generate_contingency_plans(objective, phase_plans)
            
            # Determine resource requirements
            resource_requirements = self._calculate_resource_requirements(phase_plans)
            
            # Define success metrics
            success_metrics = self._define_success_metrics(objective)
            
            plan = OperationPlan(
                plan_id=plan_id,
                objective=objective,
                phases=phase_plans,
                overall_strategy=overall_strategy,
                estimated_total_duration=total_duration,
                risk_assessment=risk_assessment,
                contingency_plans=contingency_plans,
                resource_requirements=resource_requirements,
                success_metrics=success_metrics,
                created_at=datetime.now(),
                last_modified=datetime.now()
            )
            
            logger.info(f"Created operation plan {plan_id} with {len(phase_plans)} phases")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create operation plan: {e}")
            raise
    
    def _determine_phases(self, objective: OperationObjective) -> List[OperationPhase]:
        """Determine which phases are needed for the operation"""
        phases = []
        
        if objective.operation_type == OperationType.PENETRATION_TEST:
            phases = [
                OperationPhase.RECONNAISSANCE,
                OperationPhase.SCANNING,
                OperationPhase.EXPLOITATION,
                OperationPhase.POST_EXPLOITATION,
                OperationPhase.REPORTING
            ]
        elif objective.operation_type == OperationType.OSINT_INVESTIGATION:
            phases = [
                OperationPhase.OSINT,
                OperationPhase.REPORTING
            ]
        elif objective.operation_type == OperationType.WEB_APPLICATION_AUDIT:
            phases = [
                OperationPhase.RECONNAISSANCE,
                OperationPhase.WEB_AUDIT,
                OperationPhase.REPORTING
            ]
        elif objective.operation_type == OperationType.NETWORK_ASSESSMENT:
            phases = [
                OperationPhase.RECONNAISSANCE,
                OperationPhase.SCANNING,
                OperationPhase.NETWORK_COMPROMISE,
                OperationPhase.REPORTING
            ]
        else:
            # Default phases
            phases = [
                OperationPhase.RECONNAISSANCE,
                OperationPhase.SCANNING,
                OperationPhase.REPORTING
            ]
        
        return phases
    
    async def _create_phase_plan(self, phase: OperationPhase, objective: OperationObjective) -> OperationPhaseplan:
        """Create detailed plan for a specific phase"""
        template = self.phase_templates.get(phase, {})
        
        # Generate phase-specific steps using LLM
        steps = await self._generate_phase_steps(phase, objective, template)
        
        # Estimate phase duration
        duration = self._estimate_phase_duration(steps, template)
        
        # Define success criteria
        success_criteria = self._define_phase_success_criteria(phase, objective)
        
        # Generate failure handling strategies
        failure_handling = self._generate_failure_handling(phase, steps)
        
        phase_plan = OperationPhaseplan(
            phase=phase,
            name=template.get("name", phase.value.title()),
            description=template.get("description", f"Execute {phase.value} phase"),
            steps=steps,
            estimated_duration=duration,
            success_criteria=success_criteria,
            failure_handling=failure_handling
        )
        
        return phase_plan
    
    async def _generate_phase_steps(self, phase: OperationPhase, objective: OperationObjective, 
                                   template: Dict[str, Any]) -> List[OperationPhaseStep]:
        """Generate detailed steps for a phase using LLM"""
        try:
            # Prepare LLM prompt
            prompt = self._create_phase_planning_prompt(phase, objective, template)
            
            # Request LLM planning
            request = InferenceRequest(
                model_type=ModelType.LLM_PLANNER,
                input_data=prompt,
                parameters={"temperature": 0.3, "max_tokens": 1000}
            )
            
            result = await self.model_manager.infer(request)
            if not result:
                logger.warning(f"LLM planning failed for phase {phase}, using template")
                return self._generate_template_steps(phase, objective, template)
            
            # Parse LLM response into steps
            steps = self._parse_llm_steps_response(result.output, phase, objective)
            return steps
            
        except Exception as e:
            logger.error(f"Failed to generate LLM steps for phase {phase}: {e}")
            return self._generate_template_steps(phase, objective, template)
    
    def _create_phase_planning_prompt(self, phase: OperationPhase, objective: OperationObjective, 
                                    template: Dict[str, Any]) -> str:
        """Create LLM prompt for phase planning"""
        prompt = f"""
Plan the {phase.value} phase for a {objective.operation_type.value} operation.

Objective: {objective.raw_input}
Targets: {', '.join(objective.primary_targets)}
Risk Level: {objective.risk_level.value}
Stealth Requirements: {objective.constraints.stealth_requirements.value}

Phase Template:
- Name: {template.get('name', phase.value)}
- Description: {template.get('description', '')}
- Typical Tools: {', '.join(template.get('typical_tools', []))}
- Typical Duration: {template.get('typical_duration', 'Unknown')}

Constraints:
- Prohibited Actions: {', '.join(objective.constraints.prohibited_actions)}
- Scope Boundaries: {', '.join(objective.constraints.scope_boundaries)}
- Compliance Requirements: {', '.join(objective.constraints.compliance_requirements)}

Generate a detailed step-by-step plan for this phase. For each step, include:
1. Step name and description
2. Required tools
3. Estimated duration
4. Success criteria
5. Failure alternatives
6. Stealth considerations (if applicable)

Format as JSON with the following structure:
{{
  "steps": [
    {{
      "name": "Step name",
      "description": "Detailed description",
      "tools_required": ["tool1", "tool2"],
      "estimated_duration": "X minutes",
      "success_criteria": ["criteria1", "criteria2"],
      "failure_alternatives": ["alternative1", "alternative2"],
      "stealth_considerations": ["consideration1", "consideration2"],
      "requires_approval": false
    }}
  ]
}}
"""
        return prompt
    
    def _parse_llm_steps_response(self, response: Any, phase: OperationPhase, 
                                 objective: OperationObjective) -> List[OperationPhaseStep]:
        """Parse LLM response into operation steps"""
        steps = []
        
        try:
            if isinstance(response, dict) and 'steps' in response:
                step_data = response['steps']
            elif isinstance(response, str):
                # Try to parse as JSON
                parsed = json.loads(response)
                step_data = parsed.get('steps', [])
            else:
                raise ValueError("Invalid response format")
            
            for i, step_info in enumerate(step_data):
                step_id = f"{phase.value}_{i+1:02d}"
                
                step = OperationPhaseStep(
                    step_id=step_id,
                    name=step_info.get('name', f'Step {i+1}'),
                    description=step_info.get('description', ''),
                    tools_required=step_info.get('tools_required', []),
                    estimated_duration=step_info.get('estimated_duration', '30 minutes'),
                    success_criteria=step_info.get('success_criteria', []),
                    failure_alternatives=step_info.get('failure_alternatives', []),
                    stealth_considerations=step_info.get('stealth_considerations', []),
                    requires_approval=step_info.get('requires_approval', False),
                    risk_level=RiskLevel(step_info.get('risk_level', 'medium'))
                )
                
                steps.append(step)
                
        except Exception as e:
            logger.error(f"Failed to parse LLM steps response: {e}")
            # Fallback to template-based steps
            return self._generate_template_steps(phase, objective, {})
        
        return steps
    
    def _generate_template_steps(self, phase: OperationPhase, objective: OperationObjective, 
                               template: Dict[str, Any]) -> List[OperationPhaseStep]:
        """Generate basic steps using templates as fallback"""
        steps = []
        
        if phase == OperationPhase.RECONNAISSANCE:
            steps = [
                OperationPhaseStep(
                    step_id="recon_01",
                    name="Target Discovery",
                    description="Discover and enumerate target systems",
                    tools_required=["nmap", "dig"],
                    estimated_duration="30 minutes",
                    success_criteria=["Active hosts identified", "Services enumerated"],
                    failure_alternatives=["Try alternative discovery methods"]
                ),
                OperationPhaseStep(
                    step_id="recon_02",
                    name="Service Enumeration",
                    description="Enumerate services and versions",
                    tools_required=["nmap", "banner_grabbing"],
                    estimated_duration="45 minutes",
                    success_criteria=["Service versions identified", "Potential vulnerabilities noted"],
                    failure_alternatives=["Use alternative enumeration techniques"]
                )
            ]
        elif phase == OperationPhase.SCANNING:
            steps = [
                OperationPhaseStep(
                    step_id="scan_01",
                    name="Vulnerability Scanning",
                    description="Scan for known vulnerabilities",
                    tools_required=["nmap", "nikto"],
                    estimated_duration="60 minutes",
                    success_criteria=["Vulnerabilities identified", "Risk assessment completed"],
                    failure_alternatives=["Manual vulnerability research"]
                )
            ]
        
        return steps
    
    async def _generate_overall_strategy(self, objective: OperationObjective, 
                                       phases: List[OperationPhaseplan]) -> str:
        """Generate overall operation strategy using LLM"""
        try:
            prompt = f"""
Generate an overall strategy for a {objective.operation_type.value} operation.

Objective: {objective.raw_input}
Targets: {', '.join(objective.primary_targets)}
Risk Level: {objective.risk_level.value}

Phases planned:
{chr(10).join([f"- {phase.name}: {phase.description}" for phase in phases])}

Constraints:
- Stealth Requirements: {objective.constraints.stealth_requirements.value}
- Prohibited Actions: {', '.join(objective.constraints.prohibited_actions)}
- Time Limit: {objective.constraints.max_duration_hours} hours

Provide a concise overall strategy that explains:
1. The approach and methodology
2. Key success factors
3. Risk mitigation strategies
4. Adaptation points for changing conditions
"""
            
            request = InferenceRequest(
                model_type=ModelType.LLM_PLANNER,
                input_data=prompt,
                parameters={"temperature": 0.5, "max_tokens": 500}
            )
            
            result = await self.model_manager.infer(request)
            if result and isinstance(result.output, str):
                return result.output
            
        except Exception as e:
            logger.error(f"Failed to generate overall strategy: {e}")
        
        # Fallback strategy
        return f"Execute {objective.operation_type.value} against specified targets using a phased approach with {objective.constraints.stealth_requirements.value} stealth requirements."
    
    def _estimate_total_duration(self, phases: List[OperationPhaseplan]) -> str:
        """Estimate total operation duration"""
        total_minutes = 0
        
        for phase in phases:
            # Parse duration from phase
            duration_str = phase.estimated_duration
            minutes = self._parse_duration_to_minutes(duration_str)
            total_minutes += minutes
        
        hours = total_minutes // 60
        minutes = total_minutes % 60
        
        if hours > 0:
            return f"{hours} hours {minutes} minutes"
        else:
            return f"{minutes} minutes"
    
    def _parse_duration_to_minutes(self, duration_str: str) -> int:
        """Parse duration string to minutes"""
        try:
            # Extract numbers and units
            import re
            pattern = r'(\d+)\s*(hour|minute|hr|min)'
            matches = re.findall(pattern, duration_str.lower())
            
            total_minutes = 0
            for value, unit in matches:
                if unit in ['hour', 'hr']:
                    total_minutes += int(value) * 60
                else:
                    total_minutes += int(value)
            
            return total_minutes if total_minutes > 0 else 60  # Default 1 hour
            
        except:
            return 60  # Default 1 hour
    
    def _perform_risk_assessment(self, objective: OperationObjective, 
                               phases: List[OperationPhaseplan]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        risk_factors = []
        mitigation_strategies = []
        
        # Assess target-based risks
        if any('production' in target.lower() for target in objective.primary_targets):
            risk_factors.append("Production environment target")
            mitigation_strategies.append("Use minimal impact techniques")
        
        # Assess operation-based risks
        if objective.operation_type == OperationType.PENETRATION_TEST:
            risk_factors.append("Active exploitation planned")
            mitigation_strategies.append("Implement rollback procedures")
        
        # Assess stealth risks
        if objective.constraints.stealth_requirements == StealthLevel.MAXIMUM:
            risk_factors.append("Maximum stealth required")
            mitigation_strategies.append("Use advanced evasion techniques")
        
        return {
            "overall_risk_level": objective.risk_level.value,
            "risk_factors": risk_factors,
            "mitigation_strategies": mitigation_strategies,
            "approval_required": len([p for phase in phases for p in phase.steps if p.requires_approval]) > 0
        }
    
    async def _generate_contingency_plans(self, objective: OperationObjective, 
                                        phases: List[OperationPhaseplan]) -> List[Dict[str, Any]]:
        """Generate contingency plans for common failure scenarios"""
        contingencies = []
        
        # Detection contingency
        contingencies.append({
            "scenario": "Operation detected by security systems",
            "triggers": ["IDS alerts", "Unusual network activity", "Account lockouts"],
            "response": "Switch to maximum stealth mode, pause operations, assess detection scope",
            "fallback_strategy": "Resume with alternative techniques or abort if necessary"
        })
        
        # Tool failure contingency
        contingencies.append({
            "scenario": "Primary tools fail or are blocked",
            "triggers": ["Tool crashes", "Network blocks", "AV detection"],
            "response": "Switch to alternative tools, modify approach",
            "fallback_strategy": "Manual techniques or custom tool development"
        })
        
        # Time constraint contingency
        if objective.constraints.max_duration_hours:
            contingencies.append({
                "scenario": "Operation exceeding time limits",
                "triggers": [f"Operation running longer than {objective.constraints.max_duration_hours} hours"],
                "response": "Prioritize critical objectives, skip non-essential phases",
                "fallback_strategy": "Generate partial report with completed findings"
            })
        
        return contingencies
    
    def _calculate_resource_requirements(self, phases: List[OperationPhaseplan]) -> Dict[str, Any]:
        """Calculate resource requirements for the operation"""
        tools_needed = set()
        estimated_cpu_hours = 0
        estimated_memory_gb = 2  # Base memory requirement
        
        for phase in phases:
            for step in phase.steps:
                tools_needed.update(step.tools_required)
                # Estimate resource usage based on tools
                if 'nmap' in step.tools_required:
                    estimated_cpu_hours += 0.5
                if 'metasploit' in step.tools_required:
                    estimated_memory_gb += 1
                    estimated_cpu_hours += 1
        
        return {
            "tools_required": list(tools_needed),
            "estimated_cpu_hours": estimated_cpu_hours,
            "estimated_memory_gb": estimated_memory_gb,
            "network_bandwidth": "Moderate",
            "storage_gb": 5  # For logs and evidence
        }
    
    def _define_success_metrics(self, objective: OperationObjective) -> List[str]:
        """Define measurable success metrics"""
        metrics = []
        
        # Add objective-specific metrics
        metrics.extend(objective.success_criteria)
        
        # Add operation-type specific metrics
        if objective.operation_type == OperationType.PENETRATION_TEST:
            metrics.extend([
                "Vulnerability discovery rate > 80%",
                "Successful exploitation of critical vulnerabilities",
                "Complete documentation of attack paths"
            ])
        
        # Add general metrics
        metrics.extend([
            "Operation completed within time constraints",
            "No unauthorized damage or disruption",
            "Comprehensive report generated"
        ])
        
        return metrics
    
    def _generate_plan_id(self, objective: OperationObjective) -> str:
        """Generate unique plan ID"""
        content = f"{objective.raw_input}{objective.operation_type.value}{datetime.now().isoformat()}"
        hash_obj = hashlib.md5(content.encode())
        return f"plan_{hash_obj.hexdigest()[:8]}"
    
    def _define_phase_success_criteria(self, phase: OperationPhase, objective: OperationObjective) -> List[str]:
        """Define success criteria for a specific phase"""
        criteria = []
        
        if phase == OperationPhase.RECONNAISSANCE:
            criteria = [
                "Target systems identified and cataloged",
                "Network topology mapped",
                "Initial attack surface documented"
            ]
        elif phase == OperationPhase.SCANNING:
            criteria = [
                "Vulnerabilities identified and prioritized",
                "Service enumeration completed",
                "Exploitation targets selected"
            ]
        elif phase == OperationPhase.EXPLOITATION:
            criteria = [
                "At least one successful exploitation",
                "Access level documented",
                "Evidence collected"
            ]
        elif phase == OperationPhase.POST_EXPLOITATION:
            criteria = [
                "Privilege escalation attempted",
                "Lateral movement explored",
                "Persistence mechanisms tested"
            ]
        elif phase == OperationPhase.REPORTING:
            criteria = [
                "Comprehensive report generated",
                "Executive summary completed",
                "Remediation recommendations provided"
            ]
        
        return criteria
    
    def _generate_failure_handling(self, phase: OperationPhase, steps: List[OperationPhaseStep]) -> List[str]:
        """Generate failure handling strategies for a phase"""
        strategies = []
        
        # General strategies
        strategies.append("Log failure details for analysis")
        strategies.append("Attempt alternative approaches")
        strategies.append("Escalate to manual intervention if needed")
        
        # Phase-specific strategies
        if phase == OperationPhase.EXPLOITATION:
            strategies.append("Fall back to information gathering if exploitation fails")
            strategies.append("Document failed attempts for report")
        
        return strategies
    
    def _estimate_phase_duration(self, steps: List[OperationPhaseStep], template: Dict[str, Any]) -> str:
        """Estimate duration for a phase based on its steps"""
        total_minutes = 0
        
        for step in steps:
            minutes = self._parse_duration_to_minutes(step.estimated_duration)
            total_minutes += minutes
        
        # Add buffer time (20%)
        total_minutes = int(total_minutes * 1.2)
        
        hours = total_minutes // 60
        minutes = total_minutes % 60
        
        if hours > 0:
            return f"{hours} hours {minutes} minutes"
        else:
            return f"{minutes} minutes"


class AdaptiveStrategyModifier:
    """Handles adaptive strategy modification based on changing conditions"""
    
    def __init__(self, model_manager: AIModelManager):
        self.model_manager = model_manager
        self.adaptation_history = []
    
    async def adapt_strategy(self, current_plan: OperationPlan, 
                           context: AdaptationContext) -> OperationPlan:
        """Adapt operation strategy based on current context"""
        try:
            logger.info(f"Adapting strategy for plan {current_plan.plan_id}")
            
            # Analyze need for adaptation
            adaptation_needed = self._assess_adaptation_need(current_plan, context)
            
            if not adaptation_needed:
                logger.info("No strategy adaptation needed")
                return current_plan
            
            # Generate adaptation recommendations using LLM
            recommendations = await self._generate_adaptation_recommendations(current_plan, context)
            
            # Apply adaptations
            adapted_plan = await self._apply_adaptations(current_plan, recommendations, context)
            
            # Record adaptation
            self._record_adaptation(current_plan, adapted_plan, context, recommendations)
            
            logger.info(f"Strategy adapted for plan {current_plan.plan_id}")
            return adapted_plan
            
        except Exception as e:
            logger.error(f"Failed to adapt strategy: {e}")
            return current_plan
    
    def _assess_adaptation_need(self, plan: OperationPlan, context: AdaptationContext) -> bool:
        """Assess whether strategy adaptation is needed"""
        # Check for failed steps
        if context.failed_steps:
            return True
        
        # Check for new constraints
        if context.new_constraints:
            return True
        
        # Check for significant environmental changes
        if context.environmental_changes:
            return True
        
        # Check for time pressure
        if context.remaining_time and context.remaining_time < context.time_elapsed * 0.5:
            return True
        
        # Check for new discovered information that changes approach
        if context.discovered_information:
            critical_info = context.discovered_information.get('critical_findings', [])
            if critical_info:
                return True
        
        return False
    
    async def _generate_adaptation_recommendations(self, plan: OperationPlan, 
                                                 context: AdaptationContext) -> Dict[str, Any]:
        """Generate adaptation recommendations using LLM"""
        try:
            prompt = self._create_adaptation_prompt(plan, context)
            
            request = InferenceRequest(
                model_type=ModelType.LLM_PLANNER,
                input_data=prompt,
                parameters={"temperature": 0.4, "max_tokens": 800}
            )
            
            result = await self.model_manager.infer(request)
            if result:
                return self._parse_adaptation_response(result.output)
            
        except Exception as e:
            logger.error(f"Failed to generate adaptation recommendations: {e}")
        
        # Fallback recommendations
        return self._generate_fallback_recommendations(context)
    
    def _create_adaptation_prompt(self, plan: OperationPlan, context: AdaptationContext) -> str:
        """Create LLM prompt for strategy adaptation"""
        prompt = f"""
Analyze the current security operation and recommend strategy adaptations.

Current Operation:
- Plan ID: {plan.plan_id}
- Operation Type: {plan.objective.operation_type.value}
- Current Phase: {context.current_phase.value}
- Overall Strategy: {plan.overall_strategy}

Current Situation:
- Completed Steps: {len(context.completed_steps)}
- Failed Steps: {context.failed_steps}
- Time Elapsed: {context.time_elapsed:.1f} hours
- Remaining Time: {context.remaining_time:.1f if context.remaining_time is not None else 'Unknown'} hours

Changes Detected:
- Environmental Changes: {context.environmental_changes}
- New Constraints: {context.new_constraints}
- Discovered Information: {list(context.discovered_information.keys()) if context.discovered_information else 'None'}

Performance Metrics:
{json.dumps(context.performance_metrics, indent=2) if context.performance_metrics else 'No metrics available'}

Provide adaptation recommendations in JSON format:
{{
  "adaptation_type": "minor|major|critical",
  "recommended_changes": [
    {{
      "area": "strategy|phases|tools|timing",
      "change": "Description of change",
      "reason": "Why this change is needed",
      "priority": "high|medium|low"
    }}
  ],
  "risk_assessment": "Assessment of risks with changes",
  "expected_impact": "Expected impact on operation success"
}}
"""
        return prompt
    
    def _parse_adaptation_response(self, response: Any) -> Dict[str, Any]:
        """Parse LLM adaptation response"""
        try:
            if isinstance(response, dict):
                return response
            elif isinstance(response, str):
                return json.loads(response)
            else:
                raise ValueError("Invalid response format")
        except Exception as e:
            logger.error(f"Failed to parse adaptation response: {e}")
            return self._generate_fallback_recommendations({})
    
    def _generate_fallback_recommendations(self, context: AdaptationContext) -> Dict[str, Any]:
        """Generate fallback recommendations when LLM fails"""
        recommendations = []
        
        if context.failed_steps:
            recommendations.append({
                "area": "strategy",
                "change": "Switch to alternative approaches for failed steps",
                "reason": "Multiple step failures detected",
                "priority": "high"
            })
        
        if context.new_constraints:
            recommendations.append({
                "area": "phases",
                "change": "Adjust phases to comply with new constraints",
                "reason": "New operational constraints detected",
                "priority": "medium"
            })
        
        return {
            "adaptation_type": "minor",
            "recommended_changes": recommendations,
            "risk_assessment": "Low risk adaptations",
            "expected_impact": "Improved operation success probability"
        }
    
    async def _apply_adaptations(self, plan: OperationPlan, recommendations: Dict[str, Any], 
                               context: AdaptationContext) -> OperationPlan:
        """Apply adaptation recommendations to the plan"""
        adapted_plan = plan
        
        try:
            changes = recommendations.get('recommended_changes', [])
            
            for change in changes:
                area = change.get('area')
                priority = change.get('priority', 'medium')
                
                if priority == 'high':
                    if area == 'strategy':
                        adapted_plan = self._adapt_strategy(adapted_plan, change, context)
                    elif area == 'phases':
                        adapted_plan = self._adapt_phases(adapted_plan, change, context)
                    elif area == 'tools':
                        adapted_plan = self._adapt_tools(adapted_plan, change, context)
                    elif area == 'timing':
                        adapted_plan = self._adapt_timing(adapted_plan, change, context)
            
            # Update plan metadata
            adapted_plan.last_modified = datetime.now()
            adapted_plan.version += 1
            
        except Exception as e:
            logger.error(f"Failed to apply adaptations: {e}")
        
        return adapted_plan
    
    def _adapt_strategy(self, plan: OperationPlan, change: Dict[str, Any], 
                       context: AdaptationContext) -> OperationPlan:
        """Adapt overall strategy"""
        # Update overall strategy description
        plan.overall_strategy += f" [ADAPTED: {change['change']}]"
        return plan
    
    def _adapt_phases(self, plan: OperationPlan, change: Dict[str, Any], 
                     context: AdaptationContext) -> OperationPlan:
        """Adapt phase execution"""
        # Modify phase plans based on adaptation needs
        for phase_plan in plan.phases:
            if phase_plan.phase == context.current_phase:
                # Add alternative steps for failed approaches
                if context.failed_steps:
                    phase_plan.failure_handling.append(f"Adapted: {change['change']}")
        
        return plan
    
    def _adapt_tools(self, plan: OperationPlan, change: Dict[str, Any], 
                    context: AdaptationContext) -> OperationPlan:
        """Adapt tool selection"""
        # Update tool requirements in relevant phases
        for phase_plan in plan.phases:
            for step in phase_plan.steps:
                if step.step_id in context.failed_steps:
                    step.failure_alternatives.append(f"Tool adaptation: {change['change']}")
        
        return plan
    
    def _adapt_timing(self, plan: OperationPlan, change: Dict[str, Any], 
                     context: AdaptationContext) -> OperationPlan:
        """Adapt timing and scheduling"""
        # Adjust phase durations based on time constraints
        if context.remaining_time:
            # Compress remaining phases if time is running short
            remaining_phases = [p for p in plan.phases if p.phase.value not in [s.split('_')[0] for s in context.completed_steps]]
            if remaining_phases:
                for phase in remaining_phases:
                    phase.estimated_duration = f"Compressed: {phase.estimated_duration}"
        
        return plan
    
    def _record_adaptation(self, original_plan: OperationPlan, adapted_plan: OperationPlan, 
                          context: AdaptationContext, recommendations: Dict[str, Any]):
        """Record adaptation for learning and analysis"""
        adaptation_record = {
            "timestamp": datetime.now().isoformat(),
            "plan_id": original_plan.plan_id,
            "context": {
                "current_phase": context.current_phase.value,
                "failed_steps": context.failed_steps,
                "environmental_changes": context.environmental_changes,
                "time_pressure": context.remaining_time < context.time_elapsed if context.remaining_time else False
            },
            "recommendations": recommendations,
            "changes_applied": adapted_plan.version - original_plan.version
        }
        
        self.adaptation_history.append(adaptation_record)
        logger.info(f"Recorded adaptation for plan {original_plan.plan_id}")


class LLMPlanningEngine:
    """Main LLM Planning Engine that coordinates all planning components"""
    
    def __init__(self, model_manager: AIModelManager, config: Optional[Dict[str, Any]] = None):
        self.model_manager = model_manager
        self.config = config or {}
        
        # Initialize components
        self.objective_parser = ObjectiveParser()
        self.strategy_planner = StrategyPlanner(model_manager)
        self.adaptive_modifier = AdaptiveStrategyModifier(model_manager)
        
        # Active plans tracking
        self.active_plans: Dict[str, OperationPlan] = {}
        self.plan_contexts: Dict[str, AdaptationContext] = {}
        
        # Statistics
        self.stats = {
            "plans_created": 0,
            "adaptations_made": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_planning_time_ms": 0.0
        }
        
        logger.info("LLM Planning Engine initialized")
    
    async def parse_and_plan(self, natural_language_input: str, 
                           constraints: Optional[Dict[str, Any]] = None) -> OperationPlan:
        """Parse natural language input and create comprehensive operation plan"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing planning request: {natural_language_input}")
            
            # Parse objective from natural language
            objective = await self.objective_parser.parse_objective(natural_language_input)
            
            # Apply additional constraints if provided
            if constraints:
                objective.constraints = self._merge_constraints(objective.constraints, constraints)
            
            # Validate constraints and safety
            self._validate_operation_safety(objective)
            
            # Create comprehensive operation plan
            plan = await self.strategy_planner.create_operation_plan(objective)
            
            # Store active plan
            self.active_plans[plan.plan_id] = plan
            self.plan_contexts[plan.plan_id] = AdaptationContext(
                current_phase=plan.phases[0].phase if plan.phases else OperationPhase.RECONNAISSANCE,
                completed_steps=[],
                failed_steps=[],
                discovered_information={},
                environmental_changes=[],
                new_constraints=[],
                performance_metrics={},
                time_elapsed=0.0,
                remaining_time=None
            )
            
            # Update statistics
            planning_time = (time.time() - start_time) * 1000
            self._update_planning_time_stats(planning_time)
            self.stats["plans_created"] += 1
            
            logger.info(f"Created operation plan {plan.plan_id} in {planning_time:.2f}ms")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to parse and plan operation: {e}")
            raise
    
    async def adapt_plan(self, plan_id: str, context_updates: Dict[str, Any]) -> Optional[OperationPlan]:
        """Adapt an existing plan based on new context"""
        try:
            if plan_id not in self.active_plans:
                logger.error(f"Plan {plan_id} not found")
                return None
            
            current_plan = self.active_plans[plan_id]
            current_context = self.plan_contexts[plan_id]
            
            # Update context with new information
            updated_context = self._update_context(current_context, context_updates)
            
            # Perform adaptive modification
            adapted_plan = await self.adaptive_modifier.adapt_strategy(current_plan, updated_context)
            
            # Update stored plan and context
            self.active_plans[plan_id] = adapted_plan
            self.plan_contexts[plan_id] = updated_context
            
            self.stats["adaptations_made"] += 1
            
            logger.info(f"Adapted plan {plan_id}")
            return adapted_plan
            
        except Exception as e:
            logger.error(f"Failed to adapt plan {plan_id}: {e}")
            return None
    
    def get_plan(self, plan_id: str) -> Optional[OperationPlan]:
        """Get an active operation plan"""
        return self.active_plans.get(plan_id)
    
    def get_active_plans(self) -> List[OperationPlan]:
        """Get all active operation plans"""
        return list(self.active_plans.values())
    
    def complete_plan(self, plan_id: str, success: bool = True):
        """Mark a plan as completed"""
        if plan_id in self.active_plans:
            if success:
                self.stats["successful_operations"] += 1
            else:
                self.stats["failed_operations"] += 1
            
            # Remove from active plans
            del self.active_plans[plan_id]
            del self.plan_contexts[plan_id]
            
            logger.info(f"Plan {plan_id} marked as {'successful' if success else 'failed'}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get planning engine statistics"""
        return self.stats.copy()
    
    def _merge_constraints(self, base_constraints: OperationConstraints, 
                          additional_constraints: Dict[str, Any]) -> OperationConstraints:
        """Merge additional constraints with base constraints"""
        # Update base constraints with additional ones
        if 'scope_boundaries' in additional_constraints:
            base_constraints.scope_boundaries.extend(additional_constraints['scope_boundaries'])
        
        if 'prohibited_actions' in additional_constraints:
            base_constraints.prohibited_actions.extend(additional_constraints['prohibited_actions'])
        
        if 'stealth_requirements' in additional_constraints:
            base_constraints.stealth_requirements = StealthLevel(additional_constraints['stealth_requirements'])
        
        if 'max_duration_hours' in additional_constraints:
            base_constraints.max_duration_hours = additional_constraints['max_duration_hours']
        
        if 'compliance_requirements' in additional_constraints:
            base_constraints.compliance_requirements.extend(additional_constraints['compliance_requirements'])
        
        return base_constraints
    
    def _validate_operation_safety(self, objective: OperationObjective):
        """Validate operation safety and constraints"""
        # Check for prohibited targets
        prohibited_patterns = ['localhost', '127.0.0.1', 'internal', 'prod']
        for target in objective.primary_targets:
            if any(pattern in target.lower() for pattern in prohibited_patterns):
                if target.lower() not in [c.lower() for c in objective.constraints.authorized_targets]:
                    raise ValueError(f"Target {target} requires explicit authorization")
        
        # Check for high-risk operations
        if objective.risk_level == RiskLevel.CRITICAL:
            if not objective.constraints.approval_required_actions:
                objective.constraints.approval_required_actions.append("all_critical_operations")
        
        # Validate compliance requirements
        if objective.constraints.compliance_requirements:
            logger.info(f"Operation must comply with: {', '.join(objective.constraints.compliance_requirements)}")
    
    def _update_context(self, current_context: AdaptationContext, 
                       updates: Dict[str, Any]) -> AdaptationContext:
        """Update adaptation context with new information"""
        if 'current_phase' in updates:
            current_context.current_phase = OperationPhase(updates['current_phase'])
        
        if 'completed_steps' in updates:
            current_context.completed_steps.extend(updates['completed_steps'])
        
        if 'failed_steps' in updates:
            current_context.failed_steps.extend(updates['failed_steps'])
        
        if 'discovered_information' in updates:
            current_context.discovered_information.update(updates['discovered_information'])
        
        if 'environmental_changes' in updates:
            current_context.environmental_changes.extend(updates['environmental_changes'])
        
        if 'new_constraints' in updates:
            current_context.new_constraints.extend(updates['new_constraints'])
        
        if 'performance_metrics' in updates:
            current_context.performance_metrics.update(updates['performance_metrics'])
        
        if 'time_elapsed' in updates:
            current_context.time_elapsed = updates['time_elapsed']
        
        if 'remaining_time' in updates:
            current_context.remaining_time = updates['remaining_time']
        
        return current_context
    
    def _update_planning_time_stats(self, planning_time_ms: float):
        """Update average planning time statistics"""
        current_avg = self.stats['average_planning_time_ms']
        total_plans = self.stats['plans_created']
        
        if total_plans == 0:
            self.stats['average_planning_time_ms'] = planning_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats['average_planning_time_ms'] = (
                alpha * planning_time_ms + (1 - alpha) * current_avg
            )


# Factory function for creating planning engine
def create_planning_engine(model_manager: AIModelManager, 
                         config: Optional[Dict[str, Any]] = None) -> LLMPlanningEngine:
    """Create and return a new LLM Planning Engine instance"""
    return LLMPlanningEngine(model_manager, config)