#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - Scenario Template System
Parameterized scenario configurations with intelligent template management
"""

import asyncio
import logging
import uuid
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from enum import Enum
from collections import defaultdict
import random

from agents.base_agent import Team, Role

logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    """Types of scenarios"""
    TRAINING = "training"               # Learning and skill development
    ASSESSMENT = "assessment"           # Performance evaluation
    LIVE_EXERCISE = "live_exercise"     # Real-time operational exercises
    SIMULATION = "simulation"           # Simulation-based scenarios
    COMPETITION = "competition"         # Competitive scenarios
    RESEARCH = "research"               # Research and experimentation

class ScenarioCategory(Enum):
    """Scenario categories based on focus area"""
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    COMMAND_CONTROL = "command_control"
    IMPACT = "impact"
    INCIDENT_RESPONSE = "incident_response"
    THREAT_HUNTING = "threat_hunting"
    DIGITAL_FORENSICS = "digital_forensics"
    FULL_CAMPAIGN = "full_campaign"

class ComplexityLevel(Enum):
    """Scenario complexity levels"""
    BEGINNER = 1        # Basic scenarios for new agents
    INTERMEDIATE = 2    # Standard complexity
    ADVANCED = 3        # Complex multi-stage scenarios
    EXPERT = 4          # Highly complex scenarios
    MASTER = 5          # Extremely complex, real-world scenarios

class NetworkTopology(Enum):
    """Network topology types"""
    SIMPLE_NETWORK = "simple_network"
    ENTERPRISE_NETWORK = "enterprise_network"
    CLOUD_INFRASTRUCTURE = "cloud_infrastructure"
    HYBRID_ENVIRONMENT = "hybrid_environment"
    IOT_NETWORK = "iot_network"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    SEGMENTED_NETWORK = "segmented_network"
    FLAT_NETWORK = "flat_network"

@dataclass
class ScenarioParameter:
    """Individual scenario parameter definition"""
    name: str
    parameter_type: str  # "string", "int", "float", "bool", "enum", "list"
    default_value: Any
    
    # Validation constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = True
    
    # Documentation
    description: str = ""
    examples: List[str] = field(default_factory=list)
    
    # Advanced constraints
    depends_on: List[str] = field(default_factory=list)  # Parameter dependencies
    conditional_logic: Optional[str] = None  # Python expression for conditional validation

@dataclass
class ScenarioObjective:
    """Scenario objective definition"""
    objective_id: str
    name: str
    description: str
    
    # Objective properties
    objective_type: str = "primary"  # primary, secondary, hidden, bonus
    difficulty: ComplexityLevel = ComplexityLevel.INTERMEDIATE
    
    # Success criteria
    success_criteria: List[str] = field(default_factory=list)
    points: int = 100
    time_limit: Optional[timedelta] = None
    
    # Dependencies and prerequisites
    prerequisites: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Hints and guidance
    hints: List[str] = field(default_factory=list)
    guidance: str = ""
    
    # Validation
    validation_rules: List[str] = field(default_factory=list)

@dataclass
class ScenarioAsset:
    """Scenario asset definition (VMs, networks, services, etc.)"""
    asset_id: str
    asset_type: str  # "vm", "network", "service", "file", "user", "database"
    name: str
    
    # Asset properties
    configuration: Dict[str, Any] = field(default_factory=dict)
    initial_state: Dict[str, Any] = field(default_factory=dict)
    
    # Security properties
    vulnerabilities: List[str] = field(default_factory=list)
    security_controls: List[str] = field(default_factory=list)
    
    # Monitoring and logging
    monitoring_enabled: bool = True
    logging_level: str = "standard"  # minimal, standard, verbose, debug
    
    # Lifecycle
    startup_scripts: List[str] = field(default_factory=list)
    cleanup_scripts: List[str] = field(default_factory=list)

@dataclass
class ScenarioTemplate:
    """Complete scenario template definition"""
    template_id: str
    name: str
    description: str
    
    # Classification
    scenario_type: ScenarioType
    category: ScenarioCategory
    complexity: ComplexityLevel
    
    # Template metadata
    version: str = "1.0"
    author: str = "Archangel System"
    tags: Set[str] = field(default_factory=set)
    
    # Timing and duration
    estimated_duration: timedelta = timedelta(hours=2)
    preparation_time: timedelta = timedelta(minutes=30)
    cleanup_time: timedelta = timedelta(minutes=15)
    
    # Participants
    min_participants: int = 1
    max_participants: int = 10
    recommended_teams: List[Team] = field(default_factory=list)
    required_roles: List[Role] = field(default_factory=list)
    
    # Network and infrastructure
    network_topology: NetworkTopology = NetworkTopology.SIMPLE_NETWORK
    required_assets: List[ScenarioAsset] = field(default_factory=list)
    
    # Objectives and scoring
    objectives: List[ScenarioObjective] = field(default_factory=list)
    total_points: int = 0
    
    # Configuration parameters
    parameters: List[ScenarioParameter] = field(default_factory=list)
    
    # Execution configuration
    execution_config: Dict[str, Any] = field(default_factory=dict)
    
    # Prerequisites and dependencies
    prerequisites: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Documentation and resources
    documentation: str = ""
    resources: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    
    # Template metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0
    
    # Validation and testing
    validation_rules: List[str] = field(default_factory=list)
    test_cases: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ScenarioInstance:
    """Instantiated scenario from a template"""
    instance_id: str
    template_id: str
    name: str
    
    # Instance configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    participants: List[str] = field(default_factory=list)
    
    # Execution state
    status: str = "created"  # created, prepared, running, paused, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Results and scoring
    current_score: int = 0
    objective_progress: Dict[str, float] = field(default_factory=dict)
    completed_objectives: List[str] = field(default_factory=list)
    
    # Runtime data
    logs: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"

class ScenarioTemplateManager:
    """
    Comprehensive scenario template management system.
    
    Features:
    - Template creation, validation, and management
    - Parameterized configuration system
    - Template versioning and inheritance
    - Validation and testing framework
    - Template analytics and optimization
    """
    
    def __init__(self):
        self.templates: Dict[str, ScenarioTemplate] = {}
        self.template_categories: Dict[ScenarioCategory, List[str]] = defaultdict(list)
        self.template_analytics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Template inheritance hierarchy
        self.template_inheritance: Dict[str, str] = {}  # child_id -> parent_id
        
        # Validation cache
        self.validation_cache: Dict[str, bool] = {}
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the template manager"""
        try:
            self.logger.info("Initializing scenario template manager")
            
            # Load built-in templates
            await self._load_builtin_templates()
            
            # Load custom templates
            await self._load_custom_templates()
            
            # Build category indexes
            await self._build_category_indexes()
            
            self.initialized = True
            self.logger.info(f"Template manager initialized with {len(self.templates)} templates")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize template manager: {e}")
            raise
    
    async def create_template(self, template: ScenarioTemplate) -> str:
        """Create a new scenario template"""
        try:
            # Validate template
            validation_result = await self.validate_template(template)
            if not validation_result["valid"]:
                raise ValueError(f"Template validation failed: {validation_result['errors']}")
            
            # Store template
            self.templates[template.template_id] = template
            
            # Update category index
            self.template_categories[template.category].append(template.template_id)
            
            # Initialize analytics
            self.template_analytics[template.template_id] = {
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                "success_rate": 0.0,
                "avg_duration": None,
                "user_ratings": []
            }
            
            self.logger.info(f"Created template: {template.name} ({template.template_id})")
            return template.template_id
            
        except Exception as e:
            self.logger.error(f"Failed to create template: {e}")
            raise
    
    async def get_template(self, template_id: str) -> Optional[ScenarioTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    async def update_template(self, template_id: str, template: ScenarioTemplate) -> bool:
        """Update an existing template"""
        try:
            if template_id not in self.templates:
                raise ValueError(f"Template {template_id} not found")
            
            # Validate updated template
            validation_result = await self.validate_template(template)
            if not validation_result["valid"]:
                raise ValueError(f"Template validation failed: {validation_result['errors']}")
            
            # Update template
            template.updated_at = datetime.now()
            self.templates[template_id] = template
            
            # Clear validation cache
            if template_id in self.validation_cache:
                del self.validation_cache[template_id]
            
            self.logger.info(f"Updated template: {template.name} ({template_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update template {template_id}: {e}")
            return False
    
    async def delete_template(self, template_id: str) -> bool:
        """Delete a template"""
        try:
            if template_id not in self.templates:
                return False
            
            template = self.templates[template_id]
            
            # Remove from category index
            if template_id in self.template_categories[template.category]:
                self.template_categories[template.category].remove(template_id)
            
            # Remove template
            del self.templates[template_id]
            
            # Clean up analytics
            if template_id in self.template_analytics:
                del self.template_analytics[template_id]
            
            # Clean up validation cache
            if template_id in self.validation_cache:
                del self.validation_cache[template_id]
            
            self.logger.info(f"Deleted template: {template.name} ({template_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete template {template_id}: {e}")
            return False
    
    async def list_templates(self, 
                           category: Optional[ScenarioCategory] = None,
                           complexity: Optional[ComplexityLevel] = None,
                           scenario_type: Optional[ScenarioType] = None,
                           tags: Optional[List[str]] = None) -> List[ScenarioTemplate]:
        """List templates with optional filtering"""
        try:
            templates = list(self.templates.values())
            
            # Apply filters
            if category:
                templates = [t for t in templates if t.category == category]
            
            if complexity:
                templates = [t for t in templates if t.complexity == complexity]
            
            if scenario_type:
                templates = [t for t in templates if t.scenario_type == scenario_type]
            
            if tags:
                templates = [t for t in templates if any(tag in t.tags for tag in tags)]
            
            # Sort by usage and success rate
            templates.sort(key=lambda t: (
                self.template_analytics.get(t.template_id, {}).get("usage_count", 0),
                self.template_analytics.get(t.template_id, {}).get("success_rate", 0.0)
            ), reverse=True)
            
            return templates
            
        except Exception as e:
            self.logger.error(f"Failed to list templates: {e}")
            return []
    
    async def validate_template(self, template: ScenarioTemplate) -> Dict[str, Any]:
        """Comprehensive template validation"""
        try:
            errors = []
            warnings = []
            
            # Basic validation
            if not template.name or not template.description:
                errors.append("Template must have name and description")
            
            if not template.objectives:
                warnings.append("Template has no objectives defined")
            
            # Parameter validation
            for param in template.parameters:
                param_errors = await self._validate_parameter(param)
                errors.extend([f"Parameter {param.name}: {err}" for err in param_errors])
            
            # Objective validation
            for objective in template.objectives:
                obj_errors = await self._validate_objective(objective)
                errors.extend([f"Objective {objective.name}: {err}" for err in obj_errors])
            
            # Asset validation
            for asset in template.required_assets:
                asset_errors = await self._validate_asset(asset)
                errors.extend([f"Asset {asset.name}: {err}" for err in asset_errors])
            
            # Dependency validation
            for dep in template.dependencies:
                if dep not in self.templates:
                    errors.append(f"Dependency template {dep} not found")
            
            # Duration validation
            if template.estimated_duration.total_seconds() < 300:  # 5 minutes minimum
                warnings.append("Scenario duration is very short (< 5 minutes)")
            
            if template.estimated_duration.total_seconds() > 86400:  # 24 hours maximum
                warnings.append("Scenario duration is very long (> 24 hours)")
            
            # Participant validation
            if template.min_participants > template.max_participants:
                errors.append("Minimum participants cannot exceed maximum participants")
            
            validation_result = {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "validated_at": datetime.now().isoformat()
            }
            
            # Cache validation result
            self.validation_cache[template.template_id] = validation_result["valid"]
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Failed to validate template: {e}")
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "validated_at": datetime.now().isoformat()
            }
    
    async def instantiate_template(self, 
                                 template_id: str,
                                 parameters: Dict[str, Any],
                                 participants: List[str]) -> ScenarioInstance:
        """Create a scenario instance from a template"""
        try:
            template = await self.get_template(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Validate parameters
            param_validation = await self._validate_instance_parameters(template, parameters)
            if not param_validation["valid"]:
                raise ValueError(f"Parameter validation failed: {param_validation['errors']}")
            
            # Create instance
            instance = ScenarioInstance(
                instance_id=str(uuid.uuid4()),
                template_id=template_id,
                name=f"{template.name} - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                parameters=parameters,
                participants=participants
            )
            
            # Update template analytics
            self.template_analytics[template_id]["usage_count"] += 1
            template.usage_count += 1
            
            self.logger.info(f"Instantiated template {template.name} as {instance.instance_id}")
            return instance
            
        except Exception as e:
            self.logger.error(f"Failed to instantiate template {template_id}: {e}")
            raise
    
    async def clone_template(self, 
                           source_template_id: str,
                           new_name: str,
                           modifications: Optional[Dict[str, Any]] = None) -> str:
        """Clone an existing template with optional modifications"""
        try:
            source_template = await self.get_template(source_template_id)
            if not source_template:
                raise ValueError(f"Source template {source_template_id} not found")
            
            # Create clone
            cloned_template = ScenarioTemplate(
                template_id=str(uuid.uuid4()),
                name=new_name,
                description=f"Cloned from {source_template.name}",
                scenario_type=source_template.scenario_type,
                category=source_template.category,
                complexity=source_template.complexity,
                network_topology=source_template.network_topology,
                estimated_duration=source_template.estimated_duration,
                min_participants=source_template.min_participants,
                max_participants=source_template.max_participants,
                recommended_teams=source_template.recommended_teams.copy(),
                required_roles=source_template.required_roles.copy(),
                required_assets=[asset for asset in source_template.required_assets],
                objectives=[obj for obj in source_template.objectives],
                parameters=[param for param in source_template.parameters],
                tags=source_template.tags.copy()
            )
            
            # Apply modifications
            if modifications:
                for key, value in modifications.items():
                    if hasattr(cloned_template, key):
                        setattr(cloned_template, key, value)
            
            # Set inheritance relationship
            self.template_inheritance[cloned_template.template_id] = source_template_id
            
            # Create the cloned template
            template_id = await self.create_template(cloned_template)
            
            self.logger.info(f"Cloned template {source_template.name} as {new_name} ({template_id})")
            return template_id
            
        except Exception as e:
            self.logger.error(f"Failed to clone template {source_template_id}: {e}")
            raise
    
    async def get_template_recommendations(self, 
                                         agent_profile: Dict[str, Any],
                                         learning_objectives: List[str] = None,
                                         max_recommendations: int = 5) -> List[Tuple[ScenarioTemplate, float]]:
        """Get template recommendations based on agent profile and objectives"""
        try:
            recommendations = []
            
            # Get agent characteristics
            agent_skill_level = agent_profile.get("skill_level", 0.5)
            agent_experience = agent_profile.get("experience_count", 0)
            agent_success_rate = agent_profile.get("success_rate", 0.5)
            agent_preferences = agent_profile.get("preferred_categories", [])
            
            for template in self.templates.values():
                score = 0.0
                
                # Skill level matching
                complexity_score = 1.0 - abs(template.complexity.value / 5.0 - agent_skill_level)
                score += complexity_score * 0.3
                
                # Experience matching
                if agent_experience < 10:  # Beginner
                    if template.complexity in [ComplexityLevel.BEGINNER, ComplexityLevel.INTERMEDIATE]:
                        score += 0.2
                elif agent_experience < 50:  # Intermediate
                    if template.complexity in [ComplexityLevel.INTERMEDIATE, ComplexityLevel.ADVANCED]:
                        score += 0.2
                else:  # Advanced
                    if template.complexity in [ComplexityLevel.ADVANCED, ComplexityLevel.EXPERT, ComplexityLevel.MASTER]:
                        score += 0.2
                
                # Success rate consideration
                template_success_rate = self.template_analytics.get(template.template_id, {}).get("success_rate", 0.5)
                if 0.7 <= template_success_rate <= 0.9:  # Sweet spot for learning
                    score += 0.15
                
                # Category preferences
                if template.category.value in agent_preferences:
                    score += 0.15
                
                # Learning objectives alignment
                if learning_objectives:
                    objective_match = sum(
                        1 for obj in template.objectives
                        if any(keyword in obj.description.lower() for keyword in learning_objectives)
                    )
                    score += (objective_match / len(template.objectives)) * 0.2
                
                # Popularity and quality
                usage_count = self.template_analytics.get(template.template_id, {}).get("usage_count", 0)
                score += min(usage_count / 100.0, 0.1)  # Max 0.1 boost for popularity
                
                if score > 0.3:  # Minimum threshold
                    recommendations.append((template, score))
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:max_recommendations]
            
        except Exception as e:
            self.logger.error(f"Failed to get template recommendations: {e}")
            return []
    
    async def _validate_parameter(self, parameter: ScenarioParameter) -> List[str]:
        """Validate a scenario parameter"""
        errors = []
        
        if not parameter.name:
            errors.append("Parameter name is required")
        
        if parameter.parameter_type not in ["string", "int", "float", "bool", "enum", "list"]:
            errors.append(f"Invalid parameter type: {parameter.parameter_type}")
        
        if parameter.parameter_type == "enum" and not parameter.allowed_values:
            errors.append("Enum parameter must have allowed_values")
        
        if parameter.min_value is not None and parameter.max_value is not None:
            if parameter.min_value > parameter.max_value:
                errors.append("min_value cannot be greater than max_value")
        
        return errors
    
    async def _validate_objective(self, objective: ScenarioObjective) -> List[str]:
        """Validate a scenario objective"""
        errors = []
        
        if not objective.name or not objective.description:
            errors.append("Objective must have name and description")
        
        if not objective.success_criteria:
            errors.append("Objective must have success criteria")
        
        if objective.points < 0:
            errors.append("Objective points cannot be negative")
        
        return errors
    
    async def _validate_asset(self, asset: ScenarioAsset) -> List[str]:
        """Validate a scenario asset"""
        errors = []
        
        if not asset.name or not asset.asset_type:
            errors.append("Asset must have name and type")
        
        valid_asset_types = ["vm", "network", "service", "file", "user", "database", "container"]
        if asset.asset_type not in valid_asset_types:
            errors.append(f"Invalid asset type: {asset.asset_type}")
        
        return errors
    
    async def _validate_instance_parameters(self, 
                                          template: ScenarioTemplate,
                                          parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for template instantiation"""
        errors = []
        
        # Check required parameters
        for param in template.parameters:
            if param.required and param.name not in parameters:
                errors.append(f"Required parameter missing: {param.name}")
                continue
            
            if param.name not in parameters:
                continue  # Optional parameter not provided
            
            value = parameters[param.name]
            
            # Type validation
            if param.parameter_type == "int" and not isinstance(value, int):
                errors.append(f"Parameter {param.name} must be an integer")
            elif param.parameter_type == "float" and not isinstance(value, (int, float)):
                errors.append(f"Parameter {param.name} must be a number")
            elif param.parameter_type == "bool" and not isinstance(value, bool):
                errors.append(f"Parameter {param.name} must be a boolean")
            elif param.parameter_type == "string" and not isinstance(value, str):
                errors.append(f"Parameter {param.name} must be a string")
            elif param.parameter_type == "list" and not isinstance(value, list):
                errors.append(f"Parameter {param.name} must be a list")
            elif param.parameter_type == "enum" and value not in param.allowed_values:
                errors.append(f"Parameter {param.name} must be one of: {param.allowed_values}")
            
            # Range validation
            if param.min_value is not None and isinstance(value, (int, float)):
                if value < param.min_value:
                    errors.append(f"Parameter {param.name} must be >= {param.min_value}")
            
            if param.max_value is not None and isinstance(value, (int, float)):
                if value > param.max_value:
                    errors.append(f"Parameter {param.name} must be <= {param.max_value}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _load_builtin_templates(self) -> None:
        """Load built-in scenario templates"""
        try:
            # Create some built-in templates
            builtin_templates = [
                await self._create_reconnaissance_template(),
                await self._create_phishing_template(),
                await self._create_incident_response_template(),
                await self._create_red_team_exercise_template(),
                await self._create_threat_hunting_template()
            ]
            
            for template in builtin_templates:
                self.templates[template.template_id] = template
                self.template_categories[template.category].append(template.template_id)
                
                # Initialize analytics
                self.template_analytics[template.template_id] = {
                    "created_at": datetime.now().isoformat(),
                    "usage_count": 0,
                    "success_rate": 0.5,  # Default moderate success rate
                    "avg_duration": None,
                    "user_ratings": []
                }
            
            self.logger.info(f"Loaded {len(builtin_templates)} built-in templates")
            
        except Exception as e:
            self.logger.error(f"Failed to load built-in templates: {e}")
    
    async def _load_custom_templates(self) -> None:
        """Load custom templates from storage"""
        try:
            # In production, this would load from persistent storage
            self.logger.debug("Loading custom templates (placeholder)")
            
        except Exception as e:
            self.logger.error(f"Failed to load custom templates: {e}")
    
    async def _build_category_indexes(self) -> None:
        """Build category indexes for efficient filtering"""
        try:
            self.template_categories.clear()
            
            for template_id, template in self.templates.items():
                self.template_categories[template.category].append(template_id)
            
        except Exception as e:
            self.logger.error(f"Failed to build category indexes: {e}")
    
    async def _create_reconnaissance_template(self) -> ScenarioTemplate:
        """Create a reconnaissance scenario template"""
        return ScenarioTemplate(
            template_id="builtin_recon_basic",
            name="Basic Network Reconnaissance",
            description="Learn fundamental reconnaissance techniques including network scanning, service enumeration, and information gathering",
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.RECONNAISSANCE,
            complexity=ComplexityLevel.BEGINNER,
            estimated_duration=timedelta(hours=1),
            min_participants=1,
            max_participants=4,
            recommended_teams=[Team.RED_TEAM],
            network_topology=NetworkTopology.SIMPLE_NETWORK,
            objectives=[
                ScenarioObjective(
                    objective_id="recon_obj1",
                    name="Network Discovery",
                    description="Discover live hosts on the target network",
                    success_criteria=["Identify at least 3 live hosts", "Document discovered services"],
                    points=50
                ),
                ScenarioObjective(
                    objective_id="recon_obj2",
                    name="Service Enumeration",
                    description="Enumerate services running on discovered hosts",
                    success_criteria=["Identify running services", "Determine service versions"],
                    points=50
                )
            ],
            parameters=[
                ScenarioParameter(
                    name="target_network",
                    parameter_type="string",
                    default_value="192.168.1.0/24",
                    description="Target network range for reconnaissance"
                ),
                ScenarioParameter(
                    name="scan_intensity",
                    parameter_type="enum",
                    default_value="normal",
                    allowed_values=["stealth", "normal", "aggressive"],
                    description="Scanning intensity level"
                )
            ],
            tags={"reconnaissance", "network", "scanning", "beginner"}
        )
    
    async def _create_phishing_template(self) -> ScenarioTemplate:
        """Create a phishing scenario template"""
        return ScenarioTemplate(
            template_id="builtin_phishing_basic",
            name="Email Phishing Campaign",
            description="Design and execute a targeted phishing campaign with email social engineering",
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.INITIAL_ACCESS,
            complexity=ComplexityLevel.INTERMEDIATE,
            estimated_duration=timedelta(hours=2),
            min_participants=2,
            max_participants=6,
            recommended_teams=[Team.RED_TEAM, Team.BLUE_TEAM],
            objectives=[
                ScenarioObjective(
                    objective_id="phish_obj1",
                    name="Campaign Design",
                    description="Create convincing phishing emails",
                    success_criteria=["Design realistic email template", "Set up landing page"],
                    points=40
                ),
                ScenarioObjective(
                    objective_id="phish_obj2",
                    name="Campaign Execution",
                    description="Execute phishing campaign and track results",
                    success_criteria=["Send phishing emails", "Track click rates", "Analyze results"],
                    points=60
                )
            ],
            parameters=[
                ScenarioParameter(
                    name="target_count",
                    parameter_type="int",
                    default_value=10,
                    min_value=5,
                    max_value=50,
                    description="Number of target users"
                ),
                ScenarioParameter(
                    name="campaign_theme",
                    parameter_type="enum",
                    default_value="corporate",
                    allowed_values=["corporate", "finance", "it_support", "shipping"],
                    description="Phishing campaign theme"
                )
            ],
            tags={"phishing", "social_engineering", "email", "initial_access"}
        )
    
    async def _create_incident_response_template(self) -> ScenarioTemplate:
        """Create an incident response scenario template"""
        return ScenarioTemplate(
            template_id="builtin_ir_basic",
            name="Basic Incident Response",
            description="Practice incident response procedures for a simulated security breach",
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.INCIDENT_RESPONSE,
            complexity=ComplexityLevel.INTERMEDIATE,
            estimated_duration=timedelta(hours=3),
            min_participants=3,
            max_participants=8,
            recommended_teams=[Team.BLUE_TEAM],
            required_roles=[Role.INCIDENT_RESPONDER, Role.ANALYST],
            objectives=[
                ScenarioObjective(
                    objective_id="ir_obj1",
                    name="Incident Detection",
                    description="Detect and classify the security incident",
                    success_criteria=["Identify incident type", "Assess severity", "Document findings"],
                    points=30
                ),
                ScenarioObjective(
                    objective_id="ir_obj2",
                    name="Containment",
                    description="Contain the incident to prevent further damage",
                    success_criteria=["Isolate affected systems", "Preserve evidence"],
                    points=40
                ),
                ScenarioObjective(
                    objective_id="ir_obj3",
                    name="Recovery",
                    description="Recover systems and restore normal operations",
                    success_criteria=["Clean infected systems", "Restore services", "Validate security"],
                    points=30
                )
            ],
            parameters=[
                ScenarioParameter(
                    name="incident_type",
                    parameter_type="enum",
                    default_value="malware",
                    allowed_values=["malware", "data_breach", "ddos", "insider_threat"],
                    description="Type of security incident"
                ),
                ScenarioParameter(
                    name="incident_severity",
                    parameter_type="enum",
                    default_value="medium",
                    allowed_values=["low", "medium", "high", "critical"],
                    description="Incident severity level"
                )
            ],
            tags={"incident_response", "blue_team", "forensics", "containment"}
        )
    
    async def _create_red_team_exercise_template(self) -> ScenarioTemplate:
        """Create a red team exercise template"""
        return ScenarioTemplate(
            template_id="builtin_redteam_advanced",
            name="Advanced Red Team Exercise",
            description="Full-scale red team exercise with multiple attack phases",
            scenario_type=ScenarioType.ASSESSMENT,
            category=ScenarioCategory.FULL_CAMPAIGN,
            complexity=ComplexityLevel.EXPERT,
            estimated_duration=timedelta(hours=8),
            min_participants=4,
            max_participants=12,
            recommended_teams=[Team.RED_TEAM, Team.BLUE_TEAM],
            network_topology=NetworkTopology.ENTERPRISE_NETWORK,
            objectives=[
                ScenarioObjective(
                    objective_id="rt_obj1",
                    name="Initial Access",
                    description="Gain initial foothold in target environment",
                    success_criteria=["Compromise perimeter", "Establish persistence"],
                    points=25
                ),
                ScenarioObjective(
                    objective_id="rt_obj2",
                    name="Privilege Escalation",
                    description="Escalate privileges on compromised systems",
                    success_criteria=["Obtain admin privileges", "Maintain stealth"],
                    points=25
                ),
                ScenarioObjective(
                    objective_id="rt_obj3",
                    name="Lateral Movement",
                    description="Move laterally through the network",
                    success_criteria=["Access multiple systems", "Avoid detection"],
                    points=25
                ),
                ScenarioObjective(
                    objective_id="rt_obj4",
                    name="Objective Achievement",
                    description="Achieve primary mission objectives",
                    success_criteria=["Access target data", "Demonstrate impact"],
                    points=25
                )
            ],
            parameters=[
                ScenarioParameter(
                    name="detection_difficulty",
                    parameter_type="enum",
                    default_value="normal",
                    allowed_values=["easy", "normal", "hard", "expert"],
                    description="Blue team detection difficulty"
                ),
                ScenarioParameter(
                    name="target_objectives",
                    parameter_type="list",
                    default_value=["crown_jewels", "credentials"],
                    description="Primary target objectives"
                )
            ],
            tags={"red_team", "advanced", "full_campaign", "assessment"}
        )
    
    async def _create_threat_hunting_template(self) -> ScenarioTemplate:
        """Create a threat hunting scenario template"""
        return ScenarioTemplate(
            template_id="builtin_hunting_intermediate",
            name="Network Threat Hunting",
            description="Proactive threat hunting in enterprise network environment",
            scenario_type=ScenarioType.TRAINING,
            category=ScenarioCategory.THREAT_HUNTING,
            complexity=ComplexityLevel.ADVANCED,
            estimated_duration=timedelta(hours=4),
            min_participants=2,
            max_participants=6,
            recommended_teams=[Team.BLUE_TEAM],
            required_roles=[Role.THREAT_HUNTER, Role.ANALYST],
            network_topology=NetworkTopology.ENTERPRISE_NETWORK,
            objectives=[
                ScenarioObjective(
                    objective_id="hunt_obj1",
                    name="Hypothesis Development",
                    description="Develop threat hunting hypotheses",
                    success_criteria=["Create testable hypotheses", "Define hunt scope"],
                    points=30
                ),
                ScenarioObjective(
                    objective_id="hunt_obj2",
                    name="Data Collection",
                    description="Collect and analyze relevant data",
                    success_criteria=["Gather network logs", "Analyze behavioral patterns"],
                    points=40
                ),
                ScenarioObjective(
                    objective_id="hunt_obj3",
                    name="Threat Identification",
                    description="Identify potential threats and IOCs",
                    success_criteria=["Find suspicious activity", "Document IOCs"],
                    points=30
                )
            ],
            parameters=[
                ScenarioParameter(
                    name="hunt_scope",
                    parameter_type="enum",
                    default_value="network",
                    allowed_values=["network", "endpoint", "hybrid"],
                    description="Scope of threat hunting activity"
                ),
                ScenarioParameter(
                    name="data_timeframe",
                    parameter_type="int",
                    default_value=7,
                    min_value=1,
                    max_value=30,
                    description="Days of historical data to analyze"
                )
            ],
            tags={"threat_hunting", "blue_team", "analysis", "proactive"}
        )
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """Get comprehensive template statistics"""
        try:
            stats = {
                "total_templates": len(self.templates),
                "templates_by_category": {cat.value: len(templates) for cat, templates in self.template_categories.items()},
                "templates_by_complexity": defaultdict(int),
                "templates_by_type": defaultdict(int),
                "avg_duration": 0.0,
                "most_popular": None,
                "newest": None
            }
            
            total_duration = 0
            max_usage = 0
            newest_date = None
            
            for template in self.templates.values():
                stats["templates_by_complexity"][template.complexity.value] += 1
                stats["templates_by_type"][template.scenario_type.value] += 1
                total_duration += template.estimated_duration.total_seconds()
                
                usage_count = self.template_analytics.get(template.template_id, {}).get("usage_count", 0)
                if usage_count > max_usage:
                    max_usage = usage_count
                    stats["most_popular"] = template.name
                
                if newest_date is None or template.created_at > newest_date:
                    newest_date = template.created_at
                    stats["newest"] = template.name
            
            if self.templates:
                stats["avg_duration"] = total_duration / len(self.templates) / 3600  # Hours
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get template statistics: {e}")
            return {}

# Factory function
def create_template_manager() -> ScenarioTemplateManager:
    """Create a scenario template manager instance"""
    return ScenarioTemplateManager()