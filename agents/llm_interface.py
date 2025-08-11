#!/usr/bin/env python3
"""
Archangel Autonomous AI Evolution - LLM Interface Layer
Standardized interface for LLM reasoning with prompt template management
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
# OpenAI imports - made conditional to avoid dependency issues during testing
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Create mock classes for testing
    class AsyncOpenAI:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    OLLAMA = "ollama"

class PromptType(Enum):
    """Types of prompts for different agent functions"""
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    SAFETY_CHECK = "safety_check"

@dataclass
class LLMConfig:
    """Configuration for LLM interface"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class PromptTemplate:
    """Template for LLM prompts"""
    name: str
    prompt_type: PromptType
    system_prompt: str
    user_prompt_template: str
    required_variables: List[str]
    optional_variables: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)
    safety_constraints: List[str] = field(default_factory=list)

@dataclass
class LLMRequest:
    """Request to LLM"""
    prompt_type: PromptType
    template_name: str
    variables: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    constraints: List[str] = field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    prompt_type: PromptType
    template_name: str
    model: str
    tokens_used: int
    response_time: float
    confidence_score: float
    validation_passed: bool
    safety_passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class PromptTemplateManager:
    """Manages prompt templates for different agent functions"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates"""
        
        # Reasoning template for situation analysis
        self.templates["red_team_reasoning"] = PromptTemplate(
            name="red_team_reasoning",
            prompt_type=PromptType.REASONING,
            system_prompt="""You are an autonomous Red Team agent in a cybersecurity simulation. 
Your role is to analyze the current environment and identify attack opportunities while maintaining stealth.
You must operate within ethical boundaries and simulation constraints.
Always provide structured reasoning with confidence scores.""",
            user_prompt_template="""
Current Environment State:
- Network Topology: {network_topology}
- Active Services: {active_services}
- Security Alerts: {security_alerts}
- Threat Level: {threat_level}

Agent Context:
- Role: {agent_role}
- Team: {agent_team}
- Current Objectives: {current_objectives}
- Available Tools: {available_tools}

Previous Actions: {previous_actions}

Analyze this situation and provide:
1. Situation Assessment (what's happening)
2. Threat Analysis (current security posture)
3. Opportunity Identification (potential attack vectors)
4. Risk Assessment (likelihood of detection/success)
5. Recommended Actions (prioritized list)
6. Confidence Score (0.0-1.0)

Format your response as JSON with these exact keys.
""",
            required_variables=["network_topology", "active_services", "security_alerts", 
                              "threat_level", "agent_role", "agent_team", "current_objectives", 
                              "available_tools", "previous_actions"],
            safety_constraints=["no_real_world_attacks", "simulation_only", "ethical_boundaries"]
        )
        
        # Blue team reasoning template
        self.templates["blue_team_reasoning"] = PromptTemplate(
            name="blue_team_reasoning",
            prompt_type=PromptType.REASONING,
            system_prompt="""You are an autonomous Blue Team agent in a cybersecurity simulation.
Your role is to detect, analyze, and respond to security threats while maintaining system availability.
Focus on defensive strategies and incident response procedures.""",
            user_prompt_template="""
Current Environment State:
- Network Topology: {network_topology}
- Security Alerts: {security_alerts}
- System Logs: {system_logs}
- Threat Level: {threat_level}

Agent Context:
- Role: {agent_role}
- Team: {agent_team}
- Current Objectives: {current_objectives}
- Available Tools: {available_tools}

Recent Incidents: {recent_incidents}

Analyze this situation and provide:
1. Situation Assessment (current security status)
2. Threat Analysis (identified threats and indicators)
3. Response Opportunities (defensive actions available)
4. Risk Assessment (impact and urgency levels)
5. Recommended Actions (prioritized response plan)
6. Confidence Score (0.0-1.0)

Format your response as JSON with these exact keys.
""",
            required_variables=["network_topology", "security_alerts", "system_logs",
                              "threat_level", "agent_role", "agent_team", "current_objectives",
                              "available_tools", "recent_incidents"],
            safety_constraints=["maintain_availability", "minimize_disruption", "follow_procedures"]
        )
        
        # Planning template
        self.templates["action_planning"] = PromptTemplate(
            name="action_planning",
            prompt_type=PromptType.PLANNING,
            system_prompt="""You are creating an action plan based on situation analysis.
Create detailed, executable plans with clear success criteria and fallback options.
Consider resource constraints and time limitations.""",
            user_prompt_template="""
Reasoning Results:
{reasoning_results}

Available Resources:
- Tools: {available_tools}
- Time Constraints: {time_constraints}
- Resource Limits: {resource_limits}

Create an action plan with:
1. Primary Action (main action to take)
2. Action Type (category of action)
3. Target (what/who to target)
4. Parameters (specific configuration)
5. Expected Outcome (what should happen)
6. Success Criteria (how to measure success)
7. Fallback Actions (alternatives if primary fails)
8. Estimated Duration (time to complete)
9. Risk Level (low/medium/high)

Format as JSON with these exact keys.
""",
            required_variables=["reasoning_results", "available_tools", "time_constraints", "resource_limits"],
            safety_constraints=["feasible_actions", "resource_aware", "time_bounded"]
        )
        
        # Safety validation template
        self.templates["safety_validation"] = PromptTemplate(
            name="safety_validation",
            prompt_type=PromptType.SAFETY_CHECK,
            system_prompt="""You are a safety validator for autonomous agent actions.
Evaluate proposed actions for safety, ethics, and simulation boundaries.
Reject any actions that could cause harm or violate constraints.""",
            user_prompt_template="""
Proposed Action:
{proposed_action}

Agent Context:
- Team: {agent_team}
- Role: {agent_role}
- Constraints: {agent_constraints}

Safety Constraints:
- Simulation Only: Must not affect real systems
- Ethical Boundaries: No harmful or malicious intent
- Resource Limits: Must respect system limitations
- Legal Compliance: Must follow applicable laws

Evaluate this action and provide:
1. Safety Assessment (safe/unsafe with reasoning)
2. Constraint Violations (list any violations)
3. Risk Level (low/medium/high/critical)
4. Recommendations (modifications if needed)
5. Approval Status (approved/rejected/modified)

Format as JSON with these exact keys.
""",
            required_variables=["proposed_action", "agent_team", "agent_role", "agent_constraints"],
            validation_rules=["must_be_json", "all_fields_required"],
            safety_constraints=["simulation_only", "no_harm", "ethical_boundaries"]
        )
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name"""
        return self.templates.get(template_name)
    
    def add_template(self, template: PromptTemplate) -> None:
        """Add a new prompt template"""
        self.templates[template.name] = template
    
    def list_templates(self, prompt_type: Optional[PromptType] = None) -> List[str]:
        """List available templates, optionally filtered by type"""
        if prompt_type:
            return [name for name, template in self.templates.items() 
                   if template.prompt_type == prompt_type]
        return list(self.templates.keys())
    
    def validate_variables(self, template_name: str, variables: Dict[str, Any]) -> List[str]:
        """Validate that all required variables are provided"""
        template = self.get_template(template_name)
        if not template:
            return [f"Template '{template_name}' not found"]
        
        errors = []
        for required_var in template.required_variables:
            if required_var not in variables:
                errors.append(f"Missing required variable: {required_var}")
        
        return errors

class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""
    
    @abstractmethod
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    async def validate_response(self, response: LLMResponse) -> bool:
        """Validate LLM response"""
        pass

class OpenAIInterface(LLMInterface):
    """OpenAI LLM interface implementation"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        if OPENAI_AVAILABLE:
            self.client = AsyncOpenAI(api_key=config.api_key)
        else:
            self.client = AsyncOpenAI()  # Mock client for testing
        self.template_manager = PromptTemplateManager()
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API"""
        start_time = time.time()
        
        try:
            # Get template
            template = self.template_manager.get_template(request.template_name)
            if not template:
                raise ValueError(f"Template '{request.template_name}' not found")
            
            # Validate variables
            validation_errors = self.template_manager.validate_variables(
                request.template_name, request.variables
            )
            if validation_errors:
                raise ValueError(f"Template validation failed: {validation_errors}")
            
            # Format prompt
            user_prompt = template.user_prompt_template.format(**request.variables)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": template.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Add context if provided
            if request.context:
                context_msg = f"Additional Context: {json.dumps(request.context, indent=2)}"
                messages.append({"role": "user", "content": context_msg})
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=request.max_tokens or self.config.max_tokens,
                temperature=request.temperature or self.config.temperature,
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            
            # Create response object
            llm_response = LLMResponse(
                content=response.choices[0].message.content,
                prompt_type=request.prompt_type,
                template_name=request.template_name,
                model=self.config.model,
                tokens_used=response.usage.total_tokens,
                response_time=response_time,
                confidence_score=0.8,  # Default confidence, can be improved
                validation_passed=False,  # Will be set by validation
                safety_passed=False,     # Will be set by safety check
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
            # Validate response
            llm_response.validation_passed = await self.validate_response(llm_response)
            
            # Safety check
            llm_response.safety_passed = await self.safety_check(llm_response, template)
            
            return llm_response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    async def validate_response(self, response: LLMResponse) -> bool:
        """Validate LLM response format and content"""
        try:
            # Check if response is valid JSON for structured responses
            if response.prompt_type in [PromptType.REASONING, PromptType.PLANNING, PromptType.SAFETY_CHECK]:
                try:
                    json.loads(response.content)
                except json.JSONDecodeError:
                    logger.warning(f"Response is not valid JSON: {response.content[:100]}...")
                    return False
            
            # Check response length
            if len(response.content.strip()) < 10:
                logger.warning("Response too short")
                return False
            
            # Check for harmful content (basic check)
            harmful_keywords = ["hack", "attack", "exploit", "breach", "penetrate"]
            if response.prompt_type == PromptType.SAFETY_CHECK:
                # Safety checks can mention these terms
                pass
            elif any(keyword in response.content.lower() for keyword in harmful_keywords):
                # For simulation context, these are acceptable
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            return False
    
    async def safety_check(self, response: LLMResponse, template: PromptTemplate) -> bool:
        """Perform safety check on LLM response"""
        try:
            # Check against template safety constraints
            for constraint in template.safety_constraints:
                if constraint == "simulation_only":
                    # Ensure response doesn't reference real systems
                    real_world_indicators = ["production", "live system", "real network"]
                    if any(indicator in response.content.lower() for indicator in real_world_indicators):
                        logger.warning(f"Response may reference real systems: {constraint}")
                        return False
                
                elif constraint == "ethical_boundaries":
                    # Check for unethical content
                    unethical_indicators = ["cause harm", "illegal", "malicious"]
                    if any(indicator in response.content.lower() for indicator in unethical_indicators):
                        logger.warning(f"Response may violate ethical boundaries: {constraint}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return False

class LLMReasoningEngine:
    """Main engine for LLM-powered agent reasoning"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.interface = self._create_interface(config)
        self.template_manager = PromptTemplateManager()
        self.response_cache = {}
        self.safety_validator = SafetyValidator()
    
    def _create_interface(self, config: LLMConfig) -> LLMInterface:
        """Create appropriate LLM interface based on provider"""
        if config.provider == LLMProvider.OPENAI:
            return OpenAIInterface(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    async def reason_about_situation(self, agent_context: Dict[str, Any], 
                                   environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reasoning about current situation"""
        
        # Determine template based on agent team
        template_name = f"{agent_context['team']}_team_reasoning"
        
        # Prepare variables
        variables = {
            **environment_state,
            **agent_context,
            "previous_actions": agent_context.get("previous_actions", [])
        }
        
        # Create request
        request = LLMRequest(
            prompt_type=PromptType.REASONING,
            template_name=template_name,
            variables=variables
        )
        
        # Generate response
        response = await self.interface.generate_response(request)
        
        if not response.validation_passed or not response.safety_passed:
            raise ValueError("LLM response failed validation or safety checks")
        
        # Parse JSON response
        try:
            reasoning_result = json.loads(response.content)
            reasoning_result["_metadata"] = {
                "model": response.model,
                "tokens_used": response.tokens_used,
                "response_time": response.response_time,
                "confidence_score": response.confidence_score
            }
            return reasoning_result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reasoning response: {e}")
            raise
    
    async def create_action_plan(self, reasoning_results: Dict[str, Any],
                               agent_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create action plan based on reasoning results"""
        
        variables = {
            "reasoning_results": json.dumps(reasoning_results, indent=2),
            "available_tools": agent_context.get("available_tools", []),
            "time_constraints": agent_context.get("time_constraints", "30 minutes"),
            "resource_limits": agent_context.get("resource_limits", {})
        }
        
        request = LLMRequest(
            prompt_type=PromptType.PLANNING,
            template_name="action_planning",
            variables=variables
        )
        
        response = await self.interface.generate_response(request)
        
        if not response.validation_passed or not response.safety_passed:
            raise ValueError("Action plan failed validation or safety checks")
        
        try:
            action_plan = json.loads(response.content)
            action_plan["_metadata"] = {
                "model": response.model,
                "tokens_used": response.tokens_used,
                "response_time": response.response_time
            }
            return action_plan
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse action plan response: {e}")
            raise
    
    async def validate_action_safety(self, proposed_action: Dict[str, Any],
                                   agent_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate action safety using LLM"""
        
        variables = {
            "proposed_action": json.dumps(proposed_action, indent=2),
            "agent_team": agent_context.get("team", "unknown"),
            "agent_role": agent_context.get("role", "unknown"),
            "agent_constraints": agent_context.get("constraints", [])
        }
        
        request = LLMRequest(
            prompt_type=PromptType.SAFETY_CHECK,
            template_name="safety_validation",
            variables=variables
        )
        
        response = await self.interface.generate_response(request)
        
        try:
            safety_result = json.loads(response.content)
            safety_result["_metadata"] = {
                "model": response.model,
                "validation_passed": response.validation_passed,
                "safety_passed": response.safety_passed
            }
            return safety_result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse safety validation response: {e}")
            raise

class SafetyValidator:
    """Additional safety validation for LLM responses"""
    
    def __init__(self):
        self.blocked_patterns = [
            r"rm -rf /",
            r"format c:",
            r"delete from.*where.*1=1",
            r"drop table",
            r"shutdown -h now"
        ]
        self.simulation_keywords = ["simulation", "mock", "test", "demo", "sandbox"]
    
    async def validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate that action is safe for simulation"""
        try:
            action_str = json.dumps(action).lower()
            
            # Check for dangerous patterns
            import re
            for pattern in self.blocked_patterns:
                if re.search(pattern, action_str):
                    logger.warning(f"Blocked dangerous pattern: {pattern}")
                    return False
            
            # Ensure simulation context
            if not any(keyword in action_str for keyword in self.simulation_keywords):
                # Add simulation context warning
                logger.info("Action lacks explicit simulation context")
            
            return True
            
        except Exception as e:
            logger.error(f"Safety validation failed: {e}")
            return False