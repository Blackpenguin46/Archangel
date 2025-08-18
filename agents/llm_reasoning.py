"""
LLM Reasoning and Behavior Tree Integration

This module provides the core LLM reasoning capabilities and behavior tree integration
for autonomous agents in the Archangel system.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import asyncio

# Optional imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning operations"""
    TACTICAL = "tactical"
    STRATEGIC = "strategic"
    REACTIVE = "reactive"
    ANALYTICAL = "analytical"


class NodeResult(Enum):
    """Behavior tree node execution results"""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


@dataclass
class ReasoningContext:
    """Context information for LLM reasoning"""
    agent_id: str
    team: str
    role: str
    current_phase: str
    environment_state: Dict[str, Any]
    objectives: List[str]
    available_tools: List[str]
    memory_context: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ReasoningResult:
    """Result of LLM reasoning operation"""
    reasoning_type: ReasoningType
    decision: str
    confidence: float
    reasoning_chain: List[str]
    recommended_actions: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PromptTemplate:
    """Standardized prompt template for LLM interactions"""
    
    def __init__(self, template_name: str, template: str, variables: List[str]):
        self.template_name = template_name
        self.template = template
        self.variables = variables
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables"""
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        return self.template.format(**kwargs)


class PromptTemplateManager:
    """Manages standardized prompt templates for different agent roles and scenarios"""
    
    def __init__(self):
        self.templates = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates"""
        
        # Red Team Reconnaissance Template
        self.templates["red_recon"] = PromptTemplate(
            "red_recon",
            """You are a Red Team Reconnaissance Agent in a cybersecurity simulation.

ROLE: {role}
TEAM: Red Team
CURRENT PHASE: {current_phase}
OBJECTIVES: {objectives}

ENVIRONMENT STATE:
{environment_state}

AVAILABLE TOOLS: {available_tools}

MEMORY CONTEXT:
{memory_context}

Your task is to analyze the current situation and determine the best reconnaissance approach.
Consider stealth, effectiveness, and information gathering potential.

Provide your reasoning in the following format:
1. Situation Analysis
2. Risk Assessment
3. Recommended Actions
4. Expected Outcomes

Be tactical, methodical, and prioritize operational security.""",
            ["role", "current_phase", "objectives", "environment_state", "available_tools", "memory_context"]
        )
        
        # Blue Team SOC Analyst Template
        self.templates["blue_soc"] = PromptTemplate(
            "blue_soc",
            """You are a Blue Team SOC Analyst Agent in a cybersecurity simulation.

ROLE: {role}
TEAM: Blue Team
CURRENT PHASE: {current_phase}
OBJECTIVES: {objectives}

ENVIRONMENT STATE:
{environment_state}

AVAILABLE TOOLS: {available_tools}

MEMORY CONTEXT:
{memory_context}

Your task is to monitor for threats, analyze alerts, and coordinate defensive responses.
Focus on detection accuracy, response speed, and threat containment.

Provide your reasoning in the following format:
1. Threat Analysis
2. Alert Prioritization
3. Response Strategy
4. Coordination Requirements

Be vigilant, systematic, and prioritize threat containment.""",
            ["role", "current_phase", "objectives", "environment_state", "available_tools", "memory_context"]
        )
    
    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a prompt template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]
    
    def add_template(self, template: PromptTemplate):
        """Add a new prompt template"""
        self.templates[template.template_name] = template


class LLMInterface(ABC):
    """Abstract interface for LLM providers"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    def validate_response(self, response: str) -> bool:
        """Validate the LLM response for safety and correctness"""
        pass


class OpenAIInterface(LLMInterface):
    """OpenAI GPT interface implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response using OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def validate_response(self, response: str) -> bool:
        """Validate OpenAI response"""
        # Basic validation - check for harmful content, prompt injection, etc.
        harmful_patterns = [
            "ignore previous instructions",
            "system prompt",
            "jailbreak",
            "real world attack",
            "actual network"
        ]
        
        response_lower = response.lower()
        for pattern in harmful_patterns:
            if pattern in response_lower:
                logger.warning(f"Potentially harmful content detected: {pattern}")
                return False
        
        return True


class LocalLLMInterface(LLMInterface):
    """Local LLM interface for offline operation"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        # In a real implementation, this would load a local model like Ollama
        logger.info(f"Local LLM interface initialized with model: {model_path}")
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response using local LLM"""
        # Placeholder implementation - would integrate with actual local LLM
        await asyncio.sleep(0.1)  # Simulate processing time
        return f"Local LLM response to: {prompt[:50]}..."
    
    def validate_response(self, response: str) -> bool:
        """Validate local LLM response"""
        return len(response) > 0 and len(response) < 5000


class LLMReasoningEngine:
    """Core LLM reasoning engine with safety validation"""
    
    def __init__(self, primary_llm: LLMInterface, fallback_llm: Optional[LLMInterface] = None):
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm
        self.template_manager = PromptTemplateManager()
        self.reasoning_history = []
    
    async def reason(self, context: ReasoningContext, reasoning_type: ReasoningType) -> ReasoningResult:
        """Perform LLM reasoning with the given context"""
        try:
            # Select appropriate template based on agent role and team
            template_name = self._select_template(context.team.lower(), context.role.lower())
            template = self.template_manager.get_template(template_name)
            
            # Format prompt with context
            prompt = template.format(
                role=context.role,
                current_phase=context.current_phase,
                objectives=json.dumps(context.objectives, indent=2),
                environment_state=json.dumps(context.environment_state, indent=2),
                available_tools=json.dumps(context.available_tools, indent=2),
                memory_context=json.dumps(context.memory_context, indent=2)
            )
            
            # Generate response
            response = await self._generate_with_fallback(prompt)
            
            # Parse and validate response
            reasoning_result = self._parse_response(response, reasoning_type, context)
            
            # Store in history
            self.reasoning_history.append({
                'context': asdict(context),
                'result': asdict(reasoning_result),
                'prompt': prompt,
                'raw_response': response
            })
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            # Return safe fallback result
            return ReasoningResult(
                reasoning_type=reasoning_type,
                decision="Error in reasoning - using safe fallback",
                confidence=0.0,
                reasoning_chain=["Error occurred during reasoning"],
                recommended_actions=[{"action": "wait", "priority": "low"}],
                risk_assessment={"overall": 1.0},
                metadata={"error": str(e)}
            )
    
    def _select_template(self, team: str, role: str) -> str:
        """Select appropriate template based on team and role"""
        if team == "red":
            if "recon" in role:
                return "red_recon"
            else:
                return "red_recon"  # Default for now
        elif team == "blue":
            if "soc" in role or "analyst" in role:
                return "blue_soc"
            else:
                return "blue_soc"  # Default for now
        else:
            return "red_recon"  # Safe default
    
    async def _generate_with_fallback(self, prompt: str) -> str:
        """Generate response with fallback to secondary LLM if needed"""
        try:
            response = await self.primary_llm.generate_response(prompt)
            if self.primary_llm.validate_response(response):
                return response
            else:
                logger.warning("Primary LLM response failed validation")
                if self.fallback_llm:
                    return await self.fallback_llm.generate_response(prompt)
                else:
                    raise ValueError("Response validation failed and no fallback available")
        except Exception as e:
            logger.error(f"Primary LLM failed: {e}")
            if self.fallback_llm:
                return await self.fallback_llm.generate_response(prompt)
            else:
                raise
    
    def _parse_response(self, response: str, reasoning_type: ReasoningType, context: ReasoningContext) -> ReasoningResult:
        """Parse LLM response into structured reasoning result"""
        # Simple parsing - in production this would be more sophisticated
        lines = response.split('\n')
        
        # Extract key components
        reasoning_chain = []
        recommended_actions = []
        confidence = 0.7  # Default confidence
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "situation analysis" in line.lower() or "threat analysis" in line.lower():
                current_section = "analysis"
            elif "risk assessment" in line.lower():
                current_section = "risk"
            elif "recommended actions" in line.lower() or "response strategy" in line.lower():
                current_section = "actions"
            elif current_section == "analysis":
                reasoning_chain.append(line)
            elif current_section == "actions" and line.startswith(('-', '*', '1.', '2.', '3.')):
                action_text = line.lstrip('-*123456789. ')
                recommended_actions.append({
                    "action": action_text,
                    "priority": "medium",
                    "confidence": confidence
                })
        
        # If no structured actions found, create a default one
        if not recommended_actions:
            recommended_actions = [{
                "action": "analyze_situation",
                "priority": "medium",
                "confidence": 0.5
            }]
        
        return ReasoningResult(
            reasoning_type=reasoning_type,
            decision=response[:200] + "..." if len(response) > 200 else response,
            confidence=confidence,
            reasoning_chain=reasoning_chain if reasoning_chain else ["Analysis completed"],
            recommended_actions=recommended_actions,
            risk_assessment={"overall": 0.5, "detection": 0.3, "impact": 0.4},
            metadata={"agent_id": context.agent_id, "response_length": len(response)}
        )


# Behavior Tree Implementation
class BehaviorTreeNode(ABC):
    """Abstract base class for behavior tree nodes"""
    
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.children = []
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> NodeResult:
        """Execute the node and return result"""
        pass
    
    def add_child(self, child: 'BehaviorTreeNode'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
    
    def reset(self):
        """Reset node state"""
        for child in self.children:
            child.reset()


class SequenceNode(BehaviorTreeNode):
    """Sequence node - executes children in order until one fails"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.current_child = 0
    
    async def execute(self, context: Dict[str, Any]) -> NodeResult:
        """Execute children in sequence"""
        while self.current_child < len(self.children):
            result = await self.children[self.current_child].execute(context)
            
            if result == NodeResult.FAILURE:
                self.reset()
                return NodeResult.FAILURE
            elif result == NodeResult.RUNNING:
                return NodeResult.RUNNING
            else:  # SUCCESS
                self.current_child += 1
        
        self.reset()
        return NodeResult.SUCCESS
    
    def reset(self):
        """Reset sequence state"""
        self.current_child = 0
        super().reset()


class SelectorNode(BehaviorTreeNode):
    """Selector node - executes children until one succeeds"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.current_child = 0
    
    async def execute(self, context: Dict[str, Any]) -> NodeResult:
        """Execute children until one succeeds"""
        while self.current_child < len(self.children):
            result = await self.children[self.current_child].execute(context)
            
            if result == NodeResult.SUCCESS:
                self.reset()
                return NodeResult.SUCCESS
            elif result == NodeResult.RUNNING:
                return NodeResult.RUNNING
            else:  # FAILURE
                self.current_child += 1
        
        self.reset()
        return NodeResult.FAILURE
    
    def reset(self):
        """Reset selector state"""
        self.current_child = 0
        super().reset()


class ConditionNode(BehaviorTreeNode):
    """Condition node - evaluates a condition function"""
    
    def __init__(self, name: str, condition_func: Callable[[Dict[str, Any]], bool]):
        super().__init__(name)
        self.condition_func = condition_func
    
    async def execute(self, context: Dict[str, Any]) -> NodeResult:
        """Evaluate condition"""
        try:
            if self.condition_func(context):
                return NodeResult.SUCCESS
            else:
                return NodeResult.FAILURE
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return NodeResult.FAILURE


class ActionNode(BehaviorTreeNode):
    """Action node - executes an action function"""
    
    def __init__(self, name: str, action_func: Callable[[Dict[str, Any]], Any]):
        super().__init__(name)
        self.action_func = action_func
        self.is_running = False
    
    async def execute(self, context: Dict[str, Any]) -> NodeResult:
        """Execute action"""
        try:
            if not self.is_running:
                self.is_running = True
                
            result = await self._execute_action(context)
            
            if result is True:
                self.is_running = False
                return NodeResult.SUCCESS
            elif result is False:
                self.is_running = False
                return NodeResult.FAILURE
            else:  # Still running
                return NodeResult.RUNNING
                
        except Exception as e:
            logger.error(f"Action execution error: {e}")
            self.is_running = False
            return NodeResult.FAILURE
    
    async def _execute_action(self, context: Dict[str, Any]) -> Union[bool, None]:
        """Execute the action function"""
        if asyncio.iscoroutinefunction(self.action_func):
            return await self.action_func(context)
        else:
            return self.action_func(context)
    
    def reset(self):
        """Reset action state"""
        self.is_running = False
        super().reset()


class LLMActionNode(ActionNode):
    """Action node that uses LLM reasoning to determine actions"""
    
    def __init__(self, name: str, reasoning_engine: LLMReasoningEngine, reasoning_type: ReasoningType):
        super().__init__(name, self._llm_action)
        self.reasoning_engine = reasoning_engine
        self.reasoning_type = reasoning_type
    
    async def _llm_action(self, context: Dict[str, Any]) -> bool:
        """Use LLM reasoning to determine action"""
        try:
            # Create reasoning context from behavior tree context
            reasoning_context = ReasoningContext(
                agent_id=context.get('agent_id', 'unknown'),
                team=context.get('team', 'unknown'),
                role=context.get('role', 'unknown'),
                current_phase=context.get('current_phase', 'unknown'),
                environment_state=context.get('environment_state', {}),
                objectives=context.get('objectives', []),
                available_tools=context.get('available_tools', []),
                memory_context=context.get('memory_context', {})
            )
            
            # Get LLM reasoning
            result = await self.reasoning_engine.reason(reasoning_context, self.reasoning_type)
            
            # Update context with reasoning result
            context['last_reasoning'] = result
            context['recommended_actions'] = result.recommended_actions
            
            # Consider action successful if confidence is above threshold
            return result.confidence > 0.5
            
        except Exception as e:
            logger.error(f"LLM action error: {e}")
            return False


class BehaviorTreeBuilder:
    """Builder for creating behavior trees"""
    
    def __init__(self, reasoning_engine: LLMReasoningEngine):
        self.reasoning_engine = reasoning_engine
    
    def build_red_team_tree(self) -> BehaviorTreeNode:
        """Build behavior tree for Red Team agents"""
        root = SelectorNode("RedTeamRoot")
        
        # Reconnaissance sequence
        recon_seq = SequenceNode("ReconSequence")
        recon_seq.add_child(ConditionNode("CheckReconPhase", 
                                        lambda ctx: ctx.get('current_phase') == 'recon'))
        recon_seq.add_child(LLMActionNode("PlanReconnaissance", 
                                        self.reasoning_engine, ReasoningType.TACTICAL))
        recon_seq.add_child(ActionNode("ExecuteRecon", self._execute_recon))
        
        # Exploitation sequence
        exploit_seq = SequenceNode("ExploitSequence")
        exploit_seq.add_child(ConditionNode("CheckExploitPhase",
                                          lambda ctx: ctx.get('current_phase') == 'exploit'))
        exploit_seq.add_child(LLMActionNode("PlanExploitation",
                                          self.reasoning_engine, ReasoningType.STRATEGIC))
        exploit_seq.add_child(ActionNode("ExecuteExploit", self._execute_exploit))
        
        root.add_child(recon_seq)
        root.add_child(exploit_seq)
        
        return root
    
    def build_blue_team_tree(self) -> BehaviorTreeNode:
        """Build behavior tree for Blue Team agents"""
        root = SelectorNode("BlueTeamRoot")
        
        # Monitoring sequence
        monitor_seq = SequenceNode("MonitorSequence")
        monitor_seq.add_child(LLMActionNode("AnalyzeThreat",
                                          self.reasoning_engine, ReasoningType.ANALYTICAL))
        monitor_seq.add_child(ActionNode("UpdateDefenses", self._update_defenses))
        
        # Response sequence
        response_seq = SequenceNode("ResponseSequence")
        response_seq.add_child(ConditionNode("ThreatDetected",
                                           lambda ctx: ctx.get('threat_detected', False)))
        response_seq.add_child(LLMActionNode("PlanResponse",
                                           self.reasoning_engine, ReasoningType.REACTIVE))
        response_seq.add_child(ActionNode("ExecuteResponse", self._execute_response))
        
        root.add_child(monitor_seq)
        root.add_child(response_seq)
        
        return root
    
    async def _execute_recon(self, context: Dict[str, Any]) -> bool:
        """Execute reconnaissance action"""
        logger.info("Executing reconnaissance action")
        # Placeholder implementation
        await asyncio.sleep(0.1)
        return True
    
    async def _execute_exploit(self, context: Dict[str, Any]) -> bool:
        """Execute exploitation action"""
        logger.info("Executing exploitation action")
        # Placeholder implementation
        await asyncio.sleep(0.1)
        return True
    
    async def _update_defenses(self, context: Dict[str, Any]) -> bool:
        """Update defensive measures"""
        logger.info("Updating defenses")
        # Placeholder implementation
        await asyncio.sleep(0.1)
        return True
    
    async def _execute_response(self, context: Dict[str, Any]) -> bool:
        """Execute incident response"""
        logger.info("Executing incident response")
        # Placeholder implementation
        await asyncio.sleep(0.1)
        return True


# Example usage and testing
async def main():
    """Example usage of the LLM reasoning and behavior tree system"""
    
    # Initialize LLM interfaces
    # Note: In production, you would provide actual API keys
    primary_llm = LocalLLMInterface("llama3-70b")  # Using local as example
    fallback_llm = LocalLLMInterface("codellama-34b")
    
    # Create reasoning engine
    reasoning_engine = LLMReasoningEngine(primary_llm, fallback_llm)
    
    # Create behavior tree builder
    tree_builder = BehaviorTreeBuilder(reasoning_engine)
    
    # Build Red Team behavior tree
    red_tree = tree_builder.build_red_team_tree()
    
    # Example context for Red Team agent
    context = {
        'agent_id': 'red_recon_001',
        'team': 'red',
        'role': 'reconnaissance',
        'current_phase': 'recon',
        'environment_state': {
            'network_discovered': False,
            'services_enumerated': False,
            'vulnerabilities_found': []
        },
        'objectives': ['Discover network topology', 'Identify vulnerable services'],
        'available_tools': ['nmap', 'masscan', 'dirb'],
        'memory_context': {'previous_scans': [], 'known_hosts': []}
    }
    
    # Execute behavior tree
    print("Executing Red Team behavior tree...")
    result = await red_tree.execute(context)
    print(f"Execution result: {result}")
    
    # Test direct LLM reasoning
    reasoning_context = ReasoningContext(
        agent_id='red_recon_001',
        team='red',
        role='reconnaissance',
        current_phase='recon',
        environment_state=context['environment_state'],
        objectives=context['objectives'],
        available_tools=context['available_tools'],
        memory_context=context['memory_context']
    )
    
    print("\nTesting direct LLM reasoning...")
    reasoning_result = await reasoning_engine.reason(reasoning_context, ReasoningType.TACTICAL)
    print(f"Reasoning decision: {reasoning_result.decision}")
    print(f"Confidence: {reasoning_result.confidence}")
    print(f"Recommended actions: {reasoning_result.recommended_actions}")


if __name__ == "__main__":
    asyncio.run(main())