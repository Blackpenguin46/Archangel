"""
Archangel Unified AI Orchestrator
Combines multiple AI capabilities into a seamless, production-ready system
"""

import asyncio
import os
import json
import logging
import time
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import our existing components with fallbacks
try:
    from .huggingface_orchestrator import HuggingFaceAIOrchestrator, AIResponse, ModelCapability, ModelType
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("⚠️ HuggingFace orchestrator not available - using lightweight mode")

try:
    from .kernel_interface import create_kernel_interface
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False
    print("⚠️ Kernel interface not available")

try:
    from ..tools.tool_integration import create_ai_orchestrator
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    print("⚠️ Tool integration not available")

# Import lightweight fallback
from .lightweight_orchestrator import LightweightAIOrchestrator

class AITaskType(Enum):
    SECURITY_ANALYSIS = "security_analysis"
    CONVERSATIONAL_CHAT = "conversational_chat"
    CODE_ANALYSIS = "code_analysis"
    THREAT_ASSESSMENT = "threat_assessment"
    TOOL_ORCHESTRATION = "tool_orchestration"
    KERNEL_DECISION = "kernel_decision"

class AICapabilityLevel(Enum):
    BASIC = "basic"           # Simple text generation
    ADVANCED = "advanced"     # Complex reasoning
    EXPERT = "expert"         # Specialist knowledge
    AUTONOMOUS = "autonomous" # Independent decision making

@dataclass
class UnifiedAIRequest:
    """Standardized request format for all AI operations"""
    task_type: AITaskType
    content: str
    context: Optional[Dict[str, Any]] = None
    target: Optional[str] = None
    capability_level: AICapabilityLevel = AICapabilityLevel.ADVANCED
    session_id: str = "default"
    metadata: Dict[str, Any] = None

@dataclass
class UnifiedAIResponse:
    """Comprehensive response from the unified AI system"""
    request_id: str
    task_type: AITaskType
    content: str
    confidence: float
    reasoning: Optional[str] = None
    recommendations: List[str] = None
    next_actions: List[str] = None
    metadata: Dict[str, Any] = None
    execution_time: float = 0.0
    model_used: str = "unknown"
    error: Optional[str] = None

class UnifiedAIOrchestrator:
    """
    Unified AI orchestrator that coordinates all AI capabilities:
    - Hugging Face model management
    - Claude agent integration  
    - Tool orchestration
    - Kernel interface
    - Conversation management
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        # Core AI components
        self.hf_orchestrator = None
        self.kernel_interface = None
        self.tool_orchestrator = None
        self.lightweight_orchestrator = None
        self.fallback_mode = False
        
        # Claude agent placeholders (to be implemented)
        self.claude_agents = {
            "security_expert": None,
            "code_analyzer": None, 
            "threat_hunter": None,
            "incident_responder": None,
            "vulnerability_researcher": None
        }
        
        # State management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.request_history: List[UnifiedAIRequest] = []
        self.response_cache: Dict[str, UnifiedAIResponse] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "tasks_by_type": {},
            "errors_by_type": {}
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> bool:
        """Initialize all AI components with graceful fallbacks"""
        self.logger.info("🚀 Initializing Unified AI Orchestrator...")
        
        try:
            # Try full HuggingFace orchestrator first
            if HUGGINGFACE_AVAILABLE:
                self.logger.info("🤗 Setting up HuggingFace AI models...")
                try:
                    self.hf_orchestrator = HuggingFaceAIOrchestrator(self.hf_token)
                    hf_success = await self.hf_orchestrator.initialize()
                    
                    if hf_success:
                        self.logger.info("✅ HuggingFace models ready")
                    else:
                        self.logger.warning("⚠️ HuggingFace initialization issues - trying lightweight mode")
                        raise Exception("HuggingFace initialization failed")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ HuggingFace orchestrator failed: {e}")
                    self.hf_orchestrator = None
            
            # Fallback to lightweight orchestrator if needed
            if not self.hf_orchestrator:
                self.logger.info("🎆 Setting up Lightweight AI Orchestrator...")
                self.lightweight_orchestrator = LightweightAIOrchestrator(self.hf_token)
                lightweight_success = await self.lightweight_orchestrator.initialize()
                
                if lightweight_success:
                    self.logger.info("✅ Lightweight AI Orchestrator ready")
                    self.fallback_mode = True
                else:
                    self.logger.error("❌ Both orchestrators failed to initialize")
                    return False
            
            # Initialize kernel interface
            if KERNEL_AVAILABLE:
                try:
                    self.logger.info("⚡ Connecting to kernel interface...")
                    self.kernel_interface = create_kernel_interface()
                    self.kernel_interface.init_communication()
                    self.logger.info("✅ Kernel interface connected")
                except Exception as e:
                    self.logger.warning(f"⚠️ Kernel interface unavailable: {e}")
                    self.kernel_interface = None
            
            # Initialize tool orchestration
            if TOOLS_AVAILABLE:
                try:
                    self.logger.info("🛠️ Setting up tool orchestration...")
                    self.tool_orchestrator = create_ai_orchestrator()
                    self.logger.info("✅ Tool orchestration ready")
                except Exception as e:
                    self.logger.warning(f"⚠️ Tool orchestration limited: {e}")
                    self.tool_orchestrator = None
            
            # Initialize Claude agents (placeholder)
            await self._initialize_claude_agents()
            
            mode = "Lightweight" if self.fallback_mode else "Full"
            self.logger.info(f"✅ Unified AI Orchestrator initialized successfully! (Mode: {mode})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Unified AI Orchestrator: {e}")
            return False
    
    async def _initialize_claude_agents(self):
        """Initialize Claude agent connections (placeholder for future implementation)"""
        self.logger.info("🧠 Setting up Claude agent integration...")
        
        # Placeholder - would integrate with Claude API for specialized agents
        for agent_name in self.claude_agents:
            self.claude_agents[agent_name] = {
                "available": False,
                "specialization": agent_name,
                "last_used": None
            }
        
        self.logger.info("📝 Claude agent integration ready (development mode)")
    
    async def process_request(self, request: UnifiedAIRequest) -> UnifiedAIResponse:
        """Process a unified AI request using the most appropriate capabilities"""
        start_time = time.time()
        request_id = f"req_{int(time.time())}_{len(self.request_history)}"
        
        # Track request
        self.request_history.append(request)
        self.performance_metrics["total_requests"] += 1
        
        # Update task type tracking
        task_name = request.task_type.value
        if task_name not in self.performance_metrics["tasks_by_type"]:
            self.performance_metrics["tasks_by_type"][task_name] = 0
        self.performance_metrics["tasks_by_type"][task_name] += 1
        
        try:
            # Route request to appropriate AI capability
            if request.task_type == AITaskType.SECURITY_ANALYSIS:
                response_content = await self._handle_security_analysis(request)
            elif request.task_type == AITaskType.CONVERSATIONAL_CHAT:
                response_content = await self._handle_conversational_chat(request)
            elif request.task_type == AITaskType.CODE_ANALYSIS:
                response_content = await self._handle_code_analysis(request)
            elif request.task_type == AITaskType.THREAT_ASSESSMENT:
                response_content = await self._handle_threat_assessment(request)
            elif request.task_type == AITaskType.TOOL_ORCHESTRATION:
                response_content = await self._handle_tool_orchestration(request)
            elif request.task_type == AITaskType.KERNEL_DECISION:
                response_content = await self._handle_kernel_decision(request)
            else:
                response_content = await self._handle_general_request(request)
            
            # Create successful response
            execution_time = time.time() - start_time
            
            response = UnifiedAIResponse(
                request_id=request_id,
                task_type=request.task_type,
                content=response_content.get("content", "No response generated"),
                confidence=response_content.get("confidence", 0.7),
                reasoning=response_content.get("reasoning"),
                recommendations=response_content.get("recommendations", []),
                next_actions=response_content.get("next_actions", []),
                metadata=response_content.get("metadata", {}),
                execution_time=execution_time,
                model_used=response_content.get("model_used", "unified_orchestrator")
            )
            
            # Update performance metrics
            self.performance_metrics["successful_requests"] += 1
            self._update_average_response_time(execution_time)
            
            # Cache response
            self.response_cache[request_id] = response
            
            return response
            
        except Exception as e:
            # Handle errors
            execution_time = time.time() - start_time
            error_type = type(e).__name__
            
            if error_type not in self.performance_metrics["errors_by_type"]:
                self.performance_metrics["errors_by_type"][error_type] = 0
            self.performance_metrics["errors_by_type"][error_type] += 1
            
            self.logger.error(f"❌ Request processing failed: {e}")
            
            return UnifiedAIResponse(
                request_id=request_id,
                task_type=request.task_type,
                content=f"Request processing failed: {str(e)}",
                confidence=0.0,
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _handle_security_analysis(self, request: UnifiedAIRequest) -> Dict[str, Any]:
        """Handle security analysis requests"""
        if self.hf_orchestrator:
            # Use full HuggingFace orchestrator for security analysis
            ai_response = await self.hf_orchestrator.security_analysis(
                request.content, 
                context=json.dumps(request.context) if request.context else None
            )
            
            return {
                "content": ai_response.content,
                "confidence": ai_response.confidence,
                "reasoning": ai_response.reasoning,
                "model_used": ai_response.model_used,
                "metadata": ai_response.metadata or {}
            }
        elif self.lightweight_orchestrator:
            # Use lightweight orchestrator
            ai_response = await self.lightweight_orchestrator.process_request(request)
            
            return {
                "content": ai_response.content,
                "confidence": ai_response.confidence,
                "reasoning": ai_response.reasoning,
                "model_used": ai_response.model_used,
                "metadata": ai_response.metadata or {}
            }
        else:
            # Ultimate fallback
            return {
                "content": f"Security analysis for: {request.content}\n\nThis is a fallback response. AI models are not available.",
                "confidence": 0.2,
                "reasoning": "Using fallback mode - no AI models available",
                "model_used": "fallback_analyzer"
            }
    
    async def _handle_conversational_chat(self, request: UnifiedAIRequest) -> Dict[str, Any]:
        """Handle conversational chat requests"""
        if self.hf_orchestrator:
            ai_response = await self.hf_orchestrator.conversational_security_chat(
                request.content, 
                session_id=request.session_id
            )
            
            return {
                "content": ai_response.content,
                "confidence": ai_response.confidence,
                "model_used": ai_response.model_used,
                "metadata": ai_response.metadata or {}
            }
        elif self.lightweight_orchestrator:
            # Use lightweight orchestrator
            ai_response = await self.lightweight_orchestrator.process_request(request)
            
            return {
                "content": ai_response.content,
                "confidence": ai_response.confidence,
                "model_used": ai_response.model_used,
                "metadata": ai_response.metadata or {}
            }
        else:
            return {
                "content": f"I understand you're asking: {request.content}\n\nI'm running in fallback mode. AI models are not available.",
                "confidence": 0.3,
                "model_used": "fallback_chat"
            }
    
    async def _handle_code_analysis(self, request: UnifiedAIRequest) -> Dict[str, Any]:
        """Handle code analysis requests"""
        # Could use specialized Claude agent for code analysis
        return await self._handle_security_analysis(request)  # Fallback for now
    
    async def _handle_threat_assessment(self, request: UnifiedAIRequest) -> Dict[str, Any]:
        """Handle threat assessment requests"""
        return await self._handle_security_analysis(request)  # Fallback for now
    
    async def _handle_tool_orchestration(self, request: UnifiedAIRequest) -> Dict[str, Any]:
        """Handle tool orchestration requests"""
        if self.tool_orchestrator:
            # Extract target from request
            target = request.target or request.content
            
            # Create simple strategy for demonstration
            strategy = {
                "phase_1": {
                    "name": "Reconnaissance",
                    "tools": ["nmap"],
                    "duration": "5-10 minutes",
                    "risk_level": "low"
                }
            }
            
            # Execute strategy
            results = await self.tool_orchestrator.ai_execute_strategy(target, strategy)
            
            # Format results
            result_summary = f"Executed tool orchestration for {target}\n\n"
            for result in results:
                result_summary += f"Tool: {result.tool_name}\n"
                result_summary += f"Command: {result.command}\n"
                result_summary += f"Status: {'Success' if result.exit_code == 0 else 'Failed'}\n"
                if result.parsed_data:
                    result_summary += f"Results: {json.dumps(result.parsed_data, indent=2)}\n"
                result_summary += "\n"
            
            return {
                "content": result_summary,
                "confidence": 0.8,
                "recommendations": ["Review tool outputs for security insights", "Analyze discovered services for vulnerabilities"],
                "model_used": "tool_orchestrator",
                "metadata": {"tools_executed": len(results)}
            }
        else:
            return {
                "content": "Tool orchestration unavailable - tools not initialized",
                "confidence": 0.0,
                "model_used": "none"
            }
    
    async def _handle_kernel_decision(self, request: UnifiedAIRequest) -> Dict[str, Any]:
        """Handle kernel-level security decisions"""
        if self.kernel_interface:
            # This would integrate with kernel for real-time decisions
            return {
                "content": f"Kernel decision analysis: {request.content}",
                "confidence": 0.9,
                "recommendations": ["MONITOR - Allow with logging", "Consider process reputation"],
                "model_used": "kernel_ai"
            }
        else:
            return {
                "content": "Kernel interface unavailable for real-time decisions",
                "confidence": 0.0,
                "model_used": "none"
            }
    
    async def _handle_general_request(self, request: UnifiedAIRequest) -> Dict[str, Any]:
        """Handle general requests that don't fit specific categories"""
        return await self._handle_security_analysis(request)  # Default fallback
    
    def _update_average_response_time(self, execution_time: float):
        """Update running average of response times"""
        current_avg = self.performance_metrics["average_response_time"]
        total_requests = self.performance_metrics["successful_requests"]
        
        if total_requests == 1:
            self.performance_metrics["average_response_time"] = execution_time
        else:
            self.performance_metrics["average_response_time"] = (
                (current_avg * (total_requests - 1) + execution_time) / total_requests
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        hf_status = None
        if self.hf_orchestrator:
            hf_status = self.hf_orchestrator.get_orchestrator_status()
        
        lightweight_status = None
        if self.lightweight_orchestrator:
            lightweight_status = self.lightweight_orchestrator.get_system_status()
        
        return {
            "unified_orchestrator": {
                "initialized": True,
                "mode": "lightweight" if self.fallback_mode else "full",
                "performance_metrics": self.performance_metrics,
                "active_sessions": len(self.active_sessions),
                "cached_responses": len(self.response_cache),
                "fallback_mode": self.fallback_mode
            },
            "huggingface_orchestrator": hf_status,
            "lightweight_orchestrator": lightweight_status,
            "kernel_interface": {
                "available": self.kernel_interface is not None,
                "status": "connected" if self.kernel_interface else "unavailable"
            },
            "tool_orchestrator": {
                "available": self.tool_orchestrator is not None,
                "tools": self.tool_orchestrator.get_available_tools() if self.tool_orchestrator else []
            },
            "claude_agents": self.claude_agents,
            "capabilities": {
                "security_analysis": True,
                "conversational_chat": True,
                "tool_orchestration": self.tool_orchestrator is not None,
                "kernel_decisions": self.kernel_interface is not None,
                "full_ai_models": not self.fallback_mode
            }
        }
    
    async def cleanup(self):
        """Cleanup all resources"""
        self.logger.info("🧹 Cleaning up Unified AI Orchestrator...")
        
        if self.hf_orchestrator:
            await self.hf_orchestrator.cleanup()
        
        if self.lightweight_orchestrator:
            await self.lightweight_orchestrator.cleanup()
        
        # Clear caches
        self.response_cache.clear()
        self.active_sessions.clear()
        
        self.logger.info("✅ Cleanup completed")

# Factory function
def create_unified_ai_orchestrator(hf_token: Optional[str] = None) -> UnifiedAIOrchestrator:
    """Factory function to create unified AI orchestrator"""
    return UnifiedAIOrchestrator(hf_token=hf_token)

# Convenience functions for common operations
async def analyze_security_target(target: str, hf_token: Optional[str] = None) -> UnifiedAIResponse:
    """Convenience function for security analysis"""
    orchestrator = create_unified_ai_orchestrator(hf_token)
    await orchestrator.initialize()
    
    request = UnifiedAIRequest(
        task_type=AITaskType.SECURITY_ANALYSIS,
        content=f"Perform comprehensive security analysis of: {target}",
        target=target,
        capability_level=AICapabilityLevel.EXPERT
    )
    
    return await orchestrator.process_request(request)

async def chat_with_security_ai(message: str, session_id: str = "default", hf_token: Optional[str] = None) -> UnifiedAIResponse:
    """Convenience function for security chat"""
    orchestrator = create_unified_ai_orchestrator(hf_token)
    await orchestrator.initialize()
    
    request = UnifiedAIRequest(
        task_type=AITaskType.CONVERSATIONAL_CHAT,
        content=message,
        session_id=session_id,
        capability_level=AICapabilityLevel.ADVANCED
    )
    
    return await orchestrator.process_request(request)