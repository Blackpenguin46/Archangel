"""
Archangel Lightweight AI Orchestrator
Minimal dependency version using only requests and standard library
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

class AITaskType(Enum):
    SECURITY_ANALYSIS = "security_analysis"
    CONVERSATIONAL_CHAT = "conversational_chat"
    CODE_ANALYSIS = "code_analysis"
    THREAT_ASSESSMENT = "threat_assessment"
    TOOL_ORCHESTRATION = "tool_orchestration"
    KERNEL_DECISION = "kernel_decision"

class AICapabilityLevel(Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"
    AUTONOMOUS = "autonomous"

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

class LightweightAIOrchestrator:
    """
    Lightweight AI orchestrator using only HuggingFace Inference API
    No heavy dependencies - just requests and standard library
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.api_url = "https://api-inference.huggingface.co/models/"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
        # Available models in order of preference
        self.models = [
            "microsoft/DialoGPT-medium",
            "gpt2-medium",
            "gpt2",
            "distilgpt2",
            "facebook/opt-350m"
        ]
        
        self.active_model = None
        self.conversation_history = {}
        self.request_count = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> bool:
        """Initialize the lightweight orchestrator"""
        self.logger.info("🚀 Initializing Lightweight AI Orchestrator...")
        
        # Find working model
        for model in self.models:
            if await self._test_model(model):
                self.active_model = model
                self.logger.info(f"✅ Using model: {model}")
                return True
        
        self.logger.error("❌ No working model found")
        return False
    
    async def _test_model(self, model_name: str) -> bool:
        """Test if a model is working"""
        try:
            response = requests.post(
                f"{self.api_url}{model_name}",
                headers=self.headers,
                json={
                    "inputs": "Hello",
                    "parameters": {"max_new_tokens": 5, "do_sample": False},
                    "options": {"wait_for_model": True}
                },
                timeout=15
            )
            return response.status_code == 200
        except:
            return False
    
    async def process_request(self, request: UnifiedAIRequest) -> UnifiedAIResponse:
        """Process a unified AI request"""
        start_time = time.time()
        request_id = f"req_{int(time.time())}_{self.request_count}"
        self.request_count += 1
        
        try:
            # Route request based on type
            if request.task_type == AITaskType.SECURITY_ANALYSIS:
                content = await self._handle_security_analysis(request)
            elif request.task_type == AITaskType.CONVERSATIONAL_CHAT:
                content = await self._handle_conversation(request)
            elif request.task_type == AITaskType.TOOL_ORCHESTRATION:
                content = await self._handle_tool_orchestration(request)
            elif request.task_type == AITaskType.KERNEL_DECISION:
                content = await self._handle_kernel_decision(request)
            else:
                content = await self._handle_general_request(request)
            
            execution_time = time.time() - start_time
            
            return UnifiedAIResponse(
                request_id=request_id,
                task_type=request.task_type,
                content=content,
                confidence=0.75,  # Default confidence
                execution_time=execution_time,
                model_used=self.active_model or "fallback",
                recommendations=self._generate_recommendations(request),
                next_actions=self._generate_next_actions(request)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Request processing failed: {e}")
            
            return UnifiedAIResponse(
                request_id=request_id,
                task_type=request.task_type,
                content=f"AI processing failed: {str(e)}",
                confidence=0.0,
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _handle_security_analysis(self, request: UnifiedAIRequest) -> str:
        """Handle security analysis requests"""
        prompt = self._create_security_prompt(request.content, request.target)
        return await self._query_model(prompt)
    
    async def _handle_conversation(self, request: UnifiedAIRequest) -> str:
        """Handle conversational requests"""
        session_id = request.session_id
        
        # Maintain conversation history
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        # Add context from history
        history = self.conversation_history[session_id][-3:]  # Last 3 exchanges
        context = "\\n".join([f"Human: {h['human']}\\nAI: {h['ai']}" for h in history])
        
        prompt = f"""You are Archangel, an AI cybersecurity expert. You provide helpful, accurate information about security topics.
        
{context}

Human: {request.content}
Archangel:"""
        
        response = await self._query_model(prompt)
        
        # Store in history
        self.conversation_history[session_id].append({
            "human": request.content,
            "ai": response
        })
        
        # Keep history manageable
        if len(self.conversation_history[session_id]) > 10:
            self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
        
        return response
    
    async def _handle_tool_orchestration(self, request: UnifiedAIRequest) -> str:
        """Handle tool orchestration requests"""
        # Simulate tool orchestration for now
        target = request.target or request.content.split()[-1]
        
        return f"""Tool Orchestration Analysis for {target}:

🔍 Reconnaissance Phase:
- Nmap port scan: 22/tcp (SSH), 80/tcp (HTTP), 443/tcp (HTTPS) open
- Service detection: Apache 2.4.41, OpenSSH 8.2p1
- Directory enumeration: /admin, /login, /api endpoints discovered

📊 Analysis Results:
- Standard web server configuration detected
- SSH service running on default port
- HTTPS properly configured

💡 Security Recommendations:
1. Verify SSH key-based authentication
2. Check for admin panel security
3. Review API endpoint authorization
4. Scan for common web vulnerabilities

(Note: This is a simulated tool execution - actual tools would be integrated in production)"""
    
    async def _handle_kernel_decision(self, request: UnifiedAIRequest) -> str:
        """Handle kernel-level decisions"""
        prompt = f"""Analyze this kernel security context and recommend an action (ALLOW/DENY/MONITOR):

{request.content}

Consider: Process reputation, user privileges, system call type, and potential security risks.

Decision and reasoning:"""
        
        return await self._query_model(prompt)
    
    async def _handle_general_request(self, request: UnifiedAIRequest) -> str:
        """Handle general requests"""
        prompt = f"""You are Archangel, a cybersecurity AI expert. Respond to this request:

{request.content}

Response:"""
        
        return await self._query_model(prompt)
    
    def _create_security_prompt(self, content: str, target: Optional[str] = None) -> str:
        """Create security analysis prompt"""
        return f"""You are an expert cybersecurity consultant performing a security analysis.

Target: {target or 'Provided content'}
Request: {content}

Provide a comprehensive security assessment including:
1. Target identification and classification
2. Potential security risks and attack vectors
3. Recommended methodology and approach
4. Ethical and legal considerations
5. Specific next steps

Analysis:"""
    
    async def _query_model(self, prompt: str, max_retries: int = 3) -> str:
        """Query the active model with retry logic"""
        if not self.active_model:
            return "AI model not available. Please check HuggingFace token and connection."
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "inputs": prompt[:1500],  # Limit prompt length
                    "parameters": {
                        "max_new_tokens": 256,
                        "temperature": 0.7,
                        "do_sample": True,
                        "return_full_text": False
                    },
                    "options": {
                        "wait_for_model": True,
                        "use_cache": True
                    }
                }
                
                response = requests.post(
                    f"{self.api_url}{self.active_model}",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict):
                            return result[0].get('generated_text', str(result[0])).strip()
                        else:
                            return str(result[0]).strip()
                    elif isinstance(result, dict):
                        return result.get('generated_text', str(result)).strip()
                    else:
                        return str(result).strip()
                elif response.status_code == 503 and attempt < max_retries - 1:
                    # Model loading, wait and retry
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    return f"API Error: {response.status_code} - {response.text[:100]}"
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return f"Query failed: {str(e)}"
        
        return "Failed to get AI response after all retries"
    
    def _generate_recommendations(self, request: UnifiedAIRequest) -> List[str]:
        """Generate context-aware recommendations"""
        if request.task_type == AITaskType.SECURITY_ANALYSIS:
            return [
                "Start with passive reconnaissance to minimize impact",
                "Use rate limiting to avoid overwhelming target systems",
                "Document all findings for proper reporting",
                "Follow responsible disclosure practices",
                "Ensure proper authorization before testing"
            ]
        elif request.task_type == AITaskType.KERNEL_DECISION:
            return [
                "Monitor process behavior for anomalies",
                "Check process against known threat signatures",
                "Review user privilege escalation patterns"
            ]
        else:
            return [
                "Review findings with security team",
                "Implement recommended security controls",
                "Schedule regular security assessments"
            ]
    
    def _generate_next_actions(self, request: UnifiedAIRequest) -> List[str]:
        """Generate context-aware next actions"""
        if request.task_type == AITaskType.SECURITY_ANALYSIS:
            return [
                "Execute reconnaissance phase with passive scanning",
                "Perform active enumeration with appropriate tools",
                "Analyze discovered services for vulnerabilities",
                "Generate comprehensive security report"
            ]
        elif request.task_type == AITaskType.TOOL_ORCHESTRATION:
            return [
                "Review tool execution results",
                "Correlate findings across different tools",
                "Prioritize vulnerabilities by risk level",
                "Plan remediation activities"
            ]
        else:
            return [
                "Continue monitoring and analysis",
                "Implement security improvements",
                "Schedule follow-up assessment"
            ]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "lightweight_orchestrator": {
                "initialized": self.active_model is not None,
                "active_model": self.active_model,
                "request_count": self.request_count,
                "conversation_sessions": len(self.conversation_history),
                "api_available": bool(self.hf_token)
            },
            "huggingface_integration": {
                "token_available": bool(self.hf_token),
                "api_endpoint": self.api_url,
                "models_tested": len(self.models)
            },
            "capabilities": {
                "security_analysis": True,
                "conversational_chat": True,
                "tool_orchestration": "simulated",
                "kernel_decisions": True
            }
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("🧹 Cleaning up Lightweight AI Orchestrator...")
        self.conversation_history.clear()
        self.logger.info("✅ Cleanup completed")

# Factory function
def create_lightweight_orchestrator(hf_token: Optional[str] = None) -> LightweightAIOrchestrator:
    """Factory function to create lightweight orchestrator"""
    return LightweightAIOrchestrator(hf_token=hf_token)

# Alias for compatibility
create_unified_ai_orchestrator = create_lightweight_orchestrator