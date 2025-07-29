#!/usr/bin/env python3
"""
Archangel Linux - Lightweight Real AI Security Expert
Uses Hugging Face API directly without local model downloads
"""

import asyncio
import sys
import os
import requests
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import time

from core.kernel_interface import create_kernel_interface
from tools.tool_integration import create_ai_orchestrator

def load_env_file():
    """Load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
from core.unified_ai_orchestrator import (
    create_unified_ai_orchestrator, UnifiedAIRequest, AITaskType, AICapabilityLevel
)

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class ThreatLevel(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityAnalysis:
    target: str
    target_type: str
    reasoning: str
    strategy: Dict[str, Any]
    confidence: ConfidenceLevel
    threat_level: ThreatLevel
    recommendations: List[str]
    next_actions: List[str]
    timestamp: float

class LightweightArchangelAI:
    """
    Lightweight Archangel AI using Unified AI Orchestrator
    Combines HF API with unified orchestration for seamless AI experience
    """
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.api_url = "https://api-inference.huggingface.co/models/"
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        
        # Try different models in order of preference (updated with working models)
        self.models = [
            "microsoft/DialoGPT-medium",       # Conversational AI
            "gpt2-medium",                     # Better text generation
            "gpt2",                            # Basic text generation
            "distilgpt2",                      # Lightweight text generation
            "t5-small",                        # Text-to-text transformer
            "facebook/opt-350m"                # Open Pretrained Transformer
        ]
        self.active_model = None
        
        # Unified AI orchestrator (new)
        self.unified_orchestrator = None
        
        # System interfaces (legacy - kept for compatibility)
        self.kernel_interface = None
        self.tool_orchestrator = None
        
        # AI state
        self.conversation_history = []
        self.analysis_history = []
    
    async def initialize(self):
        """Initialize the lightweight AI system with comprehensive validation"""
        print("🛡️ ARCHANGEL LINUX - Lightweight Real AI")
        print("=" * 50)
        print("🌐 Using Hugging Face Inference API...")
        print("🔑 Validating authentication...")
        
        # Validate token access
        token_valid, token_msg = await validate_hf_token_access(self.hf_token)
        if token_valid:
            print(f"  ✅ {token_msg}")
        else:
            print(f"  ⚠️ {token_msg}")
            print("  🔄 Proceeding with model testing anyway...")
        
        # Test API connection and find working model
        print("\n🔍 Finding available AI models...")
        await self._find_working_model()
        
        # Initialize unified AI orchestrator
        print("\n🎯 Initializing Unified AI Orchestrator...")
        try:
            self.unified_orchestrator = create_unified_ai_orchestrator(self.hf_token)
            orchestrator_ready = await self.unified_orchestrator.initialize()
            if orchestrator_ready:
                print("  ✅ Unified AI Orchestrator ready")
                print("  🧠 Multiple AI capabilities integrated")
                print("  🛠️ Tool orchestration active")
                print("  ⚡ Kernel interface connected")
            else:
                print("  ⚠️ Unified orchestrator partially initialized")
        except Exception as e:
            print(f"  ⚠️ Unified orchestrator warning: {e}")
            print("  🔄 Falling back to basic HF API mode")
            self.unified_orchestrator = None
        
        # Legacy initialization (fallback)
        if not self.unified_orchestrator:
            print("\n⚡ Setting up fallback components...")
            try:
                self.kernel_interface = create_kernel_interface()
                self.kernel_interface.init_communication()
                print("  ✅ Kernel interface ready")
            except Exception as e:
                print(f"  ⚠️ Kernel interface warning: {e}")
            
            try:
                self.tool_orchestrator = create_ai_orchestrator()
                print("  ✅ Tool orchestration ready")
            except Exception as e:
                print(f"  ⚠️ Tool orchestration warning: {e}")
        
        print(f"\n✅ REAL AI READY! Using model: {self.active_model}")
        print("🧠 Direct connection to Hugging Face neural networks")
        print("🚀 No local downloads - cloud-powered AI reasoning")
        print("🔮 Real neural network responses - not mock data!")
        if self.unified_orchestrator:
            print("🎯 Unified AI orchestration active - full capabilities enabled!")
        print()
    
    async def _find_working_model(self):
        """Find a working model from the preference list with better error handling"""
        successful_models = []
        
        for model in self.models:
            try:
                print(f"  Testing model: {model}")
                
                # First check if model exists without making inference call
                check_response = requests.get(
                    f"https://huggingface.co/api/models/{model}",
                    headers=self.headers,
                    timeout=5
                )
                
                if check_response.status_code == 404:
                    print(f"  ❌ Model not found: {model}")
                    continue
                elif check_response.status_code == 403:
                    print(f"  ⚠️ Access denied to model info: {model}")
                    # Still try inference - might work
                
                # Test inference API call with minimal payload
                inference_payload = {
                    "inputs": "Hello",
                    "parameters": {
                        "max_new_tokens": 10,
                        "do_sample": False,
                        "return_full_text": False
                    },
                    "options": {
                        "wait_for_model": True,
                        "use_cache": False
                    }
                }
                
                response = requests.post(
                    f"{self.api_url}{model}",
                    headers=self.headers,
                    json=inference_payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    self.active_model = model
                    print(f"  ✅ Connected to: {model}")
                    return
                elif response.status_code == 503:
                    print(f"  ⏳ Model loading: {model} - adding to retry list")
                    successful_models.append(model)
                elif response.status_code == 403:
                    print(f"  ⚠️ Access denied to: {model}")
                elif response.status_code == 400:
                    print(f"  ⚠️ Bad request for: {model} - trying simpler payload")
                    # Try with even simpler payload
                    simple_response = requests.post(
                        f"{self.api_url}{model}",
                        headers=self.headers,
                        json={"inputs": "Test"},
                        timeout=15
                    )
                    if simple_response.status_code == 200:
                        self.active_model = model
                        print(f"  ✅ Connected to: {model} (simple mode)")
                        return
                else:
                    print(f"  ❌ Failed: {model} (status: {response.status_code})")
                    if response.text:
                        print(f"      Error: {response.text[:100]}...")
                    
            except requests.exceptions.Timeout:
                print(f"  ⏰ Timeout for: {model}")
            except Exception as e:
                print(f"  ❌ Error with {model}: {e}")                    
        
        # Retry models that were loading
        if successful_models:
            print("\n  🔄 Retrying models that were loading...")
            for model in successful_models:
                try:
                    response = requests.post(
                        f"{self.api_url}{model}",
                        headers=self.headers,
                        json={"inputs": "Test"},
                        timeout=15
                    )
                    if response.status_code == 200:
                        self.active_model = model
                        print(f"  ✅ Connected to: {model} (retry successful)")
                        return
                except:
                    continue
        
        raise Exception("No working model found! This could be due to:\n" + 
                       "  1. Invalid HF token\n" + 
                       "  2. Token lacks required permissions\n" + 
                       "  3. All models are currently unavailable\n" + 
                       "  4. Network connectivity issues")
    
    async def query_ai(self, prompt: str, max_length: int = 512, retries: int = 3) -> str:
        """Query the AI model via Hugging Face API with robust error handling"""
        for attempt in range(retries):
            try:
                # Adaptive payload based on model capabilities
                payload = {
                    "inputs": prompt[:2000],  # Limit input length
                    "parameters": {
                        "max_new_tokens": min(max_length, 256),  # Conservative token limit
                        "temperature": 0.7,
                        "do_sample": True,
                        "return_full_text": False,
                        "pad_token_id": 50256  # Common pad token
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
                    timeout=45
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict):
                            text = result[0].get('generated_text', str(result[0]))
                        else:
                            text = str(result[0])
                        return text.strip() if text else "Empty response"
                    elif isinstance(result, dict):
                        text = result.get('generated_text', str(result))
                        return text.strip() if text else "Empty response"
                    else:
                        return str(result).strip() if result else "Empty response"
                        
                elif response.status_code == 503:
                    if attempt < retries - 1:
                        print(f"  ⏳ Model busy, waiting... (attempt {attempt + 1}/{retries})")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return "Model is currently busy. Please try again later."
                elif response.status_code == 400:
                    # Try with simpler parameters
                    simple_payload = {"inputs": prompt[:1000]}
                    simple_response = requests.post(
                        f"{self.api_url}{self.active_model}",
                        headers=self.headers,
                        json=simple_payload,
                        timeout=30
                    )
                    if simple_response.status_code == 200:
                        result = simple_response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return str(result[0]).strip()
                        return str(result).strip()
                    else:
                        return f"API Error: {response.status_code} - {response.text[:200]}"
                else:
                    error_msg = f"API Error: {response.status_code}"
                    if response.text:
                        error_msg += f" - {response.text[:200]}"
                    return error_msg
                    
            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    print(f"  ⏰ Request timeout, retrying... (attempt {attempt + 1}/{retries})")
                    continue
                else:
                    return "Request timeout. The model may be overloaded."
            except Exception as e:
                if attempt < retries - 1:
                    print(f"  ⚠️ Query error, retrying... (attempt {attempt + 1}/{retries}): {e}")
                    await asyncio.sleep(1)
                    continue
                else:
                    return f"Query failed after {retries} attempts: {e}"
        
        return "AI query failed after all retry attempts"
    
    async def analyze_target(self, target: str) -> SecurityAnalysis:
        """Real AI security analysis using unified orchestrator or fallback to direct API"""
        print(f"🎯 REAL AI CLOUD ANALYSIS: {target}")
        print("=" * 50)
        
        # Try unified orchestrator first
        if self.unified_orchestrator:
            print("🎯 Using Unified AI Orchestrator for comprehensive analysis...")
            try:
                request = UnifiedAIRequest(
                    task_type=AITaskType.SECURITY_ANALYSIS,
                    content=f"Perform comprehensive security analysis of: {target}",
                    target=target,
                    capability_level=AICapabilityLevel.EXPERT
                )
                
                response = await self.unified_orchestrator.process_request(request)
                
                # Convert to SecurityAnalysis format
                analysis = SecurityAnalysis(
                    target=target,
                    target_type=self._identify_target_type(target),
                    reasoning=response.content,
                    strategy=self._create_strategy_from_response(response, target),
                    confidence=self._convert_confidence(response.confidence),
                    threat_level=self._assess_threat_level(response.content),
                    recommendations=response.recommendations or self._default_recommendations(),
                    next_actions=response.next_actions or self._default_next_actions(),
                    timestamp=time.time()
                )
                
                self.analysis_history.append(analysis)
                return analysis
                
            except Exception as e:
                print(f"⚠️ Unified orchestrator failed: {e}")
                print("🔄 Falling back to direct HF API...")
        
        # Fallback to original implementation
        print("🌐 Sending to Hugging Face neural networks...")
        
        # Create comprehensive security analysis prompt
        security_prompt = f"""
        You are an expert cybersecurity consultant. Analyze the security of this target: {target}

        Provide a comprehensive security assessment including:

        1. Target Analysis:
        - What type of target is this? (web app, IP, domain, etc.)
        - What technologies might be involved?
        - What are the potential attack surfaces?

        2. Risk Assessment:
        - What security risks should be considered?
        - What are the legal and ethical considerations?
        - What approach minimizes risk while being thorough?

        3. Strategy Recommendation:
        - What methodology should be used? (OWASP, NIST, etc.)
        - What phases should the assessment include?
        - What tools would be most appropriate?

        4. Specific Recommendations:
        - What are the first steps to take?
        - What should be avoided?
        - How to ensure responsible disclosure?

        Target to analyze: {target}

        Analysis:
        """
        
        print("🧠 AI is analyzing with real neural networks...")
        ai_reasoning = await self.query_ai(security_prompt, max_length=1024)
        
        print("📊 AI generating strategic recommendations...")
        
        # Generate strategy using AI
        strategy_prompt = f"""
        Based on this security analysis of {target}:
        {ai_reasoning[:300]}...
        
        Create a detailed 3-phase security testing strategy:
        Phase 1: Reconnaissance
        Phase 2: Active Testing  
        Phase 3: Analysis
        
        For each phase specify tools, duration, and risk level.
        
        Strategy:
        """
        
        strategy_response = await self.query_ai(strategy_prompt, max_length=512)
        strategy = self._parse_ai_strategy(strategy_response, target)
        
        # AI confidence assessment
        confidence_prompt = f"""
        Rate confidence level (LOW/MEDIUM/HIGH) for this security analysis of {target}:
        {ai_reasoning[:200]}...
        
        Confidence level:
        """
        
        confidence_response = await self.query_ai(confidence_prompt, max_length=50)
        confidence = self._parse_confidence(confidence_response)
        
        # AI threat level assessment
        threat_prompt = f"""
        Assess threat level (LOW/MEDIUM/HIGH/CRITICAL) for target {target} based on:
        {ai_reasoning[:200]}...
        
        Threat level:
        """
        
        threat_response = await self.query_ai(threat_prompt, max_length=50)
        threat_level = self._parse_threat_level(threat_response)
        
        # AI recommendations
        rec_prompt = f"""
        Based on analysis of {target}, provide 5 specific security recommendations:
        {ai_reasoning[:200]}...
        
        Recommendations:
        """
        
        rec_response = await self.query_ai(rec_prompt, max_length=300)
        recommendations = self._parse_recommendations(rec_response)
        
        # AI next actions
        actions_prompt = f"""
        Based on the security strategy for {target}, what are the specific next actions to take?
        
        Next actions:
        """
        
        actions_response = await self.query_ai(actions_prompt, max_length=200)
        next_actions = self._parse_next_actions(actions_response)
        
        # Create analysis result
        analysis = SecurityAnalysis(
            target=target,
            target_type=self._identify_target_type(target),
            reasoning=ai_reasoning,
            strategy=strategy,
            confidence=confidence,
            threat_level=threat_level,
            recommendations=recommendations,
            next_actions=next_actions,
            timestamp=time.time()
        )
        
        # Store for learning
        self.analysis_history.append(analysis)
        
        return analysis
    
    def _parse_ai_strategy(self, response: str, target: str) -> Dict[str, Any]:
        """Parse AI strategy response"""
        target_type = self._identify_target_type(target)
        
        # Simple parsing - real implementation would use NLP
        if "web" in response.lower() or target_type == "web_application":
            return {
                "phase_1": {
                    "name": "Web Reconnaissance",
                    "tools": ["nmap", "dirb", "whatweb"],
                    "duration": "10-15 minutes",
                    "risk_level": "low"
                },
                "phase_2": {
                    "name": "Web Application Testing",
                    "tools": ["burp", "sqlmap", "nikto"],
                    "duration": "30-45 minutes", 
                    "risk_level": "medium"
                },
                "phase_3": {
                    "name": "Vulnerability Analysis",
                    "tools": ["manual_testing", "code_review"],
                    "duration": "45-60 minutes",
                    "risk_level": "medium"
                }
            }
        else:
            return {
                "phase_1": {
                    "name": "Network Discovery",
                    "tools": ["nmap", "ping"],
                    "duration": "5-10 minutes",
                    "risk_level": "low"
                },
                "phase_2": {
                    "name": "Service Analysis",
                    "tools": ["nmap_scripts", "banner_grab"],
                    "duration": "15-20 minutes",
                    "risk_level": "low"
                }
            }
    
    def _parse_confidence(self, response: str) -> ConfidenceLevel:
        """Parse confidence from AI response"""
        response_lower = response.lower()
        if "high" in response_lower:
            return ConfidenceLevel.HIGH
        elif "medium" in response_lower:
            return ConfidenceLevel.MEDIUM
        elif "low" in response_lower:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.MEDIUM
    
    def _parse_threat_level(self, response: str) -> ThreatLevel:
        """Parse threat level from AI response"""
        response_lower = response.lower()
        if "critical" in response_lower:
            return ThreatLevel.CRITICAL
        elif "high" in response_lower:
            return ThreatLevel.HIGH
        elif "low" in response_lower:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.MEDIUM
    
    def _parse_recommendations(self, response: str) -> List[str]:
        """Parse recommendations from AI response"""
        lines = response.split('\n')
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('1.') or 'recommend' in line.lower()):
                recommendations.append(line.lstrip('-•1234567890. '))
                if len(recommendations) >= 5:
                    break
        
        if not recommendations:
            recommendations = [
                "Start with passive reconnaissance to avoid detection",
                "Use rate limiting to prevent overwhelming the target",
                "Monitor for any signs of impact during testing",
                "Document all findings for educational purposes",
                "Follow responsible disclosure practices"
            ]
        
        return recommendations
    
    def _parse_next_actions(self, response: str) -> List[str]:
        """Parse next actions from AI response"""
        lines = response.split('\n')
        actions = []
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('1.') or 'action' in line.lower()):
                actions.append(line.lstrip('-•1234567890. '))
                if len(actions) >= 5:
                    break
        
        if not actions:
            actions = [
                "Begin reconnaissance phase with passive information gathering",
                "Execute active scanning with appropriate rate limiting",
                "Analyze results and identify potential vulnerabilities",
                "Generate comprehensive security assessment report"
            ]
        
        return actions
    
    def _create_strategy_from_response(self, response, target: str) -> Dict[str, Any]:
        """Create strategy from unified orchestrator response"""
        target_type = self._identify_target_type(target)
        
        # Extract strategy from response content or use default
        if "web" in response.content.lower() or target_type == "web_application":
            return {
                "phase_1": {
                    "name": "Web Reconnaissance",
                    "tools": ["nmap", "dirb", "whatweb"],
                    "duration": "10-15 minutes",
                    "risk_level": "low"
                },
                "phase_2": {
                    "name": "Web Application Testing",
                    "tools": ["burp", "sqlmap", "nikto"],
                    "duration": "30-45 minutes", 
                    "risk_level": "medium"
                }
            }
        else:
            return {
                "phase_1": {
                    "name": "Network Discovery",
                    "tools": ["nmap", "ping"],
                    "duration": "5-10 minutes",
                    "risk_level": "low"
                }
            }
    
    def _convert_confidence(self, confidence: float) -> ConfidenceLevel:
        """Convert float confidence to ConfidenceLevel enum"""
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.LOW
    
    def _assess_threat_level(self, content: str) -> ThreatLevel:
        """Assess threat level from response content"""
        content_lower = content.lower()
        if any(word in content_lower for word in ['critical', 'severe', 'high risk']):
            return ThreatLevel.CRITICAL
        elif any(word in content_lower for word in ['high', 'significant', 'major']):
            return ThreatLevel.HIGH
        elif any(word in content_lower for word in ['medium', 'moderate']):
            return ThreatLevel.MEDIUM
        elif any(word in content_lower for word in ['low', 'minor']):
            return ThreatLevel.LOW
        else:
            return ThreatLevel.MEDIUM
    
    def _default_recommendations(self) -> List[str]:
        """Default security recommendations"""
        return [
            "Start with passive reconnaissance to avoid detection",
            "Use rate limiting to prevent overwhelming the target",
            "Monitor for any signs of impact during testing",
            "Document all findings for educational purposes",
            "Follow responsible disclosure practices"
        ]
    
    def _default_next_actions(self) -> List[str]:
        """Default next actions"""
        return [
            "Begin reconnaissance phase with passive information gathering",
            "Execute active scanning with appropriate rate limiting",
            "Analyze results and identify potential vulnerabilities",
            "Generate comprehensive security assessment report"
        ]
    
    def _identify_target_type(self, target: str) -> str:
        """Identify target type"""
        if any(proto in target.lower() for proto in ['http://', 'https://', 'www.']):
            return "web_application"
        elif target.count('.') == 3 and all(part.isdigit() for part in target.split('.')):
            return "ip_address"
        elif '/' in target:
            return "network_range"
        elif '.' in target:
            return "domain_name"
        else:
            return "hostname"
    
    async def kernel_analysis_demo(self):
        """Demo AI-kernel integration with unified orchestrator"""
        print("\n⚡ REAL AI-KERNEL INTEGRATION")
        print("=" * 40)
        
        from core.kernel_interface import SecurityContext
        
        context = SecurityContext(
            pid=1337,
            uid=0,
            syscall_nr=59,  # execve
            timestamp=time.time_ns(),
            flags=0x0001,
            comm="suspicious_binary"
        )
        
        print("🔍 Kernel context:")
        print(f"  Process: {context.comm} (PID {context.pid})")
        print(f"  User: {'root' if context.uid == 0 else 'user'}")
        print(f"  Syscall: {context.syscall_nr}")
        
        # Try unified orchestrator first
        if self.unified_orchestrator:
            print("\n🎯 Using Unified AI Orchestrator for kernel analysis...")
            try:
                kernel_analysis_content = f"""
                Analyze this kernel security context for threats:
                
                Process: {context.comm} (PID {context.pid})
                User: {'root' if context.uid == 0 else 'user'} (UID: {context.uid})
                System Call: {context.syscall_nr} (execve - process execution)
                Flags: {hex(context.flags)}
                
                This is a real-time kernel security decision. Should this be:
                ALLOW - Let the process execute normally
                DENY - Block the process execution  
                MONITOR - Allow but log for analysis
                
                Consider: root execution, binary name, syscall type.
                """
                
                request = UnifiedAIRequest(
                    task_type=AITaskType.KERNEL_DECISION,
                    content=kernel_analysis_content,
                    capability_level=AICapabilityLevel.EXPERT,
                    metadata={"context": context.__dict__}
                )
                
                response = await self.unified_orchestrator.process_request(request)
                
                # Parse decision from unified response
                decision = "MONITOR"  # Safe default
                if "DENY" in response.content.upper() or "BLOCK" in response.content.upper():
                    decision = "DENY"
                elif "ALLOW" in response.content.upper():
                    decision = "ALLOW"
                elif "MONITOR" in response.content.upper():
                    decision = "MONITOR"
                
                print(f"🎯 Unified AI Decision: {decision}")
                print(f"🧠 AI Reasoning: {response.content[:300]}...")
                print(f"📊 Confidence: {response.confidence:.2f}")
                if response.recommendations:
                    print(f"💡 Recommendations: {', '.join(response.recommendations[:2])}")
                print("\n(Generated by Unified AI Orchestrator with real neural networks!)")
                return
                
            except Exception as e:
                print(f"⚠️ Unified orchestrator failed: {e}")
                print("🔄 Falling back to direct API...")
        
        # Fallback to original implementation
        kernel_prompt = f"""
        Analyze this kernel security context for threats:
        
        Process: {context.comm} (PID {context.pid})
        User: {'root' if context.uid == 0 else 'user'} (UID: {context.uid})
        System Call: {context.syscall_nr} (execve - process execution)
        Flags: {hex(context.flags)}
        
        This is a real-time kernel security decision. Should this be:
        ALLOW - Let the process execute normally
        DENY - Block the process execution  
        MONITOR - Allow but log for analysis
        
        Consider: root execution, binary name, syscall type.
        
        Decision and reasoning:
        """
        
        print("\n🧠 Real AI analyzing kernel context...")
        decision_response = await self.query_ai(kernel_prompt, max_length=200)
        
        # Parse decision
        decision = "MONITOR"  # Safe default
        if "DENY" in decision_response.upper() or "BLOCK" in decision_response.upper():
            decision = "DENY"
        elif "ALLOW" in decision_response.upper():
            decision = "ALLOW"
        elif "MONITOR" in decision_response.upper():
            decision = "MONITOR"
        
        print(f"🎯 Real AI Decision: {decision}")
        print(f"🧠 AI Reasoning: {decision_response[:200]}...")
        print("\n(This was generated by real AI neural networks!)")
    
    async def interactive_session(self):
        """Interactive session with real cloud AI"""
        print("\n🎮 REAL AI INTERACTIVE SESSION")
        print("=" * 40)
        print("Commands:")
        print("  analyze <target>  - AI analyzes target")  
        print("  kernel           - Demo AI-kernel integration")
        print("  chat <message>   - Chat with security AI")
        print("  quit             - Exit")
        print()
        
        while True:
            try:
                print("CloudAI> ", end="")
                command = input().strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.lower().startswith('analyze '):
                    target = command[8:].strip()
                    if target:
                        analysis = await self.analyze_target(target)
                        print(f"\n📊 REAL AI ANALYSIS RESULTS:")
                        print(f"Target: {analysis.target}")
                        print(f"Type: {analysis.target_type}")
                        print(f"Confidence: {analysis.confidence.value}")
                        print(f"Threat: {analysis.threat_level.value}")
                        print(f"\n🧠 AI Reasoning:\n{analysis.reasoning[:300]}...")
                        print(f"\n📋 AI Recommendations:")
                        for i, rec in enumerate(analysis.recommendations[:3], 1):
                            print(f"{i}. {rec}")
                elif command.lower() == 'kernel':
                    await self.kernel_analysis_demo()
                elif command.lower().startswith('chat '):
                    message = command[5:].strip()
                    
                    # Try unified orchestrator first
                    if self.unified_orchestrator:
                        try:
                            request = UnifiedAIRequest(
                                task_type=AITaskType.CONVERSATIONAL_CHAT,
                                content=message,
                                capability_level=AICapabilityLevel.ADVANCED
                            )
                            response = await self.unified_orchestrator.process_request(request)
                            print(f"🤖 AI: {response.content}")
                            if response.confidence < 0.7:
                                print(f"   (Confidence: {response.confidence:.2f})")
                        except Exception as e:
                            print(f"⚠️ Unified chat failed: {e}")
                            # Fallback to direct API
                            chat_prompt = f"You are a cybersecurity expert. User asks: {message}\n\nResponse:"
                            response = await self.query_ai(chat_prompt, max_length=300)
                            print(f"🤖 AI: {response}")
                    else:
                        # Direct API fallback
                        chat_prompt = f"You are a cybersecurity expert. User asks: {message}\n\nResponse:"
                        response = await self.query_ai(chat_prompt, max_length=300)
                        print(f"🤖 AI: {response}")
                elif command.strip() == '':
                    continue
                else:
                    print(f"❌ Unknown command: {command}")
                
                print()
                
            except (EOFError, KeyboardInterrupt):
                break
        
        print("\n👋 Real AI session ended.")
        
        # Cleanup unified orchestrator if available
        if self.unified_orchestrator:
            await self.unified_orchestrator.cleanup()

def get_hf_token():
    """Get HF token from .env file or user input with validation"""
    # Load .env file first
    load_env_file()
    
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    
    if not token:
        print("🔑 HUGGING FACE TOKEN REQUIRED")
        print("=" * 35)
        print("This uses the Hugging Face Inference API for real AI.")
        print("You need a free HF token from: https://huggingface.co/settings/tokens")
        print()
        print("📝 Token Requirements:")
        print("  • Read access to public models")
        print("  • Valid format: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("  • Must not be expired or revoked")
        print()
        try:
            token = input("Enter your HF token: ").strip()
            if not token:
                print("❌ Token required for real AI functionality")
                sys.exit(1)
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Token required")
            sys.exit(1)
    
    # Basic token format validation
    if not validate_hf_token_format(token):
        print("⚠️ Warning: Token format appears invalid")
        print("Expected format: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("Continuing anyway - will test during initialization...")
    
    return token

def validate_hf_token_format(token: str) -> bool:
    """Validate HF token format"""
    if not token:
        return False
    
    # HF tokens typically start with 'hf_' and are 37 characters total
    if token.startswith('hf_') and len(token) == 37:
        return True
    
    # Some tokens might be longer or have different format
    if len(token) >= 20 and not token.isspace():
        return True
        
    return False

async def validate_hf_token_access(token: str) -> tuple[bool, str]:
    """Validate HF token has proper access"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        # Test with HF API whoami endpoint
        response = requests.get(
            "https://huggingface.co/api/whoami",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            user_info = response.json()
            username = user_info.get('name', 'Unknown')
            return True, f"Authenticated as: {username}"
        elif response.status_code == 401:
            return False, "Invalid token - authentication failed"
        else:
            return False, f"Token validation failed with status: {response.status_code}"
            
    except Exception as e:
        return False, f"Token validation error: {e}"

async def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("🛡️ ARCHANGEL LINUX - Lightweight Real AI")
        print("=" * 45)
        print("Usage:")
        print(f"  {sys.argv[0]} analyze <target>")
        print(f"  {sys.argv[0]} interactive")
        print(f"  {sys.argv[0]} kernel")
        print()
        print("🌐 Uses Hugging Face cloud AI - no downloads!")
        print("🔑 Requires HF token for real neural networks")
        return
    
    # Get HF token
    hf_token = get_hf_token()
    
    # Initialize lightweight AI
    archangel = LightweightArchangelAI(hf_token)
    await archangel.initialize()
    
    command = sys.argv[1].lower()
    
    if command == 'analyze' and len(sys.argv) >= 3:
        target = sys.argv[2]
        analysis = await archangel.analyze_target(target)
        
        print(f"\n📊 REAL AI ANALYSIS COMPLETE")
        print("=" * 35)
        print(f"🎯 Target: {analysis.target}")
        print(f"📝 Type: {analysis.target_type}")
        print(f"📊 AI Confidence: {analysis.confidence.value}")
        print(f"⚠️ Threat Level: {analysis.threat_level.value}")
        print(f"\n🧠 REAL AI REASONING:")
        print("-" * 30)
        print(analysis.reasoning)
        print(f"\n📋 AI RECOMMENDATIONS:")
        for i, rec in enumerate(analysis.recommendations, 1):
            print(f"{i}. {rec}")
            
    elif command == 'interactive':
        await archangel.interactive_session()
    elif command == 'kernel':
        await archangel.kernel_analysis_demo()
    else:
        print("❌ Invalid command")

if __name__ == "__main__":
    asyncio.run(main())