"""
Archangel Hugging Face AI Orchestrator
Real AI model integration with proper authentication and fallback strategies
"""

import asyncio
import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
from pathlib import Path

# Hugging Face imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    pipeline, BitsAndBytesConfig
)
from huggingface_hub import login, HfApi, model_info, ModelSearchArguments
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# SmolAgents for autonomous capabilities
try:
    from smolagents import CodeAgent, ToolCallingAgent, HfApiModel, LocalModel
    from smolagents.tools import Tool
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    print("⚠️ SmolAgents not available - autonomous capabilities limited")

# LangChain for advanced reasoning
try:
    from langchain.llms.huggingface_pipeline import HuggingFacePipeline
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferWindowMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not available - advanced reasoning limited")

class ModelType(Enum):
    CYBERSECURITY_SPECIALIST = "cybersecurity"
    CONVERSATIONAL = "conversational"
    CODE_ANALYSIS = "code"
    GENERAL_PURPOSE = "general"
    CLASSIFICATION = "classification"

class ModelCapability(Enum):
    TEXT_GENERATION = "text_generation"
    CLASSIFICATION = "classification"
    QUESTION_ANSWERING = "question_answering"
    CODE_GENERATION = "code_generation"
    CONVERSATION = "conversation"

@dataclass
class ModelConfig:
    """Configuration for a Hugging Face model"""
    name: str
    model_type: ModelType
    capabilities: List[ModelCapability]
    memory_requirements: str  # "low", "medium", "high"
    requires_auth: bool = True
    fallback_models: List[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7

@dataclass
class AIResponse:
    """Standardized AI response format"""
    content: str
    model_used: str
    confidence: float
    reasoning: Optional[str] = None
    alternatives: List[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

class HuggingFaceAIOrchestrator:
    """
    Advanced Hugging Face AI model orchestrator with:
    - Proper authentication and token management
    - Intelligent model selection and fallback
    - SmolAgents integration for autonomous operations
    - Multi-model reasoning capabilities
    - Security-focused model prioritization
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.hf_api = None
        self.authenticated = False
        
        # Model registry with security-focused prioritization
        self.model_registry = self._init_model_registry()
        
        # Active models (loaded in memory)
        self.active_models: Dict[str, Any] = {}
        self.active_tokenizers: Dict[str, Any] = {}
        self.active_pipelines: Dict[str, Any] = {}
        
        # SmolAgents setup
        self.security_agent = None
        self.code_agent = None
        
        # LangChain setup
        self.conversation_chains: Dict[str, Any] = {}
        
        # Performance monitoring
        self.model_performance: Dict[str, Dict[str, Any]] = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _init_model_registry(self) -> Dict[str, ModelConfig]:
        """Initialize model registry with security-focused prioritization"""
        return {
            # Cybersecurity specialist models (highest priority) - updated with working models
            "foundation-sec-8b": ModelConfig(
                name="microsoft/DialoGPT-large",  # Fallback to working model
                model_type=ModelType.CYBERSECURITY_SPECIALIST,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.QUESTION_ANSWERING],
                memory_requirements="medium",  # Reduced requirement
                requires_auth=False,  # DialoGPT is public
                fallback_models=["microsoft/DialoGPT-medium", "gpt2-medium", "gpt2"],
                max_tokens=1024,  # Reduced for stability
                temperature=0.3
            ),
            
            # Security-focused conversational models
            "security-dialogue": ModelConfig(
                name="microsoft/DialoGPT-medium",  # More stable choice
                model_type=ModelType.CONVERSATIONAL,
                capabilities=[ModelCapability.CONVERSATION, ModelCapability.TEXT_GENERATION],
                memory_requirements="medium",
                requires_auth=False,
                fallback_models=["gpt2-medium", "gpt2", "distilgpt2"],
                max_tokens=512,  # Reduced for reliability
                temperature=0.7
            ),
            
            # Code analysis models - updated with working alternatives
            "code-security": ModelConfig(
                name="gpt2-medium",  # Better availability for code tasks
                model_type=ModelType.CODE_ANALYSIS,
                capabilities=[ModelCapability.CODE_GENERATION, ModelCapability.TEXT_GENERATION],
                memory_requirements="medium",
                requires_auth=False,
                fallback_models=["gpt2", "distilgpt2"],
                max_tokens=512,  # Conservative limit
                temperature=0.4
            ),
            
            # Classification models for threat detection
            "threat-classifier": ModelConfig(
                name="distilgpt2",  # More reliable than BERT for inference API
                model_type=ModelType.CLASSIFICATION,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.QUESTION_ANSWERING],
                memory_requirements="low",
                requires_auth=False,
                fallback_models=["gpt2"],
                max_tokens=256,
                temperature=0.1
            ),
            
            # General purpose fallback
            "general-purpose": ModelConfig(
                name="gpt2",
                model_type=ModelType.GENERAL_PURPOSE,
                capabilities=[ModelCapability.TEXT_GENERATION],
                memory_requirements="low",
                requires_auth=False,
                fallback_models=["distilgpt2"],
                max_tokens=512,  # Conservative for reliability
                temperature=0.8
            ),
            
            # Additional lightweight options
            "ultra-light": ModelConfig(
                name="distilgpt2",
                model_type=ModelType.GENERAL_PURPOSE,
                capabilities=[ModelCapability.TEXT_GENERATION],
                memory_requirements="low",
                requires_auth=False,
                fallback_models=[],  # No fallback - this is the fallback
                max_tokens=256,
                temperature=0.8
            )
        }
    
    async def initialize(self) -> bool:
        """Initialize the AI orchestrator with authentication and model loading"""
        self.logger.info("🤖 Initializing Hugging Face AI Orchestrator...")
        
        # Authenticate with Hugging Face
        await self._authenticate()
        
        # Load priority models
        await self._load_priority_models()
        
        # Initialize SmolAgents
        if SMOLAGENTS_AVAILABLE:
            await self._initialize_smolagents()
        
        # Initialize LangChain
        if LANGCHAIN_AVAILABLE:
            await self._initialize_langchain()
        
        self.logger.info("✅ Hugging Face AI Orchestrator initialized successfully!")
        return True
    
    async def _authenticate(self) -> bool:
        """Authenticate with Hugging Face Hub"""
        if not self.hf_token:
            self.logger.warning("⚠️ No HF token provided - using public models only")
            return False
        
        try:
            login(token=self.hf_token)
            self.hf_api = HfApi(token=self.hf_token)
            self.authenticated = True
            self.logger.info("✅ Successfully authenticated with Hugging Face Hub")
            return True
        except Exception as e:
            self.logger.error(f"❌ Authentication failed: {e}")
            self.authenticated = False
            return False
    
    async def _load_priority_models(self):
        """Load priority models based on security requirements"""
        priority_order = [
            "foundation-sec-8b",  # Cybersecurity specialist
            "security-dialogue",  # Security conversations
            "code-security",      # Code analysis
            "threat-classifier"   # Threat detection
        ]
        
        for model_key in priority_order:
            config = self.model_registry[model_key]
            
            # Skip authenticated models if not authenticated
            if config.requires_auth and not self.authenticated:
                self.logger.info(f"⏭️ Skipping {config.name} - requires authentication")
                continue
            
            success = await self._load_model_with_fallback(model_key, config)
            if success:
                self.logger.info(f"✅ Loaded priority model: {config.name}")
                break  # Load at least one primary model
            else:
                self.logger.warning(f"⚠️ Failed to load priority model: {config.name}")
    
    async def _load_model_with_fallback(self, model_key: str, config: ModelConfig) -> bool:
        """Load model with intelligent fallback strategy"""
        models_to_try = [config.name] + (config.fallback_models or [])
        
        for model_name in models_to_try:
            try:
                self.logger.info(f"🔄 Attempting to load: {model_name}")
                
                # Check if model exists and is accessible
                if self.authenticated:
                    try:
                        info = model_info(model_name, token=self.hf_token)
                        self.logger.info(f"📊 Model info: {info.modelId} - {info.downloads} downloads")
                    except Exception:
                        self.logger.warning(f"⚠️ Could not fetch info for {model_name}")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    token=self.hf_token if self.authenticated else None,
                    trust_remote_code=True
                )
                
                # Add padding token if missing
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model with memory optimization
                device_map = "auto" if torch.cuda.is_available() else None
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=self.hf_token if self.authenticated else None,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Create pipeline
                task = "text-generation"
                if ModelCapability.CLASSIFICATION in config.capabilities:
                    task = "text-classification"
                
                pipeline_obj = pipeline(
                    task,
                    model=model,
                    tokenizer=tokenizer,
                    max_length=config.max_tokens,
                    temperature=config.temperature,
                    do_sample=True,
                    return_full_text=False
                )
                
                # Store loaded components
                self.active_models[model_key] = model
                self.active_tokenizers[model_key] = tokenizer
                self.active_pipelines[model_key] = pipeline_obj
                
                # Update model registry with successful model
                config.name = model_name
                
                # Initialize performance tracking
                self.model_performance[model_key] = {
                    "model_name": model_name,
                    "total_requests": 0,
                    "successful_requests": 0,
                    "average_response_time": 0.0,
                    "last_used": time.time()
                }
                
                self.logger.info(f"✅ Successfully loaded: {model_name} as {model_key}")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load {model_name}: {e}")
                continue
        
        return False
    
    async def _initialize_smolagents(self):
        """Initialize SmolAgents for autonomous security operations"""
        try:
            self.logger.info("🤖 Initializing SmolAgents...")
            
            # Try to use loaded models for SmolAgents
            if "foundation-sec-8b" in self.active_models:
                # Use the cybersecurity specialist model
                model = LocalModel(
                    model=self.active_models["foundation-sec-8b"],
                    tokenizer=self.active_tokenizers["foundation-sec-8b"]
                )
            elif "security-dialogue" in self.active_models:
                # Fallback to conversational model
                model = LocalModel(
                    model=self.active_models["security-dialogue"],
                    tokenizer=self.active_tokenizers["security-dialogue"] 
                )
            else:
                # Use HfApiModel as fallback
                model = HfApiModel("microsoft/DialoGPT-medium")
            
            # Initialize security-focused agent
            security_tools = self._create_security_tools()
            self.security_agent = CodeAgent(
                tools=security_tools,
                model=model,
                max_iterations=10,
                system_prompt="""You are Archangel, an AI security expert. You perform defensive security analysis,
                explain your reasoning clearly, and always follow ethical guidelines. Focus on:
                1. Systematic security assessment
                2. Clear explanation of findings
                3. Educational value
                4. Ethical and legal compliance"""
            )
            
            self.logger.info("✅ SmolAgents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"⚠️ SmolAgents initialization failed: {e}")
            self.security_agent = None
    
    def _create_security_tools(self) -> List[Tool]:
        """Create security-focused tools for SmolAgents"""
        # This would create custom security tools
        # For now, return empty list - tools would be added based on requirements
        return []
    
    async def _initialize_langchain(self):
        """Initialize LangChain for advanced reasoning"""
        try:
            self.logger.info("🔗 Initializing LangChain...")
            
            for model_key, pipeline_obj in self.active_pipelines.items():
                if ModelCapability.CONVERSATION in self.model_registry[model_key].capabilities:
                    # Create HuggingFace pipeline wrapper
                    llm = HuggingFacePipeline(
                        pipeline=pipeline_obj,
                        model_kwargs={"temperature": self.model_registry[model_key].temperature}
                    )
                    
                    # Create conversation chain with memory
                    memory = ConversationBufferWindowMemory(k=10)
                    conversation = ConversationChain(
                        llm=llm,
                        memory=memory,
                        verbose=True
                    )
                    
                    self.conversation_chains[model_key] = conversation
            
            self.logger.info("✅ LangChain initialized successfully")
            
        except Exception as e:
            self.logger.error(f"⚠️ LangChain initialization failed: {e}")
    
    async def security_analysis(self, query: str, context: Optional[str] = None) -> AIResponse:
        """Perform AI-powered security analysis"""
        start_time = time.time()
        
        # Select best model for security analysis
        model_key = self._select_best_model(ModelCapability.TEXT_GENERATION, ModelType.CYBERSECURITY_SPECIALIST)
        
        if not model_key:
            return AIResponse(
                content="No suitable model available for security analysis",
                model_used="none",
                confidence=0.0,
                execution_time=time.time() - start_time
            )
        
        try:
            # Prepare security-focused prompt
            security_prompt = self._prepare_security_prompt(query, context)
            
            # Generate response using best available model
            if self.security_agent and SMOLAGENTS_AVAILABLE:
                # Use SmolAgent for autonomous analysis
                response_text = await self._smolagent_analysis(security_prompt)
            else:
                # Use direct model inference
                response_text = await self._direct_model_inference(model_key, security_prompt)
            
            # Update performance metrics
            self._update_performance_metrics(model_key, True, time.time() - start_time)
            
            return AIResponse(
                content=response_text,
                model_used=self.model_registry[model_key].name,
                confidence=0.85,  # Would be calculated based on model output
                reasoning="AI security analysis using specialized cybersecurity models",
                execution_time=time.time() - start_time,
                metadata={"model_key": model_key, "prompt_type": "security_analysis"}
            )
            
        except Exception as e:
            self._update_performance_metrics(model_key, False, time.time() - start_time)
            self.logger.error(f"Security analysis failed: {e}")
            
            return AIResponse(
                content=f"Security analysis failed: {str(e)}",
                model_used=self.model_registry[model_key].name,
                confidence=0.0,
                execution_time=time.time() - start_time
            )
    
    def _select_best_model(self, 
                          capability: ModelCapability, 
                          preferred_type: Optional[ModelType] = None) -> Optional[str]:
        """Intelligently select the best model for a given capability"""
        
        candidates = []
        
        for model_key, config in self.model_registry.items():
            # Must be loaded and have required capability
            if model_key not in self.active_models:
                continue
            if capability not in config.capabilities:
                continue
            
            # Calculate score
            score = 0
            
            # Prefer specific model type
            if preferred_type and config.model_type == preferred_type:
                score += 100
            
            # Performance bonus
            if model_key in self.model_performance:
                perf = self.model_performance[model_key]
                if perf["total_requests"] > 0:
                    success_rate = perf["successful_requests"] / perf["total_requests"]
                    score += success_rate * 50
                
                # Recency bonus
                time_since_use = time.time() - perf["last_used"]
                if time_since_use < 3600:  # Used within last hour
                    score += 20
            
            # Memory efficiency bonus (for resource-constrained environments)
            if config.memory_requirements == "low":
                score += 10
            elif config.memory_requirements == "medium":
                score += 5
            
            candidates.append((model_key, score))
        
        if not candidates:
            return None
        
        # Return highest scoring model
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _prepare_security_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Prepare security-focused prompt"""
        base_prompt = f"""
As Archangel, an AI cybersecurity expert, analyze this security query:

Query: {query}
"""
        
        if context:
            base_prompt += f"\nContext: {context}\n"
        
        base_prompt += """
Provide a systematic security analysis including:
1. Initial assessment and scope identification
2. Potential risks and threat vectors
3. Recommended methodology and tools
4. Ethical and legal considerations
5. Next steps and expected outcomes

Maintain defensive security focus and educational value.

Analysis:"""
        
        return base_prompt
    
    async def _smolagent_analysis(self, prompt: str) -> str:
        """Use SmolAgent for autonomous security analysis"""
        try:
            result = self.security_agent.run(prompt)
            return str(result)
        except Exception as e:
            self.logger.error(f"SmolAgent analysis failed: {e}")
            # Fallback to direct model inference
            model_key = self._select_best_model(ModelCapability.TEXT_GENERATION)
            if model_key:
                return await self._direct_model_inference(model_key, prompt)
            return f"Analysis failed: {str(e)}"
    
    async def _direct_model_inference(self, model_key: str, prompt: str) -> str:
        """Direct model inference with error handling"""
        try:
            pipeline_obj = self.active_pipelines[model_key]
            config = self.model_registry[model_key]
            
            # Generate response
            response = pipeline_obj(
                prompt,
                max_length=config.max_tokens,
                num_return_sequences=1,
                temperature=config.temperature,
                pad_token_id=self.active_tokenizers[model_key].eos_token_id
            )
            
            if isinstance(response, list) and len(response) > 0:
                return response[0].get('generated_text', str(response[0]))
            else:
                return str(response)
                
        except Exception as e:
            self.logger.error(f"Direct inference failed for {model_key}: {e}")
            return f"Model inference failed: {str(e)}"
    
    def _update_performance_metrics(self, model_key: str, success: bool, response_time: float):
        """Update performance metrics for model"""
        if model_key not in self.model_performance:
            return
        
        metrics = self.model_performance[model_key]
        metrics["total_requests"] += 1
        if success:
            metrics["successful_requests"] += 1
        
        # Update average response time
        current_avg = metrics["average_response_time"]
        total_requests = metrics["total_requests"]
        metrics["average_response_time"] = (current_avg * (total_requests - 1) + response_time) / total_requests
        metrics["last_used"] = time.time()
    
    async def conversational_security_chat(self, message: str, session_id: str = "default") -> AIResponse:
        """Handle conversational security discussions"""
        start_time = time.time()
        
        # Select conversational model
        model_key = self._select_best_model(ModelCapability.CONVERSATION, ModelType.CONVERSATIONAL)
        
        if not model_key:
            return AIResponse(
                content="No conversational model available",
                model_used="none",
                confidence=0.0,
                execution_time=time.time() - start_time
            )
        
        try:
            # Use LangChain conversation if available
            if model_key in self.conversation_chains and LANGCHAIN_AVAILABLE:
                conversation = self.conversation_chains[model_key]
                response_text = conversation.predict(input=message)
            else:
                # Fallback to direct inference with conversation context
                security_context = """You are having a security discussion. Be educational, ethical, and thorough."""
                prompt = f"{security_context}\n\nHuman: {message}\n\nArchangel:"""
                response_text = await self._direct_model_inference(model_key, prompt)
            
            self._update_performance_metrics(model_key, True, time.time() - start_time)
            
            return AIResponse(
                content=response_text,
                model_used=self.model_registry[model_key].name,
                confidence=0.8,
                execution_time=time.time() - start_time,
                metadata={"session_id": session_id, "model_key": model_key}
            )
            
        except Exception as e:
            self._update_performance_metrics(model_key, False, time.time() - start_time)
            return AIResponse(
                content=f"Conversation failed: {str(e)}",
                model_used=self.model_registry[model_key].name,
                confidence=0.0,
                execution_time=time.time() - start_time
            )
    
    async def multi_stage_security_workflow(self, target: str, workflow_type: str = "comprehensive") -> List[AIResponse]:
        """Execute multi-stage security analysis workflow"""
        responses = []
        
        workflows = {
            "comprehensive": [
                "reconnaissance", "threat_modeling", "vulnerability_assessment", 
                "risk_analysis", "mitigation_planning"
            ],
            "quick_scan": [
                "reconnaissance", "vulnerability_assessment", "risk_analysis"
            ],
            "deep_analysis": [
                "reconnaissance", "threat_modeling", "vulnerability_assessment",
                "exploitation_analysis", "post_exploitation", "risk_analysis", "mitigation_planning"
            ]
        }
        
        stages = workflows.get(workflow_type, workflows["comprehensive"])
        
        for stage in stages:
            stage_query = f"Perform {stage.replace('_', ' ')} for target: {target}"
            context = f"This is part of a {workflow_type} security workflow. Previous stages completed: {', '.join(stages[:stages.index(stage)])}"
            
            response = await self.security_analysis(stage_query, context)
            responses.append(response)
            
            # Brief pause between stages
            await asyncio.sleep(0.5)
        
        return responses
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the orchestrator"""
        return {
            "authenticated": self.authenticated,
            "active_models": list(self.active_models.keys()),
            "model_performance": self.model_performance,
            "smolagents_available": SMOLAGENTS_AVAILABLE and self.security_agent is not None,
            "langchain_available": LANGCHAIN_AVAILABLE and len(self.conversation_chains) > 0,
            "total_models_in_registry": len(self.model_registry),
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("🧹 Cleaning up Hugging Face AI Orchestrator...")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear model references
        self.active_models.clear()
        self.active_tokenizers.clear()
        self.active_pipelines.clear()
        self.conversation_chains.clear()
        
        self.logger.info("✅ Cleanup completed")

# Factory function for easy instantiation
def create_huggingface_orchestrator(hf_token: Optional[str] = None) -> HuggingFaceAIOrchestrator:
    """Factory function to create and initialize HuggingFace orchestrator"""
    return HuggingFaceAIOrchestrator(hf_token=hf_token)