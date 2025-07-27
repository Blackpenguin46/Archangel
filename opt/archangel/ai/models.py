"""
AI Model Manager for Archangel

Manages AI models for LLM planner, analyzer, generator, and reporter components.
Handles model loading, inference, and resource management.
"""

import asyncio
import logging
import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of AI models supported"""
    LLM_PLANNER = "llm_planner"
    ANALYZER = "analyzer"
    GENERATOR = "generator"
    REPORTER = "reporter"
    CLASSIFIER = "classifier"
    EMBEDDER = "embedder"


@dataclass
class ModelConfig:
    """Configuration for an AI model"""
    name: str
    type: ModelType
    path: str
    device: str = "cpu"
    max_memory_mb: int = 1024
    max_tokens: int = 2048
    temperature: float = 0.7
    batch_size: int = 1
    enabled: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class InferenceRequest:
    """Request for model inference"""
    model_type: ModelType
    input_data: Union[str, Dict[str, Any], List[Any]]
    parameters: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    priority: int = 1  # 1=low, 2=normal, 3=high, 4=critical


@dataclass
class InferenceResult:
    """Result from model inference"""
    output: Any
    confidence: float
    processing_time_ms: float
    model_name: str
    metadata: Dict[str, Any] = None


class ModelInterface:
    """Base interface for AI models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.loaded = False
        self.model = None
        self.lock = threading.Lock()
        
    async def load(self) -> bool:
        """Load the model"""
        raise NotImplementedError
        
    async def unload(self):
        """Unload the model"""
        raise NotImplementedError
        
    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Perform inference"""
        raise NotImplementedError
        
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.loaded


class LLMPlannerModel(ModelInterface):
    """LLM model for operation planning and strategy"""
    
    async def load(self) -> bool:
        """Load the LLM planner model"""
        try:
            with self.lock:
                if self.loaded:
                    return True
                
                logger.info(f"Loading LLM planner model: {self.config.name}")
                
                # In a real implementation, this would load CodeLlama or similar
                # For now, we'll simulate the model loading
                self.model = {
                    'name': self.config.name,
                    'type': 'llm_planner',
                    'loaded': True
                }
                
                self.loaded = True
                logger.info(f"LLM planner model loaded: {self.config.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load LLM planner model: {e}")
            return False
    
    async def unload(self):
        """Unload the LLM planner model"""
        with self.lock:
            if self.loaded:
                self.model = None
                self.loaded = False
                logger.info(f"LLM planner model unloaded: {self.config.name}")
    
    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Perform LLM planning inference"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Simulate LLM planning
            input_text = request.input_data
            if isinstance(input_text, dict):
                input_text = json.dumps(input_text)
            
            # Mock planning output
            planning_output = {
                'objective': input_text,
                'phases': [
                    'reconnaissance',
                    'scanning',
                    'exploitation',
                    'post_exploitation',
                    'reporting'
                ],
                'strategy': 'comprehensive_assessment',
                'estimated_duration': '2-4 hours',
                'risk_level': 'medium'
            }
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return InferenceResult(
                output=planning_output,
                confidence=0.85,
                processing_time_ms=processing_time,
                model_name=self.config.name,
                metadata={'tokens_generated': 150}
            )
            
        except Exception as e:
            logger.error(f"LLM planning inference failed: {e}")
            raise


class AnalyzerModel(ModelInterface):
    """Model for analyzing security data and findings"""
    
    async def load(self) -> bool:
        """Load the analyzer model"""
        try:
            with self.lock:
                if self.loaded:
                    return True
                
                logger.info(f"Loading analyzer model: {self.config.name}")
                
                # Simulate Security-BERT or similar model loading
                self.model = {
                    'name': self.config.name,
                    'type': 'analyzer',
                    'loaded': True
                }
                
                self.loaded = True
                logger.info(f"Analyzer model loaded: {self.config.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load analyzer model: {e}")
            return False
    
    async def unload(self):
        """Unload the analyzer model"""
        with self.lock:
            if self.loaded:
                self.model = None
                self.loaded = False
                logger.info(f"Analyzer model unloaded: {self.config.name}")
    
    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Perform analysis inference"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Simulate security analysis
            analysis_output = {
                'severity': 'high',
                'category': 'vulnerability',
                'confidence': 0.92,
                'recommendations': [
                    'Apply security patches',
                    'Update configuration',
                    'Monitor for exploitation'
                ],
                'related_cves': ['CVE-2023-1234'],
                'attack_vectors': ['network', 'local']
            }
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return InferenceResult(
                output=analysis_output,
                confidence=0.92,
                processing_time_ms=processing_time,
                model_name=self.config.name
            )
            
        except Exception as e:
            logger.error(f"Analysis inference failed: {e}")
            raise


class GeneratorModel(ModelInterface):
    """Model for generating exploits, payloads, and scripts"""
    
    async def load(self) -> bool:
        """Load the generator model"""
        try:
            with self.lock:
                if self.loaded:
                    return True
                
                logger.info(f"Loading generator model: {self.config.name}")
                
                self.model = {
                    'name': self.config.name,
                    'type': 'generator',
                    'loaded': True
                }
                
                self.loaded = True
                logger.info(f"Generator model loaded: {self.config.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load generator model: {e}")
            return False
    
    async def unload(self):
        """Unload the generator model"""
        with self.lock:
            if self.loaded:
                self.model = None
                self.loaded = False
                logger.info(f"Generator model unloaded: {self.config.name}")
    
    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Perform generation inference"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Simulate code/exploit generation
            generation_output = {
                'type': 'exploit',
                'language': 'python',
                'code': '''
import socket
import struct

def exploit_target(host, port):
    # Generated exploit code
    payload = b"A" * 100
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.send(payload)
    sock.close()
''',
                'description': 'Buffer overflow exploit for target service',
                'safety_checks': True
            }
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return InferenceResult(
                output=generation_output,
                confidence=0.78,
                processing_time_ms=processing_time,
                model_name=self.config.name
            )
            
        except Exception as e:
            logger.error(f"Generation inference failed: {e}")
            raise


class ReporterModel(ModelInterface):
    """Model for generating security reports and documentation"""
    
    async def load(self) -> bool:
        """Load the reporter model"""
        try:
            with self.lock:
                if self.loaded:
                    return True
                
                logger.info(f"Loading reporter model: {self.config.name}")
                
                self.model = {
                    'name': self.config.name,
                    'type': 'reporter',
                    'loaded': True
                }
                
                self.loaded = True
                logger.info(f"Reporter model loaded: {self.config.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load reporter model: {e}")
            return False
    
    async def unload(self):
        """Unload the reporter model"""
        with self.lock:
            if self.loaded:
                self.model = None
                self.loaded = False
                logger.info(f"Reporter model unloaded: {self.config.name}")
    
    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Perform report generation inference"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Simulate report generation
            report_output = {
                'title': 'Security Assessment Report',
                'executive_summary': 'The assessment identified several critical vulnerabilities...',
                'findings': [
                    {
                        'title': 'SQL Injection Vulnerability',
                        'severity': 'Critical',
                        'description': 'The application is vulnerable to SQL injection attacks...',
                        'recommendation': 'Implement parameterized queries and input validation.'
                    }
                ],
                'recommendations': [
                    'Implement security patches',
                    'Conduct regular security assessments',
                    'Improve security monitoring'
                ],
                'format': 'markdown'
            }
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return InferenceResult(
                output=report_output,
                confidence=0.88,
                processing_time_ms=processing_time,
                model_name=self.config.name
            )
            
        except Exception as e:
            logger.error(f"Report generation inference failed: {e}")
            raise


class AIModelManager:
    """Manages all AI models for the Archangel system"""
    
    MODEL_CLASSES = {
        ModelType.LLM_PLANNER: LLMPlannerModel,
        ModelType.ANALYZER: AnalyzerModel,
        ModelType.GENERATOR: GeneratorModel,
        ModelType.REPORTER: ReporterModel,
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[ModelType, ModelInterface] = {}
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ai-model")
        self.inference_queue = asyncio.Queue(maxsize=100)
        self.stats = {
            'models_loaded': 0,
            'total_inferences': 0,
            'average_inference_time_ms': 0.0,
            'failed_inferences': 0,
        }
        
        logger.info("AIModelManager initialized")
    
    async def initialize(self):
        """Initialize the model manager and load configured models"""
        try:
            logger.info("Initializing AI models")
            
            # Load default model configurations
            default_configs = self._get_default_model_configs()
            
            # Merge with user configuration
            model_configs = {**default_configs, **self.config.get('models', {})}
            
            # Load each configured model
            for model_name, model_config in model_configs.items():
                await self._load_model(model_name, model_config)
            
            logger.info(f"AI model initialization complete. Loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            raise
    
    def _get_default_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default model configurations"""
        return {
            'llm_planner': {
                'type': 'llm_planner',
                'path': '/opt/archangel/models/codellama-13b',
                'device': 'cpu',
                'max_memory_mb': 2048,
                'enabled': True
            },
            'security_analyzer': {
                'type': 'analyzer',
                'path': '/opt/archangel/models/security-bert',
                'device': 'cpu',
                'max_memory_mb': 512,
                'enabled': True
            },
            'exploit_generator': {
                'type': 'generator',
                'path': '/opt/archangel/models/code-generator',
                'device': 'cpu',
                'max_memory_mb': 1024,
                'enabled': True
            },
            'report_generator': {
                'type': 'reporter',
                'path': '/opt/archangel/models/report-generator',
                'device': 'cpu',
                'max_memory_mb': 512,
                'enabled': True
            }
        }
    
    async def _load_model(self, model_name: str, model_config: Dict[str, Any]):
        """Load a specific model"""
        try:
            model_type = ModelType(model_config['type'])
            
            if not model_config.get('enabled', True):
                logger.info(f"Model {model_name} is disabled, skipping")
                return
            
            config = ModelConfig(
                name=model_name,
                type=model_type,
                path=model_config['path'],
                device=model_config.get('device', 'cpu'),
                max_memory_mb=model_config.get('max_memory_mb', 1024),
                max_tokens=model_config.get('max_tokens', 2048),
                temperature=model_config.get('temperature', 0.7),
                batch_size=model_config.get('batch_size', 1),
                enabled=model_config.get('enabled', True),
                metadata=model_config.get('metadata', {})
            )
            
            model_class = self.MODEL_CLASSES.get(model_type)
            if not model_class:
                logger.warning(f"Unknown model type: {model_type}")
                return
            
            model = model_class(config)
            
            # Load the model
            if await model.load():
                self.models[model_type] = model
                self.stats['models_loaded'] += 1
                logger.info(f"Successfully loaded model: {model_name} ({model_type.value})")
            else:
                logger.error(f"Failed to load model: {model_name}")
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
    
    async def infer(self, request: InferenceRequest) -> Optional[InferenceResult]:
        """Perform inference using the specified model type"""
        try:
            model = self.models.get(request.model_type)
            if not model:
                logger.error(f"Model not available: {request.model_type}")
                return None
            
            if not model.is_loaded():
                logger.error(f"Model not loaded: {request.model_type}")
                return None
            
            # Perform inference
            result = await model.infer(request)
            
            # Update statistics
            self.stats['total_inferences'] += 1
            self._update_inference_time_stats(result.processing_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed for {request.model_type}: {e}")
            self.stats['failed_inferences'] += 1
            return None
    
    async def get_model_info(self, model_type: ModelType) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        model = self.models.get(model_type)
        if not model:
            return None
        
        return {
            'name': model.config.name,
            'type': model.config.type.value,
            'loaded': model.is_loaded(),
            'device': model.config.device,
            'max_memory_mb': model.config.max_memory_mb,
            'max_tokens': model.config.max_tokens,
            'metadata': model.config.metadata
        }
    
    def get_available_models(self) -> List[ModelType]:
        """Get list of available model types"""
        return list(self.models.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model manager statistics"""
        return self.stats.copy()
    
    def _update_inference_time_stats(self, inference_time_ms: float):
        """Update average inference time statistics"""
        current_avg = self.stats['average_inference_time_ms']
        total_inferences = self.stats['total_inferences']
        
        if total_inferences == 1:
            self.stats['average_inference_time_ms'] = inference_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats['average_inference_time_ms'] = (
                alpha * inference_time_ms + (1 - alpha) * current_avg
            )
    
    async def cleanup(self):
        """Cleanup all models and resources"""
        logger.info("Cleaning up AI models")
        
        # Unload all models
        for model in self.models.values():
            await model.unload()
        
        self.models.clear()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("AI model cleanup complete")