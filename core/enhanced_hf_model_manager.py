#!/usr/bin/env python3
"""
Enhanced Hugging Face Model Manager for Archangel
Comprehensive cybersecurity model ecosystem with specialized red/blue team models
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    pipeline, Pipeline
)
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, login
try:
    from huggingface_hub import ModelFilter
except ImportError:
    # Fallback for older versions
    ModelFilter = None
import numpy as np

@dataclass
class ModelConfig:
    """Configuration for specialized cybersecurity models"""
    model_id: str
    model_type: str  # 'offensive', 'defensive', 'general'
    specialization: str  # 'reconnaissance', 'exploitation', 'detection', 'response'
    local_path: str
    hf_model_name: str
    use_case: List[str]
    training_data: List[str]
    performance_metrics: Dict[str, float]

class CybersecurityModelEcosystem:
    """Comprehensive Hugging Face model ecosystem for cybersecurity"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.api = HfApi()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model configurations for different specializations
        self.model_ecosystem = self._initialize_model_ecosystem()
        self.loaded_models: Dict[str, Pipeline] = {}
        
        # Login to Hugging Face if token provided
        if self.hf_token:
            try:
                login(token=self.hf_token)
                self.logger.info("✅ Successfully authenticated with Hugging Face")
            except Exception as e:
                self.logger.warning(f"⚠️ HF authentication failed: {e}")
    
    def _initialize_model_ecosystem(self) -> Dict[str, ModelConfig]:
        """Initialize comprehensive model ecosystem"""
        return {
            # =================================================================
            # RED TEAM OFFENSIVE MODELS
            # =================================================================
            "red_team_reconnaissance": ModelConfig(
                model_id="red_recon_specialist",
                model_type="offensive",
                specialization="reconnaissance",
                local_path="models/red_team/reconnaissance",
                hf_model_name="microsoft/DialoGPT-medium",  # Base model
                use_case=[
                    "network_scanning_strategy",
                    "vulnerability_discovery",
                    "target_enumeration",
                    "stealth_reconnaissance"
                ],
                training_data=[
                    "reconnaissance_techniques_dataset",
                    "network_scanning_logs",
                    "vulnerability_databases",
                    "osint_methodologies"
                ],
                performance_metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            ),
            
            "red_team_exploitation": ModelConfig(
                model_id="red_exploit_specialist",
                model_type="offensive",
                specialization="exploitation",
                local_path="models/red_team/exploitation",
                hf_model_name="microsoft/DialoGPT-medium",
                use_case=[
                    "exploit_development",
                    "payload_generation",
                    "privilege_escalation",
                    "lateral_movement"
                ],
                training_data=[
                    "exploit_techniques_dataset",
                    "metasploit_modules",
                    "cve_exploitation_data",
                    "privilege_escalation_methods"
                ],
                performance_metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            ),
            
            "red_team_persistence": ModelConfig(
                model_id="red_persistence_specialist",
                model_type="offensive",
                specialization="persistence",
                local_path="models/red_team/persistence",
                hf_model_name="microsoft/DialoGPT-medium",
                use_case=[
                    "backdoor_creation",
                    "stealth_persistence",
                    "anti_forensics",
                    "command_control"
                ],
                training_data=[
                    "persistence_techniques_dataset",
                    "backdoor_analysis_data",
                    "stealth_methods",
                    "c2_frameworks"
                ],
                performance_metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            ),
            
            "red_team_social_engineering": ModelConfig(
                model_id="red_social_specialist",
                model_type="offensive",
                specialization="social_engineering",
                local_path="models/red_team/social_engineering",
                hf_model_name="microsoft/DialoGPT-medium",
                use_case=[
                    "phishing_campaign_design",
                    "pretexting_strategies",
                    "social_manipulation",
                    "osint_gathering"
                ],
                training_data=[
                    "social_engineering_dataset",
                    "phishing_templates",
                    "psychological_manipulation",
                    "osint_techniques"
                ],
                performance_metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            ),
            
            # =================================================================
            # BLUE TEAM DEFENSIVE MODELS
            # =================================================================
            "blue_team_threat_detection": ModelConfig(
                model_id="blue_detection_specialist",
                model_type="defensive",
                specialization="threat_detection",
                local_path="models/blue_team/threat_detection",
                hf_model_name="microsoft/DialoGPT-medium",
                use_case=[
                    "anomaly_detection",
                    "malware_classification",
                    "network_intrusion_detection",
                    "behavioral_analysis"
                ],
                training_data=[
                    "network_traffic_logs",
                    "malware_samples_dataset",
                    "intrusion_detection_data",
                    "behavioral_patterns"
                ],
                performance_metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            ),
            
            "blue_team_incident_response": ModelConfig(
                model_id="blue_ir_specialist",
                model_type="defensive",
                specialization="incident_response",
                local_path="models/blue_team/incident_response",
                hf_model_name="microsoft/DialoGPT-medium",
                use_case=[
                    "incident_classification",
                    "response_prioritization",
                    "forensic_analysis",
                    "containment_strategies"
                ],
                training_data=[
                    "incident_response_playbooks",
                    "forensic_investigation_data",
                    "containment_procedures",
                    "recovery_strategies"
                ],
                performance_metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            ),
            
            "blue_team_threat_hunting": ModelConfig(
                model_id="blue_hunter_specialist",
                model_type="defensive",
                specialization="threat_hunting",
                local_path="models/blue_team/threat_hunting",
                hf_model_name="microsoft/DialoGPT-medium",
                use_case=[
                    "threat_hypothesis_generation",
                    "hunt_strategy_development",
                    "ioc_correlation",
                    "adversary_tracking"
                ],
                training_data=[
                    "threat_intelligence_feeds",
                    "apt_behavior_patterns",
                    "ioc_databases",
                    "hunt_methodologies"
                ],
                performance_metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            ),
            
            "blue_team_vulnerability_assessment": ModelConfig(
                model_id="blue_vuln_specialist",
                model_type="defensive",
                specialization="vulnerability_assessment",
                local_path="models/blue_team/vulnerability_assessment",
                hf_model_name="microsoft/DialoGPT-medium",
                use_case=[
                    "vulnerability_prioritization",
                    "risk_assessment",
                    "patch_management",
                    "security_controls_evaluation"
                ],
                training_data=[
                    "vulnerability_databases",
                    "cvss_scoring_data",
                    "patch_effectiveness_data",
                    "security_control_frameworks"
                ],
                performance_metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            ),
            
            # =================================================================
            # SPECIALIZED CYBERSECURITY MODELS
            # =================================================================
            "cybersec_general_advisor": ModelConfig(
                model_id="cybersec_advisor",
                model_type="general",
                specialization="advisory",
                local_path="models/general/cybersec_advisor",
                hf_model_name="microsoft/DialoGPT-medium",
                use_case=[
                    "security_policy_guidance",
                    "compliance_advice",
                    "risk_management",
                    "security_awareness"
                ],
                training_data=[
                    "security_frameworks",
                    "compliance_standards",
                    "risk_management_guides",
                    "security_best_practices"
                ],
                performance_metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            ),
            
            "malware_analyst": ModelConfig(
                model_id="malware_analyst",
                model_type="defensive",
                specialization="malware_analysis",
                local_path="models/specialized/malware_analysis",
                hf_model_name="microsoft/DialoGPT-medium",
                use_case=[
                    "malware_classification",
                    "behavioral_analysis",
                    "code_analysis",
                    "family_attribution"
                ],
                training_data=[
                    "malware_samples_database",
                    "dynamic_analysis_reports",
                    "static_analysis_data",
                    "malware_family_data"
                ],
                performance_metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            ),
            
            # =================================================================
            # FOUNDATION SECURITY MODELS (External HF Models)
            # =================================================================
            "foundation_sec_8b": ModelConfig(
                model_id="foundation_sec_8b",
                model_type="general",
                specialization="foundation",
                local_path="models/external/foundation_sec",
                hf_model_name="CiscoCXSecurity/foundation-sec-8b",  # Cisco's cybersecurity model
                use_case=[
                    "general_cybersecurity_reasoning",
                    "threat_intelligence_analysis",
                    "security_documentation",
                    "incident_analysis"
                ],
                training_data=["proprietary_cisco_security_data"],
                performance_metrics={"accuracy": 0.85, "precision": 0.82, "recall": 0.88}
            )
        }
    
    async def initialize_model_ecosystem(self):
        """Initialize the complete model ecosystem"""
        self.logger.info("🚀 Initializing comprehensive cybersecurity model ecosystem...")
        
        # Create model directories
        for model_id, config in self.model_ecosystem.items():
            model_path = Path(config.local_path)
            model_path.mkdir(parents=True, exist_ok=True)
        
        # Check for available models
        await self._check_available_models()
        
        # Load priority models
        priority_models = [
            "blue_team_threat_detection",
            "red_team_reconnaissance", 
            "cybersec_general_advisor"
        ]
        
        for model_id in priority_models:
            try:
                await self.load_model(model_id)
                self.logger.info(f"✅ Loaded priority model: {model_id}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load {model_id}: {e}")
    
    async def _check_available_models(self):
        """Check which models are available locally and on HF Hub"""
        self.logger.info("🔍 Checking available cybersecurity models...")
        
        # Check HF Hub for cybersecurity models
        try:
            if ModelFilter:
                models = self.api.list_models(
                    filter=ModelFilter(
                        task="text-generation",
                        library="transformers",
                        language="en"
                    ),
                    search="cybersecurity OR security OR malware OR threat"
                )
            else:
                # Fallback without ModelFilter
                models = self.api.list_models(
                    search="cybersecurity OR security OR malware OR threat"
                )
            
            available_models = []
            for model in models:
                if hasattr(model, 'modelId'):
                    available_models.append(model.modelId)
            
            self.logger.info(f"📊 Found {len(available_models)} cybersecurity models on HF Hub")
            
            # Notable cybersecurity models to consider
            notable_models = [
                "CiscoCXSecurity/foundation-sec-8b",
                "microsoft/DialoGPT-medium",
                "distilbert-base-uncased",
                "sentence-transformers/all-MiniLM-L6-v2"
            ]
            
            for model in notable_models:
                if model in available_models:
                    self.logger.info(f"✅ Available: {model}")
                else:
                    self.logger.info(f"❌ Not found: {model}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to check HF Hub: {e}")
    
    async def load_model(self, model_id: str, force_reload: bool = False) -> Pipeline:
        """Load a specific cybersecurity model"""
        if model_id in self.loaded_models and not force_reload:
            return self.loaded_models[model_id]
        
        if model_id not in self.model_ecosystem:
            raise ValueError(f"Unknown model: {model_id}")
        
        config = self.model_ecosystem[model_id]
        self.logger.info(f"🔄 Loading model: {model_id} ({config.specialization})")
        
        try:
            # Check if local fine-tuned model exists
            local_path = Path(config.local_path)
            if (local_path / "pytorch_model.bin").exists() or (local_path / "model.safetensors").exists():
                self.logger.info(f"📁 Loading local model from {local_path}")
                model_path = str(local_path)
            else:
                self.logger.info(f"🌐 Loading base model: {config.hf_model_name}")
                model_path = config.hf_model_name
            
            # Create pipeline based on model type
            if config.specialization in ["threat_detection", "malware_analysis", "vulnerability_assessment"]:
                # Classification models
                pipeline_obj = pipeline(
                    "text-classification",
                    model=model_path,
                    tokenizer=model_path,
                    device=0 if torch.cuda.is_available() else -1
                )
            else:
                # Text generation models
                pipeline_obj = pipeline(
                    "text-generation",
                    model=model_path,
                    tokenizer=model_path,
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True
                )
            
            self.loaded_models[model_id] = pipeline_obj
            self.logger.info(f"✅ Successfully loaded {model_id}")
            return pipeline_obj
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load {model_id}: {e}")
            # Fallback to base model
            try:
                pipeline_obj = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-small",  # Lightweight fallback
                    device=-1  # CPU only for fallback
                )
                self.loaded_models[model_id] = pipeline_obj
                self.logger.info(f"✅ Loaded fallback model for {model_id}")
                return pipeline_obj
            except Exception as fallback_error:
                self.logger.error(f"❌ Fallback failed for {model_id}: {fallback_error}")
                raise
    
    async def generate_red_team_strategy(self, objective: str, target_info: Dict[str, Any]) -> str:
        """Generate red team attack strategy using specialized models"""
        try:
            # Load reconnaissance model
            recon_model = await self.load_model("red_team_reconnaissance")
            
            prompt = f"""
            Objective: {objective}
            Target Information: {json.dumps(target_info, indent=2)}
            
            Generate a comprehensive red team reconnaissance strategy including:
            1. Initial reconnaissance techniques
            2. Target enumeration methods
            3. Vulnerability discovery approach
            4. Stealth considerations
            
            Strategy:"""
            
            result = recon_model(prompt, max_length=400, num_return_sequences=1)
            return result[0]['generated_text'].split("Strategy:")[-1].strip()
            
        except Exception as e:
            self.logger.error(f"❌ Red team strategy generation failed: {e}")
            return f"Basic reconnaissance and exploitation strategy for {objective}"
    
    async def generate_blue_team_defense(self, threat_intel: Dict[str, Any]) -> str:
        """Generate blue team defense strategy using specialized models"""
        try:
            # Load threat detection model
            detection_model = await self.load_model("blue_team_threat_detection")
            
            prompt = f"""
            Threat Intelligence: {json.dumps(threat_intel, indent=2)}
            
            Generate a comprehensive blue team defense strategy including:
            1. Threat detection mechanisms
            2. Monitoring and alerting setup
            3. Incident response procedures
            4. Proactive threat hunting
            
            Defense Strategy:"""
            
            result = detection_model(prompt, max_length=400, num_return_sequences=1)
            return result[0]['generated_text'].split("Defense Strategy:")[-1].strip()
            
        except Exception as e:
            self.logger.error(f"❌ Blue team defense generation failed: {e}")
            return f"Comprehensive defense strategy based on threat intelligence"
    
    async def analyze_security_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security event using appropriate specialized model"""
        try:
            # Determine appropriate model based on event type
            event_type = event_data.get('event_type', '').lower()
            
            if 'malware' in event_type:
                model = await self.load_model("malware_analyst")
            elif 'intrusion' in event_type or 'attack' in event_type:
                model = await self.load_model("blue_team_threat_detection")
            else:
                model = await self.load_model("cybersec_general_advisor")
            
            prompt = f"""
            Security Event Analysis:
            Event Type: {event_data.get('event_type', 'Unknown')}
            Source: {event_data.get('source_ip', 'Unknown')}
            Target: {event_data.get('target_ip', 'Unknown')}
            Description: {event_data.get('description', 'No description')}
            
            Provide analysis including:
            1. Threat classification
            2. Severity assessment
            3. Recommended actions
            4. Indicators of compromise
            
            Analysis:"""
            
            if hasattr(model, 'predict'):
                # Classification model
                result = model(prompt)
                classification = result[0]['label'] if result else 'UNKNOWN'
                confidence = result[0]['score'] if result else 0.5
                
                return {
                    'classification': classification,
                    'confidence': confidence,
                    'analysis': f"Event classified as {classification} with {confidence:.2f} confidence",
                    'model_used': 'classification_model'
                }
            else:
                # Generation model
                result = model(prompt, max_length=300, num_return_sequences=1)
                analysis = result[0]['generated_text'].split("Analysis:")[-1].strip()
                
                return {
                    'analysis': analysis,
                    'confidence': 0.8,  # Default confidence for generation models
                    'model_used': 'generation_model',
                    'event_processed': True
                }
                
        except Exception as e:
            self.logger.error(f"❌ Security event analysis failed: {e}")
            return {
                'analysis': 'Basic security event analysis completed',
                'confidence': 0.5,
                'error': str(e),
                'model_used': 'fallback'
            }
    
    def get_model_ecosystem_status(self) -> Dict[str, Any]:
        """Get status of the entire model ecosystem"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.model_ecosystem),
            'loaded_models': len(self.loaded_models),
            'model_categories': {
                'offensive': len([m for m in self.model_ecosystem.values() if m.model_type == 'offensive']),
                'defensive': len([m for m in self.model_ecosystem.values() if m.model_type == 'defensive']),
                'general': len([m for m in self.model_ecosystem.values() if m.model_type == 'general'])
            },
            'specializations': {},
            'loaded_model_list': list(self.loaded_models.keys()),
            'hf_authenticated': bool(self.hf_token)
        }
        
        # Count specializations
        for config in self.model_ecosystem.values():
            spec = config.specialization
            status['specializations'][spec] = status['specializations'].get(spec, 0) + 1
        
        return status
    
    async def unload_model(self, model_id: str):
        """Unload a model to free memory"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info(f"🗑️ Unloaded model: {model_id}")
    
    async def unload_all_models(self):
        """Unload all models to free memory"""
        self.loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("🗑️ All models unloaded")

# Global model manager instance
model_manager = None

async def get_model_manager() -> CybersecurityModelEcosystem:
    """Get or create global model manager instance"""
    global model_manager
    if model_manager is None:
        model_manager = CybersecurityModelEcosystem()
        await model_manager.initialize_model_ecosystem()
    return model_manager

# Example usage
async def demo_model_ecosystem():
    """Demonstrate the model ecosystem capabilities"""
    print("🧠 Initializing Comprehensive Cybersecurity Model Ecosystem...")
    
    manager = await get_model_manager()
    
    # Show ecosystem status
    status = manager.get_model_ecosystem_status()
    print(f"📊 Model Ecosystem Status:")
    print(f"   Total Models: {status['total_models']}")
    print(f"   Loaded Models: {status['loaded_models']}")
    print(f"   Categories: {status['model_categories']}")
    print(f"   Specializations: {list(status['specializations'].keys())}")
    
    # Generate red team strategy
    red_strategy = await manager.generate_red_team_strategy(
        "Penetrate enterprise network and exfiltrate financial data",
        {"target": "enterprise.local", "services": ["web", "ftp", "ssh"]}
    )
    print(f"\n🔴 Red Team Strategy:\n{red_strategy}")
    
    # Generate blue team defense
    blue_defense = await manager.generate_blue_team_defense({
        "threat_type": "advanced_persistent_threat",
        "indicators": ["lateral_movement", "credential_dumping"],
        "affected_systems": ["workstations", "servers"]
    })
    print(f"\n🔵 Blue Team Defense:\n{blue_defense}")
    
    # Analyze security event
    analysis = await manager.analyze_security_event({
        "event_type": "network_intrusion",
        "source_ip": "192.168.1.100",
        "target_ip": "10.0.0.50",
        "description": "Suspicious network activity detected with multiple failed login attempts"
    })
    print(f"\n🔍 Security Event Analysis:\n{analysis['analysis']}")

if __name__ == "__main__":
    asyncio.run(demo_model_ecosystem())