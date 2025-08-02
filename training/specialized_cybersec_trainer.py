#!/usr/bin/env python3
"""
Specialized Cybersecurity Model Trainer for Red and Blue Teams
Advanced training pipeline for offensive and defensive cybersecurity models
"""

import asyncio
import json
import logging
import os
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback, get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb  # For experiment tracking

@dataclass
class TrainingConfiguration:
    """Configuration for specialized cybersecurity model training"""
    
    # Model Configuration
    base_model: str = "microsoft/DialoGPT-medium"  # 345M params - good balance
    model_type: str = "red_team"  # red_team, blue_team, general
    specialization: str = "reconnaissance"  # specific specialization
    
    # Training Parameters
    output_dir: str = "./trained_models"
    max_length: int = 512
    batch_size: int = 2  # Optimized for M2 MacBook
    learning_rate: float = 2e-4
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # LoRA Configuration for efficient fine-tuning
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Data Configuration
    train_split: float = 0.8
    eval_split: float = 0.1
    test_split: float = 0.1
    
    # Hardware Configuration
    use_gpu: bool = torch.cuda.is_available()
    fp16: bool = False  # Disabled for MPS compatibility
    dataloader_num_workers: int = 2
    
    # Experiment Tracking
    use_wandb: bool = False
    project_name: str = "archangel-cybersec-models"

class CybersecurityDatasetBuilder:
    """Build specialized datasets for cybersecurity model training"""
    
    def __init__(self, config: TrainingConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.datasets_path = Path("data/training_datasets")
        self.datasets_path.mkdir(parents=True, exist_ok=True)
    
    async def build_red_team_datasets(self) -> Dict[str, Dataset]:
        """Build comprehensive red team training datasets"""
        self.logger.info("🔴 Building Red Team specialized datasets...")
        
        datasets = {}
        
        # Reconnaissance Dataset
        if self.config.specialization in ["reconnaissance", "all"]:
            recon_data = await self._build_reconnaissance_dataset()
            datasets["reconnaissance"] = recon_data
        
        # Exploitation Dataset
        if self.config.specialization in ["exploitation", "all"]:
            exploit_data = await self._build_exploitation_dataset()
            datasets["exploitation"] = exploit_data
        
        # Persistence Dataset
        if self.config.specialization in ["persistence", "all"]:
            persistence_data = await self._build_persistence_dataset()
            datasets["persistence"] = persistence_data
        
        # Social Engineering Dataset
        if self.config.specialization in ["social_engineering", "all"]:
            social_data = await self._build_social_engineering_dataset()
            datasets["social_engineering"] = social_data
        
        return datasets
    
    async def build_blue_team_datasets(self) -> Dict[str, Dataset]:
        """Build comprehensive blue team training datasets"""
        self.logger.info("🔵 Building Blue Team specialized datasets...")
        
        datasets = {}
        
        # Threat Detection Dataset
        if self.config.specialization in ["threat_detection", "all"]:
            detection_data = await self._build_threat_detection_dataset()
            datasets["threat_detection"] = detection_data
        
        # Incident Response Dataset
        if self.config.specialization in ["incident_response", "all"]:
            ir_data = await self._build_incident_response_dataset()
            datasets["incident_response"] = ir_data
        
        # Threat Hunting Dataset
        if self.config.specialization in ["threat_hunting", "all"]:
            hunting_data = await self._build_threat_hunting_dataset()
            datasets["threat_hunting"] = hunting_data
        
        # Vulnerability Assessment Dataset
        if self.config.specialization in ["vulnerability_assessment", "all"]:
            vuln_data = await self._build_vulnerability_assessment_dataset()
            datasets["vulnerability_assessment"] = vuln_data
        
        return datasets
    
    async def _build_reconnaissance_dataset(self) -> Dataset:
        """Build reconnaissance training dataset"""
        data = []
        
        # Network Scanning Techniques
        scanning_scenarios = [
            {
                "instruction": "Plan network reconnaissance for target enterprise.local",
                "input": "Target: enterprise.local, Scope: Internal network discovery",
                "output": "1. Passive DNS enumeration\n2. Port scanning with nmap -sS\n3. Service enumeration\n4. SNMP community string testing\n5. SMB null session testing"
            },
            {
                "instruction": "Design stealth reconnaissance approach",
                "input": "Target: high-security environment with IDS/IPS",
                "output": "1. Use slow scan timing (-T1)\n2. Fragment packets for evasion\n3. Randomize source ports\n4. Use decoy addresses\n5. Implement scan delays"
            },
            {
                "instruction": "Enumerate web application attack surface",
                "input": "Target: https://webapp.enterprise.local",
                "output": "1. Directory brute forcing with gobuster\n2. Parameter discovery with ffuf\n3. Technology stack identification\n4. Subdomain enumeration\n5. Certificate analysis"
            }
        ]
        
        # OSINT Gathering
        osint_scenarios = [
            {
                "instruction": "Gather OSINT on target organization",
                "input": "Company: Enterprise Corp, Industry: Finance",
                "output": "1. LinkedIn employee profiling\n2. Social media reconnaissance\n3. DNS record analysis\n4. Email format identification\n5. Technology stack research"
            },
            {
                "instruction": "Identify attack vectors through OSINT",
                "input": "Target: Enterprise Corp executive team",
                "output": "1. Social engineering targets identification\n2. Phishing campaign planning\n3. Credential stuffing possibilities\n4. Supply chain analysis\n5. Third-party service enumeration"
            }
        ]
        
        data.extend(scanning_scenarios + osint_scenarios)
        
        # Add realistic network scanning logs and responses
        for i, item in enumerate(data):
            item['id'] = f"recon_{i:04d}"
            item['category'] = "reconnaissance"
            item['difficulty'] = np.random.choice(["beginner", "intermediate", "advanced"])
            item['timestamp'] = datetime.now().isoformat()
        
        return Dataset.from_list(data)
    
    async def _build_exploitation_dataset(self) -> Dataset:
        """Build exploitation training dataset"""
        data = []
        
        # Vulnerability Exploitation
        exploit_scenarios = [
            {
                "instruction": "Exploit SQL injection vulnerability",
                "input": "Vulnerable parameter: /search?q=', SQL Server backend",
                "output": "1. Confirm injection with ' OR 1=1--\n2. Enumerate database structure\n3. Extract sensitive data with UNION\n4. Attempt privilege escalation\n5. Establish persistence if possible"
            },
            {
                "instruction": "Exploit buffer overflow vulnerability",
                "input": "Target: Legacy service on port 9999, Windows Server 2016",
                "output": "1. Fuzzing to identify crash point\n2. Control EIP register\n3. Generate shellcode payload\n4. Find JMP ESP address\n5. Achieve code execution"
            },
            {
                "instruction": "Exploit deserialization vulnerability",
                "input": "Java application with unsafe deserialization",
                "output": "1. Identify deserialization points\n2. Generate malicious payload with ysoserial\n3. Achieve remote code execution\n4. Establish reverse shell\n5. Privilege escalation attempts"
            }
        ]
        
        # Post-Exploitation Techniques
        post_exploit_scenarios = [
            {
                "instruction": "Perform privilege escalation on Windows",
                "input": "Current user: limited privileges, Windows 10 target",
                "output": "1. Enumerate system information\n2. Check for unquoted service paths\n3. Search for vulnerable services\n4. Exploit token privileges\n5. Achieve SYSTEM access"
            },
            {
                "instruction": "Establish lateral movement",
                "input": "Compromised workstation, Active Directory environment",
                "output": "1. Enumerate domain structure\n2. Dump credentials with Mimikatz\n3. Identify high-value targets\n4. Use Pass-the-Hash techniques\n5. Move to domain controllers"
            }
        ]
        
        data.extend(exploit_scenarios + post_exploit_scenarios)
        
        for i, item in enumerate(data):
            item['id'] = f"exploit_{i:04d}"
            item['category'] = "exploitation"
            item['difficulty'] = np.random.choice(["intermediate", "advanced", "expert"])
            item['timestamp'] = datetime.now().isoformat()
        
        return Dataset.from_list(data)
    
    async def _build_persistence_dataset(self) -> Dataset:
        """Build persistence training dataset"""
        data = []
        
        # Persistence Mechanisms
        persistence_scenarios = [
            {
                "instruction": "Establish Windows persistence",
                "input": "Compromised Windows 10 workstation, need stealth persistence",
                "output": "1. Create scheduled task with schtasks\n2. Registry run key modification\n3. WMI event subscription\n4. Service creation for persistence\n5. Startup folder placement"
            },
            {
                "instruction": "Linux persistence techniques",
                "input": "Compromised Linux server, maintain access",
                "output": "1. SSH key injection\n2. Cron job backdoor\n3. Init script modification\n4. Library hijacking\n5. Kernel module rootkit"
            },
            {
                "instruction": "Web application backdoor",
                "input": "Compromised web application, PHP backend",
                "output": "1. Web shell deployment\n2. Database trigger backdoor\n3. Include file poisoning\n4. .htaccess manipulation\n5. Session hijacking mechanism"
            }
        ]
        
        data.extend(persistence_scenarios)
        
        for i, item in enumerate(data):
            item['id'] = f"persistence_{i:04d}"
            item['category'] = "persistence"
            item['difficulty'] = np.random.choice(["intermediate", "advanced"])
            item['timestamp'] = datetime.now().isoformat()
        
        return Dataset.from_list(data)
    
    async def _build_social_engineering_dataset(self) -> Dataset:
        """Build social engineering training dataset"""
        data = []
        
        # Phishing Campaigns
        phishing_scenarios = [
            {
                "instruction": "Design targeted phishing campaign",
                "input": "Target: Finance department, Company: Enterprise Corp",
                "output": "1. Research finance team members\n2. Create convincing invoice email\n3. Host malicious attachment\n4. Set up credential harvesting site\n5. Track campaign effectiveness"
            },
            {
                "instruction": "Create spear phishing attack",
                "input": "Target: CEO, Industry: Healthcare",
                "output": "1. OSINT on CEO's interests\n2. Craft personalized message\n3. Use trusted sender spoofing\n4. Include business-relevant attachment\n5. Set up C2 infrastructure"
            }
        ]
        
        # Pretexting Scenarios
        pretext_scenarios = [
            {
                "instruction": "Execute vishing attack",
                "input": "Target: IT helpdesk, Goal: Password reset",
                "output": "1. Research company structure\n2. Impersonate executive assistant\n3. Create urgent business scenario\n4. Request password reset\n5. Maintain believable conversation"
            },
            {
                "instruction": "Physical security bypass",
                "input": "Target: Office building, Goal: Badge cloning",
                "output": "1. Reconnaissance of entry points\n2. Tailgating technique\n3. Badge cloning with Proxmark\n4. Authority impersonation\n5. Information gathering"
            }
        ]
        
        data.extend(phishing_scenarios + pretext_scenarios)
        
        for i, item in enumerate(data):
            item['id'] = f"social_{i:04d}"
            item['category'] = "social_engineering"
            item['difficulty'] = np.random.choice(["beginner", "intermediate", "advanced"])
            item['timestamp'] = datetime.now().isoformat()
        
        return Dataset.from_list(data)
    
    async def _build_threat_detection_dataset(self) -> Dataset:
        """Build threat detection training dataset"""
        data = []
        
        # Network Anomaly Detection
        detection_scenarios = [
            {
                "instruction": "Analyze suspicious network traffic",
                "input": "Unusual outbound traffic to 185.220.101.42 on port 443, 2GB transferred",
                "output": "THREAT DETECTED: Data exfiltration activity\nSeverity: HIGH\nRecommendation: Block traffic, investigate endpoint\nIOCs: IP 185.220.101.42, large data transfer, non-business hours"
            },
            {
                "instruction": "Identify lateral movement indicators",
                "input": "Multiple failed RDP attempts from 10.0.1.50 to various servers",
                "output": "THREAT DETECTED: Lateral movement attempt\nSeverity: HIGH\nRecommendation: Isolate source host, reset credentials\nIOCs: RDP brute force, multiple targets, credential stuffing"
            },
            {
                "instruction": "Detect malware communication",
                "input": "Periodic DNS queries to randomized subdomains of suspicious-domain.com",
                "output": "THREAT DETECTED: Domain Generation Algorithm (DGA) activity\nSeverity: MEDIUM\nRecommendation: Block domain, scan affected host\nIOCs: DGA pattern, C2 communication, malware infection"
            }
        ]
        
        # Behavioral Analysis
        behavioral_scenarios = [
            {
                "instruction": "Analyze user behavior anomaly",
                "input": "User accessing files outside normal patterns, 3 AM login, VPN from new location",
                "output": "ANOMALY DETECTED: Unusual user behavior\nSeverity: MEDIUM\nRecommendation: Verify user identity, monitor closely\nIndicators: Time anomaly, location anomaly, access pattern change"
            }
        ]
        
        data.extend(detection_scenarios + behavioral_scenarios)
        
        for i, item in enumerate(data):
            item['id'] = f"detection_{i:04d}"
            item['category'] = "threat_detection"
            item['severity'] = np.random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
            item['timestamp'] = datetime.now().isoformat()
        
        return Dataset.from_list(data)
    
    async def _build_incident_response_dataset(self) -> Dataset:
        """Build incident response training dataset"""
        data = []
        
        # Incident Classification
        ir_scenarios = [
            {
                "instruction": "Respond to ransomware incident",
                "input": "Multiple encrypted files, ransom note, systems unavailable",
                "output": "1. IMMEDIATE: Isolate affected systems\n2. Activate incident response team\n3. Preserve forensic evidence\n4. Assess backup integrity\n5. Coordinate with legal/PR teams\n6. Begin recovery procedures"
            },
            {
                "instruction": "Handle data breach incident",
                "input": "Unauthorized access to customer database, 10,000 records compromised",
                "output": "1. Contain the breach immediately\n2. Assess scope of data compromised\n3. Notify relevant authorities (GDPR/CCPA)\n4. Prepare customer notifications\n5. Conduct forensic investigation\n6. Implement additional controls"
            },
            {
                "instruction": "Respond to insider threat",
                "input": "Employee accessing unauthorized files, unusual data downloads",
                "output": "1. Discrete monitoring activation\n2. Preserve evidence of activity\n3. Coordinate with HR/Legal\n4. Assess data exposure risk\n5. Plan confrontation strategy\n6. Implement containment measures"
            }
        ]
        
        data.extend(ir_scenarios)
        
        for i, item in enumerate(data):
            item['id'] = f"ir_{i:04d}"
            item['category'] = "incident_response"
            item['priority'] = np.random.choice(["P1-CRITICAL", "P2-HIGH", "P3-MEDIUM", "P4-LOW"])
            item['timestamp'] = datetime.now().isoformat()
        
        return Dataset.from_list(data)
    
    async def _build_threat_hunting_dataset(self) -> Dataset:
        """Build threat hunting training dataset"""
        data = []
        
        # Hunting Hypotheses
        hunting_scenarios = [
            {
                "instruction": "Hunt for APT29 indicators",
                "input": "Intelligence suggests APT29 activity, look for CozyBear TTPs",
                "output": "1. Search for PowerShell obfuscation\n2. Look for WMI persistence\n3. Check for DNS tunneling\n4. Analyze memory artifacts\n5. Correlate with known IOCs"
            },
            {
                "instruction": "Hunt for insider threats",
                "input": "Unusual data access patterns, potential insider activity",
                "output": "1. Analyze file access logs\n2. Correlate with HR data\n3. Check for USB activity\n4. Monitor email patterns\n5. Review authentication logs"
            }
        ]
        
        data.extend(hunting_scenarios)
        
        for i, item in enumerate(data):
            item['id'] = f"hunt_{i:04d}"
            item['category'] = "threat_hunting"
            item['confidence'] = np.random.uniform(0.6, 0.95)
            item['timestamp'] = datetime.now().isoformat()
        
        return Dataset.from_list(data)
    
    async def _build_vulnerability_assessment_dataset(self) -> Dataset:
        """Build vulnerability assessment training dataset"""
        data = []
        
        # Vulnerability Analysis
        vuln_scenarios = [
            {
                "instruction": "Assess critical vulnerability impact",
                "input": "CVE-2023-12345: Remote code execution in web server, CVSS 9.8",
                "output": "CRITICAL PRIORITY\nImpact: Complete system compromise\nExploitability: High (public exploits available)\nRecommendation: Emergency patching required\nMitigation: WAF rules, network segmentation"
            },
            {
                "instruction": "Prioritize vulnerability remediation",
                "input": "50 vulnerabilities identified, limited maintenance window",
                "output": "Priority 1: RCE vulnerabilities (CVE-2023-xxx)\nPriority 2: Privilege escalation (CVE-2023-yyy)\nPriority 3: Information disclosure\nPriority 4: DoS vulnerabilities"
            }
        ]
        
        data.extend(vuln_scenarios)
        
        for i, item in enumerate(data):
            item['id'] = f"vuln_{i:04d}"
            item['category'] = "vulnerability_assessment"
            item['cvss_score'] = np.random.uniform(3.0, 10.0)
            item['timestamp'] = datetime.now().isoformat()
        
        return Dataset.from_list(data)

class SpecializedCybersecurityTrainer:
    """Advanced trainer for specialized cybersecurity models"""
    
    def __init__(self, config: TrainingConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.dataset_builder = CybersecurityDatasetBuilder(config)
        self.tokenizer = None
        self.model = None
        
        # Initialize experiment tracking
        if self.config.use_wandb:
            wandb.init(
                project=self.config.project_name,
                config=asdict(self.config),
                name=f"{self.config.model_type}_{self.config.specialization}"
            )
    
    async def initialize_model_and_tokenizer(self):
        """Initialize model and tokenizer for training"""
        self.logger.info(f"🚀 Initializing {self.config.base_model} for {self.config.specialization}")
        
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.config.use_gpu else torch.float32
            )
            
            # Apply LoRA if configured
            if self.config.use_lora:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.lora_target_modules or ["q_proj", "v_proj"]
                )
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
            
            self.logger.info("✅ Model and tokenizer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model: {e}")
            return False
    
    async def prepare_training_data(self) -> DatasetDict:
        """Prepare comprehensive training data"""
        self.logger.info(f"📊 Preparing {self.config.model_type} training data...")
        
        if self.config.model_type == "red_team":
            datasets = await self.dataset_builder.build_red_team_datasets()
        elif self.config.model_type == "blue_team":
            datasets = await self.dataset_builder.build_blue_team_datasets()
        else:
            # General cybersecurity model
            red_datasets = await self.dataset_builder.build_red_team_datasets()
            blue_datasets = await self.dataset_builder.build_blue_team_datasets()
            datasets = {**red_datasets, **blue_datasets}
        
        # Combine all datasets
        all_data = []
        for dataset_name, dataset in datasets.items():
            for item in dataset:
                all_data.append(item)
        
        # Create comprehensive dataset
        full_dataset = Dataset.from_list(all_data)
        
        # Split dataset
        train_size = int(len(full_dataset) * self.config.train_split)
        eval_size = int(len(full_dataset) * self.config.eval_split)
        
        dataset_split = full_dataset.train_test_split(
            train_size=train_size,
            test_size=len(full_dataset) - train_size
        )
        
        eval_test = dataset_split['test'].train_test_split(
            train_size=eval_size,
            test_size=len(dataset_split['test']) - eval_size
        )
        
        final_dataset = DatasetDict({
            'train': dataset_split['train'],
            'validation': eval_test['train'],
            'test': eval_test['test']
        })
        
        self.logger.info(f"📊 Dataset prepared: Train={len(final_dataset['train'])}, "
                        f"Val={len(final_dataset['validation'])}, Test={len(final_dataset['test'])}")
        
        return final_dataset
    
    def tokenize_function(self, examples):
        """Tokenize training examples"""
        # Format as instruction-following
        formatted_texts = []
        for i in range(len(examples['instruction'])):
            text = f"Instruction: {examples['instruction'][i]}\n"
            if 'input' in examples and examples['input'][i]:
                text += f"Input: {examples['input'][i]}\n"
            text += f"Response: {examples['output'][i]}"
            formatted_texts.append(text)
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    async def train_specialized_model(self):
        """Train the specialized cybersecurity model"""
        self.logger.info(f"🎯 Starting {self.config.model_type} {self.config.specialization} training...")
        
        # Initialize model and tokenizer
        if not await self.initialize_model_and_tokenizer():
            raise RuntimeError("Failed to initialize model and tokenizer")
        
        # Prepare training data
        dataset = await self.prepare_training_data()
        
        # Tokenize datasets
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        # Create output directory
        output_dir = Path(self.config.output_dir) / f"{self.config.model_type}_{self.config.specialization}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_dir=str(output_dir / "logs"),
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=self.config.dataloader_num_workers,
            fp16=self.config.fp16,
            report_to="wandb" if self.config.use_wandb else None,
            run_name=f"{self.config.model_type}_{self.config.specialization}"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        self.logger.info("🚀 Beginning model training...")
        training_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Evaluate on test set
        test_results = trainer.evaluate(tokenized_dataset['test'])
        
        # Save training results
        results = {
            'training_result': training_result.metrics if hasattr(training_result, 'metrics') else {},
            'test_results': test_results,
            'model_config': asdict(self.config),
            'dataset_stats': {
                'train_size': len(tokenized_dataset['train']),
                'eval_size': len(tokenized_dataset['validation']),
                'test_size': len(tokenized_dataset['test'])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"✅ Training completed! Model saved to {output_dir}")
        self.logger.info(f"📊 Final eval loss: {test_results.get('eval_loss', 'N/A')}")
        
        return results

# Training orchestrator
async def train_cybersecurity_models():
    """Train all specialized cybersecurity models"""
    print("🧠 Starting Comprehensive Cybersecurity Model Training...")
    
    # Red Team Models
    red_team_specializations = ["reconnaissance", "exploitation", "persistence", "social_engineering"]
    
    for specialization in red_team_specializations:
        config = TrainingConfiguration(
            model_type="red_team",
            specialization=specialization,
            base_model="microsoft/DialoGPT-medium",
            num_epochs=3,  # Quick training for demo
            batch_size=1   # Memory efficient
        )
        
        trainer = SpecializedCybersecurityTrainer(config)
        
        try:
            results = await trainer.train_specialized_model()
            print(f"✅ Completed Red Team {specialization} training")
        except Exception as e:
            print(f"❌ Failed Red Team {specialization} training: {e}")
    
    # Blue Team Models
    blue_team_specializations = ["threat_detection", "incident_response", "threat_hunting", "vulnerability_assessment"]
    
    for specialization in blue_team_specializations:
        config = TrainingConfiguration(
            model_type="blue_team",
            specialization=specialization,
            base_model="microsoft/DialoGPT-medium",
            num_epochs=3,
            batch_size=1
        )
        
        trainer = SpecializedCybersecurityTrainer(config)
        
        try:
            results = await trainer.train_specialized_model()
            print(f"✅ Completed Blue Team {specialization} training")
        except Exception as e:
            print(f"❌ Failed Blue Team {specialization} training: {e}")
    
    print("🎉 Cybersecurity model training pipeline completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(train_cybersecurity_models())