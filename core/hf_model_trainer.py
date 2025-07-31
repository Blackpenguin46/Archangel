#!/usr/bin/env python3
"""
Archangel Linux - HuggingFace Model Training Integration
Train custom security models using HuggingFace datasets and transformers
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# HuggingFace imports
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AutoModelForCausalLM, TrainingArguments, Trainer,
        DataCollatorWithPadding, DataCollatorForLanguageModeling
    )
    from datasets import Dataset, load_dataset, DatasetDict
    from huggingface_hub import HfApi, create_repo, upload_folder
    import torch
    from torch.utils.data import DataLoader
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    HF_TRAINING_AVAILABLE = True
except ImportError as e:
    HF_TRAINING_AVAILABLE = False
    logging.warning(f"HuggingFace training dependencies not available: {e}")

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_name: str
    task_type: str  # 'classification', 'generation', 'vulnerability_detection'
    dataset_name: str
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 500
    max_length: int = 512
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

@dataclass
class TrainingResult:
    """Results from model training"""
    model_path: str
    training_loss: float
    eval_loss: Optional[float]
    eval_metrics: Dict[str, float]
    training_time: float
    model_size_mb: float
    success: bool
    error_message: Optional[str] = None

class SecurityDatasetProcessor:
    """Process security datasets for training"""
    
    def __init__(self):
        self.vulnerability_labels = {
            'sql_injection': 0,
            'xss': 1,
            'csrf': 2,
            'rce': 3,
            'lfi': 4,
            'safe': 5
        }
    
    def process_vulnerability_dataset(self, dataset_name: str) -> Optional[DatasetDict]:
        """Process vulnerability detection dataset"""
        try:
            # Load dataset
            if dataset_name.startswith('local:'):
                # Load local dataset
                dataset_path = dataset_name.replace('local:', '')
                dataset = Dataset.from_json(dataset_path)
            else:
                # Load from HuggingFace Hub
                dataset = load_dataset(dataset_name)
            
            # Process for vulnerability classification
            def process_example(example):
                # Extract code and vulnerability type
                code = example.get('code', example.get('text', ''))
                vuln_type = example.get('vulnerability_type', example.get('label', 'safe'))
                
                return {
                    'text': code,
                    'label': self.vulnerability_labels.get(vuln_type, 5)  # Default to 'safe'
                }
            
            if isinstance(dataset, DatasetDict):
                processed = DatasetDict({
                    split: split_dataset.map(process_example)
                    for split, split_dataset in dataset.items()
                })
            else:
                processed = dataset.map(process_example)
                # Split into train/test if not already split
                processed = processed.train_test_split(test_size=0.2)
            
            return processed
            
        except Exception as e:
            logger.error(f"Failed to process dataset {dataset_name}: {e}")
            return None
    
    def create_security_qa_dataset(self, qa_pairs: List[Dict[str, str]]) -> Dataset:
        """Create Q&A dataset for security chatbot training"""
        def format_qa(example):
            return {
                'input_text': f"Question: {example['question']}",
                'target_text': f"Answer: {example['answer']}"
            }
        
        dataset = Dataset.from_list(qa_pairs)
        return dataset.map(format_qa)
    
    def augment_code_dataset(self, dataset: Dataset, augmentation_factor: int = 2) -> Dataset:
        """Augment code dataset with variations"""
        augmented_examples = []
        
        for example in dataset:
            # Original example
            augmented_examples.append(example)
            
            # Create variations
            for _ in range(augmentation_factor - 1):
                code = example['text']
                
                # Simple augmentations
                variations = [
                    code.replace('  ', '\t'),  # Change indentation
                    code.replace("'", '"'),    # Change quotes
                    code.replace('==', ' == '), # Add spaces
                    code.upper() if len(code) < 100 else code,  # Uppercase short code
                ]
                
                for variation in variations:
                    if variation != code:
                        aug_example = example.copy()
                        aug_example['text'] = variation
                        augmented_examples.append(aug_example)
                        break
        
        return Dataset.from_list(augmented_examples)

class HuggingFaceModelTrainer:
    """Train custom security models using HuggingFace"""
    
    def __init__(self, cache_dir: str = "models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_processor = SecurityDatasetProcessor()
        self.api = HfApi() if HF_TRAINING_AVAILABLE else None
    
    async def train_vulnerability_classifier(self, config: TrainingConfig) -> TrainingResult:
        """Train a vulnerability classification model"""
        if not HF_TRAINING_AVAILABLE:
            return TrainingResult(
                model_path="",
                training_loss=0.0,
                eval_loss=None,
                eval_metrics={},
                training_time=0.0,
                model_size_mb=0.0,
                success=False,
                error_message="HuggingFace training dependencies not available"
            )
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting vulnerability classifier training: {config.model_name}")
            
            # Load and process dataset
            dataset = self.dataset_processor.process_vulnerability_dataset(config.dataset_name)
            if not dataset:
                raise ValueError(f"Failed to load dataset: {config.dataset_name}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=len(self.dataset_processor.vulnerability_labels)
            )
            
            # Tokenize dataset
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    padding=True,
                    max_length=config.max_length
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=config.output_dir,
                num_train_epochs=config.num_epochs,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                warmup_steps=config.warmup_steps,
                weight_decay=config.weight_decay,
                logging_dir=f"{config.output_dir}/logs",
                logging_steps=100,
                save_steps=config.save_steps,
                eval_steps=config.eval_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                push_to_hub=config.push_to_hub,
                hub_model_id=config.hub_model_id,
                report_to=None  # Disable wandb/tensorboard
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # Metrics function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, predictions, average='weighted'
                )
                accuracy = accuracy_score(labels, predictions)
                
                return {
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                }
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset['train'],
                eval_dataset=tokenized_dataset['test'],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )
            
            # Train model
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Evaluate model
            logger.info("Evaluating model...")
            eval_result = trainer.evaluate()
            
            # Save model
            trainer.save_model()
            tokenizer.save_pretrained(config.output_dir)
            
            # Calculate model size
            model_size = sum(
                os.path.getsize(os.path.join(config.output_dir, f))
                for f in os.listdir(config.output_dir)
                if os.path.isfile(os.path.join(config.output_dir, f))
            ) / (1024 * 1024)  # Convert to MB
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Training completed successfully in {training_time:.2f}s")
            
            return TrainingResult(
                model_path=config.output_dir,
                training_loss=train_result.training_loss,
                eval_loss=eval_result.get('eval_loss'),
                eval_metrics=eval_result,
                training_time=training_time,
                model_size_mb=model_size,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                model_path="",
                training_loss=0.0,
                eval_loss=None,
                eval_metrics={},
                training_time=(datetime.now() - start_time).total_seconds(),
                model_size_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def train_security_chatbot(self, config: TrainingConfig, qa_dataset: List[Dict[str, str]]) -> TrainingResult:
        """Train a security-focused chatbot"""
        if not HF_TRAINING_AVAILABLE:
            return TrainingResult(
                model_path="",
                training_loss=0.0,
                eval_loss=None,
                eval_metrics={},
                training_time=0.0,
                model_size_mb=0.0,
                success=False,
                error_message="HuggingFace training dependencies not available"
            )
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting security chatbot training: {config.model_name}")
            
            # Create dataset
            dataset = self.dataset_processor.create_security_qa_dataset(qa_dataset)
            dataset = dataset.train_test_split(test_size=0.1)
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(config.model_name)
            
            # Tokenize dataset
            def tokenize_function(examples):
                inputs = examples['input_text']
                targets = examples['target_text']
                
                # Combine input and target for causal LM
                combined = [f"{inp} {tgt}" for inp, tgt in zip(inputs, targets)]
                
                return tokenizer(
                    combined,
                    truncation=True,
                    padding=True,
                    max_length=config.max_length
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=config.output_dir,
                num_train_epochs=config.num_epochs,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=config.batch_size,
                warmup_steps=config.warmup_steps,
                weight_decay=config.weight_decay,
                logging_dir=f"{config.output_dir}/logs",
                logging_steps=100,
                save_steps=config.save_steps,
                eval_steps=config.eval_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                push_to_hub=config.push_to_hub,
                hub_model_id=config.hub_model_id,
                report_to=None
            )
            
            # Data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False  # Causal LM, not masked LM
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset['train'],
                eval_dataset=tokenized_dataset['test'],
                tokenizer=tokenizer,
                data_collator=data_collator
            )
            
            # Train model
            logger.info("Starting chatbot training...")
            train_result = trainer.train()
            
            # Evaluate model
            eval_result = trainer.evaluate()
            
            # Save model
            trainer.save_model()
            tokenizer.save_pretrained(config.output_dir)
            
            # Calculate model size
            model_size = sum(
                os.path.getsize(os.path.join(config.output_dir, f))
                for f in os.listdir(config.output_dir)
                if os.path.isfile(os.path.join(config.output_dir, f))
            ) / (1024 * 1024)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Chatbot training completed in {training_time:.2f}s")
            
            return TrainingResult(
                model_path=config.output_dir,
                training_loss=train_result.training_loss,
                eval_loss=eval_result.get('eval_loss'),
                eval_metrics=eval_result,
                training_time=training_time,
                model_size_mb=model_size,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Chatbot training failed: {e}")
            return TrainingResult(
                model_path="",
                training_loss=0.0,
                eval_loss=None,
                eval_metrics={},
                training_time=(datetime.now() - start_time).total_seconds(),
                model_size_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def fine_tune_for_security(self, base_model: str, security_dataset: str, output_dir: str) -> TrainingResult:
        """Fine-tune a model specifically for security tasks"""
        config = TrainingConfig(
            model_name=base_model,
            task_type='classification',
            dataset_name=security_dataset,
            output_dir=output_dir,
            num_epochs=5,
            batch_size=16,
            learning_rate=1e-5,
            max_length=512
        )
        
        return await self.train_vulnerability_classifier(config)
    
    def get_recommended_models(self) -> Dict[str, List[str]]:
        """Get recommended models for different security tasks"""
        return {
            'vulnerability_detection': [
                'microsoft/codebert-base',
                'microsoft/graphcodebert-base',
                'huggingface/CodeBERTa-small-v1'
            ],
            'security_chatbot': [
                'microsoft/DialoGPT-medium',
                'facebook/blenderbot-400M-distill',
                'microsoft/DialoGPT-small'
            ],
            'code_analysis': [
                'microsoft/codebert-base',
                'Salesforce/codet5-base',
                'microsoft/unixcoder-base'
            ],
            'threat_intelligence': [
                'sentence-transformers/all-MiniLM-L6-v2',
                'sentence-transformers/all-mpnet-base-v2',
                'distilbert-base-uncased'
            ]
        }
    
    def get_recommended_datasets(self) -> Dict[str, List[str]]:
        """Get recommended datasets for security training"""
        return {
            'vulnerability_detection': [
                'code_x_glue_cc_defect_detection',
                'microsoft/CodeXGLUE',
                'local:data/vulnerability_samples.json'
            ],
            'security_qa': [
                'local:data/security_qa_pairs.json',
                'squad',  # Can be adapted for security
                'ms_marco'  # Can be adapted
            ],
            'malware_analysis': [
                'local:data/malware_samples.json',
                'drebin',  # If available
                'local:data/pe_analysis.json'
            ]
        }

# Global trainer instance
_model_trainer = None

def get_model_trainer() -> HuggingFaceModelTrainer:
    """Get global model trainer instance"""
    global _model_trainer
    if _model_trainer is None:
        _model_trainer = HuggingFaceModelTrainer()
    return _model_trainer