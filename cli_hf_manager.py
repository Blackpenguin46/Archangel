#!/usr/bin/env python3
"""
Archangel Linux - HuggingFace CLI Manager
Command-line interface for managing HuggingFace models and training
"""

import asyncio
import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Rich imports for beautiful CLI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Core imports
from core.session_manager import get_session_manager, InteractionMode
from core.hf_model_trainer import get_model_trainer, TrainingConfig

console = Console() if RICH_AVAILABLE else None

class HuggingFaceCLI:
    """CLI for HuggingFace model management"""
    
    def __init__(self):
        self.session_manager = get_session_manager()
        self.model_trainer = get_model_trainer()
        self.session_id = None
    
    def print_message(self, message: str, style: str = "white"):
        """Print message with or without rich formatting"""
        if RICH_AVAILABLE and console:
            console.print(message, style=style)
        else:
            print(message)
    
    def print_panel(self, content: str, title: str, style: str = "blue"):
        """Print panel with or without rich formatting"""
        if RICH_AVAILABLE and console:
            console.print(Panel(content, title=title, border_style=style))
        else:
            print(f"\n=== {title} ===")
            print(content)
            print("=" * (len(title) + 8))
    
    async def authenticate(self, token: str = None) -> bool:
        """Authenticate with HuggingFace"""
        if not token:
            # Try to get token from environment
            token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
            
            if not token:
                if RICH_AVAILABLE:
                    token = Prompt.ask("Enter your HuggingFace token", password=True)
                else:
                    token = input("Enter your HuggingFace token: ")
        
        if not token:
            self.print_message("❌ No token provided", "red")
            return False
        
        # Create session if not exists
        if not self.session_id:
            self.session_id = self.session_manager.create_session("hf_cli_user")
        
        # Authenticate
        success = await self.session_manager.authenticate_huggingface(self.session_id, token)
        
        if success:
            self.print_message("✅ Successfully authenticated with HuggingFace", "green")
            return True
        else:
            self.print_message("❌ Authentication failed", "red")
            return False
    
    async def list_models(self, task: str = None):
        """List available HuggingFace models"""
        if not self.session_id:
            self.print_message("❌ Please authenticate first", "red")
            return
        
        self.print_message("🔍 Fetching available models...", "blue")
        
        models = await self.session_manager.get_available_models(self.session_id, task)
        
        if not models:
            self.print_message("❌ No models found or authentication required", "red")
            return
        
        if RICH_AVAILABLE and console:
            table = Table(title=f"Available Models{f' for {task}' if task else ''}")
            table.add_column("Model ID", style="cyan")
            table.add_column("Downloads", style="green")
            table.add_column("Pipeline", style="yellow")
            table.add_column("Library", style="magenta")
            
            for model in models[:10]:  # Show top 10
                table.add_row(
                    model['id'],
                    str(model.get('downloads', 'N/A')),
                    model.get('pipeline_tag', 'N/A'),
                    model.get('library_name', 'N/A')
                )
            
            console.print(table)
        else:
            print(f"\nAvailable Models{f' for {task}' if task else ''}:")
            print("-" * 50)
            for i, model in enumerate(models[:10], 1):
                print(f"{i}. {model['id']}")
                print(f"   Downloads: {model.get('downloads', 'N/A')}")
                print(f"   Pipeline: {model.get('pipeline_tag', 'N/A')}")
                print(f"   Library: {model.get('library_name', 'N/A')}")
                print()
    
    async def show_recommended_models(self):
        """Show recommended models for security tasks"""
        recommended = self.model_trainer.get_recommended_models()
        
        if RICH_AVAILABLE and console:
            for task, models in recommended.items():
                table = Table(title=f"Recommended Models for {task.replace('_', ' ').title()}")
                table.add_column("Model", style="cyan")
                table.add_column("Description", style="white")
                
                descriptions = {
                    'microsoft/codebert-base': 'Pre-trained model for code understanding',
                    'microsoft/graphcodebert-base': 'Graph-based code representation',
                    'microsoft/DialoGPT-medium': 'Conversational AI model',
                    'sentence-transformers/all-MiniLM-L6-v2': 'Sentence embeddings',
                    'huggingface/CodeBERTa-small-v1': 'Lightweight code analysis'
                }
                
                for model in models:
                    table.add_row(
                        model,
                        descriptions.get(model, 'Specialized model for security tasks')
                    )
                
                console.print(table)
                console.print()
        else:
            for task, models in recommended.items():
                print(f"\n{task.replace('_', ' ').title()}:")
                print("-" * 30)
                for model in models:
                    print(f"  • {model}")
    
    async def show_recommended_datasets(self):
        """Show recommended datasets for security training"""
        recommended = self.model_trainer.get_recommended_datasets()
        
        self.print_panel(
            "Recommended datasets for training security models",
            "Security Datasets",
            "green"
        )
        
        for task, datasets in recommended.items():
            self.print_message(f"\n📊 {task.replace('_', ' ').title()}:", "cyan")
            for dataset in datasets:
                if dataset.startswith('local:'):
                    self.print_message(f"  • {dataset} (local dataset)", "yellow")
                else:
                    self.print_message(f"  • {dataset}", "white")
    
    async def train_model(self, model_name: str, dataset_name: str, task_type: str, output_dir: str):
        """Train a custom security model"""
        if not self.session_id:
            self.print_message("❌ Please authenticate first", "red")
            return
        
        self.print_panel(
            f"Starting training for {task_type} task\n"
            f"Base Model: {model_name}\n"
            f"Dataset: {dataset_name}\n"
            f"Output: {output_dir}",
            "Model Training",
            "blue"
        )
        
        # Create training config
        config = TrainingConfig(
            model_name=model_name,
            task_type=task_type,
            dataset_name=dataset_name,
            output_dir=output_dir,
            num_epochs=3,
            batch_size=8,
            learning_rate=2e-5
        )
        
        # Start training with progress
        if RICH_AVAILABLE and console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Training model...", total=100)
                
                if task_type == 'classification':
                    result = await self.model_trainer.train_vulnerability_classifier(config)
                else:
                    # For now, default to classification
                    result = await self.model_trainer.train_vulnerability_classifier(config)
                
                progress.update(task, completed=100)
        else:
            self.print_message("🚀 Training started...", "blue")
            if task_type == 'classification':
                result = await self.model_trainer.train_vulnerability_classifier(config)
            else:
                result = await self.model_trainer.train_vulnerability_classifier(config)
        
        # Show results
        if result.success:
            self.print_panel(
                f"✅ Training completed successfully!\n\n"
                f"Model saved to: {result.model_path}\n"
                f"Training loss: {result.training_loss:.4f}\n"
                f"Evaluation loss: {result.eval_loss:.4f if result.eval_loss else 'N/A'}\n"
                f"Training time: {result.training_time:.2f}s\n"
                f"Model size: {result.model_size_mb:.2f} MB\n\n"
                f"Evaluation metrics:\n" +
                "\n".join([f"  {k}: {v:.4f}" for k, v in result.eval_metrics.items()]),
                "Training Results",
                "green"
            )
        else:
            self.print_panel(
                f"❌ Training failed!\n\n"
                f"Error: {result.error_message}\n"
                f"Training time: {result.training_time:.2f}s",
                "Training Results",
                "red"
            )
    
    async def create_security_dataset(self, output_file: str):
        """Create a sample security dataset"""
        self.print_message("📝 Creating sample security dataset...", "blue")
        
        # Sample security Q&A pairs
        security_qa = [
            {
                "question": "What is SQL injection?",
                "answer": "SQL injection is a code injection technique that exploits vulnerabilities in an application's software when user input is not properly sanitized before being included in SQL queries."
            },
            {
                "question": "How can I prevent XSS attacks?",
                "answer": "To prevent XSS attacks: 1) Validate and sanitize all user input, 2) Use output encoding, 3) Implement Content Security Policy (CSP), 4) Use secure coding practices."
            },
            {
                "question": "What is CSRF?",
                "answer": "Cross-Site Request Forgery (CSRF) is an attack that forces an end user to execute unwanted actions on a web application in which they're currently authenticated."
            },
            {
                "question": "How do I secure API endpoints?",
                "answer": "Secure API endpoints by: 1) Using authentication and authorization, 2) Implementing rate limiting, 3) Validating input, 4) Using HTTPS, 5) Implementing proper error handling."
            },
            {
                "question": "What is a buffer overflow?",
                "answer": "A buffer overflow occurs when a program writes more data to a buffer than it can hold, potentially overwriting adjacent memory and leading to crashes or security vulnerabilities."
            }
        ]
        
        # Sample vulnerability code examples
        vulnerability_samples = [
            {
                "code": "SELECT * FROM users WHERE id = " + user_input,
                "vulnerability_type": "sql_injection",
                "severity": "high"
            },
            {
                "code": "document.innerHTML = user_input",
                "vulnerability_type": "xss",
                "severity": "medium"
            },
            {
                "code": "os.system('rm ' + filename)",
                "vulnerability_type": "command_injection",
                "severity": "high"
            },
            {
                "code": "if (password == 'admin123'):",
                "vulnerability_type": "weak_authentication",
                "severity": "high"
            },
            {
                "code": "def secure_function(data): return hashlib.md5(data).hexdigest()",
                "vulnerability_type": "weak_crypto",
                "severity": "medium"
            }
        ]
        
        # Create dataset structure
        dataset = {
            "metadata": {
                "name": "Archangel Security Dataset",
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "description": "Sample security dataset for training Archangel models"
            },
            "qa_pairs": security_qa,
            "vulnerability_samples": vulnerability_samples
        }
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        self.print_message(f"✅ Dataset created: {output_file}", "green")
        self.print_message(f"   Q&A pairs: {len(security_qa)}", "white")
        self.print_message(f"   Vulnerability samples: {len(vulnerability_samples)}", "white")
    
    async def interactive_mode(self):
        """Interactive mode for HuggingFace management"""
        self.print_panel(
            "🤗 **HuggingFace Model Manager**\n\n"
            "Manage models, datasets, and training for Archangel security AI.\n"
            "Type 'help' for available commands or 'quit' to exit.",
            "Archangel HF Manager",
            "blue"
        )
        
        # Try to authenticate automatically
        if os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN'):
            await self.authenticate()
        
        while True:
            try:
                if RICH_AVAILABLE:
                    command = Prompt.ask("\n[bold cyan]hf>[/bold cyan]").strip()
                else:
                    command = input("\nhf> ").strip()
                
                if command.lower() in ['quit', 'exit']:
                    self.print_message("👋 Goodbye!", "green")
                    break
                
                elif command.lower() == 'help':
                    self.show_help()
                
                elif command.lower() == 'auth':
                    await self.authenticate()
                
                elif command.lower() == 'models':
                    await self.list_models()
                
                elif command.lower() == 'recommended':
                    await self.show_recommended_models()
                
                elif command.lower() == 'datasets':
                    await self.show_recommended_datasets()
                
                elif command.lower().startswith('models '):
                    task = command.split(' ', 1)[1]
                    await self.list_models(task)
                
                elif command.lower() == 'create-dataset':
                    output_file = "data/security_dataset.json"
                    if RICH_AVAILABLE:
                        output_file = Prompt.ask("Output file", default=output_file)
                    await self.create_security_dataset(output_file)
                
                elif command.lower().startswith('train'):
                    await self.handle_train_command(command)
                
                elif command.lower() == 'status':
                    await self.show_status()
                
                elif command == '':
                    continue
                
                else:
                    self.print_message(f"❌ Unknown command: {command}", "red")
                    self.print_message("Type 'help' for available commands", "yellow")
            
            except KeyboardInterrupt:
                self.print_message("\n👋 Goodbye!", "green")
                break
            except Exception as e:
                self.print_message(f"❌ Error: {e}", "red")
    
    async def handle_train_command(self, command: str):
        """Handle train command with parameters"""
        parts = command.split()
        if len(parts) < 4:
            self.print_message("❌ Usage: train <model> <dataset> <task_type> [output_dir]", "red")
            return
        
        model_name = parts[1]
        dataset_name = parts[2]
        task_type = parts[3]
        output_dir = parts[4] if len(parts) > 4 else f"models/{model_name.replace('/', '_')}_{task_type}"
        
        await self.train_model(model_name, dataset_name, task_type, output_dir)
    
    async def show_status(self):
        """Show current status"""
        if self.session_id:
            session = self.session_manager.get_session(self.session_id)
            if session:
                self.print_panel(
                    f"Session ID: {self.session_id}\n"
                    f"HF Authenticated: {'✅' if session.hf_authenticated else '❌'}\n"
                    f"Current Model: {session.current_model or 'None'}\n"
                    f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Last Activity: {session.last_activity.strftime('%Y-%m-%d %H:%M:%S')}",
                    "Session Status",
                    "blue"
                )
            else:
                self.print_message("❌ No active session", "red")
        else:
            self.print_message("❌ No session created", "red")
    
    def show_help(self):
        """Show help information"""
        help_text = """
🤗 **HuggingFace Manager Commands**

**Authentication:**
  auth                    - Authenticate with HuggingFace token
  status                  - Show current session status

**Models:**
  models                  - List available models
  models <task>           - List models for specific task
  recommended             - Show recommended models for security

**Datasets:**
  datasets                - Show recommended security datasets
  create-dataset          - Create sample security dataset

**Training:**
  train <model> <dataset> <task> [output]
                         - Train custom model
  
**Examples:**
  train microsoft/codebert-base local:data/vuln.json classification
  models text-classification
  create-dataset

**General:**
  help                    - Show this help
  quit/exit              - Exit the manager
        """
        
        self.print_panel(help_text.strip(), "Help", "green")

async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Archangel HuggingFace Model Manager")
    parser.add_argument('--auth', help='HuggingFace token for authentication')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--task', help='Filter models by task type')
    parser.add_argument('--recommended', action='store_true', help='Show recommended models')
    parser.add_argument('--create-dataset', help='Create sample dataset (specify output file)')
    parser.add_argument('--train', nargs=4, metavar=('MODEL', 'DATASET', 'TASK', 'OUTPUT'), 
                       help='Train model: MODEL DATASET TASK OUTPUT_DIR')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    
    args = parser.parse_args()
    
    cli = HuggingFaceCLI()
    
    # Handle authentication
    if args.auth:
        await cli.authenticate(args.auth)
    
    # Handle commands
    if args.list_models:
        await cli.list_models(args.task)
    elif args.recommended:
        await cli.show_recommended_models()
    elif args.create_dataset:
        await cli.create_security_dataset(args.create_dataset)
    elif args.train:
        model, dataset, task, output = args.train
        await cli.train_model(model, dataset, task, output)
    elif args.interactive or len(sys.argv) == 1:
        await cli.interactive_mode()
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())