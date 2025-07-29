#!/usr/bin/env python3
"""
Archangel Linux - Production AI Security Expert System
Complete production-ready implementation with unified AI orchestration
"""

import asyncio
import sys
import os
import json
import logging
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.unified_ai_orchestrator import (
    create_unified_ai_orchestrator, 
    UnifiedAIRequest, 
    AITaskType, 
    AICapabilityLevel
)
from archangel_lightweight import get_hf_token, validate_hf_token_format, validate_hf_token_access

class ArchangelProduction:
    """
    Production-ready Archangel AI Security Expert System
    
    Features:
    - Unified AI orchestration
    - Production error handling
    - Comprehensive logging
    - User session management
    - Graceful shutdown
    - Configuration management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.orchestrator = None
        self.session_active = False
        self.graceful_shutdown = False
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "log_level": "INFO",
            "session_timeout": 3600,  # 1 hour
            "max_requests_per_session": 100,
            "enable_kernel_integration": True,
            "enable_tool_orchestration": True,
            "default_capability_level": "ADVANCED",
            "model_preferences": {
                "security_analysis": "expert",
                "conversational": "advanced", 
                "code_analysis": "expert"
            }
        }
        
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    print(f"✅ Loaded configuration from {self.config_path}")
            except Exception as e:
                print(f"⚠️ Failed to load config from {self.config_path}: {e}")
                print("Using default configuration...")
        
        return default_config
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_level = getattr(logging, self.config.get("log_level", "INFO"))
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "archangel.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        self.graceful_shutdown = True
        if self.session_active:
            print("⏳ Waiting for current operation to complete...")
    
    async def initialize(self, hf_token: str) -> bool:
        """Initialize the production system"""
        self.logger.info("🚀 Initializing Archangel Production System...")
        
        try:
            # Validate token
            self.logger.info("🔐 Validating HuggingFace token...")
            if not validate_hf_token_format(hf_token):
                self.logger.warning("⚠️ Token format appears invalid")
            
            token_valid, token_msg = await validate_hf_token_access(hf_token)
            if token_valid:
                self.logger.info(f"✅ Token validation: {token_msg}")
            else:
                self.logger.warning(f"⚠️ Token validation: {token_msg}")
                self.logger.info("Continuing - models may still work...")
            
            # Initialize unified orchestrator
            self.logger.info("🎯 Initializing Unified AI Orchestrator...")
            self.orchestrator = create_unified_ai_orchestrator(hf_token)
            
            success = await self.orchestrator.initialize()
            
            if success:
                self.logger.info("✅ Archangel Production System ready!")
                return True
            else:
                self.logger.warning("⚠️ Partial initialization - some features limited")
                return True  # Continue with limited functionality
                
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False
    
    async def security_analysis(self, target: str, detailed: bool = True) -> Dict[str, Any]:
        """Perform comprehensive security analysis"""
        if self.graceful_shutdown:
            return {"error": "System shutting down"}
        
        self.session_active = True
        try:
            self.logger.info(f"🎯 Starting security analysis for: {target}")
            
            capability_level = AICapabilityLevel.EXPERT if detailed else AICapabilityLevel.ADVANCED
            
            request = UnifiedAIRequest(
                task_type=AITaskType.SECURITY_ANALYSIS,
                content=f"Perform {'detailed' if detailed else 'standard'} security analysis of: {target}",
                target=target,
                capability_level=capability_level,
                metadata={
                    "analysis_type": "detailed" if detailed else "standard",
                    "timestamp": asyncio.get_event_loop().time()
                }
            )
            
            response = await self.orchestrator.process_request(request)
            
            result = {
                "success": True,
                "target": target,
                "analysis": response.content,
                "confidence": response.confidence,
                "model_used": response.model_used,
                "execution_time": response.execution_time,
                "recommendations": response.recommendations or [],
                "next_actions": response.next_actions or [],
                "metadata": response.metadata or {}
            }
            
            self.logger.info(f"✅ Security analysis completed for {target} (confidence: {response.confidence:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Security analysis failed for {target}: {e}")
            return {
                "success": False,
                "target": target,
                "error": str(e)
            }
        finally:
            self.session_active = False
    
    async def chat_session(self, session_id: str = "default") -> None:
        """Interactive chat session with AI security expert"""
        self.logger.info(f"💬 Starting chat session: {session_id}")
        
        print("\\n🤖 ARCHANGEL AI SECURITY EXPERT")
        print("=" * 40)
        print("Ask me about cybersecurity, threat analysis, or security best practices.")
        print("Type 'quit' to exit, 'help' for commands.\\n")
        
        conversation_count = 0
        max_conversations = self.config.get("max_requests_per_session", 100)
        
        while not self.graceful_shutdown and conversation_count < max_conversations:
            try:
                self.session_active = True
                
                print("You: ", end="")
                user_input = input().strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._show_chat_help()
                    continue
                elif user_input.lower() == 'status':
                    await self._show_system_status()
                    continue
                elif not user_input:
                    continue
                
                # Process chat request
                request = UnifiedAIRequest(
                    task_type=AITaskType.CONVERSATIONAL_CHAT,
                    content=user_input,
                    session_id=session_id,
                    capability_level=AICapabilityLevel.ADVANCED
                )
                
                response = await self.orchestrator.process_request(request)
                
                print(f"🤖 Archangel: {response.content}")
                
                if response.confidence < 0.6:
                    print(f"   (Note: Lower confidence response - {response.confidence:.2f})")
                
                conversation_count += 1
                print()
                
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                self.logger.error(f"Chat error: {e}")
                print(f"❌ Chat error: {e}")
            finally:
                self.session_active = False
        
        if conversation_count >= max_conversations:
            print(f"ℹ️ Session limit reached ({max_conversations} conversations)")
        
        print("👋 Chat session ended.")
        self.logger.info(f"Chat session {session_id} ended after {conversation_count} conversations")
    
    def _show_chat_help(self):
        """Show chat help"""
        print("\\n📖 CHAT COMMANDS:")
        print("  help     - Show this help")
        print("  status   - Show system status")
        print("  quit     - Exit chat")
        print("\\nYou can ask about:")
        print("  • Security vulnerabilities and threats")
        print("  • Penetration testing methodologies")
        print("  • Security best practices")
        print("  • Incident response procedures")
        print("  • Risk assessment strategies")
        print()
    
    async def _show_system_status(self):
        """Show system status in chat"""
        status = self.orchestrator.get_system_status()
        
        print("\\n📊 SYSTEM STATUS:")
        unified = status.get("unified_orchestrator", {})
        metrics = unified.get("performance_metrics", {})
        
        print(f"  • Total Requests: {metrics.get('total_requests', 0)}")
        print(f"  • Success Rate: {metrics.get('successful_requests', 0)}/{metrics.get('total_requests', 0)}")
        print(f"  • Avg Response Time: {metrics.get('average_response_time', 0):.2f}s")
        
        hf_status = status.get("huggingface_orchestrator", {})
        if hf_status:
            print(f"  • Active AI Models: {len(hf_status.get('active_models', []))}")
            print(f"  • GPU Available: {hf_status.get('gpu_available', False)}")
        
        print()
    
    async def batch_analysis(self, targets: list, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Perform batch security analysis on multiple targets"""
        self.logger.info(f"📊 Starting batch analysis for {len(targets)} targets")
        
        results = {}
        successful = 0
        failed = 0
        
        for i, target in enumerate(targets, 1):
            if self.graceful_shutdown:
                break
                
            print(f"\\nAnalyzing {i}/{len(targets)}: {target}")
            
            result = await self.security_analysis(target, detailed=False)
            results[target] = result
            
            if result.get("success"):
                successful += 1
                print(f"  ✅ Completed (confidence: {result.get('confidence', 0):.2f})")
            else:
                failed += 1
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Save results if output file specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\\n💾 Results saved to: {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save results: {e}")
        
        summary = {
            "total_targets": len(targets),
            "successful": successful,
            "failed": failed,
            "results": results
        }
        
        self.logger.info(f"Batch analysis completed: {successful}/{len(targets)} successful")
        return summary
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("🧹 Cleaning up Archangel Production System...")
        
        if self.orchestrator:
            await self.orchestrator.cleanup()
        
        self.logger.info("✅ Cleanup completed")
    
    def print_banner(self):
        """Print application banner"""
        print("\\n" + "=" * 70)
        print("🛡️  ARCHANGEL LINUX - AI SECURITY EXPERT SYSTEM")
        print("=" * 70)
        print("🧠 Powered by Unified AI Orchestration")
        print("🔗 HuggingFace Neural Networks • Tool Integration • Kernel AI")
        print("🎯 Production-Ready Security Analysis and Consultation")
        print("=" * 70)

async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Archangel Linux - AI Security Expert System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --analyze example.com
  %(prog)s --chat
  %(prog)s --batch targets.txt --output results.json
  %(prog)s --analyze 192.168.1.1 --detailed
        """
    )
    
    parser.add_argument("--analyze", metavar="TARGET", help="Analyze security of target")
    parser.add_argument("--detailed", action="store_true", help="Perform detailed analysis")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat session")
    parser.add_argument("--batch", metavar="FILE", help="Batch analyze targets from file")
    parser.add_argument("--output", metavar="FILE", help="Output file for batch results")
    parser.add_argument("--config", metavar="FILE", help="Configuration file path")
    parser.add_argument("--session-id", default="default", help="Chat session ID")
    
    args = parser.parse_args()
    
    # Initialize production system
    archangel = ArchangelProduction(config_path=args.config)
    archangel.print_banner()
    
    # Get HF token
    hf_token = get_hf_token()
    
    try:
        # Initialize system
        success = await archangel.initialize(hf_token)
        if not success:
            print("❌ Failed to initialize Archangel system")
            return 1
        
        # Execute requested operation
        if args.analyze:
            result = await archangel.security_analysis(args.analyze, detailed=args.detailed)
            
            if result.get("success"):
                print(f"\\n🎯 SECURITY ANALYSIS RESULTS: {args.analyze}")
                print("=" * 50)
                print(f"📊 Confidence: {result['confidence']:.2f}")
                print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")
                print(f"🤖 Model Used: {result['model_used']}")
                print(f"\\n🧠 Analysis:")
                print("-" * 30)
                print(result['analysis'])
                
                if result.get('recommendations'):
                    print(f"\\n💡 Recommendations:")
                    for i, rec in enumerate(result['recommendations'], 1):
                        print(f"  {i}. {rec}")
                
                if result.get('next_actions'):
                    print(f"\\n📋 Next Actions:")
                    for i, action in enumerate(result['next_actions'], 1):
                        print(f"  {i}. {action}")
            else:
                print(f"❌ Analysis failed: {result.get('error')}")
                return 1
        
        elif args.batch:
            try:
                with open(args.batch, 'r') as f:
                    targets = [line.strip() for line in f if line.strip()]
                
                if not targets:
                    print(f"❌ No targets found in {args.batch}")
                    return 1
                
                print(f"📊 Starting batch analysis of {len(targets)} targets...")
                summary = await archangel.batch_analysis(targets, args.output)
                
                print(f"\\n📊 BATCH ANALYSIS SUMMARY")
                print("=" * 30)
                print(f"Total Targets: {summary['total_targets']}")
                print(f"Successful: {summary['successful']}")
                print(f"Failed: {summary['failed']}")
                print(f"Success Rate: {summary['successful']/summary['total_targets']*100:.1f}%")
                
            except FileNotFoundError:
                print(f"❌ File not found: {args.batch}")
                return 1
        
        elif args.chat:
            await archangel.chat_session(args.session_id)
        
        else:
            # Interactive mode selection
            print("\\n🎮 SELECT MODE:")
            print("1. Security Analysis")
            print("2. Interactive Chat")
            print("3. System Status")
            print("4. Exit")
            
            try:
                choice = input("\\nChoose option (1-4): ").strip()
                
                if choice == "1":
                    target = input("Enter target to analyze: ").strip()
                    if target:
                        detailed = input("Detailed analysis? (y/N): ").strip().lower() == 'y'
                        result = await archangel.security_analysis(target, detailed)
                        print(json.dumps(result, indent=2, default=str))
                
                elif choice == "2":
                    await archangel.chat_session()
                
                elif choice == "3":
                    status = archangel.orchestrator.get_system_status()
                    print("\\n📊 SYSTEM STATUS:")
                    print(json.dumps(status, indent=2, default=str))
                
                elif choice == "4":
                    print("👋 Goodbye!")
                
                else:
                    print(f"❌ Invalid choice: {choice}")
                    
            except (EOFError, KeyboardInterrupt):
                print("\\n👋 Goodbye!")
        
    except Exception as e:
        archangel.logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        return 1
    
    finally:
        await archangel.cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))