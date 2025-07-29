#!/usr/bin/env python3
"""
Archangel Unified AI Orchestration Demo
Complete demonstration of AI orchestration capabilities
"""

import asyncio
import sys
import os
import json
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.unified_ai_orchestrator import (
    create_unified_ai_orchestrator, 
    UnifiedAIRequest, 
    AITaskType, 
    AICapabilityLevel
)
from archangel_lightweight import get_hf_token, validate_hf_token_format

async def demo_security_analysis(orchestrator, target: str):
    """Demo comprehensive security analysis"""
    print(f"\n🎯 UNIFIED AI SECURITY ANALYSIS: {target}")
    print("=" * 60)
    
    request = UnifiedAIRequest(
        task_type=AITaskType.SECURITY_ANALYSIS,
        content=f"Perform comprehensive security analysis of: {target}",
        target=target,
        capability_level=AICapabilityLevel.EXPERT,
        metadata={"demo_mode": True}
    )
    
    response = await orchestrator.process_request(request)
    
    print(f"📊 Analysis Results:")
    print(f"  Target: {target}")
    print(f"  Confidence: {response.confidence:.2f}")
    print(f"  Model Used: {response.model_used}")
    print(f"  Execution Time: {response.execution_time:.2f}s")
    
    print(f"\n🧠 AI Analysis:")
    print("-" * 40)
    print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
    
    if response.recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(response.recommendations[:3], 1):
            print(f"  {i}. {rec}")
    
    if response.next_actions:
        print(f"\n📋 Next Actions:")
        for i, action in enumerate(response.next_actions[:3], 1):
            print(f"  {i}. {action}")
    
    return response

async def demo_conversational_ai(orchestrator, message: str):
    """Demo conversational AI capabilities"""
    print(f"\n💬 CONVERSATIONAL AI DEMO")
    print("=" * 40)
    
    request = UnifiedAIRequest(
        task_type=AITaskType.CONVERSATIONAL_CHAT,
        content=message,
        capability_level=AICapabilityLevel.ADVANCED,
        session_id="demo_session"
    )
    
    response = await orchestrator.process_request(request)
    
    print(f"👤 Human: {message}")
    print(f"🤖 AI: {response.content}")
    print(f"   Confidence: {response.confidence:.2f} | Time: {response.execution_time:.2f}s")
    
    return response

async def demo_tool_orchestration(orchestrator, target: str):
    """Demo AI-driven tool orchestration"""
    print(f"\n🛠️ TOOL ORCHESTRATION DEMO: {target}")
    print("=" * 50)
    
    request = UnifiedAIRequest(
        task_type=AITaskType.TOOL_ORCHESTRATION,
        content=f"Execute security tools against target: {target}",
        target=target,
        capability_level=AICapabilityLevel.AUTONOMOUS
    )
    
    response = await orchestrator.process_request(request)
    
    print(f"🔧 Tool Execution Results:")
    print(response.content)
    
    if response.metadata and "tools_executed" in response.metadata:
        print(f"📊 Tools Executed: {response.metadata['tools_executed']}")
    
    return response

async def demo_kernel_integration(orchestrator):
    """Demo kernel-level AI decision making"""
    print(f"\n⚡ KERNEL AI INTEGRATION DEMO")
    print("=" * 40)
    
    kernel_context = {
        "process": "suspicious_binary",
        "pid": 1337,
        "uid": 0,
        "syscall": 59,  # execve
        "action": "process_execution"
    }
    
    request = UnifiedAIRequest(
        task_type=AITaskType.KERNEL_DECISION,
        content=f"Analyze kernel security context: {json.dumps(kernel_context, indent=2)}",
        capability_level=AICapabilityLevel.EXPERT,
        metadata={"kernel_context": kernel_context}
    )
    
    response = await orchestrator.process_request(request)
    
    print(f"🛡️ Kernel Security Decision:")
    print(response.content)
    
    # Extract decision from response
    decision = "MONITOR"  # Default
    if "DENY" in response.content.upper() or "BLOCK" in response.content.upper():
        decision = "DENY"
    elif "ALLOW" in response.content.upper():
        decision = "ALLOW"
    
    print(f"\n🎯 Final Decision: {decision}")
    print(f"🧠 Confidence: {response.confidence:.2f}")
    
    return response

async def show_system_status(orchestrator):
    """Show comprehensive system status"""
    print(f"\n📊 SYSTEM STATUS")
    print("=" * 30)
    
    status = orchestrator.get_system_status()
    
    print(f"🎯 Unified Orchestrator:")
    unified_status = status.get("unified_orchestrator", {})
    print(f"  Initialized: {unified_status.get('initialized', False)}")
    print(f"  Total Requests: {unified_status.get('performance_metrics', {}).get('total_requests', 0)}")
    print(f"  Success Rate: {unified_status.get('performance_metrics', {}).get('successful_requests', 0)}/{unified_status.get('performance_metrics', {}).get('total_requests', 0)}")
    print(f"  Avg Response Time: {unified_status.get('performance_metrics', {}).get('average_response_time', 0):.2f}s")
    
    print(f"\n🤗 HuggingFace Integration:")
    hf_status = status.get("huggingface_orchestrator", {})
    if hf_status:
        print(f"  Authenticated: {hf_status.get('authenticated', False)}")
        print(f"  Active Models: {len(hf_status.get('active_models', []))}")
        print(f"  GPU Available: {hf_status.get('gpu_available', False)}")
        print(f"  SmolAgents Ready: {hf_status.get('smolagents_available', False)}")
    else:
        print(f"  Status: Not initialized")
    
    print(f"\n⚡ Kernel Interface:")
    kernel_status = status.get("kernel_interface", {})
    print(f"  Available: {kernel_status.get('available', False)}")
    print(f"  Status: {kernel_status.get('status', 'unknown')}")
    
    print(f"\n🛠️ Tool Orchestration:")
    tool_status = status.get("tool_orchestrator", {})
    print(f"  Available: {tool_status.get('available', False)}")
    if tool_status.get('tools'):
        print(f"  Tools: {', '.join(tool_status['tools'])}")
    
    print(f"\n🧠 Claude Agents:")
    claude_agents = status.get("claude_agents", {})
    for agent_name, agent_info in claude_agents.items():
        if isinstance(agent_info, dict):
            print(f"  {agent_name}: {'Available' if agent_info.get('available') else 'Development Mode'}")

async def interactive_demo(orchestrator):
    """Interactive demo session"""
    print(f"\n🎮 INTERACTIVE UNIFIED AI DEMO")
    print("=" * 40)
    print("Commands:")
    print("  analyze <target>     - Security analysis")
    print("  chat <message>       - Conversational AI")
    print("  tools <target>       - Tool orchestration") 
    print("  kernel               - Kernel integration demo")
    print("  status               - System status")
    print("  quit                 - Exit")
    print()
    
    while True:
        try:
            print("UnifiedAI> ", end="")
            command = input().strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            elif command.lower().startswith('analyze '):
                target = command[8:].strip()
                if target:
                    await demo_security_analysis(orchestrator, target)
            elif command.lower().startswith('chat '):
                message = command[5:].strip()
                if message:
                    await demo_conversational_ai(orchestrator, message)
            elif command.lower().startswith('tools '):
                target = command[6:].strip()
                if target:
                    await demo_tool_orchestration(orchestrator, target)
            elif command.lower() == 'kernel':
                await demo_kernel_integration(orchestrator)
            elif command.lower() == 'status':
                await show_system_status(orchestrator)
            elif command.strip() == '':
                continue
            else:
                print(f"❌ Unknown command: {command}")
            
            print()
            
        except (EOFError, KeyboardInterrupt):
            break
    
    print("\n👋 Interactive demo ended.")

async def full_capabilities_demo():
    """Complete demonstration of all unified AI capabilities"""
    print("🚀 ARCHANGEL UNIFIED AI ORCHESTRATION DEMO")
    print("=" * 60)
    print("Demonstrating complete AI orchestration capabilities")
    print("Real neural networks • Tool integration • Kernel AI")
    print()
    
    # Get HF token
    hf_token = get_hf_token()
    
    # Initialize unified orchestrator
    print("🎯 Initializing Unified AI Orchestrator...")
    orchestrator = create_unified_ai_orchestrator(hf_token)
    
    try:
        success = await orchestrator.initialize()
        if not success:
            print("⚠️ Partial initialization - some features may be limited")
            print("Continuing with available capabilities...\n")
        
        # Show system status
        await show_system_status(orchestrator)
        
        # Demo 1: Security Analysis
        await demo_security_analysis(orchestrator, "example.com")
        
        # Demo 2: Conversational AI
        await demo_conversational_ai(
            orchestrator, 
            "What are the most critical security vulnerabilities to check for in web applications?"
        )
        
        # Demo 3: Tool Orchestration
        await demo_tool_orchestration(orchestrator, "192.168.1.1")
        
        # Demo 4: Kernel Integration
        await demo_kernel_integration(orchestrator)
        
        # Interactive session
        print(f"\n🎮 Starting interactive demo...")
        await interactive_demo(orchestrator)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up resources...")
        await orchestrator.cleanup()
        print("✅ Demo completed successfully!")

async def quick_test():
    """Quick test of core functionality"""
    print("⚡ QUICK UNIFIED AI TEST")
    print("=" * 30)
    
    hf_token = get_hf_token()
    orchestrator = create_unified_ai_orchestrator(hf_token)
    
    try:
        await orchestrator.initialize()
        
        # Quick security analysis test
        request = UnifiedAIRequest(
            task_type=AITaskType.SECURITY_ANALYSIS,
            content="Quick security assessment of localhost",
            target="localhost",
            capability_level=AICapabilityLevel.BASIC
        )
        
        response = await orchestrator.process_request(request)
        
        print(f"✅ Test successful!")
        print(f"   Model: {response.model_used}")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Time: {response.execution_time:.2f}s")
        print(f"   Response: {response.content[:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await orchestrator.cleanup()

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(quick_test())
    else:
        asyncio.run(full_capabilities_demo())

if __name__ == "__main__":
    main()