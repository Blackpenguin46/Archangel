#!/usr/bin/env python3
"""
Live Combat System Test - M2 MacBook Optimized
Demonstrates red team AI agents attacking blue team AI agents in real-time
with container isolation and learning capabilities.
"""

import asyncio
import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from environments.live_adversarial_environment import LiveAdversarialEnvironment
from agents.live_combat_agents import LiveCombatOrchestrator

async def test_container_setup():
    """Test container environment setup."""
    print("🔧 Testing Container Environment Setup...")
    
    env = LiveAdversarialEnvironment()
    
    # Test container creation
    success = await env.setup_combat_environment()
    if success:
        print("✅ Container environment ready")
        return env
    else:
        print("❌ Container setup failed")
        return None

async def test_ai_agent_initialization():
    """Test AI agent initialization with M2-optimized models."""
    print("🧠 Testing AI Agent Initialization...")
    
    orchestrator = LiveCombatOrchestrator()
    
    # Initialize combat simulation
    success = await orchestrator.setup_live_combat_simulation()
    if success:
        print("✅ AI agents initialized successfully")
        return orchestrator
    else:
        print("❌ AI agent initialization failed")
        return None

async def run_quick_combat_demo():
    """Run a quick 5-minute combat demonstration."""
    print("\n🚀 Starting Live Combat Demonstration")
    print("=" * 50)
    
    # Initialize environment
    env = await test_container_setup()
    if not env:
        return False
    
    # Initialize agents
    orchestrator = await test_ai_agent_initialization()
    if not orchestrator:
        return False
    
    print("\n🥊 Starting Red Team vs Blue Team Combat")
    print("Duration: 5 minutes")
    print("Red Team: Kali Linux container (192.168.100.10)")
    print("Blue Team: Ubuntu container (192.168.100.20)")
    
    # Start combat exercise
    combat_results = await orchestrator.execute_live_adversarial_exercise(
        scenario_name="Network Reconnaissance",
        duration_minutes=5
    )
    
    # Display results
    print("\n📊 Combat Results Summary")
    print("-" * 30)
    
    if combat_results:
        print(f"Total Attacks: {combat_results.get('total_attacks', 0)}")
        print(f"Successful Attacks: {combat_results.get('successful_attacks', 0)}")
        print(f"Blocked Attacks: {combat_results.get('blocked_attacks', 0)}")
        print(f"Detection Rate: {combat_results.get('detection_rate', 0):.1f}%")
        
        # Show learning progression
        if 'learning_metrics' in combat_results:
            learning = combat_results['learning_metrics']
            print(f"\n🧠 AI Learning Progress:")
            print(f"Red Team Strategy Evolution: {learning.get('red_team_improvement', 0):.1f}%")
            print(f"Blue Team Defense Improvement: {learning.get('blue_team_improvement', 0):.1f}%")
    
    return True

async def test_memory_usage():
    """Test memory usage on M2 MacBook."""
    print("\n💾 Testing Memory Usage...")
    
    import psutil
    import torch
    
    # Check system memory
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"Memory Usage: {memory.percent}%")
    
    # Check PyTorch/MPS availability
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) available")
    else:
        print("💻 Using CPU (still fast on M2)")
    
    return memory.percent < 80  # Ensure we're not using too much memory

async def main():
    """Main test function."""
    print("🍎 Archangel Live Combat System - M2 MacBook Test")
    print("=" * 60)
    
    # System checks
    memory_ok = await test_memory_usage()
    if not memory_ok:
        print("⚠️  High memory usage detected. Consider closing other applications.")
    
    # Run combat demonstration
    success = await run_quick_combat_demo()
    
    if success:
        print("\n🎉 Live Combat System Test Complete!")
        print("\nWhat you just saw:")
        print("• AI red team agents autonomously attacking")
        print("• AI blue team agents autonomously defending")
        print("• Real-time learning and strategy adaptation")
        print("• Container isolation for safe testing")
        print("• M2 MacBook optimized performance")
        
        print("\n🚀 Ready for full autonomous security operations!")
    else:
        print("\n❌ Test failed. Check logs for details.")
        print("\nTroubleshooting:")
        print("• Ensure Docker is running")
        print("• Check internet connection for model downloads")
        print("• Verify sufficient memory available")

if __name__ == "__main__":
    asyncio.run(main())