#!/usr/bin/env python3
"""
Quick test of the autonomous enterprise scenario
"""

import asyncio
import sys
from scenarios.autonomous_enterprise_scenario import AutonomousScenarioOrchestrator

async def test_autonomous_scenario():
    """Test autonomous scenario with mock containers"""
    
    print("🧪 Testing Autonomous Enterprise Scenario Components...")
    
    # Use mock container names for testing
    red_container = "test-red-container"
    blue_container = "test-blue-container"
    
    try:
        # Initialize orchestrator
        orchestrator = AutonomousScenarioOrchestrator(red_container, blue_container)
        print("✅ Orchestrator initialized successfully")
        
        # Test data generator
        print("\n🏢 Testing enterprise data generation...")
        await orchestrator.data_generator.generate_enterprise_data(blue_container)
        print("✅ Enterprise data generation test completed")
        
        # Test agents (this will fail with mock containers but we can test initialization)
        print("\n🔴 Testing red team agent initialization...")
        red_agent = orchestrator.red_agent
        print(f"✅ Red team agent initialized for container: {red_agent.container_name}")
        
        print("\n🔵 Testing blue team agent initialization...")
        blue_agent = orchestrator.blue_agent
        print(f"✅ Blue team agent initialized for container: {blue_agent.container_name}")
        
        print("\n🎭 All autonomous scenario components initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🎭 AUTONOMOUS ENTERPRISE SCENARIO - COMPONENT TEST")
    print("=" * 60)
    
    success = await test_autonomous_scenario()
    
    if success:
        print("\n🎉 All tests passed! Autonomous enterprise scenario is ready.")
        print("\nTo run the full scenario:")
        print("python3 archangel.py --enterprise --duration 10")
        return 0
    else:
        print("\n❌ Tests failed. Check component initialization.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))